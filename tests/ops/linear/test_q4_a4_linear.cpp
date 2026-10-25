// quantize_a4_g64 and q4_a4_linear (ninfer/ops/q4_a4_linear.h) on the GPU.
//
//   1. quantize_a4_g64 is an exact transform: xq and xs must equal the host reference
//      (a4_g64_reference.h) byte for byte.
//   2. q4_a4_linear against an exact oracle over the same int4 inputs: both operands decoded to
//      their represented values on the host and summed in FP64, independently of any tiling.
//      Criterion kExactPath: relative L2 <= 3e-3 and no element off by more than two BF16 steps
//      (2^-7) of the largest reference (FP32 group accumulation plus the output's own rounding is
//      ~1.1e-3). Token magnitudes are hashed powers of two in [2^-3, 2^3], so a wrong token's (or a
//      shared) activation scale is off by up to 64x and fails clearly.
//   3. Agreement with the existing integer-A8 routes on the same weight and BF16 activation
//      (gate_up: q4a8 LinearSwiGLU, compared after applying SwiGLU to this Op's rows; down: the
//      A8 LinearAdd into a zero residual): cosine >= 0.995 on the outlier-free fixture. The
//      outlier fixture (the A8 route test's own: every 431st channel x24) is reported, not gated:
//      a group holding a 24x channel quantises its other 63 channels to zero at 4 bits, and the
//      host simulation of exactly these fixtures predicts a cosine of about 0.98 there.
//   4. Timing at T = 4096 against the A8 routes (printed; the brief's target is >= 1.4x).
//   5. The lossy scout's routes through the policy-bearing Ops (LinearPolicy::ScoutA4Int): the
//      LinearSwiGLU route (int4 GEMM over all gate/up rows, then a separate SwiGLU) and the
//      LinearAdd route (int4 GEMM, then residual_add) must agree with this Op's own output
//      (cosine >= 0.9999), and at T = 4096 they are timed against the same Ops under AllowA8Int,
//      whose gate_up route fuses SwiGLU. That ratio is the scout GEMM gate (>= 1.4x).
//
// Shapes: the 27B mlp gate_up [34816,5120] and down [5120,17408] at T = 512 and 4096, plus small
// cases for the N/T tails, odd N, T = 1, the RowSplitPanel layout and FP16 input/output.

#include "ninfer/ops/linear_add.h"
#include "ninfer/ops/linear_swiglu.h"
#include "ninfer/ops/q4_a4_linear.h"

#include "ops/linear_swiglu/q4a8/q4a8_linear_swiglu.h"

#include "ops/linear/a4_g64_reference.h"
#include "ops/op_check.h"
#include "ops/op_tester.h"
#include "ops/quantized_weight.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <exception>
#include <functional>
#include <iostream>
#include <memory>
#include <set>
#include <string>
#include <vector>

namespace {

using namespace ninfer;
using ninfer::test::GuardedDeviceBuffer;
using ninfer::test::ReductionCriterion;
namespace qw = ninfer::test::quantized_weight;
namespace a4 = ninfer::test::a4_g64;

// Same int4 inputs on both sides, so only FP32 accumulation order (~1e-6 relative at K = 17408)
// and the output's own rounding remain: for a BF16 output at most 2^-9 per element, about 1.1e-3
// relative L2. 3e-3 leaves ~2.7x for that; the gross term is the house two-BF16-step floor.
// Tightened from 1e-2/1e-2 together with the non-uniform token magnitudes; compiled, not yet run on
// a GPU at this limit.
constexpr ReductionCriterion kExactPath{3.0e-3, 0.0, ninfer::test::kBf16GrossRelativeFloor};
constexpr double kA8CosineTarget = 0.995;
constexpr double kSpeedupTarget  = 1.4;
constexpr double kRouteCosine    = 0.9999;

enum class Fixture { Uniform, Outlier };

const char* fixture_name(Fixture f) { return f == Fixture::Uniform ? "uniform" : "outlier"; }

std::uint16_t f32_to_bf16(float v) {
    std::uint32_t bits;
    std::memcpy(&bits, &v, sizeof(bits));
    bits += 0x7fffU + ((bits >> 16) & 1U);
    return static_cast<std::uint16_t>(bits >> 16);
}

float bf16_to_f32(std::uint16_t h) { return qw::detail::bf16_to_f32(h); }

// x [K,T] as 16-bit words (BF16 or FP16) and the FP32 values they represent.
struct HostActivation {
    std::vector<std::uint16_t> bits;
    std::vector<float> values;
};

HostActivation make_activation(std::int32_t k, std::int32_t t, std::uint32_t seed,
                               Fixture fixture, DType dtype) {
    HostActivation a;
    a.bits.resize(static_cast<std::size_t>(k) * t);
    a.values.resize(a.bits.size());
    for (std::int32_t token = 0; token < t; ++token) {
        // Each token's magnitude is a hashed power of two in [2^-3, 2^3], so its per-group A4 scales
        // differ from its neighbours' by up to 64x: a kernel that applied another token's scale (or
        // a tile-shared one) would be off by that factor, far outside the exact-path bound, instead
        // of hiding in the rounding noise of uniformly scaled tokens.
        const int exponent = static_cast<int>(
            qw::detail::mix64(static_cast<std::uint64_t>(token) * 0x9e3779b97f4a7c15ULL ^ seed) % 7U) - 3;
        const float magnitude = std::ldexp(1.0F, exponent);
        for (std::int32_t r = 0; r < k; ++r) {
            const std::size_t i = static_cast<std::size_t>(token) * k + r;
            const std::uint64_t h =
                qw::detail::mix64((static_cast<std::uint64_t>(token) << 32) ^
                                  static_cast<std::uint64_t>(r) ^ seed);
            float v = magnitude *
                      static_cast<float>(static_cast<std::int32_t>(h & 0xffffU) - 32768) / 32768.0F;
            if (fixture == Fixture::Outlier && (r % 431) == 7) { v *= 24.0F; }
            if (dtype == DType::BF16) {
                a.bits[i]   = f32_to_bf16(v);
                a.values[i] = bf16_to_f32(a.bits[i]);
            } else {
                a.bits[i]   = qw::detail::f32_to_f16(v);
                a.values[i] = qw::detail::f16_to_f32(a.bits[i]);
            }
        }
    }
    return a;
}

// RowSplit -> RowSplitPanel: within each k-group the 32-byte records of 4 consecutive rows become
// contiguous. Scales are unchanged. The oracle keeps decoding the original row-split payload.
std::vector<std::uint8_t> to_panel(const qw::PackedWeight& w) {
    std::vector<std::uint8_t> out = w.payload;
    const std::size_t groups      = static_cast<std::size_t>(w.weight.k) / 64;
    const std::size_t row_bytes   = groups * 32;
    for (std::size_t row = 0; row < static_cast<std::size_t>(w.weight.n); ++row) {
        const std::size_t panel = row / kRowSplitPanelRows;
        const std::size_t inner = row % kRowSplitPanelRows;
        for (std::size_t g = 0; g < groups; ++g) {
            std::memcpy(out.data() + panel * kRowSplitPanelRows * row_bytes +
                            g * 32 * kRowSplitPanelRows + inner * 32,
                        w.payload.data() + row * row_bytes + g * 32, 32);
        }
    }
    return out;
}

std::vector<double> read_output(const GuardedDeviceBuffer& buffer, std::size_t elements,
                                DType dtype) {
    std::vector<std::uint16_t> host(elements);
    buffer.copy_to_host(host.data(), elements * sizeof(std::uint16_t));
    std::vector<double> out(elements);
    for (std::size_t i = 0; i < elements; ++i) {
        out[i] = dtype == DType::BF16 ? bf16_to_f32(host[i]) : qw::detail::f16_to_f32(host[i]);
    }
    return out;
}

double cosine(const std::vector<double>& a, const std::vector<double>& b) {
    long double ab = 0, aa = 0, bb = 0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        ab += static_cast<long double>(a[i]) * b[i];
        aa += static_cast<long double>(a[i]) * a[i];
        bb += static_cast<long double>(b[i]) * b[i];
    }
    return static_cast<double>(ab / std::sqrt(aa * bb));
}

float elapsed_ms(const std::function<void()>& launch, int iterations) {
    for (int i = 0; i < 3; ++i) { launch(); }
    cudaEvent_t start = nullptr;
    cudaEvent_t stop  = nullptr;
    test::cuda_check(cudaEventCreate(&start), "cudaEventCreate");
    test::cuda_check(cudaEventCreate(&stop), "cudaEventCreate");
    test::cuda_check(cudaEventRecord(start), "cudaEventRecord");
    for (int i = 0; i < iterations; ++i) { launch(); }
    test::cuda_check(cudaEventRecord(stop), "cudaEventRecord");
    test::cuda_check(cudaEventSynchronize(stop), "cudaEventSynchronize");
    float ms = 0.0F;
    test::cuda_check(cudaEventElapsedTime(&ms, start, stop), "cudaEventElapsedTime");
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return ms / static_cast<float>(iterations);
}

enum class A8Route { None, GateUp, Down };

struct Case {
    std::int32_t n;
    std::int32_t k;
    std::int32_t t;
    Fixture fixture;
    DType in_dtype;
    DType out_dtype;
    bool panel;
    A8Route a8;
    bool time;
};

std::string label_of(const Case& c) {
    return "q4_a4_linear N=" + std::to_string(c.n) + " K=" + std::to_string(c.k) +
           " T=" + std::to_string(c.t) + " " + fixture_name(c.fixture) +
           (c.in_dtype == DType::FP16 ? " in=FP16" : " in=BF16") +
           (c.out_dtype == DType::FP16 ? " out=FP16" : " out=BF16") + (c.panel ? " panel" : "");
}

// Rows and tokens sampled for the oracle: both ends of every axis, the first tile seam, and
// hashed interior points, so a tail or a seam that goes wrong is always looked at.
std::vector<std::int32_t> sample_axis(std::int32_t extent, std::uint32_t seed, int interior) {
    std::set<std::int32_t> picks;
    for (std::int32_t v : {0, 1, 7, 8, 15, 16, 63, 64, 127, 128, 129, extent - 2, extent - 1}) {
        if (v >= 0 && v < extent) { picks.insert(v); }
    }
    for (int i = 0; i < interior; ++i) {
        picks.insert(static_cast<std::int32_t>(qw::detail::mix64(seed * 7919U + i) %
                                               static_cast<std::uint64_t>(extent)));
    }
    return {picks.begin(), picks.end()};
}

int run_case(const Case& c) {
    const std::string label = label_of(c);
    int failures            = 0;

    const qw::PackedWeight host_weight = qw::make_patterned_weight(
        QType::Q4_G64_FP16, c.n, c.k, 4801U + static_cast<std::uint32_t>(c.n),
        {qw::RowSplitScalePattern::Unit, qw::RowSplitCodePattern::Hashed});
    const std::vector<std::uint8_t> payload = c.panel ? to_panel(host_weight) : host_weight.payload;
    GuardedDeviceBuffer device_weight(payload.size());
    device_weight.copy_from_host(payload.data(), payload.size());
    Weight weight = host_weight.device_weight(device_weight.data());
    if (c.panel) { weight.layout = QuantLayout::RowSplitPanel; }

    const HostActivation x = make_activation(c.k, c.t, 91U + static_cast<std::uint32_t>(c.k),
                                             c.fixture, c.in_dtype);
    GuardedDeviceBuffer device_x(x.bits.size() * sizeof(std::uint16_t));
    device_x.copy_from_host(x.bits.data(), x.bits.size() * sizeof(std::uint16_t));

    const std::size_t code_bytes   = static_cast<std::size_t>(c.k / 2) * c.t;
    const std::size_t scale_words  = static_cast<std::size_t>(c.k / 64) * c.t;
    const std::size_t out_elements = static_cast<std::size_t>(c.n) * c.t;
    GuardedDeviceBuffer device_xq(code_bytes);
    GuardedDeviceBuffer device_xs(scale_words * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_out(out_elements * sizeof(std::uint16_t));
    device_xq.fill(0xee);
    device_xs.fill(0xee);
    device_out.fill(0xff);

    Tensor tx(device_x.data(), c.in_dtype, {c.k, c.t});
    Tensor txq(device_xq.data(), DType::U8, {c.k / 2, c.t});
    Tensor txs(device_xs.data(), DType::FP16, {c.k / 64, c.t});
    Tensor tout(device_out.data(), c.out_dtype, {c.n, c.t});
    ops::quantize_a4_g64(tx, txq, txs, nullptr);
    ops::q4_a4_linear(txq, txs, weight, tout, nullptr);
    test::cuda_check(cudaDeviceSynchronize(), "synchronize q4_a4_linear");
    failures += device_xq.verify_guards(label + " xq");
    failures += device_xs.verify_guards(label + " xs");
    failures += device_out.verify_guards(label + " out");
    failures += device_weight.verify_guards(label + " weight");
    failures += device_x.verify_guards(label + " x");

    // 1. The quantiser, exactly.
    const a4::Activation ref = a4::quantize(x.values, c.k, c.t);
    {
        std::vector<std::uint8_t> got_codes(code_bytes);
        std::vector<std::uint16_t> got_scales(scale_words);
        device_xq.copy_to_host(got_codes.data(), code_bytes);
        device_xs.copy_to_host(got_scales.data(), scale_words * sizeof(std::uint16_t));
        std::size_t bad_codes = 0, bad_scales = 0, first = 0;
        for (std::size_t i = 0; i < code_bytes; ++i) {
            if (got_codes[i] != ref.codes[i] && bad_codes++ == 0) { first = i; }
        }
        for (std::size_t i = 0; i < scale_words; ++i) {
            if (got_scales[i] != ref.scales[i]) { ++bad_scales; }
        }
        if (bad_codes != 0 || bad_scales != 0) {
            std::cerr << label << ": quantize_a4_g64 differs from the reference: " << bad_codes
                      << " code bytes (first at " << first << "), " << bad_scales << " scales\n";
            ++failures;
        }
    }

    // 2. The product against the exact oracle over the same int4 inputs.
    const std::vector<double> got = read_output(device_out, out_elements, c.out_dtype);
    const std::vector<std::int32_t> rows   = sample_axis(c.n, 11U, 40);
    const std::vector<std::int32_t> tokens = sample_axis(c.t, 13U, 40);
    std::vector<double> actual;
    std::vector<double> expected;
    double truth_err = 0.0, truth_ref = 0.0;
    for (const std::int32_t row : rows) {
        const std::vector<double> w_row = a4::decode_weight_row(host_weight, row);
        for (const std::int32_t token : tokens) {
            expected.push_back(a4::exact_dot(w_row, ref, token));
            actual.push_back(got[static_cast<std::size_t>(token) * c.n + row]);
            double unquantised = 0.0;
            for (std::int32_t k = 0; k < c.k; ++k) {
                unquantised += w_row[static_cast<std::size_t>(k)] *
                               x.values[static_cast<std::size_t>(token) * c.k + k];
            }
            truth_err += (expected.back() - unquantised) * (expected.back() - unquantised);
            truth_ref += unquantised * unquantised;
        }
    }
    failures += test::verify_reduction(label + " vs exact int4 oracle", actual, expected, kExactPath);
    const test::ReductionStats stats = test::compute_reduction_stats(
        actual.data(), expected.data(), static_cast<std::int64_t>(actual.size()));
    std::cout << "  " << label << ": relative_l2 vs exact int4 oracle " << stats.relative_l2
              << " (limit " << kExactPath.relative_l2 << ", " << actual.size()
              << " samples); A4 cast vs unquantised input " << std::sqrt(truth_err / truth_ref)
              << " (informational)\n";

    if (c.a8 == A8Route::None) { return failures; }

    // 3. Agreement with the A8 route, and 4. timing against it.
    Tensor tx_bf16(device_x.data(), DType::BF16, {c.k, c.t});
    std::vector<double> a4_view;
    std::vector<double> a8_view;
    std::function<void()> run_a8;
    std::size_t a8_bytes = 0;
    std::unique_ptr<GuardedDeviceBuffer> a8_out;
    std::unique_ptr<WorkspaceArena> a8_ws;
    if (c.a8 == A8Route::GateUp) {
        const std::int32_t half = c.n / 2;
        a8_bytes                = static_cast<std::size_t>(half) * c.t * sizeof(std::uint16_t);
        a8_out                  = std::make_unique<GuardedDeviceBuffer>(a8_bytes);
        a8_ws = std::make_unique<WorkspaceArena>(std::max<std::size_t>(
            ops::detail::q4a8_swiglu_workspace_capacity_bytes(c.t, c.t), 256));
        run_a8 = [&, half] {
            Tensor dst(a8_out->data(), DType::BF16, {half, c.t});
            ops::detail::q4a8_swiglu_launch(tx_bf16, weight, dst, *a8_ws, nullptr);
        };
        run_a8();
        test::cuda_check(cudaDeviceSynchronize(), "synchronize a8 swiglu");
        a8_view = read_output(*a8_out, static_cast<std::size_t>(half) * c.t, DType::BF16);
        a4_view.resize(a8_view.size());
        for (std::int32_t token = 0; token < c.t; ++token) {
            for (std::int32_t r = 0; r < half; ++r) {
                const double g = got[static_cast<std::size_t>(token) * c.n + r];
                const double u = got[static_cast<std::size_t>(token) * c.n + half + r];
                a4_view[static_cast<std::size_t>(token) * half + r] = g / (1.0 + std::exp(-g)) * u;
            }
        }
    } else {
        a8_bytes = out_elements * sizeof(std::uint16_t);
        a8_out   = std::make_unique<GuardedDeviceBuffer>(a8_bytes);
        a8_out->fill(0); // a zero residual, so the LinearAdd result is the projection itself
        a8_ws = std::make_unique<WorkspaceArena>(std::max<std::size_t>(
            ops::detail::a8_add_workspace_capacity_bytes(c.k, c.t, c.t), 256));
        run_a8 = [&] {
            Tensor residual(a8_out->data(), DType::BF16, {c.n, c.t});
            ops::detail::a8_add_launch(tx_bf16, weight, residual, *a8_ws, nullptr);
        };
        run_a8();
        test::cuda_check(cudaDeviceSynchronize(), "synchronize a8 add");
        a8_view = read_output(*a8_out, out_elements, DType::BF16);
        a4_view = got;
    }
    failures += a8_out->verify_guards(label + " a8 out");
    const double cos = cosine(a4_view, a8_view);
    const bool gated = c.fixture == Fixture::Uniform;
    std::cout << "  " << label << ": cosine vs the A8 route " << cos << " (target "
              << kA8CosineTarget << (gated ? ", gated" : ", reported only") << ")"
              << (cos < kA8CosineTarget ? "  BELOW TARGET" : "") << '\n';
    if (gated && !(cos >= kA8CosineTarget)) {
        std::cerr << label << ": cosine vs the A8 route " << cos << " < " << kA8CosineTarget
                  << '\n';
        ++failures;
    }

    // 5. The scout's routes through the policy-bearing Ops, against this Op's own output.
    std::unique_ptr<GuardedDeviceBuffer> route_out;
    std::unique_ptr<WorkspaceArena> route_ws;
    std::function<void(ops::LinearPolicy)> run_route;
    {
        const bool gate_up       = c.a8 == A8Route::GateUp;
        const std::int32_t rows  = gate_up ? c.n / 2 : c.n;
        const std::size_t words  = static_cast<std::size_t>(rows) * c.t;
        route_out                = std::make_unique<GuardedDeviceBuffer>(words * sizeof(std::uint16_t));
        std::size_t route_bytes  = 0;
        for (const auto policy : {ops::LinearPolicy::ScoutA4Int, ops::LinearPolicy::AllowA8Int}) {
            route_bytes = std::max(
                route_bytes, gate_up ? ops::linear_swiglu_workspace_capacity_bytes(
                                           QType::Q4_G64_FP16, c.n, c.k, policy, c.t, c.t)
                                     : ops::linear_add_workspace_capacity_bytes(
                                           QType::Q4_G64_FP16, c.n, c.k, policy, c.t, c.t));
        }
        route_ws  = std::make_unique<WorkspaceArena>(std::max<std::size_t>(route_bytes, 256));
        run_route = [&, gate_up, rows](ops::LinearPolicy policy) {
            route_ws->reset();
            Tensor dst(route_out->data(), DType::BF16, {rows, c.t});
            if (gate_up) {
                ops::linear_swiglu(tx_bf16, weight, dst, policy, *route_ws, nullptr);
            } else {
                ops::linear_add(tx_bf16, weight, dst, policy, *route_ws, nullptr);
            }
        };
        if (!gate_up) { route_out->fill(0); } // a zero residual
        run_route(ops::LinearPolicy::ScoutA4Int);
        test::cuda_check(cudaDeviceSynchronize(), "synchronize scout a4 route");
        failures += route_out->verify_guards(label + " scout route out");
        const double route_cos = cosine(read_output(*route_out, words, DType::BF16), a4_view);
        std::cout << "  " << label << ": ScoutA4Int "
                  << (gate_up ? "linear_swiglu" : "linear_add") << " route vs this Op "
                  << route_cos << " (gated >= " << kRouteCosine << ")\n";
        if (!(route_cos >= kRouteCosine)) {
            std::cerr << label << ": the ScoutA4Int route disagrees with q4_a4_linear (cosine "
                      << route_cos << ")\n";
            ++failures;
        }
    }

    if (c.time) {
        constexpr int kIterations = 20;
        const float ms_gemm  = elapsed_ms([&] { ops::q4_a4_linear(txq, txs, weight, tout, nullptr); },
                                          kIterations);
        const float ms_a4    = elapsed_ms(
            [&] {
                ops::quantize_a4_g64(tx, txq, txs, nullptr);
                ops::q4_a4_linear(txq, txs, weight, tout, nullptr);
            },
            kIterations);
        const float ms_a8    = elapsed_ms(run_a8, kIterations);
        const double tops    = 2.0 * c.n * static_cast<double>(c.k) * c.t / (ms_gemm * 1.0e-3) / 1.0e12;
        std::cout << "  PERF " << label << ": A4 GEMM " << ms_gemm << " ms (" << tops
                  << " TOP/s), A4 quantize+GEMM " << ms_a4 << " ms, A8 route " << ms_a8
                  << " ms (includes its own s8 quantiser"
                  << (c.a8 == A8Route::GateUp ? " and the fused SwiGLU" : "") << "); speedup "
                  << ms_a8 / ms_a4 << "x with quantize, " << ms_a8 / ms_gemm
                  << "x GEMM only (target >= " << kSpeedupTarget << "x)\n";
        // The scout as it runs: the Op under ScoutA4Int against the same Op under AllowA8Int.
        // For gate_up that is quantise + int4 GEMM + separate SwiGLU against the fused A8 route.
        const float ms_scout = elapsed_ms([&] { run_route(ops::LinearPolicy::ScoutA4Int); },
                                          kIterations);
        const float ms_int8  = elapsed_ms([&] { run_route(ops::LinearPolicy::AllowA8Int); },
                                          kIterations);
        std::cout << "  PERF " << label << ": scout "
                  << (c.a8 == A8Route::GateUp ? "linear_swiglu (A4 + SwiGLU)" : "linear_add (A4)")
                  << " " << ms_scout << " ms vs AllowA8Int "
                  << (c.a8 == A8Route::GateUp ? "(A8 fused SwiGLU) " : "") << ms_int8
                  << " ms: " << ms_int8 / ms_scout << "x (scout GEMM gate >= " << kSpeedupTarget
                  << "x)\n";
    }
    return failures;
}

// The Op must refuse what its contract does not cover, before touching any pointer.
int run_validation() {
    int failures = 0;
    auto* fake   = reinterpret_cast<void*>(static_cast<std::uintptr_t>(1) << 20);
    const qw::PackedWeight q4 = qw::make_patterned_weight(QType::Q4_G64_FP16, 64, 256, 1U);
    const qw::PackedWeight q5 = qw::make_patterned_weight(QType::Q5_G64_FP16, 64, 256, 1U);
    const qw::PackedWeight q4_k192 = qw::make_patterned_weight(QType::Q4_G64_FP16, 64, 192, 1U);
    const Weight w     = q4.device_weight(fake);
    auto* other        = static_cast<std::uint8_t*>(fake) + (std::size_t{1} << 24);
    auto* third        = static_cast<std::uint8_t*>(fake) + (std::size_t{1} << 25);
    auto* fourth       = static_cast<std::uint8_t*>(fake) + (std::size_t{1} << 26);
    const Tensor xq(other, DType::U8, {128, 8});
    const Tensor xs(third, DType::FP16, {4, 8});
    Tensor out(fourth, DType::BF16, {64, 8});

    struct Reject {
        const char* what;
        std::function<void()> call;
    };
    const Reject rejects[] = {
        {"a Q5 weight", [&] { ops::q4_a4_linear(xq, xs, q5.device_weight(fake), out, nullptr); }},
        {"K % 128 != 0", [&] {
             const Tensor xq192(other, DType::U8, {96, 8});
             const Tensor xs192(third, DType::FP16, {3, 8});
             ops::q4_a4_linear(xq192, xs192, q4_k192.device_weight(fake), out, nullptr);
         }},
        {"xq of the wrong width", [&] {
             const Tensor bad(other, DType::U8, {256, 8});
             ops::q4_a4_linear(bad, xs, w, out, nullptr);
         }},
        {"xq not U8", [&] {
             const Tensor bad(other, DType::I8, {128, 8});
             ops::q4_a4_linear(bad, xs, w, out, nullptr);
         }},
        {"xs of another token count", [&] {
             const Tensor bad(third, DType::FP16, {4, 9});
             ops::q4_a4_linear(xq, bad, w, out, nullptr);
         }},
        {"FP32 out", [&] {
             Tensor bad(fourth, DType::FP32, {64, 8});
             ops::q4_a4_linear(xq, xs, w, bad, nullptr);
         }},
        {"out of the wrong height", [&] {
             Tensor bad(fourth, DType::BF16, {65, 8});
             ops::q4_a4_linear(xq, xs, w, bad, nullptr);
         }},
        {"out aliasing xq", [&] {
             Tensor bad(other, DType::BF16, {64, 8});
             ops::q4_a4_linear(xq, xs, w, bad, nullptr);
         }},
        {"a null weight", [&] { ops::q4_a4_linear(xq, xs, q4.device_weight(nullptr), out, nullptr); }},
        {"quantize: FP32 x", [&] {
             const Tensor x(fake, DType::FP32, {256, 8});
             Tensor q = xq, s = xs;
             ops::quantize_a4_g64(x, q, s, nullptr);
         }},
        {"quantize: K % 64 != 0", [&] {
             const Tensor x(fake, DType::BF16, {96, 8});
             Tensor q(other, DType::U8, {48, 8});
             Tensor s(third, DType::FP16, {1, 8});
             ops::quantize_a4_g64(x, q, s, nullptr);
         }},
        {"quantize: xs of the wrong width", [&] {
             const Tensor x(fake, DType::BF16, {256, 8});
             Tensor q = xq;
             Tensor s(third, DType::FP16, {8, 8});
             ops::quantize_a4_g64(x, q, s, nullptr);
         }},
    };
    for (const Reject& r : rejects) {
        bool threw = false;
        try {
            r.call();
        } catch (const std::invalid_argument&) { threw = true; }
        if (!threw) {
            std::cerr << "validation: accepted " << r.what << '\n';
            ++failures;
        }
    }
    return failures;
}

} // namespace

int main() {
    try {
        int failures = run_validation();
        constexpr std::int32_t kGateUpN = 34816, kGateUpK = 5120;
        constexpr std::int32_t kDownN = 5120, kDownK = 17408;
        const Case cases[] = {
            // The registered shapes, both T, outlier-free fixture (cosine gated), timed at 4096.
            {kGateUpN, kGateUpK, 512, Fixture::Uniform, DType::BF16, DType::BF16, false,
             A8Route::GateUp, false},
            {kGateUpN, kGateUpK, 4096, Fixture::Uniform, DType::BF16, DType::BF16, false,
             A8Route::GateUp, true},
            {kDownN, kDownK, 512, Fixture::Uniform, DType::BF16, DType::BF16, true,
             A8Route::Down, false},
            {kDownN, kDownK, 4096, Fixture::Uniform, DType::BF16, DType::BF16, false,
             A8Route::Down, true},
            // The A8 route test's outlier fixture: cosine reported, oracle still exact.
            {kGateUpN, kGateUpK, 512, Fixture::Outlier, DType::BF16, DType::BF16, false,
             A8Route::GateUp, false},
            {kDownN, kDownK, 512, Fixture::Outlier, DType::BF16, DType::BF16, false,
             A8Route::Down, false},
            // Tails and formats.
            {200, 384, 77, Fixture::Outlier, DType::BF16, DType::BF16, true, A8Route::None, false},
            {203, 256, 1, Fixture::Uniform, DType::BF16, DType::BF16, false, A8Route::None, false},
            {130, 128, 129, Fixture::Uniform, DType::BF16, DType::FP16, false, A8Route::None,
             false},
            {256, 512, 300, Fixture::Uniform, DType::FP16, DType::BF16, false, A8Route::None,
             false},
        };
        for (const Case& c : cases) { failures += run_case(c); }
        std::cout << (failures == 0 ? "OK" : "FAIL") << " q4_a4_linear\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "q4_a4_linear test failed: " << error.what() << '\n';
        return 1;
    }
}
