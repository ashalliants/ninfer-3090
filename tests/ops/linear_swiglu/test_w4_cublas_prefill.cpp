// Correctness cover for the cuBLAS prefill route, reached through the public Ops with
// LinearPolicy::AllowPrefillCublas: linear_swiglu (Q4 gate_up), linear_add (Q5 down and the mixer
// output projection), and the attention and GDN split input projections.
//
// The oracle is independent of the route: an FP64 dot product over the exactly decoded group-64
// weight and the represented BF16 activation, not another kernel and not the integer route. What the
// route computes differs from it by design -- one int8 scale per weight row, one per token, and a
// channel equalisation between them -- so the pass bound is the fork's A8 activation allowance
// rather than a rounding bound, and the observed relative L2 is printed on every case so drift
// inside the allowance stays visible.
//
// What the bound cannot see, the sampling is built to: every destination range of the split
// projections (a row that lands in the wrong destination still produces plausible numbers in the
// others), token counts that are not a multiple of anything, and the tokens either side of each
// tile edge, since the epilogues index by `token_base`. The last case of each family sends the
// route one token past a tile so that a stale or short second tile shows up as a wrong value.
//
// Below kCublasPrefillMinTokens the route declines; those widths are checked too, because a policy
// that silently produced garbage for a narrow call would pass everything above.

#include "core/arena.h"
#include "core/weight.h"
#include "ninfer/ops/attn_input_proj.h"
#include "ninfer/ops/gdn_input_proj.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/linear_add.h"
#include "ninfer/ops/linear_swiglu.h"
#include "ops/linear_swiglu/q4cublas/w4_cublas_prefill.h"
#include "ops/op_check.h"
#include "ops/op_tester.h"
#include "ops/quantized_weight.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <exception>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

namespace {

using namespace ninfer;
using ninfer::test::ReductionCriterion;
using ninfer::test::ReductionStats;
using ninfer::test::compute_reduction_stats;
using ninfer::test::verify_reduction;
using ninfer::test::quantized_weight::PackedWeight;
namespace qw = ninfer::test::quantized_weight;

constexpr double kBf16UnitRoundoff = 1.0 / 256.0;
// The A8 activation-quantisation allowance tests/ops/linear/linear_test_common.cpp holds the
// integer-activation routes to. The route's weight side (one scale per row) adds to it rather than
// replacing it, and the equalisation keeps the sum inside it on this fixture.
constexpr double kAllowance = 0.04;
constexpr ReductionCriterion kCriterion{kAllowance, kBf16UnitRoundoff, 1.5 * kAllowance};

constexpr ops::LinearPolicy kPolicy = ops::LinearPolicy::AllowPrefillCublas;

// Roughly Gaussian hidden states with a few fixed channels an order of magnitude larger. Those
// outlier channels are what per-token activation scaling gets wrong and what the equalisation
// exists to absorb, so they belong in the fixture.
std::uint16_t to_bf16(float v) {
    std::uint32_t bits;
    std::memcpy(&bits, &v, sizeof(bits));
    bits += 0x7fffu + ((bits >> 16) & 1u); // round to nearest even, as the kernel reads it
    return static_cast<std::uint16_t>(bits >> 16);
}

std::vector<std::uint16_t> make_activation(std::int32_t rows, std::int32_t tokens,
                                           std::uint32_t seed, float scale = 1.0F) {
    std::vector<std::uint16_t> out(static_cast<std::size_t>(rows) * tokens);
    for (std::int32_t t = 0; t < tokens; ++t) {
        for (std::int32_t r = 0; r < rows; ++r) {
            const std::uint64_t h = qw::detail::mix64(
                (static_cast<std::uint64_t>(t) << 32) ^ static_cast<std::uint64_t>(r) ^ seed);
            float v = static_cast<float>(static_cast<std::int32_t>(h & 0xffffu) - 32768) / 32768.0F;
            if ((r % 431) == 7) { v *= 24.0F; }
            out[static_cast<std::size_t>(t) * rows + r] = to_bf16(v * scale);
        }
    }
    return out;
}

// Whether the route ran, judged from outside: it materialises the weight as int8 in the
// caller's workspace, so peak use of at least one weight matrix means it did, and its absence below
// the width gate means the call went to the integer or A16 route as documented. Accuracy alone
// cannot say, since an exact A16 fallback would pass every bound in this file.
int expect_route(const std::string& label, const WorkspaceArena& workspace,
                 std::size_t weight_bytes, std::int32_t tokens,
                 ops::LinearPolicy policy = kPolicy, std::int32_t gate = -1) {
    const bool engaged  = workspace.peak_used() >= weight_bytes;
    const bool expected =
        tokens >= (gate < 0 ? ops::cublas_prefill_min_tokens(policy) : gate);
    if (engaged == expected) { return 0; }
    std::cerr << label << ": cuBLAS route " << (engaged ? "ran" : "did not run") << " at T=" << tokens
              << ", expected it to " << (expected ? "run" : "decline") << " (peak workspace "
              << workspace.peak_used() << " bytes, one weight " << weight_bytes << ")\n";
    return 1;
}

float bf16_value(std::uint16_t bits) {
    const std::uint32_t u = static_cast<std::uint32_t>(bits) << 16;
    float f;
    std::memcpy(&f, &u, sizeof(f));
    return f;
}

std::vector<double> read_bf16(const test::GuardedDeviceBuffer& buffer, std::size_t elements) {
    std::vector<std::uint16_t> host(elements);
    buffer.copy_to_host(host.data(), elements * sizeof(std::uint16_t));
    std::vector<double> out(elements);
    for (std::size_t i = 0; i < elements; ++i) { out[i] = bf16_value(host[i]); }
    return out;
}

std::vector<float> token_column(const std::vector<std::uint16_t>& activation, std::int32_t rows,
                                std::int32_t token) {
    std::vector<float> column(static_cast<std::size_t>(rows));
    for (std::int32_t r = 0; r < rows; ++r) {
        column[static_cast<std::size_t>(r)] =
            bf16_value(activation[static_cast<std::size_t>(token) * rows + r]);
    }
    return column;
}

// Tokens to check: the ends, the tokens either side of every tile edge the route can use at the
// registered shapes (gate_up 1920, the split projections 5376, the down/out projections 13056),
// and a deterministic scatter. Edges past `tokens` drop out.
std::vector<std::int32_t> sample_tokens(std::int32_t tokens, std::uint32_t seed) {
    std::vector<std::int32_t> picks = {0,    1,     127,   128,   1919,  1920,  1921,  3839,
                                       3840, 3841,  5375,  5376,  5377,  13055, 13056, 13057};
    picks.push_back(tokens - 1);
    for (std::uint32_t i = 0; i < 12; ++i) {
        picks.push_back(static_cast<std::int32_t>(qw::detail::mix64(seed + i * 7919U) %
                                                  static_cast<std::uint64_t>(tokens)));
    }
    picks.erase(std::remove_if(picks.begin(), picks.end(),
                               [&](std::int32_t t) { return t < 0 || t >= tokens; }),
                picks.end());
    std::sort(picks.begin(), picks.end());
    picks.erase(std::unique(picks.begin(), picks.end()), picks.end());
    return picks;
}

std::int32_t pick_row(std::uint32_t salt, std::int32_t token, std::int32_t rows) {
    return static_cast<std::int32_t>(qw::detail::mix64(salt * 104729ULL + token * 65521ULL + 11U) %
                                     static_cast<std::uint64_t>(rows));
}

struct Samples {
    std::vector<double> actual;
    std::vector<double> reference;
};

int finish(const std::string& label, const Samples& samples) {
    const ReductionStats stats = compute_reduction_stats(
        samples.actual.data(), samples.reference.data(),
        static_cast<std::int64_t>(samples.actual.size()));
    std::cout << "  " << label << " relative_l2=" << stats.relative_l2 << " (allowance "
              << kAllowance << ", " << samples.actual.size() << " samples)\n";
    return verify_reduction(label, samples.actual, samples.reference, kCriterion);
}

// NVFP4 carries a tensor-wide divisor; the values tests/ops/linear_add/test_nvfp4.cpp uses.
qw::PatternedWeightOptions fixture_options(QType qtype) {
    qw::PatternedWeightOptions options;
    if (qtype == QType::NVFP4) {
        options.weight_scale_divisor = 0.125F;
        options.input_scale_divisor  = 3.5F;
    }
    return options;
}

struct DeviceWeight {
    PackedWeight host;
    test::GuardedDeviceBuffer device;
    Weight weight;
    DeviceWeight(QType qtype, std::int32_t rows, std::int32_t cols, std::uint32_t seed)
        : host(qw::make_patterned_weight(qtype, rows, cols, seed, fixture_options(qtype))), device(host.payload.size()) {
        device.copy_from_host(host.payload.data(), host.payload.size());
        weight = host.device_weight(device.data());
    }
};

int run_gate_up(std::int32_t tokens) {
    constexpr std::int32_t kRows = 34816;
    constexpr std::int32_t kCols = 5120;
    constexpr std::int32_t kOut  = kRows / 2;

    DeviceWeight w(QType::Q4_G64_FP16, kRows, kCols, 4801U);
    const std::vector<std::uint16_t> activation = make_activation(kCols, tokens, 91U);
    test::GuardedDeviceBuffer device_x(activation.size() * sizeof(std::uint16_t));
    device_x.copy_from_host(activation.data(), activation.size() * sizeof(std::uint16_t));
    const std::size_t out_elements = static_cast<std::size_t>(kOut) * tokens;
    test::GuardedDeviceBuffer output(out_elements * sizeof(std::uint16_t));
    output.fill(0xff);

    // Sized by the public query, so a route that takes more than it reports throws here.
    WorkspaceArena workspace(std::max<std::size_t>(
        ops::linear_swiglu_workspace_capacity_bytes(QType::Q4_G64_FP16, kRows, kCols, kPolicy,
                                                    tokens, tokens),
        256));
    Tensor x(device_x.data(), DType::BF16, {kCols, tokens});
    Tensor destination(output.data(), DType::BF16, {kOut, tokens});
    ops::linear_swiglu(x, w.weight, destination, kPolicy, workspace, nullptr);
    test::cuda_check(cudaDeviceSynchronize(), "synchronize cublas linear_swiglu");

    const std::string label = "LinearSwiGLU Q4 cuBLAS T=" + std::to_string(tokens);
    int failures            = output.verify_guards(label);
    failures += expect_route(label, workspace, static_cast<std::size_t>(kRows) * kCols, tokens);
    const std::vector<double> got = read_bf16(output, out_elements);
    Samples s;
    // Twelve rows per token rather than three. SwiGLU multiplies two quantised products, so this
    // family sits nearest its allowance (0.031-0.034 relative L2 on a fixed set of tokens, flat from
    // 129 to 4133 tokens), and with three rows a narrow call's ~45 samples scattered 0.032-0.045 by
    // which outputs happened to be drawn rather than by anything the route did.
    for (const std::int32_t token : sample_tokens(tokens, 3U)) {
        const std::vector<float> input = token_column(activation, kCols, token);
        for (std::uint32_t r = 0; r < 12; ++r) {
            const std::int32_t row = pick_row(r, token, kOut);
            const double gate      = qw::dot_fp64(w.host, row, input.data(), kCols);
            const double up        = qw::dot_fp64(w.host, kOut + row, input.data(), kCols);
            s.reference.push_back(gate / (1.0 + std::exp(-gate)) * up);
            s.actual.push_back(got[static_cast<std::size_t>(token) * kOut + row]);
        }
    }
    return failures + finish(label, s);
}

// kCols selects the registered profile: 17408 is mlp/down, 6144 the attention o_proj and GDN
// out_proj. Both go through one route, so both widths are held to the oracle.
int run_down(std::int32_t tokens, std::int32_t kCols) {
    constexpr std::int32_t kRows = 5120;

    DeviceWeight w(QType::Q5_G64_FP16, kRows, kCols, 5501U);
    const std::vector<std::uint16_t> activation = make_activation(kCols, tokens, 77U);
    // Scaled by a power of two, so exact, to put the residual on the order of the product: the
    // unscaled fixture leaves it a thousandth of the result, where a route that overwrote instead
    // of accumulating would still pass.
    const std::vector<std::uint16_t> residual0 = make_activation(kRows, tokens, 13U, 1024.0F);
    test::GuardedDeviceBuffer device_x(activation.size() * sizeof(std::uint16_t));
    device_x.copy_from_host(activation.data(), activation.size() * sizeof(std::uint16_t));
    const std::size_t res_elements = static_cast<std::size_t>(kRows) * tokens;
    test::GuardedDeviceBuffer device_residual(res_elements * sizeof(std::uint16_t));
    device_residual.copy_from_host(residual0.data(), res_elements * sizeof(std::uint16_t));

    WorkspaceArena workspace(std::max<std::size_t>(
        ops::linear_add_workspace_capacity_bytes(QType::Q5_G64_FP16, kRows, kCols, kPolicy,
                                                 tokens, tokens),
        256));
    Tensor x(device_x.data(), DType::BF16, {kCols, tokens});
    Tensor residual(device_residual.data(), DType::BF16, {kRows, tokens});
    ops::linear_add(x, w.weight, residual, kPolicy, workspace, nullptr);
    test::cuda_check(cudaDeviceSynchronize(), "synchronize cublas linear_add");

    const std::string label =
        "LinearAdd Q5 cuBLAS K=" + std::to_string(kCols) + " T=" + std::to_string(tokens);
    int failures            = device_residual.verify_guards(label);
    const std::vector<double> got = read_bf16(device_residual, res_elements);
    Samples s;
    double acc_sq = 0.0, base_sq = 0.0;
    for (const std::int32_t token : sample_tokens(tokens, 5U)) {
        const std::vector<float> input = token_column(activation, kCols, token);
        for (std::uint32_t r = 0; r < 3; ++r) {
            const std::int32_t row = pick_row(r, token, kRows);
            const double base =
                bf16_value(residual0[static_cast<std::size_t>(token) * kRows + row]);
            const double acc = qw::dot_fp64(w.host, row, input.data(), kCols);
            acc_sq += acc * acc;
            base_sq += base * base;
            // The add is in place, so a route that wrote instead of accumulated, or accumulated
            // twice, shows as a residual that is not `residual0 + acc`.
            s.reference.push_back(base + acc);
            s.actual.push_back(got[static_cast<std::size_t>(token) * kRows + row]);
        }
    }
    // The fixture is only sensitive to a route that wrote instead of accumulated, or accumulated
    // twice, if the residual is a real share of the result.
    if (base_sq < 0.1 * acc_sq) {
        std::cerr << label << ": fixture residual is negligible next to the product\n";
        ++failures;
    }
    failures += expect_route(label, workspace, static_cast<std::size_t>(kRows) * kCols, tokens);
    return failures + finish(label, s);
}

// Two parents over one activation, scattered into three destinations: qk -> qkv[0,4096), value_z
// rows [0,6144) -> qkv[4096,10240) (a destination offset into a wider tensor), rows [6144,12288)
// -> z. The destination row offset is the part nothing but this test exercises.
int run_gdn_input(std::int32_t tokens) {
    constexpr std::int32_t kHidden     = 5120;
    constexpr std::int32_t kQkRows     = 4096;
    constexpr std::int32_t kValueRows  = 6144;
    constexpr std::int32_t kZRows      = 6144;
    constexpr std::int32_t kQkvRows    = kQkRows + kValueRows;
    constexpr std::int32_t kParentRows = kValueRows + kZRows;

    DeviceWeight qk(QType::Q4_G64_FP16, kQkRows, kHidden, 909U);
    DeviceWeight vz(QType::Q5_G64_FP16, kParentRows, kHidden, 313U);
    const std::vector<std::uint16_t> activation = make_activation(kHidden, tokens, 41U);
    test::GuardedDeviceBuffer device_x(activation.size() * sizeof(std::uint16_t));
    device_x.copy_from_host(activation.data(), activation.size() * sizeof(std::uint16_t));
    test::GuardedDeviceBuffer device_qkv(static_cast<std::size_t>(kQkvRows) * tokens *
                                         sizeof(std::uint16_t));
    test::GuardedDeviceBuffer device_z(static_cast<std::size_t>(kZRows) * tokens *
                                       sizeof(std::uint16_t));
    device_qkv.fill(0xff);
    device_z.fill(0xff);

    WorkspaceArena workspace(std::max<std::size_t>(
        ops::gdn_input_proj_split_workspace_capacity_bytes(QType::Q4_G64_FP16, kQkRows,
                                                           QType::Q5_G64_FP16, kParentRows, kHidden,
                                                           kPolicy, tokens, tokens),
        256));
    Tensor x(device_x.data(), DType::BF16, {kHidden, tokens});
    Tensor qkv(device_qkv.data(), DType::BF16, {kQkvRows, tokens});
    Tensor z(device_z.data(), DType::BF16, {kZRows, tokens});
    ops::gdn_input_proj(x, qk.weight, vz.weight, qkv, z, kPolicy, workspace, nullptr);
    test::cuda_check(cudaDeviceSynchronize(), "synchronize cublas gdn input");

    const std::string label = "GdnInputProj cuBLAS T=" + std::to_string(tokens);
    int failures            = device_qkv.verify_guards(label) + device_z.verify_guards(label);
    failures += expect_route(label, workspace, static_cast<std::size_t>(kParentRows) * kHidden,
                             tokens);
    const std::vector<double> got_qkv =
        read_bf16(device_qkv, static_cast<std::size_t>(kQkvRows) * tokens);
    const std::vector<double> got_z = read_bf16(device_z, static_cast<std::size_t>(kZRows) * tokens);

    Samples s;
    for (const std::int32_t token : sample_tokens(tokens, 7U)) {
        const std::vector<float> input = token_column(activation, kHidden, token);
        const auto qkv_at = [&](std::int32_t row) {
            return got_qkv[static_cast<std::size_t>(token) * kQkvRows + row];
        };
        // First and last row of each destination range, then a scattered one: an off-by-a-range
        // mapping is caught at the edges, a wrong stride in the middle.
        const std::int32_t qk_rows[] = {0, kQkRows - 1, pick_row(0, token, kQkRows)};
        for (const std::int32_t row : qk_rows) {
            s.reference.push_back(qw::dot_fp64(qk.host, row, input.data(), kHidden));
            s.actual.push_back(qkv_at(row));
        }
        const std::int32_t value_rows[] = {0, kValueRows - 1, pick_row(1, token, kValueRows)};
        for (const std::int32_t row : value_rows) {
            s.reference.push_back(qw::dot_fp64(vz.host, row, input.data(), kHidden));
            s.actual.push_back(qkv_at(kQkRows + row));
        }
        const std::int32_t z_rows[] = {0, kZRows - 1, pick_row(2, token, kZRows)};
        for (const std::int32_t row : z_rows) {
            s.reference.push_back(qw::dot_fp64(vz.host, kValueRows + row, input.data(), kHidden));
            s.actual.push_back(got_z[static_cast<std::size_t>(token) * kZRows + row]);
        }
    }
    return failures + finish(label, s);
}

// Two parents over one activation, each split into a wide half and a 1024-row half, four
// destinations. The narrow halves start mid-parent (row 6144) and are the smallest destinations the
// route writes.
int run_attn_input(std::int32_t tokens) {
    constexpr std::int32_t kHidden     = 5120;
    constexpr std::int32_t kQRows      = 6144;
    constexpr std::int32_t kKvRows     = 1024;
    constexpr std::int32_t kParentRows = kQRows + kKvRows;

    DeviceWeight qk(QType::Q4_G64_FP16, kParentRows, kHidden, 4441U);
    DeviceWeight gv(QType::Q5_G64_FP16, kParentRows, kHidden, 977U);
    const std::vector<std::uint16_t> activation = make_activation(kHidden, tokens, 59U);
    test::GuardedDeviceBuffer device_x(activation.size() * sizeof(std::uint16_t));
    device_x.copy_from_host(activation.data(), activation.size() * sizeof(std::uint16_t));
    const std::size_t q_elements  = static_cast<std::size_t>(kQRows) * tokens;
    const std::size_t kv_elements = static_cast<std::size_t>(kKvRows) * tokens;
    test::GuardedDeviceBuffer device_q(q_elements * sizeof(std::uint16_t));
    test::GuardedDeviceBuffer device_gate(q_elements * sizeof(std::uint16_t));
    test::GuardedDeviceBuffer device_k(kv_elements * sizeof(std::uint16_t));
    test::GuardedDeviceBuffer device_v(kv_elements * sizeof(std::uint16_t));
    device_q.fill(0xff);
    device_gate.fill(0xff);
    device_k.fill(0xff);
    device_v.fill(0xff);

    WorkspaceArena workspace(std::max<std::size_t>(
        ops::attn_input_proj_split_workspace_capacity_bytes(QType::Q4_G64_FP16, kParentRows,
                                                            QType::Q5_G64_FP16, kParentRows,
                                                            kHidden, kPolicy, tokens, tokens),
        256));
    Tensor x(device_x.data(), DType::BF16, {kHidden, tokens});
    Tensor q(device_q.data(), DType::BF16, {kQRows, tokens});
    Tensor gate(device_gate.data(), DType::BF16, {kQRows, tokens});
    Tensor k(device_k.data(), DType::BF16, {kKvRows, tokens});
    Tensor v(device_v.data(), DType::BF16, {kKvRows, tokens});
    ops::attn_input_proj(x, qk.weight, gv.weight, q, gate, k, v, kPolicy, workspace, nullptr);
    test::cuda_check(cudaDeviceSynchronize(), "synchronize cublas attn input");

    const std::string label = "AttnInputProj cuBLAS T=" + std::to_string(tokens);
    int failures = device_q.verify_guards(label) + device_gate.verify_guards(label) +
                   device_k.verify_guards(label) + device_v.verify_guards(label);
    failures += expect_route(label, workspace, static_cast<std::size_t>(kParentRows) * kHidden,
                             tokens);
    const std::vector<double> got_q    = read_bf16(device_q, q_elements);
    const std::vector<double> got_gate = read_bf16(device_gate, q_elements);
    const std::vector<double> got_k    = read_bf16(device_k, kv_elements);
    const std::vector<double> got_v    = read_bf16(device_v, kv_elements);

    Samples s;
    for (const std::int32_t token : sample_tokens(tokens, 11U)) {
        const std::vector<float> input = token_column(activation, kHidden, token);
        const std::int32_t wide[]   = {0, kQRows - 1, pick_row(0, token, kQRows)};
        const std::int32_t narrow[] = {0, kKvRows - 1, pick_row(1, token, kKvRows)};
        for (const std::int32_t row : wide) {
            s.reference.push_back(qw::dot_fp64(qk.host, row, input.data(), kHidden));
            s.actual.push_back(got_q[static_cast<std::size_t>(token) * kQRows + row]);
            s.reference.push_back(qw::dot_fp64(gv.host, row, input.data(), kHidden));
            s.actual.push_back(got_gate[static_cast<std::size_t>(token) * kQRows + row]);
        }
        for (const std::int32_t row : narrow) {
            s.reference.push_back(qw::dot_fp64(qk.host, kQRows + row, input.data(), kHidden));
            s.actual.push_back(got_k[static_cast<std::size_t>(token) * kKvRows + row]);
            s.reference.push_back(qw::dot_fp64(gv.host, kQRows + row, input.data(), kHidden));
            s.actual.push_back(got_v[static_cast<std::size_t>(token) * kKvRows + row]);
        }
    }
    return failures + finish(label, s);
}

// sm_70's Qwen3.6-35B-A3B Q8 projections on the route's FP16 form (the route declines Q8 elsewhere,
// so these run only where it admits it). One parent each, split into the destinations the Q8
// kernels write: GDN [0,8192) -> qkv and [8192,12288) -> z; attention q, k, gate, v in parent order.
// The oracle is the same FP64 dot product; the FP16 route is far inside the A8 allowance.
std::string narrow_tag(ops::LinearPolicy policy) {
    return policy == ops::LinearPolicy::AllowPrefillCublasNarrow ? "narrow " : "";
}

int run_q8_gdn_input(std::int32_t tokens, ops::LinearPolicy policy = kPolicy) {
    constexpr std::int32_t kHidden = 2048, kQkvRows = 8192, kZRows = 4096;
    constexpr std::int32_t kParentRows = kQkvRows + kZRows;
    DeviceWeight w(QType::Q8_G32_FP16, kParentRows, kHidden, 7121U);
    const std::vector<std::uint16_t> activation = make_activation(kHidden, tokens, 43U);
    test::GuardedDeviceBuffer device_x(activation.size() * sizeof(std::uint16_t));
    device_x.copy_from_host(activation.data(), activation.size() * sizeof(std::uint16_t));
    test::GuardedDeviceBuffer device_qkv(static_cast<std::size_t>(kQkvRows) * tokens * 2);
    test::GuardedDeviceBuffer device_z(static_cast<std::size_t>(kZRows) * tokens * 2);
    device_qkv.fill(0xff);
    device_z.fill(0xff);
    WorkspaceArena workspace(std::max<std::size_t>(
        ops::gdn_input_proj_workspace_capacity_bytes(QType::Q8_G32_FP16, kParentRows, kHidden,
                                                     policy, tokens, tokens),
        256));
    Tensor x(device_x.data(), DType::BF16, {kHidden, tokens});
    Tensor qkv(device_qkv.data(), DType::BF16, {kQkvRows, tokens});
    Tensor z(device_z.data(), DType::BF16, {kZRows, tokens});
    ops::gdn_input_proj(x, w.weight, qkv, z, policy, workspace, nullptr);
    test::cuda_check(cudaDeviceSynchronize(), "synchronize cublas q8 gdn input");

    const std::string label =
        narrow_tag(policy) + "GdnInputProj Q8 cuBLAS T=" + std::to_string(tokens);
    int failures = device_qkv.verify_guards(label) + device_z.verify_guards(label);
    failures += expect_route(label, workspace, static_cast<std::size_t>(kParentRows) * kHidden * 2,
                             tokens, policy);
    const std::vector<double> got_qkv =
        read_bf16(device_qkv, static_cast<std::size_t>(kQkvRows) * tokens);
    const std::vector<double> got_z = read_bf16(device_z, static_cast<std::size_t>(kZRows) * tokens);
    Samples s;
    for (const std::int32_t token : sample_tokens(tokens, 13U)) {
        const std::vector<float> input = token_column(activation, kHidden, token);
        for (const std::int32_t row : {0, kQkvRows - 1, pick_row(0, token, kQkvRows)}) {
            s.reference.push_back(qw::dot_fp64(w.host, row, input.data(), kHidden));
            s.actual.push_back(got_qkv[static_cast<std::size_t>(token) * kQkvRows + row]);
        }
        for (const std::int32_t row : {0, kZRows - 1, pick_row(1, token, kZRows)}) {
            s.reference.push_back(qw::dot_fp64(w.host, kQkvRows + row, input.data(), kHidden));
            s.actual.push_back(got_z[static_cast<std::size_t>(token) * kZRows + row]);
        }
    }
    return failures + finish(label, s);
}

int run_q8_attn_input(std::int32_t tokens, ops::LinearPolicy policy = kPolicy) {
    constexpr std::int32_t kHidden = 2048, kQRows = 4096, kKvRows = 512;
    constexpr std::int32_t kParentRows = 2 * kQRows + 2 * kKvRows;
    DeviceWeight w(QType::Q8_G32_FP16, kParentRows, kHidden, 6007U);
    const std::vector<std::uint16_t> activation = make_activation(kHidden, tokens, 61U);
    test::GuardedDeviceBuffer device_x(activation.size() * sizeof(std::uint16_t));
    device_x.copy_from_host(activation.data(), activation.size() * sizeof(std::uint16_t));
    const std::size_t q_elements  = static_cast<std::size_t>(kQRows) * tokens;
    const std::size_t kv_elements = static_cast<std::size_t>(kKvRows) * tokens;
    test::GuardedDeviceBuffer device_q(q_elements * 2), device_gate(q_elements * 2);
    test::GuardedDeviceBuffer device_k(kv_elements * 2), device_v(kv_elements * 2);
    device_q.fill(0xff);
    device_gate.fill(0xff);
    device_k.fill(0xff);
    device_v.fill(0xff);
    WorkspaceArena workspace(std::max<std::size_t>(
        ops::attn_input_proj_workspace_capacity_bytes(QType::Q8_G32_FP16, kParentRows, kHidden,
                                                      policy, tokens, tokens),
        256));
    Tensor x(device_x.data(), DType::BF16, {kHidden, tokens});
    Tensor q(device_q.data(), DType::BF16, {kQRows, tokens});
    Tensor gate(device_gate.data(), DType::BF16, {kQRows, tokens});
    Tensor k(device_k.data(), DType::BF16, {kKvRows, tokens});
    Tensor v(device_v.data(), DType::BF16, {kKvRows, tokens});
    ops::attn_input_proj(x, w.weight, q, gate, k, v, policy, workspace, nullptr);
    test::cuda_check(cudaDeviceSynchronize(), "synchronize cublas q8 attn input");

    const std::string label =
        narrow_tag(policy) + "AttnInputProj Q8 cuBLAS T=" + std::to_string(tokens);
    int failures = device_q.verify_guards(label) + device_gate.verify_guards(label) +
                   device_k.verify_guards(label) + device_v.verify_guards(label);
    failures += expect_route(label, workspace, static_cast<std::size_t>(kParentRows) * kHidden * 2,
                             tokens, policy);
    const std::vector<double> got_q    = read_bf16(device_q, q_elements);
    const std::vector<double> got_gate = read_bf16(device_gate, q_elements);
    const std::vector<double> got_k    = read_bf16(device_k, kv_elements);
    const std::vector<double> got_v    = read_bf16(device_v, kv_elements);
    Samples s;
    for (const std::int32_t token : sample_tokens(tokens, 17U)) {
        const std::vector<float> input = token_column(activation, kHidden, token);
        for (const std::int32_t row : {0, kQRows - 1, pick_row(0, token, kQRows)}) {
            s.reference.push_back(qw::dot_fp64(w.host, row, input.data(), kHidden));
            s.actual.push_back(got_q[static_cast<std::size_t>(token) * kQRows + row]);
            s.reference.push_back(
                qw::dot_fp64(w.host, kQRows + kKvRows + row, input.data(), kHidden));
            s.actual.push_back(got_gate[static_cast<std::size_t>(token) * kQRows + row]);
        }
        for (const std::int32_t row : {0, kKvRows - 1, pick_row(1, token, kKvRows)}) {
            s.reference.push_back(qw::dot_fp64(w.host, kQRows + row, input.data(), kHidden));
            s.actual.push_back(got_k[static_cast<std::size_t>(token) * kKvRows + row]);
            s.reference.push_back(
                qw::dot_fp64(w.host, 2 * kQRows + kKvRows + row, input.data(), kHidden));
            s.actual.push_back(got_v[static_cast<std::size_t>(token) * kKvRows + row]);
        }
    }
    return failures + finish(label, s);
}

int run_q8_out(std::int32_t tokens, ops::LinearPolicy policy = kPolicy) {
    constexpr std::int32_t kRows = 2048, kCols = 4096;
    DeviceWeight w(QType::Q8_G32_FP16, kRows, kCols, 3301U);
    const std::vector<std::uint16_t> activation = make_activation(kCols, tokens, 83U);
    const std::vector<std::uint16_t> residual0  = make_activation(kRows, tokens, 19U, 64.0F);
    test::GuardedDeviceBuffer device_x(activation.size() * sizeof(std::uint16_t));
    device_x.copy_from_host(activation.data(), activation.size() * sizeof(std::uint16_t));
    const std::size_t res_elements = static_cast<std::size_t>(kRows) * tokens;
    test::GuardedDeviceBuffer device_residual(res_elements * sizeof(std::uint16_t));
    device_residual.copy_from_host(residual0.data(), res_elements * sizeof(std::uint16_t));
    WorkspaceArena workspace(std::max<std::size_t>(
        ops::linear_add_workspace_capacity_bytes(QType::Q8_G32_FP16, kRows, kCols, policy, tokens,
                                                 tokens),
        256));
    Tensor x(device_x.data(), DType::BF16, {kCols, tokens});
    Tensor residual(device_residual.data(), DType::BF16, {kRows, tokens});
    ops::linear_add(x, w.weight, residual, policy, workspace, nullptr);
    test::cuda_check(cudaDeviceSynchronize(), "synchronize cublas q8 linear_add");

    const std::string label =
        narrow_tag(policy) + "LinearAdd Q8 cuBLAS T=" + std::to_string(tokens);
    int failures = device_residual.verify_guards(label);
    failures +=
        expect_route(label, workspace, static_cast<std::size_t>(kRows) * kCols * 2, tokens, policy);
    const std::vector<double> got = read_bf16(device_residual, res_elements);
    Samples s;
    for (const std::int32_t token : sample_tokens(tokens, 23U)) {
        const std::vector<float> input = token_column(activation, kCols, token);
        for (std::uint32_t r = 0; r < 3; ++r) {
            const std::int32_t row = pick_row(r, token, kRows);
            s.reference.push_back(
                bf16_value(residual0[static_cast<std::size_t>(token) * kRows + row]) +
                qw::dot_fp64(w.host, row, input.data(), kCols));
            s.actual.push_back(got[static_cast<std::size_t>(token) * kRows + row]);
        }
    }
    return failures + finish(label, s);
}

// The FP16 route's NVFP4 and FP8 forms (sm_70; detail::cublas_prefill_admits_fp), at the shapes
// the NVIDIA QAD 27B uses them: NVFP4 MLP, GDN in/out projections; FP8 attention q/k/v/o and the
// remaining GDN projections. The weight is materialised as FP16, so the route is judged engaged at
// two bytes per weight.
std::string fp_tag(QType qtype, ops::LinearPolicy policy) {
    return narrow_tag(policy) + (qtype == QType::NVFP4 ? "NVFP4" : "FP8");
}

int run_fp_swiglu(QType qtype, std::int32_t tokens, ops::LinearPolicy policy) {
    constexpr std::int32_t kRows = 34816, kCols = 5120, kOut = kRows / 2;
    DeviceWeight w(qtype, kRows, kCols, 8111U);
    const std::vector<std::uint16_t> activation = make_activation(kCols, tokens, 97U);
    test::GuardedDeviceBuffer device_x(activation.size() * sizeof(std::uint16_t));
    device_x.copy_from_host(activation.data(), activation.size() * sizeof(std::uint16_t));
    const std::size_t out_elements = static_cast<std::size_t>(kOut) * tokens;
    test::GuardedDeviceBuffer output(out_elements * sizeof(std::uint16_t));
    output.fill(0xff);
    WorkspaceArena workspace(std::max<std::size_t>(
        ops::linear_swiglu_workspace_capacity_bytes(qtype, kRows, kCols, policy, tokens, tokens),
        256));
    Tensor x(device_x.data(), DType::BF16, {kCols, tokens});
    Tensor destination(output.data(), DType::BF16, {kOut, tokens});
    ops::linear_swiglu(x, w.weight, destination, policy, workspace, nullptr);
    test::cuda_check(cudaDeviceSynchronize(), "synchronize cublas fp linear_swiglu");

    const std::string label = "LinearSwiGLU " + fp_tag(qtype, policy) +
                              " cuBLAS T=" + std::to_string(tokens);
    int failures = output.verify_guards(label);
    failures += expect_route(label, workspace, static_cast<std::size_t>(kRows) * kCols * 2, tokens,
                             policy, ops::cublas_prefill_min_tokens(policy));
    const std::vector<double> got = read_bf16(output, out_elements);
    Samples s;
    for (const std::int32_t token : sample_tokens(tokens, 29U)) {
        const std::vector<float> input = token_column(activation, kCols, token);
        for (std::uint32_t r = 0; r < 6; ++r) {
            const std::int32_t row = pick_row(r, token, kOut);
            const double gate      = qw::dot_fp64(w.host, row, input.data(), kCols);
            const double up        = qw::dot_fp64(w.host, kOut + row, input.data(), kCols);
            s.reference.push_back(gate / (1.0 + std::exp(-gate)) * up);
            s.actual.push_back(got[static_cast<std::size_t>(token) * kOut + row]);
        }
    }
    return failures + finish(label, s);
}

int run_fp_add(QType qtype, std::int32_t tokens, std::int32_t kCols, ops::LinearPolicy policy) {
    constexpr std::int32_t kRows = 5120;
    DeviceWeight w(qtype, kRows, kCols, 9203U);
    const std::vector<std::uint16_t> activation = make_activation(kCols, tokens, 71U);
    const std::vector<std::uint16_t> residual0  = make_activation(kRows, tokens, 31U, 64.0F);
    test::GuardedDeviceBuffer device_x(activation.size() * sizeof(std::uint16_t));
    device_x.copy_from_host(activation.data(), activation.size() * sizeof(std::uint16_t));
    const std::size_t res_elements = static_cast<std::size_t>(kRows) * tokens;
    test::GuardedDeviceBuffer device_residual(res_elements * sizeof(std::uint16_t));
    device_residual.copy_from_host(residual0.data(), res_elements * sizeof(std::uint16_t));
    WorkspaceArena workspace(std::max<std::size_t>(
        ops::linear_add_workspace_capacity_bytes(qtype, kRows, kCols, policy, tokens, tokens),
        256));
    Tensor x(device_x.data(), DType::BF16, {kCols, tokens});
    Tensor residual(device_residual.data(), DType::BF16, {kRows, tokens});
    ops::linear_add(x, w.weight, residual, policy, workspace, nullptr);
    test::cuda_check(cudaDeviceSynchronize(), "synchronize cublas fp linear_add");

    const std::string label = "LinearAdd " + fp_tag(qtype, policy) +
                              " cuBLAS K=" + std::to_string(kCols) + " T=" + std::to_string(tokens);
    int failures = device_residual.verify_guards(label);
    failures += expect_route(label, workspace, static_cast<std::size_t>(kRows) * kCols * 2, tokens,
                             policy, ops::cublas_prefill_min_tokens(policy));
    const std::vector<double> got = read_bf16(device_residual, res_elements);
    Samples s;
    for (const std::int32_t token : sample_tokens(tokens, 37U)) {
        const std::vector<float> input = token_column(activation, kCols, token);
        for (std::uint32_t r = 0; r < 3; ++r) {
            const std::int32_t row = pick_row(r, token, kRows);
            s.reference.push_back(
                bf16_value(residual0[static_cast<std::size_t>(token) * kRows + row]) +
                qw::dot_fp64(w.host, row, input.data(), kCols));
            s.actual.push_back(got[static_cast<std::size_t>(token) * kRows + row]);
        }
    }
    return failures + finish(label, s);
}

int run_fp_gdn_input(QType qtype, std::int32_t tokens, ops::LinearPolicy policy) {
    constexpr std::int32_t kHidden = 5120, kQkvRows = 10240, kZRows = 6144;
    constexpr std::int32_t kParentRows = kQkvRows + kZRows;
    DeviceWeight w(qtype, kParentRows, kHidden, 7333U);
    const std::vector<std::uint16_t> activation = make_activation(kHidden, tokens, 47U);
    test::GuardedDeviceBuffer device_x(activation.size() * sizeof(std::uint16_t));
    device_x.copy_from_host(activation.data(), activation.size() * sizeof(std::uint16_t));
    test::GuardedDeviceBuffer device_qkv(static_cast<std::size_t>(kQkvRows) * tokens * 2);
    test::GuardedDeviceBuffer device_z(static_cast<std::size_t>(kZRows) * tokens * 2);
    device_qkv.fill(0xff);
    device_z.fill(0xff);
    WorkspaceArena workspace(std::max<std::size_t>(
        ops::gdn_input_proj_workspace_capacity_bytes(qtype, kParentRows, kHidden, policy, tokens,
                                                     tokens),
        256));
    Tensor x(device_x.data(), DType::BF16, {kHidden, tokens});
    Tensor qkv(device_qkv.data(), DType::BF16, {kQkvRows, tokens});
    Tensor z(device_z.data(), DType::BF16, {kZRows, tokens});
    ops::gdn_input_proj(x, w.weight, qkv, z, policy, workspace, nullptr);
    test::cuda_check(cudaDeviceSynchronize(), "synchronize cublas fp gdn input");

    const std::string label =
        "GdnInputProj " + fp_tag(qtype, policy) + " cuBLAS T=" + std::to_string(tokens);
    int failures = device_qkv.verify_guards(label) + device_z.verify_guards(label);
    failures += expect_route(label, workspace, static_cast<std::size_t>(kParentRows) * kHidden * 2,
                             tokens, policy,
                             ops::cublas_prefill_min_tokens(policy));
    const std::vector<double> got_qkv =
        read_bf16(device_qkv, static_cast<std::size_t>(kQkvRows) * tokens);
    const std::vector<double> got_z = read_bf16(device_z, static_cast<std::size_t>(kZRows) * tokens);
    Samples s;
    for (const std::int32_t token : sample_tokens(tokens, 41U)) {
        const std::vector<float> input = token_column(activation, kHidden, token);
        for (const std::int32_t row : {0, kQkvRows - 1, pick_row(0, token, kQkvRows)}) {
            s.reference.push_back(qw::dot_fp64(w.host, row, input.data(), kHidden));
            s.actual.push_back(got_qkv[static_cast<std::size_t>(token) * kQkvRows + row]);
        }
        for (const std::int32_t row : {0, kZRows - 1, pick_row(1, token, kZRows)}) {
            s.reference.push_back(qw::dot_fp64(w.host, kQkvRows + row, input.data(), kHidden));
            s.actual.push_back(got_z[static_cast<std::size_t>(token) * kZRows + row]);
        }
    }
    return failures + finish(label, s);
}

int run_fp_attn_input(QType qtype, std::int32_t tokens, ops::LinearPolicy policy) {
    constexpr std::int32_t kHidden = 5120, kQRows = 6144, kKvRows = 1024;
    constexpr std::int32_t kParentRows = 2 * kQRows + 2 * kKvRows;
    DeviceWeight w(qtype, kParentRows, kHidden, 6619U);
    const std::vector<std::uint16_t> activation = make_activation(kHidden, tokens, 53U);
    test::GuardedDeviceBuffer device_x(activation.size() * sizeof(std::uint16_t));
    device_x.copy_from_host(activation.data(), activation.size() * sizeof(std::uint16_t));
    const std::size_t q_elements  = static_cast<std::size_t>(kQRows) * tokens;
    const std::size_t kv_elements = static_cast<std::size_t>(kKvRows) * tokens;
    test::GuardedDeviceBuffer device_q(q_elements * 2), device_gate(q_elements * 2);
    test::GuardedDeviceBuffer device_k(kv_elements * 2), device_v(kv_elements * 2);
    device_q.fill(0xff);
    device_gate.fill(0xff);
    device_k.fill(0xff);
    device_v.fill(0xff);
    WorkspaceArena workspace(std::max<std::size_t>(
        ops::attn_input_proj_workspace_capacity_bytes(qtype, kParentRows, kHidden, policy, tokens,
                                                      tokens),
        256));
    Tensor x(device_x.data(), DType::BF16, {kHidden, tokens});
    Tensor q(device_q.data(), DType::BF16, {kQRows, tokens});
    Tensor gate(device_gate.data(), DType::BF16, {kQRows, tokens});
    Tensor k(device_k.data(), DType::BF16, {kKvRows, tokens});
    Tensor v(device_v.data(), DType::BF16, {kKvRows, tokens});
    ops::attn_input_proj(x, w.weight, q, gate, k, v, policy, workspace, nullptr);
    test::cuda_check(cudaDeviceSynchronize(), "synchronize cublas fp attn input");

    const std::string label =
        "AttnInputProj " + fp_tag(qtype, policy) + " cuBLAS T=" + std::to_string(tokens);
    int failures = device_q.verify_guards(label) + device_gate.verify_guards(label) +
                   device_k.verify_guards(label) + device_v.verify_guards(label);
    failures += expect_route(label, workspace, static_cast<std::size_t>(kParentRows) * kHidden * 2,
                             tokens, policy,
                             ops::cublas_prefill_min_tokens(policy));
    const std::vector<double> got_q    = read_bf16(device_q, q_elements);
    const std::vector<double> got_gate = read_bf16(device_gate, q_elements);
    const std::vector<double> got_k    = read_bf16(device_k, kv_elements);
    const std::vector<double> got_v    = read_bf16(device_v, kv_elements);
    Samples s;
    // Parent rows are q, k, gate, v.
    for (const std::int32_t token : sample_tokens(tokens, 43U)) {
        const std::vector<float> input = token_column(activation, kHidden, token);
        for (const std::int32_t row : {0, kQRows - 1, pick_row(0, token, kQRows)}) {
            s.reference.push_back(qw::dot_fp64(w.host, row, input.data(), kHidden));
            s.actual.push_back(got_q[static_cast<std::size_t>(token) * kQRows + row]);
            s.reference.push_back(
                qw::dot_fp64(w.host, kQRows + kKvRows + row, input.data(), kHidden));
            s.actual.push_back(got_gate[static_cast<std::size_t>(token) * kQRows + row]);
        }
        for (const std::int32_t row : {0, kKvRows - 1, pick_row(1, token, kKvRows)}) {
            s.reference.push_back(qw::dot_fp64(w.host, kQRows + row, input.data(), kHidden));
            s.actual.push_back(got_k[static_cast<std::size_t>(token) * kKvRows + row]);
            s.reference.push_back(
                qw::dot_fp64(w.host, 2 * kQRows + kKvRows + row, input.data(), kHidden));
            s.actual.push_back(got_v[static_cast<std::size_t>(token) * kKvRows + row]);
        }
    }
    return failures + finish(label, s);
}

} // namespace

int main() {
    try {
        int failures = 0;
        // Either side of the width gate: one below must resolve to a route that still answers
        // correctly, the gate itself and one past it to the route. 1000 and 1921 are unaligned, and
        // 1921 puts one token in a second gate_up tile.
        constexpr std::int32_t gate = ops::kCublasPrefillMinTokens;
        for (const std::int32_t tokens : {gate - 1, gate, gate + 1, 1000, 1921}) {
            failures += run_gate_up(tokens);
            failures += run_down(tokens, 17408);
            failures += run_down(tokens, 6144);
        }
        // Three tiles at gate_up's 1920: two full and a partial. The larger shapes need far more
        // tokens to tile, so they get their own.
        failures += run_gate_up(4133);
        failures += run_down(13057, 6144);
        for (const std::int32_t tokens : {gate - 1, gate, gate + 1, 1000, 5377}) {
            failures += run_gdn_input(tokens);
            failures += run_attn_input(tokens);
        }
        // The FP16 route's Q8 form (sm_70 builds only; the route declines Q8 elsewhere).
        if (ops::detail::cublas_prefill_admits_q8()) {
            for (const std::int32_t tokens : {gate - 1, gate, gate + 1, 1000, 5377}) {
                failures += run_q8_gdn_input(tokens);
                failures += run_q8_attn_input(tokens);
                failures += run_q8_out(tokens);
            }
            // Production prefill calls carry prefill_phase_policy's narrow form, whose sm_70 gate
            // is lower (65): the route and its scratch must engage from there, not from 129.
            constexpr ops::LinearPolicy narrow = ops::prefill_phase_policy(kPolicy);
            constexpr std::int32_t narrow_gate = ops::cublas_prefill_min_tokens(narrow);
            for (const std::int32_t tokens : {narrow_gate - 1, narrow_gate, gate - 1, gate}) {
                failures += run_q8_gdn_input(tokens, narrow);
                failures += run_q8_attn_input(tokens, narrow);
                failures += run_q8_out(tokens, narrow);
            }
        }
        // The FP16 route's NVFP4 / FP8 forms (sm_70 builds only), under both prefill policies:
        // below each gate the formats' own plans must still answer.
        if (ops::detail::cublas_prefill_admits_fp()) {
            constexpr ops::LinearPolicy narrow = ops::prefill_phase_policy(kPolicy);
            // Below the narrow gate the formats' streaming tiles answer (16 is a tile width).
            const std::int32_t wide_gate   = ops::cublas_prefill_min_tokens(kPolicy);
            const std::int32_t narrow_gate = ops::cublas_prefill_min_tokens(narrow);
            const std::pair<ops::LinearPolicy, std::vector<std::int32_t>> sweeps[] = {
                {kPolicy, {wide_gate - 1, wide_gate, 1000}},
                {narrow, {16, narrow_gate - 1, narrow_gate}}};
            for (const auto& [policy, widths] : sweeps) {
                for (const std::int32_t tokens : widths) {
                    failures += run_fp_swiglu(QType::NVFP4, tokens, policy);
                    failures += run_fp_add(QType::NVFP4, tokens, 17408, policy);
                    failures += run_fp_add(QType::NVFP4, tokens, 6144, policy);
                    failures += run_fp_add(QType::FP8_E4M3FN_ROW_BF16, tokens, 6144, policy);
                    failures += run_fp_gdn_input(QType::NVFP4, tokens, policy);
                    failures += run_fp_gdn_input(QType::FP8_E4M3FN_ROW_BF16, tokens, policy);
                    failures += run_fp_attn_input(QType::FP8_E4M3FN_ROW_BF16, tokens, policy);
                }
            }
        }
        std::cout << (failures == 0 ? "OK" : "FAIL") << " cuBLAS prefill route correctness\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "cuBLAS prefill route test failed: " << error.what() << '\n';
        return 1;
    }
}
