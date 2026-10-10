// Linear and LinearAdd over GGML block-format weights (ggml_iq4_xs, ggml_iq3_s, ggml_q6_k,
// ggml_iq4_nl, ggml_q8_0, ggml_q2_0) at every registered Qwen3.8-Flash-Next problem.
//
// Oracle: the weight is decoded by the exact host decoder (ops/ggml_blocks_decode.h, itself
// qualified bit-for-bit against ggml b11316), the activation passes through an independent host
// copy of the contract's Q8_1 cast, and every output is the naive FP64 dot product of the two
// (plus the BF16 residual for LinearAdd). The cast is a semantic boundary of these formats
// (linear.h), so it belongs to the oracle; it is qualified exactly on its own below, both directly
// (the private wide-route cast, byte for byte) and through the public Op on both routes (a one-hot
// Q8_0 probe that reads every q * d back out).
//
// T covers both routes and their seams: 1..8 (decode), 9 (first wide), 63/64/65 (one wide column
// tile and its edges) and 300 (interior, partial last tile).
//
// Usage: ninfer_linear_ggml_test [--format NAME] [--shape N K]

#include "ninfer/ops/linear.h"
#include "ninfer/ops/linear_add.h"

#include "core/arena.h"
#include "core/weight.h"
#include "ops/ggml_blocks_decode.h"
#include "ops/linear/ggml/ggml_launch.h"
#include "ops/linear/ggml/ggml_shapes.h"
#include "ops/op_check.h"
#include "ops/op_tester.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

namespace {

using namespace ninfer;
using ninfer::ops::LinearPolicy;
using ninfer::test::bf16_to_f32;
using ninfer::test::cuda_check;
using ninfer::test::f32_to_bf16;
using ninfer::test::ReductionCriterion;

// One arithmetic profile for both routes: exact int32 products of int8 codes per 32 values, FP32
// scaling and accumulation, BF16 output. Against an oracle that already contains the Q8_1 cast this
// is A16-class arithmetic: the relative-L2 allowance is one BF16 unit roundoff and the gross bound
// is the house two-step BF16 floor.
constexpr ReductionCriterion kGgmlQ81Criterion{1.0 / 256.0, 0.0, test::kBf16GrossRelativeFloor};

constexpr std::int32_t kTokens[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 63, 64, 65, 300};
constexpr std::int32_t kMaxTokens = 300;

const char* format_name(QType format) {
    switch (format) {
    case QType::GGML_IQ4_XS: return "ggml_iq4_xs";
    case QType::GGML_IQ3_S: return "ggml_iq3_s";
    case QType::GGML_Q6_K: return "ggml_q6_k";
    case QType::GGML_IQ4_NL: return "ggml_iq4_nl";
    case QType::GGML_Q8_0: return "ggml_q8_0";
    case QType::GGML_Q2_0: return "ggml_q2_0";
    default: return "?";
    }
}

// --- the contract's Q8_1 cast, host side ---------------------------------------------------------

struct Q81 {
    std::vector<std::int8_t> q; // [T][K]
    std::vector<float> d;       // [T][K / 32]
};

Q81 cast_q81(const std::vector<std::uint16_t>& x, std::int32_t k, std::int32_t t) {
    Q81 out;
    out.q.resize(static_cast<std::size_t>(k) * t);
    out.d.resize(static_cast<std::size_t>(k / 32) * t);
    for (std::size_t g = 0; g < out.d.size(); ++g) {
        float amax = 0.0F;
        for (int i = 0; i < 32; ++i) amax = std::max(amax, std::fabs(bf16_to_f32(x[32 * g + i])));
        const float inverse = amax == 0.0F ? 0.0F : 127.0F / amax;
        for (int i = 0; i < 32; ++i) {
            // Default rounding mode: nearest, ties to even.
            out.q[32 * g + i] =
                static_cast<std::int8_t>(std::nearbyint(bf16_to_f32(x[32 * g + i]) * inverse));
        }
        out.d[g] = amax / 127.0F;
    }
    return out;
}

// --- fixtures --------------------------------------------------------------------------------------

std::uint16_t fp16_bits(bool negative, int exponent, int mantissa) {
    return static_cast<std::uint16_t>((negative ? 0x8000 : 0) | ((exponent + 15) << 10) |
                                      (mantissa & 0x3FF));
}

// Random code bytes (every byte pattern is a valid code in these formats) with finite binary16
// scales of either sign.
std::vector<std::uint8_t> make_blocks(QType format, std::int32_t n, std::int32_t k,
                                      std::uint32_t seed) {
    const GgmlBlock block = ggml_block(format);
    const std::size_t blocks = static_cast<std::size_t>(n) * (k / block.values);
    std::vector<std::uint8_t> bytes(blocks * block.bytes);
    std::mt19937 rng(seed);
    for (auto& b : bytes) b = static_cast<std::uint8_t>(rng());
    const std::size_t d_offset = format == QType::GGML_Q6_K ? 208 : 0;
    // IQ4_XS and IQ3_S scale codes multiply d by up to ~32; Q6_K by up to 128 and every code by 32.
    const int base = format == QType::GGML_Q8_0 || format == QType::GGML_IQ4_NL ? -12 : -14;
    for (std::size_t i = 0; i < blocks; ++i) {
        const auto h = fp16_bits(rng() & 1, base + static_cast<int>(rng() % 4),
                                 static_cast<int>(rng() & 0x3FF));
        std::memcpy(&bytes[i * block.bytes + d_offset], &h, 2);
    }
    return bytes;
}

// BF16 [K,T]; each 32-value group gets its own magnitude, and the first group of every column is
// all zero so the cast's zero-group rule is exercised.
std::vector<std::uint16_t> make_activation(std::int32_t k, std::int32_t t, std::uint32_t seed) {
    std::vector<std::uint16_t> x(static_cast<std::size_t>(k) * t);
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> unit(-0.5F, 0.5F);
    for (std::int32_t col = 0; col < t; ++col) {
        for (std::int32_t g = 0; g < k / 32; ++g) {
            const float scale = std::ldexp(1.0F, static_cast<int>(rng() % 9) - 6);
            for (int i = 0; i < 32; ++i) {
                const float v = g == 0 ? 0.0F : unit(rng) * scale;
                x[static_cast<std::size_t>(col) * k + 32 * g + i] = f32_to_bf16(v);
            }
        }
    }
    return x;
}

// out[t][n] = sum_k w[n][k] * q[t][k] * d[t][k / 32] (+ residual), naive FP64.
std::vector<double> oracle(const std::vector<float>& w, const Q81& x, std::int32_t n,
                           std::int32_t k, std::int32_t t,
                           const std::vector<std::uint16_t>* residual) {
    std::vector<double> xr(static_cast<std::size_t>(k) * t);
    for (std::size_t i = 0; i < xr.size(); ++i) {
        xr[i] = static_cast<double>(x.q[i]) * static_cast<double>(x.d[i / 32]);
    }
    std::vector<double> out(static_cast<std::size_t>(n) * t);
    const unsigned workers = std::max(1U, std::min(16U, std::thread::hardware_concurrency()));
    std::vector<std::thread> pool;
    for (unsigned id = 0; id < workers; ++id) {
        pool.emplace_back([&, id] {
            for (std::int32_t row = static_cast<std::int32_t>(id); row < n;
                 row += static_cast<std::int32_t>(workers)) {
                const float* wr = w.data() + static_cast<std::size_t>(row) * k;
                for (std::int32_t col = 0; col < t; ++col) {
                    const double* xc = xr.data() + static_cast<std::size_t>(col) * k;
                    double sum       = 0.0;
                    for (std::int32_t i = 0; i < k; ++i) sum += static_cast<double>(wr[i]) * xc[i];
                    const std::size_t o = static_cast<std::size_t>(col) * n + row;
                    if (residual) sum += static_cast<double>(bf16_to_f32((*residual)[o]));
                    out[o] = sum;
                }
            }
        });
    }
    for (auto& worker : pool) worker.join();
    return out;
}

Weight device_weight(QType format, std::int32_t n, std::int32_t k, const DeviceBuffer& storage) {
    const GgmlBlock block = ggml_block(format);
    Weight w;
    w.payload       = storage.p;
    w.payload_bytes = static_cast<std::uint64_t>(n) * (k / block.values) * block.bytes;
    w.qtype         = format;
    w.layout        = QuantLayout::GgmlBlocks;
    w.ndim          = 2;
    w.n = w.shape[0] = w.padded_shape[0] = n;
    w.k = w.shape[1] = w.padded_shape[1] = k;
    w.qdata                              = storage.p;
    w.group = static_cast<std::int32_t>(block.values);
    w.group_size                         = block.values;
    return w;
}

std::vector<std::uint16_t> download_bf16(const void* device, std::size_t count) {
    std::vector<std::uint16_t> out(count);
    cuda_check(cudaMemcpy(out.data(), device, count * 2, cudaMemcpyDeviceToHost), "download");
    return out;
}

// Runs every T on one (format, shape, op) and compares against the FP64 oracle.
int run_problem(QType format, std::int32_t n, std::int32_t k, bool add) {
    const std::uint32_t seed = static_cast<std::uint32_t>(n * 131 + k * 7 + static_cast<int>(format));
    const auto blocks        = make_blocks(format, n, k, seed);
    std::vector<float> w(static_cast<std::size_t>(n) * k);
    test::ggml::decode_blocks(format, std::as_bytes(std::span(blocks)), w);
    const auto x_bits = make_activation(k, kMaxTokens, seed ^ 0x5bd1e995U);
    const Q81 xq      = cast_q81(x_bits, k, kMaxTokens);
    std::vector<std::uint16_t> residual;
    if (add) {
        std::mt19937 rng(seed + 1);
        std::uniform_real_distribution<float> unit(-4.0F, 4.0F);
        residual.resize(static_cast<std::size_t>(n) * kMaxTokens);
        for (auto& r : residual) r = f32_to_bf16(unit(rng));
    }
    const auto expected = oracle(w, xq, n, k, kMaxTokens, add ? &residual : nullptr);

    DeviceBuffer d_w(blocks.size());
    d_w.copy_from_host(blocks.data(), blocks.size());
    DeviceBuffer d_x(x_bits.size() * 2);
    d_x.copy_from_host(x_bits.data(), x_bits.size() * 2);
    DeviceBuffer d_out(static_cast<std::size_t>(n) * kMaxTokens * 2);
    const Weight weight = device_weight(format, n, k, d_w);

    const std::string label = std::string(add ? "linear_add " : "linear ") + format_name(format) +
                              " [" + std::to_string(n) + "," + std::to_string(k) + "]";
    int failures = 0;
    for (const std::int32_t t : kTokens) {
        const auto policy = LinearPolicy::AllowA8;
        const std::size_t capacity =
            add ? ops::linear_add_workspace_capacity_bytes(format, n, k, policy, t, t)
                : ops::linear_workspace_capacity_bytes(format, n, k, policy, t, t);
        if ((t <= ops::detail::kGgmlDecodeMaxTokens) != (capacity == 0)) {
            std::cerr << label << " T=" << t << ": unexpected workspace " << capacity << '\n';
            ++failures;
        }
        DeviceArena workspace(std::max<std::size_t>(capacity, 256));
        Tensor x(d_x.p, DType::BF16, {k, t});
        Tensor out(d_out.p, DType::BF16, {n, t});
        if (add) {
            d_out.copy_from_host(residual.data(), static_cast<std::size_t>(n) * t * 2);
            ops::linear_add(x, weight, out, policy, workspace, nullptr);
        } else {
            cuda_check(cudaMemset(d_out.p, 0xFF, static_cast<std::size_t>(n) * t * 2),
                       "poison");
            ops::linear(x, weight, out, policy, workspace, nullptr);
        }
        cuda_check(cudaDeviceSynchronize(), "linear");
        const auto got_bits = download_bf16(d_out.p, static_cast<std::size_t>(n) * t);
        std::vector<double> got(got_bits.size());
        for (std::size_t i = 0; i < got.size(); ++i) got[i] = bf16_to_f32(got_bits[i]);
        const std::span<const double> ref(expected.data(), got.size());
        failures += test::verify_reduction(label + " T=" + std::to_string(t), got, ref,
                                           kGgmlQ81Criterion);
    }
    return failures;
}

// --- the cast, exactly -------------------------------------------------------------------------

// Groups that pin the rounding rule: with max|x| = 127 the scale is exactly 1, so x * 127 / max
// lands on .5 ties (2.5 -> 2, 3.5 -> 4, -0.5 -> -0) and the cast must round them to even.
std::vector<std::uint16_t> make_cast_activation(std::int32_t k, std::int32_t t) {
    auto x = make_activation(k, t, 0xC0FFEEU);
    const float ties[32] = {127.0F, 2.5F, -2.5F, 3.5F, -3.5F, 0.5F, -0.5F, 1.5F,  -1.5F, 4.5F, 5.5F,
                            6.5F,   7.5F, -7.5F, 8.5F, 9.5F,  1.0F, -1.0F, 0.0F,  -0.0F, 10.5F,
                            11.5F,  12.5F, 13.5F, 14.5F, 15.5F, 16.5F, 17.5F, 18.5F, 19.5F, 20.5F,
                            -21.5F};
    for (std::int32_t col = 0; col < t; ++col) {
        for (int i = 0; i < 32; ++i) x[static_cast<std::size_t>(col) * k + 32 + i] = f32_to_bf16(ties[i]);
        // A group whose maximum is subnormal-adjacent and one far above 1.
        for (int i = 0; i < 32; ++i) {
            x[static_cast<std::size_t>(col) * k + 64 + i] =
                f32_to_bf16(std::ldexp(static_cast<float>(i - 16), -120));
            x[static_cast<std::size_t>(col) * k + 96 + i] =
                f32_to_bf16(std::ldexp(static_cast<float>(3 * i - 47), 90));
        }
    }
    return x;
}

// The wide route's cast through its private launcher, compared byte for byte.
int cast_exact(std::int32_t k, std::int32_t t) {
    const auto x  = make_cast_activation(k, t);
    const Q81 ref = cast_q81(x, k, t);
    DeviceBuffer d_x(x.size() * 2);
    d_x.copy_from_host(x.data(), x.size() * 2);
    DeviceBuffer d_q(ref.q.size());
    DeviceBuffer d_d(ref.d.size() * 4);
    ops::detail::ggml_quantize_q8_1_launch(static_cast<const __nv_bfloat16*>(d_x.p), k, t,
                                           static_cast<std::int8_t*>(d_q.p),
                                           static_cast<float*>(d_d.p), nullptr);
    cuda_check(cudaDeviceSynchronize(), "cast");
    std::vector<std::int8_t> q(ref.q.size());
    std::vector<float> d(ref.d.size());
    cuda_check(cudaMemcpy(q.data(), d_q.p, q.size(), cudaMemcpyDeviceToHost), "q");
    cuda_check(cudaMemcpy(d.data(), d_d.p, d.size() * 4, cudaMemcpyDeviceToHost), "d");
    std::size_t bad_q = 0, bad_d = 0;
    for (std::size_t i = 0; i < q.size(); ++i) bad_q += q[i] != ref.q[i];
    for (std::size_t i = 0; i < d.size(); ++i) {
        bad_d += std::bit_cast<std::uint32_t>(d[i]) != std::bit_cast<std::uint32_t>(ref.d[i]);
    }
    const std::string label = "Q8_1 cast K=" + std::to_string(k) + " T=" + std::to_string(t);
    if (bad_q || bad_d) {
        std::cerr << label << ": " << bad_q << " codes and " << bad_d << " scales differ\n";
        return 1;
    }
    std::cout << label << ": exact (" << q.size() << " codes, " << d.size() << " scales)\n";
    return 0;
}

// The cast through the public Op on both routes. A Q8_0 [2560, 640] weight whose row r is one code
// of 1 at column r % 640 with d = 1 makes y[r, t] = q[t, r % 640] d[t, (r % 640) / 32]. One code
// step is at least 1 / 127 of that value, far above the 2^-9 BF16 rounding, so a single
// mis-rounded or mis-scaled code fails the check.
int cast_probe() {
    constexpr std::int32_t n = 2560, k = 640;
    std::vector<std::uint8_t> blocks(static_cast<std::size_t>(n) * (k / 32) * 34, 0);
    const std::uint16_t one = fp16_bits(false, 0, 0);
    for (std::int32_t r = 0; r < n; ++r) {
        for (std::int32_t b = 0; b < k / 32; ++b) {
            std::memcpy(&blocks[(static_cast<std::size_t>(r) * (k / 32) + b) * 34], &one, 2);
        }
        const std::int32_t col = r % k;
        blocks[(static_cast<std::size_t>(r) * (k / 32) + col / 32) * 34 + 2 + col % 32] = 1;
    }
    DeviceBuffer d_w(blocks.size());
    d_w.copy_from_host(blocks.data(), blocks.size());
    const Weight weight = device_weight(QType::GGML_Q8_0, n, k, d_w);
    const auto x        = make_cast_activation(k, 65);
    const Q81 ref       = cast_q81(x, k, 65);
    DeviceBuffer d_x(x.size() * 2);
    d_x.copy_from_host(x.data(), x.size() * 2);
    DeviceBuffer d_out(static_cast<std::size_t>(n) * 65 * 2);
    int failures = 0;
    for (const std::int32_t t : {1, 3, 8, 9, 65}) {
        const std::size_t capacity =
            ops::linear_workspace_capacity_bytes(QType::GGML_Q8_0, n, k, LinearPolicy::AllowA8, t, t);
        DeviceArena workspace(std::max<std::size_t>(capacity, 256));
        Tensor xt(d_x.p, DType::BF16, {k, t});
        Tensor out(d_out.p, DType::BF16, {n, t});
        ops::linear(xt, weight, out, LinearPolicy::AllowA8, workspace, nullptr);
        cuda_check(cudaDeviceSynchronize(), "probe");
        const auto got = download_bf16(d_out.p, static_cast<std::size_t>(n) * t);
        std::size_t bad = 0;
        for (std::int32_t col = 0; col < t; ++col) {
            for (std::int32_t r = 0; r < n; ++r) {
                const std::size_t xi = static_cast<std::size_t>(col) * k + r % k;
                const double want    = static_cast<double>(ref.q[xi]) * ref.d[xi / 32];
                const double y       = bf16_to_f32(got[static_cast<std::size_t>(col) * n + r]);
                const bool ok = want == 0.0 ? y == 0.0 : std::fabs(y - want) <= std::fabs(want) * 0x1p-8;
                bad += !ok;
            }
        }
        if (bad) {
            std::cerr << "Q8_1 cast probe T=" << t << ": " << bad << " outputs off the cast\n";
            ++failures;
        } else {
            std::cout << "Q8_1 cast probe T=" << t << ": every q d reproduced\n";
        }
    }
    return failures;
}

// --- admission ----------------------------------------------------------------------------------

template <class F>
int expect_throw(const char* what, F&& f) {
    try {
        f();
    } catch (const std::invalid_argument&) {
        return 0;
    }
    std::cerr << "admission: expected rejection: " << what << '\n';
    return 1;
}

int admission() {
    int failures = 0;
    const auto policy = LinearPolicy::AllowA8;
    failures += expect_throw("A16Only GGML linear", [] {
        (void)ops::linear_workspace_capacity_bytes(QType::GGML_IQ4_XS, 10240, 2560,
                                                   LinearPolicy::A16Only, 1, 1);
    });
    failures += expect_throw("A16Only GGML linear_add", [] {
        (void)ops::linear_add_workspace_capacity_bytes(QType::GGML_IQ4_XS, 2560, 6144, 1, 1);
    });
    failures += expect_throw("unregistered shape", [&] {
        (void)ops::linear_workspace_capacity_bytes(QType::GGML_IQ4_XS, 4096, 2560, policy, 1, 1);
    });
    failures += expect_throw("unregistered format at a registered shape", [&] {
        (void)ops::linear_workspace_capacity_bytes(QType::GGML_Q8_0, 10240, 2560, policy, 1, 1);
    });
    failures += expect_throw("linear_add on a linear-only shape", [&] {
        (void)ops::linear_add_workspace_capacity_bytes(QType::GGML_IQ4_XS, 10240, 2560, policy, 1,
                                                       1);
    });
    failures += expect_throw("IQ2_S has no linear", [&] {
        (void)ops::linear_workspace_capacity_bytes(QType::GGML_IQ2_S, 640, 2560, policy, 1, 1);
    });
    // An interval reserves for its widest point.
    const auto wide = ops::linear_workspace_capacity_bytes(QType::GGML_IQ4_XS, 10240, 2560, policy,
                                                           1, 4096);
    const auto point = ops::linear_workspace_capacity_bytes(QType::GGML_IQ4_XS, 10240, 2560, policy,
                                                            4096, 4096);
    if (wide != point || point < static_cast<std::size_t>(4096) * 2560 + 4096 * 80 * 4) {
        std::cerr << "admission: workspace interval " << wide << " vs point " << point << '\n';
        ++failures;
    }
    return failures;
}

QType parse_format(std::string_view name) {
    for (const QType f : {QType::GGML_IQ4_XS, QType::GGML_IQ3_S, QType::GGML_Q6_K,
                          QType::GGML_IQ4_NL, QType::GGML_Q8_0, QType::GGML_Q2_0}) {
        if (name == format_name(f)) return f;
    }
    throw std::invalid_argument("unknown --format " + std::string(name));
}

} // namespace

int main(int argc, char** argv) {
    if (ninfer::test::cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    try {
        bool have_format = false;
        QType format     = QType::GGML_IQ4_XS;
        std::int32_t n = 0, k = 0;
        for (int i = 1; i < argc; ++i) {
            const std::string_view arg(argv[i]);
            if (arg == "--format" && i + 1 < argc) {
                format      = parse_format(argv[++i]);
                have_format = true;
            } else if (arg == "--shape" && i + 2 < argc) {
                n = std::stoi(argv[++i]);
                k = std::stoi(argv[++i]);
            } else {
                throw std::invalid_argument(
                    "usage: ninfer_linear_ggml_test [--format NAME] [--shape N K]");
            }
        }
        int failures = 0;
        if (!have_format && n == 0) {
            failures += admission();
            for (const std::int32_t kk : {640, 2560, 6144}) failures += cast_exact(kk, 300);
            failures += cast_probe();
        }
        for (const auto& shape : ops::detail::kGgmlLinearShapes) {
            if (have_format && shape.format != format) continue;
            if (n != 0 && (shape.n != n || shape.k != k)) continue;
            failures += run_problem(shape.format, shape.n, shape.k, false);
            if (shape.linear_add) failures += run_problem(shape.format, shape.n, shape.k, true);
        }
        std::cout << (failures == 0 ? "OK" : "FAIL") << " GGML Linear/LinearAdd\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "GGML Linear: " << error.what() << '\n';
        return 1;
    }
}
