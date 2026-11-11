// Linear and LinearAdd over GGML block-format weights (ggml_iq4_xs, ggml_iq3_s, ggml_q6_k,
// ggml_q5_k, ggml_q4_k, ggml_iq4_nl, ggml_q8_0, ggml_q2_0) at every registered Qwen3.8-Flash-Next
// problem.
//
// Oracle: the weight is decoded by the exact host decoder (ops/ggml_blocks_decode.h, itself
// qualified bit-for-bit against ggml b11316), the activation passes through an independent host
// copy of the contract's Q8_1 cast, and every output is the naive FP64 dot product of the two
// (plus the BF16 residual for LinearAdd). The cast is a semantic boundary of these formats
// (linear.h), so it belongs to the oracle; it is qualified exactly on its own below, both directly
// (the private wide-route cast, byte for byte) and through the public Op on both routes (a one-hot
// Q8_0 probe that reads every q * d back out).
//
// The affine K-quants (Q5_K, Q4_K) are also compared bit for bit against the host emulation of
// both routes' arithmetic (ops/ggml_linear_emulation.h, qualified against FP64 on the host by
// ninfer_ggml_linear_affine_test): the decode route at T = 1..8 and the wide route beyond.
//
// T covers both routes and their seams: 1..8 (decode), 9 (first wide), 63/64/65 (one wide column
// tile and its edges) and 300 (interior, partial last tile).
//
// The int8 tensor-core decode route (LinearPolicy::AllowA8GgmlDecodeMma, every format, Linear and
// LinearAdd) at T = 1..9: T <= 8 bit for bit against its host emulation
// (ops/linear/ggml/ggml_mma_emulation.h, qualified against FP64 on the host by
// ninfer_ggml_mma_emulation_test) and within the FP64 criterion; T = 9 bit for bit the AllowA8 wide
// route. linear_pair takes no policy and keeps the dp4a route: its outputs stay linear()'s under
// AllowA8 bit for bit (the Forward runs two linears instead under --ggml-decode-mma). The policy's
// admission: GGML formats only, and a 16-byte aligned weight for the route.
//
// linear_pair over GGML weights (its decode route, T = 1..8): every ordered pair of IQ4_XS, IQ3_S,
// Q6_K, Q5_K and Q4_K at both registered pair shapes, each output against the same FP64 oracle and bit for
// bit against linear() on its own weight; and its admission.
//
// Usage: ninfer_linear_ggml_test [--time] [--format NAME] [--shape N K]
//   --time: per-problem timing of the decode route against the tensor-core route (time_problem);
//   no checks.

#include "ninfer/ops/linear.h"
#include "ninfer/ops/linear_add.h"
#include "ninfer/ops/linear_pair.h"

#include "core/arena.h"
#include "core/weight.h"
#include "ops/ggml_blocks_decode.h"
#include "ops/ggml_linear_emulation.h"
#include "ops/linear/ggml/ggml_launch.h"
#include "ops/linear/ggml/ggml_mma_emulation.h"
#include "ops/linear/ggml/ggml_shapes.h"
#include "ops/op_check.h"
#include "ops/op_tester.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <memory>
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
    case QType::GGML_Q5_K: return "ggml_q5_k";
    case QType::GGML_Q4_K: return "ggml_q4_k";
    case QType::GGML_IQ4_NL: return "ggml_iq4_nl";
    case QType::GGML_Q8_0: return "ggml_q8_0";
    case QType::GGML_Q2_0: return "ggml_q2_0";
    default: return "?";
    }
}

// --- the contract's Q8_1 cast, host side (ops/ggml_linear_emulation.h) ---------------------------

using test::ggml_linear::Q81;

Q81 cast_q81(const std::vector<std::uint16_t>& x, std::int32_t k, std::int32_t t) {
    return test::ggml_linear::cast_q81(x, k, t);
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
    // IQ4_XS and IQ3_S scale codes multiply d by up to ~32; Q6_K by up to 128 and every code by 32;
    // Q4_K and Q5_K by up to 63 and every code by up to 31, and their dmin (after d) by up to 63.
    const int base = format == QType::GGML_Q8_0 || format == QType::GGML_IQ4_NL ? -12 : -14;
    const bool affine = test::ggml_linear::is_affine(format);
    for (std::size_t i = 0; i < blocks; ++i) {
        for (int field = 0; field < (affine ? 2 : 1); ++field) {
            const auto h = fp16_bits(rng() & 1, base + static_cast<int>(rng() % 4),
                                     static_cast<int>(rng() & 0x3FF));
            std::memcpy(&bytes[i * block.bytes + d_offset + 2 * field], &h, 2);
        }
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

// Q5_K and Q4_K: the BF16 outputs [t][n] of both routes' arithmetic for every column, by the host
// emulation (a column's bits depend on neither T nor its position): [0] the decode route, [1] the
// wide route. Empty for the other formats.
std::vector<std::vector<std::uint16_t>> emulate(QType format, const std::vector<std::uint8_t>& blocks,
                                                std::int32_t n, std::int32_t k, const Q81& x,
                                                const std::vector<std::uint16_t>* residual) {
    namespace em = test::ggml_linear;
    if (!em::is_affine(format)) { return {}; }
    const auto xm = em::activation_sums(x);
    const std::size_t row_bytes = static_cast<std::size_t>(k) / 256 * ggml_block(format).bytes;
    std::vector<std::vector<std::uint16_t>> out(2, std::vector<std::uint16_t>(static_cast<std::size_t>(n) * kMaxTokens));
    const unsigned workers = std::max(1U, std::min(16U, std::thread::hardware_concurrency()));
    std::vector<std::thread> pool;
    for (unsigned id = 0; id < workers; ++id) {
        pool.emplace_back([&, id] {
            for (std::int32_t row = static_cast<std::int32_t>(id); row < n; row += static_cast<std::int32_t>(workers)) {
                const auto subs = em::affine_row(format, blocks.data() + row * row_bytes, k);
                for (std::int32_t col = 0; col < kMaxTokens; ++col) {
                    const std::size_t o   = static_cast<std::size_t>(col) * n + row;
                    const std::uint16_t* r = residual ? &(*residual)[o] : nullptr;
                    if (col < ops::detail::kGgmlDecodeMaxTokens) {
                        out[0][o] = em::store(em::row_dot(em::Route::Decode, subs, x, xm, k, col), r);
                    }
                    out[1][o] = em::store(em::row_dot(em::Route::Wide, subs, x, xm, k, col), r);
                }
            }
        });
    }
    for (auto& worker : pool) worker.join();
    return out;
}

// The int8 tensor-core decode route's BF16 outputs [t][n] for the first kGgmlDecodeMaxTokens columns,
// by its host emulation (a column's bits depend on neither T nor its position).
std::vector<std::uint16_t> emulate_mma(QType format, const std::vector<std::uint8_t>& blocks, std::int32_t n,
                                       std::int32_t k, const Q81& x, const std::vector<std::uint16_t>* residual) {
    namespace em                = test::ggml_linear;
    constexpr std::int32_t kCols = ops::detail::kGgmlDecodeMaxTokens;
    const std::size_t row_bytes = ops::ggml_mma::row_bytes(format, k);
    std::vector<std::uint16_t> out(static_cast<std::size_t>(n) * kCols);
    const unsigned workers = std::max(1U, std::min(16U, std::thread::hardware_concurrency()));
    std::vector<std::thread> pool;
    for (unsigned id = 0; id < workers; ++id) {
        pool.emplace_back([&, id] {
            for (std::int32_t row = static_cast<std::int32_t>(id); row < n; row += static_cast<std::int32_t>(workers)) {
                const auto* bytes = reinterpret_cast<const std::byte*>(blocks.data() + row * row_bytes);
                for (std::int32_t col = 0; col < kCols; ++col) {
                    float acc = 0.0F;
                    ops::ggml_mma_emulate(format, bytes, x.q.data() + static_cast<std::size_t>(col) * k,
                                          x.d.data() + static_cast<std::size_t>(col) * (k / 32), k, &acc);
                    const std::size_t o = static_cast<std::size_t>(col) * n + row;
                    out[o]              = em::store(acc, residual ? &(*residual)[o] : nullptr);
                }
            }
        });
    }
    for (auto& worker : pool) worker.join();
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
    const auto emulated = emulate(format, blocks, n, k, xq, add ? &residual : nullptr);

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
        if (!emulated.empty()) {
            const auto& route = emulated[t <= ops::detail::kGgmlDecodeMaxTokens ? 0 : 1];
            std::size_t differ = 0;
            for (std::size_t i = 0; i < got_bits.size(); ++i) differ += got_bits[i] != route[i];
            if (differ != 0) {
                std::cerr << label << " T=" << t << ": " << differ << " outputs differ from the host emulation\n";
                ++failures;
            }
        }
    }

    // The int8 tensor-core decode route: T = 1..8 against its emulation, T = 9 the AllowA8 wide route.
    const auto mma_emulated = emulate_mma(format, blocks, n, k, xq, add ? &residual : nullptr);
    const auto run          = [&](LinearPolicy policy, std::int32_t t) {
        const std::size_t capacity = add ? ops::linear_add_workspace_capacity_bytes(format, n, k, policy, t, t)
                                         : ops::linear_workspace_capacity_bytes(format, n, k, policy, t, t);
        DeviceArena workspace(std::max<std::size_t>(capacity, 256));
        Tensor x(d_x.p, DType::BF16, {k, t});
        Tensor out(d_out.p, DType::BF16, {n, t});
        if (add) {
            d_out.copy_from_host(residual.data(), static_cast<std::size_t>(n) * t * 2);
            ops::linear_add(x, weight, out, policy, workspace, nullptr);
        } else {
            cuda_check(cudaMemset(d_out.p, 0xFF, static_cast<std::size_t>(n) * t * 2), "poison");
            ops::linear(x, weight, out, policy, workspace, nullptr);
        }
        cuda_check(cudaDeviceSynchronize(), "linear");
        return download_bf16(d_out.p, static_cast<std::size_t>(n) * t);
    };
    int mma_failures = 0;
    for (std::int32_t t = 1; t <= ops::detail::kGgmlDecodeMaxTokens + 1; ++t) {
        const std::string what = label + " AllowA8GgmlDecodeMma T=" + std::to_string(t);
        const auto bits        = run(LinearPolicy::AllowA8GgmlDecodeMma, t);
        std::vector<double> got(bits.size());
        for (std::size_t i = 0; i < got.size(); ++i) got[i] = bf16_to_f32(bits[i]);
        mma_failures += test::verify_reduction(what, got, std::span<const double>(expected.data(), got.size()),
                                               kGgmlQ81Criterion);
        std::size_t differ = 0;
        if (t <= ops::detail::kGgmlDecodeMaxTokens) {
            for (std::size_t i = 0; i < bits.size(); ++i) differ += bits[i] != mma_emulated[i];
            if (differ != 0) { std::cerr << what << ": " << differ << " outputs differ from the host emulation\n"; }
        } else {
            const auto wide = run(LinearPolicy::AllowA8, t);
            for (std::size_t i = 0; i < bits.size(); ++i) differ += bits[i] != wide[i];
            if (differ != 0) { std::cerr << what << ": " << differ << " outputs differ from AllowA8's wide route\n"; }
        }
        mma_failures += differ != 0;
    }
    if (mma_failures == 0) {
        std::cout << label << " AllowA8GgmlDecodeMma: T=1..8 equal to the emulation bit for bit, T=9 to AllowA8, "
                  << "within the FP64 criterion\n";
    }
    return failures + mma_failures;
}

// linear_pair on (first, second) at [n, k], T = 1..8: both outputs against the FP64 oracle and bit
// for bit against linear() of each weight alone.
int run_pair(QType first, QType second, std::int32_t n, std::int32_t k) {
    const std::uint32_t seed = static_cast<std::uint32_t>(n * 17 + k + 97 * static_cast<int>(first) + static_cast<int>(second));
    const auto blocks1       = make_blocks(first, n, k, seed);
    const auto blocks2       = make_blocks(second, n, k, seed ^ 0x9e3779b9U);
    std::vector<float> w1(static_cast<std::size_t>(n) * k), w2(w1.size());
    test::ggml::decode_blocks(first, std::as_bytes(std::span(blocks1)), w1);
    test::ggml::decode_blocks(second, std::as_bytes(std::span(blocks2)), w2);
    constexpr std::int32_t kPairTokens = 8;
    const auto x_bits = make_activation(k, kPairTokens, seed ^ 0x5bd1e995U);
    const Q81 xq      = cast_q81(x_bits, k, kPairTokens);
    const auto want1  = oracle(w1, xq, n, k, kPairTokens, nullptr);
    const auto want2  = oracle(w2, xq, n, k, kPairTokens, nullptr);

    DeviceBuffer d_w1(blocks1.size()), d_w2(blocks2.size());
    d_w1.copy_from_host(blocks1.data(), blocks1.size());
    d_w2.copy_from_host(blocks2.data(), blocks2.size());
    DeviceBuffer d_x(x_bits.size() * 2);
    d_x.copy_from_host(x_bits.data(), x_bits.size() * 2);
    const std::size_t out_bytes = static_cast<std::size_t>(n) * kPairTokens * 2;
    DeviceBuffer d_o1(out_bytes), d_o2(out_bytes), d_r1(out_bytes), d_r2(out_bytes);
    const Weight weight1 = device_weight(first, n, k, d_w1);
    const Weight weight2 = device_weight(second, n, k, d_w2);
    DeviceArena workspace(256);

    const std::string label = std::string("linear_pair ") + format_name(first) + "+" + format_name(second) + " [" +
                              std::to_string(n) + "," + std::to_string(k) + "]";
    int failures = 0;
    for (std::int32_t t = 1; t <= kPairTokens; ++t) {
        if (ops::linear_pair_workspace_capacity_bytes(weight1, weight2, t, t) != 0) {
            std::cerr << label << " T=" << t << ": nonzero workspace\n";
            ++failures;
        }
        Tensor x(d_x.p, DType::BF16, {k, t});
        Tensor o1(d_o1.p, DType::BF16, {n, t}), o2(d_o2.p, DType::BF16, {n, t});
        Tensor r1(d_r1.p, DType::BF16, {n, t}), r2(d_r2.p, DType::BF16, {n, t});
        for (const auto* buffer : {&d_o1, &d_o2, &d_r1, &d_r2}) {
            cuda_check(cudaMemset(buffer->p, 0xFF, out_bytes), "poison");
        }
        ops::linear_pair(x, weight1, weight2, o1, o2, nullptr);
        ops::linear(x, weight1, r1, LinearPolicy::AllowA8, workspace, nullptr);
        ops::linear(x, weight2, r2, LinearPolicy::AllowA8, workspace, nullptr);
        cuda_check(cudaDeviceSynchronize(), "linear_pair");
        const std::size_t count = static_cast<std::size_t>(n) * t;
        const auto got1 = download_bf16(d_o1.p, count), got2 = download_bf16(d_o2.p, count);
        const auto ref1 = download_bf16(d_r1.p, count), ref2 = download_bf16(d_r2.p, count);
        if (got1 != ref1 || got2 != ref2) {
            std::cerr << label << " T=" << t << ": outputs differ from linear() bit for bit\n";
            ++failures;
        }
        for (int which = 0; which < 2; ++which) {
            const auto& bits = which == 0 ? got1 : got2;
            std::vector<double> got(bits.size());
            for (std::size_t i = 0; i < got.size(); ++i) got[i] = bf16_to_f32(bits[i]);
            const auto& want = which == 0 ? want1 : want2;
            failures += test::verify_reduction(label + (which == 0 ? " first" : " second") + " T=" + std::to_string(t),
                                               got, std::span<const double>(want.data(), got.size()), kGgmlQ81Criterion);
        }
    }
    if (failures == 0) { std::cout << label << ": T=1..8 equal to linear() bit for bit, within the FP64 criterion\n"; }
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
        // A group of true BF16 subnormals (multiples of 2^-133, where 127 / amax overflows FP32)
        // and one far above 1.
        for (int i = 0; i < 32; ++i) {
            x[static_cast<std::size_t>(col) * k + 64 + i] =
                f32_to_bf16(std::ldexp(static_cast<float>(i - 16), -133));
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
// of 1 at column r % 640 with block scale s makes y[r, t] = q[t, r % 640] d[t, (r % 640) / 32] s.
// One code step is at least 1 / 127 of that value, far above the 2^-9 BF16 rounding, so a single
// mis-rounded or mis-scaled code fails the check, provided y is a normal BF16. s = 1 except over
// the subnormal group (columns 64..95), where q d is itself a BF16 subnormal (spacing 2^-133, a
// whole step or more) and would hide its codes; there s = 2^15 lifts y into the normal range. s is
// a power of two, so the product is exact on both routes.
int cast_probe() {
    constexpr std::int32_t n = 2560, k = 640;
    constexpr std::int32_t kSubnormalBlock = 2;
    constexpr double kSubnormalLift        = 0x1p+15;
    std::vector<std::uint8_t> blocks(static_cast<std::size_t>(n) * (k / 32) * 34, 0);
    const std::uint16_t one  = fp16_bits(false, 0, 0);
    const std::uint16_t lift = fp16_bits(false, 15, 0);
    for (std::int32_t r = 0; r < n; ++r) {
        for (std::int32_t b = 0; b < k / 32; ++b) {
            std::memcpy(&blocks[(static_cast<std::size_t>(r) * (k / 32) + b) * 34],
                        b == kSubnormalBlock ? &lift : &one, 2);
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
                const double s       = (r % k) / 32 == kSubnormalBlock ? kSubnormalLift : 1.0;
                const double want    = static_cast<double>(ref.q[xi]) * ref.d[xi / 32] * s;
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
    // AllowA8GgmlDecodeMma: the GGML formats' policy only, and the route needs a 16-byte aligned weight.
    failures += expect_throw("AllowA8GgmlDecodeMma on a BF16 linear", [] {
        (void)ops::linear_workspace_capacity_bytes(QType::BF16, 2560, 2560, LinearPolicy::AllowA8GgmlDecodeMma, 1, 1);
    });
    failures += expect_throw("AllowA8GgmlDecodeMma on an FP8 linear_add", [] {
        (void)ops::linear_add_workspace_capacity_bytes(QType::FP8_E4M3FN_ROW_BF16, 5120, 6144,
                                                       LinearPolicy::AllowA8GgmlDecodeMma, 1, 1);
    });
    if (ops::linear_workspace_capacity_bytes(QType::GGML_Q8_0, 2560, 640, LinearPolicy::AllowA8GgmlDecodeMma, 1, 8) !=
            0 ||
        ops::linear_workspace_capacity_bytes(QType::GGML_Q8_0, 2560, 640, LinearPolicy::AllowA8GgmlDecodeMma, 1, 9) !=
            ops::linear_workspace_capacity_bytes(QType::GGML_Q8_0, 2560, 640, policy, 1, 9)) {
        std::cerr << "admission: AllowA8GgmlDecodeMma workspace differs from AllowA8's\n";
        ++failures;
    }
    {
        // A Q8_0 matrix two bytes past a 16-byte boundary: admitted by AllowA8 (2-byte rows), refused
        // by the tensor-core route at T <= 8.
        constexpr std::int32_t n = 2560, k = 640;
        const std::size_t bytes  = static_cast<std::size_t>(n) * (k / 32) * 34;
        DeviceBuffer storage(bytes + 16);
        cuda_check(cudaMemset(storage.p, 0, bytes + 16), "zero");
        Weight weight      = device_weight(QType::GGML_Q8_0, n, k, storage);
        weight.payload     = static_cast<std::byte*>(storage.p) + 2;
        weight.qdata       = weight.payload;
        DeviceBuffer d_x(static_cast<std::size_t>(k) * 2 * 2), d_out(static_cast<std::size_t>(n) * 2 * 2);
        cuda_check(cudaMemset(d_x.p, 0, d_x.bytes), "zero x");
        DeviceArena workspace(256);
        Tensor x(d_x.p, DType::BF16, {k, 1});
        Tensor out(d_out.p, DType::BF16, {n, 1});
        ops::linear(x, weight, out, policy, workspace, nullptr);
        cuda_check(cudaDeviceSynchronize(), "misaligned AllowA8");
        failures += expect_throw("AllowA8GgmlDecodeMma with a misaligned weight", [&] {
            ops::linear(x, weight, out, LinearPolicy::AllowA8GgmlDecodeMma, workspace, nullptr);
        });
    }
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

int pair_admission() {
    int failures = 0;
    DeviceBuffer a(static_cast<std::size_t>(640) * 2560), b(a.bytes);
    const Weight iq4  = device_weight(QType::GGML_IQ4_XS, 640, 2560, a);
    const Weight q6   = device_weight(QType::GGML_Q6_K, 640, 2560, b);
    const Weight q8_0 = device_weight(QType::GGML_Q8_0, 2560, 640, b);
    const Weight wide = device_weight(QType::GGML_IQ4_XS, 10240, 2560, b);
    failures += expect_throw("pair beyond the decode route", [&] { (void)ops::linear_pair_workspace_capacity_bytes(iq4, q6, 1, 9); });
    failures += expect_throw("pair with an unpaired format", [&] { (void)ops::linear_pair_workspace_capacity_bytes(iq4, q8_0, 1, 1); });
    failures += expect_throw("pair of different shapes", [&] { (void)ops::linear_pair_workspace_capacity_bytes(iq4, wide, 1, 1); });
    failures += expect_throw("pair at an unregistered pair shape", [&] { (void)ops::linear_pair_workspace_capacity_bytes(wide, wide, 1, 1); });
    return failures;
}

// --time: the decode route (AllowA8, dp4a) against the int8 tensor-core route (AllowA8GgmlDecodeMma)
// on one registered problem at T = 1, 2, 4 and 8, for --ggml-decode-mma's per-shape on/off timing.
// The weight is copied until the copies exceed the L2 cache several times over and the launches
// cycle through them, so every launch reads its weight from VRAM as a decode step does. Five rounds
// per policy, alternating which policy goes first; each round is 100 launches between two CUDA
// events. Prints one tab-separated line per T: label, T, median microseconds per launch of each
// route and their ratio. Measures only; it checks no outputs (the default mode does).
void time_problem(QType format, std::int32_t n, std::int32_t k, bool add) {
    constexpr int kLaunches = 100, kRounds = 5;
    const auto blocks       = make_blocks(format, n, k, static_cast<std::uint32_t>(n * 131 + k * 7));
    const auto x_bits       = make_activation(k, ops::detail::kGgmlDecodeMaxTokens, 0x5bd1e995U);
    const std::size_t copies =
        std::clamp<std::size_t>((std::size_t{96} << 20) / std::max<std::size_t>(blocks.size(), 1) + 1, 1, 64);
    std::vector<std::unique_ptr<DeviceBuffer>> storage;
    std::vector<Weight> weights;
    for (std::size_t c = 0; c < copies; ++c) {
        storage.push_back(std::make_unique<DeviceBuffer>(blocks.size()));
        storage.back()->copy_from_host(blocks.data(), blocks.size());
        weights.push_back(device_weight(format, n, k, *storage.back()));
    }
    DeviceBuffer d_x(x_bits.size() * 2);
    d_x.copy_from_host(x_bits.data(), x_bits.size() * 2);
    DeviceBuffer d_out(static_cast<std::size_t>(n) * ops::detail::kGgmlDecodeMaxTokens * 2);
    cuda_check(cudaMemset(d_out.p, 0, static_cast<std::size_t>(n) * ops::detail::kGgmlDecodeMaxTokens * 2), "zero");
    cudaEvent_t start = nullptr, stop = nullptr;
    cuda_check(cudaEventCreate(&start), "event");
    cuda_check(cudaEventCreate(&stop), "event");
    const std::string label = std::string(add ? "linear_add " : "linear ") + format_name(format) + " [" +
                              std::to_string(n) + "," + std::to_string(k) + "]";
    for (const std::int32_t t : {1, 2, 4, 8}) {
        const LinearPolicy policies[2] = {LinearPolicy::AllowA8, LinearPolicy::AllowA8GgmlDecodeMma};
        std::vector<double> us[2];
        for (int round = 0; round < kRounds; ++round) {
            for (int order = 0; order < 2; ++order) {
                const int p              = (round + order) % 2;
                const std::size_t needed = add ? ops::linear_add_workspace_capacity_bytes(format, n, k, policies[p], t, t)
                                               : ops::linear_workspace_capacity_bytes(format, n, k, policies[p], t, t);
                DeviceArena workspace(std::max<std::size_t>(needed, 256));
                Tensor x(d_x.p, DType::BF16, {k, t});
                Tensor out(d_out.p, DType::BF16, {n, t});
                const auto launch = [&](int i) {
                    const Weight& w = weights[static_cast<std::size_t>(i) % copies];
                    if (add) {
                        ops::linear_add(x, w, out, policies[p], workspace, nullptr);
                    } else {
                        ops::linear(x, w, out, policies[p], workspace, nullptr);
                    }
                };
                for (int i = 0; i < 5; ++i) launch(i); // warm-up
                cuda_check(cudaEventRecord(start, nullptr), "record");
                for (int i = 0; i < kLaunches; ++i) launch(i);
                cuda_check(cudaEventRecord(stop, nullptr), "record");
                cuda_check(cudaEventSynchronize(stop), "sync");
                float ms = 0.0F;
                cuda_check(cudaEventElapsedTime(&ms, start, stop), "elapsed");
                us[p].push_back(1000.0 * ms / kLaunches);
            }
        }
        for (auto& v : us) std::sort(v.begin(), v.end());
        const double dp4a = us[0][kRounds / 2], mma = us[1][kRounds / 2];
        std::printf("time\t%s\tT=%d\tdp4a %.2f us\tmma %.2f us\tmma/dp4a %.3f\n", label.c_str(), t, dp4a, mma,
                    mma / dp4a);
    }
    cuda_check(cudaEventDestroy(start), "event");
    cuda_check(cudaEventDestroy(stop), "event");
}

QType parse_format(std::string_view name) {
    for (const QType f : {QType::GGML_IQ4_XS, QType::GGML_IQ3_S, QType::GGML_Q6_K, QType::GGML_Q5_K,
                          QType::GGML_Q4_K, QType::GGML_IQ4_NL, QType::GGML_Q8_0, QType::GGML_Q2_0}) {
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
        bool have_format = false, timing = false;
        QType format     = QType::GGML_IQ4_XS;
        std::int32_t n = 0, k = 0;
        for (int i = 1; i < argc; ++i) {
            const std::string_view arg(argv[i]);
            if (arg == "--time") {
                timing = true;
            } else if (arg == "--format" && i + 1 < argc) {
                format      = parse_format(argv[++i]);
                have_format = true;
            } else if (arg == "--shape" && i + 2 < argc) {
                n = std::stoi(argv[++i]);
                k = std::stoi(argv[++i]);
            } else {
                throw std::invalid_argument(
                    "usage: ninfer_linear_ggml_test [--time] [--format NAME] [--shape N K]");
            }
        }
        if (timing) {
            for (const auto& shape : ops::detail::kGgmlLinearShapes) {
                if (have_format && shape.format != format) continue;
                if (n != 0 && (shape.n != n || shape.k != k)) continue;
                time_problem(shape.format, shape.n, shape.k, false);
                if (shape.linear_add) time_problem(shape.format, shape.n, shape.k, true);
            }
            return 0;
        }
        int failures = 0;
        if (!have_format && n == 0) {
            failures += admission();
            failures += pair_admission();
            for (const auto& shape : ops::detail::kGgmlPairShapes) {
                for (const QType f1 : ops::detail::kGgmlPairFormats) {
                    for (const QType f2 : ops::detail::kGgmlPairFormats) { failures += run_pair(f1, f2, shape.n, shape.k); }
                }
            }
            for (const std::int32_t kk : {640, 2560, 6144}) failures += cast_exact(kk, 300);
            failures += cast_probe();
        }
        for (const auto& shape : ops::detail::kGgmlLinearShapes) {
            if (have_format && shape.format != format) continue;
            if (n != 0 && (shape.n != n || shape.k != k)) continue;
            failures += run_problem(shape.format, shape.n, shape.k, false);
            if (shape.linear_add) failures += run_problem(shape.format, shape.n, shape.k, true);
        }
        std::cout << (failures == 0 ? "OK" : "FAIL") << " GGML Linear/LinearAdd/LinearPair\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "GGML Linear: " << error.what() << '\n';
        return 1;
    }
}
