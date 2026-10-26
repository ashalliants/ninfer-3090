#pragma once

// Host emulation of the GGML linears' FP32 arithmetic (src/ops/linear/ggml) for the affine
// K-quants GGML_Q4_K and GGML_Q5_K, plus the contract's Q8_1 activation cast (linear.h).
//
// A Q4_K/Q5_K value is d * sc * q - dmin * m per 32-value sub-block (tensor-formats.md Section
// 3.5), so a sub-block's dot product with the Q8_1 activation (codes a, scale xd) is
//
//     xd * (scale * sum(q a) - offset * sum(a)),   scale = d * sc, offset = dmin * m,
//
// with both integer sums exact. Both routes add a sub-block's scale term and offset term as
//
//     acc = fma(float(sum(q a)), scale * xd, acc)   and   acc = fma(-offset, xm, acc),   xm = xd * float(sum(a))
//
// (each name one FP32 rounding). The decode route (T <= 8, ggml_mmvq.cuh) splits a row's
// sub-blocks over lanes_per_row(K / 32) lanes, lane l taking sub-blocks l, l + lanes, ... in order,
// each sub-block's scale term then its offset term, and adds the lanes' sums by an xor butterfly.
// The wide route (ggml_mmq.cuh) walks K in tiles of four sub-blocks: the tile's four scale terms in
// order, then its four offset terms. The output is y = acc (+ the BF16 residual for LinearAdd), rounded to BF16
// to nearest even. Every step is reproduced here one IEEE operation at a time, so the emulation is
// the routes' bit pattern; the including translation unit is built without FP contraction
// (ninfer_op_oracle_options).
//
// Host-only (no CUDA): the CPU qualification of the arithmetic links nothing but ninfer_core.

#include "core/weight.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <vector>

namespace ninfer::test::ggml_linear {

[[nodiscard]] inline float bf16_value(std::uint16_t h) {
    return std::bit_cast<float>(static_cast<std::uint32_t>(h) << 16);
}

// Round to nearest even; a NaN stays a NaN.
[[nodiscard]] inline std::uint16_t bf16_bits(float f) {
    std::uint32_t u = std::bit_cast<std::uint32_t>(f);
    if ((u & 0x7FFFFFFFU) > 0x7F800000U) { return static_cast<std::uint16_t>((u >> 16) | 0x0040U); }
    u += 0x7FFFU + ((u >> 16) & 1U);
    return static_cast<std::uint16_t>(u >> 16);
}

// The contract's Q8_1 cast (linear.h) of a BF16 [K,T] activation: codes q [T][K] and one scale per
// 32 values d [T][K / 32].
struct Q81 {
    std::vector<std::int8_t> q;
    std::vector<float> d;
};

[[nodiscard]] inline Q81 cast_q81(std::span<const std::uint16_t> x, std::int32_t k, std::int32_t t) {
    Q81 out;
    out.q.resize(static_cast<std::size_t>(k) * t);
    out.d.resize(static_cast<std::size_t>(k / 32) * t);
    for (std::size_t g = 0; g < out.d.size(); ++g) {
        float amax = 0.0F;
        for (int i = 0; i < 32; ++i) amax = std::max(amax, std::fabs(bf16_value(x[32 * g + i])));
        // Pre-scaled by a power of two when 127 / amax would overflow (BF16 subnormals).
        const float scale   = amax < 0x1p-100F ? 0x1p+100F : 1.0F;
        const float inverse = amax == 0.0F ? 0.0F : 127.0F / (amax * scale);
        for (int i = 0; i < 32; ++i) {
            // Default rounding mode: nearest, ties to even.
            out.q[32 * g + i] =
                static_cast<std::int8_t>(std::nearbyint((bf16_value(x[32 * g + i]) * scale) * inverse));
        }
        out.d[g] = amax / 127.0F;
    }
    return out;
}

// sum(a) of each 32-value group, exact.
[[nodiscard]] inline std::vector<std::int32_t> code_sums(const Q81& x) {
    std::vector<std::int32_t> out(x.d.size());
    for (std::size_t g = 0; g < out.size(); ++g) {
        std::int32_t sum = 0;
        for (int i = 0; i < 32; ++i) sum += x.q[32 * g + i];
        out[g] = sum;
    }
    return out;
}

// xm = xd * float(sum(a)) of each group: the activation term of the min, one FP32 product.
[[nodiscard]] inline std::vector<float> activation_sums(const Q81& x) {
    const auto sums = code_sums(x);
    std::vector<float> out(sums.size());
    for (std::size_t g = 0; g < out.size(); ++g) out[g] = x.d[g] * static_cast<float>(sums[g]);
    return out;
}

[[nodiscard]] constexpr bool is_affine(QType format) {
    return format == QType::GGML_Q4_K || format == QType::GGML_Q5_K;
}

// One decoded sub-block of an affine format, as the device decoder (ggml_blocks.cuh,
// Format<F>::decode) forms it: unsigned codes q in [0, 31], scale = d * sc and offset = dmin * m.
struct AffineSub {
    std::array<std::int32_t, 32> q{};
    float scale  = 0.0F;
    float offset = 0.0F;
};

namespace detail {

[[nodiscard]] inline float fp16_value(const std::uint8_t* p) {
    const std::uint32_t h        = static_cast<std::uint32_t>(p[0]) | (static_cast<std::uint32_t>(p[1]) << 8);
    const std::uint32_t sign     = (h & 0x8000U) << 16;
    const std::uint32_t exponent = (h >> 10) & 0x1FU;
    const std::uint32_t mantissa = h & 0x3FFU;
    if (exponent == 0x1F) { return std::bit_cast<float>(sign | 0x7F800000U | (mantissa << 13)); }
    if (exponent != 0) { return std::bit_cast<float>(sign | ((exponent + 112) << 23) | (mantissa << 13)); }
    return std::bit_cast<float>(sign | std::bit_cast<std::uint32_t>(static_cast<float>(mantissa) * 0x1p-24F));
}

} // namespace detail

// Sub-block s (0-based along the row) of a row of `format` blocks.
[[nodiscard]] inline AffineSub affine_sub(QType format, const std::uint8_t* row, int s) {
    if (!is_affine(format)) { throw std::invalid_argument("affine_sub: not Q4_K or Q5_K"); }
    const bool q5           = format == QType::GGML_Q5_K;
    const std::uint8_t* b   = row + static_cast<std::size_t>(s / 8) * (q5 ? 176 : 144);
    const int j             = s % 8;
    const std::uint8_t* sc6 = b + 4;
    int sc = 0, m = 0;
    if (j < 4) {
        sc = sc6[j] & 63;
        m  = sc6[j + 4] & 63;
    } else {
        sc = (sc6[j + 4] & 0xF) | ((sc6[j - 4] >> 6) << 4);
        m  = (sc6[j + 4] >> 4) | ((sc6[j] >> 6) << 4);
    }
    AffineSub out;
    out.scale              = detail::fp16_value(b) * static_cast<float>(sc);
    out.offset             = detail::fp16_value(b + 2) * static_cast<float>(m);
    const std::uint8_t* qs = b + (q5 ? 48 : 16) + 32 * (j / 2);
    for (int l = 0; l < 32; ++l) {
        const int low = j % 2 ? qs[l] >> 4 : qs[l] & 0xF;
        out.q[l]      = q5 ? low + (((b[16 + l] >> j) & 1) << 4) : low;
    }
    return out;
}

// ggml_mmvq.cuh's mmvq_lanes_per_row: the widest power of two that divides the row's sub-block
// count and leaves each lane at least four sub-blocks.
[[nodiscard]] constexpr int lanes_per_row(int sub_blocks) {
    for (int lanes = 32; lanes > 1; lanes /= 2) {
        if (sub_blocks % lanes == 0 && sub_blocks / lanes >= 4) { return lanes; }
    }
    return 1;
}

// A row's decoded sub-blocks (K / 32 of them).
[[nodiscard]] inline std::vector<AffineSub> affine_row(QType format, const std::uint8_t* row, std::int32_t k) {
    std::vector<AffineSub> out(static_cast<std::size_t>(k / 32));
    for (std::size_t s = 0; s < out.size(); ++s) out[s] = affine_sub(format, row, static_cast<int>(s));
    return out;
}

// A sub-block's scale term and offset term, added to acc as both routes add them.
[[nodiscard]] inline float add_scale_term(float acc, const AffineSub& sub, const std::int8_t* a, float xd) {
    std::int32_t dot = 0;
    for (int l = 0; l < 32; ++l) dot += sub.q[l] * a[l];
    const float sx = sub.scale * xd;
    return std::fma(static_cast<float>(dot), sx, acc);
}

[[nodiscard]] inline float add_offset_term(float acc, const AffineSub& sub, float xm) {
    return std::fma(-sub.offset, xm, acc);
}

// The wide route's K tile: four sub-blocks (ggml_mmq.cuh kMmqK = 128).
inline constexpr int kWideTileSubs = 4;

enum class Route { Decode, Wide };

// The FP32 accumulator of row `row` (its decoded sub-blocks) and column t of the activation.
[[nodiscard]] inline float row_dot(Route route, std::span<const AffineSub> row, const Q81& x,
                                   std::span<const float> xm, std::int32_t k, std::int32_t t) {
    const int subs          = k / 32;
    const std::int8_t* a    = x.q.data() + static_cast<std::size_t>(t) * k;
    const float* xd         = x.d.data() + static_cast<std::size_t>(t) * subs;
    const float* xmt        = xm.data() + static_cast<std::size_t>(t) * subs;
    if (route == Route::Wide) {
        if (subs % kWideTileSubs != 0) { throw std::invalid_argument("row_dot: the wide route needs K % 128 == 0"); }
        float acc = 0.0F;
        for (int tile = 0; tile < subs; tile += kWideTileSubs) {
            for (int s = tile; s < tile + kWideTileSubs; ++s) acc = add_scale_term(acc, row[s], a + 32 * s, xd[s]);
            for (int s = tile; s < tile + kWideTileSubs; ++s) acc = add_offset_term(acc, row[s], xmt[s]);
        }
        return acc;
    }
    const int lanes = lanes_per_row(subs);
    std::array<float, 32> acc{};
    for (int lane = 0; lane < lanes; ++lane) {
        for (int s = lane; s < subs; s += lanes) {
            acc[lane] = add_offset_term(add_scale_term(acc[lane], row[s], a + 32 * s, xd[s]), row[s], xmt[s]);
        }
    }
    for (int offset = lanes / 2; offset > 0; offset >>= 1) {
        std::array<float, 32> next{};
        for (int lane = 0; lane < lanes; ++lane) next[lane] = acc[lane] + acc[lane ^ offset];
        acc = next;
    }
    return acc[t % lanes]; // every lane holds the same sum; lane t % lanes stores column t
}

// The BF16 output of an accumulator (residual: LinearAdd's old output word, else null).
[[nodiscard]] inline std::uint16_t store(float acc, const std::uint16_t* residual) {
    float y = acc;
    if (residual != nullptr) y += bf16_value(*residual);
    return bf16_bits(y);
}

// The magnitude that bounds a row and column's FP32 error: sum over k of (|scale q_k| + |offset|) *
// |a_k| * xd, in FP64. The routes add 2 K / 32 terms, each at most this sub-block's share, rounding
// each once (and scale * xd, xd * sum(a) once), and ggml's decode of a value rounds d1 * q and the
// difference once each, so the result is within (K / 16 + 4) * 2^-24 of this magnitude of the
// FP64 dot product of the exactly decoded row and the cast activation: K * 2^-24 of it is the
// documented bound (projection_fp32.h), with room to spare.
[[nodiscard]] inline double term_magnitude(std::span<const AffineSub> row, const Q81& x, std::int32_t k,
                                           std::int32_t t) {
    const int subs = k / 32;
    double sum     = 0.0;
    for (int s = 0; s < subs; ++s) {
        const std::int8_t* a = x.q.data() + static_cast<std::size_t>(t) * k + 32 * s;
        const double xd      = x.d[static_cast<std::size_t>(t) * subs + s];
        for (int l = 0; l < 32; ++l) {
            sum += (std::fabs(static_cast<double>(row[s].scale) * row[s].q[l]) + std::fabs(static_cast<double>(row[s].offset))) *
                   std::fabs(static_cast<double>(a[l])) * xd;
        }
    }
    return sum;
}

} // namespace ninfer::test::ggml_linear
