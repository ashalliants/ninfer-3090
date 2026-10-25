#pragma once

// Test-owned host reference for the A4G64 activation representation (ninfer/ops/q4_a4_linear.h)
// and for the exact W4A4 product that q4_a4_linear evaluates.
//
// Everything here is written from the contract, not from the kernels: the quantiser is the formula
// in the header comment, applied one group at a time in host FP32; the packer is the row-split Q4
// nibble rule (storage-layouts.md Section 3.3); and the product decodes both operands to their
// represented values and sums them in FP64, with no knowledge of tiles, fragments or K order.

#include "ops/quantized_weight.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace ninfer::test::a4_g64 {

inline constexpr int kGroup        = 64;
inline constexpr int kGroupBytes   = kGroup / 2;
inline constexpr int kQmax         = 7;
inline constexpr std::uint16_t kF16MaxFinite = 0x7bffU; // 65504
inline constexpr std::uint16_t kF16Inf       = 0x7c00U;

// One token's activations quantised: `codes` holds K/2 bytes per token, tokens consecutive, and
// `scales` K/64 binary16 words per token. This is exactly the device layout of xq [K/2,T] and
// xs [K/64,T].
struct Activation {
    std::int32_t k = 0;
    std::int32_t t = 0;
    std::vector<std::uint8_t> codes;
    std::vector<std::uint16_t> scales;

    [[nodiscard]] std::int32_t groups() const { return k / kGroup; }

    [[nodiscard]] int code(std::int32_t column, std::int32_t token) const {
        const std::uint8_t byte =
            codes[static_cast<std::size_t>(token) * (k / 2) + static_cast<std::size_t>(column / 2)];
        const unsigned u = (column & 1) == 0 ? (byte & 0x0fU) : (byte >> 4);
        return (u & 0x8U) != 0 ? static_cast<int>(u) - 16 : static_cast<int>(u);
    }

    [[nodiscard]] float scale(std::int32_t group, std::int32_t token) const {
        return quantized_weight::detail::f16_to_f32(
            scales[static_cast<std::size_t>(token) * groups() + static_cast<std::size_t>(group)]);
    }

    // The represented activation value: code times its stored binary16 scale.
    [[nodiscard]] double value(std::int32_t column, std::int32_t token) const {
        return static_cast<double>(code(column, token)) *
               static_cast<double>(scale(column / kGroup, token));
    }
};

// Packs 64 signed codes (each in [-8,7]) in the Q4 row-split order: lane 2j in the low nibble of
// byte j, lane 2j+1 in the high nibble, two's complement.
inline void pack_group(const int* q, std::uint8_t* out) {
    for (int j = 0; j < kGroupBytes; ++j) {
        const unsigned lo = static_cast<unsigned>(q[2 * j]) & 0x0fU;
        const unsigned hi = static_cast<unsigned>(q[2 * j + 1]) & 0x0fU;
        out[j]            = static_cast<std::uint8_t>(lo | (hi << 4));
    }
}

// The quantiser of the contract, for one group of 64 FP32 values (the values an F16/BF16 input
// represents). Host FP32 arithmetic is IEEE binary32 with round-to-nearest-even here (x86-64 SSE,
// no fast-math), so every step below is the correctly rounded operation the contract names.
inline void quantize_group(const float* x, int* q, std::uint16_t& scale_bits) {
    float a = 0.0F;
    for (int i = 0; i < kGroup; ++i) { a = std::max(a, std::fabs(x[i])); }
    const float s = a / 7.0F;
    std::uint16_t d = quantized_weight::detail::f32_to_f16(s);
    if (d == kF16Inf) { d = kF16MaxFinite; }
    scale_bits = d;
    if (d == 0) {
        std::fill(q, q + kGroup, 0);
        return;
    }
    const float r = 1.0F / quantized_weight::detail::f16_to_f32(d);
    for (int i = 0; i < kGroup; ++i) {
        // |x * r| <= 7 * (1 + 2^-11) except when the scale saturated at 65504; clamping the product
        // first keeps lrint in range and cannot change the clamped code.
        const float p = std::clamp(x[i] * r, -8.0F, 8.0F);
        const long v  = std::lrint(p); // ties to even under the default rounding mode
        q[i]          = static_cast<int>(std::clamp(v, static_cast<long>(-kQmax),
                                                    static_cast<long>(kQmax)));
    }
}

// x holds T tokens of K values, token-major (the [K,T] tensor's own memory order).
inline Activation quantize(const std::vector<float>& x, std::int32_t k, std::int32_t t) {
    if (k <= 0 || t <= 0 || k % kGroup != 0 ||
        x.size() != static_cast<std::size_t>(k) * static_cast<std::size_t>(t)) {
        throw std::invalid_argument("a4_g64 reference: invalid activation geometry");
    }
    Activation out;
    out.k = k;
    out.t = t;
    out.codes.resize(static_cast<std::size_t>(t) * (k / 2));
    out.scales.resize(static_cast<std::size_t>(t) * (k / kGroup));
    int q[kGroup];
    const std::int32_t groups = k / kGroup;
    for (std::int32_t token = 0; token < t; ++token) {
        for (std::int32_t g = 0; g < groups; ++g) {
            const std::size_t index = static_cast<std::size_t>(token) * groups + g;
            quantize_group(x.data() + index * kGroup, q, out.scales[index]);
            pack_group(q, out.codes.data() + index * kGroupBytes);
        }
    }
    return out;
}

// One row of a Q4 weight decoded to its represented values (code times stored binary16 scale),
// which are exact in FP32.
inline std::vector<double> decode_weight_row(const quantized_weight::PackedWeight& w,
                                             std::int32_t row) {
    std::vector<double> out(static_cast<std::size_t>(w.weight.k));
    for (std::int32_t c = 0; c < w.weight.k; ++c) {
        out[static_cast<std::size_t>(c)] = quantized_weight::logical_weight_fp64(w, row, c);
    }
    return out;
}

// sum_k W[row,k] * A[k,token] over the represented values, accumulated in FP64. Every product is
// exact in FP64 (at most 30 significant bits); only the K-term sum rounds, at about 1e-16 relative.
inline double exact_dot(const std::vector<double>& weight_row, const Activation& a,
                        std::int32_t token) {
    double acc = 0.0;
    for (std::int32_t c = 0; c < a.k; ++c) {
        acc += weight_row[static_cast<std::size_t>(c)] * a.value(c, token);
    }
    return acc;
}

} // namespace ninfer::test::a4_g64
