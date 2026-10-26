#pragma once

// Adapted from Basalt 8d73a39 src/kernels/cuda/head_proxy.cuh (MIT, third_party/basalt/LICENSE).
// Modified for NInfer-3090: Basalt's 2-bit group fit (fit_group, the 6-candidate scale search) as
// a header the host and the device share, in IEEE binary32 operations only (canonical_math.h), the
// binary16 scale held as its bits (the host-only build has no CUDA headers). Three departures:
//   - the stored scale takes the full least-squares correction, scale x |w|^2 / <w, w_hat>, which
//     makes <w, w_hat'> = |w|^2 (Basalt multiplies by the square root of that ratio, "half the
//     correction", to keep a softmax over the copy near the exact one). Here no probability is ever
//     read from the copy; the copy only decides which rows are rescored, so an unbiased projection
//     is what the error margin wants;
//   - the copy is laid out for the int8 `mma.sync.m16n8k32` (16 rows by one 32-column group per
//     instruction) instead of Basalt's dp4a lane-per-row layout, and is built on the host at load,
//     from the row-split Q8 head, instead of by a kernel;
//   - which rows are rescored is not Basalt's "within 5 of the best approximate score": it is every
//     row at or above a column threshold that is a lower bound on the 64th best exact score
//     whenever each row's approximation error is within the margin (threshold_from_block_maxima).
//
// The two-stage output head (ops::two_stage_head, --lm-head-two-stage). A row's weights are stored
// as one 2-bit level l in {0,1,2,3} per weight and one binary16 scale s per 32 consecutive weights,
// representing (l - 1.5) * s; the int8 tensor cores read the level as the odd code c = 2l - 3 in
// {-3,-1,1,3} and the scale as s / 2. The activation is cast to int8 per 32-column group
// (quantize_activation_group). A row's approximate score is
//   sum over groups g, in ascending order, of  fma(s_g / 2 * d_g, P_g, acc),
//   P_g = sum_k c_{g,k} * q_{g,k}   (exact in int32, |P_g| <= 3 * 127 * 32),
// which the kernel and proxy_row_score compute to the same bits.

#include "ops/common/canonical_math.h"
#include "ops/hyper_connection/hc_int8_codec.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <vector>

#if !defined(NINFER_HD)
#    if defined(__CUDACC__)
#        define NINFER_HD __host__ __device__ __forceinline__
#    else
#        define NINFER_HD inline
#    endif
#endif

namespace ninfer::ops::proxy_head {

inline constexpr int kGroup           = 32; // weights per scale, columns per int8 MMA k-step
inline constexpr int kTileRows        = 16; // rows per MMA tile (one warp's rows)
inline constexpr int kQuadGroups      = 4;  // groups one lane reads with one 16-byte load
inline constexpr int kScaleCandidates = 6;  // Basalt's scale search
inline constexpr int kCodeMax         = 127;
// The default rescoring rule: every row whose approximate score reaches a lower bound on the 64th
// best exact score is rescored, with block maxima binned at 8 per logit unit and a margin of 4
// logits on every row's approximation error.
inline constexpr int kTopRows        = 64;
inline constexpr float kBinsPerUnit  = 8.0F;
inline constexpr float kMargin       = 4.0F;
inline constexpr int kThresholdBins  = 1024; // the histogram spans 1024 / bins_per_unit logits

// ---------------------------------------------------------------------------------------------
// Geometry. The copy of an [N,K] head (N a multiple of 16, K a multiple of 128) is two planes:
//   codes:  N * K / 4 bytes, ordered [N/16 tiles][K/128 quads][32 lanes][4 groups] of 32-bit
//           words. Lane L = 4 * gid + tig (gid = L / 4, tig = L % 4) holds, for group g,
//           the word whose byte 0 is row gid's columns 4tig..4tig+3, byte 1 row gid + 8's, byte 2
//           row gid's columns 16 + 4tig..16 + 4tig + 3 and byte 3 row gid + 8's; column i of a byte
//           is its bits 2i..2i+1. These are the four A registers of mma.m16n8k32 (row-major A).
//   scales: N * K / 32 binary16 values right after the codes, ordered
//           [N/16 tiles][K/128 quads][8 gid][2 halves][4 groups]: half 0 is row gid, half 1 row
//           gid + 8, so a lane's scales for one quad are one 16-byte load.
// ---------------------------------------------------------------------------------------------

[[nodiscard]] constexpr bool proxy_shape_supported(std::int64_t rows, std::int64_t columns) noexcept {
    return rows > 0 && columns > 0 && rows % kTileRows == 0 && columns % (kGroup * kQuadGroups) == 0 &&
           rows <= std::numeric_limits<std::int32_t>::max() &&
           columns <= std::numeric_limits<std::int32_t>::max();
}

[[nodiscard]] constexpr std::uint64_t proxy_code_bytes(std::int64_t rows, std::int64_t columns) noexcept {
    return static_cast<std::uint64_t>(rows) * static_cast<std::uint64_t>(columns) / 4U;
}

[[nodiscard]] constexpr std::uint64_t proxy_scale_bytes(std::int64_t rows, std::int64_t columns) noexcept {
    return static_cast<std::uint64_t>(rows) * static_cast<std::uint64_t>(columns / kGroup) * 2U;
}

// The whole copy; the scale plane starts at proxy_code_bytes (a multiple of 512).
[[nodiscard]] constexpr std::uint64_t proxy_bytes(std::int64_t rows, std::int64_t columns) noexcept {
    return proxy_code_bytes(rows, columns) + proxy_scale_bytes(rows, columns);
}

// Byte of the code plane holding (row, column), and the shift of its two bits.
NINFER_HD std::uint64_t proxy_code_byte(std::int64_t row, std::int64_t column, std::int64_t columns,
                                        int* shift) {
    const std::int64_t quads = columns / (kGroup * kQuadGroups);
    const std::int64_t tile  = row / kTileRows;
    const int r              = static_cast<int>(row % kTileRows);
    const int gid            = r % 8;
    const int half_row       = r / 8;
    const std::int64_t group = column / kGroup;
    const int k              = static_cast<int>(column % kGroup);
    const int tig            = (k % 16) / 4;
    const int byte_in_word   = (k / 16) * 2 + half_row;
    const int lane           = gid * 4 + tig;
    const std::int64_t word =
        ((tile * quads + group / kQuadGroups) * 32 + lane) * kQuadGroups + group % kQuadGroups;
    *shift = 2 * (k % 4);
    return static_cast<std::uint64_t>(word) * 4U + static_cast<std::uint64_t>(byte_in_word);
}

// Index (in binary16 values) of the scale of (row, group) inside the scale plane.
NINFER_HD std::uint64_t proxy_scale_index(std::int64_t row, std::int64_t group, std::int64_t columns) {
    const std::int64_t quads = columns / (kGroup * kQuadGroups);
    const std::int64_t tile  = row / kTileRows;
    const int r              = static_cast<int>(row % kTileRows);
    return static_cast<std::uint64_t>(
        (((tile * quads + group / kQuadGroups) * 8 + r % 8) * 2 + r / 8) * kQuadGroups +
        group % kQuadGroups);
}

// ---------------------------------------------------------------------------------------------
// The 2-bit fit of one 32-weight group (host, at load).
// ---------------------------------------------------------------------------------------------

// The level (0..3) of weight w at quantization step `step`: floor(w / step + 2), clamped. A zero
// step (a group too small for binary16) gives level 2 everywhere, whose value is 0.5 * 0 = 0.
inline int proxy_level(float w, float step) {
    if (!(step > 0.0F)) { return 2; }
    const float t = std::floor(canon::add_rn(canon::div_rn(w, step), 2.0F));
    return t < 0.0F ? 0 : (t > 3.0F ? 3 : static_cast<int>(t));
}

// The group's levels and the binary16 bits of its stored scale. Basalt's search: of the six steps
// f16(amax / 1.5 * (1 - 0.1 t)), t = 0..5, the one (first on a tie) whose levels leave the least
// squared error; then the stored scale s' = f16(step * |w|^2 / <w, w_hat>) with
// w_hat = (level - 1.5) * step, kept at the step when <w, w_hat> is not positive or s' is not a
// finite binary16. Non-finite weights are not supported (a Q8 head has none).
inline void fit_group_2bit(const float* w, std::uint8_t* levels, std::uint16_t* scale) {
    float amax = 1e-9F;
    for (int i = 0; i < kGroup; ++i) { amax = std::max(amax, std::fabs(w[i])); }
    const float base = canon::div_rn(amax, 1.5F);
    float best_s     = base;
    float best_e     = std::numeric_limits<float>::infinity();
    for (int t = 0; t < kScaleCandidates; ++t) {
        const float s = canon::mul_rn(
            base, canon::add_rn(1.0F, -canon::mul_rn(0.1F, static_cast<float>(t))));
        const float step = hc_int8::f16_to_f32(hc_int8::f32_to_f16_rn(s));
        float e          = 0.0F;
        for (int i = 0; i < kGroup; ++i) {
            const float level = static_cast<float>(proxy_level(w[i], step));
            const float r     = canon::add_rn(canon::mul_rn(canon::add_rn(level, -1.5F), step), -w[i]);
            e                 = canon::add_rn(e, canon::mul_rn(r, r));
        }
        if (e < best_e) {
            best_e = e;
            best_s = s;
        }
    }
    std::uint16_t bits = hc_int8::f32_to_f16_rn(best_s);
    const float step   = hc_int8::f16_to_f32(bits);
    float ww           = 0.0F;
    float wr           = 0.0F;
    for (int i = 0; i < kGroup; ++i) {
        const int level = proxy_level(w[i], step);
        levels[i]       = static_cast<std::uint8_t>(level);
        const float r   = canon::mul_rn(canon::add_rn(static_cast<float>(level), -1.5F), step);
        ww              = canon::add_rn(ww, canon::mul_rn(w[i], w[i]));
        wr              = canon::add_rn(wr, canon::mul_rn(w[i], r));
    }
    if (wr > 0.0F) {
        const std::uint16_t unshrunk =
            hc_int8::f32_to_f16_rn(canon::mul_rn(step, canon::div_rn(ww, wr)));
        if ((unshrunk & 0x7C00U) != 0x7C00U) { bits = unshrunk; }
    }
    *scale = bits;
}

// Writes the copy of rows [row_begin, row_end) of a row-split Q8_G32_FP16 [rows, columns] matrix
// (`q8_codes`: rows * columns int8 codes, row-major; `q8_scales`: rows * columns / 32 binary16
// little-endian scales, row-major) into `proxy` (proxy_bytes(rows, columns), which the caller
// zeroed: levels are OR-ed into their bytes). Rows are independent, so disjoint row ranges may be
// written concurrently.
inline void pack_proxy_rows(std::span<const std::byte> q8_codes, std::span<const std::byte> q8_scales,
                            std::int64_t rows, std::int64_t columns, std::span<std::byte> proxy,
                            std::int64_t row_begin, std::int64_t row_end) {
    if (!proxy_shape_supported(rows, columns) ||
        q8_codes.size() < static_cast<std::size_t>(rows * columns) ||
        q8_scales.size() < static_cast<std::size_t>(rows * (columns / kGroup) * 2) ||
        proxy.size() != proxy_bytes(rows, columns) || row_begin < 0 || row_end > rows ||
        row_begin > row_end) {
        throw std::invalid_argument("pack_proxy_rows: shape or buffer sizes do not match");
    }
    const std::int64_t groups = columns / kGroup;
    auto* scale_plane         = proxy.data() + proxy_code_bytes(rows, columns);
    float w[kGroup];
    std::uint8_t levels[kGroup];
    for (std::int64_t row = row_begin; row < row_end; ++row) {
        for (std::int64_t g = 0; g < groups; ++g) {
            const std::size_t scale_at = static_cast<std::size_t>((row * groups + g) * 2);
            const auto scale_bits      = static_cast<std::uint16_t>(
                std::to_integer<unsigned>(q8_scales[scale_at]) |
                (std::to_integer<unsigned>(q8_scales[scale_at + 1]) << 8U));
            const float s = hc_int8::f16_to_f32(scale_bits);
            for (int i = 0; i < kGroup; ++i) {
                const auto code = static_cast<std::int8_t>(std::to_integer<unsigned>(
                    q8_codes[static_cast<std::size_t>(row * columns + g * kGroup + i)]));
                w[i] = canon::mul_rn(static_cast<float>(code), s);
            }
            std::uint16_t stored = 0;
            fit_group_2bit(w, levels, &stored);
            for (int i = 0; i < kGroup; ++i) {
                int shift                = 0;
                const std::uint64_t byte = proxy_code_byte(row, g * kGroup + i, columns, &shift);
                proxy[byte] |= static_cast<std::byte>(levels[i] << shift);
            }
            const std::uint64_t index = proxy_scale_index(row, g, columns);
            scale_plane[index * 2]     = static_cast<std::byte>(stored & 0xFFU);
            scale_plane[index * 2 + 1] = static_cast<std::byte>(stored >> 8U);
        }
    }
}

// The level of (row, column) and the scale bits of (row, group) in a copy, for tests and tools.
inline int proxy_level_at(std::span<const std::byte> proxy, std::int64_t row, std::int64_t column,
                          std::int64_t columns) {
    int shift                = 0;
    const std::uint64_t byte = proxy_code_byte(row, column, columns, &shift);
    return static_cast<int>((std::to_integer<unsigned>(proxy[byte]) >> shift) & 3U);
}

inline std::uint16_t proxy_scale_at(std::span<const std::byte> proxy, std::int64_t rows,
                                    std::int64_t row, std::int64_t group, std::int64_t columns) {
    const std::uint64_t index = proxy_scale_index(row, group, columns);
    const auto* plane         = proxy.data() + proxy_code_bytes(rows, columns);
    return static_cast<std::uint16_t>(std::to_integer<unsigned>(plane[index * 2]) |
                                      (std::to_integer<unsigned>(plane[index * 2 + 1]) << 8U));
}

// ---------------------------------------------------------------------------------------------
// The activation cast and the approximate score (host and device, the same bits).
// ---------------------------------------------------------------------------------------------

NINFER_HD float positive_infinity() { return canon::f32_from_bits(0x7F800000U); }

// A group's scale from its largest magnitude a: d = a / 127, and the reciprocal step 127 / a the
// codes are taken at (both FP32, rounded to nearest even). d is 0 when a is 0 and NaN when a is
// not finite (the codes are then 0 and the group's scores NaN, as the exact head's would be).
NINFER_HD float activation_scale(float amax) {
    if (amax == 0.0F) { return 0.0F; }
    if (!(amax < positive_infinity())) { return canon::f32_from_bits(canon::kCanonicalNan); }
    return canon::div_rn(amax, static_cast<float>(kCodeMax));
}

NINFER_HD std::int8_t activation_code(float x, float amax) {
    if (!(amax > 0.0F) || !(amax < positive_infinity())) { return 0; }
    float q = canon::rint_rn(canon::mul_rn(x, canon::div_rn(static_cast<float>(kCodeMax), amax)));
    q       = q > static_cast<float>(kCodeMax) ? static_cast<float>(kCodeMax) : q;
    q       = q < -static_cast<float>(kCodeMax) ? -static_cast<float>(kCodeMax) : q;
    return static_cast<std::int8_t>(q);
}

// One group of 32 FP32 activation values (BF16 values promoted exactly): its codes and scale.
NINFER_HD void quantize_activation_group(const float* x, std::int8_t* q, float* d) {
    float amax = 0.0F;
    for (int i = 0; i < kGroup; ++i) {
        const float a = x[i] < 0.0F ? -x[i] : x[i];
        amax          = (a > amax || a != a) ? a : amax; // a NaN propagates
    }
    for (int i = 0; i < kGroup; ++i) { q[i] = activation_code(x[i], amax); }
    *d = activation_scale(amax);
}

// The exact integer dot product the int8 MMA computes for one group: sum of (2l - 3) * q.
NINFER_HD std::int32_t proxy_dot_exact(const std::uint8_t* levels, const std::int8_t* q) {
    std::int32_t sum = 0;
    for (int i = 0; i < kGroup; ++i) {
        sum += (2 * static_cast<std::int32_t>(levels[i]) - 3) * static_cast<std::int32_t>(q[i]);
    }
    return sum;
}

// The FP32 factor a group's integer dot product is scaled by: (s / 2) * d.
NINFER_HD float proxy_group_coefficient(std::uint16_t scale_bits, float d) {
    return canon::mul_rn(canon::mul_rn(hc_int8::f16_to_f32(scale_bits), 0.5F), d);
}

NINFER_HD float proxy_accumulate(float acc, float coefficient, std::int32_t dot) {
    return canon::fma_rn(coefficient, static_cast<float>(dot), acc);
}

// The approximate score of one row of a copy against one column's cast activation (`q`: columns
// int8 codes; `d`: columns / 32 scales). The kernel produces these bits.
inline float proxy_row_score(std::span<const std::byte> proxy, std::int64_t rows, std::int64_t row,
                             std::int64_t columns, const std::int8_t* q, const float* d) {
    float acc = 0.0F;
    std::uint8_t levels[kGroup];
    for (std::int64_t g = 0; g < columns / kGroup; ++g) {
        for (int i = 0; i < kGroup; ++i) {
            levels[i] = static_cast<std::uint8_t>(proxy_level_at(proxy, row, g * kGroup + i, columns));
        }
        const std::int32_t dot = proxy_dot_exact(levels, q + g * kGroup);
        acc = proxy_accumulate(acc, proxy_group_coefficient(proxy_scale_at(proxy, rows, row, g, columns),
                                                            d[g]),
                               dot);
    }
    return acc;
}

// ---------------------------------------------------------------------------------------------
// The column threshold (host and device, the same bits).
// ---------------------------------------------------------------------------------------------
//
// Let every row's approximate score a_r be within `margin` of its exact score e_r, and let the
// block maxima be the largest a_r of disjoint row blocks. Bin the finite maxima below the largest,
// m, at `bins_per_unit` bins per unit (bin floor((m - v) * bins_per_unit), kThresholdBins bins),
// and let j be the first bin at which the running count reaches k. Every maximum in bins 0..j is at
// least L = m - (j + 1) / bins_per_unit (up to the rounding of the bin's arithmetic), and they
// belong to k distinct rows, so the k-th best approximate score is at least L, the k-th best exact
// score E_k is at least L - margin, and every row with e_r >= E_k has a_r >= L - 2 * margin. The
// threshold is L - 2 * margin less two more bins, which cover the FP32 rounding of the bin index,
// of L and of the subtraction for |m| < 2^16. With fewer than k maxima inside the histogram, or
// no finite maximum, the threshold is -inf (every row is rescored).

// Bin of block maximum v below the column's largest maximum m, or -1 outside the histogram (a
// non-finite v, or one kThresholdBins bins or more below m).
NINFER_HD int threshold_bin(float m, float v, float bins_per_unit) {
    if (!(v > -positive_infinity()) || !(v < positive_infinity())) { return -1; }
    const float offset = canon::mul_rn(canon::add_rn(m, -v), bins_per_unit);
    if (!(offset < static_cast<float>(kThresholdBins))) { return -1; }
    return offset > 0.0F ? static_cast<int>(offset) : 0;
}

NINFER_HD float threshold_value(float m, int bin, float bins_per_unit, float margin) {
    const float below = canon::div_rn(static_cast<float>(bin + 3), bins_per_unit);
    return canon::add_rn(canon::add_rn(m, -below), -canon::mul_rn(2.0F, margin));
}

// The threshold of one column from its block maxima (see above).
inline float threshold_from_block_maxima(std::span<const float> block_max, int k, float bins_per_unit,
                                         float margin) {
    const float infinity = positive_infinity();
    float m              = -infinity;
    for (const float v : block_max) {
        if (v > m) { m = v; } // NaN never wins
    }
    if (!(m > -infinity) || !(m < infinity) || k < 1) { return -infinity; }
    std::vector<int> histogram(kThresholdBins, 0);
    for (const float v : block_max) {
        const int bin = threshold_bin(m, v, bins_per_unit);
        if (bin >= 0) { ++histogram[static_cast<std::size_t>(bin)]; }
    }
    int count = 0;
    for (int bin = 0; bin < kThresholdBins; ++bin) {
        count += histogram[static_cast<std::size_t>(bin)];
        if (count >= k) { return threshold_value(m, bin, bins_per_unit, margin); }
    }
    return -infinity;
}

} // namespace ninfer::ops::proxy_head
