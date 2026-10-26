#pragma once

// Adapted from Basalt 8d73a39 src/kernels/cuda/head_proxy.cuh (MIT, third_party/basalt/LICENSE).
// Modified for NInfer-3090: Basalt's 2-bit group fit (fit_group, the 6-candidate scale search) as
// a header the host and the device share, in IEEE binary32 operations only (canonical_math.h), the
// binary16 scale held as its bits (the host-only build has no CUDA headers). Departures:
//   - the stored scale takes the full least-squares correction, scale x |w|^2 / <w, w_hat>, which
//     makes <w, w_hat'> = |w|^2 (Basalt multiplies by the square root of that ratio, "half the
//     correction", to keep a softmax over the copy near the exact one; nothing here reads a
//     probability from the copy);
//   - each group also stores its residual norm, so every row's approximate score comes with a
//     rigorous interval for the dense head's score (proxy_interval below), and the rows rescored
//     are chosen by those intervals, not by Basalt's "within 5 of the best approximate score";
//   - the copy is laid out for the int8 `mma.sync.m16n8k32` (16 rows by one 32-column group per
//     instruction) instead of Basalt's dp4a lane-per-row layout, and is built on the host at load,
//     from the row-split Q8 head, instead of by a kernel.
//
// The two-stage output head (ops::two_stage_head, --lm-head-two-stage). A row's weights are stored
// as one 2-bit level l in {0,1,2,3} per weight, one binary16 scale s per 32 consecutive weights,
// representing w_hat = (l - 1.5) * s, and one binary16 residual norm rho >= |w - w_hat|_2 per
// group. The int8 tensor cores read the level as the odd code c = 2l - 3 in {-3,-1,1,3} and the
// scale as s / 2. The activation is cast to int8 per 32-column group (quantize_activation_group).
// A row's approximate score is
//   a = sum over groups g, in ascending order, of  fma(s_g / 2 * d_g, P_g, acc),
//   P_g = sum_k c_{g,k} * q_{g,k}   (exact in int32, |P_g| <= 3 * 127 * 32),
// which the kernel and proxy_interval compute to the same bits.
//
// The interval (proxy_interval). Write e = sum_k w_k x_k for the exact dot product of the Q8 row
// (w, its FP32 dequantized values) with the BF16 activation x, a~ = sum_k w_hat_k x^_k for the
// copy's value in exact arithmetic (x^_k = q_k * d the cast activation), a for the FP32 value the
// kernel computes and D for the dense head's BF16 output (ops::linear). Per group g:
//   e - a~ = sum (w - w_hat) x + sum w_hat (x - x^), so
//   |e - a~| <= sum_g rho_g |x_g|_2 + |w_hat_g|_1 max|x_g - x^_g|               (Cauchy-Schwarz)
//            <= sum_g rho_g n_g + lambda_g h_g =: E,
// with n_g = |x_g|_2, lambda_g = |w_hat_g|_1 = s_g (16 + o_g) (o_g the number of outer levels 0
// and 3: |l - 1.5| is 1.5 for them and 0.5 for the others) and h_g = (1/2 + 2^-12) d_g: the cast
// rounds x / d to the nearest code, and the FP32 reciprocal, product and scale move x / d by at most
// 4 * 127 units of 2^-24 from x * (127 / amax), well under 2^-12. Every product in both sums is
// bounded by
//   sum_g S_g,   S_g = lambda_g amax_g + rho_g n_g  >=  sum_k |w_hat_k x^_k| and sum_k |w_k x_k|,
// (|w| <= |w_hat| + |w - w_hat|, |x^| <= amax(1 + 2^-23)), so with S = sum_g S_g:
//   |a - a~| <= gamma_copy S:  one rounding of the group coefficient and one of the fma per group,
//               gamma_copy = (G + 2) 2^-24 (1 + 2^-8);
//   |f - e|  <= gamma_dense S: the dense head's FP32 result f, whose weights may be rounded once to
//               BF16 (2^-8) and whose sums err by at most 4 units of 2^-24 each over K + 64
//               additions; gamma_dense = 2^-7 covers both for K <= 8192 (an assumption about the
//               tensor-core accumulation, stated here because it is the only one);
//   |D - f|  <= 2^-8 |f| <= 2^-8 (|a| + E + (gamma_copy + gamma_dense) S)   (BF16 rounding).
// So D lies in [a - B, a + B] with
//   B = E + (gamma_copy + gamma_dense) S + 2^-8 (|a| + E + (gamma_copy + gamma_dense) S).
// E, S and B are summed in FP32 from non-negative terms; the result is multiplied by 1 + 2^-10
// and gets 2^-20 |a| + 2^-126 added, which covers every FP32 rounding of E, S, B (relative error
// at most (2G + 8) 2^-24 < 2^-10 for G <= 256 groups, K <= 8192), of n_g and h_g, and of a - B
// and a + B. rho_g and the scale-derived lambda_g are stored rounded up, never down.

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
inline constexpr int kMaxColumns      = 8192; // the interval's gamma_dense covers K up to this
// The one head the exact stage has a route for: the dense head's registered vocabulary shape.
inline constexpr std::int32_t kHeadRows    = 248320;
inline constexpr std::int32_t kHeadColumns = 5120;
// The rescoring rule: every row whose interval can reach a lower bound on the 64th best dense
// score is rescored, with the block maxima binned at 8 per logit unit.
inline constexpr int kTopRows       = 64;
inline constexpr float kBinsPerUnit = 8.0F;
inline constexpr int kThresholdBins = 1024; // the histogram spans 1024 / bins_per_unit logits

// ---------------------------------------------------------------------------------------------
// Geometry. The copy of an [N,K] head (N a multiple of 16, K a multiple of 128, K <= 8192) is
// three planes:
//   codes:     N * K / 4 bytes, ordered [N/16 tiles][K/128 quads][32 lanes][4 groups] of 32-bit
//              words. Lane L = 4 * gid + tig holds, for group g, the word whose byte 0 is row
//              gid's columns 4tig..4tig+3, byte 1 row gid + 8's, byte 2 row gid's columns
//              16 + 4tig..16 + 4tig + 3 and byte 3 row gid + 8's; column i of a byte is its bits
//              2i..2i+1. These are the four A registers of mma.m16n8k32 (row-major A).
//   scales:    N * K / 32 binary16 values right after the codes, ordered
//              [N/16 tiles][K/128 quads][8 gid][2 halves][4 groups]: half 0 is row gid, half 1
//              row gid + 8, so a lane's scales for one quad are one 16-byte load.
//   residuals: N * K / 32 binary16 residual norms right after the scales, in the scales' order.
// ---------------------------------------------------------------------------------------------

[[nodiscard]] constexpr bool proxy_shape_supported(std::int64_t rows, std::int64_t columns) noexcept {
    return rows > 0 && columns > 0 && rows % kTileRows == 0 && columns % (kGroup * kQuadGroups) == 0 &&
           columns <= kMaxColumns && rows <= std::numeric_limits<std::int32_t>::max();
}

[[nodiscard]] constexpr std::uint64_t proxy_code_bytes(std::int64_t rows, std::int64_t columns) noexcept {
    return static_cast<std::uint64_t>(rows) * static_cast<std::uint64_t>(columns) / 4U;
}

[[nodiscard]] constexpr std::uint64_t proxy_scale_bytes(std::int64_t rows, std::int64_t columns) noexcept {
    return static_cast<std::uint64_t>(rows) * static_cast<std::uint64_t>(columns / kGroup) * 2U;
}

// The residual plane starts at proxy_residual_offset; both offsets are multiples of 512.
[[nodiscard]] constexpr std::uint64_t proxy_residual_offset(std::int64_t rows, std::int64_t columns) noexcept {
    return proxy_code_bytes(rows, columns) + proxy_scale_bytes(rows, columns);
}

[[nodiscard]] constexpr std::uint64_t proxy_bytes(std::int64_t rows, std::int64_t columns) noexcept {
    return proxy_residual_offset(rows, columns) + proxy_scale_bytes(rows, columns);
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

// Index (in binary16 values) of the scale (or residual) of (row, group) inside its plane.
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

// The smallest binary16 at or above a non-negative value (infinity past the range).
inline std::uint16_t f16_round_up(double value) {
    if (!(value >= 0.0)) { return 0x7E00U; }
    float f = static_cast<float>(value);
    if (static_cast<double>(f) < value) { f = std::nextafter(f, std::numeric_limits<float>::infinity()); }
    std::uint16_t bits = hc_int8::f32_to_f16_rn(f);
    if ((bits & 0x7C00U) != 0x7C00U && hc_int8::f16_to_f32(bits) < f) { ++bits; }
    return bits;
}

// The group's levels, the binary16 bits of its stored scale and of its residual norm. Basalt's
// search: of the six steps f16(amax / 1.5 * (1 - 0.1 t)), t = 0..5, the one (first on a tie) whose
// levels leave the least squared error; then the stored scale s' = f16(step * |w|^2 / <w, w_hat>)
// with w_hat = (level - 1.5) * step, kept at the step when <w, w_hat> is not positive or s' is not
// a finite binary16. The residual is |w - (level - 1.5) s'|_2, computed in FP64 and rounded up to
// binary16. Non-finite weights are not supported (a Q8 head has none).
inline void fit_group_2bit(const float* w, std::uint8_t* levels, std::uint16_t* scale,
                           std::uint16_t* residual) {
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
    *scale               = bits;
    const double stored  = hc_int8::f16_to_f32(bits);
    double squared       = 0.0;
    for (int i = 0; i < kGroup; ++i) {
        const double r = static_cast<double>(w[i]) - (static_cast<double>(levels[i]) - 1.5) * stored;
        squared += r * r;
    }
    // FP64 rounding of the 32 squares and the root is far below one binary16 step; one more ulp of
    // margin keeps the stored value an upper bound.
    *residual = f16_round_up(std::sqrt(squared) * (1.0 + 0x1p-40));
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
    auto* residual_plane      = proxy.data() + proxy_residual_offset(rows, columns);
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
            std::uint16_t stored   = 0;
            std::uint16_t residual = 0;
            fit_group_2bit(w, levels, &stored, &residual);
            for (int i = 0; i < kGroup; ++i) {
                int shift                = 0;
                const std::uint64_t byte = proxy_code_byte(row, g * kGroup + i, columns, &shift);
                proxy[byte] |= static_cast<std::byte>(levels[i] << shift);
            }
            const std::uint64_t index     = proxy_scale_index(row, g, columns);
            scale_plane[index * 2]        = static_cast<std::byte>(stored & 0xFFU);
            scale_plane[index * 2 + 1]    = static_cast<std::byte>(stored >> 8U);
            residual_plane[index * 2]     = static_cast<std::byte>(residual & 0xFFU);
            residual_plane[index * 2 + 1] = static_cast<std::byte>(residual >> 8U);
        }
    }
}

// The level of (row, column) and the scale or residual bits of (row, group) in a copy, for tests
// and the host oracle.
inline int proxy_level_at(std::span<const std::byte> proxy, std::int64_t row, std::int64_t column,
                          std::int64_t columns) {
    int shift                = 0;
    const std::uint64_t byte = proxy_code_byte(row, column, columns, &shift);
    return static_cast<int>((std::to_integer<unsigned>(proxy[byte]) >> shift) & 3U);
}

inline std::uint16_t proxy_half_at(std::span<const std::byte> proxy, std::uint64_t plane,
                                   std::int64_t row, std::int64_t group, std::int64_t columns) {
    const std::uint64_t index = proxy_scale_index(row, group, columns);
    const auto* base          = proxy.data() + plane;
    return static_cast<std::uint16_t>(std::to_integer<unsigned>(base[index * 2]) |
                                      (std::to_integer<unsigned>(base[index * 2 + 1]) << 8U));
}

inline std::uint16_t proxy_scale_at(std::span<const std::byte> proxy, std::int64_t rows,
                                    std::int64_t row, std::int64_t group, std::int64_t columns) {
    return proxy_half_at(proxy, proxy_code_bytes(rows, columns), row, group, columns);
}

inline std::uint16_t proxy_residual_at(std::span<const std::byte> proxy, std::int64_t rows,
                                       std::int64_t row, std::int64_t group, std::int64_t columns) {
    return proxy_half_at(proxy, proxy_residual_offset(rows, columns), row, group, columns);
}

// ---------------------------------------------------------------------------------------------
// The activation cast, the approximate score and its interval (host and device, the same bits).
// ---------------------------------------------------------------------------------------------

NINFER_HD float positive_infinity() { return canon::f32_from_bits(0x7F800000U); }

NINFER_HD float f32_sqrt(float x) {
#if defined(__CUDA_ARCH__)
    return __fsqrt_rn(x);
#else
    return std::sqrt(x);
#endif
}

// What one cast group carries into the score and its interval.
// Trivial (no member initializers) so kernels can stage it in shared memory; ActivationGroup{}
// is all zero.
struct ActivationGroup {
    float d;    // the scale: code q stands for q * d
    float h;    // (1/2 + 2^-12) d, a bound on |x - q d| (see the interval above)
    float norm; // |x_g|_2 in FP32
    float amax; // max |x_g|
};

// A group's scale from its largest magnitude a: d = a / 127, and the reciprocal step 127 / a the
// codes are taken at (both FP32, rounded to nearest even). d is 0 when a is 0 and NaN when a is
// not finite (the codes are then 0 and every score and bound of the column NaN: every row is
// rescored, as the exact head's NaN scores would demand).
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

// The group's bound terms once amax, d and the squares' FP32 sum (ascending order) are known.
NINFER_HD ActivationGroup activation_group(float amax, float sum_squares) {
    ActivationGroup out{};
    out.d    = activation_scale(amax);
    out.h    = canon::mul_rn(out.d, 0.500244140625F); // 1/2 + 2^-12
    out.norm = f32_sqrt(sum_squares);
    out.amax = amax;
    return out;
}

// One group of 32 FP32 activation values (BF16 values promoted exactly): its codes and terms.
NINFER_HD void quantize_activation_group(const float* x, std::int8_t* q, ActivationGroup* group) {
    float amax = 0.0F;
    float sum  = 0.0F;
    for (int i = 0; i < kGroup; ++i) {
        const float a = x[i] < 0.0F ? -x[i] : x[i];
        amax          = (a > amax || a != a) ? a : amax; // a NaN propagates
        sum           = canon::fma_rn(x[i], x[i], sum);
    }
    for (int i = 0; i < kGroup; ++i) { q[i] = activation_code(x[i], amax); }
    *group = activation_group(amax, sum);
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

// The three running sums of one (row, column): the score, E and S of the interval.
struct ProxyAccumulator {
    float score = 0.0F;
    float error = 0.0F;
    float terms = 0.0F;
};

// One group's step, in this order on the host and the device. `outer` counts the group's levels
// 0 and 3, so lambda = s (16 + outer) = |w_hat|_1 exactly in FP32 (11 by 6 significant bits).
NINFER_HD void proxy_accumulate(ProxyAccumulator& acc, std::uint16_t scale_bits,
                                std::uint16_t residual_bits, int outer, std::int32_t dot,
                                const ActivationGroup& x) {
    const float scale  = hc_int8::f16_to_f32(scale_bits);
    const float rho    = hc_int8::f16_to_f32(residual_bits);
    const float lambda = canon::mul_rn(scale, static_cast<float>(16 + outer));
    acc.score = canon::fma_rn(canon::mul_rn(canon::mul_rn(scale, 0.5F), x.d), static_cast<float>(dot),
                              acc.score);
    acc.error = canon::fma_rn(lambda, x.h, canon::fma_rn(rho, x.norm, acc.error));
    acc.terms = canon::fma_rn(rho, x.norm, canon::fma_rn(lambda, x.amax, acc.terms));
}

// The interval [lower, upper] that holds the dense head's BF16 score of the row (see above).
struct ProxyInterval {
    float score = 0.0F;
    float bound = 0.0F;
    float lower = 0.0F;
    float upper = 0.0F;
};

NINFER_HD ProxyInterval proxy_finish(const ProxyAccumulator& acc, int groups) {
    const float gamma = canon::add_rn(
        canon::mul_rn(static_cast<float>(groups + 2), 0x1.01p-24F), // 2^-24 (1 + 2^-8)
        0.0078125F);                                                                // 2^-7
    const float magnitude = acc.score < 0.0F ? -acc.score : acc.score;
    const float base      = canon::fma_rn(gamma, acc.terms, acc.error);
    const float rounded   = canon::fma_rn(0.00390625F, canon::add_rn(magnitude, base), base); // 2^-8
    float bound           = canon::mul_rn(rounded, 1.0009765625F);                             // 1 + 2^-10
    bound = canon::add_rn(canon::fma_rn(9.5367431640625e-07F, magnitude, bound),               // 2^-20
                          1.1754943508222875e-38F);                                            // 2^-126
    ProxyInterval out;
    out.score = acc.score;
    out.bound = bound;
    out.lower = canon::add_rn(acc.score, -bound);
    out.upper = canon::add_rn(acc.score, bound);
    return out;
}

// The number of outer levels (0 and 3) among the 32 of a group.
inline int proxy_outer_levels(const std::uint8_t* levels) {
    int outer = 0;
    for (int i = 0; i < kGroup; ++i) { outer += (levels[i] == 0 || levels[i] == 3) ? 1 : 0; }
    return outer;
}

// The approximate score and interval of one row of a copy against one column's cast activation
// (`q`: columns int8 codes; `x`: columns / 32 group terms). The kernel produces these bits.
inline ProxyInterval proxy_interval(std::span<const std::byte> proxy, std::int64_t rows,
                                    std::int64_t row, std::int64_t columns, const std::int8_t* q,
                                    const ActivationGroup* x) {
    ProxyAccumulator acc;
    std::uint8_t levels[kGroup];
    const auto groups = static_cast<int>(columns / kGroup);
    for (int g = 0; g < groups; ++g) {
        for (int i = 0; i < kGroup; ++i) {
            levels[i] = static_cast<std::uint8_t>(proxy_level_at(proxy, row, g * kGroup + i, columns));
        }
        proxy_accumulate(acc, proxy_scale_at(proxy, rows, row, g, columns),
                         proxy_residual_at(proxy, rows, row, g, columns), proxy_outer_levels(levels),
                         proxy_dot_exact(levels, q + g * kGroup), x[g]);
    }
    return proxy_finish(acc, groups);
}

// ---------------------------------------------------------------------------------------------
// The column threshold (host and device, the same bits).
// ---------------------------------------------------------------------------------------------
//
// Let every row's interval hold its dense score, and let the block maxima be the largest lower
// bounds of disjoint row blocks. Bin the finite maxima below the largest, m, at `bins_per_unit`
// bins per unit (bin floor((m - v) * bins_per_unit), kThresholdBins bins), and let j be the first
// bin at which the running count reaches k. Every maximum in bins 0..j is at least
// m - (j + 1) / bins_per_unit (up to the rounding of the bin's arithmetic), and they belong to k
// distinct rows, whose dense scores are at least their lower bounds: the k-th best dense score D_k
// is at least theta = m - (j + 3) / bins_per_unit, two bins below covering the FP32 rounding of
// the bin index and of theta for |m| < 2^16. Every row with a dense score of at least D_k then has
// an upper bound of at least theta, so selecting the rows whose upper bound reaches theta selects
// all of them; a row not selected has an upper bound, and a dense score, below theta <= D_k. With
// fewer than k maxima inside the histogram, or no finite maximum, theta is -inf (every row is
// selected).

// Bin of block maximum v below the column's largest maximum m, or -1 outside the histogram (a
// non-finite v, or one kThresholdBins bins or more below m).
NINFER_HD int threshold_bin(float m, float v, float bins_per_unit) {
    if (!(v > -positive_infinity()) || !(v < positive_infinity())) { return -1; }
    const float offset = canon::mul_rn(canon::add_rn(m, -v), bins_per_unit);
    if (!(offset < static_cast<float>(kThresholdBins))) { return -1; }
    return offset > 0.0F ? static_cast<int>(offset) : 0;
}

NINFER_HD float threshold_value(float m, int bin, float bins_per_unit) {
    return canon::add_rn(m, -canon::div_rn(static_cast<float>(bin + 3), bins_per_unit));
}

// The threshold of one column from its block maxima of lower bounds (see above).
inline float threshold_from_block_maxima(std::span<const float> block_max, int k, float bins_per_unit) {
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
        if (count >= k) { return threshold_value(m, bin, bins_per_unit); }
    }
    return -infinity;
}

} // namespace ninfer::ops::proxy_head
