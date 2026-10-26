#pragma once

// Host emulation of the int8 tensor-core decode route of the GGML linears (ggml_mma_decode.cuh,
// LinearPolicy::AllowA8GgmlDecodeMma, T <= 8), one IEEE operation at a time. The GPU suite
// (tests/ops/linear/test_ggml.cpp) checks the kernel against it bit for bit; the host-only
// ninfer_ggml_mma_emulation_test qualifies it against the FP64 oracle.
//
// The arithmetic of one row r and one activation column (codes q, one scale d_g per 32 values, the
// contract's Q8_1 cast) is canonical_ggml.h's:
//
//   1. Sub-block s (32 values) of the row decodes exactly to integer weights w_j, half multipliers
//      m_0 (values 0-15) and m_1 (values 16-31) and an FP32 scale c: W[r, 32 s + j] = c m w_j. The
//      affine K-quants (Q4_K, Q5_K) also carry an offset c_min m_min: W = c m w_j - c_min m_min.
//      Per format (d, dmin the block's binary16 scales):
//        Q8_0     c = d, m = 1, w = code                                 |w| <= 128
//        Q2_0     c = d, m = 1, w = code - 1                             |w| <= 2
//        IQ4_NL   c = d, m = 1, w = kvalues_iq4nl[code]                  |w| <= 127
//        IQ4_XS   c = d, m = ls - 32, w = kvalues_iq4nl[code]            |w| <= 127, |m| <= 32
//        IQ3_S    c = d, m = 1 + 2 nibble, w = +-grid                    |w| <= 15,  m <= 31
//        Q6_K     c = d, m_h = scales[2 s + h] (int8), w = q - 32        |w| <= 32,  |m| <= 128
//        Q4_K     c = d, m = sc, w = q (4 bits); c_min = dmin, m_min = m  w <= 15, m <= 63
//        Q5_K     as Q4_K with w = q (5 bits)                             w <= 31
//   2. The tensor cores form S_h = sum over the 16 values of half h of w_j q_j, an exact int32
//      (one m16n8k32 per sub-block, two m16n8k16 for Q6_K's halves): |S_0 + S_1| <= 32 * 128 * 127
//      < 2^22. P = m_0 S_0 + m_1 S_1 is exact too, |P| < 2^24, so float(P) is exact.
//   3. Lane partials: partial[l] = +0 for l < 32, then for s = 0, 1, ..., K/32 - 1 in order
//        partial[s % 32] = fma(c * d_s, float(P), partial[s % 32])
//      and for the affine formats, right after, with A = sum of the group's 32 codes,
//        partial[s % 32] = fma(c_min * d_s, float(-(m_min A)), partial[s % 32]).
//      Each product c * d_s is one FP32 rounding.
//   4. The row value is reduce_lanes of the 32 partials (o = 16, 8, 4, 2, 1). The output is that
//      value (+ the BF16 residual for LinearAdd) rounded to BF16, nearest even.
//
// A column's arithmetic depends on nothing but its own codes and scales, so the route is column
// invariant: the bits of a column do not depend on T or on the column's position.
//
// The kernel distributes this over a 16-row tile and eight warps: warp w takes sub-blocks w, w + 8,
// ... (one per 256-value chunk), so lane partials w, w + 8, w + 16 and w + 24 live in its registers
// and its first two butterfly steps (o = 16, 8) are in-register; the eight warps' sums then take the
// last three steps through shared memory. emulate_tile below follows that distribution, and the host
// test checks it equals emulate_row bit for bit.
//
// Host-only (no CUDA). The including translation unit is built without FP contraction
// (ninfer_op_oracle_options).

#include "core/weight.h"
#include "ggml-common-tables.h"
#include "ops/common/canonical_ggml.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::ggml_mma {

// The tile of the kernel: 16 rows (the m16 of the MMA), up to 8 columns (its n8), 8 warps, one
// sub-block per warp per chunk.
inline constexpr int kRows       = 16;
inline constexpr int kColumns    = 8;
inline constexpr int kWarps      = 8;
inline constexpr int kChunkSubs  = kWarps;
inline constexpr int kLaneClass  = canon::kLanes / kWarps; // the lane partials a warp holds: 4

// The formats the route takes: every format of the GGML linear registry (ggml_shapes.h).
[[nodiscard]] constexpr bool supported(QType format) {
    return format == QType::GGML_Q8_0 || format == QType::GGML_Q2_0 || format == QType::GGML_IQ4_NL ||
           format == QType::GGML_IQ4_XS || format == QType::GGML_IQ3_S || format == QType::GGML_Q6_K ||
           format == QType::GGML_Q4_K || format == QType::GGML_Q5_K;
}

[[nodiscard]] constexpr bool affine(QType format) {
    return format == QType::GGML_Q4_K || format == QType::GGML_Q5_K;
}

// One sub-block decoded exactly (step 1).
struct Sub {
    std::array<std::int32_t, 32> w{};
    std::int32_t m[2] = {0, 0};
    float c           = 0.0F;
    float c_min       = 0.0F; // affine formats only
    std::int32_t m_min = 0;
};

namespace detail {

[[nodiscard]] inline std::uint32_t u16(const std::uint8_t* p) {
    return static_cast<std::uint32_t>(p[0]) | (static_cast<std::uint32_t>(p[1]) << 8);
}

// Exact binary16 -> binary32 widening.
[[nodiscard]] inline float fp16(const std::uint8_t* p) {
    const std::uint32_t h        = u16(p);
    const std::uint32_t sign     = (h & 0x8000U) << 16;
    const std::uint32_t exponent = (h >> 10) & 0x1FU;
    const std::uint32_t mantissa = h & 0x3FFU;
    if (exponent == 0x1F) { return canon::f32_from_bits(sign | 0x7F800000U | (mantissa << 13)); }
    if (exponent != 0) { return canon::f32_from_bits(sign | ((exponent + 112) << 23) | (mantissa << 13)); }
    const float magnitude = static_cast<float>(mantissa) * 0x1p-24F; // exact
    return canon::f32_from_bits(sign | canon::f32_bits(magnitude));
}

[[nodiscard]] inline std::int32_t iq4(int code) { return ggml_tables::kvalues_iq4nl[code & 15]; }

} // namespace detail

[[nodiscard]] inline int block_values(QType format) {
    switch (format) {
    case QType::GGML_Q8_0:
    case QType::GGML_IQ4_NL: return 32;
    case QType::GGML_Q2_0: return 64;
    default: return 256;
    }
}

[[nodiscard]] inline int block_bytes(QType format) {
    switch (format) {
    case QType::GGML_Q8_0: return 34;
    case QType::GGML_IQ4_NL:
    case QType::GGML_Q2_0: return 18;
    case QType::GGML_IQ4_XS: return 136;
    case QType::GGML_IQ3_S: return 110;
    case QType::GGML_Q6_K: return 210;
    case QType::GGML_Q4_K: return 144;
    case QType::GGML_Q5_K: return 176;
    default: break;
    }
    throw std::invalid_argument("ggml_mma: unsupported format");
}

[[nodiscard]] inline std::size_t row_bytes(QType format, int k) {
    return static_cast<std::size_t>(k / block_values(format)) * static_cast<std::size_t>(block_bytes(format));
}

// Sub-block s (0-based along K) of a row of `format`.
[[nodiscard]] inline Sub decode_sub(QType format, const std::uint8_t* row, int s) {
    using detail::fp16;
    using detail::iq4;
    using detail::u16;
    Sub out;
    const int per_block = block_values(format) / 32;
    const std::uint8_t* b = row + static_cast<std::size_t>(s / per_block) * static_cast<std::size_t>(block_bytes(format));
    const int ib          = s % per_block;
    out.m[0] = out.m[1] = 1;
    switch (format) {
    case QType::GGML_Q8_0:
        out.c = fp16(b);
        for (int j = 0; j < 32; ++j) out.w[j] = static_cast<std::int8_t>(b[2 + j]);
        break;
    case QType::GGML_Q2_0: {
        out.c                  = fp16(b);
        const std::uint8_t* qs = b + 2 + 8 * ib;
        for (int j = 0; j < 32; ++j) out.w[j] = ((qs[j / 4] >> (2 * (j % 4))) & 3) - 1;
        break;
    }
    case QType::GGML_IQ4_NL:
        out.c = fp16(b);
        for (int j = 0; j < 16; ++j) {
            out.w[j]      = iq4(b[2 + j]);
            out.w[j + 16] = iq4(b[2 + j] >> 4);
        }
        break;
    case QType::GGML_IQ4_XS: {
        out.c        = fp16(b);
        const int ls = ((b[4 + ib / 2] >> (4 * (ib % 2))) & 15) | (static_cast<int>((u16(b + 2) >> (2 * ib)) & 3) << 4);
        out.m[0] = out.m[1]    = ls - 32;
        const std::uint8_t* qs = b + 8 + 16 * ib;
        for (int j = 0; j < 16; ++j) {
            out.w[j]      = iq4(qs[j]);
            out.w[j + 16] = iq4(qs[j] >> 4);
        }
        break;
    }
    case QType::GGML_IQ3_S: {
        out.c = fp16(b);
        out.m[0] = out.m[1] = 1 + 2 * ((b[106 + ib / 2] >> (4 * (ib % 2))) & 15);
        for (int l = 0; l < 8; ++l) { // values 4 l .. 4 l + 3: one grid word
            const int index          = b[2 + 8 * ib + l] | (((b[66 + ib] >> l) & 1) << 8);
            const std::uint32_t grid = ggml_tables::iq3s_grid[index];
            const int signs          = (b[74 + 4 * ib + l / 2] >> (4 * (l % 2))) & 15;
            for (int i = 0; i < 4; ++i) {
                const int magnitude = static_cast<int>((grid >> (8 * i)) & 0xFFU);
                out.w[4 * l + i]    = (signs >> i) & 1 ? -magnitude : magnitude;
            }
        }
        break;
    }
    case QType::GGML_Q6_K: {
        const int h = ib / 4, qd = ib % 4;
        out.c       = fp16(b + 208);
        out.m[0]    = static_cast<std::int8_t>(b[192 + 2 * ib]);
        out.m[1]    = static_cast<std::int8_t>(b[193 + 2 * ib]);
        for (int j = 0; j < 32; ++j) {
            const int low  = (b[64 * h + 32 * (qd % 2) + j] >> (4 * (qd / 2))) & 15;
            const int high = (b[128 + 32 * h + j] >> (2 * qd)) & 3;
            out.w[j]       = (low | (high << 4)) - 32;
        }
        break;
    }
    case QType::GGML_Q4_K:
    case QType::GGML_Q5_K: {
        // ggml's get_scale_min_k4 over the 12 scale bytes at b + 4.
        const std::uint8_t* q = b + 4;
        int sc = 0, mn = 0;
        if (ib < 4) {
            sc = q[ib] & 63;
            mn = q[ib + 4] & 63;
        } else {
            sc = (q[ib + 4] & 15) | ((q[ib - 4] >> 6) << 4);
            mn = (q[ib + 4] >> 4) | ((q[ib] >> 6) << 4);
        }
        const bool q5 = format == QType::GGML_Q5_K;
        out.c         = fp16(b);
        out.c_min     = fp16(b + 2);
        out.m[0] = out.m[1] = sc;
        out.m_min           = mn;
        const std::uint8_t* qs = b + (q5 ? 48 : 16) + 32 * (ib / 2);
        for (int j = 0; j < 32; ++j) {
            const int low = (qs[j] >> (4 * (ib % 2))) & 15;
            out.w[j]      = q5 ? low | (((b[16 + j] >> ib) & 1) << 4) : low;
        }
        break;
    }
    default: throw std::invalid_argument("ggml_mma: unsupported format");
    }
    return out;
}

// The exact int32 sums of the two halves (step 2): what one m16n8k32 (or two m16n8k16) gives.
struct HalfSums {
    std::int32_t s0 = 0, s1 = 0;
};

[[nodiscard]] inline HalfSums half_sums(const Sub& sub, const std::int8_t* q) {
    HalfSums out;
    for (int j = 0; j < 16; ++j) out.s0 += sub.w[j] * q[j];
    for (int j = 16; j < 32; ++j) out.s1 += sub.w[j] * q[j];
    return out;
}

[[nodiscard]] inline std::int32_t product(const Sub& sub, const std::int8_t* q) {
    const HalfSums h = half_sums(sub, q);
    return sub.m[0] * h.s0 + sub.m[1] * h.s1;
}

[[nodiscard]] inline std::int32_t code_sum(const std::int8_t* q) {
    std::int32_t sum = 0;
    for (int j = 0; j < 32; ++j) sum += q[j];
    return sum;
}

// Step 3 for one sub-block against its activation group (codes q, scale d).
[[nodiscard]] inline float accumulate(float partial, const Sub& sub, const std::int8_t* q, float d, bool is_affine) {
    partial = canon::fma_rn(canon::mul_rn(sub.c, d), static_cast<float>(product(sub, q)), partial);
    if (is_affine) {
        partial = canon::fma_rn(canon::mul_rn(sub.c_min, d), static_cast<float>(-(sub.m_min * code_sum(q))), partial);
    }
    return partial;
}

// The FP32 row value of one row and one column (steps 3-4): x_codes holds K codes and x_scales
// K / 32 scales of the column.
inline void emulate_row(QType format, const std::byte* row, const std::int8_t* x_codes, const float* x_scales, int k,
                        float* out) {
    if (!supported(format) || k <= 0 || k % block_values(format) != 0) {
        throw std::invalid_argument("ggml_mma_emulate: unsupported format or K");
    }
    const auto* bytes = reinterpret_cast<const std::uint8_t*>(row);
    const bool is_affine = affine(format);
    float partial[canon::kLanes] = {};
    for (int s = 0; s < k / 32; ++s) {
        const Sub sub = decode_sub(format, bytes, s);
        partial[s % canon::kLanes] = accumulate(partial[s % canon::kLanes], sub, x_codes + 32 * s, x_scales[s], is_affine);
    }
    *out = canon::reduce_lanes(partial);
}

// The kernel's distribution of one 16-row tile over its eight warps (see the header comment), for
// T <= 8 columns: x_codes [T][K], x_scales [T][K / 32], rows kRows consecutive rows of row_bytes
// each, out [T][kRows]. Absent columns (T..7) enter the MMA as zero codes with zero scales, as in the
// kernel; their outputs are dropped.
inline void emulate_tile(QType format, const std::byte* rows, const std::int8_t* x_codes, const float* x_scales, int k,
                         int t, float* out) {
    if (!supported(format) || k <= 0 || k % block_values(format) != 0 || t < 1 || t > kColumns) {
        throw std::invalid_argument("ggml_mma_emulate_tile: unsupported format, K or T");
    }
    const auto* bytes    = reinterpret_cast<const std::uint8_t*>(rows);
    const std::size_t rb = row_bytes(format, k);
    const int subs       = k / 32;
    const int chunks     = (subs + kChunkSubs - 1) / kChunkSubs;
    const bool is_affine = affine(format);
    static const std::int8_t zeros[32] = {};
    for (int r = 0; r < kRows; ++r) {
        for (int col = 0; col < kColumns; ++col) {
            // Each warp's four lane partials, by class (lane partial w + 8 class).
            float acc[kWarps][kLaneClass] = {};
            for (int c = 0; c < chunks; ++c) {
                for (int w = 0; w < kWarps; ++w) {
                    const int s = kChunkSubs * c + w;
                    if (s >= subs) { continue; }
                    const Sub sub        = decode_sub(format, bytes + r * rb, s);
                    const bool present   = col < t;
                    const std::int8_t* q = present ? x_codes + static_cast<std::size_t>(col) * k + 32 * s : zeros;
                    const float d        = present ? x_scales[static_cast<std::size_t>(col) * subs + s] : 0.0F;
                    acc[w][c % kLaneClass] = accumulate(acc[w][c % kLaneClass], sub, q, d, is_affine);
                }
            }
            // The in-register butterfly steps (o = 16, then 8), then the three across warps.
            float warp_sum[kWarps];
            for (int w = 0; w < kWarps; ++w) {
                const float p0 = canon::add_rn(acc[w][0], acc[w][2]);
                const float p1 = canon::add_rn(acc[w][1], acc[w][3]);
                warp_sum[w]    = canon::add_rn(p0, p1);
            }
            for (int o = kWarps / 2; o > 0; o >>= 1) {
                for (int i = 0; i < o; ++i) warp_sum[i] = canon::add_rn(warp_sum[i], warp_sum[i + o]);
            }
            if (col < t) { out[static_cast<std::size_t>(col) * kRows + r] = warp_sum[0]; }
        }
    }
}

} // namespace ninfer::ops::ggml_mma

namespace ninfer::ops {

// The emulated FP32 value of one row against one A8-cast activation column (ggml_mma::emulate_row).
inline void ggml_mma_emulate(QType format, const std::byte* row, const std::int8_t* x_codes, const float* x_scales,
                             int k, float* out) {
    ggml_mma::emulate_row(format, row, x_codes, x_scales, k, out);
}

} // namespace ninfer::ops
