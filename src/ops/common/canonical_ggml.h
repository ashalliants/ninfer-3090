#pragma once

// Canonical A8 arithmetic of GGML block formats, shared by the host and device routes of a bit-exact
// contract (offloaded_sparse_moe over GGML expert records; the Q8_1 activation cast of the GGML
// linears). The method is Infernix's (src/ops/common/canonical_math.h at a3edb450: one header for
// both sides, IEEE operations rounded to nearest even, exact integer products); the codes are
// ggml's. Every function returns identical bits on the CPU and the GPU.
//
// The arithmetic of one row r of a GGML weight matrix W [N, K] against one activation column x [K]:
//
//   1. The A8 cast of x, per group g of 32 consecutive values (a8_quantize_*): a = max |x_j| (NaN
//      ignored), d_g = a / 127, q_j = clamp(rint(x_j * (127 / a)), -127, 127) with ties to even and
//      q = 0 for a NaN product, q = 0 and d_g = 0 when a = 0. Divisions and the product are FP32.
//   2. Each 32-value sub-block s of row r decodes exactly to integer weights w_j (|w_j| <= 43), two
//      integer half multipliers m_0 (values 0-15) and m_1 (values 16-31) and an FP32 scale c_s, with
//      W[r, 32 s + j] = c_s * m_{j / 16} * w_j exactly (ExpertSub, per format below).
//   3. P_s = m_0 sum_{j<16} w_j q_j + m_1 sum_{j>=16} w_j q_j, an exact int32.
//   4. Lane partials: partial[l] = +0 for l < 32, then for s = 0, 1, ..., K/32 - 1 in order
//      partial[s % 32] = fma(c_s * d_s, float(P_s), partial[s % 32]) (one FP32 product, one fma).
//   5. The row value is the butterfly of the 32 partials: for o = 16, 8, 4, 2, 1 each lane i adds
//      lane i ^ o (reduce_lanes). This is a warp's __shfl_xor reduction, lane 0's value.
//
// Steps 4-5 are ggml's own MMVQ order (lane-strided sub-blocks, then a warp butterfly) with the
// float steps pinned: a GPU warp computes a row directly, and a CPU route keeps 32 partials.

#include "ops/common/canonical_math.h"

#include <cstdint>

namespace ninfer::ops::canon {

inline constexpr int kA8Values = 32; // values per A8 group and per weight sub-block
inline constexpr int kLanes    = 32; // lane partials of a row sum

// |x| of a binary32 by its bits (NaN stays NaN and is ignored by a8_amax_step).
NINFER_CANON_HD float abs_bits(float x) { return f32_from_bits(f32_bits(x) & 0x7FFFFFFFU); }

// One step of the group maximum: a NaN magnitude never replaces the running maximum.
NINFER_CANON_HD float a8_amax_step(float amax, float x) {
    const float a = abs_bits(x);
    return a > amax ? a : amax;
}

NINFER_CANON_HD float a8_scale(float amax) { return div_rn(amax, 127.0F); }

NINFER_CANON_HD float a8_inverse(float amax) { return amax == 0.0F ? 0.0F : div_rn(127.0F, amax); }

// The code of one value: rint(x * inverse), saturated to [-127, 127] (only an overflowing 127 / a
// can exceed it), 0 for a NaN product.
NINFER_CANON_HD int a8_code(float x, float inverse) {
    const float r = mul_rn(x, inverse);
    if (r != r) { return 0; }
    if (r >= 127.0F) { return 127; }
    if (r <= -127.0F) { return -127; }
    return static_cast<int>(rint_rn(r));
}

// The A8 cast of one group of 32 values given as binary32: codes to q, returns d.
NINFER_CANON_HD float a8_quantize_f32(const float* v, std::int8_t* q) {
    float amax = 0.0F;
    for (int j = 0; j < kA8Values; ++j) { amax = a8_amax_step(amax, v[j]); }
    const float inverse = a8_inverse(amax);
    for (int j = 0; j < kA8Values; ++j) { q[j] = static_cast<std::int8_t>(a8_code(v[j], inverse)); }
    return a8_scale(amax);
}

// The A8 cast of one group of 32 BF16 values.
NINFER_CANON_HD float a8_quantize_bf16(const std::uint16_t* v, std::int8_t* q) {
    float amax = 0.0F;
    for (int j = 0; j < kA8Values; ++j) { amax = a8_amax_step(amax, bf16_to_f32(v[j])); }
    const float inverse = a8_inverse(amax);
    for (int j = 0; j < kA8Values; ++j) {
        q[j] = static_cast<std::int8_t>(a8_code(bf16_to_f32(v[j]), inverse));
    }
    return a8_scale(amax);
}

// Step 4 for one sub-block: the partial after adding sub-block s with scale c, activation scale d
// and exact product sum p.
NINFER_CANON_HD float accumulate(float partial, float c, float d, std::int32_t p) {
    return fma_rn(mul_rn(c, d), static_cast<float>(p), partial);
}

// Step 5 on 32 partials held in an array (the host form; a warp uses __shfl_xor_sync with the
// same offsets). Destroys `partial`.
NINFER_CANON_HD float reduce_lanes(float* partial) {
    for (int o = kLanes / 2; o > 0; o >>= 1) {
        for (int i = 0; i < o; ++i) { partial[i] = add_rn(partial[i], partial[i + o]); }
    }
    return partial[0];
}

// The SwiGLU product of the expert arithmetic: h = SiLU(g) * u in FP32.
NINFER_CANON_HD float swiglu_f32(float g, float u) { return mul_rn(silu_c(g), u); }

} // namespace ninfer::ops::canon
