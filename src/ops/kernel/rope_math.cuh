#pragma once

// ninfer::ops - the generic RoPE route's per-pair math (rope_generic_kernel), shared with the
// fused unit-offset norm + RoPE of attention's q and k (rmsnorm_pair_rope) so both compile the
// same expressions.

#include <cstdint>

namespace ninfer::ops {

__device__ __forceinline__ void generic_axis_frequency(int axes, int head_dim, int rotary_dim,
                                                       int pair, int* axis, float* exponent) {
    if (axes == 2 && head_dim == 72 && rotary_dim == 72) {
        *axis           = pair / 18;
        const int local = pair % 18;
        *exponent       = -2.0F * static_cast<float>(local) / 36.0F;
    } else {
        *axis     = axes == 3 ? pair % 3 : 0;
        *exponent = -2.0F * static_cast<float>(pair) / static_cast<float>(rotary_dim);
    }
}

// sin and cos of pair `pair`'s angle for `token` (positions [tokens, axes], axis-major).
__device__ __forceinline__ void generic_sincos(const std::int32_t* positions, int axes, int head_dim,
                                               int rotary_dim, float theta, int tokens, int token, int pair,
                                               float* sin_out, float* cos_out) {
    int axis       = 0;
    float exponent = 0.0F;
    generic_axis_frequency(axes, head_dim, rotary_dim, pair, &axis, &exponent);
    const float frequency = powf(theta, exponent);
    const float angle = static_cast<float>(positions[static_cast<std::int64_t>(axis) * tokens + token]) * frequency;
    sincosf(angle, sin_out, cos_out);
}

// The rotated pair (first, second) = (x[i], x[i + R/2]).
// The contraction is spelled out (rope_generic_kernel's own): left to the compiler, another
// kernel can fuse the other product and round differently.
__device__ __forceinline__ float rope_first(float first, float second, float c, float s) {
    return __fmaf_rn(first, c, -__fmul_rn(second, s));
}
__device__ __forceinline__ float rope_second(float first, float second, float c, float s) {
    return __fmaf_rn(second, c, __fmul_rn(first, s));
}

} // namespace ninfer::ops
