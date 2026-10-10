#pragma once

// Adapted from Infernix a3edb450 src/ops/common/canonical_math.h (Apache-2.0).
// Modified for NInfer-3090: kept the IEEE helpers, BF16 conversions, exp and SiLU; dropped the
// E2M1/E4M3 codecs and the A4/A16 activation encodings, which GGML expert records do not use.
//
// Canonical floating-point functions shared by the host and device routes of a bit-exact
// contract (offloaded_sparse_moe). The CPU and GPU builds of this header return identical bits for
// identical inputs: only IEEE binary32 operations rounded to nearest even, integer operations and
// explicitly fused fma are used. Host translation units that include it are compiled without
// floating-point contraction or fast math (-ffp-contract=off -fno-fast-math, MSVC /fp:precise).

#include <cmath>
#include <cstdint>
#include <cstring>

#if defined(__CUDACC__)
#    define NINFER_CANON_HD __host__ __device__ __forceinline__
#else
#    define NINFER_CANON_HD inline
#endif

namespace ninfer::ops::canon {

NINFER_CANON_HD std::uint32_t f32_bits(float x) {
#if defined(__CUDA_ARCH__)
    return __float_as_uint(x);
#else
    std::uint32_t u;
    std::memcpy(&u, &x, sizeof u);
    return u;
#endif
}

NINFER_CANON_HD float f32_from_bits(std::uint32_t u) {
#if defined(__CUDA_ARCH__)
    return __uint_as_float(u);
#else
    float x;
    std::memcpy(&x, &u, sizeof x);
    return x;
#endif
}

NINFER_CANON_HD float mul_rn(float a, float b) {
#if defined(__CUDA_ARCH__)
    return __fmul_rn(a, b);
#else
    return a * b;
#endif
}

NINFER_CANON_HD float add_rn(float a, float b) {
#if defined(__CUDA_ARCH__)
    return __fadd_rn(a, b);
#else
    return a + b;
#endif
}

NINFER_CANON_HD float div_rn(float a, float b) {
#if defined(__CUDA_ARCH__)
    return __fdiv_rn(a, b);
#else
    return a / b;
#endif
}

NINFER_CANON_HD float fma_rn(float a, float b, float c) {
#if defined(__CUDA_ARCH__)
    return __fmaf_rn(a, b, c);
#else
    return std::fmaf(a, b, c); // correctly rounded on every supported C++ runtime
#endif
}

// Round-to-nearest-even integer of a binary32 with |x| < 2^22.
NINFER_CANON_HD float rint_rn(float x) {
    const float magic = 12582912.0F; // 1.5 * 2^23
    return add_rn(add_rn(x, magic), -magic);
}

NINFER_CANON_HD float bf16_to_f32(std::uint16_t h) { return f32_from_bits(std::uint32_t{h} << 16); }

// Round-to-nearest-even binary32 -> bfloat16. Every NaN becomes the canonical quiet NaN 0x7FC0:
// NaN payloads and signs differ between x86 and CUDA arithmetic, so they are not part of the bits.
NINFER_CANON_HD std::uint16_t f32_to_bf16_rn(float x) {
    const std::uint32_t u = f32_bits(x);
    if ((u & 0x7FFFFFFFU) > 0x7F800000U) { return 0x7FC0U; }
    const std::uint32_t lsb = (u >> 16) & 1U;
    return static_cast<std::uint16_t>((u + 0x7FFFU + lsb) >> 16);
}

// The one NaN the transcendental functions return: CUDA arithmetic produces a canonical NaN where
// x86 propagates the input payload, so NaN inputs are mapped explicitly to keep the bits equal.
inline constexpr std::uint32_t kCanonicalNan = 0x7FC00000U;

// exp(x) with Cody-Waite reduction and a fixed degree-7 Taylor polynomial evaluated by explicit fma.
// Overflow returns +inf, deep underflow +0, and NaN the canonical NaN.
NINFER_CANON_HD float exp_c(float x) {
    if (x != x) { return f32_from_bits(kCanonicalNan); }
    if (x > 88.72283935546875F) { return f32_from_bits(0x7F800000U); }
    if (x < -103.97208404541015625F) { return 0.0F; }
    const float n  = rint_rn(mul_rn(x, 1.44269502162933349609375F));
    const float r0 = fma_rn(-n, 0.693145751953125F, x);         // ln2 high part (11 bits)
    const float r  = fma_rn(-n, 1.428606765330187045e-06F, r0); // ln2 low part
    float p        = 1.0F / 5040.0F;
    p              = fma_rn(p, r, 1.0F / 720.0F);
    p              = fma_rn(p, r, 1.0F / 120.0F);
    p              = fma_rn(p, r, 1.0F / 24.0F);
    p              = fma_rn(p, r, 1.0F / 6.0F);
    p              = fma_rn(p, r, 0.5F);
    p              = fma_rn(p, r, 1.0F);
    p              = fma_rn(p, r, 1.0F);
    // Scale by 2^n in two exact power-of-two steps so subnormal results are rounded once.
    const int k    = static_cast<int>(n);
    const int k1   = k / 2;
    const int k2   = k - k1;
    const float s1 = f32_from_bits(static_cast<std::uint32_t>(k1 + 127) << 23);
    const float s2 = f32_from_bits(static_cast<std::uint32_t>(k2 + 127) << 23);
    return mul_rn(mul_rn(p, s1), s2);
}

// SiLU(g) on a binary32, in the form that never overflows exp:
//   g >= 0: g / (1 + exp_c(-g));   g < 0: (g * e) / (1 + e), e = exp_c(g).
// A NaN result (a NaN input, or -inf * 0 at g = -inf) is the canonical NaN.
NINFER_CANON_HD float silu_c(float g) {
    float r;
    if (g >= 0.0F) {
        r = div_rn(g, add_rn(1.0F, exp_c(-g)));
    } else {
        const float e = exp_c(g);
        r             = div_rn(mul_rn(g, e), add_rn(1.0F, e));
    }
    return r != r ? f32_from_bits(kCanonicalNan) : r;
}

} // namespace ninfer::ops::canon
