#pragma once

// Adapted from Basalt 8d73a39 src/kernels/cuda/hc_fused.cu (MIT, third_party/basalt/LICENSE).
// Modified for NInfer-3090: Basalt's hc8_quant_kernel group rule as a header the host and the
// device share, in IEEE binary32 operations only (canonical_math.h), with the binary16 scale held
// as its bits (the host-only build has no CUDA headers). Two departures: a non-finite value or a
// scale past binary16's range refuses the group instead of writing codes, and a scale whose
// rounding leaves the group's largest value more than half a step past code 127 (only possible
// for a binary16 subnormal scale; Basalt writes s = 0 and codes 0 when it rounds to zero) is
// raised to the next binary16, so every value is within half a step of its code's value.
//
// The int8 copy of a hyper-connection matrix (hyper_connection_mix's int8 form, --hc-int8): each
// row's values in groups of kHcInt8Group consecutive values, each group one binary16 scale s and
// one code q in [-127, 127] per value, representing q * s. This is a runtime copy made from the
// BF16 weights at load, not the persistent q8_g32_fp16 format (whose encoder selects codes by a
// reciprocal multiply and never writes a zero scale for a nonzero group).

#include "ops/common/canonical_math.h"

#include <cstdint>

#if defined(__CUDACC__)
#    define NINFER_HD __host__ __device__ __forceinline__
#else
#    define NINFER_HD inline
#endif

namespace ninfer::ops::hc_int8 {

inline constexpr int kGroup         = 32;  // values per scale
inline constexpr int kCodeMax       = 127; // codes lie in [-kCodeMax, kCodeMax]
inline constexpr std::uint16_t kInf = 0x7C00U;

// Round-to-nearest-even binary32 -> binary16 bits. Overflow gives infinity, a NaN 0x7E00.
NINFER_HD std::uint16_t f32_to_f16_rn(float x) {
    const std::uint32_t u    = canon::f32_bits(x);
    const std::uint32_t sign = (u >> 16) & 0x8000U;
    const std::uint32_t a    = u & 0x7FFFFFFFU;
    if (a > 0x7F800000U) { return static_cast<std::uint16_t>(sign | 0x7E00U); }
    if (a >= 0x477FF000U) {
        return static_cast<std::uint16_t>(sign | kInf);
    } // >= 65520 rounds to inf
    if (a >= 0x38800000U) {
        // Normal: rebias the exponent (127 -> 15) and round the 13 dropped mantissa bits.
        const std::uint32_t lsb = (a >> 13) & 1U;
        return static_cast<std::uint16_t>(sign | (((a + 0xFFFU + lsb) >> 13) - (112U << 10)));
    }
    // Subnormal (or zero): the value in units of 2^-24, rounded to nearest even.
    const int exponent = static_cast<int>(a >> 23);
    const int shift    = 126 - exponent; // >= 14 here
    if (shift > 24) {
        return static_cast<std::uint16_t>(sign);
    } // below 2^-25 (or exactly it): zero
    const std::uint32_t mantissa = (a & 0x7FFFFFU) | 0x800000U;
    const std::uint32_t q        = mantissa >> shift;
    const std::uint32_t rest     = mantissa & ((1U << shift) - 1U);
    const std::uint32_t half     = 1U << (shift - 1);
    return static_cast<std::uint16_t>(
        sign | (q + ((rest > half || (rest == half && (q & 1U) != 0)) ? 1U : 0U)));
}

// binary16 bits -> binary32, exactly.
NINFER_HD float f16_to_f32(std::uint16_t h) {
    const std::uint32_t sign     = (std::uint32_t{h} & 0x8000U) << 16;
    const std::uint32_t exponent = (std::uint32_t{h} >> 10) & 0x1FU;
    const std::uint32_t mantissa = std::uint32_t{h} & 0x3FFU;
    if (exponent == 0) {
        const float v =
            canon::mul_rn(static_cast<float>(mantissa), 5.9604644775390625e-08F); // 2^-24
        return sign != 0 ? -v : v;
    }
    if (exponent == 31) { return canon::f32_from_bits(sign | 0x7F800000U | (mantissa << 13)); }
    return canon::f32_from_bits(sign | ((exponent + 112U) << 23) | (mantissa << 13));
}

// The value a code represents: q * s, exact in binary32 (8 x 11 significant bits).
NINFER_HD float dequant(std::int8_t q, std::uint16_t s) {
    return canon::mul_rn(static_cast<float>(q), f16_to_f32(s));
}

// One group of kGroup BF16 values (their bits): its scale and codes.
//   a = max |w|;  s = f16(f32(a / 127)), raised to the next binary16 when 127.5 * s < a;
//   q = clamp(rint(f32(w / s)), -127, 127), every q = 0 when s = 0 (exactly when a = 0).
// f32() and f16() round to nearest even and rint rounds half to even. Returns false, with every
// code and the scale zero, when a value is not finite or s would be infinite
// (f32(a / 127) >= 65520).
NINFER_HD bool quantize_group(const std::uint16_t* w, std::int8_t* q, std::uint16_t* s) {
    float amax  = 0.0F;
    bool finite = true;
    for (int i = 0; i < kGroup; ++i) {
        const std::uint16_t magnitude = static_cast<std::uint16_t>(w[i] & 0x7FFFU);
        finite                        = finite && magnitude < 0x7F80U;
        const float a                 = canon::bf16_to_f32(magnitude);
        amax                          = a > amax ? a : amax;
    }
    std::uint16_t scale =
        finite ? f32_to_f16_rn(canon::div_rn(amax, static_cast<float>(kCodeMax))) : kInf;
    // At a normal scale the rounding leaves a / s <= 127 * (1 + 2^-11); only a subnormal one can
    // leave the largest value more than half a step past 127 (and a zero one leaves it unencoded).
    if (scale < kInf && canon::mul_rn(f16_to_f32(scale), 127.5F) < amax) { ++scale; }
    if (scale >= kInf) {
        for (int i = 0; i < kGroup; ++i) { q[i] = 0; }
        *s = 0;
        return false;
    }
    *s             = scale;
    const float sf = f16_to_f32(scale);
    for (int i = 0; i < kGroup; ++i) {
        float code = 0.0F;
        if (sf > 0.0F) {
            code = canon::rint_rn(canon::div_rn(canon::bf16_to_f32(w[i]), sf)); // |w / s| <= 127.5
            code = code > static_cast<float>(kCodeMax) ? static_cast<float>(kCodeMax) : code;
            code = code < -static_cast<float>(kCodeMax) ? -static_cast<float>(kCodeMax) : code;
        }
        q[i] = static_cast<std::int8_t>(code);
    }
    return true;
}

} // namespace ninfer::ops::hc_int8
