#pragma once

// Decode primitives for the streaming GEMV (ops/linear/common/stream_gemv.cuh).
//
// The streaming GEMV turns each stored integer field into an fp32 operand with ONE three-input
// LOP3 and no conversion instruction: it masks the field into the top of an fp32 mantissa under a
// fixed exponent, so a 4-bit field n at bits 15..18 reads as 1 + n/256 and at bits 19..22 as
// 1 + n/16. The constant part and the per-position scale are folded into the activation instead,
// which the GEMV stages once per CTA for every row it streams:
//
//   x'[k] = x[k] * kStreamEvenScale   (even k)        x'[k] = x[k] * kStreamOddScale   (odd k)
//
// so sum_k (1 + n_k / s_k) * x'[k] = sum_k x'[k] + sum_k n_k * x[k], and the GEMV subtracts a
// per-32-weight constant that the format derives from the same activation. Integer-to-float and
// half-to-float conversions run at a quarter of the FP32 rate on sm_70, and they, not the weight
// stream, bounded the earlier decode GEMVs there; this form has none.
//
// Every value here is exact in fp32: the fields are small integers, and the scales are powers of
// two. What it gives up is the cancellation of the folded constant, about 4-5 of fp32's 24 mantissa
// bits for these formats (relative error ~1e-6 against an fp64 reference on random weights).

#include <cstdint>

namespace ninfer::ops::detail {

inline constexpr float kStreamEvenScale = 256.0f; // 2^8: a field at bits 15.. reads as 1 + n / 2^8
inline constexpr float kStreamOddScale  = 16.0f;  // 2^4: a field at bits 19.. reads as 1 + n / 2^4

// (a & mask) | magic in one LOP3. Written in PTX because the compiler splits the expression into an
// AND and an OR whenever both constants are immediates (one LOP3 takes at most one), which doubles
// the decode's instruction count.
__device__ __forceinline__ float stream_lop_magic(std::uint32_t a, std::uint32_t mask,
                                                  std::uint32_t magic) {
    std::uint32_t d;
    asm("lop3.b32 %0, %1, %2, %3, 0xEA;" : "=r"(d) : "r"(a), "r"(mask), "r"(magic));
    return __uint_as_float(d);
}

// Masks and magic exponents, held in registers so that every decode is a single LOP3.
struct StreamDecodeConstants {
    std::uint32_t low_nibble;  // bits 15..18
    std::uint32_t high_nibble; // bits 19..22
    std::uint32_t one;         // fp32 1.0
    std::uint32_t two;         // fp32 2.0
    std::uint32_t four;        // fp32 4.0
    std::uint32_t bit19;
    std::uint32_t bit22;

    __device__ __forceinline__ static StreamDecodeConstants make() {
        StreamDecodeConstants c{0x00078000u, 0x00780000u, 0x3f800000u, 0x40000000u,
                                0x40800000u, 1u << 19,    1u << 22};
        // Opaque to the compiler, so it keeps them in registers instead of folding immediates.
        asm volatile(""
                     : "+r"(c.low_nibble), "+r"(c.high_nibble), "+r"(c.one), "+r"(c.two),
                       "+r"(c.four), "+r"(c.bit19), "+r"(c.bit22));
        return c;
    }
};

// sum over eight weights of nibble * x', for one packed code word: nibble i (bits 4i..4i+3) is the
// weight at k offset i, and x0 / x1 hold x' for offsets 0..3 / 4..7. Returns
// sum(x') + sum(nibble * x) over the eight, the caller folding sum(x') into its correction.
__device__ __forceinline__ float stream_dot_nibbles8(std::uint32_t word, const float4& x0,
                                                     const float4& x1,
                                                     const StreamDecodeConstants& c, float acc) {
    // One shift places two nibbles: nibble 2j at bits 15..18 and nibble 2j+1 at bits 19..22.
    const std::uint32_t s0 = word << 15, s1 = word << 7, s2 = word >> 1, s3 = word >> 9;
    acc = fmaf(stream_lop_magic(s0, c.low_nibble, c.one), x0.x, acc);
    acc = fmaf(stream_lop_magic(s0, c.high_nibble, c.one), x0.y, acc);
    acc = fmaf(stream_lop_magic(s1, c.low_nibble, c.one), x0.z, acc);
    acc = fmaf(stream_lop_magic(s1, c.high_nibble, c.one), x0.w, acc);
    acc = fmaf(stream_lop_magic(s2, c.low_nibble, c.one), x1.x, acc);
    acc = fmaf(stream_lop_magic(s2, c.high_nibble, c.one), x1.y, acc);
    acc = fmaf(stream_lop_magic(s3, c.low_nibble, c.one), x1.z, acc);
    acc = fmaf(stream_lop_magic(s3, c.high_nibble, c.one), x1.w, acc);
    return acc;
}

// Adds or subtracts Scale * sum(bit_i * x[i]) over eight weights, where the bit of the weight at
// k offset i sits at bit Stride * i + Offset of `bits` (Q5's high plane: stride 1; Q6's two high
// planes: stride 2, offsets 0 and 1). Scale is 16 or 32.
//
// Each bit is shifted to bit 19 (even i) or bit 22 (odd i) and read under a magic exponent:
// 1 + b/16 and 2 + b against x' = 256x and 16x give 16x per set bit; doubling the magics (2.0 and
// 4.0) gives 32x. Every set bit is one LOP3 and one FFMA, and shifts shared between weights are
// computed once. The constant part, Mult * (sum_even(x') + 2 sum_odd(x')) with Mult = Scale / 16,
// enters with the same sign and is folded into the caller's correction.
template <int Stride, int Offset, int Scale, bool Subtract>
__device__ __forceinline__ float stream_bits8(std::uint32_t bits, const float4& x0, const float4& x1,
                                              const StreamDecodeConstants& c, float acc) {
    static_assert(Scale == 16 || Scale == 32, "magics exist for 16x and 32x per set bit");
    constexpr bool kDouble = Scale == 32;
    const float xs[8]      = {x0.x, x0.y, x0.z, x0.w, x1.x, x1.y, x1.z, x1.w};
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        constexpr int kEvenTarget = 19, kOddTarget = 22;
        const int target          = (i & 1) ? kOddTarget : kEvenTarget;
        const int shift           = target - (Stride * i + Offset); // folded per unrolled i
        const std::uint32_t moved = shift >= 0 ? bits << shift : bits >> -shift;
        const std::uint32_t mask  = (i & 1) ? c.bit22 : c.bit19;
        const std::uint32_t magic =
            kDouble ? ((i & 1) ? c.four : c.two) : ((i & 1) ? c.two : c.one);
        acc = fmaf(stream_lop_magic(moved, mask, magic), Subtract ? -xs[i] : xs[i], acc);
    }
    return acc;
}

} // namespace ninfer::ops::detail
