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
    std::uint32_t bit19;
    std::uint32_t bit22;

    __device__ __forceinline__ static StreamDecodeConstants make() {
        StreamDecodeConstants c{0x00078000u, 0x00780000u, 0x3f800000u,
                                0x40000000u, 1u << 19,    1u << 22};
        // Opaque to the compiler, so it keeps them in registers instead of folding immediates.
        asm volatile(""
                     : "+r"(c.low_nibble), "+r"(c.high_nibble), "+r"(c.one), "+r"(c.two),
                       "+r"(c.bit19), "+r"(c.bit22));
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

// Subtracts 16 * sum(bit_i * x[i]) over eight weights, bit i of `bits` being the weight at k offset
// i (Q5's high plane). Even bits go to bit 19 under 1.0 (1 + b/16, times x' = 256x gives 16x per set
// bit) and odd bits to bit 22 under 2.0 (2 + b, times x' = 16x gives 16x per set bit), so every set
// bit costs one LOP3 and one FFMA and contributes exactly 16 * x. The constant part is
// sum_even(x') + 2 * sum_odd(x'), which the caller folds into its correction.
__device__ __forceinline__ float stream_sub_bits8(std::uint32_t bits, const float4& x0,
                                                  const float4& x1, const StreamDecodeConstants& c,
                                                  float acc) {
    const std::uint32_t b21 = bits << 21, b19 = bits << 19, b17 = bits << 17, b15 = bits << 15,
                        b13 = bits << 13;
    acc = fmaf(stream_lop_magic(b19, c.bit19, c.one), -x0.x, acc); // bit 0 -> 19
    acc = fmaf(stream_lop_magic(b21, c.bit22, c.two), -x0.y, acc); // bit 1 -> 22
    acc = fmaf(stream_lop_magic(b17, c.bit19, c.one), -x0.z, acc); // bit 2 -> 19
    acc = fmaf(stream_lop_magic(b19, c.bit22, c.two), -x0.w, acc); // bit 3 -> 22
    acc = fmaf(stream_lop_magic(b15, c.bit19, c.one), -x1.x, acc); // bit 4 -> 19
    acc = fmaf(stream_lop_magic(b17, c.bit22, c.two), -x1.y, acc); // bit 5 -> 22
    acc = fmaf(stream_lop_magic(b13, c.bit19, c.one), -x1.z, acc); // bit 6 -> 19
    acc = fmaf(stream_lop_magic(b15, c.bit22, c.two), -x1.w, acc); // bit 7 -> 22
    return acc;
}

} // namespace ninfer::ops::detail
