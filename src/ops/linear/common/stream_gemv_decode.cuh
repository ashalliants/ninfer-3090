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
    std::uint32_t sixteen;     // fp32 16.0
    std::uint32_t byte;        // bits 15..22
    std::uint32_t bit19;
    std::uint32_t bit22;

    __device__ __forceinline__ static StreamDecodeConstants make() {
        StreamDecodeConstants c{0x00078000u, 0x00780000u, 0x3f800000u, 0x40000000u, 0x40800000u,
                                0x41800000u, 0x007f8000u, 1u << 19,    1u << 22};
        // Opaque to the compiler, so it keeps them in registers instead of folding immediates.
        asm volatile(""
                     : "+r"(c.low_nibble), "+r"(c.high_nibble), "+r"(c.one), "+r"(c.two),
                       "+r"(c.four), "+r"(c.sixteen), "+r"(c.byte), "+r"(c.bit19), "+r"(c.bit22));
        return c;
    }
};

// A format decodes each packed word of a lane slice into eight fp32 operands f[i] (weight at k
// offset i), and the GEMV forms sum_i f[i] * x'[i] for every token it streams: one decode serves
// all of them. The helpers below build f.

// f[i] = the magic float of nibble i (bits 4i..4i+3) of `word`: 1 + n/256 for even i, 1 + n/16 for
// odd i, so sum_i f[i] x'[i] = sum(x') + sum(n x). One shift places two nibbles (2j at bits 15..18,
// 2j+1 at 19..22), so a word costs four shifts and eight LOP3s.
__device__ __forceinline__ void stream_decode_nibbles8(std::uint32_t word,
                                                       const StreamDecodeConstants& c,
                                                       float (&f)[8]) {
    const std::uint32_t s0 = word << 15, s1 = word << 7, s2 = word >> 1, s3 = word >> 9;
    f[0] = stream_lop_magic(s0, c.low_nibble, c.one);
    f[1] = stream_lop_magic(s0, c.high_nibble, c.one);
    f[2] = stream_lop_magic(s1, c.low_nibble, c.one);
    f[3] = stream_lop_magic(s1, c.high_nibble, c.one);
    f[4] = stream_lop_magic(s2, c.low_nibble, c.one);
    f[5] = stream_lop_magic(s2, c.high_nibble, c.one);
    f[6] = stream_lop_magic(s3, c.low_nibble, c.one);
    f[7] = stream_lop_magic(s3, c.high_nibble, c.one);
}

// f[j] for the four bytes of `word` (byte j = the weight at k offset j, value u in 0..255): u sits
// at bits 15..22 under 1.0 for even j (1 + u/256, times x' = 256x) and under 16.0 for odd j
// (16 + u/16, times x' = 16x), so every weight contributes 256x + u x. One shift and one LOP3 per
// weight.
__device__ __forceinline__ void stream_decode_bytes4(std::uint32_t word,
                                                     const StreamDecodeConstants& c, float* f) {
    f[0] = stream_lop_magic(word << 15, c.byte, c.one);
    f[1] = stream_lop_magic(word << 7, c.byte, c.sixteen);
    f[2] = stream_lop_magic(word >> 1, c.byte, c.one);
    f[3] = stream_lop_magic(word >> 9, c.byte, c.sixteen);
}

// Folds a plane of one bit per weight into f: f[i] += or -= the magic float of the bit of weight i,
// which sits at bit Stride * i + Offset of `bits` (Q5's high plane: stride 1; Q6's two high planes:
// stride 2, offsets 0 and 1).
//
// Each bit is shifted to bit 19 (even i) or bit 22 (odd i) and read under a magic exponent:
// 1 + b/16 and 2 + b, against x' = 256x and 16x, give 16x per set bit; doubled magics (2.0 and 4.0)
// give 32x (Scale). The constant part, (Scale / 16) * (sum_even(x') + 2 sum_odd(x')), enters with
// the same sign and is folded into the format's correction. Both terms of every sum are small
// dyadic numbers, so f stays exact.
template <int Stride, int Offset, int Scale, bool Subtract>
__device__ __forceinline__ void stream_fold_bits8(std::uint32_t bits,
                                                  const StreamDecodeConstants& c, float (&f)[8]) {
    static_assert(Scale == 16 || Scale == 32, "magics exist for 16x and 32x per set bit");
    constexpr bool kDouble = Scale == 32;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        constexpr int kEvenTarget = 19, kOddTarget = 22;
        const int target          = (i & 1) ? kOddTarget : kEvenTarget;
        const int shift           = target - (Stride * i + Offset); // folded per unrolled i
        const std::uint32_t moved = shift >= 0 ? bits << shift : bits >> -shift;
        const std::uint32_t mask  = (i & 1) ? c.bit22 : c.bit19;
        const std::uint32_t magic =
            kDouble ? ((i & 1) ? c.four : c.two) : ((i & 1) ? c.two : c.one);
        const float term = stream_lop_magic(moved, mask, magic);
        f[i]             = Subtract ? f[i] - term : f[i] + term;
    }
}

} // namespace ninfer::ops::detail
