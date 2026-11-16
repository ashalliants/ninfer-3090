#pragma once

#include "ops/common/math.cuh"
#include "ops/linear/common/stream_gemv_decode.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>

namespace ninfer::ops::detail {

struct Q4RowSplitStorage {
    static constexpr int kGroupK             = 64;
    static constexpr int kCodeBytesPerGroup  = 32;
    static constexpr int kScaleBytesPerGroup = 2;
};

struct Q4SimtDecodeAtom {
    __device__ static __forceinline__ void
    decode_eight(std::uint32_t packed, std::uint16_t scale_bits, float (&weights)[8]) {
        const std::uint32_t word = packed ^ 0x88888888u;
        const float scale        = __half2float(__ushort_as_half(scale_bits));
        const __half2 bias       = __half2half2(__ushort_as_half(0x6408)); // 1032.0
#pragma unroll
        for (int pair = 0; pair < 4; ++pair) {
            const std::uint32_t bits = ((word >> (4 * pair)) & 0x000f000fu) | 0x64006400u;
            const __half2 decoded    = __hsub2(half2_from_bits(bits), bias);
            const float2 values      = __half22float2(decoded);
            weights[pair]            = values.x * scale;
            weights[pair + 4]        = values.y * scale;
        }
    }

    __device__ static __forceinline__ void
    decode_pair(std::uint8_t packed, std::uint16_t scale_bits, float& w0, float& w1) {
        const float scale = __half2float(__ushort_as_half(scale_bits));
        const int q0      = (static_cast<int>(packed & 0x0fu) ^ 0x08) - 0x08;
        const int q1      = (static_cast<int>(packed >> 4) ^ 0x08) - 0x08;
        w0                = static_cast<float>(q0) * scale;
        w1                = static_cast<float>(q1) * scale;
    }
};

// Q4 for the streaming GEMV (ops/linear/common/stream_gemv.cuh). One lane slice is 16 code bytes,
// half a group: 32 weights at consecutive k, nibble i of word w at k offset 8w + i, value
// (n ^ 8) - 8. XORing the word with 0x88888888 gives m = value + 8 in [0, 15], which
// stream_dot_nibbles8 decodes; the -8 joins the activation correction, so per weight pair
// (x_even, x_odd) the correction is x' + 8x = 264 x_even + 24 x_odd.
struct Q4StreamFormat {
    static constexpr const char* kName = "q4";
    static constexpr int kGroupK       = Q4RowSplitStorage::kGroupK;

    struct Planes {
        const std::uint8_t* codes;
        const std::uint8_t* scales;
    };
    struct Row {
        const uint4* codes;
        const std::uint16_t* scales;
    };
    struct Slice {
        uint4 codes;
        std::uint16_t scale;
    };

    __device__ static __forceinline__ Row row(const Planes& planes, int row, int padded_k) {
        const std::int64_t groups = static_cast<std::int64_t>(row) * (padded_k / kGroupK);
        return {reinterpret_cast<const uint4*>(planes.codes +
                                               groups * Q4RowSplitStorage::kCodeBytesPerGroup),
                reinterpret_cast<const std::uint16_t*>(
                    planes.scales + groups * Q4RowSplitStorage::kScaleBytesPerGroup)};
    }
    // Slice v covers k 32v .. 32v + 31 (group v / 2).
    __device__ static __forceinline__ Slice load(const Row& row, int v) {
        return {__ldg(&row.codes[v]), __ldg(&row.scales[v >> 1])};
    }
    __device__ static __forceinline__ Slice empty() { return {make_uint4(0u, 0u, 0u, 0u), 0}; }
    __device__ static __forceinline__ float scale(const Slice& slice) {
        return __half2float(__ushort_as_half(slice.scale));
    }
    __device__ static __forceinline__ float correction(float x_even, float x_odd) {
        return fmaf(x_even, kStreamEvenScale + 8.0f, x_odd * (kStreamOddScale + 8.0f));
    }
    // x' for k offsets 8w .. 8w + 7 of the slice are chunks[2w] and chunks[2w + 1].
    __device__ static __forceinline__ float dot(const Slice& slice, const float4 (&chunks)[8],
                                                const StreamDecodeConstants& c) {
        const std::uint32_t words[4] = {slice.codes.x, slice.codes.y, slice.codes.z,
                                        slice.codes.w};
        float acc = 0.0f;
#pragma unroll
        for (int w = 0; w < 4; ++w) {
            acc = stream_dot_nibbles8(words[w] ^ 0x88888888u, chunks[2 * w], chunks[2 * w + 1], c,
                                      acc);
        }
        return acc;
    }
};

struct Q4MmaDecodeAtom {
    // Eight packed codes -> four bf16 pairs, in weight order; out[i] holds the pair for
    // lane = 4 * chunk + i. A nibble n decodes to the integer (n ^ 8) - 8, which lies in [-8, 7]
    // and which bf16 represents exactly, so forming 128 + (n ^ 8) and subtracting 136 lands on it.
    // The values are unscaled: callers of this overload fold the group scale into the
    // accumulation, unlike decode_pair below.
    static __device__ __forceinline__ void decode_eight(unsigned word, unsigned (&out)[4]) {
        const unsigned kBias  = 0x43084308u; // bf16 136.0 in both halves
        const unsigned kMagic = 0x43004300u; // bf16 128.0 in both halves
        word ^= 0x88888888u;
        const unsigned lo        = word & 0x0f0f0f0fu;
        const unsigned hi        = (word >> 4) & 0x0f0f0f0fu;
        const unsigned even      = __byte_perm(lo, hi, 0x5140);
        const unsigned odd       = __byte_perm(lo, hi, 0x7362);
        const unsigned biased[4] = {
            __byte_perm(even, kMagic, 0x7150), __byte_perm(even, kMagic, 0x7372),
            __byte_perm(odd, kMagic, 0x7150), __byte_perm(odd, kMagic, 0x7372)};
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const __nv_bfloat162 value =
                __hsub2(*reinterpret_cast<const __nv_bfloat162*>(&biased[i]),
                        *reinterpret_cast<const __nv_bfloat162*>(&kBias));
            out[i] = *reinterpret_cast<const unsigned*>(&value);
        }
    }

    static __device__ __forceinline__ __nv_bfloat162 decode_pair(const std::uint8_t* codes,
                                                                 const std::uint8_t* scale_ptr,
                                                                 std::int64_t group_index,
                                                                 int lane) {
        const float scale =
            __half2float(__ushort_as_half(*reinterpret_cast<const std::uint16_t*>(scale_ptr)));
        const std::uint8_t packed =
            codes[group_index * Q4RowSplitStorage::kCodeBytesPerGroup + lane];
        const int q0 = (static_cast<int>(packed & 0x0fu) ^ 0x08) - 0x08;
        const int q1 = (static_cast<int>(packed >> 4) ^ 0x08) - 0x08;
        return __floats2bfloat162_rn(static_cast<float>(q0) * scale,
                                     static_cast<float>(q1) * scale);
    }
};

} // namespace ninfer::ops::detail
