#pragma once

#include "ops/common/math.cuh"
#include "ops/linear/common/stream_gemv_decode.cuh"
#include "ops/linear/common/stream_hmma.cuh"

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
// stream_decode_nibbles8 decodes; the -8 joins the activation correction, so per weight pair
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
    // The eight operands of packed word W (k offsets 8W .. 8W + 7 of the slice).
    template <int W>
    __device__ static __forceinline__ void decode_word(const Slice& slice,
                                                       const StreamDecodeConstants& c,
                                                       float (&f)[8]) {
        const std::uint32_t words[4] = {slice.codes.x, slice.codes.y, slice.codes.z,
                                        slice.codes.w};
        stream_decode_nibbles8(words[W] ^ 0x88888888u, c, f);
    }

    // Tensor-core form (ops/linear/common/stream_hmma.cuh). A step is kHmmaGroups groups of one
    // row: their code bytes, read by lane pairs, and their scales.
    static constexpr bool kHmma       = true;
    static constexpr int kHmmaGroups  = 4;
    struct HmmaStep {
        uint4 codes[2 * kHmmaGroups]; // slice s: k 32s .. 32s + 31
        uint2 scales;                 // groups 0..3
    };
    struct HmmaRaw {
        StreamHmmaPair codes[kHmmaGroups];
        uint2 scales;
    };
    __device__ static __forceinline__ HmmaRaw hmma_fetch(const Row& even, const Row& odd,
                                                         bool is_odd, int group) {
        HmmaRaw raw;
#pragma unroll
        for (int g = 0; g < kHmmaGroups; ++g) {
            raw.codes[g] = stream_hmma_pair_fetch(even.codes + 2 * (group + g),
                                                  odd.codes + 2 * (group + g), is_odd);
        }
        const Row& own = is_odd ? odd : even;
        raw.scales     = __ldg(reinterpret_cast<const uint2*>(own.scales + group));
        return raw;
    }
    __device__ static __forceinline__ HmmaStep hmma_finish(const HmmaRaw& raw, bool is_odd) {
        HmmaStep step;
        step.scales = raw.scales;
#pragma unroll
        for (int g = 0; g < kHmmaGroups; ++g) {
            stream_hmma_pair_swap(raw.codes[g], is_odd, step.codes[2 * g], step.codes[2 * g + 1]);
        }
        return step;
    }
    // Moves slice s + 1 into slice s, so a rolled slice loop always decodes slice 0 (keeps the
    // loop body inside the instruction cache, without indexing registers at run time).
    __device__ static __forceinline__ void hmma_advance(HmmaStep& step) {
#pragma unroll
        for (int s = 0; s + 1 < 2 * kHmmaGroups; ++s) step.codes[s] = step.codes[s + 1];
    }
    __device__ static __forceinline__ float hmma_scale(const HmmaStep& step, int slice) {
        const std::uint32_t pair = (slice >> 2) ? step.scales.y : step.scales.x;
        return __half2float(__ushort_as_half(
            static_cast<std::uint16_t>(((slice >> 1) & 1) ? pair >> 16 : pair & 0xffffu)));
    }
    // The four FP16 pairs (k 8w + 2j, 8w + 2j + 1) of word w of a slice, as exact integers:
    // m = n ^ 8 sits in the mantissa of 1024 + m, and subtracting 1032 leaves (n ^ 8) - 8.
    __device__ static __forceinline__ void hmma_decode(const HmmaStep& step, int slice, int w,
                                                       unsigned (&b)[4]) {
        const uint4 codes             = step.codes[slice];
        const std::uint32_t words[4]  = {codes.x, codes.y, codes.z, codes.w};
        const std::uint32_t m  = words[w] ^ 0x88888888u;
        const std::uint32_t lo = m & 0x0f0f0f0fu;        // byte j: k 2j
        const std::uint32_t hi = (m >> 4) & 0x0f0f0f0fu; // byte j: k 2j + 1
        stream_hmma_pairs(__byte_perm(lo, hi, 0x5140), __byte_perm(lo, hi, 0x7362), 0x6408u, b);
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
