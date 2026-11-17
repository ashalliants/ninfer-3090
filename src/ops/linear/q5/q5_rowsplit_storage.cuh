#pragma once

#include "ops/common/math.cuh"
#include "ops/linear/common/stream_gemv_decode.cuh"
#include "ops/linear/common/stream_hmma.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>

namespace ninfer::ops::detail {

struct Q5RowSplitStorage {
    static constexpr int kGroupK             = 64;
    static constexpr int kCodeBytesPerGroup  = 32;
    static constexpr int kHighBytesPerGroup  = 8;
    static constexpr int kScaleBytesPerGroup = 2;

    // A chunk is the eight codes one thread decodes, so it spans four packed bytes.
    static constexpr int kCodeBytesPerChunk = 4;
    static constexpr int kHighBytesPerChunk =
        kHighBytesPerGroup / (kCodeBytesPerGroup / kCodeBytesPerChunk);
    static_assert(kHighBytesPerChunk * (kCodeBytesPerGroup / kCodeBytesPerChunk) ==
                      kHighBytesPerGroup,
                  "the high-bit plane must divide evenly across a group's chunks");
};

struct Q5ScalarDecodeAtom {
    static constexpr int kGroupK = Q5RowSplitStorage::kGroupK;

    __device__ static __forceinline__ void
    load_pair(const std::uint8_t* codes, const std::uint8_t* high, const std::uint8_t* scales,
              std::int64_t group_index, int lane, float& w0, float& w1) {
        decode_pair(codes, high, scales, group_index, lane, w0, w1);
    }

    __device__ static __forceinline__ void
    decode_pair(const std::uint8_t* codes, const std::uint8_t* high, const std::uint8_t* scales,
                std::int64_t group_index, int lane, float& w0, float& w1) {
        const float scale = __half2float(__ushort_as_half(*reinterpret_cast<const std::uint16_t*>(
            scales + group_index * Q5RowSplitStorage::kScaleBytesPerGroup)));
        const std::uint8_t packed =
            codes[group_index * Q5RowSplitStorage::kCodeBytesPerGroup + lane];
        const std::uint8_t high_byte =
            high[group_index * Q5RowSplitStorage::kHighBytesPerGroup + (lane >> 2)];
        const int shift = (lane & 3) * 2;
        const int q0 =
            ((static_cast<int>(packed & 0x0fu) | (((high_byte >> shift) & 1) << 4)) ^ 0x10) - 0x10;
        const int q1 =
            ((static_cast<int>(packed >> 4) | (((high_byte >> (shift + 1)) & 1) << 4)) ^ 0x10) -
            0x10;
        w0 = static_cast<float>(q0) * scale;
        w1 = static_cast<float>(q1) * scale;
    }
};

struct Q5SimtDecodeAtom {
    __device__ static __forceinline__ void decode_eight(std::uint32_t packed,
                                                        std::uint8_t high_bits,
                                                        std::uint16_t scale_bits,
                                                        float (&weights)[8]) {
        const std::uint32_t high = static_cast<std::uint32_t>(high_bits) ^ 0xffu;
        const float scale        = __half2float(__ushort_as_half(scale_bits));
        const __half2 bias       = __half2half2(__ushort_as_half(0x6410)); // 1040.0
#pragma unroll
        for (int pair = 0; pair < 4; ++pair) {
            std::uint32_t bits = ((packed >> (4 * pair)) & 0x000f000fu) | 0x64006400u;
            bits |= (((high >> pair) & 1u) << 4) | (((high >> (pair + 4)) & 1u) << 20);
            const __half2 decoded = __hsub2(half2_from_bits(bits), bias);
            const float2 values   = __half22float2(decoded);
            weights[pair]         = values.x * scale;
            weights[pair + 4]     = values.y * scale;
        }
    }
};

// Q5 for the streaming GEMV (ops/linear/common/stream_gemv.cuh). One lane slice is 16 code bytes
// (32 weights, nibble i of word w at k offset 8w + i) plus their four high bytes (byte w, bit i ->
// k offset 8w + i), value n - 16 * high. Against x', stream_decode_nibbles8 contributes
// sum(x') + sum(n x) and stream_fold_bits8<1, 0, 16, true> takes away
// sum_even(x') + 2 sum_odd(x') + 16 sum(high x), so the constant left is -sum_odd(x'): the
// correction per weight pair is -16 x_odd.
struct Q5StreamFormat {
    static constexpr const char* kName = "q5";
    static constexpr int kGroupK       = Q5RowSplitStorage::kGroupK;

    struct Planes {
        const std::uint8_t* codes;
        const std::uint8_t* high;
        const std::uint8_t* scales;
    };
    struct Row {
        const uint4* codes;
        const std::uint32_t* high;
        const std::uint16_t* scales;
    };
    struct Slice {
        uint4 codes;
        std::uint32_t high;
        std::uint16_t scale;
    };

    __device__ static __forceinline__ Row row(const Planes& planes, int row, int padded_k) {
        const std::int64_t groups = static_cast<std::int64_t>(row) * (padded_k / kGroupK);
        return {reinterpret_cast<const uint4*>(planes.codes +
                                               groups * Q5RowSplitStorage::kCodeBytesPerGroup),
                reinterpret_cast<const std::uint32_t*>(
                    planes.high + groups * Q5RowSplitStorage::kHighBytesPerGroup),
                reinterpret_cast<const std::uint16_t*>(
                    planes.scales + groups * Q5RowSplitStorage::kScaleBytesPerGroup)};
    }
    // Slice v covers k 32v .. 32v + 31: code bytes 16v.., high bytes 4v.. (group v / 2).
    __device__ static __forceinline__ Slice load(const Row& row, int v) {
        return {__ldg(&row.codes[v]), __ldg(&row.high[v]), __ldg(&row.scales[v >> 1])};
    }
    __device__ static __forceinline__ Slice empty() { return {make_uint4(0u, 0u, 0u, 0u), 0u, 0}; }
    __device__ static __forceinline__ float scale(const Slice& slice) {
        return __half2float(__ushort_as_half(slice.scale));
    }
    __device__ static __forceinline__ float correction(float, float x_odd) {
        return -kStreamOddScale * x_odd;
    }
    // The eight operands of packed word W (k offsets 8W .. 8W + 7 of the slice).
    template <int W>
    __device__ static __forceinline__ void decode_word(const Slice& slice,
                                                       const StreamDecodeConstants& c,
                                                       float (&f)[8]) {
        const std::uint32_t words[4] = {slice.codes.x, slice.codes.y, slice.codes.z,
                                        slice.codes.w};
        stream_decode_nibbles8(words[W], c, f);
        stream_fold_bits8<1, 0, 16, true>((slice.high >> (8 * W)) & 0xffu, c, f);
    }

    // Tensor-core form (ops/linear/common/stream_hmma.cuh): n - 16 high = m - 16 with
    // m = n + 16 (1 - high) in [0, 31], so the inverted high bit becomes bit 4 of each byte
    // before the same FP16 magic as Q4, with 1040 subtracted.
    // Tensor-core form (ops/linear/common/stream_hmma.cuh). A step is two groups (128 k) of one
    // row: 64 code bytes and 16 high bytes, both read by lane pairs, and the two scales.
    //
    // Off (kHmma = false): correct, but slower than the SIMT kernel at every verify width on a
    // V100 (us SIMT / tensor cores; NINFER_STREAM_HMMA=0 against =2 with this set to true):
    //   12288 x 5120   T=4 103 / 158   T=6 156 / 205   T=8 188 / 203
    //   5120 x 6144    T=4  67 /  93   T=6  83 /  95   T=8 104 / 102
    // The high-bit spread costs ~1 instruction per weight on top of Q4's decode, and the half-
    // sector high-plane reads cost bandwidth.
    static constexpr bool kHmma      = false;
    static constexpr int kHmmaGroups = 2;
    struct HmmaStep {
        uint4 codes[4]; // slice s: k 32s .. 32s + 31
        uint4 high;     // word s: slice s's 32 high bits
        std::uint32_t scales;
    };
    struct HmmaRaw {
        StreamHmmaPair codes[2];
        uint2 high_from_even;
        uint2 high_from_odd;
        std::uint32_t scales;
    };
    __device__ static __forceinline__ HmmaRaw hmma_fetch(const Row& even, const Row& odd,
                                                         bool is_odd, int group) {
        HmmaRaw raw;
#pragma unroll
        for (int g = 0; g < 2; ++g) {
            raw.codes[g] = stream_hmma_pair_fetch(even.codes + 2 * (group + g),
                                                  odd.codes + 2 * (group + g), is_odd);
        }
        // 16 high bytes per row: each lane of the pair reads an 8-byte half of both rows'.
        const int half     = is_odd ? 1 : 0;
        raw.high_from_even = __ldg(reinterpret_cast<const uint2*>(even.high + 2 * group) + half);
        raw.high_from_odd  = __ldg(reinterpret_cast<const uint2*>(odd.high + 2 * group) + half);
        const Row& own     = is_odd ? odd : even;
        raw.scales         = __ldg(reinterpret_cast<const std::uint32_t*>(own.scales + group));
        return raw;
    }
    __device__ static __forceinline__ HmmaStep hmma_finish(const HmmaRaw& raw, bool is_odd) {
        HmmaStep step;
        step.scales = raw.scales;
#pragma unroll
        for (int g = 0; g < 2; ++g) {
            stream_hmma_pair_swap(raw.codes[g], is_odd, step.codes[2 * g], step.codes[2 * g + 1]);
        }
        const uint2 send = is_odd ? raw.high_from_even : raw.high_from_odd;
        const uint2 recv = make_uint2(__shfl_xor_sync(0xffffffffu, send.x, 1),
                                      __shfl_xor_sync(0xffffffffu, send.y, 1));
        const uint2 lo   = is_odd ? recv : raw.high_from_even;
        const uint2 hi   = is_odd ? raw.high_from_odd : recv;
        step.high        = make_uint4(lo.x, lo.y, hi.x, hi.y);
        return step;
    }
    // Moves group 1 into group 0 (stream_hmma.cuh's rolled group loop decodes group 0).
    __device__ static __forceinline__ void hmma_advance(HmmaStep& step) {
        step.codes[0] = step.codes[2];
        step.codes[1] = step.codes[3];
        step.high     = make_uint4(step.high.z, step.high.w, 0u, 0u);
        step.scales >>= 16;
    }
    __device__ static __forceinline__ float hmma_scale(const HmmaStep& step, int) {
        return __half2float(__ushort_as_half(static_cast<std::uint16_t>(step.scales & 0xffffu)));
    }
    __device__ static __forceinline__ void hmma_decode(const HmmaStep& step, int slice, int w,
                                                       unsigned (&b)[4]) {
        const uint4 codes            = step.codes[slice];
        const std::uint32_t words[4] = {codes.x, codes.y, codes.z, codes.w};
        const std::uint32_t highs[4] = {step.high.x, step.high.y, step.high.z, step.high.w};
        const std::uint32_t lo  = words[w] & 0x0f0f0f0fu;
        const std::uint32_t hi  = (words[w] >> 4) & 0x0f0f0f0fu;
        const std::uint32_t inv = ~(highs[slice] >> (8 * w));
        // Bit i of a nibble lands in bit 4 of byte i: i + 4 + 7i = 8i + 4, and the cross terms
        // (i + 7j + 4 for i != j) never meet a byte's bit 4 or carry into it.
        const std::uint32_t first  = ((inv & 0xfu) * 0x02040810u) & 0x10101010u;
        const std::uint32_t second = (((inv >> 4) & 0xfu) * 0x02040810u) & 0x10101010u;
        stream_hmma_pairs(__byte_perm(lo, hi, 0x5140) | first,
                          __byte_perm(lo, hi, 0x7362) | second, 0x6410u, b);
    }
};

struct Q5MmaDecodeAtom {
    // Four packed bytes plus its high-bit byte -> four bf16 pairs, in weight order; out[i] holds
    // the pair decode_pair produces at lane = 4 * chunk + i.
    static __device__ __forceinline__ void
    decode_eight(unsigned word, const std::uint8_t* high_chunk, float scale, unsigned (&out)[4]) {
        const unsigned high = high_chunk[0];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const unsigned byte        = (word >> (8 * i)) & 0xffu;
            const int q0               = ((static_cast<int>(byte & 0x0fu) |
                                           static_cast<int>(((high >> (2 * i)) & 1u) << 4)) ^
                                          0x10) -
                                         0x10;
            const int q1               = ((static_cast<int>(byte >> 4) |
                                           static_cast<int>(((high >> (2 * i + 1)) & 1u) << 4)) ^
                                          0x10) -
                                         0x10;
            const __nv_bfloat162 value = __floats2bfloat162_rn(static_cast<float>(q0) * scale,
                                                               static_cast<float>(q1) * scale);
            out[i]                     = *reinterpret_cast<const unsigned*>(&value);
        }
    }

    static __device__ __forceinline__ __nv_bfloat162 decode_pair(const std::uint8_t* staged_codes,
                                                                 const std::uint8_t* staged_high,
                                                                 const std::uint8_t* scale_ptr,
                                                                 std::int64_t staged_group_index,
                                                                 int lane) {
        const float scale =
            __half2float(__ushort_as_half(*reinterpret_cast<const std::uint16_t*>(scale_ptr)));
        const std::uint8_t packed =
            staged_codes[staged_group_index * Q5RowSplitStorage::kCodeBytesPerGroup + lane];
        const std::uint8_t high_byte =
            staged_high[staged_group_index * Q5RowSplitStorage::kHighBytesPerGroup + (lane >> 2)];
        const int shift = (lane & 3) * 2;
        const int q0 =
            ((static_cast<int>(packed & 0x0fu) | (((high_byte >> shift) & 1) << 4)) ^ 0x10) - 0x10;
        const int q1 =
            ((static_cast<int>(packed >> 4) | (((high_byte >> (shift + 1)) & 1) << 4)) ^ 0x10) -
            0x10;
        return __floats2bfloat162_rn(static_cast<float>(q0) * scale,
                                     static_cast<float>(q1) * scale);
    }
};

} // namespace ninfer::ops::detail
