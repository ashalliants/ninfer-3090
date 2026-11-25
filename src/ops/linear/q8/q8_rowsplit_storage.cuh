#pragma once

#include "ops/linear/common/stream_gemv_decode.cuh"

#include <cuda_fp16.h>

#include <cstdint>

namespace ninfer::ops::detail {

struct Q8RowSplitStorage {
    static constexpr int kGroupK             = 32;
    static constexpr int kCodeBytesPerGroup  = 32;
    static constexpr int kHighBytesPerGroup  = 0;
    static constexpr int kScaleBytesPerGroup = 2;
};

// Q8 for the streaming GEMV (ops/linear/common/stream_gemv.cuh). One lane slice is one 32-weight
// group: 32 int8 codes at consecutive k (32 bytes, two 16-byte loads) and its f16 scale. XORing a
// word with 0x80808080 gives u = code + 128 in [0, 255], which stream_decode_bytes4 reads so that
// every weight contributes 256x + u x; the constant and the -128 offset join the correction,
// 384 (x_even + x_odd) per weight pair.
struct Q8StreamFormat {
    static constexpr const char* kName = "q8";
    static constexpr int kGroupK       = Q8RowSplitStorage::kGroupK;

    struct Planes {
        const std::uint8_t* codes;
        const std::uint8_t* scales;
    };
    struct Row {
        const uint4* codes;
        const std::uint16_t* scales;
    };
    struct Slice {
        uint4 lo;
        uint4 hi;
        std::uint16_t scale;
    };

    __device__ static __forceinline__ Row row(const Planes& planes, int row, int padded_k) {
        const std::int64_t groups = static_cast<std::int64_t>(row) * (padded_k / kGroupK);
        return {reinterpret_cast<const uint4*>(planes.codes +
                                               groups * Q8RowSplitStorage::kCodeBytesPerGroup),
                reinterpret_cast<const std::uint16_t*>(
                    planes.scales + groups * Q8RowSplitStorage::kScaleBytesPerGroup)};
    }
    // Slice v is group v: k 32v .. 32v + 31.
    __device__ static __forceinline__ Slice load(const Row& row, int v) {
        return {__ldg(&row.codes[2 * v]), __ldg(&row.codes[2 * v + 1]), __ldg(&row.scales[v])};
    }
    __device__ static __forceinline__ Slice empty() {
        return {make_uint4(0u, 0u, 0u, 0u), make_uint4(0u, 0u, 0u, 0u), 0};
    }
    __device__ static __forceinline__ float scale(const Slice& slice) {
        return __half2float(__ushort_as_half(slice.scale));
    }
    __device__ static __forceinline__ float correction(float x_even, float x_odd) {
        return 384.0f * (x_even + x_odd);
    }
    // The eight operands of k offsets 8W .. 8W + 7 of the slice: code words 2W and 2W + 1.
    template <int W>
    __device__ static __forceinline__ void decode_word(const Slice& slice,
                                                       const StreamDecodeConstants& c,
                                                       float (&f)[8]) {
        const std::uint32_t words[8] = {slice.lo.x, slice.lo.y, slice.lo.z, slice.lo.w,
                                        slice.hi.x, slice.hi.y, slice.hi.z, slice.hi.w};
        stream_decode_bytes4(words[2 * W] ^ 0x80808080u, c, f);
        stream_decode_bytes4(words[2 * W + 1] ^ 0x80808080u, c, f + 4);
    }

    // dp4a form (ops/linear/common/stream_dp4a.cuh): the codes are signed int8 already, so word W's
    // even and odd k (8W + 0, 2, 4, 6 and 8W + 1, 3, 5, 7) are two byte permutes, with no bias.
    // Drafter-only: a speculative drafter's Q8 linears take it inside a ScopedDraftA8
    // (stream_draft_dp4a.cuh); the verify route (EngineOptions::verify_a8) leaves Q8 on A16, except
    // the shapes try_launch_q8_verify_dp4a admits (the 35B-A3B's, through its own entry).
    static constexpr bool kDp4a            = true;
    static constexpr bool kDp4aDrafterOnly = true;
    static constexpr int kDp4aBias         = 0;
    template <int W>
    __device__ static __forceinline__ void dp4a_word(const Slice& slice, std::uint32_t& even,
                                                     std::uint32_t& odd) {
        const std::uint32_t words[8] = {slice.lo.x, slice.lo.y, slice.lo.z, slice.lo.w,
                                        slice.hi.x, slice.hi.y, slice.hi.z, slice.hi.w};
        even = __byte_perm(words[2 * W], words[2 * W + 1], 0x6420);
        odd  = __byte_perm(words[2 * W], words[2 * W + 1], 0x7531);
    }
};

struct Q8ScalarDecodeAtom {
    static constexpr int kGroupK = Q8RowSplitStorage::kGroupK;

    __device__ static __forceinline__ void
    load_pair(const std::uint8_t* codes, const std::uint8_t* /*high*/, const std::uint8_t* scales,
              std::int64_t group_index, int lane, float& w0, float& w1) {
        if (lane >= kGroupK / 2) {
            w0 = 0.0f;
            w1 = 0.0f;
            return;
        }
        const float scale = __half2float(
            __ushort_as_half(*reinterpret_cast<const std::uint16_t*>(scales + group_index * 2)));
        const std::uint8_t* packed = codes + group_index * Q8RowSplitStorage::kCodeBytesPerGroup +
                                     static_cast<std::int64_t>(lane) * 2;
        w0 = static_cast<float>(static_cast<std::int8_t>(packed[0])) * scale;
        w1 = static_cast<float>(static_cast<std::int8_t>(packed[1])) * scale;
    }
};

} // namespace ninfer::ops::detail
