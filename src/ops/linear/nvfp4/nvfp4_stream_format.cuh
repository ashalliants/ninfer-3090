#pragma once

// NVFP4 for the dp4a streaming GEMV (ops/linear/common/stream_dp4a.cuh): the 1..8-token linears
// of an NVFP4 artifact on sm_70 under EngineOptions::verify_a8, the same route Q4/Q5/Q6 take.
//
// Why: sm_70 has no FP4/FP8 conversion instructions, so the A16 GEMV's exact FP32 expansion goes
// through the vendor software fallback for every code and scale, and measured ~7x the Q4 dp4a
// kernel on the same 5120 x 17408 shape (813.8 us against 118.2 us at T = 1, V100).
//
// Weights stay exact: an E2M1 code doubled is an integer in {0, +-1, 2, 3, 4, 6, 8, 12}, so two
// byte permutes against constant tables give each code's signed int8, and the 1/2 joins the scale.
// An E4M3 scale shifted into FP16 bits is the same value times 2^-8 (normal and subnormal alike),
// so it costs a shift and a convert. The approximation is the activation, rounded to int8 per 32
// values as for the other formats.
//
// Layout read: codes are (rows, K / 2) bytes, low nibble first, so slice v of a row (k 32v ..
// 32v + 31) is 16 contiguous bytes and nibble i of its word W is k 8W + i. Scales use the M128
// swizzle (nvfp4_scale_byte_offset): the two groups of a slice are adjacent bytes.

#include "ops/linear/common/stream_dp4a.cuh"
#include "ops/linear/nvfp4/nvfp4_codec.cuh"

#include <cuda_fp16.h>

#include <cstdint>

namespace ninfer::ops::detail {

struct Nvfp4StreamFormat {
    static constexpr const char* kName = "nvfp4";
    static constexpr int kGroupK       = 16;

    struct Planes {
        const std::uint8_t* codes;
        const std::uint8_t* scales;
        float scale_factor; // 128 / weight_scale_divisor: 2^8 for the FP16 scale, 1/2 for codes
    };
    struct Row {
        const uint4* codes;
        const std::uint8_t* scales; // this row's scale bytes for groups 0..3 of K-tile 0
        float scale_factor;
    };
    struct Slice {
        uint4 codes;
        std::uint32_t scales; // the slice's two E4M3 scales in bytes 0 and 1
        float scale_factor;
    };

    __device__ static __forceinline__ Row row(const Planes& planes, int row, int padded_k) {
        return {reinterpret_cast<const uint4*>(planes.codes +
                                               static_cast<std::int64_t>(row) * (padded_k / 2)),
                planes.scales + nvfp4_scale_byte_offset(row, 0, padded_k), planes.scale_factor};
    }
    // Groups 2v, 2v + 1: K-tile v / 2 (512 bytes apart), bytes 2 (v & 1) and 2 (v & 1) + 1.
    __device__ static __forceinline__ Slice load(const Row& row, int v) {
        const auto* scales =
            reinterpret_cast<const std::uint16_t*>(row.scales + (v >> 1) * 512 + (v & 1) * 2);
        return {__ldg(&row.codes[v]), __ldg(scales), row.scale_factor};
    }
    __device__ static __forceinline__ Slice empty() { return {make_uint4(0u, 0u, 0u, 0u), 0u, 0.0f}; }

    static constexpr bool kDp4a            = true;
    static constexpr int kDp4aBias         = 0;
    static constexpr bool kDp4aHalfScales  = true;
    static constexpr bool kDp4aOnly        = true;

    // 2 x E2M1 for codes 0..7 and their negations (codes 8..15), one signed byte each.
    static constexpr std::uint32_t kPositive0 = 0x03020100u, kPositive1 = 0x0c080604u;
    static constexpr std::uint32_t kNegative0 = 0xfdfeff00u, kNegative1 = 0xf4f8fafcu;

    // Four codes (the low 16 bits of `codes`, k ascending) to four signed bytes.
    __device__ static __forceinline__ std::uint32_t decode_quad(std::uint32_t magnitude,
                                                                std::uint32_t sign_selector) {
        const std::uint32_t positive = __byte_perm(kPositive0, kPositive1, magnitude);
        const std::uint32_t negative = __byte_perm(kNegative0, kNegative1, magnitude);
        // Selector nibble 0 picks byte 0 (0x00), 1 picks byte 1 (0xff). __byte_perm reads only
        // the low three bits of a nibble, so prmt's sign-replicate mode is not available here.
        const std::uint32_t mask = __byte_perm(0x0000ff00u, 0u, sign_selector);
        return (positive & ~mask) | (negative & mask);
    }

    template <int W>
    __device__ static __forceinline__ void dp4a_word(const Slice& slice, std::uint32_t& even,
                                                     std::uint32_t& odd) {
        const std::uint32_t words[4] = {slice.codes.x, slice.codes.y, slice.codes.z,
                                        slice.codes.w};
        const std::uint32_t word      = words[W];
        const std::uint32_t magnitude = word & 0x77777777u;
        const std::uint32_t sign      = (word >> 3) & 0x11111111u; // 1 per negative code
        const std::uint32_t low       = decode_quad(magnitude, sign);              // k 0..3
        const std::uint32_t high      = decode_quad(magnitude >> 16, sign >> 16); // k 4..7
        even = __byte_perm(low, high, 0x6420);
        odd  = __byte_perm(low, high, 0x7531);
    }

    __device__ static __forceinline__ float2 half_scales(const Slice& slice) {
        // E4M3 s.eeee.mmm -> FP16 s.0eeee.mmm0000000 is the same value times 2^-8.
        const std::uint32_t spread = __byte_perm(slice.scales, 0u, 0x4140);
        const std::uint32_t bits =
            ((spread & 0x00800080u) << 8) | ((spread & 0x007f007fu) << 7);
        const float2 values = __half22float2(*reinterpret_cast<const __half2*>(&bits));
        return {values.x * slice.scale_factor, values.y * slice.scale_factor};
    }
};

inline Nvfp4StreamFormat::Planes nvfp4_stream_planes(const std::uint8_t* codes,
                                                     const std::uint8_t* scales, float alpha) {
    return {codes, scales, alpha * 128.0f};
}

} // namespace ninfer::ops::detail
