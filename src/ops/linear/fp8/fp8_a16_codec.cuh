#pragma once

// Exact E4M3-to-BF16 operand widening shared by the small-T and GEMM A16 routes.

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

#include <cstdint>

namespace ninfer::ops::detail {

// Two E4M3 codes (bytes 0 and 1 of `packed`) to FP32, exactly, without the vendor conversion: below
// sm_89 there is no FP8 conversion instruction, and __nv_fp8x2_e4m3's software fallback cost the
// sm_70 decode GEMV about 2.5x the weight stream. E4M3 s.eeee.mmm moved into FP16 bits as
// s.0eeee.mmm0000000 is the same value times 2^-8 for every finite code, subnormals included
// (FP16 keeps its subnormals in hardware whatever the FTZ mode), so one conversion and a power-of-two
// multiply finish it. NaN (0x7f / 0xff) does not occur in weights and is not preserved.
__device__ __forceinline__ float2 fp8_e4m3x2_to_float2_exact(std::uint32_t packed) {
    const std::uint32_t spread = __byte_perm(packed, 0u, 0x4140); // byte 0 -> half 0, 1 -> half 1
    const std::uint32_t bits   = ((spread & 0x00800080u) << 8) | ((spread & 0x007f007fu) << 7);
    const float2 scaled        = __half22float2(*reinterpret_cast<const __half2*>(&bits));
    return {scaled.x * 256.0f, scaled.y * 256.0f};
}

__device__ __forceinline__ unsigned fp8_e4m3x2_to_bf16x2_bits(unsigned packed) {
// The native widening is Blackwell-only; sm_86 always takes the exact FP16 bridge.
#if (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) &&   \
    defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
    // PTX 9.2 exposes the native pair widening on the project's sm_120a target.
    unsigned result;
    asm("cvt.rn.bf16x2.e4m3x2 %0, %1;" : "=r"(result) : "h"(static_cast<std::uint16_t>(packed)));
    return result;
#elif defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 890
    // No FP8 conversion instruction: the exact bit-level widening. An E4M3 value has at most four
    // significant bits and lies in BF16's normal range, so the FP32 top half is it exactly.
    const float2 value = fp8_e4m3x2_to_float2_exact(packed);
    return (__float_as_uint(value.x) >> 16) | (__float_as_uint(value.y) & 0xffff0000u);
#else
    // CUDA 13.1 retains the exact FP16 bridge.
    __nv_fp8x2_e4m3 fp8;
    fp8.__x                 = static_cast<std::uint16_t>(packed);
    const __half2 half_pair = static_cast<__half2>(fp8);

    union {
        __nv_bfloat162 pair;
        unsigned bits;
    } result;

    result.pair = __halves2bfloat162(__nv_bfloat16(__low2half(half_pair)),
                                     __nv_bfloat16(__high2half(half_pair)));
    return result.bits;
#endif
}

__device__ __forceinline__ int fp8_a16_shared_col_64(int row, int col) {
    return (((col >> 3) ^ (row & 7)) << 3) | (col & 7);
}

} // namespace ninfer::ops::detail
