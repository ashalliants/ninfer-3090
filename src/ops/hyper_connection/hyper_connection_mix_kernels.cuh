#pragma once

// Adapted from Infernix a3edb450 src/ops/hyper_connection/hyper_connection_mix_fused.cu
// (Apache-2.0) and Basalt 8d73a39 src/kernels/cuda/hc_fused.cu (MIT, third_party/basalt/LICENSE).
// Modified for NInfer-3090: the device helpers the BF16 and int8 fused routes of
// hyper_connection_mix share (hyper_connection_mix_fused.cu, hyper_connection_mix_int8.cu), and
// their launch check. Private to the Op.

#include <cuda_runtime.h>

#include <cstdint>
#include <stdexcept>
#include <string>

namespace ninfer::ops::detail {

inline void check_mix_launch(const char* what) {
    const cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        throw std::runtime_error(std::string("hyper_connection_mix ") + what + ": " +
                                 cudaGetErrorString(error));
    }
}

// Read-only global load that does not allocate in L1: weights streamed once per call (Infernix's
// ops/common/memory.cuh ld_nc_na, 16-byte form; sm_70+).
__device__ __forceinline__ uint4 ld_nc_na_16(const void* ptr) {
    uint4 value;
    asm("ld.global.nc.L1::no_allocate.v4.u32 {%0, %1, %2, %3}, [%4];"
        : "=r"(value.x), "=r"(value.y), "=r"(value.z), "=r"(value.w)
        : "l"(ptr));
    return value;
}

// Its 8-byte form, for an int8 copy's eight codes.
__device__ __forceinline__ uint2 ld_nc_na_8(const void* ptr) {
    uint2 value;
    asm("ld.global.nc.L1::no_allocate.v2.u32 {%0, %1}, [%2];"
        : "=r"(value.x), "=r"(value.y)
        : "l"(ptr));
    return value;
}

__device__ __forceinline__ float sigmoidf(float x) { return 1.0F / (1.0F + expf(-x)); }

// Four int8 codes as floats, exactly: q + 128 as the low byte of 2^23 + (q + 128), less 2^23 + 128
// (Basalt i8x4_f).
__device__ __forceinline__ float4 i8x4_to_float4(std::uint32_t q) {
    const std::uint32_t u = q ^ 0x80808080U;
    return make_float4(__uint_as_float(__byte_perm(u, 0x4B000000U, 0x7650U)) - 8388736.0F,
                       __uint_as_float(__byte_perm(u, 0x4B000000U, 0x7651U)) - 8388736.0F,
                       __uint_as_float(__byte_perm(u, 0x4B000000U, 0x7652U)) - 8388736.0F,
                       __uint_as_float(__byte_perm(u, 0x4B000000U, 0x7653U)) - 8388736.0F);
}

} // namespace ninfer::ops::detail
