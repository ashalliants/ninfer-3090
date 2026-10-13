#pragma once

// Volta (sm_70) emulation of the Ampere instructions the kernels are written against.
//
// Volta has no cp.async, ldmatrix, or m16n8kK mma.sync, and its tensor cores have no BF16 or INT8
// mode. Each helper here reproduces the exact PTX fragment layout of the instruction it replaces,
// so every kernel using ops/common/mma.cuh and ops/common/memory.cuh runs unchanged. The mma
// emulations accumulate in FP32 (or INT32 via dp4a) with all-lane shuffles; they are a correct
// baseline, not a fast path. Fast Volta GEMMs belong in dedicated kernels.

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
#define NINFER_VOLTA_EMULATION 1
#endif

#if defined(NINFER_VOLTA_EMULATION)

#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace ninfer::ops::volta {

constexpr unsigned kFullMask = 0xffffffffu;

__device__ __forceinline__ unsigned lds32(unsigned addr) {
    unsigned v;
    asm volatile("ld.shared.b32 %0, [%1];\n" : "=r"(v) : "r"(addr));
    return v;
}

__device__ __forceinline__ unsigned lds16(unsigned addr) {
    unsigned short v;
    asm volatile("ld.shared.u16 %0, [%1];\n" : "=h"(v) : "r"(addr));
    return v;
}

// Matrix i (0..3) of an ldmatrix: lane 8*i+r supplies the address of row r; each lane receives
// row lane/4, columns 2*(lane%4), 2*(lane%4)+1.
__device__ __forceinline__ unsigned ldmatrix_word(unsigned addr, int i) {
    const int lane = threadIdx.x & 31;
    const unsigned row_addr = __shfl_sync(kFullMask, addr, 8 * i + (lane >> 2));
    return lds32(row_addr + 4 * (lane & 3));
}

// Transposed: each lane receives M[2*(lane%4)][lane/4] and M[2*(lane%4)+1][lane/4].
__device__ __forceinline__ unsigned ldmatrix_word_t(unsigned addr, int i) {
    const int lane = threadIdx.x & 31;
    const int r    = 2 * (lane & 3);
    const unsigned a0 = __shfl_sync(kFullMask, addr, 8 * i + r);
    const unsigned a1 = __shfl_sync(kFullMask, addr, 8 * i + r + 1);
    const unsigned c  = 2u * static_cast<unsigned>(lane >> 2);
    return lds16(a0 + c) | (lds16(a1 + c) << 16);
}

struct Bf16Unpack {
    __device__ static float2 apply(unsigned x) {
        return make_float2(__uint_as_float(x << 16), __uint_as_float(x & 0xffff0000u));
    }
};

struct F16Unpack {
    __device__ static float2 apply(unsigned x) {
        return __half22float2(*reinterpret_cast<const __half2*>(&x));
    }
};

// D = A(16x16, row) * B(16x8, col) + C, PTX m16n8k16 fragment layout. g = lane/4, t = lane%4:
//   a0:(row g,   k 2t..2t+1)  a1:(row g+8, k 2t..)  a2:(row g, k 2t+8..)  a3:(row g+8, k 2t+8..)
//   b0:(k 2t..2t+1, n g)      b1:(k 2t+8.., n g)
//   c0,c1:(row g, n 2t..2t+1) c2,c3:(row g+8, n 2t..2t+1)
template <class Unpack>
__device__ __forceinline__ void mma_m16n8k16(float& c0, float& c1, float& c2, float& c3,
                                             unsigned a0, unsigned a1, unsigned a2, unsigned a3,
                                             unsigned b0, unsigned b1) {
    const int lane = threadIdx.x & 31;
    const int g    = lane >> 2;
    const int t    = lane & 3;
#pragma unroll
    for (int tp = 0; tp < 4; ++tp) {
        const int src_a = 4 * g + tp;
        const float2 A0 = Unpack::apply(__shfl_sync(kFullMask, a0, src_a));
        const float2 A1 = Unpack::apply(__shfl_sync(kFullMask, a1, src_a));
        const float2 A2 = Unpack::apply(__shfl_sync(kFullMask, a2, src_a));
        const float2 A3 = Unpack::apply(__shfl_sync(kFullMask, a3, src_a));
#pragma unroll
        for (int j = 0; j < 2; ++j) {
            const int src_b = 4 * (2 * t + j) + tp;
            const float2 B0 = Unpack::apply(__shfl_sync(kFullMask, b0, src_b));
            const float2 B1 = Unpack::apply(__shfl_sync(kFullMask, b1, src_b));
            float& lo       = (j == 0) ? c0 : c1;
            float& hi       = (j == 0) ? c2 : c3;
            lo = fmaf(A0.x, B0.x, lo);
            lo = fmaf(A0.y, B0.y, lo);
            lo = fmaf(A2.x, B1.x, lo);
            lo = fmaf(A2.y, B1.y, lo);
            hi = fmaf(A1.x, B0.x, hi);
            hi = fmaf(A1.y, B0.y, hi);
            hi = fmaf(A3.x, B1.x, hi);
            hi = fmaf(A3.y, B1.y, hi);
        }
    }
}

// m16n8k32 s8: a0:(row g, k 4t..4t+3) a1:(row g+8, same) a2:(row g, k 16+4t..) a3:(row g+8, 16+4t..)
//              b0:(k 4t..4t+3, n g)   b1:(k 16+4t.., n g)
__device__ __forceinline__ void mma_m16n8k32_s8(int& c0, int& c1, int& c2, int& c3, unsigned a0,
                                                unsigned a1, unsigned a2, unsigned a3,
                                                unsigned b0, unsigned b1) {
    const int lane = threadIdx.x & 31;
    const int g    = lane >> 2;
    const int t    = lane & 3;
#pragma unroll
    for (int tp = 0; tp < 4; ++tp) {
        const int src_a = 4 * g + tp;
        const int A0    = static_cast<int>(__shfl_sync(kFullMask, a0, src_a));
        const int A1    = static_cast<int>(__shfl_sync(kFullMask, a1, src_a));
        const int A2    = static_cast<int>(__shfl_sync(kFullMask, a2, src_a));
        const int A3    = static_cast<int>(__shfl_sync(kFullMask, a3, src_a));
#pragma unroll
        for (int j = 0; j < 2; ++j) {
            const int src_b = 4 * (2 * t + j) + tp;
            const int B0    = static_cast<int>(__shfl_sync(kFullMask, b0, src_b));
            const int B1    = static_cast<int>(__shfl_sync(kFullMask, b1, src_b));
            int& lo         = (j == 0) ? c0 : c1;
            int& hi         = (j == 0) ? c2 : c3;
            lo = __dp4a(A0, B0, lo);
            lo = __dp4a(A2, B1, lo);
            hi = __dp4a(A1, B0, hi);
            hi = __dp4a(A3, B1, hi);
        }
    }
}

// m16n8k8 tf32: a0:(row g, k t) a1:(row g+8, k t) a2:(row g, k t+4) a3:(row g+8, k t+4)
//               b0:(k t, n g)   b1:(k t+4, n g)
__device__ __forceinline__ void mma_m16n8k8_tf32(float& c0, float& c1, float& c2, float& c3,
                                                 unsigned a0, unsigned a1, unsigned a2,
                                                 unsigned a3, unsigned b0, unsigned b1) {
    constexpr unsigned kTf32Mask = 0xffffe000u;
    const int lane               = threadIdx.x & 31;
    const int g                  = lane >> 2;
    const int t                  = lane & 3;
#pragma unroll
    for (int tp = 0; tp < 4; ++tp) {
        const int src_a = 4 * g + tp;
        const float A0  = __uint_as_float(__shfl_sync(kFullMask, a0, src_a) & kTf32Mask);
        const float A1  = __uint_as_float(__shfl_sync(kFullMask, a1, src_a) & kTf32Mask);
        const float A2  = __uint_as_float(__shfl_sync(kFullMask, a2, src_a) & kTf32Mask);
        const float A3  = __uint_as_float(__shfl_sync(kFullMask, a3, src_a) & kTf32Mask);
#pragma unroll
        for (int j = 0; j < 2; ++j) {
            const int src_b = 4 * (2 * t + j) + tp;
            const float B0  = __uint_as_float(__shfl_sync(kFullMask, b0, src_b) & kTf32Mask);
            const float B1  = __uint_as_float(__shfl_sync(kFullMask, b1, src_b) & kTf32Mask);
            float& lo       = (j == 0) ? c0 : c1;
            float& hi       = (j == 0) ? c2 : c3;
            lo = fmaf(A0, B0, lo);
            lo = fmaf(A2, B1, lo);
            hi = fmaf(A1, B0, hi);
            hi = fmaf(A3, B1, hi);
        }
    }
}

} // namespace ninfer::ops::volta

#endif // NINFER_VOLTA_EMULATION
