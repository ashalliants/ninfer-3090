#pragma once

// Volta (sm_70) emulation of the Ampere instructions the kernels are written against.
//
// Volta has no cp.async, ldmatrix, or m16n8kK mma.sync, and its tensor cores have no BF16, INT8 or
// INT4 mode. Each helper here reproduces the exact PTX fragment layout of the instruction it
// replaces, so every kernel using ops/common/mma.cuh and ops/common/memory.cuh runs unchanged. The
// mma emulations accumulate in FP32 (or INT32 via dp4a) with all-lane shuffles; they are a correct
// baseline, not a fast path. Fast Volta GEMMs belong in dedicated kernels
// (docs/maintainer/volta-port.md). Kernels must not name the Ampere instructions themselves:
// scripts/check-volta-compat.sh enforces that.

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
#define NINFER_VOLTA_EMULATION 1
#endif

#if defined(NINFER_VOLTA_EMULATION)

#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace ninfer::ops::volta {

constexpr unsigned kFullMask = 0xffffffffu;

__device__ __forceinline__ int laneid() {
    unsigned lane;
    asm("mov.u32 %0, %%laneid;\n" : "=r"(lane));
    return static_cast<int>(lane);
}

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
    const int lane = laneid();
    const unsigned row_addr = __shfl_sync(kFullMask, addr, 8 * i + (lane >> 2));
    return lds32(row_addr + 4 * (lane & 3));
}

// Transposed: each lane receives M[2*(lane%4)][lane/4] and M[2*(lane%4)+1][lane/4].
__device__ __forceinline__ unsigned ldmatrix_word_t(unsigned addr, int i) {
    const int lane = laneid();
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
// Not inlined: expanding ~100 instructions at every call site made ptxas and cicc take over half
// an hour on the attention kernels. By-value float4 keeps the accumulators in registers.
template <class Unpack>
__device__ __noinline__ float4 mma_m16n8k16_impl(float4 c, unsigned a0, unsigned a1, unsigned a2,
                                                 unsigned a3, unsigned b0, unsigned b1) {
    float c0 = c.x, c1 = c.y, c2 = c.z, c3 = c.w;
    const int lane = laneid();
    const int g    = lane >> 2;
    const int t    = lane & 3;
    // Rolled to keep the non-inlined body small.
#pragma unroll 1
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
    return make_float4(c0, c1, c2, c3);
}

template <class Unpack>
__device__ __forceinline__ void mma_m16n8k16(float& c0, float& c1, float& c2, float& c3, unsigned a0,
                                             unsigned a1, unsigned a2, unsigned a3, unsigned b0,
                                             unsigned b1) {
    const float4 r = mma_m16n8k16_impl<Unpack>(make_float4(c0, c1, c2, c3), a0, a1, a2, a3, b0, b1);
    c0 = r.x; c1 = r.y; c2 = r.z; c3 = r.w;
}

// m16n8k32 s8: a0:(row g, k 4t..4t+3) a1:(row g+8, same) a2:(row g, k 16+4t..) a3:(row g+8, 16+4t..)
//              b0:(k 4t..4t+3, n g)   b1:(k 16+4t.., n g)
__device__ __noinline__ int4 mma_m16n8k32_s8_impl(int4 c, unsigned a0, unsigned a1, unsigned a2,
                                                  unsigned a3, unsigned b0, unsigned b1) {
    int c0 = c.x, c1 = c.y, c2 = c.z, c3 = c.w;
    const int lane = laneid();
    const int g    = lane >> 2;
    const int t    = lane & 3;
    // Rolled to keep the non-inlined body small.
#pragma unroll 1
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
    return make_int4(c0, c1, c2, c3);
}

__device__ __forceinline__ void mma_m16n8k32_s8(int& c0, int& c1, int& c2, int& c3, unsigned a0,
                                                unsigned a1, unsigned a2, unsigned a3,
                                                unsigned b0, unsigned b1) {
    const int4 r = mma_m16n8k32_s8_impl(make_int4(c0, c1, c2, c3), a0, a1, a2, a3, b0, b1);
    c0 = r.x; c1 = r.y; c2 = r.z; c3 = r.w;
}

// m16n8k16 s8: a0:(row g, k 4t..4t+3) a1:(row g+8, same)  b0:(k 4t..4t+3, n g). The k32 shape's
// first half.
__device__ __noinline__ int4 mma_m16n8k16_s8_impl(int4 c, unsigned a0, unsigned a1, unsigned b0) {
    int c0 = c.x, c1 = c.y, c2 = c.z, c3 = c.w;
    const int lane = laneid();
    const int g    = lane >> 2;
    const int t    = lane & 3;
#pragma unroll 1
    for (int tp = 0; tp < 4; ++tp) {
        const int src_a = 4 * g + tp;
        const int A0    = static_cast<int>(__shfl_sync(kFullMask, a0, src_a));
        const int A1    = static_cast<int>(__shfl_sync(kFullMask, a1, src_a));
#pragma unroll
        for (int j = 0; j < 2; ++j) {
            const int B0 = static_cast<int>(__shfl_sync(kFullMask, b0, 4 * (2 * t + j) + tp));
            int& lo      = (j == 0) ? c0 : c1;
            int& hi      = (j == 0) ? c2 : c3;
            lo           = __dp4a(A0, B0, lo);
            hi           = __dp4a(A1, B0, hi);
        }
    }
    return make_int4(c0, c1, c2, c3);
}

__device__ __forceinline__ void mma_m16n8k16_s8(int& c0, int& c1, int& c2, int& c3, unsigned a0,
                                                unsigned a1, unsigned b0) {
    const int4 r = mma_m16n8k16_s8_impl(make_int4(c0, c1, c2, c3), a0, a1, b0);
    c0 = r.x; c1 = r.y; c2 = r.z; c3 = r.w;
}

// Eight signed 4-bit values (lowest k in the lowest nibble) as two registers of four sign-extended
// bytes: the even-k nibbles and the odd-k nibbles. A dot product is the same sum in any k order as
// long as both operands are split the same way.
__device__ __forceinline__ void split_s4(unsigned x, int& even, int& odd) {
    constexpr unsigned kLow  = 0x0F0F0F0Fu;
    constexpr unsigned kSign = 0x08080808u;
    // (v ^ 8) - 8 sign-extends a nibble held in a byte; the per-byte subtract cannot borrow
    // across bytes because every byte is at least 8 after the xor.
    even = static_cast<int>(__vsub4((x & kLow) ^ kSign, kSign));
    odd  = static_cast<int>(__vsub4(((x >> 4) & kLow) ^ kSign, kSign));
}

__device__ __forceinline__ int dot8_s4(unsigned x, unsigned y, int acc) {
    int xe, xo, ye, yo;
    split_s4(x, xe, xo);
    split_s4(y, ye, yo);
    return __dp4a(xo, yo, __dp4a(xe, ye, acc));
}

// m16n8k64 s4: a0:(row g, k 8t..8t+7) a1:(row g+8, same) a2:(row g, k 32+8t..) a3:(row g+8, 32+8t..)
//              b0:(k 8t..8t+7, n g)   b1:(k 32+8t.., n g). D = A*B + C with C separate from D.
__device__ __noinline__ int4 mma_m16n8k64_s4_impl(int4 c, unsigned a0, unsigned a1, unsigned a2,
                                                  unsigned a3, unsigned b0, unsigned b1) {
    int c0 = c.x, c1 = c.y, c2 = c.z, c3 = c.w;
    const int lane = laneid();
    const int g    = lane >> 2;
    const int t    = lane & 3;
#pragma unroll 1
    for (int tp = 0; tp < 4; ++tp) {
        const int src_a   = 4 * g + tp;
        const unsigned A0 = __shfl_sync(kFullMask, a0, src_a);
        const unsigned A1 = __shfl_sync(kFullMask, a1, src_a);
        const unsigned A2 = __shfl_sync(kFullMask, a2, src_a);
        const unsigned A3 = __shfl_sync(kFullMask, a3, src_a);
#pragma unroll
        for (int j = 0; j < 2; ++j) {
            const int src_b   = 4 * (2 * t + j) + tp;
            const unsigned B0 = __shfl_sync(kFullMask, b0, src_b);
            const unsigned B1 = __shfl_sync(kFullMask, b1, src_b);
            int& lo           = (j == 0) ? c0 : c1;
            int& hi           = (j == 0) ? c2 : c3;
            lo                = dot8_s4(A2, B1, dot8_s4(A0, B0, lo));
            hi                = dot8_s4(A3, B1, dot8_s4(A1, B0, hi));
        }
    }
    return make_int4(c0, c1, c2, c3);
}

__device__ __forceinline__ void mma_m16n8k64_s4(int& d0, int& d1, int& d2, int& d3, unsigned a0,
                                                unsigned a1, unsigned a2, unsigned a3,
                                                unsigned b0, unsigned b1, int c0, int c1, int c2,
                                                int c3) {
    const int4 r = mma_m16n8k64_s4_impl(make_int4(c0, c1, c2, c3), a0, a1, a2, a3, b0, b1);
    d0 = r.x; d1 = r.y; d2 = r.z; d3 = r.w;
}

// m16n8k8 tf32: a0:(row g, k t) a1:(row g+8, k t) a2:(row g, k t+4) a3:(row g+8, k t+4)
//               b0:(k t, n g)   b1:(k t+4, n g)
__device__ __noinline__ float4 mma_m16n8k8_tf32_impl(float4 c, unsigned a0, unsigned a1,
                                                     unsigned a2, unsigned a3, unsigned b0,
                                                     unsigned b1) {
    float c0 = c.x, c1 = c.y, c2 = c.z, c3 = c.w;
    constexpr unsigned kTf32Mask = 0xffffe000u;
    const int lane               = laneid();
    const int g                  = lane >> 2;
    const int t                  = lane & 3;
    // Rolled to keep the non-inlined body small.
#pragma unroll 1
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
    return make_float4(c0, c1, c2, c3);
}

__device__ __forceinline__ void mma_m16n8k8_tf32(float& c0, float& c1, float& c2, float& c3,
                                                 unsigned a0, unsigned a1, unsigned a2,
                                                 unsigned a3, unsigned b0, unsigned b1) {
    const float4 r = mma_m16n8k8_tf32_impl(make_float4(c0, c1, c2, c3), a0, a1, a2, a3, b0, b1);
    c0 = r.x; c1 = r.y; c2 = r.z; c3 = r.w;
}

} // namespace ninfer::ops::volta

#endif // NINFER_VOLTA_EMULATION
