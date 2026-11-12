#pragma once

#include <cuda_runtime.h>

namespace ninfer::ops {

inline constexpr int kWarpSize          = 32;
inline constexpr unsigned kFullWarpMask = 0xffffffffu;

template <int Width = kWarpSize, class T>
__device__ __forceinline__ T warp_sum(T x, unsigned mask = kFullWarpMask) {
    static_assert(Width > 0 && Width <= kWarpSize && (Width & (Width - 1)) == 0);
#pragma unroll
    for (int offset = Width / 2; offset > 0; offset >>= 1) {
        x += __shfl_xor_sync(mask, x, offset, Width);
    }
    return x;
}

template <int Width = kWarpSize, class T>
__device__ __forceinline__ T warp_reduce_sum(T x, unsigned mask = kFullWarpMask) {
    static_assert(Width > 0 && Width <= kWarpSize && (Width & (Width - 1)) == 0);
#pragma unroll
    for (int offset = Width / 2; offset > 0; offset >>= 1) {
        x += __shfl_down_sync(mask, x, offset, Width);
    }
    return x;
}

template <int Width = kWarpSize>
__device__ __forceinline__ float warp_max(float x, unsigned mask = kFullWarpMask) {
    static_assert(Width > 0 && Width <= kWarpSize && (Width & (Width - 1)) == 0);
#pragma unroll
    for (int offset = Width / 2; offset > 0; offset >>= 1) {
        x = fmaxf(x, __shfl_xor_sync(mask, x, offset, Width));
    }
    return x;
}

template <int BlockSize>
__device__ __forceinline__ float block_reduce_sum(float x, float* sums) {
    static_assert(BlockSize >= kWarpSize && BlockSize <= 1024);
    static_assert((BlockSize & (BlockSize - 1)) == 0);
    constexpr int Warps = BlockSize / kWarpSize;

    x = warp_reduce_sum(x);
    if constexpr (Warps == 1) { return x; }

    const int lane = threadIdx.x & (kWarpSize - 1);
    const int warp = threadIdx.x / kWarpSize;
    if (lane == 0) { sums[warp] = x; }
    __syncthreads();

    x = threadIdx.x < Warps ? sums[lane] : 0.0f;
    if (warp == 0) { x = warp_reduce_sum<Warps>(x); }
    return x;
}

// Unsigned reductions over the lanes named by `mask`. sm_80+ has the single-instruction
// __reduce_*_sync intrinsics; Volta has none, so there they are built from shuffles. Every lane in
// `mask` must call with the same mask and receives the same result, as with the intrinsics.

namespace detail {

struct ReduceMax {
    __device__ static unsigned apply(unsigned a, unsigned b) { return a > b ? a : b; }
};
struct ReduceAnd {
    __device__ static unsigned apply(unsigned a, unsigned b) { return a & b; }
};
struct ReduceOr {
    __device__ static unsigned apply(unsigned a, unsigned b) { return a | b; }
};
struct ReduceAdd {
    __device__ static unsigned apply(unsigned a, unsigned b) { return a + b; }
};

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ < 800
template <class Op>
__device__ __forceinline__ unsigned shuffle_reduce(unsigned mask, unsigned value) {
    if (mask == 0xFFFFFFFFu) {
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            value = Op::apply(value, __shfl_xor_sync(mask, value, offset));
        }
        return value;
    }
    // A partial mask (the sampler merges one lane per warp): fold the member lanes in order.
    unsigned rest = mask & (mask - 1);
    unsigned acc  = __shfl_sync(mask, value, __ffs(static_cast<int>(mask)) - 1);
    while (rest != 0) {
        acc = Op::apply(acc, __shfl_sync(mask, value, __ffs(static_cast<int>(rest)) - 1));
        rest &= rest - 1;
    }
    return acc;
}
#endif

} // namespace detail

__device__ __forceinline__ unsigned masked_reduce_max(unsigned mask, unsigned value) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    return __reduce_max_sync(mask, value);
#else
    return detail::shuffle_reduce<detail::ReduceMax>(mask, value);
#endif
}

__device__ __forceinline__ unsigned masked_reduce_and(unsigned mask, unsigned value) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    return __reduce_and_sync(mask, value);
#else
    return detail::shuffle_reduce<detail::ReduceAnd>(mask, value);
#endif
}

__device__ __forceinline__ unsigned masked_reduce_or(unsigned mask, unsigned value) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    return __reduce_or_sync(mask, value);
#else
    return detail::shuffle_reduce<detail::ReduceOr>(mask, value);
#endif
}

__device__ __forceinline__ unsigned masked_reduce_add(unsigned mask, unsigned value) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    return __reduce_add_sync(mask, value);
#else
    return detail::shuffle_reduce<detail::ReduceAdd>(mask, value);
#endif
}

} // namespace ninfer::ops
