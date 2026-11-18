#pragma once

// Implements: include/ninfer/ops/top_logprobs.h
// Match: contiguous BF16 [physical_rows,C]; I32 [C]; FP32 [C]; I32 and FP32 [k,C].
// Algorithm assumptions: one CTA per column. A stable two-pass logsumexp, then k selection rounds,
// each a block-wide maximum of a 64-bit key that orders (logit descending, row ascending).

#include "ops/common/warp.cuh"
#include "ops/kernel/target_logprobs.cuh"

#include <cuda_bf16.h>
#include <math_constants.h>

#include <cstdint>

namespace ninfer::ops {

inline constexpr int kTopLogprobsBlock = 1024;

// Monotone map of a finite float to an unsigned key (larger float, larger key).
__device__ __forceinline__ std::uint32_t top_logprobs_float_key(float value) {
    const std::uint32_t bits = __float_as_uint(value);
    return (bits & 0x80000000u) != 0u ? ~bits : (bits | 0x80000000u);
}

// Larger key means larger logit, and among equal logits the lower row.
__device__ __forceinline__ unsigned long long top_logprobs_key(float value, std::int32_t row) {
    return (static_cast<unsigned long long>(top_logprobs_float_key(value)) << 32) |
           static_cast<unsigned long long>(0xffffffffu - static_cast<std::uint32_t>(row));
}

template <int BlockSize>
__device__ __forceinline__ unsigned long long top_logprobs_block_max_key(unsigned long long key) {
    constexpr int kWarps = BlockSize / kWarpSize;
    __shared__ unsigned long long warp_keys[kWarps];
    __shared__ unsigned long long result;

    const int lane = static_cast<int>(threadIdx.x) & (kWarpSize - 1);
    const int warp = static_cast<int>(threadIdx.x) / kWarpSize;
#pragma unroll
    for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
        const unsigned long long other = __shfl_xor_sync(kFullWarpMask, key, offset);
        key                            = other > key ? other : key;
    }
    if (lane == 0) { warp_keys[warp] = key; }
    __syncthreads();
    if (warp == 0) {
        key = lane < kWarps ? warp_keys[lane] : 0ull;
#pragma unroll
        for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
            const unsigned long long other = __shfl_xor_sync(kFullWarpMask, key, offset);
            key                            = other > key ? other : key;
        }
        if (lane == 0) { result = key; }
    }
    __syncthreads();
    return result;
}

template <int BlockSize>
__launch_bounds__(BlockSize) __global__
    void top_logprobs_kernel(const __nv_bfloat16* logits, const std::int32_t* chosen_ids,
                             float* chosen_logprobs, std::int32_t* top_ids, float* top_logprobs,
                             std::int32_t valid_rows, std::int32_t physical_rows, std::int32_t k) {
    const std::int32_t column = static_cast<std::int32_t>(blockIdx.x);
    const __nv_bfloat16* col  = logits + static_cast<std::int64_t>(column) * physical_rows;

    float local_max = -CUDART_INF_F;
    for (std::int32_t row = static_cast<std::int32_t>(threadIdx.x); row < valid_rows;
         row += BlockSize) {
        local_max = fmaxf(local_max, __bfloat162float(col[row]));
    }
    const float maximum = target_logprobs_block_max<BlockSize>(local_max);

    float local_sum = 0.0f;
    for (std::int32_t row = static_cast<std::int32_t>(threadIdx.x); row < valid_rows;
         row += BlockSize) {
        local_sum += expf(__bfloat162float(col[row]) - maximum);
    }
    __shared__ float warp_sums[BlockSize / kWarpSize];
    const float sum       = block_reduce_sum<BlockSize>(local_sum, warp_sums);
    const float log_denom = maximum + logf(sum);

    if (threadIdx.x == 0) {
        const std::int32_t chosen = chosen_ids[column];
        chosen_logprobs[column] =
            chosen >= 0 && chosen < valid_rows ? __bfloat162float(col[chosen]) - log_denom
                                               : __int_as_float(0x7fc00000);
    }

    // Round i keeps the best key strictly below round i-1's winner.
    unsigned long long previous = ~0ull;
    for (std::int32_t i = 0; i < k; ++i) {
        unsigned long long best = 0ull;
        for (std::int32_t row = static_cast<std::int32_t>(threadIdx.x); row < valid_rows;
             row += BlockSize) {
            const unsigned long long key = top_logprobs_key(__bfloat162float(col[row]), row);
            if (key < previous && key > best) { best = key; }
        }
        previous = top_logprobs_block_max_key<BlockSize>(best);
        if (threadIdx.x == 0) {
            const std::int32_t row = static_cast<std::int32_t>(
                0xffffffffu - static_cast<std::uint32_t>(previous & 0xffffffffull));
            const std::int64_t at = static_cast<std::int64_t>(column) * k + i;
            top_ids[at]           = row;
            top_logprobs[at]      = __bfloat162float(col[row]) - log_denom;
        }
    }
}

} // namespace ninfer::ops
