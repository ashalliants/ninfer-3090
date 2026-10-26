#pragma once

// Implements: include/ninfer/ops/argmax_probability.h
// Match: contiguous FP32 [physical_rows,C] (any values), valid_rows <= physical_rows, I32 [C], FP32 [C].
// Algorithm assumptions: a first pass reduces each kArgmaxProbabilityTileRows-row tile of a column
// to its BF16-key winner (value, lower id), its FP32 maximum and its sum of exp(l - tile max); a
// second pass, one CTA per column, combines the tiles in a fixed order (deterministic, no atomics)
// and seeds the winner with row 0 as ops::argmax does, so a NaN key at row 0 selects row 0.
// Each term's argument l - max is rounded to FP32 before expf; its rounding residual (TwoSum) is
// added back to first order, so a coherent argument error (many equal rows, |l - max| in [8, 16))
// does not reach the sum. The terms accumulate in FP64.

#include <cuda_bf16.h>
#include <math_constants.h>

#include <climits>
#include <cstdint>

namespace ninfer::ops {

inline constexpr int kArgmaxProbabilityBlock    = 256;
inline constexpr int kArgmaxProbabilityItems    = 8;
inline constexpr int kArgmaxProbabilityTileRows = kArgmaxProbabilityBlock * kArgmaxProbabilityItems;
inline constexpr int kArgmaxProbabilityWarps    = kArgmaxProbabilityBlock / 32;

// The argmax key: the logit's BF16 rounding (ops::cast_fp32_to_bf16's), so ties follow the BF16 argmax.
__device__ __forceinline__ float argmax_probability_key(float logit) {
    return __bfloat162float(__float2bfloat16_rn(logit));
}

__device__ __forceinline__ bool argmax_probability_better(float key, std::int32_t index, float best_key,
                                                          std::int32_t best_index) {
    return key > best_key || (key == best_key && index < best_index);
}

// Block-wide (key, lower index) winner and maximum; every thread receives the result.
__device__ __forceinline__ void argmax_probability_block_winner(float& key, std::int32_t& index, float& maximum) {
    __shared__ float warp_keys[kArgmaxProbabilityWarps];
    __shared__ std::int32_t warp_indices[kArgmaxProbabilityWarps];
    __shared__ float warp_maxima[kArgmaxProbabilityWarps];
    constexpr unsigned kMask = 0xffffffffu;
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    for (int offset = 16; offset > 0; offset >>= 1) {
        const float other_key          = __shfl_down_sync(kMask, key, offset);
        const std::int32_t other_index = __shfl_down_sync(kMask, index, offset);
        maximum                        = fmaxf(maximum, __shfl_down_sync(kMask, maximum, offset));
        if (argmax_probability_better(other_key, other_index, key, index)) {
            key   = other_key;
            index = other_index;
        }
    }
    if (lane == 0) {
        warp_keys[warp]    = key;
        warp_indices[warp] = index;
        warp_maxima[warp]  = maximum;
    }
    __syncthreads();
    if (warp == 0) {
        key     = lane < kArgmaxProbabilityWarps ? warp_keys[lane] : -CUDART_INF_F;
        index   = lane < kArgmaxProbabilityWarps ? warp_indices[lane] : INT32_MAX;
        maximum = lane < kArgmaxProbabilityWarps ? warp_maxima[lane] : -CUDART_INF_F;
        for (int offset = 16; offset > 0; offset >>= 1) {
            const float other_key          = __shfl_down_sync(kMask, key, offset);
            const std::int32_t other_index = __shfl_down_sync(kMask, index, offset);
            maximum                        = fmaxf(maximum, __shfl_down_sync(kMask, maximum, offset));
            if (argmax_probability_better(other_key, other_index, key, index)) {
                key   = other_key;
                index = other_index;
            }
        }
        if (lane == 0) {
            warp_keys[0]    = key;
            warp_indices[0] = index;
            warp_maxima[0]  = maximum;
        }
    }
    __syncthreads();
    key     = warp_keys[0];
    index   = warp_indices[0];
    maximum = warp_maxima[0];
}

// exp(logit - maximum) for logit <= maximum: expf of the FP32-rounded difference, corrected by the
// difference's rounding residual (exp(d + r) ~ exp(d) (1 + r)). -inf contributes 0, and so does a
// finite logit whose difference overflows FP32 to -inf (its residual would be NaN); NaN, and +inf
// (inf - inf), give NaN.
__device__ __forceinline__ float argmax_probability_term(float logit, float maximum) {
    if (logit == -CUDART_INF_F) { return 0.0f; }
    const float negated  = -maximum;
    const float d        = logit + negated;
    if (d == -CUDART_INF_F) { return 0.0f; }
    const float virtual_ = d - logit;
    const float residual = (logit - (d - virtual_)) + (negated - virtual_);
    const float e        = expf(d);
    return fmaf(e, residual, e);
}

// Block-wide sum; thread 0 receives the result.
__device__ __forceinline__ double argmax_probability_block_sum(double value) {
    __shared__ double warp_sums[kArgmaxProbabilityWarps];
    constexpr unsigned kMask = 0xffffffffu;
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    for (int offset = 16; offset > 0; offset >>= 1) { value += __shfl_down_sync(kMask, value, offset); }
    if (lane == 0) { warp_sums[warp] = value; }
    __syncthreads();
    if (warp == 0) {
        value = lane < kArgmaxProbabilityWarps ? warp_sums[lane] : 0.0;
        for (int offset = 16; offset > 0; offset >>= 1) { value += __shfl_down_sync(kMask, value, offset); }
    }
    return value;
}

// grid (tiles, C): tile blockIdx.x of column blockIdx.y. Partial p = column * tiles + tile.
__launch_bounds__(kArgmaxProbabilityBlock) __global__
    void argmax_probability_tiles_kernel(const float* logits, float* keys, std::int32_t* indices, float* maxima,
                                         double* sums, std::int32_t valid_rows, std::int32_t physical_rows) {
    const std::int64_t base = static_cast<std::int64_t>(blockIdx.y) * physical_rows;
    const std::int32_t first = static_cast<std::int32_t>(blockIdx.x) * kArgmaxProbabilityTileRows;

    float values[kArgmaxProbabilityItems];
    float key = -CUDART_INF_F, maximum = -CUDART_INF_F;
    std::int32_t index = INT32_MAX;
#pragma unroll
    for (int item = 0; item < kArgmaxProbabilityItems; ++item) {
        const std::int32_t v = first + item * kArgmaxProbabilityBlock + static_cast<std::int32_t>(threadIdx.x);
        values[item]         = -CUDART_INF_F;
        if (v < valid_rows) {
            const float logit = logits[base + v];
            values[item]      = logit;
            maximum           = fmaxf(maximum, logit);
            const float k     = argmax_probability_key(logit);
            if (argmax_probability_better(k, v, key, index)) {
                key   = k;
                index = v;
            }
        }
    }
    argmax_probability_block_winner(key, index, maximum);

    double sum = 0.0;
#pragma unroll
    for (int item = 0; item < kArgmaxProbabilityItems; ++item) {
        const std::int32_t v = first + item * kArgmaxProbabilityBlock + static_cast<std::int32_t>(threadIdx.x);
        if (v < valid_rows) { sum += static_cast<double>(argmax_probability_term(values[item], maximum)); }
    }
    sum = argmax_probability_block_sum(sum);
    if (threadIdx.x == 0) {
        const std::size_t p = static_cast<std::size_t>(blockIdx.y) * gridDim.x + blockIdx.x;
        keys[p]             = key;
        indices[p]          = index;
        maxima[p]           = maximum;
        sums[p]             = sum;
    }
}

// grid (C): column blockIdx.x combines its `tiles` partials into ids and probs.
__launch_bounds__(kArgmaxProbabilityBlock) __global__
    void argmax_probability_combine_kernel(const float* logits, const float* keys, const std::int32_t* indices,
                                           const float* maxima, const double* sums, std::int32_t* ids, float* probs,
                                           std::int32_t tiles, std::int32_t physical_rows) {
    const std::int32_t t    = static_cast<std::int32_t>(blockIdx.x);
    const std::size_t first = static_cast<std::size_t>(t) * tiles;
    float key = -CUDART_INF_F, maximum = -CUDART_INF_F;
    std::int32_t index = INT32_MAX;
    for (std::int32_t p = static_cast<std::int32_t>(threadIdx.x); p < tiles; p += kArgmaxProbabilityBlock) {
        maximum = fmaxf(maximum, maxima[first + p]);
        if (argmax_probability_better(keys[first + p], indices[first + p], key, index)) {
            key   = keys[first + p];
            index = indices[first + p];
        }
    }
    argmax_probability_block_winner(key, index, maximum);

    double sum = 0.0;
    for (std::int32_t p = static_cast<std::int32_t>(threadIdx.x); p < tiles; p += kArgmaxProbabilityBlock) {
        // A tile without a finite or +inf maximum (every row -inf or NaN) adds its sum as is: 0, or NaN.
        const float tile_max = maxima[first + p];
        sum += tile_max == -CUDART_INF_F
                   ? sums[first + p]
                   : sums[first + p] * exp(static_cast<double>(tile_max) - static_cast<double>(maximum));
    }
    sum = argmax_probability_block_sum(sum);
    if (threadIdx.x == 0) {
        const std::int64_t base = static_cast<std::int64_t>(t) * physical_rows;
        // ops::argmax's winner starts at row 0 and only a strictly better key replaces it: a NaN key at
        // row 0 is never replaced, and a NaN key elsewhere never wins.
        if (!argmax_probability_better(key, index, argmax_probability_key(logits[base]), 0)) { index = 0; }
        const float winner = logits[base + index];
        ids[t]             = index;
        probs[t]           = static_cast<float>(exp(static_cast<double>(winner) - static_cast<double>(maximum)) / sum);
    }
}

} // namespace ninfer::ops
