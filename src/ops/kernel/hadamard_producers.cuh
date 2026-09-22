#pragma once

// ninfer::ops - producers of a Hadamard-rotated projection input that apply the forward transform
// on the way out. Each kernel computes its values exactly as the unfused op does, rounds them to
// BF16 exactly as that op stores them, and hands them to hadamard_1024_forward_store, so the fused
// output is bit-identical to the op followed by hadamard_transform.

#include "ops/common/math.cuh"
#include "ops/common/memory.cuh"
#include "ops/common/warp.cuh"
#include "ops/kernel/hadamard_transform.cuh"
#include "ops/kernel/rmsnorm.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops {

inline constexpr int kRmsnormHadamardWidth = 5120;

// rmsnorm_cta_bf16x2_kernel<Epilogue, 256, 10, true, 5120> (the route rmsnorm takes for every
// 5120-wide row), then the forward transform of the row: the normalised row is staged in shared
// memory as BF16 and five warps transform one 1024-block each.
template <RmsEpilogue Epilogue>
__launch_bounds__(256) __global__
    void rmsnorm_hadamard_5120_kernel(const __nv_bfloat162* __restrict__ x,
                                      const __nv_bfloat162* __restrict__ weight,
                                      const __nv_bfloat16* __restrict__ signs,
                                      __nv_bfloat16* __restrict__ out, std::int64_t rows,
                                      float eps) {
    constexpr int kBlock          = 256;
    constexpr int kWidth          = kRmsnormHadamardWidth;
    constexpr int kPairs          = kWidth / 2;
    constexpr int kPairsPerThread = kPairs / kBlock;
    const std::int64_t row        = static_cast<std::int64_t>(blockIdx.x);
    if (row >= rows) { return; }

    const std::int64_t row_base = row * static_cast<std::int64_t>(kPairs);
    __nv_bfloat162 values[kPairsPerThread];
    __nv_bfloat162 weights[kPairsPerThread];
    float sum = 0.0f;

#pragma unroll
    for (int k = 0; k < kPairsPerThread; ++k) {
        const int pair  = static_cast<int>(threadIdx.x) + k * kBlock;
        values[k]       = x[row_base + pair];
        weights[k]      = weight[pair];
        const float2 xf = __bfloat1622float2(values[k]);
        sum += xf.x * xf.x + xf.y * xf.y;
    }

    __shared__ float warp_sums[kBlock / kWarpSize];
    __shared__ float inv_shared;
    __shared__ alignas(16) __nv_bfloat162 staged[kPairs];
    const float block_sum = block_reduce_sum<kBlock>(sum, warp_sums);
    if (threadIdx.x == 0) { inv_shared = rsqrtf(block_sum / static_cast<float>(kWidth) + eps); }
    __syncthreads();
    const float inv = inv_shared;

#pragma unroll
    for (int k = 0; k < kPairsPerThread; ++k) {
        const int pair  = static_cast<int>(threadIdx.x) + k * kBlock;
        const float2 xf = __bfloat1622float2(values[k]);
        const float2 wf = __bfloat1622float2(weights[k]);
        staged[pair]    = __floats2bfloat162_rn(rmsnorm_epilogue<Epilogue>(xf.x, inv, wf.x, 0.0f),
                                                rmsnorm_epilogue<Epilogue>(xf.y, inv, wf.y, 0.0f));
    }
    __syncthreads();

    const int lane = static_cast<int>(threadIdx.x) & (kWarpSize - 1);
    const int warp = static_cast<int>(threadIdx.x) / kWarpSize;
    if (warp < kWidth / kHadamardTransformBlock) {
        const int block_base = warp * kHadamardTransformBlock;
        float v[kHadamardTransformLaneVectors][8];
        hadamard_1024_load_shared(reinterpret_cast<const __nv_bfloat16*>(staged) + block_base, v,
                                  lane);
        hadamard_1024_forward_store(v, signs + block_base, out + row * kWidth + block_base, lane);
    }
}

inline constexpr int kGatedRmsnormHadamardHeadDim    = 128;
inline constexpr int kGatedRmsnormHadamardRowsPerCta = 16;

// rmsnorm_warp_bf16x2_kernel<Gated, 512, *> over 128-wide head rows (one warp per row), then the
// forward transform over each token's heads laid end to end. Sixteen rows per CTA are two
// 1024-blocks of one token, so the launcher requires the per-token head count to be a multiple of
// sixteen.
__launch_bounds__(kGatedRmsnormHadamardRowsPerCta* kWarpSize) __global__
    void gated_rmsnorm_hadamard_d128_kernel(const __nv_bfloat162* __restrict__ x,
                                            const __nv_bfloat162* __restrict__ weight,
                                            const __nv_bfloat162* __restrict__ z,
                                            const __nv_bfloat16* __restrict__ signs,
                                            __nv_bfloat16* __restrict__ out,
                                            std::int32_t heads_per_column, std::int64_t rows,
                                            float eps) {
    constexpr int kD               = kGatedRmsnormHadamardHeadDim;
    constexpr int kPairs           = kD / 2;
    constexpr int kMaxPairsPerLane = 4;
    constexpr int kRows            = kGatedRmsnormHadamardRowsPerCta;
    const int lane                 = static_cast<int>(threadIdx.x) & (kWarpSize - 1);
    const int warp                 = static_cast<int>(threadIdx.x) / kWarpSize;
    const std::int64_t first_row   = static_cast<std::int64_t>(blockIdx.x) * kRows;
    const std::int64_t row         = first_row + warp;

    __shared__ alignas(16) __nv_bfloat162 staged[kRows * kPairs];
    if (row < rows) {
        const std::int64_t row_base = row * kPairs;
        __nv_bfloat162 values[kMaxPairsPerLane];
        float sum = 0.0f;
#pragma unroll
        for (int k = 0; k < kMaxPairsPerLane; ++k) {
            const int pair = lane + k * kWarpSize;
            if (pair < kPairs) {
                values[k]       = x[row_base + pair];
                const float2 xf = __bfloat1622float2(values[k]);
                sum += xf.x * xf.x + xf.y * xf.y;
            }
        }
        sum       = warp_reduce_sum(sum);
        float inv = lane == 0 ? rsqrtf(sum / static_cast<float>(kD) + eps) : 0.0f;
        inv       = __shfl_sync(kFullWarpMask, inv, 0);
#pragma unroll
        for (int k = 0; k < kMaxPairsPerLane; ++k) {
            const int pair = lane + k * kWarpSize;
            if (pair < kPairs) {
                const float2 xf              = __bfloat1622float2(values[k]);
                const float2 wf              = __bfloat1622float2(weight[pair]);
                const float2 zf              = __bfloat1622float2(z[row_base + pair]);
                staged[warp * kPairs + pair] = __floats2bfloat162_rn(
                    rmsnorm_epilogue<RmsEpilogue::Gated>(xf.x, inv, wf.x, zf.x),
                    rmsnorm_epilogue<RmsEpilogue::Gated>(xf.y, inv, wf.y, zf.y));
            }
        }
    }
    __syncthreads();

    constexpr int kBlocksPerCta = kRows * kD / kHadamardTransformBlock;
    if (warp < kBlocksPerCta && first_row + (warp + 1) * (kHadamardTransformBlock / kD) <= rows) {
        const std::int64_t column = first_row / heads_per_column;
        const int head            = static_cast<int>(first_row - column * heads_per_column);
        const std::int64_t width  = static_cast<std::int64_t>(heads_per_column) * kD;
        const std::int64_t offset =
            static_cast<std::int64_t>(head) * kD + warp * kHadamardTransformBlock;
        float v[kHadamardTransformLaneVectors][8];
        hadamard_1024_load_shared(reinterpret_cast<const __nv_bfloat16*>(staged) +
                                      warp * kHadamardTransformBlock,
                                  v, lane);
        hadamard_1024_forward_store(v, signs + offset, out + column * width + offset, lane);
    }
}

// sigmoid_mul's BF16 product x * sigmoid(gate), then the forward transform: one warp per
// (column, 1024-block). out may alias x exactly; each warp reads its block before writing it.
template <int WarpsPerCta>
__launch_bounds__(WarpsPerCta* kWarpSize) __global__
    void sigmoid_mul_hadamard_1024_kernel(const __nv_bfloat16* __restrict__ gate,
                                          const __nv_bfloat16* x,
                                          const __nv_bfloat16* __restrict__ signs,
                                          __nv_bfloat16* out, std::int64_t items,
                                          std::int32_t blocks_per_column) {
    const int lane          = static_cast<int>(threadIdx.x) & (kWarpSize - 1);
    const int warp          = static_cast<int>(threadIdx.x) / kWarpSize;
    const std::int64_t item = static_cast<std::int64_t>(blockIdx.x) * WarpsPerCta + warp;
    if (item >= items) { return; }
    const std::int64_t column = item / blocks_per_column;
    const int block           = static_cast<int>(item - column * blocks_per_column);
    const std::int64_t base =
        column * static_cast<std::int64_t>(blocks_per_column) * kHadamardTransformBlock +
        static_cast<std::int64_t>(block) * kHadamardTransformBlock;

    float v[kHadamardTransformLaneVectors][8];
#pragma unroll
    for (int r = 0; r < kHadamardTransformLaneVectors; ++r) {
        const int offset = hadamard_lane_offset(lane, r);
        float g[8];
        float a[8];
        hadamard_unpack8(load_vec<uint4>(gate + base + offset), g);
        hadamard_unpack8(*reinterpret_cast<const uint4*>(x + base + offset), a);
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            v[r][j] = __bfloat162float(__float2bfloat16_rn(a[j] * sigmoid(g[j])));
        }
    }
    hadamard_1024_forward_store(
        v, signs + static_cast<std::int64_t>(block) * kHadamardTransformBlock, out + base, lane);
}

} // namespace ninfer::ops
