#pragma once

// ninfer::ops - normalized 1024-point Sylvester Walsh-Hadamard transform with a sign vector, one
// warp per (column, 1024-block). Lane l holds elements 8l + j + 256r in v[r][j] (j = 0..7,
// r = 0..3), so every warp-wide load or store is 512 contiguous bytes: index bits 0..2 and 8..9 are
// butterflied within each lane's registers, bits 3..7 across lanes with shuffles. The same
// butterfly and sign convention as ops/kv_cache/hadamard_d256.cuh: the low element of a pair takes
// x + y, the high element x - y.

#include "ops/common/math.cuh"
#include "ops/common/memory.cuh"
#include "ops/common/warp.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops {

inline constexpr int kHadamardTransformBlock        = 1024;
inline constexpr int kHadamardTransformLaneElements = kHadamardTransformBlock / kWarpSize;
inline constexpr int kHadamardTransformLaneVectors  = kHadamardTransformLaneElements / 8;
inline constexpr float kHadamardTransformNormalizer = 0.03125f; // 2^-5 = 1 / sqrt(1024)

// Byte offset of lane vector r inside a 1024-element block.
__device__ __forceinline__ int hadamard_lane_offset(int lane, int r) {
    return r * (kWarpSize * 8) + lane * 8;
}

__device__ __forceinline__ void hadamard_unpack8(const uint4& bits, float (&values)[8]) {
    const auto* pairs = reinterpret_cast<const __nv_bfloat162*>(&bits);
#pragma unroll
    for (int p = 0; p < 4; ++p) {
        const float2 f    = __bfloat1622float2(pairs[p]);
        values[2 * p]     = f.x;
        values[2 * p + 1] = f.y;
    }
}

__device__ __forceinline__ uint4 hadamard_pack8(const float (&values)[8]) {
    uint4 bits;
    auto* pairs = reinterpret_cast<__nv_bfloat162*>(&bits);
#pragma unroll
    for (int p = 0; p < 4; ++p) {
        pairs[p] = __floats2bfloat162_rn(values[2 * p], values[2 * p + 1]);
    }
    return bits;
}

// The ten butterfly stages over one lane's 32 elements of a 1024-block.
__device__ __forceinline__ void
hadamard_1024_butterfly(float (&v)[kHadamardTransformLaneVectors][8], int lane) {
    // Index bits 0..2 live in j.
#pragma unroll
    for (int stride = 1; stride < 8; stride <<= 1) {
#pragma unroll
        for (int r = 0; r < kHadamardTransformLaneVectors; ++r) {
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                if ((j & stride) == 0) {
                    const float low  = v[r][j];
                    const float high = v[r][j + stride];
                    v[r][j]          = __fadd_rn(low, high);
                    v[r][j + stride] = __fsub_rn(low, high);
                }
            }
        }
    }
    // Index bits 3..7 live in the lane id: partner lanes differ in one bit.
#pragma unroll
    for (int stride = 1; stride < kWarpSize; stride <<= 1) {
#pragma unroll
        for (int r = 0; r < kHadamardTransformLaneVectors; ++r) {
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const float peer = __shfl_xor_sync(kFullWarpMask, v[r][j], stride);
                v[r][j] =
                    (lane & stride) == 0 ? __fadd_rn(v[r][j], peer) : __fsub_rn(peer, v[r][j]);
            }
        }
    }
    // Index bits 8..9 live in r.
#pragma unroll
    for (int stride = 1; stride < kHadamardTransformLaneVectors; stride <<= 1) {
#pragma unroll
        for (int r = 0; r < kHadamardTransformLaneVectors; ++r) {
            if ((r & stride) == 0) {
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    const float low  = v[r][j];
                    const float high = v[r + stride][j];
                    v[r][j]          = __fadd_rn(low, high);
                    v[r + stride][j] = __fsub_rn(low, high);
                }
            }
        }
    }
}

template <bool SignsAfter, int WarpsPerCta>
__launch_bounds__(WarpsPerCta* kWarpSize) __global__
    void hadamard_transform_1024_kernel(const __nv_bfloat16* __restrict__ x,
                                        const __nv_bfloat16* __restrict__ signs, __nv_bfloat16* out,
                                        std::int64_t items, std::int32_t blocks_per_column) {
    const int lane          = static_cast<int>(threadIdx.x) & (kWarpSize - 1);
    const int warp          = static_cast<int>(threadIdx.x) / kWarpSize;
    const std::int64_t item = static_cast<std::int64_t>(blockIdx.x) * WarpsPerCta + warp;
    if (item >= items) { return; }
    const std::int64_t column = item / blocks_per_column;
    const int block           = static_cast<int>(item - column * blocks_per_column);
    const std::int64_t base =
        column * static_cast<std::int64_t>(blocks_per_column) * kHadamardTransformBlock +
        static_cast<std::int64_t>(block) * kHadamardTransformBlock;
    const __nv_bfloat16* block_signs =
        signs + static_cast<std::int64_t>(block) * kHadamardTransformBlock;

    float v[kHadamardTransformLaneVectors][8];
    float s[kHadamardTransformLaneVectors][8];
#pragma unroll
    for (int r = 0; r < kHadamardTransformLaneVectors; ++r) {
        const int offset = hadamard_lane_offset(lane, r);
        hadamard_unpack8(load_vec<uint4>(x + base + offset), v[r]);
        hadamard_unpack8(load_vec<uint4>(block_signs + offset), s[r]);
        if constexpr (!SignsAfter) {
#pragma unroll
            for (int j = 0; j < 8; ++j) { v[r][j] *= s[r][j]; }
        }
    }

    hadamard_1024_butterfly(v, lane);

#pragma unroll
    for (int r = 0; r < kHadamardTransformLaneVectors; ++r) {
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            float value = __fmul_rn(v[r][j], kHadamardTransformNormalizer);
            if constexpr (SignsAfter) { value *= s[r][j]; }
            v[r][j] = value;
        }
        store_vec(out + base + hadamard_lane_offset(lane, r), hadamard_pack8(v[r]));
    }
}

// out[b*1024+i, t] = 2^-5 * sum_j H[i][j] * signs[b*1024+j] * silu(gate[b*1024+j, t]) *
// up[b*1024+j, t], with gate and up the two halves of one [2*I, T] plane: the down-projection
// input of a Hadamard-rotated checkpoint in one pass over the SwiGLU plane.
template <int WarpsPerCta>
__launch_bounds__(WarpsPerCta* kWarpSize) __global__
    void silu_mul_hadamard_1024_kernel(const __nv_bfloat16* __restrict__ plane,
                                       const __nv_bfloat16* __restrict__ signs,
                                       __nv_bfloat16* __restrict__ out, std::int64_t items,
                                       std::int32_t blocks_per_column) {
    const int lane          = static_cast<int>(threadIdx.x) & (kWarpSize - 1);
    const int warp          = static_cast<int>(threadIdx.x) / kWarpSize;
    const std::int64_t item = static_cast<std::int64_t>(blockIdx.x) * WarpsPerCta + warp;
    if (item >= items) { return; }
    const std::int64_t column = item / blocks_per_column;
    const int block           = static_cast<int>(item - column * blocks_per_column);
    const std::int64_t width =
        static_cast<std::int64_t>(blocks_per_column) * kHadamardTransformBlock;
    const std::int64_t block_base    = static_cast<std::int64_t>(block) * kHadamardTransformBlock;
    const __nv_bfloat16* gate        = plane + column * 2 * width + block_base;
    const __nv_bfloat16* up          = gate + width;
    const __nv_bfloat16* block_signs = signs + block_base;
    __nv_bfloat16* out_block         = out + column * width + block_base;

    float v[kHadamardTransformLaneVectors][8];
#pragma unroll
    for (int r = 0; r < kHadamardTransformLaneVectors; ++r) {
        const int offset = hadamard_lane_offset(lane, r);
        float g[8];
        float u[8];
        float s[8];
        hadamard_unpack8(load_vec<uint4>(gate + offset), g);
        hadamard_unpack8(load_vec<uint4>(up + offset), u);
        hadamard_unpack8(load_vec<uint4>(block_signs + offset), s);
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            // The SwiGLU value is rounded to BF16 first, exactly as the two-op composition
            // (silu_mul then hadamard_transform) stores it.
            const float activation = __bfloat162float(__float2bfloat16_rn(silu(g[j]) * u[j]));
            v[r][j]                = activation * s[j];
        }
    }

    hadamard_1024_butterfly(v, lane);

#pragma unroll
    for (int r = 0; r < kHadamardTransformLaneVectors; ++r) {
#pragma unroll
        for (int j = 0; j < 8; ++j) { v[r][j] = __fmul_rn(v[r][j], kHadamardTransformNormalizer); }
        store_vec(out_block + hadamard_lane_offset(lane, r), hadamard_pack8(v[r]));
    }
}

} // namespace ninfer::ops
