#pragma once

// ninfer::ops - RMSNorm kernels over contiguous BF16 rows.

#include "ops/common/math.cuh"
#include "ops/common/warp.cuh"
#include "ops/kernel/rope_math.cuh"

#include <cuda_bf16.h>

#include <cstdint>

namespace ninfer::ops {

enum class RmsEpilogue {
    Offset,
    Plain,
    Gated,        // weight * SiLU(z)
    SigmoidGated, // weight * sigmoid(z)
};

// Whether an epilogue reads the gate z.
template <RmsEpilogue Epilogue>
inline constexpr bool kRmsGated = Epilogue == RmsEpilogue::Gated || Epilogue == RmsEpilogue::SigmoidGated;

template <RmsEpilogue Epilogue>
__device__ __forceinline__ float rmsnorm_epilogue(float x, float inv, float weight, float z) {
    if constexpr (Epilogue == RmsEpilogue::Offset) { weight += 1.0f; }
    float value = x * inv * weight;
    if constexpr (Epilogue == RmsEpilogue::Gated) { value *= silu(z); }
    if constexpr (Epilogue == RmsEpilogue::SigmoidGated) { value *= sigmoid(z); }
    return value;
}

// Fast geometry for D in {64, 128, 192, 256}. One warp owns one row, keeps the input in
// registers, and uses only warp shuffles for the reduction. Block is a scheduling choice rather
// than part of the row geometry.
// The stored pair of slot 0 as it is (Slot0 of rmsnorm_warp_row).
struct RmsSlot0Plain {
    __device__ __forceinline__ __nv_bfloat162 operator()(__nv_bfloat162 value) const { return value; }
};

// One row by one warp (the kernels below). slot0 maps each lane's first output pair before it is
// stored (the fused RoPE rotates the rows' first 64 features there).
template <RmsEpilogue Epilogue, bool Prefetch, int FixedD = 0, class Slot0 = RmsSlot0Plain>
__device__ __forceinline__ void rmsnorm_warp_row(const __nv_bfloat162* x, const __nv_bfloat162* weight,
                                                 const __nv_bfloat162* z, __nv_bfloat162* out,
                                                 std::int32_t input_d, std::int64_t row, float eps,
                                                 Slot0 slot0 = {}) {
    const int d = FixedD ? FixedD : input_d;
    constexpr int kMaxPairsPerLane = 4;
    const int lane                 = static_cast<int>(threadIdx.x) & (kWarpSize - 1);

    const int pairs             = d / 2;
    const std::int64_t row_base = row * static_cast<std::int64_t>(pairs);
    __nv_bfloat162 values[kMaxPairsPerLane];
    __nv_bfloat162 weights[kMaxPairsPerLane];
    __nv_bfloat162 gates[kMaxPairsPerLane];
    float sum = 0.0f;

#pragma unroll
    for (int k = 0; k < kMaxPairsPerLane; ++k) {
        const int pair = lane + k * kWarpSize;
        if (pair < pairs) {
            values[k] = x[row_base + pair];
            if constexpr (Prefetch) {
                weights[k] = weight[pair];
                if constexpr (kRmsGated<Epilogue>) { gates[k] = z[row_base + pair]; }
            }
            const float2 xf = __bfloat1622float2(values[k]);
            sum += xf.x * xf.x + xf.y * xf.y;
        }
    }

    sum       = warp_reduce_sum(sum);
    float inv = lane == 0 ? rsqrtf(sum / static_cast<float>(d) + eps) : 0.0f;
    inv       = __shfl_sync(kFullWarpMask, inv, 0);

#pragma unroll
    for (int k = 0; k < kMaxPairsPerLane; ++k) {
        const int pair = lane + k * kWarpSize;
        if (pair < pairs) {
            const float2 xf = __bfloat1622float2(values[k]);
            __nv_bfloat162 w_pair;
            __nv_bfloat162 z_pair{};
            if constexpr (Prefetch) {
                w_pair = weights[k];
                if constexpr (kRmsGated<Epilogue>) { z_pair = gates[k]; }
            } else {
                w_pair = weight[pair];
                if constexpr (kRmsGated<Epilogue>) { z_pair = z[row_base + pair]; }
            }
            const float2 wf = __bfloat1622float2(w_pair);
            float2 zf{0.0f, 0.0f};
            if constexpr (kRmsGated<Epilogue>) { zf = __bfloat1622float2(z_pair); }
            const __nv_bfloat162 stored =
                __floats2bfloat162_rn(rmsnorm_epilogue<Epilogue>(xf.x, inv, wf.x, zf.x),
                                      rmsnorm_epilogue<Epilogue>(xf.y, inv, wf.y, zf.y));
            out[row_base + pair] = k == 0 ? slot0(stored) : stored;
        }
    }
}

template <RmsEpilogue Epilogue, int Block, bool Prefetch, int FixedD = 0>
__launch_bounds__(Block) __global__
    void rmsnorm_warp_bf16x2_kernel(const __nv_bfloat162* x, const __nv_bfloat162* weight,
                                    const __nv_bfloat162* z, __nv_bfloat162* out,
                                    std::int32_t input_d, std::int64_t rows, float eps) {
    static_assert(Block % kWarpSize == 0);
    constexpr int kWarpsPerBlock = Block / kWarpSize;
    const int warp               = static_cast<int>(threadIdx.x) / kWarpSize;
    const std::int64_t row       = static_cast<std::int64_t>(blockIdx.x) * kWarpsPerBlock + warp;
    if (row >= rows) { return; }
    rmsnorm_warp_row<Epilogue, Prefetch, FixedD>(x, weight, z, out, input_d, row, eps);
}

// Two unit-offset D = 256 norms in one launch (rmsnorm_pair): rows [0, rows1) of the first, then
// the second's; each row is rmsnorm_warp_bf16x2_kernel's.
template <int Block>
__launch_bounds__(Block) __global__
    void rmsnorm_pair_offset256_kernel(const __nv_bfloat162* x1, const __nv_bfloat162* w1, __nv_bfloat162* o1,
                                       std::int64_t rows1, const __nv_bfloat162* x2, const __nv_bfloat162* w2,
                                       __nv_bfloat162* o2, std::int64_t rows2, float eps) {
    constexpr int kWarpsPerBlock = Block / kWarpSize;
    const int warp               = static_cast<int>(threadIdx.x) / kWarpSize;
    const std::int64_t row       = static_cast<std::int64_t>(blockIdx.x) * kWarpsPerBlock + warp;
    if (row < rows1) {
        rmsnorm_warp_row<RmsEpilogue::Offset, true, 256>(x1, w1, nullptr, o1, 256, row, eps);
    } else if (row - rows1 < rows2) {
        rmsnorm_warp_row<RmsEpilogue::Offset, true, 256>(x2, w2, nullptr, o2, 256, row - rows1, eps);
    }
}

// rope_generic_kernel's rotation of features [0, 64) (rotary_dim 64) applied to the normalized,
// BF16-rounded values: lane l holds features 2l and 2l + 1 of slot 0, so pair p's two features sit
// 16 lanes apart (lanes below 16 hold the first of each pair). c/s: pairs 2 (l % 16) and + 1.
struct RmsSlot0Rope64 {
    float c0, c1, s0, s1;
    int lane;
    __device__ __forceinline__ __nv_bfloat162 operator()(__nv_bfloat162 value) const {
        const float2 f  = __bfloat1622float2(value);
        const float px  = __shfl_xor_sync(kFullWarpMask, f.x, 16);
        const float py  = __shfl_xor_sync(kFullWarpMask, f.y, 16);
        const bool head = lane < 16;
        const float a   = head ? rope_first(f.x, px, c0, s0) : rope_second(px, f.x, c0, s0);
        const float b   = head ? rope_first(f.y, py, c1, s1) : rope_second(py, f.y, c1, s1);
        return __halves2bfloat162(__float2bfloat16_rn(a), __float2bfloat16_rn(b));
    }
};

// rmsnorm_pair_offset256_kernel followed by rope_generic_kernel over q (rows1 = heads1 x T) and k
// (rows2 = heads2 x T), rotary_dim 64: each row's sin/cos as that kernel forms them.
template <int Block>
__launch_bounds__(Block) __global__
    void rmsnorm_pair_rope256_kernel(const __nv_bfloat162* x1, const __nv_bfloat162* w1, __nv_bfloat162* o1,
                                     std::int64_t rows1, int heads1, const __nv_bfloat162* x2,
                                     const __nv_bfloat162* w2, __nv_bfloat162* o2, std::int64_t rows2, int heads2,
                                     float eps, const std::int32_t* positions, int axes, float theta, int tokens) {
    constexpr int kWarpsPerBlock = Block / kWarpSize;
    const int warp               = static_cast<int>(threadIdx.x) / kWarpSize;
    const int lane               = static_cast<int>(threadIdx.x) & (kWarpSize - 1);
    const std::int64_t row       = static_cast<std::int64_t>(blockIdx.x) * kWarpsPerBlock + warp;
    const bool first             = row < rows1;
    const std::int64_t local     = first ? row : row - rows1;
    if (!first && local >= rows2) { return; }
    const int token = static_cast<int>(local / (first ? heads1 : heads2));
    const int pair  = 2 * (lane & 15);
    RmsSlot0Rope64 rope{0.0F, 0.0F, 0.0F, 0.0F, lane};
    generic_sincos(positions, axes, 256, 64, theta, tokens, token, pair, &rope.s0, &rope.c0);
    generic_sincos(positions, axes, 256, 64, theta, tokens, token, pair + 1, &rope.s1, &rope.c1);
    if (first) {
        rmsnorm_warp_row<RmsEpilogue::Offset, true, 256>(x1, w1, nullptr, o1, 256, local, eps, rope);
    } else {
        rmsnorm_warp_row<RmsEpilogue::Offset, true, 256>(x2, w2, nullptr, o2, 256, local, eps, rope);
    }
}

// Implements: include/ninfer/ops/rmsnorm.h
// Match: aligned contiguous BF16, plain epilogue, D=128, sm_120a.
// Algorithm assumptions: exactly two BF16x2 values per lane; one warp owns one logical row.
template <RmsEpilogue Epilogue, int Block>
__launch_bounds__(Block) __global__
    void rmsnorm_d128_bf16x2_kernel(const __nv_bfloat162* x, const __nv_bfloat162* weight,
                                    const __nv_bfloat162* z, __nv_bfloat162* out, std::int64_t rows,
                                    float eps) {
    static_assert(Block % kWarpSize == 0);
    constexpr int kWarpsPerBlock = Block / kWarpSize;
    constexpr int kPairsPerRow   = 64;
    const int lane               = static_cast<int>(threadIdx.x) & (kWarpSize - 1);
    const int warp               = static_cast<int>(threadIdx.x) / kWarpSize;
    const std::int64_t row       = static_cast<std::int64_t>(blockIdx.x) * kWarpsPerBlock + warp;
    if (row >= rows) { return; }

    const std::int64_t row_base = row * kPairsPerRow;
    const int pair0             = lane;
    const int pair1             = lane + kWarpSize;
    const __nv_bfloat162 value0 = x[row_base + pair0];
    const __nv_bfloat162 value1 = x[row_base + pair1];
    const float2 x0             = __bfloat1622float2(value0);
    const float2 x1             = __bfloat1622float2(value1);
    float sum                   = x0.x * x0.x + x0.y * x0.y + x1.x * x1.x + x1.y * x1.y;
    sum                         = warp_reduce_sum(sum);
    float inv                   = lane == 0 ? rsqrtf(sum * (1.0f / 128.0f) + eps) : 0.0f;
    inv                         = __shfl_sync(kFullWarpMask, inv, 0);

    const float2 w0 = __bfloat1622float2(weight[pair0]);
    const float2 w1 = __bfloat1622float2(weight[pair1]);
    float2 z0{0.0f, 0.0f};
    float2 z1{0.0f, 0.0f};
    if constexpr (kRmsGated<Epilogue>) {
        z0 = __bfloat1622float2(z[row_base + pair0]);
        z1 = __bfloat1622float2(z[row_base + pair1]);
    }
    out[row_base + pair0] =
        __floats2bfloat162_rn(rmsnorm_epilogue<Epilogue>(x0.x, inv, w0.x, z0.x),
                              rmsnorm_epilogue<Epilogue>(x0.y, inv, w0.y, z0.y));
    out[row_base + pair1] =
        __floats2bfloat162_rn(rmsnorm_epilogue<Epilogue>(x1.x, inv, w1.x, z1.x),
                              rmsnorm_epilogue<Epilogue>(x1.y, inv, w1.y, z1.y));
}

// Fast geometry for wide rows. One CTA owns one row and keeps up to MaxPairsPerThread BF16x2
// values per lane. The launcher admits only widths evenly divisible by the CTA vector span.
template <RmsEpilogue Epilogue, int Block, int MaxPairsPerThread, bool Prefetch, int FixedD = 0>
__launch_bounds__(Block) __global__
    void rmsnorm_cta_bf16x2_kernel(const __nv_bfloat162* x, const __nv_bfloat162* weight,
                                   const __nv_bfloat162* z, __nv_bfloat162* out,
                                   std::int32_t input_d, std::int64_t rows, float eps) {
    const int d = FixedD ? FixedD : input_d;
    static_assert(Block % kWarpSize == 0);
    const std::int64_t row = static_cast<std::int64_t>(blockIdx.x);
    if (row >= rows) { return; }

    const int pairs             = d / 2;
    const int pairs_per_thread  = pairs / Block;
    const std::int64_t row_base = row * static_cast<std::int64_t>(pairs);
    __nv_bfloat162 values[MaxPairsPerThread];
    // Neither the weight nor the gate depends on the sum of squares, so under Prefetch both
    // are issued in the same pass as x and the two trips to memory overlap instead of running
    // back to back across the reduction. They cost registers, so the launcher turns this off
    // once the grid is wide enough that occupancy is worth more than the overlap.
    __nv_bfloat162 weights[MaxPairsPerThread];
    __nv_bfloat162 gates[MaxPairsPerThread];
    float sum = 0.0f;

#pragma unroll
    for (int k = 0; k < MaxPairsPerThread; ++k) {
        if (k < pairs_per_thread) {
            const int pair = static_cast<int>(threadIdx.x) + k * Block;
            values[k]      = x[row_base + pair];
            if constexpr (Prefetch) {
                weights[k] = weight[pair];
                if constexpr (kRmsGated<Epilogue>) { gates[k] = z[row_base + pair]; }
            }
            const float2 xf = __bfloat1622float2(values[k]);
            sum += xf.x * xf.x + xf.y * xf.y;
        }
    }

    __shared__ float warp_sums[Block / kWarpSize];
    __shared__ float inv_shared;
    const float block_sum = block_reduce_sum<Block>(sum, warp_sums);
    if (threadIdx.x == 0) { inv_shared = rsqrtf(block_sum / static_cast<float>(d) + eps); }
    __syncthreads();
    const float inv = inv_shared;

#pragma unroll
    for (int k = 0; k < MaxPairsPerThread; ++k) {
        if (k < pairs_per_thread) {
            const int pair  = static_cast<int>(threadIdx.x) + k * Block;
            const float2 xf = __bfloat1622float2(values[k]);
            __nv_bfloat162 w_pair;
            __nv_bfloat162 z_pair{};
            if constexpr (Prefetch) {
                w_pair = weights[k];
                if constexpr (kRmsGated<Epilogue>) { z_pair = gates[k]; }
            } else {
                w_pair = weight[pair];
                if constexpr (kRmsGated<Epilogue>) { z_pair = z[row_base + pair]; }
            }
            const float2 wf = __bfloat1622float2(w_pair);
            float2 zf{0.0f, 0.0f};
            if constexpr (kRmsGated<Epilogue>) { zf = __bfloat1622float2(z_pair); }
            out[row_base + pair] =
                __floats2bfloat162_rn(rmsnorm_epilogue<Epilogue>(xf.x, inv, wf.x, zf.x),
                                      rmsnorm_epilogue<Epilogue>(xf.y, inv, wf.y, zf.y));
        }
    }
}

// Implements: include/ninfer/ops/rmsnorm.h
// Match: aligned contiguous BF16, plain epilogue, D=2048, sm_120a.
// Algorithm assumptions: exactly two BF16x2 values per thread; one 512-thread CTA owns one row.
template <RmsEpilogue Epilogue>
__launch_bounds__(512) __global__
    void rmsnorm_d2048_bf16x2_kernel(const __nv_bfloat162* x, const __nv_bfloat162* weight,
                                     const __nv_bfloat162* z, __nv_bfloat162* out,
                                     std::int64_t rows, float eps) {
    constexpr int kBlock       = 512;
    constexpr int kPairsPerRow = 1024;
    const std::int64_t row     = static_cast<std::int64_t>(blockIdx.x);
    if (row >= rows) { return; }

    const std::int64_t row_base = row * kPairsPerRow;
    const int pair0             = static_cast<int>(threadIdx.x);
    const int pair1             = pair0 + kBlock;
    const __nv_bfloat162 value0 = x[row_base + pair0];
    const __nv_bfloat162 value1 = x[row_base + pair1];
    const float2 x0             = __bfloat1622float2(value0);
    const float2 x1             = __bfloat1622float2(value1);
    const float local_sum       = x0.x * x0.x + x0.y * x0.y + x1.x * x1.x + x1.y * x1.y;

    __shared__ float warp_sums[kBlock / kWarpSize];
    __shared__ float inv_shared;
    const float sum = block_reduce_sum<kBlock>(local_sum, warp_sums);
    if (threadIdx.x == 0) { inv_shared = rsqrtf(sum * (1.0f / 2048.0f) + eps); }
    __syncthreads();
    const float inv = inv_shared;

    const float2 w0 = __bfloat1622float2(weight[pair0]);
    const float2 w1 = __bfloat1622float2(weight[pair1]);
    float2 z0{0.0f, 0.0f};
    float2 z1{0.0f, 0.0f};
    if constexpr (kRmsGated<Epilogue>) {
        z0 = __bfloat1622float2(z[row_base + pair0]);
        z1 = __bfloat1622float2(z[row_base + pair1]);
    }
    out[row_base + pair0] =
        __floats2bfloat162_rn(rmsnorm_epilogue<Epilogue>(x0.x, inv, w0.x, z0.x),
                              rmsnorm_epilogue<Epilogue>(x0.y, inv, w0.y, z0.y));
    out[row_base + pair1] =
        __floats2bfloat162_rn(rmsnorm_epilogue<Epilogue>(x1.x, inv, w1.x, z1.x),
                              rmsnorm_epilogue<Epilogue>(x1.y, inv, w1.y, z1.y));
}

// Functional fallback outside the aligned fast domains. It intentionally favors a simple complete
// implementation over another family of shape-specific paths.
template <RmsEpilogue Epilogue>
__launch_bounds__(256) __global__
    void rmsnorm_generic_kernel(const __nv_bfloat16* x, const __nv_bfloat16* weight,
                                const __nv_bfloat16* z, __nv_bfloat16* out, std::int32_t d,
                                std::int64_t rows, float eps) {
    const std::int64_t row = static_cast<std::int64_t>(blockIdx.x);
    if (row >= rows) { return; }

    const std::int64_t base = row * static_cast<std::int64_t>(d);
    float sum               = 0.0f;
    for (std::int64_t i = threadIdx.x; i < static_cast<std::int64_t>(d); i += blockDim.x) {
        const float xv = __bfloat162float(x[base + i]);
        sum += xv * xv;
    }

    __shared__ float scratch[256];
    scratch[threadIdx.x] = sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) { scratch[threadIdx.x] += scratch[threadIdx.x + stride]; }
        __syncthreads();
    }

    const float inv = rsqrtf(scratch[0] / static_cast<float>(d) + eps);
    for (std::int64_t i = threadIdx.x; i < static_cast<std::int64_t>(d); i += blockDim.x) {
        const std::int64_t index = base + i;
        const float xv           = __bfloat162float(x[index]);
        const float wv           = __bfloat162float(weight[i]);
        float zv                 = 0.0f;
        if constexpr (kRmsGated<Epilogue>) { zv = __bfloat162float(z[index]); }
        out[index] = __float2bfloat16_rn(rmsnorm_epilogue<Epilogue>(xv, inv, wv, zv));
    }
}

} // namespace ninfer::ops
