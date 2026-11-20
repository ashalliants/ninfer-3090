#pragma once

// Implements: include/ninfer/ops/lora.h
// Two stages. Shrink: one warp per rank row, a block per (8-column tile, 8 rank rows, K split of
// kLoraSplitK values); each split writes its FP32 partial dot products. Expand: one thread per
// output row, a block per (256 rows, 16-column tile, output); it sums the partials in split order
// and adds B * u to the BF16 output. The split length is a constant, so a column's arithmetic never
// depends on the call's other columns or on the device.

#include <cuda_bf16.h>

#include <cstdint>

namespace ninfer::ops {

inline constexpr int kLoraThreads         = 256;
inline constexpr int kLoraShrinkColumns   = 8;
inline constexpr int kLoraShrinkRows      = kLoraThreads / 32;
inline constexpr int kLoraSplitK          = 2048;
inline constexpr int kLoraExpandColumns   = 16;
inline constexpr int kLoraKernelAdapters  = 8;
inline constexpr int kLoraKernelRows      = 64;
inline constexpr int kLoraKernelOutputs   = 4;

struct LoraShrinkParams {
    const __nv_bfloat16* x;
    std::int64_t x_ld; // elements between columns
    const __nv_bfloat16* a[kLoraKernelAdapters];
    std::int8_t slot[kLoraKernelRows];
    std::int32_t rank;
    std::int32_t k;
    std::int32_t columns;
    std::int32_t width;
    float* partial; // [splits, rank, columns]
};

struct LoraExpandOutput {
    __nv_bfloat16* out;
    std::int64_t ld; // elements between columns
    const __nv_bfloat16* b[kLoraKernelAdapters];
    std::int32_t n;
    std::int32_t rank_offset;
    std::int32_t rank_count;
};

struct LoraExpandParams {
    LoraExpandOutput outputs[kLoraKernelOutputs];
    std::int8_t slot[kLoraKernelRows];
    std::int32_t rank;
    std::int32_t columns;
    std::int32_t width;
    std::int32_t splits;
    const float* partial;
};

__device__ __forceinline__ void lora_unpack8(const uint4 v, float out[8]) {
    const std::uint32_t w[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        out[2 * i]     = __uint_as_float(w[i] << 16);
        out[2 * i + 1] = __uint_as_float(w[i] & 0xffff0000u);
    }
}

__global__ __launch_bounds__(kLoraThreads) void lora_shrink_kernel(const LoraShrinkParams p) {
    const int lane   = static_cast<int>(threadIdx.x & 31U);
    const int row    = static_cast<int>(blockIdx.y) * kLoraShrinkRows + static_cast<int>(threadIdx.x >> 5U);
    const int col0   = static_cast<int>(blockIdx.x) * kLoraShrinkColumns;
    const int split  = static_cast<int>(blockIdx.z);
    const int k0     = split * kLoraSplitK;
    const int k1     = min(p.k, k0 + kLoraSplitK);

    const __nv_bfloat16* a_row[kLoraShrinkColumns];
#pragma unroll
    for (int c = 0; c < kLoraShrinkColumns; ++c) {
        const int col = col0 + c;
        a_row[c]      = nullptr;
        if (col < p.columns) {
            const int s = p.slot[col / p.width];
            if (s >= 0 && p.a[s] != nullptr) {
                a_row[c] = p.a[s] + static_cast<std::int64_t>(row) * p.k;
            }
        }
    }

    float acc[kLoraShrinkColumns];
#pragma unroll
    for (int c = 0; c < kLoraShrinkColumns; ++c) { acc[c] = 0.0f; }

    for (int k = k0 + lane * 8; k < k1; k += 32 * 8) {
        const __nv_bfloat16* loaded = nullptr;
        float av[8];
#pragma unroll
        for (int c = 0; c < kLoraShrinkColumns; ++c) {
            if (a_row[c] == nullptr) { continue; }
            if (a_row[c] != loaded) {
                lora_unpack8(*reinterpret_cast<const uint4*>(a_row[c] + k), av);
                loaded = a_row[c];
            }
            float xv[8];
            lora_unpack8(*reinterpret_cast<const uint4*>(p.x + static_cast<std::int64_t>(col0 + c) * p.x_ld + k),
                         xv);
#pragma unroll
            for (int q = 0; q < 8; ++q) { acc[c] = fmaf(av[q], xv[q], acc[c]); }
        }
    }

#pragma unroll
    for (int c = 0; c < kLoraShrinkColumns; ++c) {
        float v = acc[c];
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) { v += __shfl_xor_sync(0xffffffffU, v, offset); }
        const int col = col0 + c;
        if (lane == 0 && col < p.columns) {
            p.partial[(static_cast<std::int64_t>(split) * p.rank + row) * p.columns + col] = v;
        }
    }
}

__global__ __launch_bounds__(kLoraThreads) void lora_expand_kernel(const LoraExpandParams p) {
    extern __shared__ float u[]; // [rank_count, kLoraExpandColumns]
    const LoraExpandOutput& o = p.outputs[blockIdx.z];
    const int col0            = static_cast<int>(blockIdx.y) * kLoraExpandColumns;

    for (int index = static_cast<int>(threadIdx.x); index < o.rank_count * kLoraExpandColumns;
         index += kLoraThreads) {
        const int i   = index / kLoraExpandColumns;
        const int c   = index % kLoraExpandColumns;
        const int col = col0 + c;
        float v       = 0.0f;
        if (col < p.columns) {
            for (int s = 0; s < p.splits; ++s) {
                v += p.partial[(static_cast<std::int64_t>(s) * p.rank + o.rank_offset + i) * p.columns + col];
            }
        }
        u[index] = v;
    }
    __syncthreads();

    const int n = static_cast<int>(blockIdx.x) * kLoraThreads + static_cast<int>(threadIdx.x);
    if (n >= o.n) { return; }

    const __nv_bfloat16* b_row[kLoraExpandColumns];
#pragma unroll
    for (int c = 0; c < kLoraExpandColumns; ++c) {
        const int col = col0 + c;
        b_row[c]      = nullptr;
        if (col < p.columns) {
            const int s = p.slot[col / p.width];
            if (s >= 0 && o.b[s] != nullptr) {
                b_row[c] = o.b[s] + static_cast<std::int64_t>(n) * o.rank_count;
            }
        }
    }

    float acc[kLoraExpandColumns];
#pragma unroll
    for (int c = 0; c < kLoraExpandColumns; ++c) { acc[c] = 0.0f; }
    for (int i0 = 0; i0 < o.rank_count; i0 += 8) {
        const __nv_bfloat16* loaded = nullptr;
        float bv[8];
#pragma unroll
        for (int c = 0; c < kLoraExpandColumns; ++c) {
            if (b_row[c] == nullptr) { continue; }
            if (b_row[c] != loaded) {
                lora_unpack8(*reinterpret_cast<const uint4*>(b_row[c] + i0), bv);
                loaded = b_row[c];
            }
#pragma unroll
            for (int q = 0; q < 8; ++q) {
                acc[c] = fmaf(bv[q], u[(i0 + q) * kLoraExpandColumns + c], acc[c]);
            }
        }
    }
#pragma unroll
    for (int c = 0; c < kLoraExpandColumns; ++c) {
        if (b_row[c] == nullptr) { continue; }
        __nv_bfloat16* y = o.out + static_cast<std::int64_t>(col0 + c) * o.ld + n;
        *y               = __float2bfloat16_rn(__bfloat162float(*y) + acc[c]);
    }
}

} // namespace ninfer::ops
