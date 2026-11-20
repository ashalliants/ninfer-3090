#pragma once

// Implements: include/ninfer/ops/lora.h
// Two stages. Shrink: a warp per RowsPerWarp rank rows, a block per (Cols-column tile, 8 warps, K
// split of kLoraSplitK values); its lanes split the K chunk and each split writes its FP32 partial
// dot products. Expand: a block per (256 * Rows output rows, Cols-column tile, output); it sums the
// partials in split order and adds B * u to the BF16 output. Every (row, column) value takes the
// same operations in the same order whatever the tile, the rows per warp or thread, or the call's
// other columns, and the split length is a constant: a column's arithmetic never depends on the
// call's width or neighbours, or on the device.
//
// The launcher sizes the tiles to the call. Decode and verify widths are latency-bound (a few
// columns, rank 32-96): narrow tiles and 512-wide K splits keep enough blocks in flight. Prefill
// widths are issue-bound: two rank rows per warp share each x load, four output rows per thread
// share each u read. Measured on a V100 against the first version (2048-wide splits, 8- and 16-column
// tiles), per call, ffn gate/up site (K 5120, rank 64, N 2 x 17408):
//
//   columns        4 (MTP verify)          2048 (prefill chunk)
//   shrink      14.0 ->  5.5 us          598 ->  394 us
//   expand      12.5 ->  9.0 us         2165 ->  983 us

#include <cuda_bf16.h>

#include <cstdint>

namespace ninfer::ops {

inline constexpr int kLoraThreads         = 256;
inline constexpr int kLoraShrinkColumns   = 8;  // the widest shrink tile
inline constexpr int kLoraShrinkRows      = kLoraThreads / 32;
inline constexpr int kLoraSplitK          = 512;
inline constexpr int kLoraExpandColumns   = 8;  // the widest expand tile
inline constexpr int kLoraExpandRows      = 4;  // output rows per thread at prefill widths
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

// Sums each of a lane's Count values over the warp, leaving value `lane >> (5 - log2(Count))` with
// that lane (every lane, once Count < 32, as the plain butterfly does). Each value takes exactly the
// xor butterfly's adds, own value first, so the total is bit-for-bit the butterfly's whatever Count
// is: halving the values a lane keeps at each step only drops the copies it would never read.
template <int Count>
__device__ __forceinline__ float lora_warp_reduce_scatter(float (&v)[Count], int lane) {
    static_assert(Count >= 1 && Count <= 32 && (Count & (Count - 1)) == 0);
    int half = Count / 2;
    int offset = 16;
#pragma unroll
    for (; half >= 1; half /= 2, offset /= 2) {
        const bool upper = (lane & offset) != 0;
#pragma unroll
        for (int i = 0; i < half; ++i) {
            const float send = upper ? v[i] : v[i + half];
            const float keep = upper ? v[i + half] : v[i];
            v[i]             = keep + __shfl_xor_sync(0xffffffffU, send, offset);
        }
    }
    float total = v[0];
#pragma unroll
    for (; offset >= 1; offset /= 2) { total += __shfl_xor_sync(0xffffffffU, total, offset); }
    return total;
}

template <int Cols, int RowsPerWarp>
__global__ __launch_bounds__(kLoraThreads) void lora_shrink_kernel(const LoraShrinkParams p) {
    const int lane   = static_cast<int>(threadIdx.x & 31U);
    const int row0   = (static_cast<int>(blockIdx.y) * kLoraShrinkRows + static_cast<int>(threadIdx.x >> 5U)) *
                     RowsPerWarp;
    const int col0   = static_cast<int>(blockIdx.x) * Cols;
    const int split  = static_cast<int>(blockIdx.z);
    const int k0     = split * kLoraSplitK;
    const int k1     = min(p.k, k0 + kLoraSplitK);

    const __nv_bfloat16* a_base[Cols];
#pragma unroll
    for (int c = 0; c < Cols; ++c) {
        const int col = col0 + c;
        a_base[c]     = nullptr;
        if (col < p.columns) {
            const int s = p.slot[col / p.width];
            if (s >= 0 && p.a[s] != nullptr) { a_base[c] = p.a[s] + static_cast<std::int64_t>(row0) * p.k; }
        }
    }

    float acc[RowsPerWarp * Cols];
#pragma unroll
    for (int i = 0; i < RowsPerWarp * Cols; ++i) { acc[i] = 0.0f; }

    for (int k = k0 + lane * 8; k < k1; k += 32 * 8) {
        const __nv_bfloat16* loaded = nullptr;
        float av[RowsPerWarp][8];
#pragma unroll
        for (int c = 0; c < Cols; ++c) {
            if (a_base[c] == nullptr) { continue; }
            if (a_base[c] != loaded) {
#pragma unroll
                for (int r = 0; r < RowsPerWarp; ++r) {
                    lora_unpack8(*reinterpret_cast<const uint4*>(a_base[c] + static_cast<std::int64_t>(r) * p.k + k),
                                 av[r]);
                }
                loaded = a_base[c];
            }
            float xv[8];
            lora_unpack8(*reinterpret_cast<const uint4*>(p.x + static_cast<std::int64_t>(col0 + c) * p.x_ld + k),
                         xv);
#pragma unroll
            for (int r = 0; r < RowsPerWarp; ++r) {
#pragma unroll
                for (int q = 0; q < 8; ++q) { acc[r * Cols + c] = fmaf(av[r][q], xv[q], acc[r * Cols + c]); }
            }
        }
    }

    constexpr int kCount = RowsPerWarp * Cols;
    constexpr int kShift = kCount >= 32 ? 0 : (kCount >= 16 ? 1 : kCount >= 8 ? 2 : kCount >= 4 ? 3 : kCount >= 2 ? 4 : 5);
    const float v = lora_warp_reduce_scatter<kCount>(acc, lane);
    if ((lane & ((1 << kShift) - 1)) == 0) {
        const int index = lane >> kShift;
        const int r     = index / Cols;
        const int col   = col0 + index % Cols;
        if (col < p.columns) {
            p.partial[(static_cast<std::int64_t>(split) * p.rank + row0 + r) * p.columns + col] = v;
        }
    }
}

template <int Cols>
__global__ __launch_bounds__(kLoraThreads) void lora_expand_kernel(const LoraExpandParams p) {
    extern __shared__ float u[]; // [rank_count, Cols]
    const LoraExpandOutput& o = p.outputs[blockIdx.z];
    const int col0            = static_cast<int>(blockIdx.y) * Cols;

    for (int index = static_cast<int>(threadIdx.x); index < o.rank_count * Cols;
         index += kLoraThreads) {
        const int i   = index / Cols;
        const int c   = index % Cols;
        const int col = col0 + c;
        float v       = 0.0f;
        if (col < p.columns) {
#pragma unroll 8
            for (int s = 0; s < p.splits; ++s) {
                v += p.partial[(static_cast<std::int64_t>(s) * p.rank + o.rank_offset + i) * p.columns + col];
            }
        }
        u[index] = v;
    }
    __syncthreads();

    const int n = static_cast<int>(blockIdx.x) * kLoraThreads + static_cast<int>(threadIdx.x);
    if (n >= o.n) { return; }

    const __nv_bfloat16* b_row[Cols];
#pragma unroll
    for (int c = 0; c < Cols; ++c) {
        const int col = col0 + c;
        b_row[c]      = nullptr;
        if (col < p.columns) {
            const int s = p.slot[col / p.width];
            if (s >= 0 && o.b[s] != nullptr) {
                b_row[c] = o.b[s] + static_cast<std::int64_t>(n) * o.rank_count;
            }
        }
    }

    // The old outputs are read up front: the stores at the end may alias them, so reading each
    // after the previous column's store would serialise their latencies behind the B stream.
    float old[Cols];
    float acc[Cols];
#pragma unroll
    for (int c = 0; c < Cols; ++c) {
        acc[c] = 0.0f;
        old[c] = b_row[c] == nullptr
                     ? 0.0f
                     : __bfloat162float(o.out[static_cast<std::int64_t>(col0 + c) * o.ld + n]);
    }
#pragma unroll 4
    for (int i0 = 0; i0 < o.rank_count; i0 += 8) {
        const __nv_bfloat16* loaded = nullptr;
        float bv[8];
#pragma unroll
        for (int c = 0; c < Cols; ++c) {
            if (b_row[c] == nullptr) { continue; }
            if (b_row[c] != loaded) {
                lora_unpack8(*reinterpret_cast<const uint4*>(b_row[c] + i0), bv);
                loaded = b_row[c];
            }
#pragma unroll
            for (int q = 0; q < 8; ++q) {
                acc[c] = fmaf(bv[q], u[(i0 + q) * Cols + c], acc[c]);
            }
        }
    }
#pragma unroll
    for (int c = 0; c < Cols; ++c) {
        if (b_row[c] == nullptr) { continue; }
        o.out[static_cast<std::int64_t>(col0 + c) * o.ld + n] = __float2bfloat16_rn(old[c] + acc[c]);
    }
}

// Wide calls (prefill): each thread takes Rows output rows, n and n + 256, ..., so every u value
// read from shared memory feeds Rows FMAs. A row's arithmetic is lora_expand_kernel's exactly:
// the same FMAs in the same order into its own accumulator, then one add to the old value.
template <int Cols, int Rows>
__global__ __launch_bounds__(kLoraThreads) void lora_expand_rows_kernel(const LoraExpandParams p) {
    extern __shared__ float u[]; // [rank_count, Cols]
    const LoraExpandOutput& o = p.outputs[blockIdx.z];
    const int col0            = static_cast<int>(blockIdx.y) * Cols;

    for (int index = static_cast<int>(threadIdx.x); index < o.rank_count * Cols;
         index += kLoraThreads) {
        const int i   = index / Cols;
        const int c   = index % Cols;
        const int col = col0 + c;
        float v       = 0.0f;
        if (col < p.columns) {
#pragma unroll 8
            for (int s = 0; s < p.splits; ++s) {
                v += p.partial[(static_cast<std::int64_t>(s) * p.rank + o.rank_offset + i) * p.columns + col];
            }
        }
        u[index] = v;
    }
    __syncthreads();

    const int n0 = static_cast<int>(blockIdx.x) * kLoraThreads * Rows + static_cast<int>(threadIdx.x);
    if (n0 >= o.n) { return; }

    const __nv_bfloat16* b_base[Cols];
#pragma unroll
    for (int c = 0; c < Cols; ++c) {
        const int col = col0 + c;
        b_base[c]     = nullptr;
        if (col < p.columns) {
            const int s = p.slot[col / p.width];
            if (s >= 0 && o.b[s] != nullptr) { b_base[c] = o.b[s]; }
        }
    }
    bool live[Rows];
#pragma unroll
    for (int r = 0; r < Rows; ++r) { live[r] = n0 + r * kLoraThreads < o.n; }

    float acc[Rows][Cols];
    __nv_bfloat16 old[Rows][Cols];
#pragma unroll
    for (int r = 0; r < Rows; ++r) {
#pragma unroll
        for (int c = 0; c < Cols; ++c) {
            acc[r][c] = 0.0f;
            old[r][c] = (live[r] && b_base[c] != nullptr)
                            ? o.out[static_cast<std::int64_t>(col0 + c) * o.ld + n0 + r * kLoraThreads]
                            : __float2bfloat16_rn(0.0f);
        }
    }
    // Every adapted column of the tile on one adapter (a prefill chunk, one sequence's verify
    // columns): one B read per rank chunk, and u read a row at a time as float4s. Each acc[r][c]
    // takes the same FMAs in the same order as on the general path below.
    const __nv_bfloat16* lead = nullptr;
    bool uniform              = true;
#pragma unroll
    for (int c = 0; c < Cols; ++c) {
        if (b_base[c] == nullptr) { continue; }
        if (lead == nullptr) { lead = b_base[c]; }
        uniform = uniform && b_base[c] == lead;
    }
    if (uniform && Cols % 4 == 0) {
        for (int i0 = 0; i0 < o.rank_count; i0 += 8) {
            float bv[Rows][8];
#pragma unroll
            for (int r = 0; r < Rows; ++r) {
                const int n = live[r] ? n0 + r * kLoraThreads : n0;
                lora_unpack8(*reinterpret_cast<const uint4*>(
                                 lead + static_cast<std::int64_t>(n) * o.rank_count + i0),
                             bv[r]);
            }
#pragma unroll
            for (int q = 0; q < 8; ++q) {
#pragma unroll
                for (int c4 = 0; c4 < Cols; c4 += 4) {
                    const float4 uq = *reinterpret_cast<const float4*>(&u[(i0 + q) * Cols + c4]);
#pragma unroll
                    for (int r = 0; r < Rows; ++r) {
                        acc[r][c4 + 0] = fmaf(bv[r][q], uq.x, acc[r][c4 + 0]);
                        acc[r][c4 + 1] = fmaf(bv[r][q], uq.y, acc[r][c4 + 1]);
                        acc[r][c4 + 2] = fmaf(bv[r][q], uq.z, acc[r][c4 + 2]);
                        acc[r][c4 + 3] = fmaf(bv[r][q], uq.w, acc[r][c4 + 3]);
                    }
                }
            }
        }
    } else
    for (int i0 = 0; i0 < o.rank_count; i0 += 8) {
        const __nv_bfloat16* loaded = nullptr;
        float bv[Rows][8];
#pragma unroll
        for (int c = 0; c < Cols; ++c) {
            if (b_base[c] == nullptr) { continue; }
            if (b_base[c] != loaded) {
#pragma unroll
                for (int r = 0; r < Rows; ++r) {
                    const int n = live[r] ? n0 + r * kLoraThreads : n0;
                    lora_unpack8(*reinterpret_cast<const uint4*>(
                                     b_base[c] + static_cast<std::int64_t>(n) * o.rank_count + i0),
                                 bv[r]);
                }
                loaded = b_base[c];
            }
#pragma unroll
            for (int q = 0; q < 8; ++q) {
                const float uq = u[(i0 + q) * Cols + c];
#pragma unroll
                for (int r = 0; r < Rows; ++r) { acc[r][c] = fmaf(bv[r][q], uq, acc[r][c]); }
            }
        }
    }
#pragma unroll
    for (int r = 0; r < Rows; ++r) {
        if (!live[r]) { continue; }
        const int n = n0 + r * kLoraThreads;
#pragma unroll
        for (int c = 0; c < Cols; ++c) {
            if (b_base[c] == nullptr) { continue; }
            o.out[static_cast<std::int64_t>(col0 + c) * o.ld + n] =
                __float2bfloat16_rn(__bfloat162float(old[r][c]) + acc[r][c]);
        }
    }
}

} // namespace ninfer::ops
