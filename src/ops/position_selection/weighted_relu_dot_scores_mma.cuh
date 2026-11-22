#pragma once

// The tensor-core route of weighted_relu_dot_scores (contract: include/ninfer/ops/
// position_selection.h), for chunks of query columns. A block serves eight columns, one warp each,
// over a range of positions. Every position's key is the same for every column, so a tile of 32 keys
// is staged once in shared memory (cp.async, XOR-swizzled 16-byte chunks) and read by the eight warps.
//
// A warp multiplies its column's query heads (the M dimension, held in registers as A fragments for
// the whole range) by the staged keys (the N dimension, 8 positions per MMA, K = the head dimension).
// The accumulator holds head dots with heads as rows, so the epilogue is a relu, a multiply by the
// row's weight and a sum over rows: in the thread over the head tiles, then over the 8 row groups of
// the warp by shuffles. A tile that no column of the block can reach is not loaded.

#include "ops/common/memory.cuh"
#include "ops/common/mma.cuh"

#include <cuda_bf16.h>

#include <cmath>
#include <cstddef>
#include <cstdint>

namespace ninfer::ops::scores_mma {

inline constexpr int kWarps   = 8; // columns per block
inline constexpr int kThreads = kWarps * 32;
inline constexpr int kKeys    = 32; // positions per tile
inline constexpr int kStages  = 3;

// Heads H = 16 * MT and head dimension D: the A fragments take MT * D / 4 registers per thread.
template <int MT, int D>
struct Shape {
    static_assert((D == 64 || D == 128 || D == 256) && MT >= 1 && MT <= 4 && MT * D <= 256);
    static constexpr int chunks      = D / 8;
    static constexpr int loads       = kKeys * chunks / kThreads;
    static constexpr int row_stride  = kThreads / chunks;
    static constexpr int stage_bytes = kKeys * D * 2;
    static constexpr int bytes       = kStages * stage_bytes;
};

template <int MT, int D>
__global__ void __launch_bounds__(kThreads, 2)
scores_mma_kernel(const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ k,
                  const __nv_bfloat16* __restrict__ weights, const std::int32_t* __restrict__ valid, int columns,
                  int n, int tiles_per_block, float weight_scale, float* __restrict__ scores) {
    using S = Shape<MT, D>;
    constexpr int kH = MT * 16;
    extern __shared__ __align__(16) unsigned char smem[];

    const int tid  = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int g    = lane >> 2;
    const int tq   = lane & 3;
    const int s    = blockIdx.x * kWarps + warp; // this warp's column
    const bool live = s < columns;

    // The block's reach: the largest number of reachable positions of its columns.
    int reach = 0;
    for (int w = 0; w < kWarps; ++w) {
        const int c = blockIdx.x * kWarps + w;
        if (c < columns) { reach = max(reach, min(max(valid[c], 0), n)); }
    }
    const int lim = live ? min(max(valid[s], 0), n) : 0;

    const int tile_begin = blockIdx.y * tiles_per_block;
    const int tile_end   = min(tile_begin + tiles_per_block, (n + kKeys - 1) / kKeys);
    const int loaded_end = min(tile_end, (reach + kKeys - 1) / kKeys); // tiles with a reachable position
    float* out           = scores + static_cast<std::size_t>(s) * n;

    // Queries as A fragments (row = head, column = head-dimension index) and each row's weight.
    unsigned qa[MT][D / 16][4];
    float wt[MT][2];
    if (live) {
#pragma unroll
        for (int m = 0; m < MT; ++m) {
#pragma unroll
            for (int r = 0; r < 2; ++r) {
                const int h = m * 16 + g + 8 * r;
                const __nv_bfloat16* row = q + (static_cast<std::size_t>(s) * kH + h) * D;
#pragma unroll
                for (int ks = 0; ks < D / 16; ++ks) {
                    qa[m][ks][r]     = *reinterpret_cast<const unsigned*>(row + 16 * ks + 2 * tq);
                    qa[m][ks][2 + r] = *reinterpret_cast<const unsigned*>(row + 16 * ks + 8 + 2 * tq);
                }
                wt[m][r] = __bfloat162float(weights[static_cast<std::size_t>(s) * kH + h]);
            }
        }
    }

    const int ld_chunk = tid % S::chunks;
    const int ld_row   = tid / S::chunks;
    auto issue = [&](int tile) {
        unsigned char* stage = smem + ((tile - tile_begin) % kStages) * S::stage_bytes;
#pragma unroll
        for (int i = 0; i < S::loads; ++i) {
            const int key = ld_row + S::row_stride * i;
            const int pos = tile * kKeys + key;
            const bool ok = pos < n;
            const __nv_bfloat16* src = k + static_cast<std::size_t>(ok ? pos : 0) * D + ld_chunk * 8;
            cp_async_zfill_mem<16, Cache::cg>(stage + key * (D * 2) + ((ld_chunk ^ (key & 7)) * 16), src, ok ? 16 : 0);
        }
    };

    if (tile_begin < loaded_end) { issue(tile_begin); }
    cp_commit_mem();
    if (tile_begin + 1 < loaded_end) { issue(tile_begin + 1); }
    cp_commit_mem();

    for (int tile = tile_begin; tile < loaded_end; ++tile) {
        cp_wait_mem<1>();
        __syncthreads();
        if (tile + 2 < loaded_end) { issue(tile + 2); }
        cp_commit_mem();
        if (!live) { continue; }
        const unsigned char* stage = smem + ((tile - tile_begin) % kStages) * S::stage_bytes;

        float total[4][2]; // per 8-position group: the weighted sums of the two columns this thread owns
#pragma unroll
        for (int nt = 0; nt < 4; ++nt) {
            float c[MT][4];
#pragma unroll
            for (int m = 0; m < MT; ++m) {
#pragma unroll
                for (int i = 0; i < 4; ++i) { c[m][i] = 0.0F; }
            }
#pragma unroll
            for (int kk = 0; kk < D / 32; ++kk) {
                const int mi    = lane >> 3;
                const int key   = nt * 8 + (lane & 7);
                const int chunk = 4 * kk + mi;
                unsigned b0, b1, b2, b3;
                ldmatrix_x4(b0, b1, b2, b3, smem_addr(stage + key * (D * 2) + ((chunk ^ (key & 7)) * 16)));
#pragma unroll
                for (int m = 0; m < MT; ++m) {
                    mma_bf16(c[m][0], c[m][1], c[m][2], c[m][3], qa[m][2 * kk][0], qa[m][2 * kk][1], qa[m][2 * kk][2],
                             qa[m][2 * kk][3], b0, b1);
                    mma_bf16(c[m][0], c[m][1], c[m][2], c[m][3], qa[m][2 * kk + 1][0], qa[m][2 * kk + 1][1],
                             qa[m][2 * kk + 1][2], qa[m][2 * kk + 1][3], b2, b3);
                }
            }
#pragma unroll
            for (int col = 0; col < 2; ++col) {
                float sum = 0.0F;
#pragma unroll
                for (int m = 0; m < MT; ++m) {
                    sum = fmaf(wt[m][0], fmaxf(c[m][col], 0.0F), sum);
                    sum = fmaf(wt[m][1], fmaxf(c[m][2 + col], 0.0F), sum);
                }
                sum += __shfl_xor_sync(0xFFFFFFFFU, sum, 4);
                sum += __shfl_xor_sync(0xFFFFFFFFU, sum, 8);
                sum += __shfl_xor_sync(0xFFFFFFFFU, sum, 16);
                total[nt][col] = sum;
            }
        }
        // Lane (g, tq) writes position 8 * (g & 3) + 2 * tq + (g >> 2) of the tile: 32 consecutive floats.
        float mine = 0.0F;
        const int own = g & 3, col = g >> 2;
#pragma unroll
        for (int nt = 0; nt < 4; ++nt) {
#pragma unroll
            for (int cc = 0; cc < 2; ++cc) {
                if (own == nt && col == cc) { mine = total[nt][cc]; }
            }
        }
        const int pos = tile * kKeys + 8 * own + 2 * tq + col;
        if (pos < n) { out[pos] = pos < lim ? mine * weight_scale : -INFINITY; }
    }
    // The tiles past the block's reach hold no reachable position for any of its columns.
    if (live) {
        for (int pos = max(loaded_end, tile_begin) * kKeys + lane; pos < tile_end * kKeys; pos += 32) {
            if (pos < n) { out[pos] = -INFINITY; }
        }
    }
}

} // namespace ninfer::ops::scores_mma
