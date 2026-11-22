#pragma once

// The tensor-core route of gathered_sink_attention (contract: include/ninfer/ops/
// gathered_sink_attention.h), for chunks of columns. One block serves one query column and 32 heads.
// The 32 heads are the M dimension of the MMAs: every head of a column reads the same gathered rows,
// so a tile of 16 listed rows is staged once in shared memory (cp.async, XOR-swizzled 16-byte
// chunks) and feeds both products, and nothing needs the columns to share rows.
//
// Eight warps split the head dimension: warp w keeps its D/8 slice of the 32 queries in registers.
//   S = Q K^T  : each warp multiplies its slice (a partial sum over D/8 of the 512), the partials of
//                the 8 warps are added through shared memory, and the softmax runs on the sum
//                (online: running maximum and denominator per head, in base 2).
//   O += P V   : P (BF16, rounded once; the denominator sums the same rounded values so the weights
//                still add to one) multiplies the same staged tile transposed, and each warp keeps
//                the O columns of its own slice for all 32 heads in registers.
// A tile that holds only skipped rows contributes nothing, so lists padded with -1 cost only their
// copies.

#include "ops/common/mma.cuh"
#include "ops/common/memory.cuh"

#include <cuda_bf16.h>

#include <cmath>
#include <cstddef>
#include <cstdint>

namespace ninfer::ops::gathered_mma {

inline constexpr int kWarps   = 8;
inline constexpr int kThreads = kWarps * 32;
inline constexpr int kHeads   = 32; // heads of one block: two m16 tiles
inline constexpr int kKeys    = 16; // listed rows per tile: the k16 of the PV product
inline constexpr int kStages  = 3;
inline constexpr int kSPad    = 20; // floats per head row of the partial-score buffer
inline constexpr int kPPad    = 24; // BF16 per head row of the probability tile

template <int D>
struct Layout {
    static_assert(D == 256 || D == 512);
    static constexpr int chunks      = D / 8;        // 16-byte chunks of one row
    static constexpr int slice       = D / kWarps;   // head-dimension columns of one warp
    static constexpr int ksteps      = slice / 16;   // k16 steps of the score product
    static constexpr int ntiles      = slice / 8;    // n8 tiles of the output product
    static constexpr int loads       = kKeys * chunks / kThreads; // 16-byte copies per thread and tile
    static constexpr int row_stride  = kThreads / chunks;        // rows between a thread's copies
    static constexpr int stage_bytes = kKeys * D * 2;
    static constexpr int stage_off   = 0;
    static constexpr int part_off    = kStages * stage_bytes;
    static constexpr int part_bytes  = kWarps * kHeads * kSPad * 4;
    static constexpr int prob_off    = part_off + part_bytes;
    static constexpr int prob_bytes  = kHeads * kPPad * 2;
    static constexpr int alpha_off   = prob_off + prob_bytes;
    static constexpr int valid_off   = alpha_off + kHeads * 4;
    static constexpr int bytes       = valid_off + kStages * kKeys * 4;
};

__device__ __forceinline__ float round_bf16(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }

template <int D>
__global__ void __launch_bounds__(kThreads, 1)
attention_mma_kernel(const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ kv,
                     const std::int32_t* __restrict__ idx, const float* __restrict__ sink, int heads, int list,
                     int rows, float scale_log2, __nv_bfloat16* __restrict__ out) {
    using L = Layout<D>;
    extern __shared__ __align__(16) unsigned char smem[];
    auto* stage_base = smem + L::stage_off;
    auto* part       = reinterpret_cast<float*>(smem + L::part_off);
    auto* prob       = reinterpret_cast<__nv_bfloat16*>(smem + L::prob_off);
    auto* alpha_s    = reinterpret_cast<float*>(smem + L::alpha_off);
    auto* valid_s    = reinterpret_cast<int*>(smem + L::valid_off);

    const int tid  = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int g    = lane >> 2; // fragment row group
    const int tq   = lane & 3;
    const int t    = blockIdx.x; // query column
    const int h0   = blockIdx.y * kHeads;
    const std::int32_t* column = idx + static_cast<std::size_t>(t) * list;
    const int tiles            = (list + kKeys - 1) / kKeys;

    // This warp's slice of the queries as A fragments: row = head, column = head-dimension index.
    unsigned qa[L::ksteps][2][4];
#pragma unroll
    for (int m = 0; m < 2; ++m) {
#pragma unroll
        for (int r = 0; r < 2; ++r) {
            const int h = h0 + m * 16 + g + 8 * r;
            const __nv_bfloat16* row = q + (static_cast<std::size_t>(t) * heads + (h < heads ? h : 0)) * D + warp * L::slice;
#pragma unroll
            for (int ks = 0; ks < L::ksteps; ++ks) {
                unsigned lo = 0, hi = 0;
                if (h < heads) {
                    lo = *reinterpret_cast<const unsigned*>(row + 16 * ks + 2 * tq);
                    hi = *reinterpret_cast<const unsigned*>(row + 16 * ks + 8 + 2 * tq);
                }
                qa[ks][m][r]     = lo; // a0 (row g) and a1 (row g + 8)
                qa[ks][m][2 + r] = hi; // a2 (row g, columns + 8) and a3
            }
        }
    }
    float acc[L::ntiles][2][4];
#pragma unroll
    for (int n = 0; n < L::ntiles; ++n) {
#pragma unroll
        for (int m = 0; m < 2; ++m) {
#pragma unroll
            for (int i = 0; i < 4; ++i) { acc[n][m][i] = 0.0F; }
        }
    }

    // Copy of tile `tile`: thread tid moves chunk tid % chunks of rows tid / chunks + row_stride * i.
    int next_rows[L::loads];
    const int ld_chunk = tid % L::chunks;
    const int ld_row   = tid / L::chunks;
    auto fetch_rows = [&](int tile) {
#pragma unroll
        for (int i = 0; i < L::loads; ++i) {
            const int j = tile * kKeys + ld_row + L::row_stride * i;
            next_rows[i] = (tile < tiles && j < list) ? __ldg(column + j) : -1;
        }
    };
    auto issue = [&](int tile) {
        unsigned char* stage = stage_base + (tile % kStages) * L::stage_bytes;
#pragma unroll
        for (int i = 0; i < L::loads; ++i) {
            const int key    = ld_row + L::row_stride * i;
            const int r      = next_rows[i];
            const bool valid = r >= 0 && r < rows;
            const __nv_bfloat16* src = kv + static_cast<std::size_t>(valid ? r : 0) * D + ld_chunk * 8;
            cp_async_zfill_mem<16, Cache::cg>(stage + key * (D * 2) + ((ld_chunk ^ (key & 7)) * 16), src, valid ? 16 : 0);
            if (ld_chunk == 0) { valid_s[(tile % kStages) * kKeys + key] = valid ? 1 : 0; }
        }
    };

    fetch_rows(0);
    if (0 < tiles) { issue(0); }
    cp_commit_mem();
    fetch_rows(1);
    if (1 < tiles) { issue(1); }
    cp_commit_mem();
    fetch_rows(2);

    float m_run = -INFINITY, l_run = 0.0F; // per head of this thread's softmax group (8 threads per head)
    const int sm_head = tid >> 3;
    const int sm_sub  = tid & 7;

    for (int tile = 0; tile < tiles; ++tile) {
        cp_wait_mem<1>();
        __syncthreads();
        if (tile + 2 < tiles) { issue(tile + 2); }
        cp_commit_mem();
        fetch_rows(tile + 3);
        const unsigned char* stage = stage_base + (tile % kStages) * L::stage_bytes;

        // S partial = Q[:, slice] K[tile, slice]^T
        float s[2][2][4];
#pragma unroll
        for (int m = 0; m < 2; ++m) {
#pragma unroll
            for (int n = 0; n < 2; ++n) {
#pragma unroll
                for (int i = 0; i < 4; ++i) { s[m][n][i] = 0.0F; }
            }
        }
#pragma unroll
        for (int ks = 0; ks < L::ksteps; ++ks) {
            const int mi    = lane >> 3;
            const int key   = (mi >> 1) * 8 + (lane & 7);
            const int chunk = (L::slice / 8) * warp + 2 * ks + (mi & 1);
            unsigned b00, b01, b10, b11;
            ldmatrix_x4(b00, b01, b10, b11, smem_addr(stage + key * (D * 2) + ((chunk ^ (key & 7)) * 16)));
#pragma unroll
            for (int m = 0; m < 2; ++m) {
                mma_bf16(s[m][0][0], s[m][0][1], s[m][0][2], s[m][0][3], qa[ks][m][0], qa[ks][m][1], qa[ks][m][2],
                         qa[ks][m][3], b00, b01);
                mma_bf16(s[m][1][0], s[m][1][1], s[m][1][2], s[m][1][3], qa[ks][m][0], qa[ks][m][1], qa[ks][m][2],
                         qa[ks][m][3], b10, b11);
            }
        }
#pragma unroll
        for (int m = 0; m < 2; ++m) {
#pragma unroll
            for (int n = 0; n < 2; ++n) {
                float* base = part + (warp * kHeads + m * 16 + g) * kSPad + n * 8 + 2 * tq;
                *reinterpret_cast<float2*>(base)             = make_float2(s[m][n][0], s[m][n][1]);
                *reinterpret_cast<float2*>(base + 8 * kSPad) = make_float2(s[m][n][2], s[m][n][3]);
            }
        }
        __syncthreads();

        // Softmax of the summed scores: 8 threads per head, two keys each.
        {
            float s0 = 0.0F, s1 = 0.0F;
#pragma unroll
            for (int w = 0; w < kWarps; ++w) {
                const float2 v = *reinterpret_cast<const float2*>(part + (w * kHeads + sm_head) * kSPad + 2 * sm_sub);
                s0 += v.x;
                s1 += v.y;
            }
            const int* ok = valid_s + (tile % kStages) * kKeys + 2 * sm_sub;
            s0 = ok[0] ? s0 * scale_log2 : -INFINITY;
            s1 = ok[1] ? s1 * scale_log2 : -INFINITY;
            float tile_max = fmaxf(s0, s1);
#pragma unroll
            for (int o = 1; o < 8; o <<= 1) { tile_max = fmaxf(tile_max, __shfl_xor_sync(0xFFFFFFFFU, tile_max, o)); }
            const float new_m = fmaxf(m_run, tile_max);
            float alpha = 1.0F, p0 = 0.0F, p1 = 0.0F;
            if (new_m != -INFINITY) {
                alpha = exp2f(m_run - new_m);
                p0    = round_bf16(exp2f(s0 - new_m));
                p1    = round_bf16(exp2f(s1 - new_m));
            }
            float p_sum = p0 + p1;
#pragma unroll
            for (int o = 1; o < 8; o <<= 1) { p_sum += __shfl_xor_sync(0xFFFFFFFFU, p_sum, o); }
            l_run = l_run * alpha + p_sum;
            m_run = new_m;
            *reinterpret_cast<__nv_bfloat162*>(prob + sm_head * kPPad + 2 * sm_sub) = __floats2bfloat162_rn(p0, p1);
            if (sm_sub == 0) { alpha_s[sm_head] = alpha; }
        }
        __syncthreads();

        // O = O * alpha + P V[tile, slice]
#pragma unroll
        for (int m = 0; m < 2; ++m) {
            const float a0 = alpha_s[m * 16 + g], a1 = alpha_s[m * 16 + g + 8];
#pragma unroll
            for (int n = 0; n < L::ntiles; ++n) {
                acc[n][m][0] *= a0;
                acc[n][m][1] *= a0;
                acc[n][m][2] *= a1;
                acc[n][m][3] *= a1;
            }
        }
        unsigned pa[2][4];
        {
            const int mi = lane >> 3;
#pragma unroll
            for (int m = 0; m < 2; ++m) {
                const int row = m * 16 + (mi & 1) * 8 + (lane & 7);
                ldmatrix_x4(pa[m][0], pa[m][1], pa[m][2], pa[m][3],
                            smem_addr(prob + row * kPPad + (mi >> 1) * 8));
            }
        }
#pragma unroll
        for (int pair = 0; pair < L::ntiles / 2; ++pair) {
            const int mi    = lane >> 3;
            const int key   = (mi & 1) * 8 + (lane & 7);
            const int chunk = (L::slice / 8) * warp + 2 * pair + (mi >> 1);
            unsigned b00, b01, b10, b11;
            ldmatrix_x4_t(b00, b01, b10, b11, smem_addr(stage + key * (D * 2) + ((chunk ^ (key & 7)) * 16)));
#pragma unroll
            for (int m = 0; m < 2; ++m) {
                mma_bf16(acc[2 * pair][m][0], acc[2 * pair][m][1], acc[2 * pair][m][2], acc[2 * pair][m][3],
                         pa[m][0], pa[m][1], pa[m][2], pa[m][3], b00, b01);
                mma_bf16(acc[2 * pair + 1][m][0], acc[2 * pair + 1][m][1], acc[2 * pair + 1][m][2],
                         acc[2 * pair + 1][m][3], pa[m][0], pa[m][1], pa[m][2], pa[m][3], b10, b11);
            }
        }
    }

    // Normalise: the sink joins the denominator only; a head with no valid row writes zeros.
    __syncthreads(); // the last tile's alpha reads precede this write
    if (sm_sub == 0) {
        float factor = 0.0F;
        const int h  = h0 + sm_head;
        if (m_run != -INFINITY && h < heads) {
            const float sink_log2 = sink[h] * 1.4426950408889634F;
            const float top       = fmaxf(m_run, sink_log2);
            const float a         = exp2f(m_run - top);
            factor                = a / (l_run * a + exp2f(sink_log2 - top));
        }
        alpha_s[sm_head] = factor;
    }
    __syncthreads();
#pragma unroll
    for (int m = 0; m < 2; ++m) {
#pragma unroll
        for (int r = 0; r < 2; ++r) {
            const int h = h0 + m * 16 + g + 8 * r;
            if (h >= heads) { continue; }
            const float f = alpha_s[m * 16 + g + 8 * r];
            __nv_bfloat16* row = out + (static_cast<std::size_t>(t) * heads + h) * D + warp * L::slice;
#pragma unroll
            for (int n = 0; n < L::ntiles; ++n) {
                *reinterpret_cast<__nv_bfloat162*>(row + n * 8 + 2 * tq) =
                    __floats2bfloat162_rn(acc[n][m][2 * r] * f, acc[n][m][2 * r + 1] * f);
            }
        }
    }
}

} // namespace ninfer::ops::gathered_mma
