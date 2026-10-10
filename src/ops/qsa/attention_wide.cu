// QSA attention, wide route (launch.h): one CTA per (column, KV head), unsplit, Int8Group64.
// Adapted from Infernix fd2aa93c9f8bc716d7ec1fea3c36abba067ad8de src/ops/qsa/qsa_prompt.cu
// (Apache-2.0): qsa_prompt_kernel<Int8Group64, Stages> and its launch rule. Modified for NInfer: the
// INT8-G64 path only, one sequence, no page spaces, and the dense sentinel (count -1) replaced by an
// explicit selection.

#include "ops/qsa/launch.h"

#include "ops/common/math.cuh"
#include "ops/common/memory.cuh"
#include "ops/common/mma.cuh"
#include "ops/common/warp.cuh"
#include "ops/kernel/paged_kv_address.cuh"
#include "ops/kv_cache/hadamard_d256.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <math_constants.h>

#include <stdexcept>
#include <string>

namespace ninfer::ops::detail {
namespace {

using bf16 = __nv_bfloat16;

constexpr int kD       = kQsaHeadDim;
constexpr int kTile    = 16;
constexpr int kRows    = 16;
constexpr int kQRow    = 2 * kD;
constexpr int kWarps   = 4;
constexpr int kThreads = kWarps * 32;
constexpr int kRatio   = kQsaBlockTokens;
constexpr int kGroups  = 4;
constexpr float kMaxScale = 256.0f;

// Stage: [K rows | V rows | K scales | V scales], 16 tokens each; rows keep 16-byte chunks swizzled.
constexpr int kKeyBytes    = kD;
constexpr int kValues      = kTile * kKeyBytes;
constexpr int kKeyScales   = kValues + kTile * kD;
constexpr int kValueScales = kKeyScales + kTile * 8;
constexpr int kStageBytes  = kValueScales + kTile * 8;

template <int Stages>
__host__ __device__ constexpr int prompt_smem_bytes() {
    constexpr int stages = kWarps * Stages * kStageBytes;
    constexpr int merge  = (kRows * kD + kWarps * kRows * 2) * 4;
    return kQRow * kRows + (stages > merge ? stages : merge);
}

__device__ __forceinline__ int swizzled(int row, int row_bytes, int chunk) {
    return row * row_bytes + ((chunk ^ (row & 7)) << 4);
}

__device__ __forceinline__ int attended_token(int i, int count, const std::int32_t* list, int p) {
    const int block_tokens = count * kRatio;
    if (i < block_tokens) { return list[i / kRatio] * kRatio + i % kRatio; }
    return (p + 1) / kRatio * kRatio + (i - block_tokens);
}

__device__ __forceinline__ int int8_query_position(int d) {
    const int local = d & 15;
    const int p     = ((local & 2) != 0 ? 8 : 0) + 2 * (local >> 2) + (local & 1);
    return (d & ~15) + p;
}

__device__ __forceinline__ void widen_codes(unsigned codes, unsigned& lo, unsigned& hi) {
    const unsigned biased = codes ^ 0x80808080u;
    const unsigned e      = __byte_perm(biased, 0x64646464u, 0x5140);
    const unsigned o      = __byte_perm(biased, 0x64646464u, 0x7362);
    const __half2 offset  = __float2half2_rn(1152.0f);
    const __half2 l       = __hsub2(load_vec<__half2>(&e), offset);
    const __half2 h       = __hsub2(load_vec<__half2>(&o), offset);
    lo                    = load_vec<unsigned>(&l);
    hi                    = load_vec<unsigned>(&h);
}

__device__ __forceinline__ void decode_v_pair(unsigned codes, __half2 scales, unsigned& even, unsigned& odd) {
    const unsigned biased = codes ^ 0x80808080u;
    const unsigned e      = __byte_perm(biased, 0x64646464u, 0x4240);
    const unsigned o      = __byte_perm(biased, 0x64646464u, 0x4341);
    const __half2 offset  = __float2half2_rn(1152.0f);
    const __half2 ve      = __hmul2(__hsub2(load_vec<__half2>(&e), offset), scales);
    const __half2 vo      = __hmul2(__hsub2(load_vec<__half2>(&o), offset), scales);
    even                  = load_vec<unsigned>(&ve);
    odd                   = load_vec<unsigned>(&vo);
}

template <int Stages>
__global__ void __launch_bounds__(kThreads, Stages == 1 ? 2 : 1)
    wide_attention_kernel(const bf16* __restrict__ q, const std::int8_t* __restrict__ k_pages,
                     const std::int8_t* __restrict__ v_pages, const __half* __restrict__ k_scales,
                     const __half* __restrict__ v_scales, const std::int32_t* __restrict__ table,
                     const std::int32_t* __restrict__ positions, const std::int32_t* __restrict__ selected,
                     const std::int32_t* __restrict__ counts, float scale, bf16* __restrict__ out) {
    constexpr int heads = kQsaQueryHeads, kv_heads = kQsaKvHeads, group = kQsaGroup;
    constexpr float Log2E       = 1.4426950408889634074f;
    constexpr unsigned FullMask = 0xffffffffu;
    extern __shared__ __align__(16) unsigned char smem[];

    const int t = static_cast<int>(blockIdx.x), kv_head = static_cast<int>(blockIdx.y);
    const int tid = static_cast<int>(threadIdx.x), warp = tid >> 5, lane = tid & 31;
    const int gid = lane >> 2, lid = lane & 3;
    const int p     = positions[t];
    const int count = counts[t];
    const int total = count * kRatio + (p + 1) % kRatio;
    const std::int32_t* list = selected + static_cast<std::size_t>(t) * kQsaTopBlocks;

    unsigned char* q_s = smem;
    for (int r = warp; r < kRows; r += kWarps) {
        float v[8];
#pragma unroll
        for (int k = 0; k < 8; ++k) {
            v[k] = r < group ? __bfloat162float(q[(static_cast<std::size_t>(t) * heads + kv_head * group + r) * kD + lane + 32 * k])
                             : 0.0f;
        }
        normalized_hadamard_d256_inplace(v, lane);
#pragma unroll
        for (int k = 0; k < 8; ++k) {
            const int d = lane + 32 * k;
            const int e = int8_query_position(d);
            unsigned char* at = q_s + swizzled(r, kQRow, e >> 3) + 2 * (e & 7);
            *reinterpret_cast<__half*>(at) = __float2half_rn(v[k]);
        }
    }
    __syncthreads();

    unsigned char* stages = smem + kQRow * kRows + warp * (Stages * kStageBytes);
    const int tiles       = (total + kTile - 1) / kTile;

    const auto issue = [&](int i, int s) {
        unsigned char* k_s = stages + s * kStageBytes;
        unsigned char* v_s = k_s + kValues;
        const int row      = lane & 15;
        const int index    = i * kTile + row;
        const bool valid   = index < total;
        const int token    = valid ? attended_token(index, count, list, p) : 0;
        const int page     = valid ? table[token >> kPagedKVPageShift] : 0;
        const std::size_t row_at =
            static_cast<std::size_t>(kv_head + kv_heads * page) * kPagedKVPageSize + (token & kPagedKVPageMask);
        const auto copy_rows = [&](unsigned char* dst_rows, const std::int8_t* own_row) {
            constexpr int Chunks = kD / 16;
            const auto own       = reinterpret_cast<unsigned long long>(own_row);
#pragma unroll
            for (int j = 0; j < kTile * Chunks / 32; ++j) {
                const int chunk = lane + 32 * j;
                const int r = chunk / Chunks, c = chunk % Chunks;
                const auto* src = reinterpret_cast<const unsigned char*>(__shfl_sync(FullMask, own, r));
                const bool ok   = __shfl_sync(FullMask, static_cast<int>(valid), r) != 0;
                cp_async_zfill<16, Cache::cg>(dst_rows + swizzled(r, kD, c), src + 16 * c, ok ? 16 : 0);
            }
        };
        copy_rows(k_s, k_pages + row_at * kD);
        copy_rows(v_s, v_pages + row_at * kD);
        if (lane < 16) {
            cp_async_zfill<8>(k_s + kKeyScales + row * 8, k_scales + row_at * kGroups, valid ? 8 : 0);
        } else {
            cp_async_zfill<8>(k_s + kValueScales + row * 8, v_scales + row_at * kGroups, valid ? 8 : 0);
        }
        cp_commit();
    };

    const int a_mat = lane >> 3, a_rin = lane & 7;
    const int a_row = a_rin + ((a_mat & 1) << 3);
    const auto query_fragment = [&](int kb, unsigned (&a)[4]) {
        ldmatrix_x4(a[0], a[1], a[2], a[3], smem_addr(q_s + swizzled(a_row, kQRow, 2 * kb + (a_mat >> 1))));
    };

    float o[16][2][4];
#pragma unroll
    for (int b = 0; b < 16; ++b) {
#pragma unroll
        for (int h = 0; h < 2; ++h) {
#pragma unroll
            for (int i = 0; i < 4; ++i) { o[b][h][i] = 0.0f; }
        }
    }
    float m0 = -CUDART_INF_F, m1 = -CUDART_INF_F, l0 = 0.0f, l1 = 0.0f;
    const float scale_l2 = scale * Log2E;

    const auto compute = [&](int i, int s) {
        const unsigned char* k_s = stages + s * kStageBytes;
        const unsigned char* v_s = k_s + kValues;
        float score[2][4];
#pragma unroll
        for (int nt = 0; nt < 2; ++nt) {
#pragma unroll
            for (int j = 0; j < 4; ++j) { score[nt][j] = 0.0f; }
        }
        constexpr int Steps = 16 / kGroups;
#pragma unroll
        for (int g = 0; g < kGroups; ++g) {
            float acc[2][4];
#pragma unroll
            for (int nt = 0; nt < 2; ++nt) {
#pragma unroll
                for (int j = 0; j < 4; ++j) { acc[nt][j] = 0.0f; }
            }
#pragma unroll
            for (int kk = 0; kk < Steps; ++kk) {
                const int kb = g * Steps + kk;
                unsigned a[4];
                query_fragment(kb, a);
                unsigned r0, r1;
                ldmatrix_x2(r0, r1, smem_addr(k_s + swizzled(lane & 15, kKeyBytes, kb)));
                unsigned lo, hi;
                widen_codes(r0, lo, hi);
                mma_f16(acc[0][0], acc[0][1], acc[0][2], acc[0][3], a[0], a[1], a[2], a[3], lo, hi);
                widen_codes(r1, lo, hi);
                mma_f16(acc[1][0], acc[1][1], acc[1][2], acc[1][3], a[0], a[1], a[2], a[3], lo, hi);
            }
#pragma unroll
            for (int nt = 0; nt < 2; ++nt) {
                const int key    = nt * 8 + 2 * lid;
                const __half* ks = reinterpret_cast<const __half*>(k_s + kKeyScales);
                const float ks0  = __half2float(ks[key * kGroups + g]);
                const float ks1  = __half2float(ks[(key + 1) * kGroups + g]);
                score[nt][0]     = __fmaf_rn(ks0, acc[nt][0], score[nt][0]);
                score[nt][1]     = __fmaf_rn(ks1, acc[nt][1], score[nt][1]);
                score[nt][2]     = __fmaf_rn(ks0, acc[nt][2], score[nt][2]);
                score[nt][3]     = __fmaf_rn(ks1, acc[nt][3], score[nt][3]);
            }
        }
        float bm0 = -CUDART_INF_F, bm1 = -CUDART_INF_F;
#pragma unroll
        for (int nt = 0; nt < 2; ++nt) {
            const int index = i * kTile + nt * 8 + 2 * lid;
            if (index >= total) { score[nt][0] = score[nt][2] = -CUDART_INF_F; }
            if (index + 1 >= total) { score[nt][1] = score[nt][3] = -CUDART_INF_F; }
            bm0 = fmaxf(bm0, fmaxf(score[nt][0], score[nt][1]));
            bm1 = fmaxf(bm1, fmaxf(score[nt][2], score[nt][3]));
        }
        bm0 = warp_max<4>(bm0, FullMask);
        bm1 = warp_max<4>(bm1, FullMask);
        const float nm0 = fmaxf(m0, bm0), nm1 = fmaxf(m1, bm1);
        const float alpha0 = m0 == -CUDART_INF_F ? 0.0f : exp2_approx((m0 - nm0) * scale_l2);
        const float alpha1 = m1 == -CUDART_INF_F ? 0.0f : exp2_approx((m1 - nm1) * scale_l2);
        m0 = nm0;
        m1 = nm1;
        const float base0 = nm0 * scale_l2, base1 = nm1 * scale_l2;
        float pr[2][4];
        float bl0 = 0.0f, bl1 = 0.0f;
#pragma unroll
        for (int nt = 0; nt < 2; ++nt) {
            pr[nt][0] = exp2_approx(__fmaf_rn(score[nt][0], scale_l2, -base0));
            pr[nt][1] = exp2_approx(__fmaf_rn(score[nt][1], scale_l2, -base0));
            pr[nt][2] = exp2_approx(__fmaf_rn(score[nt][2], scale_l2, -base1));
            pr[nt][3] = exp2_approx(__fmaf_rn(score[nt][3], scale_l2, -base1));
            bl0 += pr[nt][0] + pr[nt][1];
            bl1 += pr[nt][2] + pr[nt][3];
        }
        l0 = __fmaf_rn(l0, alpha0, bl0);
        l1 = __fmaf_rn(l1, alpha1, bl1);
#pragma unroll
        for (int b = 0; b < 16; ++b) {
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                o[b][h][0] *= alpha0;
                o[b][h][1] *= alpha0;
                o[b][h][2] *= alpha1;
                o[b][h][3] *= alpha1;
            }
        }
        float own;
        {
            const __half* vs   = reinterpret_cast<const __half*>(k_s + kValueScales);
            const __half2 pair = load_vec<__half2>(&vs[2 * lane]);
            own                = fmaxf(fabsf(__low2float(pair)), fabsf(__high2float(pair)));
        }
        const float vmax = warp_max(own, FullMask);
        int shift        = 0;
        while (shift < 14 && ldexpf(vmax, -shift) > kMaxScale) { ++shift; }
        const float up = ldexpf(1.0f, shift);
        unsigned pa[4];
        pa[0] = pack_f16x2(pr[0][0] * up, pr[0][1] * up);
        pa[1] = pack_f16x2(pr[0][2] * up, pr[0][3] * up);
        pa[2] = pack_f16x2(pr[1][0] * up, pr[1][1] * up);
        pa[3] = pack_f16x2(pr[1][2] * up, pr[1][3] * up);
        const __half2 down = __float2half2_rn(ldexpf(1.0f, -shift));
        const int v_mat = lane >> 3, v_rin = lane & 7;
        const int v_row = v_rin + ((v_mat & 1) << 3);
        const auto value_scale = [&](int key, int g) {
            return reinterpret_cast<const __half*>(k_s + kValueScales)[key * kGroups + g];
        };
#pragma unroll
        for (int db = 0; db < 16; db += 2) {
            unsigned r[4];
            ldmatrix_x4_t(r[0], r[1], r[2], r[3], smem_addr(v_s + swizzled(v_row, kD, db + (v_mat >> 1))));
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int g       = (db + h) * kGroups / 16;
                const int key     = 2 * lid;
                const __half2 s01 = __hmul2(__halves2half2(value_scale(key, g), value_scale(key + 1, g)), down);
                const __half2 s89 = __hmul2(__halves2half2(value_scale(key + 8, g), value_scale(key + 9, g)), down);
                unsigned e0, o0, e1, o1;
                decode_v_pair(r[2 * h], s01, e0, o0);
                decode_v_pair(r[2 * h + 1], s89, e1, o1);
                float* even = o[db + h][0];
                float* odd  = o[db + h][1];
                mma_f16(even[0], even[1], even[2], even[3], pa[0], pa[1], pa[2], pa[3], e0, e1);
                mma_f16(odd[0], odd[1], odd[2], odd[3], pa[0], pa[1], pa[2], pa[3], o0, o1);
            }
        }
    };
    if constexpr (Stages == 1) {
        for (int i = warp; i < tiles; i += kWarps) {
            issue(i, 0);
            cp_wait<0>();
            __syncwarp();
            compute(i, 0);
            __syncwarp();
        }
    } else {
        int s = 0;
        if (warp < tiles) { issue(warp, 0); }
        for (int i = warp; i < tiles; i += kWarps) {
            if (i + kWarps < tiles) {
                issue(i + kWarps, s ^ 1);
                cp_wait<1>();
            } else {
                cp_wait<0>();
            }
            __syncwarp();
            compute(i, s);
            __syncwarp();
            s ^= 1;
        }
    }
    cp_wait<0>();
    l0 += __shfl_xor_sync(FullMask, l0, 1);
    l0 += __shfl_xor_sync(FullMask, l0, 2);
    l1 += __shfl_xor_sync(FullMask, l1, 1);
    l1 += __shfl_xor_sync(FullMask, l1, 2);
    __syncthreads();
    float* rows  = reinterpret_cast<float*>(smem + kQRow * kRows);
    float* stats = rows + kRows * kD;
    if (lid == 0) {
        stats[(warp * kRows + gid) * 2]         = m0;
        stats[(warp * kRows + gid) * 2 + 1]     = l0;
        stats[(warp * kRows + gid + 8) * 2]     = m1;
        stats[(warp * kRows + gid + 8) * 2 + 1] = l1;
    }
    __syncthreads();
    const auto row_max = [&](int row) {
        float m = -CUDART_INF_F;
#pragma unroll
        for (int w = 0; w < kWarps; ++w) { m = fmaxf(m, stats[(w * kRows + row) * 2]); }
        return m;
    };
    const float f0 = m0 == -CUDART_INF_F ? 0.0f : exp2_approx((m0 - row_max(gid)) * scale_l2);
    const float f1 = m1 == -CUDART_INF_F ? 0.0f : exp2_approx((m1 - row_max(gid + 8)) * scale_l2);
    for (int w = 0; w < kWarps; ++w) {
        if (warp == w) {
#pragma unroll
            for (int b = 0; b < 16; ++b) {
#pragma unroll
                for (int h = 0; h < 2; ++h) {
#pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        const int upper = j >> 1;
                        const int row   = gid + 8 * upper;
                        const int d     = 16 * b + 4 * lid + 2 * (j & 1) + h;
                        const bool live = (upper ? m1 : m0) != -CUDART_INF_F;
                        float& a        = rows[row * kD + d];
                        if (w == 0) {
                            a = live ? __fmaf_rn(upper ? f1 : f0, o[b][h][j], 0.0f) : 0.0f;
                        } else if (live) {
                            a = __fmaf_rn(upper ? f1 : f0, o[b][h][j], a);
                        }
                    }
                }
            }
        }
        __syncthreads();
    }
    for (int e = tid; e < group * kD; e += kThreads) {
        const int row = e / kD, d = e % kD;
        const float m = row_max(row);
        float l       = 0.0f;
#pragma unroll
        for (int w = 0; w < kWarps; ++w) {
            const float mw = stats[(w * kRows + row) * 2];
            if (mw == -CUDART_INF_F) { continue; }
            l = __fmaf_rn(exp2_approx((mw - m) * scale_l2), stats[(w * kRows + row) * 2 + 1], l);
        }
        const float value = l > 0.0f ? rows[row * kD + d] / l : 0.0f;
        out[(static_cast<std::size_t>(t) * heads + kv_head * group + row) * kD + d] = __float2bfloat16_rn(value);
    }
}

template <int Stages>
void run(const Tensor& q, const Tensor& positions, const Tensor& selected, const Tensor& counts, float scale,
         const PagedKVLayerView& cache, Tensor& out, cudaStream_t stream) {
    constexpr int bytes = prompt_smem_bytes<Stages>();
    static const bool configured =
        cudaFuncSetAttribute(wide_attention_kernel<Stages>, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes) == cudaSuccess;
    if (!configured) { throw std::runtime_error("qsa wide attention: cannot reserve shared memory"); }
    const dim3 grid(static_cast<unsigned>(q.ne[2]), kQsaKvHeads);
    wide_attention_kernel<Stages><<<grid, kThreads, bytes, stream>>>(
        static_cast<const bf16*>(q.data), static_cast<const std::int8_t*>(cache.k_pages.data),
        static_cast<const std::int8_t*>(cache.v_pages.data), static_cast<const __half*>(cache.k_scale_pages.data),
        static_cast<const __half*>(cache.v_scale_pages.data), static_cast<const std::int32_t*>(cache.block_table.data),
        static_cast<const std::int32_t*>(positions.data), static_cast<const std::int32_t*>(selected.data),
        static_cast<const std::int32_t*>(counts.data), scale, static_cast<bf16*>(out.data));
}

} // namespace

void qsa_attention_wide_launch(const Tensor& q, const Tensor& positions, const Tensor& selected,
                               const Tensor& counts, float scale, const PagedKVLayerView& cache,
                               Tensor& out, std::int32_t sms, cudaStream_t stream) {
    constexpr bool two_ctas = 2 * (prompt_smem_bytes<1>() + 1024) <= 100 * 1024;
    if (two_ctas && q.ne[2] * kQsaKvHeads >= sms) {
        run<1>(q, positions, selected, counts, scale, cache, out, stream);
    } else {
        run<2>(q, positions, selected, counts, scale, cache, out, stream);
    }
    const cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) { throw std::runtime_error(std::string("qsa wide attention: ") + cudaGetErrorString(error)); }
}

} // namespace ninfer::ops::detail
