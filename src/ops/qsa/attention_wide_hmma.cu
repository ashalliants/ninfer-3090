// QSA attention, wide route, on Volta's FP16 tensor cores (mma.sync.m8n8k4): the sm_70 counterpart
// of attention_wide.cu, taken when NINFER_SM70_TENSOR_CORES=1 (ops/linear/common/sm70_tensor_core.h).
//
// attention_wide.cu is written in Ampere m16n8k16 fragments, which sm_70 emulates with shuffles and
// FMAs (ops/common/volta_compat.cuh); this kernel keeps its structure -- one CTA per (column, KV
// head), the 12 query rows of the head's group padded to 16, four warps taking 16-key tiles in turn,
// an online softmax per warp and a merge at the end -- and does the two products natively:
//
//   * m8n8k4 is four independent 8x8x4 products per warp instruction, one per quadpair (lanes
//     4q..4q+3 and 16+4q..16+4q+3). Lane position p = (lane & 3) + 4 ((lane >> 4) & 1). Measured
//     fragment maps (f16 inputs, f32 accumulator): a row-major A row is lane p's four consecutive
//     k; a column-major B column is lane p's four consecutive k; a row-major B row k = lane & 3
//     holds columns 4 ((lane >> 4) & 1) + 0..3; accumulator register i holds row
//     (i & 2) | (lane & 1) | 4 ((lane >> 4) & 1) and column (i & 1) | 2 ((lane >> 1) & 1) | 4 (i >> 2).
//   * QK: quadpair q takes query rows 8 (q & 1).. and keys 8 (q >> 1).. of the tile. A is the FP16
//     query, Hadamard-rotated as in attention_wide.cu but in natural order (that kernel permutes it
//     to m16n8k16's byte pairing of the INT8 rows; four consecutive codes here are four k); B
//     the key's INT8 codes widened exactly to FP16. Each 64-wide group is summed in FP32 by the MMA
//     and scaled by its key's group scale in FP32, as in attention_wide.cu.
//   * Softmax: the scores go through the warp's shared scratch; lanes 0..15 own one row each.
//   * PV: quadpair q takes rows 8 (q & 1).. and dims 128 (q >> 1)..; B is the value codes as a
//     row-major fragment (four consecutive bytes of one key row, widened exactly), and A folds the
//     value scale into the probability: A = FP16(p * vscale[key][group] * 2^u), with 2^u per tile
//     putting the tile's largest scale at 2^7..2^8. The running output carries 2^u, rescaled with
//     the softmax's alpha when u changes and removed at the merge.
//
// Numerics against attention_wide.cu: the same FP16 query and exact codes, FP32 accumulation in
// another order, and one FP16 rounding of p * vscale where that kernel rounds p and v * vscale
// separately. Not bit-identical, so it is opt-in with the rest of the tensor-core route.

#include "ops/common/smem_limit.h"
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

// Stage (as attention_wide.cu): [K rows | V rows | K scales | V scales], 16-byte chunks swizzled.
constexpr int kKeyBytes    = kD;
constexpr int kValues      = kTile * kKeyBytes;
constexpr int kKeyScales   = kValues + kTile * kD;
constexpr int kValueScales = kKeyScales + kTile * 8;
constexpr int kStageBytes  = kValueScales + kTile * 8;
// Per-warp scratch: scores / probabilities [16 rows][16 keys] and the rows' alpha.
constexpr int kScratch = (kRows * kTile + kRows) * 4;
constexpr int kStats   = 3; // m, l, u per (warp, row)

template <int Stages>
__host__ __device__ constexpr int hmma_smem_bytes() {
    constexpr int stages = kWarps * Stages * kStageBytes;
    constexpr int merge  = (kRows * kD + kWarps * kRows * kStats) * 4;
    return kQRow * kRows + kWarps * kScratch + (stages > merge ? stages : merge);
}

__device__ __forceinline__ int swizzled(int row, int row_bytes, int chunk) {
    return row * row_bytes + ((chunk ^ (row & 7)) << 4);
}

__device__ __forceinline__ int attended_token(int i, int count, const std::int32_t* list, int p) {
    const int block_tokens = count * kRatio;
    if (i < block_tokens) { return list[i / kRatio] * kRatio + i % kRatio; }
    return (p + 1) / kRatio * kRatio + (i - block_tokens);
}

// Four signed codes -> FP16 pairs (c0, c1) and (c2, c3), exactly.
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

template <bool BRowMajor>
__device__ __forceinline__ void hmma(float (&d)[8], unsigned a0, unsigned a1, unsigned b0, unsigned b1) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
    if constexpr (BRowMajor) {
        asm volatile("mma.sync.aligned.m8n8k4.row.row.f32.f16.f16.f32 "
                     "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, {%0,%1,%2,%3,%4,%5,%6,%7};"
                     : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]),
                       "+f"(d[7])
                     : "r"(a0), "r"(a1), "r"(b0), "r"(b1));
    } else {
        asm volatile("mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "
                     "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, {%0,%1,%2,%3,%4,%5,%6,%7};"
                     : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]),
                       "+f"(d[7])
                     : "r"(a0), "r"(a1), "r"(b0), "r"(b1));
    }
#endif
}

template <int Stages>
__global__ void __launch_bounds__(kThreads, Stages == 1 ? 2 : 1)
    wide_attention_hmma_kernel(const bf16* __restrict__ q, const std::int8_t* __restrict__ k_pages,
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
    const int p     = positions[t];
    const int count = counts[t];
    const int total = count * kRatio + (p + 1) % kRatio;
    const std::int32_t* list = selected + static_cast<std::size_t>(t) * kQsaTopBlocks;

    // The query, rotated as attention_wide.cu does it.
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
            const int d = lane + 32 * k; // natural order: a fragment's four k are four consecutive codes
            unsigned char* at = q_s + swizzled(r, kQRow, d >> 3) + 2 * (d & 7);
            *reinterpret_cast<__half*>(at) = __float2half_rn(v[k]);
        }
    }
    __syncthreads();

    float* sp       = reinterpret_cast<float*>(smem + kQRow * kRows + warp * kScratch);
    float* alpha_s  = sp + kRows * kTile;
    unsigned char* stages = smem + kQRow * kRows + kWarps * kScratch + warp * (Stages * kStageBytes);
    const int tiles = (total + kTile - 1) / kTile;

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

    const int qp = (lane >> 2) & 3, hi = (lane >> 4) & 1, pos = (lane & 3) + 4 * hi;
    const int mq = qp & 1, half = qp >> 1;           // QK: key half; PV: dim half
    const int row_a = mq * 8 + ((lane & 1) | 4 * hi); // accumulator rows row_a, row_a + 2
    const int col_a = 2 * ((lane >> 1) & 1);          // accumulator cols col_a + (i & 1) + 4 (i >> 2)

    float o[16][8];
#pragma unroll
    for (int b = 0; b < 16; ++b) {
#pragma unroll
        for (int i = 0; i < 8; ++i) { o[b][i] = 0.0f; }
    }
    float m_run = -CUDART_INF_F, l_run = 0.0f; // row `lane` (lanes 0..15)
    int u_cur = 0;
    const float scale_l2 = scale * Log2E;

    const auto compute = [&](int i, int s) {
        const unsigned char* k_s = stages + s * kStageBytes;
        const unsigned char* v_s = k_s + kValues;
        const __half* ks = reinterpret_cast<const __half*>(k_s + kKeyScales);
        const __half* vs = reinterpret_cast<const __half*>(k_s + kValueScales);

        // QK: rows mq*8.., keys half*8..; this lane supplies query row mq*8+pos and key half*8+pos.
        float score[8];
#pragma unroll
        for (int e = 0; e < 8; ++e) { score[e] = 0.0f; }
        const int key_b = half * 8 + pos, q_row = mq * 8 + pos;
#pragma unroll
        for (int g = 0; g < kGroups; ++g) {
            float acc[8];
#pragma unroll
            for (int e = 0; e < 8; ++e) { acc[e] = 0.0f; }
#pragma unroll
            for (int c4 = 0; c4 < 4; ++c4) {
                const int ch  = 4 * g + c4;
                const uint4 kw = *reinterpret_cast<const uint4*>(k_s + swizzled(key_b, kKeyBytes, ch));
                const uint4 q0 = *reinterpret_cast<const uint4*>(q_s + swizzled(q_row, kQRow, 2 * ch));
                const uint4 q1 = *reinterpret_cast<const uint4*>(q_s + swizzled(q_row, kQRow, 2 * ch + 1));
                unsigned lo, hv;
                widen_codes(kw.x, lo, hv);
                hmma<false>(acc, q0.x, q0.y, lo, hv);
                widen_codes(kw.y, lo, hv);
                hmma<false>(acc, q0.z, q0.w, lo, hv);
                widen_codes(kw.z, lo, hv);
                hmma<false>(acc, q1.x, q1.y, lo, hv);
                widen_codes(kw.w, lo, hv);
                hmma<false>(acc, q1.z, q1.w, lo, hv);
            }
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                const int key = half * 8 + col_a + (e & 1) + 4 * (e >> 2);
                score[e]      = __fmaf_rn(__half2float(ks[key * kGroups + g]), acc[e], score[e]);
            }
        }
#pragma unroll
        for (int e = 0; e < 8; ++e) {
            sp[(row_a + (e & 2)) * kTile + half * 8 + col_a + (e & 1) + 4 * (e >> 2)] = score[e];
        }
        // The tile's value-scale exponent: its largest scale lands in [2^7, 2^8).
        float own;
        {
            const __half2 pair = load_vec<__half2>(&vs[2 * lane]);
            own                = fmaxf(fabsf(__low2float(pair)), fabsf(__high2float(pair)));
        }
        const float vmax = warp_max(own, FullMask);
        int u            = u_cur;
        if (vmax > 0.0f && isfinite(vmax)) {
            int x = 0;
            (void)frexpf(vmax, &x);
            u = 8 - x;
        }
        __syncwarp();
        if (lane < kRows) {
            float* row = sp + lane * kTile;
            float bm   = -CUDART_INF_F;
            float sc[kTile];
#pragma unroll
            for (int k = 0; k < kTile; ++k) {
                sc[k] = i * kTile + k < total ? row[k] : -CUDART_INF_F;
                bm    = fmaxf(bm, sc[k]);
            }
            const float nm    = fmaxf(m_run, bm);
            const float alpha = m_run == -CUDART_INF_F ? 0.0f : exp2_approx((m_run - nm) * scale_l2);
            const float base  = nm * scale_l2;
            float bl          = 0.0f;
#pragma unroll
            for (int k = 0; k < kTile; ++k) {
                const float pk = exp2_approx(__fmaf_rn(sc[k], scale_l2, -base));
                bl += pk;
                row[k] = pk;
            }
            l_run         = __fmaf_rn(l_run, alpha, bl);
            m_run         = nm;
            alpha_s[lane] = alpha * ldexpf(1.0f, u - u_cur);
        }
        u_cur = u;
        __syncwarp();

        // PV: rows mq*8.., dims half*128..; rescale, then A = p * vscale * 2^u for the two groups.
        const float al_a = alpha_s[row_a], al_b = alpha_s[row_a + 2];
#pragma unroll
        for (int b = 0; b < 16; ++b) {
#pragma unroll
            for (int e = 0; e < 8; ++e) { o[b][e] *= (e & 2) ? al_b : al_a; }
        }
        const float up = ldexpf(1.0f, u);
        unsigned a[2][4][2];
        const float* prow = sp + (mq * 8 + pos) * kTile;
#pragma unroll
        for (int gl = 0; gl < 2; ++gl) {
            const int g = 2 * half + gl;
#pragma unroll
            for (int st = 0; st < 4; ++st) {
                float pv[4];
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const int key = 4 * st + j;
                    pv[j]         = prow[key] * __half2float(vs[key * kGroups + g]) * up;
                }
                a[gl][st][0] = pack_f16x2(pv[0], pv[1]);
                a[gl][st][1] = pack_f16x2(pv[2], pv[3]);
            }
        }
#pragma unroll
        for (int b = 0; b < 16; ++b) {
            const int gl   = b >> 3;
            const int dim0 = half * 128 + b * 8 + 4 * hi;
#pragma unroll
            for (int st = 0; st < 4; ++st) {
                const int key    = 4 * st + (lane & 3);
                const unsigned w = *reinterpret_cast<const unsigned*>(v_s + swizzled(key, kD, dim0 >> 4) + (dim0 & 15));
                unsigned lo, hv;
                widen_codes(w, lo, hv);
                hmma<true>(o[b], a[gl][st][0], a[gl][st][1], lo, hv);
            }
        }
        __syncwarp();
    };
    if constexpr (Stages == 1) {
        for (int i = warp; i < tiles; i += kWarps) {
            issue(i, 0);
            cp_wait<0>();
            __syncwarp();
            compute(i, 0);
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
            s ^= 1;
        }
    }
    cp_wait<0>();
    __syncthreads();
    float* rows  = reinterpret_cast<float*>(smem + kQRow * kRows + kWarps * kScratch);
    float* stats = rows + kRows * kD;
    if (lane < kRows) {
        stats[(warp * kRows + lane) * kStats]     = m_run;
        stats[(warp * kRows + lane) * kStats + 1] = l_run;
        stats[(warp * kRows + lane) * kStats + 2] = static_cast<float>(u_cur);
    }
    __syncthreads();
    const auto row_max = [&](int row) {
        float m = -CUDART_INF_F;
#pragma unroll
        for (int w = 0; w < kWarps; ++w) { m = fmaxf(m, stats[(w * kRows + row) * kStats]); }
        return m;
    };
    {
        // This warp's factor per accumulator row: exp2((m_w - m) scale) * 2^-u (0 for a row it never saw).
        const float mw_a = stats[(warp * kRows + row_a) * kStats], mw_b = stats[(warp * kRows + row_a + 2) * kStats];
        const float down = ldexpf(1.0f, -u_cur);
        const float f_a  = mw_a == -CUDART_INF_F ? 0.0f : exp2_approx((mw_a - row_max(row_a)) * scale_l2) * down;
        const float f_b  = mw_b == -CUDART_INF_F ? 0.0f : exp2_approx((mw_b - row_max(row_a + 2)) * scale_l2) * down;
        for (int w = 0; w < kWarps; ++w) {
            if (warp == w) {
#pragma unroll
                for (int b = 0; b < 16; ++b) {
#pragma unroll
                    for (int e = 0; e < 8; ++e) {
                        const int row = row_a + (e & 2);
                        const int d   = half * 128 + b * 8 + col_a + (e & 1) + 4 * (e >> 2);
                        const float f = (e & 2) ? f_b : f_a;
                        float& acc    = rows[row * kD + d];
                        acc           = w == 0 ? f * o[b][e] : __fmaf_rn(f, o[b][e], acc);
                    }
                }
            }
            __syncthreads();
        }
    }
    for (int e = tid; e < group * kD; e += kThreads) {
        const int row = e / kD, d = e % kD;
        const float m = row_max(row);
        float l       = 0.0f;
#pragma unroll
        for (int w = 0; w < kWarps; ++w) {
            const float mw = stats[(w * kRows + row) * kStats];
            if (mw == -CUDART_INF_F) { continue; }
            l = __fmaf_rn(exp2_approx((mw - m) * scale_l2), stats[(w * kRows + row) * kStats + 1], l);
        }
        const float value = l > 0.0f ? rows[row * kD + d] / l : 0.0f;
        out[(static_cast<std::size_t>(t) * heads + kv_head * group + row) * kD + d] = __float2bfloat16_rn(value);
    }
}

template <int Stages>
void run(const Tensor& q, const Tensor& positions, const Tensor& selected, const Tensor& counts, float scale,
         const PagedKVLayerView& cache, Tensor& out, cudaStream_t stream) {
    constexpr int bytes = hmma_smem_bytes<Stages>();
    static_assert(bytes <= kMaxDynamicSharedBytes, "exceeds the target's per-block shared memory");
    static const bool configured =
        cudaFuncSetAttribute(wide_attention_hmma_kernel<Stages>, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes) ==
        cudaSuccess;
    if (!configured) { throw std::runtime_error("qsa wide attention (HMMA): cannot reserve shared memory"); }
    const dim3 grid(static_cast<unsigned>(q.ne[2]), kQsaKvHeads);
    wide_attention_hmma_kernel<Stages><<<grid, kThreads, bytes, stream>>>(
        static_cast<const bf16*>(q.data), static_cast<const std::int8_t*>(cache.k_pages.data),
        static_cast<const std::int8_t*>(cache.v_pages.data), static_cast<const __half*>(cache.k_scale_pages.data),
        static_cast<const __half*>(cache.v_scale_pages.data), static_cast<const std::int32_t*>(cache.block_table.data),
        static_cast<const std::int32_t*>(positions.data), static_cast<const std::int32_t*>(selected.data),
        static_cast<const std::int32_t*>(counts.data), scale, static_cast<bf16*>(out.data));
}

} // namespace

void qsa_attention_wide_hmma_launch(const Tensor& q, const Tensor& positions, const Tensor& selected,
                                    const Tensor& counts, float scale, const PagedKVLayerView& cache, Tensor& out,
                                    std::int32_t sms, cudaStream_t stream) {
    constexpr bool two_ctas = 2 * (hmma_smem_bytes<1>() + 1024) <= kSharedBytesPerSm;
    if (two_ctas && q.ne[2] * kQsaKvHeads >= sms) {
        run<1>(q, positions, selected, counts, scale, cache, out, stream);
    } else {
        run<2>(q, positions, selected, counts, scale, cache, out, stream);
    }
    const cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        throw std::runtime_error(std::string("qsa wide attention (HMMA): ") + cudaGetErrorString(error));
    }
}

} // namespace ninfer::ops::detail
