// QSA attention, split route (launch.h): calls too narrow to fill the SMs split each column's key
// list; Int8Group64 cache, D256/Hq24/Hkv2.
// Adapted from Infernix fd2aa93c9f8bc716d7ec1fea3c36abba067ad8de src/ops/qsa/qsa_prompt.cu and
// src/ops/qsa/qsa.cu (Apache-2.0): one CTA per (split, KV head, column) computes the KV head's
// query heads padded to 16 MMA rows, each warp sweeping its own 16-token tiles with its own online
// softmax, merged at the end; splits of the key list are merged by a second kernel. Modified for
// NInfer: sm_86 m16n8k16 FP16 MMA throughout. QK multiplies the FP16 Hadamard-prepared query by
// the exact FP16 key codes (a dimension order chosen so each thread's fragment is one 32-bit word
// of codes) and applies the key group scales to the FP32 partials; PV multiplies probabilities
// pre-scaled by the value group scales by exact FP16 value codes read with ldmatrix.trans.

#include "ops/qsa/launch.h"

#include "ops/common/memory.cuh"
#include "ops/common/mma.cuh"
#include "ops/kernel/paged_kv_address.cuh"
#include "ops/kv_cache/hadamard_d256.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <math_constants.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace ninfer::ops::detail {
namespace {

using bf16 = __nv_bfloat16;

constexpr int kD        = kQsaHeadDim;
constexpr int kHq       = kQsaQueryHeads;
constexpr int kHkv      = kQsaKvHeads;
constexpr int kGroup    = kQsaGroup; // 12 query heads per KV head
constexpr int kR        = kQsaBlockTokens;
constexpr int kTop      = kQsaTopBlocks;
constexpr int kWarps    = 4;
constexpr int kThreads  = 32 * kWarps;
constexpr int kTile     = 16; // tokens per warp tile
constexpr int kMaxTiles = (kQsaMaxAttended + kTile - 1) / kTile;
constexpr int kCodeRow  = kD + 16;   // bytes per staged code row (bank-conflict-free fragments)
constexpr int kQRow     = kD + 16;   // FP16 elements per staged query row
constexpr int kGroups   = kD / 64;   // scale groups per row
constexpr unsigned kAll = 0xffffffffU;
constexpr float kLog2e  = 1.4426950408889634F;

void check_launch(const char* what) {
    const cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        throw std::runtime_error(std::string("qsa ") + what + ": " + cudaGetErrorString(error));
    }
}

struct alignas(16) Stage {
    std::int8_t k[kTile][kCodeRow];
    std::int8_t v[kTile][kCodeRow];
    __half k_scale[kTile][kGroups];
    __half v_scale[kTile][kGroups];
};

template <int Stages>
struct Shared {
    __half q[16][kQRow];
    union {
        Stage stage[kWarps][Stages];
        float merge[2][kGroup][kD]; // two warps' output rows during the merge rounds
    };
    float row_max[kWarps][16];
    float row_sum[kWarps][16];
};

// Four signed INT8 codes (one word) to two FP16 pairs: bytes (sel) placed over the exponent of
// 1024 after a bias of 128, then 1152 subtracted. Exact for every code.
__device__ __forceinline__ unsigned codes_to_half2(unsigned biased, unsigned selector) {
    const unsigned raw = __byte_perm(biased, 0x64646464U, selector);
    __half2 h          = *reinterpret_cast<const __half2*>(&raw);
    h                  = __hsub2(h, __half2half2(__ushort_as_half(0x6480))); // 1152
    return *reinterpret_cast<const unsigned*>(&h);
}

__device__ __forceinline__ unsigned pack_half2(float lo, float hi) {
    const __half2 h = __floats2half2_rn(lo, hi);
    return *reinterpret_cast<const unsigned*>(&h);
}

// The token at index i of a column's attended list: the selected blocks' tokens, then the tail.
__device__ __forceinline__ int attended_position(const std::int32_t* __restrict__ ids, int count,
                                                 int tail_begin, int length, int i) {
    if (i >= length) { return -1; }
    const int block_tokens = count * kR;
    if (i < block_tokens) { return ids[i / kR] * kR + (i % kR); }
    return tail_begin + (i - block_tokens);
}

struct Planes {
    const std::int8_t* k;
    const std::int8_t* v;
    const __half* k_scale;
    const __half* v_scale;
    const std::int32_t* table;
};

// Stage tile `tile` of the list into `st` (zero-filled past the list).
__device__ __forceinline__ void stage_tile(Stage& st, const Planes& planes, int kv_head,
                                           const std::int32_t* __restrict__ ids, int count,
                                           int tail_begin, int length, int tile, int lane) {
    // Lanes 0..15 resolve the tile's tokens to cache rows.
    int row = 0, live = 0;
    if (lane < kTile) {
        const int position = attended_position(ids, count, tail_begin, length, tile * kTile + lane);
        if (position >= 0) {
            const int page = paged_kv_physical_page(planes.table, position);
            row  = (page * kHkv + kv_head) * kPagedKVPageSize + (position & kPagedKVPageMask);
            live = 1;
        }
    }
    // Codes: 16 tokens x 256 B per plane = 256 chunks of 16 B per plane.
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const int chunk = lane + 32 * j;
        const int token = chunk >> 4, part = chunk & 15;
        const int r     = __shfl_sync(kAll, row, token);
        const int ok    = __shfl_sync(kAll, live, token);
        const std::int64_t offset = static_cast<std::int64_t>(r) * kD + part * 16;
        cp_async_zfill<16>(&st.k[token][part * 16], planes.k + offset, ok ? 16 : 0);
        cp_async_zfill<16>(&st.v[token][part * 16], planes.v + offset, ok ? 16 : 0);
    }
    // Scales: 8 B per token and plane.
    {
        const int token = lane & 15;
        const int r     = __shfl_sync(kAll, row, token);
        const int ok    = __shfl_sync(kAll, live, token);
        const std::int64_t offset = static_cast<std::int64_t>(r) * kGroups;
        if (lane < 16) {
            cp_async_zfill<8>(&st.k_scale[token][0], planes.k_scale + offset, ok ? 8 : 0);
        } else {
            cp_async_zfill<8>(&st.v_scale[token][0], planes.v_scale + offset, ok ? 8 : 0);
        }
    }
}

template <int Stages>
__global__ void __launch_bounds__(kThreads)
    attention_kernel(const bf16* __restrict__ q, const std::int32_t* __restrict__ positions,
                     const std::int32_t* __restrict__ selected,
                     const std::int32_t* __restrict__ counts, Planes planes, float scale_log2,
                     int tiles_per_split, float* __restrict__ partial, bf16* __restrict__ out) {
    extern __shared__ __align__(16) unsigned char smem_raw[];
    auto& sh            = *reinterpret_cast<Shared<Stages>*>(smem_raw);
    const int split     = blockIdx.x;
    const int kv_head   = blockIdx.y;
    const int column    = blockIdx.z;
    const int splits    = gridDim.x;
    const int warp      = threadIdx.x >> 5;
    const int lane      = threadIdx.x & 31;
    const int g         = lane >> 2;
    const int quad      = lane & 3;

    const int p          = positions[column];
    const int n          = (p + 1) / kR;
    const int count      = counts[column];
    const int tail_begin = n * kR;
    const int length     = count * kR + (p + 1 - tail_begin);
    const int tiles      = (length + kTile - 1) / kTile;
    const int tile_begin = split * tiles_per_split;
    const int tile_end   = min(tiles, tile_begin + tiles_per_split);
    const std::int32_t* ids = selected + static_cast<std::int64_t>(column) * kTop;

    // Query rows: Hadamard-prepared, FP16; rows 12..15 are zero.
    for (int r = warp; r < 16; r += kWarps) {
        if (r < kGroup) {
            const bf16* row = q + (static_cast<std::int64_t>(column) * kHq + kv_head * kGroup + r) * kD;
            float values[8];
#pragma unroll
            for (int j = 0; j < 8; ++j) { values[j] = __bfloat162float(row[lane + 32 * j]); }
            normalized_hadamard_d256_inplace(values, lane);
#pragma unroll
            for (int j = 0; j < 8; ++j) { sh.q[r][lane + 32 * j] = __float2half_rn(values[j]); }
        } else {
#pragma unroll
            for (int j = 0; j < 8; ++j) { sh.q[r][lane + 32 * j] = __float2half_rn(0.0F); }
        }
    }
    __syncthreads();

    float o[32][4];
#pragma unroll
    for (int i = 0; i < 32; ++i) {
#pragma unroll
        for (int e = 0; e < 4; ++e) { o[i][e] = 0.0F; }
    }
    float m_lo = -CUDART_INF_F, m_hi = -CUDART_INF_F; // running max of rows g and g + 8 (log2 units)
    float l_lo = 0.0F, l_hi = 0.0F;           // this thread's partial row sums

    Stage* stages = sh.stage[warp];
    int issued    = tile_begin + warp;
#pragma unroll
    for (int s = 0; s < Stages - 1; ++s) {
        if (issued + s * kWarps < tile_end) {
            stage_tile(stages[s], planes, kv_head, ids, count, tail_begin, length,
                       issued + s * kWarps, lane);
        }
        cp_commit();
    }
    int slot = 0;
    for (int tile = tile_begin + warp; tile < tile_end; tile += kWarps) {
        if constexpr (Stages > 1) {
            const int ahead = tile + (Stages - 1) * kWarps;
            if (ahead < tile_end) {
                stage_tile(stages[(slot + Stages - 1) % Stages], planes, kv_head, ids, count,
                           tail_begin, length, ahead, lane);
            }
            cp_commit();
            cp_wait<Stages - 1>();
        } else {
            stage_tile(stages[0], planes, kv_head, ids, count, tail_begin, length, tile, lane);
            cp_commit();
            cp_wait<0>();
        }
        __syncwarp();
        const Stage& st = stages[slot];

        // S = Q K^T over 16 tokens, group by group.
        float s_acc[2][4] = {{0.0F, 0.0F, 0.0F, 0.0F}, {0.0F, 0.0F, 0.0F, 0.0F}};
#pragma unroll
        for (int group = 0; group < kGroups; ++group) {
            float c[2][4] = {{0.0F, 0.0F, 0.0F, 0.0F}, {0.0F, 0.0F, 0.0F, 0.0F}};
#pragma unroll
            for (int step = 0; step < 4; ++step) {
                const int d   = 64 * group + 16 * step + 4 * quad;
                const uint2 lo = *reinterpret_cast<const uint2*>(&sh.q[g][d]);
                const uint2 hi = *reinterpret_cast<const uint2*>(&sh.q[g + 8][d]);
#pragma unroll
                for (int nt = 0; nt < 2; ++nt) {
                    const unsigned word =
                        *reinterpret_cast<const unsigned*>(&st.k[8 * nt + g][d]) ^ 0x80808080U;
                    const unsigned b0 = codes_to_half2(word, 0x4140U);
                    const unsigned b1 = codes_to_half2(word, 0x4342U);
                    mma_f16(c[nt][0], c[nt][1], c[nt][2], c[nt][3], lo.x, hi.x, lo.y, hi.y, b0, b1);
                }
            }
#pragma unroll
            for (int nt = 0; nt < 2; ++nt) {
                const float s0 = __half2float(st.k_scale[8 * nt + 2 * quad][group]);
                const float s1 = __half2float(st.k_scale[8 * nt + 2 * quad + 1][group]);
                s_acc[nt][0] += c[nt][0] * s0;
                s_acc[nt][1] += c[nt][1] * s1;
                s_acc[nt][2] += c[nt][2] * s0;
                s_acc[nt][3] += c[nt][3] * s1;
            }
        }

        // Online softmax in log2 units; tokens past the list are -inf.
        const int token0 = tile * kTile;
        float tile_lo = -CUDART_INF_F, tile_hi = -CUDART_INF_F;
#pragma unroll
        for (int nt = 0; nt < 2; ++nt) {
#pragma unroll
            for (int e = 0; e < 2; ++e) {
                const bool live = token0 + 8 * nt + 2 * quad + e < length;
                s_acc[nt][e]     = live ? s_acc[nt][e] * scale_log2 : -CUDART_INF_F;
                s_acc[nt][e + 2] = live ? s_acc[nt][e + 2] * scale_log2 : -CUDART_INF_F;
                tile_lo          = fmaxf(tile_lo, s_acc[nt][e]);
                tile_hi          = fmaxf(tile_hi, s_acc[nt][e + 2]);
            }
        }
        tile_lo = fmaxf(tile_lo, __shfl_xor_sync(kAll, tile_lo, 1));
        tile_lo = fmaxf(tile_lo, __shfl_xor_sync(kAll, tile_lo, 2));
        tile_hi = fmaxf(tile_hi, __shfl_xor_sync(kAll, tile_hi, 1));
        tile_hi = fmaxf(tile_hi, __shfl_xor_sync(kAll, tile_hi, 2));
        const float new_lo   = fmaxf(m_lo, tile_lo);
        const float new_hi   = fmaxf(m_hi, tile_hi);
        // A tile holds at least one live token, so the new maxima are finite.
        const float alpha_lo = exp2f(m_lo - new_lo);
        const float alpha_hi = exp2f(m_hi - new_hi);
        m_lo                 = new_lo;
        m_hi                 = new_hi;
        float pr[2][4];
#pragma unroll
        for (int nt = 0; nt < 2; ++nt) {
            pr[nt][0] = exp2f(s_acc[nt][0] - new_lo);
            pr[nt][1] = exp2f(s_acc[nt][1] - new_lo);
            pr[nt][2] = exp2f(s_acc[nt][2] - new_hi);
            pr[nt][3] = exp2f(s_acc[nt][3] - new_hi);
        }
        l_lo = l_lo * alpha_lo + (pr[0][0] + pr[0][1]) + (pr[1][0] + pr[1][1]);
        l_hi = l_hi * alpha_hi + (pr[0][2] + pr[0][3]) + (pr[1][2] + pr[1][3]);
#pragma unroll
        for (int i = 0; i < 32; ++i) {
            o[i][0] *= alpha_lo;
            o[i][1] *= alpha_lo;
            o[i][2] *= alpha_hi;
            o[i][3] *= alpha_hi;
        }

        // O += (P x value scale) V: tokens 2q, 2q+1 (n-tile 0) and 2q+8, 2q+9 (n-tile 1).
#pragma unroll
        for (int group = 0; group < kGroups; ++group) {
            const float v0 = __half2float(st.v_scale[2 * quad][group]);
            const float v1 = __half2float(st.v_scale[2 * quad + 1][group]);
            const float v2 = __half2float(st.v_scale[8 + 2 * quad][group]);
            const float v3 = __half2float(st.v_scale[8 + 2 * quad + 1][group]);
            const unsigned a0 = pack_half2(pr[0][0] * v0, pr[0][1] * v1);
            const unsigned a1 = pack_half2(pr[0][2] * v0, pr[0][3] * v1);
            const unsigned a2 = pack_half2(pr[1][0] * v2, pr[1][1] * v3);
            const unsigned a3 = pack_half2(pr[1][2] * v2, pr[1][3] * v3);
#pragma unroll
            for (int chunk = 0; chunk < 2; ++chunk) {
                // ldmatrix.trans of four 8x8 b16 matrices: tokens 0-7 / 8-15 by dims
                // [base, base + 16) and [base + 16, base + 32).
                const int base  = 64 * group + 32 * chunk;
                const int mat   = lane >> 3;
                const int token = (lane & 7) + 8 * (mat & 1);
                const int dim   = base + 16 * (mat >> 1);
                unsigned r0, r1, r2, r3;
                ldmatrix_x4_t(r0, r1, r2, r3, smem_addr(&st.v[token][dim]));
                const unsigned w[4] = {r0 ^ 0x80808080U, r1 ^ 0x80808080U, r2 ^ 0x80808080U,
                                       r3 ^ 0x80808080U};
#pragma unroll
                for (int sub = 0; sub < 2; ++sub) {
#pragma unroll
                    for (int parity = 0; parity < 2; ++parity) {
                        const unsigned selector = parity == 0 ? 0x4240U : 0x4341U;
                        const unsigned b0       = codes_to_half2(w[2 * sub], selector);
                        const unsigned b1       = codes_to_half2(w[2 * sub + 1], selector);
                        float* acc = o[((group * 2 + chunk) * 2 + sub) * 2 + parity];
                        mma_f16(acc[0], acc[1], acc[2], acc[3], a0, a1, a2, a3, b0, b1);
                    }
                }
            }
        }
        __syncwarp();
        slot = (slot + 1) % Stages;
    }
    cp_wait<0>();

    // Row sums across the quad; rows g and g + 8.
    l_lo += __shfl_xor_sync(kAll, l_lo, 1);
    l_lo += __shfl_xor_sync(kAll, l_lo, 2);
    l_hi += __shfl_xor_sync(kAll, l_hi, 1);
    l_hi += __shfl_xor_sync(kAll, l_hi, 2);
    if (quad == 0) {
        sh.row_max[warp][g]     = m_lo;
        sh.row_max[warp][g + 8] = m_hi;
        sh.row_sum[warp][g]     = l_lo;
        sh.row_sum[warp][g + 8] = l_hi;
    }
    __syncthreads(); // every warp done with its stages: the merge area may be reused

    // Rescale each warp's rows to the CTA maximum.
    float cta_lo = -CUDART_INF_F, cta_hi = -CUDART_INF_F, sum_lo = 0.0F, sum_hi = 0.0F;
#pragma unroll
    for (int w = 0; w < kWarps; ++w) {
        cta_lo = fmaxf(cta_lo, sh.row_max[w][g]);
        cta_hi = fmaxf(cta_hi, sh.row_max[w][g + 8]);
    }
#pragma unroll
    for (int w = 0; w < kWarps; ++w) {
        const float flo = cta_lo == -CUDART_INF_F ? 0.0F : exp2f(sh.row_max[w][g] - cta_lo);
        const float fhi = cta_hi == -CUDART_INF_F ? 0.0F : exp2f(sh.row_max[w][g + 8] - cta_hi);
        sum_lo += sh.row_sum[w][g] * flo;
        sum_hi += sh.row_sum[w][g + 8] * fhi;
    }
    {
        const float flo = cta_lo == -CUDART_INF_F ? 0.0F : exp2f(m_lo - cta_lo);
        const float fhi = cta_hi == -CUDART_INF_F ? 0.0F : exp2f(m_hi - cta_hi);
#pragma unroll
        for (int i = 0; i < 32; ++i) {
            o[i][0] *= flo;
            o[i][1] *= flo;
            o[i][2] *= fhi;
            o[i][3] *= fhi;
        }
    }

    // Output element (row, tile i, entry e): dims base + 2 (2 quad + e % 2) + parity.
    const auto dim_of = [&](int i, int e) {
        const int parity = i & 1, sub = (i >> 1) & 1, chunk = (i >> 2) & 1, group = i >> 3;
        return 64 * group + 32 * chunk + 16 * sub + 2 * (2 * quad + (e & 1)) + parity;
    };
    // Two merge rounds: warps 2, 3 deposit, warps 0, 1 add; warp 1 deposits, warp 0 adds.
#pragma unroll
    for (int round = 0; round < 2; ++round) {
        const int depositors = round == 0 ? 2 : 1;
        if (warp >= depositors && warp < 2 * depositors) {
            float(*area)[kD] = sh.merge[warp - depositors];
#pragma unroll
            for (int i = 0; i < 32; ++i) {
#pragma unroll
                for (int e = 0; e < 4; ++e) {
                    const int row = g + 8 * (e >> 1);
                    if (row < kGroup) { area[row][dim_of(i, e)] = o[i][e]; }
                }
            }
        }
        __syncthreads();
        if (warp < depositors) {
            float(*area)[kD] = sh.merge[warp];
#pragma unroll
            for (int i = 0; i < 32; ++i) {
#pragma unroll
                for (int e = 0; e < 4; ++e) {
                    const int row = g + 8 * (e >> 1);
                    if (row < kGroup) { o[i][e] += area[row][dim_of(i, e)]; }
                }
            }
        }
        __syncthreads();
    }
    if (warp != 0) { return; }

    if (splits == 1) {
        const float inv_lo = 1.0F / sum_lo, inv_hi = 1.0F / sum_hi;
#pragma unroll
        for (int i = 0; i < 32; ++i) {
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                const int row = g + 8 * (e >> 1);
                if (row < kGroup) {
                    const float value = o[i][e] * ((e >> 1) == 0 ? inv_lo : inv_hi);
                    out[(static_cast<std::int64_t>(column) * kHq + kv_head * kGroup + row) * kD +
                        dim_of(i, e)] = __float2bfloat16_rn(value);
                }
            }
        }
        return;
    }
    // Partial: [column][split][head] rows of D values, then (max, sum) per row.
    const std::int64_t rows_base =
        (static_cast<std::int64_t>(column) * splits + split) * kHq + kv_head * kGroup;
#pragma unroll
    for (int i = 0; i < 32; ++i) {
#pragma unroll
        for (int e = 0; e < 4; ++e) {
            const int row = g + 8 * (e >> 1);
            if (row < kGroup) { partial[(rows_base + row) * kD + dim_of(i, e)] = o[i][e]; }
        }
    }
    if (quad == 0) {
        const std::int64_t stats_offset =
            static_cast<std::int64_t>(gridDim.z) * splits * kHq * kD;
        if (g < kGroup) {
            partial[stats_offset + 2 * (rows_base + g)]     = cta_lo;
            partial[stats_offset + 2 * (rows_base + g) + 1] = sum_lo;
        }
        if (g + 8 < kGroup) {
            partial[stats_offset + 2 * (rows_base + g + 8)]     = cta_hi;
            partial[stats_offset + 2 * (rows_base + g + 8) + 1] = sum_hi;
        }
    }
}

// One CTA per (head, column), one thread per dimension.
__global__ void __launch_bounds__(kD) merge_kernel(const float* __restrict__ partial, int splits,
                                                    int columns, bf16* __restrict__ out) {
    const int head = blockIdx.x, column = blockIdx.y, d = threadIdx.x;
    const std::int64_t stats_offset = static_cast<std::int64_t>(columns) * splits * kHq * kD;
    float best = -CUDART_INF_F;
    for (int s = 0; s < splits; ++s) {
        const std::int64_t row = (static_cast<std::int64_t>(column) * splits + s) * kHq + head;
        best = fmaxf(best, partial[stats_offset + 2 * row]);
    }
    float sum = 0.0F, value = 0.0F;
    for (int s = 0; s < splits; ++s) {
        const std::int64_t row = (static_cast<std::int64_t>(column) * splits + s) * kHq + head;
        const float m          = partial[stats_offset + 2 * row];
        if (m == -CUDART_INF_F) { continue; } // an empty split
        const float f = exp2f(m - best);
        sum += partial[stats_offset + 2 * row + 1] * f;
        value += partial[row * kD + d] * f;
    }
    out[(static_cast<std::int64_t>(column) * kHq + head) * kD + d] = __float2bfloat16_rn(value / sum);
}

template <int Stages>
void launch_attention(const dim3& grid, const bf16* q, const std::int32_t* positions,
                      const std::int32_t* selected, const std::int32_t* counts,
                      const Planes& planes, float scale_log2, int tiles_per_split, float* partial,
                      bf16* out, cudaStream_t stream) {
    constexpr int kBytes = static_cast<int>(sizeof(Shared<Stages>));
    static const bool reserved = [] {
        return cudaFuncSetAttribute(attention_kernel<Stages>,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    kBytes) == cudaSuccess;
    }();
    if (!reserved) { throw std::runtime_error("qsa attention: cannot reserve shared memory"); }
    attention_kernel<Stages><<<grid, kThreads, kBytes, stream>>>(
        q, positions, selected, counts, planes, scale_log2, tiles_per_split, partial, out);
}

constexpr int kStages = 2;

} // namespace

QsaAttentionPlan qsa_attention_plan(std::int32_t tokens, std::int32_t multiprocessor_count) {
    // Columns x KV heads that already fill the SMs run unsplit; otherwise split the key list so
    // that about two CTAs land on every SM, keeping at least one tile per warp.
    QsaAttentionPlan plan;
    const std::int32_t ctas = tokens * kHkv;
    const std::int32_t max_splits = (kMaxTiles + kWarps - 1) / kWarps;
    if (ctas >= multiprocessor_count) {
        plan.splits = 1;
    } else {
        plan.splits = std::clamp((2 * multiprocessor_count + ctas - 1) / ctas, 1, max_splits);
    }
    plan.tiles_per_split = (kMaxTiles + plan.splits - 1) / plan.splits;
    plan.splits          = (kMaxTiles + plan.tiles_per_split - 1) / plan.tiles_per_split;
    return plan;
}

std::size_t qsa_attention_partial_bytes(std::int32_t tokens, const QsaAttentionPlan& plan) {
    if (plan.splits <= 1) { return 0; }
    const std::size_t rows = static_cast<std::size_t>(tokens) * plan.splits * kHq;
    return rows * (kD + 2) * sizeof(float);
}

void qsa_attention_split_launch(const Tensor& q, const Tensor& positions, const Tensor& selected,
                                const Tensor& counts, float scale, const PagedKVLayerView& cache,
                                const QsaAttentionPlan& plan, float* partial, Tensor& out,
                                cudaStream_t stream) {
    const int tokens = q.ne[2];
    const Planes planes{static_cast<const std::int8_t*>(cache.k_pages.data),
                        static_cast<const std::int8_t*>(cache.v_pages.data),
                        static_cast<const __half*>(cache.k_scale_pages.data),
                        static_cast<const __half*>(cache.v_scale_pages.data),
                        static_cast<const std::int32_t*>(cache.block_table.data)};
    const dim3 grid(static_cast<unsigned>(plan.splits), kHkv, static_cast<unsigned>(tokens));
    launch_attention<kStages>(grid, static_cast<const bf16*>(q.data),
                              static_cast<const std::int32_t*>(positions.data),
                              static_cast<const std::int32_t*>(selected.data),
                              static_cast<const std::int32_t*>(counts.data), planes,
                              scale * kLog2e, plan.tiles_per_split, partial,
                              static_cast<bf16*>(out.data), stream);
    check_launch("attention");
    if (plan.splits > 1) {
        merge_kernel<<<dim3(kHq, tokens), kD, 0, stream>>>(partial, plan.splits, tokens,
                                                          static_cast<bf16*>(out.data));
        check_launch("attention merge");
    }
}

} // namespace ninfer::ops::detail
