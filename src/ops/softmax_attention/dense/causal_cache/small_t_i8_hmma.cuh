#pragma once

// ninfer::ops - split-KV causal small-T attention over the INT8 G64 KV cache on Volta's FP16 tensor
// cores (mma.sync.m8n8k4, FP32 accumulation): the sm_70 route for 1..8 query columns of the 27B
// geometry (which widths and key counts take it: causal_sm70_verify_hmma in launch.h).
//
// The general INT8 partial kernel (small_t_i8.cuh) is written in Ampere m16n8k32.s8 / m16n8k16
// fragments. sm_70 has neither, so there it serves one row tile (two tokens on the 27B) on the CUDA
// cores (dp4a QK, FP32 FMA PV), and a wider verify runs as two-token launches
// (causal_attention_chunk_tokens), each re-reading every cached key and value. At long context that
// was most of a verify round: 18.6 of 46.5 ms per MTP3 round at 42.7K tokens (nsys), the PV FMAs
// alone 6 * T * 256 per key and KV head on the CUDA cores.
//
// This kernel serves the whole width in one launch, so the cache is read once:
//
//   * A CTA owns one (KV head, split) as the general kernel does, writes the same partials (FP32
//     accumulator, m and l in natural-log units) and is merged by the same reducer. Its split
//     capacity is lower (causal_attention_split_capacity): one wave at one CTA per SM.
//   * The CTA's warps form KeyGroups groups of RowWarps warps. Each warp computes 16 query rows --
//     token-major, row = token * GroupSize + local head, as in the general kernel -- of every 16-key
//     tile its group stages. A group copies the next tile into the second of its two shared stages
//     before computing the current one, so no tile is held in registers across the math (holding
//     it there took the kernel to 255 registers with spills and ran ~1.8x slower). Groups take
//     tiles in turn and are merged in shared memory at the end.
//   * Per warp and tile the products follow qsa/attention_wide_hmma.cu, whose measured m8n8k4
//     fragment maps this reuses: QK takes the Hadamard-rotated query in FP16 (each row scaled by a
//     power of two putting its absmax in [2^7, 2^8), removed again in FP32) against the key codes
//     widened exactly to FP16; each 64-wide group is summed by the MMA in FP32 and scaled by the
//     key's group scale in FP32. PV folds the value scale into the probability,
//     A = FP16(p * vscale * 2^u), against the value codes widened exactly; the running output
//     carries 2^u (see the exponent's rule in the loop).
//
// QK with dp4a instead (the query quantized to INT8 per 64-group as the general kernel does, int32
// dots in shared memory, PV unchanged) measured no faster in the same structure (27B, T=4, 164K
// keys: 2.81 vs 2.80 ms; T=8: 5.77 vs 5.49) and keeps the query at ~7 significant bits instead
// of 11, so the FP16 product is the one kept.
//
// Numerics against the general kernel: the FP16 query keeps more bits than its INT8 one, the
// summation order differs, and p * vscale is rounded once to FP16 where the other kernels round p
// to FP16 and dequantize v * vscale to FP16 separately. Close, not bit-identical.

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <math_constants.h>

#include "ops/common/math.cuh"
#include "ops/common/memory.cuh"
#include "ops/common/warp.cuh"
#include "ops/kv_cache/int8_g64_codec.cuh"
#include "ops/softmax_attention/dense/causal_cache/small_t.cuh"

#include <cstdint>

namespace ninfer::ops {

inline constexpr int kVoltaVerifyTile       = 16;
inline constexpr int kVoltaVerifyStageBytes = 2 * kVoltaVerifyTile * kCausalHeadDim + // K, V codes
                                              2 * kVoltaVerifyTile * kKVCacheInt8Groups * 2; // scales
inline constexpr int kVoltaVerifyScratchBytes = (16 * kVoltaVerifyTile + 16) * 4;

template <typename Geometry, int TokenTile, int KeyGroups>
struct VoltaVerifyLayout {
    static constexpr int Rows         = TokenTile * Geometry::GroupSize;
    static constexpr int RowWarps     = (Rows + 15) / 16;
    static constexpr int Br           = RowWarps * 16;
    static constexpr int GroupThreads = RowWarps * 32;
    static constexpr int Threads      = GroupThreads * KeyGroups;
    // Loop phase: [query | query row scales | two stages per group | warp scratch].
    static constexpr int QBytes        = Br * kCausalHeadDim * 2;
    static constexpr int QScaleBytes   = Br * 4;
    static constexpr int StageOffset   = QBytes + QScaleBytes;
    static constexpr int ScratchOffset = StageOffset + KeyGroups * 2 * kVoltaVerifyStageBytes;
    static constexpr int LoopBytes = ScratchOffset + KeyGroups * RowWarps * kVoltaVerifyScratchBytes;
    // Merge phase (reuses everything): [rows Br x D floats | (m, l, u) per group and row].
    static constexpr int MergeBytes = Br * kCausalHeadDim * 4 + KeyGroups * Br * 3 * 4;
    // The fused append's rotation exchange (TokenTile x D floats) lives in the stage region.
    static constexpr int AppendBytes = StageOffset + TokenTile * kCausalHeadDim * 4;
    static constexpr int SmemBytes   = LoopBytes > MergeBytes
                                           ? (LoopBytes > AppendBytes ? LoopBytes : AppendBytes)
                                           : (MergeBytes > AppendBytes ? MergeBytes : AppendBytes);
    static constexpr int StageChunks         = 2 * kVoltaVerifyTile * (kCausalHeadDim / 16);
    static constexpr int StageChunksPerThread = (StageChunks + GroupThreads - 1) / GroupThreads;
    static_assert(GroupThreads >= 2 * kVoltaVerifyTile, "one thread stages each key's scale row");
};

namespace volta_verify {

__device__ __forceinline__ int swizzled(int row, int row_bytes, int chunk) {
    return row * row_bytes + ((chunk ^ (row & 7)) << 4);
}

// Four signed codes -> FP16 pairs (c0, c1) and (c2, c3), exactly (qsa/attention_wide_hmma.cu).
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
__device__ __forceinline__ void hmma(float (&d)[8], unsigned a0, unsigned a1, unsigned b0,
                                     unsigned b1) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
    if constexpr (BRowMajor) {
        asm volatile("mma.sync.aligned.m8n8k4.row.row.f32.f16.f16.f32 "
                     "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, {%0,%1,%2,%3,%4,%5,%6,%7};"
                     : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]),
                       "+f"(d[6]), "+f"(d[7])
                     : "r"(a0), "r"(a1), "r"(b0), "r"(b1));
    } else {
        asm volatile("mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "
                     "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, {%0,%1,%2,%3,%4,%5,%6,%7};"
                     : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]),
                       "+f"(d[6]), "+f"(d[7])
                     : "r"(a0), "r"(a1), "r"(b0), "r"(b1));
    }
#endif
}

__device__ __forceinline__ void group_barrier(int id, int threads) {
    asm volatile("bar.sync %0, %1;" ::"r"(id), "r"(threads) : "memory");
}

} // namespace volta_verify

template <typename Geometry, int TokenTile, int KeyGroups, bool MultiBatch, bool Masked,
          typename CacheInput>
__launch_bounds__(VoltaVerifyLayout<Geometry, TokenTile, KeyGroups>::Threads, 1)
    __global__ void causal_attention_small_t_i8_hmma_kernel(
        const __nv_bfloat16* q, CacheInput input, const std::int32_t* pos, std::int8_t* cache_k_i8,
        std::int8_t* cache_v_i8, __half* cache_k_scale, __half* cache_v_scale,
        const std::int32_t* block_tables, const std::int32_t* valid_columns,
        const std::int32_t* table_rows, std::int32_t table_stride, std::int32_t full_width,
        std::int32_t column_begin, std::int32_t logical_capacity, float scale, float* partial_acc,
        float* partial_m, float* partial_l) {
    using namespace volta_verify;
    using Layout                 = VoltaVerifyLayout<Geometry, TokenTile, KeyGroups>;
    constexpr int D              = kCausalHeadDim;
    constexpr int Groups         = kKVCacheInt8Groups;
    constexpr int Tile           = kVoltaVerifyTile;
    constexpr int Rows           = Layout::Rows;
    constexpr int RowWarps       = Layout::RowWarps;
    constexpr int Br             = Layout::Br;
    constexpr int GroupThreads   = Layout::GroupThreads;
    constexpr int Threads        = Layout::Threads;
    constexpr int Wc             = Threads / 32;
    constexpr int SplitTile      = 32; // the general kernel's split granularity (its Bc)
    constexpr float Log2E        = 1.4426950408889634074f;
    constexpr unsigned FullMask  = 0xffffffffu;
    static_assert(Groups == 4 && D == 256);

    extern __shared__ __align__(16) unsigned char smem[];

    const int kv_head     = static_cast<int>(blockIdx.x);
    const int split       = static_cast<int>(blockIdx.y);
    const int batch       = MultiBatch ? static_cast<int>(blockIdx.z) : 0;
    const int split_count = static_cast<int>(gridDim.y);
    const int tid         = static_cast<int>(threadIdx.x);
    const int warp        = tid >> 5;
    const int lane        = tid & 31;

    int valid_tokens = TokenTile;
    if constexpr (Masked) {
        const int remaining = valid_columns[batch] - column_begin;
        valid_tokens        = remaining <= 0 ? 0 : (remaining < TokenTile ? remaining : TokenTile);
    }
    std::int64_t column_base = column_begin;
    if constexpr (MultiBatch) { column_base += static_cast<std::int64_t>(batch) * full_width; }
    q += static_cast<std::int64_t>(D) * Geometry::QHeads * column_base;
    pos += column_base;
    if constexpr (CacheInput::writes_cache) {
        input.k += static_cast<std::int64_t>(D) * Geometry::KVHeads * column_base;
        input.v += static_cast<std::int64_t>(D) * Geometry::KVHeads * column_base;
    }
    const int table_row = table_rows == nullptr ? 0 : table_rows[batch];
    const std::int32_t* block_table =
        block_tables + static_cast<std::int64_t>(table_row) * table_stride;
    if constexpr (MultiBatch) {
        partial_acc +=
            static_cast<std::int64_t>(batch) * D * Geometry::QHeads * TokenTile * split_count;
        partial_m += static_cast<std::int64_t>(batch) * Geometry::QHeads * TokenTile * split_count;
        partial_l += static_cast<std::int64_t>(batch) * Geometry::QHeads * TokenTile * split_count;
    }

    auto write_neutral = [&]() {
        for (int row = tid; row < Rows; row += Threads) {
            int q_head = 0, token = 0;
            causal_small_t_tc_row_to_qt<Geometry>(row, TokenTile, kv_head, q_head, token);
            partial_m[causal_partial_stat_index<Geometry>(q_head, token, split, TokenTile)] =
                -CUDART_INF_F;
            partial_l[causal_partial_stat_index<Geometry>(q_head, token, split, TokenTile)] = 0.0f;
        }
        for (int idx = tid; idx < Rows * D; idx += Threads) {
            const int row = idx / D;
            const int d   = idx - row * D;
            int q_head = 0, token = 0;
            causal_small_t_tc_row_to_qt<Geometry>(row, TokenTile, kv_head, q_head, token);
            partial_acc[causal_partial_acc_index<Geometry>(q_head, d, token, split, TokenTile)] =
                0.0f;
        }
    };

    if (kv_head < 0 || kv_head >= Geometry::KVHeads || split_count <= 0) { return; }
    if (valid_tokens == 0) {
        write_neutral();
        return;
    }
    const std::int32_t first_pos = pos[0];
    const std::int32_t last_pos  = pos[TokenTile - 1];
    if (first_pos < 0 || last_pos < 0 || last_pos >= logical_capacity) {
        write_neutral();
        return;
    }

    // Split boundaries exactly as the general kernel draws them (multiples of its 32-key tile when
    // the window spans at least one tile per split).
    const int window = last_pos + 1;
    const int active_split_count =
        causal_small_t_active_splits<Geometry, true>(window, split_count, TokenTile);
    if (split >= active_split_count) { return; }
    const int logical_tiles = div_up(window, SplitTile);
    const bool tile_split   = logical_tiles >= active_split_count;
    const int units_per_split =
        tile_split ? div_up(logical_tiles, active_split_count) : div_up(window, active_split_count);
    const int split_start = split * units_per_split * (tile_split ? SplitTile : 1);
    const int split_limit = split_start + units_per_split * (tile_split ? SplitTile : 1);
    const int split_end   = (split_limit < window) ? split_limit : window;
    if (split_start >= split_end) {
        write_neutral();
        return;
    }
    const int first_key = (split_start / Tile) * Tile;
    const int tiles     = div_up(split_end - first_key, Tile);

    if constexpr (CacheInput::writes_cache) {
        // The general kernel's fused append (INT8 G64 keys and values): the new tokens of this
        // split's key range are rotated, quantized and written first, then read back below.
        float* k_h64_s = reinterpret_cast<float*>(smem + Layout::StageOffset);
        for (int pair = warp; pair < valid_tokens * Groups; pair += Wc) {
            const int token    = pair / Groups;
            const int grp      = pair - token * Groups;
            const int position = pos[token];
            if (position < split_start || position >= split_end) { continue; }
            const int d0            = grp * kKVCacheInt8Group + lane;
            const int d1            = d0 + 32;
            const std::int64_t src0 = kv_cache_int8_new_index<Geometry>(kv_head, d0, token);
            const std::int64_t src1 = kv_cache_int8_new_index<Geometry>(kv_head, d1, token);
            float k_h64[2] = {__bfloat162float(input.k[src0]), __bfloat162float(input.k[src1])};
            hadamard_d64_fragment_inplace(k_h64, lane);
            k_h64_s[token * D + d0] = k_h64[0];
            k_h64_s[token * D + d1] = k_h64[1];
        }
        __syncthreads();
        for (int pair = warp; pair < valid_tokens * Groups; pair += Wc) {
            const int token    = pair / Groups;
            const int grp      = pair - token * Groups;
            const int position = pos[token];
            if (position < split_start || position >= split_end) { continue; }
            const int physical_page = block_table[position >> kPagedKVPageShift];
            const int page_offset   = position & kPagedKVPageMask;
            const int d0            = grp * kKVCacheInt8Group + lane;
            const int d1            = d0 + 32;
            const std::int64_t src0 = kv_cache_int8_new_index<Geometry>(kv_head, d0, token);
            const std::int64_t src1 = kv_cache_int8_new_index<Geometry>(kv_head, d1, token);
            float k_out[2];
#pragma unroll
            for (int half = 0; half < 2; ++half) {
                const int dh   = lane + half * 32;
                const float x0 = k_h64_s[token * D + dh];
                const float x1 = k_h64_s[token * D + kKVCacheInt8Group + dh];
                const float x2 = k_h64_s[token * D + 2 * kKVCacheInt8Group + dh];
                const float x3 = k_h64_s[token * D + 3 * kKVCacheInt8Group + dh];
                k_out[half]    = normalized_hadamard_d256_group_value_from_h64(x0, x1, x2, x3, grp);
            }
            kv_cache_i8_family_store_key_group<Geometry, false>(cache_k_i8, cache_k_scale,
                                                                physical_page, kv_head, grp,
                                                                page_offset, lane, k_out[0],
                                                                k_out[1]);
            const float vv0    = __bfloat162float(input.v[src0]);
            const float vv1    = __bfloat162float(input.v[src1]);
            const auto v_quant = kv_cache_int8_quant_params(warp_max(fmaxf(fabsf(vv0), fabsf(vv1)), FullMask));
            cache_v_i8[kv_cache_int8_quant_code_index<Geometry>(physical_page, kv_head, d0,
                                                                page_offset)] =
                kv_cache_int8_quant_code(vv0, v_quant.inverse_scale);
            cache_v_i8[kv_cache_int8_quant_code_index<Geometry>(physical_page, kv_head, d1,
                                                                page_offset)] =
                kv_cache_int8_quant_code(vv1, v_quant.inverse_scale);
            if (lane == 0) {
                cache_v_scale[kv_cache_int8_quant_scale_index<Geometry>(physical_page, kv_head, grp,
                                                                        page_offset)] =
                    v_quant.scale;
            }
        }
        __syncthreads();
    }

    // The query, rotated as the cache's keys are: [Br][D] halves, 16-byte chunks swizzled, one
    // power-of-two scale per row.
    unsigned char* q_s = smem;
    float* q_scale     = reinterpret_cast<float*>(smem + Layout::QBytes);
    for (int i = tid; i < Layout::QBytes / 16; i += Threads) {
        reinterpret_cast<int4*>(q_s)[i] = make_int4(0, 0, 0, 0);
    }
    for (int i = tid; i < Br; i += Threads) { q_scale[i] = 0.0f; }
    __syncthreads();
    for (int row = warp; row < Rows; row += Wc) {
        int q_head = 0, token = 0;
        causal_small_t_tc_row_to_qt<Geometry>(row, TokenTile, kv_head, q_head, token);
        float v[8];
#pragma unroll
        for (int r = 0; r < 8; ++r) {
            v[r] = __bfloat162float(q[causal_q_index<Geometry>(q_head, lane + 32 * r, token)]);
        }
        normalized_hadamard_d256_inplace(v, lane);
        float amax = 0.0f;
#pragma unroll
        for (int r = 0; r < 8; ++r) { amax = fmaxf(amax, fabsf(v[r])); }
        amax      = warp_max(amax, FullMask);
        float mul = 0.0f;
        if (amax > 0.0f && isfinite(amax)) {
            int x = 0;
            (void)frexpf(amax, &x);
            mul = ldexpf(1.0f, 8 - x);
            if (lane == 0) { q_scale[row] = ldexpf(1.0f, x - 8); }
        }
#pragma unroll
        for (int r = 0; r < 8; ++r) {
            const int d = lane + 32 * r;
            *reinterpret_cast<__half*>(q_s + swizzled(row, 2 * D, d >> 3) + 2 * (d & 7)) =
                __float2half_rn(v[r] * mul);
        }
    }
    __syncthreads();

    const int key_group  = warp / RowWarps;
    const int row_warp   = warp - key_group * RowWarps;
    const int group_tid  = tid - key_group * GroupThreads;
    unsigned char* stage_base =
        smem + Layout::StageOffset + key_group * 2 * kVoltaVerifyStageBytes;
    float* sp            = reinterpret_cast<float*>(smem + Layout::ScratchOffset +
                                                    warp * kVoltaVerifyScratchBytes);
    float* alpha_s       = sp + 16 * Tile;

    // A tile's K and V codes (16-byte chunks) and scale rows: loaded, then stored to a stage.
    constexpr int ChunksPerThread = Layout::StageChunksPerThread;
    int4 staged[ChunksPerThread];
    uint2 staged_scale = make_uint2(0u, 0u);
    const auto load_tile = [&](int t) {
        const int k0 = first_key + t * Tile;
#pragma unroll
        for (int i = 0; i < ChunksPerThread; ++i) {
            const int chunk = group_tid + i * GroupThreads;
            staged[i]       = make_int4(0, 0, 0, 0);
            if (chunk < Layout::StageChunks) {
                const int plane = chunk / (Tile * (D / 16));
                const int key_l = (chunk / (D / 16)) % Tile;
                const int c     = chunk % (D / 16);
                const int key   = k0 + key_l;
                if (key >= split_start && key < split_end) {
                    const int page = block_table[key >> kPagedKVPageShift];
                    const std::int64_t off = kv_cache_int8_quant_code_index<Geometry>(
                        page, kv_head, c * 16, key & kPagedKVPageMask);
                    // Plain loads: the fused append above may have written these bytes.
                    staged[i] = *reinterpret_cast<const int4*>((plane ? cache_v_i8 : cache_k_i8) + off);
                }
            }
        }
        staged_scale = make_uint2(0u, 0u);
        if (group_tid < 2 * Tile) {
            const int plane = group_tid / Tile;
            const int key   = k0 + (group_tid % Tile);
            if (key >= split_start && key < split_end) {
                const int page = block_table[key >> kPagedKVPageShift];
                const std::int64_t off = kv_cache_int8_quant_scale_index<Geometry>(
                    page, kv_head, 0, key & kPagedKVPageMask);
                staged_scale =
                    *reinterpret_cast<const uint2*>((plane ? cache_v_scale : cache_k_scale) + off);
            }
        }
    };
    const auto store_tile = [&](unsigned char* stage) {
#pragma unroll
        for (int i = 0; i < ChunksPerThread; ++i) {
            const int chunk = group_tid + i * GroupThreads;
            if (chunk < Layout::StageChunks) {
                const int plane = chunk / (Tile * (D / 16));
                const int key_l = (chunk / (D / 16)) % Tile;
                const int c     = chunk % (D / 16);
                *reinterpret_cast<int4*>(stage + plane * Tile * D + swizzled(key_l, D, c)) = staged[i];
            }
        }
        if (group_tid < 2 * Tile) {
            *reinterpret_cast<uint2*>(stage + 2 * Tile * D + group_tid * Groups * 2) = staged_scale;
        }
    };

    // Fragment coordinates (qsa/attention_wide_hmma.cu).
    const int qp = (lane >> 2) & 3, hi = (lane >> 4) & 1, p = (lane & 3) + 4 * hi;
    const int mq = qp & 1, half = qp >> 1;
    const int row_a = mq * 8 + ((lane & 1) | 4 * hi);
    const int col_a = 2 * ((lane >> 1) & 1);
    const int row_base = row_warp * 16;

    // Softmax state: lane l < 16 owns row row_base + l.
    const int own_row = row_base + (lane & 15);
    int own_qabs      = -1;
    if (lane < 16 && own_row < Rows) { own_qabs = pos[own_row / Geometry::GroupSize]; }
    const float own_qscale = q_scale[own_row];

    float o[16][8];
#pragma unroll
    for (int b = 0; b < 16; ++b) {
#pragma unroll
        for (int i = 0; i < 8; ++i) { o[b][i] = 0.0f; }
    }
    float m_run = -CUDART_INF_F, l_run = 0.0f;
    int u_cur   = 0;
    bool u_set  = false;

    const int group_tiles = tiles > key_group ? div_up(tiles - key_group, KeyGroups) : 0;
    const int barrier_id  = 1 + key_group;
    if (group_tiles > 0) {
        load_tile(key_group);
        store_tile(stage_base);
    }
    for (int it = 0; it < group_tiles; ++it) {
        const int t                = key_group + it * KeyGroups;
        const int k0               = first_key + t * Tile;
        const unsigned char* stage = stage_base + (it & 1) * kVoltaVerifyStageBytes;
        // This tile's stage is complete, and the other one was consumed by the previous tile.
        group_barrier(barrier_id, GroupThreads);
        if (it + 1 < group_tiles) {
            load_tile(t + KeyGroups);
            store_tile(stage_base + ((it + 1) & 1) * kVoltaVerifyStageBytes);
        }
        const __half* ks = reinterpret_cast<const __half*>(stage + 2 * Tile * D);
        const __half* vs = ks + Tile * Groups;

        float sc[Tile];
        {
            // QK: rows mq*8.., keys half*8..; this lane supplies query row mq*8+p, key half*8+p.
            float score[8];
#pragma unroll
            for (int e = 0; e < 8; ++e) { score[e] = 0.0f; }
            const int key_b = half * 8 + p;
            const int q_row = row_base + mq * 8 + p;
#pragma unroll
            for (int g = 0; g < Groups; ++g) {
                float acc[8];
#pragma unroll
                for (int e = 0; e < 8; ++e) { acc[e] = 0.0f; }
#pragma unroll
                for (int c4 = 0; c4 < 4; ++c4) {
                    const int ch   = 4 * g + c4;
                    const uint4 kw = *reinterpret_cast<const uint4*>(stage + swizzled(key_b, D, ch));
                    const uint4 q0 = *reinterpret_cast<const uint4*>(q_s + swizzled(q_row, 2 * D, 2 * ch));
                    const uint4 q1 = *reinterpret_cast<const uint4*>(q_s + swizzled(q_row, 2 * D, 2 * ch + 1));
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
                    score[e]      = __fmaf_rn(__half2float(ks[key * Groups + g]), acc[e], score[e]);
                }
            }
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                sp[(row_a + (e & 2)) * Tile + half * 8 + col_a + (e & 1) + 4 * (e >> 2)] = score[e];
            }
            __syncwarp();
            if (lane < 16) {
#pragma unroll
                for (int k = 0; k < Tile; ++k) { sc[k] = sp[lane * Tile + k] * own_qscale; }
            }
        }

        // The value-scale exponent u: set from the first tile with values so its largest scale x 2^u
        // lands in [2^7, 2^8), and moved again (to that position) only when a tile's largest
        // scale x 2^u leaves [2^0, 2^15): A = p * vscale * 2^u then stays a normal FP16 below
        // 2^15 for p <= 1, and the running output is rarely rescaled for u.
        float own_v;
        {
            const __half2 pair = load_vec<__half2>(&vs[2 * lane]);
            own_v              = fmaxf(fabsf(__low2float(pair)), fabsf(__high2float(pair)));
        }
        const float vmax = warp_max(own_v, FullMask);
        int u            = u_cur;
        if (vmax > 0.0f && isfinite(vmax)) {
            int x = 0;
            (void)frexpf(vmax, &x); // vmax in [2^(x-1), 2^x)
            if (!u_set || x - 1 + u_cur < 0 || x + u_cur > 15) { u = 8 - x; }
            u_set = true;
        }
        if (lane < 16) {
            float bm = -CUDART_INF_F;
#pragma unroll
            for (int k = 0; k < Tile; ++k) {
                const int key = k0 + k;
                const bool ok = own_row < Rows && key >= split_start && key < split_end &&
                                key <= own_qabs;
                sc[k]         = ok ? sc[k] * scale : -CUDART_INF_F;
                bm            = fmaxf(bm, sc[k]);
            }
            const float nm    = fmaxf(m_run, bm);
            // A row with no keys yet has a zero output and l, so any alpha serves: 1 keeps the
            // rescale skippable (padding rows never get keys).
            const float alpha = m_run == -CUDART_INF_F ? 1.0f : exp2_approx((m_run - nm) * Log2E);
            float bl          = 0.0f;
            float* row        = sp + lane * Tile;
#pragma unroll
            for (int k = 0; k < Tile; ++k) {
                const float pk = sc[k] == -CUDART_INF_F ? 0.0f : exp2_approx((sc[k] - nm) * Log2E);
                bl += pk;
                row[k] = pk;
            }
            l_run         = __fmaf_rn(l_run, alpha, bl);
            m_run         = nm;
            // o holds sum(p * v * 2^u_cur); in this tile's units it is o * 2^(u - u_cur).
            alpha_s[lane] = alpha * ldexpf(1.0f, u - u_cur);
        }
        u_cur = u;
        __syncwarp();

        // PV: rows mq*8.., dims half*128..; rescale, then A = p * vscale * 2^u for the two groups.
        const float al_a = alpha_s[row_a], al_b = alpha_s[row_a + 2];
        // Once a row's maximum settles, alpha is exactly 1 (ex2(0)) and u rarely moves: skip the
        // rescale when every row of the warp keeps its scale.
        if (__any_sync(FullMask, al_a != 1.0f || al_b != 1.0f)) {
#pragma unroll
            for (int b = 0; b < 16; ++b) {
#pragma unroll
                for (int e = 0; e < 8; ++e) { o[b][e] *= (e & 2) ? al_b : al_a; }
            }
        }
        const float up = ldexpf(1.0f, u);
        unsigned a[2][4][2];
        const float* prow = sp + (mq * 8 + p) * Tile;
#pragma unroll
        for (int gl = 0; gl < 2; ++gl) {
            const int g = 2 * half + gl;
#pragma unroll
            for (int st = 0; st < 4; ++st) {
                float pv[4];
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const int key = 4 * st + j;
                    pv[j]         = prow[key] * __half2float(vs[key * Groups + g]) * up;
                }
                a[gl][st][0] = pack_f16x2(pv[0], pv[1]);
                a[gl][st][1] = pack_f16x2(pv[2], pv[3]);
            }
        }
        const unsigned char* v_s = stage + Tile * D;
#pragma unroll
        for (int b = 0; b < 16; ++b) {
            const int gl   = b >> 3;
            const int dim0 = half * 128 + b * 8 + 4 * hi;
#pragma unroll
            for (int st = 0; st < 4; ++st) {
                const int key    = 4 * st + (lane & 3);
                const unsigned w = *reinterpret_cast<const unsigned*>(
                    v_s + swizzled(key, D, dim0 >> 4) + (dim0 & 15));
                unsigned lo, hv;
                widen_codes(w, lo, hv);
                hmma<true>(o[b], a[gl][st][0], a[gl][st][1], lo, hv);
            }
        }
        __syncwarp();
    }

    // Merge the key groups: rows[Br][D] = sum_g exp(m_g - M) 2^-u_g o_g, then the partials.
    __syncthreads();
    float* rows  = reinterpret_cast<float*>(smem);
    float* stats = rows + Br * D;
    if (lane < 16) {
        float* s = stats + (key_group * Br + own_row) * 3;
        s[0]     = m_run;
        s[1]     = l_run;
        s[2]     = static_cast<float>(u_cur);
    }
    __syncthreads();
    const auto row_max = [&](int row) {
        float m = -CUDART_INF_F;
#pragma unroll
        for (int g = 0; g < KeyGroups; ++g) { m = fmaxf(m, stats[(g * Br + row) * 3]); }
        return m;
    };
    {
        const int ra   = row_base + row_a;
        const int rb   = ra + 2;
        const float ma = stats[(key_group * Br + ra) * 3];
        const float mb = stats[(key_group * Br + rb) * 3];
        const float down = ldexpf(1.0f, -u_cur);
        const float fa = ma == -CUDART_INF_F ? 0.0f : exp2_approx((ma - row_max(ra)) * Log2E) * down;
        const float fb = mb == -CUDART_INF_F ? 0.0f : exp2_approx((mb - row_max(rb)) * Log2E) * down;
        for (int g = 0; g < KeyGroups; ++g) {
            if (key_group == g) {
#pragma unroll
                for (int b = 0; b < 16; ++b) {
#pragma unroll
                    for (int e = 0; e < 8; ++e) {
                        const int row = row_base + row_a + (e & 2);
                        const int d   = half * 128 + b * 8 + col_a + (e & 1) + 4 * (e >> 2);
                        const float f = (e & 2) ? fb : fa;
                        float& acc    = rows[row * D + d];
                        acc           = g == 0 ? f * o[b][e] : __fmaf_rn(f, o[b][e], acc);
                    }
                }
            }
            __syncthreads();
        }
    }
    for (int row = tid; row < Rows; row += Threads) {
        const float m = row_max(row);
        float l       = 0.0f;
#pragma unroll
        for (int g = 0; g < KeyGroups; ++g) {
            const float mg = stats[(g * Br + row) * 3];
            if (mg == -CUDART_INF_F) { continue; }
            l = __fmaf_rn(exp2_approx((mg - m) * Log2E), stats[(g * Br + row) * 3 + 1], l);
        }
        int q_head = 0, token = 0;
        causal_small_t_tc_row_to_qt<Geometry>(row, TokenTile, kv_head, q_head, token);
        partial_m[causal_partial_stat_index<Geometry>(q_head, token, split, TokenTile)] = m;
        partial_l[causal_partial_stat_index<Geometry>(q_head, token, split, TokenTile)] = l;
    }
    for (int idx = tid; idx < Rows * D; idx += Threads) {
        const int row = idx / D;
        const int d   = idx - row * D;
        int q_head = 0, token = 0;
        causal_small_t_tc_row_to_qt<Geometry>(row, TokenTile, kv_head, q_head, token);
        partial_acc[causal_partial_acc_index<Geometry>(q_head, d, token, split, TokenTile)] =
            rows[row * D + d];
    }
}

} // namespace ninfer::ops
