#pragma once

// FlashAttention-2 style causal prompt kernel for the INT8 cache family (INT8-G64, rk8v4, rk4v4)
// on sm_80/86/89.
//
// Adapted from Infernix's fast INT8 prompt kernel and its key-split merge
// (github.com/wallawalla47/infernix, commits 60e67d06 and dbd1f374,
// src/ops/softmax_attention/dense/causal_cache/int8/fast_tiled_mma.cuh and fast_prompt_common.cuh),
// Copyright the Infernix authors, licensed under the Apache License, Version 2.0. Changes for this
// fork: the 8-bit PV form is dropped, the split merge reads the batch metadata instead of a raw
// valid-column pointer, the FP16 MMA comes from ops/common/mma.cuh, launch planning lives in
// prompt_i8_fa2_plan.h with constants measured on the RTX 3090, Q is encoded by the whole CTA,
// and the rk8v4 packed INT4 values and rk4v4 Lloyd-Max keys are decoded in-kernel.
//
//   * Each warp owns 16 query rows of one head for the whole key sweep (eight warps per CTA, or
//     four for launches too narrow to occupy every SM with 128-row CTAs), so scores,
//     probabilities and the D256 output accumulator never leave registers and no warp waits for
//     another between QK and PV. The only barrier in the key sweep is the one that publishes a
//     key tile.
//   * Q and cached K use the same fixed register-only D256 rotation before their private G64
//     encoders; QK stays INT8 through m16n8k32.s8 Tensor Cores with the per-group scales applied
//     in FP32. Every warp of the CTA encodes Q rows, so a launch narrower than its CTA encodes at
//     most a few rows per warp.
//   * K and V codes are double-buffered one 64-key page per tile. INT8 V is decoded in registers:
//     ldmatrix.trans on byte pairs yields, per lane, two adjacent keys for two adjacent dimensions,
//     which is exactly one FP16 B fragment for an even-dimension and an odd-dimension n8 tile.
//     Packed INT4 V (rk8v4, rk4v4) is decoded the same way from nibble quads: one b16 lane holds
//     two adjacent keys for four adjacent dimensions of one 32-value scale group, the B fragments
//     of four n8 tiles. Codes widen exactly to FP16 and take their represented FP16 group scale
//     with one rounding, as the dequantizing kernels do.
//   * rk4v4 keys are 4-bit Lloyd-Max indices; the CTA expands each tile once into the INT8 K tile
//     the QK path reads. Each thread loads its share of the next tile's packed keys into registers
//     where the other planes start their copies, holds them across this tile's QK and PV, and
//     expands them into the next stage before the barrier that publishes it. Measured against
//     staging the packed keys through shared memory with cp.async (RTX 3090, 2026-10-10, both
//     geometries), this was 1.0-1.4 % faster on 4096-token chunks and 6-8 % on H24 first chunks,
//     4-5 % slower only at H16 257-320 columns over 4K-32K keys, spill-free in the eight-warp
//     instances, and 16 KB smaller.
//   * PV runs FP16 Tensor Cores with FP16 accumulation over the 64 keys of a tile and is promoted
//     into the FP32 output accumulator once per tile. Every probability is at most one, so a tile
//     partial is bounded by 64 * max|code| * max_scale; a tile whose largest V scale could take
//     that bound past the FP16 range decodes V with its scales divided by an exact power of two
//     and multiplies the partial back at promotion.
//   * CTAs issue longest-first, so a causal prompt's heaviest row blocks do not form the tail.
//   * A split launch (gridDim.z > 1, eight-warp CTAs only) gives each CTA a contiguous run of key
//     pages; each CTA publishes its normalized FP32 rows and their (max, sum), and
//     causal_attention_prompt_i8_fa2_merge_kernel combines the splits in FP32.
//
// This kernel replaced the sixteen-warp tiled prompt kernel (FP32 PV accumulation, P staged through
// shared memory) for every INT8-family launch. Measured against it on the RTX 3090 (2026-10-10,
// both geometries, rk4v4 and rk8v4): 0.54-0.64x on 4096-token chunks, 0.12-0.81x on follow-ups of
// up to 256 tokens over 32K-184K cached keys, 0.65-1.03x on 7-64-column launches over at most 4K
// keys; slower only for 65-320-column launches over at most about 512 keys (rk4v4 up to 1.31x, at
// most 11 us per layer; rk8v4 up to 1.07x).

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <math_constants.h>

#include "ops/common/math.h"
#include "ops/kv_cache/int8_g64_codec.cuh"
#include "ops/softmax_attention/dense/causal_cache/prompt_common.cuh"

#include <cstdint>

namespace ninfer::ops {

inline constexpr int kCausalPromptFa2Bc     = 64;
inline constexpr int kCausalPromptFa2Groups = kCausalPromptHeadDim / kKVCacheInt8Group;

// One stage of the double buffer holds one page of one KV head: the INT8 K tile the QK path reads
// (copied, or expanded from rk4v4's packed keys), the V codes and both scale rows.
template <bool PackedValues, bool PackedKeys>
struct CausalPromptFa2Stage {
    static_assert(PackedValues || !PackedKeys, "rk4v4 pairs packed keys with packed values");
    static constexpr int KBytes       = kCausalPromptFa2Bc * kCausalPromptHeadDim;
    static constexpr int VBytes       = PackedValues ? KBytes / 2 : KBytes;
    static constexpr int VGroups      = PackedValues ? kKVCacheInt4ValueGroups : kKVCacheInt8Groups;
    static constexpr int KScaleBytes  = kCausalPromptFa2Bc * kCausalPromptFa2Groups * 2;
    static constexpr int VScaleBytes  = kCausalPromptFa2Bc * VGroups * 2;
    static constexpr int VOffset      = KBytes;
    static constexpr int KScaleOffset = VOffset + VBytes;
    static constexpr int VScaleOffset = KScaleOffset + KScaleBytes;
    static constexpr int Bytes        = VScaleOffset + VScaleBytes;
    static_assert(Bytes % 16 == 0);
};

template <int Warps, bool PackedValues, bool PackedKeys>
struct CausalPromptFa2Shape {
    static_assert(Warps == 4 || Warps == 8);
    using Stage                    = CausalPromptFa2Stage<PackedValues, PackedKeys>;
    static constexpr int Threads   = Warps * 32;
    static constexpr int Br        = Warps * 16;
    static constexpr int QBytes    = Br * kCausalPromptHeadDim;
    static constexpr int SmemBytes = QBytes + 2 * Stage::Bytes;
    // The CTA's FP32 Q group scales pass through the stage the prologue does not fill.
    static_assert(Br * kCausalPromptFa2Groups * 4 <= Stage::KBytes);
};

// A 64-key FP16 partial is bounded by 64 * max|code| * max_scale (every probability is at most
// one). INT8 codes reach 127, so a scale of at most 8 leaves that bound at 65,024; packed INT4
// codes decode to at most 8 in magnitude, so a scale of at most 127 leaves it at 65,024. Either
// keeps FP16 rounding slack under the largest finite FP16 value 65,504.
template <bool PackedValues>
inline constexpr float kCausalPromptFa2F16PartialScaleLimit = PackedValues ? 127.0f : 8.0f;

static_assert(kCausalPromptFa2Bc == kPagedKVPageSize);
static_assert(kCausalPromptFa2Groups == 4);
// sm_86 allows 101,376 B of dynamic shared memory per block: one CTA per SM every shape.
static_assert(CausalPromptFa2Shape<8, false, false>::SmemBytes == 100352);
static_assert(CausalPromptFa2Shape<4, false, false>::SmemBytes == 83968);
static_assert(CausalPromptFa2Shape<8, true, false>::SmemBytes == 84992);
static_assert(CausalPromptFa2Shape<8, true, true>::SmemBytes == 84992);

// One ldmatrix.trans b16 lane of an INT8 [key][d] tile holds the codes
// {V[k][d], V[k][d+1], V[k+1][d], V[k+1][d+1]}. Returns the FP16 B-fragment halves
// {V[k][d], V[k+1][d]} and {V[k][d+1], V[k+1][d+1]}, each code widened exactly and multiplied
// once by its key's represented group scale.
__device__ __forceinline__ void causal_prompt_fa2_decode_v_pair(unsigned codes, unsigned scales,
                                                                unsigned& even, unsigned& odd) {
    // code ^ 0x80 is code + 128; 0x6400 | byte is the FP16 value 1024 + byte, so subtracting 1152
    // recovers the signed code exactly.
    const unsigned biased = codes ^ 0x80808080u;
    unsigned e            = __byte_perm(biased, 0x64646464u, 0x4240);
    unsigned o            = __byte_perm(biased, 0x64646464u, 0x4341);
    const __half2 offset  = __float2half2_rn(1152.0f);
    const __half2 s2      = load_vec<__half2>(&scales);
    const __half2 ve      = __hmul2(__hsub2(load_vec<__half2>(&e), offset), s2);
    const __half2 vo      = __hmul2(__hsub2(load_vec<__half2>(&o), offset), s2);
    even                  = load_vec<unsigned>(&ve);
    odd                   = load_vec<unsigned>(&vo);
}

// One ldmatrix.trans b16 lane of a packed INT4 [key][d/2] tile holds the nibbles of dimensions
// d..d+3 for key k (low half) and key k+1 (high half), even dimension in the low nibble. Returns
// the FP16 B-fragment halves {V[k][d+n], V[k+1][d+n]} for n = 0..3, each code widened exactly and
// multiplied once by its key's represented group scale.
__device__ __forceinline__ void causal_prompt_fa2_decode_v_quad(unsigned codes, unsigned scales,
                                                                unsigned (&out)[4]) {
    // A two's-complement nibble x ^ 8 is code + 8; 0x6400 | (code + 8) is the FP16 value
    // 1024 + code + 8, so subtracting 1032 recovers the signed code exactly.
    const __half2 offset = __float2half2_rn(1032.0f);
    const __half2 s2     = load_vec<__half2>(&scales);
#pragma unroll
    for (int n = 0; n < 4; ++n) {
        const unsigned biased = ((codes >> (4 * n)) & 0x000F000Fu) ^ 0x64086408u;
        const __half2 v       = __hmul2(__hsub2(load_vec<__half2>(&biased), offset), s2);
        out[n]                = load_vec<unsigned>(&v);
    }
}

// Split partial layout for key split s, column c and query head h: the D256 row at
// ((s * width + c) * QHeads + h) * 256 and its (max, sum) pair at that row index.
template <typename Geometry>
__host__ __device__ __forceinline__ std::int64_t causal_prompt_fa2_partial_row(int split,
                                                                               int column,
                                                                               int q_head,
                                                                               int width) {
    return (static_cast<std::int64_t>(split) * width + column) * Geometry::QHeads + q_head;
}

// PackedValues selects the packed INT4 G32 value plane (rk8v4, rk4v4); PackedKeys selects rk4v4's
// Lloyd-Max key plane. cache_k and cache_v are the raw planes either way.
template <typename Geometry, typename Metadata, int Warps, bool Split, bool PackedValues,
          bool PackedKeys>
__global__ __launch_bounds__(CausalPromptFa2Shape<Warps, PackedValues, PackedKeys>::Threads, 1) void
causal_attention_prompt_i8_fa2_kernel(const __nv_bfloat16* __restrict__ q,
                                      const std::int8_t* __restrict__ cache_k,
                                      const std::int8_t* __restrict__ cache_v,
                                      const __half* __restrict__ cache_k_scale,
                                      const __half* __restrict__ cache_v_scale, Metadata metadata,
                                      const std::int32_t* __restrict__ positions, float scale,
                                      __nv_bfloat16* __restrict__ out, std::int32_t width,
                                      float* __restrict__ partial_rows,
                                      float2* __restrict__ partial_stats) {
    constexpr int D             = kCausalPromptHeadDim;
    constexpr int DB16          = D / 2;
    using Shape                 = CausalPromptFa2Shape<Warps, PackedValues, PackedKeys>;
    using Stage                 = typename Shape::Stage;
    constexpr int Threads       = Shape::Threads;
    constexpr int Br            = Shape::Br;
    constexpr int Bc            = kCausalPromptFa2Bc;
    constexpr int Groups        = kCausalPromptFa2Groups;
    constexpr int VGroups       = Stage::VGroups;
    constexpr int GroupKc       = kKVCacheInt8Group / 32;
    constexpr int QKNt          = Bc / 8;
    constexpr int PVKs          = Bc / 16;
    constexpr int GroupDBlocks  = kKVCacheInt8Group / 16;
    constexpr int PassDBlocks   = 2;
    // The FP32 accumulator holds 128 values per lane either way: INT8 V as 16 d-blocks of an
    // even and an odd n8 tile, packed V as 8 value groups of four n8 tiles.
    constexpr int AccBlocks     = PackedValues ? VGroups : D / 16;
    constexpr int AccTiles      = PackedValues ? 4 : 2;
    constexpr int PackedRow     = D / 2;
    constexpr float Log2E       = 1.4426950408889634074f;
    constexpr unsigned FullMask = 0xffffffffu;

    static_assert(GroupKc == 2);
    static_assert(GroupDBlocks == 4);
    static_assert(AccBlocks * AccTiles == 32);
    static_assert(!Split || Warps == 8, "only eight-warp CTAs split");

    extern __shared__ __align__(16) unsigned char smem_raw[];
    std::int8_t* q_i8    = reinterpret_cast<std::int8_t*>(smem_raw);
    __nv_bfloat16* q_b16 = reinterpret_cast<__nv_bfloat16*>(q_i8);

    const int tid  = static_cast<int>(threadIdx.x);
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int gid  = lane >> 2;
    const int lid  = lane & 3;

    // Longest-first issue order: the first QHeads CTAs take the last row block of every head.
    const int row_blocks = static_cast<int>(gridDim.x);
    const int linear     = static_cast<int>(blockIdx.x + blockIdx.y * gridDim.x);
    const int q_head     = linear % Geometry::QHeads;
    const int q0         = (row_blocks - 1 - linear / Geometry::QHeads) * Br;
    const int kv_head    = q_head / Geometry::GroupSize;
    const int tokens     = metadata.valid_tokens(width);

    // Rows past the valid token count are inactive columns and publish zeros.
    const auto store_row = [&](int row, int d0, float v0, float v1, float v2, float v3) {
        const int token = q0 + row;
        if (token >= width) { return; }
        const bool valid   = token < tokens;
        const uint2 packed = make_uint2(pack_bf16x2(valid ? v0 : 0.0f, valid ? v1 : 0.0f),
                                        pack_bf16x2(valid ? v2 : 0.0f, valid ? v3 : 0.0f));
        store_vec(&out[causal_prompt_q_index<Geometry>(q_head, d0, token)], packed);
    };

    if (q0 >= tokens) {
        // The merge publishes a split launch's inactive columns.
        if constexpr (!Split) {
            for (int element = tid; element < Br * (D / 4); element += Threads) {
                const int row = element / (D / 4);
                store_row(row, (element - row * (D / 4)) * 4, 0.0f, 0.0f, 0.0f, 0.0f);
            }
        }
        return;
    }

    const int base_pos              = positions[0];
    const std::int32_t* block_table = metadata.block_table();
    const int rows                  = min(Br, tokens - q0);
    const int max_query_abs         = base_pos + q0 + rows - 1;
    const int key_blocks            = max_query_abs / Bc + 1;
    // Key pages [kb_begin, kb_end) of this split; an empty run publishes neutral statistics.
    const int split           = Split ? static_cast<int>(blockIdx.z) : 0;
    const int pages_per_split = Split ? div_up(key_blocks, static_cast<int>(gridDim.z)) : 0;
    const int kb_begin        = Split ? min(key_blocks, split * pages_per_split) : 0;
    const int kb_end          = Split ? min(key_blocks, kb_begin + pages_per_split) : key_blocks;

    const auto stage_base = [&](int stage) {
        return smem_raw + Shape::QBytes + stage * Stage::Bytes;
    };

    // rk4v4: the next tile's packed keys, 16 bytes (32 dimensions of one key) per chunk, held in
    // registers from the tile's issue until its expansion.
    constexpr int PackedKChunks = PackedKeys ? Stage::KBytes / 2 / 16 / Threads : 1;
    uint4 k_packed[PackedKChunks];

    // One tile is one physical page of this KV head; every plane of a page is contiguous in the
    // cache. Keys past the CTA's last visible key are zero-filled so masked columns stay finite
    // (an expanded zero Lloyd-Max index is not a zero code, but its zero-filled scale is).
    const auto issue_tile = [&](int kb) {
        unsigned char* base = stage_base(kb & 1);
        const int page      = block_table[kb];
        const int valid     = min(Bc, max_query_abs + 1 - kb * Bc);
        if constexpr (PackedKeys) {
            // Both packed planes are 128 bytes per key, so one offset addresses either.
            const std::int64_t packed_base =
                kv_cache_int4_value_code_index<Geometry>(page, kv_head, 0, 0);
#pragma unroll
            for (int i = 0; i < PackedKChunks; ++i) {
                const int chunk = tid + i * Threads;
                k_packed[i]     = make_uint4(0u, 0u, 0u, 0u);
                if ((chunk >> 3) < valid) {
                    const auto* src =
                        reinterpret_cast<const std::uint8_t*>(cache_k) + packed_base + chunk * 16;
                    asm volatile("ld.global.nc.v4.u32 {%0, %1, %2, %3}, [%4];\n"
                                 : "=r"(k_packed[i].x), "=r"(k_packed[i].y), "=r"(k_packed[i].z),
                                   "=r"(k_packed[i].w)
                                 : "l"(src));
                }
            }
        } else {
            const std::int64_t code_base =
                kv_cache_int8_quant_code_index<Geometry>(page, kv_head, 0, 0);
#pragma unroll
            for (int i = 0; i < Stage::KBytes / 16 / Threads; ++i) {
                const int chunk = tid + i * Threads;
                const int key   = chunk >> 4;
                const int c     = chunk & 15;
                cp_async_zfill<16, Cache::cg>(base + key * D + ((c ^ (key & 7)) << 4),
                                              cache_k + code_base + key * D + c * 16,
                                              key < valid ? 16 : 0);
            }
        }
        if constexpr (PackedValues) {
            // A packed row is eight 16-byte chunks, one per 32-value scale group.
            const std::int64_t packed_base =
                kv_cache_int4_value_code_index<Geometry>(page, kv_head, 0, 0);
#pragma unroll
            for (int i = 0; i < Stage::VBytes / 16 / Threads; ++i) {
                const int chunk = tid + i * Threads;
                const int key   = chunk >> 3;
                const int c     = chunk & 7;
                cp_async_zfill<16, Cache::cg>(
                    base + Stage::VOffset + key * PackedRow + ((c ^ (key & 7)) << 4),
                    reinterpret_cast<const std::uint8_t*>(cache_v) + packed_base + chunk * 16,
                    key < valid ? 16 : 0);
            }
        } else {
            const std::int64_t code_base =
                kv_cache_int8_quant_code_index<Geometry>(page, kv_head, 0, 0);
#pragma unroll
            for (int i = 0; i < Stage::VBytes / 16 / Threads; ++i) {
                const int chunk = tid + i * Threads;
                const int key   = chunk >> 4;
                const int c     = chunk & 15;
                cp_async_zfill<16, Cache::cg>(base + Stage::VOffset + key * D +
                                                  ((c ^ (key & 7)) << 4),
                                              cache_v + code_base + key * D + c * 16,
                                              key < valid ? 16 : 0);
            }
        }
        if (tid < Bc) {
            const std::int64_t src = kv_cache_int8_quant_scale_index<Geometry>(page, kv_head, 0, tid);
            cp_async_zfill<8>(base + Stage::KScaleOffset + tid * Groups * 2, cache_k_scale + src,
                              tid < valid ? 8 : 0);
        } else if (tid < 2 * Bc) {
            const int key = tid - Bc;
            if constexpr (PackedValues) {
                const std::int64_t src =
                    kv_cache_int4_value_scale_index<Geometry>(page, kv_head, 0, key);
                cp_async_zfill<16>(base + Stage::VScaleOffset + key * VGroups * 2,
                                   cache_v_scale + src, key < valid ? 16 : 0);
            } else {
                const std::int64_t src =
                    kv_cache_int8_quant_scale_index<Geometry>(page, kv_head, 0, key);
                cp_async_zfill<8>(base + Stage::VScaleOffset + key * VGroups * 2,
                                  cache_v_scale + src, key < valid ? 8 : 0);
            }
        }
        cp_commit();
    };

    // rk4v4: each thread expands the packed key chunks it loaded for tile kb into the INT8 K tile,
    // in the swizzled layout the INT8 copy uses. The next barrier publishes them with the rest of
    // the tile.
    const auto expand_keys = [&](int kb) {
        if constexpr (PackedKeys) {
            unsigned char* base = stage_base(kb & 1);
#pragma unroll
            for (int i = 0; i < PackedKChunks; ++i) {
                const int chunk     = tid + i * Threads;
                const int key       = chunk >> 3;
                const int c         = (chunk & 7) * 2;
                const uint4 packed  = k_packed[i];
                store_vec(base + key * D + ((c ^ (key & 7)) << 4),
                          kv_cache_lloyd4_expand16(make_uint2(packed.x, packed.y)));
                store_vec(base + key * D + (((c + 1) ^ (key & 7)) << 4),
                          kv_cache_lloyd4_expand16(make_uint2(packed.z, packed.w)));
            }
        }
    };

    // The first tile's copy overlaps the Q encoding below; the first loop barrier publishes both.
    if (kb_begin < kb_end) { issue_tile(kb_begin); }

    // The whole CTA rotates and encodes the Q rows of its active warps, QRows rows per warp at a
    // time: one row is a chain of dependent shuffles (the rotation's five lane stages and four
    // group maxima) that interleaving independent rows overlaps. Spreading the rows over every
    // warp keeps a launch narrower than its CTA from encoding sixteen rows in one warp. The loop
    // stays rolled and four rows wide: unrolled, or eight rows wide, the once-per-CTA prologue's
    // larger code, fetched cold by every CTA, made one-tile launches slower (16 columns: 15.8 us
    // at four rows, 31.1 us at eight). The group scales pass through the stage the first tile does
    // not fill.
    constexpr int QRows = 4;
    float* q_scale_s    = reinterpret_cast<float*>(stage_base((kb_begin + 1) & 1));
    const int active_q_groups = div_up(rows, 16) * (16 / QRows);
#pragma unroll 1
    for (int group = warp; group < active_q_groups; group += Warps) {
        const int r0 = group * QRows;
        float q_values[8 * QRows];
#pragma unroll
        for (int i = 0; i < QRows; ++i) {
            const int row    = r0 + i;
            const bool valid = row < rows;
#pragma unroll
            for (int k = 0; k < 8; ++k) {
                q_values[8 * i + k] =
                    valid ? __bfloat162float(
                                q[causal_prompt_q_index<Geometry>(q_head, lane + 32 * k, q0 + row)])
                          : 0.0f;
            }
        }
        normalized_hadamard_d256_rows_inplace<QRows>(q_values, lane);
        float mx[QRows * Groups];
#pragma unroll
        for (int j = 0; j < QRows * Groups; ++j) {
            mx[j] = fmaxf(fabsf(q_values[2 * j]), fabsf(q_values[2 * j + 1]));
        }
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
#pragma unroll
            for (int j = 0; j < QRows * Groups; ++j) {
                mx[j] = fmaxf(mx[j], __shfl_xor_sync(FullMask, mx[j], offset));
            }
        }
#pragma unroll
        for (int i = 0; i < QRows; ++i) {
            const int row = r0 + i;
#pragma unroll
            for (int grp = 0; grp < Groups; ++grp) {
                const int d0    = grp * kKVCacheInt8Group + lane;
                const float x0  = q_values[8 * i + 2 * grp];
                const float x1  = q_values[8 * i + 2 * grp + 1];
                const float m   = mx[i * Groups + grp];
                const float qs  = m > 0.0f ? m / 127.0f : 0.0f;
                const float inv = qs > 0.0f ? 1.0f / qs : 0.0f;
                causal_prompt_store_byte_swizzled(q_i8, row, d0,
                                                  kv_cache_int8_quant_code(x0, inv));
                causal_prompt_store_byte_swizzled(q_i8, row, d0 + 32,
                                                  kv_cache_int8_quant_code(x1, inv));
                if (lane == 0) { q_scale_s[row * Groups + grp] = qs; }
            }
        }
    }
    if (kb_begin < kb_end) { expand_keys(kb_begin); }
    __syncthreads();

    const int a_mat    = lane >> 3;
    const int a_rin    = lane & 7;
    const int a_rowoff = a_rin + ((a_mat & 1) << 3);
    const int a_coloff = (a_mat >> 1) << 3;
    const int b_koff   = ((lane >> 3) & 1) << 3;
    const int row_base = warp * 16;

    // Rows this warp owns, for warp-uniform tile skipping and mask elision.
    const bool warp_active  = row_base < rows;
    const int warp_min_qabs = base_pos + q0 + row_base;
    const int warp_max_qabs = base_pos + q0 + min(row_base + 15, rows - 1);
    const int row0          = row_base + gid;
    const int row1          = row0 + 8;
    const int qabs0         = row0 < rows ? base_pos + q0 + row0 : -1;
    const int qabs1         = row1 < rows ? base_pos + q0 + row1 : -1;

    // The two rows each lane owns in the MMA C layout keep their group scales in registers; the
    // first loop barrier orders these reads before the next tile's copy reuses the stage.
    float q_scale_r[2][Groups];
#pragma unroll
    for (int grp = 0; grp < Groups; ++grp) {
        q_scale_r[0][grp] = warp_active ? q_scale_s[row0 * Groups + grp] : 0.0f;
        q_scale_r[1][grp] = warp_active ? q_scale_s[row1 * Groups + grp] : 0.0f;
    }

    float acc[AccBlocks][AccTiles][4];
#pragma unroll
    for (int b = 0; b < AccBlocks; ++b) {
#pragma unroll
        for (int p = 0; p < AccTiles; ++p) {
#pragma unroll
            for (int i = 0; i < 4; ++i) { acc[b][p][i] = 0.0f; }
        }
    }
    float running_m0     = -CUDART_INF_F;
    float running_m1     = -CUDART_INF_F;
    float running_l0     = 0.0f;
    float running_l1     = 0.0f;
    const float scale_l2 = scale * Log2E;

    // What QK hands to PV for one tile: the probability A fragments, the row rescale factors, and
    // the exact power of two that keeps the tile's FP16 partials representable.
    unsigned pa[PVKs][4];
    float tile_alpha0 = 0.0f;
    float tile_alpha1 = 0.0f;
    int tile_shift    = 0;
    bool tile_live    = false;

    const auto qk_softmax = [&](int kb) {
        tile_live    = false;
        const int k0 = kb * Bc;
        if (!warp_active || k0 > warp_max_qabs) { return; }
        const unsigned char* base  = stage_base(kb & 1);
        const __nv_bfloat16* k_b16 = reinterpret_cast<const __nv_bfloat16*>(base);
        const __half* ks_s         = reinterpret_cast<const __half*>(base + Stage::KScaleOffset);
        const __half* vs_s         = reinterpret_cast<const __half*>(base + Stage::VScaleOffset);

        float score[QKNt][4];
#pragma unroll
        for (int nt = 0; nt < QKNt; ++nt) {
            score[nt][0] = score[nt][1] = score[nt][2] = score[nt][3] = 0.0f;
        }
#pragma unroll
        for (int grp = 0; grp < Groups; ++grp) {
            unsigned af[GroupKc][4];
#pragma unroll
            for (int kk = 0; kk < GroupKc; ++kk) {
                const int acol = (grp * GroupKc + kk) * 16 + a_coloff;
                ldmatrix_x4(af[kk][0], af[kk][1], af[kk][2], af[kk][3],
                            smem_addr(&q_b16[(row_base + a_rowoff) * DB16 +
                                             causal_prompt_swz(row_base + a_rowoff, acol)]));
            }
#pragma unroll
            for (int nt = 0; nt < QKNt; ++nt) {
                int c0 = 0, c1 = 0, c2 = 0, c3 = 0;
#pragma unroll
                for (int kk = 0; kk < GroupKc; ++kk) {
                    const int brow = nt * 8 + a_rin;
                    const int bcol = (grp * GroupKc + kk) * 16 + b_koff;
                    unsigned bf[2];
                    ldmatrix_x2(bf[0], bf[1],
                                smem_addr(&k_b16[brow * DB16 + causal_prompt_swz(brow, bcol)]));
                    mma_s8(c0, c1, c2, c3, af[kk][0], af[kk][1], af[kk][2], af[kk][3], bf[0],
                           bf[1]);
                }
                const int keya  = nt * 8 + 2 * lid;
                const float ks0 = __half2float(ks_s[keya * Groups + grp]);
                const float ks1 = __half2float(ks_s[(keya + 1) * Groups + grp]);
                const float qs0 = q_scale_r[0][grp];
                const float qs1 = q_scale_r[1][grp];
                score[nt][0]    = __fmaf_rn(qs0 * ks0, static_cast<float>(c0), score[nt][0]);
                score[nt][1]    = __fmaf_rn(qs0 * ks1, static_cast<float>(c1), score[nt][1]);
                score[nt][2]    = __fmaf_rn(qs1 * ks0, static_cast<float>(c2), score[nt][2]);
                score[nt][3]    = __fmaf_rn(qs1 * ks1, static_cast<float>(c3), score[nt][3]);
            }
        }

        // A tile wholly at or below the warp's first row needs no causal mask.
        const bool full_tile = k0 + Bc - 1 <= warp_min_qabs;
        float bm0            = -CUDART_INF_F;
        float bm1            = -CUDART_INF_F;
#pragma unroll
        for (int nt = 0; nt < QKNt; ++nt) {
            if (!full_tile) {
                const int key0 = k0 + nt * 8 + 2 * lid;
                const int key1 = key0 + 1;
                score[nt][0]   = key0 <= qabs0 ? score[nt][0] : -CUDART_INF_F;
                score[nt][1]   = key1 <= qabs0 ? score[nt][1] : -CUDART_INF_F;
                score[nt][2]   = key0 <= qabs1 ? score[nt][2] : -CUDART_INF_F;
                score[nt][3]   = key1 <= qabs1 ? score[nt][3] : -CUDART_INF_F;
            }
            bm0 = fmaxf(bm0, fmaxf(score[nt][0], score[nt][1]));
            bm1 = fmaxf(bm1, fmaxf(score[nt][2], score[nt][3]));
        }
        bm0 = warp_max<4>(bm0, FullMask);
        bm1 = warp_max<4>(bm1, FullMask);

        // A row with no visible key yet (an inactive row, or a split run wholly past this row's
        // position) keeps max -inf; exp2(-inf - 0) is then an exact zero probability.
        const float nm0        = fmaxf(running_m0, bm0);
        const float nm1        = fmaxf(running_m1, bm1);
        const float nm0_scaled = nm0 == -CUDART_INF_F ? 0.0f : nm0 * scale_l2;
        const float nm1_scaled = nm1 == -CUDART_INF_F ? 0.0f : nm1 * scale_l2;
        tile_alpha0            = running_m0 == -CUDART_INF_F
                                     ? 0.0f
                                     : exp2_approx(__fmaf_rn(running_m0, scale_l2, -nm0_scaled));
        tile_alpha1            = running_m1 == -CUDART_INF_F
                                     ? 0.0f
                                     : exp2_approx(__fmaf_rn(running_m1, scale_l2, -nm1_scaled));
        running_m0             = nm0;
        running_m1             = nm1;

        // Probabilities become the PV A fragments directly: k-step j covers keys 16j..16j+15,
        // which are score tiles 2j (keys 2t, 2t+1) and 2j+1 (keys 8+2t, 9+2t).
        float bl0 = 0.0f;
        float bl1 = 0.0f;
#pragma unroll
        for (int nt = 0; nt < QKNt; ++nt) {
            const float p00 = exp2_approx(__fmaf_rn(score[nt][0], scale_l2, -nm0_scaled));
            const float p01 = exp2_approx(__fmaf_rn(score[nt][1], scale_l2, -nm0_scaled));
            const float p10 = exp2_approx(__fmaf_rn(score[nt][2], scale_l2, -nm1_scaled));
            const float p11 = exp2_approx(__fmaf_rn(score[nt][3], scale_l2, -nm1_scaled));
            bl0 += p00 + p01;
            bl1 += p10 + p11;
            pa[nt >> 1][(nt & 1) * 2 + 0] = pack_f16x2(p00, p01);
            pa[nt >> 1][(nt & 1) * 2 + 1] = pack_f16x2(p10, p11);
        }
        // Row sums stay lane-partial; the quad is reduced once after the sweep.
        running_l0 = __fmaf_rn(running_l0, tile_alpha0, bl0);
        running_l1 = __fmaf_rn(running_l1, tile_alpha1, bl1);

        // A tile whose largest V scale exceeds the limit decodes V with its scales divided by an
        // exact power of two and multiplies the partial back at promotion.
        constexpr int ScaleWords = Bc * VGroups / 32 / 8;
        static_assert(ScaleWords == 1 || ScaleWords == 2);
        __half2 vmax2 = __float2half2_rn(0.0f);
#pragma unroll
        for (int w = 0; w < ScaleWords; ++w) {
            const uint4 v8 = load_vec<uint4>(&vs_s[8 * (lane + 32 * w)]);
            vmax2          = __hmax2(
                vmax2,
                __hmax2(__hmax2(__habs2(load_vec<__half2>(&v8.x)), __habs2(load_vec<__half2>(&v8.y))),
                        __hmax2(__habs2(load_vec<__half2>(&v8.z)),
                                __habs2(load_vec<__half2>(&v8.w)))));
        }
        const float vmax = warp_max(fmaxf(__low2float(vmax2), __high2float(vmax2)), FullMask);
        tile_shift       = 0;
        if (vmax > kCausalPromptFa2F16PartialScaleLimit<PackedValues>) {
            // An infinite or NaN scale would never halve below the limit; the largest finite FP16
            // value takes at most 13 halvings.
            const float bounded = fminf(vmax, 65504.0f);
            while (ldexpf(bounded, -tile_shift) > kCausalPromptFa2F16PartialScaleLimit<PackedValues>) {
                ++tile_shift;
            }
        }
        tile_live = true;
    };

    // Promotes one n8 tile's FP16 partial (two packed rows) into its FP32 accumulator.
    const auto promote = [&](float (&a)[4], const unsigned (&h)[2], float unscale) {
        float2 r0 = __half22float2(load_vec<__half2>(&h[0]));
        float2 r1 = __half22float2(load_vec<__half2>(&h[1]));
        if (tile_shift != 0) {
            r0.x *= unscale;
            r0.y *= unscale;
            r1.x *= unscale;
            r1.y *= unscale;
        }
        a[0] = __fmaf_rn(a[0], tile_alpha0, r0.x);
        a[1] = __fmaf_rn(a[1], tile_alpha0, r0.y);
        a[2] = __fmaf_rn(a[2], tile_alpha1, r1.x);
        a[3] = __fmaf_rn(a[3], tile_alpha1, r1.y);
    };

    const auto pv = [&](int kb) {
        if (!tile_live) { return; }
        const unsigned char* base = stage_base(kb & 1);
        const unsigned char* v_s  = base + Stage::VOffset;
        const __half* vs_s        = reinterpret_cast<const __half*>(base + Stage::VScaleOffset);
        const __half2 mul         = __float2half2_rn(ldexpf(1.0f, -tile_shift));
        const float unscale       = ldexpf(1.0f, tile_shift);
        // Group scales of the keys each lane supplies: (2t, 2t+1) and (8+2t, 9+2t) per k-step.
        const auto load_scales = [&](int grp, unsigned (&vsc)[PVKs][2]) {
#pragma unroll
            for (int j = 0; j < PVKs; ++j) {
                const int key = j * 16 + 2 * lid;
                __half2 lo    = __halves2half2(vs_s[key * VGroups + grp],
                                               vs_s[(key + 1) * VGroups + grp]);
                __half2 hi    = __halves2half2(vs_s[(key + 8) * VGroups + grp],
                                               vs_s[(key + 9) * VGroups + grp]);
                if (tile_shift != 0) {
                    lo = __hmul2(lo, mul);
                    hi = __hmul2(hi, mul);
                }
                vsc[j][0] = load_vec<unsigned>(&lo);
                vsc[j][1] = load_vec<unsigned>(&hi);
            }
        };
        if constexpr (PackedValues) {
            // One 32-value scale group per pass: four n8 tiles, dimensions 32g + 4c + n for
            // ldmatrix column c and tile n. Each x4 load covers two k-steps.
#pragma unroll
            for (int grp = 0; grp < VGroups; ++grp) {
                unsigned vsc[PVKs][2];
                load_scales(grp, vsc);
                unsigned h[4][2];
#pragma unroll
                for (int n = 0; n < 4; ++n) { h[n][0] = h[n][1] = 0u; }
#pragma unroll
                for (int jj = 0; jj < PVKs / 2; ++jj) {
                    // Matrices: keys 32jj + 0..7, 8..15, 16..23, 24..31 of group grp.
                    const int key = jj * 32 + a_mat * 8 + a_rin;
                    unsigned r[4];
                    ldmatrix_x4_t(r[0], r[1], r[2], r[3],
                                  smem_addr(&v_s[key * PackedRow + ((grp ^ (key & 7)) << 4)]));
#pragma unroll
                    for (int s = 0; s < 2; ++s) {
                        const int j = 2 * jj + s;
                        unsigned lo[4];
                        unsigned hi[4];
                        causal_prompt_fa2_decode_v_quad(r[2 * s], vsc[j][0], lo);
                        causal_prompt_fa2_decode_v_quad(r[2 * s + 1], vsc[j][1], hi);
#pragma unroll
                        for (int n = 0; n < 4; ++n) {
                            mma_f16_acc16(h[n][0], h[n][1], pa[j][0], pa[j][1], pa[j][2], pa[j][3],
                                          lo[n], hi[n]);
                        }
                    }
                }
#pragma unroll
                for (int n = 0; n < 4; ++n) { promote(acc[grp][n], h[n], unscale); }
            }
        } else {
#pragma unroll
            for (int grp = 0; grp < Groups; ++grp) {
                unsigned vsc[PVKs][2];
                load_scales(grp, vsc);
#pragma unroll
                for (int pass = 0; pass < GroupDBlocks / PassDBlocks; ++pass) {
                    const int db0 = grp * GroupDBlocks + pass * PassDBlocks;
                    unsigned h[PassDBlocks][2][2];
#pragma unroll
                    for (int b = 0; b < PassDBlocks; ++b) {
#pragma unroll
                        for (int p = 0; p < 2; ++p) { h[b][p][0] = h[b][p][1] = 0u; }
                    }
#pragma unroll
                    for (int j = 0; j < PVKs; ++j) {
#pragma unroll
                        for (int q2 = 0; q2 < PassDBlocks / 2; ++q2) {
                            // Matrices: keys 16j+0..7 and 16j+8..15 of d-block db, then of db + 1.
                            const int db    = db0 + 2 * q2;
                            const int key   = j * 16 + ((a_mat & 1) << 3) + a_rin;
                            const int chunk = db + (a_mat >> 1);
                            unsigned r[4];
                            ldmatrix_x4_t(r[0], r[1], r[2], r[3],
                                          smem_addr(&v_s[key * D + ((chunk ^ (key & 7)) << 4)]));
#pragma unroll
                            for (int b = 0; b < 2; ++b) {
                                unsigned even_lo, odd_lo, even_hi, odd_hi;
                                causal_prompt_fa2_decode_v_pair(r[2 * b], vsc[j][0], even_lo,
                                                                odd_lo);
                                causal_prompt_fa2_decode_v_pair(r[2 * b + 1], vsc[j][1], even_hi,
                                                                odd_hi);
                                unsigned(&he)[2] = h[2 * q2 + b][0];
                                unsigned(&ho)[2] = h[2 * q2 + b][1];
                                mma_f16_acc16(he[0], he[1], pa[j][0], pa[j][1], pa[j][2],
                                              pa[j][3], even_lo, even_hi);
                                mma_f16_acc16(ho[0], ho[1], pa[j][0], pa[j][1], pa[j][2],
                                              pa[j][3], odd_lo, odd_hi);
                            }
                        }
                    }
#pragma unroll
                    for (int b = 0; b < PassDBlocks; ++b) {
#pragma unroll
                        for (int p = 0; p < 2; ++p) { promote(acc[db0 + b][p], h[b][p], unscale); }
                    }
                }
            }
        }
    };

    // Every warp publishes one tile per barrier; the next tile's copy overlaps this tile's math.
#pragma unroll 1
    for (int kb = kb_begin; kb < kb_end; ++kb) {
        cp_wait<0>();
        __syncthreads();
        const bool has_next = kb + 1 < kb_end;
        if (has_next) { issue_tile(kb + 1); }
        qk_softmax(kb);
        pv(kb);
        if (has_next) { expand_keys(kb + 1); }
    }

    running_l0         = warp_sum<4>(running_l0, FullMask);
    running_l1         = warp_sum<4>(running_l1, FullMask);
    const float inv_l0 = running_l0 > 0.0f ? __frcp_rn(running_l0) : 0.0f;
    const float inv_l1 = running_l1 > 0.0f ? __frcp_rn(running_l1) : 0.0f;
    // INT8 V: even n8 tiles hold dimensions 16b + 2n and odd tiles 16b + 2n + 1, so lane (g, t)
    // owns the four contiguous dimensions 16b + 4t .. 16b + 4t + 3 of its two rows. Packed V: tile
    // n of group b holds dimensions 32b + 4c + n, so lane (g, t) owns the eight contiguous
    // dimensions 32b + 8t .. 32b + 8t + 7 (C columns 2t and 2t + 1 over the four tiles).
    const auto block_values = [&](int b, int e, float inv, int half) {
        if constexpr (PackedValues) {
            return make_float4(acc[b][0][e + half] * inv, acc[b][1][e + half] * inv,
                               acc[b][2][e + half] * inv, acc[b][3][e + half] * inv);
        } else {
            return make_float4(acc[b][0][e] * inv, acc[b][1][e] * inv, acc[b][0][e + 1] * inv,
                               acc[b][1][e + 1] * inv);
        }
    };
    constexpr int BlockDims  = PackedValues ? 32 : 16;
    constexpr int LaneDims   = PackedValues ? 8 : 4;
    constexpr int LaneHalves = LaneDims / 4;
    if constexpr (Split) {
        // Each split row is normalized by its own sum; the merge weights the splits.
        const auto publish = [&](int row, int e, float maximum, float sum, float inv) {
            if (row >= rows) { return; }
            const std::int64_t index =
                causal_prompt_fa2_partial_row<Geometry>(split, q0 + row, q_head, width);
            if (lid == 0) { partial_stats[index] = make_float2(maximum, sum); }
            float* target = partial_rows + index * D;
#pragma unroll
            for (int b = 0; b < AccBlocks; ++b) {
#pragma unroll
                for (int half = 0; half < LaneHalves; ++half) {
                    store_vec(target + b * BlockDims + LaneDims * lid + 4 * half,
                              block_values(b, e, inv, half));
                }
            }
        };
        publish(row0, 0, running_m0, running_l0, inv_l0);
        publish(row1, 2, running_m1, running_l1, inv_l1);
    } else {
#pragma unroll
        for (int b = 0; b < AccBlocks; ++b) {
#pragma unroll
            for (int half = 0; half < LaneHalves; ++half) {
                const int d0     = b * BlockDims + LaneDims * lid + 4 * half;
                const float4 v0 = block_values(b, 0, inv_l0, half);
                const float4 v1 = block_values(b, 2, inv_l1, half);
                store_row(row0, d0, v0.x, v0.y, v0.z, v0.w);
                store_row(row1, d0, v1.x, v1.y, v1.z, v1.w);
            }
        }
    }
}

// Combines the key splits of one column and query head in FP32. Every split row is normalized by
// its own sum, so the merged row is sum_s w_s * row_s / sum_s w_s with
// w_s = sum_s * 2^((m_s - M) * scale * log2 e). Columns past the valid count publish zeros.
template <typename Geometry, typename Metadata>
__global__ __launch_bounds__(kCausalPromptHeadDim) void causal_attention_prompt_i8_fa2_merge_kernel(
    const float* __restrict__ partial_rows, const float2* __restrict__ partial_stats,
    Metadata metadata, std::int32_t width, std::int32_t splits, float scale_l2,
    __nv_bfloat16* __restrict__ out) {
    const int column = static_cast<int>(blockIdx.x);
    const int q_head = static_cast<int>(blockIdx.y);
    const int d      = static_cast<int>(threadIdx.x);
    const int tokens = metadata.valid_tokens(width);
    const auto index = causal_prompt_q_index<Geometry>(q_head, d, column);
    if (column >= tokens) {
        out[index] = __float2bfloat16(0.0f);
        return;
    }
    float maximum = -CUDART_INF_F;
    for (int split = 0; split < splits; ++split) {
        const float2 stats =
            partial_stats[causal_prompt_fa2_partial_row<Geometry>(split, column, q_head, width)];
        if (stats.y > 0.0f) { maximum = fmaxf(maximum, stats.x); }
    }
    float numerator   = 0.0f;
    float denominator = 0.0f;
    for (int split = 0; split < splits; ++split) {
        const std::int64_t row =
            causal_prompt_fa2_partial_row<Geometry>(split, column, q_head, width);
        const float2 stats = partial_stats[row];
        if (stats.y > 0.0f) {
            const float weight = stats.y * exp2f((stats.x - maximum) * scale_l2);
            numerator = __fmaf_rn(weight, partial_rows[row * kCausalPromptHeadDim + d], numerator);
            denominator += weight;
        }
    }
    out[index] = __float2bfloat16(denominator > 0.0f ? numerator / denominator : 0.0f);
}

} // namespace ninfer::ops
