// Adapted from Basalt 8d73a39 src/kernels/cuda/dense_tc.cu (MIT, third_party/basalt/LICENSE).
#pragma once

// The int8 tensor-core decode route of the GGML linears (T <= 8, LinearPolicy::AllowA8GgmlDecodeMma):
// a 16-row by 8-column tile per step, one mma.sync.m16n8k32.s8 per 32-value sub-block (two m16n8k16
// for Q6_K's 16-value halves) for the exact integer dots, the scale terms in the fixed order of
// ggml_mma_emulation.h (canonical_ggml.h's lane partials and butterfly), so a column's bits do not
// depend on T.
//
// Basalt's structure: persistent CTAs take tiles b, b + G, ...; a tile's rows stream through a ring
// of S = 3 shared-memory slots, one 256-value chunk (all 16 rows) per slot, by 16-byte cp.async from
// the 16-byte unit holding each row's first byte, issued S - 1 chunks ahead across tile boundaries;
// rows sit at an odd number of 16-byte units so the rows a load touches fall in different banks;
// IQ3_S is decoded from shared memory (grid and staged codes) with Basalt's branchless sign mask.
//
// Modified for NInfer: sm_86 (no PDL, no bulk copies); the activation is cast to Q8_1 inside the
// kernel into shared memory (ggml_blocks.cuh's canonical cast, as the dp4a route does) instead of a
// separate q8_1 launch; eight warps share a chunk, warp w taking sub-block w, so the lane partials
// w, w + 8, w + 16, w + 24 of canonical_ggml.h step 4 live in its registers (the class of chunk c is
// c % 4, kept current by rotating the four accumulators) and the butterfly's first two steps are
// in-register, the last three across warps through shared memory; the exact decode (c, m, w) per
// format with the multipliers applied to the int32 sums (canonical_ggml.h step 3); the affine
// K-quants' offset term from the group's integer code sum.

#include "ops/linear/ggml/ggml_blocks.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace ninfer::ops::ggml {

inline constexpr int kMmaThreads    = 256;
inline constexpr int kMmaWarps      = kMmaThreads / 32;
inline constexpr int kMmaRows       = 16; // the MMA's m
inline constexpr int kMmaColumns    = 8;  // the MMA's n: T <= 8
inline constexpr int kMmaStages     = 3;  // ring slots
inline constexpr int kMmaChunkSubs  = kMmaWarps; // sub-blocks per chunk: one per warp

// One lane's view of one row of one sub-block: values 4t..4t+3 (lo) and 16+4t..16+4t+3 (hi) as
// signed bytes, and the exact scale terms (ggml_mma_emulation.h step 1).
struct MmaRow {
    std::uint32_t lo, hi;
    float c;
    int m0, m1;
    float c_min; // affine formats
    int m_min;
};

__device__ __forceinline__ std::uint32_t mma_ld32(const std::uint8_t* p) { return *reinterpret_cast<const std::uint32_t*>(p); }

__device__ __forceinline__ std::uint32_t mma_ld32_b2(const std::uint8_t* p) {
    return static_cast<std::uint32_t>(load_i32_b2(p));
}

// IQ3_S: the grid word g with byte j negated where bit j of nib is set (Basalt's iq3s_signed; the
// negation is per byte, so it never carries).
__device__ __forceinline__ std::uint32_t iq3s_signed(std::uint32_t g, std::uint32_t nib) {
    const std::uint32_t b = (nib * 0x204081U) & 0x01010101U; // bit j of nib to bit 0 of byte j
    const std::uint32_t m = (b << 8) - b;                   // 0xFF in those bytes
    return __vsub4(g ^ m, m);
}

// The bytes of one row's 256-value chunk (8 sub-blocks) and the lane's decode of sub-block j of it.
template <QType F>
struct MmaFormat;

template <>
struct MmaFormat<QType::GGML_Q8_0> {
    static constexpr int kChunkBytes = 8 * 34;
    __device__ static void row(const std::uint8_t* p, int j, int t, const Tables&, MmaRow& r) {
        const std::uint8_t* b = p + 34 * j;
        r.c  = load_half(b);
        r.m0 = r.m1 = 1;
        r.lo = mma_ld32_b2(b + 2 + 4 * t);
        r.hi = mma_ld32_b2(b + 18 + 4 * t);
    }
};

template <>
struct MmaFormat<QType::GGML_IQ4_NL> {
    static constexpr int kChunkBytes = 8 * 18;
    __device__ static void row(const std::uint8_t* p, int j, int t, const Tables& tables, MmaRow& r) {
        const std::uint8_t* b = p + 18 * j;
        r.c          = load_half(b);
        r.m0 = r.m1  = 1;
        const int2 v = iq4_lookup(load_i32_b2(b + 2 + 4 * t), tables.kv);
        r.lo         = static_cast<std::uint32_t>(v.x);
        r.hi         = static_cast<std::uint32_t>(v.y);
    }
};

template <>
struct MmaFormat<QType::GGML_IQ4_XS> {
    static constexpr int kChunkBytes = 136;
    __device__ static void row(const std::uint8_t* b, int j, int t, const Tables& tables, MmaRow& r) {
        const std::uint32_t h = load_u16(b + 2);
        const int ls = ((b[4 + (j >> 1)] >> (4 * (j & 1))) & 0x0F) | (static_cast<int>((h >> (2 * j)) & 3) << 4);
        r.c          = load_half(b);
        r.m0 = r.m1  = ls - 32;
        // 136-byte blocks keep the codes of an 8-byte aligned row 8-byte aligned (staging keeps the
        // address modulo 16).
        const int2 v = iq4_lookup(static_cast<int>(mma_ld32(b + 8 + 16 * j + 4 * t)), tables.kv);
        r.lo         = static_cast<std::uint32_t>(v.x);
        r.hi         = static_cast<std::uint32_t>(v.y);
    }
};

// Q2_0's words: the byte permute of ggml_blocks.cuh's decoder, values 8 i .. 8 i + 7 from the u16
// at qs + 2 i, word 0 the first four.
__device__ __forceinline__ std::uint32_t q2_0_word(const std::uint8_t* qs, int w) {
    const std::uint32_t q  = load_u16(qs + 2 * (w >> 1));
    const std::uint32_t qe = __byte_perm(0x020100FFU, 0x020100FFU, q >> 0);
    const std::uint32_t qo = __byte_perm(0x020100FFU, 0x020100FFU, q >> 2);
    return (w & 1) ? __byte_perm(qe, qo, 0x7362) : __byte_perm(qe, qo, 0x5140);
}

template <>
struct MmaFormat<QType::GGML_Q2_0> {
    static constexpr int kChunkBytes = 4 * 18;
    __device__ static void row(const std::uint8_t* p, int j, int t, const Tables&, MmaRow& r) {
        const std::uint8_t* b  = p + 18 * (j >> 1);
        const std::uint8_t* qs = b + 2 + 8 * (j & 1);
        r.c  = load_half(b);
        r.m0 = r.m1 = 1;
        r.lo = q2_0_word(qs, t);
        r.hi = q2_0_word(qs, 4 + t);
    }
};

template <>
struct MmaFormat<QType::GGML_IQ3_S> {
    static constexpr int kChunkBytes = 110;
    __device__ static void row(const std::uint8_t* b, int j, int t, const Tables& tables, MmaRow& r) {
        r.c                 = load_half(b);
        r.m0 = r.m1         = 1 + 2 * ((b[106 + (j >> 1)] >> (4 * (j & 1))) & 0x0F);
        const std::uint32_t qh = b[66 + j];
        const std::uint32_t sg = mma_ld32_b2(b + 74 + 4 * j); // the four sign bytes: nibble l signs word l
        r.lo = iq3s_signed(tables.iq3s[b[2 + 8 * j + t] | (((qh >> t) & 1U) << 8)], (sg >> (4 * t)) & 15U);
        r.hi = iq3s_signed(tables.iq3s[b[6 + 8 * j + t] | (((qh >> (4 + t)) & 1U) << 8)], (sg >> (16 + 4 * t)) & 15U);
    }
};

template <>
struct MmaFormat<QType::GGML_Q6_K> {
    static constexpr int kChunkBytes = 210;
    __device__ static void row(const std::uint8_t* b, int j, int t, const Tables&, MmaRow& r) {
        const int h            = j >> 2;
        const int qd           = j & 3;
        const std::uint8_t* ql = b + 64 * h + 32 * (qd & 1);
        const std::uint8_t* qh = b + 128 + 32 * h;
        const int low_shift    = 4 * (qd >> 1);
        const int high_shift   = 2 * qd;
        r.c  = load_half(b + 208);
        r.m0 = static_cast<std::int8_t>(b[192 + 2 * j]);
        r.m1 = static_cast<std::int8_t>(b[193 + 2 * j]);
        const auto word = [&](int w) {
            const std::uint32_t vl = (mma_ld32_b2(ql + 4 * w) >> low_shift) & 0x0F0F0F0FU;
            const std::uint32_t vh = ((mma_ld32_b2(qh + 4 * w) >> high_shift) & 0x03030303U) << 4;
            return static_cast<std::uint32_t>(__vsubss4(static_cast<int>(vl | vh), 0x20202020));
        };
        r.lo = word(t);
        r.hi = word(4 + t);
    }
};

// get_scale_min_k4 of sub-block j of a Q4_K/Q5_K block, without a branch on j.
__device__ __forceinline__ void mma_k_quant_scales(const std::uint8_t* b, int j, MmaRow& r) {
    const int i  = j & 3;
    const int a0 = b[4 + i];
    const int a1 = b[8 + i];
    const int a2 = b[12 + i];
    r.m0 = r.m1 = j < 4 ? (a0 & 63) : ((a2 & 0x0F) | ((a0 >> 6) << 4));
    r.m_min     = j < 4 ? (a1 & 63) : ((a2 >> 4) | ((a1 >> 6) << 4));
    r.c         = load_half(b);
    r.c_min     = load_half(b + 2);
}

template <>
struct MmaFormat<QType::GGML_Q4_K> {
    static constexpr int kChunkBytes = 144;
    __device__ static void row(const std::uint8_t* b, int j, int t, const Tables&, MmaRow& r) {
        mma_k_quant_scales(b, j, r);
        const std::uint8_t* qs = b + 16 + 32 * (j >> 1);
        const int shift        = 4 * (j & 1);
        r.lo = (mma_ld32(qs + 4 * t) >> shift) & 0x0F0F0F0FU;
        r.hi = (mma_ld32(qs + 16 + 4 * t) >> shift) & 0x0F0F0F0FU;
    }
};

template <>
struct MmaFormat<QType::GGML_Q5_K> {
    static constexpr int kChunkBytes = 176;
    __device__ static void row(const std::uint8_t* b, int j, int t, const Tables&, MmaRow& r) {
        mma_k_quant_scales(b, j, r);
        const std::uint8_t* qh = b + 16;
        const std::uint8_t* qs = b + 48 + 32 * (j >> 1);
        const int shift        = 4 * (j & 1);
        r.lo = ((mma_ld32(qs + 4 * t) >> shift) & 0x0F0F0F0FU) | (((mma_ld32(qh + 4 * t) >> j) & 0x01010101U) << 4);
        r.hi = ((mma_ld32(qs + 16 + 4 * t) >> shift) & 0x0F0F0F0FU) |
               (((mma_ld32(qh + 16 + 4 * t) >> j) & 0x01010101U) << 4);
    }
};

// A staged row of a slot: the chunk's bytes from the 16-byte unit holding its first byte, padded to
// an odd number of 16-byte units.
__host__ __device__ constexpr int mma_row_stride(int chunk_bytes) {
    const int r = (chunk_bytes + 15 + 15) & ~15;
    return ((r / 16) & 1) ? r : r + 16;
}

template <QType F, int K>
struct MmaGeometry {
    static constexpr int kSubs       = K / kSubValues;
    static constexpr int kChunks     = (kSubs + kMmaChunkSubs - 1) / kMmaChunkSubs;
    static constexpr int kChunkBytes = MmaFormat<F>::kChunkBytes;
    static constexpr int kRowBytes   = K / Format<F>::kBlockValues * Format<F>::kBlockBytes;
    static constexpr int kStride     = mma_row_stride(kChunkBytes);
    static constexpr int kUnits      = kStride / 16; // the most 16-byte units a staged row takes
    static constexpr int kSlotBytes  = kMmaRows * kStride;
    static constexpr int kXStride    = K + 16; // a column's codes, padded so the 8 columns hit distinct banks
    static_assert(K % Format<F>::kBlockValues == 0);
    static_assert(kSubs % kMmaChunkSubs == 0 || (kSubs % kMmaChunkSubs) % (Format<F>::kBlockValues / 32) == 0,
                  "a partial chunk holds whole blocks");
};

// Shared memory: the ring, the cast columns (codes, scales, and the affine formats' code sums), the
// IQ3_S grid, the warps' sums.
template <QType F, int K>
constexpr std::size_t mma_shared_bytes(int t) {
    using G = MmaGeometry<F, K>;
    return static_cast<std::size_t>(kMmaStages) * G::kSlotBytes + static_cast<std::size_t>(t) * G::kXStride +
           static_cast<std::size_t>(t) * G::kSubs * 4 * (kAffine<F> ? 2 : 1) + (kNeedsIq3sGrid<F> ? 512 * 4 : 0) +
           static_cast<std::size_t>(kMmaWarps) * kMmaRows * kMmaColumns * 4;
}

__device__ __forceinline__ void mma_k32(int (&d)[4], std::uint32_t a0, std::uint32_t a1, std::uint32_t a2,
                                        std::uint32_t a3, std::uint32_t b0, std::uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                 "{%10,%10,%10,%10};\n"
                 : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(0));
}

__device__ __forceinline__ void mma_k16(int (&d)[4], std::uint32_t a0, std::uint32_t a1, std::uint32_t b0) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%7,%7,%7,%7};\n"
                 : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
                 : "r"(a0), "r"(a1), "r"(b0), "r"(0));
}

__device__ __forceinline__ void mma_cp_async16(void* smem, const void* global) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(static_cast<unsigned>(__cvta_generic_to_shared(smem))),
                 "l"(global)
                 : "memory");
}

__device__ __forceinline__ void mma_cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::: "memory"); }

template <int N>
__device__ __forceinline__ void mma_cp_async_wait() {
    asm volatile("cp.async.wait_group %0;\n" ::"n"(N) : "memory");
}

// x is BF16 [K,T] (column t at x + t K), out BF16 [N,T]; w is 16-byte aligned and n % 16 == 0.
template <QType F, int K, bool Residual>
__global__ void __launch_bounds__(kMmaThreads)
    ggml_mma_decode_kernel(const std::uint8_t* __restrict__ w, const __nv_bfloat16* __restrict__ x,
                           __nv_bfloat16* __restrict__ out, int n, int t_cols) {
    using G = MmaGeometry<F, K>;
    extern __shared__ __align__(16) unsigned char smem[];
    unsigned char* ring = smem;
    auto* xq            = reinterpret_cast<std::int8_t*>(smem + kMmaStages * G::kSlotBytes);
    auto* xd            = reinterpret_cast<float*>(xq + static_cast<std::size_t>(t_cols) * G::kXStride);
    auto* xs            = reinterpret_cast<int*>(xd + static_cast<std::size_t>(t_cols) * G::kSubs); // affine only
    auto* grid_s        = reinterpret_cast<std::uint32_t*>(xd + static_cast<std::size_t>(t_cols) * G::kSubs *
                                                                   (kAffine<F> ? 2 : 1));
    auto* red           = reinterpret_cast<float*>(reinterpret_cast<unsigned char*>(grid_s) + (kNeedsIq3sGrid<F> ? 512 * 4 : 0));

    const int tid  = static_cast<int>(threadIdx.x);
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int g    = lane >> 2; // the MMA's group: rows g, g + 8 and column g of B
    const int t    = lane & 3;  // its thread in group: columns 2t, 2t + 1 of D

    const int tiles = n / kMmaRows;
    const int ctas  = static_cast<int>(gridDim.x);
    const int cta   = static_cast<int>(blockIdx.x);
    const int mine  = cta < tiles ? (tiles - 1 - cta) / ctas + 1 : 0;
    const int units = mine * G::kChunks;

    // Chunk u of this CTA (tile cta + (u / kChunks) ctas, chunk u % kChunks) into slot u % S.
    const auto issue = [&](int u) {
        const int tile          = cta + (u / G::kChunks) * ctas;
        const int c             = u % G::kChunks;
        unsigned char* slot     = ring + (u % kMmaStages) * G::kSlotBytes;
        const int chunk_bytes   = c * G::kChunkBytes + G::kChunkBytes <= G::kRowBytes ? G::kChunkBytes
                                                                                      : G::kRowBytes - c * G::kChunkBytes;
        for (int q = tid; q < kMmaRows * G::kUnits; q += kMmaThreads) {
            const int r              = q / G::kUnits;
            const int k              = q - r * G::kUnits;
            const std::size_t offset = static_cast<std::size_t>(tile * kMmaRows + r) * G::kRowBytes +
                                       static_cast<std::size_t>(c) * G::kChunkBytes;
            const int lead           = static_cast<int>(offset & 15);
            if (16 * k < lead + chunk_bytes) {
                mma_cp_async16(slot + r * G::kStride + 16 * k, w + (offset - lead) + 16 * k);
            }
        }
    };
#pragma unroll
    for (int u = 0; u < kMmaStages - 1; ++u) {
        if (u < units) issue(u);
        mma_cp_async_commit();
    }

    // The Q8_1 cast of the columns while the first chunks land.
    Tables tables;
    load_tables<F>(tables, grid_s);
    for (int q = tid; q < t_cols * G::kSubs; q += kMmaThreads) {
        const int col = q / G::kSubs;
        const int s   = q - col * G::kSubs;
        int words[8];
        xd[q] = quantize_q8_1_group(x + static_cast<std::size_t>(col) * K + kSubValues * s, words);
        if constexpr (kAffine<F>) {
            int sum = 0;
#pragma unroll
            for (int i = 0; i < 8; ++i) sum = __dp4a(words[i], 0x01010101, sum);
            xs[q] = sum;
        }
        auto* dst = reinterpret_cast<int4*>(xq + static_cast<std::size_t>(col) * G::kXStride + kSubValues * s);
        dst[0]    = make_int4(words[0], words[1], words[2], words[3]);
        dst[1]    = make_int4(words[4], words[5], words[6], words[7]);
    }

    // acc[p][q]: lane partial class (p + chunks so far) % 4 of this lane's (row, column) q.
    float acc[4][4];
#pragma unroll
    for (int p = 0; p < 4; ++p) {
#pragma unroll
        for (int q = 0; q < 4; ++q) acc[p][q] = 0.0F;
    }

    for (int u = 0; u < units; ++u) {
        const int tile = cta + (u / G::kChunks) * ctas;
        const int c    = u % G::kChunks;
        mma_cp_async_wait<kMmaStages - 2>();
        __syncthreads(); // chunk u is in for every thread (and the cast, at u = 0); slot u - 1 is free
        if (u + kMmaStages - 1 < units) issue(u + kMmaStages - 1);
        mma_cp_async_commit();

        const int s = kMmaChunkSubs * c + warp;
        if (s < G::kSubs) {
            const unsigned char* slot = ring + (u % kMmaStages) * G::kSlotBytes;
            MmaRow rows[2];
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int r              = g + 8 * h;
                const std::size_t offset = static_cast<std::size_t>(tile * kMmaRows + r) * G::kRowBytes +
                                           static_cast<std::size_t>(c) * G::kChunkBytes;
                rows[h].c_min = 0.0F;
                rows[h].m_min = 0;
                MmaFormat<F>::row(slot + r * G::kStride + static_cast<int>(offset & 15), warp, t, tables, rows[h]);
            }
            std::uint32_t b0 = 0, b1 = 0;
            if (g < t_cols) {
                const std::int8_t* xc = xq + static_cast<std::size_t>(g) * G::kXStride + kSubValues * s + 4 * t;
                b0                    = *reinterpret_cast<const std::uint32_t*>(xc);
                b1                    = *reinterpret_cast<const std::uint32_t*>(xc + 16);
            }
            float d[2];
            int a[2];
#pragma unroll
            for (int e = 0; e < 2; ++e) {
                const int col = 2 * t + e;
                d[e]          = col < t_cols ? xd[col * G::kSubs + s] : 0.0F;
                if constexpr (kAffine<F>) { a[e] = col < t_cols ? xs[col * G::kSubs + s] : 0; }
            }
            int p[4];
            if constexpr (F == QType::GGML_Q6_K) {
                int e0[4], e1[4];
                mma_k16(e0, rows[0].lo, rows[1].lo, b0);
                mma_k16(e1, rows[0].hi, rows[1].hi, b1);
#pragma unroll
                for (int q = 0; q < 4; ++q) p[q] = rows[q >> 1].m0 * e0[q] + rows[q >> 1].m1 * e1[q];
            } else {
                int e[4];
                mma_k32(e, rows[0].lo, rows[1].lo, rows[0].hi, rows[1].hi, b0, b1);
#pragma unroll
                for (int q = 0; q < 4; ++q) p[q] = rows[q >> 1].m0 * e[q];
            }
#pragma unroll
            for (int q = 0; q < 4; ++q) {
                const MmaRow& r = rows[q >> 1];
                acc[0][q]       = __fmaf_rn(__fmul_rn(r.c, d[q & 1]), __int2float_rn(p[q]), acc[0][q]);
                if constexpr (kAffine<F>) {
                    acc[0][q] = __fmaf_rn(__fmul_rn(r.c_min, d[q & 1]), __int2float_rn(-(r.m_min * a[q & 1])), acc[0][q]);
                }
            }
        }
        // The next chunk's class to the front.
#pragma unroll
        for (int q = 0; q < 4; ++q) {
            const float front = acc[0][q];
            acc[0][q]         = acc[1][q];
            acc[1][q]         = acc[2][q];
            acc[2][q]         = acc[3][q];
            acc[3][q]         = front;
        }

        if (c == G::kChunks - 1) {
            // After kChunks rotations class k sits at (k - kChunks) mod 4. Butterfly steps 16 and 8
            // (partial w + 8 k with w + 8 (k + 2), then w with w + 8), then the warps' through `red`.
            constexpr int kShift = (4 - G::kChunks % 4) % 4;
            float* own           = red + warp * kMmaRows * kMmaColumns;
#pragma unroll
            for (int q = 0; q < 4; ++q) {
                const float p0 = __fadd_rn(acc[(0 + kShift) % 4][q], acc[(2 + kShift) % 4][q]);
                const float p1 = __fadd_rn(acc[(1 + kShift) % 4][q], acc[(3 + kShift) % 4][q]);
                own[(2 * t + (q & 1)) * kMmaRows + g + 8 * (q >> 1)] = __fadd_rn(p0, p1);
#pragma unroll
                for (int k = 0; k < 4; ++k) acc[k][q] = 0.0F;
            }
            __syncthreads();
            if (tid < kMmaRows * kMmaColumns) {
                const int col = tid / kMmaRows;
                const int row = tid - col * kMmaRows;
                if (col < t_cols) {
                    float v[kMmaWarps];
#pragma unroll
                    for (int i = 0; i < kMmaWarps; ++i) v[i] = red[i * kMmaRows * kMmaColumns + tid];
#pragma unroll
                    for (int o = kMmaWarps / 2; o > 0; o >>= 1) {
#pragma unroll
                        for (int i = 0; i < o; ++i) v[i] = __fadd_rn(v[i], v[i + o]);
                    }
                    const std::size_t at = static_cast<std::size_t>(col) * n + tile * kMmaRows + row;
                    float y              = v[0];
                    if constexpr (Residual) y = __fadd_rn(y, __bfloat162float(out[at]));
                    out[at] = __float2bfloat16_rn(y);
                }
            }
            // `red` is next written after at least one more __syncthreads (the loop's top).
        }
    }
    mma_cp_async_wait<0>();
}

inline void mma_check(cudaError_t error, const char* what) {
    if (error != cudaSuccess) {
        throw std::runtime_error(std::string("ggml mma decode: ") + what + ": " + cudaGetErrorString(error));
    }
}

// Persistent CTAs: as many as fit at once at this T, never more than there are tiles.
template <QType F, int K, bool Residual>
void launch_mma_decode(const std::uint8_t* w, const __nv_bfloat16* x, __nv_bfloat16* out, int n, int t,
                       cudaStream_t stream) {
    if (t < 1 || t > kMmaColumns) { throw std::invalid_argument("ggml mma decode: T must be in [1, 8]"); }
    struct Resident {
        int ctas[kMmaColumns + 1] = {};
    };
    static const Resident resident = [] {
        auto kernel = ggml_mma_decode_kernel<F, K, Residual>;
        mma_check(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                       static_cast<int>(mma_shared_bytes<F, K>(kMmaColumns))),
                  "shared memory opt-in");
        int device = 0;
        int sms    = 0;
        mma_check(cudaGetDevice(&device), "device");
        mma_check(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device), "SMs");
        Resident out;
        for (int c = 1; c <= kMmaColumns; ++c) {
            int per_sm = 0;
            mma_check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, kernel, kMmaThreads, mma_shared_bytes<F, K>(c)),
                      "occupancy");
            out.ctas[c] = sms * (per_sm > 0 ? per_sm : 1);
        }
        return out;
    }();
    const int tiles = n / kMmaRows;
    const int grid  = tiles < resident.ctas[t] ? tiles : resident.ctas[t];
    ggml_mma_decode_kernel<F, K, Residual><<<grid, kMmaThreads, mma_shared_bytes<F, K>(t), stream>>>(w, x, out, n, t);
    mma_check(cudaGetLastError(), "launch");
}

} // namespace ninfer::ops::ggml
