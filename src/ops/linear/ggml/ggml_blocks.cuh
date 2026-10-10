#pragma once

// Device decoders of the GGML block formats and the Q8_1 activation cast, shared by the decode
// (MMVQ) and wide (MMQ) routes of the GGML linears.
//
// Adapted from Strata src/kernels/cuda/native_mmvq.cu and src/kernels/cuda/iq_kernels.cu (Strata
// commit 99f3dbd, MIT, Copyright (c) 2026 Niko1221 and the Strata contributors; notice in
// third_party/strata/LICENSE; iq_kernels.cu carries an attribution line only). Both transcribe
// llama.cpp 3cf03257f219afbe7334045ff7c6a06ac68c627d ggml/src/ggml-cuda/{quantize.cu,vecdotq.cuh,
// common.cuh}. native_mmvq.cu's notice follows verbatim.
//
// Modified for NInfer: each format is split into a decoder that turns one 32-value sub-block into
// eight words of four signed int8 values plus its FP32 scale(s), so the dp4a GEMV and the int8 MMA
// GEMM share one decode; the Q8_1 cast keeps its scale in FP32, multiplies by the reciprocal of
// the scale and rounds ties to even (ggml divides and rounds ties away), and carries no sum term,
// because none of the formats here has an offset that needs it.
//
// MIT License
// Copyright (c) 2023-2026 The ggml authors
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "core/weight.h"
#include "ops/common/canonical_ggml.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

// The device copy of ggml's IQ3_S grid (third_party/ggml/ggml-common-tables.h is the one table
// set; tests read the same header as host arrays). Static per translation unit.
#define NINFER_GGML_TABLE_BEGIN(type, name, size) static __device__ const type name[size] = {
#define NINFER_GGML_TABLE_END() };
#include "ggml-common-tables.h"

namespace ninfer::ops::ggml {

inline constexpr int kSubValues = 32;

// One decoded 32-value sub-block: v[w] packs values 4w..4w+3 as signed bytes (little endian).
// scale[0] multiplies values 0..15 and scale[1] values 16..31; they differ only for Q6_K.
struct SubBlock {
    int v[8];
    float scale[2];
};

// Tables a format needs in registers or shared memory. kv holds ggml's IQ4_NL codebook as four
// words for the byte-permute lookup; iq3s points at a shared-memory copy of iq3s_grid.
struct Tables {
    std::uint32_t kv[4];
    const std::uint32_t* iq3s;
};

__device__ __forceinline__ std::uint32_t load_u16(const std::uint8_t* p) {
    return *reinterpret_cast<const std::uint16_t*>(p);
}

// Four bytes from a two-byte aligned address (ggml's get_int_b2).
__device__ __forceinline__ int load_i32_b2(const std::uint8_t* p) {
    return static_cast<int>(load_u16(p) | (load_u16(p + 2) << 16));
}

__device__ __forceinline__ int load_i32_b4(const std::uint8_t* p) {
    return *reinterpret_cast<const int*>(p);
}

__device__ __forceinline__ float load_half(const std::uint8_t* p) {
    return __half2float(*reinterpret_cast<const __half*>(p));
}

// ggml's get_int_from_table_16: eight 4-bit codes -> two words of codebook bytes (low nibbles in
// .x, high nibbles in .y).
__device__ __forceinline__ int2 iq4_lookup(int q4, const std::uint32_t (&table)[4]) {
    std::uint32_t tmp[2];
    const std::uint32_t select = 0x32103210U | ((static_cast<std::uint32_t>(q4) & 0x88888888U) >> 1);
#pragma unroll
    for (std::uint32_t i = 0; i < 2; ++i) {
        const std::uint32_t shift = 16 * i;
        const std::uint32_t low   = __byte_perm(table[0], table[1], static_cast<std::uint32_t>(q4) >> shift);
        const std::uint32_t high  = __byte_perm(table[2], table[3], static_cast<std::uint32_t>(q4) >> shift);
        tmp[i]                    = __byte_perm(low, high, select >> shift);
    }
    return make_int2(static_cast<int>(__byte_perm(tmp[0], tmp[1], 0x6420)),
                     static_cast<int>(__byte_perm(tmp[0], tmp[1], 0x7531)));
}

template <QType F>
struct Format;

// Q8_0: {fp16 d; int8 qs[32]}.
template <>
struct Format<QType::GGML_Q8_0> {
    static constexpr int kBlockValues = 32, kBlockBytes = 34;
    static constexpr bool kSplitScale = false;
    __device__ static void decode(const std::uint8_t* row, int s, const Tables&, SubBlock& out) {
        const std::uint8_t* b = row + s * kBlockBytes;
        out.scale[0] = out.scale[1] = load_half(b);
#pragma unroll
        for (int w = 0; w < 8; ++w) out.v[w] = load_i32_b2(b + 2 + 4 * w);
    }
};

// IQ4_NL: {fp16 d; u8 qs[16]}; value j = kvalues[qs[j] & 15], value j + 16 = kvalues[qs[j] >> 4].
template <>
struct Format<QType::GGML_IQ4_NL> {
    static constexpr int kBlockValues = 32, kBlockBytes = 18;
    static constexpr bool kSplitScale = false;
    __device__ static void decode(const std::uint8_t* row, int s, const Tables& t, SubBlock& out) {
        const std::uint8_t* b = row + s * kBlockBytes;
        out.scale[0] = out.scale[1] = load_half(b);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int2 v = iq4_lookup(load_i32_b2(b + 2 + 4 * j), t.kv);
            out.v[j]     = v.x;
            out.v[4 + j] = v.y;
        }
    }
};

// IQ4_XS: {fp16 d; u16 scales_h; u8 scales_l[4]; u8 qs[128]} per 256 values. Sub-block ib has
// scale d * (ls - 32) with the 6-bit ls split across scales_l and scales_h.
template <>
struct Format<QType::GGML_IQ4_XS> {
    static constexpr int kBlockValues = 256, kBlockBytes = 136;
    static constexpr bool kSplitScale = false;
    __device__ static void decode(const std::uint8_t* row, int s, const Tables& t, SubBlock& out) {
        const std::uint8_t* b = row + (s >> 3) * kBlockBytes;
        const int ib          = s & 7;
        const std::uint32_t h = load_u16(b + 2);
        const int ls          = ((b[4 + (ib >> 1)] >> (4 * (ib & 1))) & 0x0F) |
                       (static_cast<int>((h >> (2 * ib)) & 3) << 4);
        out.scale[0] = out.scale[1] = load_half(b) * static_cast<float>(ls - 32);
        // Blocks are 136 bytes, so the codes of an 8-byte aligned row are 8-byte aligned.
        const auto* qs = reinterpret_cast<const uint2*>(b + 8 + 16 * ib);
        const uint2 q01 = qs[0];
        const uint2 q23 = qs[1];
        const int words[4] = {static_cast<int>(q01.x), static_cast<int>(q01.y),
                              static_cast<int>(q23.x), static_cast<int>(q23.y)};
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int2 v = iq4_lookup(words[j], t.kv);
            out.v[j]     = v.x;
            out.v[4 + j] = v.y;
        }
    }
};

// Q2_0: {fp16 d; u8 qs[16]} per 64 values; code j = (qs[j / 4] >> 2 (j % 4)) & 3, value (code - 1) d.
// The byte permute maps codes {0,1,2,3} to {-1,0,1,2} (vec_dot_q2_0_q8_1).
template <>
struct Format<QType::GGML_Q2_0> {
    static constexpr int kBlockValues = 64, kBlockBytes = 18;
    static constexpr bool kSplitScale = false;
    __device__ static void decode(const std::uint8_t* row, int s, const Tables&, SubBlock& out) {
        const std::uint8_t* b = row + (s >> 1) * kBlockBytes;
        out.scale[0] = out.scale[1] = load_half(b);
        const std::uint8_t* qs      = b + 2 + 8 * (s & 1);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const std::uint32_t q  = load_u16(qs + 2 * j);
            const std::uint32_t qe = __byte_perm(0x020100FFU, 0x020100FFU, q >> 0);
            const std::uint32_t qo = __byte_perm(0x020100FFU, 0x020100FFU, q >> 2);
            out.v[2 * j]     = static_cast<int>(__byte_perm(qe, qo, 0x5140));
            out.v[2 * j + 1] = static_cast<int>(__byte_perm(qe, qo, 0x7362));
        }
    }
};

// IQ3_S: {fp16 d; u8 qs[64]; u8 qh[8]; u8 signs[32]; u8 scales[4]} per 256 values. Sub-block ib
// reads qs[8 ib..], qh[ib], signs[4 ib..] and scale nibble ib; its scale is d (1 + 2 nibble).
template <>
struct Format<QType::GGML_IQ3_S> {
    static constexpr int kBlockValues = 256, kBlockBytes = 110;
    static constexpr bool kSplitScale = false;
    __device__ static void decode(const std::uint8_t* row, int s, const Tables& t, SubBlock& out) {
        const std::uint8_t* b = row + (s >> 3) * kBlockBytes;
        const int ib          = s & 7;
        const int nibble      = (b[106 + (ib >> 1)] >> (4 * (ib & 1))) & 0x0F;
        out.scale[0] = out.scale[1] = load_half(b) * static_cast<float>(1 + 2 * nibble);
        const int qs_packed[2] = {load_i32_b2(b + 2 + 8 * ib), load_i32_b2(b + 6 + 8 * ib)};
        const auto* qs         = reinterpret_cast<const std::uint8_t*>(qs_packed);
        const int qh           = b[66 + ib];
        const int signs_packed = load_i32_b2(b + 74 + 4 * ib);
        const auto* signs      = reinterpret_cast<const std::uint8_t*>(&signs_packed);
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const int g1 = static_cast<int>(t.iq3s[qs[2 * l + 0] | ((qh << (8 - 2 * l)) & 0x100)]);
            const int g2 = static_cast<int>(t.iq3s[qs[2 * l + 1] | ((qh << (7 - 2 * l)) & 0x100)]);
            const int sg = signs[l];
            const int m1 = static_cast<int>(__vcmpne4(((sg & 0x03) << 7) | ((sg & 0x0C) << 21), 0));
            const int m2 = static_cast<int>(__vcmpne4(((sg & 0x30) << 3) | ((sg & 0xC0) << 17), 0));
            out.v[2 * l]     = static_cast<int>(__vsub4(g1 ^ m1, m1));
            out.v[2 * l + 1] = static_cast<int>(__vsub4(g2 ^ m2, m2));
        }
    }
};

// Q6_K: {u8 ql[128]; u8 qh[64]; int8 scales[16]; fp16 d} per 256 values. Sub-block ib = 4 h + qd
// reads ql[64 h + 32 (qd & 1) + l] (nibble qd >> 1) and qh[32 h + l] (bits 2 qd), values q - 32;
// values 16 g .. 16 g + 15 of the block use scales[g].
template <>
struct Format<QType::GGML_Q6_K> {
    static constexpr int kBlockValues = 256, kBlockBytes = 210;
    static constexpr bool kSplitScale = true;
    __device__ static void decode(const std::uint8_t* row, int s, const Tables&, SubBlock& out) {
        const std::uint8_t* b  = row + (s >> 3) * kBlockBytes;
        const int ib           = s & 7;
        const int h            = ib >> 2;
        const int qd           = ib & 3;
        const std::uint8_t* ql = b + 64 * h + 32 * (qd & 1);
        const std::uint8_t* qh = b + 128 + 32 * h;
        const float d          = load_half(b + 208);
        out.scale[0]           = d * static_cast<float>(static_cast<std::int8_t>(b[192 + 2 * ib]));
        out.scale[1]           = d * static_cast<float>(static_cast<std::int8_t>(b[193 + 2 * ib]));
        const int low_shift    = 4 * (qd >> 1);
        const int high_shift   = 2 * qd;
#pragma unroll
        for (int w = 0; w < 8; ++w) {
            const int vl = (load_i32_b2(ql + 4 * w) >> low_shift) & 0x0F0F0F0F;
            const int vh = ((load_i32_b2(qh + 4 * w) >> high_shift) & 0x03030303) << 4;
            out.v[w]     = static_cast<int>(__vsubss4(vl | vh, 0x20202020));
        }
    }
};

template <QType F>
inline constexpr bool kNeedsIq3sGrid = F == QType::GGML_IQ3_S;

template <QType F>
inline constexpr bool kNeedsIq4Codebook = F == QType::GGML_IQ4_NL || F == QType::GGML_IQ4_XS;

// Fills `tables` for format F. A format that reads the IQ3_S grid copies it into `grid_smem`
// (512 words) with the whole block; the caller synchronizes before the first decode.
template <QType F>
__device__ __forceinline__ void load_tables(Tables& tables, std::uint32_t* grid_smem) {
    if constexpr (kNeedsIq4Codebook<F>) {
        const auto* kv = reinterpret_cast<const std::uint8_t*>(ggml_tables::kvalues_iq4nl);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            tables.kv[i] = static_cast<std::uint32_t>(kv[4 * i]) |
                           (static_cast<std::uint32_t>(kv[4 * i + 1]) << 8) |
                           (static_cast<std::uint32_t>(kv[4 * i + 2]) << 16) |
                           (static_cast<std::uint32_t>(kv[4 * i + 3]) << 24);
        }
    }
    if constexpr (kNeedsIq3sGrid<F>) {
        for (int i = static_cast<int>(threadIdx.x); i < 512; i += static_cast<int>(blockDim.x)) {
            grid_smem[i] = ggml_tables::iq3s_grid[i];
        }
        tables.iq3s = grid_smem;
    } else {
        tables.iq3s = nullptr;
    }
}

// The Q8_1 activation cast of one 32-value group (linear.h), the only implementation both routes
// use. It is the canonical A8 cast (ops/common/canonical_ggml.h, shared with offloaded_sparse_moe's
// CPU and GPU routes): a = max|x|, d = a / 127, q = rint(x * (127 / a)) with ties to even, saturated
// to [-127, 127], q = 0 when a = 0. `x` is 16-byte aligned; q lands in `words` (values 4w..4w+3 in
// word w) and d is returned.
__device__ __forceinline__ float quantize_q8_1_group(const __nv_bfloat16* x, int (&words)[8]) {
    std::uint32_t bits[16];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const uint4 raw  = *reinterpret_cast<const uint4*>(x + 8 * i);
        bits[4 * i + 0]  = raw.x;
        bits[4 * i + 1]  = raw.y;
        bits[4 * i + 2]  = raw.z;
        bits[4 * i + 3]  = raw.w;
    }
    float value[32];
    float amax = 0.0F;
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        value[2 * i]     = __uint_as_float(bits[i] << 16);
        value[2 * i + 1] = __uint_as_float(bits[i] & 0xFFFF0000U);
        amax             = canon::a8_amax_step(canon::a8_amax_step(amax, value[2 * i]), value[2 * i + 1]);
    }
    const float inverse = canon::a8_inverse(amax);
#pragma unroll
    for (int w = 0; w < 8; ++w) {
        std::uint32_t word = 0;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int q = canon::a8_code(value[4 * w + j], inverse);
            word |= (static_cast<std::uint32_t>(q) & 0xFFU) << (8 * j);
        }
        words[w] = static_cast<int>(word);
    }
    return canon::a8_scale(amax);
}

} // namespace ninfer::ops::ggml
