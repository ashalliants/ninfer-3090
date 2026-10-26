#pragma once

// Device decoders of the GGML expert formats for offloaded_sparse_moe's narrow and wide routes: one
// 32-value sub-block into eight words of four signed int8 weights, its two integer half multipliers
// and its FP32 scale (canonical_ggml.h step 2), the same integers the host decoder
// (cpu/expert_cpu.cpp) produces. Each format names the table its decode reads (Table, from table()
// once per kernel): a grid in global memory, or the IQ4_NL codebook in registers.
//
// Adapted from Strata src/kernels/cuda/iq_kernels.cu (Strata commit 99f3dbd, MIT, Copyright (c)
// 2026 Niko1221 and the Strata contributors; notice in third_party/strata/LICENSE; that file
// carries an attribution line only), which transcribes llama.cpp 3cf03257f219afbe7334045ff7c6a06ac68c627d
// ggml/src/ggml-cuda/vecdotq.cuh (MIT, Copyright (c) 2023-2026 The ggml authors; full notice in
// src/ops/linear/ggml/ggml_blocks.cuh and third_party/ggml/LICENSE).
//
// Modified for NInfer: Strata's decode-once `Split<>::load` of IQ2_XXS, IQ2_S and IQ1_M, with
// ggml's integer scale steps (sumi * ls / 8, (sumi0 ls0 + sumi1 ls1 + ...) / 4) replaced by exact
// half multipliers 2 ls + 1 and the factor 1/8 folded into the FP32 scale, and IQ1_M's float delta
// replaced by integer weights 8 grid +- 1. Q2_0, IQ2_XS, IQ3_XXS, IQ3_S, IQ4_XS and IQ4_NL reuse the
// GGML linears' decoders for their values, with the exact multipliers and scale formed here (IQ4_XS's
// multiplier ls - 32 is signed).

#include "ops/linear/ggml/ggml_blocks.cuh"

#include <cstdint>

namespace ninfer::ops::offloaded_moe::device {

struct Sub {
    int v[8]; // values 4w..4w+3 as signed bytes
    int m0, m1;
    float c;
};

__device__ __forceinline__ float half_scale(const std::uint8_t* p) {
    return __half2float(*reinterpret_cast<const __half*>(p));
}

template <QType F>
struct ExpertFormat;

// IQ2_XXS: 66 bytes per 256 values.
template <>
struct ExpertFormat<QType::GGML_IQ2_XXS> {
    using Table = const uint2*;
    __device__ static Table table() { return reinterpret_cast<const uint2*>(ggml_tables::iq2xxs_grid); }
    __device__ static Sub decode(const std::uint8_t* row, int s, Table grid_table) {
        const std::uint8_t* b     = row + (s >> 3) * 66;
        const int ib              = s & 7;
        const int q2              = ggml::load_i32_b2(b + 2 + 8 * ib);
        const std::uint32_t aux32 = static_cast<std::uint32_t>(ggml::load_i32_b2(b + 6 + 8 * ib));
        Sub out;
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const auto index = static_cast<std::uint32_t>(q2 >> (8 * l)) & 0xFFU;
            const uint2 grid = grid_table[index];
            // ggml::signed_grid's arithmetic written out: calling it compiles these kernels with
            // commuted LOP3 operands, so their SASS would no longer match the shipped build's.
            const std::uint32_t signs = ggml::ksigns_word(aux32 >> (7 * l));
            const int s0 = static_cast<int>(__vcmpne4(signs & 0x08040201U, 0));
            const int s1 = static_cast<int>(__vcmpne4(signs & 0x80402010U, 0));
            out.v[2 * l]     = static_cast<int>(__vsub4(static_cast<int>(grid.x) ^ s0, s0));
            out.v[2 * l + 1] = static_cast<int>(__vsub4(static_cast<int>(grid.y) ^ s1, s1));
        }
        out.m0 = out.m1 = 2 * static_cast<int>(aux32 >> 28) + 1;
        out.c           = __fmul_rn(half_scale(b), 0.125F);
        return out;
    }
};

// IQ2_S: 82 bytes per 256 values.
template <>
struct ExpertFormat<QType::GGML_IQ2_S> {
    using Table = const uint2*;
    __device__ static Table table() { return reinterpret_cast<const uint2*>(ggml_tables::iq2s_grid); }
    __device__ static Sub decode(const std::uint8_t* row, int s, Table grid_table) {
        const std::uint8_t* b = row + (s >> 3) * 82;
        const int ib          = s & 7;
        const int qs_packed   = ggml::load_i32_b2(b + 2 + 4 * ib);
        const int signs_packed = ggml::load_i32_b2(b + 34 + 4 * ib);
        const int qh          = b[66 + ib];
        const int scale       = b[74 + ib];
        Sub out;
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const int index = ((qs_packed >> (8 * l)) & 0xFF) | ((qh << (8 - 2 * l)) & 0x300);
            const uint2 grid = grid_table[index];
            const int sb     = (signs_packed >> (8 * l)) & 0xFF;
            const int s0 = static_cast<int>(__vcmpne4(((sb & 0x03) << 7) | ((sb & 0x0C) << 21), 0));
            const int s1 = static_cast<int>(__vcmpne4(((sb & 0x30) << 3) | ((sb & 0xC0) << 17), 0));
            out.v[2 * l]     = static_cast<int>(__vsub4(static_cast<int>(grid.x) ^ s0, s0));
            out.v[2 * l + 1] = static_cast<int>(__vsub4(static_cast<int>(grid.y) ^ s1, s1));
        }
        out.m0 = 2 * (scale & 0xF) + 1;
        out.m1 = 2 * (scale >> 4) + 1;
        out.c  = __fmul_rn(half_scale(b), 0.125F);
        return out;
    }
};

// IQ1_M: 56 bytes per 256 values; weights 8 grid +- 1.
template <>
struct ExpertFormat<QType::GGML_IQ1_M> {
    using Table = const uint2*;
    __device__ static Table table() { return reinterpret_cast<const uint2*>(ggml_tables::iq1s_grid); }
    __device__ static Sub decode(const std::uint8_t* row, int s, Table grid_table) {
        const std::uint8_t* b = row + (s >> 3) * 56;
        const int ib          = s & 7;
        const int qs_packed   = ggml::load_i32_b2(b + 4 * ib);
        const std::uint32_t qh = ggml::load_u16(b + 32 + 2 * ib);
        Sub out;
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const std::uint32_t qhl = qh >> (4 * l); // l = 0, 1: first qh byte; 2, 3: second
            const int index  = ((qs_packed >> (8 * l)) & 0xFF) | static_cast<int>((qhl & 7U) << 8);
            const uint2 grid = grid_table[index];
            const std::uint32_t delta = (qhl & 0x08U) != 0 ? 0xFFFFFFFFU : 0x01010101U;
            out.v[2 * l]     = static_cast<int>(__vadd4((grid.x & 0x1F1F1F1FU) << 3, delta));
            out.v[2 * l + 1] = static_cast<int>(__vadd4((grid.y & 0x1F1F1F1FU) << 3, delta));
        }
        const std::uint32_t sc0 = ggml::load_u16(b + 48), sc1 = ggml::load_u16(b + 50);
        const std::uint32_t sc2 = ggml::load_u16(b + 52), sc3 = ggml::load_u16(b + 54);
        const auto d16 = static_cast<unsigned short>((sc0 >> 12) | ((sc1 >> 8) & 0x00F0U) | ((sc2 >> 4) & 0x0F00U) |
                                                     (sc3 & 0xF000U));
        const std::uint32_t sc = ggml::load_u16(b + 48 + 2 * (ib >> 1));
        const int shift        = 6 * (ib & 1);
        out.m0 = 2 * static_cast<int>((sc >> shift) & 7U) + 1;
        out.m1 = 2 * static_cast<int>((sc >> (shift + 3)) & 7U) + 1;
        out.c  = __fmul_rn(__half2float(__ushort_as_half(d16)), 0.125F);
        return out;
    }
};

// Q2_0: 18 bytes per 64 values; weights code - 1.
template <>
struct ExpertFormat<QType::GGML_Q2_0> {
    using Table = const uint2*;
    __device__ static Table table() { return nullptr; }
    __device__ static Sub decode(const std::uint8_t* row, int s, Table) {
        ggml::SubBlock block;
        const ggml::Tables none{};
        ggml::Format<QType::GGML_Q2_0>::decode(row, s, none, block);
        Sub out;
#pragma unroll
        for (int w = 0; w < 8; ++w) { out.v[w] = block.v[w]; }
        out.m0 = out.m1 = 1;
        out.c           = block.scale[0];
        return out;
    }
};

// The formats below take their values from the GGML linears' decoders (ggml_blocks.cuh) with the
// tables those read: the IQ4_NL codebook in registers (IQ4_XS, IQ4_NL) or the IQ3_S grid, here in
// global memory; IQ2_XS and IQ3_XXS read their grids from global memory themselves.
__device__ __forceinline__ Sub from_linear(const ggml::SubBlock& block, int m0, int m1, float c) {
    Sub out;
#pragma unroll
    for (int w = 0; w < 8; ++w) { out.v[w] = block.v[w]; }
    out.m0 = m0;
    out.m1 = m1;
    out.c  = c;
    return out;
}

// IQ2_XS: 74 bytes per 256 values; scale byte ib holds the halves' ls (multipliers 2 ls + 1), and
// c = d / 8.
template <>
struct ExpertFormat<QType::GGML_IQ2_XS> {
    using Table = ggml::Tables;
    __device__ static Table table() { return Table{{}, nullptr}; }
    __device__ static Sub decode(const std::uint8_t* row, int s, const Table& tables) {
        ggml::SubBlock block;
        ggml::Format<QType::GGML_IQ2_XS>::decode(row, s, tables, block);
        const std::uint8_t* b = row + (s >> 3) * 74;
        const int scale       = b[66 + (s & 7)];
        return from_linear(block, 2 * (scale & 0xF) + 1, 2 * (scale >> 4) + 1, __fmul_rn(half_scale(b), 0.125F));
    }
};

// IQ3_XXS: 98 bytes per 256 values; the sub-block's word holds ls in bits 28..31 (multiplier
// 2 ls + 1 for both halves), and c = d / 4.
template <>
struct ExpertFormat<QType::GGML_IQ3_XXS> {
    using Table = ggml::Tables;
    __device__ static Table table() { return Table{{}, nullptr}; }
    __device__ static Sub decode(const std::uint8_t* row, int s, const Table& tables) {
        ggml::SubBlock block;
        ggml::Format<QType::GGML_IQ3_XXS>::decode(row, s, tables, block);
        const std::uint8_t* b     = row + (s >> 3) * 98;
        const std::uint32_t aux32 = static_cast<std::uint32_t>(ggml::load_i32_b2(b + 66 + 4 * (s & 7)));
        const int m               = 2 * static_cast<int>(aux32 >> 28) + 1;
        return from_linear(block, m, m, __fmul_rn(half_scale(b), 0.25F));
    }
};

// IQ3_S: 110 bytes per 256 values; scale nibble ib gives the multiplier 1 + 2 nibble, and c = d.
template <>
struct ExpertFormat<QType::GGML_IQ3_S> {
    using Table = ggml::Tables;
    __device__ static Table table() { return Table{{}, ggml_tables::iq3s_grid}; }
    __device__ static Sub decode(const std::uint8_t* row, int s, const Table& tables) {
        ggml::SubBlock block;
        ggml::Format<QType::GGML_IQ3_S>::decode(row, s, tables, block);
        const std::uint8_t* b = row + (s >> 3) * 110;
        const int ib          = s & 7;
        const int m           = 1 + 2 * ((b[106 + (ib >> 1)] >> (4 * (ib & 1))) & 0x0F);
        return from_linear(block, m, m, half_scale(b));
    }
};

// The IQ4_NL codebook as four words in registers (ggml::load_tables), for IQ4_XS and IQ4_NL.
__device__ __forceinline__ ggml::Tables iq4_tables() {
    ggml::Tables tables;
    ggml::load_tables<QType::GGML_IQ4_NL>(tables, nullptr);
    return tables;
}

// IQ4_XS: 136 bytes per 256 values; the sub-block's 6-bit ls gives the signed multiplier ls - 32
// in [-32, 31] for both halves, and c = d.
template <>
struct ExpertFormat<QType::GGML_IQ4_XS> {
    using Table = ggml::Tables;
    __device__ static Table table() { return iq4_tables(); }
    __device__ static Sub decode(const std::uint8_t* row, int s, const Table& tables) {
        ggml::SubBlock block;
        ggml::Format<QType::GGML_IQ4_XS>::decode(row, s, tables, block);
        const std::uint8_t* b = row + (s >> 3) * 136;
        const int ib          = s & 7;
        const int ls          = ((b[4 + (ib >> 1)] >> (4 * (ib & 1))) & 0x0F) |
                       (static_cast<int>((ggml::load_u16(b + 2) >> (2 * ib)) & 3U) << 4);
        return from_linear(block, ls - 32, ls - 32, half_scale(b));
    }
};

// IQ4_NL: 18 bytes per 32 values (one block per sub-block); multiplier 1, c = d.
template <>
struct ExpertFormat<QType::GGML_IQ4_NL> {
    using Table = ggml::Tables;
    __device__ static Table table() { return iq4_tables(); }
    __device__ static Sub decode(const std::uint8_t* row, int s, const Table& tables) {
        ggml::SubBlock block;
        ggml::Format<QType::GGML_IQ4_NL>::decode(row, s, tables, block);
        return from_linear(block, 1, 1, half_scale(row + s * 18));
    }
};

} // namespace ninfer::ops::offloaded_moe::device
