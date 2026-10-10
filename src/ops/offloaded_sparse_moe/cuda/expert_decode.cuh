#pragma once

// Device decoders of the GGML expert formats for offloaded_sparse_moe's narrow route: one 32-value
// sub-block into eight words of four signed int8 weights, its two integer half multipliers and its
// FP32 scale (canonical_ggml.h step 2), the same integers the host decoder (cpu/expert_cpu.cpp)
// produces.
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
// replaced by integer weights 8 grid +- 1. Q2_0 reuses the GGML linears' decoder.

#include "ops/linear/ggml/ggml_blocks.cuh"

#include <cstdint>

namespace ninfer::ops::offloaded_moe::device {

struct Sub {
    int v[8]; // values 4w..4w+3 as signed bytes
    int m0, m1;
    float c;
};

__device__ __forceinline__ std::uint32_t unpack_ksigns(std::uint32_t v) {
    v &= 0xFFU;
    const std::uint32_t p = __popc(v) & 1U;
    const std::uint32_t s = v ^ (p << 7);
    return s * 0x01010101U;
}

__device__ __forceinline__ float half_scale(const std::uint8_t* p) {
    return __half2float(*reinterpret_cast<const __half*>(p));
}

template <QType F>
struct ExpertFormat;

// IQ2_XXS: 66 bytes per 256 values.
template <>
struct ExpertFormat<QType::GGML_IQ2_XXS> {
    __device__ static const uint2* global_grid() { return reinterpret_cast<const uint2*>(ggml_tables::iq2xxs_grid); }
    __device__ static Sub decode(const std::uint8_t* row, int s, const uint2* grid_table) {
        const std::uint8_t* b     = row + (s >> 3) * 66;
        const int ib              = s & 7;
        const int q2              = ggml::load_i32_b2(b + 2 + 8 * ib);
        const std::uint32_t aux32 = static_cast<std::uint32_t>(ggml::load_i32_b2(b + 6 + 8 * ib));
        Sub out;
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const auto index = static_cast<std::uint32_t>(q2 >> (8 * l)) & 0xFFU;
            const uint2 grid = grid_table[index];
            const std::uint32_t signs = unpack_ksigns(aux32 >> (7 * l));
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
    __device__ static const uint2* global_grid() { return reinterpret_cast<const uint2*>(ggml_tables::iq2s_grid); }
    __device__ static Sub decode(const std::uint8_t* row, int s, const uint2* grid_table) {
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
    __device__ static const uint2* global_grid() { return reinterpret_cast<const uint2*>(ggml_tables::iq1s_grid); }
    __device__ static Sub decode(const std::uint8_t* row, int s, const uint2* grid_table) {
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
    __device__ static const uint2* global_grid() { return nullptr; }
    __device__ static Sub decode(const std::uint8_t* row, int s, const uint2* grid_table) {
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

} // namespace ninfer::ops::offloaded_moe::device
