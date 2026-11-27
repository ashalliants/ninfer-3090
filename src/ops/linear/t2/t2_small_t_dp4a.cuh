#pragma once

// sm_70 form of the T2G128 x int8 small-T GEMM (t2_small_t_i8.cuh): same activation planes (s8
// codes [T, K] token-major, one binary16 scale per (64-k group, column) at [K / 64, 192]), same
// parents and epilogues, but no mma.sync. Volta has no integer tensor cores, and the m16n8k32 s8
// mma that kernel issues is emulated there (volta_compat.cuh) at several times the weight
// stream's cost. Here every product is a native dp4a: four ternary codes expand to four s8 lanes
// with one prmt against a {0, +1, -2, -1} byte table, and one dp4a takes four k of one row and one
// column.
//
// Layout: eight warps per CTA, each warp owns Rows consecutive output rows and walks K in 32-k
// chunks, lane l taking chunks l, l + 32, ... (every registered K -- 5120, 6144, 17408 -- is a
// whole number of 1024-k warp steps, so no lane idles). A chunk is 8 code bytes per row (one
// 8-byte load, the warp reading 256 contiguous bytes of each row), decoded once into registers and
// reused for every column; a column's 32 activation bytes are read through L1 once per chunk and
// reused for every row. The chunk's int32 sum is rescaled by (row scale of its 128-k group) x
// (column scale of its 64-k group) and accumulated in FP32; the lanes' partials are folded by a
// butterfly at the end. A column's result depends only on K, never on how many columns or rows a
// launch takes, so a verify round reproduces decode bit for bit.

#include "ops/common/memory.cuh"
#include "ops/linear/t2/t2_rowsplit_storage.cuh"
#include "ops/linear/t2/t2_small_t_i8.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops::detail {

inline constexpr int kT2Dp4aWarps   = 8;
inline constexpr int kT2Dp4aThreads = kT2Dp4aWarps * 32;
inline constexpr int kT2Dp4aChunkK  = 32;
inline constexpr int kT2Dp4aMagic   = 0x4B400000; // 1.5 * 2^23

template <int Rows, int Columns, class Epilogue>
__global__ __launch_bounds__(kT2Dp4aThreads) void t2_small_t_dp4a_kernel(
    const std::int8_t* __restrict__ x_codes, const __half* __restrict__ x_scales,
    const __grid_constant__ T2I8Parents<Epilogue> parents, std::int32_t k, std::int32_t col0,
    std::int32_t cols) {
    constexpr int kRowsPerBlock = kT2Dp4aWarps * Rows;
    const int warp              = static_cast<int>(threadIdx.x) >> 5;
    const int lane              = static_cast<int>(threadIdx.x) & 31;
    const bool second           = static_cast<int>(blockIdx.x) >= parents.first_blocks;
    const int row0 =
        (static_cast<int>(blockIdx.x) - (second ? parents.first_blocks : 0)) * kRowsPerBlock +
        warp * Rows;
    const std::uint8_t* __restrict__ codes  = parents.codes[second ? 1 : 0];
    const std::uint8_t* __restrict__ scales = parents.scales[second ? 1 : 0];
    const Epilogue& epilogue                = parents.epilogue[second ? 1 : 0];

    const int chunks                  = k / kT2Dp4aChunkK;
    const std::int64_t code_row_bytes = k / T2RowSplitStorage::kCodesPerByte;
    const std::int64_t scale_row      = k / T2RowSplitStorage::kGroupK;
    const auto* __restrict__ w_scales = reinterpret_cast<const __half*>(scales);

    float acc[Rows][Columns];
#pragma unroll
    for (int r = 0; r < Rows; ++r) {
#pragma unroll
        for (int c = 0; c < Columns; ++c) { acc[r][c] = 0.0F; }
    }

#pragma unroll 2
    for (int chunk = lane; chunk < chunks; chunk += 32) {
        // Rows: eight s8x4 words each, k order {0,2,4,6}{1,3,5,7}{8,..}{9,..} per 16-code word.
        unsigned wq[Rows][8];
        float ws[Rows];
#pragma unroll
        for (int r = 0; r < Rows; ++r) {
            const uint2 packed = load_ldg<uint2>(codes + static_cast<std::int64_t>(row0 + r) *
                                                             code_row_bytes +
                                                 static_cast<std::int64_t>(chunk) * 8);
            ws[r] = __half2float(w_scales[static_cast<std::int64_t>(row0 + r) * scale_row +
                                          (chunk >> 2)]);
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const std::uint32_t word = h == 0 ? packed.x : packed.y;
                const std::uint32_t even = word & 0x33333333u;
                const std::uint32_t odd  = (word >> 2) & 0x33333333u;
                wq[r][4 * h + 0]         = t2_i8_codes(even);
                wq[r][4 * h + 1]         = t2_i8_codes(odd);
                wq[r][4 * h + 2]         = t2_i8_codes(even >> 16);
                wq[r][4 * h + 3]         = t2_i8_codes(odd >> 16);
            }
        }
#pragma unroll
        for (int c = 0; c < Columns; ++c) {
            if (c < cols) {
                const std::int8_t* xp = x_codes + static_cast<std::int64_t>(col0 + c) * k +
                                        static_cast<std::int64_t>(chunk) * kT2Dp4aChunkK;
                const uint4 lo = load_ldg<uint4>(xp);
                const uint4 hi = load_ldg<uint4>(xp + 16);
                const float xs = __half2float(x_scales[(chunk >> 1) * kT2I8MaxColumns + col0 + c]);
                unsigned xq[8];
                xq[0] = __byte_perm(lo.x, lo.y, 0x6420);
                xq[1] = __byte_perm(lo.x, lo.y, 0x7531);
                xq[2] = __byte_perm(lo.z, lo.w, 0x6420);
                xq[3] = __byte_perm(lo.z, lo.w, 0x7531);
                xq[4] = __byte_perm(hi.x, hi.y, 0x6420);
                xq[5] = __byte_perm(hi.x, hi.y, 0x7531);
                xq[6] = __byte_perm(hi.z, hi.w, 0x6420);
                xq[7] = __byte_perm(hi.z, hi.w, 0x7531);
#pragma unroll
                for (int r = 0; r < Rows; ++r) {
                    // The int32 sum rides on the bit pattern of 1.5 * 2^23, so the float it
                    // decodes to is 1.5 * 2^23 + sum exactly (|sum| <= 32 * 127 < 2^22): one FADD
                    // instead of a quarter-rate I2F.
                    int sum = kT2Dp4aMagic;
#pragma unroll
                    for (int j = 0; j < 8; ++j) {
                        sum = __dp4a(static_cast<int>(wq[r][j]), static_cast<int>(xq[j]), sum);
                    }
                    const float value = __int_as_float(sum) - 12582912.0F;
                    acc[r][c]         = fmaf(value, ws[r] * xs, acc[r][c]);
                }
            }
        }
    }

#pragma unroll
    for (int r = 0; r < Rows; ++r) {
#pragma unroll
        for (int c = 0; c < Columns; ++c) {
            float v = acc[r][c];
#pragma unroll
            for (int mask = 16; mask > 0; mask >>= 1) {
                v += __shfl_xor_sync(0xffffffffu, v, mask);
            }
            if (lane == ((r * Columns + c) & 31) && c < cols) { epilogue(row0 + r, col0 + c, v); }
        }
    }
}

} // namespace ninfer::ops::detail
