#pragma once
#include "ops/common/smem_limit.h"
#include "core/device.h"
#include "ops/common/mma.cuh"
#include "ops/linear/bf16/bf16_schedule.cuh"
#include "ops/linear/common/epilogue.cuh"

namespace ninfer::ops::detail {

template <int Bytes, auto Kernel>
int bf16_prepare_shared() {
    static_assert(Bytes <= kMaxDynamicSharedBytes);
    if constexpr (Bytes > 48 * 1024) {
        // Per device: function attributes do not carry over to a second GPU.
        configure_cuda_device_once([] {
            return cudaFuncSetAttribute(Kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        Bytes);
        });
    }
    return Bytes;
}

template <class Schedule, class Epilogue>
inline constexpr int bf16_epilogue_bytes = [] {
    if constexpr (requires { Epilogue::template kSharedBytes<Schedule>; })
        return Epilogue::template kSharedBytes<Schedule>;
    else
        return 0;
}();

template <class Schedule, class Epilogue>
inline constexpr int bf16_mma_shared_bytes =
    Schedule::kSharedBytes > bf16_epilogue_bytes<Schedule, Epilogue>
        ? Schedule::kSharedBytes
        : bf16_epilogue_bytes<Schedule, Epilogue>;

template <class Schedule>
__device__ __forceinline__ int bf16_mma_shared_col(int row, int col) {
    if constexpr (Schedule::kSwizzle == Bf16MmaSwizzle::Plain) {
        return col;
    } else if constexpr (Schedule::kSwizzle == Bf16MmaSwizzle::Tma128) {
        const int chunk = (row * Schedule::kBlockK + col) / 64;
        return (col & ~63) + ((((col & 63) >> 3) ^ (chunk & 7)) << 3) + (col & 7);
    } else {
        return (col & ~63) + ((((col & 63) >> 3) ^ (row & 7)) << 3) + (col & 7);
    }
}

template <class Schedule>
__device__ __forceinline__ void
bf16_mma_tile_coordinates(std::int32_t linear, std::int32_t tiles_m, std::int32_t tiles_n,
                          std::int32_t& tile_m, std::int32_t& tile_n) {
    if constexpr (Schedule::kRaster == Bf16MmaRaster::TokenFast) {
        tile_m = linear / tiles_n;
        tile_n = linear - tile_m * tiles_n;
    } else if constexpr (Schedule::kRaster == Bf16MmaRaster::RowFast) {
        tile_n = linear / tiles_m;
        tile_m = linear - tile_n * tiles_m;
    } else {
        constexpr int group_rows      = Schedule::kRasterGroupRows;
        const std::int32_t group_span = group_rows * tiles_n;
        const std::int32_t group      = linear / group_span;
        const std::int32_t first_m    = group * group_rows;
        const std::int32_t active_m   = min(group_rows, tiles_m - first_m);
        const std::int32_t within     = linear - group * group_span;
        tile_m                        = first_m + within % active_m;
        tile_n                        = within / active_m;
    }
}

template <class Schedule>
__device__ __forceinline__ void
bf16_mma_compute_stage(const __nv_bfloat16* As, const __nv_bfloat16* Bs,
                       float (&accum)[Schedule::kMmaRows][Schedule::kMmaTokens][4], int warp,
                       int lane) {
    constexpr int BK = Schedule::kBlockK, WM = Schedule::kWarpRows, WN = Schedule::kWarpTokens;
    constexpr int MT = Schedule::kMmaRows, NT = Schedule::kMmaTokens, KSUB = BK / 16;
    const int wm = warp / Schedule::kWarpsTokens, wn = warp % Schedule::kWarpsTokens;
    const int a_matrix = lane >> 3, a_inner_row = lane & 7;
    const int a_row_offset = a_inner_row + ((a_matrix & 1) << 3);
    const int a_col_offset = (a_matrix >> 1) << 3;
    const int b_inner_row = lane & 7, b_k_offset = ((lane >> 3) & 1) << 3;
    auto load_fragments = [&](int k_step, unsigned(&a_frag)[MT][4], unsigned(&b_frag)[NT][2]) {
#pragma unroll
        for (int mi = 0; mi < MT; ++mi) {
            const int row = wm * WM + mi * 16 + a_row_offset;
            const int col = k_step * 16 + a_col_offset;
            ldmatrix_x4(a_frag[mi][0], a_frag[mi][1], a_frag[mi][2], a_frag[mi][3],
                        smem_addr(&As[row * BK + bf16_mma_shared_col<Schedule>(row, col)]));
        }
#pragma unroll
        for (int ni = 0; ni < NT; ++ni) {
            const int row = wn * WN + ni * 8 + b_inner_row;
            const int col = k_step * 16 + b_k_offset;
            ldmatrix_x2(b_frag[ni][0], b_frag[ni][1],
                        smem_addr(&Bs[row * BK + bf16_mma_shared_col<Schedule>(row, col)]));
        }
    };

#if defined(NINFER_SM70)
    // sm_70 has no BF16 tensor cores: the emulated ldmatrix and m16n8k16 (volta_compat.cuh) spend
    // ~44 shuffles per 64 FMAs a lane does. Each thread instead reads its own fragment rows
    // (wm WM + mi 16 + g + 8 rh) and tokens (wn WN + ni 8 + 2 t + j) from the swizzled tiles, 16
    // bytes per 8-value chunk, and runs each output's FMAs in the emulation's order: per k16 step,
    // for tp = 0..3, k = 2 tp, 2 tp + 1, 2 tp + 8, 2 tp + 9, as fmaf(weight, activation, acc).
    // Same operands, same order: bit for bit the emulated mma.
    if constexpr (true) {
        (void)a_row_offset; (void)a_col_offset; (void)b_inner_row; (void)b_k_offset;
        const int g = lane >> 2, t = lane & 3;
        const auto chunk = [&](const __nv_bfloat16* tile, int row, int col) {
            return *reinterpret_cast<const uint4*>(&tile[row * BK + bf16_mma_shared_col<Schedule>(row, col)]);
        };
        const auto word = [](const uint4& v, int i) { return i == 0 ? v.x : i == 1 ? v.y : i == 2 ? v.z : v.w; };
        // Rolled: unrolled, the hoisted loads of every k step spill under the 80-register schedules.
#pragma unroll 1
        for (int k_step = 0; k_step < KSUB; ++k_step) {
#pragma unroll
            for (int mi = 0; mi < MT; ++mi) {
                uint4 ar[2][2]; // [rh][chunk]
#pragma unroll
                for (int rh = 0; rh < 2; ++rh) {
                    const int row = wm * WM + mi * 16 + g + 8 * rh;
                    ar[rh][0]     = chunk(As, row, k_step * 16);
                    ar[rh][1]     = chunk(As, row, k_step * 16 + 8);
                }
#pragma unroll
                for (int ni = 0; ni < NT; ++ni) {
                    uint4 bt[2][2]; // [j][chunk]
#pragma unroll
                    for (int j = 0; j < 2; ++j) {
                        const int tok = wn * WN + ni * 8 + 2 * t + j;
                        bt[j][0]      = chunk(Bs, tok, k_step * 16);
                        bt[j][1]      = chunk(Bs, tok, k_step * 16 + 8);
                    }
#pragma unroll
                    for (int e = 0; e < 4; ++e) {
                        const uint4& w0 = ar[e >> 1][0];
                        const uint4& w1 = ar[e >> 1][1];
                        const uint4& x0 = bt[e & 1][0];
                        const uint4& x1 = bt[e & 1][1];
                        float acc       = accum[mi][ni][e];
#pragma unroll
                        for (int tp = 0; tp < 4; ++tp) {
                            const unsigned a0 = word(w0, tp), b0 = word(x0, tp), a2 = word(w1, tp), b1 = word(x1, tp);
                            acc = fmaf(__uint_as_float(a0 << 16), __uint_as_float(b0 << 16), acc);
                            acc = fmaf(__uint_as_float(a0 & 0xFFFF0000U), __uint_as_float(b0 & 0xFFFF0000U), acc);
                            acc = fmaf(__uint_as_float(a2 << 16), __uint_as_float(b1 << 16), acc);
                            acc = fmaf(__uint_as_float(a2 & 0xFFFF0000U), __uint_as_float(b1 & 0xFFFF0000U), acc);
                        }
                        accum[mi][ni][e] = acc;
                    }
                }
            }
        }
    } else
#endif
    if constexpr (Schedule::kFragmentPipeline == Bf16MmaFragmentPipeline::PingPong) {
        unsigned a_frag[2][MT][4];
        unsigned b_frag[2][NT][2];
        load_fragments(0, a_frag[0], b_frag[0]);
#pragma unroll
        for (int k_step = 0; k_step < KSUB; ++k_step) {
            const int slot = k_step & 1;
            if (k_step + 1 < KSUB) {
                load_fragments(k_step + 1, a_frag[slot ^ 1], b_frag[slot ^ 1]);
            }
#pragma unroll
            for (int mi = 0; mi < MT; ++mi) {
#pragma unroll
                for (int ni = 0; ni < NT; ++ni) {
                    mma_bf16(accum[mi][ni][0], accum[mi][ni][1], accum[mi][ni][2], accum[mi][ni][3],
                             a_frag[slot][mi][0], a_frag[slot][mi][1], a_frag[slot][mi][2],
                             a_frag[slot][mi][3], b_frag[slot][ni][0], b_frag[slot][ni][1]);
                }
            }
        }
    } else {
        unsigned a_frag[MT][4];
        unsigned b_frag[NT][2];
#pragma unroll
        for (int k_step = 0; k_step < KSUB; ++k_step) {
            load_fragments(k_step, a_frag, b_frag);
#pragma unroll
            for (int mi = 0; mi < MT; ++mi) {
#pragma unroll
                for (int ni = 0; ni < NT; ++ni) {
                    mma_bf16(accum[mi][ni][0], accum[mi][ni][1], accum[mi][ni][2], accum[mi][ni][3],
                             a_frag[mi][0], a_frag[mi][1], a_frag[mi][2], a_frag[mi][3],
                             b_frag[ni][0], b_frag[ni][1]);
                }
            }
        }
    }
}

template <bool FullTokens, class Output, class Epilogue, bool FullRows = true>
__device__ __forceinline__ void bf16_finish_fragment(const Output& output, const Epilogue& epilogue,
                                                     int row, int token, float4 value, int rows,
                                                     int tokens) {
    if constexpr (requires { epilogue.store_fragment(output, row, token, value, rows, tokens); }) {
        static_assert(FullRows, "row-tail MMA requires a scalar epilogue");
        // A fragment consumer receives the atom origin and owns its token-tail predicates.
        epilogue.store_fragment(output, row, token, value, rows, tokens);
    } else {
        if (FullTokens || token < tokens) {
            if (FullRows || row < rows)
                output.store(row, token, epilogue.apply(row, token, value.x));
            if (FullRows || row + 8 < rows)
                output.store(row + 8, token, epilogue.apply(row + 8, token, value.z));
        }
        if (FullTokens || token + 1 < tokens) {
            if (FullRows || row < rows)
                output.store(row, token + 1, epilogue.apply(row, token + 1, value.y));
            if (FullRows || row + 8 < rows)
                output.store(row + 8, token + 1, epilogue.apply(row + 8, token + 1, value.w));
        }
    }
}

template <class Schedule, bool FullTokens, class Output, class Epilogue>
__device__ __forceinline__ void
bf16_finish_mma_tile(const Output& output, const Epilogue& epilogue, unsigned char* scratch,
                     float (&accum)[Schedule::kMmaRows][Schedule::kMmaTokens][4], int row_begin,
                     int token_begin, int rows, int tokens, int warp, int lane) {
    const auto destination = linear_output_tile<Schedule::kBlockRows>(output, row_begin);
    if constexpr (requires {
                      epilogue.template finish_tile<Schedule, FullTokens>(
                          destination, scratch, accum, row_begin, token_begin, rows, tokens);
                  }) {
        static_assert(!bf16_predicated_rows<Schedule>, "row-tail MMA requires a scalar epilogue");
        // All staging reads must have finished before a collective consumer reuses scratch.
        __syncthreads();
        epilogue.template finish_tile<Schedule, FullTokens>(destination, scratch, accum, row_begin,
                                                            token_begin, rows, tokens);
    } else {
        const int wm = warp / Schedule::kWarpsTokens, wn = warp % Schedule::kWarpsTokens;
#pragma unroll
        for (int mi = 0; mi < Schedule::kMmaRows; ++mi) {
            const int row = row_begin + wm * Schedule::kWarpRows + mi * 16 + (lane >> 2);
#pragma unroll
            for (int ni = 0; ni < Schedule::kMmaTokens; ++ni) {
                const int token =
                    token_begin + wn * Schedule::kWarpTokens + ni * 8 + 2 * (lane & 3);
                const auto& v = accum[mi][ni];
                bf16_finish_fragment<FullTokens, decltype(destination), Epilogue,
                                     !bf16_predicated_rows<Schedule>>(
                    destination, epilogue, row, token, make_float4(v[0], v[1], v[2], v[3]), rows,
                    tokens);
            }
        }
    }
}
} // namespace ninfer::ops::detail
