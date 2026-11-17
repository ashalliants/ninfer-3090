#pragma once

// Short-prompt GEMM on the Volta tensor cores: 9..32 tokens per pass over the quantized weights.
//
// Below the cuBLAS prefill route's width gate (kCublasPrefillNarrowMinTokens: its weight-sized
// FP16 dequantise only pays from ~100 tokens), sm_70 ran 9..32-token prompts as passes of eight
// through the streaming GEMV, re-reading every weight per pass at ~40 ms per pass of the 27B. Here
// a CTA dequantises a tile of weight rows into shared memory once per 64-wide K step and multiplies
// it against all of the pass's tokens with WMMA (m16n16k16, FP16 in, FP32 accumulation), so the
// weights are read once per pass and the per-token work is tensor-core math. A pass of fewer
// tokens than its 16- or 32-token tile has zeros for the rest, which are not written. The weight
// tile's raw codes are loaded kTileGemmDepth steps ahead, the activation one step ahead.
//
// Numerics are the cuBLAS FP16 prefill route's (ops/linear_swiglu/q4cublas/w4_cublas_prefill.cu):
// each weight is the FP16 rounding of its exact integer times its group's FP16 scale (__hmul2 is
// correctly rounded, so the same value dequantise_row_to_fp16 writes), products accumulate in FP32.
// The activation is staged as FP16 without the route's per-token power-of-two scaling, as the
// tensor-core streaming GEMV stages it (stream_hmma.cuh): exact for BF16 values in FP16's normal
// range.
//
// K order: a format decodes each packed word of eight k into FP16 pairs (k 8w + i, 8w + i + 4),
// i = 0..3 -- the pairs its nibble layout yields with one mask per pair -- and the activation is
// staged in the same permuted order, so the dot products are unchanged.
//
// Work split: a CTA of 8 warps owns TileRows weight rows (whole items: an item's Rows rows are
// consecutive tile rows) and all Cols tokens. Its warps form `Groups` groups that split the K
// steps between them (step s to group s % Groups) for matrices too short to fill the card with
// row tiles alone; each group's warps own 16 x 16 output fragments of the tile. The groups' sums
// are added in a fixed order in shared memory and every item is finished through the family's
// Sink, like the streaming GEMVs.

#include "core/device.h"
#include "ops/common/math.h"
#include "ops/common/smem_limit.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <mma.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>

namespace ninfer::ops::detail {

#if defined(NINFER_SM70)
inline constexpr bool kStreamTileGemmDefault = true;
#else
inline constexpr bool kStreamTileGemmDefault = false;
#endif

// NINFER_STREAM_TILE_GEMM=0 turns it off at runtime in sm_70 builds (for A/B measurement).
inline bool stream_tile_gemm_enabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("NINFER_STREAM_TILE_GEMM");
        if (value == nullptr || *value == '\0') return kStreamTileGemmDefault;
        return std::strcmp(value, "0") != 0;
    }();
    return enabled;
}

template <class Format>
concept StreamTileGemmFormat = requires { requires Format::kTileGemm; };

inline constexpr int kTileGemmWarps   = 8;
inline constexpr int kTileGemmThreads = kTileGemmWarps * 32;
inline constexpr int kTileGemmStepK   = 64;               // one quantization group
inline constexpr int kTileGemmLd      = kTileGemmStepK + 8; // halves; staggers rows across banks
// Weight steps in flight per thread (register ring).
inline constexpr int kTileGemmDepth = 2;

template <int TileRows, int Cols, int Groups>
struct TileGemmShape {
    static_assert(TileRows % 16 == 0 && Cols % 16 == 0 && kTileGemmWarps % Groups == 0);
    static constexpr int kGroupWarps = kTileGemmWarps / Groups;
    static constexpr int kFragments  = (TileRows / 16) * (Cols / 16); // per group
    static_assert(kFragments % kGroupWarps == 0);
    static constexpr int kWarpFragments = kFragments / kGroupWarps;
    // A warp's fragments: kWarpCols token blocks of kWarpRows 16-row blocks, so each A fragment is
    // loaded once for kWarpCols products and each B fragment once for kWarpRows.
    static constexpr int kWarpCols = kWarpFragments < Cols / 16 ? kWarpFragments : Cols / 16;
    static constexpr int kWarpRows = kWarpFragments / kWarpCols;
    static_assert((Cols / 16) % kWarpCols == 0 && kWarpRows * kWarpCols == kWarpFragments);
    // Per group: the dequantised weight tile and the staged activation of one step.
    static constexpr int kWeightHalves = TileRows * kTileGemmLd;
    static constexpr int kInputHalves  = Cols * kTileGemmLd;
    static constexpr std::size_t kStageBytes =
        static_cast<std::size_t>(Groups) * (kWeightHalves + kInputHalves) * sizeof(__half);
    // The groups' FP32 sums, [group][tile row][token] (reuses the stage).
    static constexpr std::size_t kReduceBytes =
        static_cast<std::size_t>(Groups) * TileRows * Cols * sizeof(float);
    static constexpr std::size_t kSharedBytes = kStageBytes > kReduceBytes ? kStageBytes
                                                                           : kReduceBytes;
    // Weight half-groups (32 k of one row) per group per step, and per thread of the group.
    static constexpr int kGroupThreads = kGroupWarps * 32;
    static constexpr int kHalves       = TileRows * 2;
    static constexpr int kHalvesPerThread = (kHalves + kGroupThreads - 1) / kGroupThreads;
    // Activation vectors (eight k of one token) per group per step, and per thread of the group.
    static constexpr int kInputVectors   = Cols * (kTileGemmStepK / 8);
    static constexpr int kInputPerThread = (kInputVectors + kGroupThreads - 1) / kGroupThreads;
};

// Eight BF16 activations (k 8w .. 8w + 7) as FP16 pairs in the formats' order: (x_i, x_{i+4}).
__device__ __forceinline__ uint4 tile_gemm_stage_input(uint4 raw) {
    const std::uint32_t w[4] = {raw.x, raw.y, raw.z, raw.w};
    float f[8];
#pragma unroll
    for (int e = 0; e < 4; ++e) {
        f[2 * e]     = __uint_as_float(w[e] << 16);
        f[2 * e + 1] = __uint_as_float(w[e] & 0xffff0000u);
    }
    std::uint32_t h[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const __half2 v = __floats2half2_rn(f[i], f[i + 4]);
        h[i]            = *reinterpret_cast<const std::uint32_t*>(&v);
    }
    return make_uint4(h[0], h[1], h[2], h[3]);
}

// CTAs per SM: three for a 16-token tile (its registers capped to fit, with the shared-memory
// carveout raised to hold three stages) -- 6144 x 5120 at 16 tokens 94 -> 73 us; a 32-token tile
// spills under that cap and keeps two.
template <int Cols>
inline constexpr int kTileGemmBlocksPerSm = Cols <= 16 ? 3 : 2;

template <class Format, int Rows, int Cols, int TileRows, int Groups, class Sink>
__global__ __launch_bounds__(kTileGemmThreads, kTileGemmBlocksPerSm<Cols>) void stream_tile_gemm_kernel(
    const __nv_bfloat16* __restrict__ x, std::int64_t x_stride, typename Format::Planes planes,
    Sink sink, std::int32_t items, std::int32_t row_offset, std::int32_t k,
    std::int32_t padded_k) {
    using namespace nvcuda;
    // Cols tokens (9..32) on a tile of kTileCols: tokens past Cols are zero and not written.
    constexpr int kTileCols = Cols <= 16 ? 16 : 32;
    using Shape = TileGemmShape<TileRows, kTileCols, Groups>;
    constexpr int kItemsPerTile = TileRows / Rows;
    extern __shared__ __align__(16) unsigned char tile_gemm_shared[];

    const int warp       = static_cast<int>(threadIdx.x) >> 5;
    const int group      = warp / Shape::kGroupWarps;
    const int group_warp = warp % Shape::kGroupWarps;
    const int group_tid  = static_cast<int>(threadIdx.x) - group * Shape::kGroupThreads;
    // A group's stage is its own: its warps meet at a named barrier (ids 1..Groups), the CTA only
    // for the final reduction.
    const auto group_sync = [&] {
        if constexpr (Groups == 1) {
            __syncthreads();
        } else if constexpr (Shape::kGroupWarps == 1) {
            __syncwarp();
        } else {
            asm volatile("bar.sync %0, %1;" ::"r"(group + 1), "r"(Shape::kGroupThreads));
        }
    };
    auto* weights = reinterpret_cast<__half*>(tile_gemm_shared) +
                    group * (Shape::kWeightHalves + Shape::kInputHalves);
    auto* inputs  = weights + Shape::kWeightHalves;

    const int item0  = static_cast<int>(blockIdx.x) * kItemsPerTile;
    const int steps  = k / kTileGemmStepK;
    const int rounds = div_up(steps, Groups);

    // This thread's weight half-groups: tile row h / 2, k half h % 2 of each step.
    typename Format::GemmRow rows_of[Shape::kHalvesPerThread];
    bool live_half[Shape::kHalvesPerThread];
#pragma unroll
    for (int j = 0; j < Shape::kHalvesPerThread; ++j) {
        const int h    = group_tid + j * Shape::kGroupThreads;
        const int t    = h >> 1;
        const int item = item0 + t / Rows;
        live_half[j]   = h < Shape::kHalves && item < items;
        const int row  = live_half[j] ? item + (t % Rows) * row_offset : 0;
        rows_of[j]     = Format::gemm_row(planes, row, padded_k);
    }

    // The next kTileGemmDepth rounds' weights are in flight in registers while one multiplies.
    typename Format::GemmRaw pending[kTileGemmDepth][Shape::kHalvesPerThread];
    const auto fetch = [&](typename Format::GemmRaw (&slot)[Shape::kHalvesPerThread], int step) {
#pragma unroll
        for (int j = 0; j < Shape::kHalvesPerThread; ++j) {
            const int h = group_tid + j * Shape::kGroupThreads;
            slot[j]     = live_half[j] ? Format::gemm_fetch(rows_of[j], step, h & 1)
                                       : Format::gemm_empty();
        }
    };

    // This warp's fragments (Shape::kWarpRows x kWarpCols): fragment f is row block tm0 + f / cols,
    // token block tn0 + f % cols.
    const int warp_blocks_n = kTileCols / 16 / Shape::kWarpCols; // warps across the tokens
    const int tm0 = (group_warp / warp_blocks_n) * Shape::kWarpRows;
    const int tn0 = (group_warp % warp_blocks_n) * Shape::kWarpCols;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[Shape::kWarpFragments];
#pragma unroll
    for (int f = 0; f < Shape::kWarpFragments; ++f) wmma::fill_fragment(acc[f], 0.0f);

    // Round r multiplies step r * Groups + group.
    const auto round_step = [&](int round) { return round * Groups + group; };
    // The activation of the next round's step, likewise loaded a round ahead.
    uint4 pending_x[Shape::kInputPerThread];
    const auto fetch_x = [&](int step) {
        const int k0 = step * kTileGemmStepK;
#pragma unroll
        for (int j = 0; j < Shape::kInputPerThread; ++j) {
            const int i = group_tid + j * Shape::kGroupThreads;
            if (i < Shape::kInputVectors) {
                const int col = i / (kTileGemmStepK / 8);
                const int v   = i % (kTileGemmStepK / 8);
                pending_x[j] = col < Cols ? __ldg(reinterpret_cast<const uint4*>(
                                                x + col * x_stride + k0) + v)
                                          : make_uint4(0u, 0u, 0u, 0u);
            }
        }
    };
    if (group < steps) fetch_x(group);
#pragma unroll
    for (int d = 0; d < kTileGemmDepth; ++d) {
        if (round_step(d) < steps) fetch(pending[d], round_step(d));
    }
    for (int base = 0; base < rounds; base += kTileGemmDepth) {
#pragma unroll
        for (int d = 0; d < kTileGemmDepth; ++d) {
            const int round = base + d;
            if (round >= rounds) break; // uniform across the CTA
            const int step    = round_step(round);
            const bool active = step < steps;
            if (round > 0) group_sync(); // the group's warps are done with the previous step's tile
            if (active) {
#pragma unroll
                for (int j = 0; j < Shape::kHalvesPerThread; ++j) {
                    const int h = group_tid + j * Shape::kGroupThreads;
                    if (h < Shape::kHalves) {
                        __half2 w[16];
                        Format::gemm_decode(pending[d][j], w);
                        auto* dst = reinterpret_cast<uint4*>(weights + (h >> 1) * kTileGemmLd +
                                                             (h & 1) * 32);
#pragma unroll
                        for (int q = 0; q < 4; ++q) {
                            dst[q] = make_uint4(
                                *reinterpret_cast<const std::uint32_t*>(&w[4 * q]),
                                *reinterpret_cast<const std::uint32_t*>(&w[4 * q + 1]),
                                *reinterpret_cast<const std::uint32_t*>(&w[4 * q + 2]),
                                *reinterpret_cast<const std::uint32_t*>(&w[4 * q + 3]));
                        }
                    }
                }
                if (round_step(round + kTileGemmDepth) < steps) {
                    fetch(pending[d], round_step(round + kTileGemmDepth));
                }
#pragma unroll
                for (int j = 0; j < Shape::kInputPerThread; ++j) {
                    const int i = group_tid + j * Shape::kGroupThreads;
                    if (i < Shape::kInputVectors) {
                        const int col = i / (kTileGemmStepK / 8);
                        const int v   = i % (kTileGemmStepK / 8);
                        *reinterpret_cast<uint4*>(inputs + col * kTileGemmLd + 8 * v) =
                            tile_gemm_stage_input(pending_x[j]);
                    }
                }
                if (step + Groups < steps) fetch_x(step + Groups);
            }
            group_sync();
            if (active) {
#pragma unroll
                for (int kk = 0; kk < kTileGemmStepK; kk += 16) {
                    wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major>
                        b[Shape::kWarpCols];
#pragma unroll
                    for (int c = 0; c < Shape::kWarpCols; ++c) {
                        wmma::load_matrix_sync(b[c], inputs + (tn0 + c) * 16 * kTileGemmLd + kk,
                                               kTileGemmLd);
                    }
#pragma unroll
                    for (int r = 0; r < Shape::kWarpRows; ++r) {
                        wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a;
                        wmma::load_matrix_sync(a, weights + (tm0 + r) * 16 * kTileGemmLd + kk,
                                               kTileGemmLd);
#pragma unroll
                        for (int c = 0; c < Shape::kWarpCols; ++c) {
                            wmma::mma_sync(acc[r * Shape::kWarpCols + c], a, b[c],
                                           acc[r * Shape::kWarpCols + c]);
                        }
                    }
                }
            }
        }
    }

    __syncthreads();
    // The groups' sums, [group][tile row][token], added in group order.
    auto* reduce = reinterpret_cast<float*>(tile_gemm_shared);
#pragma unroll
    for (int f = 0; f < Shape::kWarpFragments; ++f) {
        const int tm = tm0 + f / Shape::kWarpCols;
        const int tn = tn0 + f % Shape::kWarpCols;
        wmma::store_matrix_sync(reduce + (group * TileRows + tm * 16) * kTileCols + tn * 16,
                                acc[f], kTileCols, wmma::mem_row_major);
    }
    __syncthreads();
    for (int t = static_cast<int>(threadIdx.x); t < kItemsPerTile; t += kTileGemmThreads) {
        const int item = item0 + t;
        if (item >= items) continue;
        float sum[Rows][Cols];
#pragma unroll
        for (int r = 0; r < Rows; ++r) {
#pragma unroll
            for (int c = 0; c < Cols; ++c) {
                float s = 0.0f;
#pragma unroll
                for (int g = 0; g < Groups; ++g) {
                    s += reduce[(g * TileRows + t * Rows + r) * kTileCols + c];
                }
                sum[r][c] = s;
            }
        }
        sink.finish(item, sum);
    }
}

// Whether this build's tile GEMM serves the format at all (sm_70 builds, Q4 and Q5).
template <class Format>
constexpr bool stream_tile_gemm_format() {
    if constexpr (!kStreamTileGemmDefault) {
        return false;
    } else {
        return StreamTileGemmFormat<Format>;
    }
}

// Whether try_launch_stream_tile_gemm takes this format and shape (any Cols it serves).
template <class Format>
bool stream_tile_gemm_supported(std::int32_t k, std::int32_t padded_k, std::int64_t x_stride) {
    if constexpr (!kStreamTileGemmDefault || !StreamTileGemmFormat<Format>) {
        return false;
    } else {
        return stream_tile_gemm_enabled() && k % kTileGemmStepK == 0 &&
               padded_k % kTileGemmStepK == 0 && x_stride % 8 == 0;
    }
}

// Launches the short-prompt tensor-core GEMM for exactly Cols tokens, 9..32 (on a 16- or 32-token
// tile), if it is enabled and the format and shape fit; returns false otherwise.
template <class Format, int Rows, int Cols, class Sink>
bool try_launch_stream_tile_gemm(const __nv_bfloat16* x, std::int64_t x_stride,
                                 const typename Format::Planes& planes, Sink sink,
                                 std::int32_t items, std::int32_t row_offset, std::int32_t k,
                                 std::int32_t padded_k, cudaStream_t stream) {
    if constexpr (!kStreamTileGemmDefault || !StreamTileGemmFormat<Format> ||
                  Cols < 9 || Cols > 32) {
        return false;
    } else {
        if (!stream_tile_gemm_supported<Format>(k, padded_k, x_stride)) return false;
        int device = 0;
        int sms    = 0;
        CUDA_CHECK(cudaGetDevice(&device));
        CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device));
        const std::int64_t rows_total = static_cast<std::int64_t>(items) * Rows;
        const auto launch = [&]<int TileRows, int Groups>() {
            using Shape       = TileGemmShape<TileRows, (Cols <= 16 ? 16 : 32), Groups>;
            const auto kernel = stream_tile_gemm_kernel<Format, Rows, Cols, TileRows, Groups, Sink>;
            configure_cuda_device_once([&] {
                const cudaError_t e = cudaFuncSetAttribute(
                    kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                    static_cast<int>(Shape::kSharedBytes));
                if (e != cudaSuccess || kTileGemmBlocksPerSm<Cols> < 3) return e;
                return cudaFuncSetAttribute(kernel, cudaFuncAttributePreferredSharedMemoryCarveout,
                                            cudaSharedmemCarveoutMaxShared);
            });
            const int blocks = static_cast<int>(div_up(items, TileRows / Rows));
            kernel<<<blocks, kTileGemmThreads, Shape::kSharedBytes, stream>>>(
                x, x_stride, planes, sink, items, row_offset, k, padded_k);
            CUDA_CHECK(cudaGetLastError());
            return true;
        };
        // Measured on the 27B shapes (V100-PCIE-32GB): short tiles whose four warp groups split K
        // win everywhere at 16 tokens; at 32 only the 34816-row SwiGLU matrix fills the card
        // better with tall tiles (473 against 511 us).
        if (Cols > 16 && rows_total >= static_cast<std::int64_t>(2 * sms) * 128) {
            return launch.template operator()<128, 1>();
        }
        return launch.template operator()<32, 4>();
    }
}

} // namespace ninfer::ops::detail
