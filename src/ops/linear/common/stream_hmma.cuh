#pragma once

// Streaming GEMV on the Volta tensor cores, for the speculative-verification widths (2..8 tokens).
//
// The SIMT streaming GEMV (stream_gemv.cuh) decodes a weight once and pays one FFMA per token, so
// at T = 4..8 it is bound by issue, not by the weight stream: a Qwen3.8-27B MTP3 verify round
// spent ~26 ms in it against a ~19 ms read of the weights (V100-PCIE-32GB, nsys). Here the format
// decodes each packed word straight into FP16 operand pairs and mma.sync.m8n8k4 does the
// per-token work: about 2 instructions per weight at any T up to 8 instead of 1.5 + T.
//
// Mapping (the quadpair-split-N form; the fragment maps follow geoffwatts/ninfer-v100's
// q4_volta_qpn_gemm.cuh, Apache-2.0, which measured them against a host oracle):
//
//   * m8n8k4 is four independent 8x8x4 products per warp instruction, one per quadpair (lanes
//     4q..4q+3 and 16+4q..16+4q+3). Lane position p = (lane & 3) + 4 * ((lane >> 4) & 1) inside
//     its quadpair is both the A row it supplies (token p) and the B column (weight row
//     q * 8 + p of the warp's 32-row tile). So each lane streams one weight row, sequentially,
//     and the four quadpairs read the same activation (an LDS broadcast).
//   * A row-major A fragment is four consecutive k of one token and a column-major B fragment is
//     four consecutive k of one weight row, so a format's packed word of eight k is two MMAs.
//   * Accumulator register i of lane L holds token (i & 2) | 4 * ((L >> 4) & 1) | (L & 1) and
//     quadpair-local row (i & 1) | 2 * ((L >> 1) & 1) | 4 * (i >> 2).
//
// Numerics: the formats decode to exact small integers (exact in FP16); each 32-weight slice is
// summed in FP32 by the MMA and scaled by its row's group scale in FP32 afterwards, the scale of
// the other rows a lane holds coming from its quadpair over four shuffles. The activation is
// staged per CTA as FP16, which is exact for every BF16 value in FP16's normal range (2^-14 to
// 65504: activations entering these GEMVs are normalised or gated); smaller magnitudes keep
// FP16's subnormal precision and larger ones saturate rather than overflow.
//
// Work split: a CTA's warps split each 32-row tile's K and reduce through shared memory, and the
// reduced rows go through the family's Sink (StreamGemvRowSink) like the SIMT kernel's, so every
// Output / Epilogue works unchanged. K that does not fit the activation stage is processed in
// chunks with per-tile partial sums, as in stream_gemv.cuh.
//
// A format opts in with `kHmma = true` and `decode_half2<W>(slice, b)`: the four FP16 pairs
// (k 8W + 2j, 8W + 2j + 1), j = 0..3, of packed word W, as exact integers.

#include "core/device.h"
#include "ops/common/math.h"
#include "ops/common/smem_limit.h"
#include "ops/linear/common/epilogue.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>

namespace ninfer::ops::detail {

#if defined(NINFER_SM70)
inline constexpr bool kStreamHmmaDefault = true;
#else
inline constexpr bool kStreamHmmaDefault = false;
#endif

// NINFER_STREAM_HMMA=0 turns it off at runtime in sm_70 builds, for A/B measurement.
inline bool stream_hmma_enabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("NINFER_STREAM_HMMA");
        if (value == nullptr || *value == '\0') return kStreamHmmaDefault;
        return std::strcmp(value, "0") != 0;
    }();
    return enabled;
}

// The narrowest verify width that takes the tensor cores (see the route in
// try_launch_stream_hmma_rows): at one token the SIMT kernel's single FFMA per weight is already
// below the weight stream.
inline constexpr int kStreamHmmaMinCols = 4;

template <class Format>
concept StreamHmmaFormat = requires { requires Format::kHmma; };

inline constexpr int kStreamHmmaGroups     = 4;  // tiles a CTA works on at once
inline constexpr int kStreamHmmaGroupWarps = 4;  // warps splitting one tile's K
inline constexpr int kStreamHmmaWarps      = kStreamHmmaGroups * kStreamHmmaGroupWarps;
inline constexpr int kStreamHmmaTileRows   = 32; // weight rows per tile: one per lane
inline constexpr int kStreamHmmaStagePad   = 8;  // halves; staggers the 8 token rows across banks
inline constexpr int kStreamHmmaStepK      = 128; // k per load step: two 64-weight groups

// Lanes `lane` and `lane ^ 1` read 32 contiguous bytes of two rows -- `even` for the even lane's
// row, `odd` for the odd lane's -- each lane one 16-byte half of each, so every 32-byte sector is
// read whole by a single instruction (stream_hmma_pair_fetch), and then swap halves so each lane
// holds its own row's 32 bytes (stream_hmma_pair_swap). A format issues every fetch of a step
// before the first swap: a shuffle waits for its operand, and the compiler does not move loads
// across it, so interleaving them would serialise the step's memory latencies.
struct StreamHmmaPair {
    uint4 from_even;
    uint4 from_odd;
};
__device__ __forceinline__ StreamHmmaPair stream_hmma_pair_fetch(const uint4* even,
                                                                 const uint4* odd, bool is_odd) {
    return {__ldg(even + (is_odd ? 1 : 0)), __ldg(odd + (is_odd ? 1 : 0))};
}
__device__ __forceinline__ void stream_hmma_pair_swap(const StreamHmmaPair& pair, bool is_odd,
                                                      uint4& lo, uint4& hi) {
    const uint4 send = is_odd ? pair.from_even : pair.from_odd;
    uint4 recv;
    recv.x = __shfl_xor_sync(0xffffffffu, send.x, 1);
    recv.y = __shfl_xor_sync(0xffffffffu, send.y, 1);
    recv.z = __shfl_xor_sync(0xffffffffu, send.z, 1);
    recv.w = __shfl_xor_sync(0xffffffffu, send.w, 1);
    lo     = is_odd ? recv : pair.from_even;
    hi     = is_odd ? pair.from_odd : recv;
}

__device__ __forceinline__ void stream_hmma_m8n8k4(float (&d)[8], unsigned a0, unsigned a1,
                                                   unsigned b0, unsigned b1) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
    asm volatile("mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "
                 "{%0, %1, %2, %3, %4, %5, %6, %7}, {%8, %9}, {%10, %11}, "
                 "{%0, %1, %2, %3, %4, %5, %6, %7};"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]),
                   "+f"(d[6]), "+f"(d[7])
                 : "r"(a0), "r"(a1), "r"(b0), "r"(b1));
#endif
}

// Shared memory for one K-chunk: the FP16 activation of Cols tokens, the cross-warp reduction
// buffer (per group: [warp][tile row][token]), and (chunked) the partial sums of `tiles` tiles.
__host__ __device__ constexpr std::size_t stream_hmma_stage_bytes(std::int32_t chunk_k, int cols) {
    return static_cast<std::size_t>(cols) *
           static_cast<std::size_t>(chunk_k + kStreamHmmaStagePad) * sizeof(__half);
}
__host__ __device__ constexpr std::size_t stream_hmma_reduce_bytes() {
    return static_cast<std::size_t>(kStreamHmmaWarps) * kStreamHmmaTileRows * 8 * sizeof(float);
}
__host__ __device__ constexpr std::size_t stream_hmma_partial_bytes(std::int32_t tiles, int cols) {
    return static_cast<std::size_t>(tiles) * kStreamHmmaTileRows * cols * sizeof(float);
}

template <class Format, int Rows, int Cols, bool Chunked, class Sink>
__global__ __launch_bounds__(kStreamHmmaWarps * 32, 1) void stream_hmma_kernel(
    const __nv_bfloat16* __restrict__ x, std::int64_t x_stride, typename Format::Planes planes,
    Sink sink, std::int32_t items, std::int32_t row_offset, std::int32_t k, std::int32_t padded_k,
    std::int32_t chunk_k, int groups) {
    constexpr int kThreads      = kStreamHmmaWarps * 32;
    constexpr int kItemsPerTile = kStreamHmmaTileRows / Rows; // a paired item's rows are lane pairs
    extern __shared__ __align__(16) unsigned char stream_hmma_shared[];
    const int stage_stride = chunk_k + kStreamHmmaStagePad; // halves per token row
    auto* staged           = reinterpret_cast<__half*>(stream_hmma_shared);
    auto* reduce           = reinterpret_cast<float*>(stream_hmma_shared +
                                                      stream_hmma_stage_bytes(chunk_k, Cols));
    float* partials        = reduce + kStreamHmmaWarps * kStreamHmmaTileRows * 8;

    const int lane       = static_cast<int>(threadIdx.x) & 31;
    const int warp       = static_cast<int>(threadIdx.x) >> 5;
    // `groups` tiles at once (4, 2 or 1: the launcher picks fewer when there are few tiles), each
    // with group_warps warps splitting its K.
    const int group_warps = kStreamHmmaWarps / groups;
    const int group       = warp / group_warps; // the tile this warp works on
    const int group_warp  = warp % group_warps; // its share of that tile's K
    const int quad       = (lane >> 2) & 3;
    const int pos        = (lane & 3) + ((lane >> 4) & 1) * 4; // A row (token) and B column
    const int column     = quad * 8 + pos;                     // this lane's row of the tile
    // Tile row and token of accumulator register i (see the header comment).
    const int row_bit = ((lane >> 1) & 1) << 1;
    const int tok_hi  = ((lane >> 4) & 1) << 2;
    // The lanes that own quadpair-local rows c0 = row_bit, c0 + 1, c0 + 4, c0 + 5.
    const auto owner = [&](int c) { return (c & 3) | ((c & 4) << 2) | (quad << 2); };
    const int src0   = owner(row_bit);
    const int src1   = owner(row_bit + 1);
    const int src4   = owner(row_bit + 4);
    const int src5   = owner(row_bit + 5);
    const bool odd   = (lane & 1) != 0; // pos & 1: the odd row of the lane pair

    const int tiles       = div_up(items, kItemsPerTile);
    const int tile_stride = static_cast<int>(gridDim.x) * groups;
    const int chunk_count = Chunked ? div_up(k, chunk_k) : 1;
    float* group_reduce   = reduce + group * group_warps * kStreamHmmaTileRows * 8;

    using Row = decltype(Format::row(planes, 0, padded_k));
    // The weight row of tile row c: item c / Rows of the tile, its row c % Rows.
    const auto weight_row = [&](int tile, int c, bool& live) {
        const int item = tile * kItemsPerTile + c / Rows;
        live           = item < items;
        return Format::row(planes, live ? item + (c % Rows) * row_offset : 0, padded_k);
    };

    for (int chunk = 0; chunk < chunk_count; ++chunk) {
        const int k0    = chunk * chunk_k;
        const int width = min(chunk_k, k - k0);

        if (chunk > 0) __syncthreads();
        // Stage x[k0, k0 + width) of every token as FP16, eight elements per thread and pass.
        {
            const int vectors = width / 8;
            for (int i = static_cast<int>(threadIdx.x); i < Cols * vectors; i += kThreads) {
                const int col = i / vectors;
                const int j   = i - col * vectors;
                const uint4 raw =
                    __ldg(reinterpret_cast<const uint4*>(x + col * x_stride + k0) + j);
                const std::uint32_t w[4] = {raw.x, raw.y, raw.z, raw.w};
                std::uint32_t h[4];
#pragma unroll
                for (int e = 0; e < 4; ++e) {
                    const __half2 v = __floats2half2_rn(__uint_as_float(w[e] << 16),
                                                        __uint_as_float(w[e] & 0xffff0000u));
                    h[e]            = *reinterpret_cast<const std::uint32_t*>(&v);
                }
                *reinterpret_cast<uint4*>(staged + col * stage_stride + 8 * j) =
                    make_uint4(h[0], h[1], h[2], h[3]);
            }
        }
        __syncthreads();

        constexpr int kStepK = Format::kHmmaGroups * Format::kGroupK;
        const int steps      = width / kStepK;
        const int step_first = steps * group_warp / group_warps;
        const int step_last  = steps * (group_warp + 1) / group_warps;
        const __half* a_row  = staged + (pos < Cols ? pos : 0) * stage_stride;

        int n = 0; // this group's n-th tile
        for (int tile = static_cast<int>(blockIdx.x) * groups + group; tile < tiles;
             tile += tile_stride, ++n) {
            // The lane pair (lane, lane ^ 1) owns tile rows column & ~1 and column | 1 and loads
            // them together, one 16-byte half of each (Format::hmma_fetch), so every sector it
            // requests is read whole by one instruction.
            bool live, even_live, odd_live;
            (void)weight_row(tile, column, live);
            const Row even_row = weight_row(tile, column & ~1, even_live);
            const Row odd_row  = weight_row(tile, column | 1, odd_live);

            float acc[8];
#pragma unroll
            for (int i = 0; i < 8; ++i) acc[i] = 0.0f;

            // Software pipeline: step s + 1's loads are in flight while step s decodes.
            typename Format::HmmaRaw pending;
            const auto fetch = [&](int step) {
                pending = Format::hmma_fetch(even_row, odd_row, odd,
                                             (k0 + step * kStepK) / Format::kGroupK);
            };
            if (step_first < step_last) fetch(step_first);
            for (int step = step_first; step < step_last; ++step) {
                typename Format::HmmaStep loaded = Format::hmma_finish(pending, odd);
                if (step + 1 < step_last) fetch(step + 1);
                const __half* a_step = a_row + step * kStepK;
                // One group (two slices) per iteration: the format shifts the next group into
                // group 0 between iterations (hmma_advance).
#pragma unroll 1
                for (int group_index = 0; group_index < Format::kHmmaGroups; ++group_index) {
#pragma unroll
                for (int slice = 0; slice < 2; ++slice) {
                    float d[8];
#pragma unroll
                    for (int i = 0; i < 8; ++i) d[i] = 0.0f;
#pragma unroll
                    for (int w = 0; w < 4; ++w) {
                        uint4 a = make_uint4(0u, 0u, 0u, 0u);
                        if (pos < Cols) {
                            a = *reinterpret_cast<const uint4*>(a_step + (group_index * 2 + slice) * 32 + 8 * w);
                        }
                        unsigned b[4];
                        Format::hmma_decode(loaded, slice, w, b);
                        stream_hmma_m8n8k4(d, a.x, a.y, b[0], b[1]);
                        stream_hmma_m8n8k4(d, a.z, a.w, b[2], b[3]);
                    }
                    const float own = live ? Format::hmma_scale(loaded, slice) : 0.0f;
                    const float s0  = __shfl_sync(0xffffffffu, own, src0);
                    const float s1  = __shfl_sync(0xffffffffu, own, src1);
                    const float s4  = __shfl_sync(0xffffffffu, own, src4);
                    const float s5  = __shfl_sync(0xffffffffu, own, src5);
                    const float scale[8] = {s0, s1, s0, s1, s4, s5, s4, s5};
#pragma unroll
                    for (int i = 0; i < 8; ++i) acc[i] = fmaf(scale[i], d[i], acc[i]);
                }
                Format::hmma_advance(loaded);
                }
            }

            // The group's cross-warp reduction: group_reduce[group warp][tile row][token].
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const int c     = quad * 8 + ((i & 1) | row_bit | ((i >> 2) << 2));
                const int token = (i & 2) | tok_hi | (lane & 1);
                group_reduce[(group_warp * kStreamHmmaTileRows + c) * 8 + token] = acc[i];
            }
            // Named barrier: the group's four warps only.
            asm volatile("bar.sync %0, %1;" ::"r"(group + 1), "r"(group_warps * 32));
            if (group_warp == 0 && lane < kItemsPerTile) {
                float sum[Rows][Cols];
#pragma unroll
                for (int r = 0; r < Rows; ++r) {
#pragma unroll
                    for (int t = 0; t < Cols; ++t) {
                        float s = 0.0f;
#pragma unroll
                        for (int w = 0; w < group_warps; ++w) {
                            s += group_reduce[(w * kStreamHmmaTileRows + lane * Rows + r) * 8 + t];
                        }
                        sum[r][t] = s;
                    }
                }
                const int out_item = tile * kItemsPerTile + lane;
                if constexpr (!Chunked) {
                    if (out_item < items) sink.finish(out_item, sum);
                } else {
                    float* p = partials +
                               ((n * groups + group) * kItemsPerTile + lane) * Rows * Cols;
#pragma unroll
                    for (int r = 0; r < Rows; ++r) {
#pragma unroll
                        for (int t = 0; t < Cols; ++t) {
                            p[r * Cols + t] =
                                chunk == 0 ? sum[r][t] : p[r * Cols + t] + sum[r][t];
                        }
                    }
                    if (chunk + 1 == chunk_count && out_item < items) {
                        float total[Rows][Cols];
#pragma unroll
                        for (int r = 0; r < Rows; ++r) {
#pragma unroll
                            for (int t = 0; t < Cols; ++t) total[r][t] = p[r * Cols + t];
                        }
                        sink.finish(out_item, total);
                    }
                }
            }
            asm volatile("bar.sync %0, %1;" ::"r"(group + 1), "r"(group_warps * 32));
        }
    }
}

// Launches the tensor-core streaming GEMV (see the header comment) if it is enabled and the shape
// fits; returns false otherwise, and the caller falls back to the SIMT streaming GEMV.
template <class Format, int Rows, int Cols, class Sink>
bool try_launch_stream_hmma_rows(const __nv_bfloat16* x, std::int64_t x_stride,
                                 const typename Format::Planes& planes, Sink sink,
                                 std::int32_t items, std::int32_t row_offset, std::int32_t k,
                                 std::int32_t padded_k, cudaStream_t stream) {
    // Compiled for sm_70 builds only: other cards have native mma routes for these widths.
    if constexpr (!kStreamHmmaDefault || !StreamHmmaFormat<Format> || Cols < kStreamHmmaMinCols ||
                  Cols > 8) {
        return false;
    } else {
        if (!stream_hmma_enabled()) return false;
        // Where it beats the SIMT kernel (Q4, V100-PCIE-32GB, us SIMT / tensor cores):
        //
        //   shape (rows x K)        T=3          T=4          T=5          T=6          T=8
        //   34816 x 5120 (SwiGLU)  163 / 190    195 / 190    252 / 237    343 / 226    503 / 305
        //   6144 x 5120             44 /  52     51 /  53     75 /  67     86 /  65    101 /  84
        //   4096 x 5120             33 /  48     37 /  49     55 /  62     63 /  61     72 /  77
        //   5120 x 17408           105 / 184    115 / 194    146 / 209    169 / 227    225 / 236
        //
        // The SIMT kernel's cost grows with T (one FFMA per weight and token) and this one's
        // barely does, but it reads each row from one lane, so it needs many rows in flight: it
        // loses on the long, few-row down projection at every width.
        const std::int64_t rows_total = static_cast<std::int64_t>(items) * Rows;
        if (k > 6144 || !(Cols >= 5 || rows_total >= 16384)) return false;
        // Whole steps, and Q5's high plane read in 32-byte pairs of steps.
        if (k % 256 != 0 || padded_k % 256 != 0 || x_stride % 8 != 0) {
            return false;
        }

        const auto single  = stream_hmma_kernel<Format, Rows, Cols, false, Sink>;
        const auto chunked = stream_hmma_kernel<Format, Rows, Cols, true, Sink>;
        configure_cuda_device_once([&] {
            const cudaError_t first = cudaFuncSetAttribute(
                single, cudaFuncAttributeMaxDynamicSharedMemorySize, kMaxDynamicSharedBytes);
            return first != cudaSuccess
                       ? first
                       : cudaFuncSetAttribute(chunked, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                              kMaxDynamicSharedBytes);
        });
        int device = 0;
        int sms    = 0;
        CUDA_CHECK(cudaGetDevice(&device));
        CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device));

        const std::int32_t tiles = div_up(items, kStreamHmmaTileRows / Rows);
        const std::size_t budget = static_cast<std::size_t>(kMaxDynamicSharedBytes);
        const std::size_t fixed  = stream_hmma_reduce_bytes();
        std::int32_t chunk_k     = k;
        std::size_t shared_bytes = stream_hmma_stage_bytes(k, Cols) + fixed;
        // Up to four tiles per CTA, fewer when that would leave SMs idle (each then splits its K
        // across more warps).
        const int groups = kStreamHmmaGroups;
        int blocks = std::max(1, std::min(div_up(tiles, groups), sms));
        if (shared_bytes > budget) {
            const std::size_t partial = stream_hmma_partial_bytes(
                div_up(tiles, blocks * groups) * groups, Cols);
            if (fixed + partial + stream_hmma_stage_bytes(256, Cols) > budget) return false;
            const std::size_t room = budget - fixed - partial - stream_hmma_stage_bytes(0, Cols);
            chunk_k = static_cast<std::int32_t>(room / (Cols * sizeof(__half)) / 256 * 256);
            chunk_k      = std::min(chunk_k, k);
            shared_bytes = stream_hmma_stage_bytes(chunk_k, Cols) + fixed + partial;
        }
        const auto kernel = chunk_k == k ? single : chunked;
        kernel<<<blocks, kStreamHmmaWarps * 32, shared_bytes, stream>>>(
            x, x_stride, planes, sink, items, row_offset, k, padded_k, chunk_k, groups);
        CUDA_CHECK(cudaGetLastError());
        return true;
    }
}

} // namespace ninfer::ops::detail
