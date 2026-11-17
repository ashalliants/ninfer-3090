#pragma once

// Streaming GEMV: y = W x for one to eight tokens, over row-split weights (Q4, Q5, Q6), written to
// be bound by the weight stream on cards whose integer/half conversions are slow (sm_70).
//
// Shape of the kernel, and why:
//
//   * Persistent CTAs (about one wave), each streaming many rows. The activation is staged into
//     shared memory once per CTA, so its cost is paid per SM, not per row.
//   * The activation is staged as fp32 x', pre-scaled per k parity, with a per-32-weight correction
//     (ops/linear/common/stream_gemv_decode.cuh): each weight then costs one LOP3 and one FFMA,
//     with no conversion instruction. The formats live with their storage (Q4StreamFormat,
//     Q5StreamFormat, Q6StreamFormat in the q4/q5/q6 row-split storage headers).
//   * x' is stored in 16-byte chunks swizzled by (slice & 7), so the eight lanes of an LDS.128
//     phase read eight different bank groups.
//   * One warp per row; each lane streams 16-byte code vectors (a 512-byte warp transaction) and
//     prefetches the next step's vectors into registers -- across row boundaries too -- before
//     decoding the current ones, which keeps loads in flight across the decode (Volta has no
//     cp.async to do it).
//   * Several tokens (Cols, up to 8: speculative verification) share every decoded weight: a
//     format decodes a packed word into eight fp32 operands once, and each token adds one FFMA per
//     weight. When Cols * K of x' does not fit in shared memory (K = 17408 at T >= 2), K is staged
//     in chunks and each warp keeps its rows' partial sums in shared memory between chunks.
//   * The fully reduced row goes through linear_finish_row, so every Output / Epilogue the
//     family's other GEMVs accept (split outputs, residual add, ...) works unchanged.
//
// Measured on a V100-PCIE-32GB (879 GB/s sustained read) against the decode GEMVs it replaces
// (bandwidth share of the read rate; microbenchmark with the same layout, 2026-11-16):
//
//   shape (rows x K)    Q4 stream  Q4 before     Q5 stream  Q5 before
//   4096 x 5120              64%      40%            -          -
//   5120 x 6144              72%      41%           72%        51%
//   12288 x 5120              -        -            78%        52%
//   5120 x 17408             79%      35%           79%        61%
//
// Routing: on by default on sm_70 (kStreamGemvDefault); NINFER_STREAM_GEMV=0/1 overrides it at
// runtime, for A/B measurement on any card.

#include "core/device.h"
#include "ops/common/math.h"
#include "ops/common/smem_limit.h"
#include "ops/common/warp.cuh"
#include "ops/linear/common/epilogue.cuh"
#include "ops/linear/common/stream_gemv_decode.cuh"
#include "ops/linear/common/stream_hmma.cuh"
#include "ops/linear/common/stream_tile_gemm.cuh"

#include <cuda_bf16.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <unordered_map>
#include <utility>

namespace ninfer::ops::detail {

#if defined(NINFER_SM70)
inline constexpr bool kStreamGemvDefault = true;
#else
inline constexpr bool kStreamGemvDefault = false;
#endif

inline bool stream_gemv_enabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("NINFER_STREAM_GEMV");
        if (value == nullptr || *value == '\0') return kStreamGemvDefault;
        return std::strcmp(value, "0") != 0;
    }();
    return enabled;
}

// Shared memory for one K-chunk of x' over Cols tokens: chunk_k floats plus one correction per 32
// weights, per token.
constexpr std::size_t stream_gemv_stage_bytes(std::int32_t chunk_k, int cols) {
    return static_cast<std::size_t>(cols) *
           (static_cast<std::size_t>(chunk_k) * sizeof(float) +
            static_cast<std::size_t>(chunk_k / 32) * sizeof(float));
}

// Stages elements [k0, k0 + chunk_k) of Cols tokens of x (bf16; token t at x + t * x_stride) as x'
// with the slice swizzle, and Format::correction summed per 32-weight slice. Every thread issues
// all of its 16-byte loads for a pass before using any of them, so the prologue costs about one
// memory latency per pass instead of one per element.
// Token and position of flat vector i of a Cols x col_vectors staging grid, without an integer
// division (a runtime IDIV is ~20 instructions on the prologue's critical path).
template <int Cols>
__device__ __forceinline__ void stream_gemv_split_vector(int i, int col_vectors, int& col, int& j) {
    col = 0;
    j   = i;
    if constexpr (Cols > 1) {
#pragma unroll
        for (int t = 1; t < Cols; ++t) {
            if (j >= col_vectors) {
                j -= col_vectors;
                ++col;
            }
        }
    }
}

template <class Format, int Threads, int Cols>
__device__ __forceinline__ void stream_gemv_stage(const __nv_bfloat16* __restrict__ x,
                                                  std::int64_t x_stride, int k0, int chunk_k,
                                                  float4* __restrict__ chunks,
                                                  float* __restrict__ corrections) {
    static_assert(Threads % 32 == 0, "four consecutive threads share a slice");
    constexpr int kPassVectors = 4;
    const int col_vectors      = chunk_k / 8; // 8 bf16 each; four per 32-weight slice
    const int col_slices       = chunk_k / 32;
    const int vector_count     = Cols * col_vectors;
    for (int pass = 0; pass < vector_count; pass += Threads * kPassVectors) {
        uint4 loaded[kPassVectors];
#pragma unroll
        for (int p = 0; p < kPassVectors; ++p) {
            const int i = pass + p * Threads + static_cast<int>(threadIdx.x);
            loaded[p]   = make_uint4(0u, 0u, 0u, 0u);
            if (i < vector_count) {
                int col, j;
                stream_gemv_split_vector<Cols>(i, col_vectors, col, j);
                loaded[p] = __ldg(reinterpret_cast<const uint4*>(x + col * x_stride + k0) + j);
            }
        }
#pragma unroll
        for (int p = 0; p < kPassVectors; ++p) {
            if (pass + p * Threads >= vector_count) break; // uniform across the CTA
            const int i              = pass + p * Threads + static_cast<int>(threadIdx.x);
            const std::uint32_t w[4] = {loaded[p].x, loaded[p].y, loaded[p].z, loaded[p].w};
            float scaled[8];
            float correction = 0.0f;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const float even  = __uint_as_float(w[j] << 16);
                const float odd   = __uint_as_float(w[j] & 0xffff0000u);
                scaled[2 * j]     = even * kStreamEvenScale;
                scaled[2 * j + 1] = odd * kStreamOddScale;
                correction += Format::correction(even, odd);
            }
            // Four consecutive vectors are one slice of one token (col_vectors is a multiple of 4).
            correction += __shfl_xor_sync(0xffffffffu, correction, 1);
            correction += __shfl_xor_sync(0xffffffffu, correction, 2);
            if (i < vector_count) {
                int col, j;
                stream_gemv_split_vector<Cols>(i, col_vectors, col, j);
                const int slice = j >> 2;
                const int first = (j & 3) * 2; // chunks first, first + 1 of the slice
                float4* base    = chunks + (col * col_slices + slice) * 8;
                base[first ^ (slice & 7)] =
                    make_float4(scaled[0], scaled[1], scaled[2], scaled[3]);
                base[(first + 1) ^ (slice & 7)] =
                    make_float4(scaled[4], scaled[5], scaled[6], scaled[7]);
                if ((j & 3) == 0) corrections[col * col_slices + slice] = correction;
            }
        }
    }
}

// Receives fully reduced rows through the GEMV family's Output / Epilogue sink, all Cols tokens of
// a row at once, as the family's multi-token GEMVs do. An item of Rows rows holds rows item,
// item + row_offset, ...
template <class Output, class Epilogue>
struct StreamGemvRowSink {
    Output output;
    Epilogue epilogue;
    std::int32_t row_offset;
    std::int32_t token_begin = 0; // the first token of this pass (try_launch_stream_gemv_passes)

    template <int Rows, int Cols>
    __device__ __forceinline__ void finish(int item, const float (&acc)[Rows][Cols]) const {
#pragma unroll
        for (int r = 0; r < Rows; ++r) {
            linear_finish_row(output, epilogue, item + r * row_offset, token_begin, acc[r], Cols);
        }
    }
    StreamGemvRowSink with_token_offset(std::int32_t begin) const {
        StreamGemvRowSink sink = *this;
        sink.token_begin += begin;
        return sink;
    }
};

// One warp streams `Rows` weight rows per item -- rows item, item + row_offset, ... -- against the
// staged activation of `Cols` tokens, and hands their dot products to sink.finish(item, acc[Rows]
// [Cols]). Rows = 1 is the plain GEMV; Rows = 2 serves fused pairs such as SwiGLU's gate and up
// halves, which read each x' chunk once for both rows.
//
// K is processed in chunks of chunk_k (a multiple of 256) when it does not fit: each warp then adds
// its items' per-chunk sums into `partials` (its own items only, so no other warp touches them) and
// finishes them after the last chunk. With one chunk, items are finished as they complete.
// Blocks per SM the single-token kernel must fit: it is latency-bound and wants every resident
// warp it can get, so its registers are capped at 64 per thread (1024 threads per SM). The Q6
// decode would otherwise take 75 and drop to three CTAs per SM, costing ~10% at T = 1.
template <int Warps, int Cols>
inline constexpr int kStreamGemvMinBlocks = Cols == 1 ? 1024 / (Warps * 32) : 1;

template <class Format, int Warps, int Unroll, int Rows, int Cols, bool Chunked, class Sink>
__global__ __launch_bounds__(Warps * 32, (kStreamGemvMinBlocks<Warps, Cols>))
void stream_gemv_kernel(
    const __nv_bfloat16* __restrict__ x, std::int64_t x_stride, typename Format::Planes planes,
    Sink sink, std::int32_t items, std::int32_t row_offset, std::int32_t k, std::int32_t padded_k,
    std::int32_t chunk_k) {
    extern __shared__ __align__(16) unsigned char stream_shared[];
    const int chunk_slices = chunk_k / 32;
    auto* chunks           = reinterpret_cast<float4*>(stream_shared);
    auto* corrections      = reinterpret_cast<float*>(stream_shared) + Cols * chunk_k;
    auto* partials         = corrections + Cols * chunk_slices; // [item index][Rows][Cols]

    const int lane        = static_cast<int>(threadIdx.x) & 31;
    const int warp        = static_cast<int>(threadIdx.x) >> 5;
    const auto c          = StreamDecodeConstants::make();
    const int item_stride = static_cast<int>(gridDim.x) * Warps;
    const int first_item  = static_cast<int>(blockIdx.x) * Warps + warp;
    // Chunked is a compile-time property so that the single-chunk kernel (every T = 1 launch)
    // carries no chunk loop or partial-sum code.
    const int chunk_count = Chunked ? div_up(k, chunk_k) : 1;
    // A lane's slices are lane + 32n, so their swizzle (slice & 7) is fixed per lane.
    int chunk_of[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) chunk_of[i] = i ^ (lane & 7);

    using Slice = typename Format::Slice;
    using Row   = decltype(Format::row(planes, 0, padded_k));

    for (int chunk = 0; chunk < chunk_count; ++chunk) {
        const int k0             = chunk * chunk_k;
        const int slice_base     = k0 / 32;
        const int slices         = min(chunk_k, k - k0) / 32;
        const int steps_per_item = div_up(slices, 32 * Unroll);
        // The warp walks one flat sequence of steps -- each item's slices of this chunk, 32 *
        // Unroll at a time, then the next item's -- with the following step's weights always in
        // flight while it decodes the current one, across item boundaries too. A short row
        // (K = 5120 is five slices per lane) would otherwise wait a full memory latency per item.
        const auto fetch = [&](int item, int step, Slice(&slice)[Rows][Unroll]) {
            Row row_planes[Rows];
#pragma unroll
            for (int r = 0; r < Rows; ++r) {
                row_planes[r] =
                    Format::row(planes, item < items ? item + r * row_offset : 0, padded_k);
            }
#pragma unroll
            for (int u = 0; u < Unroll; ++u) {
                const int v = step * 32 * Unroll + u * 32 + lane;
#pragma unroll
                for (int r = 0; r < Rows; ++r) {
                    slice[r][u] = item < items && v < slices
                                      ? Format::load(row_planes[r], slice_base + v)
                                      : Format::empty();
                }
            }
        };

        int item = first_item;
        int step = 0;
        Slice current[Rows][Unroll];
        // The first step's loads are issued before the chunk is staged, so the two latencies
        // overlap. The barrier first lets every warp finish reading the previous chunk.
        fetch(item, step, current);
        if (chunk > 0) __syncthreads();
        stream_gemv_stage<Format, Warps * 32, Cols>(x, x_stride, k0, chunk_k, chunks, corrections);
        __syncthreads();

        float acc[Rows][Cols];
#pragma unroll
        for (int r = 0; r < Rows; ++r) {
#pragma unroll
            for (int t = 0; t < Cols; ++t) acc[r][t] = 0.0f;
        }
        int item_index = warp; // this warp's n-th item has partial index n * Warps + warp
        while (item < items) {
            const bool last_step = step + 1 == steps_per_item;
            const int next_item  = last_step ? item + item_stride : item;
            const int next_step  = last_step ? 0 : step + 1;
            Slice next[Rows][Unroll];
            fetch(next_item, next_step, next);
#pragma unroll
            for (int u = 0; u < Unroll; ++u) {
                const int v = step * 32 * Unroll + u * 32 + lane;
                if (v < slices) {
                    float part[Rows][Cols];
#pragma unroll
                    for (int r = 0; r < Rows; ++r) {
#pragma unroll
                        for (int t = 0; t < Cols; ++t) part[r][t] = 0.0f;
                    }
                    const auto word = [&]<int W>() {
                        float f[Rows][8];
#pragma unroll
                        for (int r = 0; r < Rows; ++r) {
                            Format::template decode_word<W>(current[r][u], c, f[r]);
                        }
#pragma unroll
                        for (int t = 0; t < Cols; ++t) {
                            const float4* xv = chunks + (t * chunk_slices + v) * 8;
                            const float4 a   = xv[chunk_of[2 * W]];
                            const float4 b   = xv[chunk_of[2 * W + 1]];
                            const float xs[8] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
#pragma unroll
                            for (int r = 0; r < Rows; ++r) {
#pragma unroll
                                for (int i = 0; i < 8; ++i) {
                                    part[r][t] = fmaf(f[r][i], xs[i], part[r][t]);
                                }
                            }
                        }
                    };
                    word.template operator()<0>();
                    word.template operator()<1>();
                    word.template operator()<2>();
                    word.template operator()<3>();
#pragma unroll
                    for (int t = 0; t < Cols; ++t) {
                        const float correction = corrections[t * chunk_slices + v];
#pragma unroll
                        for (int r = 0; r < Rows; ++r) {
                            acc[r][t] = fmaf(Format::scale(current[r][u]),
                                             part[r][t] - correction, acc[r][t]);
                        }
                    }
                }
            }
            if (last_step) {
#pragma unroll
                for (int r = 0; r < Rows; ++r) {
#pragma unroll
                    for (int t = 0; t < Cols; ++t) acc[r][t] = warp_reduce_sum(acc[r][t]);
                }
                if (lane == 0) {
                    if constexpr (!Chunked) {
                        sink.finish(item, acc);
                    } else {
                        float* p = partials + item_index * Rows * Cols;
#pragma unroll
                        for (int r = 0; r < Rows; ++r) {
#pragma unroll
                            for (int t = 0; t < Cols; ++t) {
                                p[r * Cols + t] =
                                    chunk == 0 ? acc[r][t] : p[r * Cols + t] + acc[r][t];
                            }
                        }
                    }
                }
#pragma unroll
                for (int r = 0; r < Rows; ++r) {
#pragma unroll
                    for (int t = 0; t < Cols; ++t) acc[r][t] = 0.0f;
                }
                item_index += Warps;
            }
#pragma unroll
            for (int u = 0; u < Unroll; ++u) {
#pragma unroll
                for (int r = 0; r < Rows; ++r) current[r][u] = next[r][u];
            }
            item = next_item;
            step = next_step;
        }
    }

    if constexpr (Chunked) {
        // Each warp finishes its own items, one per lane.
        __syncwarp();
        int n = lane;
        for (int item = first_item + lane * item_stride; item < items;
             item += 32 * item_stride, n += 32) {
            const float* p = partials + (n * Warps + warp) * Rows * Cols;
            float acc[Rows][Cols];
#pragma unroll
            for (int r = 0; r < Rows; ++r) {
#pragma unroll
                for (int t = 0; t < Cols; ++t) acc[r][t] = p[r * Cols + t];
            }
            sink.finish(item, acc);
        }
    }
}

// Launches the streaming GEMV over `items` items of `Rows` rows each for `Cols` tokens (token t of
// x at x + t * x_stride) if it is enabled and the shapes fit. Returns false otherwise, and the
// caller launches its own schedule; the two compute the same function.
template <class Format, int Rows, int Cols, class Sink>
bool try_launch_stream_gemv_rows(const __nv_bfloat16* x, std::int64_t x_stride,
                                 const typename Format::Planes& planes, Sink sink,
                                 std::int32_t items, std::int32_t row_offset, std::int32_t k,
                                 std::int32_t padded_k, cudaStream_t stream) {
    static_assert(Cols >= 1 && Cols <= 8);
    if (!stream_gemv_enabled()) return false;
    // Verify widths run on the tensor cores where the format supports it (stream_hmma.cuh).
    if (try_launch_stream_hmma_rows<Format, Rows, Cols>(x, x_stride, planes, sink, items,
                                                        row_offset, k, padded_k, stream)) {
        return true;
    }
    if (k % 256 != 0 && Cols > 1) return false; // chunks are whole multiples of 256
    if (k % 128 != 0 || x_stride % 8 != 0) return false;

    const auto launch = [&]<int Warps, int Unroll>() {
        const auto single  = stream_gemv_kernel<Format, Warps, Unroll, Rows, Cols, false, Sink>;
        const auto chunked = stream_gemv_kernel<Format, Warps, Unroll, Rows, Cols, true, Sink>;
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

        // One chunk when all of K fits; then about one wave of resident CTAs.
        std::int32_t chunk_k     = k;
        std::size_t shared_bytes = stream_gemv_stage_bytes(k, Cols);
        int blocks               = 0;
        if (shared_bytes <= static_cast<std::size_t>(kMaxDynamicSharedBytes)) {
            int per_sm = 0;
            CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, single, Warps * 32,
                                                                     shared_bytes));
            blocks = std::max(1, std::min(div_up(items, Warps), sms * std::max(per_sm, 1)));
        } else {
            // K-chunked: one CTA per SM; the partials take what the chunk leaves.
            blocks = std::max(1, std::min(div_up(items, Warps), sms));
            const std::size_t partial_bytes =
                static_cast<std::size_t>(div_up(items, blocks * Warps)) * Warps * Rows * Cols *
                sizeof(float);
            const std::size_t budget = static_cast<std::size_t>(kMaxDynamicSharedBytes);
            if (partial_bytes + stream_gemv_stage_bytes(256, Cols) > budget) return false;
            chunk_k = static_cast<std::int32_t>(
                (budget - partial_bytes) / stream_gemv_stage_bytes(256, Cols) * 256);
            chunk_k      = std::min(chunk_k, k);
            shared_bytes = stream_gemv_stage_bytes(chunk_k, Cols) + partial_bytes;
        }
        const auto kernel = chunk_k == k ? single : chunked;
        kernel<<<blocks, Warps * 32, shared_bytes, stream>>>(x, x_stride, planes, sink, items,
                                                            row_offset, k, padded_k, chunk_k);
        CUDA_CHECK(cudaGetLastError());
        return true;
    };
    // T = 1: K <= 6144 leaves room for several CTAs per SM, and short rows favour more CTAs with
    // one slice in flight; the long K = 17408 rows want a full SM's worth of warps and deeper
    // prefetch. A pair, or several tokens, already keep more work per weight in flight.
    if constexpr (Cols == 1) {
        if (k <= 6144) return launch.template operator()<8, 1>();
        if constexpr (Rows == 1) {
            return launch.template operator()<16, 2>();
        } else {
            return launch.template operator()<16, 1>();
        }
    } else {
        return launch.template operator()<16, 1>();
    }
}

// The most tokens one call takes: past eight (the kernels' widest instantiation) it runs as passes
// over the same weights, each finished at its token offset -- whole passes of 32 and 16 tokens on
// the tensor cores for the formats stream_tile_gemm.cuh serves, passes of eight otherwise. Both
// beat the sm_70 alternatives at short-prompt widths: the emulated small-T mma kernels, or
// materialising every weight in FP16 for cuBLAS (see kCublasPrefillNarrowMinTokens). Formats
// without the tile GEMM stop at kStreamGemvMaxUntiledTokens: four passes of eight still win there.
inline constexpr std::int32_t kStreamGemvMaxTokens         = 64;
inline constexpr std::int32_t kStreamGemvMaxUntiledTokens  = 32;

// `tokens` (1..kStreamGemvMaxTokens) tokens of `items` items of Rows rows: one launch per eight
// tokens. `sink` must offer with_token_offset(begin). Declines (returns false, nothing launched)
// unless every pass would run.
template <class Format, int Rows, class Sink>
bool try_launch_stream_gemv_passes(const __nv_bfloat16* x, std::int64_t x_stride,
                                   const typename Format::Planes& planes, Sink sink,
                                   std::int32_t items, std::int32_t row_offset, std::int32_t k,
                                   std::int32_t padded_k, cudaStream_t stream,
                                   std::int32_t tokens) {
    if (tokens < 1 || tokens > kStreamGemvMaxTokens) return false;
    // A format the tile GEMM serves takes every width up to the maximum (as passes of eight when
    // NINFER_STREAM_TILE_GEMM=0 turns the tiles off, so routes sized for it still run).
    if (tokens > kStreamGemvMaxUntiledTokens && !stream_tile_gemm_format<Format>()) {
        return false;
    }
    if (tokens > 8) {
        // The checks of the widest pass cover the narrower ones (try_launch_stream_gemv_rows).
        if (!stream_gemv_enabled() || k % 256 != 0 || k % 128 != 0 || x_stride % 8 != 0) {
            return false;
        }
    }
    std::int32_t begin = 0;
    // Whole passes of 32 tokens on the tensor cores where the format allows, then one pass of the
    // last 9..31 (stream_tile_gemm.cuh: 1.3-3.6x the passes of eight it replaces); a last one to
    // eight tokens runs as one pass of eight.
    if (tokens >= 9 && stream_tile_gemm_supported<Format>(k, padded_k, x_stride)) {
        for (; tokens - begin >= 32; begin += 32) {
            (void)try_launch_stream_tile_gemm<Format, Rows, 32>(x + begin * x_stride, x_stride,
                                                               planes, sink.with_token_offset(begin),
                                                               items, row_offset, k, padded_k,
                                                               stream);
        }
        const std::int32_t last = tokens - begin;
        if (last >= 9) {
            const __nv_bfloat16* pass_x = x + begin * x_stride;
            const Sink pass_sink        = sink.with_token_offset(begin);
            [&]<int... I>(std::integer_sequence<int, I...>) {
                // One instantiation per width 9..31 (32 is the loop above).
                (void)((last == I + 9 &&
                        (void(try_launch_stream_tile_gemm<Format, Rows, I + 9>(
                             pass_x, x_stride, planes, pass_sink, items, row_offset, k, padded_k,
                             stream)),
                         true)) ||
                       ...);
            }(std::make_integer_sequence<int, 23>{});
            begin = tokens;
        }
    }
    const std::int32_t tiled = begin;
    for (; begin < tokens; begin += 8) {
        const std::int32_t count = std::min<std::int32_t>(8, tokens - begin);
        const __nv_bfloat16* pass_x = x + begin * x_stride;
        const Sink pass_sink        = sink.with_token_offset(begin);
        bool launched               = false;
        switch (count) {
#define NINFER_STREAM_GEMV_PASS(T)                                                                     case T:                                                                                                launched = try_launch_stream_gemv_rows<Format, Rows, T>(                                               pass_x, x_stride, planes, pass_sink, items, row_offset, k, padded_k, stream);                  break
            NINFER_STREAM_GEMV_PASS(1);
            NINFER_STREAM_GEMV_PASS(2);
            NINFER_STREAM_GEMV_PASS(3);
            NINFER_STREAM_GEMV_PASS(4);
            NINFER_STREAM_GEMV_PASS(5);
            NINFER_STREAM_GEMV_PASS(6);
            NINFER_STREAM_GEMV_PASS(7);
            NINFER_STREAM_GEMV_PASS(8);
#undef NINFER_STREAM_GEMV_PASS
        }
        if (!launched) {
            if (begin > 0 || tiled > 0) {
                throw std::logic_error("streaming GEMV declined a pass after launching others");
            }
            return false;
        }
    }
    return true;
}

// The plain GEMV over `tokens` (1..kStreamGemvMaxTokens) tokens of a contiguous x: rows rows,
// each finished through the family's Output / Epilogue with every token of a pass at once.
template <class Format, class Output, class Epilogue>
bool try_launch_stream_gemv(const __nv_bfloat16* x, const typename Format::Planes& planes,
                            Output output, Epilogue epilogue, std::int32_t rows, std::int32_t k,
                            std::int32_t padded_k, cudaStream_t stream, std::int32_t tokens = 1,
                            std::int64_t x_stride = -1) {
    const std::int64_t stride = x_stride < 0 ? k : x_stride;
    // From two tokens on, shared-memory reads of x' (one per token per weight) bound the kernel, so
    // each warp streams two rows -- r and r + rows / 2 -- against every x' read: 20-36% at T = 4
    // and 30-35% at T = 8 on a V100 (four rows spill registers and lose).
    const bool paired = tokens >= 2 && rows % 2 == 0;
    if (paired) {
        return try_launch_stream_gemv_passes<Format, 2>(
            x, stride, planes, StreamGemvRowSink<Output, Epilogue>{output, epilogue, rows / 2},
            rows / 2, rows / 2, k, padded_k, stream, tokens);
    }
    return try_launch_stream_gemv_passes<Format, 1>(
        x, stride, planes, StreamGemvRowSink<Output, Epilogue>{output, epilogue, 0}, rows, 0, k,
        padded_k, stream, tokens);
}

} // namespace ninfer::ops::detail
