#pragma once

// Streaming GEMV: y = W x for one token (T = 1), over row-split weights (Q4, Q5), written to be
// bound by the weight stream on cards whose integer/half conversions are slow (sm_70).
//
// Shape of the kernel, and why:
//
//   * Persistent CTAs (about one wave), each streaming many rows. The activation is staged into
//     shared memory once per CTA, so its cost is paid per SM, not per row.
//   * The activation is staged as fp32 x', pre-scaled per k parity, with a per-32-weight correction
//     (ops/linear/common/stream_gemv_decode.cuh): each weight then costs one LOP3 and one FFMA,
//     with no conversion instruction. The formats live with their storage
//     (Q4StreamFormat in q4_rowsplit_storage.cuh, Q5StreamFormat in q5_rowsplit_storage.cuh).
//   * x' is stored in 16-byte chunks swizzled by (slice & 7), so the eight lanes of an LDS.128
//     phase read eight different bank groups.
//   * One warp per row; each lane streams 16-byte code vectors (a 512-byte warp transaction) and
//     prefetches the next iteration's vectors into registers before decoding the current ones,
//     which keeps loads in flight across the decode (Volta has no cp.async to do it).
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

#include <cuda_bf16.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <unordered_map>

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

// Shared memory one CTA needs: x' (k floats) plus one correction per 32 weights.
constexpr std::size_t stream_gemv_shared_bytes(std::int32_t k) {
    return static_cast<std::size_t>(k) * sizeof(float) +
           static_cast<std::size_t>(k / 32) * sizeof(float);
}

// Stages x (bf16, length k) as x' with the slice swizzle, and Format::correction summed per
// 32-weight slice. Every thread issues all of its 16-byte loads for a pass before using any of
// them, so the prologue costs about one memory latency per pass instead of one per element.
template <class Format, int Threads>
__device__ __forceinline__ void stream_gemv_stage(const __nv_bfloat16* __restrict__ x, int k,
                                                  float4* __restrict__ chunks,
                                                  float* __restrict__ corrections) {
    static_assert(Threads % 32 == 0, "four consecutive threads share a slice");
    constexpr int kPassVectors = 4;
    const auto* vectors        = reinterpret_cast<const uint4*>(x);
    const int vector_count     = k / 8; // 8 bf16 each; four per 32-weight slice
    for (int pass = 0; pass < vector_count; pass += Threads * kPassVectors) {
        uint4 loaded[kPassVectors];
#pragma unroll
        for (int p = 0; p < kPassVectors; ++p) {
            const int i = pass + p * Threads + static_cast<int>(threadIdx.x);
            loaded[p]   = i < vector_count ? __ldg(&vectors[i]) : make_uint4(0u, 0u, 0u, 0u);
        }
#pragma unroll
        for (int p = 0; p < kPassVectors; ++p) {
            if (pass + p * Threads >= vector_count) break; // uniform across the CTA
            const int i               = pass + p * Threads + static_cast<int>(threadIdx.x);
            const std::uint32_t w[4]  = {loaded[p].x, loaded[p].y, loaded[p].z, loaded[p].w};
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
            correction += __shfl_xor_sync(0xffffffffu, correction, 1);
            correction += __shfl_xor_sync(0xffffffffu, correction, 2);
            if (i < vector_count) {
                const int slice  = i >> 2;
                const int first  = (i & 3) * 2; // chunks first, first + 1 of the slice
                float4* base     = chunks + slice * 8;
                base[first ^ (slice & 7)] =
                    make_float4(scaled[0], scaled[1], scaled[2], scaled[3]);
                base[(first + 1) ^ (slice & 7)] =
                    make_float4(scaled[4], scaled[5], scaled[6], scaled[7]);
                if ((i & 3) == 0) corrections[slice] = correction;
            }
        }
    }
}

// Receives each fully reduced row (Rows = 1) through the GEMV family's Output / Epilogue sink.
template <class Output, class Epilogue>
struct StreamGemvRowSink {
    Output output;
    Epilogue epilogue;

    __device__ __forceinline__ void finish(int row, const float (&acc)[1]) const {
        const float values[1]{acc[0]};
        linear_finish_row(output, epilogue, row, 0, values, 1);
    }
};

// One warp streams `Rows` weight rows per item -- rows item, item + row_offset, ... -- against the
// same staged activation, and hands their dot products to sink.finish(item, acc). Rows = 1 is the
// plain GEMV; Rows = 2 serves fused pairs such as SwiGLU's gate and up halves, which read each x'
// chunk once for both rows.
template <class Format, int Warps, int Unroll, int Rows, class Sink>
__global__ __launch_bounds__(Warps * 32) void stream_gemv_kernel(
    const __nv_bfloat16* __restrict__ x, typename Format::Planes planes, Sink sink,
    std::int32_t items, std::int32_t row_offset, std::int32_t k, std::int32_t padded_k) {
    extern __shared__ __align__(16) unsigned char stream_shared[];
    auto* chunks      = reinterpret_cast<float4*>(stream_shared);
    auto* corrections = reinterpret_cast<float*>(stream_shared) + k;

    const int lane   = static_cast<int>(threadIdx.x) & 31;
    const int warp   = static_cast<int>(threadIdx.x) >> 5;
    const int slices = k / 32;
    const auto c     = StreamDecodeConstants::make();
    // A lane's slices are lane + 32n, so their swizzle (slice & 7) is fixed per lane.
    int chunk_of[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) chunk_of[i] = i ^ (lane & 7);

    // The warp walks one flat sequence of steps -- each item's slices, 32 * Unroll at a time, then
    // the next item's -- and always has the following step's weights in flight while it decodes
    // the current one, across item boundaries too. A short row (K = 5120 is five slices per lane)
    // would otherwise wait a full memory latency at the start of every item. The first step's
    // loads are issued before the activation is staged, so the two latencies overlap.
    using Slice                = typename Format::Slice;
    using Row                  = decltype(Format::row(planes, 0, padded_k));
    const int steps_per_item   = div_up(slices, 32 * Unroll);
    const int item_stride      = static_cast<int>(gridDim.x) * Warps;
    const auto fetch           = [&](int item, int step, Slice(&slice)[Rows][Unroll]) {
        Row row_planes[Rows];
#pragma unroll
        for (int r = 0; r < Rows; ++r) {
            row_planes[r] = Format::row(planes, item < items ? item + r * row_offset : 0, padded_k);
        }
#pragma unroll
        for (int u = 0; u < Unroll; ++u) {
            const int v = step * 32 * Unroll + u * 32 + lane;
#pragma unroll
            for (int r = 0; r < Rows; ++r) {
                slice[r][u] =
                    item < items && v < slices ? Format::load(row_planes[r], v) : Format::empty();
            }
        }
    };

    int item = static_cast<int>(blockIdx.x) * Warps + warp;
    int step = 0;
    Slice current[Rows][Unroll];
    fetch(item, step, current);

    stream_gemv_stage<Format, Warps * 32>(x, k, chunks, corrections);
    __syncthreads();

    float acc[Rows];
#pragma unroll
    for (int r = 0; r < Rows; ++r) acc[r] = 0.0f;
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
                float4 xs[8];
#pragma unroll
                for (int i = 0; i < 8; ++i) xs[i] = chunks[v * 8 + chunk_of[i]];
                const float correction = corrections[v];
#pragma unroll
                for (int r = 0; r < Rows; ++r) {
                    const float dot = Format::dot(current[r][u], xs, c);
                    acc[r]          = fmaf(Format::scale(current[r][u]), dot - correction, acc[r]);
                }
            }
        }
        if (last_step) {
#pragma unroll
            for (int r = 0; r < Rows; ++r) acc[r] = warp_reduce_sum(acc[r]);
            if (lane == 0) sink.finish(item, acc);
#pragma unroll
            for (int r = 0; r < Rows; ++r) acc[r] = 0.0f;
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

// Launches the streaming GEMV over `items` items of `Rows` rows each if it is enabled and K fits
// the shared-memory budget. Returns false otherwise, and the caller launches its own schedule; the
// two compute the same function.
template <class Format, int Rows, class Sink>
bool try_launch_stream_gemv_rows(const __nv_bfloat16* x, const typename Format::Planes& planes,
                                 Sink sink, std::int32_t items, std::int32_t row_offset,
                                 std::int32_t k, std::int32_t padded_k, cudaStream_t stream) {
    if (!stream_gemv_enabled()) return false;
    const std::size_t shared_bytes = stream_gemv_shared_bytes(k);
    if (k % 128 != 0 || shared_bytes > static_cast<std::size_t>(kMaxDynamicSharedBytes))
        return false;

    const auto launch = [&]<int Warps, int Unroll>() {
        const auto kernel = stream_gemv_kernel<Format, Warps, Unroll, Rows, Sink>;
        configure_cuda_device_once([&] {
            return cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        kMaxDynamicSharedBytes);
        });
        // About one wave of resident CTAs, each streaming items until none are left.
        int device = 0;
        int sms    = 0;
        int per_sm = 0;
        CUDA_CHECK(cudaGetDevice(&device));
        CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device));
        CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, kernel, Warps * 32,
                                                                 shared_bytes));
        const int blocks = std::max(1, std::min(div_up(items, Warps), sms * std::max(per_sm, 1)));
        kernel<<<blocks, Warps * 32, shared_bytes, stream>>>(x, planes, sink, items, row_offset, k,
                                                            padded_k);
        CUDA_CHECK(cudaGetLastError());
    };
    // K <= 6144 leaves room for several CTAs per SM, and short rows favour more CTAs with one
    // slice in flight; the long K = 17408 rows want a full SM's worth of warps and deeper prefetch.
    // A pair already keeps two rows' slices in flight per lane.
    if (k <= 6144) {
        launch.template operator()<8, 1>();
    } else if constexpr (Rows == 1) {
        launch.template operator()<16, 2>();
    } else {
        launch.template operator()<16, 1>();
    }
    return true;
}

// The plain GEMV: rows rows, each finished through the family's Output / Epilogue.
template <class Format, class Output, class Epilogue>
bool try_launch_stream_gemv(const __nv_bfloat16* x, const typename Format::Planes& planes,
                            Output output, Epilogue epilogue, std::int32_t rows, std::int32_t k,
                            std::int32_t padded_k, cudaStream_t stream) {
    return try_launch_stream_gemv_rows<Format, 1>(
        x, planes, StreamGemvRowSink<Output, Epilogue>{output, epilogue}, rows, 0, k, padded_k,
        stream);
}

} // namespace ninfer::ops::detail
