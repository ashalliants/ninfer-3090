#pragma once

// Streaming GEMV on int8 activations with dp4a, for 1..8 tokens of a Q4 / Q5 / Q6 row-split linear on
// sm_70 (V100). The flat-cost verify GEMV of speculative decoding.
//
// The SIMT streaming GEMV (stream_gemv.cuh) decodes each weight once and then pays one FFMA per
// weight per token, so its cost grows with the verify width: a Qwen3.8-27B MTP3 round (T = 4)
// and a DFlash2 round (T = 8) were bound by issue, not by the weight stream. Here each CTA
// quantises its staged activation to int8 (one FP32 scale per 32-value slice and token, the scale
// of llama.cpp's q8_1), the format unpacks a packed word of eight weights into two words of four
// biased unsigned bytes (3 instructions for Q4), and each token then costs one dp4a per four
// weights: about 0.4 + 0.45 T instructions per weight instead of 1.5 + 1.25 T. That keeps the
// kernel at the weight stream through T = 8.
//
// Numerics: weights are exact (the formats' integer codes, plus a bias the activation sum
// removes); the activation is rounded to int8 per 32-value slice (absmax / 127), which is the
// approximation. Each slice's integer dot product is exact, then scaled in FP32.
//
// Every token's sum is formed in the same order at any width: lane l sums its slices l, l + 32,
// ... of a K-chunk, the warp reduces the chunk, and chunks add in order. The chunk length depends
// only on K (stream_dp4a_chunk_k), not on the number of tokens, so a token's result is bit-for-bit
// the same at T = 1 as in an eight-token verify. That keeps speculative decoding's output equal to
// plain decoding's wherever both run this kernel.
//
// A format opts in with `kDp4a = true`, `kDp4aBias` (the code bias: value = code - bias) and
// `dp4a_word<W>(slice, even, odd)`: the biased codes of k offsets 8W + 0, 2, 4, 6 in the bytes of
// `even` and of 8W + 1, 3, 5, 7 in those of `odd`.
//
// The same kernel serves a speculative drafter's Q8 and Q4 linears (stream_draft_dp4a.cuh, through
// launch_stream_dp4a_rows, inside a ScopedDraftA8 only). A format whose form is for the drafter
// alone sets `kDp4aDrafterOnly` (Q8): the verify route above never takes it, so the target's Q8
// linears keep their A16 kernels whatever --verify-a8 says.

#include "core/device.h"
#include "ops/common/math.h"
#include "ops/common/smem_limit.h"
#include "ops/common/warp.cuh"
#include "ops/linear/common/stream_a8.h"

#include <cuda_bf16.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>

namespace ninfer::ops::detail {

// Compiled into sm_70 builds only; whether a call takes it is the engine's choice, per stream
// (EngineOptions::verify_a8 through ops::StreamA8Registration, stream_a8.h).
#if defined(NINFER_SM70)
inline constexpr bool kStreamDp4aCompiled = true;
#else
inline constexpr bool kStreamDp4aCompiled = false;
#endif

inline bool stream_dp4a_enabled(cudaStream_t stream) {
    return ninfer::ops::stream_a8_activations(stream);
}

template <class Format>
concept StreamDp4aFormat = requires { requires Format::kDp4a; };

// A format whose dp4a form serves only drafters (`kDp4aDrafterOnly`, the Q8 drafter weights:
// stream_draft_dp4a.cuh) never takes the verify route, which would otherwise also move the Q8
// linears of whatever else shares a registered stream.
template <class Format>
concept StreamDp4aVerifyFormat =
    StreamDp4aFormat<Format> && !requires { requires Format::kDp4aDrafterOnly; };

inline constexpr int kStreamDp4aMaxCols = 8;

// Shared memory per token and K value: one int8 plus a (scale, bias sum) pair per 32 values.
__host__ __device__ constexpr std::size_t stream_dp4a_stage_bytes(std::int32_t chunk_k, int cols) {
    return static_cast<std::size_t>(cols) *
           (static_cast<std::size_t>(chunk_k) + static_cast<std::size_t>(chunk_k / 32) * 8);
}

// The K-chunk length, a function of K alone (see the header comment): all of K when eight tokens'
// int8 stage fits the stage budget, else K split into equal multiples of 256.
inline constexpr std::size_t kStreamDp4aStageBudget = 80 * 1024;
__host__ __device__ constexpr std::int32_t stream_dp4a_chunk_k(std::int32_t k) {
    std::int32_t parts = 1;
    while (stream_dp4a_stage_bytes(div_up(div_up(k, parts), 256) * 256, kStreamDp4aMaxCols) >
           kStreamDp4aStageBudget) {
        ++parts;
    }
    return std::min(k, div_up(div_up(k, parts), 256) * 256);
}

// Bytes 0..3 of the result are the low bytes of a, b, c, d.
__device__ __forceinline__ std::uint32_t stream_dp4a_pack(int a, int b, int c, int d) {
    return __byte_perm(__byte_perm(a, b, 0x0040), __byte_perm(c, d, 0x0040), 0x5410);
}

// Stages x[k0, k0 + chunk_k) of Cols tokens as int8, one FP32 scale per 32 values. Four
// consecutive threads take one slice, eight values each (packed word W = thread & 3). Slice v of
// token t is 32 bytes at q8 + (t * chunk_slices + v) * 2 (uint4 units): word W's even values then
// its odd ones, the two 16-byte halves swapped when (v >> 2) & 1 so that the eight lanes of an
// LDS.128 phase (consecutive v) hit eight different bank groups. meta[t * chunk_slices + v] holds
// the slice's scale and -Bias * sum(q) (as int bits), the dp4a chain's starting value.
template <int Bias, int Threads, int Cols>
__device__ __forceinline__ void stream_dp4a_stage(const __nv_bfloat16* __restrict__ x,
                                                  std::int64_t x_stride, int k0, int chunk_k,
                                                  int valid_k, uint4* __restrict__ q8,
                                                  float2* __restrict__ meta) {
    static_assert(Threads % 32 == 0);
    const int chunk_slices = chunk_k / 32;
    const int total        = Cols * chunk_slices * 4;
    for (int base = 0; base < total; base += Threads) { // uniform: every lane shuffles
        const int i     = base + static_cast<int>(threadIdx.x);
        const int col   = i / (chunk_slices * 4);
        const int rest  = i - col * chunk_slices * 4;
        const int slice = rest >> 2;
        const int w     = rest & 3;
        const int kk    = slice * 32 + w * 8; // within the chunk
        uint4 raw       = make_uint4(0u, 0u, 0u, 0u);
        if (i < total && kk < valid_k) {
            raw = __ldg(reinterpret_cast<const uint4*>(x + col * x_stride + k0 + kk));
        }
        const std::uint32_t words[4] = {raw.x, raw.y, raw.z, raw.w};
        float v[8];
        float amax = 0.0f;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            v[2 * j]     = __uint_as_float(words[j] << 16);
            v[2 * j + 1] = __uint_as_float(words[j] & 0xffff0000u);
            amax         = fmaxf(amax, fmaxf(fabsf(v[2 * j]), fabsf(v[2 * j + 1])));
        }
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 1));
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 2));
        const float d   = amax * (1.0f / 127.0f);
        const float inv = amax > 0.0f ? 127.0f / amax : 0.0f;
        int q[8];
        int sum = 0;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            q[j] = __float2int_rn(v[j] * inv);
            sum += q[j];
        }
        sum += __shfl_xor_sync(0xffffffffu, sum, 1);
        sum += __shfl_xor_sync(0xffffffffu, sum, 2);
        if (i < total) {
            const std::uint32_t even = stream_dp4a_pack(q[0], q[2], q[4], q[6]);
            const std::uint32_t odd  = stream_dp4a_pack(q[1], q[3], q[5], q[7]);
            // Word W sits at bytes 8W.. of the slice's 32: half W / 2, 8-byte lane W & 1.
            const int half = (w >> 1) ^ ((slice >> 2) & 1);
            auto* dst      = reinterpret_cast<uint2*>(q8 + (col * chunk_slices + slice) * 2 + half);
            dst[w & 1]     = make_uint2(even, odd);
            if (w == 0) {
                meta[col * chunk_slices + slice] = make_float2(d, __int_as_float(-Bias * sum));
            }
        }
    }
}

// One warp streams `Rows` weight rows per item (rows item, item + row_offset, ...) against the
// staged int8 activation of `Cols` tokens, and hands their dot products to sink.finish(item,
// acc[Rows][Cols]), as stream_gemv_kernel does. K runs in chunks of chunk_k (stream_dp4a_chunk_k):
// with more than one, each warp keeps its items' per-chunk sums in `partials` and finishes them
// after the last chunk.
template <class Format, int Warps, int Unroll, int Rows, int Cols, bool Chunked, class Sink>
__global__ __launch_bounds__(Warps * 32, (Cols == 1 ? 1024 / (Warps * 32) : 1))
void stream_dp4a_kernel(const __nv_bfloat16* __restrict__ x, std::int64_t x_stride,
                        typename Format::Planes planes, Sink sink, std::int32_t items,
                        std::int32_t row_offset, std::int32_t k, std::int32_t padded_k,
                        std::int32_t chunk_k) {
    extern __shared__ __align__(16) unsigned char stream_dp4a_shared[];
    const int chunk_slices = chunk_k / 32;
    auto* q8               = reinterpret_cast<uint4*>(stream_dp4a_shared);
    auto* meta             = reinterpret_cast<float2*>(stream_dp4a_shared +
                                                       static_cast<std::size_t>(Cols) * chunk_k);
    auto* partials         = reinterpret_cast<float*>(meta + Cols * chunk_slices);

    const int lane        = static_cast<int>(threadIdx.x) & 31;
    const int warp        = static_cast<int>(threadIdx.x) >> 5;
    const int item_stride = static_cast<int>(gridDim.x) * Warps;
    const int first_item  = static_cast<int>(blockIdx.x) * Warps + warp;
    const int chunk_count = Chunked ? div_up(k, chunk_k) : 1;

    using Slice = typename Format::Slice;
    using Row   = decltype(Format::row(planes, 0, padded_k));

    for (int chunk = 0; chunk < chunk_count; ++chunk) {
        const int k0             = chunk * chunk_k;
        const int slice_base     = k0 / 32;
        const int slices         = min(chunk_k, k - k0) / 32;
        const int steps_per_item = div_up(slices, 32 * Unroll);
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
        fetch(item, step, current);
        if (chunk > 0) __syncthreads();
        stream_dp4a_stage<Format::kDp4aBias, Warps * 32, Cols>(x, x_stride, k0, chunk_k,
                                                               min(chunk_k, k - k0), q8, meta);
        __syncthreads();

        float acc[Rows][Cols];
#pragma unroll
        for (int r = 0; r < Rows; ++r) {
#pragma unroll
            for (int t = 0; t < Cols; ++t) acc[r][t] = 0.0f;
        }
        int item_index = warp;
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
                    std::uint32_t even[Rows][4], odd[Rows][4];
                    float scale[Rows];
#pragma unroll
                    for (int r = 0; r < Rows; ++r) {
                        Format::template dp4a_word<0>(current[r][u], even[r][0], odd[r][0]);
                        Format::template dp4a_word<1>(current[r][u], even[r][1], odd[r][1]);
                        Format::template dp4a_word<2>(current[r][u], even[r][2], odd[r][2]);
                        Format::template dp4a_word<3>(current[r][u], even[r][3], odd[r][3]);
                        scale[r] = Format::scale(current[r][u]);
                    }
                    const int swap = (v >> 2) & 1;
#pragma unroll
                    for (int t = 0; t < Cols; ++t) {
                        const uint4* slot  = q8 + (t * chunk_slices + v) * 2;
                        const uint4 first  = slot[swap];     // words 0, 1
                        const uint4 second = slot[swap ^ 1]; // words 2, 3
                        const float2 m     = meta[t * chunk_slices + v];
#pragma unroll
                        for (int r = 0; r < Rows; ++r) {
                            int s = __float_as_int(m.y);
                            s     = __dp4a(static_cast<int>(even[r][0]), static_cast<int>(first.x), s);
                            s     = __dp4a(static_cast<int>(odd[r][0]), static_cast<int>(first.y), s);
                            s     = __dp4a(static_cast<int>(even[r][1]), static_cast<int>(first.z), s);
                            s     = __dp4a(static_cast<int>(odd[r][1]), static_cast<int>(first.w), s);
                            s     = __dp4a(static_cast<int>(even[r][2]), static_cast<int>(second.x), s);
                            s     = __dp4a(static_cast<int>(odd[r][2]), static_cast<int>(second.y), s);
                            s     = __dp4a(static_cast<int>(even[r][3]), static_cast<int>(second.z), s);
                            s     = __dp4a(static_cast<int>(odd[r][3]), static_cast<int>(second.w), s);
                            acc[r][t] = fmaf(scale[r] * m.x, static_cast<float>(s), acc[r][t]);
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

// Launches the dp4a streaming GEMV if the format has the form and the shape fits, whatever the
// stream's setting; returns false otherwise. The verify route (try_launch_stream_dp4a_rows) and
// the drafter route (stream_draft_dp4a.cuh) each decide whether a call takes it.
template <class Format, int Rows, int Cols, class Sink>
bool launch_stream_dp4a_rows(const __nv_bfloat16* x, std::int64_t x_stride,
                             const typename Format::Planes& planes, Sink sink, std::int32_t items,
                             std::int32_t row_offset, std::int32_t k, std::int32_t padded_k,
                             cudaStream_t stream) {
    if constexpr (!kStreamDp4aCompiled || !StreamDp4aFormat<Format> || Cols > kStreamDp4aMaxCols) {
        return false;
    } else {
        if (k % 256 != 0 || x_stride % 8 != 0) return false;

        const std::int32_t chunk_k = stream_dp4a_chunk_k(k);
        const auto launch = [&]<int Warps, int Unroll>() {
            const auto single  = stream_dp4a_kernel<Format, Warps, Unroll, Rows, Cols, false, Sink>;
            const auto chunked = stream_dp4a_kernel<Format, Warps, Unroll, Rows, Cols, true, Sink>;
            configure_cuda_device_once([&] {
                const cudaError_t first = cudaFuncSetAttribute(
                    single, cudaFuncAttributeMaxDynamicSharedMemorySize, kMaxDynamicSharedBytes);
                return first != cudaSuccess
                           ? first
                           : cudaFuncSetAttribute(chunked,
                                                  cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                  kMaxDynamicSharedBytes);
            });
            int device = 0;
            int sms    = 0;
            CUDA_CHECK(cudaGetDevice(&device));
            CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device));
            const std::size_t stage = stream_dp4a_stage_bytes(chunk_k, Cols);
            const std::size_t budget = static_cast<std::size_t>(kMaxDynamicSharedBytes);
            if (chunk_k == k) {
                int per_sm = 0;
                CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, single,
                                                                         Warps * 32, stage));
                const int blocks =
                    std::max(1, std::min(div_up(items, Warps), sms * std::max(per_sm, 1)));
                single<<<blocks, Warps * 32, stage, stream>>>(x, x_stride, planes, sink, items,
                                                             row_offset, k, padded_k, chunk_k);
            } else {
                // One CTA per SM, or further waves while a CTA's partial sums would take more
                // than what the stage leaves.
                const auto partial_for = [&](int b) {
                    return static_cast<std::size_t>(div_up(items, b * Warps)) * Warps * Rows *
                           Cols * sizeof(float);
                };
                const int max_blocks = std::max(1, div_up(items, Warps));
                int blocks           = std::max(1, std::min(max_blocks, sms));
                while (stage + partial_for(blocks) > budget && blocks < max_blocks) {
                    blocks = std::min(max_blocks, blocks + sms);
                }
                if (stage + partial_for(blocks) > budget) return false;
                chunked<<<blocks, Warps * 32, stage + partial_for(blocks), stream>>>(
                    x, x_stride, planes, sink, items, row_offset, k, padded_k, chunk_k);
            }
            CUDA_CHECK(cudaGetLastError());
            return true;
        };
        if constexpr (Cols == 1) {
            if (k <= 6144) return launch.template operator()<8, 1>();
            return launch.template operator()<16, Rows == 1 ? 2 : 1>();
        } else {
            return launch.template operator()<16, 1>();
        }
    }
}

// Launches the dp4a streaming GEMV if the stream takes the verify route (stream_a8.h), the format
// serves it and the shape fits; returns false otherwise (the caller falls back to the other
// streaming GEMVs).
template <class Format, int Rows, int Cols, class Sink>
bool try_launch_stream_dp4a_rows(const __nv_bfloat16* x, std::int64_t x_stride,
                                 const typename Format::Planes& planes, Sink sink,
                                 std::int32_t items, std::int32_t row_offset, std::int32_t k,
                                 std::int32_t padded_k, cudaStream_t stream) {
    if constexpr (!kStreamDp4aCompiled || !StreamDp4aVerifyFormat<Format>) {
        return false;
    } else {
        if (!stream_dp4a_enabled(stream)) return false;
        return launch_stream_dp4a_rows<Format, Rows, Cols>(x, x_stride, planes, sink, items,
                                                           row_offset, k, padded_k, stream);
    }
}

} // namespace ninfer::ops::detail
