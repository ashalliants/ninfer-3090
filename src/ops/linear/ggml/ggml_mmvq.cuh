#pragma once

// Decode route of the GGML linears (T <= 8): the Q8_1 cast of x into shared memory, then a dp4a
// GEMV. Adapted from Strata src/kernels/cuda/native_mmvq.cu (MIT, see ggml_blocks.cuh for the
// notice); modified for NInfer: persistent CTAs that cast x once, a fixed lanes-per-row split
// derived from K so every lane owns whole 32-value sub-blocks, and the decode-once multi-column
// form for every T.

#include "ops/linear/ggml/ggml_blocks.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace ninfer::ops::ggml {

inline constexpr int kMmvqThreads = 256;
inline constexpr int kMmvqWarps   = kMmvqThreads / 32;

// Lanes that share one row: the widest power of two that divides the row's sub-block count and
// still leaves each lane at least four sub-blocks to keep loads in flight.
__host__ __device__ constexpr int mmvq_lanes_per_row(int sub_blocks) {
    for (int lanes = 32; lanes > 1; lanes /= 2) {
        if (sub_blocks % lanes == 0 && sub_blocks / lanes >= 4) { return lanes; }
    }
    return 1;
}

template <QType F, int K, int T>
constexpr std::size_t mmvq_shared_bytes() {
    return static_cast<std::size_t>(T) * K + static_cast<std::size_t>(T) * (K / 32) * 4 +
           (kNeedsIq3sGrid<F> ? 512 * 4 : 0);
}

// x is BF16 [K,T] (column t at x + t K); out is BF16 [N,T]. The cast stores sub-block s of column
// t as two 16-byte halves at t K + 16 s and t K + K / 2 + 16 s, so the lanes of a row, which read
// consecutive sub-blocks, hit distinct banks.
template <QType F, int K, int T, bool Residual>
__global__ void __launch_bounds__(kMmvqThreads)
    ggml_mmvq_kernel(const std::uint8_t* __restrict__ w, const __nv_bfloat16* __restrict__ x,
                     __nv_bfloat16* __restrict__ out, int n) {
    using Fmt                 = Format<F>;
    constexpr int kSubs       = K / kSubValues;
    constexpr int kLanes      = mmvq_lanes_per_row(kSubs);
    constexpr int kRowsWarp   = 32 / kLanes;
    constexpr int kSubsLane   = kSubs / kLanes;
    constexpr int kRowBytes   = K / Fmt::kBlockValues * Fmt::kBlockBytes;
    static_assert(K % Fmt::kBlockValues == 0 && kSubs % kLanes == 0);

    extern __shared__ __align__(16) unsigned char smem[];
    auto* xq_s   = reinterpret_cast<std::int8_t*>(smem);
    auto* xd_s   = reinterpret_cast<float*>(smem + static_cast<std::size_t>(T) * K);
    auto* grid_s = reinterpret_cast<std::uint32_t*>(smem + static_cast<std::size_t>(T) * K +
                                                    static_cast<std::size_t>(T) * kSubs * 4);

    const int lane     = static_cast<int>(threadIdx.x) & 31;
    const int warp     = static_cast<int>(threadIdx.x) >> 5;
    const int sub_lane = lane % kLanes;
    const int groups   = n / kRowsWarp;
    const int stride   = static_cast<int>(gridDim.x) * kMmvqWarps;
    int rg             = static_cast<int>(blockIdx.x) * kMmvqWarps + warp;

    Tables tables;
    load_tables<F>(tables, grid_s);
    if constexpr (kNeedsIq3sGrid<F>) __syncthreads();

    // The first row group's weight loads go out before the cast, so the cast's reads of x overlap
    // them instead of delaying every CTA's first DRAM request.
    SubBlock sb[kSubsLane];
    const auto decode = [&](int group) {
        const std::uint8_t* wrow =
            w + static_cast<std::size_t>(group * kRowsWarp + lane / kLanes) * kRowBytes;
#pragma unroll
        for (int i = 0; i < kSubsLane; ++i) Fmt::decode(wrow, sub_lane + kLanes * i, tables, sb[i]);
    };
    if (rg < groups) decode(rg);

    for (int g = static_cast<int>(threadIdx.x); g < T * kSubs; g += kMmvqThreads) {
        const int t  = g / kSubs;
        const int gi = g - t * kSubs;
        int words[8];
        xd_s[g] = quantize_q8_1_group(x + static_cast<std::size_t>(t) * K + kSubValues * gi, words);
        *reinterpret_cast<int4*>(xq_s + t * K + 16 * gi) =
            make_int4(words[0], words[1], words[2], words[3]);
        *reinterpret_cast<int4*>(xq_s + t * K + K / 2 + 16 * gi) =
            make_int4(words[4], words[5], words[6], words[7]);
    }
    __syncthreads();

    for (; rg < groups; rg += stride) {
        const int row = rg * kRowsWarp + lane / kLanes;
        float acc[T];
#pragma unroll
        for (int t = 0; t < T; ++t) acc[t] = 0.0F;
#pragma unroll
        for (int i = 0; i < kSubsLane; ++i) {
            const int s = sub_lane + kLanes * i;
#pragma unroll
            for (int t = 0; t < T; ++t) {
                const int4 a   = *reinterpret_cast<const int4*>(xq_s + t * K + 16 * s);
                const int4 b   = *reinterpret_cast<const int4*>(xq_s + t * K + K / 2 + 16 * s);
                const float xd = xd_s[t * kSubs + s];
                int lo         = __dp4a(sb[i].v[0], a.x, 0);
                lo             = __dp4a(sb[i].v[1], a.y, lo);
                lo             = __dp4a(sb[i].v[2], a.z, lo);
                lo             = __dp4a(sb[i].v[3], a.w, lo);
                int hi         = __dp4a(sb[i].v[4], b.x, 0);
                hi             = __dp4a(sb[i].v[5], b.y, hi);
                hi             = __dp4a(sb[i].v[6], b.z, hi);
                hi             = __dp4a(sb[i].v[7], b.w, hi);
                if constexpr (Fmt::kSplitScale) {
                    acc[t] = fmaf(xd,
                                  fmaf(sb[i].scale[0], static_cast<float>(lo),
                                       sb[i].scale[1] * static_cast<float>(hi)),
                                  acc[t]);
                } else {
                    acc[t] = fmaf(static_cast<float>(lo + hi), sb[i].scale[0] * xd, acc[t]);
                }
            }
        }
        // The next group's loads are in flight while this one reduces and stores.
        if (rg + stride < groups) decode(rg + stride);
#pragma unroll
        for (int offset = kLanes / 2; offset > 0; offset >>= 1) {
#pragma unroll
            for (int t = 0; t < T; ++t) acc[t] += __shfl_xor_sync(0xFFFFFFFFU, acc[t], offset);
        }
#pragma unroll
        for (int t = 0; t < T; ++t) {
            if (sub_lane == t % kLanes) {
                const std::size_t o = static_cast<std::size_t>(t) * n + row;
                float y             = acc[t];
                if constexpr (Residual) y += __bfloat162float(out[o]);
                out[o] = __float2bfloat16_rn(y);
            }
        }
    }
}

inline void mmvq_check(cudaError_t error, const char* what) {
    if (error != cudaSuccess) {
        throw std::runtime_error(std::string("ggml mmvq: ") + what + ": " +
                                 cudaGetErrorString(error));
    }
}

// Grid of persistent CTAs: as many as fit at once, never more than there are row groups.
template <QType F, int K, int T, bool Residual>
void launch_mmvq_instance(const std::uint8_t* w, const __nv_bfloat16* x, __nv_bfloat16* out, int n,
                          cudaStream_t stream) {
    constexpr std::size_t kShared = mmvq_shared_bytes<F, K, T>();
    constexpr int kRowsCta =
        kMmvqWarps * (32 / mmvq_lanes_per_row(K / kSubValues));
    static const int resident = [] {
        auto kernel = ggml_mmvq_kernel<F, K, T, Residual>;
        if (kShared > 48 * 1024) {
            mmvq_check(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                            static_cast<int>(kShared)),
                       "shared memory opt-in");
        }
        int device = 0;
        int sms    = 0;
        int per_sm = 0;
        mmvq_check(cudaGetDevice(&device), "device");
        mmvq_check(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device), "SMs");
        mmvq_check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, kernel, kMmvqThreads,
                                                                 kShared),
                   "occupancy");
        return sms * (per_sm > 0 ? per_sm : 1);
    }();
    const int ctas_needed = (n + kRowsCta - 1) / kRowsCta;
    const int grid        = ctas_needed < resident ? ctas_needed : resident;
    ggml_mmvq_kernel<F, K, T, Residual><<<grid, kMmvqThreads, kShared, stream>>>(w, x, out, n);
    mmvq_check(cudaGetLastError(), "launch");
}

template <QType F, int K, bool Residual>
void launch_mmvq_k(const std::uint8_t* w, const __nv_bfloat16* x, __nv_bfloat16* out, int n, int t,
                   cudaStream_t stream) {
    switch (t) {
    case 1: return launch_mmvq_instance<F, K, 1, Residual>(w, x, out, n, stream);
    case 2: return launch_mmvq_instance<F, K, 2, Residual>(w, x, out, n, stream);
    case 3: return launch_mmvq_instance<F, K, 3, Residual>(w, x, out, n, stream);
    case 4: return launch_mmvq_instance<F, K, 4, Residual>(w, x, out, n, stream);
    case 5: return launch_mmvq_instance<F, K, 5, Residual>(w, x, out, n, stream);
    case 6: return launch_mmvq_instance<F, K, 6, Residual>(w, x, out, n, stream);
    case 7: return launch_mmvq_instance<F, K, 7, Residual>(w, x, out, n, stream);
    case 8: return launch_mmvq_instance<F, K, 8, Residual>(w, x, out, n, stream);
    default: break;
    }
    throw std::invalid_argument("ggml mmvq: T must be in [1, 8]");
}

} // namespace ninfer::ops::ggml
