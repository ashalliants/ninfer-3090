// Adapted from Infernix a3edb450 src/ops/hyper_connection/hyper_connection_mix_fused.cu
// (Apache-2.0).
// Modified for NInfer-3090: BF16 contiguous weights only (Infernix's Q8 codec is dropped), its
// no-L1-allocate weight load kept local to this file, the norm weight is the stored FP32
// multiplier (Rn = R * inv * w), and the namespaces and includes are NInfer's. Re-fitted to the
// RTX 3090's 82 SMs: K1a stages 4 columns per pass (40 KiB) at two CTAs per SM, and K1b spreads a
// warp's S * rank / 8 W_up chunks over all lanes (5 per lane at rank 320, where Infernix's layout
// left 24 of 32 lanes idle on the second chunk) at two 512-thread CTAs per SM, so each kernel's
// 164 / 160 CTAs run in one wave instead of two. K1b's summation order changes accordingly.
//
// The fused route of hyper_connection_mix: two kernels instead of norm, the down projection,
// gates, the up projection and collapse.
//
// K1a hc_down_partials: one CTA per (stream s, 8 rows of W_down), a row per warp. The CTA stages
// its stream's slice of R for up to 4 columns per pass, reduces each column's sum of squares in a fixed
// order (per-thread sums over its elements in index order, a warp butterfly, warps in order),
// forms Rn = R * inv * w in FP32 shared memory and computes the slice's dot products in the
// lane-chunk order (lane l owns weights k = 8l + 256m, fmaf in k order, a warp butterfly).
// Outputs: FP32 partial[s][row][t]; the row-group-0 CTA of each stream writes inv[s][t].
//
// K1b hc_up_collapse: one CTA per 16 features d, a warp per d. Every CTA sums the partials in the
// fixed order s = 0..S-1 into z, forms m = SiLU(z / S) in FP32 shared memory (CTA 0 also writes
// inject = 2 sigmoid(z[rank + s] / S)), computes the S rows (s*H + d) of W_up against m (per
// lane, each of its chunks' eight products in k order added to that stream's sum in chunk order,
// then a warp butterfly per stream), and lane 0 writes x[d] = bf16((1/S) * sum_s sigmoid(u_s) * Rn_s) with Rn_s
// formed in FP32 from R, inv and w.
//
// No intermediate is rounded to BF16: Rn, z, m and u stay FP32. Each column is computed
// independently in a fixed order, so a column's result does not depend on how many columns share
// the call.

#include "ops/hyper_connection/hyper_connection_mix_fused.h"

#include "ops/common/math.cuh"
#include "ops/common/memory.cuh"

#include <cuda_bf16.h>

#include <stdexcept>
#include <string>

namespace ninfer::ops::detail {
namespace {

using bf16 = __nv_bfloat16;

constexpr int kStreams     = 4;
constexpr int kHidden      = 2560;
constexpr int kChunks      = kHidden / 256; // a lane's 8-weight chunks per stream slice
constexpr int kMaxRank     = 512;
constexpr int kDownRows    = 8; // rows of W_down per CTA (a row per warp)
constexpr int kDownThreads = kDownRows * 32;
constexpr int kPassColumns = 4; // 40 KiB of Rn: two CTAs per SM
constexpr int kUpFeatures  = 16; // features d per CTA (a d per warp)
constexpr int kUpThreads   = kUpFeatures * 32;
constexpr int kUpMinBlocks = 2; // 160 CTAs in one wave on 82 SMs
constexpr std::size_t kDownSharedBytes =
    static_cast<std::size_t>(kPassColumns) * kHidden * sizeof(float);

void check_launch(const char* what) {
    const cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        throw std::runtime_error(std::string("hyper_connection_mix ") + what + ": " +
                                 cudaGetErrorString(error));
    }
}

// Read-only global load that does not allocate in L1: weights streamed once per call (Infernix's
// ops/common/memory.cuh ld_nc_na, 16-byte form; sm_70+).
__device__ __forceinline__ uint4 ld_nc_na_16(const void* ptr) {
    uint4 value;
    asm("ld.global.nc.L1::no_allocate.v4.u32 {%0, %1, %2, %3}, [%4];"
        : "=r"(value.x), "=r"(value.y), "=r"(value.z), "=r"(value.w)
        : "l"(ptr));
    return value;
}

__device__ __forceinline__ float sigmoidf(float x) { return 1.0F / (1.0F + expf(-x)); }

__device__ __forceinline__ float warp_sum(float value) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_xor_sync(0xFFFFFFFFU, value, offset);
    }
    return value;
}

// Fixed-order block sum: warp butterflies, then the warps' sums in warp order.
template <int Threads>
__device__ __forceinline__ float block_sum(float value, float* scratch) {
    value          = warp_sum(value);
    const int warp = static_cast<int>(threadIdx.x) >> 5, lane = static_cast<int>(threadIdx.x) & 31;
    if (lane == 0) { scratch[warp] = value; }
    __syncthreads();
    float total = 0.0F;
#pragma unroll
    for (int w = 0; w < Threads / 32; ++w) { total += scratch[w]; }
    __syncthreads();
    return total;
}

// BF16 contiguous: rows of `stride` elements.
struct Bf16Codec {
    struct Params {
        const bf16* data;
        int stride;
    };
    struct Chunk {
        uint4 bits;
    };
    __device__ static Chunk load(const Params& p, std::int64_t row, int k) {
        return {ld_nc_na_16(p.data + row * p.stride + k)};
    }
    __device__ static void decode(const Chunk& c, float (&w)[8]) {
        const float2 a = bf16x2_bits_to_float2(c.bits.x), b = bf16x2_bits_to_float2(c.bits.y);
        const float2 d = bf16x2_bits_to_float2(c.bits.z), e = bf16x2_bits_to_float2(c.bits.w);
        w[0] = a.x, w[1] = a.y, w[2] = b.x, w[3] = b.y, w[4] = d.x, w[5] = d.y, w[6] = e.x,
        w[7] = e.y;
    }
};

template <class Codec>
__global__ __launch_bounds__(kDownThreads, 2) void hc_down_partials_kernel(
    const bf16* __restrict__ residual, const float* __restrict__ norm_weight,
    typename Codec::Params down, int rows, int columns, float eps, float* __restrict__ partial,
    float* __restrict__ inv_out) {
    extern __shared__ __align__(16) float rn[]; // [kPassColumns][kHidden]
    __shared__ float scratch[kDownThreads / 32];
    __shared__ float inv[kPassColumns];
    const int lane = static_cast<int>(threadIdx.x) & 31, warp = static_cast<int>(threadIdx.x) >> 5;
    const int s    = static_cast<int>(blockIdx.y);
    const int row  = static_cast<int>(blockIdx.x) * kDownRows + warp;
    const bool live     = row < rows;
    constexpr int width = kStreams * kHidden;

    // 1. Weights first: this warp's row of the stream's K slice, before any activation.
    const std::int64_t source = live ? row : 0;
    typename Codec::Chunk chunk[kChunks];
#pragma unroll
    for (int m = 0; m < kChunks; ++m) {
        chunk[m] = Codec::load(down, source, s * kHidden + m * 256 + lane * 8);
    }
    const float* w = norm_weight + static_cast<std::int64_t>(s) * kHidden;

#pragma unroll 1
    for (int c0 = 0; c0 < columns; c0 += kPassColumns) {
        const int pass = min(kPassColumns, columns - c0);
        // 2. Stage the slice in FP32 and reduce each column's sum of squares in a fixed order.
#pragma unroll 1
        for (int c = 0; c < pass; ++c) {
            const bf16* r = residual + static_cast<std::int64_t>(c0 + c) * width +
                            static_cast<std::int64_t>(s) * kHidden;
            float sum = 0.0F;
            for (int i = static_cast<int>(threadIdx.x) * 8; i < kHidden; i += kDownThreads * 8) {
                const uint4 values = load_ldg<uint4>(r + i);
                const float2 v0 = bf16x2_bits_to_float2(values.x), v1 = bf16x2_bits_to_float2(values.y);
                const float2 v2 = bf16x2_bits_to_float2(values.z), v3 = bf16x2_bits_to_float2(values.w);
                const float v[8] = {v0.x, v0.y, v1.x, v1.y, v2.x, v2.y, v3.x, v3.y};
                float* out       = rn + c * kHidden + i;
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    out[j] = v[j];
                    sum += v[j] * v[j];
                }
            }
            const float total = block_sum<kDownThreads>(sum, scratch);
            if (threadIdx.x == 0) { inv[c] = rsqrtf(total / static_cast<float>(kHidden) + eps); }
        }
        __syncthreads();
        // Rn = R * inv * w, FP32.
        for (int i = static_cast<int>(threadIdx.x); i < pass * kHidden; i += kDownThreads) {
            const int c = i / kHidden, d = i - c * kHidden;
            rn[i]       = rn[i] * inv[c] * w[d];
        }
        if (blockIdx.x == 0 && static_cast<int>(threadIdx.x) < pass) {
            inv_out[static_cast<std::int64_t>(s) * columns + c0 + static_cast<int>(threadIdx.x)] =
                inv[threadIdx.x];
        }
        __syncthreads();
        // 3. The row's dot products: the lane-chunk order, then a warp butterfly.
        float acc[kPassColumns];
#pragma unroll
        for (int c = 0; c < kPassColumns; ++c) { acc[c] = 0.0F; }
#pragma unroll
        for (int m = 0; m < kChunks; ++m) {
            const int kk = m * 256 + lane * 8;
            float wv[8];
            Codec::decode(chunk[m], wv);
#pragma unroll
            for (int c = 0; c < kPassColumns; ++c) {
                if (c < pass) {
                    const float4 a = *reinterpret_cast<const float4*>(rn + c * kHidden + kk);
                    const float4 b = *reinterpret_cast<const float4*>(rn + c * kHidden + kk + 4);
                    acc[c]         = fmaf(wv[0], a.x, acc[c]);
                    acc[c]         = fmaf(wv[1], a.y, acc[c]);
                    acc[c]         = fmaf(wv[2], a.z, acc[c]);
                    acc[c]         = fmaf(wv[3], a.w, acc[c]);
                    acc[c]         = fmaf(wv[4], b.x, acc[c]);
                    acc[c]         = fmaf(wv[5], b.y, acc[c]);
                    acc[c]         = fmaf(wv[6], b.z, acc[c]);
                    acc[c]         = fmaf(wv[7], b.w, acc[c]);
                }
            }
        }
#pragma unroll
        for (int c = 0; c < kPassColumns; ++c) {
            if (c < pass) {
                const float total = warp_sum(acc[c]);
                if (lane == 0 && live) {
                    partial[(static_cast<std::int64_t>(s) * rows + row) * columns + c0 + c] = total;
                }
            }
        }
        __syncthreads(); // the next pass overwrites rn
    }
}

// A warp's S rows (s*H + d) of W_up hold S * rank / 8 chunks of 8 weights; lane l owns chunks
// l, l + 32, ... of that list (chunk j is stream j / (rank / 8), weights 8 (j % (rank / 8)) on), so
// a lane carries at most PerLane of them and, for rank 320, no lane idles.
template <class Codec, int PerLane>
__global__ __launch_bounds__(kUpThreads, kUpMinBlocks) void hc_up_collapse_kernel(
    const float* __restrict__ partial, const float* __restrict__ inv_in,
    const bf16* __restrict__ residual, const float* __restrict__ norm_weight,
    typename Codec::Params up, int rank, int down_rows, int columns, float* __restrict__ inject,
    bf16* __restrict__ x) {
    __shared__ __align__(16) float m[kHcMixFusedMaxColumns * kMaxRank]; // [T][rank]
    __shared__ float inv[kStreams * kHcMixFusedMaxColumns];             // [S][T]
    const int lane = static_cast<int>(threadIdx.x) & 31, warp = static_cast<int>(threadIdx.x) >> 5;
    const int d    = static_cast<int>(blockIdx.x) * kUpFeatures + warp;
    const bool live     = d < kHidden;
    constexpr int width = kStreams * kHidden;
    const int chunks    = rank / 8;
    const int total     = kStreams * chunks;

    // 1. Weights first: this lane's chunks of the S rows (s*H + d) of W_up.
    typename Codec::Chunk chunk[PerLane];
#pragma unroll
    for (int i = 0; i < PerLane; ++i) {
        const int j = lane + 32 * i;
        if (j < total) {
            const int s = j / chunks, c = j - s * chunks;
            chunk[i] = Codec::load(up, static_cast<std::int64_t>(s) * kHidden + (live ? d : 0), c * 8);
        } else {
            chunk[i] = typename Codec::Chunk{};
        }
    }

    // 2. z = sum_s partial[s] in order s = 0..S-1; m = SiLU(z / S); CTA 0 writes the injects.
    const float inv_streams = 1.0F / static_cast<float>(kStreams);
    for (int i = static_cast<int>(threadIdx.x); i < down_rows * columns; i += kUpThreads) {
        const int k = i / columns, t = i - k * columns;
        float z     = 0.0F;
#pragma unroll
        for (int s = 0; s < kStreams; ++s) {
            z += partial[(static_cast<std::int64_t>(s) * down_rows + k) * columns + t];
        }
        const float v = z * inv_streams;
        if (k < rank) {
            m[t * rank + k] = v * sigmoidf(v);
        } else if (inject != nullptr && blockIdx.x == 0) {
            inject[static_cast<std::int64_t>(t) * kStreams + (k - rank)] = 2.0F * sigmoidf(v);
        }
    }
    for (int i = static_cast<int>(threadIdx.x); i < kStreams * columns; i += kUpThreads) {
        inv[i] = inv_in[i];
    }
    __syncthreads();
    if (!live) { return; }

    // 3. u_s = W_up[s*H + d] . m: per lane, each chunk's eight products in k order, added to its
    //    stream's sum in chunk order; a warp butterfly per stream, then the collapse in lane 0.
    float w_norm[kStreams];
#pragma unroll
    for (int s = 0; s < kStreams; ++s) {
        w_norm[s] = norm_weight[static_cast<std::int64_t>(s) * kHidden + d];
    }
#pragma unroll 1
    for (int t = 0; t < columns; ++t) {
        const float* mt = m + t * rank;
        float acc[kStreams];
#pragma unroll
        for (int s = 0; s < kStreams; ++s) { acc[s] = 0.0F; }
#pragma unroll
        for (int i = 0; i < PerLane; ++i) {
            const int j = lane + 32 * i;
            if (j < total) {
                const int s    = j / chunks, c = j - s * chunks;
                const float4 a = *reinterpret_cast<const float4*>(mt + c * 8);
                const float4 b = *reinterpret_cast<const float4*>(mt + c * 8 + 4);
                float wv[8];
                Codec::decode(chunk[i], wv);
                float p = wv[0] * a.x;
                p       = fmaf(wv[1], a.y, p);
                p       = fmaf(wv[2], a.z, p);
                p       = fmaf(wv[3], a.w, p);
                p       = fmaf(wv[4], b.x, p);
                p       = fmaf(wv[5], b.y, p);
                p       = fmaf(wv[6], b.z, p);
                p       = fmaf(wv[7], b.w, p);
#pragma unroll
                for (int q = 0; q < kStreams; ++q) {
                    if (q == s) { acc[q] += p; }
                }
            }
        }
#pragma unroll
        for (int s = 0; s < kStreams; ++s) { acc[s] = warp_sum(acc[s]); }
        if (lane == 0) {
            float sum = 0.0F;
#pragma unroll
            for (int s = 0; s < kStreams; ++s) {
                const float r = __bfloat162float(
                    residual[static_cast<std::int64_t>(t) * width +
                             static_cast<std::int64_t>(s) * kHidden + d]);
                const float rn = r * inv[s * columns + t] * w_norm[s];
                sum += sigmoidf(acc[s]) * rn;
            }
            x[static_cast<std::int64_t>(t) * kHidden + d] = __float2bfloat16_rn(sum * inv_streams);
        }
    }
}

template <class Codec, int PerLane>
void launch_up(const float* partial, const float* inv, const bf16* residual,
               const float* norm_weight, typename Codec::Params up, int rank, int down_rows,
               int columns, float* inject, bf16* x, cudaStream_t stream) {
    hc_up_collapse_kernel<Codec, PerLane>
        <<<(kHidden + kUpFeatures - 1) / kUpFeatures, kUpThreads, 0, stream>>>(
            partial, inv, residual, norm_weight, up, rank, down_rows, columns, inject, x);
}

int bf16_stride(const Weight& w) noexcept {
    return w.padded_shape[1] >= w.k ? w.padded_shape[1] : w.k;
}

bool bf16_contiguous(const Weight& w) noexcept {
    return w.qtype == QType::BF16 && w.layout == QuantLayout::Contiguous && w.qdata != nullptr &&
           reinterpret_cast<std::uintptr_t>(w.qdata) % 16 == 0 && bf16_stride(w) % 8 == 0;
}

Bf16Codec::Params bf16_params(const Weight& w) {
    return {static_cast<const bf16*>(w.qdata), bf16_stride(w)};
}

template <class Codec>
void launch(const Tensor& residual, const Tensor& norm_weight, typename Codec::Params down,
            typename Codec::Params up, int down_rows, int rank, int streams, float eps, Tensor& x,
            Tensor* inject, void* partials, cudaStream_t stream) {
    static const bool configured = [] {
        const cudaError_t error = cudaFuncSetAttribute(
            hc_down_partials_kernel<Codec>, cudaFuncAttributeMaxDynamicSharedMemorySize,
            static_cast<int>(kDownSharedBytes));
        if (error != cudaSuccess) {
            throw std::runtime_error(std::string("hyper_connection_mix: shared memory: ") +
                                     cudaGetErrorString(error));
        }
        return true;
    }();
    (void)configured;
    const int columns = residual.ne[1];
    auto* partial     = static_cast<float*>(partials);
    const std::size_t partial_bytes =
        (static_cast<std::size_t>(streams) * down_rows * columns * sizeof(float) + 255) / 256 * 256;
    auto* inv = reinterpret_cast<float*>(static_cast<std::byte*>(partials) + partial_bytes);
    hc_down_partials_kernel<Codec>
        <<<dim3((down_rows + kDownRows - 1) / kDownRows, streams), kDownThreads, kDownSharedBytes,
           stream>>>(static_cast<const bf16*>(residual.data),
                     static_cast<const float*>(norm_weight.data), down, down_rows, columns, eps,
                     partial, inv);
    check_launch("down partials");
    using UpLaunch = void (*)(const float*, const float*, const bf16*, const float*,
                              typename Codec::Params, int, int, int, float*, bf16*, cudaStream_t);
    // Chunks per lane: ceil(S * rank / 8 / 32), 1..8 for rank 8..512.
    constexpr UpLaunch kUp[] = {launch_up<Codec, 1>, launch_up<Codec, 2>, launch_up<Codec, 3>,
                                launch_up<Codec, 4>, launch_up<Codec, 5>, launch_up<Codec, 6>,
                                launch_up<Codec, 7>, launch_up<Codec, 8>};
    const int per_lane = (kStreams * (rank / 8) + 31) / 32;
    kUp[per_lane - 1](partial, inv, static_cast<const bf16*>(residual.data),
                      static_cast<const float*>(norm_weight.data), up, rank, down_rows, columns,
                      inject != nullptr ? static_cast<float*>(inject->data) : nullptr,
                      static_cast<bf16*>(x.data), stream);
    check_launch("up collapse");
}

} // namespace

bool hc_mix_fused_supported(const Weight& down, const Weight& up, std::int32_t hidden,
                            std::int32_t streams, std::int32_t rank,
                            std::int32_t columns) noexcept {
    return bf16_contiguous(down) && bf16_contiguous(up) && columns >= 1 &&
           columns <= kHcMixFusedMaxColumns && hidden == kHidden && streams == kStreams &&
           rank > 0 && rank % 8 == 0 && rank <= kMaxRank && down.k == kStreams * kHidden &&
           (down.n == rank || down.n == rank + kStreams) && up.n == kStreams * kHidden &&
           up.k == rank;
}

std::size_t hc_mix_fused_workspace_bytes(std::int32_t down_rows, std::int32_t streams,
                                         std::int32_t columns) noexcept {
    const auto align = [](std::size_t bytes) { return (bytes + 255) / 256 * 256; };
    return align(static_cast<std::size_t>(streams) * down_rows * columns * sizeof(float)) +
           align(static_cast<std::size_t>(streams) * columns * sizeof(float));
}

void hc_mix_fused(const Tensor& residual, const Tensor& norm_weight, const Weight& down,
                  const Weight& up, std::int32_t streams, std::int32_t rank, float eps, Tensor& x,
                  Tensor* inject, void* partials, cudaStream_t stream) {
    if (!hc_mix_fused_supported(down, up, x.ne[0], streams, rank, residual.ne[1])) {
        throw std::invalid_argument("hyper_connection_mix: the fused route does not serve this problem");
    }
    if (inject != nullptr && down.n != rank + streams) {
        throw std::invalid_argument(
            "hyper_connection_mix: injects need the down projection's injection rows");
    }
    launch<Bf16Codec>(residual, norm_weight, bf16_params(down), bf16_params(up), down.n, rank,
                      streams, eps, x, inject, partials, stream);
}

} // namespace ninfer::ops::detail
