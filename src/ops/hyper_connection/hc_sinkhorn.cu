// DeepSeek-V4's hyper-connection arithmetic (Sinkhorn mixing between the residual streams); the
// contract is include/ninfer/ops/hc_sinkhorn.h. Written for NInfer-3090 against the model repo's
// `Block.hc_mixes / hc_pre / hc_post` and `hc_split_sinkhorn` (inference/model.py, kernel.py).

#include "ninfer/ops/hc_sinkhorn.h"

#include <cuda_bf16.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace ninfer::ops {

namespace {

using bf16 = __nv_bfloat16;

constexpr int kS        = kHcSinkhornStreams;
constexpr int kMixRows  = kHcSinkhornMixRows;
constexpr int kMixThreads = 512;
constexpr int kMixWarps   = kMixThreads / 32;
constexpr int kElementThreads = 256;
constexpr std::int32_t kMaxStreamWidth = 32768;
constexpr std::int32_t kMaxIters       = 64;
// Feature tiles of an element-wise launch ride grid.y (at most 65,535); columns ride grid.x.
constexpr std::int32_t kMaxHidden      = 65535 * kElementThreads;

void require(bool condition, const char* message) {
    if (!condition) { throw std::invalid_argument(std::string("hc_sinkhorn: ") + message); }
}

void check_launch(const char* what) {
    const cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        throw std::runtime_error(std::string("hc_sinkhorn ") + what + ": " + cudaGetErrorString(error));
    }
}

bool contiguous_2d(const Tensor& t, DType dtype) {
    return t.data != nullptr && t.dtype == dtype && t.is_contiguous() && t.ne[2] == 1 && t.ne[3] == 1;
}

__device__ __forceinline__ float sigmoidf(float x) { return 1.0F / (1.0F + expf(-x)); }

// One block per column: the statistic over the whole K and all (2 + S) * S projection rows in one
// pass over the column (each residual value is read once and meets every row), then the split and the
// Sinkhorn iterations on thread 0 (a 4 x 4 matrix).
__global__ void __launch_bounds__(kMixThreads)
mixes_kernel(const bf16* __restrict__ residual, const float* __restrict__ fn,
             const float* __restrict__ scale, const float* __restrict__ base, int width,
             float norm_eps, float eps, int iters, float* __restrict__ pre, float* __restrict__ post,
             float* __restrict__ comb) {
    __shared__ float partial[kMixWarps][kMixRows + 1];
    __shared__ float total[kMixRows + 1];

    const int t = blockIdx.x;
    const bf16* column = residual + static_cast<std::size_t>(t) * width;

    float acc[kMixRows];
    float sumsq = 0.0F;
#pragma unroll
    for (int n = 0; n < kMixRows; ++n) { acc[n] = 0.0F; }

    for (int c = threadIdx.x; c < width / 4; c += kMixThreads) {
        const uint2 raw = *reinterpret_cast<const uint2*>(column + 4 * c);
        const __nv_bfloat162 lo = *reinterpret_cast<const __nv_bfloat162*>(&raw.x);
        const __nv_bfloat162 hi = *reinterpret_cast<const __nv_bfloat162*>(&raw.y);
        const float v[4] = {__low2float(lo), __high2float(lo), __low2float(hi), __high2float(hi)};
        sumsq += v[0] * v[0] + v[1] * v[1] + v[2] * v[2] + v[3] * v[3];
#pragma unroll
        for (int n = 0; n < kMixRows; ++n) {
            const float4 w = *reinterpret_cast<const float4*>(fn + static_cast<std::size_t>(n) * width + 4 * c);
            acc[n] += w.x * v[0] + w.y * v[1] + w.z * v[2] + w.w * v[3];
        }
    }

    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
#pragma unroll
    for (int n = 0; n <= kMixRows; ++n) {
        float value = n < kMixRows ? acc[n] : sumsq;
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            value += __shfl_xor_sync(0xFFFFFFFFU, value, offset);
        }
        if (lane == 0) { partial[warp][n] = value; }
    }
    __syncthreads();
    if (threadIdx.x <= kMixRows) {
        float sum = 0.0F;
        for (int w = 0; w < kMixWarps; ++w) { sum += partial[w][threadIdx.x]; }
        total[threadIdx.x] = sum;
    }
    __syncthreads();
    if (threadIdx.x != 0) { return; }

    const float r = 1.0F / sqrtf(total[kMixRows] / static_cast<float>(width) + norm_eps);
    float m[kMixRows];
    for (int n = 0; n < kMixRows; ++n) { m[n] = total[n] * r; }

    for (int i = 0; i < kS; ++i) {
        pre[static_cast<std::size_t>(t) * kS + i] = sigmoidf(m[i] * scale[0] + base[i]) + eps;
        post[static_cast<std::size_t>(t) * kS + i] = 2.0F * sigmoidf(m[kS + i] * scale[1] + base[kS + i]);
    }

    float a[kS][kS];
    for (int i = 0; i < kS; ++i) {
        float peak = -INFINITY;
        for (int j = 0; j < kS; ++j) {
            a[i][j] = m[2 * kS + i * kS + j] * scale[2] + base[2 * kS + i * kS + j];
            peak    = fmaxf(peak, a[i][j]);
        }
        float sum = 0.0F;
        for (int j = 0; j < kS; ++j) {
            a[i][j] = expf(a[i][j] - peak);
            sum += a[i][j];
        }
        for (int j = 0; j < kS; ++j) { a[i][j] = a[i][j] / sum + eps; }
    }
    for (int round = 0; round < iters; ++round) {
        if (round > 0) {
            for (int i = 0; i < kS; ++i) {
                float sum = 0.0F;
                for (int j = 0; j < kS; ++j) { sum += a[i][j]; }
                for (int j = 0; j < kS; ++j) { a[i][j] = a[i][j] / (sum + eps); }
            }
        }
        for (int j = 0; j < kS; ++j) {
            float sum = 0.0F;
            for (int i = 0; i < kS; ++i) { sum += a[i][j]; }
            for (int i = 0; i < kS; ++i) { a[i][j] = a[i][j] / (sum + eps); }
        }
    }
    for (int i = 0; i < kS; ++i) {
        for (int j = 0; j < kS; ++j) { comb[static_cast<std::size_t>(t) * kS * kS + i * kS + j] = a[i][j]; }
    }
}

__global__ void collapse_kernel(const bf16* __restrict__ residual, const float* __restrict__ pre,
                                int hidden, bf16* __restrict__ x) {
    const int d = blockIdx.y * blockDim.x + threadIdx.x;
    const int t = blockIdx.x;
    if (d >= hidden) { return; }
    const std::size_t column = static_cast<std::size_t>(t) * kS * hidden;
    float sum = 0.0F;
#pragma unroll
    for (int s = 0; s < kS; ++s) {
        sum += pre[static_cast<std::size_t>(t) * kS + s] * __bfloat162float(residual[column + static_cast<std::size_t>(s) * hidden + d]);
    }
    x[static_cast<std::size_t>(t) * hidden + d] = __float2bfloat16_rn(sum);
}

__global__ void expand_mix_kernel(const bf16* __restrict__ y, const float* __restrict__ post,
                                  const float* __restrict__ comb, int hidden, bf16* residual) {
    const int d = blockIdx.y * blockDim.x + threadIdx.x;
    const int t = blockIdx.x;
    if (d >= hidden) { return; }
    const std::size_t column = static_cast<std::size_t>(t) * kS * hidden;
    float old[kS];
#pragma unroll
    for (int i = 0; i < kS; ++i) { old[i] = __bfloat162float(residual[column + static_cast<std::size_t>(i) * hidden + d]); }
    const float v = __bfloat162float(y[static_cast<std::size_t>(t) * hidden + d]);
#pragma unroll
    for (int j = 0; j < kS; ++j) {
        float sum = post[static_cast<std::size_t>(t) * kS + j] * v;
#pragma unroll
        for (int i = 0; i < kS; ++i) { sum += comb[static_cast<std::size_t>(t) * kS * kS + i * kS + j] * old[i]; }
        residual[column + static_cast<std::size_t>(j) * hidden + d] = __float2bfloat16_rn(sum);
    }
}

bool aligned16(const void* p) { return reinterpret_cast<std::uintptr_t>(p) % 16 == 0; }

bool overlaps(const Tensor& a, const Tensor& b) {
    const auto a0 = reinterpret_cast<std::uintptr_t>(a.data), b0 = reinterpret_cast<std::uintptr_t>(b.data);
    return a0 < b0 + b.bytes() && b0 < a0 + a.bytes();
}

} // namespace

void hc_sinkhorn_mixes(const Tensor& residual, const Tensor& fn, const Tensor& scale, const Tensor& base,
                       float norm_eps, float sinkhorn_eps, std::int32_t sinkhorn_iters, Tensor& pre,
                       Tensor& post, Tensor& comb, cudaStream_t stream) {
    require(contiguous_2d(residual, DType::BF16) && contiguous_2d(fn, DType::FP32) &&
                contiguous_2d(pre, DType::FP32) && contiguous_2d(post, DType::FP32) &&
                contiguous_2d(comb, DType::FP32) && contiguous_2d(scale, DType::FP32) &&
                contiguous_2d(base, DType::FP32),
            "mixes need contiguous BF16 residual and FP32 weights and outputs");
    const int width = residual.ne[0], columns = residual.ne[1];
    require(width % 8 == 0 && width > 0 && width <= kMaxStreamWidth && columns > 0 && width % kS == 0,
            "the residual width must be a multiple of 8 (and of S) up to 32,768");
    require(fn.ne[0] == width && fn.ne[1] == kMixRows && scale.numel() == 3 && base.numel() == kMixRows,
            "fn, scale or base do not match the residual width");
    require(pre.ne[0] == kS && pre.ne[1] == columns && post.ne[0] == kS && post.ne[1] == columns &&
                comb.ne[0] == kS * kS && comb.ne[1] == columns,
            "pre, post or comb have the wrong shape");
    require(aligned16(residual.data) && aligned16(fn.data), "the residual and fn must be 16-byte aligned");
    require(std::isfinite(norm_eps) && norm_eps > 0.0F && std::isfinite(sinkhorn_eps) && sinkhorn_eps >= 0.0F &&
                sinkhorn_iters >= 1 && sinkhorn_iters <= kMaxIters,
            "epsilons or iteration count out of range");
    for (const Tensor* out : {&pre, &post, &comb}) {
        for (const Tensor* in : {&residual, &fn, &scale, &base}) {
            require(!overlaps(*out, *in), "an output overlaps an input");
        }
    }
    require(!overlaps(pre, post) && !overlaps(pre, comb) && !overlaps(post, comb), "outputs overlap");
    mixes_kernel<<<columns, kMixThreads, 0, stream>>>(
        static_cast<const bf16*>(residual.data), static_cast<const float*>(fn.data),
        static_cast<const float*>(scale.data), static_cast<const float*>(base.data), width, norm_eps,
        sinkhorn_eps, sinkhorn_iters, static_cast<float*>(pre.data), static_cast<float*>(post.data),
        static_cast<float*>(comb.data));
    check_launch("mixes");
}

void hc_collapse(const Tensor& residual, const Tensor& pre, Tensor& x, cudaStream_t stream) {
    require(contiguous_2d(residual, DType::BF16) && contiguous_2d(pre, DType::FP32) &&
                contiguous_2d(x, DType::BF16),
            "collapse needs contiguous tensors");
    const int hidden = x.ne[0], columns = x.ne[1];
    require(hidden > 0 && hidden <= kMaxHidden && columns > 0 && residual.ne[0] == kS * hidden &&
                residual.ne[1] == columns && pre.ne[0] == kS && pre.ne[1] == columns,
            "collapse shapes disagree");
    require(!overlaps(x, residual) && !overlaps(x, pre), "collapse output overlaps an input");
    collapse_kernel<<<dim3(columns, (hidden + kElementThreads - 1) / kElementThreads), kElementThreads, 0, stream>>>(
        static_cast<const bf16*>(residual.data), static_cast<const float*>(pre.data), hidden,
        static_cast<bf16*>(x.data));
    check_launch("collapse");
}

void hc_expand_mix(const Tensor& y, const Tensor& post, const Tensor& comb, Tensor& residual,
                   cudaStream_t stream) {
    require(contiguous_2d(y, DType::BF16) && contiguous_2d(post, DType::FP32) &&
                contiguous_2d(comb, DType::FP32) && contiguous_2d(residual, DType::BF16),
            "expand_mix needs contiguous tensors");
    const int hidden = y.ne[0], columns = y.ne[1];
    require(hidden > 0 && hidden <= kMaxHidden && columns > 0 && residual.ne[0] == kS * hidden &&
                residual.ne[1] == columns && post.ne[0] == kS && post.ne[1] == columns && comb.ne[0] == kS * kS && comb.ne[1] == columns,
            "expand_mix shapes disagree");
    require(!overlaps(residual, y) && !overlaps(residual, post) && !overlaps(residual, comb),
            "the residual overlaps an input");
    expand_mix_kernel<<<dim3(columns, (hidden + kElementThreads - 1) / kElementThreads), kElementThreads, 0, stream>>>(
        static_cast<const bf16*>(y.data), static_cast<const float*>(post.data),
        static_cast<const float*>(comb.data), hidden, static_cast<bf16*>(residual.data));
    check_launch("expand_mix");
}

} // namespace ninfer::ops
