// Written for NInfer-3090. The contract is include/ninfer/ops/inkling2_adapters.h.

#include "ninfer/ops/inkling2_adapters.h"

#include <cuda_bf16.h>

#include <cstdint>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

constexpr int kThreads = 256;
constexpr int kWarps   = kThreads / 32;

void require(bool condition, const char* op, const std::string& message) {
    if (!condition) { throw std::invalid_argument(std::string(op) + ": " + message); }
}

// The column stride in elements of a [rows, cols] matrix of `dtype` (columns at any stride, rows contiguous).
std::int64_t column_stride(const Tensor& t, DType dtype, std::int32_t rows, std::int32_t cols, const char* op, const char* name) {
    const std::int64_t width = static_cast<std::int64_t>(dtype_size(dtype));
    require(t.data != nullptr && t.dtype == dtype, op, std::string(name) + " has the wrong type");
    require(t.ne[0] == rows && t.ne[1] == cols && t.ne[2] == 1 && t.ne[3] == 1, op, std::string(name) + " has the wrong shape");
    require(t.nb[0] == width && t.nb[1] % width == 0 && (cols == 1 || t.nb[1] >= rows * width), op,
            std::string(name) + " has a bad layout");
    return cols == 1 ? rows : t.nb[1] / width;
}

void launched(const char* op) {
    const cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) { throw std::runtime_error(std::string(op) + ": launch failed: " + cudaGetErrorString(error)); }
}

// Sums `v` over the block; every thread gets the total. `scratch` holds kWarps floats.
__device__ __forceinline__ float block_sum(float v, float* scratch) {
    for (int o = 16; o > 0; o >>= 1) { v += __shfl_xor_sync(0xffffffffU, v, o); }
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    __syncthreads(); // scratch may still be read by an earlier call
    if (lane == 0) { scratch[warp] = v; }
    __syncthreads();
    float total = 0.0F;
#pragma unroll
    for (int i = 0; i < kWarps; ++i) { total += scratch[i]; }
    return total;
}

// One block per column.
__global__ void __launch_bounds__(kThreads) low_rank_kernel(const __nv_bfloat16* x, std::int64_t xs, const float* a, const float* b,
                                                            int rank, float scale, int k_dim, int n_dim, float* y, std::int64_t ys) {
    __shared__ float scratch[kWarps];
    __shared__ float u[kLowRankMaxRank];
    const std::int64_t t = blockIdx.x;
    const __nv_bfloat16* xc = x + t * xs;
    for (int j = 0; j < rank; ++j) {
        float acc = 0.0F;
        for (int k = threadIdx.x; k < k_dim; k += kThreads) { acc = fmaf(a[static_cast<std::int64_t>(j) * k_dim + k], __bfloat162float(xc[k]), acc); }
        const float total = block_sum(acc, scratch);
        if (threadIdx.x == 0) { u[j] = scale * total; }
    }
    __syncthreads();
    float* yc = y + t * ys;
    for (int n = threadIdx.x; n < n_dim; n += kThreads) {
        float s = 0.0F;
        for (int j = 0; j < rank; ++j) { s = fmaf(b[static_cast<std::int64_t>(n) * rank + j], u[j], s); }
        yc[n] += s;
    }
}

__global__ void __launch_bounds__(kThreads) ablate_kernel(float* y, std::int64_t ys, const float* r, float w, const float* residual,
                                                          std::int64_t rs, float* sum, std::int64_t ss, int channels) {
    __shared__ float scratch[kWarps];
    const std::int64_t t = blockIdx.x;
    float* yc = y + t * ys;
    float acc = 0.0F;
    for (int c = threadIdx.x; c < channels; c += kThreads) { acc = fmaf(r[c], yc[c], acc); }
    const float wd = w * block_sum(acc, scratch);
    // Each thread rewrites only the channels it read, so sum may be y.
    float* sc = sum + t * ss;
    const float* rc = residual != nullptr ? residual + t * rs : nullptr;
    for (int c = threadIdx.x; c < channels; c += kThreads) {
        const float v = yc[c] - wd * r[c];
        yc[c] = v;
        sc[c] = rc != nullptr ? rc[c] + v : v;
    }
}

} // namespace

void low_rank_update_fp32(const Tensor& x, const float* a, const float* b, std::int32_t rank, float scale, Tensor& y,
                          cudaStream_t stream) {
    constexpr const char* op = "low_rank_update_fp32";
    const std::int32_t k = x.ne[0], cols = x.ne[1], n = y.ne[0];
    require(rank >= 1 && rank <= kLowRankMaxRank, op, "rank outside 1.." + std::to_string(kLowRankMaxRank));
    require(a != nullptr && b != nullptr, op, "null A or B");
    require(k > 0 && n > 0 && cols > 0, op, "empty operands");
    const std::int64_t xs = column_stride(x, DType::BF16, k, cols, op, "x");
    const std::int64_t ys = column_stride(y, DType::FP32, n, cols, op, "y");
    low_rank_kernel<<<cols, kThreads, 0, stream>>>(static_cast<const __nv_bfloat16*>(x.data), xs, a, b, rank, scale, k, n,
                                                   static_cast<float*>(y.data), ys);
    launched(op);
}

void direction_ablate_fp32(Tensor& y, const float* r, float w, const Tensor* residual, Tensor& sum, cudaStream_t stream) {
    constexpr const char* op = "direction_ablate_fp32";
    const std::int32_t c = y.ne[0], cols = y.ne[1];
    require(r != nullptr, op, "null direction");
    require(c > 0 && cols > 0, op, "empty operands");
    const std::int64_t ys = column_stride(y, DType::FP32, c, cols, op, "y");
    const std::int64_t ss = column_stride(sum, DType::FP32, c, cols, op, "sum");
    std::int64_t rs = 0;
    if (residual != nullptr) {
        rs = column_stride(*residual, DType::FP32, c, cols, op, "residual");
        require(sum.data != y.data && sum.data != residual->data, op, "sum overlaps y or the residual");
    } else if (sum.data == y.data) {
        require(ss == ys, op, "sum is y at another stride");
    }
    ablate_kernel<<<cols, kThreads, 0, stream>>>(static_cast<float*>(y.data), ys, r, w,
                                                 residual != nullptr ? static_cast<const float*>(residual->data) : nullptr, rs,
                                                 static_cast<float*>(sum.data), ss, c);
    launched(op);
}

} // namespace ninfer::ops
