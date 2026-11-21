// DeepSeek-V4's Engram gate and residual write; the contract is include/ninfer/ops/engram.h.
// Written for NInfer-3090 against the model repo's `Engram` (inference/model.py) and
// tools/dsv4/engram_ref.py `engram_forward`.

#include "ninfer/ops/engram.h"

#include <cuda_bf16.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace ninfer::ops {

namespace {

constexpr int kThreads = 512;
constexpr int kWarps   = kThreads / 32;
constexpr int kMaxH    = 16384;
constexpr float kClamp = 1e-6F;

void require(bool condition, const char* message) {
    if (!condition) { throw std::invalid_argument(std::string("engram_gate_inject: ") + message); }
}

__device__ float load_w(const __nv_bfloat16* w, int i) { return __bfloat162float(w[i]); }

__device__ float load_w(const float* w, int i) { return w[i]; }

__device__ float block_sum(float v, float* shared) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_xor_sync(0xFFFFFFFFU, v, offset);
    }
    __syncthreads(); // the previous use of `shared`
    if ((threadIdx.x & 31) == 0) { shared[threadIdx.x >> 5] = v; }
    __syncthreads();
    float total = 0.0F;
    for (int i = 0; i < kWarps; ++i) { total += shared[i]; }
    return total;
}

// One block per (stream, column). Each thread owns adjacent pairs of features, strided by the
// block. The summation order depends only on H, so a column's result does not depend on T.
template <class W>
__global__ void __launch_bounds__(kThreads)
    engram_gate_inject_kernel(const __nv_bfloat16* __restrict__ kv, const W* __restrict__ w,
                              int streams, int h, float eps, __nv_bfloat16* __restrict__ residual) {
    __shared__ float shared[kWarps];
    // One flat grid dimension (x allows 2^31 - 1 blocks; y would stop at 65,535 columns).
    const int s                  = static_cast<int>(blockIdx.x % streams);
    const long long t            = blockIdx.x / streams;
    const std::size_t kv_stride  = static_cast<std::size_t>(streams + 1) * h;
    const std::size_t res_stride = static_cast<std::size_t>(streams) * h;
    const __nv_bfloat16* key     = kv + t * kv_stride + static_cast<std::size_t>(s) * h;
    const __nv_bfloat16* value   = kv + t * kv_stride + static_cast<std::size_t>(streams) * h;
    __nv_bfloat16* x             = residual + t * res_stride + static_cast<std::size_t>(s) * h;
    const W* ws                  = w + static_cast<std::size_t>(s) * h;

    float xx = 0.0F, kk = 0.0F, xwk = 0.0F;
    for (int i = 2 * threadIdx.x; i < h; i += 2 * kThreads) {
        const float2 xv = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(x + i));
        const float2 kx = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(key + i));
        xx += xv.x * xv.x + xv.y * xv.y;
        kk += kx.x * kx.x + kx.y * kx.y;
        xwk += xv.x * load_w(ws, i) * kx.x + xv.y * load_w(ws, i + 1) * kx.y;
    }
    const float sum_xx  = block_sum(xx, shared);
    const float sum_kk  = block_sum(kk, shared);
    const float sum_xwk = block_sum(xwk, shared);
    const float inv_h   = 1.0F / static_cast<float>(h);
    const float rx      = 1.0F / sqrtf(sum_xx * inv_h + eps);
    const float rk      = 1.0F / sqrtf(sum_kk * inv_h + eps);
    const float dot     = sum_xwk * rx * rk / sqrtf(static_cast<float>(h));
    const float g =
        copysignf(sqrtf(fmaxf(fabsf(dot), kClamp)), dot == 0.0F ? 1.0F : dot); // sign(+-0) = +1
    const float gate = 1.0F / (1.0F + expf(-g));
    for (int i = 2 * threadIdx.x; i < h; i += 2 * kThreads) {
        const float2 xv = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(x + i));
        const float2 vv = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(value + i));
        *reinterpret_cast<__nv_bfloat162*>(x + i) =
            __floats2bfloat162_rn(fmaf(gate, vv.x, xv.x), fmaf(gate, vv.y, xv.y));
    }
}

bool overlaps(const Tensor& a, const Tensor& b) {
    const auto a0 = reinterpret_cast<std::uintptr_t>(a.data),
               b0 = reinterpret_cast<std::uintptr_t>(b.data);
    return a0 < b0 + b.bytes() && b0 < a0 + a.bytes();
}

bool plain(const Tensor& t, DType dtype) {
    return t.data != nullptr && t.dtype == dtype && t.is_contiguous() && t.ne[2] == 1 &&
           t.ne[3] == 1 && reinterpret_cast<std::uintptr_t>(t.data) % 16 == 0;
}

} // namespace

void engram_gate_inject(const Tensor& kv, const Tensor& w, std::int32_t streams, float eps,
                        Tensor& residual, cudaStream_t stream) {
    require(streams >= 1, "streams must be at least 1");
    require(eps > 0.0F && std::isfinite(eps), "eps must be positive");
    require(plain(kv, DType::BF16) && plain(residual, DType::BF16),
            "contiguous 16-byte aligned BF16 kv and residual are required");
    require(w.data != nullptr && (w.dtype == DType::BF16 || w.dtype == DType::FP32) &&
                w.is_contiguous() && reinterpret_cast<std::uintptr_t>(w.data) % 4 == 0,
            "the gate weight must be contiguous, 4-byte aligned BF16 or FP32");
    require(residual.ne[0] % streams == 0, "the residual's rows are not a multiple of the streams");
    const std::int64_t h = residual.ne[0] / streams;
    require(h >= 8 && h % 8 == 0 && h <= kMaxH,
            "the stream width must be a multiple of 8 up to 16384");
    const std::int64_t tokens = residual.ne[1];
    require(tokens >= 1 && tokens * streams <= 2147483647LL,
            "streams * tokens must be at most 2^31 - 1");
    require(kv.ne[0] == static_cast<std::int64_t>(streams + 1) * h && kv.ne[1] == tokens,
            "kv must be [(streams + 1) * H, T]");
    require(static_cast<std::int64_t>(w.ne[0]) * w.ne[1] * w.ne[2] * w.ne[3] ==
                static_cast<std::int64_t>(streams) * h,
            "the gate weight must hold streams * H elements");
    require(!overlaps(residual, kv) && !overlaps(residual, w), "the residual overlaps an input");
    const unsigned grid = static_cast<unsigned>(tokens * streams);
    const auto* kv_data = static_cast<const __nv_bfloat16*>(kv.data);
    auto* res           = static_cast<__nv_bfloat16*>(residual.data);
    if (w.dtype == DType::BF16) {
        engram_gate_inject_kernel<<<grid, kThreads, 0, stream>>>(
            kv_data, static_cast<const __nv_bfloat16*>(w.data), streams, static_cast<int>(h), eps,
            res);
    } else {
        engram_gate_inject_kernel<<<grid, kThreads, 0, stream>>>(
            kv_data, static_cast<const float*>(w.data), streams, static_cast<int>(h), eps, res);
    }
    const cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        throw std::runtime_error(std::string("engram_gate_inject: ") + cudaGetErrorString(error));
    }
}

} // namespace ninfer::ops
