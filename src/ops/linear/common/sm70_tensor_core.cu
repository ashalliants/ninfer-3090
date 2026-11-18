// See sm70_tensor_core.h.
#include "ops/linear/common/sm70_tensor_core.h"

#include "core/device.h" // CUDA_CHECK

#include <cublas_v2.h>
#include <cuda_bf16.h>

#include <cstdlib>
#include <stdexcept>
#include <string>

namespace ninfer::ops::detail {
namespace {

constexpr int kThreads     = 256;
constexpr int kSliceTokens = 512; // tokens per GEMM (bounds the FP16 activation and FP32 buffers: ~36 MB at the widest)

void check_blas(cublasStatus_t status, const char* what) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(std::string("sm_70 tensor-core route: ") + what + " failed: " +
                                 std::to_string(static_cast<int>(status)));
    }
}

cublasHandle_t handle() {
    static cublasHandle_t h = nullptr;
    if (h == nullptr) {
        check_blas(cublasCreate(&h), "cublasCreate");
        check_blas(cublasSetMathMode(h, CUBLAS_TENSOR_OP_MATH), "cublasSetMathMode");
    }
    return h;
}

// A device buffer that grows on demand to the largest call seen (~100 MB for Flash-Next in all:
// the [12800, 2560] FP16 weight and a 512-token slice). Not workspace-planned: the route is opt-in.
struct Buffer {
    void* p          = nullptr;
    std::size_t size = 0;
    void* reserve(std::size_t bytes) {
        if (bytes > size) {
            if (p != nullptr) { CUDA_CHECK(cudaFree(p)); }
            CUDA_CHECK(cudaMalloc(&p, bytes));
            size = bytes;
        }
        return p;
    }
};
Buffer g_weight, g_weight_inverse, g_acts, g_accum, g_token_inverse;

__device__ __forceinline__ float block_absmax(float v, float* red) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) { v = fmaxf(v, __shfl_xor_sync(0xFFFFFFFFU, v, o)); }
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    if (lane == 0) { red[warp] = v; }
    __syncthreads();
    if (warp == 0) {
        v = lane < kThreads / 32 ? red[lane] : 0.0F;
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) { v = fmaxf(v, __shfl_xor_sync(0xFFFFFFFFU, v, o)); }
        if (lane == 0) { red[0] = v; }
    }
    __syncthreads();
    return red[0];
}

// One block per token: X16 = FP16(x 2^e), absmax(x) 2^e in [2^14, 2^15); inverse = 2^-e.
__global__ void __launch_bounds__(kThreads)
    tokens_to_f16(const __nv_bfloat16* __restrict__ src, int k, __half* __restrict__ dst, float* __restrict__ inverse) {
    __shared__ float red[kThreads / 32];
    const __nv_bfloat16* row = src + static_cast<std::size_t>(blockIdx.x) * k;
    __half* out              = dst + static_cast<std::size_t>(blockIdx.x) * k;
    float m = 0.0F;
    for (int i = threadIdx.x; i < k; i += kThreads) { m = fmaxf(m, fabsf(__bfloat162float(row[i]))); }
    m = block_absmax(m, red);
    int e = 0;
    if (m > 0.0F && isfinite(m)) {
        int x = 0;
        (void)frexpf(m, &x);
        e = 15 - x;
    }
    for (int i = threadIdx.x; i < k; i += kThreads) { out[i] = __float2half_rn(ldexpf(__bfloat162float(row[i]), e)); }
    if (threadIdx.x == 0) { inverse[blockIdx.x] = ldexpf(1.0F, -e); }
}

template <bool kResidual>
__global__ void __launch_bounds__(kThreads)
    finish(const float* __restrict__ c, int rows, int tokens, const float* __restrict__ row_inverse,
           const float* __restrict__ token_inverse, __nv_bfloat16* __restrict__ out) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * kThreads + threadIdx.x;
    if (i >= static_cast<std::size_t>(rows) * tokens) { return; }
    const int t = static_cast<int>(i / rows), m = static_cast<int>(i % rows);
    float v = c[i] * row_inverse[m] * token_inverse[t];
    if constexpr (kResidual) { v += __bfloat162float(out[i]); }
    out[i] = __float2bfloat16_rn(v);
}

} // namespace

bool sm70_tensor_cores_enabled() noexcept {
#if defined(NINFER_SM70)
    static const bool on = [] {
        const char* v = std::getenv("NINFER_SM70_TENSOR_CORES");
        return v != nullptr && v[0] == '1';
    }();
    return on;
#else
    return false;
#endif
}

Sm70HalfWeight sm70_half_weight(int rows, int k) {
    return {static_cast<__half*>(g_weight.reserve(static_cast<std::size_t>(rows) * k * sizeof(__half))),
            static_cast<float*>(g_weight_inverse.reserve(static_cast<std::size_t>(rows) * sizeof(float)))};
}

void sm70_tensor_core_gemm(const Sm70HalfWeight& w, int rows, int k, const void* x, int tokens, void* out,
                           bool residual, cudaStream_t stream) {
    const int slice = tokens < kSliceTokens ? tokens : kSliceTokens;
    auto* xh  = static_cast<__half*>(g_acts.reserve(static_cast<std::size_t>(k) * slice * sizeof(__half)));
    auto* c   = static_cast<float*>(g_accum.reserve(static_cast<std::size_t>(rows) * slice * sizeof(float)));
    auto* inv = static_cast<float*>(g_token_inverse.reserve(static_cast<std::size_t>(slice) * sizeof(float)));
    cublasHandle_t h = handle();
    check_blas(cublasSetStream(h, stream), "cublasSetStream");
    const float one = 1.0F, zero = 0.0F;
    const auto* xs = static_cast<const __nv_bfloat16*>(x);
    auto* ys       = static_cast<__nv_bfloat16*>(out);
    for (int t0 = 0; t0 < tokens; t0 += slice) {
        const int n = tokens - t0 < slice ? tokens - t0 : slice;
        tokens_to_f16<<<n, kThreads, 0, stream>>>(xs + static_cast<std::size_t>(t0) * k, k, xh, inv);
        CUDA_CHECK(cudaGetLastError());
        check_blas(cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, rows, n, k, &one, w.values, CUDA_R_16F, k, xh, CUDA_R_16F,
                                k, &zero, c, CUDA_R_32F, rows, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP),
                   "cublasGemmEx");
        const std::size_t count = static_cast<std::size_t>(rows) * n;
        const auto blocks       = static_cast<unsigned>((count + kThreads - 1) / kThreads);
        auto* dst               = ys + static_cast<std::size_t>(t0) * rows;
        if (residual) {
            finish<true><<<blocks, kThreads, 0, stream>>>(c, rows, n, w.inverse, inv, dst);
        } else {
            finish<false><<<blocks, kThreads, 0, stream>>>(c, rows, n, w.inverse, inv, dst);
        }
        CUDA_CHECK(cudaGetLastError());
    }
}

} // namespace ninfer::ops::detail
