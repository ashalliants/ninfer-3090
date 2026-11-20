// See sm70_tensor_core.h.
#include "ops/linear/common/sm70_tensor_core.h"

#include "core/device.h" // CUDA_CHECK

#include <cublas_v2.h>
#include <cuda_bf16.h>

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <stdexcept>
#include <string>

namespace ninfer::ops::detail {
namespace {

constexpr int kThreads     = 256;

void check_blas(cublasStatus_t status, const char* what) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(std::string("sm_70 tensor-core route: ") + what + " failed: " +
                                 std::to_string(static_cast<int>(status)));
    }
}

// One cuBLAS handle per device (a handle belongs to the device current when it was created),
// created on first use there. Each is published only once fully configured, and created under a
// mutex, so threads racing to a device's first call share one handle.
constexpr int kMaxDevices = 64;
std::atomic<cublasHandle_t> g_handles[kMaxDevices]{};
std::mutex g_handle_mutex;

int current_device() {
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    if (device < 0 || device >= kMaxDevices) {
        throw std::runtime_error("sm_70 tensor-core route: device " + std::to_string(device) +
                                 " is past the route's handle table");
    }
    return device;
}

// The current device's handle, or null before its first create.
cublasHandle_t existing_handle(int device) {
    return g_handles[device].load(std::memory_order_acquire);
}

cublasHandle_t handle() {
    const int device = current_device();
    cublasHandle_t h = existing_handle(device);
    if (h != nullptr) { return h; }
    std::lock_guard<std::mutex> lock(g_handle_mutex);
    h = g_handles[device].load(std::memory_order_relaxed);
    if (h == nullptr) {
        check_blas(cublasCreate(&h), "cublasCreate");
        const cublasStatus_t mode = cublasSetMathMode(h, CUBLAS_TENSOR_OP_MATH);
        if (mode != CUBLAS_STATUS_SUCCESS) {
            cublasDestroy(h);
            check_blas(mode, "cublasSetMathMode");
        }
        g_handles[device].store(h, std::memory_order_release);
    }
    return h;
}


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

namespace {
std::atomic<std::uint64_t> g_sm70_fallbacks{0};
} // namespace

bool sm70_tensor_core_can_launch(cudaStream_t stream) {
    // The handle of the device this call runs on: another device's handle cannot serve it.
    if (existing_handle(current_device()) != nullptr) { return true; }
    cudaStreamCaptureStatus status = cudaStreamCaptureStatusNone;
    CUDA_CHECK(cudaStreamIsCapturing(stream, &status));
    if (status != cudaStreamCaptureStatusNone) { return false; }
    (void)handle();
    return true;
}

bool sm70_staging(WorkspaceArena* arena, int rows, int k, int tokens, Sm70Staging& out) {
    const std::size_t need = sm70_tensor_core_workspace_bytes(rows, k, tokens);
    if (need == 0) { return false; }
    if (arena == nullptr || arena->capacity() - arena->used() < need) {
        g_sm70_fallbacks.fetch_add(1, std::memory_order_relaxed); // ops::sm70_tensor_core_fallbacks
        return false;
    }
    const std::size_t r = static_cast<std::size_t>(rows), kk = static_cast<std::size_t>(k);
    const std::size_t slice = static_cast<std::size_t>(std::min(tokens, kSm70TensorCoreSliceTokens));
    out.weight           = static_cast<__half*>(arena->alloc_bytes(r * kk * 2).data);
    out.weight_inverse   = static_cast<float*>(arena->alloc_bytes(r * 4).data);
    out.acts             = static_cast<__half*>(arena->alloc_bytes(slice * kk * 2).data);
    out.accum            = static_cast<float*>(arena->alloc_bytes(slice * r * 4).data);
    out.token_inverse    = static_cast<float*>(arena->alloc_bytes(slice * 4).data);
    out.cublas_workspace = arena->alloc_bytes(kSm70CublasWorkspaceBytes).data;
    return true;
}

void sm70_tensor_core_gemm(const Sm70Staging& st, int rows, int k, const void* x, int tokens, void* out,
                           bool residual, cudaStream_t stream) {
    const int slice = std::min(tokens, kSm70TensorCoreSliceTokens);
    cublasHandle_t h = handle();
    check_blas(cublasSetStream(h, stream), "cublasSetStream");
    // After the stream: cublasSetStream resets the workspace to the handle's default pool.
    check_blas(cublasSetWorkspace(h, st.cublas_workspace, kSm70CublasWorkspaceBytes), "cublasSetWorkspace");
    const float one = 1.0F, zero = 0.0F;
    const auto* xs = static_cast<const __nv_bfloat16*>(x);
    auto* ys       = static_cast<__nv_bfloat16*>(out);
    for (int t0 = 0; t0 < tokens; t0 += slice) {
        const int n = tokens - t0 < slice ? tokens - t0 : slice;
        tokens_to_f16<<<n, kThreads, 0, stream>>>(xs + static_cast<std::size_t>(t0) * k, k, st.acts, st.token_inverse);
        CUDA_CHECK(cudaGetLastError());
        check_blas(cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, rows, n, k, &one, st.weight, CUDA_R_16F, k, st.acts,
                                CUDA_R_16F, k, &zero, st.accum, CUDA_R_32F, rows, CUBLAS_COMPUTE_32F,
                                CUBLAS_GEMM_DEFAULT_TENSOR_OP),
                   "cublasGemmEx");
        const std::size_t count = static_cast<std::size_t>(rows) * n;
        const auto blocks       = static_cast<unsigned>((count + kThreads - 1) / kThreads);
        auto* dst               = ys + static_cast<std::size_t>(t0) * rows;
        if (residual) {
            finish<true><<<blocks, kThreads, 0, stream>>>(st.accum, rows, n, st.weight_inverse, st.token_inverse, dst);
        } else {
            finish<false><<<blocks, kThreads, 0, stream>>>(st.accum, rows, n, st.weight_inverse, st.token_inverse, dst);
        }
        CUDA_CHECK(cudaGetLastError());
    }
}

} // namespace ninfer::ops::detail

namespace ninfer::ops {

std::uint64_t sm70_tensor_core_fallbacks() noexcept {
    return detail::g_sm70_fallbacks.load(std::memory_order_relaxed);
}

} // namespace ninfer::ops
