// sm_70 opt-in: BF16 linears on the FP16 tensor cores (ops/linear/common/sm70_tensor_core.h). The
// BF16 weight converts to FP16 exactly under its row's power of two.
#include "ops/linear/bf16/bf16_tensor_core_sm70.h"

#include "core/device.h" // CUDA_CHECK
#include "ops/linear/common/sm70_tensor_core.h"

#include <cuda_bf16.h>

namespace ninfer::ops::detail {
namespace {

constexpr int kThreads = 256;

// One block per weight row: FP16(w 2^e), absmax(w) 2^e in [2^14, 2^15); inverse = 2^-e.
__global__ void __launch_bounds__(kThreads)
    bf16_rows_to_f16(const __nv_bfloat16* __restrict__ src, int k, __half* __restrict__ dst, float* __restrict__ inverse) {
    __shared__ float red[kThreads / 32];
    const __nv_bfloat16* row = src + static_cast<std::size_t>(blockIdx.x) * k;
    __half* out              = dst + static_cast<std::size_t>(blockIdx.x) * k;
    float m = 0.0F;
    for (int i = threadIdx.x; i < k; i += kThreads) { m = fmaxf(m, fabsf(__bfloat162float(row[i]))); }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) { m = fmaxf(m, __shfl_xor_sync(0xFFFFFFFFU, m, o)); }
    if ((threadIdx.x & 31) == 0) { red[threadIdx.x >> 5] = m; }
    __syncthreads();
    m = 0.0F;
#pragma unroll
    for (int i = 0; i < kThreads / 32; ++i) { m = fmaxf(m, red[i]); }
    int e = 0;
    if (m > 0.0F && isfinite(m)) {
        int x = 0;
        (void)frexpf(m, &x);
        e = 15 - x;
    }
    for (int i = threadIdx.x; i < k; i += kThreads) { out[i] = __float2half_rn(ldexpf(__bfloat162float(row[i]), e)); }
    if (threadIdx.x == 0) { inverse[blockIdx.x] = ldexpf(1.0F, -e); }
}

} // namespace

bool bf16_tensor_core_linear(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    const int tokens = x.ne[1];
    if (!sm70_tensor_core_admits(w.n, w.k, tokens, sm70_tensor_cores_enabled()) || out.dtype != DType::BF16 ||
        x.dtype != DType::BF16) {
        return false;
    }
    const Sm70HalfWeight half = sm70_half_weight(w.n, w.k);
    bf16_rows_to_f16<<<w.n, kThreads, 0, stream>>>(static_cast<const __nv_bfloat16*>(w.qdata), w.k, half.values,
                                                   half.inverse);
    CUDA_CHECK(cudaGetLastError());
    sm70_tensor_core_gemm(half, w.n, w.k, x.data, tokens, out.data, false, stream);
    return true;
}

} // namespace ninfer::ops::detail
