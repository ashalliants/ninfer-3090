#pragma once

// ninfer::ops - gdn_gating kernel: elementwise GDN gate prep over [48,T].
// Transcendentals use fp32 CUDA math functions, not polynomial approximations.

#include "ops/common/math.cuh"
#include "ops/launcher/gdn_gate_job.h"

#include <cuda_bf16.h>

#include <cmath>
#include <cstdint>

namespace ninfer::ops {

// Decay = false: rate[h] holds A_log and the decay is -exp(A_log). Decay = true: rate[h] is the
// decay itself (GGUF ssm_a = -exp(A_log)).
// Element (h, t) of a and b is at t * ab_stride + h (48 for separate [48, T] tensors, 96 for the
// halves of one [96, T] projection); g and beta are [48, T].
// One element i = t * 48 + h of the gates.
template <bool Decay>
__device__ __forceinline__ void gdn_gating_element(const __nv_bfloat16* a, const __nv_bfloat16* b,
                                                   const float* rate, const float* dt_bias, float* g,
                                                   float* beta, std::int64_t i, int ab_stride) {
    const int h    = static_cast<int>(i % 48);
    const std::int64_t at = (i / 48) * ab_stride + h;
    const float av = __bfloat162float(a[at]);
    const float bv = __bfloat162float(b[at]);
    const float sp = softplus(av + dt_bias[h]);
    if constexpr (Decay) {
        g[i] = rate[h] * sp;
    } else {
        g[i] = -expf(rate[h]) * sp;
    }
    beta[i] = sigmoid(bv);
}

// A GdnGateJob's elements start, start + stride, ...
__device__ __forceinline__ void run_gdn_gate_job(const GdnGateJob& job, std::int64_t start, std::int64_t stride) {
    for (std::int64_t i = start; i < job.n; i += stride) {
        gdn_gating_element<true>(job.a, job.b, job.decay, job.dt_bias, job.g, job.beta, i, job.ab_stride);
    }
}

template <bool Decay>
__global__ void gdn_gating_kernel(const __nv_bfloat16* a, const __nv_bfloat16* b,
                                  const float* rate, const float* dt_bias, float* g, float* beta,
                                  std::int64_t n, int ab_stride) {
    const std::int64_t start  = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x;
    const std::int64_t stride = static_cast<std::int64_t>(gridDim.x) * blockDim.x;
    for (std::int64_t i = start; i < n; i += stride) {
        gdn_gating_element<Decay>(a, b, rate, dt_bias, g, beta, i, ab_stride);
    }
}

} // namespace ninfer::ops
