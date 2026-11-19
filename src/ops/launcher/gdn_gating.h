#pragma once

// ninfer::ops::detail - private launch prototype for gdn_gating.

#include "core/tensor.h"
#include "ops/launcher/gdn_gate_job.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace ninfer::ops::detail {

// decay: `rate` holds the decay itself rather than A_log (gdn_gating_decay).
void gdn_gating_launch(const Tensor& a, const Tensor& b, const Tensor& rate, const Tensor& dt_bias,
                       bool decay, Tensor& g, Tensor& beta, cudaStream_t stream);
// a and b as raw planes: element (h, t) at t * ab_stride + h.
void gdn_gating_launch(const __nv_bfloat16* a, const __nv_bfloat16* b, int ab_stride, const Tensor& rate,
                       const Tensor& dt_bias, bool decay, Tensor& g, Tensor& beta, cudaStream_t stream);
// The job's gates in their own launch (gdn_gating_kernel<true>).
void gdn_gate_job_launch(const GdnGateJob& job, cudaStream_t stream);
// gdn_gating_decay_packed's operands checked and bound as a job.
GdnGateJob packed_gate_job(const Tensor& ab, const Tensor& decay, const Tensor& dt_bias, Tensor& g, Tensor& beta);

} // namespace ninfer::ops::detail
