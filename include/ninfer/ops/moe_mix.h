#pragma once

// The two elementwise steps of DeepSeek-V4's MoE layer that the existing Ops do not serve: the
// clamped SwiGLU of the shared expert (silu_mul has no limit) and the mix of the routed experts'
// outputs with the shared expert's (moe_combine gates the shared output by a sigmoid and is
// specific to Qwen's shapes). The model repo's `Expert.forward` and `MoE.forward`
// (inference/model.py); the NumPy oracle is tools/dsv4/moe.py.

#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::ops {

/**
 * Op: clamped_swiglu
 *
 * Math / indexing:
 *   With limit L > 0:  out[i] = silu(min(gate[i], L)) * clamp(up[i], -L, L),
 *   with L == 0 no clamp:  out[i] = silu(gate[i]) * up[i];     silu(x) = x / (1 + exp(-x)).
 *
 * Logical shapes:
 *   gate, up and out BF16 of one shape (any rank up to 4), contiguous.
 *
 * Supported domain:
 *   Finite limit >= 0; finite gate and up.
 *
 * Numeric:
 *   FP32 arithmetic on the BF16 inputs, one BF16 rounding of the result: against the FP64 oracle
 *   on the represented inputs, half a BF16 ulp plus a few FP32 ulps (expf).
 *
 * Effects:
 *   Writes out. gate and up are read only (they may overlap each other); out must not overlap
 *   either.
 *
 * Workspace: none. Execution: stream-ordered; graph capturable.
 */
void clamped_swiglu(const Tensor& gate, const Tensor& up, float limit, Tensor& out, cudaStream_t stream);

/**
 * Op: weighted_slot_sum
 *
 * Math / indexing:
 *   For every column t and row h, with k slots:
 *     y[h, t] = bf16( sum_{s < k} weights[s, t] * outputs[h, t * k + s] + extra[h, t] )
 *   summed in slot order as an FP32 fused-multiply-add chain, then `extra` added.
 *
 * Logical shapes:
 *   outputs BF16 [H, k * T] (column t * k + s is routed expert slot s of token t, as
 *   moe_experts_narrow writes it); weights FP32 [k, T]; extra BF16 [H, T] (the shared expert's
 *   output); y BF16 [H, T]. 1 <= k <= 64; T >= 1; 1 <= H <= 65,535 x 256 (one grid-y tile per 256 rows).
 *
 * Supported domain:
 *   Contiguous tensors; finite values.
 *
 * Numeric:
 *   FP32 chain, one BF16 rounding: against the FP64 oracle on the represented inputs, half a BF16
 *   ulp of the result plus k FP32 ulps of the largest term.
 *
 * Effects:
 *   Writes y. Inputs are read only; y must not overlap an input.
 *
 * Workspace: none. Execution: stream-ordered; graph capturable.
 */
void weighted_slot_sum(const Tensor& outputs, const Tensor& weights, const Tensor& extra, Tensor& y,
                       cudaStream_t stream);

} // namespace ninfer::ops
