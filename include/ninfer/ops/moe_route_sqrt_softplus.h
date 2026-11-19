#pragma once

// DeepSeek-V4's expert router: sqrt-softplus scores, an aux-loss-free bias that picks the experts
// but does not weight them, and weights from the unbiased scores. The model repo's `Gate` with
// `score_func == "sqrtsoftplus"` (inference/model.py); the NumPy oracle is
// tools/dsv4/reference.py `gate_forward`. moe_route (offloaded_sparse_moe.h) is the softmax router
// of Qwen's models and does not serve it.

#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops {

/// The most experts and the most selected experts the router serves.
inline constexpr std::int32_t kRouteSqrtSoftplusMaxExperts = 512;
inline constexpr std::int32_t kRouteSqrtSoftplusMaxTopK    = 16;

/**
 * Op: moe_route_sqrt_softplus
 *
 * Math / indexing:
 *   For every column t independently, with E experts and logits z[e] (the router projection already
 *   divided by the gate temperature by the caller):
 *     s[e]   = sqrt(softplus(z[e])),   softplus(x) = x for x > 20, else log(1 + exp(x))
 *     key[e] = s[e] + bias[e]
 *     ids[0..k)  = the k experts with the largest key, in descending key order, ties to the lower
 *                  expert id
 *     w[i]       = s[ids[i]]                                  (the unbiased score)
 *     w[i]       = w[i] / (sum_i' w[i'] + 1e-20)              when norm_topk_prob and k > 1
 *     weights[i] = w[i] * route_scale
 *
 * Logical shapes:
 *   logits FP32 [E, T]; bias FP32 [E]; ids I32 [k, T]; weights FP32 [k, T] (slot i of column t at
 *   i + k * t). 1 <= k <= min(E, kRouteSqrtSoftplusMaxTopK); 1 <= E <=
 *   kRouteSqrtSoftplusMaxExperts; T >= 1.
 *
 * Supported domain:
 *   Contiguous tensors; finite logits and bias; route_scale finite.
 *
 * Numeric:
 *   FP32 arithmetic. The selection is exact on the FP32 keys the formula defines; a pair of experts
 *   whose keys differ by less than FP32 rounding of the score evaluation may order either way
 *   (expf, log1pf and sqrtf are not correctly rounded). The weights match the FP64 oracle given the
 *   selected ids to a few FP32 ulps.
 *
 * Effects:
 *   Writes ids and weights. Inputs are read only; no output may overlap an input or the other output.
 *
 * Workspace: none. Execution: stream-ordered; graph capturable.
 */
void moe_route_sqrt_softplus(const Tensor& logits, const Tensor& bias, std::int32_t top_k,
                             float route_scale, bool norm_topk_prob, Tensor& ids, Tensor& weights,
                             cudaStream_t stream);

} // namespace ninfer::ops
