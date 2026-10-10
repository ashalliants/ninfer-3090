#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h> // cudaStream_t

namespace ninfer::ops {

/**
 * Prepares Gated DeltaNet decay and update gates:
 *
 *   g[h,t]    = -exp(A_log[h]) * softplus(a[h,t] + dt_bias[h])
 *   beta[h,t] = sigmoid(b[h,t]).
 *
 * `a` and `b` are contiguous BF16 [48,T], `A_log` and `dt_bias` are contiguous FP32 [48], and
 * `g` and `beta` are contiguous FP32 [48,T]. The oracle evaluates the formula naively in FP64;
 * transcendental implementation and intermediate precision are private kernel choices. Inputs and
 * the two outputs must be mutually non-overlapping. There is no workspace or persistent state side
 * effect.
 */
void gdn_gating(const Tensor& a, const Tensor& b, const Tensor& A_log, const Tensor& dt_bias,
                Tensor& g, Tensor& beta, cudaStream_t stream);

/**
 * The same gates from the decay rate itself, as GGUF exports store it (ssm_a = -exp(A_log)):
 *
 *   g[h,t]    = decay[h] * softplus(a[h,t] + dt_bias[h])
 *   beta[h,t] = sigmoid(b[h,t]).
 *
 * `decay` is contiguous FP32 [48]; every other operand, the oracle and the effects are those of
 * gdn_gating. The decay is applied as stored: no logarithm or exponential of it is formed.
 */
void gdn_gating_decay(const Tensor& a, const Tensor& b, const Tensor& decay, const Tensor& dt_bias,
                      Tensor& g, Tensor& beta, cudaStream_t stream);

} // namespace ninfer::ops
