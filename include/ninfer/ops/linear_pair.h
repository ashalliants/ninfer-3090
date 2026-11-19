#pragma once

#include "core/weight.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::ops {

// Validates the native weight pair and inclusive T interval. Existing routes use no scratch.
[[nodiscard]] std::size_t linear_pair_workspace_capacity_bytes(const Weight& first_weight,
                                                               const Weight& second_weight,
                                                               std::int32_t min_tokens,
                                                               std::int32_t max_tokens);

/**
 * Computes two projections of one input:
 *
 *   first_out[:,t]  = linear(x[:,t], first_weight)
 *   second_out[:,t] = linear(x[:,t], second_weight).
 *
 * `x` is contiguous BF16 [K,T] and the outputs are distinct contiguous BF16 [N1,T] and [N2,T] for
 * weights of logical shapes [N1,K] and [N2,K] (the same N except where listed). Registered problems:
 *   - RowSplit Q8_G32_FP16 with FP16 scales: the paired [1024,5120] physical projection and the
 *     exact adjacent [1024,2048] K/V row views (parent rows [4096,5120) and [5120,6144)), for
 *     every positive T;
 *   - ggml_blocks_v1 weights of [640,2560] or [512,2560] (Qwen3.8-Flash-Next's shared-expert
 *     gate/up and attention key/value), or a [10240,2560] first and a [6144,2560] second (its
 *     GDN qkv and z), each in GGML_IQ4_XS, GGML_IQ3_S, GGML_Q6_K, GGML_Q5_K or GGML_Q4_K, or in
 *     GGML_Q8_0 at the shapes where linear.h registers it ([640,2560], and the [10240,2560] and
 *     [6144,2560] GDN pair; not [512,2560]) (the two may differ), for 1 <= T <= 8. They take linear()'s Q8_1 (A8) profile, the only one GGML
 *     formats register, and every output equals linear()'s bit for bit.
 * Numeric semantics are those of linear(). Inputs, outputs, and weight planes must be mutually
 * non-overlapping. The Op requires no caller-owned transient storage and has no persistent state
 * side effect.
 */
void linear_pair(const Tensor& x, const Weight& first_weight, const Weight& second_weight,
                 Tensor& first_out, Tensor& second_out, cudaStream_t stream);

} // namespace ninfer::ops
