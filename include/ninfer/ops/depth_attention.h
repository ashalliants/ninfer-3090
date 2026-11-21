#pragma once

// Written for NInfer-3090. Depth attention over a short list of residual states, and the gated message
// combine of a loop controller (Inkling-2-Mini's attention residuals and loop messages,
// docs/maintainer/inkling2-model.md 3b, 3c, 3i and 4). Columns follow fp32_rows.h: [D, T], ne[0]
// contiguous, any column stride.

#include "core/arena.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <span>

namespace ninfer::ops {

/// The most sources and queries depth_attention_mix takes.
inline constexpr int kDepthAttentionMaxSources = 8;
inline constexpr int kDepthAttentionMaxQueries = 4;

/**
 * Op: depth_attention_mix (softmax over normalized sources, values normalized too)
 *
 * Math / indexing:
 *   s_j[:, t] = sources[j][:, t] / sqrt(mean_d sources[j][d, t]^2 + eps)
 *   a_ij[t]   = softmax_j( sum_d queries[d, i] * s_j[d, t] / sqrt(D) )
 *   outputs[i][:, t] = sum_j a_ij[t] * s_j[:, t]
 *
 * Logical shapes:
 *   sources: S FP32 [D, T]; queries FP32 [D, Q] contiguous; outputs: Q FP32 [D, T].
 *
 * Supported domain:
 *   1 <= S <= kDepthAttentionMaxSources, 1 <= Q <= kDepthAttentionMaxQueries, D at most 8192, T at most 65535.
 *
 * Numeric:
 *   FP32 reductions, max-subtracted softmax; oracle FP64 (tests/ops/inkling2_reference.h mix).
 *
 * Effects: writes the outputs, which must not overlap a source or each other.
 * Workspace: depth_attention_mix_workspace_bytes(D, T, S) from `workspace`, released on return.
 * Execution: stream-ordered; graph capturable.
 */
[[nodiscard]] std::size_t depth_attention_mix_workspace_bytes(std::int32_t rows, std::int64_t columns, int sources);
void depth_attention_mix(std::span<const Tensor* const> sources, const Tensor& queries, float eps, DeviceArena& workspace,
                         std::span<Tensor* const> outputs, cudaStream_t stream);

/**
 * Op: forget_gain_combine (the output of a gated message controller)
 *
 * Math / indexing:
 *   out[d, t] = (1 - tanh(gf[D + d, t])) * c[d, t] + tanh(gf[d, t]) * m[d, t]
 *   (gf holds the gain pre-activations in rows 0..D-1 and the forget pre-activations in rows D..2D-1)
 *
 * Logical shapes: c, m, out FP32 [D, T]; gf FP32 [2D, T].
 * Numeric: FP32 tanh; oracle FP64 (inkling2_reference.h control, given the same gf).
 * Effects: writes out, which may equal c or m. Workspace: none. Execution: graph capturable.
 */
void forget_gain_combine(const Tensor& c, const Tensor& m, const Tensor& gf, Tensor& out, cudaStream_t stream);

} // namespace ninfer::ops
