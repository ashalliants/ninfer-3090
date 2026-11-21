#pragma once

// Written for NInfer-3090. Depth attention over a short list of residual states, and the gated message
// controller of the loop (Inkling-2-Mini's attention residuals and loop messages,
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
 *   1 <= S <= kDepthAttentionMaxSources, 1 <= Q <= kDepthAttentionMaxQueries, D at most 8192.
 *
 * Numeric:
 *   FP32 reductions, max-subtracted softmax; oracle FP64 (tests/ops/inkling2_reference.h mix).
 *
 * Effects: writes the outputs, which must not overlap a source or each other.
 * Workspace: depth_attention_mix_workspace_bytes(D, T, S) from `workspace`, released on return; the query
 *   throws std::invalid_argument outside the supported domain.
 * Execution: stream-ordered; graph capturable.
 */
[[nodiscard]] std::size_t depth_attention_mix_workspace_bytes(std::int32_t rows, std::int64_t columns, int sources);
void depth_attention_mix(std::span<const Tensor* const> sources, const Tensor& queries, float eps, DeviceArena& workspace,
                         std::span<Tensor* const> outputs, cudaStream_t stream);

/// One gated message controller of a loop_controller_fp32 batch.
struct LoopControllerJob {
    const Tensor* c      = nullptr; // FP32 [H, T]
    const Tensor* m      = nullptr; // FP32 [H, T]
    const Tensor* down   = nullptr; // BF16 [2H, R]: row r holds the 2H inputs' weights
    const Tensor* gain   = nullptr; // BF16 [R, H]: row d holds the R hidden values' weights
    const Tensor* forget = nullptr; // BF16 [R, H]
    Tensor* out          = nullptr; // FP32 [H, T]
};
inline constexpr int kLoopControllerMaxJobs = 3;
inline constexpr int kLoopControllerMaxRank = 64;

/**
 * Op: loop_controller_fp32 (Inkling-2's gated message controller, 1 to kLoopControllerMaxJobs at once)
 *
 * Math / indexing (each job, each column t):
 *   z = down [u(c); u(m)], u(x) = x / sqrt(mean(x^2) + eps)   (R values)
 *   out[d, t] = (1 - tanh(forget[d] . z)) * c[d, t] + tanh(gain[d] . z) * m[d, t]
 *
 * Logical shapes: per job as LoopControllerJob; every job has the same H, R and T.
 *
 * Supported domain:
 *   H a multiple of 256; R a multiple of 8, at most kLoopControllerMaxRank; T >= 1; c, m and out at any
 *   column stride (ne[0] contiguous); the weights contiguous and 16-byte aligned.
 *
 * Numeric:
 *   FP32 throughout, nothing rounded to BF16: since u is a per-column scale, down [u(c); u(m)] is formed as
 *   (down_c . c) / rms(c) + (down_m . m) / rms(m), each dot product summed in 256-input slices (8 inputs a
 *   lane, a fixed butterfly) and the slices added in order; the sums of squares likewise. Oracle FP64
 *   (inkling2_reference.h control) over the BF16 weights.
 *
 * Effects: writes out, which may equal its job's c or m but no other job's operands.
 * Workspace: loop_controller_workspace_bytes(H, R, T, jobs). Execution: three launches; graph capturable.
 */
[[nodiscard]] std::size_t loop_controller_workspace_bytes(int hidden, int rank, int cols, int jobs);
void loop_controller_fp32(std::span<const LoopControllerJob> jobs, float eps, DeviceArena& workspace, cudaStream_t stream);

} // namespace ninfer::ops
