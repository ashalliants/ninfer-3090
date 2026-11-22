#pragma once

// Written for NInfer-3090. The two opt-in Inkling-2-Mini adapters' Ops (docs/maintainer/inkling2-model.md,
// "Adapters"): a low-rank update of a projection's result (a PEFT LoRA on attn.wo_ud) and the removal of one
// direction from a residual write (the per-visit ablation graft). Neither runs unless a request selects an adapter.

#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops {

inline constexpr std::int32_t kLowRankMaxRank = 16;

/**
 * Op: low_rank_update_fp32 (y += scale * B (A x), a LoRA applied unmerged)
 *
 * Math / indexing, per column t:
 *   u[j]    = scale * sum_k a[j * K + k] * x[k, t]      (j < rank)
 *   y[n, t] = y[n, t] + sum_j b[n * rank + j] * u[j]    (n < N)
 *
 * Logical shapes:
 *   x BF16 [K, T] (the projection's input); a FP32 [rank, K] row-major and b FP32 [N, rank] row-major, device
 *   pointers; y FP32 [N, T] (the projection's result), updated in place. Columns at any stride.
 *
 * Supported domain: 1 <= rank <= kLowRankMaxRank.
 *
 * Numeric: FP32 products and sums (the order of u's sum is implementation-defined); oracle FP64.
 *
 * Effects: updates y. Workspace: none. Execution: stream-ordered; graph capturable.
 */
void low_rank_update_fp32(const Tensor& x, const float* a, const float* b, std::int32_t rank, float scale, Tensor& y,
                          cudaStream_t stream);

/**
 * Op: direction_ablate_fp32 (y <- y - w r (r . y), then the residual add)
 *
 * Math / indexing, per column t:
 *   d          = sum_c r[c] * y[c, t]
 *   y'[c, t]   = y[c, t] - (w * d) * r[c]
 *   sum[c, t]  = residual[c, t] + y'[c, t]      (residual given)
 *   sum[c, t]  = y'[c, t]                       (residual null)
 *
 * Logical shapes:
 *   y FP32 [C, T]; r FP32 [C] (a device pointer, a unit vector: the Op does not normalise it); residual FP32 [C, T]
 *   or null; sum FP32 [C, T]. Columns at any stride.
 *
 * Numeric: FP32 (the order of d's sum is implementation-defined); oracle FP64. The residual add is the one FP32
 * addition, as add_fp32's.
 *
 * Effects: writes y' into y and the sum into sum. sum may be y itself (residual null); otherwise it must not overlap
 * y or residual. Workspace: none. Execution: stream-ordered; graph capturable.
 */
void direction_ablate_fp32(Tensor& y, const float* r, float w, const Tensor* residual, Tensor& sum, cudaStream_t stream);

} // namespace ninfer::ops
