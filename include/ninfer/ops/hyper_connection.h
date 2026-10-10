#pragma once

// Adapted from Infernix a3edb450 include/infernix/ops/hyper_connection.h (Apache-2.0).
// Modified for NInfer-3090: the norm weight is the stored FP32 multiplier instead of a BF16
// unit-offset weight, the Q8 weight form is dropped, and the contract follows op-development §3.

#include "core/arena.h"
#include "core/tensor.h"
#include "core/weight.h"
#include "ninfer/ops/linear.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops {

/**
 * Op: hyper_connection_mix (the mixer that reads a block's input from hyper-connection residual
 * streams)
 *
 * Math / indexing:
 *   The residual R holds S streams of H features per column, stream-major: R[s*H + d, t] is
 *   feature d of stream s. For every column t independently:
 *     Rn[s*H + d] = R[s*H + d] * (mean_d' R[s*H + d']^2 + eps)^(-1/2) * w[s*H + d]
 *     z           = W_down Rn                       (rank rows, then S injection rows if present)
 *     m[k]        = SiLU(z[k] / S)                  for k < rank
 *     inject[s]   = 2 * sigmoid(z[rank + s] / S)    for s < S, only when inject is requested
 *     u           = W_up m
 *     x[d]        = (1/S) * sum_s sigmoid(u[s*H + d]) * Rn[s*H + d]
 *   w is the norm's full multiplier (a checkpoint's unit-offset 1 + gamma is stored as such).
 *
 * Logical shapes:
 *   residual BF16 [S*H, T]; norm_weight FP32 [S*H]; down [rank (+ S), S*H]; up [S*H, rank];
 *   x BF16 [H, T]; inject FP32 [S, T]. down has rank + S rows exactly when inject is requested.
 *   T >= 1.
 *
 * Supported domain:
 *   down and up are contiguous BF16 weights registered with Linear for their [N, K]; every T. The
 *   geometry S = 4, H = 2560, rank a multiple of 8 up to 512 is the one served without Linear at
 *   T <= 16. eps > 0.
 *
 * Numeric:
 *   Arithmetic is FP32 from the represented inputs. Two profiles: at T <= 16 with the S = 4,
 *   H = 2560 geometry every intermediate (Rn, z, m, u) stays FP32 and only x is rounded to BF16,
 *   so a column's x and inject do not depend on T within 1..16; otherwise Rn, z, m and u are
 *   rounded to BF16 (the composed profile) and Linear's BF16 criterion applies to z and u. The
 *   oracle is the formula in FP64.
 *
 * Effects:
 *   Writes x and, when requested, inject. residual and the weights are read only. No output may
 *   overlap an input.
 *
 * Workspace:
 *   hyper_connection_mix_workspace_capacity_bytes() for every T in [1, max_tokens]. It must not
 *   overlap any operand.
 *
 * Execution:
 *   Stream-ordered; graph capturable.
 */
void hyper_connection_mix(const Tensor& residual, const Tensor& norm_weight, const Weight& down,
                          const Weight& up, LinearPolicy policy, std::int32_t streams,
                          std::int32_t rank, float eps, Tensor& x, Tensor* inject,
                          WorkspaceArena& workspace, cudaStream_t stream);

/// The workspace hyper_connection_mix needs for every T in [1, max_tokens].
[[nodiscard]] std::size_t hyper_connection_mix_workspace_capacity_bytes(
    const Weight& down, const Weight& up, LinearPolicy policy, std::int32_t streams,
    std::int32_t rank, std::int32_t max_tokens);

/**
 * Op: hyper_connection_inject
 *
 * Math / indexing:
 *   residual[s*H + d, t] = bf16(residual[s*H + d, t] + inject[s, t] * y[d, t])
 *
 * Logical shapes:
 *   y BF16 [H, T]; inject FP32 [S, T]; residual BF16 [S*H, T]; T >= 1.
 *
 * Numeric:
 *   One FP32 multiply-add per element, rounded once to BF16.
 *
 * Effects:
 *   Updates residual in place; y and inject are read only and must not overlap it.
 *
 * Workspace: none. Execution: stream-ordered; graph capturable.
 */
void hyper_connection_inject(const Tensor& y, const Tensor& inject, Tensor& residual,
                             cudaStream_t stream);

/**
 * Op: hyper_connection_expand
 *
 * Math / indexing:
 *   residual[s*H + d, t] = x[d, t] for every stream s (the first block's residual from the
 *   embedding). Exact copy.
 *
 * Logical shapes:
 *   x BF16 [H, T]; residual BF16 [S*H, T]; T >= 1.
 *
 * Effects:
 *   Writes residual; x must not overlap it.
 *
 * Workspace: none. Execution: stream-ordered; graph capturable.
 */
void hyper_connection_expand(const Tensor& x, std::int32_t streams, Tensor& residual,
                             cudaStream_t stream);

} // namespace ninfer::ops
