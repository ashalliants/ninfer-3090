#pragma once

// Adapted from Infernix a3edb450 include/infernix/ops/rows.h (Apache-2.0).
// Modified for NInfer-3090: the contract follows op-development §3.

#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <span>

namespace ninfer::ops {

/**
 * Op: split_rows
 *
 * Math / indexing:
 *   Output i receives rows [sum_{j<i} r_j, sum_{j<=i} r_j) of in: outs[i][r, t] =
 *   in[sum_{j<i} r_j + r, t]. Exact copy.
 *
 * Logical shapes:
 *   in BF16 [R, T]; outs[i] BF16 [r_i, T] with sum r_i = R; one to eight outputs; T >= 1.
 *
 * Supported domain: contiguous BF16 tensors; any element alignment.
 * Effects: writes every output; in is read only and no output overlaps it or another output.
 * Workspace: none. Execution: stream-ordered; graph capturable.
 */
void split_rows(const Tensor& in, std::span<Tensor* const> outs, cudaStream_t stream);

/**
 * Op: gather_columns
 *
 * Math / indexing:
 *   out[:, i] = in[:, columns[i]]. Exact copy.
 *
 * Logical shapes:
 *   in BF16 [R, T]; columns I32 [N], each in [0, T); out BF16 [R, N].
 *
 * Supported domain: contiguous tensors.
 * Effects: writes out; in and columns are read only and must not overlap out.
 * Workspace: none. Execution: stream-ordered; graph capturable.
 */
void gather_columns(const Tensor& in, const Tensor& columns, Tensor& out, cudaStream_t stream);

} // namespace ninfer::ops
