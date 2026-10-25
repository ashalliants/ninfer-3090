#pragma once

#include "core/arena.h"
#include "core/tensor.h"

#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h> // cudaStream_t

namespace ninfer::ops {

// Capacity for every call with at most `valid_rows` valid rows and a column count in the inclusive
// interval [min_columns, max_columns] (1 <= min_columns <= max_columns, valid_rows >= 1).
[[nodiscard]] std::size_t argmax_probability_workspace_capacity_bytes(std::int32_t valid_rows,
                                                                      std::int32_t min_columns,
                                                                      std::int32_t max_columns);

/**
 * Op: argmax_probability (one vocabulary argmax per column and the winner's softmax probability)
 *
 * Math / indexing:
 *   Let l[v,t] be the FP32 value of logits[v,t] and b[v,t] = round_to_bf16_rne(l[v,t]) its BF16
 *   rounding (the rounding of ops::cast_fp32_to_bf16). Over the valid rows 0 <= v < valid_rows,
 *
 *     ids[t]   = min argmax_v b[v,t]
 *     m[t]     = max_v l[v,t]
 *     probs[t] = exp(l[ids[t],t] - m[t]) / sum_v exp(l[v,t] - m[t]).
 *
 *   ids is therefore exactly ops::argmax(cast_fp32_to_bf16(logits), valid_rows), and probs is the
 *   softmax probability of that winner under the FP32 logits. When the BF16 winner is also the FP32
 *   maximum (always unless several FP32 values round to the same largest BF16 value), probs[t] is
 *   1 / sum_v exp(l[v,t] - m[t]).
 *
 * Logical shapes:
 *   logits is [physical_rows,T], ids and probs are [T], with T > 0 and
 *   1 <= valid_rows <= physical_rows. Physical rows [valid_rows,physical_rows) do not participate.
 *
 * Supported domain:
 *   logits is contiguous finite FP32; ids is contiguous I32; probs is contiguous FP32. T is any
 *   positive extent.
 *
 * Numeric:
 *   The BF16 rounding of the argmax key is a semantic boundary: ties between equal BF16 keys select
 *   the lowest row, so the ids are bitwise those of the BF16 argmax. probs is an FP32
 *   approximation of the formula computed from the FP32 logits; reduction association and private
 *   accumulator precision are implementation choices, and the oracle evaluates the formula in FP64.
 *   Non-finite logits are outside the domain.
 *
 * Effects:
 *   Writes every element of ids and probs and preserves logits. ids, probs and logits must not
 *   overlap one another or the workspace.
 *
 * Workspace:
 *   Transient scratch from `workspace` of at most argmax_probability_workspace_capacity_bytes
 *   (valid_rows, T, T), scoped to the call.
 *
 * Execution:
 *   Enqueues work on stream (capturable in a CUDA graph) and owns no persistent state.
 */
void argmax_probability(const Tensor& logits, Tensor& ids, Tensor& probs, std::int32_t valid_rows,
                        WorkspaceArena& workspace, cudaStream_t stream);

} // namespace ninfer::ops
