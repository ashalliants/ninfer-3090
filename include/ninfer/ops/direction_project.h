#pragma once

#include "core/tensor.h"

#include <cstdint>
#include <cuda_runtime.h> // cudaStream_t

namespace ninfer::ops {

/**
 * Activation steering along one direction per selected column (refusal control vectors):
 *
 *   for each column t with k = row_select[t / row_columns] >= 0, and each stream s of `streams`:
 *     ideal[s, :, t] = x[s, :, t] - p_k (x[s, :, t] . v_k) v_k + a_k v_k,
 *
 * where x[s, :, t] is the H = residual.ne[0] / streams values of stream s, v_k is the FP32 vector
 * `directions + k * set_stride` (H values), and (p_k, a_k) are the two FP32 values at
 * `coefficients + k * coefficient_stride`: the projection strength and the additive scale. A null
 * `coefficients` means (1, 0) for every set, the plain projection x - (x . v) v. For p = 1 and a
 * unit v the result is orthogonal to v; the Op does not normalise v. Columns with a negative
 * selection, and columns whose set has p = a = 0, are left bit-for-bit unchanged.
 *
 * `row_select` is I32 [T / row_columns]: one selection per row of `row_columns` consecutive
 * columns (a sequence's verification window, or 1 for one selection per column); T must be a
 * multiple of `row_columns`. `residual` is contiguous BF16 [streams * H, T] and is updated in
 * place. `directions`, `coefficients` and `row_select` are device pointers read by the kernel
 * only, so a captured graph sees the selection of each replay. The oracle evaluates `ideal` in FP64
 * from the represented inputs; the BF16 result is compared with it directly, so the one output
 * rounding belongs to the Op's criterion. Private kernel arithmetic is implementation-defined
 * (FP32 dot and update). The Op uses no workspace or other persistent state.
 */
void direction_project(Tensor& residual, std::int32_t streams, const float* directions,
                       std::int64_t set_stride, const float* coefficients,
                       std::int64_t coefficient_stride, const std::int32_t* row_select,
                       std::int32_t row_columns, cudaStream_t stream);

} // namespace ninfer::ops
