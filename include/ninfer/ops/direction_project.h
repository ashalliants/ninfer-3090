#pragma once

#include "core/tensor.h"

#include <cstdint>
#include <cuda_runtime.h> // cudaStream_t

namespace ninfer::ops {

/**
 * Removes one direction from every hyper-connection stream of selected columns (refusal steering):
 *
 *   for each column t with k = row_select[t] >= 0, and each stream s of `streams`:
 *     ideal[s, :, t] = x[s, :, t] - (x[s, :, t] . v_k) v_k,
 *
 * where x[s, :, t] is the H = residual.ne[0] / streams values of stream s and v_k is the FP32
 * vector `directions + k * set_stride` (H values). The vector is the caller's: it should be a unit
 * vector, which makes the result orthogonal to v_k; the Op does not normalise it. Columns with a
 * negative `row_select` are left bit-for-bit unchanged.
 *
 * `residual` is contiguous BF16 [streams * H, T] and is updated in place. `directions` and
 * `row_select` (I32 [T]) are device pointers read by the kernel only, so a captured graph sees the
 * selection of each replay. The oracle evaluates `ideal` in FP64 from the represented inputs; the
 * BF16 result is compared with it directly, so the one output rounding belongs to the Op's
 * criterion. Private kernel arithmetic is implementation-defined (FP32 dot and update). The Op uses
 * no workspace or other persistent state.
 */
void direction_project(Tensor& residual, std::int32_t streams, const float* directions,
                       std::int64_t set_stride, const std::int32_t* row_select,
                       cudaStream_t stream);

} // namespace ninfer::ops
