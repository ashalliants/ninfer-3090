#pragma once

// The pooling step of DeepSeek-V4's KV compressor: every channel pools each group of `ratio`
// consecutive tokens with a softmax over the group, scored by a second projection. The model repo's
// `Compressor` (inference/model.py); the NumPy oracle is tools/dsv4/attention.py `compressor`
// (the projections and the norm after it are the linear and rmsnorm Ops).

#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::ops {

/// The largest group `softmax_pool` serves.
inline constexpr int kSoftmaxPoolMaxRatio = 256;

/**
 * Op: softmax_pool
 *
 * Math / indexing:
 *   For every channel d and group g in [0, G), with r = ratio and the group's tokens
 *   n = g * r + i, i in [0, r):
 *     m           = max_i score[d, n]
 *     out[d, g]   = sum_i kv[d, n] * exp(score[d, n] - m) / sum_i exp(score[d, n] - m)
 *   Tokens at or past G * r (a trailing partial group) are not read.
 *
 * Logical shapes:
 *   kv BF16 [D, T]; score BF16 [D, T]; out BF16 [D, G]. 2 <= ratio <= kSoftmaxPoolMaxRatio;
 *   G >= 1 and G * ratio <= T.
 *
 * Supported domain:
 *   Contiguous tensors; finite kv and score.
 *
 * Numeric:
 *   FP32 arithmetic on the BF16 inputs; one BF16 rounding of the result. Against the FP64 oracle on
 *   the represented inputs: half a BF16 ulp plus FP32 evaluation error (a few ulps of the group's
 *   largest |kv|).
 *
 * Effects:
 *   Writes out. Inputs are read only; out must not overlap an input.
 *
 * Workspace: none. Execution: stream-ordered; graph capturable.
 */
void softmax_pool(const Tensor& kv, const Tensor& score, int ratio, Tensor& out, cudaStream_t stream);

} // namespace ninfer::ops
