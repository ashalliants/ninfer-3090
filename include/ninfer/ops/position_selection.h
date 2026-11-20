#pragma once

// The selection half of DeepSeek-V4's indexer: which compressed positions each query attends to.
// `weighted_relu_dot_scores` scores every (query, position) pair and `top_k_positions` keeps the
// best k of each query's reachable positions. The model repo's `Indexer` and `select_topk`
// (inference/model.py); the NumPy oracle is tools/dsv4/attention.py `indexer` (the query and key
// projections, norms and RoPE around them are other Ops, and the second-level candidate-block
// mask of the later index layers is the caller's: it writes -inf into the scores it rules out).

#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops {

/// The largest head dimension `weighted_relu_dot_scores` serves, and the most positions (the position
/// axis is tiled over a grid dimension, 65,535 tiles of 128).
inline constexpr std::int32_t kScoreMaxHeadDim  = 256;
inline constexpr std::int32_t kScoreMaxPositions = 65535 * 128;

/**
 * Op: weighted_relu_dot_scores
 *
 * Math / indexing:
 *   For every query column s, position t and head h, with D the head dimension:
 *     dot[h]      = sum_d q[d, h, s] * k[d, t]
 *     score[t, s] = weight_scale * sum_h weights[h, s] * max(dot[h], 0)        for t < valid[s]
 *     score[t, s] = -inf                                                         for t >= valid[s]
 *
 * Logical shapes:
 *   q BF16 [D, H, T]; k BF16 [D, N]; weights BF16 [H, T]; valid I32 [T] (any value; clamped to
 *   [0, N]); scores FP32 [N, T] (position t of column s at t + N * s). H in {8, 16, 32, 64};
 *   D a multiple of 8 with D <= kScoreMaxHeadDim and D * H <= 8192; 1 <= N <= kScoreMaxPositions; T >= 1.
 *
 * Supported domain:
 *   Contiguous tensors with 16-byte aligned q and k; finite q, k and weights; finite weight_scale.
 *
 * Numeric:
 *   FP32 accumulation of the BF16 products; against the FP64 oracle on the represented inputs
 *   |got - ref| <= 2e-6 * sum_h |weights[h]| * |dot[h]| * |weight_scale| (a few FP32 ulps of the
 *   magnitude of the terms).
 *
 * Effects:
 *   Writes scores. Inputs are read only; scores must not overlap an input.
 *
 * Workspace: none. Execution: stream-ordered; graph capturable.
 */
void weighted_relu_dot_scores(const Tensor& q, const Tensor& k, const Tensor& weights, const Tensor& valid,
                              float weight_scale, Tensor& scores, cudaStream_t stream);

/// The most positions `top_k_positions` selects per column.
inline constexpr std::int32_t kTopKPositionsMax = 4096;

/**
 * Op: top_k_positions
 *
 * Math / indexing:
 *   For every column s, with n = clamp(valid[s], 0, N) reachable positions 0 <= t < n (the score of
 *   a reachable position may be anything, including -inf) and m = min(k, n): the set of m positions
 *   with the largest scores, ties going to the LOWER position, written in ascending position order,
 *   each plus `offset`:
 *     out[i, s] = (the i-th smallest selected position) + offset    for i < m
 *     out[i, s] = -1                                                for i >= m
 *
 * Logical shapes:
 *   scores FP32 [N, T]; valid I32 [T]; out I32 [k, T]. 1 <= k <= kTopKPositionsMax; N >= 1; T >= 1;
 *   offset >= 0 and offset + N - 1 <= INT32_MAX (every emitted position fits the I32 output).
 *
 * Supported domain:
 *   Contiguous tensors; no NaN among the reachable scores (-inf is allowed).
 *
 * Numeric:
 *   Exact: the selection is a function of the FP32 scores only.
 *
 * Effects:
 *   Writes out. Inputs are read only; out must not overlap an input.
 *
 * Workspace: none. Execution: stream-ordered; graph capturable.
 */
void top_k_positions(const Tensor& scores, const Tensor& valid, std::int32_t k, std::int32_t offset, Tensor& out,
                     cudaStream_t stream);

} // namespace ninfer::ops
