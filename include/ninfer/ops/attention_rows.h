#pragma once

// Row bookkeeping of DeepSeek-V4's attention layer that depends on where a chunk of tokens sits in
// its sequence: which compressed rows each query can reach, the list of KV-table rows each query
// attends to, and the second-level candidate-block selection of the later index layers. They are
// small exact (integer or max/compare) kernels with no arithmetic to round; the oracles are the
// host formulas below and tools/dsv4/attention.py (`window_indices`, the reachability rule
// `j < (t + 1) // ratio`, `select_candidate_blocks` and the `np.where(shared.candidates, ...)`
// mask of `indexer`). Position-dependent values are kernel arguments, so a captured graph is
// replayed for the same chunk position (re-capture, or update the kernel node, for the next one).

#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops {

/**
 * Op: compressed_lengths
 *
 * Math: out[s] = (start + s + 1) / ratio (integer division): the compressed rows complete when the
 * query at position start + s runs, i.e. what `weighted_relu_dot_scores` and `top_k_positions` take
 * as `valid`.
 *
 * Shapes: out I32 [T]; start >= 0; ratio >= 1; start + T <= INT32_MAX.
 * Workspace: none. Execution: stream-ordered; graph capturable.
 */
void compressed_lengths(std::int32_t start, std::int32_t ratio, Tensor& out, cudaStream_t stream);

// clang-format off
/**
 * Op: attention_row_lists
 *
 * Math: the row list of query s (position start + s) of a chunk, for `gathered_sink_attention`.
 * The layer's raw KV rows sit in the table from row `raw_base`: row raw_base + r holds position
 * start - (window - 1) + r (the previous window - 1 positions, then the chunk), so the query's
 * window is the table rows raw_base + s + j for j in [0, window). Selected compressed positions
 * index the table from row `comp_base`.
 *   rows[j, s] = raw_base + s + j                 if j < window and start + s - (window - 1) + j >= 0
 *   rows[j, s] = -1                               if j < window and that position is before the sequence
 *   rows[window + i, s] = comp_base + topk[i, s]  if topk[i, s] >= 0
 *   rows[window + i, s] = -1                      if topk[i, s] < 0
 *
 * Shapes: rows I32 [window + Kt, T]; topk I32 [Kt, T], or null for no compressed part (Kt = 0).
 * window >= 1; start, raw_base, comp_base >= 0; raw_base + T + window fits I32.
 * Workspace: none. Execution: stream-ordered; graph capturable.
 */
void attention_row_lists(const Tensor* topk, std::int32_t start, std::int32_t window,
                         std::int32_t raw_base, std::int32_t comp_base, Tensor& rows,
                         cudaStream_t stream);

/**
 * Op: block_max_scores
 *
 * Math: level one of the two-level selection (`select_candidate_blocks`). For query s with
 * n = clamp(valid[s], 0, N) reachable positions and block b covering positions [b bs, (b + 1) bs):
 *   out[b, s] = +inf    for the block holding the newest reachable position, b = (n - 1) / bs (n >= 1)
 *   out[b, s] = -inf    for b * bs >= n (no reachable position)
 *   out[b, s] = max over t in the block of scores[t, s]    otherwise (every such t is reachable)
 *
 * Shapes: scores FP32 [N, T]; valid I32 [T]; out FP32 [ceil(N / bs), T]; bs >= 1.
 * Workspace: none. Execution: stream-ordered; graph capturable.
 */
void block_max_scores(const Tensor& scores, const Tensor& valid, std::int32_t block_size,
                      Tensor& out, cudaStream_t stream);

/**
 * Op: mask_scores_outside_blocks
 *
 * Math: scores[t, s] = -inf unless block t / bs appears in column s of `selected` (the block ids
 * `top_k_positions` chose over `block_max_scores`: ascending, -1 padded at the end); other scores
 * are unchanged. The later index layers search only inside the candidate blocks.
 *
 * Shapes: scores FP32 [N, T] (updated in place); selected I32 [Kb, T]; bs >= 1.
 * Workspace: none. Execution: stream-ordered; graph capturable.
 */
void mask_scores_outside_blocks(const Tensor& selected, std::int32_t block_size, Tensor& scores,
                                cudaStream_t stream);

// clang-format on

} // namespace ninfer::ops
