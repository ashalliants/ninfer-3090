#pragma once

// Attention of DeepSeek-V4: many query heads over ONE shared key/value head, where each query reads
// its own list of rows from a table of KV rows (its sliding window plus the compressed positions
// the indexer chose), and a per-head "sink" logit joins the softmax denominator only. This is the
// `sparse_attn` kernel of the model repo (inference/kernel.py); the NumPy oracle is
// tools/dsv4/attention.py `sparse_attention`. The existing attention Ops
// (softmax_attention.h, sliding_window_attention.h) serve grouped heads with separate keys and
// values and fixed windows, and do not.

#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::ops {

/// The most query heads `gathered_sink_attention` serves (the head index is a grid dimension).
inline constexpr int kGatheredSinkAttentionMaxHeads = 1024;

/**
 * Op: gathered_sink_attention
 *
 * Math / indexing:
 *   For every query column t and head h, with D the head dimension, K the list length, and
 *   rows(t) = { idx[j, t] : 0 <= idx[j, t] < N } (entries outside [0, N) are skipped; duplicates
 *   count each time):
 *     s[j]    = scale * sum_d q[d, h, t] * kv[d, idx[j, t]]            for the j in rows(t)
 *     m       = max(max_j s[j], sink[h])
 *     out[:, h, t] = sum_j exp(s[j] - m) * kv[:, idx[j, t]] / (sum_j exp(s[j] - m) + exp(sink[h] - m))
 *   The key and the value of a row are the same vector. A column whose list has no valid entry
 *   produces exact zeros (not the sink's share of nothing).
 *
 * Logical shapes:
 *   q BF16 [D, H, T]; kv BF16 [D, N] (N >= 1); idx I32 [K, T] (K >= 1, slot j of column t at
 *   j + K * t); sink FP32 [H]; out BF16 [D, H, T]. D in {64, 128, 256, 512}; 1 <= H <= kGatheredSinkAttentionMaxHeads; T >= 1.
 *
 * Supported domain:
 *   Contiguous tensors with 4-byte aligned data; finite q and kv and sink; scale positive and finite.
 *
 * Numeric:
 *   FP32 accumulation of the BF16 products and an online softmax; the output is rounded to BF16
 *   once. Against the FP64 oracle on the represented inputs the result differs by that rounding and
 *   by FP32 summation order (a few FP32 ulps relative to the output's scale).
 *   Chunks of 8 or more columns with D = 256 or 512, kv 16-byte aligned, on a device of compute capability 8.0 or newer
 *   run on tensor cores: the scores keep that accuracy, but the softmax weights multiply the values
 *   as BF16 (each weight rounded once, the denominator summing the rounded weights), so each output
 *   element also differs by at most 2^-8 * (sum_j w_j |kv[d, row j]| / sum_j w_j), w_j the
 *   unnormalised weights, on top of the BF16 rounding.
 *
 * Effects:
 *   Writes out. Inputs are read only; out must not overlap an input.
 *
 * Workspace: none. Execution: stream-ordered; graph capturable.
 */
void gathered_sink_attention(const Tensor& q, const Tensor& kv, const Tensor& idx, const Tensor& sink,
                             float scale, Tensor& out, cudaStream_t stream);

} // namespace ninfer::ops
