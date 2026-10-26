#pragma once

#include "core/tensor.h"
#include "core/weight.h"

#include <cuda_runtime.h>

namespace ninfer::ops {

/**
 * Op: linear_rows
 *
 * Math / indexing:
 *   For every slot c in [0,C) and column t in [0,T):
 *     out[c,t] = sum_{k=0}^{K-1} FP32Dequant(head)_{ids[c],k} * FP32(hidden[k,t])
 *   when 0 <= ids[c] < N, and out[c,t] = -inf otherwise (an empty slot; callers use -1).
 *   Only the selected rows of `head` are read.
 *
 * Logical shapes:
 *   hidden is BF16 [K,T]; head is [N,K]; ids is I32 [C]; out is FP32 [C,T]. C, N, K and T are
 *   positive; dimension zero is stored fastest.
 *
 * Supported domain:
 *   head is RowSplit Q8_G32_FP16, Q6_G64_FP16 or Q4_G64_FP16 with FP16 scales and K a multiple
 *   of 128 (row_split_k128_v1, storage-layouts.md Section 3, with no padded columns). hidden, ids
 *   and out are contiguous and non-null; hidden and the code planes are 16-byte aligned.
 *
 * Numeric:
 *   The ideal result is evaluated with FP32 products and accumulation in an unspecified order and
 *   returned in FP32. The oracle is the FP64 dot product of the dequantized rows and the BF16
 *   activation. Empty slots are exactly -inf; a valid slot is finite for finite inputs.
 *
 * Effects:
 *   Writes every element of out. hidden, ids and every weight plane are preserved. out must not
 *   overlap hidden, ids or the weight.
 *
 * Workspace: none.
 * Execution: stream-ordered; graph capturable (the ids are read on the device at replay).
 */
void linear_rows(const Tensor& hidden, const Weight& head, const Tensor& ids, Tensor& out,
                 cudaStream_t stream);

/**
 * Op: merge_row_argmax
 *
 * Math / indexing:
 *   For every column t in [0,T), with r = tokens[t] (input), p = FP32(logits[r,t]) and
 *   g = row_to_global ? row_to_global[r] : r, and over the valid slots c (0 <= row_ids[c]):
 *     s* = max_c row_scores[c,t], c* the lowest slot reaching it;
 *     tokens[t] (output) = row_ids[c*] if a valid slot exists and s* > p, otherwise g.
 *   A tie keeps the proposal's id; NaN scores never win.
 *
 * Logical shapes:
 *   logits is BF16 [R,T]; row_to_global, when given, is I32 [R]; row_scores is FP32 [C,T];
 *   row_ids is I32 [C]; tokens is I32 [T]. R, C and T are positive.
 *
 * Supported domain:
 *   Contiguous, non-null tensors. Every input tokens[t] is in [0,R) (an argmax row of logits).
 *
 * Numeric: exact (comparisons and copies only).
 *
 * Effects:
 *   Updates tokens in place; the other tensors are preserved and must not overlap tokens.
 *
 * Workspace: none.
 * Execution: stream-ordered; graph capturable.
 */
void merge_row_argmax(const Tensor& logits, const Tensor* row_to_global, const Tensor& row_scores,
                      const Tensor& row_ids, Tensor& tokens, cudaStream_t stream);

/**
 * Op: merge_row_topk
 *
 * Math / indexing:
 *   For every column u in [0,U), let L be the union of the 16 input candidates
 *   (candidate_ids[i,u], candidate_scores[i,u]) and the valid slots (row_ids[c], row_scores[c,u])
 *   with 0 <= row_ids[c] and a score that is not NaN. The outputs are the 16 members of L ranked
 *   first by descending score, exact score ties ranked by lower id, stored in rank order.
 *
 * Logical shapes:
 *   row_scores is FP32 [C,U]; row_ids is I32 [C]; candidate_ids is I32 [16,U] and
 *   candidate_scores is FP32 [16,U]. C and U are positive.
 *
 * Supported domain:
 *   Contiguous, non-null tensors; C at most 2048. Input candidate scores are not NaN, and no valid
 *   row id equals an input candidate id of the same column (a context row is never a proposal
 *   row).
 *
 * Numeric: exact (comparisons and copies only).
 *
 * Effects:
 *   Overwrites both candidate tensors in place; row_scores and row_ids are preserved and must not
 *   overlap the candidates.
 *
 * Workspace: none.
 * Execution: stream-ordered; graph capturable.
 */
void merge_row_topk(const Tensor& row_scores, const Tensor& row_ids, Tensor& candidate_ids,
                    Tensor& candidate_scores, cudaStream_t stream);

} // namespace ninfer::ops
