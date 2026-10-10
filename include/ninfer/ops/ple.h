#pragma once

// Adapted from Infernix a3edb450 include/infernix/ops/ple.h (Apache-2.0).
// Modified for NInfer-3090: n-gram rows are GGML IQ4_NL blocks (Infernix: FP8 E4M3 with one
// per-tensor scale), the three norm weights are stored FP32 multipliers instead of BF16 unit-offset
// weights, the convolution weight is FP16 with each channel's taps contiguous, and the contract
// follows op-development §3.

#include "core/tensor.h"
#include "core/weight.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops {

/**
 * Per-layer n-gram embedding injection (PLE) into hyper-connection residual streams. For each
 * column t, with S streams of H features (residual R [S*H, T], stream-major):
 *
 *   e        = decode of the column's n-gram rows                              -> ple_embed
 *   key      = W_key e  [S*H],  value = W_value e  [H]                         (caller's Linear)
 *   g_s      = <norm(key_s; w_key), norm(R_s; w_query)> / sqrt(H)
 *   g_s      = sign(g_s) * sqrt(max(|g_s|, 1e-6))
 *   gated_s  = sigmoid(g_s) * value                                            -> ple_gate
 *   nrm      = norm(gated; w_conv)                                             -> ple_gate
 *   R       += gated + SiLU(sum_j w_conv1d[c, j] * nrm[t - (K-1-j) * dilation]) -> ple_conv_inject
 *
 * norm(v_s; w) is the per-stream RMSNorm v_s * (mean v_s^2 + eps)^(-1/2) * w_s, with w the stored
 * full multiplier (a checkpoint's 1 + gamma). The convolution is depthwise over the S*H channels
 * and causal; its state holds the last (K-1)*dilation columns of nrm of each sequence.
 */

/**
 * Op: ple_embed (decode GGML IQ4_NL n-gram rows to BF16)
 *
 * Math / indexing:
 *   Each row is B / 18 IQ4_NL blocks of 32 values (tensor-formats.md, GGML formats): block b is an
 *   FP16 scale d_b followed by 16 bytes q; value 32b + j is d_b * kvalues_iq4nl[q[j] & 15] for
 *   j < 16 and d_b * kvalues_iq4nl[q[j - 16] >> 4] for j >= 16.
 *   out[h*E + i, t] = bf16(value i of rows[:, h, t]), E = 32 * B / 18.
 *
 * Logical shapes:
 *   rows U8 [B, heads, T] with B a positive multiple of 18; out BF16 [heads*E, T]; T >= 1.
 *
 * Supported domain:
 *   format == QType::GGML_IQ4_NL.
 *
 * Numeric:
 *   The FP32 product d_b * kvalue is exact; out is its round-to-nearest-even BF16. Exact output.
 *
 * Effects: writes out; rows is read only and must not overlap it.
 * Workspace: none. Execution: stream-ordered; graph capturable.
 */
void ple_embed(const Tensor& rows, QType format, Tensor& out, cudaStream_t stream);

/**
 * Op: ple_gate
 *
 * Math / indexing:
 *   For each stream s and column t, g_s, gated_s and nrm_s of the PLE formula above:
 *   gated[s*H + d, t] = sigmoid(g_s) * value[d, t]; normalized = norm(gated; w_conv).
 *
 * Logical shapes:
 *   key BF16 [S*H, T]; value BF16 [H, T]; residual BF16 [S*H, T]; key_norm, query_norm,
 *   conv_norm FP32 [S*H]; gated and normalized BF16 [S*H, T]. T >= 1, eps > 0.
 *
 * Numeric:
 *   FP32 from the represented inputs; gated and normalized are each rounded once to BF16
 *   (normalized is formed from the unrounded FP32 gated value). Oracle: the formula in FP64.
 *
 * Effects: writes gated and normalized; no output overlaps an input.
 * Workspace: none. Execution: stream-ordered; graph capturable.
 */
void ple_gate(const Tensor& key, const Tensor& value, const Tensor& residual,
              const Tensor& key_norm, const Tensor& query_norm, const Tensor& conv_norm,
              std::int32_t streams, float eps, Tensor& gated, Tensor& normalized,
              cudaStream_t stream);

/**
 * Op: ple_conv_inject (stateful causal depthwise convolution and residual injection)
 *
 * Math / indexing:
 *   The T columns form `sequences` runs of W = T / sequences consecutive columns. For sequence i,
 *   local column u and channel c, with span = (K-1) * dilation and history h_i[c, 0..span) (oldest
 *   first) read from slot source_slots[i]:
 *     src(p)          = normalized[c, i*W + p] for p >= 0, h_i[c, span + p] for p < 0
 *     a               = sum_{j<K} weight[c, j] * src(u - (K-1-j) * dilation)
 *     residual[c, t] += gated[c, t] + SiLU(a)
 *   New state, when destination_slots is non-empty: slot destination_slots[i] receives
 *   h'[c, k] = src(W - span + k) for k < span (the trailing span columns of history || normalized).
 *
 * Logical shapes:
 *   gated, normalized, residual BF16 [C, T]; weight FP16 [K, C] (tap j of channel c at c*K + j);
 *   states BF16 [C, span, slots]; source_slots, destination_slots I32 [sequences]. K >= 2,
 *   dilation >= 1, T a multiple of sequences.
 *
 * Numeric:
 *   FP32 accumulation from the represented inputs; residual rounded once to BF16 per call. The new
 *   state copies BF16 values exactly. Oracle: the formula in FP64.
 *
 * Effects:
 *   Updates residual in place. Every convolution read sees the old history. With a non-empty
 *   destination_slots each destination slot is written after all reads (equal source and
 *   destination slots update in place; destination slots must be distinct, and no sequence's
 *   destination may be another sequence's source); an empty
 *   destination_slots (data == nullptr) leaves every state unchanged (verification, committed later
 *   by ple_conv_commit). Slots not named are untouched.
 *
 * Workspace: none. Execution: stream-ordered; graph capturable.
 */
void ple_conv_inject(const Tensor& gated, const Tensor& normalized, const Tensor& weight,
                     std::int32_t dilation, Tensor& states, const Tensor& source_slots,
                     const Tensor& destination_slots, Tensor& residual, cudaStream_t stream);

/**
 * Op: ple_conv_commit
 *
 * Math / indexing:
 *   For row b with n = commit_columns[b] in [0, W]: when n > 0, slot slots[b] becomes the trailing
 *   span columns of (old history || normalized[:, 0:n, b]). n = 0 leaves the slot unchanged. The
 *   result equals ple_conv_inject's in-place state update over those n columns.
 *
 * Logical shapes:
 *   normalized BF16 [C, W, B]; commit_columns, slots I32 [B]; states BF16 [C, span, slots].
 *
 * Numeric: exact copies. Effects: updates the named slots; slots must be distinct.
 * Workspace: none. Execution: stream-ordered; graph capturable.
 */
void ple_conv_commit(const Tensor& normalized, const Tensor& commit_columns, Tensor& states,
                     const Tensor& slots, cudaStream_t stream);

} // namespace ninfer::ops
