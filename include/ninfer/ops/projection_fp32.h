#pragma once

// Adapted from Infernix a3edb450 include/infernix/ops/projection_fp32.h (Apache-2.0).
// Modified for NInfer-3090: the BF16 form plus a GGML IQ4_XS head form (Infernix's
// q8_g32_fp16/q4_g64_fp16 heads and its BF16 vocabulary-head tall and tensor-core mappings are not
// ported), and the contract follows op-development §3.

#include "core/tensor.h"
#include "core/weight.h"

#include <cuda_runtime.h>

#include <span>

namespace ninfer::ops {

/**
 * Op: projection_fp32 (BF16 activations by BF16 weight rows, FP32 outputs)
 *
 * For outputs whose consumers need more than BF16 resolution, such as MoE router logits (BF16
 * logits turn experts within one BF16 step of the top-k boundary into index-biased ties).
 *
 * Math / indexing:
 *   out[n, t] = sum_k FP32(w[n, k]) * FP32(x[k, t]) for the rows n of `weights` concatenated in
 *   order.
 *
 * Logical shapes:
 *   x BF16 [K, T]; weight i BF16 [K, N_i] (one row per column of the view); out FP32
 *   [sum N_i, T]. T >= 1.
 *
 * Supported domain:
 *   K a multiple of 8 and at most 3072; one to four weights; every operand contiguous and
 *   16-byte aligned.
 *
 * Numeric:
 *   Each output is one FP32 dot product whose summation order depends only on K, so a column's
 *   results are bitwise identical in every batch width and column position. Oracle: the FP64 dot
 *   product, with the FP32 accumulation bound sum_k |w x| * K * 2^-24.
 *
 * Effects: writes out; it must not overlap x or a weight.
 * Workspace: none. Execution: stream-ordered; graph capturable.
 */
void projection_fp32(const Tensor& x, std::span<const Tensor* const> weights, Tensor& out,
                     cudaStream_t stream);

/**
 * Op: projection_fp32, GGML block-format weight (an LM head stored as GGUF IQ4_XS)
 *
 * For vocabulary logits: BF16 logits quantize token probabilities by up to 2^-8 of the logit.
 *
 * Math / indexing:
 *   out[n, t] = sum_k w[n, k] * xq[k, t], where w is the exact decode of the weight's blocks
 *   (tensor-formats.md, GGML formats) and xq is the Q8_1 cast of x, the activation cast of every
 *   GGML weight format (linear.h): per 32 values of a column, d = amax / 127 and
 *   xq = d * rne(x * (127 / amax)), xq = 0 when amax = 0.
 *
 * Logical shapes:
 *   x BF16 [K, T]; weight [N, K]; out FP32 [N, T]. T >= 1.
 *
 * Supported domain:
 *   QType::GGML_IQ4_XS in the ggml_blocks_v1 layout (8-byte aligned rows) with K = 2560 and N even.
 *
 * Numeric:
 *   Integer products per 32 values are exact; scaling and the sum over K are FP32, within
 *   sum_k |w xq| * K * 2^-24 of the FP64 dot product of w and xq. A column's results do not depend
 *   on T or on the column's position.
 *
 * Effects: writes out; it must not overlap x or the weight.
 * Workspace: none. Execution: stream-ordered; graph capturable.
 */
void projection_fp32(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream);

} // namespace ninfer::ops
