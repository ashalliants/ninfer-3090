#pragma once

// Adapted from Infernix a3edb450 include/infernix/ops/projection_fp32.h (Apache-2.0).
// Modified for NInfer-3090: the BF16 form plus a GGML IQ4_XS, Q6_K or Q5_K head form (Infernix's
// q8_g32_fp16/q4_g64_fp16 heads and its BF16 vocabulary-head tall and tensor-core mappings are not
// ported), and the contract follows op-development §3.

#include "core/tensor.h"
#include "core/weight.h"

#include <cuda_runtime.h>

#include <cstdint>

#include <span>

namespace ninfer::ops {

/// The most weights (row segments) one projection call takes: a visit's projections of one input in one launch.
inline constexpr int kProjectionMaxWeights = 8;

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
 *   K a multiple of 8 and at most 8192 (K up to 5120, DeepSeek-V4's hidden size among them, takes
 *   the narrow mapping at one to eight columns; wider calls and K above 5120 take the wide
 *   mapping; both give the same bits; the BF16 split form below keeps the limit of 3072); one to
 *   kProjectionMaxWeights weights; every operand contiguous and 16-byte aligned.
 *
 * Numeric:
 *   Each output is one FP32 dot product whose summation order depends only on K, so a column's
 *   results are bitwise identical in every batch width and column position. Oracle: the FP64 dot
 *   product, with the FP32 accumulation bound sum_k |w x| * K * 2^-24.
 *
 * Effects: writes out; it must not overlap x or a weight.
 * Workspace: none. Execution: stream-ordered; graph capturable.
 */
/// projection_fp32 with each FP32 result rounded to BF16 (round to nearest even, as
/// cast_fp32_to_bf16 rounds) and the rows split in order: first [N1, T] then second [N2, T],
/// N1 + N2 the weights' rows. The same bits as projection_fp32, cast_fp32_to_bf16 and split_rows.
/// T from 1 to 8 (projection_bf16_split_supported); one launch.
[[nodiscard]] bool projection_bf16_split_supported(std::int32_t columns) noexcept;
void projection_bf16_split(const Tensor& x, std::span<const Tensor* const> weights, Tensor& first, Tensor& second,
                           cudaStream_t stream);

void projection_fp32(const Tensor& x, std::span<const Tensor* const> weights, Tensor& out,
                     cudaStream_t stream);

/// projection_fp32 with an input per weight: out's rows of weight i are weight i by inputs[i], each the same bits as
/// projection_fp32(inputs[i], weight i). One launch for a layer's projections of different inputs. Every input has
/// the same K and T, with T from 1 to 8 (projection_bf16_split_supported) and K at most 3072; with more than one
/// distinct input every weight's N is a multiple of 4 (a CTA's rows come from one input).
void projection_fp32_inputs(std::span<const Tensor* const> inputs, std::span<const Tensor* const> weights, Tensor& out,
                            cudaStream_t stream);

/**
 * Op: projection_fp32, GGML block-format weight (an LM head stored as GGUF IQ4_XS, Q6_K or Q5_K)
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
 *   QType::GGML_IQ4_XS (8-byte aligned rows), GGML_Q5_K (16-byte aligned rows) or GGML_Q6_K
 *   (2-byte aligned rows) in the ggml_blocks_v1 layout with K = 2560 and N even, or GGML_Q5_K with
 *   K = 5120 (DeepSeek-V4.1-Flash's head, any N);
 *   x and out contiguous, x 16-byte aligned.
 *
 * Numeric:
 *   Integer products per 32 values are exact; scaling and the sum over K are FP32, within
 *   sum_k |w xq| * K * 2^-24 of the FP64 dot product of w and xq. Q5_K is affine (w = d sc q -
 *   dmin m per 32 values, the min term formed from the sum of the group's Q8_1 codes), and its
 *   bound sums (|d sc q| + |dmin m|) |xq| instead of |w xq|. A column's results do not depend
 *   on T or on the column's position.
 *
 * Effects: writes out; it must not overlap x or the weight.
 * Workspace: none. Execution: stream-ordered; graph capturable.
 */
void projection_fp32(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream);

/// A BF16 weight [K, N] re-encoded with Q8_0's values (an opt-in, lossy dense form): per 32 consecutive values of
/// a row an FP16 scale d and int8 codes q, stored split: values int8 [N, K] (row n's K codes contiguous, 16-byte
/// aligned rows) and scales FP16 [N, K / 32]. w[n, k] = FP32(d[n, k / 32]) * q[n, k], exact in FP32.
struct Q8Weight {
    const std::int8_t* values = nullptr;
    const void* scales        = nullptr; // __half
    std::int32_t k = 0, rows = 0;
};

/**
 * Op: quantize_q8_g32 (BF16 weight rows to Q8Weight, ggml's quantize_row_q8_0_ref rounding)
 *
 *   amax = max |w| over the 32 values; d = amax / 127 (FP32); q = roundf(w * (d ? 1 / d : 0)); the stored scale
 *   is FP16(d) (round to nearest even).
 *
 * w BF16 [K, N] contiguous with K a multiple of 32; values int8 [N, K] and scales FP16 [N, K / 32] device memory,
 * values 16-byte aligned with K a multiple of 16. Stream-ordered.
 */
void quantize_q8_g32(const Tensor& w, std::int8_t* values, void* scales, cudaStream_t stream);

/**
 * Op: projection_q8_fp32 (BF16 activations by Q8Weight rows, FP32 outputs)
 *
 *   out[n, t] = sum_k w[n, k] * FP32(x[k, t]) for the rows n of `weights` concatenated in order.
 *
 * x BF16 [K, T] contiguous, 16-byte aligned; one to kProjectionMaxWeights weights of that K; out FP32 [sum N, T] contiguous. K a
 * multiple of 32 and at most 3072; T >= 1. Per 16 values of a row the products q x (exact) are summed in FP32 and
 * the partial is scaled by d into the FP32 sum, in an order that depends only on K, so a column's results are bitwise
 * the same at every T and column position. Oracle: the FP64 dot product of the decoded weights and x, within
 * sum_k |w x| * K * 2^-24.
 * Effects: writes out; it must not overlap x or a weight. Workspace: none. Graph capturable.
 */
void projection_q8_fp32(const Tensor& x, std::span<const Q8Weight> weights, Tensor& out, cudaStream_t stream);

} // namespace ninfer::ops
