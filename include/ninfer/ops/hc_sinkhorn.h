#pragma once

// The residual-stream arithmetic of DeepSeek-V4's hyper-connections: a Sinkhorn-normalised mixing
// matrix between the streams, as the model repo's `Block.hc_mixes / hc_pre / hc_post` and its
// `hc_split_sinkhorn` kernel define it. It is not hyper_connection.h's mixer (Flash-Next's gated
// low-rank form): the formulas differ, so these are their own Ops. The NumPy oracle is
// tools/dsv4/hyper_connection.py.

#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops {

/// The streams of DeepSeek-V4's hyper-connections (hc_mult).
inline constexpr std::int32_t kHcSinkhornStreams = 4;

/// Rows of the mixing projection: (2 + S) * S (S pre, S post and S * S comb logits).
inline constexpr std::int32_t kHcSinkhornMixRows = (2 + kHcSinkhornStreams) * kHcSinkhornStreams;

/**
 * Op: hc_sinkhorn_mixes (the pre, post and comb coefficients of one residual stream)
 *
 * Math / indexing:
 *   The residual holds S streams of H features per column, stream-major: R[s*H + d, t] is feature d
 *   of stream s. For every column t independently, with K = S*H:
 *     r      = (mean_k R[k, t]^2 + norm_eps)^(-1/2)           (one statistic over the whole K)
 *     m[n]   = r * sum_k fn[n, k] * R[k, t]                    for n < (2 + S) * S
 *     pre[i]  = sigmoid(m[i] * scale[0] + base[i]) + eps                                  i < S
 *     post[i] = 2 * sigmoid(m[S + i] * scale[1] + base[S + i])                            i < S
 *     c[i, j] = m[2S + i*S + j] * scale[2] + base[2S + i*S + j]                           i, j < S
 *     A       = softmax_j(c[i, j]) + eps                       (row-wise softmax)
 *     A       = A / (sum_i A[i, j] + eps)                      (column normalisation)
 *     repeat (iters - 1) times:
 *       A = A / (sum_j A[i, j] + eps)                          (rows), then
 *       A = A / (sum_i A[i, j] + eps)                          (columns)
 *     comb[i, j] = A[i, j]
 *   with eps = sinkhorn_eps.
 *
 * Logical shapes:
 *   residual BF16 [S*H, T]; fn FP32 [S*H, (2+S)*S] (row n contiguous); scale FP32 [3]; base FP32
 *   [(2+S)*S]; pre FP32 [S, T]; post FP32 [S, T]; comb FP32 [S*S, T], comb[i*S + j, t] = A[i, j].
 *   S = kHcSinkhornStreams; T >= 1.
 *
 * Supported domain:
 *   Contiguous tensors; S*H a multiple of 8 and at most 32,768; 16-byte aligned residual and fn;
 *   norm_eps > 0, sinkhorn_eps >= 0, 1 <= iters <= 64.
 *
 * Numeric:
 *   FP32 arithmetic from the represented BF16 residual and FP32 weights; the statistic and the
 *   projection are FP32 sums. The oracle is the formula in FP64.
 *
 * Effects:
 *   Writes pre, post and comb. The inputs are read only and no output may overlap an input.
 *
 * Workspace: none. Execution: stream-ordered; graph capturable.
 */
void hc_sinkhorn_mixes(const Tensor& residual, const Tensor& fn, const Tensor& scale,
                       const Tensor& base, float norm_eps, float sinkhorn_eps,
                       std::int32_t sinkhorn_iters, Tensor& pre, Tensor& post, Tensor& comb,
                       cudaStream_t stream);

/**
 * Op: hc_collapse (the streams summed into one sublayer input)
 *
 * Math / indexing:
 *   x[d, t] = bf16( sum_s pre[s, t] * R[s*H + d, t] )
 *
 * Logical shapes:
 *   residual BF16 [S*H, T]; pre FP32 [S, T]; x BF16 [H, T]. S = kHcSinkhornStreams; T >= 1.
 *
 * Supported domain: contiguous tensors; H >= 1.
 *
 * Numeric: the sum is FP32 from the represented values, rounded once to BF16.
 *
 * Effects: writes x; residual and pre are read only and x may not overlap them.
 *
 * Workspace: none. Execution: stream-ordered; graph capturable.
 */
void hc_collapse(const Tensor& residual, const Tensor& pre, Tensor& x, cudaStream_t stream);

/**
 * Op: hc_expand_mix (a sublayer's output written back into the streams, the old streams mixed in)
 *
 * Math / indexing:
 *   R'[j*H + d, t] = bf16( post[j, t] * y[d, t] + sum_i comb[i*S + j, t] * R[i*H + d, t] )
 *   for every stream j < S, feature d and column t, where R is the residual before the call.
 *
 * Logical shapes:
 *   y BF16 [H, T]; post FP32 [S, T]; comb FP32 [S*S, T] (hc_sinkhorn_mixes' layout); residual BF16
 *   [S*H, T]. S = kHcSinkhornStreams; T >= 1.
 *
 * Supported domain: contiguous tensors.
 *
 * Numeric: the products and sum are FP32 from the represented values, rounded once to BF16.
 *
 * Effects: updates residual in place (every output element reads only its own feature d of all S
 *   streams); y, post and comb are read only and must not overlap it.
 *
 * Workspace: none. Execution: stream-ordered; graph capturable.
 */
void hc_expand_mix(const Tensor& y, const Tensor& post, const Tensor& comb, Tensor& residual,
                   cudaStream_t stream);

} // namespace ninfer::ops
