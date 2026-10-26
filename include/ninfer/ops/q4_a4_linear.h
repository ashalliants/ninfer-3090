#pragma once

// ninfer::ops - 4-bit groupwise activations (A4G64) and their product with Q4_G64_FP16 weights.

#include "core/tensor.h"
#include "core/weight.h"

#include <cuda_runtime.h>

namespace ninfer::ops {

/**
 * Op: quantize_a4_g64
 *
 * Math / indexing:
 *   For every token t and every group g of 64 consecutive input rows (k = 64g .. 64g+63), with
 *   x_k the FP32 value represented by x[k,t]:
 *
 *     a      = max_k |x_k|                              (exact)
 *     d      = binary16_rn(a / 7)                       (FP32 division, then round to nearest
 *                                                        even binary16; +inf saturates to 65504)
 *     q_k    = 0                                        if d == 0
 *     q_k    = clamp(rint(x_k * (1 / d)), -7, 7)        otherwise: FP32 reciprocal and FP32
 *                                                        product, each rounded to nearest even,
 *                                                        rint ties to even
 *
 *   xs[g,t] = d and the codes q_k are stored in xq. The represented activation is q_k * d.
 *   The codes are symmetric: -8 is never produced.
 *
 * Logical shapes:
 *   x [K,T], xq [K/2,T], xs [K/64,T], dimension zero fastest. K and T are positive and
 *   K % 64 == 0.
 *
 * Supported domain:
 *   x is contiguous BF16 or FP16 with 16-byte-aligned data; its values are finite. xq is
 *   contiguous U8 with 4-byte-aligned data. xs is contiguous FP16.
 *
 * Numeric:
 *   An exact transform: every output bit is determined by the formula above.
 *
 *   Packed layout (the A4G64 representation, shared with q4_a4_linear): token t owns K/2
 *   consecutive bytes of xq starting at t*K/2. Byte j holds q_{2j} in its low nibble and
 *   q_{2j+1} in its high nibble, each as a 4-bit two's-complement word. This is exactly the
 *   Q4_G64_FP16 row-split base-code order (storage-layouts.md Section 3.3) with tokens in place of
 *   weight rows, so 4 consecutive bytes hold 8 consecutive k in nibble order -- the s4 operand
 *   word of mma.m16n8k64. Token t's group scales are xs[0..K/64-1, t], consecutive binary16.
 *
 * Effects:
 *   Writes all of xq and xs. None of x, xq, xs may alias.
 *
 * Workspace:
 *   None.
 *
 * Execution:
 *   Asynchronous on `stream`; capturable.
 */
void quantize_a4_g64(const Tensor& x, Tensor& xq, Tensor& xs, cudaStream_t stream);

/**
 * Op: q4_a4_linear
 *
 * Math / indexing:
 *   out[n,t] = sum_k W[n,k] * A[k,t], where W[n,k] = c_{n,k} * s_{n,k/64} is the represented
 *   Q4_G64_FP16 weight (signed code times its binary16 group scale) and A[k,t] = q_{k,t} *
 *   xs[k/64,t] is the represented A4G64 activation produced by quantize_a4_g64.
 *
 * Logical shapes:
 *   xq [K/2,T], xs [K/64,T] as defined by quantize_a4_g64; w [N,K]; out [N,T], dimension zero
 *   fastest. N, K and T are positive and K % 128 == 0.
 *
 * Supported domain:
 *   w is Q4_G64_FP16 in RowSplit or RowSplitPanel layout with FP16 scales, padded K equal to K,
 *   16-byte-aligned codes and 4-byte-aligned scales (RowSplitPanel also requires N % 4 == 0).
 *   xq is contiguous U8 with 16-byte-aligned data, xs contiguous FP16 with 4-byte-aligned data,
 *   out contiguous BF16 or FP16 with 4-byte-aligned data. Every N, K % 128 == 0 and T admitted
 *   above is served; there are no registered shapes. Codes of -8 in xq are accepted and
 *   computed exactly although quantize_a4_g64 never writes them.
 *
 * Numeric:
 *   Within each group of 64 the integer dot product sum c * q is exact. Group results are scaled
 *   and summed in FP32 (the scaled group term is exact before the accumulation rounds); out is
 *   the FP32 result rounded to nearest even in out's dtype. The oracle decodes both operands to
 *   their represented values and sums the K products in FP64; there is no activation-quantisation
 *   allowance, because the A4 cast is the explicit quantize_a4_g64 boundary, not part of this Op.
 *   The A4 cast itself is lossy -- about 7% relative L2 against the unquantised product on the
 *   test's outlier-free synthetic activations, and about 20% when a group holds a 24x outlier
 *   channel, which zeroes the other 63 codes of its group -- so callers own where that is
 *   acceptable.
 *
 * Effects:
 *   Writes all of out. out must not alias xq, xs or w.
 *
 * Workspace:
 *   None.
 *
 * Execution:
 *   Asynchronous on `stream`; capturable. Requires an sm_80+ device (s4 integer tensor cores).
 */
void q4_a4_linear(const Tensor& xq, const Tensor& xs, const Weight& w, Tensor& out,
                  cudaStream_t stream);

} // namespace ninfer::ops
