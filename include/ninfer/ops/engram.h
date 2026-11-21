#pragma once

// DeepSeek-V4's Engram injection (inference/model.py `Engram`, tools/dsv4/engram_ref.py
// `engram_forward`): the gated write of an n-gram value into the hyper-connection streams. The
// rows' decode is ple_embed (ops/ple.h, IQ4_NL), the projection of the 24 decoded rows to a key per
// stream and one shared value is Linear; this Op is what follows them.

#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops {

/**
 * Op: engram_gate_inject
 *
 * Math / indexing:
 *   The residual R holds S streams of H features per column, stream-major (R[s*H + d, t] is feature
 *   d of stream s); kv holds the projection of the column's n-gram rows: S keys of H features, then
 *   one value of H features (kv[s*H + d, t] is key s, kv[S*H + d, t] the value). For every column t
 *   and stream s independently:
 *     rx     = (mean_d R[s*H + d]^2 + eps)^(-1/2)
 *     rk     = (mean_d key_s[d]^2 + eps)^(-1/2)
 *     dot    = rx * rk * sum_d R[s*H + d] * w[s*H + d] * key_s[d] / sqrt(H)
 *     g      = sign(dot) * sqrt(max(|dot|, 1e-6))          (sign(0) = +1)
 *     R[s*H + d] <- R[s*H + d] + sigmoid(g) * value[d]
 *   w is the gate weight (the checkpoint's engram_gate: the product of the model's query and key
 *   norm weights), one row of H per stream.
 *
 * Logical shapes:
 *   residual BF16 [S*H, T] (updated in place); kv BF16 [(S + 1)*H, T]; w BF16 or FP32 with S*H
 *   elements, stream-major. S >= 1, H a positive multiple of 8 up to 16384, T >= 1 with S * T at most 2^31 - 1, eps > 0.
 *
 * Supported domain:
 *   Contiguous tensors; kv and residual 16-byte aligned, w 4-byte aligned; finite inputs.
 *
 * Numeric:
 *   FP32 arithmetic from the represented (BF16) inputs and one BF16 rounding of each output.
 * Against the FP64 oracle on the same inputs each output is within half a BF16 ulp of the exact
 * result plus the FP32 evaluation error: the gate's relative error is below (3 H + 40) 2^-24 (three
 * FP32 sums over H terms, then the square roots, sigmoid and exp, a few ulps), and the sigmoid's
 *   sensitivity to g is at most 1/4 per unit g, so the added term sigmoid(g) * value is off by well
 *   under 2^-18 |value| (H = 5120). A column's result does not depend on T or its position.
 *
 * Effects:
 *   Updates residual in place; kv and w are read only and must not overlap it.
 *
 * Workspace: none. Execution: stream-ordered; graph capturable.
 */
void engram_gate_inject(const Tensor& kv, const Tensor& w, std::int32_t streams, float eps,
                        Tensor& residual, cudaStream_t stream);

} // namespace ninfer::ops
