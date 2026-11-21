#pragma once

// Written for NInfer-3090. The causal short convolution of Inkling-2-Mini's K, V, attention-output and
// MoE-output streams (docs/maintainer/inkling2-model.md 3e, 3g and 3j), with its persistent history.

#include "core/tensor.h"

#include <cuda_runtime.h>

#include <span>

namespace ninfer::ops {

/**
 * Op: short_conv_fp32 (depthwise causal convolution plus the input, scaled; optional per-head dynamic taps)
 *
 * Math / indexing:
 *   Let z_{t-l} be x[:, t-l] for t-l >= 0 and history[:, K-1-(l-t)] otherwise (the state holds the last K-1
 *   inputs of the previous call, oldest first: history[:, 0] is the oldest).
 *   static:   out[c, t] = ( x[c, t] + sum_{l=0..K-1} taps[K-1-l, c] * z_{t-l}[c] ) / sqrt(K + 1)
 *   dynamic:  out[c, t] = ( x[c, t] + sum_l (taps[K-1-l, c] + dynamic[(c / head_width) * K + l, t]) * z_{t-l}[c] )
 *                         / sqrt(2K + 1)
 *   new history[:, i] = the K-1 latest inputs z_{T-K+1+i} (i = 0..K-2).
 *
 * Logical shapes:
 *   x FP32 [C, T]; taps FP32 [K, C] contiguous (tap K-1 multiplies the current input); dynamic FP32
 *   [(C / head_width) * K, T] or null; history FP32 [C, K-1] contiguous; out FP32 [C, T].
 *
 * Supported domain:
 *   2 <= K <= 8; C a multiple of head_width when dynamic; columns at any stride.
 *
 * Numeric: FP32 multiply-adds; oracle FP64 (inkling2_reference.h short_conv). History stays FP32.
 *
 * Effects: writes out and replaces history. out must not overlap x, dynamic or history.
 * Workspace: none. Execution: stream-ordered; graph capturable.
 */
void short_conv_fp32(const Tensor& x, const Tensor& taps, const Tensor* dynamic, int head_width, Tensor& history,
                     Tensor& out, cudaStream_t stream);

/// One short_conv_fp32 of a batch, optionally also writing sum = residual + out (out may then be null).
struct ShortConvJob {
    const Tensor* x        = nullptr;
    const Tensor* taps     = nullptr;
    const Tensor* dynamic  = nullptr;
    Tensor* history        = nullptr;
    Tensor* out            = nullptr;
    const Tensor* residual = nullptr; // FP32 [C, T], with sum
    Tensor* sum            = nullptr; // FP32 [C, T]: residual + the convolution, the same bits as add_fp32
};
inline constexpr int kShortConvMaxJobs = 4;

/**
 * Op: short_conv_fp32_batch: 1 to kShortConvMaxJobs short_conv_fp32 calls in one launch (each job's out
 * the same bits as its own call), each optionally fused with the residual add that follows it. Every job
 * has the same C, T, K and dynamic-ness (and head_width); sum must not overlap x, dynamic, history or out,
 * and no job's outputs may overlap another job's operands.
 */
void short_conv_fp32_batch(std::span<const ShortConvJob> jobs, int head_width, cudaStream_t stream);

} // namespace ninfer::ops
