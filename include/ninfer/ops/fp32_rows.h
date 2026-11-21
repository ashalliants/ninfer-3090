#pragma once

// Written for NInfer-3090. FP32 column primitives for models whose residual state is FP32 (Inkling-2-Mini,
// docs/maintainer/inkling2-model.md). Every tensor is [D, T] with D = ne[0] contiguous and the columns at
// any byte stride nb[1] (a row slice of a wider tensor is accepted), so a projection's concatenated output
// can be consumed in place.

#include "core/tensor.h"

#include <cuda_runtime.h>

#include <span>

namespace ninfer::ops {

/**
 * Op: rmsnorm_fp32 (RMS normalization of FP32 columns, optional FP32 gain)
 *
 * Math / indexing:
 *   out[d, t] = x[d, t] / sqrt(mean_d x[d, t]^2 + eps) * (weight ? weight[d] : 1)
 *
 * Logical shapes:
 *   x FP32 [D, T]; weight FP32 [D] or null; out FP32 or BF16 [D, T]. D >= 1, T >= 1.
 *
 * Supported domain:
 *   ne[0] contiguous in every operand, columns at any stride; D at most 65536.
 *
 * Numeric:
 *   FP32 sum of squares; oracle FP64. A BF16 output is the round to nearest even of the FP32 result.
 *
 * Effects: writes out; out may equal x (same view, FP32) but must not partially overlap it.
 * Workspace: none. Execution: stream-ordered; graph capturable.
 */
void rmsnorm_fp32(const Tensor& x, const Tensor* weight, float eps, Tensor& out, cudaStream_t stream);

/// One rmsnorm_fp32 of a batch.
struct RmsNormJob {
    const Tensor* x      = nullptr;
    const Tensor* weight = nullptr;
    Tensor* out          = nullptr;
};
inline constexpr int kRmsNormMaxJobs = 4;

/**
 * Op: rmsnorm_fp32_batch: 1 to kRmsNormMaxJobs rmsnorm_fp32 calls in one launch, each job's result the
 * same bits as its own call. Every job has the same D and T and one output dtype; the in-place rule of
 * rmsnorm_fp32 holds per job, and no job's out may overlap another job's operands.
 */
void rmsnorm_fp32_batch(std::span<const RmsNormJob> jobs, float eps, cudaStream_t stream);

/**
 * Op: add_fp32: out[d, t] = a[d, t] + b[d, t]. FP32 [D, T]; out may equal a or b. No workspace.
 */
void add_fp32(const Tensor& a, const Tensor& b, Tensor& out, cudaStream_t stream);

/**
 * Op: sigmoid_gate_fp32: out[d, t] = x[d, t] / (1 + exp(-gate[d, t])), written as FP32 or BF16 (round to
 * nearest even). x, gate FP32 [D, T]; out may equal x when FP32. No workspace.
 */
void sigmoid_gate_fp32(const Tensor& x, const Tensor& gate, Tensor& out, cudaStream_t stream);

/**
 * Op: tanh_softcap_fp32: out[d, t] = cap * tanh(x[d, t] / cap). FP32 [D, T], cap > 0; out may equal x.
 */
void tanh_softcap_fp32(const Tensor& x, float cap, Tensor& out, cudaStream_t stream);

/**
 * Op: embedding_bf16_fp32 (row gather): out[d, t] = FP32(table[d, ids[t]]).
 *
 * ids I32 [T], every id in [0, V); table BF16 [D, V] contiguous; out FP32 [D, T]. An id out of range is
 * not checked on the device (the caller's tokens are validated on the host). No workspace.
 */
void embedding_bf16_fp32(const Tensor& ids, const Tensor& table, Tensor& out, cudaStream_t stream);

} // namespace ninfer::ops
