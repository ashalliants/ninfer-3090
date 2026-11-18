#pragma once

#include "core/device.h"
#include "core/tensor.h"

#include <cuda_runtime.h> // cudaStream_t

namespace ninfer::ops {

/**
 * Applies RMSNorm over the fastest dimension D=ne[0]. For each logical row r:
 *
 *   inv_r    = 1 / sqrt((1/D) * sum_d x[d,r]^2 + eps)
 *   gain[d]  = unit_offset ? 1 + weight[d] : weight[d]
 *   ideal[d,r] = x[d,r] * inv_r * gain[d].
 *
 * `x` and `out` are same-shaped contiguous BF16 tensors, weight is contiguous BF16 [D], and eps
 * is positive and finite. Input, weight, and output must not overlap. The oracle evaluates `ideal`
 * naively in FP64 from the represented inputs. The BF16 output is promoted and compared directly
 * with that result; output storage rounding belongs to the Op's numerical criterion, not the
 * oracle. Kernel reduction, staging, and accumulator precision are implementation choices. There
 * is no workspace or persistent state side effect.
 */
void rmsnorm(const Tensor& x, const Tensor& weight, float eps, bool unit_offset, Tensor& out,
             cudaStream_t stream);

/// rmsnorm(x1, w1, eps, unit_offset, out1) and rmsnorm(x2, w2, eps, unit_offset, out2), in one
/// launch where both take the same row kernel (unit offset, D = 256); the same bits. The four
/// operands of each norm follow rmsnorm's rules, and neither norm's output may overlap the other's
/// operands.
void rmsnorm_pair(const Tensor& x1, const Tensor& w1, Tensor& out1, const Tensor& x2, const Tensor& w2, Tensor& out2,
                  float eps, bool unit_offset, cudaStream_t stream);

/// rmsnorm_pair(x1, w1, out1, x2, w2, out2, eps, unit_offset = true) and then
/// rope(positions, rotary_dim, theta, out1, out2) (rope.h; out1/out2 [D, heads, T]), in one launch
/// where the norms take the D = 256 row kernel and the RoPE its generic kernel at rotary_dim 64;
/// the same bits as the two calls.
void rmsnorm_pair_rope(const Tensor& x1, const Tensor& w1, Tensor& out1, const Tensor& x2, const Tensor& w2,
                       Tensor& out2, float eps, const Tensor& positions, int rotary_dim, float theta,
                       DeviceExecutionView execution);

} // namespace ninfer::ops
