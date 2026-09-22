#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::ops {

/**
 * Op: hadamard_transform
 *
 * Math / indexing:
 *   For every column t and every 1024-block b of the fastest axis, with i, j in [0,1024):
 *     forward: out[b*1024+i, t] = 2^-5 * sum_j H[i][j] * signs[b*1024+j] * x[b*1024+j, t]
 *     inverse: out[b*1024+i, t] = 2^-5 * signs[b*1024+i] * sum_j H[i][j] * x[b*1024+j, t]
 *   where H[i][j] = (-1)^popcount(i & j) is the order-1024 Sylvester Walsh-Hadamard matrix.
 *   The forward form is the activation-side transform of a Hadamard-rotated checkpoint
 *   (`y = W_r · FWHT(s ⊙ a)`); the inverse form restores a rotated table row to the primal
 *   basis (`e = s ⊙ FWHT(z)`). H is symmetric and 2^-5 H is orthonormal, so the two forms are
 *   mutually inverse up to rounding.
 *
 * Logical shapes / supported domain:
 *   x and out are same-shaped contiguous BF16 tensors whose fastest extent K = ne[0] is a
 *   positive multiple of 1024; every other extent is a batch axis (T = numel / K columns).
 *   signs is contiguous BF16 [K] holding +1 or -1. The registered checkpoint uses
 *   K in {5120, 6144, 17408}; the Op accepts every positive multiple of 1024. x, signs and out
 *   are 16-byte aligned (the lanes load and store eight elements at a time).
 *
 * Numeric:
 *   The oracle evaluates the definition naively in FP64 from the represented BF16 inputs. The
 *   BF16 output is promoted and compared directly with that result; output storage rounding
 *   belongs to the Op's named reduction criterion, not the oracle. Butterfly association and
 *   accumulator precision are implementation choices.
 *
 * Effects:
 *   Writes all of out. out may alias x exactly (in place); otherwise the two must not overlap.
 *   x (when not aliased) and signs are preserved.
 *
 * Workspace:
 *   None. No persistent state side effect.
 */
void hadamard_transform(const Tensor& x, const Tensor& signs, bool inverse, Tensor& out,
                        cudaStream_t stream);

/**
 * Op: silu_mul_hadamard
 *
 * Math / indexing:
 *   With gate = plane[0:I, t] and up = plane[I:2I, t] for every column t, and a = the BF16
 *   rounding of silu(gate) * up (silu(x) = x / (1 + e^-x), exact fp32):
 *     out[b*1024+i, t] = 2^-5 * sum_j H[i][j] * signs[b*1024+j] * a[b*1024+j, t]
 *   i.e. the forward hadamard_transform of silu_mul's output, the down-projection input of a
 *   Hadamard-rotated checkpoint, in one pass over the SwiGLU plane.
 *
 * Logical shapes / supported domain:
 *   plane is contiguous BF16 [2*I, T...], out contiguous BF16 [I, T...] with I = out.ne[0] a
 *   positive multiple of 1024 and the batch extents equal; signs is contiguous BF16 [I] of
 *   +1 / -1. All three are 16-byte aligned. The registered checkpoint uses I = 17408.
 *
 * Numeric:
 *   The oracle is the composition's definition in FP64 from the represented BF16 inputs with the
 *   intermediate SwiGLU value rounded to BF16; the same reduction criterion as hadamard_transform.
 *
 * Effects:
 *   Writes all of out; out must not overlap the plane. plane and signs are preserved.
 *
 * Workspace:
 *   None. No persistent state side effect.
 */
void silu_mul_hadamard(const Tensor& plane, const Tensor& signs, Tensor& out, cudaStream_t stream);

} // namespace ninfer::ops
