#pragma once

// Rotary embedding in the layout DeepSeek-V4 uses: adjacent element pairs rotate as complex numbers
// (x[2i] + i x[2i+1]), only the LAST `rotary_dim` elements of each head, with a caller-supplied
// frequency table (V4's layers differ: YaRN where a layer compresses, plain frequencies where it
// does not) and an inverse mode (the attention output has its rotation removed before the output
// projection). `apply_rotary_emb` in the model repo (inference/model.py); the NumPy oracle is
// tools/dsv4/attention.py `apply_rope`. rope.h rotates split halves of a fixed set of head
// geometries and does not serve it.

#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::ops {

/**
 * Op: rope_pairs
 *
 * Math / indexing:
 *   For every column t, head h and pair i in [0, R/2), with R = rotary_dim, base = D - R and
 *   angle = positions[t] * inv_freq[i] (evaluated in FP64):
 *     z      = x[base + 2i, h, t] + i * x[base + 2i + 1, h, t]
 *     z'     = z * (cos(angle) + i * sin(angle))                 forward
 *     z'     = z * (cos(angle) - i * sin(angle))                 inverse
 *   x[:base, h, t] is unchanged.
 *
 * Logical shapes:
 *   x BF16 [D, H, T] (updated in place); positions I32 [T]; inv_freq FP32 [R/2]. R even,
 *   2 <= R <= D, D even and <= 1024 (each pair is read as one 4-byte unit); H >= 1; T >= 1; positions >= 0.
 *
 * Supported domain:
 *   Contiguous tensors with 4-byte aligned data; finite inv_freq >= 0.
 *
 * Numeric:
 *   cos and sin of the FP64 angle rounded to FP32; the rotation in FP32; one BF16 rounding of the
 *   result (so the error against the FP64 oracle is half a BF16 ulp plus FP32 rounding). Exact for
 *   the unrotated elements.
 *
 * Effects:
 *   Updates x in place. positions and inv_freq are read only and must not overlap x.
 *
 * Workspace: none. Execution: stream-ordered; graph capturable.
 */
void rope_pairs(const Tensor& positions, const Tensor& inv_freq, int rotary_dim, bool inverse, Tensor& x,
                cudaStream_t stream);

} // namespace ninfer::ops
