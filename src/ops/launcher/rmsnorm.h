#pragma once

// ninfer::ops::detail - private launch prototype for rmsnorm.

#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

// z selects the gated forms: SiLU(z), or sigmoid(z) when sigmoid_gate.
void rmsnorm_launch(const Tensor& x, const Tensor& weight, float eps, bool unit_offset,
                    const Tensor* z, bool sigmoid_gate, Tensor& out,
                    std::int32_t multiprocessor_count, cudaStream_t stream);
// Both unit-offset norms in one launch when both are D = 256 and 4-byte aligned (the route
// rmsnorm_launch takes for each); false, launching nothing, otherwise.
// rmsnorm_pair_launch and then the generic RoPE (rotary_dim 64) of out1/out2 ([256, heads, T],
// contiguous) in one launch; false, launching nothing, unless both are D = 256 and aligned.
bool rmsnorm_pair_rope_launch(const Tensor& x1, const Tensor& w1, Tensor& out1, const Tensor& x2, const Tensor& w2,
                              Tensor& out2, float eps, const Tensor& positions, float theta, cudaStream_t stream);
bool rmsnorm_pair_launch(const Tensor& x1, const Tensor& w1, Tensor& out1, const Tensor& x2, const Tensor& w2,
                         Tensor& out2, float eps, cudaStream_t stream);

} // namespace ninfer::ops::detail
