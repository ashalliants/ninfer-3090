#pragma once

// ninfer::ops::detail - private launch prototype for rmsnorm.

#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

// z selects the gated forms: SiLU(z), or sigmoid(z) when sigmoid_gate.
void rmsnorm_launch(const Tensor& x, const Tensor& weight, float eps, bool unit_offset,
                    const Tensor* z, bool sigmoid_gate, Tensor& out,
                    std::int32_t multiprocessor_count, cudaStream_t stream);

} // namespace ninfer::ops::detail
