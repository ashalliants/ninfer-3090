#pragma once

// ninfer::ops::detail - private launch prototype for hadamard_transform.

#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

void hadamard_transform_launch(const Tensor& x, const Tensor& signs, bool inverse, Tensor& out,
                               cudaStream_t stream);
void silu_mul_hadamard_launch(const Tensor& plane, const Tensor& signs, Tensor& out,
                              cudaStream_t stream);

} // namespace ninfer::ops::detail
