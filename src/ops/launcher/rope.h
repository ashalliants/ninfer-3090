#pragma once

// ninfer::ops::detail - private launch prototype for rope. Included by the wrapper
// and defined by the CUDA launcher.

#include "core/device.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

// Whether rope_launch takes the generic kernel (rope_generic_kernel) for this pair.
bool rope_pair_generic_route(const Tensor& positions, int rotary_dim, float theta, const Tensor& q, const Tensor& k);
void rope_launch(const Tensor& positions, int rotary_dim, float theta, Tensor& q, Tensor& k,
                 DeviceExecutionView execution);

void rope_single_launch(const Tensor& positions, int rotary_dim, float theta, Tensor& x,
                        DeviceExecutionView execution);

} // namespace ninfer::ops::detail
