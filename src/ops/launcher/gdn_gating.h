#pragma once

// ninfer::ops::detail - private launch prototype for gdn_gating.

#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

// decay: `rate` holds the decay itself rather than A_log (gdn_gating_decay).
void gdn_gating_launch(const Tensor& a, const Tensor& b, const Tensor& rate, const Tensor& dt_bias,
                       bool decay, Tensor& g, Tensor& beta, cudaStream_t stream);

} // namespace ninfer::ops::detail
