#pragma once

// ninfer::ops::detail - private launch prototype for direction_project. Included by the wrapper
// (host) and defined by the launcher (.cu). Not part of the public api.
// See docs/maintainer/op-development.md §2.

#include "core/tensor.h"

#include <cstdint>
#include <cuda_runtime.h>

namespace ninfer::ops::detail {

// Host entry; assumes inputs already validated by the wrapper.
void direction_project_launch(Tensor& residual, std::int32_t streams, const float* directions,
                              std::int64_t set_stride, const std::int32_t* row_select,
                              cudaStream_t stream);

} // namespace ninfer::ops::detail
