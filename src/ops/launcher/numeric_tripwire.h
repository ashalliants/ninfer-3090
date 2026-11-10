#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops::detail {

void numeric_tripwire_scan_launch(const Tensor& values, std::uint32_t site, float abs_limit,
                                  Tensor& record, cudaStream_t stream);

} // namespace ninfer::ops::detail
