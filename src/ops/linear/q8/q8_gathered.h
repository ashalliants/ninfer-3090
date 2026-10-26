#pragma once

#include "core/tensor.h"
#include "core/weight.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops::detail {

// The exact stage of two_stage_head on the 27B output head [248320, 5120]: the route the dense
// head takes at this column count (select_q8_n248320_k5120), over the rows ids[0, *count) (all
// rows [0, *count) with null ids), each written to its row of `out` with the dense head's bits.
// Rows not listed are left alone. count is device memory, read at replay.
void launch_q8_gathered_n248320_k5120(const Tensor& x, const Weight& w, Tensor& out,
                                      const std::int32_t* ids, const std::int32_t* count,
                                      cudaStream_t stream);

} // namespace ninfer::ops::detail
