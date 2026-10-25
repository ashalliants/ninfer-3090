#pragma once

// ninfer::ops::detail - private launch prototypes for argmax_probability.

#include "core/arena.h"
#include "core/tensor.h"

#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

// The scratch of one call over `columns` columns (the tile partials), from a 256-byte-aligned cursor.
[[nodiscard]] std::size_t argmax_probability_workspace_bytes(std::int32_t valid_rows, std::int32_t columns);

void argmax_probability_launch(const Tensor& logits, Tensor& ids, Tensor& probs, std::int32_t valid_rows,
                               WorkspaceArena& workspace, cudaStream_t stream);

} // namespace ninfer::ops::detail
