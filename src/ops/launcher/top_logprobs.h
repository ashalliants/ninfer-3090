#pragma once

// ninfer::ops::detail - private launch prototype for top_logprobs.

#include "core/tensor.h"

#include <cstdint>

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

void top_logprobs_launch(const Tensor& logits, const Tensor& chosen_ids, std::int32_t valid_rows,
                         std::int32_t k, Tensor& chosen_logprobs, Tensor& top_ids,
                         Tensor& top_logprobs, cudaStream_t stream);

} // namespace ninfer::ops::detail
