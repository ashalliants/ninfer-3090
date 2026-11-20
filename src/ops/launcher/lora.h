#pragma once

// ninfer::ops::detail - private launch prototype for lora_apply. Included by the wrapper (host)
// and defined by the launcher (.cu). Not part of the public api.
// See docs/maintainer/op-development.md §2.

#include "ninfer/ops/lora.h"

#include <cstdint>
#include <span>
#include <cuda_runtime.h>

namespace ninfer::ops::detail {

// The K-split count of the shrink stage for an input of `k` values.
[[nodiscard]] std::int32_t lora_splits(std::int32_t k) noexcept;

// Host entry; assumes inputs already validated by the wrapper. `partial` holds
// lora_splits(k) * rank * T FP32 values.
void lora_apply_launch(const Tensor& x, const LoraDown& down, std::span<const LoraUp> up,
                       const LoraColumns& columns, float* partial, cudaStream_t stream);

} // namespace ninfer::ops::detail
