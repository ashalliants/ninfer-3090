#pragma once

// Integer-activation prefill route for T2G128 row-split weights on sm_86: the shared int8 GEMM of
// ops/common/rowsplit_a8_mma.cuh with the ternary codec (the codes are exact int8 {-1, 0, +1}; the
// activations are quantised to int8 with one scale per token and 64-wide group).

#include "core/arena.h"
#include "core/tensor.h"
#include "core/weight.h"
#include "ninfer/ops/linear.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::detail {

[[nodiscard]] bool t2_a8_admits(LinearPolicy policy);
[[nodiscard]] bool t2_a8_shape_supported(std::int32_t output_rows, std::int32_t input_rows);
[[nodiscard]] bool t2_a8_supported(const Weight& w, std::int32_t tokens);
// Activation planes for T in [1, max_tokens]; zero when no admitted call can take the route.
[[nodiscard]] std::size_t t2_a8_workspace_bytes(std::int32_t output_rows, std::int32_t input_rows,
                                                LinearPolicy policy, std::int32_t max_tokens);

void t2_a8_linear(const Tensor& x, const Weight& w, Tensor& out, WorkspaceArena& workspace,
                  cudaStream_t stream);
void t2_a8_linear_add(const Tensor& x, const Weight& w, Tensor& residual, WorkspaceArena& workspace,
                      cudaStream_t stream);

} // namespace ninfer::ops::detail
