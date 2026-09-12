#pragma once

#include "core/arena.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::detail {

void q4_linear_swiglu_gemv_pair_launch(const Tensor& x, const Weight& w, Tensor& out,
                                       cudaStream_t stream);
void q4_linear_swiglu_mma_split_half_pair_r32_c128_launch(const Tensor& x, const Weight& w,
                                                          Tensor& out, cudaStream_t stream);
void q4_linear_swiglu_mma_split_half_pair_r32_c40_launch(const Tensor& x, const Weight& w,
                                                         Tensor& out, cudaStream_t stream);
void q4_linear_swiglu_mma_split_half_pair_r32_c48_launch(const Tensor& x, const Weight& w,
                                                         Tensor& out, cudaStream_t stream);
void q4_linear_swiglu_small_t_tiled_launch(const Tensor& x, const Weight& w, Tensor& out,
                                           cudaStream_t stream);

// int8 tensor-core small-T route, T=32 only for this first pass. Not routed by
// q4_linear_swiglu_resolve_plan; reachable only via q4_linear_swiglu_execute_schedule for
// exploratory measurement (bench/ops/q4_linear_swiglu_schedule_bench.cu) and its correctness test,
// until it is measured against small_t_tiled and either wins or is deleted.
[[nodiscard]] std::size_t q4_linear_swiglu_small_t_tiled_i8_workspace_bytes(std::int32_t tokens);
void q4_linear_swiglu_small_t_tiled_i8_launch(const Tensor& x, const Weight& w, Tensor& out,
                                              WorkspaceArena& workspace, cudaStream_t stream);

} // namespace ninfer::ops::detail
