#pragma once
// sm_70 opt-in: Q4/Q5/Q6/Q8 (ninfer row-split group formats) A16 linears of 64+ tokens on the FP16
// tensor cores (ops/linear/common/sm70_tensor_core.h). Each weight value (code x group scale, exact
// in FP32) rounds once to FP16 under its row's power of two. Only A16Only calls take it: their
// default prefill kernels are Ampere BF16 mma.sync, emulated on Volta (the vision tower's linears).
#include "core/arena.h"
#include "core/tensor.h"
#include "core/weight.h"
#include "ninfer/ops/linear.h"

#include <cuda_runtime.h>

#include <cstddef>

namespace ninfer::ops::detail {

// Whether a call with this policy and shape would take the route (ignoring the arena).
bool rowsplit_tensor_core_admits(QType qtype, LinearPolicy policy, int rows, int k, int tokens) noexcept;

// The arena bytes the widest call of `max_tokens` takes (0 when not admitted).
std::size_t rowsplit_tensor_core_workspace_bytes(QType qtype, LinearPolicy policy, int rows, int k,
                                                 int max_tokens) noexcept;

// Runs the call and returns true, or returns false (nothing launched) when not admitted or the
// arena lacks room.
bool rowsplit_tensor_core_linear(const Tensor& x, const Weight& w, Tensor& out, LinearPolicy policy,
                                 WorkspaceArena* workspace, cudaStream_t stream);

} // namespace ninfer::ops::detail
