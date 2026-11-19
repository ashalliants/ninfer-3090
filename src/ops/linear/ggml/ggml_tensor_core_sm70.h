#pragma once
// sm_70 opt-in route of the GGML linear on the FP16 tensor cores (ggml_tensor_core_sm70.cu).
#include "core/arena.h"
#include "core/tensor.h"
#include "core/weight.h"

#include <cuda_runtime.h>

namespace ninfer::ops::detail {
// Runs the linear (out = x W, or out += x W when `residual`) and returns true when the route applies
// (sm_70 build, NINFER_SM70_TENSOR_CORES=1, sm70_tensor_core_admits, room in `workspace`); otherwise returns false.
bool ggml_tensor_core_linear(const Tensor& x, const Weight& w, Tensor& out, bool residual, WorkspaceArena* workspace,
                             cudaStream_t stream);
} // namespace ninfer::ops::detail
