#pragma once
// sm_70: a plain Q4 / Q5 linear of 2..kStreamGemvMaxTokens tokens on the streaming GEMV
// (stream_gemv.cuh; its tile GEMM past eight tokens). The shape tables' small-T routes are
// Ampere mma.sync kernels, emulated and compute-bound on Volta: the Q4 MTP proposal head
// (131072 x 5120) at two lanes ran q4_ksplit_mma at 2.8 ms a call against ~0.75 ms streamed.
// Every other target, and any call the streaming GEMV declines, keeps the tables.
#include "core/tensor.h"
#include "core/weight.h"

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

bool try_launch_q4_stream_tokens(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream);
bool try_launch_q5_stream_tokens(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream);

} // namespace ninfer::ops::detail
