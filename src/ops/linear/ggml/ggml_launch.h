#pragma once

// Private launchers of the GGML block-format linears (src/ops/linear/ggml). Not a contract: the
// wrappers in linear.cpp and wrapper/linear_add.cpp validate and dispatch through ggml_dispatch.h.

#include "core/weight.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops::detail {

// Decode route, 1 <= t <= kGgmlDecodeMaxTokens: one kernel that applies the Q8_1 cast to x in
// shared memory and runs a dp4a GEMV over the weight rows. `residual` selects the LinearAdd
// epilogue (out += W x); otherwise out = W x.
template <QType F>
void ggml_mmvq_launch(const std::uint8_t* w, const __nv_bfloat16* x, __nv_bfloat16* out,
                      std::int32_t n, std::int32_t k, std::int32_t t, bool residual,
                      cudaStream_t stream);

// Wide route: int8 tensor-core GEMM over the Q8_1 activation in (xq, xd), written by
// ggml_quantize_q8_1_launch. Requires n % 128 == 0 and k % 128 == 0.
template <QType F>
void ggml_mmq_launch(const std::uint8_t* w, const std::int8_t* xq, const float* xd,
                     __nv_bfloat16* out, std::int32_t n, std::int32_t k, std::int32_t t,
                     bool residual, cudaStream_t stream);

// The Q8_1 cast of a BF16 [K,T] activation: xq int8 [T][K] (row t at xq + t k) and one FP32 scale
// per 32 values, xd [T][K / 32]. k % 32 == 0.
void ggml_quantize_q8_1_launch(const __nv_bfloat16* x, std::int32_t k, std::int32_t t,
                               std::int8_t* xq, float* xd, cudaStream_t stream);

} // namespace ninfer::ops::detail
