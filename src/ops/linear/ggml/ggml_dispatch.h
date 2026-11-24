#pragma once

// Private dispatch of the GGML block-format Linear and LinearAdd (contracts in
// include/ninfer/ops/linear.h and linear_add.h). The wrappers call these after their own
// tensor checks.

#include "core/arena.h"
#include "core/tensor.h"
#include "core/weight.h"
#include "ninfer/ops/linear.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::detail {

// Throws unless (format, n, k) is a registered problem of the Op (LinearAdd registers a subset)
// and the policy admits the Q8_1 (A8) profile, the only one GGML formats register.
void ggml_require_registered(QType format, std::int32_t n, std::int32_t k, bool linear_add,
                             bool a8_admitted);

// The staged decode route's workspace for calls of up to max_tokens (<= 8) columns: x cast once
// for the GEMV to read (ggml_launch.h). Nonzero only where that route runs: sm_70 builds (unless
// NINFER_GGML_STAGED=0), K = 2560 or 6144, a format other than Q4_K / Q5_K, max_tokens >= 2.
std::size_t ggml_decode_staging_bytes(QType format, std::int32_t k, std::int32_t max_tokens);

// Caller workspace for every T in [min_tokens, max_tokens]: the staged decode route's staging
// (ggml_decode_staging_bytes, often zero) when the interval stays on the decode route, otherwise
// the Q8_1 activation of max_tokens columns.
std::size_t ggml_linear_workspace_capacity_bytes(QType format, std::int32_t n, std::int32_t k,
                                                 bool linear_add, std::int32_t min_tokens,
                                                 std::int32_t max_tokens);

// Validates the weight view and runs out = W x (residual == false) or out += W x. `decode_mma`
// (LinearPolicy::AllowA8GgmlDecodeMma) takes T <= 8 to the int8 tensor-core decode route, which
// also needs a 16-byte aligned weight; wider calls are unchanged.
void ggml_linear_dispatch(const Tensor& x, const Weight& w, Tensor& out, bool residual, bool decode_mma,
                          WorkspaceArena* workspace, cudaStream_t stream);

// linear_pair over GGML weights: validates both weights, the registered pair problem and
// 1 <= T <= kGgmlDecodeMaxTokens, then runs first_out = W1 x and second_out = W2 x in one launch.
void ggml_pair_dispatch(const Tensor& x, const Weight& first, const Weight& second, Tensor& first_out,
                        Tensor& second_out, cudaStream_t stream);

} // namespace ninfer::ops::detail
