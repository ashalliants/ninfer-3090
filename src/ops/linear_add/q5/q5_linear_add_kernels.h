#pragma once

#include "core/weight.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

// The streaming GEMV (ops/linear/common/stream_gemv.cuh) for T = 1..8 when it is enabled (sm_70 by
// default); false when it declines, and the caller launches its own schedule.
bool try_q5_linear_add_stream(const Tensor& x, const Weight& w, Tensor& residual_out,
                              cudaStream_t stream);
void q5_linear_add_split2_exact_launch(const Tensor& x, const Weight& w, Tensor& residual_out,
                                       cudaStream_t stream);
void q5_linear_add_small_t_mma_launch(const Tensor& x, const Weight& w, Tensor& residual_out,
                                      cudaStream_t stream);
void q5_linear_add_mma_r64_c16_launch(const Tensor& x, const Weight& w, Tensor& residual_out,
                                      cudaStream_t stream);
void q5_linear_add_mma_r64_c24_launch(const Tensor& x, const Weight& w, Tensor& residual_out,
                                      cudaStream_t stream);
void q5_linear_add_mma_r64_c32_launch(const Tensor& x, const Weight& w, Tensor& residual_out,
                                      cudaStream_t stream);
void q5_linear_add_mma_r64_c64_launch(const Tensor& x, const Weight& w, Tensor& residual_out,
                                      cudaStream_t stream);
void q5_linear_add_mma_r64_c32_s3_launch(const Tensor& x, const Weight& w, Tensor& residual_out,
                                         cudaStream_t stream);
void q5_linear_add_mma_r64_c32_s4_launch(const Tensor& x, const Weight& w, Tensor& residual_out,
                                         cudaStream_t stream);
void q5_linear_add_mma_r64_c128_launch(const Tensor& x, const Weight& w, Tensor& residual_out,
                                       cudaStream_t stream);

void q5_linear_add_sliced_r16_t8_launch(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void q5_linear_add_sliced_r16_t16_launch(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void q5_linear_add_sliced_r16_t24_launch(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void q5_linear_add_sliced_r32_t32_w4_s2_launch(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void q5_linear_add_sliced_r32_t24_pairwise_launch(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void q5_linear_add_sliced_r32_t32_w4_s1_launch(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void q5_linear_add_sliced_r32_t32_w2_s2_launch(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void q5_linear_add_sliced_r32_t64_w2_s1_launch(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void q5_linear_add_mma_r32_t32_k128_launch(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void q5_linear_add_mma_r32_t128_launch(const Tensor&, const Weight&, Tensor&, cudaStream_t);

} // namespace ninfer::ops::detail
