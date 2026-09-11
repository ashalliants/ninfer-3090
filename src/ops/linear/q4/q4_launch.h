#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops::detail {

using Q4Launch = void (*)(const Tensor&, const Weight&, Tensor&, cudaStream_t);

// Row counts a leading prefix of the 27B's 131072-row draft head (text/draft_head, K = 5120) may
// take. Its rows are ordered by token frequency, so a prefix is the N most frequent proposal
// tokens; multiples of 8192 keep every row tile of every schedule whole.
inline constexpr bool q4_draft_head_prefix_rows(std::int32_t n) noexcept {
    return n >= 8192 && n < 131072 && (n % 8192) == 0;
}

void launch_q4_gemv_r4_w1_direct(const Tensor& x, const Weight& w, Tensor& out,
                                 cudaStream_t stream);
void launch_q4_gemv_r1_w8_direct(const Tensor& x, const Weight& w, Tensor& out,
                                 cudaStream_t stream);
void launch_q4_simt_r8_c4(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream);
void launch_q4_simt_r8_c8(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream);
void launch_q4_draft_head_small_t(const Tensor& x, const Weight& w, Tensor& out,
                                  cudaStream_t stream);
void launch_q4_mma_r64_c32(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream);
void launch_q4_mma_r64_c48(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream);
void launch_q4_mma_r64_c56(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream);
void launch_q4_mma_r64_c64(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream);
void launch_q4_mma_r64_c64_endpoint(const Tensor& x, const Weight& w, Tensor& out,
                                    cudaStream_t stream);
void launch_q4_mma_r64_c72(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream);
void launch_q4_mma_r64_c80(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream);
void launch_q4_mma_r64_c96(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream);
void launch_q4_mma_r64_c104_bounded(const Tensor& x, const Weight& w, Tensor& out,
                                    cudaStream_t stream);
void launch_q4_mma_r64_c112_partial(const Tensor& x, const Weight& w, Tensor& out,
                                    cudaStream_t stream);
void launch_q4_mma_r64_c112(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream);
void launch_q4_mma_r64_c120_partial(const Tensor& x, const Weight& w, Tensor& out,
                                    cudaStream_t stream);
void launch_q4_mma_r64_c120(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream);
void launch_q4_mma_r64_c128(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream);

} // namespace ninfer::ops::detail
