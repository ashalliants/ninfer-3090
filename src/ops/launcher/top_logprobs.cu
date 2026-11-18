// Implements: include/ninfer/ops/top_logprobs.h
// Match: wrapper-validated contiguous tensors, valid vocabulary rows and k in [1,20].
// Algorithm assumptions: one independent CTA per column; no global workspace.
#include "ops/launcher/top_logprobs.h"

#include "core/device.h"
#include "ops/kernel/top_logprobs.cuh"

namespace ninfer::ops::detail {

void top_logprobs_launch(const Tensor& logits, const Tensor& chosen_ids, std::int32_t valid_rows,
                         std::int32_t k, Tensor& chosen_logprobs, Tensor& top_ids,
                         Tensor& top_logprobs, cudaStream_t stream) {
    const auto columns = static_cast<unsigned int>(logits.ne[1]);
    top_logprobs_kernel<kTopLogprobsBlock><<<columns, kTopLogprobsBlock, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(logits.data),
        static_cast<const std::int32_t*>(chosen_ids.data),
        static_cast<float*>(chosen_logprobs.data), static_cast<std::int32_t*>(top_ids.data),
        static_cast<float*>(top_logprobs.data), valid_rows, logits.ne[0], k);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
