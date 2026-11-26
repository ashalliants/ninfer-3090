#pragma once

#include "core/weight.h"
#include "core/arena.h"
#include "core/tensor.h"
#include "ninfer/ops/linear.h"
#include "ops/linear/nvfp4/nvfp4_a4_plan.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::detail {

[[nodiscard]] std::size_t nvfp4_gdn_input_workspace_capacity_bytes(LinearPolicy policy,
                                                                   std::int32_t min_tokens,
                                                                   std::int32_t max_tokens);

void nvfp4_gdn_input_a16_launch(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                                cudaStream_t stream);

// sm_70: 1..64 tokens as the dp4a / tensor-core streaming passes (nvfp4_stream_format.cuh). False
// where that route does not apply, and the caller runs its own plan.
// Whether a `tokens`-wide NVFP4 call on `stream` takes the streaming route (dp4a to 8 tokens, the
// tensor-core tiles to 64): sm_70 under EngineOptions::verify_a8. The GDN conv forms then compose
// that projection with the separate conv rather than run their fused small-T kernels.
bool nvfp4_stream_admits(std::int32_t tokens, cudaStream_t stream);
// sm_70 composes the A16 small-T GDN conv forms (projection, then the conv) rather than run the
// fused kernel, whose mma is emulated there.
#if defined(NINFER_SM70)
inline constexpr bool kNvfp4ComposesSmallT = true;
#else
inline constexpr bool kNvfp4ComposesSmallT = false;
#endif
bool nvfp4_gdn_input_stream_launch(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                                   cudaStream_t stream);
void nvfp4_gdn_input_decode_launch(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                                   cudaStream_t stream);

void nvfp4_gdn_input_small_t_launch(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                                    cudaStream_t stream);

void nvfp4_gdn_input_a4_launch(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                               Nvfp4A4Workspace workspace, cudaStream_t stream);

void nvfp4_gdn_input_dispatch(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                              LinearPolicy policy, WorkspaceArena* workspace, cudaStream_t stream);

} // namespace ninfer::ops::detail
