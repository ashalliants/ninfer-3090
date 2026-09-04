#include "ops/attn_input_proj/nvfp4/nvfp4_attn_input_plan.h"
#include "ops/gdn_input_proj/nvfp4/nvfp4_gdn_input_plan.h"
#include "ops/kv_cache/append/launch.h"
#include "ops/linear/nvfp4/nvfp4_w4a4_plan.h"
#include "ops/linear_add/nvfp4/nvfp4_linear_add_plan.h"
#include "ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_plan.h"
#include "ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_w4a4_tma_launch.h"
#include "ops/softmax_attention/dense/causal_cache/launch.h"

#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

[[noreturn]] void reject_nvfp4_a4() {
    throw std::runtime_error("NVFP4 A4 execution requires an sm_120a GPU");
}

// NVFP4's causal-attention *prompt* (long-context prefill) kernel is warp-specialized
// (setmaxnreg.{dec,inc}.sync.aligned dynamic producer/consumer register reallocation), a genuine
// Hopper+ hardware feature with no sm_86/sm_89 fallback -- unlike the value codec, this is not a
// swap-one-instruction port. Its decode kernel (small_t, T<=6) has no such dependency and is
// compiled for real; only the prompt path is stubbed here.
[[noreturn]] void reject_nvfp4_prompt_kv() {
    throw std::runtime_error(
        "NVFP4 KV-cache prompt (long-context) attention requires an sm_100a or sm_120a GPU; use "
        "--kv-dtype bf16, int8, or rk8v4 for prompts longer than 6 tokens");
}

} // namespace

void launch_nvfp4_w4a4_quantize(const Tensor&, const Weight&, Nvfp4W4a4Workspace, cudaStream_t) {
    reject_nvfp4_a4();
}

void launch_nvfp4_w4a4(const Tensor&, const Weight&, Tensor&, Nvfp4W4a4Workspace, cudaStream_t) {
    reject_nvfp4_a4();
}

void nvfp4_linear_swiglu_w4a4_launch(const Tensor&, const Weight&, Tensor&, WorkspaceArena&,
                                     cudaStream_t) {
    reject_nvfp4_a4();
}

void launch_nvfp4_linear_swiglu_w4a4_tma(const std::uint8_t*, const std::uint8_t*,
                                         const std::uint8_t*, const std::uint8_t*,
                                         __nv_bfloat16*, std::int32_t, float, cudaStream_t) {
    reject_nvfp4_a4();
}

void nvfp4_linear_add_w4a4_launch(const Tensor&, const Weight&, Tensor&, Nvfp4W4a4Workspace,
                                  cudaStream_t) {
    reject_nvfp4_a4();
}

void nvfp4_attn_input_w4a4_launch(const Tensor&, const Weight&, Tensor&, Tensor&, Tensor&, Tensor&,
                                  Nvfp4W4a4Workspace, cudaStream_t) {
    reject_nvfp4_a4();
}

void nvfp4_gdn_input_w4a4_launch(const Tensor&, const Weight&, Tensor&, Tensor&,
                                 Nvfp4W4a4Workspace, cudaStream_t) {
    reject_nvfp4_a4();
}

// --- NVFP4 causal-attention prompt path -----------------------------------------------------------

void causal_attention_prompt_nvfp4_launch(const Tensor&, const Tensor&, const Tensor&,
                                          const Tensor&, const Tensor&, const Tensor&, float,
                                          PagedKVBatchLayerView, Tensor&, cudaStream_t) {
    reject_nvfp4_prompt_kv();
}

void causal_attention_prompt_nvfp4_attention_launch(const Tensor&, const Tensor&, float,
                                                    const PagedKVLayerView&, Tensor&,
                                                    cudaStream_t) {
    reject_nvfp4_prompt_kv();
}

} // namespace ninfer::ops::detail
