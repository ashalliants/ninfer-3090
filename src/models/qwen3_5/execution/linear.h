#pragma once

#include "models/qwen3_5/execution/parameters.h"
#include "models/qwen3_5/execution/rotation.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/linear_add.h"
#include "ninfer/ops/linear_swiglu.h"
#include "ninfer/ops/two_stage_head.h"

#include <cstddef>

namespace ninfer::models::qwen3_5::execution {

inline void project(const Tensor& input, const LinearParameters& p, Tensor& output,
                    WorkspaceArena& workspace, cudaStream_t stream,
                    InputBasis basis = InputBasis::Primal) {
    auto scope     = workspace.scope();
    const Tensor x = rotated_input(input, p.hadamard_signs, workspace, stream, basis);
    ops::linear(x, p.weight, output, p.policy, workspace, stream);
}

inline void project_add(const Tensor& input, const LinearParameters& p, Tensor& residual,
                        WorkspaceArena& workspace, cudaStream_t stream,
                        InputBasis basis = InputBasis::Primal) {
    auto scope     = workspace.scope();
    const Tensor x = rotated_input(input, p.hadamard_signs, workspace, stream, basis);
    ops::linear_add(x, p.weight, residual, p.policy, workspace, stream);
}

inline void project_swiglu(const Tensor& input, const LinearParameters& p, Tensor& output,
                           WorkspaceArena& workspace, cudaStream_t stream,
                           InputBasis basis = InputBasis::Primal) {
    auto scope     = workspace.scope();
    const Tensor x = rotated_input(input, p.hadamard_signs, workspace, stream, basis);
    ops::linear_swiglu(x, p.weight, output, p.policy, workspace, stream);
}

// --lm-head-two-stage: the scratch the Text output head's two-stage route takes at `tokens`
// columns (the rotated input and the Op's workspace), or 0 without the head's 2-bit copy. Startup
// plans it for every ordinary decode and target verify body.
[[nodiscard]] inline std::size_t output_head_two_stage_scratch_bytes(const TextParameters& text,
                                                                     std::int32_t tokens) {
    if (!text.output_head_proxy) { return 0; }
    const auto& head = text.output_head;
    return rotated_workspace_bytes(
        head.hadamard_signs, head.weight.k, tokens,
        ops::two_stage_head_workspace_capacity_bytes(head.weight.n, head.weight.k, tokens));
}

} // namespace ninfer::models::qwen3_5::execution
