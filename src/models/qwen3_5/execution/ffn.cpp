#include "models/qwen3_5/execution/ffn.h"
#include "models/qwen3_5/execution/lora.h"
#include "models/qwen3_5/execution/rotation.h"

#include "core/layout.h"
#include "ninfer/ops/hadamard_transform.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/linear_add.h"
#include "ninfer/ops/linear_swiglu.h"
#include "ninfer/ops/residual_add.h"
#include "ninfer/ops/silu_mul.h"

#include <array>
#include <stdexcept>

namespace ninfer::models::qwen3_5::execution {
namespace {

bool rotated_dense(const DenseParameters& p) {
    return rotated(p.gate_up.hadamard_signs) || rotated(p.down.hadamard_signs);
}

// A Hadamard-rotated Dense FFN projects gate/up into one plane from the rotated input; when down
// is rotated, silu_mul_hadamard writes its input already in the rotated basis.
std::size_t rotated_dense_workspace_bytes(const DenseParameters& p, std::int32_t first,
                                          std::int32_t last, bool prefill) {
    const auto policy = [&](ops::LinearPolicy own) {
        return prefill ? ops::prefill_phase_policy(own) : own;
    };
    const auto& gu   = p.gate_up.weight;
    const auto& down = p.down.weight;
    WorkspaceLayoutBuilder layout;
    (void)layout.alloc(DType::BF16, {gu.n, last});
    {
        auto scope = layout.scope();
        (void)layout.alloc_bytes(
            rotated_workspace_bytes(p.gate_up.hadamard_signs, gu.k, last,
                                    ops::linear_workspace_capacity_bytes(
                                        gu.qtype, gu.n, gu.k, policy(p.gate_up.policy), first, last)));
    }
    (void)layout.alloc(DType::BF16, {gu.n / 2, last});
    (void)layout.alloc_bytes(ops::linear_add_workspace_capacity_bytes(
        down.qtype, down.n, down.k, policy(p.down.policy), first, last));
    return layout.peak_bytes(1);
}

void rotated_dense_ffn(const Tensor& hidden, const DenseParameters& p, Tensor& residual,
                       WorkspaceArena& workspace, cudaStream_t stream, InputBasis basis,
                       bool prefill) {
    const auto policy = [&](ops::LinearPolicy own) {
        return prefill ? ops::prefill_phase_policy(own) : own;
    };
    const auto columns      = hidden.ne[1];
    const auto& gu          = p.gate_up.weight;
    const auto intermediate = gu.n / 2;
    Tensor plane            = workspace.alloc(DType::BF16, {gu.n, columns});
    {
        auto call      = workspace.scope();
        const Tensor x = rotated_input(hidden, p.gate_up.hadamard_signs, workspace, stream, basis);
        ops::linear(x, gu, plane, policy(p.gate_up.policy), workspace, stream);
    }
    Tensor activation = workspace.alloc(DType::BF16, {intermediate, columns});
    if (rotated(p.down.hadamard_signs)) {
        ops::silu_mul_hadamard(plane, p.down.hadamard_signs, activation, stream);
    } else {
        ops::silu_mul(plane.slice(0, 0, intermediate), plane.slice(0, intermediate, intermediate),
                      activation, stream);
    }
    ops::linear_add(activation, p.down.weight, residual, policy(p.down.policy), workspace, stream);
}

// The gate/up policy of the base route lora_ffn stands in for: ffn()'s linear_swiglu policy on an
// ordinary model, rotated_dense_ffn()'s linear policy on a rotated one.
ops::LinearPolicy lora_gate_up_policy(const DenseParameters& p, bool verify, bool prefill) {
    if (rotated_dense(p)) {
        return prefill ? ops::prefill_phase_policy(p.gate_up.policy) : p.gate_up.policy;
    }
    return verify    ? p.verify_gate_up_policy
           : prefill ? ops::prefill_phase_policy(p.gate_up.policy)
                     : p.gate_up.policy;
}

} // namespace

std::size_t lora_ffn_workspace_bytes(const DenseParameters& p, const LoraParameters& lora,
                                     std::int32_t first, std::int32_t last, bool prefill) {
    const auto policy = [&](ops::LinearPolicy own) {
        return prefill ? ops::prefill_phase_policy(own) : own;
    };
    const auto& gu   = p.gate_up.weight;
    const auto& down = p.down.weight;
    WorkspaceLayoutBuilder layout;
    (void)layout.alloc(DType::BF16, {gu.n, last});
    {
        auto scope = layout.scope();
        if (rotated(p.gate_up.hadamard_signs)) { (void)layout.alloc(DType::BF16, {gu.k, last}); }
        {
            auto call = layout.scope();
            // Either phase's gate/up policy (decode and verify share a plan, so both are covered).
            std::size_t bytes = 0;
            for (const bool verify : {false, true}) {
                const ops::LinearPolicy own = lora_gate_up_policy(p, verify && !prefill, prefill);
                bytes = std::max(bytes, rotated_dense(p)
                                            ? ops::linear_workspace_capacity_bytes(
                                                  gu.qtype, gu.n, gu.k, own, first, last)
                                            : ops::linear_swiglu_plane_workspace_capacity_bytes(
                                                  gu.qtype, gu.n, gu.k, own, first, last));
            }
            (void)layout.alloc_bytes(bytes);
        }
        (void)layout.alloc_bytes(ops::lora_workspace_bytes(2 * lora.rank, gu.k, last));
    }
    (void)layout.alloc(DType::BF16, {gu.n / 2, last});
    {
        auto call = layout.scope();
        (void)layout.alloc_bytes(ops::linear_add_workspace_capacity_bytes(
            down.qtype, down.n, down.k, policy(p.down.policy), first, last));
    }
    (void)layout.alloc_bytes(ops::lora_workspace_bytes(lora.rank, down.k, last));
    return layout.peak_bytes(1);
}

void lora_ffn(const Tensor& hidden, const DenseParameters& p, const LoraBlock& lora,
              const ops::LoraColumns& columns, Tensor& residual, WorkspaceArena& workspace,
              cudaStream_t stream, InputBasis basis, bool verify, bool prefill) {
    const auto policy = [&](ops::LinearPolicy own) {
        return prefill ? ops::prefill_phase_policy(own) : own;
    };
    auto scope              = workspace.scope();
    const auto count        = hidden.ne[1];
    const auto& gu          = p.gate_up.weight;
    const auto intermediate = gu.n / 2;
    Tensor plane            = workspace.alloc(DType::BF16, {gu.n, count});
    {
        auto call      = workspace.scope();
        const Tensor x = rotated_input(hidden, p.gate_up.hadamard_signs, workspace, stream, basis);
        {
            // The base route's projection: an ordinary model's fused linear_swiglu runs its own
            // routes (sm_70's streaming GEMV, the cuBLAS prefill GEMM), and linear_swiglu_plane
            // takes the same ones up to the plane; a rotated model's base route is ops::linear.
            auto linear_scope               = workspace.scope();
            const ops::LinearPolicy own     = lora_gate_up_policy(p, verify, prefill);
            if (rotated_dense(p)) {
                ops::linear(x, gu, plane, own, workspace, stream);
            } else {
                ops::linear_swiglu_plane(x, gu, plane, own, workspace, stream);
            }
        }
        std::array<Tensor, 2> halves{plane.slice(0, 0, intermediate),
                                     plane.slice(0, intermediate, intermediate)};
        apply_lora_site(lora.ffn_input, x, halves, columns, workspace, stream);
    }
    Tensor activation = workspace.alloc(DType::BF16, {intermediate, count});
    if (rotated(p.down.hadamard_signs)) {
        ops::silu_mul_hadamard(plane, p.down.hadamard_signs, activation, stream);
    } else {
        ops::silu_mul(plane.slice(0, 0, intermediate), plane.slice(0, intermediate, intermediate),
                      activation, stream);
    }
    {
        auto call = workspace.scope();
        ops::linear_add(activation, p.down.weight, residual, policy(p.down.policy), workspace, stream);
    }
    std::array<Tensor, 1> out{residual};
    apply_lora_site(lora.ffn_output, activation, out, columns, workspace, stream);
}

std::size_t ffn_workspace_bytes(const FfnParameters& parameters, std::int32_t first,
                                std::int32_t last, bool mtp, bool verify, bool prefill) {
    if (first <= 0 || last < first) { throw std::invalid_argument("FFN: invalid column interval"); }
    if (const auto* moe = std::get_if<ops::SparseMoeWeights>(&parameters)) {
        return ops::sparse_moe_workspace_capacity_bytes(moe->routed_gate_up.qtype,
                                                        moe->routed_down.qtype, first, last);
    }
    const auto& p = std::get<DenseParameters>(parameters);
    if (rotated_dense(p)) { return rotated_dense_workspace_bytes(p, first, last, prefill); }
    // The policies ffn() / rotated_dense_ffn() run with (prefill: ops::prefill_phase_policy).
    const auto policy = [&](ops::LinearPolicy own) {
        return prefill ? ops::prefill_phase_policy(own) : own;
    };
    const auto& gu   = p.gate_up.weight;
    const auto& down = p.down.weight;
    WorkspaceLayoutBuilder layout;
    if (mtp) {
        (void)layout.alloc(DType::BF16, {gu.n, last});
        {
            auto scope = layout.scope();
            (void)layout.alloc_bytes(ops::linear_workspace_capacity_bytes(
                gu.qtype, gu.n, gu.k, p.gate_up.policy, first, last));
        }
        (void)layout.alloc(DType::BF16, {gu.n / 2, last});
        (void)layout.alloc(DType::BF16, {down.n, last});
        (void)layout.alloc_bytes(ops::linear_workspace_capacity_bytes(down.qtype, down.n, down.k,
                                                                      p.down.policy, first, last));
    } else {
        (void)layout.alloc(DType::BF16, {gu.n / 2, last});
        {
            auto scope = layout.scope();
            (void)layout.alloc_bytes(ops::linear_swiglu_workspace_capacity_bytes(
                gu.qtype, gu.n, gu.k, verify ? p.verify_gate_up_policy : policy(p.gate_up.policy),
                first, last));
        }
        {
            auto scope = layout.scope();
            (void)layout.alloc_bytes(ops::linear_add_workspace_capacity_bytes(
                down.qtype, down.n, down.k, policy(p.down.policy), first, last));
        }
    }
    return layout.peak_bytes(1);
}

const Tensor* ffn_input_signs(const FfnParameters& parameters) {
    const auto* dense = std::get_if<DenseParameters>(&parameters);
    return dense != nullptr && rotated(dense->gate_up.hadamard_signs)
               ? &dense->gate_up.hadamard_signs
               : nullptr;
}

void ffn(const Tensor& hidden, const FfnParameters& parameters, Tensor& residual,
         const ops::SparseMoeHints& hints, WorkspaceArena& workspace, DeviceExecutionView execution,
         bool mtp, bool verify, InputBasis basis, bool prefill) {
    const auto stream  = execution.stream;
    auto scope         = workspace.scope();
    const auto columns = hidden.ne[1];
    if (const auto* moe = std::get_if<ops::SparseMoeWeights>(&parameters)) {
        const auto storage =
            workspace.alloc_bytes(ffn_workspace_bytes(parameters, columns, columns));
        WorkspaceArena scratch(storage);
        ops::sparse_moe(hidden, *moe, ops::SparseMoeEpilogue::AddResidual, residual, hints, scratch,
                        execution);
        return;
    }
    const auto& p = std::get<DenseParameters>(parameters);
    if (rotated_dense(p)) {
        rotated_dense_ffn(hidden, p, residual, workspace, stream, basis, prefill);
        return;
    }
    const auto& gu   = p.gate_up.weight;
    const auto& down = p.down.weight;
    if (mtp) {
        Tensor gate_up = workspace.alloc(DType::BF16, {gu.n, columns});
        {
            auto call = workspace.scope();
            ops::linear(hidden, gu, gate_up, p.gate_up.policy, workspace, stream);
        }
        Tensor activation = workspace.alloc(DType::BF16, {gu.n / 2, columns});
        ops::silu_mul(gate_up.slice(0, 0, gu.n / 2), gate_up.slice(0, gu.n / 2, gu.n / 2),
                      activation, stream);
        Tensor delta = workspace.alloc(DType::BF16, {down.n, columns});
        ops::linear(activation, down, delta, p.down.policy, workspace, stream);
        ops::residual_add(delta, residual, stream);
        return;
    }
    Tensor activation = workspace.alloc(DType::BF16, {gu.n / 2, columns});
    {
        auto call = workspace.scope();
        const ops::LinearPolicy gate_up_policy =
            verify    ? p.verify_gate_up_policy
            : prefill ? ops::prefill_phase_policy(p.gate_up.policy)
                      : p.gate_up.policy;
        ops::linear_swiglu(hidden, gu, activation, gate_up_policy, workspace, stream);
    }
    ops::linear_add(activation, down, residual,
                    prefill ? ops::prefill_phase_policy(p.down.policy) : p.down.policy, workspace,
                    stream);
}

} // namespace ninfer::models::qwen3_5::execution
