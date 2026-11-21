#pragma once

// Written for NInfer-3090. Sigmoid top-k routing and softcapped-SwiGLU ("SITU") routed experts over a resident
// bank, for Inkling-2-Mini (docs/maintainer/inkling2-model.md 3j). Columns follow fp32_rows.h.

#include "core/arena.h"
#include "core/tensor.h"
#include "core/weight.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops {

/**
 * Op: moe_route_sigmoid_topk
 *
 * Math / indexing:
 *   L[e, t] = sum_d router[d, e] * z[d, t];  s = 1 / (1 + exp(-L))
 *   ids[:, t] = the top_k experts by s + bias (descending; the lower id wins a tie)
 *   weights[i, t] = scale * s[ids[i, t], t] / (sum_i s[ids[i, t], t] + 1e-20)
 *
 * Logical shapes: z FP32 [D, T] (any column stride); router FP32 [D, E] contiguous; bias FP32 [E];
 *   ids I32 [top_k, T] and weights FP32 [top_k, T], contiguous.
 * Supported domain: E at most 256, 1 <= top_k <= min(E, 16), T at most 65535.
 * Numeric: FP32 logits; oracle FP64 (inkling2_reference.h route). A near tie at the k-th place may select
 *   differently from the oracle; the test accepts that only within the FP32 bound of the two scores.
 * Workspace: moe_route_sigmoid_topk_workspace_bytes(E, T) from `workspace`, released on return.
 * Execution: graph capturable.
 */
[[nodiscard]] std::size_t moe_route_sigmoid_topk_workspace_bytes(std::int32_t experts, std::int64_t columns);
void moe_route_sigmoid_topk(const Tensor& z, const Tensor& router, const Tensor& bias, int top_k, float scale, DeviceArena& workspace,
                            Tensor& ids, Tensor& weights, cudaStream_t stream);

/// A layer's routed experts, resident in device memory. Either a ggml_expert_record_v1 bank
/// (`records`, `format` GGML_REC_NVFP4_NVFP4: gate [I, H], up [I, H] and down [H, I] per record,
/// storage-layouts.md section 7) with optional per-expert FP32 multipliers of the gate/up and down
/// products, or two BF16 banks: gate_up [H, 2I, E] (rows 0..I-1 gate, I..2I-1 up) and down [I, H, E].
struct SituExpertBank {
    QType format                = QType::BF16;
    const std::uint8_t* records = nullptr;
    std::uint64_t record_stride = 0;
    const float* gate_up_scale  = nullptr; // [E] or null
    const float* down_scale     = nullptr; // [E] or null
    const void* gate_up         = nullptr; // BF16 banks
    const void* down            = nullptr;
    std::int32_t experts = 0, hidden = 0, intermediate = 0;
};

/**
 * Op: moe_situ_experts
 *
 * Math / indexing:
 *   For column t and slot i with e = ids[i, t]:
 *     [g; u] = (Wgu_e z[:, t]) * gate_up_scale[e]          (scale 1 when absent)
 *     a      = gate_cap * tanh(g / gate_cap) * sigmoid(g) * up_cap * tanh(u / up_cap)
 *     y_i    = (Wd_e a) * down_scale[e]
 *   out[:, t] = sum_i weights[i, t] * y_i
 *   where Wgu_e, Wd_e are the exact decode of the bank's blocks (tensor-formats.md ggml_nvfp4) or its BF16 values.
 *
 * Logical shapes: z FP32 [H, T] (any column stride); ids I32 [k, T], weights FP32 [k, T] contiguous; out FP32
 *   [H, T] contiguous.
 * Supported domain: format GGML_REC_NVFP4_NVFP4 (H and I multiples of 64) or BF16 (H and I multiples of 8, banks
 *   16-byte aligned); H and I at most 8192.
 * Numeric: FP32 activations and accumulation (no activation quantization); oracle FP64 over the decoded
 *   weights (inkling2_reference.h decode_nvfp4, situ).
 * Effects: writes out; it must not overlap z.
 * Workspace: moe_situ_experts_workspace_bytes(bank, k * T) from `workspace`, released on return.
 * Execution: graph capturable.
 */
[[nodiscard]] std::size_t moe_situ_experts_workspace_bytes(const SituExpertBank& bank, std::int64_t entries);
void moe_situ_experts(const SituExpertBank& bank, const Tensor& z, const Tensor& ids, const Tensor& weights, float gate_cap,
                      float up_cap, DeviceArena& workspace, Tensor& out, cudaStream_t stream);

} // namespace ninfer::ops
