#pragma once

// Written for NInfer-3090. One sequence through Inkling-2-Mini's 48 layer visits, in order
// (docs/maintainer/inkling2-model.md, "Forward, in order"). The Program owns the mutable state: a KV ring and four
// convolution histories per layer visit, and its workspace. Every state stays FP32: this is the correctness-first
// composition of the FP32 Ops (fp32_rows, depth_attention, short_conv, relative_bias_attention, moe_situ) and
// projection_fp32 for the BF16 matrices, which the decode and prefill kernels of later phases replace.

#include "core/arena.h"
#include "core/tensor.h"
#include "models/inkling2/model.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <functional>
#include <span>
#include <string>
#include <vector>

namespace ninfer::models::inkling2 {

struct ProgramLimits {
    std::int32_t max_context = 8192; // positions a sequence may reach (sizes the global KV rings)
    std::int32_t max_chunk   = 256;  // tokens per forward call
};

class TextProgram {
public:
    // Called with llama.cpp's names (tools/inkling2/llama_golden.cpp): attn_out-v, ffn_inp-v, l_out-v per layer
    // visit v, and result_norm; each FP32 [H, T] on the device. Also route_ids-v, the experts the visit's MoE
    // chose: I32 [top_k, T]. The callback runs on the host between launches
    // (the program synchronizes the stream before it), so it may copy the tensor out.
    using Trace = std::function<void(const std::string& name, const Tensor& value)>;

    TextProgram(const Model& model, ProgramLimits limits, cudaStream_t stream);
    ~TextProgram();
    TextProgram(const TextProgram&)            = delete;
    TextProgram& operator=(const TextProgram&) = delete;

    // Forgets the sequence: position 0, zero convolution histories (the KV rings need no clearing).
    void reset();
    [[nodiscard]] std::int64_t position() const noexcept { return position_; }

    // Appends tokens at position() and returns their logits: FP32 [vocab_size, T] on the device (rows past
    // unpadded_vocab_size are padding and never sampled), softcapped. Valid until the next call.
    Tensor forward(std::span<const std::int32_t> tokens, const Trace* trace = nullptr);

private:
    struct VisitState {
        Tensor k_cache, v_cache;                       // FP32 [kv_heads * head_dim, capacity]
        Tensor k_hist, v_hist, attn_hist, mlp_hist;    // FP32 [C, K - 1]
    };

    Tensor visit(std::uint32_t v, const Tensor& cur, std::vector<Tensor>& blocks, const Tensor* messages, std::int64_t start,
                 const Trace* trace);
    void control(const ControllerWeights& w, const Tensor& c, const Tensor& m, Tensor& out);
    Tensor project(const Tensor& x_bf16, std::span<const WeightId> weights, std::span<const std::int32_t> rows);

    const Model& model_;
    const TextConfig& config_;
    ProgramLimits limits_;
    cudaStream_t stream_;
    DeviceBuffer state_memory_;
    DeviceArena workspace_;
    std::vector<VisitState> state_;
    std::int64_t position_ = 0;
};

} // namespace ninfer::models::inkling2
