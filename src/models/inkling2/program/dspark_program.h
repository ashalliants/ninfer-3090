#pragma once

// Written for NInfer-3090. Inkling-2-Mini's DSpark drafter (docs/maintainer/inkling2-model.md, "DSpark drafter";
// llama.cpp's src/models/dspark.cpp graph_inkling2 and common_speculative's draft-dspark): the context K/V of the
// committed target positions, and one block draft from the anchor. FP32 throughout (include/ninfer/ops/dspark_fp32.h).

#include "core/arena.h"
#include "core/tensor.h"
#include "models/inkling2/model.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <span>
#include <vector>

namespace ninfer::models::inkling2 {

class DSparkProgram {
public:
    // max_context sizes the global layer's ring; max_chunk is the most positions one extend() takes.
    DSparkProgram(const Model& model, std::int32_t max_context, std::int32_t max_chunk, cudaStream_t stream);
    DSparkProgram(const DSparkProgram&)            = delete;
    DSparkProgram& operator=(const DSparkProgram&) = delete;

    void reset() noexcept { position_ = 0; }
    // Target positions committed so far; the next anchor sits here.
    [[nodiscard]] std::int64_t position() const noexcept { return position_; }
    // The target visits whose hidden states extend() takes, in order (the TextProgram's tap_visits).
    [[nodiscard]] const std::vector<std::uint32_t>& tap_visits() const noexcept { return config_.target_visits; }

    // Commits the next n target positions: tap i is the target's hidden state at tap_visits()[i] for them, FP32
    // [hidden, >= n] (columns 0 .. n - 1 are used). Writes each draft layer's context K/V and keeps the Markov
    // head's context (the last position's).
    void extend(std::span<const Tensor> taps, std::int32_t n);

    // The block_size - 1 tokens drafted after `anchor`, the target's next token at position() (>= 1).
    std::vector<std::int32_t> draft(std::int32_t anchor);

private:
    struct Layer {
        Tensor k_ring, v_ring; // FP32 [kv_heads * head_dim, capacity]
    };

    const Model& model_;
    const DSparkConfig& config_;
    const TextConfig& text_;
    std::int32_t max_context_, max_chunk_;
    cudaStream_t stream_;
    DeviceBuffer state_memory_;
    DeviceArena workspace_;
    std::vector<Layer> layers_;
    Tensor masks_;      // FP32 [hidden, block]: column 0 unused, the rest the mask embedding
    Tensor joint_in_;   // FP32 [2R + 2H]: state, W1[prev], x_i, the context of the anchor (f_rnn)
    Tensor pt_, hb_;    // FP32 [V, R] (ne[0] = V) = W_head W_o, and FP32 [V] = W_head b_o
    Tensor ids_;        // I32 [block]: the anchor, then the drafts
    std::int64_t position_ = 0;
};

} // namespace ninfer::models::inkling2
