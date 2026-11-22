#pragma once

// Written for NInfer-3090. Inkling-2-Mini's DSpark drafter (docs/maintainer/inkling2-model.md, "DSpark drafter";
// llama.cpp's src/models/dspark.cpp graph_inkling2 and common_speculative's draft-dspark): the context K/V of the
// committed target positions, and one block draft from the anchor. FP32 state and sums; the weights it reads per
// round in compact forms (Q8 P and head, BF16 activations into the projections), lossy for drafts only.

#include "core/arena.h"
#include "core/tensor.h"
#include "models/inkling2/model.h"
#include "ninfer/ops/dspark_fp32.h"
#include "ninfer/ops/projection_fp32.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <span>
#include <unordered_map>
#include <vector>

namespace ninfer::models::inkling2 {

class DSparkProgram {
public:
    // max_context sizes the global layer's ring; max_chunk is the most positions one extend() takes. head_q8, when
    // the target runs its head over an 8-bit re-encoding (DenseWeightsQ8::head, a view of the owner both Programs read), is
    // read in place of the BF16 head so the drafts are proposed from the logits the target verifies them with; its
    // owner must outlive the drafter. kv_fp16: the context rings hold FP16 (round to nearest), half the bytes; only the
    // drafts can change, never the verified output. global_window, when positive and under max_context: the global
    // layers attend over only their last global_window context positions, held in a ring of that many (a fraction of
    // the bytes at a long context; again only the drafts can change). head_copy false (and no head_q8): the base
    // logits read the target's BF16 head instead of the drafter's Q8 copy of it (492 MB less, a slower draft).
    DSparkProgram(const Model& model, std::int32_t max_context, std::int32_t max_chunk, cudaStream_t stream,
                  const ops::Q8Weight* head_q8 = nullptr, bool kv_fp16 = false, std::int32_t global_window = 0,
                  bool head_copy = true);
    [[nodiscard]] std::int32_t max_chunk() const noexcept { return max_chunk_; }
    DSparkProgram(const DSparkProgram&)            = delete;
    DSparkProgram& operator=(const DSparkProgram&) = delete;

    void reset() noexcept { position_ = 0; }

    // The drafter's state at position() (prefix reuse, as TextProgram::Snapshot): its rings (the sliding layers', and the
    // global layers' under a global_window) and the anchor context. Unwindowed global rings are indexed by position and
    // not copied.
    struct Snapshot {
        DeviceBuffer memory;        // snapshot_bytes()
        std::int64_t position = -1; // -1: holds nothing
    };
    [[nodiscard]] std::size_t snapshot_bytes() const;
    void save(Snapshot& snapshot) const;
    void restore(const Snapshot& snapshot);
    // Target positions committed so far; the next anchor sits here.
    [[nodiscard]] std::int64_t position() const noexcept { return position_; }
    // The target visits whose hidden states extend() takes, in order (the TextProgram's tap_visits).
    [[nodiscard]] const std::vector<std::uint32_t>& tap_visits() const noexcept { return config_.target_visits; }

    // Commits the next n target positions: tap i is the target's hidden state at tap_visits()[i] for them, FP32
    // [hidden, >= n] (columns 0 .. n - 1 are used). Writes each draft layer's context K/V and keeps the Markov
    // head's context (the last position's).
    void extend(std::span<const Tensor> taps, std::int32_t n);

    // The first `n` (1 .. block_size - 1) tokens drafted after `anchor`, the target's next token at position() (>= 1).
    // The block forward always takes all block_size positions (its attention is non-causal); only the Markov chain
    // and the head stop at n, which leaves the n drafts the same as the full block's first n.
    // `probs`, when not null, receives the drafter's probability of each draft (ops::dspark_draft_pick, the same tokens
    // as argmax_fp32).
    std::vector<std::int32_t> draft(std::int32_t anchor, std::int32_t n, std::vector<float>* probs = nullptr);

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
    std::vector<std::int32_t> window_; // per layer: the keys it attends over (0: every context position)
    Tensor masks_;      // FP32 [hidden, block]: column 0 unused, the rest the mask embedding
    Tensor joint_in_;   // FP32 [2R + 2H]: state, W1[prev], x_i, the context of the anchor (f_rnn)
    Tensor hb_;         // FP32 [V] = W_head b_o
    Tensor ids_;        // I32 [block]: the anchor, then the drafts
    Tensor probs_;      // FP32 [block]: the drafter's probability of each draft (picked drafts)
    std::int64_t position_ = 0;
    // The drafter's compact forms (lossy; they change only drafts): P = W_head W_o as Q8, and the head as Q8 (the
    // target's re-encoded head when it has one, else the drafter's own copy in head_memory_).
    ops::MarkovHead markov_{};
    DeviceBuffer markov_memory_;
    DeviceBuffer head_memory_;
    ops::Q8Weight head_q8_{};
    ops::DenseMatrix head_bf16_{}; // the base logits' head when there is no Q8 one
    // The Q8_0 matrices in the split form (same values; a faster projection kernel), by their artifact address.
    DeviceBuffer split_memory_, copy_memory_;
    std::unordered_map<const void*, ops::DenseMatrix> split_;
    [[nodiscard]] ops::DenseMatrix matrix(const Model::Matrix& m) const;
};

} // namespace ninfer::models::inkling2
