#pragma once

// Written for NInfer-3090. Inkling-2-Mini's DFlash2 drafter (docs/maintainer/inkling2-model.md, "DFlash2 drafter";
// incoai/Inkling-2-Mini-DFlash2): five non-causal sliding-window layers over the committed positions' fused taps,
// each wrapped in a two-tap dynamic convolution, then the target's head's top 16 per draft position and a rank-256
// selector that picks each draft given the one before it. FP32 state and sums; the matrices are re-encoded to 8 bits
// at load (and the head read in its 4-bit copy), lossy for the drafts only: the target verifies every draft.

#include "core/arena.h"
#include "core/tensor.h"
#include "models/inkling2/dflash2.h"
#include "models/inkling2/model.h"
#include "models/inkling2/program/drafter.h"
#include "ninfer/ops/dspark_fp32.h"
#include "ninfer/ops/projection_fp32.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <span>
#include <vector>

namespace ninfer::models::inkling2 {

class DFlash2Program final : public Drafter {
public:
    // release: read here once (its matrices re-encoded on the device); it need not outlive the Program. max_chunk is
    // the most positions one extend() takes (at most the sliding window). head_q8, kv_fp16, head_copy and workspace as
    // DSparkProgram's (the target's Q8 head when it has one; FP16 context rings; the drafter's own 4-bit head copy).
    DFlash2Program(const Model& model, const DFlash2Release& release, std::int32_t max_context, std::int32_t max_chunk, cudaStream_t stream,
                   const ops::Q8Weight* head_q8 = nullptr, bool kv_fp16 = false, bool head_copy = true, DeviceSpan workspace = {});
    [[nodiscard]] static std::size_t workspace_bytes(const Model& model, const DFlash2Config& config, std::int32_t max_chunk);
    ~DFlash2Program() override;
    DFlash2Program(const DFlash2Program&)            = delete;
    DFlash2Program& operator=(const DFlash2Program&) = delete;

    void reset() noexcept override { position_ = 0; }
    [[nodiscard]] std::int64_t position() const noexcept override { return position_; }
    [[nodiscard]] std::int32_t max_chunk() const noexcept override { return max_chunk_; }
    [[nodiscard]] const std::vector<std::uint32_t>& tap_visits() const noexcept override { return config_.target_visits; }
    [[nodiscard]] std::int32_t max_drafts() const noexcept override { return static_cast<std::int32_t>(config_.block_size) - 1; }

    // fused = rmsnorm(fc [taps], hidden_norm); each layer's context K (k_norm, RoPE at the position) and V of it.
    void extend(std::span<const Tensor> taps, std::int32_t n) override;
    // Greedy drafts: the selector's argmax. A sampled pick draws each draft over its 16 candidates with the request's
    // processors and the target's uniform for that position (ops::dflash2_select), so a draft is the target's own
    // draw wherever the two distributions agree. probs: the selector's probability of each draft.
    std::vector<std::int32_t> draft(std::int32_t anchor, std::int32_t n, const ops::DraftPick* pick = nullptr,
                                    std::span<const double> canonical = {}, std::vector<float>* probs = nullptr) override;

    // Every layer's ring (each holds its last sliding-window positions).
    [[nodiscard]] std::size_t snapshot_bytes() const override;
    void save(Snapshot& snapshot) override;
    void save_async(Snapshot& snapshot, cudaStream_t copy) override;
    void restore(const Snapshot& snapshot) override;

private:
    struct Layer {
        ops::DenseMatrix q, k, v, output, gate, up, down, attention_conv, mlp_conv; // Q8_G32_FP16
        Tensor input_norm, post_norm, q_norm, k_norm;                               // FP32
        Tensor attention_base, mlp_base;                                            // FP32 [H, 4]
        Tensor k_ring, v_ring;                                                      // [kv_heads * head_dim, window]
    };

    const Model& model_;
    DFlash2Config config_;
    const TextConfig& text_;
    std::int32_t max_context_, max_chunk_;
    cudaStream_t stream_;
    std::vector<DeviceBuffer> weight_memory_; // one per device weight
    DeviceBuffer state_memory_;
    DeviceArena workspace_;
    std::vector<Layer> layers_;
    ops::DenseMatrix fc_{}, selector_{};
    Tensor hidden_norm_, output_norm_, predecessor_, successor_; // FP32 [H]; BF16 [256, V]
    Tensor masks_;  // FP32 [hidden, block]: column 0 unused, the rest the normed mask embedding
    Tensor tokens_; // I32 [block]: the anchor, then the drafts
    Tensor probs_;  // FP32 [block]
    Tensor draws_;  // FP64 [block] (as I32 [2 * block]): a sampled round's uniforms
    std::int64_t position_ = 0;
    // The base logits' head: the target's Q8 re-encoding, the drafter's 4-bit copy, or the BF16 head.
    DeviceBuffer head_memory_;
    ops::Q8Weight head_q8_{};
    ops::Q4Matrix head_q4_{};
    ops::DenseMatrix head_bf16_{};
    // save_async: the copy's events and descriptors.
    PinnedHostBuffer copy_descriptors_;
    cudaEvent_t copy_done_ = nullptr;
    bool copy_pending_     = false;
    void wait_copy();
    template <class F> void ring_rows(const Tensor& ring, F&& f) const;
};

} // namespace ninfer::models::inkling2
