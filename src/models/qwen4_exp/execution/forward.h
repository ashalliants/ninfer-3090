#pragma once

// Adapted from Infernix a3edb450 src/models/qwen4_exp/execution/forward.h (Apache-2.0).
// Modified for NInfer-3090: one sequence per call (M0: teacher-forced chunks and single-sequence
// decode); NInfer's single-sequence QSA Ops and paged INT8 KV; the PLE rows come from the caller
// already read; experts are served by the GPU narrow route from device frames, staging slots or the
// pinned bank (no CPU miss service, overlap streams, landing frames, prefill streaming, MTP,
// vision, verification or prefix-cache events).
//
// The Qwen4Exp forward pass: embedding, the PLE injection, 48 hyper-connection blocks of GDN or QSA
// plus the offloaded MoE, the final mixer and the LM head.

#include "core/arena.h"
#include "core/device.h"
#include "core/paged_kv_cache.h"
#include "core/tensor.h"
#include "models/qwen4_exp/execution/parameters.h"
#include "ninfer/ops/qsa.h"

#include <cstdint>
#include <vector>

namespace ninfer::models::qwen4_exp::execution {

// Recurrent state of the one sequence, owned by the caller (the Program in later milestones).
struct ForwardState {
    std::vector<Tensor> gdn_conv;      // per GDN block: BF16 [C, K - 1], oldest first
    std::vector<Tensor> gdn_recurrent; // per GDN block: FP32 [Dk, Dv, value heads]
    Tensor ple_conv;                   // BF16 [S*H, span, 1]: the PLE convolution history
    std::vector<Tensor> qsa_tails;     // per attention block: BF16 [Di, R - 1]
};

// Paged KV of the attention blocks, through one single-sequence block table.
struct ForwardKV {
    std::vector<PagedKVLayerView> layers; // per attention block (Int8Group64), block_table set
    std::vector<Tensor> pooled;           // per attention block: BF16 [Di / R, 64, 1, pages]
    Tensor block_table;                   // I32 [pages]
    std::uint32_t max_visible_keys = 0;   // the context the KV and selection are sized for
};

// Where the routed experts are read: device frames for resident experts (frames[l][e] >= 0), the
// pinned host bank otherwise, through `staging_slots` device slots (0: zero-copy).
struct ForwardExperts {
    const std::uint8_t* frame_base = nullptr;
    std::vector<const std::int32_t*> frames; // per layer, device I32 [E]: frame or -1
    std::uint8_t* staging_base  = nullptr;
    std::int32_t staging_slots  = 0;
    std::uint32_t* error        = nullptr; // optional mapped word (ops::MoeExpertSource::error)
};

// One call: T consecutive positions of the sequence.
struct ForwardBatch {
    Tensor ids;              // I32 [T]
    Tensor positions;        // I32 [T]: KV positions, consecutive
    Tensor rope_positions;   // I32 [T, 3] axis-major
    Tensor block_start_rope; // I32 [3]: RoPE position of the first pooled-block token (ops::qsa_pool_keys)
    Tensor ngram_rows;       // U8 [row bytes, heads, T]: each column's n-gram rows (NgramVolume)
    Tensor logit_columns;    // I32 [n]: the columns whose logits are produced
};

// Optional observation of every block, for reference comparison: BF16 copies per block of the
// residual after it ([S*H, T]), its routed expert ids (I32 [top k, T]), and the mixer (GDN/QSA)
// and MoE inputs and outputs ([H, T]), so each op chain can be checked on identical inputs.
struct ForwardTap {
    std::vector<Tensor>* residuals     = nullptr;
    std::vector<Tensor>* routes        = nullptr;
    std::vector<Tensor>* mixer_inputs  = nullptr;
    std::vector<Tensor>* mixer_outputs = nullptr;
    std::vector<Tensor>* moe_inputs    = nullptr;
    std::vector<Tensor>* moe_outputs   = nullptr;
};

class Forward {
public:
    Forward(const Parameters& parameters, DeviceContext& device, WorkspaceArena& work, ForwardState state,
            ForwardKV kv, ForwardExperts experts);

    // FP32 logits [V, n] of batch.logit_columns.
    void run(const ForwardBatch& batch, Tensor& logits, const ForwardTap* tap = nullptr);

    // Workspace one call of up to `columns` columns needs, at the given context.
    [[nodiscard]] static std::size_t workspace_bytes(const TextConfig& config, std::int32_t columns,
                                                     std::uint32_t max_visible_keys, DeviceExecutionView execution);

private:
    void layer(std::uint32_t layer, const ForwardBatch& batch, Tensor& residual, const ForwardTap* tap);
    void ple(const PleParameters& p, Tensor& residual, const ForwardBatch& batch);
    Tensor mix(const HyperConnectionParameters& p, const Tensor& residual, Tensor* inject);
    Tensor gdn(const GdnParameters& p, const Tensor& x, std::uint32_t index);
    Tensor attention(const AttentionParameters& p, const Tensor& x, std::uint32_t index, const ForwardBatch& batch);
    Tensor moe(const MoeParameters& p, const Tensor& x, std::uint32_t layer, Tensor* route_tap);

    const Parameters& parameters_;
    const TextConfig& config_;
    DeviceContext& device_;
    WorkspaceArena& work_;
    ForwardState state_;
    ForwardKV kv_;
    ForwardExperts experts_;
};

// The QSA indexer geometry of the configuration.
[[nodiscard]] ops::QsaIndexerGeometry qsa_geometry(const TextConfig& config);

} // namespace ninfer::models::qwen4_exp::execution
