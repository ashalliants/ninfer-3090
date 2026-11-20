#pragma once

// Written for NInfer-3090. The pattern follows src/models/qwen4_exp/execution/forward.h.
//
// The attention sublayer of a DeepSeek-V4.1-Flash block, as library code over bound weight views
// and persistent per-sequence state, so a prefill chunk of any length and a decode step (T = 1) run
// the same function. The mathematics is tools/dsv4/attention.py (`attention`, `compressor`,
// `indexer`, `select_candidate_blocks`, `window_indices`); the op chain is the one tests/ops'
// real-weights layer test validated stage by stage:
//
// clang-format off
//   x -> linear(q_a) -> rmsnorm -> linear(q_b) -> rope_pairs                      = q
//   x -> linear(kv) -> rmsnorm -> rope_pairs                                      = a raw KV row
//   kv sources: x -> projection(kv, gate) -> softmax_pool -> rmsnorm -> rope_pairs = compressed rows
//               (ratio 1: projection -> rmsnorm -> rope_pairs); index keys from the latent
//   index sources: indexer query from q_a's output, head weights from x, scores, top-k
//   gathered_sink_attention over [window rows | selected compressed rows]; the inverse rotation;
//   the grouped output projection (8 groups 4096 -> 1024) and out_b (8192 -> hidden).
// clang-format on
//
// State per sequence (AttentionState, on the device, sized from the caller's maximum context):
//   * per kv-owner group one KV table: the compressed rows [N_max, 512] first, then every member
//     layer's raw-row buffer of (window - 1 + max_chunk) rows, so one `gathered_sink_attention`
//     table serves a layer's window and the compressed rows it reads from its kv owner. Raw row r
//     of a layer holds position start - (window - 1) + r: the last window - 1 rows are shifted to
//     the front when a chunk ends, and the chunk's rows follow them;
//   * the index keys [N_max, 128] of each kv owner that has an index source;
//   * the compressor's pending partial group: the last (ratio - 1) tokens' projected kv and gate rows,
//     carried across chunk boundaries and decode steps the same way;
//   * the shared top-k of the nearest preceding index source and the candidate blocks of the
//     candidate layer (only valid during one forward pass over a chunk: every layer runs in order).
//
// Execution: stream-ordered, no host synchronization and no allocation (all transient storage comes
// from the caller's workspace arena, see attention_workspace_bytes). Which rows a chunk writes and
// which compressed rows are complete follow from the chunk's `start` position, a host value that is
// a kernel argument, so a captured graph serves one position (re-capture per step, or update the
// nodes); every shape of a decode step except the indexer's score width (the complete compressed
// rows) is constant.

#include "core/arena.h"
#include "core/tensor.h"
#include "core/weight.h"
#include "models/dsv4/config.h"
#include "models/dsv4/model.h"
#include "ninfer/ops/linear.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

namespace ninfer::models::dsv4 {

// Bound device views of one layer's attention parameters. GGML weights are [N, K] Weights; BF16
// projections are Tensors [K, N] (one row per output); norms BF16 [D]; sinks FP32 [heads].
struct CompressorViews {
    Tensor kv;                          // [hidden, 512]
    std::optional<Tensor> gate;         // [hidden, 512]; only where the ratio is above 1
    Tensor norm;                        // [512]
    std::optional<Tensor> index_k;      // [512, indexer_head_dim]: the index keys' projection
    std::optional<Tensor> index_k_norm; // [indexer_head_dim]
};

struct IndexerViews {
    Tensor q_b;  // [q_lora_rank, indexer_heads * indexer_head_dim]
    Tensor proj; // [hidden, indexer_heads]
};

struct AttentionLayerViews {
    Weight q_a, q_b, kv, out_a, out_b;
    Tensor q_a_norm, kv_norm, sinks;
    std::optional<CompressorViews> compressor; // kv source layers
    std::optional<IndexerViews> indexer;       // index source layers
    ops::LinearPolicy policy = ops::LinearPolicy::AllowA8;
};

// A layer's views bound from a materialized model: they point into the model's storage, except the
// norm weights the artifact keeps in FP32 (the GGUF's own), which `rmsnorm` takes as BF16: those
// are rounded to BF16 once into `owned`, which the views also point into (keep the binding alive).
struct AttentionBinding {
    AttentionLayerViews views;
    std::vector<DeviceBuffer> owned;
};

[[nodiscard]] AttentionBinding bind_attention_views(const Model& model, const AttentionWeights& ids,
                                                    cudaStream_t stream);

struct AttentionLimits {
    std::uint32_t max_context = 0; // positions per sequence (sizes the compressed rows)
    std::uint32_t max_chunk =
        0; // the most tokens of one call (sizes the raw rows and the selections)
};

// RoPE inverse frequencies (rope_dim / 2 entries) of a layer: plain with rope_theta in a
// window-only layer, YaRN with compress_rope_theta where it compresses (tools/dsv4/attention.py
// rope_inv_freqs).
[[nodiscard]] std::vector<float> rope_inv_freqs(const TextConfig& config, std::uint32_t layer);

class AttentionState {
public:
    AttentionState(const TextConfig& config, AttentionLimits limits);
    AttentionState(AttentionState&&) noexcept;
    AttentionState& operator=(AttentionState&&) noexcept;
    ~AttentionState();

    [[nodiscard]] const AttentionLimits& limits() const noexcept { return limits_; }

    [[nodiscard]] std::size_t device_bytes() const noexcept;

    // The compressed rows [512, N_max] (rotated) of the layer's kv owner, the owner's index keys
    // [indexer_head_dim, N_max] (rotated), the shared top-k [indexer_top_k, max_chunk] (positions,
    // -1 padded; the last chunk's) and the candidate blocks [candidate_blocks, max_chunk].
    [[nodiscard]] Tensor compressed_rows(std::uint32_t layer) const;
    [[nodiscard]] Tensor index_keys(std::uint32_t layer) const;
    [[nodiscard]] Tensor shared_topk() const;
    [[nodiscard]] Tensor candidate_blocks() const;

    struct Impl;

private:
    friend void attention_layer(const TextConfig&, std::uint32_t, const AttentionLayerViews&,
                                AttentionState&, const Tensor&, std::int32_t, Tensor&,
                                WorkspaceArena&, cudaStream_t);
    AttentionLimits limits_;
    std::unique_ptr<Impl> impl_;
};

// Upper bound of the workspace one attention_layer call takes (any layer, up to limits.max_chunk
// tokens).
[[nodiscard]] std::size_t attention_workspace_bytes(const TextConfig& config,
                                                    AttentionLimits limits);

// Runs layer `layer`'s attention over the T = x.ne[1] tokens at positions start .. start + T - 1 of
// the sequence `state` holds (state of every earlier position of that sequence must be current, and
// the layer's kv owner and index source must already have run this chunk). x is the layer's normed
// input BF16 [hidden, T]; out BF16 [hidden, T] receives the attention output (before the block's
// residual).
void attention_layer(const TextConfig& config, std::uint32_t layer,
                     const AttentionLayerViews& weights, AttentionState& state, const Tensor& x,
                     std::int32_t start, Tensor& out, WorkspaceArena& work, cudaStream_t stream);

} // namespace ninfer::models::dsv4
