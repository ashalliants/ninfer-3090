#pragma once

// Written for NInfer-3090. The pattern follows src/models/qwen4_exp/config.h.
//
// DeepSeek-V4.1-Flash instance configuration as recorded by the converter (tools/convert/dsv4.py
// `text_config`, model_type deepseek41_text). The mathematics is fixed by this implementation (see
// docs/maintainer/dsv4-flash-plan.md and tools/dsv4/*.py); the config supplies dimensions and each
// layer's structure.

#include "artifact/schema.h"

#include <cstdint>
#include <vector>

namespace ninfer::models::dsv4 {

// What one layer's attention reads and writes beyond its sliding window.
struct LayerPlan {
    std::uint32_t compress_ratio = 0; // 0: window only; 1: one latent per token; r > 1: r tokens pooled into one
    bool kv_source               = false; // owns the compressor (and its index keys): writes the shared compressed KV
    bool index_source            = false; // owns an indexer query: selects positions for itself and later layers
    bool engram                  = false; // an Engram injection precedes the layer
    float clamp_experts          = 0;     // routed experts' SwiGLU clamp (0: none)
    float clamp_shared           = 0;     // shared expert's SwiGLU clamp (0: none)
};

struct YarnConfig {
    float factor                   = 0;
    std::uint32_t original_context = 0;
    float beta_fast                = 0;
    float beta_slow                = 0;
};

struct EngramConfig {
    std::uint32_t ngram_size = 0;
    std::uint32_t heads      = 0; // per n-gram order
    std::uint32_t row_width  = 0;
    std::int32_t pad_token   = 0;
    std::vector<std::uint64_t> head_vocab_sizes; // per Engram layer, heads x (ngram_size - 1) entries
    std::uint64_t table_rows = 0;
};

struct TextConfig {
    std::uint32_t hidden_size             = 0;
    std::uint32_t vocab_size              = 0;
    std::uint32_t num_hidden_layers       = 0;
    std::uint32_t max_position_embeddings = 0;
    float rms_norm_eps                    = 0;

    std::uint32_t heads        = 0; // query heads over the one shared KV head
    std::uint32_t head_dim     = 0; // also the value width
    std::uint32_t rope_dim     = 0; // trailing elements of a head that rotate
    std::uint32_t q_lora_rank  = 0;
    std::uint32_t o_groups     = 0;
    std::uint32_t o_lora_rank  = 0;
    std::uint32_t window       = 0;
    float rope_theta           = 0; // window-only layers
    float compress_rope_theta  = 0; // layers with a compress ratio (YaRN)
    YarnConfig yarn;

    std::uint32_t indexer_heads    = 0;
    std::uint32_t indexer_head_dim = 0;
    std::uint32_t indexer_top_k    = 0;

    std::uint32_t experts           = 0;
    std::uint32_t experts_per_token = 0;
    std::uint32_t moe_intermediate  = 0;
    float route_scale               = 0;
    bool norm_topk                  = false;

    std::uint32_t hc_streams    = 0;
    std::uint32_t hc_iterations = 0;
    float hc_epsilon            = 0;

    std::vector<LayerPlan> layers;
    // The layer whose compressed KV layer i reads (itself when it is a kv source): the nearest kv source at or before i.
    std::vector<std::uint32_t> kv_owner;
    // The layer whose indexer keys layer i's selection reads: its kv owner.
    std::vector<std::uint32_t> index_owner;
    EngramConfig engram;
    // Two-level selection (tools/dsv4/attention.py `select_candidate_blocks`): the first ratio-1 kv source keeps its
    // best `candidate_blocks` blocks of `candidate_block_size` positions, and later index layers search only inside
    // them. The model repo fixes these (they are not in the GGUF); with at most 2048 x 8 positions nothing is masked.
    std::int32_t candidate_layer           = -1;
    std::uint32_t candidate_blocks         = 0;
    std::uint32_t candidate_block_size     = 0;

    [[nodiscard]] std::uint32_t residual_width() const noexcept { return hc_streams * hidden_size; }
    [[nodiscard]] std::uint32_t query_width() const noexcept { return heads * head_dim; }
};

// True when a text config names DeepSeek-V4.1-Flash (model_type deepseek41_text).
[[nodiscard]] bool is_dsv4_config(const artifact::Directory& directory);

[[nodiscard]] TextConfig parse_config(const artifact::Directory& directory);

// The same from the text component's config JSON (what the converter writes).
[[nodiscard]] TextConfig parse_config(const artifact::Json& text);

} // namespace ninfer::models::dsv4
