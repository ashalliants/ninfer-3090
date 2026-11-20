// Written for NInfer-3090. The pattern follows src/models/qwen4_exp/config.cpp.

#include "models/dsv4/config.h"

#include "artifact/schema.h"

#include <cmath>
#include <limits>
#include <string>

namespace ninfer::models::dsv4 {
namespace {

using artifact::ArtifactError;
using artifact::Json;

constexpr const char* kArchitecture = "deepseek41";
constexpr const char* kModelType    = "deepseek41_text";

// The model repo's constants for the two-level selection (inference/model.py), not carried by the GGUF.
constexpr std::uint32_t kCandidateBlocks    = 2048;
constexpr std::uint32_t kCandidateBlockSize = 8;

std::uint32_t integer(const Json& value, std::string_view label, bool positive = true) {
    const auto n = artifact::require_u64(value, label, positive);
    if (n > std::numeric_limits<std::uint32_t>::max()) { throw ArtifactError(std::string(label) + ": exceeds u32"); }
    return static_cast<std::uint32_t>(n);
}

std::uint32_t dimension(const Json& config, const char* name) { return integer(config.at(name), name); }

float real(const Json& config, const char* name, bool positive = true) {
    const auto& value = config.at(name);
    if (!value.is_number()) { throw ArtifactError(std::string(name) + " must be a real value"); }
    const float result = value.get<float>();
    if (!std::isfinite(result) || (positive ? result <= 0 : result < 0)) {
        throw ArtifactError(std::string(name) + " must be finite" + (positive ? " and positive" : " and non-negative"));
    }
    return result;
}

std::vector<std::uint32_t> integers(const Json& config, const char* name) {
    const auto& value = config.at(name);
    if (!value.is_array()) { throw ArtifactError(std::string(name) + " must be a list"); }
    std::vector<std::uint32_t> out;
    for (const auto& item : value) { out.push_back(integer(item, name, false)); }
    return out;
}

bool contains(const std::vector<std::uint32_t>& ids, std::uint32_t id) {
    for (const auto i : ids) {
        if (i == id) { return true; }
    }
    return false;
}

TextConfig text(const Json& value) {
    if (value.at("architecture") != kArchitecture || value.at("model_type") != kModelType) {
        throw ArtifactError("text component is not DeepSeek-V4.1-Flash");
    }
    TextConfig out;
    out.hidden_size             = dimension(value, "hidden_size");
    out.vocab_size              = dimension(value, "vocab_size");
    out.num_hidden_layers       = dimension(value, "num_hidden_layers");
    out.max_position_embeddings = dimension(value, "max_position_embeddings");
    out.rms_norm_eps            = real(value, "rms_norm_eps");
    out.heads                   = dimension(value, "num_attention_heads");
    out.head_dim                = dimension(value, "head_dim");
    out.rope_dim                = dimension(value, "rope_dim");
    out.q_lora_rank             = dimension(value, "q_lora_rank");
    out.o_groups                = dimension(value, "o_groups");
    out.o_lora_rank             = dimension(value, "o_lora_rank");
    out.window                  = dimension(value, "sliding_window");
    out.rope_theta              = real(value, "rope_theta");
    out.compress_rope_theta     = real(value, "compress_rope_theta");
    if (dimension(value, "num_key_value_heads") != 1 || dimension(value, "value_head_dim") != out.head_dim) {
        throw ArtifactError("one shared KV head with value width equal to the head width is required");
    }
    if (out.rope_dim % 2 || out.rope_dim > out.head_dim || (out.heads * out.head_dim) % out.o_groups) {
        throw ArtifactError("invalid attention geometry");
    }
    const auto& scaling = value.at("rope_scaling");
    if (scaling.at("type") != "yarn") { throw ArtifactError("rope_scaling must be yarn"); }
    out.yarn = {real(scaling, "factor"), dimension(scaling, "original_max_position_embeddings"),
                real(scaling, "beta_fast"), real(scaling, "beta_slow")};

    out.indexer_heads    = dimension(value, "indexer_heads");
    out.indexer_head_dim = dimension(value, "indexer_head_dim");
    out.indexer_top_k    = dimension(value, "indexer_top_k");

    out.experts           = dimension(value, "num_experts");
    out.experts_per_token = dimension(value, "num_experts_per_tok");
    out.moe_intermediate  = dimension(value, "moe_intermediate_size");
    out.route_scale       = real(value, "routed_scaling_factor");
    out.norm_topk         = value.at("norm_topk_prob") == true;
    if (out.experts_per_token > out.experts) { throw ArtifactError("selected experts exceed expert count"); }
    if (dimension(value, "expert_gating_func") != 4 || dimension(value, "num_shared_experts") != 1) {
        throw ArtifactError("sqrt-softplus gating and one shared expert are required");
    }

    out.hc_streams    = dimension(value, "hc_count");
    out.hc_iterations = dimension(value, "hc_sinkhorn_iterations");
    out.hc_epsilon    = real(value, "hc_epsilon");

    const auto ratios   = integers(value, "compress_ratios");
    const auto kv_src   = integers(value, "kv_source_layers");
    const auto idx_src  = integers(value, "indexer_source_layers");
    const auto engram   = integers(value, "engram_layers");
    const auto& clamp_e = value.at("swiglu_clamp_experts");
    const auto& clamp_s = value.at("swiglu_clamp_shared");
    const std::size_t n = out.num_hidden_layers;
    if (ratios.size() != n || clamp_e.size() != n || clamp_s.size() != n) {
        throw ArtifactError("per-layer lists must cover every layer");
    }
    for (std::uint32_t i = 0; i < n; ++i) {
        LayerPlan layer;
        layer.compress_ratio = ratios[i];
        layer.kv_source      = contains(kv_src, i);
        layer.index_source   = contains(idx_src, i);
        layer.engram         = contains(engram, i);
        layer.clamp_experts  = clamp_e[i].get<float>();
        layer.clamp_shared   = clamp_s[i].get<float>();
        if (layer.kv_source && !layer.compress_ratio) { throw ArtifactError("a kv source layer needs a compress ratio"); }
        if (layer.index_source && !layer.compress_ratio) { throw ArtifactError("an index source layer needs a compress ratio"); }
        out.layers.push_back(layer);
    }
    std::int32_t owner = -1;
    for (std::uint32_t i = 0; i < n; ++i) {
        if (out.layers[i].kv_source) { owner = static_cast<std::int32_t>(i); }
        if (out.layers[i].compress_ratio && owner < 0) { throw ArtifactError("a compressing layer precedes every kv source"); }
        out.kv_owner.push_back(out.layers[i].compress_ratio ? static_cast<std::uint32_t>(owner) : 0);
        if (out.layers[i].compress_ratio && out.layers[out.kv_owner.back()].compress_ratio != out.layers[i].compress_ratio) {
            throw ArtifactError("a layer's compress ratio differs from its kv source's");
        }
    }
    out.index_owner = out.kv_owner;
    for (std::uint32_t i = 0; i < n; ++i) {
        if (out.candidate_layer < 0 && out.layers[i].kv_source && out.layers[i].compress_ratio == 1) {
            out.candidate_layer = static_cast<std::int32_t>(i);
        }
    }
    if (out.candidate_layer >= 0) {
        out.candidate_blocks     = kCandidateBlocks;
        out.candidate_block_size = kCandidateBlockSize;
    }

    out.engram.ngram_size = dimension(value, "engram_ngram_size");
    out.engram.heads      = dimension(value, "engram_heads_per_ngram");
    out.engram.row_width  = dimension(value, "engram_row_width");
    out.engram.pad_token  = static_cast<std::int32_t>(integer(value.at("engram_pad_token_id"), "engram_pad_token_id", false));
    for (const auto& v : value.at("engram_head_vocab_sizes")) {
        out.engram.head_vocab_sizes.push_back(artifact::require_u64(v, "engram_head_vocab_sizes", true));
    }
    if (out.engram.head_vocab_sizes.size() != engram.size() * out.engram.heads * (out.engram.ngram_size - 1)) {
        throw ArtifactError("engram_head_vocab_sizes must hold one set of hashed heads per Engram layer");
    }
    if (value.contains("engram_table")) {
        out.engram.table_rows = artifact::require_u64(value.at("engram_table").at("rows"), "engram_table.rows", true);
    }
    return out;
}

} // namespace

bool is_dsv4_config(const artifact::Directory& directory) {
    if (!directory.components.contains("text")) { return false; }
    const auto& config = directory.component("text").config;
    return config.contains("model_type") && config.at("model_type") == kModelType;
}

TextConfig parse_config(const Json& config) {
    try {
        return text(config);
    } catch (const std::exception& error) {
        throw ArtifactError(std::string("DeepSeek-V4 config: ") + error.what());
    }
}

TextConfig parse_config(const artifact::Directory& directory) { return parse_config(directory.component("text").config); }

} // namespace ninfer::models::dsv4
