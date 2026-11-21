// Written for NInfer-3090. The pattern follows src/models/dsv4/config.cpp.

#include "models/inkling2/config.h"

#include <cmath>
#include <limits>
#include <string>

namespace ninfer::models::inkling2 {
namespace {

using artifact::ArtifactError;
using artifact::Json;

constexpr const char* kArchitecture = "inkling2";
constexpr const char* kModelType    = "inkling2_text";

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

TextConfig text(const Json& v) {
    if (v.at("architecture") != kArchitecture || v.at("model_type") != kModelType) {
        throw ArtifactError("text component is not Inkling-2");
    }
    TextConfig c;
    c.hidden_size             = dimension(v, "hidden_size");
    c.num_hidden_layers       = dimension(v, "num_hidden_layers");
    c.num_layer_visits        = dimension(v, "num_layer_visits");
    c.loop_core_start         = integer(v.at("loop_core_start"), "loop_core_start", false);
    c.loop_core_end           = dimension(v, "loop_core_end");
    c.loop_num_passes         = dimension(v, "loop_num_passes");
    c.loop_gate_rank          = dimension(v, "loop_gate_rank");
    c.vocab_size              = dimension(v, "vocab_size");
    c.unpadded_vocab_size     = dimension(v, "unpadded_vocab_size");
    c.max_position_embeddings = dimension(v, "max_position_embeddings");
    c.rms_norm_eps            = real(v, "rms_norm_eps");
    c.heads                   = dimension(v, "num_attention_heads");
    c.kv_heads                = dimension(v, "num_key_value_heads");
    c.head_dim                = dimension(v, "head_dim");
    c.sliding_window          = dimension(v, "sliding_window");
    c.d_rel                   = dimension(v, "d_rel");
    c.rel_extent              = dimension(v, "rel_extent");
    c.qk_clip                 = real(v, "qk_clip");
    c.rel_clip                = real(v, "rel_clip");
    c.log_scaling_n_floor     = dimension(v, "log_scaling_n_floor");
    c.log_scaling_alpha       = real(v, "log_scaling_alpha", false);
    c.conv_kernel_size        = dimension(v, "conv_kernel_size");
    c.attn_res_block_size     = dimension(v, "attn_res_block_size");
    c.experts                 = dimension(v, "num_experts");
    c.experts_per_token       = dimension(v, "num_experts_per_tok");
    c.moe_intermediate        = dimension(v, "moe_intermediate_size");
    c.route_scale             = real(v, "route_scale");
    c.gate_softcap            = real(v, "mlp_gate_softcap");
    c.up_softcap              = real(v, "mlp_up_softcap");
    c.final_logit_softcap     = real(v, "final_logit_softcapping");

    if (c.loop_core_start >= c.loop_core_end || c.loop_core_end > c.num_hidden_layers) {
        throw ArtifactError("the loop core must be a non-empty range of layers");
    }
    if (c.num_layer_visits != c.num_hidden_layers + (c.loop_num_passes - 1) * c.core_layers()) {
        throw ArtifactError("num_layer_visits differs from the loop schedule");
    }
    if (c.unpadded_vocab_size > c.vocab_size) { throw ArtifactError("unpadded_vocab_size exceeds vocab_size"); }
    if (c.heads % c.kv_heads) { throw ArtifactError("query heads are not a multiple of KV heads"); }
    if (c.experts_per_token > c.experts) { throw ArtifactError("selected experts exceed expert count"); }
    // MIX at the attention input sees the blocks pushed so far plus the current state; every block boundary
    // pushes one. The depth-attention Op takes at most eight sources.
    if (c.num_hidden_layers / c.attn_res_block_size + 1 > 8) { throw ArtifactError("too many attention-residual blocks"); }
    c.local.assign(c.num_hidden_layers, false);
    for (const auto& item : v.at("local_layers")) {
        const auto layer = integer(item, "local_layers", false);
        if (layer >= c.num_hidden_layers) { throw ArtifactError("local_layers names a layer past the last one"); }
        c.local[layer] = true;
    }
    return c;
}

} // namespace

std::uint32_t TextConfig::physical(std::uint32_t visit) const {
    if (visit >= num_layer_visits) { throw std::out_of_range("layer visit out of range"); }
    if (visit < loop_core_start) { return visit; }
    if (visit < loop_core_start + loop_num_passes * core_layers()) { return loop_core_start + (visit - loop_core_start) % core_layers(); }
    return visit - (loop_num_passes - 1) * core_layers();
}

std::uint32_t TextConfig::pass(std::uint32_t visit) const {
    if (visit >= loop_core_start && visit < loop_core_start + loop_num_passes * core_layers()) {
        return (visit - loop_core_start) / core_layers();
    }
    return 0;
}

bool is_inkling2_config(const artifact::Directory& directory) {
    if (!directory.components.contains("text")) { return false; }
    const auto& config = directory.component("text").config;
    return config.contains("model_type") && config.at("model_type") == kModelType;
}

TextConfig parse_config(const Json& config) {
    try {
        return text(config);
    } catch (const std::exception& error) {
        throw ArtifactError(std::string("Inkling-2 config: ") + error.what());
    }
}

TextConfig parse_config(const artifact::Directory& directory) { return parse_config(directory.component("text").config); }

DSparkConfig parse_dspark_config(const Json& v, const TextConfig& text) {
    try {
        DSparkConfig c;
        c.layers              = dimension(v, "num_layers");
        c.hidden_size         = dimension(v, "hidden_size");
        c.intermediate_size   = dimension(v, "intermediate_size");
        c.heads               = dimension(v, "num_attention_heads");
        c.kv_heads            = dimension(v, "num_key_value_heads");
        c.head_dim            = dimension(v, "head_dim");
        c.rms_norm_eps        = real(v, "rms_norm_eps");
        c.rope_theta          = real(v, "rope_theta");
        c.final_logit_softcap = real(v, "final_logit_softcapping");
        c.sliding_window      = dimension(v, "sliding_window");
        c.block_size          = dimension(v, "block_size");
        c.markov_rank         = dimension(v, "markov_rank");
        c.mask_token_id       = integer(v.at("mask_token_id"), "mask_token_id", false);
        c.local.assign(c.layers, false);
        for (const auto& item : v.at("local_layers")) {
            const auto layer = integer(item, "local_layers", false);
            if (layer >= c.layers) { throw ArtifactError("local_layers names a layer past the last one"); }
            c.local[layer] = true;
        }
        for (const auto& item : v.at("target_visits")) { c.target_visits.push_back(integer(item, "target_visits", false)); }
        if (c.hidden_size != text.hidden_size) { throw ArtifactError("hidden_size differs from the text model's"); }
        // The drafter's Ops (include/ninfer/ops/dspark_fp32.h) take heads of 128 and at most 8 taps.
        if (c.heads % c.kv_heads || c.head_dim != 128) { throw ArtifactError("the drafter's attention takes heads of 128"); }
        if (c.target_visits.size() > 8) { throw ArtifactError("at most 8 target_visits"); }
        if (c.block_size < 2) { throw ArtifactError("a block needs a draft position"); }
        if (c.mask_token_id >= text.unpadded_vocab_size) { throw ArtifactError("mask_token_id is not a token"); }
        if (c.final_logit_softcap != text.final_logit_softcap) { throw ArtifactError("softcap differs from the text model's"); }
        if (c.target_visits.empty() || c.target_visits.back() != text.num_layer_visits) {
            throw ArtifactError("the last tap must be the final state");
        }
        for (std::size_t i = 1; i < c.target_visits.size(); ++i) {
            if (c.target_visits[i] <= c.target_visits[i - 1]) { throw ArtifactError("target_visits must increase"); }
        }
        return c;
    } catch (const std::exception& error) {
        throw ArtifactError(std::string("Inkling-2 DSpark config: ") + error.what());
    }
}

VisionConfig parse_vision_config(const Json& v, const TextConfig& text) {
    try {
        VisionConfig c;
        c.depth             = dimension(v, "depth");
        c.hidden_size       = dimension(v, "hidden_size");
        c.intermediate_size = dimension(v, "intermediate_size");
        c.heads             = dimension(v, "num_heads");
        c.patch_size        = dimension(v, "patch_size");
        c.position_grid     = dimension(v, "position_grid");
        c.merge             = dimension(v, "spatial_merge_size");
        c.projection_size   = dimension(v, "projection_size");
        c.max_image_tokens  = dimension(v, "max_image_tokens");
        c.layer_norm_eps    = real(v, "layer_norm_eps");
        c.rope_theta        = real(v, "rope_theta");
        for (const auto* key : {"image_mean", "image_std"}) {
            const auto& list = v.at(key);
            if (!list.is_array() || list.size() != 3) { throw ArtifactError(std::string(key) + " must list three values"); }
        }
        for (int i = 0; i < 3; ++i) {
            c.image_mean[i] = v.at("image_mean").at(i).get<float>();
            c.image_std[i]  = v.at("image_std").at(i).get<float>();
            if (!std::isfinite(c.image_mean[i]) || !(c.image_std[i] > 0) || !std::isfinite(c.image_std[i])) {
                throw ArtifactError("image_mean and image_std must be finite, image_std positive");
            }
        }
        if (c.projection_size != text.hidden_size) { throw ArtifactError("projection_size differs from the text hidden size"); }
        if (c.hidden_size % c.heads || c.head_dim() != 64) { throw ArtifactError("the tower's heads must be 64 wide"); }
        if (c.merge != 2) { throw ArtifactError("the merge must be 2 x 2"); }
        return c;
    } catch (const std::exception& error) {
        throw ArtifactError(std::string("Inkling-2 vision config: ") + error.what());
    }
}

} // namespace ninfer::models::inkling2
