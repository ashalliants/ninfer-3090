#pragma once

// Written for NInfer-3090. Inkling-2-Mini's text config as the converter writes it (tools/convert/inkling2.py
// text_config), and the layer-visit schedule of its loop (docs/maintainer/inkling2-model.md, "Layer visits").

#include "artifact/schema.h"

#include <cstdint>
#include <vector>

namespace ninfer::models::inkling2 {

struct TextConfig {
    std::uint32_t hidden_size = 0, num_hidden_layers = 0, num_layer_visits = 0;
    std::uint32_t loop_core_start = 0, loop_core_end = 0, loop_num_passes = 0, loop_gate_rank = 0;
    std::uint32_t vocab_size = 0, unpadded_vocab_size = 0, max_position_embeddings = 0;
    float rms_norm_eps = 0;
    std::uint32_t heads = 0, kv_heads = 0, head_dim = 0;
    std::vector<bool> local; // per physical layer
    std::uint32_t sliding_window = 0, d_rel = 0, rel_extent = 0;
    float qk_clip = 0, rel_clip = 0;
    std::uint32_t log_scaling_n_floor = 0;
    float log_scaling_alpha = 0;
    std::uint32_t conv_kernel_size = 0, attn_res_block_size = 0;
    std::uint32_t experts = 0, experts_per_token = 0, moe_intermediate = 0;
    float route_scale = 0, gate_softcap = 0, up_softcap = 0, final_logit_softcap = 0;

    [[nodiscard]] std::uint32_t core_layers() const noexcept { return loop_core_end - loop_core_start; }
    [[nodiscard]] bool is_core(std::uint32_t layer) const noexcept { return layer >= loop_core_start && layer < loop_core_end; }
    // Rows of a layer's relative-bias table (backward distances): the window for a local layer, rel_extent for a
    // global one (tools/convert/inkling2.py expected_shapes).
    [[nodiscard]] std::uint32_t rel_extent_of(std::uint32_t layer) const { return local.at(layer) ? sliding_window : rel_extent; }
    // The physical layer whose weights layer visit `visit` applies, and the loop pass of a core visit (0 otherwise).
    [[nodiscard]] std::uint32_t physical(std::uint32_t visit) const;
    [[nodiscard]] std::uint32_t pass(std::uint32_t visit) const;
};

// The DSpark drafter (component dspark; tools/convert/inkling2_dspark.py dspark_config).
struct DSparkConfig {
    std::uint32_t layers = 0, hidden_size = 0, intermediate_size = 0, heads = 0, kv_heads = 0, head_dim = 0;
    float rms_norm_eps = 0, rope_theta = 0, final_logit_softcap = 0;
    std::uint32_t sliding_window = 0, block_size = 0, markov_rank = 0, mask_token_id = 0;
    std::vector<bool> local;                  // per draft layer
    std::vector<std::uint32_t> target_visits; // tap v < visits: the input of visit v; v == visits: the final normed state
};

// The vision tower (component vision; tools/convert/inkling2_vision.py vision_config).
struct VisionConfig {
    std::uint32_t depth = 0, hidden_size = 0, intermediate_size = 0, heads = 0, patch_size = 0, position_grid = 0;
    std::uint32_t merge = 0, projection_size = 0, max_image_tokens = 0;
    float layer_norm_eps = 0, rope_theta = 0;
    float image_mean[3] = {}, image_std[3] = {};

    [[nodiscard]] std::uint32_t head_dim() const noexcept { return hidden_size / heads; }
    // The image side an image is padded to a multiple of: one merged token's square.
    [[nodiscard]] std::uint32_t align() const noexcept { return patch_size * merge; }
};

[[nodiscard]] bool is_inkling2_config(const artifact::Directory& directory);
[[nodiscard]] TextConfig parse_config(const artifact::Json& config);
[[nodiscard]] TextConfig parse_config(const artifact::Directory& directory);
// Checked against the text config whose embedding, embedding norm and head the drafter borrows.
[[nodiscard]] DSparkConfig parse_dspark_config(const artifact::Json& config, const TextConfig& text);
// Checked against the text config it projects into.
[[nodiscard]] VisionConfig parse_vision_config(const artifact::Json& config, const TextConfig& text);

} // namespace ninfer::models::inkling2
