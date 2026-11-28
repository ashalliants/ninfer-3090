#pragma once

// Written for NInfer-3090. Inkling-2-Mini's DFlash2 drafter (docs/maintainer/inkling2-model.md, "DFlash2 drafter"),
// read from its Hugging Face release directory (incoai/Inkling-2-Mini-DFlash2: config.json and model.safetensors).
// The files are data only: the header is parsed as JSON and the tensors' BF16 words copied out by their offsets.
// The drafter borrows the target's embedding, embedding norm and head; DFlash2Program builds its device forms from
// this host copy, which the engine drops once the programs are built.

#include "models/inkling2/config.h"

#include <cstdint>
#include <filesystem>
#include <map>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace ninfer::models::inkling2 {

struct DFlash2Config {
    std::uint32_t layers = 0, hidden_size = 0, intermediate_size = 0, heads = 0, kv_heads = 0, head_dim = 0;
    float rms_norm_eps = 0, rope_theta = 0, final_logit_softcap = 0;
    std::uint32_t sliding_window = 0, block_size = 0, mask_token_id = 0, conv_group = 0;
    // The text Program's tap_visits convention (k < visits: the input of visit k; k == visits: the final normed state),
    // which the release's target_layer_ids (Hugging Face hidden_states indices) already are.
    std::vector<std::uint32_t> target_visits;
};

// Checked against the text config whose embedding, embedding norm and head the drafter borrows, and the geometry its
// kernels take (heads of 128, a two-tap convolution, the selector's top 16 of rank 256, a block of 8).
[[nodiscard]] DFlash2Config parse_dflash2_config(const std::string& config_json, const TextConfig& text);

class DFlash2Release {
public:
    // Reads `directory`'s config.json and model.safetensors (every tensor BF16, each the shape the config gives).
    [[nodiscard]] static std::unique_ptr<DFlash2Release> load(const std::filesystem::path& directory, const TextConfig& text);

    [[nodiscard]] const DFlash2Config& config() const noexcept { return config_; }
    // The release tensor `name`'s BF16 words (row-major, the shape load() checked).
    [[nodiscard]] std::span<const std::uint16_t> words(const std::string& name) const;
    [[nodiscard]] const std::vector<std::uint64_t>& shape(const std::string& name) const;
    // A short digest of the weights (the safetensors header and size), for the startup log.
    [[nodiscard]] const std::string& source() const noexcept { return source_; }

private:
    struct Entry {
        std::vector<std::uint64_t> shape;
        std::size_t offset = 0, count = 0;
    };
    DFlash2Config config_;
    std::vector<std::uint8_t> file_;
    std::map<std::string, Entry> tensors_;
    std::string source_;
};

} // namespace ninfer::models::inkling2
