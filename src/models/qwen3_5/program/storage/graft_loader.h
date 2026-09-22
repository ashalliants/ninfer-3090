#pragma once

#include "core/dtype.h"
#include "core/layout.h"
#include "core/tensor.h"
#include "models/qwen3_5/state/decoder_state.h"

#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>
#include <string>
#include <vector>

// phantom-kv graft artifact loading infrastructure.
// Reads per-layer K/V tensor data from .phantom or .bin files and validates against the active
// model configuration. Loads into host-pinned staging buffers ready for H2D transfer into device
// KV pages during activation.

namespace ninfer::models::qwen3_5 {

/** One layer's worth of raw BF16 K/V data for a phantom graft. */
struct LayerGrafitData {
    std::int32_t layer_index  = 0;
    std::int32_t head_dim     = 0;
    std::int32_t num_kv_heads = 0;
    std::uint32_t slots       = 0; // total phantom tokens this layer contributes

    // H2D staging: [slots, head_dim] each in row-major layout compatible with paged KV storage.
    std::unique_ptr<std::uint8_t[], decltype([](auto* p) { cudaFreeHost(p); })> k_raw;
    std::unique_ptr<std::uint8_t[], decltype([](auto* p) { cudaFreeHost(p); })> v_raw;
    Tensor k_tensor;   // pinned host buffer, dtype=BF16
    Tensor v_tensor;   // pinned host buffer, dtype=BF16
};

/** A validated, loaded graft that can be attached to an active sequence. */
class GraftLoader;

class LoadedGraft {
public:
    LoadedGraft(LoadedGraft&&) noexcept = default;

    [[nodiscard]] std::string const& model_id() const noexcept { return model_id_; }
    [[nodiscard]] std::uint32_t max_slots() const noexcept { return max_slots_; }
    [[nodiscard]] std::int32_t layers() const noexcept { return static_cast<std::int32_t>(layers_.size()); }
    [[nodiscard]] LayerGrafitData const& layer(std::uint32_t i) const { return layers_[i]; }
    [[nodiscard]] std::span<LayerGrafitData const> all_layers() const noexcept { return layers_; }

private:
    friend class GraftLoader;
    LoadedGraft() noexcept = default;
    std::string model_id_;
    std::uint32_t max_slots_ = 0;
    std::vector<LayerGrafitData> layers_;
};

/** Load a single graft artifact file into memory. */
class GraftLoader {
public:
    /** Read a .phantom/.bin file and validate it matches `expected_model_id`. */
    [[nodiscard]] static LoadedGraft load(std::filesystem::path const& path,
                                          std::string_view expected_model_id);

    /** Quick check whether a file looks like a valid graft artifact without full load. */
    [[nodiscard]] static bool probe_header(std::filesystem::path const& path,
                                           std::string& out_model_id);

private:
    // Internal loading implementation; declared here so it remains a friend of LoadedGraft.
    struct FileSpec { std::string model_id; };
    static LoadedGraft load_file(std::filesystem::path const&, FileSpec);
};

} // namespace ninfer::models::qwen3_5
