#pragma once

// Adapted from Infernix a3edb450 src/models/qwen4_exp/model.h and load.h (Apache-2.0).
// Modified for NInfer-3090: GGML expert record banks (each layer's record format, no per-expert
// scales); no frontend resources, expert store, vision or MTP (the M0 model runs through the
// real-artifact test; the Program, Engine and frontend arrive with PR 9).

#include "artifact/materializer.h"
#include "core/device.h"
#include "models/qwen4_exp/config.h"
#include "models/qwen4_exp/weights.h"

#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>
#include <vector>

namespace ninfer::artifact {
class Reader;
}

namespace ninfer::models::qwen4_exp {

// The routed experts of one MoE layer: the pinned host bank of ggml_expert_record_v1 records, read
// by the GPU through `device_records` (the same memory, device-addressable) and by the CPU through
// `host_records`.
struct ExpertBank {
    QType format                        = QType::BF16; // a ggml_rec_*_q2_0 record format
    std::uint32_t experts               = 0;
    std::uint64_t record_stride         = 0;
    const std::uint8_t* host_records    = nullptr;
    const std::uint8_t* device_records  = nullptr;
};

class Model {
public:
    ~Model();
    Model(const Model&)            = delete;
    Model& operator=(const Model&) = delete;

    [[nodiscard]] const TextConfig& config() const noexcept { return config_; }
    [[nodiscard]] const TextWeights& weights() const noexcept { return weights_; }
    [[nodiscard]] const BoundWeight& weight(WeightId id) const { return bound_.at(id.index); }
    [[nodiscard]] std::span<const ExpertBank> expert_banks() const noexcept { return banks_; }
    [[nodiscard]] const artifact::MaterializationStats& storage_stats() const noexcept { return backing_.stats(); }

private:
    friend std::unique_ptr<Model> load_model(const artifact::Reader&, DeviceContext&);
    Model(TextConfig config, TextWeights weights, std::vector<BoundWeight> bound, std::vector<ExpertBank> banks,
          artifact::MaterializedArtifact backing);

    // Borrowers are declared after the backing they view, so they are destroyed first.
    artifact::MaterializedArtifact backing_;
    TextConfig config_;
    TextWeights weights_;
    std::vector<BoundWeight> bound_;
    std::vector<ExpertBank> banks_;
};

// True when the artifact's text component is Qwen4Exp.
[[nodiscard]] bool is_qwen4_exp(const artifact::Reader& reader);

// Binds every parameter (dense weights on the device, expert banks in one pinned host block) and
// materializes them. Device memory is allocated before the pinned block.
[[nodiscard]] std::unique_ptr<Model> load_model(const artifact::Reader& reader, DeviceContext& device);

} // namespace ninfer::models::qwen4_exp
