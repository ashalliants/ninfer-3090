#pragma once

// Written for NInfer-3090. The pattern follows src/models/qwen4_exp/model.h: the load is planned
// first (the RAM ledger reads the plan's sizes before anything is pinned) and then materialized.

#include "artifact/materializer.h"
#include "artifact/reader.h"
#include "core/device.h"
#include "models/dsv4/config.h"
#include "models/dsv4/weights.h"
#include "models/qwen3_5/frontend/resources.h"

#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace ninfer::models::dsv4 {

// The tokenizer and chat template are the Qwen3.5 frontend's (it already carries the V4 pipeline).
using FrontendResources = qwen3_5::FrontendResources;

// The routed experts of one layer: ggml_expert_record_v1 records. Resident (the default): a pinned
// host bank the GPU reads through `device_records` and the CPU through `host_records`. Left in the
// file (plan_load with `in_file`): both are null and the Program reads record e at
// `artifact_offset` + e * `record_stride` into its VRAM frames and RAM slots.
struct ExpertBank {
    QType format                       = QType::BF16;
    std::uint32_t experts              = 0;
    std::uint64_t record_stride        = 0;
    const std::uint8_t* host_records   = nullptr;
    const std::uint8_t* device_records = nullptr;
    std::uint64_t artifact_offset      = 0;
};

struct InstanceInfo {
    std::string name;
    artifact::ArtifactId artifact_id{};
};

class LoadPlan {
public:
    LoadPlan(LoadPlan&&) noexcept;
    LoadPlan& operator=(LoadPlan&&) noexcept;
    ~LoadPlan();

    [[nodiscard]] const TextConfig& config() const;
    [[nodiscard]] const FrontendResources& resources() const;
    // Device bytes of the dense weights and page-locked host bytes (the expert banks, unless left in the file).
    [[nodiscard]] std::uint64_t device_bytes() const;
    [[nodiscard]] std::uint64_t pinned_bytes() const;
    // Each layer's expert record stride (the bank's record format: GSQ-RCO picks the types per layer).
    [[nodiscard]] const std::vector<std::uint64_t>& expert_record_strides() const;
    // Where each layer's bank lives in the artifact (offset of record 0), for banks left in the file.
    [[nodiscard]] const std::vector<ExpertBank>& file_banks() const;

    struct Impl;

private:
    explicit LoadPlan(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
    friend LoadPlan plan_load(const artifact::Reader&, bool);
    friend class Model;
    friend std::unique_ptr<class Model> materialize_model(LoadPlan&&, DeviceContext&, const StartupObserver*);
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
    [[nodiscard]] const FrontendResources& resources() const noexcept { return resources_; }
    [[nodiscard]] const InstanceInfo& info() const noexcept { return info_; }
    [[nodiscard]] const artifact::MaterializationStats& storage_stats() const noexcept { return backing_.stats(); }

private:
    friend std::unique_ptr<Model> materialize_model(LoadPlan&&, DeviceContext&, const StartupObserver*);
    Model(TextConfig config, TextWeights weights, std::vector<BoundWeight> bound, std::vector<ExpertBank> banks,
          FrontendResources resources, InstanceInfo info, artifact::MaterializedArtifact backing);

    // Borrowers are declared after the backing they view, so they are destroyed first.
    artifact::MaterializedArtifact backing_;
    TextConfig config_;
    TextWeights weights_;
    std::vector<BoundWeight> bound_;
    std::vector<ExpertBank> banks_;
    FrontendResources resources_;
    InstanceInfo info_;
};

// True when the artifact's text component is DeepSeek-V4.1-Flash.
[[nodiscard]] bool is_dsv4(const artifact::Reader& reader);

// Binds every parameter (dense weights on the device, expert banks in one pinned host block, or with
// `in_file` neither pinned nor read) and the frontend resources, reading nothing but the resources.
[[nodiscard]] LoadPlan plan_load(const artifact::Reader& reader, bool in_file = false);

// Materializes a plan. Device memory is allocated before the pinned block.
[[nodiscard]] std::unique_ptr<Model> materialize_model(LoadPlan&& plan, DeviceContext& device,
                                                       const StartupObserver* observer = nullptr);

} // namespace ninfer::models::dsv4
