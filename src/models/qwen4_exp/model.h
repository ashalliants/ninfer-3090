#pragma once

// Adapted from Infernix a3edb450 src/models/qwen4_exp/model.h and load.h (Apache-2.0).
// Modified for NInfer-3090: GGML expert record banks (each layer's record format, no per-expert
// scales); the MTP drafter's Q2_0 bank in device memory; the vision tower pinned in host memory
// with Qwen3.5's overlay layout (Infernix binds it on the device or overlays it the same way); no
// expert store. The load is planned first (the RAM ledger reads the plan's pinned bytes before
// anything is pinned) and then materialized, as Infernix's plan_load.

#include "artifact/materializer.h"
#include "artifact/reader.h"
#include "core/device.h"
#include "models/qwen3_5/config.h"
#include "models/qwen3_5/frontend/resources.h"
#include "models/qwen3_5/weights.h"
#include "models/qwen4_exp/config.h"
#include "models/qwen4_exp/weights.h"

#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace ninfer::models::qwen4_exp {

// The Qwen3.5 frontend (tokenizer, chat template, thinking control) reads the same resources.
using FrontendResources = qwen3_5::FrontendResources;

// The routed experts of one MoE layer: the pinned host bank of ggml_expert_record_v1 records, read
// by the GPU through `device_records` (the same memory, device-addressable) and by the CPU through
// `host_records`. The MTP drafter's bank lives in device memory: `host_records` is null. With partial
// RAM residency (plan_load with partial) the text banks are not loaded at all: both pointers are null
// and the Program reads each record from the artifact at `artifact_offset` + e * record_stride.
struct ExpertBank {
    QType format                        = QType::BF16; // a ggml_rec_*_q2_0 record format
    std::uint32_t experts               = 0;
    std::uint64_t record_stride         = 0;
    const std::uint8_t* host_records    = nullptr;
    const std::uint8_t* device_records  = nullptr;
    std::uint64_t artifact_offset       = 0; // partial RAM residency: record 0's logical artifact offset
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
    // Device bytes of the dense weights, and the page-locked Host bytes (the expert banks; with
    // vision the tower and with hot_embedding the token embedding too).
    [[nodiscard]] std::uint64_t device_bytes() const;
    [[nodiscard]] std::uint64_t pinned_bytes() const;
    // Each layer's expert record stride (the bank's record format).
    [[nodiscard]] const std::vector<std::uint64_t>& expert_record_strides() const;

    struct Impl;

private:
    explicit LoadPlan(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
    friend LoadPlan plan_load(const artifact::Reader&, bool, bool, bool, bool);
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
    // The MTP drafter, when it was loaded (plan_load with mtp).
    [[nodiscard]] const std::optional<MtpWeights>& mtp() const noexcept { return mtp_; }
    [[nodiscard]] const ExpertBank* mtp_bank() const noexcept { return mtp_bank_ ? &*mtp_bank_ : nullptr; }
    // The vision tower, when it was loaded (plan_load with vision): its config, its parameters in
    // the pinned host block, and the groups a weight stream stages (Qwen3.5's overlay layout).
    [[nodiscard]] const std::optional<VisionWeights>& vision() const noexcept { return vision_; }
    [[nodiscard]] const std::optional<qwen3_5::VisionConfig>& vision_config() const noexcept { return vision_config_; }
    [[nodiscard]] const std::optional<qwen3_5::VisionOverlayLayout>& vision_layout() const noexcept {
        return vision_layout_;
    }
    [[nodiscard]] std::span<const std::byte> pinned_weights() const noexcept { return backing_.pinned_block(); }
    [[nodiscard]] const FrontendResources& resources() const noexcept { return resources_; }
    [[nodiscard]] const InstanceInfo& info() const noexcept { return info_; }
    [[nodiscard]] const artifact::MaterializationStats& storage_stats() const noexcept { return backing_.stats(); }

private:
    friend std::unique_ptr<Model> materialize_model(LoadPlan&&, DeviceContext&, const StartupObserver*);
    Model(TextConfig config, TextWeights weights, std::optional<MtpWeights> mtp, std::vector<BoundWeight> bound,
          std::vector<ExpertBank> banks, std::optional<ExpertBank> mtp_bank, std::optional<VisionWeights> vision,
          std::optional<qwen3_5::VisionConfig> vision_config, std::optional<qwen3_5::VisionOverlayLayout> vision_layout,
          FrontendResources resources, InstanceInfo info, artifact::MaterializedArtifact backing);

    // Borrowers are declared after the backing they view, so they are destroyed first.
    artifact::MaterializedArtifact backing_;
    TextConfig config_;
    TextWeights weights_;
    std::optional<MtpWeights> mtp_;
    std::vector<BoundWeight> bound_;
    std::vector<ExpertBank> banks_;
    std::optional<ExpertBank> mtp_bank_;
    std::optional<VisionWeights> vision_;
    std::optional<qwen3_5::VisionConfig> vision_config_;
    std::optional<qwen3_5::VisionOverlayLayout> vision_layout_;
    FrontendResources resources_;
    InstanceInfo info_;
};

// True when the artifact's text component is Qwen4Exp.
[[nodiscard]] bool is_qwen4_exp(const artifact::Reader& reader);

// True when the artifact holds the MTP drafter (component mtp, architecture Qwen4ExpMTP).
[[nodiscard]] bool has_mtp(const artifact::Reader& reader);

// True when the artifact holds the vision tower (component vision).
[[nodiscard]] bool has_vision(const artifact::Reader& reader);

// Binds every parameter (dense weights on the device, expert banks in one pinned host block) and
// the frontend resources, reading nothing but the resources. With `mtp` it binds the MTP drafter
// too (the artifact must hold it); without, an artifact's drafter is left unread. With `vision` it
// binds the vision tower in the pinned block after the expert banks and the frontend's vision
// resources (the artifact must hold the tower). With `hot_embedding` (--hot-embedding-rows) the token
// embedding is bound in the pinned block (before the banks) instead of device memory; the Program
// copies its hot rows to the device and the forward reads the rest through the mapped block. With
// `partial` (--partial-ram-residency) the text model's expert banks are neither pinned nor read: each
// ExpertBank names its records' place in the artifact, and the Program reads them into its VRAM frames
// and RAM slots.
[[nodiscard]] LoadPlan plan_load(const artifact::Reader& reader, bool mtp = false, bool vision = false,
                                 bool hot_embedding = false, bool partial = false);

// Materializes a plan. Device memory is allocated before the pinned block.
[[nodiscard]] std::unique_ptr<Model> materialize_model(LoadPlan&& plan, DeviceContext& device,
                                                       const StartupObserver* observer = nullptr);

// plan_load then materialize_model.
[[nodiscard]] std::unique_ptr<Model> load_model(const artifact::Reader& reader, DeviceContext& device,
                                                bool mtp = false, bool vision = false, bool hot_embedding = false);

} // namespace ninfer::models::qwen4_exp
