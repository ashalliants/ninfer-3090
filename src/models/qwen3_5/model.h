#pragma once

#include "artifact/framing.h"
#include "artifact/materializer.h"
#include "core/arena.h"
#include "core/weight.h"
#include "models/qwen3_5/config.h"
#include "models/qwen3_5/frontend/resources.h"
#include "models/qwen3_5/weights.h"
#include "ninfer/ops/weight_input.h"

#include <memory>
#include <optional>
#include <span>
#include <string>

namespace ninfer::models::qwen3_5 {

struct InstanceInfo {
    std::string name;
    std::string metadata_json;
    std::string provenance_json;
    artifact::ArtifactId artifact_id{};
};

class LoadPlan;

// --hot-embedding-rows N (LoadOptions::hot_embedding_rows): the token embedding (Q4_G64_FP16
// RowSplit) is bound in the mapped pinned Host block, and the Model keeps a device copy of its first
// N rows. ops::embedding_split reads ids below N from `hot` and the rest from `cold` over PCIe.
struct HotEmbedding {
    std::int32_t rows = 0;
    // Rows [0, rows) of both planes, laid out as a RowSplit payload of `rows` rows.
    DeviceBuffer buffer;
    Weight hot;
    // The whole table, its planes addressed through their mapped device pointers.
    Weight cold;
    // Bytes of the whole table (pinned) and of the device copy.
    std::uint64_t table_bytes = 0;
    std::uint64_t hot_bytes   = 0;
};

class Model {
public:
    ~Model();
    Model(const Model&)            = delete;
    Model& operator=(const Model&) = delete;
    Model(Model&&)                 = delete;
    Model& operator=(Model&&)      = delete;

    [[nodiscard]] const Config& config() const noexcept { return config_; }

    [[nodiscard]] const LoadOptions& options() const noexcept { return options_; }

    [[nodiscard]] const ModelWeights& weights() const noexcept { return weights_; }

    [[nodiscard]] const BoundWeight& weight(WeightId id) const { return bound_.at(id.index); }

    // A Use whose matrix is Hadamard-rotated is only admitted where the execution layer rotates
    // the activation: input() refuses it, rotated_input() carries its sign vector.
    [[nodiscard]] ops::WeightInput input(WeightUseId id) const;
    [[nodiscard]] ops::WeightInput input(WeightId id) const;
    [[nodiscard]] ops::WeightInput rotated_input(WeightUseId id) const;
    [[nodiscard]] ops::WeightInput rotated_input(WeightId id) const;

    [[nodiscard]] std::span<const BoundWeight> weight_data() const noexcept { return bound_; }

    [[nodiscard]] const FrontendResources& resources() const noexcept { return resources_; }

    [[nodiscard]] const InstanceInfo& info() const noexcept { return info_; }

    [[nodiscard]] const artifact::MaterializationStats& storage_stats() const noexcept {
        return backing_.stats();
    }

    // Overlay Vision residency only: the pinned tower groups, the pinned block that holds them and
    // the eviction pool behind the weight arena. The pool is shared mutable device state; its
    // single window is arbitrated by the one Program that executes this Model.
    [[nodiscard]] const std::optional<VisionOverlayLayout>& vision_overlay() const noexcept {
        return vision_overlay_;
    }

    // Set with --hot-embedding-rows only. The device copy is not part of the weight arenas, so it
    // stays resident while the Model is suspended (release_device_weights).
    [[nodiscard]] const HotEmbedding* hot_embedding() const noexcept {
        return hot_embedding_ ? &*hot_embedding_ : nullptr;
    }

    [[nodiscard]] std::span<const std::byte> pinned_weights() const noexcept {
        return backing_.pinned_block();
    }

    [[nodiscard]] EvictableWeightPool* weight_pool() const noexcept {
        return backing_.weight_pool();
    }

    // Model suspend: the weight arenas give their device memory back and later receive the same
    // bytes again from the artifact, at the same addresses, so every bound weight and captured
    // graph stays valid. The caller has drained every stream that reads them.
    [[nodiscard]] bool weights_suspendable() const noexcept { return backing_.suspendable(); }
    [[nodiscard]] bool weights_resident() const noexcept { return backing_.device_backed(); }
    [[nodiscard]] std::uint64_t weight_backing_bytes() const noexcept {
        return backing_.device_backing_bytes();
    }
    [[nodiscard]] std::vector<artifact::MaterializedArtifact::RankArena>
    device_weight_arenas() const {
        return backing_.device_arenas();
    }
    void release_device_weights() { backing_.release_device_backing(); }
    void restore_device_weight_backing(DeviceContext& device) {
        backing_.restore_device_backing(device);
    }
    artifact::MaterializationStats reload_device_weights(DeviceContext& device) {
        return backing_.upload_device_objects_again(device);
    }

private:
    friend std::unique_ptr<Model> materialize_model(LoadPlan&&, DeviceContext&,
                                                    const StartupObserver*,
                                                    const artifact::MaterializeOptions&);
    Model(Config config, LoadOptions options, ModelWeights weights, std::vector<BoundWeight> bound,
          FrontendResources resources, InstanceInfo info, artifact::MaterializedArtifact backing,
          std::optional<VisionOverlayLayout> vision_overlay,
          std::optional<HotEmbedding> hot_embedding);

    // Destroy all borrowers before backing. The caller keeps DeviceContext alive through cleanup.
    artifact::MaterializedArtifact backing_;
    Config config_;
    LoadOptions options_;
    ModelWeights weights_;
    std::vector<BoundWeight> bound_;
    FrontendResources resources_;
    InstanceInfo info_;
    std::optional<VisionOverlayLayout> vision_overlay_;
    std::optional<HotEmbedding> hot_embedding_;
};

} // namespace ninfer::models::qwen3_5
