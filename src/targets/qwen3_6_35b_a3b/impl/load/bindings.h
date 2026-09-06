#pragma once

#include <ninfer/targets/qwen3_6_35b_a3b/package.h>
#include <ninfer/targets/qwen3_6/frontend_resources.h>
#include <ninfer/targets/qwen3_6/model_view.h>
#include <ninfer/targets/qwen3_6/startup_features.h>
#include <ninfer/targets/qwen3_6/vision.h>

#include "artifact/binder.h"
#include "artifact/materializer.h"
#include "core/pipeline_split.h"
#include "core/tensor.h"
#include "ninfer/ops/sparse_moe.h"

#include <array>
#include <cstddef>
#include <optional>
#include <utility>

namespace ninfer::targets::qwen3_6_35b_a3b::detail {

inline constexpr std::size_t kTextLayers          = 40;
inline constexpr std::size_t kFullAttentionLayers = 10;
inline constexpr std::size_t kGdnLayers           = 30;
inline constexpr std::size_t kDFlashLayers        = 6;

struct MoePlan {
    artifact::ObjectHandle router_shared_gate;
    artifact::ObjectHandle routed_gate_up;
    artifact::ObjectHandle routed_down;
    artifact::ObjectHandle shared_gate_up;
    artifact::ObjectHandle shared_down;
};

struct FullAttentionPlan {
    artifact::ObjectHandle query_key_gate_value;
    artifact::ObjectHandle query_norm;
    artifact::ObjectHandle key_norm;
    artifact::ObjectHandle output;
};

struct GdnPlan {
    artifact::ObjectHandle a_log;
    artifact::ObjectHandle dt_bias;
    artifact::ObjectHandle convolution;
    artifact::ObjectHandle a_b_projection;
    artifact::ObjectHandle query_key_value_z;
    artifact::ObjectHandle norm;
    artifact::ObjectHandle output;
};

struct TextLayerPlan {
    artifact::ObjectHandle input_norm;
    FullAttentionPlan attention{};
    GdnPlan gdn{};
    bool is_full_attention = false;
    artifact::ObjectHandle post_attention_norm;
    MoePlan moe;
};

struct MtpPlan {
    artifact::ObjectHandle input_projection;
    artifact::ObjectHandle embedding_norm;
    artifact::ObjectHandle hidden_norm;
    artifact::ObjectHandle input_norm;
    FullAttentionPlan attention;
    artifact::ObjectHandle post_attention_norm;
    MoePlan moe;
    artifact::ObjectHandle final_norm;
};

struct DFlashLayerPlan {
    artifact::ObjectHandle input_norm;
    artifact::ObjectHandle query_key_value;
    artifact::ObjectHandle query_norm;
    artifact::ObjectHandle key_norm;
    artifact::ObjectHandle attention_output;
    artifact::ObjectHandle post_attention_norm;
    artifact::ObjectHandle gate_up;
    artifact::ObjectHandle down;
};

struct DFlashPlan {
    artifact::ObjectHandle feature_projection;
    artifact::ObjectHandle context_norm;
    std::array<DFlashLayerPlan, kDFlashLayers> layers;
    artifact::ObjectHandle final_norm;
};

struct BindingPlan {
    qwen3_6::FrontendResourcePlan frontend;
    qwen3_6::StartupFeatures features;
    artifact::ObjectHandle token_embedding;
    std::array<TextLayerPlan, kTextLayers> text_layers;
    artifact::ObjectHandle final_norm;
    artifact::ObjectHandle output_head;
    artifact::ObjectHandle draft_head;
    artifact::ObjectHandle draft_head_token_ids;
    MtpPlan mtp;
    qwen3_6::VisionBackbonePlan vision_backbone;
    qwen3_6::VisionMergerInputPlan vision_merger_input;
    artifact::ObjectHandle vision_merger_fc2;
    artifact::ObjectHandle vision_merger_fc2_bias;
    qwen3_6::VisionMergerNormPlan vision_merger_norm;
    std::optional<qwen3_6::VisionOverlayLayout> vision_overlay;
    DFlashPlan dflash;
};

struct ArtifactLoadPlan {
    BindingPlan bindings;
    artifact::MaterializationPlan materialization;
};

// ownership selects which layers this pass uploads to the current device. Default-constructed it
// owns the whole model, which is the single-GPU path; for a pipeline split, call this once per
// rank with that rank's ownership and materialize each plan on its own device.
ArtifactLoadPlan bind_artifact(artifact::Binder& binder, qwen3_6::StartupFeatures features,
                               RankOwnership ownership = {});

struct SparseMoePayload {
    ops::SparseMoeWeights op;
};

struct AttentionProjectionPayload {
    Weight query_key_gate_value;
};

struct GdnProjectionPayload {
    Tensor a_log;
    Tensor dt_bias;
    Weight a_b_projection;
    Weight query_key_value_z;
};

using RuntimeModelView =
    qwen3_6::ModelView<AttentionProjectionPayload, GdnProjectionPayload, SparseMoePayload,
                       AttentionProjectionPayload, SparseMoePayload,
                       qwen3_6::DFlashWeights<kDFlashLayers>, kFullAttentionLayers, kGdnLayers>;
using FullAttentionWeights = RuntimeModelView::FullLayer;
using GdnWeights           = RuntimeModelView::GdnLayer;
using MtpWeights           = RuntimeModelView::MtpLayer;
using DFlashWeights        = RuntimeModelView::DFlash;
using DFlashLayerWeights   = qwen3_6::DFlashLayerWeights;

class LoadedModelData {
public:
    // Whole model on one device.
    LoadedModelData(BindingPlan plan, artifact::MaterializedArtifact materialized);
    // One plan and one artifact per pipeline rank. Each layer's weights are read from the artifact
    // belonging to the rank that owns it, so a single RuntimeModelView addresses tensors living in
    // two device arenas -- which works because every layer lives wholly on one device, and is why
    // a layer split needs no per-rank arena inside a single artifact.
    LoadedModelData(std::vector<BindingPlan> plans,
                    std::vector<artifact::MaterializedArtifact> materialized, PipelineSplit split);

    LoadedModelData(const LoadedModelData&)            = delete;
    LoadedModelData& operator=(const LoadedModelData&) = delete;
    LoadedModelData(LoadedModelData&&)                 = delete;
    LoadedModelData& operator=(LoadedModelData&&)      = delete;

    // One per rank; index with split.placement(layer).rank. Single-rank loads hold exactly one.
    std::vector<artifact::MaterializedArtifact> backings;
    PipelineSplit split{kTextLayers};
    qwen3_6::FrontendResources frontend;
    RuntimeModelView runtime;
};

class LoadedModel::Impl {
public:
    Impl(WeightsProfile weights_profile_in, BindingPlan plan,
         artifact::MaterializedArtifact materialized)
        : weights_profile(weights_profile_in), data(std::move(plan), std::move(materialized)) {}

    Impl(WeightsProfile weights_profile_in, std::vector<BindingPlan> plans,
         std::vector<artifact::MaterializedArtifact> materialized, PipelineSplit split)
        : weights_profile(weights_profile_in),
          data(std::move(plans), std::move(materialized), std::move(split)) {}

    WeightsProfile weights_profile;
    LoadedModelData data;
};

} // namespace ninfer::targets::qwen3_6_35b_a3b::detail
