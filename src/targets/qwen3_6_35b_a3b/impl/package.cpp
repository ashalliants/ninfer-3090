#include <ninfer/targets/qwen3_6_35b_a3b/package.h>
#include <ninfer/targets/qwen3_6/frontend_resources.h>
#include <ninfer/targets/qwen3_6/prepared_prompt.h>

#include "artifact/reader.h"
#include "targets/qwen3_6_35b_a3b/impl/load/bindings.h"
#include "targets/qwen3_6_35b_a3b/impl/variant.h"

#include <stdexcept>
#include <utility>

namespace ninfer::targets::qwen3_6_35b_a3b::detail {

class LoadPlan::Impl {
public:
    Impl(WeightsProfile weights_profile_in, ArtifactLoadPlan target_plan)
        : weights_profile(weights_profile_in), plan(std::move(target_plan)) {}

    WeightsProfile weights_profile;
    ArtifactLoadPlan plan;
};

LoadPlan::LoadPlan(std::unique_ptr<Impl> impl) noexcept : impl_(std::move(impl)) {}

LoadPlan::LoadPlan(LoadPlan&&) noexcept            = default;
LoadPlan& LoadPlan::operator=(LoadPlan&&) noexcept = default;
LoadPlan::~LoadPlan()                              = default;

std::size_t LoadPlan::overlay_staging_bytes() const {
    if (impl_ == nullptr || !impl_->plan.bindings.vision_overlay) { return 0; }
    return impl_->plan.bindings.vision_overlay->staging_bytes;
}

const artifact::MaterializationPlan& LoadPlan::materialization() const {
    if (impl_ == nullptr) { throw std::logic_error("target load plan is empty"); }
    return impl_->plan.materialization;
}

LoadedModel::LoadedModel(std::unique_ptr<Impl> impl) noexcept : impl_(std::move(impl)) {}

LoadedModel::~LoadedModel() = default;

} // namespace ninfer::targets::qwen3_6_35b_a3b::detail

namespace ninfer::targets::qwen3_6_35b_a3b {
namespace {

constexpr ModelSamplingDefaults kQwen3_6_35BA3BDefaults{
    .thinking     = {.temperature       = 1.0F,
                     .top_k             = 20,
                     .top_p             = 0.95F,
                     .min_p             = 0.0F,
                     .presence_penalty  = 1.5F,
                     .frequency_penalty = 0.0F},
    .non_thinking = {.temperature       = 0.7F,
                     .top_k             = 20,
                     .top_p             = 0.80F,
                     .min_p             = 0.0F,
                     .presence_penalty  = 1.5F,
                     .frequency_penalty = 0.0F},
};

} // namespace

ModelSamplingDefaults Package::sampling_defaults(std::string_view model) {
    if (model == model_id) { return kQwen3_6_35BA3BDefaults; }
    throw std::runtime_error("model '" + std::string(model) +
                             "' has no sampling defaults in target package '" +
                             std::string(target_key) + "'");
}

Package::WeightsProfile Package::resolve_weights(const artifact::ArtifactIdentity& identity) {
    if (identity.model_id == model_id && identity.weights_id == "groupwise-int") {
        return WeightsProfile::GroupwiseInt;
    }
    throw std::runtime_error("artifact identity '" + identity.model_id + "/" + identity.weights_id +
                             "' is not supported by target '" + std::string(target_key) + "'");
}

PipelineSplit Package::pipeline_split(std::size_t ranks) {
    // Offload every expert block. The MoE is ~88% of this model's weights (17.3 GiB of 19.6), so
    // moving all of it leaves rank 0 holding only embeddings, attention, GDN, norms and the head
    // -- roughly 2.3 GiB -- and turns the other ~21 GiB into KV on the card that serves attention.
    return PipelineSplit::experts_on_last(static_cast<std::uint32_t>(detail::kTextLayers), ranks);
}

Package::LoadPlan Package::plan_load(artifact::Binder& binder, const EngineOptions& options,
                                     WeightsProfile weights_profile) {
    return plan_load(binder, options, weights_profile, RankOwnership{});
}

Package::LoadPlan Package::plan_load(artifact::Binder& binder, const EngineOptions& options,
                                     WeightsProfile weights_profile, RankOwnership ownership) {
    return LoadPlan(std::make_unique<LoadPlan::Impl>(
        weights_profile,
        detail::bind_artifact(binder, qwen3_6::startup_features(options), ownership)));
}

std::unique_ptr<Package::LoadedModel>
Package::construct_loaded_model(LoadPlan&& plan, artifact::MaterializedArtifact&& materialized) {
    if (plan.impl_ == nullptr) { throw std::invalid_argument("target load plan is empty"); }
    auto impl = std::make_unique<LoadedModel::Impl>(
        plan.impl_->weights_profile, std::move(plan.impl_->plan.bindings), std::move(materialized));
    plan.impl_.reset();
    return std::unique_ptr<LoadedModel>(new LoadedModel(std::move(impl)));
}

std::unique_ptr<Package::LoadedModel>
Package::construct_loaded_model(std::vector<LoadPlan>&& plans,
                                std::vector<artifact::MaterializedArtifact>&& materialized,
                                PipelineSplit split) {
    if (plans.empty()) { throw std::invalid_argument("target load plan is empty"); }
    if (plans.size() != materialized.size()) {
        throw std::invalid_argument("one materialized artifact is required per pipeline rank");
    }

    std::vector<detail::BindingPlan> bindings;
    bindings.reserve(plans.size());
    const WeightsProfile weights_profile = plans.front().impl_->weights_profile;
    for (LoadPlan& plan : plans) {
        if (plan.impl_ == nullptr) { throw std::invalid_argument("target load plan is empty"); }
        if (plan.impl_->weights_profile != weights_profile) {
            throw std::invalid_argument("pipeline ranks disagree on the weights profile");
        }
        bindings.push_back(std::move(plan.impl_->plan.bindings));
        plan.impl_.reset();
    }

    auto impl = std::make_unique<LoadedModel::Impl>(weights_profile, std::move(bindings),
                                                    std::move(materialized), std::move(split));
    return std::unique_ptr<LoadedModel>(new LoadedModel(std::move(impl)));
}

Package::Frontend Package::make_frontend(const LoadedModel& model, const EngineOptions& options) {
    if (model.impl_ == nullptr) { throw std::invalid_argument("loaded model is empty"); }
    return qwen3_6::make_frontend(
        model.impl_->data.frontend,
        qwen3_6::FrontendOptions{
            .vision_enabled                = model.impl_->data.runtime.features.vision,
            .max_context                   = options.max_context,
            .media_cache_bytes             = options.media_cache_bytes,
            .media_live_bytes              = options.media_live_bytes,
            .media_preprocess_threads      = options.media_preprocess_threads,
            .max_cache_markers_per_request = *options.context_cache.max_cache_markers_per_request,
            .vision_max_merged_tokens = options.vision_max_merged_tokens,
        });
}

Package::SequencePlanner Package::make_sequence_planner(DeviceContext& device,
                                                        const EngineOptions& options,
                                                        WeightsProfile weights_profile) {
    return qwen3_6::make_sequence_planner<detail::Variant>(device, options, weights_profile);
}

std::unique_ptr<Package::Program> Package::create_program(const LoadedModel& model,
                                                          SequencePlan&& plan,
                                                          DeviceContext& device,
                                                          const StartupObserver& startup_observer) {
    if (model.impl_ == nullptr) { throw std::invalid_argument("loaded model is empty"); }
    return qwen3_6::create_program<detail::Variant>(model.impl_->data.runtime,
                                                    model.impl_->weights_profile, std::move(plan),
                                                    device, startup_observer);
}

} // namespace ninfer::targets::qwen3_6_35b_a3b
