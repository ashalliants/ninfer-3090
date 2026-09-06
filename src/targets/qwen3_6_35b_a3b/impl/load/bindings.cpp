#include "targets/qwen3_6_35b_a3b/impl/load/bindings.h"

#include "artifact/typed_binding.h"
#include "core/evictable_weight_pool.h"

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace ninfer::targets::qwen3_6_35b_a3b::detail {
namespace {

using artifact::NumericFormat;

bool is_full_layer(std::size_t layer) { return layer >= 3 && (layer - 3) % 4 == 0; }

NumericFormat routed_down_format(std::size_t layer) {
    return layer == 34 || layer == 38 || layer == 39 ? NumericFormat::Q6G64_F16S
                                                     : NumericFormat::Q5G64_F16S;
}

Weight row_view(const Weight& block, std::int32_t row_begin, std::int32_t row_count) {
    if (row_begin < 0 || row_count <= 0 || row_begin + row_count > block.n ||
        block.qtype != QType::W8G32_F16S || block.layout != QuantLayout::RowSplit ||
        block.group != 32) {
        throw std::logic_error("invalid DFlash W8 row view");
    }
    const std::uint64_t code_row = static_cast<std::uint64_t>(block.padded_shape[1]);
    const std::uint64_t scale_row =
        static_cast<std::uint64_t>(block.padded_shape[1] / block.group) * sizeof(std::uint16_t);
    Weight out = block;
    out.qdata  = static_cast<const std::byte*>(block.qdata) +
                static_cast<std::uint64_t>(row_begin) * code_row;
    out.scales = static_cast<const std::byte*>(block.scales) +
                 static_cast<std::uint64_t>(row_begin) * scale_row;
    out.n               = row_count;
    out.shape[0]        = row_count;
    out.padded_shape[0] = row_count;
    return out;
}

MoePlan bind_moe(artifact::Binder& binder, const std::string& prefix, NumericFormat routed_gate_up,
                 NumericFormat routed_down, artifact::TensorPlacement placement) {
    const auto bind = [&](std::string_view name, NumericFormat format,
                          std::initializer_list<std::uint64_t> shape) {
        return artifact::bind_tensor(binder, name, format, shape, placement);
    };
    return MoePlan{
        .router_shared_gate = bind(prefix + "router_shared_gate", NumericFormat::BF16, {257, 2048}),
        .routed_gate_up     = bind(prefix + "routed_gate_up", routed_gate_up, {262144, 2048}),
        .routed_down        = bind(prefix + "routed_down", routed_down, {524288, 512}),
        .shared_gate_up = bind(prefix + "shared_gate_up", NumericFormat::W8G32_F16S, {1024, 2048}),
        .shared_down    = bind(prefix + "shared_down", NumericFormat::W8G32_F16S, {2048, 512}),
    };
}

SparseMoePayload load_moe(const MoePlan& plan, const artifact::MaterializedArtifact& materialized,
                          NumericFormat routed_gate_up, NumericFormat routed_down) {
    return SparseMoePayload{
        .op = {
            .router_shared_gate = artifact::materialized_weight(
                materialized, plan.router_shared_gate, NumericFormat::BF16, 257, 2048),
            .routed_gate_up = artifact::materialized_weight(materialized, plan.routed_gate_up,
                                                            routed_gate_up, 262144, 2048),
            .routed_down    = artifact::materialized_weight(materialized, plan.routed_down,
                                                            routed_down, 524288, 512),
            .shared_gate_up = artifact::materialized_weight(materialized, plan.shared_gate_up,
                                                            NumericFormat::W8G32_F16S, 1024, 2048),
            .shared_down    = artifact::materialized_weight(materialized, plan.shared_down,
                                                            NumericFormat::W8G32_F16S, 2048, 512),
        }};
}

void validate_draft_ids(const artifact::Binder& binder, artifact::ObjectHandle handle) {
    constexpr std::size_t kDraftVocab     = 131072;
    constexpr std::size_t kTokenizerVocab = 248077;
    const auto bytes                      = binder.payload(handle).data;
    std::vector<bool> seen(kTokenizerVocab, false);
    for (std::size_t i = 0; i < kDraftVocab; ++i) {
        const std::byte* value = bytes.data() + i * sizeof(std::uint32_t);
        const std::uint32_t id = std::to_integer<std::uint32_t>(value[0]) |
                                 (std::to_integer<std::uint32_t>(value[1]) << 8U) |
                                 (std::to_integer<std::uint32_t>(value[2]) << 16U) |
                                 (std::to_integer<std::uint32_t>(value[3]) << 24U);
        if (id >= kTokenizerVocab || seen[id]) {
            throw artifact::ArtifactError("invalid optimized draft-head token ids");
        }
        seen[id] = true;
    }
}

} // namespace

// Overlay eviction ladder, shared semantics with the 27B target: higher ranks sit at the arena end
// and are borrowed first.
constexpr std::uint32_t kEvictRankMtp       = 400;
constexpr std::uint32_t kEvictRankDraftHead = 500;
constexpr std::uint32_t kEvictRankEmbedding = 600;
constexpr std::uint32_t kEvictRankLmHead    = 700;

ArtifactLoadPlan bind_artifact(artifact::Binder& binder, qwen3_6::StartupFeatures features,
                               RankOwnership ownership) {
    ArtifactLoadPlan load_plan;
    BindingPlan& out   = load_plan.bindings;
    out.frontend       = qwen3_6::bind_frontend_resources(binder);
    out.features       = features;
    const bool overlay = features.overlay_vision();

    // Everything this rank does not own is still bound, but ValidateOnly: the artifact is checked
    // in full against the file while only this rank's bytes are uploaded to this device.
    const auto head_placement = ownership.owns_head() ? artifact::TensorPlacement::Device
                                                      : artifact::TensorPlacement::ValidateOnly;
    const auto embedding_placement = ownership.owns_embedding()
                                         ? artifact::TensorPlacement::Device
                                         : artifact::TensorPlacement::ValidateOnly;

    out.token_embedding =
        artifact::bind_tensor(binder, "text/token_embedding", NumericFormat::W8G32_F16S,
                              {248320, 2048}, embedding_placement,
                              overlay ? kEvictRankEmbedding : 0);

    for (std::size_t layer = 0; layer < kTextLayers; ++layer) {
        TextLayerPlan& target    = out.text_layers[layer];
        const std::string prefix = "text/layers/" + std::to_string(layer) + "/";
        const auto place = ownership.owns_layer(static_cast<std::uint32_t>(layer))
                               ? artifact::TensorPlacement::Device
                               : artifact::TensorPlacement::ValidateOnly;
        const auto bind_layer_tensor = [&](std::string_view name, NumericFormat format,
                                           std::initializer_list<std::uint64_t> shape) {
            return artifact::bind_tensor(binder, name, format, shape, place);
        };

        target.input_norm = bind_layer_tensor(prefix + "input_norm", NumericFormat::BF16, {2048});
        target.is_full_attention = is_full_layer(layer);
        if (target.is_full_attention) {
            target.attention.query_key_gate_value =
                bind_layer_tensor(prefix + "attention/query_key_gate_value",
                                  NumericFormat::W8G32_F16S, {9216, 2048});
            target.attention.query_norm =
                bind_layer_tensor(prefix + "attention/query_norm", NumericFormat::BF16, {256});
            target.attention.key_norm =
                bind_layer_tensor(prefix + "attention/key_norm", NumericFormat::BF16, {256});
            target.attention.output = bind_layer_tensor(
                prefix + "attention/output", NumericFormat::W8G32_F16S, {2048, 4096});
        } else {
            target.gdn.a_log =
                bind_layer_tensor(prefix + "gdn/a_log", NumericFormat::FP32, {32});
            target.gdn.dt_bias =
                bind_layer_tensor(prefix + "gdn/dt_bias", NumericFormat::FP32, {32});
            target.gdn.convolution =
                bind_layer_tensor(prefix + "gdn/convolution", NumericFormat::BF16, {4, 8192});
            target.gdn.a_b_projection =
                bind_layer_tensor(prefix + "gdn/a_b_projection", NumericFormat::BF16, {64, 2048});
            target.gdn.query_key_value_z = bind_layer_tensor(
                prefix + "gdn/query_key_value_z", NumericFormat::W8G32_F16S, {12288, 2048});
            target.gdn.norm = bind_layer_tensor(prefix + "gdn/norm", NumericFormat::BF16, {128});
            target.gdn.output =
                bind_layer_tensor(prefix + "gdn/output", NumericFormat::W8G32_F16S, {2048, 4096});
        }
        target.post_attention_norm =
            bind_layer_tensor(prefix + "post_attention_norm", NumericFormat::BF16, {2048});
        target.moe = bind_moe(binder, prefix + "moe/", NumericFormat::Q4G64_F16S,
                              routed_down_format(layer), place);
    }

    out.final_norm = artifact::bind_tensor(binder, "text/final_norm", NumericFormat::BF16, {2048},
                                           head_placement);
    out.output_head = artifact::bind_tensor(binder, "text/output_head", NumericFormat::Q6G64_F16S,
                                            {248320, 2048}, head_placement,
                                            overlay ? kEvictRankLmHead : 0);
    // The proposal and MTP heads read the final hidden state, so they live with the rank that
    // produces it -- gated by ownership as well as by the feature being enabled at all.
    const artifact::TensorPlacement proposal_placement =
        (features.optimized_proposal() && ownership.owns_head())
            ? artifact::TensorPlacement::Device
            : artifact::TensorPlacement::ValidateOnly;
    out.draft_head = artifact::bind_tensor(binder, "text/draft_head", NumericFormat::Q4G64_F16S,
                                           {131072, 2048}, proposal_placement,
                                           overlay ? kEvictRankDraftHead : 0);
    out.draft_head_token_ids = artifact::bind_tensor(
        binder, "text/draft_head_token_ids", NumericFormat::I32, {131072}, proposal_placement);
    validate_draft_ids(binder, out.draft_head_token_ids);

    const artifact::TensorPlacement mtp_placement =
        (features.mtp() && ownership.owns_head()) ? artifact::TensorPlacement::Device
                                                  : artifact::TensorPlacement::ValidateOnly;
    const auto bind_mtp                           = [&](std::string_view name, NumericFormat format,
                              std::initializer_list<std::uint64_t> shape) {
        return artifact::bind_tensor(binder, name, format, shape, mtp_placement,
                                     overlay ? kEvictRankMtp : 0);
    };
    out.mtp.input_projection =
        bind_mtp("mtp/input_projection", NumericFormat::W8G32_F16S, {2048, 4096});
    out.mtp.embedding_norm = bind_mtp("mtp/embedding_norm", NumericFormat::BF16, {2048});
    out.mtp.hidden_norm    = bind_mtp("mtp/hidden_norm", NumericFormat::BF16, {2048});
    out.mtp.input_norm     = bind_mtp("mtp/layer/input_norm", NumericFormat::BF16, {2048});
    out.mtp.attention.query_key_gate_value = bind_mtp("mtp/layer/attention/query_key_gate_value",
                                                      NumericFormat::W8G32_F16S, {9216, 2048});
    out.mtp.attention.query_norm =
        bind_mtp("mtp/layer/attention/query_norm", NumericFormat::BF16, {256});
    out.mtp.attention.key_norm =
        bind_mtp("mtp/layer/attention/key_norm", NumericFormat::BF16, {256});
    out.mtp.attention.output =
        bind_mtp("mtp/layer/attention/output", NumericFormat::W8G32_F16S, {2048, 4096});
    out.mtp.post_attention_norm =
        bind_mtp("mtp/layer/post_attention_norm", NumericFormat::BF16, {2048});
    out.mtp.moe        = bind_moe(binder, "mtp/layer/moe/", NumericFormat::W8G32_F16S,
                                  NumericFormat::W8G32_F16S, mtp_placement);
    out.mtp.final_norm = bind_mtp("mtp/final_norm", NumericFormat::BF16, {2048});

    // Vision produces the embeddings that enter at layer 0, so the backbone belongs with the rank
    // that owns the embedding. HostPinned overlay weights are not device-resident at all, so they
    // stay pinned on whichever rank owns them rather than being demoted to ValidateOnly.
    const artifact::TensorPlacement vision_placement =
        !ownership.owns_embedding() ? artifact::TensorPlacement::ValidateOnly
        : overlay                   ? artifact::TensorPlacement::HostPinned
        : features.vision           ? artifact::TensorPlacement::Device
                                    : artifact::TensorPlacement::ValidateOnly;
    out.vision_backbone     = qwen3_6::bind_vision_backbone(binder, vision_placement);
    out.vision_merger_input = qwen3_6::bind_vision_merger_input(binder, vision_placement);
    out.vision_merger_fc2   = artifact::bind_tensor(
        binder, "vision/merger/fc2", NumericFormat::W8G32_F16S, {2048, 4608}, vision_placement);
    out.vision_merger_fc2_bias = artifact::bind_tensor(
        binder, "vision/merger/fc2_bias", NumericFormat::BF16, {2048}, vision_placement);
    out.vision_merger_norm = qwen3_6::bind_vision_merger_norm(binder, vision_placement);

    const bool artifact_has_dflash = binder.has_object("dflash/feature_projection");
    if (features.dflash() && !artifact_has_dflash) {
        throw artifact::ArtifactError("DFlash was requested but this compact artifact has no DFlash weights");
    }
    if (artifact_has_dflash) {
        // The DFlash draft model reads the final hidden state, so it lives with the head rank.
        const artifact::TensorPlacement dflash_placement =
            (features.dflash() && ownership.owns_head()) ? artifact::TensorPlacement::Device
                                                         : artifact::TensorPlacement::ValidateOnly;
        const auto bind_dflash = [&](std::string_view name, NumericFormat format,
                                     std::initializer_list<std::uint64_t> shape) {
            return artifact::bind_tensor(binder, name, format, shape, dflash_placement);
        };
        out.dflash.feature_projection =
            bind_dflash("dflash/feature_projection", NumericFormat::W8G32_F16S, {2048, 16384});
        out.dflash.context_norm = bind_dflash("dflash/context_norm", NumericFormat::BF16, {2048});
        for (std::size_t layer = 0; layer < kDFlashLayers; ++layer) {
            DFlashLayerPlan& target  = out.dflash.layers[layer];
            const std::string prefix = "dflash/layers/" + std::to_string(layer) + "/";
            target.input_norm = bind_dflash(prefix + "input_norm", NumericFormat::BF16, {2048});
            target.query_key_value = bind_dflash(prefix + "attention/query_key_value",
                                                  NumericFormat::W8G32_F16S, {6144, 2048});
            target.query_norm = bind_dflash(prefix + "attention/query_norm", NumericFormat::BF16, {128});
            target.key_norm = bind_dflash(prefix + "attention/key_norm", NumericFormat::BF16, {128});
            target.attention_output = bind_dflash(prefix + "attention/output",
                                                   NumericFormat::W8G32_F16S, {2048, 4096});
            target.post_attention_norm = bind_dflash(prefix + "post_attention_norm",
                                                      NumericFormat::BF16, {2048});
            target.gate_up = bind_dflash(prefix + "mlp/gate_up", NumericFormat::W8G32_F16S,
                                         {12288, 2048});
            target.down = bind_dflash(prefix + "mlp/down", NumericFormat::W8G32_F16S, {2048, 6144});
        }
        out.dflash.final_norm = bind_dflash("dflash/final_norm", NumericFormat::BF16, {2048});
    }

    load_plan.materialization =
        binder.finish(overlay ? ninfer::EvictableWeightPool::kChunkBytes : 1);
    if (overlay) {
        out.vision_overlay = qwen3_6::compute_vision_overlay_layout(
            out.vision_backbone, out.vision_merger_input, out.vision_merger_fc2,
            out.vision_merger_fc2_bias, out.vision_merger_norm, load_plan.materialization);
    }
    return load_plan;
}

namespace {

template <typename T>
std::vector<T> single_element(T value) {
    std::vector<T> out;
    out.reserve(1);
    out.push_back(std::move(value));
    return out;
}

} // namespace

LoadedModelData::LoadedModelData(BindingPlan plan, artifact::MaterializedArtifact materialized)
    : LoadedModelData(single_element(std::move(plan)), single_element(std::move(materialized)),
                      PipelineSplit(kTextLayers)) {}

LoadedModelData::LoadedModelData(std::vector<BindingPlan> plans,
                                 std::vector<artifact::MaterializedArtifact> materialized,
                                 PipelineSplit split_in)
    : backings(std::move(materialized)), split(std::move(split_in)) {
    if (plans.empty() || plans.size() != backings.size()) {
        throw std::invalid_argument("one binding plan is required per pipeline rank");
    }
    if (plans.size() != split.ranks()) {
        throw std::invalid_argument("binding plans do not match the pipeline split");
    }

    // The embedding and the vision backbone feed layer 0; the head objects consume the final
    // hidden state. Each therefore reads from the rank that owns that end of the model.
    const BindingPlan& embed_plan               = plans.front();
    const BindingPlan& head_plan                = plans.back();
    artifact::MaterializedArtifact& embed_backing = backings.front();
    artifact::MaterializedArtifact& head_backing  = backings.back();

    frontend = qwen3_6::take_frontend_resources(embed_backing, embed_plan.frontend);

    runtime.weights_arena = &embed_backing.device_arena();
    runtime.features      = embed_plan.features;
    auto& token_embedding = runtime.token_embedding;
    auto& full_layers     = runtime.full_layers;
    auto& gdn_layers      = runtime.gdn_layers;
    auto& final_norm      = runtime.final_norm;
    auto& output_head     = runtime.output_head;

    token_embedding = artifact::materialized_weight(embed_backing, embed_plan.token_embedding,
                                                    NumericFormat::W8G32_F16S, 248320, 2048);

    std::size_t full_index = 0;
    std::size_t gdn_index  = 0;
    for (std::size_t layer = 0; layer < kTextLayers; ++layer) {
        // Shadow `backing` and `source` with the owning rank's pair so the body below reads from
        // whichever device holds this layer without any per-tensor branching.
        const std::size_t owner                = split.placement(static_cast<std::uint32_t>(layer)).rank;
        artifact::MaterializedArtifact& backing = backings[owner];
        const TextLayerPlan& source             = plans[owner].text_layers[layer];
        if (source.is_full_attention) {
            FullAttentionWeights& target = full_layers.at(full_index++);
            target.input_norm            = artifact::materialized_tensor(backing, source.input_norm,
                                                                         NumericFormat::BF16, {2048});
            target.projection.query_key_gate_value =
                artifact::materialized_weight(backing, source.attention.query_key_gate_value,
                                              NumericFormat::W8G32_F16S, 9216, 2048);
            target.query_norm = artifact::materialized_tensor(backing, source.attention.query_norm,
                                                              NumericFormat::BF16, {256});
            target.key_norm   = artifact::materialized_tensor(backing, source.attention.key_norm,
                                                              NumericFormat::BF16, {256});
            target.output     = artifact::materialized_weight(backing, source.attention.output,
                                                              NumericFormat::W8G32_F16S, 2048, 4096);
            target.post_attention_norm = artifact::materialized_tensor(
                backing, source.post_attention_norm, NumericFormat::BF16, {2048});
            target.post_mixer =
                load_moe(source.moe, backing, NumericFormat::Q4G64_F16S, routed_down_format(layer));
        } else {
            GdnWeights& target = gdn_layers.at(gdn_index++);
            target.input_norm  = artifact::materialized_tensor(backing, source.input_norm,
                                                               NumericFormat::BF16, {2048});
            target.projection.a_log =
                artifact::materialized_tensor(backing, source.gdn.a_log, NumericFormat::FP32, {32});
            target.projection.dt_bias = artifact::materialized_tensor(backing, source.gdn.dt_bias,
                                                                      NumericFormat::FP32, {32});
            target.convolution = artifact::materialized_tensor(backing, source.gdn.convolution,
                                                               NumericFormat::BF16, {8192, 4});
            target.projection.a_b_projection = artifact::materialized_weight(
                backing, source.gdn.a_b_projection, NumericFormat::BF16, 64, 2048);
            target.projection.query_key_value_z = artifact::materialized_weight(
                backing, source.gdn.query_key_value_z, NumericFormat::W8G32_F16S, 12288, 2048);
            target.norm =
                artifact::materialized_tensor(backing, source.gdn.norm, NumericFormat::BF16, {128});
            target.output              = artifact::materialized_weight(backing, source.gdn.output,
                                                                       NumericFormat::W8G32_F16S, 2048, 4096);
            target.post_attention_norm = artifact::materialized_tensor(
                backing, source.post_attention_norm, NumericFormat::BF16, {2048});
            target.post_mixer =
                load_moe(source.moe, backing, NumericFormat::Q4G64_F16S, routed_down_format(layer));
        }
    }
    if (full_index != full_layers.size() || gdn_index != gdn_layers.size()) {
        throw std::logic_error("35B Text topology binding is incomplete");
    }

    // Everything from here reads the final hidden state -- final norm, output head, proposal, MTP
    // and DFlash -- so it comes from the head rank. Aliased rather than renamed at each use so the
    // bodies stay identical to the single-rank versions they were.
    artifact::MaterializedArtifact& backing = head_backing;
    const BindingPlan& plan                 = head_plan;

    final_norm =
        artifact::materialized_tensor(backing, plan.final_norm, NumericFormat::BF16, {2048});
    output_head = artifact::materialized_weight(backing, plan.output_head,
                                                NumericFormat::Q6G64_F16S, 248320, 2048);
    if (plan.features.optimized_proposal()) {
        auto& proposal     = runtime.optimized_proposal.emplace();
        proposal.head      = artifact::materialized_weight(backing, plan.draft_head,
                                                           NumericFormat::Q4G64_F16S, 131072, 2048);
        proposal.token_ids = artifact::materialized_tensor(backing, plan.draft_head_token_ids,
                                                           NumericFormat::I32, {131072});
    }

    if (plan.features.mtp()) {
        auto& mtp            = runtime.mtp.emplace();
        mtp.input_projection = artifact::materialized_weight(backing, plan.mtp.input_projection,
                                                             NumericFormat::W8G32_F16S, 2048, 4096);
        mtp.embedding_norm   = artifact::materialized_tensor(backing, plan.mtp.embedding_norm,
                                                             NumericFormat::BF16, {2048});
        mtp.hidden_norm      = artifact::materialized_tensor(backing, plan.mtp.hidden_norm,
                                                             NumericFormat::BF16, {2048});
        mtp.input_norm       = artifact::materialized_tensor(backing, plan.mtp.input_norm,
                                                             NumericFormat::BF16, {2048});
        mtp.attention.query_key_gate_value =
            artifact::materialized_weight(backing, plan.mtp.attention.query_key_gate_value,
                                          NumericFormat::W8G32_F16S, 9216, 2048);
        mtp.query_norm = artifact::materialized_tensor(backing, plan.mtp.attention.query_norm,
                                                       NumericFormat::BF16, {256});
        mtp.key_norm   = artifact::materialized_tensor(backing, plan.mtp.attention.key_norm,
                                                       NumericFormat::BF16, {256});
        mtp.output     = artifact::materialized_weight(backing, plan.mtp.attention.output,
                                                       NumericFormat::W8G32_F16S, 2048, 4096);
        mtp.post_attention_norm = artifact::materialized_tensor(
            backing, plan.mtp.post_attention_norm, NumericFormat::BF16, {2048});
        mtp.post_mixer =
            load_moe(plan.mtp.moe, backing, NumericFormat::W8G32_F16S, NumericFormat::W8G32_F16S);
        mtp.final_norm = artifact::materialized_tensor(backing, plan.mtp.final_norm,
                                                       NumericFormat::BF16, {2048});
    }

    // Vision feeds the embedding, so this one block reads from the embedding rank instead.
    {
    artifact::MaterializedArtifact& backing = embed_backing;
    const BindingPlan& plan                 = embed_plan;
    if (plan.features.vision) {
        qwen3_6::VisionWeights vision;
        vision.common = qwen3_6::materialize_vision_common(
            backing, plan.vision_backbone, plan.vision_merger_input, plan.vision_merger_norm);
        vision.merger_fc2      = artifact::materialized_weight(backing, plan.vision_merger_fc2,
                                                               NumericFormat::W8G32_F16S, 2048, 4608);
        vision.merger_fc2_bias = artifact::materialized_tensor(backing, plan.vision_merger_fc2_bias,
                                                               NumericFormat::BF16, {2048});
        if (plan.features.overlay_vision()) {
            if (!plan.vision_overlay || backing.eviction_pool() == nullptr ||
                backing.pinned_block().empty()) {
                throw std::logic_error(
                    "overlay vision requires a pool-backed pinned materialization");
            }
            auto& overlay                = runtime.vision_overlay.emplace();
            overlay.pool                 = backing.eviction_pool();
            overlay.pinned_block         = backing.pinned_block();
            overlay.host.weights         = vision;
            overlay.layout               = *plan.vision_overlay;
            overlay.window_capacity_bytes = backing.eviction_pool()->window_capacity_bytes();
        } else {
            runtime.vision = vision;
        }
    }
    } // vision reads the embedding rank

    if (plan.features.dflash()) {
        DFlashWeights& target     = runtime.dflash.emplace();
        target.feature_projection = artifact::materialized_weight(
            backing, plan.dflash.feature_projection, NumericFormat::W8G32_F16S, 2048, 16384);
        target.context_norm = artifact::materialized_tensor(backing, plan.dflash.context_norm,
                                                            NumericFormat::BF16, {2048});
        for (std::size_t layer = 0; layer < kDFlashLayers; ++layer) {
            const DFlashLayerPlan& source = plan.dflash.layers[layer];
            DFlashLayerWeights& weights   = target.layers[layer];
            weights.input_norm      = artifact::materialized_tensor(backing, source.input_norm,
                                                                    NumericFormat::BF16, {2048});
            weights.query_key_value = artifact::materialized_weight(
                backing, source.query_key_value, NumericFormat::W8G32_F16S, 6144, 2048);
            weights.context_key   = row_view(weights.query_key_value, 4096, 1024);
            weights.context_value = row_view(weights.query_key_value, 5120, 1024);
            weights.query_norm    = artifact::materialized_tensor(backing, source.query_norm,
                                                                  NumericFormat::BF16, {128});
            weights.key_norm =
                artifact::materialized_tensor(backing, source.key_norm, NumericFormat::BF16, {128});
            weights.attention_output = artifact::materialized_weight(
                backing, source.attention_output, NumericFormat::W8G32_F16S, 2048, 4096);
            weights.post_attention_norm = artifact::materialized_tensor(
                backing, source.post_attention_norm, NumericFormat::BF16, {2048});
            weights.gate_up = artifact::materialized_weight(backing, source.gate_up,
                                                            NumericFormat::W8G32_F16S, 12288, 2048);
            weights.down    = artifact::materialized_weight(backing, source.down,
                                                            NumericFormat::W8G32_F16S, 2048, 6144);
        }
        target.final_norm = artifact::materialized_tensor(backing, plan.dflash.final_norm,
                                                          NumericFormat::BF16, {2048});
    }
}

} // namespace ninfer::targets::qwen3_6_35b_a3b::detail
