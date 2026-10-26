#pragma once

#include "models/qwen3_5/model.h"
#include "ninfer/ops/weight_input.h"

#include <array>
#include <memory>
#include <limits>
#include <optional>
#include <stdexcept>
#include <variant>
#include <vector>

namespace ninfer::models::qwen3_5::execution {

using LinearParameters = ops::SingleProjectionWeight;

[[nodiscard]] inline std::int32_t dimension(std::uint64_t value) {
    if (value > std::uint64_t(std::numeric_limits<std::int32_t>::max())) {
        throw std::overflow_error("model dimension exceeds the Tensor integer domain");
    }
    return static_cast<std::int32_t>(value);
}

struct DenseParameters {
    LinearParameters gate_up;
    LinearParameters down;
    // gate_up policy for Verify-phase calls. --mlp-a8-decode widens this to AllowA8IntDecode
    // independently of --prefill-a8, so long as the profile is eligible; prefill and scoring
    // always use gate_up.policy and never take the decode-width integer route.
    ops::LinearPolicy verify_gate_up_policy = ops::LinearPolicy::A16Only;
};

using FfnParameters = std::variant<DenseParameters, ops::SparseMoeWeights>;

struct AttentionParameters {
    ops::ProjectionWeights projection;
    Tensor query_norm, key_norm;
    LinearParameters output;
};

struct GdnParameters {
    ops::ProjectionWeights projection;
    ops::ProjectionWeights control;
    Tensor a_log, dt_bias, convolution, norm;
    LinearParameters output;
};

struct BlockParameters {
    Tensor input_norm, post_attention_norm;
    std::variant<AttentionParameters, GdnParameters> mixer;
    FfnParameters ffn;
    ops::SparseMoeHints projection_prefetch;
    // Which stage's device holds this layer: its weights, its KV cache or recurrent state, and the
    // scratch it runs in. Always 0 on one GPU.
    std::size_t rank = 0;
};

struct TextParameters {
    Weight token_embedding;
    // A token table stored in the rotated basis (T2) is restored after the gather with the
    // hidden-width sign vector; empty for an ordinary table.
    Tensor token_embedding_signs;
    LinearParameters output_head;
    Tensor final_norm;
    std::vector<BlockParameters> layers;
    // How many stages the layer loop spans. One is the identity plan and the fast path.
    std::size_t rank_count = 1;
    // Stage s owns layers [stage_begin[s], stage_end(s)).
    std::vector<std::uint32_t> stage_begin{0};

    [[nodiscard]] bool split_execution() const noexcept { return rank_count > 1; }
    [[nodiscard]] std::uint32_t stage_end(std::size_t stage) const {
        return stage + 1 < stage_begin.size() ? stage_begin[stage + 1]
                                              : static_cast<std::uint32_t>(layers.size());
    }
};

struct MtpProjectionParameters {
    LinearParameters packed;
    // Dense MTP projects K/V and Q/gate independently in its incremental path.
    // MoE MTP uses its existing complete-parent Attention projection.
    std::optional<std::array<LinearParameters, 4>> rows;
};

struct MtpParameters {
    LinearParameters input_projection;
    Tensor embedding_norm, hidden_norm, input_norm, post_attention_norm, final_norm;
    MtpProjectionParameters projection;
    Tensor query_norm, key_norm;
    LinearParameters output;
    FfnParameters ffn;
    LinearParameters output_head;
};

struct NormParameters {
    Tensor weight, bias;
};

struct VisionBlockParameters {
    NormParameters norm1, norm2;
    LinearParameters qkv;
    Tensor qkv_bias;
    LinearParameters output, fc1, fc2;
    Tensor output_bias, fc1_bias, fc2_bias;
};

struct VisionParameters {
    LinearParameters patch_embedding;
    Tensor patch_embedding_bias, position_embedding;
    std::vector<VisionBlockParameters> layers;
    NormParameters merger_norm;
    LinearParameters merger_fc1, merger_fc2;
    Tensor merger_fc1_bias, merger_fc2_bias;
};

struct DynamicConvParameters {
    Tensor base_kernel;
    LinearParameters kernel_projection;
};

struct DraftBlockParameters {
    Tensor input_norm, post_attention_norm;
    LinearParameters query_key_value, context_key, context_value;
    Tensor query_norm, key_norm;
    LinearParameters output;
    DenseParameters mlp;
    std::optional<DynamicConvParameters> attention_conv, mlp_conv;
};

struct SelectorParameters {
    LinearParameters hidden_projection;
    Tensor predecessor_codebook, successor_codebook;
};

struct DraftParameters {
    LinearParameters feature_projection;
    Tensor context_norm, final_norm;
    std::vector<DraftBlockParameters> layers;
    std::optional<SelectorParameters> selector;
    LinearParameters output_head;
};

struct ProposalParameters {
    LinearParameters head;
    std::optional<Tensor> token_ids;
    std::uint32_t rows = 0;
};

// The lossy-prefill scout's int4 parameters (ops::LinearPolicy::ScoutA4Int,
// docs/maintainer/lossy-long-prefill.md "Int4 scout GEMMs"): copies of layers [0, filter_layer]
// in which every Q4 matrix that takes the integer-A8 prefill route takes the int4 one instead (the
// Q4 half of a split Q4/Q5 input projection too). Q5 matrices keep their route, and a matrix with
// no integer route keeps A16. The weights are the model's own: only the policies differ. These
// copies are the only place the policy is ever written; TextContext::scout_chunk runs them.
[[nodiscard]] std::vector<BlockParameters> scout_a4_layers(const TextParameters& text,
                                                           std::uint32_t filter_layer);
// Whether any projection of the block carries the scout's int4 policy.
[[nodiscard]] bool holds_scout_policy(const BlockParameters& block) noexcept;
// Whether --lossy-prefill-scout-a4 would move anything at all: some layer in [0, filter_layer] has
// a Q4 matrix on the integer-A8 prefill route. Startup refuses the option otherwise (without
// --prefill-a8 it would silently be the int8 scout).
[[nodiscard]] bool scout_a4_promotes(const TextParameters& text, std::uint32_t filter_layer);

// Cold native preparation for the fixed model implementation. This owner is stable before
// startup sizing, execution, or Graph capture; all weight addresses borrow the source Model.
// Shape-dependent kernel selection and scratch remain with the calling implementation and Op.
class Parameters {
public:
    explicit Parameters(const Model& source);
    Parameters(const Parameters&)            = delete;
    Parameters& operator=(const Parameters&) = delete;
    Parameters(Parameters&&)                 = delete;
    Parameters& operator=(Parameters&&)      = delete;

    const Model& model;
    TextParameters text;
    std::optional<MtpParameters> mtp;
    std::optional<VisionParameters> vision;
    std::optional<DraftParameters> draft;
    std::optional<ProposalParameters> proposal;
};

} // namespace ninfer::models::qwen3_5::execution
