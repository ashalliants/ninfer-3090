// Adapted from Infernix a3edb450 src/models/qwen4_exp/load.cpp (Apache-2.0).
// Modified for NInfer-3090: binds the converter's GGUF-mirroring parameters (GGML block-format
// projections, BF16 gamma attention norms, FP32 multiplier hyper-connection and PLE norms) and
// ggml_expert_record_v1 banks in pinned host memory; no expert scales, frontend resources, vision,
// MTP, proposal head or SSD tier; one-step load (no separate plan).

#include "models/qwen4_exp/model.h"

#include "artifact/binder.h"
#include "artifact/reader.h"
#include "artifact/views.h"

#include <cuda_runtime.h>

#include <map>
#include <string>
#include <utility>

namespace ninfer::models::qwen4_exp {
namespace {

using artifact::ArtifactError;
using artifact::Residency;
using artifact::Shape;

ops::LinearPolicy linear_policy(artifact::ActivationPolicy policy) {
    switch (policy) {
    case artifact::ActivationPolicy::A16Only:
        return ops::LinearPolicy::A16Only;
    case artifact::ActivationPolicy::AllowA8:
        return ops::LinearPolicy::AllowA8;
    case artifact::ActivationPolicy::AllowA4:
        return ops::LinearPolicy::AllowA4;
    }
    throw ArtifactError("unknown activation policy");
}

struct PendingWeight {
    artifact::ParameterReference reference;
    ops::LinearPolicy policy = ops::LinearPolicy::A16Only;
};

// Logical demands of the model; each parameter is declared exactly once.
class Bindings {
public:
    explicit Bindings(artifact::Binder& binder) : binder_(binder) {}

    WeightId parameter(std::string name, Shape shape, std::string_view input = {},
                       std::optional<QType> exact_format = {}, Residency residency = Residency::Device) {
        if (ids_.contains(name)) { throw ArtifactError(name + ": duplicate parameter declaration"); }
        PendingWeight pending;
        pending.reference = binder_.parameter(name, std::move(shape), residency, exact_format);
        if (!input.empty()) {
            const auto& use = binder_.use(name, input);
            if (!use.activation_policy) {
                throw ArtifactError(name + "@" + std::string(input) + ": missing activation policy");
            }
            if (!use.auxiliaries.empty()) { throw ArtifactError(name + ": Qwen4Exp weights carry no Use auxiliaries"); }
            pending.policy = linear_policy(*use.activation_policy);
        }
        const WeightId id{weights.size()};
        ids_.emplace(std::move(name), id);
        weights.push_back(std::move(pending));
        return id;
    }

    WeightId direct(std::string name, Shape shape, QType format) {
        return parameter(std::move(name), std::move(shape), {}, format);
    }

    std::vector<PendingWeight> weights;

private:
    artifact::Binder& binder_;
    std::map<std::string, WeightId, std::less<>> ids_;
};

HyperConnectionWeights bind_hc(Bindings& b, const TextConfig& c, const std::string& prefix, bool combine) {
    const std::uint64_t width = c.residual_width(), rank = c.hc.rank;
    HyperConnectionWeights out;
    out.norm = b.direct(prefix + "norm", {width}, QType::FP32);
    out.down = b.parameter(prefix + "down", {rank, width}, prefix + "input", QType::BF16);
    if (combine) { out.inject = b.parameter(prefix + "inject", {c.hc.streams, width}, prefix + "input", QType::BF16); }
    out.up = b.parameter(prefix + "up", {width, rank}, prefix + "mix_activation", QType::BF16);
    return out;
}

AttentionWeights bind_attention(Bindings& b, const TextConfig& c, const std::string& prefix) {
    const std::uint64_t h = c.hidden_size, q = c.attention.query_width(), k = c.attention.key_width();
    const std::uint64_t iq = std::uint64_t(c.qsa.index_heads) * c.qsa.index_head_dim;
    const std::string input = prefix + "mixer_input", p = prefix + "attention/", x = prefix + "indexer/";
    AttentionWeights out;
    out.query_gate       = b.parameter(p + "query_gate", {2 * q, h}, input);
    out.key              = b.parameter(p + "key", {k, h}, input);
    out.value            = b.parameter(p + "value", {k, h}, input);
    out.output           = b.parameter(p + "output", {h, q}, p + "gated_output");
    out.query_norm       = b.direct(p + "query_norm", {c.attention.head_dim}, QType::BF16);
    out.key_norm         = b.direct(p + "key_norm", {c.attention.head_dim}, QType::BF16);
    out.index_query      = b.parameter(x + "query", {iq, h}, input, QType::BF16);
    out.index_key        = b.parameter(x + "key", {c.qsa.index_head_dim, h}, input, QType::BF16);
    out.index_query_norm = b.direct(x + "query_norm", {c.qsa.index_head_dim}, QType::BF16);
    out.index_key_norm   = b.direct(x + "key_norm", {c.qsa.index_head_dim}, QType::BF16);
    return out;
}

GdnWeights bind_gdn(Bindings& b, const TextConfig& c, const std::string& prefix) {
    const std::uint64_t h = c.hidden_size, kw = c.gdn.key_width(), vw = c.gdn.value_width(), nv = c.gdn.value_heads;
    const std::string input = prefix + "mixer_input", p = prefix + "gdn/";
    GdnWeights out;
    out.qkv          = b.parameter(p + "qkv", {2 * kw + vw, h}, input);
    out.z            = b.parameter(p + "z", {vw, h}, input);
    out.a_projection = b.parameter(p + "a_projection", {nv, h}, input, QType::BF16);
    out.b_projection = b.parameter(p + "b_projection", {nv, h}, input, QType::BF16);
    out.decay        = b.direct(p + "a", {nv}, QType::FP32);
    out.dt_bias      = b.direct(p + "dt_bias", {nv}, QType::FP32);
    out.convolution  = b.direct(p + "convolution", {c.gdn.conv_kernel, c.gdn.conv_channels()}, QType::BF16);
    out.norm         = b.direct(p + "norm", {c.gdn.value_head_dim}, QType::BF16);
    out.output       = b.parameter(p + "output", {h, vw}, p + "gated_output");
    return out;
}

MoeWeights bind_moe(Bindings& b, const TextConfig& c, const std::string& prefix) {
    const std::uint64_t h = c.hidden_size, e = c.moe.experts, s = c.moe.shared_intermediate;
    const std::string input = prefix + "ffn_input", p = prefix + "moe/";
    MoeWeights out;
    out.router       = b.parameter(p + "router", {e, h}, input, QType::BF16);
    out.shared_score = b.parameter(p + "shared_score", {1, h}, input, QType::BF16);
    out.shared_gate  = b.parameter(p + "shared/gate", {s, h}, input);
    out.shared_up    = b.parameter(p + "shared/up", {s, h}, input);
    out.shared_down  = b.parameter(p + "shared/down", {h, s}, p + "shared/product");
    out.experts      = b.parameter(p + "experts", {e, h, c.moe.intermediate}, input, {}, Residency::Pinned);
    return out;
}

PleWeights bind_ple(Bindings& b, const TextConfig& c, const std::string& prefix) {
    const std::uint64_t h = c.hidden_size, width = c.residual_width(), dim = c.ple.embed_dim;
    const std::string p = prefix + "ple/", input = p + "ngram_embedding";
    PleWeights out;
    out.key_projection   = b.parameter(p + "key", {width, dim}, input, QType::BF16);
    out.value_projection = b.parameter(p + "value", {h, dim}, input, QType::BF16);
    out.key_norm         = b.direct(p + "key_norm", {width}, QType::FP32);
    out.query_norm       = b.direct(p + "query_norm", {width}, QType::FP32);
    out.conv_norm        = b.direct(p + "conv_norm", {width}, QType::FP32);
    out.convolution      = b.direct(p + "convolution", {width, c.ple.conv_kernel}, QType::FP32);
    return out;
}

TextWeights bind_text(Bindings& b, const TextConfig& c) {
    TextWeights out;
    const std::uint64_t h = c.hidden_size, v = c.vocab_size;
    out.token_embedding = b.parameter("text/token_embedding", {v, h}, {}, QType::GGML_IQ4_XS);
    out.output_head     = b.parameter("text/output_head", {v, h}, "text/final_hidden", QType::GGML_IQ4_XS);
    out.final_mixer     = bind_hc(b, c, "text/final_mixer/", false);
    for (std::uint32_t i = 0; i < c.num_hidden_layers; ++i) {
        const std::string prefix = "text/layers/" + std::to_string(i) + "/";
        BlockWeights block;
        if (i == c.ple.layer) { block.ple = bind_ple(b, c, prefix); }
        block.attn_hc = bind_hc(b, c, prefix + "attn_hc/", true);
        if (c.layer_types[i] == MixerKind::Attention) {
            block.mixer = bind_attention(b, c, prefix);
        } else {
            block.mixer = bind_gdn(b, c, prefix);
        }
        block.mlp_hc = bind_hc(b, c, prefix + "mlp_hc/", true);
        block.moe    = bind_moe(b, c, prefix);
        out.layers.push_back(std::move(block));
    }
    return out;
}

ExpertBank expert_bank(const BoundWeight& weight, const TextConfig& c) {
    const auto& view = weight.view;
    if (view.parts.size() != 1 || !is_complete_weight(view)) {
        throw ArtifactError(weight.name + ": an expert bank must be one complete parent");
    }
    const auto& parent = *view.parts.front().parent;
    const auto& g      = parent.geometry;
    if (g.layout != QuantLayout::GgmlExpertRecord || g.shape.size() != 3 || g.shape[0] != c.moe.experts ||
        g.shape[1] != c.hidden_size || g.shape[2] != c.moe.intermediate) {
        throw ArtifactError(weight.name + ": expert bank geometry differs from the text config");
    }
    ExpertBank bank;
    bank.format        = g.format;
    bank.experts       = c.moe.experts;
    bank.record_stride = g.record_stride;
    bank.host_records  = reinterpret_cast<const std::uint8_t*>(parent.data);
    void* device       = nullptr;
    if (cudaHostGetDevicePointer(&device, const_cast<std::byte*>(parent.data), 0) != cudaSuccess || device == nullptr) {
        throw ArtifactError(weight.name + ": the pinned expert bank is not device-addressable");
    }
    bank.device_records = static_cast<const std::uint8_t*>(device);
    return bank;
}

} // namespace

Model::Model(TextConfig config, TextWeights weights, std::vector<BoundWeight> bound, std::vector<ExpertBank> banks,
             artifact::MaterializedArtifact backing)
    : backing_(std::move(backing)), config_(std::move(config)), weights_(std::move(weights)),
      bound_(std::move(bound)), banks_(std::move(banks)) {}

Model::~Model() = default;

bool is_qwen4_exp(const artifact::Reader& reader) { return is_qwen4_exp_config(reader.directory()); }

std::unique_ptr<Model> load_model(const artifact::Reader& reader, DeviceContext& device) {
    TextConfig config = parse_config(reader.directory());
    artifact::Binder binder(reader);
    Bindings bindings(binder);
    TextWeights weights = bind_text(bindings, config);
    auto pending        = std::move(bindings.weights);
    auto backing        = artifact::materialize(reader, std::move(binder).finish(), device);
    std::vector<BoundWeight> bound;
    bound.reserve(pending.size());
    for (auto& item : pending) {
        bound.push_back({item.reference.name, artifact::bind_view(item.reference, backing), item.policy});
    }
    std::vector<ExpertBank> banks;
    for (const auto& layer : weights.layers) { banks.push_back(expert_bank(bound.at(layer.moe.experts.index), config)); }
    return std::unique_ptr<Model>(
        new Model(std::move(config), std::move(weights), std::move(bound), std::move(banks), std::move(backing)));
}

} // namespace ninfer::models::qwen4_exp
