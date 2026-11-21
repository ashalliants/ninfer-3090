// Written for NInfer-3090. The pattern follows src/models/dsv4/load.cpp; every parameter is device resident.

#include "models/inkling2/model.h"

#include "artifact/binder.h"
#include "artifact/views.h"

#include <string>
#include <utility>

namespace ninfer::models::inkling2 {
namespace {

using artifact::ArtifactError;
using artifact::Shape;

class Bindings {
public:
    explicit Bindings(artifact::Binder& binder) : binder_(binder) {}

    WeightId parameter(const std::string& name, Shape shape) {
        const WeightId id{references.size()};
        references.push_back(binder_.parameter(name, std::move(shape), artifact::Residency::Device));
        return id;
    }

    [[nodiscard]] bool contains(const std::string& name) const { return binder_.contains(name); }

    std::vector<artifact::ParameterReference> references;

private:
    artifact::Binder& binder_;
};

LayerWeights bind_layer(Bindings& b, const TextConfig& c, std::uint32_t layer) {
    const std::uint64_t h = c.hidden_size, q = std::uint64_t(c.heads) * c.head_dim, kv = std::uint64_t(c.kv_heads) * c.head_dim,
                        k = c.conv_kernel_size, e = c.experts, i = c.moe_intermediate;
    const std::string p = "text/layers/" + std::to_string(layer) + "/", a = p + "attention/", m = p + "moe/";
    const bool local = c.local[layer];
    LayerWeights w;
    if (local) {
        w.norm = b.parameter(a + "norm", {h});
    } else {
        w.q_norm      = b.parameter(a + "q_norm", {h});
        w.k_norm      = b.parameter(a + "k_norm", {h});
        w.v_norm      = b.parameter(a + "v_norm", {h});
        w.k_conv_proj = b.parameter(a + "k_conv_proj", {std::uint64_t(c.kv_heads) * k, h});
        w.v_conv_proj = b.parameter(a + "v_conv_proj", {std::uint64_t(c.kv_heads) * k, h});
    }
    if (layer > 0) { w.res_query = local ? b.parameter(a + "res_query", {h}) : b.parameter(a + "res_query", {3, h}); }
    w.q         = b.parameter(a + "q", {q, h});
    w.k         = b.parameter(a + "k", {kv, h});
    w.v         = b.parameter(a + "v", {kv, h});
    w.output    = b.parameter(a + "output", {h, q});
    w.gate      = b.parameter(a + "gate", {q, h});
    w.rel       = b.parameter(a + "rel", {std::uint64_t(c.heads) * c.d_rel, h});
    w.sink      = b.parameter(a + "sink", {c.heads, h});
    w.q_clip    = b.parameter(a + "q_clip", {c.heads, c.head_dim});
    w.rel_clip  = b.parameter(a + "rel_clip", {c.heads, c.d_rel});
    w.rel_bias  = b.parameter(a + "rel_bias", {c.rel_extent_of(layer), c.d_rel});
    w.k_conv    = b.parameter(a + "k_conv", {kv, k});
    w.v_conv    = b.parameter(a + "v_conv", {kv, k});
    w.post_conv = b.parameter(a + "post_conv", {h, k});
    if (c.is_core(layer)) {
        for (const char* module : {"attn", "k", "v", "mlp"}) {
            const std::string l = p + "loop/" + module + "/";
            w.loop.push_back({b.parameter(l + "down", {c.loop_gate_rank, 2 * h}), b.parameter(l + "gain", {h, c.loop_gate_rank}),
                              b.parameter(l + "forget", {h, c.loop_gate_rank})});
        }
    }
    w.ffn_res_query = b.parameter(m + "res_query", {h});
    w.ffn_norm      = b.parameter(m + "norm", {h});
    w.router        = b.parameter(m + "router", {e, h});
    w.router_bias   = b.parameter(m + "router_bias", {e});
    w.mlp_post_conv = b.parameter(m + "post_conv", {h, k});
    if (b.contains(m + "experts")) {
        w.experts       = b.parameter(m + "experts", {e, h, i});
        w.gate_up_scale = b.parameter(m + "experts_gate_up_scale", {e});
        w.down_scale    = b.parameter(m + "experts_down_scale", {e});
    } else {
        w.experts_gate_up = b.parameter(m + "experts_gate_up", {e, 2 * i, h});
        w.experts_down    = b.parameter(m + "experts_down", {e, h, i});
    }
    return w;
}

DSparkWeights bind_dspark(Bindings& b, const DSparkConfig& c, const TextConfig& text) {
    const std::uint64_t h = c.hidden_size, f = c.intermediate_size, r = c.markov_rank, d = c.head_dim;
    const std::uint64_t q = std::uint64_t(c.heads) * d, kv = std::uint64_t(c.kv_heads) * d;
    DSparkWeights w;
    w.fuse             = b.parameter("dspark/fuse", {c.layers + 1, c.target_visits.size()});
    w.rnn_context_norm = b.parameter("dspark/rnn_context_norm", {h});
    w.mask_embedding   = b.parameter("dspark/mask_embedding", {h});
    w.output_norm      = b.parameter("dspark/output_norm", {h});
    w.markov_embedding = b.parameter("dspark/markov/token_embedding", {text.vocab_size, r});
    w.joint            = b.parameter("dspark/markov/joint", {3 * r, 2 * r + 2 * h});
    w.joint_bias       = b.parameter("dspark/markov/joint_bias", {3 * r});
    w.hidden_out       = b.parameter("dspark/markov/hidden_out", {h, r});
    w.hidden_out_bias  = b.parameter("dspark/markov/hidden_out_bias", {h});
    for (std::uint32_t i = 0; i < c.layers; ++i) {
        const std::string p = "dspark/layers/" + std::to_string(i) + "/", a = p + "attention/";
        DSparkLayerWeights l;
        l.attn_norm      = b.parameter(a + "norm", {h});
        l.q              = b.parameter(a + "q", {q, h});
        l.k              = b.parameter(a + "k", {kv, h});
        l.v              = b.parameter(a + "v", {kv, h});
        l.output         = b.parameter(a + "output", {h, q});
        l.q_norm         = b.parameter(a + "q_norm", {d});
        l.k_norm         = b.parameter(a + "k_norm", {d});
        l.context_k      = b.parameter(a + "context_k", {kv, h});
        l.context_v      = b.parameter(a + "context_v", {kv, h});
        l.context_k_norm = b.parameter(a + "context_k_norm", {d});
        l.context_norm   = b.parameter(p + "context_norm", {h});
        l.ffn_norm       = b.parameter(p + "ffn/norm", {h});
        l.gate           = b.parameter(p + "ffn/gate", {f, h});
        l.up             = b.parameter(p + "ffn/up", {f, h});
        l.down           = b.parameter(p + "ffn/down", {h, f});
        w.layers.push_back(l);
    }
    return w;
}

TextWeights bind_text(Bindings& b, const TextConfig& c) {
    const std::uint64_t h = c.hidden_size, v = c.vocab_size;
    TextWeights w;
    w.embedding         = b.parameter("text/embedding", {v, h});
    w.embedding_norm    = b.parameter("text/embedding_norm", {h});
    w.head              = b.parameter("text/head", {v, h});
    w.output_norm       = b.parameter("text/output_norm", {h});
    w.output_res_query  = b.parameter("text/output_res_query", {h});
    w.loop_res_query    = b.parameter("text/loop/res_query", {2 * std::uint64_t(c.core_layers()), h});
    w.loop_kv_res_query = b.parameter("text/loop/kv_res_query", {2 * std::uint64_t(c.core_layers()), h});
    for (std::uint32_t layer = 0; layer < c.num_hidden_layers; ++layer) { w.layers.push_back(bind_layer(b, c, layer)); }
    return w;
}

const WeightParent& whole_parent(const BoundWeight& weight) {
    if (weight.view.parts.size() != 1 || !is_complete_weight(weight.view)) {
        throw ArtifactError(weight.name + ": must be one complete stored object");
    }
    return *weight.view.parts.front().parent;
}

const float* fp32_data(const BoundWeight& weight) {
    const auto& parent = whole_parent(weight);
    if (parent.geometry.format != QType::FP32 || parent.geometry.layout != QuantLayout::Contiguous) {
        throw ArtifactError(weight.name + ": must be contiguous FP32");
    }
    return reinterpret_cast<const float*>(parent.data);
}

ops::SituExpertBank bind_bank(const std::vector<BoundWeight>& bound, const LayerWeights& w, const TextConfig& c) {
    ops::SituExpertBank bank;
    bank.experts      = static_cast<std::int32_t>(c.experts);
    bank.hidden       = static_cast<std::int32_t>(c.hidden_size);
    bank.intermediate = static_cast<std::int32_t>(c.moe_intermediate);
    if (w.experts.valid()) {
        const BoundWeight& weight = bound.at(w.experts.index);
        const auto& parent        = whole_parent(weight);
        if (parent.geometry.layout != QuantLayout::GgmlExpertRecord || parent.geometry.format != QType::GGML_REC_NVFP4_NVFP4) {
            throw ArtifactError(weight.name + ": the expert bank must be ggml_rec_nvfp4_nvfp4 records");
        }
        bank.format        = parent.geometry.format;
        bank.records       = reinterpret_cast<const std::uint8_t*>(parent.data);
        bank.record_stride = parent.geometry.record_stride;
        bank.gate_up_scale = fp32_data(bound.at(w.gate_up_scale.index));
        bank.down_scale    = fp32_data(bound.at(w.down_scale.index));
    } else {
        for (const WeightId id : {w.experts_gate_up, w.experts_down}) {
            const auto& parent = whole_parent(bound.at(id.index));
            if (parent.geometry.format != QType::BF16 || parent.geometry.layout != QuantLayout::Contiguous) {
                throw ArtifactError(bound.at(id.index).name + ": BF16 expert banks must be contiguous BF16");
            }
        }
        bank.format  = QType::BF16;
        bank.gate_up = whole_parent(bound.at(w.experts_gate_up.index)).data;
        bank.down    = whole_parent(bound.at(w.experts_down.index)).data;
    }
    return bank;
}

} // namespace

Model::Model(TextConfig config, TextWeights weights, std::vector<BoundWeight> bound, artifact::MaterializedArtifact backing)
    : backing_(std::move(backing)), config_(std::move(config)), weights_(std::move(weights)), bound_(std::move(bound)) {
    for (const auto& layer : weights_.layers) { banks_.push_back(bind_bank(bound_, layer, config_)); }
}

Model::~Model() = default;

Model::Matrix Model::matrix(WeightId id) const {
    const BoundWeight& weight = bound_.at(id.index);
    const auto& parent        = whole_parent(weight);
    const auto& g             = parent.geometry;
    const bool q8 = g.format == QType::GGML_Q8_0 && g.layout == QuantLayout::GgmlBlocks;
    const bool bf16 = g.format == QType::BF16 && g.layout == QuantLayout::Contiguous;
    if ((!q8 && !bf16) || weight.view.shape.size() != 2) {
        throw ArtifactError(weight.name + ": must be a Q8_0 (GGML blocks) or BF16 matrix");
    }
    return {parent.data, g.format, static_cast<std::int32_t>(weight.view.shape[0]), static_cast<std::int32_t>(weight.view.shape[1])};
}

Tensor Model::tensor(WeightId id, std::initializer_list<std::int32_t> ne) const {
    const BoundWeight& weight = bound_.at(id.index);
    Tensor t                  = weight_tensor(weight.view, ne);
    return t;
}

bool is_inkling2(const artifact::Reader& reader) { return is_inkling2_config(reader.directory()); }

std::unique_ptr<Model> load_model(const artifact::Reader& reader, DeviceContext& device, const StartupObserver* observer) {
    TextConfig config = parse_config(reader.directory());
    artifact::Binder binder(reader);
    Bindings bindings(binder);
    TextWeights weights = bind_text(bindings, config);
    std::optional<DSparkConfig> dspark;
    DSparkWeights dspark_weights;
    if (reader.directory().components.contains("dspark")) {
        dspark         = parse_dspark_config(reader.directory().component("dspark").config, config);
        dspark_weights = bind_dspark(bindings, *dspark, config);
    }
    auto references     = std::move(bindings.references);
    auto plan           = std::move(binder).finish();
    auto backing        = artifact::materialize(*plan.source, std::move(plan), device, observer);
    std::vector<BoundWeight> bound;
    bound.reserve(references.size());
    for (auto& reference : references) { bound.push_back({reference.name, artifact::bind_view(reference, backing)}); }
    auto model = std::unique_ptr<Model>(new Model(std::move(config), std::move(weights), std::move(bound), std::move(backing)));
    model->dspark_config_  = std::move(dspark);
    model->dspark_weights_ = std::move(dspark_weights);
    return model;
}

} // namespace ninfer::models::inkling2
