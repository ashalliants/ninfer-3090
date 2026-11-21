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

VisionWeights bind_vision(Bindings& b, const VisionConfig& c) {
    const std::uint64_t d = c.hidden_size, f = c.intermediate_size, p = c.patch_size, m = 4 * d;
    VisionWeights w;
    w.class_embedding    = b.parameter("vision/class_embedding", {d});
    w.class_position     = b.parameter("vision/class_position_embedding", {d});
    w.patch_embedding    = b.parameter("vision/patch_embedding", {d, 3, p, p});
    w.position_embedding = b.parameter("vision/position_embedding", {std::uint64_t(c.position_grid) * c.position_grid, d});
    w.pre_norm_weight    = b.parameter("vision/pre_norm_weight", {d});
    w.pre_norm_bias      = b.parameter("vision/pre_norm_bias", {d});
    w.projector_fc1      = b.parameter("vision/projector/fc1", {m, m});
    w.projector_fc2      = b.parameter("vision/projector/fc2", {c.projection_size, m});
    for (std::uint32_t i = 0; i < c.depth; ++i) {
        const std::string l = "vision/layers/" + std::to_string(i) + "/", a = l + "attention/", f2 = l + "mlp/";
        VisionLayerWeights x;
        x.norm1_weight = b.parameter(l + "norm1_weight", {d});
        x.norm1_bias   = b.parameter(l + "norm1_bias", {d});
        x.q            = b.parameter(a + "query", {d, d});
        x.q_bias       = b.parameter(a + "query_bias", {d});
        x.k            = b.parameter(a + "key", {d, d});
        x.k_bias       = b.parameter(a + "key_bias", {d});
        x.v            = b.parameter(a + "value", {d, d});
        x.v_bias       = b.parameter(a + "value_bias", {d});
        x.output       = b.parameter(a + "output", {d, d});
        x.output_bias  = b.parameter(a + "output_bias", {d});
        x.attn_scale   = b.parameter(a + "layer_scale", {d});
        x.norm2_weight = b.parameter(l + "norm2_weight", {d});
        x.norm2_bias   = b.parameter(l + "norm2_bias", {d});
        x.fc1          = b.parameter(f2 + "fc1", {f, d});
        x.fc1_bias     = b.parameter(f2 + "fc1_bias", {f});
        x.fc2          = b.parameter(f2 + "fc2", {d, f});
        x.fc2_bias     = b.parameter(f2 + "fc2_bias", {d});
        x.mlp_scale    = b.parameter(f2 + "layer_scale", {d});
        w.layers.push_back(x);
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

// The drafter's direct reads (dspark_program.cpp): vectors and the fusion weights as FP32 words, the Markov
// token table, its output projection and the borrowed text token table and head as BF16 words; every other matrix
// goes through Model::matrix (Q8_0 blocks or BF16). Checked at load, whatever the converter wrote.
void check_dspark_formats(const std::vector<BoundWeight>& bound, const DSparkWeights& w, const TextWeights& text) {
    const auto require = [&](WeightId id, QType format, const char* what) {
        const BoundWeight& weight = bound.at(id.index);
        const auto& g             = whole_parent(weight).geometry;
        if (g.format != format || g.layout != QuantLayout::Contiguous) {
            throw ArtifactError(weight.name + ": the DSpark drafter reads it as contiguous " + what);
        }
    };
    for (const WeightId id : {w.fuse, w.rnn_context_norm, w.mask_embedding, w.output_norm, w.joint_bias, w.hidden_out_bias}) {
        require(id, QType::FP32, "FP32");
    }
    for (const WeightId id : {w.markov_embedding, w.hidden_out, text.embedding, text.head}) { require(id, QType::BF16, "BF16"); }
    for (const DSparkLayerWeights& l : w.layers) {
        for (const WeightId id : {l.attn_norm, l.q_norm, l.k_norm, l.context_k_norm, l.context_norm, l.ffn_norm}) {
            require(id, QType::FP32, "FP32");
        }
    }
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
    const bool words = (g.format == QType::BF16 || g.format == QType::FP32) && g.layout == QuantLayout::Contiguous;
    if ((!q8 && !words) || weight.view.shape.size() < 2) {
        throw ArtifactError(weight.name + ": must be a Q8_0 (GGML blocks), BF16 or FP32 matrix");
    }
    std::uint64_t cols = 1;
    for (std::size_t i = 1; i < weight.view.shape.size(); ++i) { cols *= weight.view.shape[i]; }
    return {parent.data, g.format, static_cast<std::int32_t>(weight.view.shape[0]), static_cast<std::int32_t>(cols)};
}

Tensor Model::tensor(WeightId id, std::initializer_list<std::int32_t> ne) const {
    const BoundWeight& weight = bound_.at(id.index);
    Tensor t                  = weight_tensor(weight.view, ne);
    return t;
}

bool is_inkling2(const artifact::Reader& reader) { return is_inkling2_config(reader.directory()); }

std::unique_ptr<Model> load_model(const artifact::Reader& reader, DeviceContext& device, const LoadComponents& components,
                                  const StartupObserver* observer) {
    TextConfig config = parse_config(reader.directory());
    artifact::Binder binder(reader);
    Bindings bindings(binder);
    TextWeights weights = bind_text(bindings, config);
    std::optional<DSparkConfig> dspark;
    DSparkWeights dspark_weights;
    if (components.dspark && reader.directory().components.contains("dspark")) {
        dspark         = parse_dspark_config(reader.directory().component("dspark").config, config);
        dspark_weights = bind_dspark(bindings, *dspark, config);
    }
    std::optional<VisionConfig> vision;
    VisionWeights vision_weights;
    if (components.vision && reader.directory().components.contains("vision")) {
        vision         = parse_vision_config(reader.directory().component("vision").config, config);
        vision_weights = bind_vision(bindings, *vision);
    }
    // The tokenizer and chat files are optional here (a Program-only artifact has none); the Frontend requires them.
    const auto& text_resources = reader.directory().component("text").resources;
    const auto resource = [&](std::string_view role) -> std::string_view {
        if (!text_resources.contains(role)) { return {}; }
        const auto bytes = binder.host_object(binder.resource("text", role));
        return std::string_view(reinterpret_cast<const char*>(bytes.data()), bytes.size());
    };
    const Model::Resources resources{.tokenizer_json         = resource("tokenizer.json"),
                                     .tokenizer_config_json  = resource("tokenizer_config.json"),
                                     .chat_template_jinja    = resource("chat_template.jinja"),
                                     .generation_config_json = resource("generation_config.json")};
    auto references     = std::move(bindings.references);
    auto plan           = std::move(binder).finish();
    auto backing        = artifact::materialize(*plan.source, std::move(plan), device, observer);
    std::vector<BoundWeight> bound;
    bound.reserve(references.size());
    for (auto& reference : references) { bound.push_back({reference.name, artifact::bind_view(reference, backing)}); }
    auto model = std::unique_ptr<Model>(new Model(std::move(config), std::move(weights), std::move(bound), std::move(backing)));
    if (dspark) {
        check_dspark_formats(model->bound_, dspark_weights, model->weights_);
        for (const DSparkLayerWeights& l : dspark_weights.layers) {
            for (const WeightId id : {l.q, l.k, l.v, l.output, l.context_k, l.context_v, l.gate, l.up, l.down}) { (void)model->matrix(id); }
        }
        (void)model->matrix(dspark_weights.joint);
    }
    model->dspark_config_  = std::move(dspark);
    model->dspark_weights_ = std::move(dspark_weights);
    model->vision_config_  = std::move(vision);
    model->vision_weights_ = std::move(vision_weights);
    model->resources_      = resources;
    return model;
}

} // namespace ninfer::models::inkling2
