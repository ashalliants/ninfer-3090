// Written for NInfer-3090. The pattern follows src/models/qwen4_exp/load.cpp: the parameters are
// the converter's (tools/convert/dsv4.py name_map), each bound in the format the GGUF stored; the
// expert banks are pinned host memory or, with `in_file`, left in the artifact.

#include "models/dsv4/model.h"

#include "artifact/binder.h"
#include "artifact/formats.h"
#include "artifact/reader.h"
#include "artifact/views.h"
#include "models/qwen3_5/config.h"

#include <cuda_runtime.h>

#include <map>
#include <string>
#include <string_view>
#include <utility>

namespace ninfer::models::dsv4 {
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

// Logical demands of the model; each parameter is declared exactly once, in the format it was stored in.
class Bindings {
public:
    explicit Bindings(artifact::Binder& binder) : binder_(binder) {}

    WeightId parameter(std::string name, Shape shape, std::string_view input = {},
                       Residency residency = Residency::Device) {
        if (ids_.contains(name)) { throw ArtifactError(name + ": duplicate parameter declaration"); }
        PendingWeight pending;
        pending.reference = binder_.parameter(name, std::move(shape), residency, std::nullopt);
        if (!input.empty()) {
            const auto& use = binder_.use(name, input);
            if (!use.activation_policy) { throw ArtifactError(name + "@" + std::string(input) + ": missing activation policy"); }
            if (!use.auxiliaries.empty()) { throw ArtifactError(name + ": DeepSeek-V4 weights carry no Use auxiliaries"); }
            pending.policy = linear_policy(*use.activation_policy);
        }
        const WeightId id{weights.size()};
        ids_.emplace(std::move(name), id);
        weights.push_back(std::move(pending));
        return id;
    }

    std::vector<PendingWeight> weights;

private:
    artifact::Binder& binder_;
    std::map<std::string, WeightId, std::less<>> ids_;
};

HyperConnectionWeights bind_hc(Bindings& b, const TextConfig& c, const std::string& p) {
    const std::uint64_t mix = std::uint64_t(c.hc_streams) * c.hc_streams + 2 * c.hc_streams;
    HyperConnectionWeights out;
    out.fn    = b.parameter(p + "fn", {mix, c.residual_width()});
    out.base  = b.parameter(p + "base", {mix});
    out.scale = b.parameter(p + "scale", {3});
    return out;
}

AttentionWeights bind_attention(Bindings& b, const TextConfig& c, std::uint32_t layer, const std::string& prefix) {
    const LayerPlan& plan = c.layers[layer];
    const std::uint64_t h = c.hidden_size, rank = c.q_lora_rank, d = c.head_dim, q = c.query_width();
    const std::string p = prefix + "attention/", input = prefix + "attention_input";
    AttentionWeights out;
    out.norm     = b.parameter(p + "norm", {h});
    out.q_a      = b.parameter(p + "q_a", {rank, h}, input);
    out.q_a_norm = b.parameter(p + "q_a_norm", {rank});
    out.q_b      = b.parameter(p + "q_b", {q, rank}, p + "q_a_output");
    out.kv       = b.parameter(p + "kv", {d, h}, input);
    out.kv_norm  = b.parameter(p + "kv_norm", {d});
    out.output_a = b.parameter(p + "output_a", {std::uint64_t(c.o_groups) * c.o_lora_rank, q / c.o_groups}, p + "context");
    out.output_b = b.parameter(p + "output_b", {h, std::uint64_t(c.o_groups) * c.o_lora_rank}, p + "output_a_output");
    out.sinks    = b.parameter(p + "sinks", {c.heads});
    if (plan.kv_source) {
        CompressorWeights comp;
        comp.kv = b.parameter(p + "compressor/kv", {d, h}, input);
        if (plan.compress_ratio > 1) { comp.gate = b.parameter(p + "compressor/gate", {d, h}, input); }
        comp.norm         = b.parameter(p + "compressor/norm", {d});
        comp.index_k      = b.parameter(prefix + "indexer/k", {c.indexer_head_dim, d}, p + "compressor/output");
        comp.index_k_norm = b.parameter(prefix + "indexer/k_norm", {c.indexer_head_dim});
        out.compressor    = comp;
    }
    if (plan.index_source) {
        IndexerWeights idx;
        idx.q_b  = b.parameter(prefix + "indexer/q_b", {std::uint64_t(c.indexer_heads) * c.indexer_head_dim, rank}, p + "q_a_output");
        idx.proj = b.parameter(prefix + "indexer/proj", {c.indexer_heads, h}, input);
        out.indexer = idx;
    }
    return out;
}

MoeWeights bind_moe(Bindings& b, const TextConfig& c, const std::string& prefix, bool in_file) {
    const std::uint64_t h = c.hidden_size, e = c.experts, s = c.moe_intermediate;
    const std::string input = prefix + "ffn_input", p = prefix + "moe/";
    MoeWeights out;
    out.norm        = b.parameter(p + "norm", {h});
    out.router      = b.parameter(p + "router", {e, h}, input);
    out.router_bias = b.parameter(p + "router_bias", {e});
    out.shared.gate = b.parameter(p + "shared/gate", {s, h}, input);
    out.shared.up   = b.parameter(p + "shared/up", {s, h}, input);
    out.shared.down = b.parameter(p + "shared/down", {h, s}, p + "shared/product");
    out.experts     = b.parameter(p + "experts", {e, h, s}, input, in_file ? Residency::File : Residency::Pinned);
    return out;
}

TextWeights bind_text(Bindings& b, const TextConfig& c, bool in_file) {
    TextWeights out;
    out.token_embedding = b.parameter("text/embedding", {c.vocab_size, c.hidden_size});
    out.output_head     = b.parameter("text/head", {c.vocab_size, c.hidden_size}, "text/final_hidden");
    out.output_norm     = b.parameter("text/output_norm", {c.hidden_size});
    for (std::uint32_t i = 0; i < c.num_hidden_layers; ++i) {
        const std::string prefix = "text/layers/" + std::to_string(i) + "/";
        BlockWeights block;
        block.attn_hc   = bind_hc(b, c, prefix + "hyper_connection/attention/");
        block.ffn_hc    = bind_hc(b, c, prefix + "hyper_connection/ffn/");
        block.attention = bind_attention(b, c, i, prefix);
        block.moe       = bind_moe(b, c, prefix, in_file);
        if (c.layers[i].engram) {
            const std::uint64_t width = std::uint64_t(c.engram.ngram_size - 1) * c.engram.heads * c.engram.row_width;
            EngramWeights engram;
            engram.kv   = b.parameter(prefix + "engram/kv", {std::uint64_t(c.hidden_size) * (c.hc_streams + 1), width}, prefix + "engram_rows");
            engram.gate = b.parameter(prefix + "engram/gate", {c.hc_streams, c.hidden_size}, prefix + "attention_input");
            block.engram = engram;
        }
        out.layers.push_back(std::move(block));
    }
    return out;
}

const WeightGeometry& bank_geometry(const artifact::Reader& reader, const PendingWeight& weight,
                                              const TextConfig& c) {
    const auto& parts = weight.reference.binding.parts;
    if (parts.size() != 1 || !weight.reference.binding.whole_object) {
        throw ArtifactError(weight.reference.name + ": an expert bank must be one complete parent");
    }
    const auto& g = reader.geometry(parts.front().object);
    if (g.layout != QuantLayout::GgmlExpertRecord || g.shape.size() != 3 || g.shape[0] != c.experts ||
        g.shape[1] != c.hidden_size || g.shape[2] != c.moe_intermediate) {
        throw ArtifactError(weight.reference.name + ": expert bank geometry differs from the text config");
    }
    return g;
}

ExpertBank file_bank(const artifact::Reader& reader, const PendingWeight& weight, const TextConfig& c) {
    const auto& g = bank_geometry(reader, weight, c);
    ExpertBank bank;
    bank.format          = g.format;
    bank.experts         = c.experts;
    bank.record_stride   = g.record_stride;
    bank.artifact_offset = artifact::object_offset(reader.directory().object(weight.reference.binding.parts.front().object));
    return bank;
}

ExpertBank resident_bank(const BoundWeight& weight, const TextConfig& c) {
    const auto& view = weight.view;
    if (view.parts.size() != 1 || !is_complete_weight(view)) {
        throw ArtifactError(weight.name + ": an expert bank must be one complete parent");
    }
    const auto& parent = *view.parts.front().parent;
    const auto& g      = parent.geometry;
    if (g.layout != QuantLayout::GgmlExpertRecord || g.shape.size() != 3 || g.shape[0] != c.experts) {
        throw ArtifactError(weight.name + ": expert bank geometry differs from the text config");
    }
    ExpertBank bank;
    bank.format        = g.format;
    bank.experts       = c.experts;
    bank.record_stride = g.record_stride;
    bank.host_records  = reinterpret_cast<const std::uint8_t*>(parent.data);
    void* device       = nullptr;
    if (cudaHostGetDevicePointer(&device, const_cast<std::byte*>(parent.data), 0) != cudaSuccess || device == nullptr) {
        throw ArtifactError(weight.name + ": the pinned expert bank is not device-addressable");
    }
    bank.device_records = static_cast<const std::uint8_t*>(device);
    return bank;
}

FrontendResources bind_resources(artifact::Binder& binder, const TextConfig& config) {
    const auto resource = [&](std::string_view role) {
        const auto bytes = binder.host_object(binder.resource("text", role));
        return std::string_view(reinterpret_cast<const char*>(bytes.data()), bytes.size());
    };
    FrontendResources out;
    out.tokenizer_json         = resource("tokenizer.json");
    out.tokenizer_config_json  = resource("tokenizer_config.json");
    out.chat_template_jinja    = resource("chat_template.jinja");
    out.generation_config_json = resource("generation_config.json");
    qwen3_5::Config domain;
    domain.text.vocab_size = config.vocab_size;
    qwen3_5::parse_resources(out, domain);
    return out;
}

} // namespace

struct LoadPlan::Impl {
    TextConfig config;
    TextWeights weights;
    std::vector<PendingWeight> pending;
    artifact::MaterializationPlan materialization;
    FrontendResources resources;
    InstanceInfo info;
    std::vector<std::uint64_t> strides;
    std::vector<ExpertBank> file_banks;
};

LoadPlan::LoadPlan(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
LoadPlan::LoadPlan(LoadPlan&&) noexcept            = default;
LoadPlan& LoadPlan::operator=(LoadPlan&&) noexcept = default;
LoadPlan::~LoadPlan()                              = default;

const TextConfig& LoadPlan::config() const { return impl_->config; }
const FrontendResources& LoadPlan::resources() const { return impl_->resources; }
std::uint64_t LoadPlan::device_bytes() const { return impl_->materialization.device_capacity(0); }
std::uint64_t LoadPlan::pinned_bytes() const { return impl_->materialization.pinned_capacity_bytes; }
const std::vector<std::uint64_t>& LoadPlan::expert_record_strides() const { return impl_->strides; }
const std::vector<ExpertBank>& LoadPlan::file_banks() const { return impl_->file_banks; }

Model::Model(TextConfig config, TextWeights weights, std::vector<BoundWeight> bound, std::vector<ExpertBank> banks,
             FrontendResources resources, InstanceInfo info, artifact::MaterializedArtifact backing)
    : backing_(std::move(backing)), config_(std::move(config)), weights_(std::move(weights)),
      bound_(std::move(bound)), banks_(std::move(banks)), resources_(std::move(resources)), info_(std::move(info)) {}

Model::~Model() = default;

bool is_dsv4(const artifact::Reader& reader) { return is_dsv4_config(reader.directory()); }

LoadPlan plan_load(const artifact::Reader& reader, bool in_file) {
    auto out    = std::make_unique<LoadPlan::Impl>();
    out->config = parse_config(reader.directory());
    artifact::Binder binder(reader);
    out->resources = bind_resources(binder, out->config);
    Bindings bindings(binder);
    out->weights = bind_text(bindings, out->config, in_file);
    out->pending = std::move(bindings.weights);
    for (const auto& layer : out->weights.layers) {
        const auto& weight = out->pending.at(layer.moe.experts.index);
        out->strides.push_back(bank_geometry(reader, weight, out->config).record_stride);
        if (in_file) { out->file_banks.push_back(file_bank(reader, weight, out->config)); }
    }
    out->materialization = std::move(binder).finish();
    out->info.name        = reader.directory().metadata.value("name", std::string("DeepSeek-V4.1-Flash"));
    out->info.artifact_id = reader.artifact_id();
    return LoadPlan(std::move(out));
}

std::unique_ptr<Model> materialize_model(LoadPlan&& plan, DeviceContext& device, const StartupObserver* observer) {
    auto& p      = *plan.impl_;
    auto backing = artifact::materialize(*p.materialization.source, std::move(p.materialization), device, observer);
    std::vector<BoundWeight> bound;
    bound.reserve(p.pending.size());
    for (auto& item : p.pending) {
        // A bank left in the file has no view: the Program reads it.
        bound.push_back({item.reference.name,
                         item.reference.residency == Residency::File ? WeightView{}
                                                                     : artifact::bind_view(item.reference, backing),
                         item.policy});
    }
    std::vector<ExpertBank> banks = std::move(p.file_banks);
    if (banks.empty()) {
        for (const auto& layer : p.weights.layers) { banks.push_back(resident_bank(bound.at(layer.moe.experts.index), p.config)); }
    }
    return std::unique_ptr<Model>(new Model(std::move(p.config), std::move(p.weights), std::move(bound), std::move(banks),
                                            std::move(p.resources), std::move(p.info), std::move(backing)));
}

} // namespace ninfer::models::dsv4
