// Adapted from Infernix a3edb450 src/models/qwen4_exp/load.cpp (Apache-2.0).
// Modified for NInfer-3090: binds the converter's GGUF-mirroring parameters (GGML block-format
// projections, BF16 gamma attention norms, FP32 multiplier hyper-connection and PLE norms) and
// ggml_expert_record_v1 banks in pinned host memory, and on request the MTP drafter (BF16
// projections, its Q2_0 expert bank in device memory) and the vision tower (Qwen3.5's parameters,
// pinned after the banks in Qwen3.5's overlay groups); no expert scales, proposal head or SSD tier;
// the frontend resources are the Qwen3.5 frontend's.

#include "models/qwen4_exp/model.h"

#include "artifact/binder.h"
#include "artifact/reader.h"
#include "artifact/views.h"
#include "models/qwen3_5/vision_config.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <limits>
#include <map>
#include <string_view>
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

MoeWeights bind_moe(Bindings& b, const TextConfig& c, const std::string& prefix, bool drafter = false) {
    const std::uint64_t h = c.hidden_size, e = c.moe.experts, s = c.moe.shared_intermediate;
    const std::string input = prefix + "ffn_input", p = prefix + "moe/";
    // The drafter's shared expert is BF16 (projected in FP32 rows); the text model's is GGML.
    const std::optional<QType> shared = drafter ? std::optional<QType>(QType::BF16) : std::nullopt;
    MoeWeights out;
    out.router       = b.parameter(p + "router", {e, h}, input, QType::BF16);
    out.shared_score = b.parameter(p + "shared_score", {1, h}, input, QType::BF16);
    out.shared_gate  = b.parameter(p + "shared/gate", {s, h}, input, shared);
    out.shared_up    = b.parameter(p + "shared/up", {s, h}, input, shared);
    out.shared_down  = b.parameter(p + "shared/down", {h, s}, p + "shared/product", shared);
    // Text banks stay in pinned host memory (the expert cache stages them); the drafter's 512
    // experts are device-resident.
    out.experts = b.parameter(p + "experts", {e, h, c.moe.intermediate}, input,
                              drafter ? std::optional<QType>(QType::GGML_REC_Q2_0_Q2_0) : std::nullopt,
                              drafter ? Residency::Device : Residency::Pinned);
    return out;
}

// The MTP drafter: every projection BF16 (the GGUF's words).
MtpWeights bind_mtp(Bindings& b, const TextConfig& c) {
    const std::uint64_t h = c.hidden_size, width = c.residual_width();
    const std::uint64_t q = c.attention.query_width(), k = c.attention.key_width();
    const std::uint64_t iq = std::uint64_t(c.qsa.index_heads) * c.qsa.index_head_dim;
    MtpWeights out;
    out.embedding_norm       = b.direct("mtp/embedding_norm", {h}, QType::BF16);
    out.hidden_norm          = b.direct("mtp/hidden_norm", {width}, QType::BF16);
    out.embedding_projection = b.parameter("mtp/embedding_projection", {h, h}, "mtp/embedding_input", QType::BF16);
    out.hidden_projection    = b.parameter("mtp/hidden_projection", {h, h}, "mtp/hidden_input", QType::BF16);
    out.final_mixer          = bind_hc(b, c, "mtp/final_mixer/", false);
    const std::string prefix = "mtp/layers/0/";
    out.attn_hc              = bind_hc(b, c, prefix + "attn_hc/", true);
    out.mlp_hc               = bind_hc(b, c, prefix + "mlp_hc/", true);
    const std::string input = prefix + "mixer_input", p = prefix + "attention/", x = prefix + "indexer/";
    auto& a            = out.attention;
    a.query_gate       = b.parameter(p + "query_gate", {2 * q, h}, input, QType::BF16);
    a.key              = b.parameter(p + "key", {k, h}, input, QType::BF16);
    a.value            = b.parameter(p + "value", {k, h}, input, QType::BF16);
    a.output           = b.parameter(p + "output", {h, q}, p + "gated_output", QType::BF16);
    a.query_norm       = b.direct(p + "query_norm", {c.attention.head_dim}, QType::BF16);
    a.key_norm         = b.direct(p + "key_norm", {c.attention.head_dim}, QType::BF16);
    a.index_query      = b.parameter(x + "query", {iq, h}, input, QType::BF16);
    a.index_key        = b.parameter(x + "key", {c.qsa.index_head_dim, h}, input, QType::BF16);
    a.index_query_norm = b.direct(x + "query_norm", {c.qsa.index_head_dim}, QType::BF16);
    a.index_key_norm   = b.direct(x + "key_norm", {c.qsa.index_head_dim}, QType::BF16);
    out.moe            = bind_moe(b, c, prefix, true);
    return out;
}

// Whether the artifact holds the MTP drafter (component mtp, architecture Qwen4ExpMTP).
bool has_mtp_component(const artifact::Directory& directory) {
    const auto found = directory.components.find("mtp");
    if (found == directory.components.end()) { return false; }
    const auto& component = found->second;
    if (component.target.value_or("") != "text") { throw ArtifactError("mtp: target must be text"); }
    const auto architectures = component.config.find("architectures");
    if (architectures == component.config.end() || !architectures->is_array() || architectures->size() != 1 ||
        architectures->front() != "Qwen4ExpMTP") {
        throw ArtifactError("mtp: the drafter's architecture must be Qwen4ExpMTP");
    }
    return true;
}

// The vision tower under Qwen3.5's names, every parameter BF16 in the pinned block. The binding
// order is the overlay layout's group order (prelude, each layer, merger), so each group's parents
// are adjacent in the block and one copy stages a group.
VisionWeights bind_vision(Bindings& b, const qwen3_5::VisionConfig& config, const TextConfig& text) {
    const std::uint64_t h = config.hidden_size, intermediate = config.intermediate_size;
    const auto parameter = [&](std::string name, Shape shape, std::string input) {
        return b.parameter(std::move(name), std::move(shape), input, QType::BF16, Residency::Pinned);
    };
    const auto direct = [&](std::string name, Shape shape) {
        return b.parameter(std::move(name), std::move(shape), {}, QType::BF16, Residency::Pinned);
    };
    VisionWeights out;
    out.patch_embedding      = parameter("vision/patch_embedding", {h, config.patch_width()}, "vision/patch_input");
    out.patch_embedding_bias = direct("vision/patch_embedding_bias", {h});
    out.position_embedding   = direct("vision/position_embedding", {config.num_position_embeddings, h});
    out.layers.reserve(config.depth);
    for (std::uint32_t i = 0; i < config.depth; ++i) {
        const auto p = "vision/layers/" + std::to_string(i) + "/";
        VisionBlockWeights layer;
        layer.norm1       = {direct(p + "norm1_weight", {h}), direct(p + "norm1_bias", {h})};
        layer.norm2       = {direct(p + "norm2_weight", {h}), direct(p + "norm2_bias", {h})};
        layer.query       = parameter(p + "attention/query", {h, h}, p + "attention_input");
        layer.key         = parameter(p + "attention/key", {h, h}, p + "attention_input");
        layer.value       = parameter(p + "attention/value", {h, h}, p + "attention_input");
        layer.query_bias  = direct(p + "attention/query_bias", {h});
        layer.key_bias    = direct(p + "attention/key_bias", {h});
        layer.value_bias  = direct(p + "attention/value_bias", {h});
        layer.output      = parameter(p + "attention/output", {h, h}, p + "attention_output");
        layer.output_bias = direct(p + "attention/output_bias", {h});
        layer.fc1         = parameter(p + "mlp/fc1", {intermediate, h}, p + "mlp_input");
        layer.fc1_bias    = direct(p + "mlp/fc1_bias", {intermediate});
        layer.fc2         = parameter(p + "mlp/fc2", {h, intermediate}, p + "mlp_activation");
        layer.fc2_bias    = direct(p + "mlp/fc2_bias", {h});
        out.layers.push_back(layer);
    }
    const std::uint64_t merger = config.merger_width();
    out.merger_norm     = {direct("vision/merger/norm_weight", {h}), direct("vision/merger/norm_bias", {h})};
    out.merger_fc1      = parameter("vision/merger/fc1", {merger, merger}, "vision/merger/input");
    out.merger_fc1_bias = direct("vision/merger/fc1_bias", {merger});
    out.merger_fc2      = parameter("vision/merger/fc2", {text.hidden_size, merger}, "vision/merger/activation");
    out.merger_fc2_bias = direct("vision/merger/fc2_bias", {text.hidden_size});
    return out;
}

bool has_vision_component(const artifact::Directory& directory) {
    return directory.components.find("vision") != directory.components.end();
}

constexpr std::size_t kStagingAlignment = 256;

std::size_t staging_align(std::size_t bytes) {
    return (bytes + kStagingAlignment - 1) / kStagingAlignment * kStagingAlignment;
}

// The pinned extent covering exactly the parents of one group (Qwen3.5's overlay group_range): no
// other pinned parent may fall inside it, so uploading the extent stages the group and nothing
// else.
qwen3_5::PinnedRange group_range(std::span<const WeightId> group, std::span<const BoundWeight> bound,
                                 std::span<const std::byte> block) {
    const auto* const base = block.data();
    std::size_t begin      = std::numeric_limits<std::size_t>::max();
    std::size_t end        = 0;
    std::vector<const WeightParent*> members;
    for (const WeightId id : group) {
        for (const WeightRegion& part : bound[id.index].view.parts) {
            const auto* data = part.parent->data;
            if (data < base || data + part.parent->geometry.bytes > base + block.size()) {
                throw ArtifactError(bound[id.index].name + ": a vision weight is outside the pinned block");
            }
            const auto offset = static_cast<std::size_t>(data - base);
            begin             = std::min(begin, offset);
            end               = std::max(end, offset + static_cast<std::size_t>(part.parent->geometry.bytes));
            members.push_back(part.parent);
        }
    }
    if (members.empty() || end <= begin) { throw ArtifactError("a vision weight group has no pinned weights"); }
    for (const BoundWeight& weight : bound) {
        for (const WeightRegion& part : weight.view.parts) {
            const auto* data = part.parent->data;
            if (data < base || data >= base + block.size() ||
                std::find(members.begin(), members.end(), part.parent) != members.end()) {
                continue;
            }
            const auto offset = static_cast<std::size_t>(data - base);
            if (offset < end && offset + part.parent->geometry.bytes > begin) {
                throw ArtifactError("a vision weight group interleaves with " + weight.name);
            }
        }
    }
    return {begin, end - begin};
}

qwen3_5::VisionOverlayLayout vision_layout(const VisionWeights& w, std::span<const BoundWeight> bound,
                                           std::span<const std::byte> block) {
    qwen3_5::VisionOverlayLayout out;
    const std::array prelude{w.patch_embedding, w.patch_embedding_bias, w.position_embedding};
    out.prelude = group_range(prelude, bound, block);
    out.layers.reserve(w.layers.size());
    for (const VisionBlockWeights& l : w.layers) {
        const std::array group{l.norm1.weight, l.norm1.bias, l.norm2.weight, l.norm2.bias, l.query, l.key,
                               l.value, l.query_bias, l.key_bias, l.value_bias, l.output, l.output_bias,
                               l.fc1, l.fc1_bias, l.fc2, l.fc2_bias};
        out.layers.push_back(group_range(group, bound, block));
        out.slot_bytes = std::max(out.slot_bytes, out.layers.back().bytes);
    }
    const std::array merger{w.merger_norm.weight, w.merger_norm.bias, w.merger_fc1, w.merger_fc1_bias,
                            w.merger_fc2, w.merger_fc2_bias};
    out.merger        = group_range(merger, bound, block);
    out.staging_bytes = staging_align(out.prelude.bytes) + staging_align(out.merger.bytes) +
                        2 * staging_align(out.slot_bytes);
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

TextWeights bind_text(Bindings& b, const TextConfig& c, bool hot_embedding) {
    TextWeights out;
    const std::uint64_t h = c.hidden_size, v = c.vocab_size;
    // --hot-embedding-rows: the whole table in the pinned block (mapped), first, before the banks.
    out.token_embedding = b.parameter("text/token_embedding", {v, h}, {}, QType::GGML_IQ4_XS,
                                      hot_embedding ? Residency::Pinned : Residency::Device);
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

ExpertBank expert_bank(const BoundWeight& weight, const TextConfig& c, bool on_device = false) {
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
    if (on_device) {
        // The drafter's bank: device memory, no host copy.
        bank.device_records = reinterpret_cast<const std::uint8_t*>(parent.data);
        return bank;
    }
    bank.host_records  = reinterpret_cast<const std::uint8_t*>(parent.data);
    void* device       = nullptr;
    if (cudaHostGetDevicePointer(&device, const_cast<std::byte*>(parent.data), 0) != cudaSuccess || device == nullptr) {
        throw ArtifactError(weight.name + ": the pinned expert bank is not device-addressable");
    }
    bank.device_records = static_cast<const std::uint8_t*>(device);
    return bank;
}

FrontendResources bind_resources(artifact::Binder& binder, const TextConfig& config,
                                 const std::optional<qwen3_5::VisionConfig>& vision) {
    const auto resource = [&](std::string_view component, std::string_view role) {
        const auto bytes = binder.host_object(binder.resource(component, role));
        return std::string_view(reinterpret_cast<const char*>(bytes.data()), bytes.size());
    };
    FrontendResources out;
    out.tokenizer_json         = resource("text", "tokenizer.json");
    out.tokenizer_config_json  = resource("text", "tokenizer_config.json");
    out.chat_template_jinja    = resource("text", "chat_template.jinja");
    out.generation_config_json = resource("text", "generation_config.json");
    if (vision) {
        out.preprocessor_config_json       = resource("vision", "preprocessor_config.json");
        out.video_preprocessor_config_json = resource("vision", "video_preprocessor_config.json");
    }
    // The shared frontend validates the token domain against the embedding rows and the vision
    // processors' patch geometry against the tower.
    qwen3_5::Config domain;
    domain.text.vocab_size = config.vocab_size;
    domain.vision          = vision;
    qwen3_5::parse_resources(out, domain);
    return out;
}

} // namespace

struct LoadPlan::Impl {
    TextConfig config;
    TextWeights weights;
    std::optional<MtpWeights> mtp;
    std::optional<VisionWeights> vision;
    std::optional<qwen3_5::VisionConfig> vision_config;
    std::vector<PendingWeight> pending;
    artifact::MaterializationPlan materialization;
    FrontendResources resources;
    InstanceInfo info;
    std::vector<std::uint64_t> strides;
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

Model::Model(TextConfig config, TextWeights weights, std::optional<MtpWeights> mtp, std::vector<BoundWeight> bound,
             std::vector<ExpertBank> banks, std::optional<ExpertBank> mtp_bank, std::optional<VisionWeights> vision,
             std::optional<qwen3_5::VisionConfig> vision_config,
             std::optional<qwen3_5::VisionOverlayLayout> vision_layout, FrontendResources resources,
             InstanceInfo info, artifact::MaterializedArtifact backing)
    : backing_(std::move(backing)), config_(std::move(config)), weights_(std::move(weights)), mtp_(std::move(mtp)),
      bound_(std::move(bound)), banks_(std::move(banks)), mtp_bank_(std::move(mtp_bank)), vision_(std::move(vision)),
      vision_config_(std::move(vision_config)), vision_layout_(std::move(vision_layout)),
      resources_(std::move(resources)), info_(std::move(info)) {}

Model::~Model() = default;

bool is_qwen4_exp(const artifact::Reader& reader) { return is_qwen4_exp_config(reader.directory()); }

bool has_mtp(const artifact::Reader& reader) { return has_mtp_component(reader.directory()); }

bool has_vision(const artifact::Reader& reader) { return has_vision_component(reader.directory()); }

LoadPlan plan_load(const artifact::Reader& reader, bool mtp, bool vision, bool hot_embedding) {
    auto out    = std::make_unique<LoadPlan::Impl>();
    out->config = parse_config(reader.directory());
    if (mtp && !has_mtp_component(reader.directory())) {
        throw ArtifactError("this artifact has no MTP drafter: convert it with --mtp-gguf to use --spec mtp");
    }
    if (vision) {
        if (!has_vision_component(reader.directory())) {
            throw ArtifactError("this artifact has no vision tower: convert it with --mmproj to use --vision");
        }
        out->vision_config = qwen3_5::parse_vision_component(reader.directory());
    }
    artifact::Binder binder(reader);
    out->resources = bind_resources(binder, out->config, out->vision_config);
    Bindings bindings(binder);
    out->weights         = bind_text(bindings, out->config, hot_embedding);
    if (mtp) { out->mtp = bind_mtp(bindings, out->config); }
    // After the expert banks: the tower's groups follow them in the pinned block.
    if (vision) { out->vision = bind_vision(bindings, *out->vision_config, out->config); }
    out->pending         = std::move(bindings.weights);
    for (const auto& layer : out->weights.layers) {
        const auto& parts = out->pending.at(layer.moe.experts.index).reference.binding.parts;
        if (parts.size() != 1) { throw ArtifactError("an expert bank must be one complete parent"); }
        out->strides.push_back(reader.geometry(parts.front().object).record_stride);
    }
    out->materialization = std::move(binder).finish();
    out->info.name        = reader.directory().metadata.value("name", std::string("Qwen3.8-Flash-Next"));
    out->info.artifact_id = reader.artifact_id();
    return LoadPlan(std::move(out));
}

std::unique_ptr<Model> materialize_model(LoadPlan&& plan, DeviceContext& device, const StartupObserver* observer) {
    auto& p      = *plan.impl_;
    auto backing = artifact::materialize(*p.materialization.source, std::move(p.materialization), device, observer);
    std::vector<BoundWeight> bound;
    bound.reserve(p.pending.size());
    for (auto& item : p.pending) {
        bound.push_back({item.reference.name, artifact::bind_view(item.reference, backing), item.policy});
    }
    std::vector<ExpertBank> banks;
    for (const auto& layer : p.weights.layers) { banks.push_back(expert_bank(bound.at(layer.moe.experts.index), p.config)); }
    std::optional<ExpertBank> mtp_bank;
    if (p.mtp) {
        mtp_bank = expert_bank(bound.at(p.mtp->moe.experts.index), p.config, /*device=*/true);
        if (mtp_bank->format != QType::GGML_REC_Q2_0_Q2_0) {
            throw ArtifactError("mtp: the drafter's experts must be ggml_rec_q2_0_q2_0 records");
        }
    }
    std::optional<qwen3_5::VisionOverlayLayout> layout;
    if (p.vision) { layout = vision_layout(*p.vision, bound, backing.pinned_block()); }
    return std::unique_ptr<Model>(new Model(std::move(p.config), std::move(p.weights), std::move(p.mtp), std::move(bound),
                                            std::move(banks), std::move(mtp_bank), std::move(p.vision),
                                            std::move(p.vision_config), std::move(layout), std::move(p.resources),
                                            std::move(p.info), std::move(backing)));
}

std::unique_ptr<Model> load_model(const artifact::Reader& reader, DeviceContext& device, bool mtp, bool vision,
                                  bool hot_embedding) {
    return materialize_model(plan_load(reader, mtp, vision, hot_embedding), device);
}

} // namespace ninfer::models::qwen4_exp
