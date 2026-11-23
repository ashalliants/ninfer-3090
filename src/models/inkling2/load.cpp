// Written for NInfer-3090. The pattern follows src/models/dsv4/load.cpp; every parameter is device resident (the BF16
// expert banks as their exact MXFP8 re-encoding with --mxfp8-experts, read from the file and never uploaded as BF16).

#include "models/inkling2/model.h"

#include "artifact/binder.h"
#include "artifact/views.h"
#include "core/arena.h"
#include "core/startup.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdlib>
#include <optional>
#include <set>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>

namespace ninfer::models::inkling2 {
namespace {

using artifact::ArtifactError;
using artifact::Shape;

class Bindings {
public:
    explicit Bindings(artifact::Binder& binder) : binder_(binder) {}

    WeightId parameter(const std::string& name, Shape shape, std::optional<artifact::Residency> where = std::nullopt) {
        const WeightId id{references.size()};
        references.push_back(binder_.parameter(name, std::move(shape), where.value_or(residency)));
        return id;
    }

    // Where the parameters bound next live: on the device, or (vision overlay) in pinned host memory.
    artifact::Residency residency = artifact::Residency::Device;
    // --mxfp8-experts: the BF16 expert banks stay in the file (load_model re-encodes them from it).
    bool bf16_experts_in_file = false;

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
        const std::optional<artifact::Residency> where =
            b.bf16_experts_in_file ? std::optional{artifact::Residency::File} : std::nullopt;
        w.experts_gate_up = b.parameter(m + "experts_gate_up", {e, 2 * i, h}, where);
        w.experts_down    = b.parameter(m + "experts_down", {e, h, i}, where);
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
    // The layers' matrices live in pinned host memory: the drafter reads them once, into its own device copies
    // (DSparkProgram's split Q8 forms), so a device copy here would sit unused.
    const auto pinned = [&](const std::string& name, Shape shape) {
        b.residency   = artifact::Residency::Pinned;
        const auto id = b.parameter(name, std::move(shape));
        b.residency   = artifact::Residency::Device;
        return id;
    };
    for (std::uint32_t i = 0; i < c.layers; ++i) {
        const std::string p = "dspark/layers/" + std::to_string(i) + "/", a = p + "attention/";
        DSparkLayerWeights l;
        l.attn_norm      = b.parameter(a + "norm", {h});
        l.q              = pinned(a + "q", {q, h});
        l.k              = pinned(a + "k", {kv, h});
        l.v              = pinned(a + "v", {kv, h});
        l.output         = pinned(a + "output", {h, q});
        l.q_norm         = b.parameter(a + "q_norm", {d});
        l.k_norm         = b.parameter(a + "k_norm", {d});
        l.context_k      = pinned(a + "context_k", {kv, h});
        l.context_v      = pinned(a + "context_v", {kv, h});
        l.context_k_norm = b.parameter(a + "context_k_norm", {d});
        l.context_norm   = b.parameter(p + "context_norm", {h});
        l.ffn_norm       = b.parameter(p + "ffn/norm", {h});
        l.gate           = pinned(p + "ffn/gate", {f, h});
        l.up             = pinned(p + "ffn/up", {f, h});
        l.down           = pinned(p + "ffn/down", {h, f});
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
    } else if (bound.at(w.experts_gate_up.index).view.parts.empty()) {
        bank.format = QType::BF16; // left in the file: load_model fills in the MXFP8 arrays
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

// --mxfp8-experts auto: re-encode when the parameters bound to the device (the text model's, and the vision tower's
// when resident) leave less than kMxfp8AutoRoom of free device memory. The artifact's BF16 expert banks only.
bool want_mxfp8(const artifact::Reader& reader, const LoadComponents& components) {
    if (components.experts_mxfp8 == ExpertsMxfp8::Off) { return false; }
    const auto& directory = reader.directory();
    bool any = false;
    for (const auto& [name, binding] : directory.bindings) {
        (void)binding;
        if (name.starts_with("text/layers/") && name.ends_with("/moe/experts_gate_up")) { any = true; }
    }
    if (!any) { return false; } // every layer's experts are NVFP4 records
    if (components.experts_mxfp8 == ExpertsMxfp8::On) { return true; }
    std::set<std::size_t> seen;
    std::uint64_t need = 0;
    const bool vision = components.vision && !components.vision_overlay;
    for (const auto& [name, binding] : directory.bindings) {
        if (!name.starts_with("text/") && !(vision && name.starts_with("vision/"))) { continue; }
        for (const auto& part : binding.parts) {
            if (seen.insert(part.object.index).second) { need += artifact::object_bytes(directory.object(part.object)); }
        }
    }
    std::size_t free_bytes = 0, total_bytes = 0;
    if (cudaMemGetInfo(&free_bytes, &total_bytes) != cudaSuccess) { throw std::runtime_error("inkling2: cudaMemGetInfo failed"); }
    return need + kMxfp8AutoRoom > free_bytes;
}

// A BF16 bank left in the file: its single object, checked to hold exactly `rows` x `cols` contiguous BF16 values.
std::uint64_t file_bank_offset(const artifact::Reader& reader, const artifact::ParameterReference& reference, std::uint64_t rows,
                               std::uint64_t cols) {
    const auto& parts = reference.binding.parts;
    if (parts.size() != 1) { throw ArtifactError(reference.name + ": a BF16 expert bank must be one stored object"); }
    const auto& g = reader.geometry(parts.front().object);
    if (g.format != QType::BF16 || g.layout != QuantLayout::Contiguous || parts.front().begin != 0 || parts.front().end != rows * cols ||
        g.elements != rows * cols) {
        throw ArtifactError(reference.name + ": BF16 expert banks must be contiguous BF16");
    }
    return artifact::object_offset(reader.directory().object(parts.front().object));
}

} // namespace

Model::Model(TextConfig config, TextWeights weights, std::vector<BoundWeight> bound, artifact::MaterializedArtifact backing)
    : backing_(std::move(backing)), config_(std::move(config)), weights_(std::move(weights)), bound_(std::move(bound)) {
    for (const auto& layer : weights_.layers) { banks_.push_back(bind_bank(bound_, layer, config_)); }
    tile_expert_records();
}

// The NVFP4 experts' records, resident in device memory, are repacked once into the order their decode reads
// (ops::nvfp4_tile_records: a permutation of the blocks, so every result is the same bits).
void Model::tile_expert_records() {
    std::uint64_t scratch_bytes = 0;
    for (const ops::SituExpertBank& bank : banks_) {
        if (bank.format == QType::GGML_REC_NVFP4_NVFP4) { scratch_bytes = std::max(scratch_bytes, bank.record_stride); }
    }
    if (scratch_bytes == 0) { return; }
    DeviceBuffer scratch(scratch_bytes);
    std::vector<const std::uint8_t*> done;
    for (ops::SituExpertBank& bank : banks_) {
        if (bank.format != QType::GGML_REC_NVFP4_NVFP4) { continue; }
        if (std::find(done.begin(), done.end(), bank.records) != done.end()) { // a bank another layer shares
            bank.tiled = true;
            continue;
        }
        cudaPointerAttributes attributes{};
        if (cudaPointerGetAttributes(&attributes, bank.records) != cudaSuccess || attributes.type != cudaMemoryTypeDevice) {
            (void)cudaGetLastError();
            throw ArtifactError("inkling2: NVFP4 expert records must be resident in device memory");
        }
        ops::nvfp4_tile_records(bank, const_cast<std::uint8_t*>(bank.records), scratch.p, nullptr);
        done.push_back(bank.records);
    }
    if (cudaStreamSynchronize(nullptr) != cudaSuccess) { throw std::runtime_error("inkling2: tiling the expert records failed"); }
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
    bindings.bf16_experts_in_file = want_mxfp8(reader, components);
    TextWeights weights = bind_text(bindings, config);
    std::optional<DSparkConfig> dspark;
    DSparkWeights dspark_weights;
    if (components.dspark && reader.directory().components.contains("dspark")) {
        dspark         = parse_dspark_config(reader.directory().component("dspark").config, config);
        dspark_weights = bind_dspark(bindings, *dspark, config);
    }
    std::optional<VisionConfig> vision;
    VisionWeights vision_weights;
    std::size_t vision_first = 0, vision_end = 0;
    if (components.vision && reader.directory().components.contains("vision")) {
        vision         = parse_vision_config(reader.directory().component("vision").config, config);
        vision_first   = bindings.references.size();
        if (components.vision_overlay) { bindings.residency = artifact::Residency::Pinned; }
        vision_weights = bind_vision(bindings, *vision);
        bindings.residency = artifact::Residency::Device;
        vision_end         = bindings.references.size();
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
    for (auto& reference : references) {
        // A bank left in the file has no view: it is re-encoded below.
        bound.push_back({reference.name, reference.residency == artifact::Residency::File ? WeightView{} : artifact::bind_view(reference, backing)});
    }
    auto model = std::unique_ptr<Model>(new Model(std::move(config), std::move(weights), std::move(bound), std::move(backing)));
    if (bindings.bf16_experts_in_file) {
        // Each BF16 bank, an expert at a time: read from the file into pinned staging (two slots, so the next read
        // overlaps this one's copy and encode), copied to the device and encoded into the layer's MXFP8 arrays.
        const TextConfig& c = model->config_;
        const std::uint64_t h = c.hidden_size, inter = c.moe_intermediate, e = c.experts;
        const std::uint64_t gu_values = e * 2 * inter * h, dn_values = e * h * inter;
        const auto align = [](std::uint64_t n) { return (n + 255) / 256 * 256; };
        const std::uint64_t gu_codes = 0, gu_scales = align(gu_values), dn_codes = gu_scales + align(gu_values / 32),
                            dn_scales = dn_codes + align(dn_values), layer_bytes = dn_scales + align(dn_values / 32);
        std::uint64_t total = 0;
        for (const LayerWeights& l : model->weights_.layers) { total += l.experts.valid() ? 0 : 2 * (gu_values + dn_values); }
        const StartupObserver none{};
        StartupPhaseScope phase(observer != nullptr ? *observer : none, StartupPhase::WeightsReencode, StartupProgressUnit::Bytes, total);
        const std::uint64_t slot_bytes = 2 * inter * h * 2; // one expert's gate_up, the larger matrix
        std::array<PinnedHostBuffer, 2> host{PinnedHostBuffer(slot_bytes), PinnedHostBuffer(slot_bytes)};
        std::array<DeviceBuffer, 2> staging{DeviceBuffer(slot_bytes), DeviceBuffer(slot_bytes)};
        DeviceBuffer count(sizeof(unsigned long long));
        count.fill(0);
        std::array<cudaEvent_t, 2> done{};
        cudaStream_t stream = nullptr;
        const auto check = [](cudaError_t error, const char* what) {
            if (error != cudaSuccess) { throw std::runtime_error(std::string("inkling2: MXFP8 experts: ") + what + ": " + cudaGetErrorString(error)); }
        };
        // Owns the stream and events from before they exist, so a failure part-way through their creation (or any
        // later read, upload or encode) releases whatever was made.
        struct Release {
            cudaStream_t& stream;
            std::array<cudaEvent_t, 2>& done;
            ~Release() {
                if (stream != nullptr) { (void)cudaStreamSynchronize(stream), (void)cudaStreamDestroy(stream); }
                for (auto& event : done) {
                    if (event != nullptr) { (void)cudaEventDestroy(event); }
                }
            }
        } release{stream, done};
        check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "stream");
        for (auto& event : done) { check(cudaEventCreateWithFlags(&event, cudaEventDisableTiming), "event"); }
        const auto started = std::chrono::steady_clock::now();
        std::uint64_t read = 0;
        int slot = 0;
        for (std::size_t p = 0; p < model->weights_.layers.size(); ++p) {
            const LayerWeights& l = model->weights_.layers[p];
            if (l.experts.valid()) { continue; }
            DeviceBuffer arrays(layer_bytes);
            auto* base = static_cast<std::uint8_t*>(arrays.p);
            const std::uint64_t gu_at = file_bank_offset(reader, references.at(l.experts_gate_up.index), e * 2 * inter, h);
            const std::uint64_t dn_at = file_bank_offset(reader, references.at(l.experts_down.index), e * h, inter);
            for (std::uint64_t x = 0; x < e; ++x) {
                for (const bool gate_up : {true, false}) {
                    const std::uint64_t rows = gate_up ? 2 * inter : h, cols = gate_up ? h : inter, bytes = rows * cols * 2;
                    check(cudaEventSynchronize(done[slot]), "staging"); // this slot's previous copy has finished
                    auto* bytes_out = static_cast<std::byte*>(host[slot].data());
                    reader.read_into((gate_up ? gu_at : dn_at) + x * bytes, std::span<std::byte>(bytes_out, bytes));
                    check(cudaMemcpyAsync(staging[slot].p, bytes_out, bytes, cudaMemcpyHostToDevice, stream), "upload");
                    const std::uint64_t first = x * rows * cols;
                    ops::mxfp8_encode_bf16_rows(staging[slot].p, static_cast<std::int64_t>(rows), static_cast<int>(cols),
                                                base + (gate_up ? gu_codes : dn_codes) + first, base + (gate_up ? gu_scales : dn_scales) + first / 32,
                                                static_cast<unsigned long long*>(count.p), stream);
                    check(cudaEventRecord(done[slot], stream), "record");
                    slot ^= 1;
                    read += bytes;
                    phase.progress(read, total);
                }
            }
            ops::SituExpertBank& bank = model->banks_.at(p);
            bank.mx_gate_up        = base + gu_codes;
            bank.mx_gate_up_scales = base + gu_scales;
            bank.mx_down           = base + dn_codes;
            bank.mx_down_scales    = base + dn_scales;
            model->mxfp8_banks_.push_back(std::move(arrays));
        }
        check(cudaStreamSynchronize(stream), "encode");
        unsigned long long inexact = 0;
        count.copy_to_host(&inexact, sizeof inexact);
        if (inexact != 0) {
            throw ArtifactError("inkling2: " + std::to_string(inexact) +
                                " values of the BF16 expert banks are not MXFP8 values, so they cannot be re-encoded exactly; "
                                "start with --mxfp8-experts off");
        }
        model->mxfp8_load_ = {.read_bytes    = read,
                              .h2d_bytes     = read,
                              .staging_bytes = 2 * slot_bytes,
                              .seconds       = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count()};
        phase.complete(read, total);
    }
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
    if (components.vision_overlay) {
        for (std::size_t i = vision_first; i < vision_end; ++i) {
            for (const WeightRegion& part : model->bound_[i].view.parts) {
                if (std::find(model->vision_overlay_.begin(), model->vision_overlay_.end(), part.parent) == model->vision_overlay_.end()) {
                    model->vision_overlay_.push_back(part.parent);
                }
            }
        }
    }
    model->resources_      = resources;
    return model;
}

} // namespace ninfer::models::inkling2
