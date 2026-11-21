// Written for NInfer-3090. See adapters.h. The files are read as data only: safetensors headers parsed as JSON, tensor
// bytes copied out by their offsets; nothing in them is executed.

#include "models/inkling2/program/adapters.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <stdexcept>
#include <string_view>

namespace ninfer::models::inkling2 {
namespace {

using Json = nlohmann::json;

// An adapter of either form is a few megabytes at most; anything near this is not one.
constexpr std::uint64_t kMaxFileBytes = 256ULL << 20;

class Refusal {
public:
    Refusal(const char* kind, std::string name) : kind_(kind), name_(std::move(name)) {}
    [[noreturn]] void fail(const std::string& message) const {
        throw std::invalid_argument(std::string(kind_) + " '" + name_ + "': " + message);
    }
    void require(bool condition, const std::string& message) const {
        if (!condition) { fail(message); }
    }

private:
    const char* kind_;
    std::string name_;
};

struct Fnv {
    std::uint64_t value = 1469598103934665603ULL;
    void bytes(const void* data, std::size_t count) {
        const auto* p = static_cast<const std::uint8_t*>(data);
        for (std::size_t i = 0; i < count; ++i) {
            value ^= p[i];
            value *= 1099511628211ULL;
        }
    }
    [[nodiscard]] std::uint64_t get() const { return value == 0 ? 1 : value; }
};

std::vector<std::uint8_t> read_file(const std::filesystem::path& path, const Refusal& refusal) {
    std::error_code error;
    const std::uint64_t size = std::filesystem::file_size(path, error);
    refusal.require(!error, "cannot read " + path.string() + (error ? ": " + error.message() : std::string()));
    refusal.require(size <= kMaxFileBytes, path.filename().string() + " is " + std::to_string(size) + " bytes, far more than an adapter");
    std::vector<std::uint8_t> bytes(static_cast<std::size_t>(size));
    std::ifstream in(path, std::ios::binary);
    refusal.require(in.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(bytes.size())).good() || bytes.empty(),
                    "cannot read " + path.string());
    return bytes;
}

struct Entry {
    std::vector<std::uint64_t> shape;
    const std::uint8_t* data = nullptr;
    std::size_t bytes        = 0;
};

struct Safetensors {
    std::map<std::string, Entry> tensors;
    Json metadata = Json::object();
};

// The header of an F32 safetensors file (every tensor F32, data in range).
Safetensors parse_safetensors(const std::vector<std::uint8_t>& file, const std::string& what, const Refusal& refusal) {
    refusal.require(file.size() >= 8, what + " is not a safetensors file (shorter than its header)");
    std::uint64_t header_bytes = 0;
    for (int byte = 7; byte >= 0; --byte) { header_bytes = (header_bytes << 8U) | file[static_cast<std::size_t>(byte)]; }
    refusal.require(header_bytes >= 2 && header_bytes <= file.size() - 8 && header_bytes <= (16ULL << 20),
                    what + " is not a safetensors file (its header runs past the file)");
    Json header;
    try {
        header = Json::parse(std::string_view(reinterpret_cast<const char*>(file.data() + 8), static_cast<std::size_t>(header_bytes)));
    } catch (const Json::exception& error) {
        refusal.fail(what + " is not a safetensors file (its header is not JSON: " + error.what() + ")");
    }
    refusal.require(header.is_object(), what + " is not a safetensors file (its header is not an object)");
    const std::uint64_t payload_offset = 8 + header_bytes, payload = file.size() - payload_offset;
    Safetensors out;
    for (const auto& [name, value] : header.items()) {
        if (name == "__metadata__") {
            if (value.is_object()) { out.metadata = value; }
            continue;
        }
        refusal.require(value.is_object() && value.contains("dtype") && value.contains("shape") && value.contains("data_offsets") &&
                            value.at("dtype").is_string() && value.at("shape").is_array() && value.at("data_offsets").is_array() &&
                            value.at("data_offsets").size() == 2,
                        what + ": tensor '" + name + "' has an incomplete entry");
        refusal.require(value.at("dtype").get<std::string>() == "F32",
                        what + ": tensor '" + name + "' is " + value.at("dtype").get<std::string>() + "; Inkling-2 adapters are F32");
        Entry entry;
        std::uint64_t count = 1;
        for (const Json& d : value.at("shape")) {
            refusal.require(d.is_number_unsigned() && d.get<std::uint64_t>() <= (1ULL << 31), what + ": tensor '" + name + "' has an invalid shape");
            entry.shape.push_back(d.get<std::uint64_t>());
            count *= entry.shape.back();
            refusal.require(count <= (1ULL << 31), what + ": tensor '" + name + "' is too large");
        }
        const Json& offsets = value.at("data_offsets");
        refusal.require(offsets[0].is_number_unsigned() && offsets[1].is_number_unsigned(), what + ": tensor '" + name + "' has invalid offsets");
        const std::uint64_t begin = offsets[0].get<std::uint64_t>(), end = offsets[1].get<std::uint64_t>();
        refusal.require(begin <= end && end <= payload && end - begin == count * 4,
                        what + ": tensor '" + name + "' offsets do not match its shape");
        entry.data  = file.data() + payload_offset + begin;
        entry.bytes = static_cast<std::size_t>(end - begin);
        out.tensors.emplace(name, std::move(entry));
    }
    return out;
}

std::vector<float> floats(const Entry& entry, const std::string& name, const Refusal& refusal) {
    std::vector<float> values(entry.bytes / 4);
    std::memcpy(values.data(), entry.data, entry.bytes);
    for (const float v : values) { refusal.require(std::isfinite(v), "tensor '" + name + "' holds a non-finite value"); }
    return values;
}

// A PEFT setting that changes the formula is refused unless it holds its plain-LoRA value.
void refuse_set(const Json& config, const char* key, const char* why, const Refusal& refusal) {
    if (!config.contains(key)) { return; }
    const Json& v    = config.at(key);
    const bool unset = v.is_null() || (v.is_boolean() && !v.get<bool>()) || (v.is_object() && v.empty()) || (v.is_array() && v.empty()) ||
                       (v.is_string() && (v.get<std::string>().empty() || v.get<std::string>() == "none"));
    refusal.require(unset, std::string("adapter_config.json sets ") + key + " (" + why + "); Inkling-2 serves plain LoRA only");
}

std::filesystem::path adapter_directory(const std::filesystem::path& path, const Refusal& refusal) {
    std::error_code error;
    if (std::filesystem::is_directory(path, error)) { return path; }
    const auto file = path.filename();
    if (file == "adapter_config.json" || file == "adapter_model.safetensors") {
        return path.parent_path().empty() ? std::filesystem::path(".") : path.parent_path();
    }
    refusal.fail("'" + path.string() + "' is not a PEFT adapter directory (one holding adapter_config.json and adapter_model.safetensors)");
}

// "base_model.model.model.llm.layers.N.attn.wo_ud.lora_A.weight" -> N and A or B; any other name is refused.
std::pair<std::uint32_t, bool> parse_lora_name(const std::string& name, std::uint32_t layers, const Refusal& refusal) {
    constexpr std::string_view kPrefix = "base_model.model.model.llm.layers.", kTarget = ".attn.wo_ud.lora_";
    const auto unsupported = [&] {
        refusal.fail("tensor '" + name + "' targets a module Inkling-2 cannot adapt; the only supported LoRA target is "
                     "model.llm.layers.N.attn.wo_ud (the attention output projection)");
    };
    if (name.rfind(kPrefix, 0) != 0) { unsupported(); }
    std::size_t at = kPrefix.size(), digits = 0;
    std::uint64_t layer = 0;
    while (at < name.size() && name[at] >= '0' && name[at] <= '9' && digits < 6) { layer = layer * 10 + static_cast<std::uint64_t>(name[at++] - '0'), ++digits; }
    if (digits == 0 || name.compare(at, kTarget.size(), kTarget) != 0) { unsupported(); }
    const std::string rest = name.substr(at + kTarget.size());
    if (rest != "A.weight" && rest != "B.weight") { unsupported(); }
    refusal.require(layer < layers, "tensor '" + name + "' names layer " + std::to_string(layer) + "; Inkling-2 has " + std::to_string(layers));
    return {static_cast<std::uint32_t>(layer), rest == "A.weight"};
}

} // namespace

LoraAdapter load_lora_adapter(const LoraSource& source, const TextConfig& config) {
    const Refusal refusal("LoRA adapter", source.name);
    const auto directory = adapter_directory(source.path, refusal);
    const auto config_bytes = read_file(directory / "adapter_config.json", refusal);
    Json settings;
    try {
        settings = Json::parse(config_bytes.begin(), config_bytes.end());
    } catch (const Json::exception& error) {
        refusal.fail(std::string("adapter_config.json is not JSON: ") + error.what());
    }
    refusal.require(settings.is_object(), "adapter_config.json is not an object");
    refusal.require(settings.contains("peft_type") && settings.at("peft_type").is_string() && settings.at("peft_type").get<std::string>() == "LORA",
                    "adapter_config.json peft_type is not LORA");
    if (settings.contains("task_type") && settings.at("task_type").is_string()) {
        refusal.require(settings.at("task_type").get<std::string>() == "CAUSAL_LM", "adapter_config.json task_type is not CAUSAL_LM");
    }
    refuse_set(settings, "use_dora", "DoRA magnitude vectors", refusal);
    refuse_set(settings, "rank_pattern", "per-module ranks", refusal);
    refuse_set(settings, "alpha_pattern", "per-module alphas", refusal);
    refuse_set(settings, "lora_bias", "a bias on lora_B", refusal);
    refuse_set(settings, "bias", "trained base biases", refusal);
    refuse_set(settings, "modules_to_save", "fully trained modules", refusal);
    refuse_set(settings, "fan_in_fan_out", "transposed weights", refusal);
    refuse_set(settings, "layer_replication", "replicated layers", refusal);
    refuse_set(settings, "alora_invocation_tokens", "activated LoRA", refusal);
    refuse_set(settings, "trainable_token_indices", "trained token embeddings", refusal);
    refuse_set(settings, "target_parameters", "parameter targets", refusal);
    refuse_set(settings, "use_qalora", "QA-LoRA pooling", refusal);
    refusal.require(settings.contains("r") && settings.at("r").is_number_integer(), "adapter_config.json has no integer r");
    refusal.require(settings.contains("lora_alpha") && settings.at("lora_alpha").is_number(), "adapter_config.json has no numeric lora_alpha");
    LoraAdapter out;
    out.name        = source.name;
    const auto rank = settings.at("r").get<std::int64_t>();
    refusal.require(rank >= 1 && rank <= 16, "r must be in [1, 16] (the low-rank update's range)");
    out.rank  = static_cast<std::int32_t>(rank);
    out.alpha = settings.at("lora_alpha").get<float>();
    refusal.require(std::isfinite(out.alpha) && out.alpha != 0.0F, "lora_alpha must be finite and nonzero");
    const bool rslora = settings.contains("use_rslora") && settings.at("use_rslora").is_boolean() && settings.at("use_rslora").get<bool>();
    out.scale = static_cast<float>(rslora ? out.alpha / std::sqrt(static_cast<double>(out.rank)) : static_cast<double>(out.alpha) / out.rank);

    const auto file = read_file(directory / "adapter_model.safetensors", refusal);
    const Safetensors st = parse_safetensors(file, "adapter_model.safetensors", refusal);
    const std::uint64_t in = std::uint64_t(config.heads) * config.head_dim, hidden = config.hidden_size;
    std::vector<std::optional<std::vector<float>>> a(config.num_hidden_layers), b(config.num_hidden_layers);
    Fnv digest;
    digest.bytes(&out.scale, sizeof out.scale);
    for (const auto& [name, entry] : st.tensors) {
        const auto [layer, is_a] = parse_lora_name(name, config.num_hidden_layers, refusal);
        const std::vector<std::uint64_t> want = is_a ? std::vector<std::uint64_t>{std::uint64_t(out.rank), in}
                                                     : std::vector<std::uint64_t>{hidden, std::uint64_t(out.rank)};
        refusal.require(entry.shape == want, "tensor '" + name + "' is not [" + std::to_string(want[0]) + ", " + std::to_string(want[1]) +
                                                 "] (r " + std::to_string(out.rank) + ", Inkling-2-Mini's wo_ud)");
        auto& slot = (is_a ? a : b)[layer];
        refusal.require(!slot, "tensor '" + name + "' appears twice");
        slot = floats(entry, name, refusal);
        digest.bytes(name.data(), name.size());
        digest.bytes(entry.data, entry.bytes);
    }
    out.layers.resize(config.num_hidden_layers);
    std::size_t adapted = 0;
    for (std::uint32_t l = 0; l < config.num_hidden_layers; ++l) {
        refusal.require(a[l].has_value() == b[l].has_value(), "layer " + std::to_string(l) + "'s wo_ud has only one of lora_A and lora_B");
        if (a[l]) {
            out.layers[l] = LoraAdapter::Pair{std::move(*a[l]), std::move(*b[l])};
            ++adapted;
        }
    }
    refusal.require(adapted > 0, "adapter_model.safetensors adapts no layer");
    out.digest = digest.get();
    return out;
}

VisitAblation load_visit_ablation(const GraftSource& source, const TextConfig& config) {
    const Refusal refusal("graft", source.name);
    const auto file = read_file(source.path, refusal);
    const std::string supported = "an Inkling-2 graft is a per-visit ablation safetensors file (attn_dirs, attn_w, mlp_dirs, mlp_w; "
                                  "metadata params.scope \"per_visit\")";
    refusal.require(file.size() < 4 || std::memcmp(file.data(), "GGUF", 4) != 0,
                    "a GGUF file (a control vector or KV graft) is not supported for Inkling-2; " + supported);
    const Safetensors st = parse_safetensors(file, source.path.filename().string(), refusal);
    std::string scope;
    if (st.metadata.contains("params") && st.metadata.at("params").is_string()) {
        try {
            const Json params = Json::parse(st.metadata.at("params").get<std::string>());
            if (params.is_object() && params.contains("scope") && params.at("scope").is_string()) { scope = params.at("scope").get<std::string>(); }
        } catch (const Json::exception&) {}
    }
    refusal.require(scope == "per_visit", "the file's metadata has no params.scope \"per_visit\" (" +
                                              (scope.empty() ? std::string("none") : "\"" + scope + "\"") + "); " + supported);
    const std::uint32_t visits = config.num_layer_visits, hidden = config.hidden_size;
    refusal.require(st.tensors.size() == 4, "the file holds " + std::to_string(st.tensors.size()) + " tensors; " + supported);
    VisitAblation out;
    out.name = source.name, out.visits = visits, out.hidden = hidden;
    const auto take = [&](const char* name, std::vector<std::uint64_t> shape) {
        const auto found = st.tensors.find(name);
        refusal.require(found != st.tensors.end(), std::string("the file has no tensor ") + name + "; " + supported);
        refusal.require(found->second.shape == shape,
                        std::string("tensor ") + name + " is not of Inkling-2-Mini's shape (" + std::to_string(visits) + " visits, hidden " +
                            std::to_string(hidden) + ")");
        return floats(found->second, name, refusal);
    };
    out.attn_dirs = take("attn_dirs", {visits, hidden});
    out.attn_w    = take("attn_w", {visits});
    out.mlp_dirs  = take("mlp_dirs", {visits, hidden});
    out.mlp_w     = take("mlp_w", {visits});
    // Unit directions, as the reference's Ablation.set (F.normalize: v / max(|v|, 1e-12)).
    for (auto* dirs : {&out.attn_dirs, &out.mlp_dirs}) {
        for (std::uint32_t v = 0; v < visits; ++v) {
            float* r    = dirs->data() + std::size_t(v) * hidden;
            double norm = 0.0;
            for (std::uint32_t c = 0; c < hidden; ++c) { norm += double(r[c]) * r[c]; }
            norm = std::max(std::sqrt(norm), 1e-12);
            for (std::uint32_t c = 0; c < hidden; ++c) { r[c] = static_cast<float>(r[c] / norm); }
        }
    }
    Fnv digest;
    digest.bytes(file.data(), file.size());
    out.digest = digest.get();
    return out;
}

std::vector<double> lora_reference_delta(const LoraAdapter::Pair& pair, std::int32_t rank, double scale, const std::vector<double>& x) {
    const std::size_t in = x.size(), out = pair.b.size() / static_cast<std::size_t>(rank);
    std::vector<double> u(static_cast<std::size_t>(rank), 0.0), y(out, 0.0);
    for (std::size_t j = 0; j < u.size(); ++j) {
        for (std::size_t k = 0; k < in; ++k) { u[j] += double(pair.a[j * in + k]) * x[k]; }
        u[j] *= scale;
    }
    for (std::size_t n = 0; n < out; ++n) {
        for (std::size_t j = 0; j < u.size(); ++j) { y[n] += double(pair.b[n * u.size() + j]) * u[j]; }
    }
    return y;
}

std::vector<double> ablation_reference(const float* r, double w, const std::vector<double>& y) {
    double d = 0.0;
    for (std::size_t c = 0; c < y.size(); ++c) { d += double(r[c]) * y[c]; }
    std::vector<double> out(y.size());
    for (std::size_t c = 0; c < y.size(); ++c) { out[c] = y[c] - w * d * double(r[c]); }
    return out;
}

DeviceLora::DeviceLora(const LoraAdapter& adapter) : rank(adapter.rank), scale(adapter.scale) {
    std::size_t total = 0;
    for (const auto& layer : adapter.layers) {
        if (layer) { total += (layer->a.size() + layer->b.size()) * 4; }
    }
    memory = DeviceBuffer(total);
    std::size_t offset = 0;
    for (const auto& layer : adapter.layers) {
        if (!layer) {
            a.push_back(nullptr), b.push_back(nullptr);
            continue;
        }
        const auto* base = static_cast<const std::byte*>(memory.p);
        memory.copy_from_host(layer->a.data(), layer->a.size() * 4, offset);
        a.push_back(reinterpret_cast<const float*>(base + offset));
        offset += layer->a.size() * 4;
        memory.copy_from_host(layer->b.data(), layer->b.size() * 4, offset);
        b.push_back(reinterpret_cast<const float*>(base + offset));
        offset += layer->b.size() * 4;
    }
}

DeviceAblation::DeviceAblation(const VisitAblation& ablation) : attn_w(ablation.attn_w), mlp_w(ablation.mlp_w) {
    const std::size_t plane = std::size_t(ablation.visits) * ablation.hidden * 4;
    memory = DeviceBuffer(2 * plane);
    memory.copy_from_host(ablation.attn_dirs.data(), plane, 0);
    memory.copy_from_host(ablation.mlp_dirs.data(), plane, plane);
    const auto* base = static_cast<const float*>(memory.p);
    for (std::uint32_t v = 0; v < ablation.visits; ++v) {
        attn_dir.push_back(base + std::size_t(v) * ablation.hidden);
        mlp_dir.push_back(base + std::size_t(ablation.visits + v) * ablation.hidden);
    }
}

} // namespace ninfer::models::inkling2
