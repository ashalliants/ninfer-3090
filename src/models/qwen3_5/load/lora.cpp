#include "models/qwen3_5/load/lora.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <stdexcept>
#include <string_view>
#include <variant>

namespace ninfer::models::qwen3_5 {
namespace {

using Json = nlohmann::json;

// Bump when the meaning of a loaded adapter changes, so cached context keyed by an older digest is
// never reused.
constexpr std::uint64_t kLoraDigestVersion = 1;

class AdapterError {
public:
    explicit AdapterError(std::string name) : name_(std::move(name)) {}
    [[noreturn]] void fail(const std::string& message) const {
        throw std::invalid_argument("LoRA adapter '" + name_ + "': " + message);
    }
    void require(bool condition, const std::string& message) const {
        if (!condition) { fail(message); }
    }

private:
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
    template <class T>
    void pod(const T& v) {
        bytes(&v, sizeof(T));
    }
    void text(std::string_view s) {
        pod(static_cast<std::uint64_t>(s.size()));
        bytes(s.data(), s.size());
    }
};

struct TensorEntry {
    std::string dtype;
    std::vector<std::uint64_t> shape;
    std::uint64_t begin = 0;
    std::uint64_t end   = 0;
};

std::uint64_t element_bytes(const std::string& dtype) {
    if (dtype == "F32") { return 4; }
    if (dtype == "BF16" || dtype == "F16") { return 2; }
    return 0;
}

float half_to_float(std::uint16_t h) {
    const std::uint32_t sign = static_cast<std::uint32_t>(h & 0x8000U) << 16U;
    std::uint32_t exponent   = (h >> 10U) & 0x1fU;
    std::uint32_t mantissa   = h & 0x3ffU;
    std::uint32_t bits       = 0;
    if (exponent == 0) {
        if (mantissa == 0) {
            bits = sign;
        } else {
            exponent = 127 - 15 + 1;
            while ((mantissa & 0x400U) == 0) {
                mantissa <<= 1U;
                --exponent;
            }
            bits = sign | (exponent << 23U) | ((mantissa & 0x3ffU) << 13U);
        }
    } else if (exponent == 31) {
        bits = sign | 0x7f800000U | (mantissa << 13U);
    } else {
        bits = sign | ((exponent + 127 - 15) << 23U) | (mantissa << 13U);
    }
    return std::bit_cast<float>(bits);
}

std::filesystem::path adapter_directory(const LoraSource& source, const AdapterError& error) {
    std::error_code ec;
    if (std::filesystem::is_directory(source.path, ec)) { return source.path; }
    const auto file = source.path.filename();
    if (file == "adapter_config.json" || file == "adapter_model.safetensors") {
        return source.path.parent_path().empty() ? std::filesystem::path(".") : source.path.parent_path();
    }
    error.fail("'" + source.path.string() +
               "' is not a PEFT adapter directory (one holding adapter_config.json and "
               "adapter_model.safetensors)");
}

Json read_json(const std::filesystem::path& path, const AdapterError& error) {
    std::ifstream in(path, std::ios::binary);
    error.require(static_cast<bool>(in), "cannot read " + path.string());
    try {
        return Json::parse(in);
    } catch (const Json::exception& parse_error) {
        error.fail(path.filename().string() + " is not JSON: " + parse_error.what());
    }
}

// A setting PEFT records that changes the adapter's formula is refused unless it holds its plain
// LoRA value.
void refuse_set(const Json& config, const char* key, const AdapterError& error, const char* why) {
    if (!config.contains(key)) { return; }
    const Json& value = config.at(key);
    const bool unset  = value.is_null() || (value.is_boolean() && !value.get<bool>()) ||
                       (value.is_object() && value.empty()) || (value.is_array() && value.empty()) ||
                       (value.is_string() && (value.get<std::string>().empty() ||
                                              value.get<std::string>() == "none"));
    error.require(unset, std::string("adapter_config.json sets ") + key + " (" + why +
                             "), which has no native route; only plain LoRA is supported");
}

struct Settings {
    std::int32_t rank = 0;
    float alpha       = 0;
    float scale       = 0;
};

Settings read_settings(const Json& config, const AdapterError& error) {
    error.require(config.is_object(), "adapter_config.json is not an object");
    error.require(config.contains("peft_type") && config.at("peft_type").is_string() &&
                      config.at("peft_type").get<std::string>() == "LORA",
                  "adapter_config.json peft_type is not LORA");
    if (config.contains("task_type") && config.at("task_type").is_string()) {
        error.require(config.at("task_type").get<std::string>() == "CAUSAL_LM",
                      "adapter_config.json task_type is not CAUSAL_LM");
    }
    refuse_set(config, "use_dora", error, "DoRA magnitude vectors");
    refuse_set(config, "rank_pattern", error, "per-module ranks");
    refuse_set(config, "alpha_pattern", error, "per-module alphas");
    refuse_set(config, "lora_bias", error, "a bias on lora_B");
    refuse_set(config, "bias", error, "trained base biases");
    refuse_set(config, "modules_to_save", error, "fully trained modules");
    refuse_set(config, "fan_in_fan_out", error, "transposed weights");
    refuse_set(config, "layer_replication", error, "replicated layers");
    refuse_set(config, "alora_invocation_tokens", error, "activated LoRA");
    refuse_set(config, "trainable_token_indices", error, "trained token embeddings");
    refuse_set(config, "target_parameters", error, "parameter targets");
    refuse_set(config, "use_qalora", error, "QA-LoRA pooling");
    refuse_set(config, "use_bdlora", error, "block-diagonal LoRA");
    error.require(config.contains("r") && config.at("r").is_number_integer(),
                  "adapter_config.json has no integer r");
    error.require(config.contains("lora_alpha") && config.at("lora_alpha").is_number(),
                  "adapter_config.json has no numeric lora_alpha");
    Settings out;
    const auto rank = config.at("r").get<std::int64_t>();
    error.require(rank >= 1 && rank <= 256, "r must be in [1, 256]");
    out.rank  = static_cast<std::int32_t>(rank);
    out.alpha = config.at("lora_alpha").get<float>();
    error.require(std::isfinite(out.alpha) && out.alpha != 0.0f, "lora_alpha must be finite and nonzero");
    const bool rslora = config.contains("use_rslora") && config.at("use_rslora").is_boolean() &&
                        config.at("use_rslora").get<bool>();
    out.scale = rslora ? static_cast<float>(out.alpha / std::sqrt(static_cast<double>(out.rank)))
                       : static_cast<float>(static_cast<double>(out.alpha) / out.rank);
    return out;
}

std::map<std::string, TensorEntry> read_header(std::ifstream& in, std::uint64_t file_bytes,
                                               std::uint64_t& payload_offset,
                                               const AdapterError& error) {
    error.require(file_bytes >= 8, "adapter_model.safetensors is shorter than its header");
    std::uint8_t prefix[8];
    in.read(reinterpret_cast<char*>(prefix), 8);
    std::uint64_t header_bytes = 0;
    for (int byte = 7; byte >= 0; --byte) { header_bytes = (header_bytes << 8U) | prefix[byte]; }
    error.require(header_bytes <= file_bytes - 8 && header_bytes <= (64ULL << 20U),
                  "adapter_model.safetensors header runs past the file");
    std::string text(static_cast<std::size_t>(header_bytes), '\0');
    in.read(text.data(), static_cast<std::streamsize>(header_bytes));
    error.require(static_cast<bool>(in), "cannot read the adapter_model.safetensors header");
    payload_offset             = 8 + header_bytes;
    const std::uint64_t payload = file_bytes - payload_offset;
    Json header;
    try {
        header = Json::parse(text);
    } catch (const Json::exception& parse_error) {
        error.fail(std::string("adapter_model.safetensors header is not JSON: ") + parse_error.what());
    }
    error.require(header.is_object(), "adapter_model.safetensors header is not an object");
    std::map<std::string, TensorEntry> tensors;
    for (const auto& [name, value] : header.items()) {
        if (name == "__metadata__") { continue; }
        error.require(value.is_object() && value.contains("dtype") && value.contains("shape") &&
                          value.contains("data_offsets") && value.at("dtype").is_string() &&
                          value.at("shape").is_array() && value.at("data_offsets").is_array(),
                      "tensor '" + name + "' has an incomplete safetensors entry");
        TensorEntry entry;
        entry.dtype = value.at("dtype").get<std::string>();
        std::uint64_t bytes = element_bytes(entry.dtype);
        error.require(bytes != 0, "tensor '" + name + "' is " + entry.dtype + ", not F32, BF16 or F16");
        for (const Json& d : value.at("shape")) {
            error.require(d.is_number_unsigned() || (d.is_number_integer() && d.get<std::int64_t>() >= 0),
                          "tensor '" + name + "' has an invalid shape");
            const auto dim = d.get<std::uint64_t>();
            error.require(dim <= (1ULL << 31U), "tensor '" + name + "' is too large");
            entry.shape.push_back(dim);
            bytes *= dim;
        }
        const Json& offsets = value.at("data_offsets");
        error.require(offsets.size() == 2 && offsets[0].is_number_unsigned() &&
                          offsets[1].is_number_unsigned(),
                      "tensor '" + name + "' has invalid data_offsets");
        entry.begin = offsets[0].get<std::uint64_t>();
        entry.end   = offsets[1].get<std::uint64_t>();
        error.require(entry.begin <= entry.end && entry.end <= payload && entry.end - entry.begin == bytes,
                      "tensor '" + name + "' data_offsets do not match its dtype and shape");
        tensors.emplace(name, std::move(entry));
    }
    return tensors;
}

std::vector<float> read_values(std::ifstream& in, std::uint64_t payload_offset, const TensorEntry& entry,
                               const std::string& name, Fnv& digest, const AdapterError& error) {
    std::vector<std::uint8_t> raw(static_cast<std::size_t>(entry.end - entry.begin));
    in.seekg(static_cast<std::streamoff>(payload_offset + entry.begin));
    in.read(reinterpret_cast<char*>(raw.data()), static_cast<std::streamsize>(raw.size()));
    error.require(static_cast<bool>(in), "cannot read tensor '" + name + "'");
    const std::size_t count = raw.size() / element_bytes(entry.dtype);
    std::vector<float> values(count);
    if (entry.dtype == "F32") {
        std::memcpy(values.data(), raw.data(), raw.size());
    } else {
        for (std::size_t i = 0; i < count; ++i) {
            std::uint16_t h;
            std::memcpy(&h, raw.data() + 2 * i, 2);
            values[i] = entry.dtype == "BF16" ? std::bit_cast<float>(static_cast<std::uint32_t>(h) << 16U)
                                              : half_to_float(h);
        }
    }
    for (const float v : values) {
        error.require(std::isfinite(v), "tensor '" + name + "' holds a non-finite value");
    }
    digest.text(name);
    digest.text(entry.dtype);
    for (const auto d : entry.shape) { digest.pod(d); }
    digest.bytes(raw.data(), raw.size());
    return values;
}

struct ParsedName {
    std::uint32_t layer = 0;
    LoraModule module   = LoraModule::Query;
    bool is_a           = false;
};

// "base_model.model." + ("model.language_model." | "model.") + "layers.N." +
// ("self_attn." | "mlp.") + module + (".lora_A.weight" | ".lora_B.weight").
ParsedName parse_name(const std::string& name, const AdapterError& error) {
    std::string_view rest(name);
    const auto take = [&](std::string_view prefix) {
        if (rest.substr(0, prefix.size()) != prefix) { return false; }
        rest.remove_prefix(prefix.size());
        return true;
    };
    const auto refuse = [&](const std::string& why) {
        error.fail("tensor '" + name + "' " + why);
    };
    if (!take("base_model.model.")) { refuse("is not a PEFT LoRA tensor (no base_model.model. prefix)"); }
    if (!take("model.language_model.") && !take("model.")) {
        refuse("is outside the text model (Vision, MTP, embedding and head adapters are not supported)");
    }
    if (!take("layers.")) {
        refuse("is not in a text layer (Vision, MTP, embedding and head adapters are not supported)");
    }
    std::size_t digits = 0;
    while (digits < rest.size() && rest[digits] >= '0' && rest[digits] <= '9') { ++digits; }
    if (digits == 0 || digits > 6 || digits >= rest.size() || rest[digits] != '.') {
        refuse("has no layer index");
    }
    ParsedName out;
    out.layer = static_cast<std::uint32_t>(std::stoul(std::string(rest.substr(0, digits))));
    rest.remove_prefix(digits + 1);
    bool attention = false;
    if (take("self_attn.")) {
        attention = true;
    } else if (take("linear_attn.")) {
        refuse("adapts a Gated DeltaNet projection, which has no LoRA route (only the full-attention "
               "q/k/v/o_proj and the MLP gate/up/down_proj are supported)");
    } else if (!take("mlp.")) {
        refuse("adapts no supported module");
    }
    static constexpr std::array<std::pair<std::string_view, LoraModule>, kLoraModuleCount> modules{{
        {"q_proj", LoraModule::Query},
        {"k_proj", LoraModule::Key},
        {"v_proj", LoraModule::Value},
        {"o_proj", LoraModule::Output},
        {"gate_proj", LoraModule::Gate},
        {"up_proj", LoraModule::Up},
        {"down_proj", LoraModule::Down},
    }};
    bool found = false;
    for (const auto& [text, module] : modules) {
        const bool attention_module = module == LoraModule::Query || module == LoraModule::Key ||
                                      module == LoraModule::Value || module == LoraModule::Output;
        if (attention_module != attention) { continue; }
        if (take(text)) {
            out.module = module;
            found      = true;
            break;
        }
    }
    if (!found) {
        refuse(attention ? "adapts an attention module other than q/k/v/o_proj"
                         : "adapts an MLP module other than gate/up/down_proj (MoE experts are not supported)");
    }
    if (rest == ".lora_A.weight") {
        out.is_a = true;
    } else if (rest != ".lora_B.weight") {
        refuse("is not a lora_A or lora_B weight (DoRA magnitudes and LoRA biases are not supported)");
    }
    return out;
}

// The [out, in] geometry of `module` in this model.
std::pair<std::uint64_t, std::uint64_t> module_shape(LoraModule module, const TextConfig& text) {
    const std::uint64_t hidden = text.hidden_size;
    const auto& attention      = *text.attention;
    const std::uint64_t ffn    = std::get<DenseConfig>(text.ffn).intermediate_size;
    switch (module) {
    case LoraModule::Query: return {2 * attention.query_width(), hidden};
    case LoraModule::Key:
    case LoraModule::Value: return {attention.key_width(), hidden};
    case LoraModule::Output: return {hidden, attention.query_width()};
    case LoraModule::Gate:
    case LoraModule::Up: return {ffn, hidden};
    case LoraModule::Down: return {hidden, ffn};
    }
    return {0, 0};
}

} // namespace

const char* lora_module_name(LoraModule module) noexcept {
    switch (module) {
    case LoraModule::Query: return "q_proj";
    case LoraModule::Key: return "k_proj";
    case LoraModule::Value: return "v_proj";
    case LoraModule::Output: return "o_proj";
    case LoraModule::Gate: return "gate_proj";
    case LoraModule::Up: return "up_proj";
    case LoraModule::Down: return "down_proj";
    }
    return "?";
}

LoraAdapter load_lora_adapter(const LoraSource& source, const TextConfig& text) {
    const AdapterError error(source.name);
    error.require(!source.name.empty(), "an adapter needs a name");
    error.require(std::holds_alternative<DenseConfig>(text.ffn),
                  "LoRA adapters need a dense model; MoE experts have no LoRA route");
    error.require(text.attention.has_value(), "the model has no attention configuration");
    const std::filesystem::path directory = adapter_directory(source, error);
    const Settings settings               = read_settings(read_json(directory / "adapter_config.json", error), error);

    const std::filesystem::path weights = directory / "adapter_model.safetensors";
    std::error_code ec;
    const auto file_bytes = std::filesystem::file_size(weights, ec);
    error.require(!ec, "cannot read " + weights.string() +
                           (std::filesystem::exists(directory / "adapter_model.bin")
                                ? " (a pickled adapter_model.bin is not supported; save it as safetensors)"
                                : ""));
    std::ifstream in(weights, std::ios::binary);
    error.require(static_cast<bool>(in), "cannot read " + weights.string());
    std::uint64_t payload_offset = 0;
    const auto tensors           = read_header(in, file_bytes, payload_offset, error);
    error.require(!tensors.empty(), "adapter_model.safetensors holds no tensors");

    LoraAdapter adapter;
    adapter.name  = source.name;
    adapter.rank  = settings.rank;
    adapter.alpha = settings.alpha;
    adapter.scale = settings.scale;
    adapter.layers.resize(text.num_hidden_layers);
    Fnv digest;
    digest.pod(kLoraDigestVersion);
    digest.pod(settings.scale);

    // Both halves of each module first, so a module with only one is refused by name.
    struct Halves {
        const TensorEntry* a = nullptr;
        const TensorEntry* b = nullptr;
        std::string a_name, b_name;
    };
    std::map<std::pair<std::uint32_t, LoraModule>, Halves> found;
    for (const auto& [name, entry] : tensors) {
        const ParsedName parsed = parse_name(name, error);
        error.require(parsed.layer < text.num_hidden_layers,
                      "tensor '" + name + "' names layer " + std::to_string(parsed.layer) + "; the model has " +
                          std::to_string(text.num_hidden_layers));
        const bool attention_module = parsed.module == LoraModule::Query || parsed.module == LoraModule::Key ||
                                      parsed.module == LoraModule::Value || parsed.module == LoraModule::Output;
        error.require(!attention_module || text.layer_types[parsed.layer] == MixerKind::FullAttention,
                      "tensor '" + name + "' adapts self_attn in layer " + std::to_string(parsed.layer) +
                          ", which is a Gated DeltaNet layer in this model");
        Halves& halves = found[{parsed.layer, parsed.module}];
        (parsed.is_a ? halves.a : halves.b)           = &entry;
        (parsed.is_a ? halves.a_name : halves.b_name) = name;
    }
    for (const auto& [key, halves] : found) {
        const auto [layer, module] = key;
        const std::string where    = "layer " + std::to_string(layer) + " " + lora_module_name(module);
        error.require(halves.a != nullptr && halves.b != nullptr,
                      where + " has only " + (halves.a != nullptr ? "lora_A" : "lora_B"));
        const auto [out, in_features] = module_shape(module, text);
        const std::vector<std::uint64_t> a_shape{static_cast<std::uint64_t>(settings.rank), in_features};
        const std::vector<std::uint64_t> b_shape{out, static_cast<std::uint64_t>(settings.rank)};
        error.require(halves.a->shape == a_shape && halves.b->shape == b_shape,
                      where + " has lora_A/lora_B shapes that do not match this model at r=" +
                          std::to_string(settings.rank) + " (an adapter for another model?)");
        LoraPair pair;
        pair.rank = settings.rank;
        pair.in   = static_cast<std::int32_t>(in_features);
        pair.out  = static_cast<std::int32_t>(out);
        pair.a    = read_values(in, payload_offset, *halves.a, halves.a_name, digest, error);
        pair.b    = read_values(in, payload_offset, *halves.b, halves.b_name, digest, error);
        for (float& v : pair.b) { v *= settings.scale; }
        adapter.parameter_count += pair.a.size() + pair.b.size();
        ++adapter.adapted_modules;
        adapter.layers[layer][module] = std::move(pair);
    }
    adapter.digest = digest.value == 0 ? 1 : digest.value;
    return adapter;
}

QueryGateRows lora_split_query_gate(const LoraPair& q_proj, std::uint32_t heads, std::uint32_t head_dim) {
    const auto rank = static_cast<std::size_t>(q_proj.rank);
    const std::size_t d = head_dim;
    if (static_cast<std::size_t>(q_proj.out) != 2 * heads * d || q_proj.b.size() != q_proj.out * rank) {
        throw std::invalid_argument("lora_split_query_gate: q_proj B does not have 2 * heads * head_dim rows");
    }
    QueryGateRows out;
    out.query.resize(heads * d * rank);
    out.gate.resize(heads * d * rank);
    for (std::size_t h = 0; h < heads; ++h) {
        std::copy_n(q_proj.b.data() + (2 * h * d) * rank, d * rank, out.query.data() + h * d * rank);
        std::copy_n(q_proj.b.data() + (2 * h * d + d) * rank, d * rank, out.gate.data() + h * d * rank);
    }
    return out;
}

std::vector<float> lora_rotate_rows(std::span<const float> a, std::int32_t rows, std::int32_t in,
                                    std::span<const float> signs) {
    constexpr std::size_t kBlock = 1024;
    if (rows < 0 || in <= 0 || static_cast<std::size_t>(in) % kBlock != 0 ||
        a.size() != static_cast<std::size_t>(rows) * in || signs.size() != static_cast<std::size_t>(in)) {
        throw std::invalid_argument("lora_rotate_rows: A must be [rows, in] with in a multiple of 1024 signs");
    }
    std::vector<float> out(a.size());
    std::vector<double> block(kBlock);
    for (std::size_t r = 0; r < static_cast<std::size_t>(rows); ++r) {
        for (std::size_t b0 = 0; b0 < static_cast<std::size_t>(in); b0 += kBlock) {
            const float* src = a.data() + r * in + b0;
            for (std::size_t j = 0; j < kBlock; ++j) { block[j] = static_cast<double>(src[j]) * signs[b0 + j]; }
            // In-place fast Walsh-Hadamard transform in Sylvester order: H[i][j] = (-1)^popcount(i & j).
            for (std::size_t h = 1; h < kBlock; h <<= 1U) {
                for (std::size_t i = 0; i < kBlock; i += 2 * h) {
                    for (std::size_t j = i; j < i + h; ++j) {
                        const double x = block[j], y = block[j + h];
                        block[j]       = x + y;
                        block[j + h]   = x - y;
                    }
                }
            }
            float* dst = out.data() + r * in + b0;
            for (std::size_t i = 0; i < kBlock; ++i) { dst[i] = static_cast<float>(block[i] / 32.0); }
        }
    }
    return out;
}

std::vector<double> lora_reference_delta(const LoraPair& pair, std::span<const double> x) {
    if (x.size() != static_cast<std::size_t>(pair.in)) {
        throw std::invalid_argument("lora_reference_delta: x does not match the module's input");
    }
    std::vector<double> u(static_cast<std::size_t>(pair.rank), 0.0);
    for (std::size_t j = 0; j < u.size(); ++j) {
        for (std::size_t k = 0; k < x.size(); ++k) { u[j] += static_cast<double>(pair.a[j * x.size() + k]) * x[k]; }
    }
    std::vector<double> y(static_cast<std::size_t>(pair.out), 0.0);
    for (std::size_t n = 0; n < y.size(); ++n) {
        for (std::size_t j = 0; j < u.size(); ++j) { y[n] += static_cast<double>(pair.b[n * u.size() + j]) * u[j]; }
    }
    return y;
}

} // namespace ninfer::models::qwen3_5
