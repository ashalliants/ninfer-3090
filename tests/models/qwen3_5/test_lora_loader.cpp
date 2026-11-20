// PEFT LoRA adapter loading (models/qwen3_5/load/lora.h) against a small hybrid geometry: name
// mapping, scale folding, the dtypes PEFT writes, every refusal, the q_proj query/gate split, the
// Hadamard basis change of A, and the FP64 reference the device tests compare against. CPU only.
#include "models/qwen3_5/config.h"
#include "models/qwen3_5/load/lora.h"

#include <nlohmann/json.hpp>

#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <random>
#include <string>
#include <vector>

#include <unistd.h>

namespace {

namespace q  = ninfer::models::qwen3_5;
using Json   = nlohmann::json;
namespace fs = std::filesystem;

int check(bool condition, const std::string& message) {
    if (condition) { return 0; }
    std::cerr << "FAIL: " << message << '\n';
    return 1;
}

// Layer 0 Gated DeltaNet, layer 1 full attention. q_proj is [2 * 2 * 4, 16] (query + gate rows).
q::TextConfig model() {
    q::TextConfig text;
    text.hidden_size       = 16;
    text.vocab_size        = 100;
    text.num_hidden_layers = 2;
    text.layer_types       = {q::MixerKind::LinearAttention, q::MixerKind::FullAttention};
    text.attention = q::AttentionConfig{.num_attention_heads = 2, .num_key_value_heads = 1,
                                        .head_dim = 4};
    text.ffn = q::DenseConfig{.intermediate_size = 24};
    return text;
}

struct Tensor {
    std::string dtype = "F32";
    std::vector<std::uint64_t> shape;
    std::vector<float> values;
};

std::uint16_t to_bf16(float v) { return static_cast<std::uint16_t>(std::bit_cast<std::uint32_t>(v) >> 16U); }

// IEEE half for the small exactly representable values the tests use.
std::uint16_t to_f16(float v) {
    if (v == 0.0f) { return 0; }
    const std::uint32_t bits = std::bit_cast<std::uint32_t>(v);
    const std::uint32_t sign = (bits >> 16U) & 0x8000U;
    const int exponent       = static_cast<int>((bits >> 23U) & 0xffU) - 127 + 15;
    const std::uint32_t mant = (bits >> 13U) & 0x3ffU;
    return static_cast<std::uint16_t>(sign | (static_cast<std::uint32_t>(exponent) << 10U) | mant);
}

// Values exact in BF16 and F16: multiples of 1/64 in [-2, 2).
std::vector<float> values(std::size_t count, std::uint32_t seed) {
    std::mt19937 rng(seed);
    std::vector<float> out(count);
    for (float& v : out) { v = static_cast<float>(static_cast<int>(rng() % 256U) - 128) / 64.0f; }
    return out;
}

struct Adapter {
    Json config = {{"peft_type", "LORA"}, {"task_type", "CAUSAL_LM"}, {"r", 4},
                   {"lora_alpha", 8},     {"use_rslora", false},     {"use_dora", false},
                   {"bias", "none"},      {"rank_pattern", Json::object()},
                   {"alpha_pattern", Json::object()},
                   {"target_modules", {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj",
                                       "up_proj", "down_proj"}}};
    std::map<std::string, Tensor> tensors;
};

std::string name(std::uint32_t layer, const std::string& block, const std::string& module,
                 const char* half, const std::string& prefix = "base_model.model.model.language_model.") {
    return prefix + "layers." + std::to_string(layer) + "." + block + "." + module + ".lora_" + half +
           ".weight";
}

void add_pair(Adapter& a, std::uint32_t layer, const std::string& block, const std::string& module,
              std::uint64_t out, std::uint64_t in, std::uint32_t seed, const std::string& dtype = "F32",
              const std::string& prefix = "base_model.model.model.language_model.") {
    a.tensors[name(layer, block, module, "A", prefix)] = {dtype, {4, in}, values(4 * in, seed)};
    a.tensors[name(layer, block, module, "B", prefix)] = {dtype, {out, 4}, values(out * 4, seed + 1)};
}

// Every supported target: attention on layer 1, the MLP on both layers, in all three dtypes.
Adapter full_adapter() {
    Adapter a;
    add_pair(a, 1, "self_attn", "q_proj", 16, 16, 10);
    add_pair(a, 1, "self_attn", "k_proj", 4, 16, 20, "BF16");
    add_pair(a, 1, "self_attn", "v_proj", 4, 16, 30, "F16");
    add_pair(a, 1, "self_attn", "o_proj", 16, 8, 40);
    for (std::uint32_t layer = 0; layer < 2; ++layer) {
        add_pair(a, layer, "mlp", "gate_proj", 24, 16, 50 + layer);
        add_pair(a, layer, "mlp", "up_proj", 24, 16, 60 + layer, "BF16");
        add_pair(a, layer, "mlp", "down_proj", 16, 24, 70 + layer);
    }
    return a;
}

void write(const fs::path& dir, const Adapter& a) {
    fs::create_directories(dir);
    std::ofstream(dir / "adapter_config.json") << a.config.dump();
    Json header = Json::object();
    std::vector<std::uint8_t> payload;
    for (const auto& [key, t] : a.tensors) {
        const std::size_t begin = payload.size();
        for (const float v : t.values) {
            if (t.dtype == "F32") {
                std::uint8_t b[4];
                std::memcpy(b, &v, 4);
                payload.insert(payload.end(), b, b + 4);
            } else {
                const std::uint16_t h = t.dtype == "BF16" ? to_bf16(v) : to_f16(v);
                payload.push_back(static_cast<std::uint8_t>(h & 0xffU));
                payload.push_back(static_cast<std::uint8_t>(h >> 8U));
            }
        }
        header[key] = {{"dtype", t.dtype}, {"shape", t.shape}, {"data_offsets", {begin, payload.size()}}};
    }
    header["__metadata__"] = {{"format", "pt"}};
    std::string text       = header.dump();
    while (text.size() % 8 != 0) { text.push_back(' '); }
    std::ofstream out(dir / "adapter_model.safetensors", std::ios::binary);
    const std::uint64_t size = text.size();
    for (int i = 0; i < 8; ++i) { out.put(static_cast<char>((size >> (8 * i)) & 0xffU)); }
    out.write(text.data(), static_cast<std::streamsize>(text.size()));
    out.write(reinterpret_cast<const char*>(payload.data()), static_cast<std::streamsize>(payload.size()));
}

std::string refusal(const std::function<void()>& body) {
    try {
        body();
    } catch (const std::invalid_argument& error) { return error.what(); }
    return {};
}

bool contains(const std::string& text, const std::string& part) { return text.find(part) != std::string::npos; }

// 2^-5 H (s * x) per 1024-block, written out from the definition (no fast transform).
std::vector<double> rotate_input(const std::vector<double>& x, const std::vector<float>& signs) {
    std::vector<double> out(x.size(), 0.0);
    for (std::size_t b0 = 0; b0 < x.size(); b0 += 1024) {
        for (std::size_t i = 0; i < 1024; ++i) {
            double sum = 0;
            for (std::size_t j = 0; j < 1024; ++j) {
                const double h = (std::popcount(i & j) & 1U) != 0 ? -1.0 : 1.0;
                sum += h * signs[b0 + j] * x[b0 + j];
            }
            out[b0 + i] = sum / 32.0;
        }
    }
    return out;
}

} // namespace

int main() {
    int failures         = 0;
    const fs::path root  = fs::temp_directory_path() / ("ninfer_lora_loader_" + std::to_string(::getpid()));
    fs::remove_all(root);
    const q::TextConfig text = model();
    int case_index           = 0;
    const auto load = [&](const Adapter& a) {
        const fs::path dir = root / std::to_string(case_index++);
        write(dir, a);
        return q::load_lora_adapter({.name = "pi", .path = dir}, text);
    };

    // A full adapter: every module mapped, the scale folded into B, all three dtypes widened.
    {
        const Adapter source  = full_adapter();
        const q::LoraAdapter adapter = load(source);
        failures += check(adapter.rank == 4 && adapter.scale == 2.0f, "r=4 alpha=8 gives scale 2");
        failures += check(adapter.adapted_modules == 10, "ten modules adapted");
        failures += check(adapter.parameter_count == 4 * (16 + 16 + 16 + 4 + 16 + 4 + 8 + 16) +
                                                          2 * 4 * (16 + 24 + 16 + 24 + 24 + 16),
                          "parameter count");
        failures += check(!adapter.layers[0][q::LoraModule::Query] && adapter.layers[0][q::LoraModule::Gate],
                          "the GDN layer has only its MLP adapted");
        const auto& query = *adapter.layers[1][q::LoraModule::Query];
        const auto& b_src = source.tensors.at(name(1, "self_attn", "q_proj", "B")).values;
        bool scaled       = query.b.size() == b_src.size();
        for (std::size_t i = 0; scaled && i < b_src.size(); ++i) { scaled = query.b[i] == 2.0f * b_src[i]; }
        failures += check(scaled, "q_proj B carries alpha / r");
        const auto& value = *adapter.layers[1][q::LoraModule::Value];
        const auto& a_src = source.tensors.at(name(1, "self_attn", "v_proj", "A")).values;
        failures += check(value.a == a_src, "F16 v_proj A widens exactly");
        const auto& key   = *adapter.layers[1][q::LoraModule::Key];
        const auto& k_src = source.tensors.at(name(1, "self_attn", "k_proj", "A")).values;
        failures += check(key.a == k_src, "BF16 k_proj A widens exactly");
        failures += check(adapter.digest != 0, "digest is nonzero");

        // The digest follows content: the same files give the same digest, one changed value
        // another.
        failures += check(load(source).digest == adapter.digest, "digest is deterministic");
        Adapter changed = source;
        changed.tensors.at(name(0, "mlp", "down_proj", "B")).values[3] += 1.0f / 64.0f;
        failures += check(load(changed).digest != adapter.digest, "digest sees a changed value");
        Adapter rescaled              = source;
        rescaled.config["lora_alpha"] = 16;
        failures += check(load(rescaled).digest != adapter.digest, "digest sees the scale");

        // q_proj splits per head into query rows then gate rows.
        const q::QueryGateRows rows = q::lora_split_query_gate(query, 2, 4);
        bool split = rows.query.size() == 8 * 4 && rows.gate.size() == 8 * 4;
        for (std::size_t h = 0; split && h < 2; ++h) {
            for (std::size_t d = 0; split && d < 4; ++d) {
                for (std::size_t j = 0; split && j < 4; ++j) {
                    split = rows.query[(h * 4 + d) * 4 + j] == query.b[((2 * h) * 4 + d) * 4 + j] &&
                            rows.gate[(h * 4 + d) * 4 + j] == query.b[((2 * h + 1) * 4 + d) * 4 + j];
                }
            }
        }
        failures += check(split, "lora_split_query_gate takes [query D | gate D] per head");

        // The reference: s * B (A x) by hand for one column.
        std::vector<double> x(16);
        for (std::size_t i = 0; i < x.size(); ++i) { x[i] = 0.25 * static_cast<double>(i) - 1.5; }
        const std::vector<double> delta = q::lora_reference_delta(query, x);
        double worst                    = 0;
        for (std::size_t n = 0; n < 16; ++n) {
            double expected = 0;
            for (std::size_t j = 0; j < 4; ++j) {
                double u = 0;
                for (std::size_t k = 0; k < 16; ++k) {
                    u += static_cast<double>(source.tensors.at(name(1, "self_attn", "q_proj", "A")).values[j * 16 + k]) * x[k];
                }
                expected += 2.0 * static_cast<double>(b_src[n * 4 + j]) * u;
            }
            worst = std::max(worst, std::abs(expected - delta[n]));
        }
        failures += check(worst < 1e-9, "lora_reference_delta is s * B (A x)");
    }

    // Partial adapters and the other PEFT prefix.
    {
        Adapter mlp_only;
        add_pair(mlp_only, 0, "mlp", "down_proj", 16, 24, 5, "F32", "base_model.model.model.");
        const q::LoraAdapter adapter = load(mlp_only);
        failures += check(adapter.adapted_modules == 1 && adapter.layers[0][q::LoraModule::Down],
                          "base_model.model.model.layers... maps like the language_model prefix");
        Adapter rslora                = mlp_only;
        rslora.config["use_rslora"]   = true;
        failures += check(load(rslora).scale == 4.0f, "use_rslora scales by alpha / sqrt(r)");
    }

    // Refusals, each named.
    const auto refused = [&](const Adapter& a, const std::string& expect, const std::string& what) {
        const std::string message = refusal([&] { (void)load(a); });
        failures += check(contains(message, expect) && contains(message, "'pi'"),
                          what + " is refused naming '" + expect + "' (got: " + message + ")");
    };
    {
        Adapter a = full_adapter();
        a.config["use_dora"] = true;
        refused(a, "use_dora", "DoRA");
        a = full_adapter();
        a.config["rank_pattern"] = {{"q_proj", 8}};
        refused(a, "rank_pattern", "a rank pattern");
        a = full_adapter();
        a.config["bias"] = "all";
        refused(a, "bias", "a trained bias");
        a = full_adapter();
        a.config["modules_to_save"] = {"lm_head"};
        refused(a, "modules_to_save", "modules_to_save");
        a = full_adapter();
        a.config["peft_type"] = "IA3";
        refused(a, "peft_type", "another PEFT method");
        a = full_adapter();
        a.config.erase("r");
        refused(a, "integer r", "a missing r");
        a = full_adapter();
        a.config["r"] = 8;
        refused(a, "shapes", "an r that disagrees with the tensors");
        a = full_adapter();
        add_pair(a, 0, "self_attn", "q_proj", 16, 16, 90);
        refused(a, "Gated DeltaNet layer", "self_attn on a GDN layer");
        a = full_adapter();
        add_pair(a, 0, "linear_attn", "in_proj_qkv", 16, 16, 91);
        refused(a, "Gated DeltaNet projection", "a linear_attn target");
        a = full_adapter();
        a.tensors["base_model.model.lm_head.lora_A.weight"] = {"F32", {4, 16}, values(64, 1)};
        refused(a, "outside the text model", "an lm_head target");
        a = full_adapter();
        a.tensors["base_model.model.model.language_model.embed_tokens.lora_embedding_A"] = {"F32", {4, 100}, values(400, 1)};
        refused(a, "not in a text layer", "an embedding target");
        a = full_adapter();
        a.tensors["base_model.model.model.visual.blocks.0.attn.qkv.lora_A.weight"] = {"F32", {4, 16}, values(64, 1)};
        refused(a, "not in a text layer", "a Vision target");
        a = full_adapter();
        add_pair(a, 2, "mlp", "up_proj", 24, 16, 92);
        refused(a, "names layer 2", "a layer past the model");
        a = full_adapter();
        a.tensors.erase(name(1, "mlp", "up_proj", "B"));
        refused(a, "has only lora_A", "a module missing lora_B");
        a = full_adapter();
        a.tensors.at(name(1, "mlp", "gate_proj", "A")).shape = {4, 8};
        a.tensors.at(name(1, "mlp", "gate_proj", "A")).values.resize(32);
        refused(a, "do not match this model", "another model's shapes");
        a = full_adapter();
        a.tensors.at(name(0, "mlp", "up_proj", "A")).values[2] = std::nanf("");
        refused(a, "non-finite", "a NaN");
        a = full_adapter();
        a.tensors["base_model.model.model.language_model.layers.1.mlp.gate_proj.lora_magnitude_vector"] = {"F32", {24}, values(24, 1)};
        refused(a, "not a lora_A or lora_B weight", "a DoRA magnitude tensor");
        a = full_adapter();
        a.tensors["base_model.model.model.language_model.layers.1.mlp.experts.0.gate_proj.lora_A.weight"] = {"F32", {4, 16}, values(64, 1)};
        refused(a, "MoE", "an expert target");

        const std::string missing = refusal(
            [&] { (void)q::load_lora_adapter({.name = "pi", .path = root / "nowhere"}, text); });
        failures += check(contains(missing, "not a PEFT adapter directory"), "a missing path is refused");

        q::TextConfig moe = text;
        moe.ffn           = q::MoeConfig{.num_experts = 4, .num_experts_per_tok = 2,
                                         .moe_intermediate_size = 8, .shared_expert_intermediate_size = 8};
        const fs::path dir = root / "moe";
        write(dir, full_adapter());
        const std::string message = refusal([&] { (void)q::load_lora_adapter({.name = "pi", .path = dir}, moe); });
        failures += check(contains(message, "dense model"), "a MoE model is refused");
    }

    // The rotated basis: A' x_r == A x where x_r = 2^-5 H (s * x) per 1024-block.
    {
        constexpr std::int32_t in = 2048, rows = 3;
        std::mt19937 rng(7);
        std::uniform_real_distribution<float> uniform(-1.0f, 1.0f);
        std::vector<float> a(static_cast<std::size_t>(rows) * in), signs(in);
        for (float& v : a) { v = uniform(rng); }
        for (float& s : signs) { s = (rng() & 1U) != 0 ? 1.0f : -1.0f; }
        std::vector<double> x(in);
        for (double& v : x) { v = uniform(rng); }
        const std::vector<float> rotated = q::lora_rotate_rows(a, rows, in, signs);
        const std::vector<double> xr     = rotate_input(x, signs);
        double worst = 0, scale = 0;
        for (std::int32_t r = 0; r < rows; ++r) {
            double plain = 0, basis = 0;
            for (std::int32_t k = 0; k < in; ++k) {
                plain += static_cast<double>(a[static_cast<std::size_t>(r) * in + k]) * x[static_cast<std::size_t>(k)];
                basis += static_cast<double>(rotated[static_cast<std::size_t>(r) * in + k]) * xr[static_cast<std::size_t>(k)];
            }
            worst = std::max(worst, std::abs(plain - basis));
            scale = std::max(scale, std::abs(plain));
        }
        failures += check(worst <= 1e-5 * std::max(1.0, scale), "lora_rotate_rows keeps A x in the rotated basis");
        const std::string bad = refusal([&] { (void)q::lora_rotate_rows(a, rows, 1000, signs); });
        failures += check(!bad.empty(), "lora_rotate_rows refuses a width that is not 1024-blocked");
    }

    fs::remove_all(root);
    if (failures == 0) { std::cout << "lora loader: all checks passed\n"; }
    return failures == 0 ? 0 : 1;
}
