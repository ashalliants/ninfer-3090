// Written for NInfer-3090. See dflash2.h.

#include "models/inkling2/dflash2.h"

#include "ninfer/ops/dspark_fp32.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string_view>

namespace ninfer::models::inkling2 {
namespace {

using Json = nlohmann::json;

// The release is about 1 GB; a file far past that is not a DFlash2 drafter for this model.
constexpr std::uint64_t kMaxFileBytes = 4ULL << 30;

[[noreturn]] void fail(const std::string& message) { throw std::invalid_argument("--draft-model: " + message); }
void require(bool condition, const std::string& message) {
    if (!condition) { fail(message); }
}

std::uint32_t dimension(const Json& v, const char* key) {
    require(v.contains(key) && v.at(key).is_number_unsigned() && v.at(key).get<std::uint64_t>() > 0 && v.at(key).get<std::uint64_t>() < (1ULL << 31),
            std::string("config.json: ") + key + " must be a positive integer");
    return v.at(key).get<std::uint32_t>();
}
float real(const Json& v, const char* key) {
    require(v.contains(key) && v.at(key).is_number(), std::string("config.json: ") + key + " must be a number");
    return v.at(key).get<float>();
}

std::vector<std::uint8_t> read_file(const std::filesystem::path& path) {
    std::error_code error;
    const std::uint64_t size = std::filesystem::file_size(path, error);
    require(!error, "cannot read " + path.string() + (error ? ": " + error.message() : std::string()));
    require(size <= kMaxFileBytes, path.filename().string() + " is " + std::to_string(size) + " bytes, far more than a DFlash2 drafter");
    std::vector<std::uint8_t> bytes(static_cast<std::size_t>(size));
    std::ifstream in(path, std::ios::binary);
    require(in.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(bytes.size())).good() || bytes.empty(),
            "cannot read " + path.string());
    return bytes;
}

} // namespace

DFlash2Config parse_dflash2_config(const std::string& config_json, const TextConfig& text) {
    Json v;
    try {
        v = Json::parse(config_json);
    } catch (const Json::exception& error) {
        fail(std::string("config.json is not JSON: ") + error.what());
    }
    require(v.is_object() && v.contains("architectures") && v.at("architectures") == Json::array({"DFlash2DraftModel"}),
            "config.json is not a DFlash2DraftModel's");
    require(v.contains("dflash_config") && v.at("dflash_config").is_object(), "config.json has no dflash_config");
    const Json& d = v.at("dflash_config");
    DFlash2Config c;
    c.layers            = dimension(v, "num_hidden_layers");
    c.hidden_size       = dimension(v, "hidden_size");
    c.intermediate_size = dimension(v, "intermediate_size");
    c.heads             = dimension(v, "num_attention_heads");
    c.kv_heads          = dimension(v, "num_key_value_heads");
    c.head_dim          = dimension(v, "head_dim");
    c.rms_norm_eps      = real(v, "rms_norm_eps");
    require(v.contains("rope_parameters") && v.at("rope_parameters").is_object(), "config.json has no rope_parameters");
    require(v.at("rope_parameters").value("rope_type", std::string("default")) == "default", "only default RoPE");
    c.rope_theta          = real(v.at("rope_parameters"), "rope_theta");
    c.sliding_window      = dimension(v, "sliding_window");
    c.block_size          = dimension(d, "block_size");
    c.conv_group          = dimension(d, "conv_group_size");
    c.final_logit_softcap = real(d, "final_logit_softcapping");
    require(d.contains("mask_token_id") && d.at("mask_token_id").is_number_unsigned(), "config.json: mask_token_id");
    c.mask_token_id = d.at("mask_token_id").get<std::uint32_t>();
    require(d.contains("target_layer_ids") && d.at("target_layer_ids").is_array(), "config.json: target_layer_ids");
    for (const Json& id : d.at("target_layer_ids")) {
        require(id.is_number_unsigned(), "config.json: target_layer_ids are indices");
        c.target_visits.push_back(id.get<std::uint32_t>());
    }
    require(v.contains("layer_types") && v.at("layer_types").is_array() && v.at("layer_types").size() == c.layers, "config.json: layer_types");
    for (const Json& t : v.at("layer_types")) { require(t == "sliding_attention", "every layer must be sliding attention"); }
    require(d.value("selector_softcap_mode", std::string("after_correction")) == "after_correction", "selector_softcap_mode must be after_correction");
    // The kernels' geometry (include/ninfer/ops/dspark_fp32.h).
    require(c.head_dim == 128 && c.heads % c.kv_heads == 0, "the drafter's attention takes heads of 128 in whole groups");
    require(dimension(d, "conv_kernel_size") == 2 && c.hidden_size % c.conv_group == 0, "a two-tap convolution in whole groups");
    require(dimension(d, "selector_top_k") == ops::kDFlash2TopK && dimension(d, "selector_rank") == ops::kDFlash2Rank,
            "a top-16 selector of rank 256");
    require(c.block_size == 8, "a block of 8");
    // The target it drafts for.
    require(c.hidden_size == text.hidden_size, "hidden_size differs from the text model's");
    require(dimension(v, "vocab_size") == text.vocab_size, "vocab_size differs from the text model's");
    require(c.mask_token_id < text.unpadded_vocab_size, "mask_token_id is not a token");
    require(c.final_logit_softcap > 0, "final_logit_softcapping must be positive");
    // As DSpark's (config.cpp): the last tap is the final normed state, which the drafter's fusion expects.
    require(!c.target_visits.empty() && c.target_visits.size() <= 8 && c.target_visits.back() == text.num_layer_visits,
            "one to eight taps, the last the final state (target_layer_ids ending in " + std::to_string(text.num_layer_visits) + ")");
    for (std::size_t i = 1; i < c.target_visits.size(); ++i) { require(c.target_visits[i] > c.target_visits[i - 1], "the taps must increase"); }
    return c;
}

std::unique_ptr<DFlash2Release> DFlash2Release::load(const std::filesystem::path& directory, const TextConfig& text) {
    require(std::filesystem::is_directory(directory),
            "'" + directory.string() + "' is not a directory (the DFlash2 release: config.json and model.safetensors)");
    auto out = std::unique_ptr<DFlash2Release>(new DFlash2Release());
    {
        const std::vector<std::uint8_t> config = read_file(directory / "config.json");
        out->config_ = parse_dflash2_config(std::string(config.begin(), config.end()), text);
    }
    out->file_                  = read_file(directory / "model.safetensors");
    const std::vector<std::uint8_t>& file = out->file_;
    require(file.size() >= 8, "model.safetensors is shorter than its header");
    std::uint64_t header_bytes = 0;
    for (int byte = 7; byte >= 0; --byte) { header_bytes = (header_bytes << 8U) | file[static_cast<std::size_t>(byte)]; }
    require(header_bytes >= 2 && header_bytes <= file.size() - 8 && header_bytes <= (16ULL << 20), "model.safetensors: its header runs past the file");
    Json header;
    try {
        header = Json::parse(std::string_view(reinterpret_cast<const char*>(file.data() + 8), static_cast<std::size_t>(header_bytes)));
    } catch (const Json::exception& error) {
        fail(std::string("model.safetensors: the header is not JSON: ") + error.what());
    }
    require(header.is_object(), "model.safetensors: the header is not an object");
    const std::uint64_t payload_offset = 8 + header_bytes, payload = file.size() - payload_offset;
    for (const auto& [name, value] : header.items()) {
        if (name == "__metadata__") { continue; }
        require(value.is_object() && value.contains("dtype") && value.contains("shape") && value.contains("data_offsets") &&
                    value.at("shape").is_array() && value.at("data_offsets").is_array() && value.at("data_offsets").size() == 2,
                "model.safetensors: tensor '" + name + "' has an incomplete entry");
        require(value.at("dtype") == "BF16", "model.safetensors: tensor '" + name + "' is not BF16");
        Entry entry;
        std::uint64_t count = 1;
        for (const Json& d : value.at("shape")) {
            require(d.is_number_unsigned() && d.get<std::uint64_t>() <= (1ULL << 31), "model.safetensors: '" + name + "' has an invalid shape");
            entry.shape.push_back(d.get<std::uint64_t>());
            count *= entry.shape.back();
            require(count <= (1ULL << 32), "model.safetensors: '" + name + "' is too large");
        }
        const Json& offsets = value.at("data_offsets");
        require(offsets[0].is_number_unsigned() && offsets[1].is_number_unsigned(), "model.safetensors: '" + name + "' has invalid offsets");
        const std::uint64_t begin = offsets[0].get<std::uint64_t>(), end = offsets[1].get<std::uint64_t>();
        require(begin <= end && end <= payload && end - begin == count * 2 && (payload_offset + begin) % 2 == 0,
                "model.safetensors: '" + name + "' offsets do not match its shape");
        entry.offset = static_cast<std::size_t>(payload_offset + begin);
        entry.count  = static_cast<std::size_t>(count);
        out->tensors_.emplace(name, std::move(entry));
    }
    // Every tensor the program reads, at its shape; nothing else.
    const DFlash2Config& c   = out->config_;
    const std::uint64_t h    = c.hidden_size, f = c.intermediate_size, v = text.vocab_size, r = ops::kDFlash2Rank;
    const std::uint64_t q    = std::uint64_t(c.heads) * c.head_dim, kv = std::uint64_t(c.kv_heads) * c.head_dim;
    const std::uint64_t conv = 4 * (h / c.conv_group);
    std::map<std::string, std::vector<std::uint64_t>> want = {
        {"fc.weight", {h, h * c.target_visits.size()}},
        {"hidden_norm.weight", {h}},
        {"norm.weight", {h}},
        {"candidate_selector.hidden_projection.weight", {r, h}},
        {"candidate_selector.predecessor_codebook", {v, r}},
        {"candidate_selector.successor_codebook", {v, r}},
    };
    for (std::uint32_t i = 0; i < c.layers; ++i) {
        const std::string p = "layers." + std::to_string(i) + ".";
        for (const auto& [name, shape] : std::map<std::string, std::vector<std::uint64_t>>{
                 {"input_layernorm.weight", {h}},
                 {"post_attention_layernorm.weight", {h}},
                 {"self_attn.q_proj.weight", {q, h}},
                 {"self_attn.k_proj.weight", {kv, h}},
                 {"self_attn.v_proj.weight", {kv, h}},
                 {"self_attn.o_proj.weight", {h, q}},
                 {"self_attn.q_norm.weight", {c.head_dim}},
                 {"self_attn.k_norm.weight", {c.head_dim}},
                 {"mlp.gate_proj.weight", {f, h}},
                 {"mlp.up_proj.weight", {f, h}},
                 {"mlp.down_proj.weight", {h, f}},
                 {"attention_conv.base_kernel", {2, 2, h}},
                 {"attention_conv.kernel_projection.weight", {conv, h}},
                 {"mlp_conv.base_kernel", {2, 2, h}},
                 {"mlp_conv.kernel_projection.weight", {conv, h}},
             }) {
            want.emplace(p + name, shape);
        }
    }
    for (const auto& [name, shape] : want) {
        const auto it = out->tensors_.find(name);
        require(it != out->tensors_.end(), "model.safetensors has no '" + name + "'");
        require(it->second.shape == shape, "model.safetensors: '" + name + "' is not the shape config.json gives");
    }
    require(out->tensors_.size() == want.size(), "model.safetensors holds tensors a DFlash2 drafter does not");
    out->source_ = (directory / "model.safetensors").string() + " (" + std::to_string(file.size() >> 20) + " MiB)";
    return out;
}

std::span<const std::uint16_t> DFlash2Release::words(const std::string& name) const {
    const auto it = tensors_.find(name);
    if (it == tensors_.end()) { throw std::out_of_range("dflash2: no tensor " + name); }
    return {reinterpret_cast<const std::uint16_t*>(file_.data() + it->second.offset), it->second.count};
}

const std::vector<std::uint64_t>& DFlash2Release::shape(const std::string& name) const {
    const auto it = tensors_.find(name);
    if (it == tensors_.end()) { throw std::out_of_range("dflash2: no tensor " + name); }
    return it->second.shape;
}

} // namespace ninfer::models::inkling2
