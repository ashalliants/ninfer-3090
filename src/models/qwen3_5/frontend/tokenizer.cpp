#include "models/qwen3_5/frontend/tokenizer.h"
#include "ops/common/relaxed_accept.h"

#include "text/unicode.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <charconv>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace ninfer::models::qwen3_5::frontend {
namespace {

using Json    = nlohmann::json;
namespace uni = ninfer::text::unicode_internal;

constexpr std::int64_t kMaxTokenId = 1'000'000;

constexpr std::string_view kQwenSplitPattern =
    R"qwen((?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+)qwen";

// The DeepSeek-V4 pre-tokenizer, as tokenizer.json spells it (Hugging Face Split patterns, Oniguruma syntax).
constexpr std::string_view kDsv4DigitsPattern = "\\p{N}{1,3}";
constexpr std::string_view kDsv4CjkPattern    = "[\u4e00-\u9fa5\u3040-\u309f\u30a0-\u30ff]+";
constexpr std::string_view kDsv4WordPattern =
    "[!\"#$%&'()*+,\\-./:;<=>?@\\[\\\\\\]^_`{|}~][A-Za-z]+|[^\r\n\\p{L}\\p{P}\\p{S}]?[\\p{L}\\p{M}]+| "
    "?[\\p{P}\\p{S}]+[\r\n]*|\\s*[\r\n]+|\\s+(?!\\S)|\\s+";

Pipeline validate_pipeline(const Json& root, const Json& model) {
    const auto require = [](bool valid, const char* field) {
        if (!valid) {
            throw std::invalid_argument(std::string("tokenizer.json ") + field +
                                        " is not supported by the Qwen tokenizer");
        }
    };
    const auto type = [](const Json& value, const char* name) {
        return value.is_object() && value.contains("type") && value["type"] == name;
    };
    const auto is_split = [&](const Json& split, std::string_view pattern) {
        return type(split, "Split") && split.contains("pattern") && split["pattern"].is_object() &&
               split["pattern"].value("Regex", std::string{}) == pattern &&
               split.value("behavior", std::string{}) == "Isolated" &&
               split.value("invert", Json(false)) == false;
    };
    const auto is_plain_byte_level = [&](const Json& bytes) {
        return type(bytes, "ByteLevel") && bytes.value("add_prefix_space", Json(true)) == false &&
               bytes.value("use_regex", Json(true)) == false;
    };
    // Omitted pipeline descriptions use the architecture's fixed tokenizer semantics.
    // An explicit description must agree with the implemented transformations.
    Pipeline pipeline = Pipeline::Qwen;
    if (root.contains("pre_tokenizer") && root["pre_tokenizer"].is_object() &&
        root["pre_tokenizer"].contains("pretokenizers") &&
        root["pre_tokenizer"]["pretokenizers"].is_array() &&
        root["pre_tokenizer"]["pretokenizers"].size() == 4) {
        const auto& stages = root["pre_tokenizer"]["pretokenizers"];
        require(type(root["pre_tokenizer"], "Sequence") && is_split(stages[0], kDsv4DigitsPattern) &&
                    is_split(stages[1], kDsv4CjkPattern) && is_split(stages[2], kDsv4WordPattern) &&
                    is_plain_byte_level(stages[3]),
                "pre_tokenizer");
        // No normalizer: text reaches the pre-tokenizer unchanged (an empty Sequence or null is the same).
        const bool no_normalizer =
            !root.contains("normalizer") || root["normalizer"].is_null() ||
            (type(root["normalizer"], "Sequence") && root["normalizer"].contains("normalizers") &&
             root["normalizer"]["normalizers"].is_array() && root["normalizer"]["normalizers"].empty());
        require(no_normalizer, "normalizer");
        pipeline = Pipeline::Dsv4;
    } else {
        if (root.contains("normalizer")) { require(type(root["normalizer"], "NFC"), "normalizer"); }
        if (root.contains("pre_tokenizer")) {
            const auto& pre = root["pre_tokenizer"];
            require(type(pre, "Sequence") && pre.contains("pretokenizers") &&
                        pre["pretokenizers"].is_array() && pre["pretokenizers"].size() == 2,
                    "pre_tokenizer");
            require(is_split(pre["pretokenizers"][0], kQwenSplitPattern), "pre_tokenizer.Split");
            require(is_plain_byte_level(pre["pretokenizers"][1]), "pre_tokenizer.ByteLevel");
        }
    }
    if (root.contains("decoder")) { require(type(root["decoder"], "ByteLevel"), "decoder"); }
    if (root.contains("post_processor") && !root["post_processor"].is_null()) {
        require(type(root["post_processor"], "ByteLevel"), "post_processor");
    }
    if (model.contains("dropout") && !model["dropout"].is_null()) {
        require(model["dropout"].is_number() && model["dropout"] == 0, "model.dropout");
    }
    for (const char* field : {"continuing_subword_prefix", "end_of_word_suffix"}) {
        require(!model.contains(field) || model[field].is_null() || model[field] == "", field);
    }
    require(model.value("ignore_merges", Json(false)) == false, "model.ignore_merges");
    return pipeline;
}

struct VocabMetadata {
    std::vector<std::string> id_to_token;
    std::unordered_map<std::string, int> token_to_id;
    std::unordered_set<int> occupied_ids;
};

Json read_json_asset(std::string_view contents, std::string_view label) {
    try {
        return Json::parse(contents);
    } catch (const nlohmann::json::exception& ex) {
        throw std::invalid_argument("malformed " + std::string(label) + ": " + ex.what());
    }
}

const Json& require_object_field(const Json& object, const char* field, std::string_view label) {
    if (!object.is_object() || !object.contains(field)) {
        throw std::invalid_argument("missing field " + std::string(field) + " in " +
                                    std::string(label));
    }
    const Json& value = object.at(field);
    if (!value.is_object()) {
        throw std::invalid_argument("field " + std::string(field) + " must be object in " +
                                    std::string(label));
    }
    return value;
}

const Json& require_array_field(const Json& object, const char* field, std::string_view label) {
    if (!object.is_object() || !object.contains(field)) {
        throw std::invalid_argument("missing field " + std::string(field) + " in " +
                                    std::string(label));
    }
    const Json& value = object.at(field);
    if (!value.is_array()) {
        throw std::invalid_argument("field " + std::string(field) + " must be array in " +
                                    std::string(label));
    }
    return value;
}

int parse_token_id(const Json& value, const char* field, std::string_view label) {
    if (!value.is_number_integer()) {
        throw std::invalid_argument("field " + std::string(field) + " must be integer in " +
                                    std::string(label));
    }
    if (value.is_number_unsigned()) {
        const std::uint64_t id = value.get<std::uint64_t>();
        if (id > static_cast<std::uint64_t>(kMaxTokenId)) {
            throw std::invalid_argument("field " + std::string(field) + " id is out of range in " +
                                        std::string(label));
        }
        return static_cast<int>(id);
    }

    const std::int64_t id = value.get<std::int64_t>();
    if (id < 0) {
        throw std::invalid_argument("field " + std::string(field) + " has negative id in " +
                                    std::string(label));
    }
    if (id > kMaxTokenId) {
        throw std::invalid_argument("field " + std::string(field) + " id is out of range in " +
                                    std::string(label));
    }
    return static_cast<int>(id);
}

std::string require_string_field(const Json& object, const char* field, std::string_view label) {
    if (!object.is_object() || !object.contains(field) || !object.at(field).is_string()) {
        throw std::invalid_argument("field " + std::string(field) + " must be string in " +
                                    std::string(label));
    }
    return object.at(field).get<std::string>();
}

bool require_bool_field(const Json& object, const char* field, std::string_view label) {
    if (!object.is_object() || !object.contains(field) || !object.at(field).is_boolean()) {
        throw std::invalid_argument("field " + std::string(field) + " must be boolean in " +
                                    std::string(label));
    }
    return object.at(field).get<bool>();
}

VocabMetadata load_vocab(const Json& model, std::string_view label) {
    if (!model.contains("type") || !model.at("type").is_string() ||
        model.at("type").get<std::string>() != "BPE") {
        throw std::invalid_argument("field model.type must be BPE in " + std::string(label));
    }
    const Json& vocab = require_object_field(model, "vocab", label);
    if (vocab.empty()) {
        throw std::invalid_argument("field model.vocab must not be empty in " + std::string(label));
    }

    int max_id = -1;
    VocabMetadata metadata;
    for (const auto& item : vocab.items()) {
        const int id = parse_token_id(item.value(), "model.vocab", label);
        if (!metadata.occupied_ids.insert(id).second) {
            throw std::invalid_argument("field model.vocab has duplicate id in " +
                                        std::string(label));
        }
        max_id = std::max(max_id, id);
    }

    metadata.id_to_token.resize(static_cast<std::size_t>(max_id + 1));
    for (const auto& item : vocab.items()) {
        const int id = parse_token_id(item.value(), "model.vocab", label);
        metadata.id_to_token.at(static_cast<std::size_t>(id)) = item.key();
        metadata.token_to_id.emplace(item.key(), id);
    }
    return metadata;
}

AddedToken parse_added_token(const Json& item, std::string_view label) {
    if (!item.is_object()) {
        throw std::invalid_argument("field added_tokens item must be object in " +
                                    std::string(label));
    }
    AddedToken token;
    if (!item.contains("id")) {
        throw std::invalid_argument("missing field added_tokens.id in " + std::string(label));
    }
    token.id          = parse_token_id(item.at("id"), "added_tokens.id", label);
    token.content     = require_string_field(item, "content", label);
    token.single_word = require_bool_field(item, "single_word", label);
    token.lstrip      = require_bool_field(item, "lstrip", label);
    token.rstrip      = require_bool_field(item, "rstrip", label);
    token.normalized  = require_bool_field(item, "normalized", label);
    token.special     = require_bool_field(item, "special", label);
    return token;
}

AddedToken parse_added_token_decoder_entry(int id, const Json& item, std::string_view label) {
    if (!item.is_object()) {
        throw std::invalid_argument("field added_tokens_decoder item must be object in " +
                                    std::string(label));
    }
    AddedToken token;
    token.id          = id;
    token.content     = require_string_field(item, "content", label);
    token.single_word = require_bool_field(item, "single_word", label);
    token.lstrip      = require_bool_field(item, "lstrip", label);
    token.rstrip      = require_bool_field(item, "rstrip", label);
    token.normalized  = require_bool_field(item, "normalized", label);
    token.special     = require_bool_field(item, "special", label);
    return token;
}

// `normalized` asks for the normalizer to run over the token before matching; a pipeline without one (Dsv4)
// makes it irrelevant, so it is accepted there.
void validate_supported_added_token(const AddedToken& token, std::string_view label, Pipeline pipeline) {
    if (token.content.empty()) {
        throw std::invalid_argument("added token content must not be empty in " +
                                    std::string(label));
    }
    if (token.single_word || token.lstrip || token.rstrip ||
        (token.normalized && pipeline != Pipeline::Dsv4)) {
        throw std::invalid_argument("Tokenizer only supports added tokens with single_word=false, "
                                    "lstrip=false, rstrip=false, and normalized=false in " +
                                    std::string(label));
    }
}

bool same_added_token(const AddedToken& lhs, const AddedToken& rhs) noexcept {
    return lhs.id == rhs.id && lhs.content == rhs.content && lhs.single_word == rhs.single_word &&
           lhs.lstrip == rhs.lstrip && lhs.rstrip == rhs.rstrip &&
           lhs.normalized == rhs.normalized && lhs.special == rhs.special;
}

int parse_added_token_decoder_id(std::string_view key, std::string_view label) {
    std::int64_t parsed     = -1;
    const auto [end, error] = std::from_chars(key.data(), key.data() + key.size(), parsed);
    if (error != std::errc{} || end != key.data() + key.size() || parsed < 0 ||
        parsed > kMaxTokenId || std::to_string(parsed) != key) {
        throw std::invalid_argument("added_tokens_decoder key must be a nonnegative token id in " +
                                    std::string(label));
    }
    return static_cast<int>(parsed);
}

std::vector<AddedToken>
load_added_tokens(const Json& root, std::string_view label, std::vector<std::string>& id_to_token,
                  const std::unordered_set<int>& occupied_vocab_ids,
                  const std::unordered_map<std::string, int>& occupied_vocab_tokens, Pipeline pipeline) {
    const Json& added = require_array_field(root, "added_tokens", label);
    std::vector<AddedToken> tokens;
    tokens.reserve(added.size());
    std::unordered_set<int> seen_added_ids;
    std::unordered_map<std::string, int> seen_added_contents;
    for (const Json& item : added) {
        AddedToken token = parse_added_token(item, label);
        validate_supported_added_token(token, label, pipeline);
        const auto index = static_cast<std::size_t>(token.id);
        // DeepSeek-V4's tokenizer.json lists its first control tokens (BOS, EOS, pad) in the vocabulary too,
        // with the same id and content: that is the same token twice, not a clash.
        const auto in_vocab = occupied_vocab_tokens.find(token.content);
        const bool same_as_vocab_entry = pipeline == Pipeline::Dsv4 && in_vocab != occupied_vocab_tokens.end() &&
                                         in_vocab->second == token.id;
        if (occupied_vocab_ids.contains(token.id) && !same_as_vocab_entry) {
            throw std::invalid_argument("field added_tokens overlaps existing id in " +
                                        std::string(label));
        }
        if (!seen_added_ids.insert(token.id).second) {
            throw std::invalid_argument("field added_tokens has duplicate id in " +
                                        std::string(label));
        }
        if ((occupied_vocab_tokens.contains(token.content) && !same_as_vocab_entry) ||
            !seen_added_contents.emplace(token.content, token.id).second) {
            throw std::invalid_argument("field added_tokens has duplicate content mapping in " +
                                        std::string(label));
        }
        if (index >= id_to_token.size()) { id_to_token.resize(index + 1); }
        id_to_token.at(static_cast<std::size_t>(token.id)) = token.content;
        tokens.push_back(std::move(token));
    }
    return tokens;
}

void merge_added_tokens_decoder(const Json& root, std::string_view label,
                                std::vector<std::string>& id_to_token,
                                const std::unordered_set<int>& occupied_vocab_ids,
                                const std::unordered_map<std::string, int>& occupied_vocab_tokens,
                                std::vector<AddedToken>& tokens, Pipeline pipeline) {
    const Json& decoder = require_object_field(root, "added_tokens_decoder", label);
    std::unordered_map<int, std::size_t> token_by_id;
    std::unordered_map<std::string, int> token_by_content;
    std::unordered_set<int> decoder_ids;
    token_by_id.reserve(tokens.size() + decoder.size());
    token_by_content.reserve(tokens.size() + decoder.size());
    for (std::size_t index = 0; index < tokens.size(); ++index) {
        token_by_id.emplace(tokens[index].id, index);
        token_by_content.emplace(tokens[index].content, tokens[index].id);
    }

    for (const auto& item : decoder.items()) {
        const int id = parse_added_token_decoder_id(item.key(), label);
        if (!decoder_ids.insert(id).second) {
            throw std::invalid_argument("added_tokens_decoder has duplicate id mapping in " +
                                        std::string(label));
        }
        AddedToken token = parse_added_token_decoder_entry(id, item.value(), label);
        validate_supported_added_token(token, label, pipeline);

        const auto existing_id = token_by_id.find(id);
        if (existing_id != token_by_id.end()) {
            if (!same_added_token(tokens.at(existing_id->second), token)) {
                throw std::invalid_argument("conflicting added-token definition for id " +
                                            std::to_string(id) +
                                            " between tokenizer.json and tokenizer_config.json");
            }
            continue;
        }
        if (occupied_vocab_ids.contains(id)) {
            throw std::invalid_argument("added_tokens_decoder overlaps vocabulary id " +
                                        std::to_string(id));
        }
        if (token_by_content.contains(token.content) ||
            occupied_vocab_tokens.contains(token.content)) {
            throw std::invalid_argument("conflicting added-token content mapping for " +
                                        token.content);
        }

        const auto index = static_cast<std::size_t>(id);
        if (index >= id_to_token.size()) { id_to_token.resize(index + 1); }
        if (!id_to_token[index].empty()) {
            throw std::invalid_argument("duplicate tokenizer mapping for id " + std::to_string(id));
        }
        id_to_token[index] = token.content;
        token_by_id.emplace(id, tokens.size());
        token_by_content.emplace(token.content, id);
        tokens.push_back(std::move(token));
    }
    std::sort(tokens.begin(), tokens.end(),
              [](const AddedToken& lhs, const AddedToken& rhs) { return lhs.id < rhs.id; });
}

std::vector<int> load_default_stop_token_ids(std::string_view contents) {
    constexpr std::string_view label = "generation_config.json";
    const Json root                  = read_json_asset(contents, label);
    if (!root.is_object() || !root.contains("eos_token_id")) {
        throw std::invalid_argument("missing field eos_token_id in generation_config.json");
    }

    const Json& eos = root.at("eos_token_id");
    if (eos.is_number_integer()) { return {parse_token_id(eos, "eos_token_id", label)}; }
    if (eos.is_array()) {
        if (eos.empty()) {
            throw std::invalid_argument(
                "field eos_token_id must not be empty in generation_config.json");
        }
        std::vector<int> ids;
        ids.reserve(eos.size());
        for (const Json& item : eos) { ids.push_back(parse_token_id(item, "eos_token_id", label)); }
        return ids;
    }
    throw std::invalid_argument(
        "field eos_token_id must be integer or array in generation_config.json");
}

std::uint64_t merge_pair_key(int left, int right) noexcept {
    return (static_cast<std::uint64_t>(static_cast<std::uint32_t>(left)) << 32U) |
           static_cast<std::uint32_t>(right);
}

BpeMergeTable load_bpe_merge_rules(const Json& model, std::string_view label,
                                   const std::unordered_map<std::string, int>& token_to_id) {
    const Json& merges = require_array_field(model, "merges", label);
    BpeMergeTable rules;
    rules.reset(merges.size());
    // Rank order matters here and is not incidental: linear probing lets the first claimant keep
    // a slot, and a low rank is a merge the encoder performs often, so walking model.merges in
    // order hands the frequent rules their home slot. Filling the table out of an unordered
    // container instead is measurably slower.
    int rank = 0;
    for (const Json& item : merges) {
        std::string left;
        std::string right;
        if (item.is_array() && item.size() == 2 && item[0].is_string() && item[1].is_string()) {
            left  = item[0].get<std::string>();
            right = item[1].get<std::string>();
        } else if (item.is_string()) {
            const std::string pair  = item.get<std::string>();
            const std::size_t space = pair.find(' ');
            if (space == std::string::npos || space == 0 || space + 1 >= pair.size() ||
                pair.find(' ', space + 1) != std::string::npos) {
                throw std::invalid_argument("malformed model.merges entry in " +
                                            std::string(label));
            }
            left  = pair.substr(0, space);
            right = pair.substr(space + 1);
        } else {
            throw std::invalid_argument("field model.merges must contain symbol pairs in " +
                                        std::string(label));
        }
        const auto left_id   = token_to_id.find(left);
        const auto right_id  = token_to_id.find(right);
        const auto result_id = token_to_id.find(left + right);
        if (left_id == token_to_id.end() || right_id == token_to_id.end() ||
            result_id == token_to_id.end()) {
            throw std::invalid_argument("model.merges references a symbol outside model.vocab in " +
                                        std::string(label));
        }
        const bool inserted =
            rules.insert(merge_pair_key(left_id->second, right_id->second),
                         BpeMergeRule{.rank = rank++, .result = result_id->second});
        if (!inserted) { throw std::invalid_argument("duplicate merge pair in model.merges"); }
    }
    return rules;
}

std::unordered_map<std::uint32_t, char> build_byte_level_decoder() {
    std::unordered_map<std::uint32_t, char> decoder;
    std::uint32_t next = 256;
    for (int byte = 0; byte <= std::numeric_limits<unsigned char>::max(); ++byte) {
        const bool visible = (byte >= 33 && byte <= 126) || (byte >= 161 && byte <= 172) ||
                             (byte >= 174 && byte <= 255);
        const std::uint32_t codepoint = visible ? static_cast<std::uint32_t>(byte) : next++;
        decoder.emplace(codepoint, static_cast<char>(static_cast<unsigned char>(byte)));
    }
    return decoder;
}

std::string decode_byte_level_token(std::string_view token, int id) {
    static const std::unordered_map<std::uint32_t, char> byte_decoder = build_byte_level_decoder();
    std::string bytes;
    const std::vector<uni::CodepointSpan> codepoints =
        uni::utf8_codepoints(token, "Tokenizer::decode token id " + std::to_string(id));
    bytes.reserve(codepoints.size());
    for (const uni::CodepointSpan& codepoint : codepoints) {
        const auto byte = byte_decoder.find(static_cast<std::uint32_t>(codepoint.value));
        if (byte == byte_decoder.end()) {
            throw std::invalid_argument("Tokenizer::decode token id " + std::to_string(id) +
                                        " contains a character outside the byte-level alphabet");
        }
        bytes.push_back(byte->second);
    }
    return bytes;
}

std::array<std::string, 256> build_byte_level_encoder() {
    std::array<std::string, 256> encoder;
    std::uint32_t next = 256;
    for (int byte = 0; byte <= std::numeric_limits<unsigned char>::max(); ++byte) {
        const bool visible = (byte >= 33 && byte <= 126) || (byte >= 161 && byte <= 172) ||
                             (byte >= 174 && byte <= 255);
        const std::uint32_t codepoint = visible ? static_cast<std::uint32_t>(byte) : next++;
        encoder[static_cast<unsigned char>(byte)] =
            uni::codepoint_to_utf8(static_cast<std::int32_t>(codepoint));
    }
    return encoder;
}

bool is_newline(std::int32_t codepoint) noexcept { return codepoint == '\r' || codepoint == '\n'; }

bool is_letter_or_mark(std::int32_t codepoint) noexcept {
    return uni::is_letter(codepoint) || uni::is_mark(codepoint);
}

bool is_non_newline_non_letter_non_number(std::int32_t codepoint) noexcept {
    return !is_newline(codepoint) && !uni::is_letter(codepoint) && !uni::is_number(codepoint);
}

bool is_non_space_non_letter_mark_number(std::int32_t codepoint) noexcept {
    return !uni::is_whitespace(codepoint) && !uni::is_letter(codepoint) &&
           !uni::is_mark(codepoint) && !uni::is_number(codepoint);
}

bool ascii_ci_matches(std::string_view text, std::size_t offset, std::string_view suffix) {
    if (offset + suffix.size() > text.size()) { return false; }
    for (std::size_t i = 0; i < suffix.size(); ++i) {
        const unsigned char lhs = static_cast<unsigned char>(text[offset + i]);
        const unsigned char rhs = static_cast<unsigned char>(suffix[i]);
        if (std::tolower(lhs) != std::tolower(rhs)) { return false; }
    }
    return true;
}

uni::CodepointSpan qwen_codepoint_at(std::string_view text, std::size_t offset) {
    return uni::utf8_codepoint_at(text, offset, "Tokenizer::encode input");
}

std::size_t qwen_word_end(std::string_view text, std::size_t begin) {
    const uni::CodepointSpan first = qwen_codepoint_at(text, begin);
    const std::size_t after_first  = begin + first.length;
    const std::int32_t cp          = first.value;

    if (cp == '\'') {
        constexpr std::string_view suffixes[] = {"s", "t", "re", "ve", "m", "ll", "d"};
        for (std::string_view suffix : suffixes) {
            if (ascii_ci_matches(text, after_first, suffix)) { return after_first + suffix.size(); }
        }
    }

    const bool has_next     = after_first < text.size();
    const std::int32_t next = has_next ? qwen_codepoint_at(text, after_first).value : 0;
    if (is_letter_or_mark(cp) ||
        (is_non_newline_non_letter_non_number(cp) && has_next && is_letter_or_mark(next))) {
        std::size_t end = is_letter_or_mark(cp) ? begin : after_first;
        while (end < text.size()) {
            const uni::CodepointSpan value = qwen_codepoint_at(text, end);
            if (!is_letter_or_mark(value.value)) { break; }
            end += value.length;
        }
        return end;
    }

    if (uni::is_number(cp)) { return after_first; }

    if ((cp == ' ' && has_next && is_non_space_non_letter_mark_number(next)) ||
        is_non_space_non_letter_mark_number(cp)) {
        std::size_t end = cp == ' ' ? after_first : begin;
        while (end < text.size()) {
            const uni::CodepointSpan value = qwen_codepoint_at(text, end);
            if (!is_non_space_non_letter_mark_number(value.value)) { break; }
            end += value.length;
        }
        while (end < text.size()) {
            const uni::CodepointSpan value = qwen_codepoint_at(text, end);
            if (!is_newline(value.value)) { break; }
            end += value.length;
        }
        return end;
    }

    if (uni::is_whitespace(cp)) {
        std::size_t end              = begin;
        std::size_t last_begin       = begin;
        std::size_t last_newline_end = std::string_view::npos;
        std::size_t count            = 0;
        while (end < text.size()) {
            const uni::CodepointSpan value = qwen_codepoint_at(text, end);
            if (!uni::is_whitespace(value.value)) { break; }
            last_begin = end;
            end += value.length;
            ++count;
            if (is_newline(value.value)) { last_newline_end = end; }
        }
        if (last_newline_end != std::string_view::npos) { return last_newline_end; }
        if (end == text.size()) { return end; }
        return count >= 2 ? last_begin : end;
    }

    return after_first;
}

// ---------------------------------------------------------------- DeepSeek-V4 pre-tokenizer
//
// tokenizer.json's three Splits (behaviour Isolated: every match and every gap between matches is a piece),
// applied one after another to the pieces the previous one made:
//   1. \p{N}{1,3}                       digits, in runs of up to three;
//   2. [一-龥぀-ゟ゠-ヿ]+   runs of CJK ideographs and kana;
//   3. the word regex, once per piece the first two leave (its end-of-text is the end of that piece).
// `dsv4_word_end` returns where the piece starting at `begin` (always a piece start) ends.

bool dsv4_is_cjk(std::int32_t cp) noexcept {
    return (cp >= 0x4e00 && cp <= 0x9fa5) || (cp >= 0x3040 && cp <= 0x309f) || (cp >= 0x30a0 && cp <= 0x30ff);
}

// Oniguruma's \s over UTF-8: Unicode White_Space.
bool dsv4_is_space(std::int32_t cp) noexcept { return uni::is_whitespace(cp) || cp == 0x85; }

bool dsv4_is_ascii_punct(std::int32_t cp) noexcept {
    constexpr std::string_view set = "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~";
    return cp > 0 && cp < 0x80 && set.find(static_cast<char>(cp)) != std::string_view::npos;
}

bool dsv4_is_ascii_letter(std::int32_t cp) noexcept { return (cp >= 'A' && cp <= 'Z') || (cp >= 'a' && cp <= 'z'); }

bool dsv4_is_punct_or_symbol(std::int32_t cp) noexcept { return uni::is_punctuation(cp) || uni::is_symbol(cp); }

// [^\r\n\p{L}\p{P}\p{S}]
bool dsv4_is_prefix_char(std::int32_t cp) noexcept {
    return !is_newline(cp) && !uni::is_letter(cp) && !dsv4_is_punct_or_symbol(cp);
}

// The word regex's alternatives tried at `at` in order, over text[0, limit): the end of the first that matches,
// or npos when none does (the position then belongs to a gap).
std::size_t dsv4_match_at(std::string_view text, std::size_t at, std::size_t limit) {
    const auto codepoint = [&](std::size_t offset) { return qwen_codepoint_at(text, offset); };
    const uni::CodepointSpan first = codepoint(at);
    const std::size_t after_first  = at + first.length;
    const bool has_next            = after_first < limit;
    const std::int32_t next        = has_next ? codepoint(after_first).value : 0;
    const std::int32_t cp          = first.value;
    const auto run_end = [&](std::size_t from, auto&& in_class) {
        std::size_t end = from;
        while (end < limit) {
            const uni::CodepointSpan value = codepoint(end);
            if (!in_class(value.value)) { break; }
            end += value.length;
        }
        return end;
    };

    // 1. [ASCII punctuation][A-Za-z]+
    if (dsv4_is_ascii_punct(cp) && has_next && dsv4_is_ascii_letter(next)) {
        return run_end(after_first, dsv4_is_ascii_letter);
    }
    // 2. [^\r\n\p{L}\p{P}\p{S}]?[\p{L}\p{M}]+
    if (dsv4_is_prefix_char(cp) && has_next && is_letter_or_mark(next)) {
        return run_end(after_first, is_letter_or_mark);
    }
    if (is_letter_or_mark(cp)) { return run_end(at, is_letter_or_mark); }
    // 3.  ?[\p{P}\p{S}]+[\r\n]*
    {
        std::size_t from = at;
        if (cp == ' ' && has_next && dsv4_is_punct_or_symbol(next)) {
            from = after_first;
        } else if (!dsv4_is_punct_or_symbol(cp)) {
            from = std::string_view::npos;
        }
        if (from != std::string_view::npos) {
            return run_end(run_end(from, dsv4_is_punct_or_symbol), is_newline);
        }
    }
    if (!dsv4_is_space(cp)) { return std::string_view::npos; }
    // 4-6. whitespace: \s*[\r\n]+ | \s+(?!\S) | \s+
    std::size_t end              = at;
    std::size_t last_begin       = at;
    std::size_t last_newline_end = std::string_view::npos;
    std::size_t count            = 0;
    while (end < limit) {
        const uni::CodepointSpan value = codepoint(end);
        if (!dsv4_is_space(value.value)) { break; }
        last_begin = end;
        end += value.length;
        ++count;
        if (is_newline(value.value)) { last_newline_end = end; }
    }
    if (last_newline_end != std::string_view::npos) { return last_newline_end; }
    if (end == limit) { return end; }
    return count >= 2 ? last_begin : end;
}

std::size_t dsv4_word_end(std::string_view text, std::size_t begin) {
    const uni::CodepointSpan first = qwen_codepoint_at(text, begin);
    const std::int32_t cp          = first.value;
    if (uni::is_number(cp)) {  // stage 1: up to three digits, and nothing else matches a digit piece
        std::size_t end = begin;
        for (int count = 0; count < 3 && end < text.size(); ++count) {
            const uni::CodepointSpan value = qwen_codepoint_at(text, end);
            if (!uni::is_number(value.value)) { break; }
            end += value.length;
        }
        return end;
    }
    // The piece stages 1 and 2 leave around `begin`: a CJK run, or the text up to the next digit or CJK char.
    const bool cjk = dsv4_is_cjk(cp);
    std::size_t limit = begin;
    while (limit < text.size()) {
        const uni::CodepointSpan value = qwen_codepoint_at(text, limit);
        if (cjk ? !dsv4_is_cjk(value.value) : (uni::is_number(value.value) || dsv4_is_cjk(value.value))) { break; }
        limit += value.length;
    }
    // Stage 3 within [begin, limit): a match starting here is the word; otherwise the gap up to the next match.
    std::size_t at = begin;
    while (at < limit) {
        const std::size_t matched = dsv4_match_at(text, at, limit);
        if (matched != std::string_view::npos) { return at == begin ? matched : at; }
        at += qwen_codepoint_at(text, at).length;
    }
    return limit;
}

std::size_t word_end(Pipeline pipeline, std::string_view text, std::size_t begin) {
    return pipeline == Pipeline::Dsv4 ? dsv4_word_end(text, begin) : qwen_word_end(text, begin);
}

// NFC for Qwen; DeepSeek-V4 has no normalizer, so its text passes through unchanged.
std::string normalize_text(Pipeline pipeline, std::string_view text) {
    return pipeline == Pipeline::Dsv4 ? std::string(text) : uni::normalize_nfc(text);
}

bool is_stop_token_id(std::span<const int> stop_token_ids, int id) {
    return std::find(stop_token_ids.begin(), stop_token_ids.end(), id) != stop_token_ids.end();
}

struct BpeNode {
    int symbol               = -1;
    int previous             = -1;
    int next                 = -1;
    std::size_t end          = 0;
    std::uint32_t generation = 0;
    bool live                = true;
};

struct BpeCandidate {
    int rank                       = 0;
    int left                       = -1;
    int right                      = -1;
    int result                     = -1;
    std::uint32_t left_generation  = 0;
    std::uint32_t right_generation = 0;
};

struct BpeWordEnd {
    std::size_t normalized_offset = 0;
    std::size_t token_frontier    = 0;
};

struct LaterBpeCandidate {
    bool operator()(const BpeCandidate& lhs, const BpeCandidate& rhs) const noexcept {
        if (lhs.rank != rhs.rank) { return lhs.rank > rhs.rank; }
        return lhs.left > rhs.left;
    }
};

// One encode call owns this scratch; words and ordinary-text segments reuse its capacity.
struct BpeScratch {
    std::vector<BpeNode> nodes;
    std::vector<BpeCandidate> candidates;
};

std::array<int, 256> load_byte_token_ids(const std::unordered_map<std::string, int>& token_to_id) {
    static const std::array<std::string, 256> byte_encoder = build_byte_level_encoder();
    std::array<int, 256> ids;
    ids.fill(-1);
    for (std::size_t byte = 0; byte < ids.size(); ++byte) {
        const auto token = token_to_id.find(byte_encoder[byte]);
        if (token != token_to_id.end()) { ids[byte] = token->second; }
    }
    return ids;
}

bool append_normalized_bpe_ids(Pipeline pipeline, std::vector<int>& ids, std::string_view normalized,
                               const BpeMergeTable& merge_rules,
                               const std::array<int, 256>& byte_token_ids, std::size_t max_tokens,
                               BpeScratch& scratch, std::vector<std::size_t>* token_ends = nullptr,
                               std::vector<BpeWordEnd>* word_ends = nullptr) {
    if (normalized.empty()) { return true; }
    if (ids.size() == max_tokens) { return false; }

    for (std::size_t begin = 0; begin < normalized.size();) {
        const std::size_t end = word_end(pipeline, normalized, begin);
        const std::string_view word(normalized.data() + begin, end - begin);
        auto& nodes = scratch.nodes;
        nodes.clear();
        nodes.reserve(word.size());
        for (std::size_t index = 0; index < word.size(); ++index) {
            const unsigned char byte = static_cast<unsigned char>(word[index]);
            const int symbol         = byte_token_ids[byte];
            if (symbol < 0) {
                throw std::invalid_argument(
                    "Tokenizer::encode produced byte symbol outside vocabulary");
            }
            nodes.push_back(
                BpeNode{.symbol   = symbol,
                        .previous = index == 0 ? -1 : static_cast<int>(index - 1),
                        .next     = index + 1 == word.size() ? -1 : static_cast<int>(index + 1),
                        .end      = index + 1});
        }

        auto& queue = scratch.candidates;
        queue.clear();
        const auto push_candidate = [&](int left, bool heap_ready) {
            if (left < 0 || !nodes[static_cast<std::size_t>(left)].live) { return; }
            const int right = nodes[static_cast<std::size_t>(left)].next;
            if (right < 0) { return; }
            const BpeMergeEntry* rule =
                merge_rules.find(merge_pair_key(nodes[static_cast<std::size_t>(left)].symbol,
                                                nodes[static_cast<std::size_t>(right)].symbol));
            if (rule == nullptr) { return; }
            queue.push_back(BpeCandidate{
                .rank             = rule->rank,
                .left             = left,
                .right            = right,
                .result           = rule->result,
                .left_generation  = nodes[static_cast<std::size_t>(left)].generation,
                .right_generation = nodes[static_cast<std::size_t>(right)].generation,
            });
            if (heap_ready) { std::push_heap(queue.begin(), queue.end(), LaterBpeCandidate{}); }
        };
        for (std::size_t index = 0; index + 1 < nodes.size(); ++index) {
            push_candidate(static_cast<int>(index), false);
        }
        std::make_heap(queue.begin(), queue.end(), LaterBpeCandidate{});
        while (!queue.empty()) {
            std::pop_heap(queue.begin(), queue.end(), LaterBpeCandidate{});
            const BpeCandidate candidate = queue.back();
            queue.pop_back();
            BpeNode& left  = nodes[static_cast<std::size_t>(candidate.left)];
            BpeNode& right = nodes[static_cast<std::size_t>(candidate.right)];
            if (!left.live || !right.live || left.next != candidate.right ||
                left.generation != candidate.left_generation ||
                right.generation != candidate.right_generation) {
                continue;
            }
            left.symbol = candidate.result;
            left.end    = right.end;
            ++left.generation;
            left.next  = right.next;
            right.live = false;
            ++right.generation;
            if (left.next >= 0) {
                nodes[static_cast<std::size_t>(left.next)].previous = candidate.left;
            }
            push_candidate(left.previous, true);
            push_candidate(candidate.left, true);
        }
        for (int node = nodes.empty() ? -1 : 0; node >= 0;
             node     = nodes[static_cast<std::size_t>(node)].next) {
            ids.push_back(nodes[static_cast<std::size_t>(node)].symbol);
            if (token_ends != nullptr) {
                token_ends->push_back(begin + nodes[static_cast<std::size_t>(node)].end);
            }
            if (ids.size() == max_tokens) { return false; }
        }
        if (word_ends != nullptr) {
            word_ends->push_back(
                BpeWordEnd{.normalized_offset = end, .token_frontier = ids.size()});
        }
        begin = end;
    }
    return true;
}

struct IndexedByteBoundary {
    std::size_t offset = 0;
    std::size_t index  = 0;
};

bool append_ordinary_text(Pipeline pipeline, BoundaryEncodedText& encoded, std::string_view text,
                          std::size_t text_offset, std::span<const IndexedByteBoundary> boundaries,
                          const BpeMergeTable& merge_rules,
                          const std::array<int, 256>& byte_token_ids, std::size_t max_tokens,
                          BpeScratch& scratch) {
    const std::size_t token_base = encoded.input_ids.size();
    if (text.empty()) {
        for (const IndexedByteBoundary boundary : boundaries) {
            encoded.boundaries[boundary.index] =
                TokenBoundaryResult{.exact_frontier = token_base, .stable_frontier = token_base};
        }
        return true;
    }

    const std::string normalized = normalize_text(pipeline, text);
    const bool has_internal_boundary =
        std::any_of(boundaries.begin(), boundaries.end(), [&](IndexedByteBoundary boundary) {
            const std::size_t local = boundary.offset - text_offset;
            return local != 0 && local != text.size();
        });
    std::vector<std::size_t> token_ends;
    std::vector<BpeWordEnd> word_ends;
    if (has_internal_boundary) { token_ends.reserve(normalized.size()); }
    if (!append_normalized_bpe_ids(pipeline, encoded.input_ids, normalized, merge_rules, byte_token_ids,
                                   max_tokens, scratch,
                                   has_internal_boundary ? &token_ends : nullptr,
                                   has_internal_boundary ? &word_ends : nullptr)) {
        return false;
    }
    const std::size_t token_end = encoded.input_ids.size();
    if (!has_internal_boundary) {
        for (const IndexedByteBoundary boundary : boundaries) {
            const std::size_t frontier = boundary.offset == text_offset ? token_base : token_end;
            encoded.boundaries[boundary.index] =
                TokenBoundaryResult{.exact_frontier = frontier, .stable_frontier = frontier};
        }
        return true;
    }

    std::vector<std::size_t> normalized_boundaries(boundaries.size());
    std::string independently_normalized;
    independently_normalized.reserve(normalized.size());
    std::size_t raw_begin = 0;
    for (std::size_t request = 0; request < boundaries.size(); ++request) {
        const std::size_t raw_end = boundaries[request].offset - text_offset;
        if (raw_end != raw_begin) {
            independently_normalized +=
                normalize_text(pipeline, text.substr(raw_begin, raw_end - raw_begin));
            raw_begin = raw_end;
        }
        normalized_boundaries[request] = independently_normalized.size();
    }
    if (raw_begin != text.size()) {
        independently_normalized += normalize_text(pipeline, text.substr(raw_begin));
    }

    if (independently_normalized != normalized) {
        // A requested marker splits a canonical-composition sequence. The full token stream stays
        // authoritative; conservatively retain only the preceding added-token/ordinary boundary.
        for (const IndexedByteBoundary boundary : boundaries) {
            const std::size_t local = boundary.offset - text_offset;
            if (local == 0 || local == text.size()) {
                const std::size_t frontier = local == 0 ? token_base : token_end;
                encoded.boundaries[boundary.index] =
                    TokenBoundaryResult{.exact_frontier = frontier, .stable_frontier = frontier};
            } else {
                encoded.boundaries[boundary.index] = TokenBoundaryResult{
                    .exact_frontier = std::nullopt, .stable_frontier = token_base};
            }
        }
        return true;
    }

    std::size_t token_cursor    = 0;
    std::size_t word_cursor     = 0;
    std::size_t stable_frontier = token_base;
    for (std::size_t request = 0; request < boundaries.size(); ++request) {
        const std::size_t normalized_boundary = normalized_boundaries[request];
        while (token_cursor < token_ends.size() &&
               token_ends[token_cursor] <= normalized_boundary) {
            ++token_cursor;
        }
        while (word_cursor < word_ends.size() &&
               word_ends[word_cursor].normalized_offset <= normalized_boundary) {
            stable_frontier = word_ends[word_cursor].token_frontier;
            ++word_cursor;
        }
        const bool exact =
            normalized_boundary == 0 ||
            (token_cursor != 0 && token_ends[token_cursor - 1] == normalized_boundary);
        const std::size_t frontier                    = token_base + token_cursor;
        encoded.boundaries[boundaries[request].index] = TokenBoundaryResult{
            .exact_frontier  = exact ? std::optional<std::size_t>(frontier) : std::nullopt,
            .stable_frontier = stable_frontier};
    }
    return true;
}

} // namespace

std::vector<std::size_t> pre_tokenize_ends(Pipeline pipeline, std::string_view text) {
    std::vector<std::size_t> ends;
    for (std::size_t begin = 0; begin < text.size();) {
        begin = word_end(pipeline, text, begin);
        ends.push_back(begin);
    }
    return ends;
}

Tokenizer::Tokenizer(TokenizerResources resources) {
    if (resources.tokenizer_json.empty() || resources.tokenizer_config_json.empty() ||
        resources.generation_config_json.empty()) {
        throw std::invalid_argument("embedded tokenizer resources are empty");
    }
    constexpr std::string_view tokenizer_label        = "tokenizer.json";
    constexpr std::string_view tokenizer_config_label = "tokenizer_config.json";
    const Json root = read_json_asset(resources.tokenizer_json, tokenizer_label);
    const Json tokenizer_config =
        read_json_asset(resources.tokenizer_config_json, tokenizer_config_label);
    const Json& model = require_object_field(root, "model", tokenizer_label);
    pipeline_ = validate_pipeline(root, model);

    VocabMetadata vocab_metadata = load_vocab(model, tokenizer_label);
    decoded_token_bytes_         = std::move(vocab_metadata.id_to_token);
    vocab_token_to_id_           = std::move(vocab_metadata.token_to_id);
    valid_token_ids_.resize(decoded_token_bytes_.size());
    for (const int id : vocab_metadata.occupied_ids) {
        valid_token_ids_.at(static_cast<std::size_t>(id)) = true;
    }
    added_tokens_ = load_added_tokens(root, tokenizer_label, decoded_token_bytes_,
                                      vocab_metadata.occupied_ids, vocab_token_to_id_, pipeline_);
    merge_added_tokens_decoder(tokenizer_config, tokenizer_config_label, decoded_token_bytes_,
                               vocab_metadata.occupied_ids, vocab_token_to_id_, added_tokens_, pipeline_);
    for (std::size_t index = 0; index < added_tokens_.size(); ++index) {
        const std::string& content = added_tokens_[index].content;
        if (!content.empty()) {
            added_token_candidates_[static_cast<unsigned char>(content.front())].push_back(index);
        }
    }
    if (valid_token_ids_.size() < decoded_token_bytes_.size()) {
        valid_token_ids_.resize(decoded_token_bytes_.size());
    }
    std::vector<bool> added_token_ids(decoded_token_bytes_.size());
    special_token_ids_.resize(decoded_token_bytes_.size());
    for (const AddedToken& token : added_tokens_) {
        const auto index             = static_cast<std::size_t>(token.id);
        valid_token_ids_.at(index)   = true;
        added_token_ids.at(index)    = true;
        special_token_ids_.at(index) = token.special;
    }
    for (std::size_t index = 0; index < decoded_token_bytes_.size(); ++index) {
        if (valid_token_ids_[index] && !added_token_ids[index]) {
            decoded_token_bytes_[index] =
                decode_byte_level_token(decoded_token_bytes_[index], static_cast<int>(index));
        }
    }
    bpe_merge_rules_        = load_bpe_merge_rules(model, tokenizer_label, vocab_token_to_id_);
    byte_token_ids_         = load_byte_token_ids(vocab_token_to_id_);
    default_stop_token_ids_ = load_default_stop_token_ids(resources.generation_config_json);
}

std::vector<int> Tokenizer::encode(std::string_view text, EncodeOptions options) const {
    BoundaryEncodedText encoded = encode_with_boundaries(text, {}, options);
    return std::move(encoded.input_ids);
}

BoundaryEncodedText Tokenizer::encode_with_boundaries(
    std::string_view text, std::span<const std::size_t> byte_boundaries, EncodeOptions options,
    std::span<const text::ByteSpan> literal_spans) const {
    std::size_t previous_end = 0;
    for (const auto span : literal_spans) {
        if (span.begin < previous_end || span.begin >= span.end || span.end > text.size())
            throw std::invalid_argument(
                "Tokenizer literal spans must be ordered disjoint input ranges");
        previous_end = span.end;
    }
    BoundaryEncodedText encoded;
    encoded.boundaries.resize(byte_boundaries.size());
    std::vector<IndexedByteBoundary> boundaries;
    boundaries.reserve(byte_boundaries.size());
    for (std::size_t index = 0; index < byte_boundaries.size(); ++index) {
        if (byte_boundaries[index] > text.size()) {
            throw std::out_of_range("Tokenizer byte boundary exceeds input text");
        }
        boundaries.push_back(IndexedByteBoundary{.offset = byte_boundaries[index], .index = index});
    }
    std::stable_sort(
        boundaries.begin(), boundaries.end(),
        [](IndexedByteBoundary lhs, IndexedByteBoundary rhs) { return lhs.offset < rhs.offset; });
    if (options.max_tokens == 0) { return encoded; }

    BpeScratch scratch;
    std::size_t boundary_cursor     = 0;
    const auto boundary_end_through = [&](std::size_t offset) {
        std::size_t end = boundary_cursor;
        while (end < boundaries.size() && boundaries[end].offset <= offset) { ++end; }
        return end;
    };
    const auto append_ordinary = [&](std::size_t begin, std::size_t end) {
        const std::size_t request_end = boundary_end_through(end);
        const bool complete =
            append_ordinary_text(pipeline_, encoded, text.substr(begin, end - begin), begin,
                                 std::span<const IndexedByteBoundary>(boundaries)
                                     .subspan(boundary_cursor, request_end - boundary_cursor),
                                 bpe_merge_rules_, byte_token_ids_, options.max_tokens, scratch);
        boundary_cursor = request_end;
        return complete;
    };

    if (!options.parse_added_tokens) {
        (void)append_ordinary(0, text.size());
        return encoded;
    }

    std::size_t ordinary_begin = 0;
    std::size_t pos            = 0;
    while (pos < text.size()) {
        const AddedToken* match_token = nullptr;
        const auto& candidates = added_token_candidates_[static_cast<unsigned char>(text[pos])];
        for (const std::size_t index : candidates) {
            const AddedToken& token = added_tokens_[index];
            if (token.content.size() <= text.size() - pos &&
                text.compare(pos, token.content.size(), token.content) == 0 &&
                !text::overlaps(literal_spans, pos, pos + token.content.size())) {
                match_token = &token;
                break;
            }
        }

        if (match_token == nullptr) {
            ++pos;
            continue;
        }

        if (!append_ordinary(ordinary_begin, pos)) { return encoded; }

        const std::size_t token_end       = pos + match_token->content.size();
        const std::size_t stable_frontier = encoded.input_ids.size();
        while (boundary_cursor < boundaries.size() &&
               boundaries[boundary_cursor].offset < token_end) {
            encoded.boundaries[boundaries[boundary_cursor].index] = TokenBoundaryResult{
                .exact_frontier = std::nullopt, .stable_frontier = stable_frontier};
            ++boundary_cursor;
        }
        encoded.input_ids.push_back(match_token->id);
        if (encoded.input_ids.size() == options.max_tokens) { return encoded; }
        while (boundary_cursor < boundaries.size() &&
               boundaries[boundary_cursor].offset == token_end) {
            encoded.boundaries[boundaries[boundary_cursor].index] =
                TokenBoundaryResult{.exact_frontier  = encoded.input_ids.size(),
                                    .stable_frontier = encoded.input_ids.size()};
            ++boundary_cursor;
        }
        pos            = token_end;
        ordinary_begin = pos;
    }
    (void)append_ordinary(ordinary_begin, text.size());
    return encoded;
}

std::string Tokenizer::decode(std::span<const int> ids, DecodeOptions options) const {
    std::string text;
    const std::size_t terminal_stop_index =
        (!ids.empty() && is_stop_token_id(options.stop_token_ids, ids.back())) ? ids.size() - 1
                                                                               : ids.size();

    for (std::size_t i = 0; i < ids.size(); ++i) {
        const int id = ids[i];
        if (i == terminal_stop_index) { continue; }
        text += decode_token_bytes(id, options.skip_special_tokens);
    }
    (void)uni::utf8_codepoints(text, "Tokenizer::decode reconstructed output");
    return text;
}

DecodedTokenView Tokenizer::decoded_token(int id) const {
    if (id < 0) {
        throw std::invalid_argument("Tokenizer::decode received negative token id " +
                                    std::to_string(id));
    }
    const auto index = static_cast<std::size_t>(id);
    if (index >= decoded_token_bytes_.size() || index >= valid_token_ids_.size() ||
        !valid_token_ids_.at(index)) {
        throw std::out_of_range("Tokenizer::decode token id " + std::to_string(id) +
                                " is outside loaded vocabulary");
    }
    return DecodedTokenView{.bytes   = decoded_token_bytes_[index],
                            .special = special_token_ids_[index]};
}

std::string_view Tokenizer::decode_token_bytes(int id, bool skip_special_tokens) const {
    const DecodedTokenView token = decoded_token(id);
    return skip_special_tokens && token.special ? std::string_view{} : token.bytes;
}

bool Tokenizer::is_special_token(int id) const noexcept {
    return id >= 0 && static_cast<std::size_t>(id) < special_token_ids_.size() &&
           special_token_ids_[static_cast<std::size_t>(id)];
}

bool Tokenizer::is_valid_token(int id) const noexcept {
    return id >= 0 && static_cast<std::size_t>(id) < valid_token_ids_.size() &&
           valid_token_ids_[static_cast<std::size_t>(id)];
}

bool Tokenizer::has_exact_token_domain(std::size_t size) const noexcept {
    return valid_token_ids_.size() == size &&
           std::find(valid_token_ids_.begin(), valid_token_ids_.end(), false) ==
               valid_token_ids_.end();
}

std::vector<int> Tokenizer::added_token_ids() const {
    std::vector<int> ids;
    ids.reserve(added_tokens_.size());
    for (const AddedToken& token : added_tokens_) { ids.push_back(token.id); }
    return ids;
}

RelaxedDraftingTokens relaxed_drafting_tokens(const Tokenizer& tokenizer) {
    const std::vector<int> added = tokenizer.added_token_ids();
    const std::vector<std::int32_t> ids(added.begin(), added.end());
    RelaxedDraftingTokens out;
    out.added_token_floor = ops::added_token_floor(ids);
    const auto tag        = [&](std::string_view text) {
        std::vector<int> encoded;
        try {
            encoded = tokenizer.encode(text);
        } catch (const std::exception&) { encoded.clear(); }
        if (encoded.size() != 1 || encoded.front() < out.added_token_floor) {
            throw std::invalid_argument("--relaxed-drafting needs " + std::string(text) +
                                        " to be a single added token of the tokenizer");
        }
        return static_cast<std::int32_t>(encoded.front());
    };
    out.tool_call_open  = tag("<tool_call>");
    out.tool_call_close = tag("</tool_call>");
    return out;
}

} // namespace ninfer::models::qwen3_5::frontend
