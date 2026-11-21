// Written for NInfer-3090 (Inkling-2-Mini bring-up).
//
// The Inkling-2 tokenizer pipeline: its pre-tokenizer (one o200k-style Split, then byte level, no normalizer) cuts
// text exactly where the Hugging Face tokenizers library does; BPE with ignore_merges emits a word that is a
// vocabulary token as that token; a tokenizer.json that describes it loads and one that describes something else is
// refused; and, with the real file, encode and decode equal the library's.

#include "models/qwen3_5/frontend/tokenizer.h"

#include <nlohmann/json.hpp>

#include <array>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using ninfer::models::qwen3_5::frontend::Pipeline;
using ninfer::models::qwen3_5::frontend::Tokenizer;
using ninfer::models::qwen3_5::frontend::TokenizerResources;
using Json = nlohmann::json;

int g_failures = 0;
void expect(bool ok, const std::string& what) {
    if (!ok) {
        std::cerr << "FAILED: " << what << '\n';
        ++g_failures;
    }
}

Json read_json(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) { throw std::runtime_error("cannot open " + path); }
    return Json::parse(in);
}

const Json& golden() {
    static const Json value = read_json(NINFER_SOURCE_DIR "/tests/fixtures/inkling2/pretokenizer_golden.json");
    return value;
}

// The golden: the Hugging Face tokenizers library applying the real tokenizer.json's Split
// (tools/inkling2/tokenizer_golden.py). Each case is [text, [pieces...]].
void pre_tokenizer_matches_hugging_face() {
    std::size_t checked = 0;
    for (const Json& item : golden().at("cases")) {
        const std::string text = item.at(0).get<std::string>();
        std::vector<std::size_t> want;
        std::size_t at = 0;
        for (const Json& piece : item.at(1)) {
            at += piece.get<std::string>().size();
            want.push_back(at);
        }
        const auto got = ninfer::models::qwen3_5::frontend::pre_tokenize_ends(Pipeline::Inkling2, text);
        if (got != want) {
            std::string shown;
            std::size_t begin = 0;
            for (const std::size_t end : got) {
                shown += "[" + text.substr(begin, end - begin) + "]";
                begin = end;
            }
            expect(false, "pre-tokenizer differs on \"" + text + "\": got " + shown + ", expected " + item.at(1).dump());
        }
        ++checked;
    }
    expect(checked > 1300, "the golden has over 1,300 cases");
}

void long_text_is_pre_tokenized_in_linear_time() {
    std::string text;
    for (int i = 0; i < 600000; ++i) { text += "Word, "; }
    const auto start     = std::chrono::steady_clock::now();
    const auto ends      = ninfer::models::qwen3_5::frontend::pre_tokenize_ends(Pipeline::Inkling2, text);
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    expect(ends.size() == 2 * 600000 + 1 && ends.back() == text.size(), "\"Word, \" repeated cuts into words, commas and a final space");
    expect(seconds < 10.0, "3.6 MB of text pre-tokenized in " + std::to_string(seconds) + " s");
}

// ByteLevel's alphabet (GPT-2's bytes_to_unicode).
std::array<std::string, 256> byte_alphabet() {
    std::array<std::string, 256> out;
    std::uint32_t next = 256;
    const auto utf8 = [](std::uint32_t cp) {
        std::string s;
        if (cp < 0x80) {
            s += static_cast<char>(cp);
        } else {
            s += static_cast<char>(0xc0 | (cp >> 6));
            s += static_cast<char>(0x80 | (cp & 0x3f));
        }
        return s;
    };
    for (int b = 0; b < 256; ++b) {
        const bool visible = (b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174 && b <= 255);
        out[static_cast<std::size_t>(b)] = utf8(visible ? static_cast<std::uint32_t>(b) : next++);
    }
    return out;
}

Json inkling2_pipeline() {
    return Json{{"type", "Sequence"},
                {"pretokenizers",
                 Json::array({Json{{"type", "Split"}, {"pattern", {{"Regex", golden().at("stages").at(0)}}},
                                   {"behavior", "Isolated"}, {"invert", false}},
                              Json{{"type", "ByteLevel"}, {"add_prefix_space", false}, {"trim_offsets", true}, {"use_regex", false}}})}};
}

struct Built {
    std::string tokenizer_json, config_json, generation_json;
};

// The 256 byte tokens, then "ab" (258) and "abc" (259) with merges "a b" and "b c" (no "ab c"): merging "abc"
// gives a, bc; ignore_merges finds "abc" whole. Added tokens <|end|> and <|tool|> after them.
Built build(const Json& pipeline, const Json& normalizer, bool ignore_merges = true) {
    const auto alphabet = byte_alphabet();
    Json vocab = Json::object();
    for (int b = 0; b < 256; ++b) { vocab[alphabet[static_cast<std::size_t>(b)]] = b; }
    vocab["ab"] = 256;
    vocab["bc"] = 257;
    vocab["abc"] = 258;
    const auto added = [](int id, const char* content, bool special) {
        return Json{{"id", id}, {"content", content}, {"single_word", false}, {"lstrip", false},
                    {"rstrip", false}, {"normalized", false}, {"special", special}};
    };
    Json added_tokens = Json::array({added(259, "<|end|>", true), added(260, "<|tool|>", true)});
    Json root = {{"version", "1.0"}, {"added_tokens", added_tokens}, {"normalizer", normalizer}, {"pre_tokenizer", pipeline},
                 {"post_processor", {{"type", "ByteLevel"}, {"add_prefix_space", true}, {"trim_offsets", false}, {"use_regex", true}}},
                 {"decoder", {{"type", "ByteLevel"}, {"add_prefix_space", true}, {"trim_offsets", true}, {"use_regex", true}}},
                 {"model", {{"type", "BPE"}, {"dropout", nullptr}, {"ignore_merges", ignore_merges}, {"byte_fallback", false},
                            {"vocab", vocab}, {"merges", Json::array({"b c", "a b"})}}}};
    Json decoder = Json::object();
    for (const Json& a : added_tokens) {
        decoder[std::to_string(a.at("id").get<int>())] = Json{{"content", a.at("content")}, {"single_word", false}, {"lstrip", false},
                                                               {"rstrip", false}, {"normalized", false}, {"special", true}};
    }
    return {root.dump(), Json{{"added_tokens_decoder", decoder}}.dump(), Json{{"eos_token_id", 259}}.dump()};
}

Tokenizer make(const Built& b) { return Tokenizer(TokenizerResources{b.tokenizer_json, b.config_json, b.generation_json}); }

bool refuses(const Built& b) {
    try {
        (void)make(b);
    } catch (const std::invalid_argument&) { return true; }
    return false;
}

void an_inkling2_tokenizer_loads_and_ignores_merges() {
    const Tokenizer tokenizer = make(build(inkling2_pipeline(), nullptr));
    expect(tokenizer.pipeline() == Pipeline::Inkling2, "the pipeline is recognised as Inkling-2's");
    expect(tokenizer.encode("abc") == std::vector<int>{258}, "ignore_merges: \"abc\" is its vocabulary token");
    expect(tokenizer.encode("abcb") == std::vector<int>{'a', 257, 'b'}, "a word not in the vocabulary is merged (b c first)");
    expect(tokenizer.encode(std::string("e\xcc\x81")).size() == 3, "no normalizer: a decomposed letter stays 3 bytes");
    const auto ids = tokenizer.encode("abc<|end|>ab<|tool|>");
    expect(ids == std::vector<int>{258, 259, 256, 260}, "added tokens are matched between words");
    expect(tokenizer.decode(ids) == "abc<|end|>ab<|tool|>", "decode round-trips");
    expect(tokenizer.default_stop_token_ids() == std::vector<int>{259}, "stop token from generation_config");
}

void other_descriptions_are_refused() {
    expect(refuses(build(inkling2_pipeline(), Json{{"type", "NFC"}})), "a normalizer with the Inkling-2 pre-tokenizer is refused");
    expect(refuses(build(inkling2_pipeline(), nullptr, false)), "the Inkling-2 pre-tokenizer without ignore_merges is refused");
    Json invert = inkling2_pipeline();
    invert["pretokenizers"][0]["invert"] = true;
    expect(refuses(build(invert, nullptr)), "an inverted split is refused");
    Json behaviour = inkling2_pipeline();
    behaviour["pretokenizers"][0]["behavior"] = "Removed";
    expect(refuses(build(behaviour, nullptr)), "another split behaviour is refused");
    Json prefix = inkling2_pipeline();
    prefix["pretokenizers"][1]["add_prefix_space"] = true;
    expect(refuses(build(prefix, nullptr)), "a byte level that adds a prefix space is refused");
    Json pattern = inkling2_pipeline();
    std::string regex = pattern["pretokenizers"][0]["pattern"]["Regex"];
    regex.replace(regex.find("{1,3}"), 5, "{1,2}");
    pattern["pretokenizers"][0]["pattern"]["Regex"] = regex;
    expect(refuses(build(pattern, nullptr)), "a different word regex is refused (ignore_merges is Inkling-2's only)");
}

std::string read_text(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) { throw std::runtime_error("cannot open " + path); }
    std::stringstream raw;
    raw << in.rdbuf();
    return raw.str();
}

// The whole encoder against the library, on the real tokenizer files (not in the repository): set
// NINFER_INKLING2_TOKENIZER_DIR to a directory with the checkpoint's tokenizer.json, tokenizer_config.json and
// generation_config.json. The expected ids are tests/fixtures/inkling2/tokenizer_encode_golden.json's.
void real_tokenizer_encodes_like_hugging_face() {
    const char* dir = std::getenv("NINFER_INKLING2_TOKENIZER_DIR");
    if (dir == nullptr || *dir == '\0') {
        std::cout << "inkling2_tokenizer: NINFER_INKLING2_TOKENIZER_DIR not set; the real-tokenizer comparison is skipped\n";
        return;
    }
    const std::string root = dir;
    const Tokenizer tokenizer(TokenizerResources{read_text(root + "/tokenizer.json"), read_text(root + "/tokenizer_config.json"),
                                                 read_text(root + "/generation_config.json")});
    expect(tokenizer.pipeline() == Pipeline::Inkling2, "the real tokenizer.json is the Inkling-2 pipeline");
    expect(tokenizer.vocab_size() == 200058 && tokenizer.has_exact_token_domain(200058), "200,058 tokens, contiguous");
    const Json encode = read_json(NINFER_SOURCE_DIR "/tests/fixtures/inkling2/tokenizer_encode_golden.json");
    std::size_t checked = 0;
    for (const Json& item : encode.at("cases")) {
        const std::string text = item.at(0).get<std::string>();
        const auto want        = item.at(1).get<std::vector<int>>();
        const auto got         = tokenizer.encode(text);
        if (got != want) {
            expect(false, "encode differs from Hugging Face on \"" + text.substr(0, 60) + "\" (" + std::to_string(got.size()) +
                              " ids vs " + std::to_string(want.size()) + ")");
        }
        expect(tokenizer.decode(got) == text, "decode does not give back \"" + text.substr(0, 60) + "\"");
        ++checked;
    }
    expect(checked == encode.at("count").get<std::size_t>(), "every golden case ran");
    expect(tokenizer.default_stop_token_ids() == std::vector<int>{200006}, "stop at <|content_model_end_sampling|>");
    expect(tokenizer.is_special_token(200006) && !tokenizer.is_special_token(1000), "special flags come from tokenizer.json");
}

} // namespace

int main() {
    try {
        pre_tokenizer_matches_hugging_face();
        long_text_is_pre_tokenized_in_linear_time();
        an_inkling2_tokenizer_loads_and_ignores_merges();
        other_descriptions_are_refused();
        real_tokenizer_encodes_like_hugging_face();
    } catch (const std::exception& error) {
        std::cerr << "FAILED: unexpected exception: " << error.what() << '\n';
        ++g_failures;
    }
    if (g_failures == 0) { std::cout << "inkling2_tokenizer: passed\n"; }
    return g_failures == 0 ? 0 : 1;
}
