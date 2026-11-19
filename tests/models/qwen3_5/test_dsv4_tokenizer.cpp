// The DeepSeek-V4 tokenizer pipeline: its pre-tokenizer (three Splits then byte level, no normalizer) cuts text
// exactly where the Hugging Face tokenizers library does, a tokenizer.json that describes it loads, and one
// that describes something else is refused.

#include "models/qwen3_5/frontend/tokenizer.h"

#include <nlohmann/json.hpp>

#include <array>
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

// The golden: the Hugging Face tokenizers library applying the real tokenizer.json's three Split stages
// (tools/dsv4: tests/fixtures/dsv4/pretokenizer_golden.json). Each case is [text, [pieces...]].
void pre_tokenizer_matches_hugging_face() {
    const Json golden = read_json(NINFER_SOURCE_DIR "/tests/fixtures/dsv4/pretokenizer_golden.json");
    std::size_t checked = 0;
    for (const Json& item : golden.at("cases")) {
        const std::string text = item.at(0).get<std::string>();
        std::vector<std::size_t> want;
        std::size_t at = 0;
        for (const Json& piece : item.at(1)) {
            at += piece.get<std::string>().size();
            want.push_back(at);
        }
        const auto got = ninfer::models::qwen3_5::frontend::pre_tokenize_ends(Pipeline::Dsv4, text);
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
    expect(checked > 1000, "the golden has over a thousand cases");
}

// ByteLevel's alphabet (GPT-2's bytes_to_unicode), to write a vocabulary of the 256 single bytes.
std::array<std::string, 256> byte_alphabet() {
    std::array<std::string, 256> out;
    std::uint32_t next = 256;
    const auto utf8 = [](std::uint32_t cp) {
        std::string s;
        if (cp < 0x80) {
            s += static_cast<char>(cp);
        } else if (cp < 0x800) {
            s += static_cast<char>(0xc0 | (cp >> 6));
            s += static_cast<char>(0x80 | (cp & 0x3f));
        } else {
            s += static_cast<char>(0xe0 | (cp >> 12));
            s += static_cast<char>(0x80 | ((cp >> 6) & 0x3f));
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

Json dsv4_pipeline(const Json& golden) {
    Json stages = Json::array();
    for (int i = 0; i < 3; ++i) {
        stages.push_back(Json{{"type", "Split"}, {"pattern", {{"Regex", golden.at("stages").at(i)}}},
                              {"behavior", "Isolated"}, {"invert", false}});
    }
    stages.push_back(Json{{"type", "ByteLevel"}, {"add_prefix_space", false}, {"trim_offsets", true}, {"use_regex", false}});
    return Json{{"type", "Sequence"}, {"pretokenizers", stages}};
}

struct Built {
    std::string tokenizer_json, config_json, generation_json;
};

// A tokenizer of the 256 byte tokens (no merges), BOS/EOS/pad also listed in the vocabulary as the real file does,
// and two more added tokens, one flagged normalized (the real file has 47 of those; with no normalizer it is moot).
Built build(const Json& pipeline, const Json& normalizer, bool normalized_added = true) {
    const auto alphabet = byte_alphabet();
    Json vocab = Json::object();
    vocab["<bos>"] = 0;
    vocab["<eos>"] = 1;
    vocab["<pad>"] = 2;
    for (int b = 0; b < 256; ++b) { vocab[alphabet[static_cast<std::size_t>(b)]] = 3 + b; }
    const auto added = [](int id, const char* content, bool special, bool normalized) {
        return Json{{"id", id}, {"content", content}, {"single_word", false}, {"lstrip", false},
                    {"rstrip", false}, {"normalized", normalized}, {"special", special}};
    };
    Json added_tokens = Json::array({added(0, "<bos>", true, false), added(1, "<eos>", true, false), added(2, "<pad>", true, false),
                                     added(259, "<tool>", false, normalized_added), added(260, "<end>", true, false)});
    Json root = {{"version", "1.0"}, {"added_tokens", added_tokens}, {"pre_tokenizer", pipeline},
                 {"post_processor", {{"type", "ByteLevel"}, {"add_prefix_space", true}, {"trim_offsets", false}, {"use_regex", true}}},
                 {"decoder", {{"type", "ByteLevel"}, {"add_prefix_space", true}, {"trim_offsets", true}, {"use_regex", true}}},
                 {"model", {{"type", "BPE"}, {"dropout", nullptr}, {"vocab", vocab}, {"merges", Json::array()}}}};
    if (!normalizer.is_null()) { root["normalizer"] = normalizer; }
    Json decoder = Json::object();
    for (const Json& a : added_tokens) { decoder[std::to_string(a.at("id").get<int>())] = Json{{"content", a.at("content")}, {"single_word", false}, {"lstrip", false}, {"rstrip", false}, {"normalized", a.at("normalized")}, {"special", a.at("special")}}; }
    Json config = {{"added_tokens_decoder", decoder}};
    Json generation = {{"eos_token_id", 1}};
    return {root.dump(), config.dump(), generation.dump()};
}

Tokenizer make(const Built& b) { return Tokenizer(TokenizerResources{b.tokenizer_json, b.config_json, b.generation_json}); }

bool refuses(const Built& b) {
    try {
        (void)make(b);
    } catch (const std::invalid_argument&) { return true; }
    return false;
}

void a_dsv4_tokenizer_loads_and_encodes_without_normalizing() {
    const Json golden = read_json(NINFER_SOURCE_DIR "/tests/fixtures/dsv4/pretokenizer_golden.json");
    const Json empty_sequence = {{"type", "Sequence"}, {"normalizers", Json::array()}};
    for (const Json& normalizer : {Json(nullptr), empty_sequence}) {
        const Tokenizer tokenizer = make(build(dsv4_pipeline(golden), normalizer));
        expect(tokenizer.pipeline() == Pipeline::Dsv4, "the pipeline is recognised as DeepSeek-V4's");
        // 'e' + U+0301 stays two code points (NFC would compose them): bytes 65, cc, 81 -> three tokens.
        const auto ids = tokenizer.encode(std::string("e\xcc\x81"));
        expect(ids.size() == 3, "no normalizer: a decomposed letter is not composed (3 byte tokens, not 2)");
        // Added tokens are found, including the one flagged normalized, and the control tokens listed in the vocab too.
        const auto with_added = tokenizer.encode("a<tool>b<bos>");
        expect(with_added.size() == 4 && with_added[1] == 259 && with_added[3] == 0, "added tokens are matched");
        expect(tokenizer.is_special_token(0) && !tokenizer.is_special_token(259), "special flags are kept");
        expect(tokenizer.decode(with_added) == "a<tool>b<bos>", "decode round-trips");
    }
}

void other_descriptions_are_refused() {
    const Json golden = read_json(NINFER_SOURCE_DIR "/tests/fixtures/dsv4/pretokenizer_golden.json");
    const Json nfc = {{"type", "NFC"}};
    expect(refuses(build(dsv4_pipeline(golden), nfc)), "a normalizer with the V4 pre-tokenizer is refused");
    Json wrong = dsv4_pipeline(golden);
    wrong["pretokenizers"][0]["pattern"]["Regex"] = "\\p{N}{1,2}";
    expect(refuses(build(wrong, nullptr)), "a different digit split is refused");
    Json swapped = dsv4_pipeline(golden);
    std::swap(swapped["pretokenizers"][1], swapped["pretokenizers"][2]);
    expect(refuses(build(swapped, nullptr)), "the stages in another order are refused");
    Json invert = dsv4_pipeline(golden);
    invert["pretokenizers"][2]["invert"] = true;
    expect(refuses(build(invert, nullptr)), "an inverted split is refused");
    Json behaviour = dsv4_pipeline(golden);
    behaviour["pretokenizers"][0]["behavior"] = "Removed";
    expect(refuses(build(behaviour, nullptr)), "another split behaviour is refused");
    Json three = dsv4_pipeline(golden);
    three["pretokenizers"].erase(1);
    expect(refuses(build(three, nullptr)), "a pipeline of the wrong length is refused");
}

// The whole encoder against the library, on the real tokenizer.json (6 MB, not in the repository): set
// NINFER_DSV4_TOKENIZER_JSON to a copy of the checkpoint's file to run it. The expected ids are the Hugging Face
// tokenizers library's (tests/fixtures/dsv4/tokenizer_encode_golden.json); decoding must give the text back.
void real_tokenizer_encodes_like_hugging_face() {
    const char* path = std::getenv("NINFER_DSV4_TOKENIZER_JSON");
    if (path == nullptr || *path == '\0') {
        std::cout << "dsv4_tokenizer: NINFER_DSV4_TOKENIZER_JSON not set; the real-tokenizer comparison is skipped\n";
        return;
    }
    std::ifstream in(path, std::ios::binary);
    std::stringstream raw;
    raw << in.rdbuf();
    const Json root = Json::parse(raw.str());
    Json decoder = Json::object();
    for (const Json& a : root.at("added_tokens")) {
        decoder[std::to_string(a.at("id").get<int>())] = Json{{"content", a.at("content")}, {"single_word", false},
                                                              {"lstrip", false}, {"rstrip", false},
                                                              {"normalized", a.at("normalized")}, {"special", a.at("special")}};
    }
    const std::string config     = Json{{"added_tokens_decoder", decoder}}.dump();
    const std::string generation = Json{{"eos_token_id", 1}}.dump();
    const std::string tokenizer_json = raw.str();
    const Tokenizer tokenizer(TokenizerResources{tokenizer_json, config, generation});
    expect(tokenizer.pipeline() == Pipeline::Dsv4, "the real tokenizer.json is the V4 pipeline");
    expect(tokenizer.vocab_size() == 129280 && tokenizer.has_exact_token_domain(129280), "129,280 tokens, contiguous");
    std::size_t checked = 0;
    const Json golden = read_json(NINFER_SOURCE_DIR "/tests/fixtures/dsv4/tokenizer_encode_golden.json");
    for (const Json& item : golden.at("cases")) {
        const std::string text = item.at(0).get<std::string>();
        const auto want = item.at(1).get<std::vector<int>>();
        const auto got  = tokenizer.encode(text);
        if (got != want) {
            expect(false, "encode differs from Hugging Face on \"" + text.substr(0, 60) + "\" (" + std::to_string(got.size()) +
                              " ids vs " + std::to_string(want.size()) + ")");
        }
        expect(tokenizer.decode(got) == text, "decode does not give back \"" + text.substr(0, 60) + "\"");
        ++checked;
    }
    expect(checked == golden.at("count").get<std::size_t>(), "every golden case ran");
    // The stop token and the special flags of the real file: <｜end▁of▁sentence｜> stops, the tool-call markers are visible.
    expect(tokenizer.default_stop_token_ids() == std::vector<int>{1}, "stop at the end-of-sentence token");
    expect(tokenizer.is_special_token(1) && !tokenizer.is_special_token(128799), "special flags come from tokenizer.json");
}

} // namespace

int main() {
    try {
        pre_tokenizer_matches_hugging_face();
        a_dsv4_tokenizer_loads_and_encodes_without_normalizing();
        other_descriptions_are_refused();
        real_tokenizer_encodes_like_hugging_face();
    } catch (const std::exception& error) {
        std::cerr << "FAILED: unexpected exception: " << error.what() << '\n';
        ++g_failures;
    }
    if (g_failures == 0) { std::cout << "dsv4_tokenizer: passed\n"; }
    return g_failures == 0 ? 0 : 1;
}
