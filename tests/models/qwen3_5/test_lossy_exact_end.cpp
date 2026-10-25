// Lossy prefill's M (the end of the product prompt) for real protocol traffic (CPU only).
//
// A request goes through the real OpenAI chat or Anthropic Messages translator with its ordinary
// cache policy, then through the Frontend with the maintained Qwen3.6 chat template and a
// byte-level tokenizer. M must be the end of the system message and tools, never the boundary
// those protocols place on the latest message: with M there, U is a few tokens and the trigger can
// never fire (docs/maintainer/lossy-long-prefill.md, "As built").
#include "models/qwen3_5/frontend/frontend.h"
#include "models/qwen3_5/frontend/prepared_prompt.h"
#include "models/qwen3_5/frontend/resources.h"
#include "models/qwen3_5/frontend/test_access.h"
#include "models/qwen3_5/frontend/tokenizer.h"
#include "models/qwen3_5/program/lossy/lossy_flow.h"
#include "serve/anthropic_messages.h"
#include "serve/openai_chat.h"
#include "serve/translate.h"
#include "text/unicode.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

namespace q = ninfer::models::qwen3_5;
using Json = ninfer::serve::RequestJson;

constexpr ninfer::TokenId kImStart = 248045;
constexpr ninfer::TokenId kImEnd   = 248046;

void require(bool ok, const std::string& what) {
    if (!ok) { throw std::runtime_error("FAILED: " + what); }
}

std::string read_file(const char* path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) { throw std::runtime_error(std::string("cannot open ") + path); }
    return {std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>()};
}

std::string byte_level_symbol(std::uint8_t target) {
    std::uint32_t next = 256;
    for (int value = 0; value <= 255; ++value) {
        const bool visible = (value >= 33 && value <= 126) || (value >= 161 && value <= 172) ||
                             (value >= 174 && value <= 255);
        const std::uint32_t codepoint = visible ? static_cast<std::uint32_t>(value) : next++;
        if (value == target) {
            return ninfer::text::unicode_internal::codepoint_to_utf8(
                static_cast<std::int32_t>(codepoint));
        }
    }
    throw std::logic_error("byte symbol");
}

nlohmann::json added(int id, std::string content, bool special) {
    return {{"id", id},          {"content", std::move(content)}, {"single_word", false},
            {"lstrip", false},   {"rstrip", false},               {"normalized", false},
            {"special", special}};
}

// One token per byte (ids 1000 + byte) and the template's added tokens: enough to count frontiers.
q::Frontend make_frontend() {
    const std::string chat_template =
        read_file(NINFER_SOURCE_DIR "/tools/chat_templates/qwen3_6.jinja");
    const nlohmann::json tokens = nlohmann::json::array(
        {added(6, "<eos>", true), added(kImStart, "<|im_start|>", true),
         added(kImEnd, "<|im_end|>", true), added(248053, "<|vision_start|>", true),
         added(248054, "<|vision_end|>", true), added(248056, "<|image_pad|>", true),
         added(248057, "<|video_pad|>", true), added(248068, "<think>", false),
         added(248069, "</think>", false)});
    nlohmann::json vocab = nlohmann::json::object();
    for (int value = 0; value <= 255; ++value) {
        vocab[byte_level_symbol(static_cast<std::uint8_t>(value))] = 1000 + value;
    }
    const std::string tokenizer_json =
        nlohmann::json{{"model",
                        {{"type", "BPE"}, {"vocab", vocab}, {"merges", nlohmann::json::array()}}},
                       {"added_tokens", tokens}}
            .dump();
    nlohmann::json decoder = nlohmann::json::object();
    for (nlohmann::json token : tokens) {
        const std::string id = std::to_string(token.at("id").get<int>());
        token.erase("id");
        decoder[id] = std::move(token);
    }
    const std::string tokenizer_config_json =
        nlohmann::json{{"add_bos_token", false},
                       {"add_prefix_space", false},
                       {"pad_token", "<|endoftext|>"},
                       {"chat_template", chat_template},
                       {"added_tokens_decoder", decoder}}
            .dump();
    const std::string generation_config_json = R"({"eos_token_id":[6]})";
    q::FrontendResources parsed{tokenizer_json, tokenizer_config_json, chat_template,
                                generation_config_json, "", ""};
    parsed.tokenizer = std::make_shared<const q::frontend::Tokenizer>(q::frontend::TokenizerResources{
        tokenizer_json, tokenizer_config_json, generation_config_json});
    parsed.public_token_count = static_cast<std::uint32_t>(parsed.tokenizer->vocab_size());
    q::FrontendOptions options;
    options.vision_enabled = false;
    options.max_context    = std::numeric_limits<std::uint32_t>::max();
    return q::make_frontend(parsed, options);
}

ninfer::serve::RequestLimits limits() {
    ninfer::serve::RequestLimits value;
    value.default_max_tokens = 1024;
    return value;
}

ninfer::PromptInput prompt_input(const ninfer::serve::GenerationRequest& request) {
    ninfer::serve::ServeOptions server;
    const auto semantics = ninfer::serve::resolve_prompt_semantics(request, server);
    return ninfer::serve::to_prompt_input(request, semantics, {});
}

const Json kTool = Json{{"type", "function"},
                        {"function",
                         {{"name", "read_file"},
                          {"description", "Read a file"},
                          {"parameters",
                           {{"type", "object"},
                            {"properties", {{"path", {{"type", "string"}}}}},
                            {"required", Json::array({"path"})}}}}}};

const std::string kSystem = "You are a careful coding agent. Follow the repository rules.";
const std::string kLong   = std::string(4000, 'r') + " what is in the log?";

struct Measured {
    std::uint32_t exact_end    = 0;
    std::uint32_t tokens       = 0;
    std::uint32_t deepest_mark = 0; // the deepest shared opportunity, the old rule's M
    std::vector<ninfer::TokenId> ids;
};

Measured measure(const q::Frontend& frontend, ninfer::PromptInput input) {
    const q::PreparedPrompt prompt   = frontend.prepare(std::move(input));
    const q::PreparedPromptData& data = q::FrontendTestAccess::inspect(prompt);
    Measured out;
    out.exact_end = q::detail::lossy_planned_exact_end(data.context_cache);
    out.tokens    = static_cast<std::uint32_t>(data.token_ids.size());
    for (const auto& opportunity : data.context_cache.opportunities) {
        if (opportunity.kind == ninfer::PromptCacheMarkerKind::SharedStablePrefix) {
            out.deepest_mark = std::max(out.deepest_mark, opportunity.frontier);
        }
    }
    out.ids = data.token_ids;
    return out;
}

// The independent oracle for "system message and tools": M is where the second turn (the first
// non-system one) opens, and exactly one turn opens before it.
void require_system_boundary(const Measured& m, const std::string& what) {
    require(m.exact_end > 0 && m.exact_end < m.tokens, what + ": M inside the prompt");
    require(m.ids[m.exact_end] == kImStart, what + ": M is where the user turn opens");
    require(std::count(m.ids.begin(), m.ids.begin() + m.exact_end, kImStart) == 1,
            what + ": only the system turn precedes M");
    require(m.tokens - m.exact_end > 4000, what + ": U holds the long user message");
}

void test_openai_default_policy(const q::Frontend& frontend) {
    const Json body{{"model", "m"},
                    {"tools", Json::array({kTool})},
                    {"messages", Json::array({Json{{"role", "system"}, {"content", kSystem}},
                                              Json{{"role", "user"}, {"content", kLong}}})}};
    const auto request = ninfer::serve::parse_chat_completion_request(body, limits());
    const Measured m   = measure(frontend, prompt_input(request.generation));
    // The default policy's automatic boundary closes the last message: the old rule took it as M.
    require(m.deepest_mark > m.exact_end + 4000, "openai: an automatic boundary ends the prompt");
    require_system_boundary(m, "openai default policy");
}

void test_anthropic_cache_control_on_last_message(const q::Frontend& frontend) {
    const Json anthropic_tool{{"name", "read_file"},
                              {"description", "Read a file"},
                              {"input_schema",
                               {{"type", "object"},
                                {"properties", {{"path", {{"type", "string"}}}}}}}};
    const Json body{
        {"model", "m"},
        {"max_tokens", 256},
        {"system", kSystem},
        {"tools", Json::array({anthropic_tool})},
        {"messages",
         Json::array({Json{{"role", "user"},
                           {"content", Json::array({Json{{"type", "text"},
                                                         {"text", kLong},
                                                         {"cache_control",
                                                          {{"type", "ephemeral"}}}}})}}})}};
    const auto request = ninfer::serve::parse_anthropic_messages_request(body, limits());
    const Measured m   = measure(frontend, prompt_input(request.generation));
    require(m.deepest_mark > m.exact_end + 4000, "anthropic: cache_control marks the last message");
    require_system_boundary(m, "anthropic cache_control on the last message");

    // Tools and no system prompt: M is the end of the tool declarations, inside the system turn the
    // template opens for them.
    Json tools_only = body;
    tools_only.erase("system");
    const auto tools_request = ninfer::serve::parse_anthropic_messages_request(tools_only, limits());
    const Measured t         = measure(frontend, prompt_input(tools_request.generation));
    const auto first_end     = std::find(t.ids.begin(), t.ids.end(), kImEnd) - t.ids.begin();
    require(t.exact_end > 0 && t.exact_end <= static_cast<std::uint32_t>(first_end),
            "anthropic tools only: M inside the tool turn");
    require(t.tokens - t.exact_end > 4000, "anthropic tools only: U holds the user message");
}

void test_no_product_prompt(const q::Frontend& frontend) {
    const Json body{{"model", "m"},
                    {"messages", Json::array({Json{{"role", "user"}, {"content", kLong}}})}};
    const auto request = ninfer::serve::parse_chat_completion_request(body, limits());
    const Measured m   = measure(frontend, prompt_input(request.generation));
    require(m.exact_end == 0, "no system message or tools: M is 0");
}

} // namespace

int main() {
    try {
        const q::Frontend frontend = make_frontend();
        test_openai_default_policy(frontend);
        test_anthropic_cache_control_on_last_message(frontend);
        test_no_product_prompt(frontend);
        std::cout << "OK lossy exact end\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}
