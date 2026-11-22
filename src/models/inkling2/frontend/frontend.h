#pragma once

// Written for NInfer-3090. Inkling-2-Mini's frontend (docs/maintainer/inkling2-model.md, "Serving"): a conversation
// to prompt tokens through the artifact's own chat template and tokenizer, images to prepared pixels, and the
// model's output tokens back to reasoning, content and tool calls.
//
// Inkling-2 marks every channel with a control token: a model turn is `<|message_model|>`, then an optional name
// (a tool's, for a call), then `<|content_thinking|>`, `<|content_text|>` or `<|content_invoke_tool_json|>`, the
// body, and `<|end_message|>`; `<|content_model_end_sampling|>` ends the turn. The template renders no generation
// prompt: the model writes its own `<|message_model|>`. An image is `<|content_image|><|unused_200054|>` in the
// template; the placeholder becomes one token per merged 2 x 2 patches, whose rows the vision tower supplies.

#include "models/inkling2/program/vision_program.h"
#include "models/qwen3_5/frontend/tokenizer.h"
#include "ninfer/types.h"
#include "text/jinja.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace ninfer::models::inkling2 {

// The control tokens the frontend reads (tokenizer_config.json's added_tokens_decoder; checked at load).
namespace token {
inline constexpr TokenId kMessageUser       = 200000;
inline constexpr TokenId kMessageModel      = 200001;
inline constexpr TokenId kContentText       = 200004;
inline constexpr TokenId kContentImage      = 200005;
inline constexpr TokenId kEndSampling       = 200006; // <|content_model_end_sampling|>: the turn's end
inline constexpr TokenId kContentThinking   = 200008;
inline constexpr TokenId kEndMessage        = 200010;
inline constexpr TokenId kContentToolError  = 200022;
inline constexpr TokenId kInvokeToolJson    = 200049;
inline constexpr TokenId kImagePlaceholder  = 200054; // <|unused_200054|>
inline constexpr TokenId kInvokeToolText    = 200057;
} // namespace token

struct PromptImage {
    std::int32_t column = 0; // the prompt position of its first image token
    PreparedImage image;
};

// A tool the prompt declares: its name and its parameters' JSON Schema ("{}" when it gives none).
struct ToolDeclaration {
    std::string name;
    std::string parameters_json = "{}";
};

struct PreparedPrompt {
    std::vector<std::int32_t> tokens;
    // The declared tools; nullopt for a token prompt (prepare_tokens), whose tools are unknown.
    std::optional<std::vector<ToolDeclaration>> tools;
    std::vector<PromptImage> images; // ascending columns
    PromptSummary summary;
    PromptPreparationStats stats;
    bool thinking = true; // the request's sampling mode
    // The adapters the request selected (program/adapters.h): indexes into FrontendOptions::lora_names and
    // graft_names, or -1 for none (the base model).
    std::int32_t lora = -1, graft = -1;
};

struct FrontendOptions {
    std::uint32_t max_context = 8192;
    // The tower's config when the artifact has one; images are refused otherwise.
    const VisionConfig* vision = nullptr;
    // The loaded adapters' names (the Engine's --lora and --graft entries, in order), which a request's
    // PromptOptions::lora and ::graft select from.
    std::vector<std::string> lora_names, graft_names;
};

class Frontend {
public:
    Frontend(const Model::Resources& resources, FrontendOptions options);

    // Renders and tokenizes the conversation. Throws RequestError for what the request asks and this model cannot
    // do (context length, media, an unknown reasoning effort, continuing an assistant turn).
    [[nodiscard]] PreparedPrompt prepare(PromptInput input, const PreparationControl& control = {}) const;
    [[nodiscard]] PreparedPrompt prepare_tokens(std::vector<TokenId> tokens) const;
    // The prompt's length as prepare() expands it (images included), also when it leaves no room for a reply.
    [[nodiscard]] std::uint32_t count_tokens(PromptInput input, const PreparationControl& control = {}) const;
    [[nodiscard]] bool is_valid_token(TokenId id) const;
    [[nodiscard]] std::vector<TokenId> tokenize_text(std::string_view text) const;
    [[nodiscard]] std::string_view token_bytes(TokenId id) const;
    [[nodiscard]] bool is_special(TokenId id) const noexcept;
    [[nodiscard]] const ModelSamplingDefaults& sampling_defaults() const noexcept { return sampling_; }
    [[nodiscard]] std::uint32_t max_context() const noexcept { return options_.max_context; }
    // Lowers the context a prompt is admitted against (the Engine fitted a smaller one per request slot at startup,
    // before any prompt is prepared).
    void set_max_context(std::uint32_t max_context) noexcept { options_.max_context = max_context; }

    // The rendered prompt text, before tokenization (for tests).
    [[nodiscard]] text::TemplateOutput render(const PromptInput& input, const PreparationControl& control = {}) const;

private:
    // prepare() with `admit` (the prompt leaves room for a reply), count_tokens() without.
    [[nodiscard]] PreparedPrompt build(PromptInput input, const PreparationControl& control, bool admit) const;

    std::shared_ptr<const qwen3_5::frontend::Tokenizer> tokenizer_;
    std::shared_ptr<const text::JinjaTemplate> template_;
    FrontendOptions options_;
    ModelSamplingDefaults sampling_;
};

// One request's output: control tokens switch the channel, the rest is text in it. The text is valid UTF-8 (a
// malformed sequence becomes U+FFFD) and published in whole characters, held back while it could still become a
// stop string.
class OutputDecoder {
public:
    // `tools` false (tool_choice none): a tool call the model writes stays content text. `declared`: the prompt's
    // tools; a call to a name it lacks stays content text (fallback UndeclaredTool), and arguments that break the
    // tool's parameter schema (an unknown or missing required property, a wrong type or enum value) are counted in
    // tool_call_parse().schema_mismatch_arguments. nullopt checks nothing (the tools are unknown).
    OutputDecoder(const Frontend& frontend, std::vector<StopString> stops, bool tools = true,
                  std::optional<std::vector<ToolDeclaration>> declared = std::nullopt);

    struct Step {
        std::vector<OutputDelta> deltas;
        bool stop = false; // a stop string matched: the request ends here
    };
    // One generated token (not a stop token).
    Step push(TokenId id);
    // The request ended: the held text, and any tool call left open.
    std::vector<OutputDelta> finish();

    [[nodiscard]] bool in_reasoning() const noexcept { return channel_ == Channel::Reasoning; }
    [[nodiscard]] std::uint32_t reasoning_tokens() const noexcept { return reasoning_tokens_; }
    [[nodiscard]] const std::string& content() const noexcept { return text_[0]; }
    [[nodiscard]] const std::string& reasoning() const noexcept { return text_[1]; }
    [[nodiscard]] const std::vector<GeneratedToolCall>& tool_calls() const noexcept { return calls_; }
    [[nodiscard]] const ToolCallParseDiagnostics& tool_call_parse() const noexcept { return parse_; }
    [[nodiscard]] const std::optional<std::string>& matched_stop() const noexcept { return matched_; }

private:
    enum class Channel : std::uint8_t { Content, Reasoning, ToolJson, Header, None };
    // Every byte of text goes through here, so a stop string is matched wherever it appears; `final` ends any
    // open character.
    void append(std::size_t channel, std::string_view bytes, Step& step, bool final = false);
    void publish(std::size_t channel, bool all, std::vector<OutputDelta>& out);
    void close_tool_call(Step& step);

    const Frontend& frontend_;
    std::vector<StopString> stops_;
    Channel channel_ = Channel::None;
    bool tools_ = true;
    std::optional<std::vector<ToolDeclaration>> declared_;
    std::string text_[2];          // content, reasoning: valid UTF-8
    std::string raw_[2];           // their last bytes that do not yet make a whole character
    std::size_t published_[2] = {0, 0};
    // Where a stop string may start: a channel the output left was published whole, so no match reaches back past it.
    std::size_t floor_[2] = {0, 0};
    std::string tool_json_;
    std::vector<GeneratedToolCall> calls_;
    ToolCallParseDiagnostics parse_;
    std::uint32_t reasoning_tokens_ = 0;
    std::optional<std::string> matched_;
};

} // namespace ninfer::models::inkling2
