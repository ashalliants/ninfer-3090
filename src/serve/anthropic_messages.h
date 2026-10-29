#pragma once

// Anthropic Messages wire adapter. Request parsing lowers executable semantics into the common
// GenerationRequest; response construction owns Anthropic aggregate, SSE, usage, and error shapes.
// Visible Thinking text owns prompt semantics. Its wire signature is response metadata and never
// enters common request or Engine state. The one exception is a replayed empty Thinking block from a
// thinking.display="omitted" response: its signature is looked up to restore the hidden reasoning
// (see AnthropicReasoningStore), so the prompt matches the tokens the Engine already cached.

#include "serve/request.h"
#include "serve/request_json.h"

#include <functional>
#include <optional>
#include <string>
#include <vector>

namespace ninfer::serve {

struct GenerationOutcome;

// Returns the reasoning hidden behind a Thinking signature, or nothing when it is unknown.
using ReasoningRestore = std::function<std::optional<std::string>(const std::string& signature)>;

struct AnthropicMessagesRequest {
    std::string model;
    GenerationRequest generation;
    bool stream                 = false;
    bool output_tokens_explicit = false;
    // thinking.display="omitted": responses carry empty Thinking text plus a restorable signature.
    bool thinking_display_omitted = false;
};

struct AnthropicCountTokensRequest {
    std::string model;
    GenerationRequest generation;
};

AnthropicMessagesRequest parse_anthropic_messages_request(const RequestJson& body,
                                                          const RequestLimits& limits,
                                                          const ReasoningRestore& restore = {});
AnthropicCountTokensRequest parse_anthropic_count_tokens_request(
    const RequestJson& body, const ReasoningRestore& restore = {});

struct AnthropicResponseIdentity {
    std::string request_id;
    std::string message_id;
    std::string model;
};

std::string new_anthropic_request_id();
AnthropicResponseIdentity make_anthropic_response_identity(std::string request_id,
                                                           std::string model);

ApiError normalize_anthropic_error(ApiError error);
std::string make_anthropic_error_body(const ApiError& error, const std::string& request_id);
std::string make_anthropic_sse_error(const ApiError& error, const std::string& request_id);

// `hide_thinking` is thinking.display="omitted": Thinking text is withheld, the signature is kept.
std::string make_anthropic_messages_response(const AnthropicResponseIdentity& identity,
                                             const GenerationOutcome& outcome,
                                             bool hide_thinking = false);
std::string make_anthropic_count_tokens_response(int input_tokens);

class AnthropicMessagesStream {
public:
    AnthropicMessagesStream(AnthropicResponseIdentity identity, int input_tokens,
                            bool hide_thinking = false);

    // The Engine start event is exact for normal streams. The no-argument form is reserved for an
    // error raised before admission, so an Anthropic error event still has a valid stream prefix.
    std::string start();
    std::string start(const ninfer::GenerationStart& generation);

    [[nodiscard]] bool started() const noexcept { return started_; }

    std::vector<std::string> reasoning_delta(const std::string& text);
    std::vector<std::string> content_delta(const std::string& text);
    std::vector<std::string> finish(const GenerationOutcome& outcome);
    std::string error(const ApiError& api_error) const;

private:
    std::string start_with_cache(std::optional<int> cache_read_input_tokens);
    std::vector<std::string> close_thinking();
    std::vector<std::string> close_text();

    AnthropicResponseIdentity identity_;
    std::string reasoning_;
    std::string content_;
    bool hide_thinking_ = false;
    int input_tokens_   = 0;
    int next_index_     = 0;
    int thinking_index_ = -1;
    int text_index_     = -1;
    bool started_       = false;
    bool finished_      = false;
    bool thinking_open_ = false;
    bool text_open_     = false;
};

} // namespace ninfer::serve
