#pragma once

#include "serve/request.h"
#include "serve/request_json.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>

namespace ninfer::text {
struct ParsedJsonNumbers;
}

namespace ninfer::serve {
void validate_schema_number_input(const text::ParsedJsonNumbers& parsed);

[[noreturn]] void bad_request(std::string message, std::string param = {}, std::string code = {});

std::optional<int> optional_int(const RequestJson& object, const char* key);
std::optional<double> optional_number(const RequestJson& object, const char* key);
bool optional_bool(const RequestJson& object, const char* key, bool fallback);
enum class JsonFormatProtocol { Chat, Responses, Anthropic };
void parse_json_output_format(const RequestJson& format, GenerationRequest& request,
                              const std::string& param, JsonFormatProtocol protocol);

void parse_structured_outputs(const RequestJson& body, GenerationRequest& request);

// The NInfer `graft` extension field: a graft name, "" for explicitly none, or absent/null (nullopt)
// to take the server's default graft. Whether the name is loaded is checked against the server's
// grafts during translation.
std::optional<std::string> parse_graft_field(const RequestJson& body);
// The NInfer `lora` extension field, read like `graft`: an adapter name loaded with --lora, "" for
// the base model, or absent/null (nullopt) to take the server's --default-lora.
std::optional<std::string> parse_lora_field(const RequestJson& body);

// The NInfer `thinking_budget` extension field: a positive cap on model-origin thinking tokens for
// this request, or absent/null (nullopt) to take the server default. Ignored when the request does
// not think. Anthropic Messages carries the same cap as `thinking.budget_tokens`.
std::optional<std::uint32_t> parse_thinking_budget_field(const RequestJson& body);

// The NInfer `relaxed_drafting` extension field (relaxed drafting; changes the output
// distribution): a threshold in [0,1], `true` for 0.5, `false` for 0 (exact), or absent/null
// (nullopt) to take the server's --relaxed-drafting default. Anything else fails with
// `invalid_relaxed_drafting`.
std::optional<double> parse_relaxed_drafting_field(const RequestJson& body);

[[nodiscard]] bool valid_tool_name(std::string_view name, std::size_t maximum_length) noexcept;

// The longest tool or namespace name a client may declare. Clients such as Codex build names out of
// MCP server and tool names, so they use dots, colons and long prefixes the Engine cannot carry.
inline constexpr std::size_t kMaxClientToolNameBytes = 512;

// Maps a client's tool name to one the Engine accepts, [A-Za-z0-9_-]{1,64}. A name that already
// fits is returned unchanged. Otherwise bytes outside the set become '_', the result is cut to
// leave room for '_' plus an 8-digit hash of the original, and the hash is appended, so two
// different originals do not share an Engine name. Deterministic: history replays and later
// requests produce the same name. The caller keeps the original to restore on the wire.
[[nodiscard]] std::string engine_tool_name(std::string_view client_name);

} // namespace ninfer::serve
