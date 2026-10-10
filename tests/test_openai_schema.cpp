#include "serve/generation_service.h"
#include "serve/openai_chat.h"
#include "serve/openai_common.h"
#include "serve/translate.h"

#include <nlohmann/json.hpp>

#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

namespace {

using Json = ninfer::serve::RequestJson;
using namespace ninfer::serve;

int check(bool condition, const std::string& label) {
    if (condition) { return 0; }
    std::cerr << "FAIL: " << label << '\n';
    return 1;
}

template <typename Function>
ApiError api_error(Function&& function) {
    try {
        function();
    } catch (const ApiException& exception) { return exception.error(); }
    return ApiError{.status = 0, .message = "no exception"};
}

template <typename Function>
bool throws_logic(Function&& function) {
    try {
        function();
    } catch (const std::logic_error&) { return true; }
    return false;
}

RequestLimits limits() { return RequestLimits{.default_max_tokens = 512}; }

Json base_request() {
    return Json{{"model", "qwen"},
                {"messages", Json::array({Json{{"role", "user"}, {"content", "hello"}}})}};
}

OpenAIChatRequest parse(Json body) { return parse_chat_completion_request(body, limits()); }

ResolvedPromptSemantics semantics(const GenerationRequest& request) {
    ServeOptions server;
    return resolve_prompt_semantics(request, server);
}

ninfer::PromptInput prompt(const GenerationRequest& request) {
    return to_prompt_input(request, semantics(request), {});
}

ninfer::RequestOptions options(const GenerationRequest& request) {
    ServeOptions server;
    return to_request_options(request, server, semantics(request), true);
}

Json parse_sse(const std::string& event) {
    constexpr std::string_view prefix = "data: ";
    if (!event.starts_with(prefix) || !event.ends_with("\n\n")) {
        throw std::runtime_error("invalid SSE framing");
    }
    return Json::parse(event.substr(prefix.size(), event.size() - prefix.size() - 2));
}

int test_request_envelope_and_sampling() {
    int failures                  = 0;
    Json body                     = base_request();
    body["stream"]                = true;
    body["stream_options"]        = Json{{"include_usage", true}, {"include_obfuscation", false}};
    body["max_completion_tokens"] = 48;
    body["max_tokens"]            = 9;
    body["temperature"]           = 0.7;
    body["top_p"]                 = 0.8;
    body["presence_penalty"]      = 0.3;
    body["frequency_penalty"]     = -0.2;
    body["seed"]                  = -1;
    body["top_k"]                 = 17;
    body["min_p"]                 = 0.05;
    body["timings_per_token"]     = true;
    body["return_progress"]       = true;

    const OpenAIChatRequest request = parse(body);
    failures += check(request.model == "qwen", "model remains in OpenAI envelope");
    failures += check(request.stream && request.include_usage, "stream metadata parsed");
    failures += check(request.timings_per_token && request.return_progress,
                      "llama.cpp response observations remain in the protocol envelope");
    failures += check(request.output_tokens_explicit && request.generation.max_tokens == 48,
                      "max_completion_tokens wins and explicitness stays in envelope");
    failures += check(request.generation.sampling.seed == std::numeric_limits<std::uint64_t>::max(),
                      "signed seed maps modulo 2^64");
    failures +=
        check(request.generation.sampling.top_k == 17 && request.generation.sampling.min_p == 0.05,
              "compatible sampler extensions parsed");
    const ninfer::RequestOptions translated = options(request.generation);
    failures +=
        check(translated.execution.sampling.top_k == 17, "top_k reaches Engine request options");
    failures +=
        check(translated.execution.sampling.min_p && *translated.execution.sampling.min_p == 0.05F,
              "min_p reaches Engine request options");
    failures +=
        check(translated.execution.sampling.seed == std::numeric_limits<std::uint64_t>::max(),
              "signed seed reaches Engine request options");

    const OpenAIChatRequest defaults = parse(base_request());
    failures +=
        check(!defaults.stream && !defaults.include_usage && !defaults.output_tokens_explicit &&
                  !defaults.timings_per_token && !defaults.return_progress &&
                  defaults.generation.max_tokens == limits().default_max_tokens,
              "protocol defaults remain outside GenerationRequest");
    failures += check(!defaults.generation.derive_output_budget,
                      "a --default-max-tokens cap was marked for derivation");
    const RequestLimits derived{.max_context = 4096};
    const OpenAIChatRequest derived_omitted = parse_chat_completion_request(base_request(), derived);
    failures += check(derived_omitted.generation.derive_output_budget &&
                          derived_omitted.generation.max_tokens == 4096,
                      "an omitted limit without a server cap was not marked for derivation");
    Json explicit_limit          = base_request();
    explicit_limit["max_tokens"] = 32;
    const OpenAIChatRequest derived_explicit = parse_chat_completion_request(explicit_limit, derived);
    failures += check(!derived_explicit.generation.derive_output_budget &&
                          derived_explicit.generation.max_tokens == 32,
                      "an explicit limit was marked for derivation");
    ServeOptions server;
    server.default_thinking_budget = 256;
    GenerationRequest gen_req = derived_omitted.generation;
    gen_req.thinking_budget = 256;
    const ninfer::RequestOptions translated_derived =
        to_request_options(gen_req, server, semantics(gen_req), true);
    failures += check(derived_omitted.generation.derive_output_budget &&
                          translated_derived.execution.thinking.budget == 256,
                      "derived output budget request retained thinking budget in translation");

    Json malformed              = base_request();
    malformed["stream_options"] = true;
    failures += check(api_error([&] { (void)parse(malformed); }).param == "stream_options",
                      "malformed stream_options rejected");
    malformed                      = base_request();
    malformed["timings_per_token"] = "yes";
    failures += check(api_error([&] { (void)parse(malformed); }).param == "timings_per_token",
                      "non-boolean timings_per_token rejected");
    malformed                    = base_request();
    malformed["return_progress"] = 1;
    failures += check(api_error([&] { (void)parse(malformed); }).param == "return_progress",
                      "non-boolean return_progress rejected");
    return failures;
}

int test_standard_field_policy() {
    int failures  = 0;
    auto rejected = [&](const char* key, Json value, const char* code) {
        Json body            = base_request();
        body[key]            = std::move(value);
        const ApiError error = api_error([&] { (void)parse(body); });
        failures += check(error.param == key && error.code == code,
                          std::string(key) + " non-neutral value rejected");
    };

    rejected("n", 2, "n_not_supported");
    rejected("logit_bias", Json{{"12", 1}}, "logit_bias_not_supported");
    rejected("logprobs", true, "logprobs_not_supported");
    rejected("top_logprobs", 2, "logprobs_not_supported");
    rejected("modalities", Json::array({"text", "audio"}), "modality_not_supported");
    rejected("web_search_options", Json::object(), "web_search_not_supported");
    rejected("moderation", Json::object(), "moderation_not_supported");
    rejected("verbosity", "high", "verbosity_not_supported");
    rejected("store", true, "store_not_supported");
    rejected("functions", Json::array({Json{{"name", "legacy"}}}), "legacy_tools_not_supported");

    Json neutral                      = base_request();
    neutral["n"]                      = 1;
    neutral["logit_bias"]             = Json{{"12", 0}, {"13", 0.0}};
    neutral["logprobs"]               = false;
    neutral["top_logprobs"]           = 0;
    neutral["response_format"]        = Json{{"type", "text"}};
    neutral["modalities"]             = Json::array({"text"});
    neutral["audio"]                  = Json{{"voice", "alloy"}};
    neutral["prediction"]             = Json{{"type", "content"}, {"content", "expected"}};
    neutral["verbosity"]              = "medium";
    neutral["store"]                  = false;
    neutral["functions"]              = Json::array();
    neutral["function_call"]          = "auto";
    neutral["metadata"]               = Json{{"trace", "client"}};
    neutral["user"]                   = "user-1";
    neutral["safety_identifier"]      = "safe-1";
    neutral["prompt_cache_key"]       = "cache-1";
    neutral["prompt_cache_options"]   = Json{{"retention", "24h"}};
    neutral["prompt_cache_retention"] = "24h";
    neutral["service_tier"]           = "priority";
    neutral["future_unknown_field"]   = Json{{"value", 1}};
    failures += check(parse(neutral).generation.messages.size() == 1,
                      "neutral controls and advisory hints are accepted");

    Json zero_limit                     = base_request();
    zero_limit["max_completion_tokens"] = 0;
    const OpenAIChatRequest zero        = parse(zero_limit);
    failures += check(zero.output_tokens_explicit && zero.generation.max_tokens == 0,
                      "an explicit zero output limit reaches Engine's no-generation path");
    return failures;
}

// ignore_eos is an NInfer extension that removes the model EOS; a constrained output can only end
// there, so the combination is refused at the extension's own parameter.
int test_ignore_eos_with_constraints() {
    int failures = 0;
    const auto refused = [&](Json body, const std::string& label) {
        const ApiError error = api_error([&] { (void)parse(std::move(body)); });
        failures += check(error.status == 400 && error.param == "ignore_eos", label);
    };
    Json format               = base_request();
    format["response_format"] = Json{{"type", "json_object"}};
    format["ignore_eos"]      = true;
    refused(format, "response_format with ignore_eos was accepted");
    Json tools          = base_request();
    tools["tools"]      = Json::array({Json{
        {"type", "function"},
        {"function", Json{{"name", "lookup"}, {"parameters", Json{{"type", "object"}}}}}}});
    tools["ignore_eos"]       = true;
    tools["tool_constraints"] = "basic";
    refused(tools, "explicitly constrained tools with ignore_eos were accepted");
    tools["tool_constraints"] = "auto";
    failures += check(!options(parse(tools).generation).stop.include_model_defaults,
                      "ignore_eos with unconstrained tools did not reach the Engine");
    return failures;
}

int test_prompt_cache_boundaries() {
    using Location = ninfer::PromptCacheMarkerLocation;
    using Evidence = ninfer::SharedCandidateEvidence;
    int failures   = 0;
    for (const char* role : {"user", "system", "developer"}) {
        Json body = base_request();
        body["messages"].push_back(Json{{"role", role}, {"content", "complete message"}});
        const auto prepared = prompt(parse(body).generation);
        const auto& markers = prepared.context_cache.markers;
        failures += check(
            markers.size() == 1 && markers[0].location == Location::MessagePartBoundary &&
                markers[0].after_message_count == 2 && markers[0].after_message_part_count == 1 &&
                markers[0].evidence == Evidence::DefaultAutomatic,
            std::string(role) + " automatic cache ends at the final source part");
    }

    Json multipart                      = base_request();
    multipart["messages"][0]["content"] = Json::array(
        {Json{{"type", "text"}, {"text", "first"}}, Json{{"type", "text"}, {"text", "second"}}});
    multipart["prompt_cache_options"] = Json{{"mode", "implicit"}};
    const auto requested              = prompt(parse(multipart).generation);
    failures +=
        check(requested.context_cache.markers.size() == 1 &&
                  requested.context_cache.markers[0].location == Location::MessagePartBoundary &&
                  requested.context_cache.markers[0].after_message_part_count == 2 &&
                  requested.context_cache.markers[0].evidence == Evidence::RequestedAutomatic,
              "requested implicit caching uses the final source part");
    multipart["prompt_cache_options"] = Json{{"mode", "explicit"}};
    const auto disabled               = prompt(parse(multipart).generation);
    failures += check(disabled.context_cache.markers.empty() &&
                          !disabled.context_cache.allow_engine_automatic_shared_prefixes,
                      "explicit mode without markers disables automatic shared writes");

    Json marked                      = base_request();
    marked["messages"][0]["content"] = Json::array();
    for (int index = 0; index < 4; ++index) {
        marked["messages"][0]["content"].push_back(
            Json{{"type", "text"},
                 {"text", "section " + std::to_string(index)},
                 {"prompt_cache_breakpoint", Json{{"mode", "explicit"}}}});
    }
    const auto merged = prompt(parse(marked).generation);
    failures += check(merged.context_cache.markers.size() == 4,
                      "automatic caching reuses an explicitly marked final part");
    for (std::size_t index = 0; index < merged.context_cache.markers.size(); ++index) {
        const auto& marker = merged.context_cache.markers[index];
        failures += check(
            marker.location == Location::MessagePartBoundary &&
                marker.after_message_part_count == index + 1 &&
                ninfer::has_shared_candidate_evidence(marker.evidence, Evidence::ExplicitBoundary),
            "explicit part locations survive automatic merging");
    }
    failures +=
        check(merged.context_cache.markers.size() == 4 &&
                  ninfer::has_shared_candidate_evidence(
                      merged.context_cache.markers.back().evidence, Evidence::DefaultAutomatic),
              "the final explicit part also carries the automatic opportunity");
    marked["messages"][0]["content"].push_back(Json{{"type", "text"}, {"text", "new tail"}});
    const auto full = prompt(parse(marked).generation);
    failures += check(full.context_cache.markers.size() == 4,
                      "automatic caching yields to four explicit boundaries");
    for (std::size_t index = 0; index < full.context_cache.markers.size(); ++index) {
        const auto& marker = full.context_cache.markers[index];
        failures += check(marker.location == Location::MessagePartBoundary &&
                              marker.after_message_part_count == index + 1 &&
                              marker.evidence == Evidence::ExplicitBoundary,
                          "a later implicit target does not displace an explicit boundary");
    }

    Json assistant = base_request();
    assistant["messages"].push_back(Json{{"role", "assistant"}, {"content", "partial answer"}});
    const auto assistant_prompt = prompt(parse(assistant).generation);
    failures += check(assistant_prompt.context_cache.markers.size() == 1 &&
                          assistant_prompt.context_cache.markers[0].location ==
                              Location::MessagePartBoundary &&
                          assistant_prompt.context_cache.markers[0].after_message_count == 2,
                      "assistant content retains its part boundary for continuation");

    Json tool = base_request();
    tool["messages"].push_back(
        Json{{"role", "assistant"},
             {"tool_calls",
              Json::array({Json{{"id", "call_cache"},
                                {"type", "function"},
                                {"function", Json{{"name", "lookup"}, {"arguments", "{}"}}}}})}});
    tool["messages"].push_back(
        Json{{"role", "tool"}, {"tool_call_id", "call_cache"}, {"content", "result"}});
    const auto tool_prompt = prompt(parse(tool).generation);
    failures +=
        check(tool_prompt.context_cache.markers.size() == 1 &&
                  tool_prompt.context_cache.markers[0].location == Location::MessagePartBoundary &&
                  tool_prompt.context_cache.markers[0].after_message_count == 3,
              "tool results retain their content boundary when a result group grows");
    return failures;
}

int test_constrained_decoding_extensions() {
    int failures               = 0;
    Json body                  = base_request();
    body["structured_outputs"] = Json{{"choice", {"", "yes", "你好"}}};
    const auto choice          = options(parse(body).generation).constraint;
    failures += check(choice == ninfer::OutputConstraint::choice({"", "yes", "你好"}),
                      "choice literals were changed in Engine translation");
    body["structured_outputs"] = Json{{"regex", ""}};
    failures +=
        check(options(parse(body).generation).constraint == ninfer::OutputConstraint::regex(""),
              "empty regex was dropped in Engine translation");
    for (const auto& value : {Json{{"choice", Json::array()}}, Json{{"choice", {"a", 1}}},
                              Json{{"regex", 7}}, Json{{"regex", "a"}, {"choice", {"a"}}}}) {
        body["structured_outputs"] = value;
        failures += check(api_error([&] { (void)parse(body); }).status == 400,
                          "invalid choice/regex request accepted");
    }
    for (const auto kind :
         {ninfer::RequestErrorKind::InvalidChoice, ninfer::RequestErrorKind::InvalidRegex}) {
        const auto param = kind == ninfer::RequestErrorKind::InvalidChoice
                               ? "structured_outputs.choice"
                               : "structured_outputs.regex";
        const auto error = request_error_to_api_error(ninfer::RequestError(kind, "invalid"), param);
        failures += check(error.status == 400 && error.param == param &&
                              error.code == (kind == ninfer::RequestErrorKind::InvalidChoice
                                                 ? "invalid_choice"
                                                 : "invalid_regex"),
                          "choice/regex error was misclassified");
    }
    body["structured_outputs"] = Json{{"grammar", "root ::= \"yes\" | \"no\""}};
    const auto parsed          = parse(body);
    failures += check(parsed.generation.constraint->source == "root ::= \"yes\" | \"no\"" &&
                          options(parsed.generation).constraint == parsed.generation.constraint,
                      "GBNF must survive protocol-to-Engine translation");
    body["stop"] = "yes";
    failures += check(api_error([&] { (void)parse(body); }).param == "structured_outputs.grammar",
                      "grammar with custom stops accepted");
    for (const auto& value : {Json{{"grammar", ""}}, Json{{"json", Json::object()}}, Json("bad")}) {
        body                       = base_request();
        body["structured_outputs"] = value;
        failures += check(api_error([&] { (void)parse(body); }).status == 400,
                          "malformed structured_outputs accepted");
    }
    for (const char* alias :
         {"grammar", "guided_json", "guided_regex", "guided_choice", "guided_grammar"}) {
        body        = base_request();
        body[alias] = "root ::= \"yes\"";
        failures += check(api_error([&] { (void)parse(body); }).param == alias,
                          "unsupported constrained-decoding alias accepted");
    }
    body                    = base_request();
    body["response_format"] = Json{{"type", "json_object"}};
    failures += check(options(parse(body).generation).constraint->kind ==
                          ninfer::OutputConstraintKind::JsonObject,
                      "JSON object mode lost in Engine translation");
    const Json schema = {
        {"type", "object"},
        {"properties", {{"description", {{"type", "string"}}}, {"a", {{"type", "integer"}}}}}};
    body["response_format"] =
        Json{{"type", "json_schema"},
             {"json_schema", {{"name", "answer"}, {"strict", false}, {"schema", schema}}}};
    const auto typed = parse(body).generation;
    failures += check(typed.constraint->kind == ninfer::OutputConstraintKind::JsonSchema &&
                          typed.constraint->source == schema.dump() &&
                          typed.constraint_param == "response_format.json_schema.schema",
                      "JSON schema source/order or diagnostic location lost");
    body["tools"] = Json::array(
        {Json{{"type", "function"},
              {"function", {{"name", "lookup"}, {"parameters", {{"type", "object"}}}}}}});
    const auto combined = parse(body).generation;
    failures +=
        check(combined.constraint && combined.uses_tools(), "JSON output blocked active tools");
    const std::vector<std::string> schema_paths{combined.tools[0].schema_param};
    const auto tool_error = request_error_to_api_error(
        ninfer::RequestError(ninfer::RequestErrorKind::UnsupportedJsonSchema, "bad tool schema",
                             "/0/parameters/properties/x/format",
                             ninfer::RequestErrorSource::Tools),
        combined.constraint_param, schema_paths);
    failures += check(tool_error.param == "tools/0/function/parameters/properties/x/format",
                      "combined request attributed tool error to body schema");
    body["structured_outputs"] = Json{{"grammar", "root ::= \"x\""}};
    failures +=
        check(api_error([&] { (void)parse(body); }).status == 400, "conflicting formats accepted");
    const auto error = request_error_to_api_error(
        ninfer::RequestError(ninfer::RequestErrorKind::UnsupportedJsonSchema, "unsupported keyword",
                             "/properties/x/format"),
        typed.constraint_param);
    failures += check(error.param == "response_format.json_schema.schema/properties/x/format" &&
                          error.code == "unsupported_json_schema",
                      "schema error lost its source path");
    return failures;
}

Json function_tool(std::string name = "weather", bool strict = false) {
    return Json{{"type", "function"},
                {"function", Json{{"name", std::move(name)},
                                  {"description", "Get weather"},
                                  {"parameters", Json{{"type", "object"}}},
                                  {"strict", strict}}}};
}

int test_tools() {
    int failures                      = 0;
    Json body                         = base_request();
    body["tools"]                     = Json::array({function_tool()});
    const OpenAIChatRequest automatic = parse(body);
    failures += check(automatic.generation.uses_tools(), "function tools default to auto");
    failures += check(automatic.generation.constrains_tools() &&
                          !options(automatic.generation).output.preserve_special_tokens,
                      "ordinary auto tools must receive basic constraints");
    failures += check(prompt(automatic.generation).options.tool_jsons.size() == 1,
                      "auto tools reach PromptInput");
    body["tool_constraints"] = "auto";
    failures += check(!parse(body).generation.constrains_tools(),
                      "explicit auto did not select request-driven constraints");
    body["tool_choice"] = "required";
    failures +=
        check(parse(body).generation.constrains_tools(), "auto relaxed required tool choice");
    body.erase("tool_choice");
    body.erase("tool_constraints");

    // Default constraints give way to a client's own stop strings instead of refusing the request;
    // an explicit request for constraints does not.
    body["stop"]               = Json::array({"</done>"});
    const auto with_stop       = parse(body).generation;
    failures += check(!with_stop.constrains_tools() && with_stop.uses_tools() &&
                          with_stop.stop_strings.size() == 1,
                      "stop strings with default tool constraints fall back to free tool calls");
    body["tool_constraints"] = "basic";
    failures += check(api_error([&] { (void)parse(body); }).param == "stop",
                      "explicit basic constraints still refuse stop strings");
    body.erase("tool_constraints");
    body["tool_choice"] = "required";
    failures += check(api_error([&] { (void)parse(body); }).param == "stop",
                      "required tool choice still refuses stop strings");
    body.erase("tool_choice");
    body.erase("stop");
    body["ignore_eos"] = true;
    failures += check(!parse(body).generation.constrains_tools(),
                      "ignore_eos with default tool constraints falls back to free tool calls");
    body.erase("ignore_eos");

    body["tools"][0]["future_item_field"]                 = "ignored";
    body["tools"][0]["function"]["future_function_field"] = "ignored";
    const std::string normalized_definition = prompt(parse(body).generation).options.tool_jsons[0];
    failures += check(normalized_definition.find("future_item_field") == std::string::npos &&
                          normalized_definition.find("future_function_field") == std::string::npos,
                      "unknown tool fields do not silently alter the model prompt");

    body["tool_choice"]          = "none";
    body["parallel_tool_calls"]  = false;
    const OpenAIChatRequest none = parse(body);
    failures += check(!none.generation.uses_tools() &&
                          prompt(none.generation).options.tool_jsons.size() == 1,
                      "tool_choice none keeps declarations in the prompt");

    body["tool_choice"] = "required";
    failures += check(parse(body).generation.tool_choice.mode == ToolChoiceMode::Required,
                      "required choice reaches generation");
    body["tool_choice"] = Json{{"type", "function"}, {"function", Json{{"name", "weather"}}}};
    failures += check(parse(body).generation.tool_choice.allowed_names ==
                              std::vector<std::string>{"weather"} &&
                          !parse(body).generation.tool_choice.parallel,
                      "named choice selects exactly one invocation");

    body          = base_request();
    body["tools"] = Json::array({function_tool(), function_tool("search")});
    body["tool_choice"] =
        Json{{"type", "allowed_tools"},
             {"allowed_tools",
              Json{{"mode", "auto"},
                   {"tools", Json::array({Json{{"type", "function"}, {"name", "search"}}})}}}};
    const GenerationRequest allowed = parse(body).generation;
    failures += check(allowed.tools.size() == 2 &&
                          allowed.tool_choice.allowed_names == std::vector<std::string>{"search"} &&
                          prompt(allowed).options.tool_jsons.size() == 2,
                      "allowed_tools preserves declarations and selects callable names");

    body["tool_choice"] =
        Json{{"type", "allowed_tools"},
             {"mode", "auto"},
             {"tools", Json::array({Json{{"type", "function"}, {"name", "weather"}}})}};
    const GenerationRequest direct_allowed = parse(body).generation;
    failures +=
        check(direct_allowed.tools.size() == 2 &&
                  direct_allowed.tool_choice.allowed_names == std::vector<std::string>{"weather"},
              "direct allowed_tools compatibility shape is accepted");
    body["tool_choice"]["mode"] = "required";
    failures += check(parse(body).generation.tool_choice.mode == ToolChoiceMode::Required,
                      "allowed_tools required reaches generation");
    body["tool_choice"]["mode"]             = "auto";
    body["tool_choice"]["tools"][0]["name"] = "missing";
    failures += check(api_error([&] { (void)parse(body); }).param == "tool_choice",
                      "allowed_tools rejects names absent from the declared tool set");

    body          = base_request();
    body["tools"] = Json::array({function_tool("weather", true)});
    failures += check(
        parse(body).generation.tools[0].strict &&
            Json::parse(
                prompt(parse(body).generation).options.tool_jsons[0])["function"]["strict"] == true,
        "strict survives prompt and request translation");
    body["tools"] = Json::array({Json{{"type", "code_runner"}, {"name", "shell"}}});
    failures += check(api_error([&] { (void)parse(body); }).code == "tool_type_not_supported",
                      "unknown client-executed tool types rejected");

    // A hosted tool is the server's to run. NInfer has no executor, and the caller is not waiting
    // on one either, so the declaration is dropped instead of failing a request the client cannot
    // change: agent harnesses declare web_search unconditionally.
    body          = base_request();
    body["tools"] = Json::array({Json{{"type", "web_search"}}, function_tool()});
    const GenerationRequest hosted = parse(body).generation;
    failures += check(hosted.tools.size() == 1 && hosted.tools[0].name == "weather",
                      "hosted web_search is dropped and the function tool survives");
    body["tools"] = Json::array({Json{{"type", "web_search_preview_2025_03_11"}}});
    failures += check(parse(body).generation.tools.empty(),
                      "dated hosted tool spellings are dropped by family");
    body["tools"] = Json::array({Json{{"type", "web_search"}}});
    failures += check(!parse(body).generation.uses_tools(),
                      "a request whose only tool is hosted still parses, with no callable tools");

    body                        = base_request();
    body["tools"]               = Json::array({function_tool()});
    body["parallel_tool_calls"] = false;
    failures += check(!parse(body).generation.tool_choice.parallel,
                      "parallel_tool_calls=false reaches generation");
    body.erase("tools");
    failures += check(parse(body).generation.tools.empty(),
                      "parallel_tool_calls=false is neutral without tools");
    body["tool_choice"] = "auto";
    failures +=
        check(parse(body).generation.tools.empty(), "tool_choice auto is neutral without tools");

    Json history = base_request();
    history["messages"] =
        Json::array({Json{{"role", "user"}, {"content", "weather?"}},
                     Json{{"role", "assistant"},
                          {"content", nullptr},
                          {"tool_calls",
                           Json::array({Json{{"id", "call_1"},
                                             {"type", "function"},
                                             {"function", Json{{"name", "weather"},
                                                               {"arguments", "not-json-yet"}}}}})}},
                     Json{{"role", "tool"}, {"tool_call_id", "call_1"}, {"content", "sunny"}}});
    failures += check(parse(history).generation.has_tool_history(),
                      "tool-call history follows wire types without inventing JSON validation");

    Json mixed_assistant        = base_request();
    mixed_assistant["messages"] = Json::array(
        {Json{{"role", "user"}, {"content", "inspect"}},
         Json{{"role", "assistant"},
              {"content", "I will inspect it"},
              {"tool_calls",
               Json::array({Json{
                   {"id", "call_2"},
                   {"type", "function"},
                   {"function", Json{{"name", "inspect"}, {"arguments", R"({"path":"a"})"}}}}})}}});
    const GenerationRequest mixed_request  = parse(mixed_assistant).generation;
    const ninfer::PromptInput mixed_prompt = prompt(mixed_request);
    failures += check(mixed_request.messages[1].cache_boundary_after &&
                          !mixed_request.messages[1].content[0].cache_boundary_after &&
                          !mixed_prompt.context_cache.markers.empty() &&
                          mixed_prompt.context_cache.markers.back().location ==
                              ninfer::PromptCacheMarkerLocation::MessageBoundary &&
                          mixed_prompt.context_cache.markers.back().after_message_count == 2,
                      "automatic caching stops after a complete assistant text/tool-call turn");

    const Json ordered = Json::parse(
        R"({"model":"qwen","messages":[{"role":"user","content":"probe"}],"tools":[{"type":"function","function":{"name":"probe","parameters":{"type":"object","properties":{"zeta":{"type":"string"},"alpha":{"type":"integer"}}}}}]})");
    const ninfer::PromptInput ordered_prompt = prompt(parse(ordered).generation);
    failures += check(
        ordered_prompt.options.tool_jsons.size() == 1 &&
            ordered_prompt.options.tool_jsons.front() ==
                R"({"type":"function","function":{"name":"probe","parameters":{"type":"object","properties":{"zeta":{"type":"string"},"alpha":{"type":"integer"}}},"strict":false}})",
        "OpenAI Chat changed tool-schema member order before PromptInput");
    return failures;
}

// Free-form `custom` tools (OpenAI Chat Completions ChatCompletionCustomToolParam). A custom tool
// is lowered exactly like on Responses: a strict function with one required string parameter,
// `input`, whose grammar is described, not enforced. Its calls are answered as
// `{type:"custom", custom:{name, input}}`, and history in that shape lowers back to `{"input"}`.
Json custom_tool(std::string name = "apply_patch") {
    return Json{{"type", "custom"},
                {"custom", Json{{"name", std::move(name)},
                                {"description", "Edit files with a patch."},
                                {"format", Json{{"type", "grammar"},
                                                {"grammar", Json{{"syntax", "lark"},
                                                                 {"definition", "start: LF"}}}}}}}};
}

int test_custom_tools() {
    int failures                    = 0;
    Json body                       = base_request();
    body["tools"]                   = Json::array({custom_tool(), function_tool("shell")});
    const OpenAIChatRequest request = parse(body);
    const ToolDefinition& tool      = request.generation.tools.at(0);
    const Json expected_schema      = {
        {"type", "object"},
        {"properties",
              Json{{"input",
                    Json{{"type", "string"},
                         {"description",
                          "The tool's complete free-form input, passed to it exactly as written (not "
                               "JSON). It must match this lark grammar:\nstart: LF"}}}}},
        {"required", Json::array({"input"})},
        {"additionalProperties", false}};
    failures +=
        check(tool.name == "apply_patch" && tool.description == "Edit files with a patch." &&
                  tool.strict && Json::parse(tool.input_schema_json) == expected_schema &&
                  !request.generation.tools.at(1).strict &&
                  request.custom_tools == std::unordered_set<std::string>{"apply_patch"},
              "a Chat custom tool is a strict function with one required string input");
    const Json rendered = Json::parse(prompt(request.generation).options.tool_jsons.at(0));
    failures += check(rendered.at("type") == "function" &&
                          rendered.at("function").at("name") == "apply_patch" &&
                          rendered.at("function").at("strict") == true &&
                          rendered.at("function").at("parameters") == expected_schema,
                      "a Chat custom tool renders as a strict function");

    body["tool_constraints"] = "auto";
    failures += check(parse(body).generation.constrains_tools(),
                      "a custom tool stays constrained under tool_constraints auto");
    body.erase("tool_constraints");
    body["stop"] = Json::array({"</done>"});
    failures += check(api_error([&] { (void)parse(body); }).param == "stop",
                      "a custom tool's strict lowering refuses custom stop strings");
    body.erase("stop");

    Json text_format                            = body;
    text_format["tools"][0]["custom"]["format"] = Json{{"type", "text"}};
    Json no_format                              = body;
    no_format["tools"][0]["custom"].erase("format");
    const std::string plain_description =
        "The tool's complete free-form input, passed to it exactly as written (not JSON).";
    failures += check(
        Json::parse(parse(text_format).generation.tools[0].input_schema_json)["properties"]["input"]
                                                                             ["description"] ==
                plain_description &&
            Json::parse(
                parse(no_format).generation.tools[0].input_schema_json)["properties"]["input"]
                                                                       ["description"] ==
                plain_description,
        "a text or absent format describes unconstrained free-form input");

    // Forced choice and allowed_tools, in OpenAI's nested and the flat compatibility shape.
    body["tool_choice"] = Json{{"type", "custom"}, {"custom", Json{{"name", "apply_patch"}}}};
    const GenerationRequest forced = parse(body).generation;
    failures +=
        check(forced.tool_choice.mode == ToolChoiceMode::Required &&
                  forced.tool_choice.allowed_names == std::vector<std::string>{"apply_patch"} &&
                  !forced.tool_choice.parallel,
              "a named custom tool_choice forces exactly one call of that tool");
    body["tool_choice"] = Json{
        {"type", "allowed_tools"},
        {"allowed_tools",
         Json{{"mode", "required"},
              {"tools",
               Json::array({Json{{"type", "custom"}, {"custom", Json{{"name", "apply_patch"}}}},
                            Json{{"type", "function"}, {"function", Json{{"name", "shell"}}}}})}}}};
    const GenerationRequest allowed = parse(body).generation;
    failures += check(allowed.tool_choice.mode == ToolChoiceMode::Required &&
                          allowed.tool_choice.allowed_names ==
                              std::vector<std::string>{"apply_patch", "shell"},
                      "allowed_tools selects custom and function tools in OpenAI's nested shape");
    body["tool_choice"]["allowed_tools"]["tools"] =
        Json::array({Json{{"type", "custom"}, {"name", "apply_patch"}}});
    failures += check(parse(body).generation.tool_choice.allowed_names ==
                          std::vector<std::string>{"apply_patch"},
                      "allowed_tools accepts a flat custom entry");

    // Rejections: malformed definitions and references that mix a tool's kind.
    auto rejected = [&](Json invalid, const std::string& param, const std::string& code,
                        const std::string& label) {
        const ApiError error = api_error([&] { (void)parse(std::move(invalid)); });
        failures += check(
            error.status == 400 && error.param == param && (code.empty() || error.code == code),
            label + " (got " + error.param + "/" + error.code + ": " + error.message + ")");
    };
    Json invalid     = base_request();
    invalid["tools"] = Json::array({Json{{"type", "custom"}, {"name", "apply_patch"}}});
    rejected(invalid, "tools", "", "a custom tool without a custom object");
    invalid["tools"]                        = Json::array({custom_tool()});
    invalid["tools"][0]["custom"]["format"] = Json{{"type", "regex"}};
    rejected(invalid, "tools", "", "an unknown custom format type");
    invalid["tools"][0]["custom"]["format"] = Json{{"type", "grammar"}};
    rejected(invalid, "tools", "", "a grammar format without a grammar object");
    invalid["tools"][0]["custom"]["format"] =
        Json{{"type", "grammar"}, {"grammar", Json{{"syntax", "ebnf"}, {"definition", "x"}}}};
    rejected(invalid, "tools", "", "a grammar syntax other than lark or regex");
    invalid["tools"][0]["custom"]["format"] =
        Json{{"type", "grammar"}, {"grammar", Json{{"syntax", "regex"}}}};
    rejected(invalid, "tools", "", "a grammar without a definition");
    invalid["tools"]                             = Json::array({custom_tool()});
    invalid["tools"][0]["custom"]["description"] = 5;
    rejected(invalid, "tools", "", "a non-string custom description");
    invalid["tools"] = Json::array({custom_tool("dup"), function_tool("dup")});
    rejected(invalid, "tools", "duplicate_tool_name", "a name declared as custom and function");
    invalid["tools"] = Json::array({function_tool("dup"), custom_tool("dup")});
    rejected(invalid, "tools", "duplicate_tool_name", "a name declared as function and custom");
    invalid = body;
    invalid["tool_choice"] =
        Json{{"type", "function"}, {"function", Json{{"name", "apply_patch"}}}};
    rejected(invalid, "tool_choice", "invalid_tool_choice",
             "a function tool_choice naming a custom tool");
    invalid["tool_choice"] = Json{{"type", "custom"}, {"custom", Json{{"name", "shell"}}}};
    rejected(invalid, "tool_choice", "invalid_tool_choice",
             "a custom tool_choice naming a function tool");
    invalid["tool_choice"] = Json{{"type", "custom"}, {"custom", Json{{"name", "missing"}}}};
    rejected(invalid, "tool_choice", "", "a custom tool_choice naming an undeclared tool");
    invalid["tool_choice"] = Json{{"type", "custom"}, {"name", "apply_patch"}};
    rejected(invalid, "tool_choice", "", "a custom tool_choice without its custom object");
    invalid["tool_choice"] =
        Json{{"type", "allowed_tools"},
             {"mode", "auto"},
             {"tools", Json::array({Json{{"type", "function"}, {"name", "apply_patch"}}})}};
    rejected(invalid, "tool_choice", "invalid_tool_choice",
             "allowed_tools selecting a custom tool as a function");
    return failures;
}

int test_messages_and_media() {
    int failures                   = 0;
    Json body                      = base_request();
    body["messages"][0]["content"] = Json::array(
        {Json{{"type", "text"}, {"text", "alpha"}}, Json{{"type", "text"}, {"text", "beta"}}});
    const ninfer::PromptInput translated = prompt(parse(body).generation);
    failures += check(translated.messages[0].parts.size() == 2 &&
                          translated.messages[0].parts[0].text == "alpha" &&
                          translated.messages[0].parts[1].text == "beta",
                      "adjacent text parts preserve exact text without inserted newline");

    body                           = base_request();
    body["messages"][0]["content"] = Json::array(
        {Json{{"type", "image_url"},
              {"image_url", Json{{"url", "https://example.test/a.png"}, {"detail", "auto"}}}},
         Json{{"type", "video_url"}, {"video_url", "https://example.test/a.mp4"}}});
    const GenerationRequest media = parse(body).generation;
    failures += check(media.media_item_count() == 2 &&
                          media.messages[0].content[0].kind == ContentKind::Image &&
                          media.messages[0].content[1].kind == ContentKind::Video,
                      "image and video compatibility inputs normalize to Engine media");

    body["messages"][0]["content"][0]["image_url"]["detail"] = "high";
    failures += check(api_error([&] { (void)parse(body); }).code == "image_detail_not_supported",
                      "explicit image preprocessing detail rejected");

    auto content_rejected = [&](const char* role, const char* type) {
        Json invalid                   = base_request();
        invalid["messages"][0]["role"] = role;
        invalid["messages"][0]["content"] =
            Json::array({Json{{"type", type}, {type, "https://example.test/x"}}});
        return api_error([&] { (void)parse(invalid); }).code == "modality_not_supported";
    };
    failures +=
        check(content_rejected("assistant", "image_url"), "assistant media history rejected");
    failures += check(content_rejected("system", "image_url"),
                      "system media rejected at protocol boundary");

    body["messages"] = Json::array(
        {Json{{"role", "user"}, {"content", "capture it"}},
         Json{{"role", "assistant"},
              {"content", nullptr},
              {"tool_calls",
               Json::array({Json{{"id", "call_capture"},
                                 {"type", "function"},
                                 {"function", Json{{"name", "capture"}, {"arguments", "{}"}}}}})}},
         Json{{"role", "tool"},
              {"tool_call_id", "call_capture"},
              {"content",
               Json::array({Json{{"type", "text"}, {"text", "captured"}},
                            Json{{"type", "image_url"},
                                 {"image_url", Json{{"url", "https://example.test/capture.png"},
                                                    {"detail", "auto"}}}}})}}});
    const GenerationRequest tool_image = parse(body).generation;
    failures += check(tool_image.messages.back().role == ninfer::ChatRole::Tool &&
                          tool_image.messages.back().tool_call_id == "call_capture" &&
                          tool_image.messages.back().content.size() == 2 &&
                          tool_image.messages.back().content[0].kind == ContentKind::Text &&
                          tool_image.messages.back().content[1].kind == ContentKind::Image,
                      "tool result text and image parts normalize to one tool turn");

    body["messages"].back()["content"] = Json::array(
        {Json{{"type", "video_url"}, {"video_url", "https://example.test/capture.mp4"}}});
    failures += check(api_error([&] { (void)parse(body); }).code == "modality_not_supported",
                      "tool result video remains outside the Chat compatibility extension");

    body = base_request();
    body["messages"][0]["content"] =
        Json::array({Json{{"type", "input_audio"}, {"input_audio", Json::object()}}});
    failures += check(api_error([&] { (void)parse(body); }).code == "modality_not_supported",
                      "input audio rejected");
    body["messages"][0]["content"] =
        Json::array({Json{{"type", "file"}, {"file", Json::object()}}});
    failures += check(api_error([&] { (void)parse(body); }).code == "modality_not_supported",
                      "file input rejected");

    body                        = base_request();
    body["messages"][0]["name"] = "speaker";
    failures += check(api_error([&] { (void)parse(body); }).code == "message_name_not_supported",
                      "message name rejected");

    body = base_request();
    body["messages"].push_back(Json{
        {"role", "assistant"},
        {"content", nullptr},
        {"tool_calls",
         Json::array({Json{{"id", "call_1"},
                           {"type", "function"},
                           {"function", Json{{"name", "get_status"}, {"arguments", "{}"}}}}})}});
    body["messages"].push_back(Json{
        {"role", "tool"}, {"name", "get_status"}, {"tool_call_id", "call_1"}, {"content", "ok"}});
    const GenerationRequest named_tool_history = parse(body).generation;
    const ChatTurn& named_tool                 = named_tool_history.messages.back();
    failures += check(named_tool.role == ninfer::ChatRole::Tool &&
                          named_tool.tool_call_id == "call_1" && !named_tool.tool_result_name &&
                          named_tool.content.size() == 1 && named_tool.content[0].text == "ok",
                      "tool message name is an ignored compatibility extension");

    body["messages"].back()["name"] = Json::array();
    failures +=
        check(api_error([&] { (void)parse(body); }).message == "message name must be a string",
              "tool message name remains type checked");

    body                           = base_request();
    body["messages"][0]["name"]    = "";
    body["messages"][0]["content"] = Json::array();
    failures += check(parse(body).generation.messages[0].content.empty(),
                      "empty names and empty content arrays remain neutral");

    body = base_request();
    body["messages"].push_back(
        Json{{"role", "assistant"},
             {"content", Json::array({Json{{"type", "refusal"}, {"refusal", "part"}}})},
             {"refusal", "top-level"}});
    const GenerationRequest refusal_history = parse(body).generation;
    const ChatTurn& refusal                 = refusal_history.messages.back();
    failures += check(refusal.content.size() == 2 && refusal.content[0].text == "part" &&
                          refusal.content[1].text == "top-level",
                      "assistant refusal history is preserved as assistant text");

    body = base_request();
    body["messages"].push_back(Json{{"role", "assistant"}});
    failures += check(parse(body).generation.messages.back().content.empty(),
                      "an empty assistant history turn is representable");

    body["messages"] = Json::array(
        {Json{{"role", "user"}, {"content", "run it"}},
         Json{{"role", "assistant"},
              {"content", nullptr},
              {"function_call", Json{{"name", "legacy"}, {"arguments", R"({"value":1})"}}}},
         Json{{"role", "function"}, {"name", "legacy"}, {"content", "done"}}});
    const GenerationRequest legacy = parse(body).generation;
    failures += check(legacy.messages[1].tool_calls.size() == 1 &&
                          legacy.messages[1].tool_calls[0].name == "legacy" &&
                          legacy.messages[2].role == ninfer::ChatRole::Tool,
                      "legacy function-call history lowers to Engine tool history");

    body["messages"] = Json::array(
        {Json{{"role", "assistant"},
              {"content", nullptr},
              {"tool_calls",
               Json::array({Json{{"id", ""},
                                 {"type", "function"},
                                 {"function", Json{{"name", "weather"}, {"arguments", "{}"}}}}})}},
         Json{{"role", "tool"}, {"tool_call_id", ""}, {"content", "done"}}});
    failures += check(parse(body).generation.has_tool_history(),
                      "string tool-call identifiers may be empty without changing history");
    return failures;
}

int test_reasoning_and_extensions() {
    int failures = 0;
    Json body    = base_request();
    body["messages"].push_back(Json{{"role", "assistant"},
                                    {"content", "answer"},
                                    {"reasoning_content", "thought"},
                                    {"reasoning", "thought"}});
    failures += check(parse(body).generation.messages.back().reasoning_content == "thought",
                      "assistant reasoning aliases normalize");
    body["messages"].back()["reasoning"] = "different";
    failures += check(api_error([&] { (void)parse(body); }).code == "conflicting_template_option",
                      "conflicting assistant reasoning aliases rejected");
    body["messages"].back()["reasoning_content"] = "";
    failures += check(parse(body).generation.messages.back().reasoning_content == "different",
                      "an empty reasoning alias does not conflict with a meaningful alias");
    body = base_request();
    body["messages"].push_back(Json{
        {"role", "assistant"}, {"content", nullptr}, {"reasoning_content", "unfinished thought"}});
    failures +=
        check(parse(body).generation.messages.back().reasoning_content == "unfinished thought",
              "reasoning-only assistant history is preserved");

    body                         = base_request();
    body["enable_thinking"]      = true;
    body["preserve_thinking"]    = false;
    body["chat_template_kwargs"] = Json{{"enable_thinking", true}, {"preserve_thinking", false}};
    const GenerationRequest normalized = parse(body).generation;
    failures += check(normalized.enable_thinking == true && normalized.preserve_thinking == false,
                      "Qwen/vLLM template aliases normalize");
    body["chat_template_kwargs"]["enable_thinking"] = false;
    failures += check(api_error([&] { (void)parse(body); }).code == "conflicting_template_option",
                      "conflicting thinking aliases rejected");
    body                         = base_request();
    body["chat_template_kwargs"] = Json{{"future", 1}};
    failures +=
        check(Json::parse(parse(body).generation.chat_template_kwargs_json).at("future") == 1,
              "custom template keyword did not survive protocol parsing");
    body["chat_template_kwargs"] = Json{{"future", nullptr}};
    failures += check(parse(body).generation.messages.size() == 1,
                      "null unknown template option is neutral");

    body                        = base_request();
    body["repetition_penalty"]  = 1.0;
    body["mm_processor_kwargs"] = Json{{"max_pixels", nullptr}};
    failures +=
        check(parse(body).generation.messages.size() == 1, "neutral ecosystem defaults accepted");
    body["repetition_penalty"] = 1.1;
    failures +=
        check(api_error([&] { (void)parse(body); }).code == "repetition_penalty_not_supported",
              "non-neutral repetition penalty rejected");
    body                        = base_request();
    body["mm_processor_kwargs"] = Json{{"max_pixels", 100}};
    failures +=
        check(api_error([&] { (void)parse(body); }).code == "mm_processor_kwargs_not_supported",
              "non-empty media processor kwargs rejected");

    // Request efforts collapse onto the template's three rungs; Claude Code sends 'high'.
    body                     = base_request();
    body["reasoning_effort"] = "high";
    ResolvedPromptSemantics resolved = semantics(parse(body).generation);
    failures += check(resolved.reasoning_effort == ninfer::ReasoningEffort::XHigh &&
                          resolved.enable_thinking == true,
                      "request effort high did not select the xhigh rung");
    body["reasoning_effort"] = "minimal";
    failures += check(semantics(parse(body).generation).reasoning_effort ==
                          ninfer::ReasoningEffort::Low,
                      "request effort minimal did not select the low rung");

    // --reasoning-effort fills in for thinking requests that state no effort, collapsed alike.
    ServeOptions server;
    server.default_reasoning_effort = RequestedReasoningEffort::Max;
    const GenerationRequest plain   = parse(base_request()).generation;
    resolved                        = resolve_prompt_semantics(plain, server);
    failures += check(resolved.reasoning_effort == ninfer::ReasoningEffort::XHigh &&
                          !resolved.enable_thinking.has_value(),
                      "server effort default did not apply to a request without one");
    server.default_reasoning_effort = RequestedReasoningEffort::Medium;
    body                            = base_request();
    body["reasoning_effort"]        = "low";
    failures += check(resolve_prompt_semantics(parse(body).generation, server).reasoning_effort ==
                          ninfer::ReasoningEffort::Low,
                      "request effort did not override the server default");
    body                         = base_request();
    body["chat_template_kwargs"] = Json{{"reasoning_effort", "xhigh"}};
    failures += check(resolve_prompt_semantics(parse(body).generation, server).reasoning_effort ==
                          ninfer::ReasoningEffort::XHigh,
                      "template-kwargs effort did not override the server default");
    body                    = base_request();
    body["enable_thinking"] = false;
    resolved                = resolve_prompt_semantics(parse(body).generation, server);
    failures += check(!resolved.reasoning_effort && resolved.enable_thinking == false,
                      "server effort default applied to a non-thinking request");
    server.enable_thinking = false;
    resolved               = resolve_prompt_semantics(plain, server);
    failures += check(!resolved.reasoning_effort && resolved.enable_thinking == false,
                      "server effort default overrode --no-thinking");
    body                     = base_request();
    body["reasoning_effort"] = "medium";
    resolved                 = resolve_prompt_semantics(parse(body).generation, server);
    failures += check(resolved.reasoning_effort == ninfer::ReasoningEffort::Medium &&
                          resolved.enable_thinking == true,
                      "request effort under --no-thinking did not enable thinking");
    return failures;
}

int test_stops_and_ranges() {
    int failures                            = 0;
    Json body                               = base_request();
    body["stop"]                            = Json::array({"A", "B"});
    const ninfer::RequestOptions translated = options(parse(body).generation);
    failures += check(translated.stop.strings.size() == 4,
                      "each stop string applies to Content and Reasoning");
    failures += check(translated.stop.strings[0].channel == ninfer::OutputChannel::Content &&
                          translated.stop.strings[1].channel == ninfer::OutputChannel::Reasoning,
                      "stop channel ordering is explicit");
    failures += check(translated.stop.include_model_defaults,
                      "checkpoint stop tokens apply when ignore_eos is absent");

    body["ignore_eos"]                       = true;
    const ninfer::RequestOptions ignore_eos = options(parse(body).generation);
    failures += check(!ignore_eos.stop.include_model_defaults,
                      "ignore_eos suppresses the checkpoint's stop tokens");
    failures += check(ignore_eos.stop.strings.size() == 4,
                      "ignore_eos keeps caller-supplied stop strings");
    body.erase("stop");
    failures += check(parse(body).generation.ignore_eos, "ignore_eos parses without a stop field");
    body["ignore_eos"] = "true";
    failures += check(api_error([&] { (void)parse(body); }).param == "ignore_eos",
                      "non-boolean ignore_eos rejected");
    body.erase("ignore_eos");

    body["stop"] = Json::array({"1", "2", "3", "4", "5"});
    failures += check(api_error([&] { (void)parse(body); }).param == "stop",
                      "more than four stop strings rejected");
    body["stop"] = "";
    failures +=
        check(api_error([&] { (void)parse(body); }).param == "stop", "empty stop string rejected");

    body                                  = base_request();
    body["top_k"]                         = 21;
    const GenerationRequest invalid_top_k = parse(body).generation;
    failures += check(api_error([&] { (void)options(invalid_top_k); }).param == "top_k",
                      "Engine translator owns sampler value range");
    body["top_k"]                         = 5;
    body["min_p"]                         = 1.1;
    const GenerationRequest invalid_min_p = parse(body).generation;
    failures += check(api_error([&] { (void)options(invalid_min_p); }).param == "min_p",
                      "min_p range enforced by common Engine translator");
    return failures;
}

GenerationOutcome sample_outcome() {
    GenerationOutcome outcome;
    outcome.text                                = "answer";
    outcome.reasoning                           = "thought";
    outcome.prompt_tokens                       = 20;
    outcome.completion_tokens                   = 7;
    outcome.reasoning_tokens                    = 3;
    outcome.finish_reason                       = ninfer::FinishReason::StopToken;
    outcome.metrics.prefix_cache_hit_tokens     = 12;
    outcome.metrics.prompt_wall_seconds         = 0.04;
    outcome.metrics.generation_wall_seconds     = 0.03;
    outcome.metrics.speculative_draft_tokens    = 9;
    outcome.metrics.speculative_accepted_tokens = 6;
    return outcome;
}

OpenAIChatResponseIdentity identity() {
    return OpenAIChatResponseIdentity{.id = "chatcmpl-test", .model = "qwen", .created = 42};
}

int test_aggregate_response() {
    int failures              = 0;
    GenerationOutcome outcome = sample_outcome();
    Json response             = Json::parse(make_chat_completion_response(identity(), outcome));
    failures += check(response["choices"][0]["message"]["content"] == "answer" &&
                          response["choices"][0]["message"]["reasoning_content"] == "thought" &&
                          response["choices"][0]["message"]["refusal"].is_null(),
                      "aggregate response separates reasoning and content");
    failures += check(response["choices"][0]["logprobs"].is_null(),
                      "aggregate choice carries nullable logprobs");
    failures += check(response["usage"]["prompt_tokens_details"]["cached_tokens"] == 12 &&
                          response["usage"]["completion_tokens_details"]["reasoning_tokens"] == 3,
                      "aggregate usage exposes cache hits and reasoning tokens");
    failures += check(
        response["timings"]["cache_n"] == 12 && response["timings"]["prompt_n"] == 8 &&
            response["timings"]["prompt_ms"] == 40.0 &&
            response["timings"]["prompt_per_second"] == 200.0 &&
            response["timings"]["predicted_n"] == 7 &&
            response["timings"]["predicted_ms"] == 30.0 &&
            response["timings"]["predicted_per_second"] == 200.0 &&
            response["timings"]["draft_n"] == 9 && response["timings"]["draft_n_accepted"] == 6,
        "aggregate timings use exact cache and N-1 generation intervals");
    failures += check(!response.contains("id_slot") && !response.contains("session_digest"),
                      "aggregate response advertised a slot identity");

    outcome.text.clear();
    outcome.tool_calls.push_back(ninfer::GeneratedToolCall{
        .name = "Edit",
        .arguments_json =
            R"({"file_path":"/tmp/probe.cpp","old_string":"old","new_string":"new"})"});
    response         = Json::parse(make_chat_completion_response(identity(), outcome));
    const Json& call = response["choices"][0]["message"]["tool_calls"][0];
    failures += check(response["choices"][0]["finish_reason"] == "tool_calls" &&
                          response["choices"][0]["message"]["content"].is_null(),
                      "aggregate tool call has OpenAI terminal shape");
    failures += check(
        call["id"].get<std::string>().starts_with("call_") && call["function"]["name"] == "Edit" &&
            !Json::parse(call["function"]["arguments"].get<std::string>()).contains("replace_all"),
        "OpenAI adapter owns wire tool-call identifiers");
    outcome.finish_reason = ninfer::FinishReason::OutputLimit;
    outcome.constraint    = ninfer::ConstraintObservation{
           .branch = ninfer::ConstraintOutputBranch::Tools, .complete = true, .terminated = false};
    response = Json::parse(make_chat_completion_response(identity(), outcome));
    failures += check(response["choices"][0]["finish_reason"] == "length" &&
                          response["choices"][0]["message"]["tool_calls"].size() == 1 &&
                          response["constraint"]["complete"] == true &&
                          response["constraint"]["terminated"] == false,
                      "length limit lost a completed call or hid the interruption");
    return failures;
}

int test_stream_response() {
    int failures = 0;
    OpenAIChatStream stream(identity(), true);
    Json role = parse_sse(stream.start());
    failures += check(role["choices"][0]["delta"]["role"] == "assistant" &&
                          role["choices"][0]["logprobs"].is_null() && role["usage"].is_null(),
                      "stream starts with role, nullable logprobs, and null usage");
    Json reasoning = parse_sse(stream.reasoning_delta("thought"));
    Json content   = parse_sse(stream.content_delta("ans"));
    failures += check(reasoning["choices"][0]["delta"]["reasoning_content"] == "thought" &&
                          content["choices"][0]["delta"]["content"] == "ans",
                      "stream separates reasoning and content deltas");

    GenerationOutcome outcome = sample_outcome();
    outcome.constraint        = ninfer::ConstraintObservation{
               .branch = ninfer::ConstraintOutputBranch::Content, .complete = true, .terminated = true};
    const std::vector<std::string> events = stream.finish(outcome);
    failures +=
        check(events.size() == 4, "finish emits buffered suffix, terminal, usage, and done");
    failures += check(parse_sse(events[0])["choices"][0]["delta"]["content"] == "wer",
                      "terminal content suffix is emitted exactly once");
    failures += check(parse_sse(events[1])["choices"][0]["finish_reason"] == "stop",
                      "stream terminal finish reason emitted");
    const Json usage = parse_sse(events[2]);
    failures += check(usage["choices"].empty() &&
                          usage["usage"]["prompt_tokens_details"]["cached_tokens"] == 12 &&
                          usage["usage"]["completion_tokens_details"]["reasoning_tokens"] == 3 &&
                          usage["timings"]["predicted_n"] == 7,
                      "dedicated stream usage carries token accounting and terminal timings");
    failures += check(events.back() == "data: [DONE]\n\n", "stream ends with DONE sentinel");
    failures += check(usage["constraint"]["branch"] == "content" &&
                          usage["constraint"]["terminated"] == true &&
                          !parse_sse(events[0]).contains("constraint") &&
                          !parse_sse(events[1]).contains("constraint"),
                      "Chat constraint state must appear once, with terminal usage");

    OpenAIChatStream without_usage(identity(), false);
    (void)without_usage.start();
    const std::vector<std::string> plain_events = without_usage.finish(sample_outcome());
    const Json plain_finish = parse_sse(plain_events[plain_events.size() - 2]);
    failures += check(plain_finish["choices"][0]["finish_reason"] == "stop" &&
                          !plain_finish.contains("id_slot") &&
                          !plain_finish.contains("session_digest"),
                      "stream finish chunk advertised a slot identity");

    OpenAIChatStream mismatch(identity(), false);
    (void)mismatch.start();
    (void)mismatch.content_delta("different");
    failures += check(throws_logic([&] { (void)mismatch.finish(outcome); }),
                      "stream encoder rejects terminal/content divergence");

    OpenAIChatStream tool_stream(identity(), false);
    (void)tool_stream.start();
    GenerationOutcome tool_outcome;
    tool_outcome.tool_calls.push_back(ninfer::GeneratedToolCall{
        .name = "Edit", .arguments_json = R"({"file_path":"/tmp/probe.cpp"})"});
    tool_outcome.finish_reason                 = ninfer::FinishReason::StopToken;
    const std::vector<std::string> tool_events = tool_stream.finish(tool_outcome);
    const Json tool_delta                      = parse_sse(tool_events[0]);
    failures += check(
        tool_delta["choices"][0]["delta"]["tool_calls"][0]["id"].get<std::string>().starts_with(
            "call_") &&
            tool_delta["choices"][0]["delta"]["tool_calls"][0]["function"]["name"] == "Edit" &&
            parse_sse(tool_events[1])["choices"][0]["finish_reason"] == "tool_calls",
        "stream encoder owns stable OpenAI tool-call shape");
    OpenAIChatStream interrupted(identity(), false);
    (void)interrupted.start();
    tool_outcome.finish_reason = ninfer::FinishReason::OutputLimit;
    const auto partial_events  = interrupted.finish(tool_outcome);
    failures += check(parse_sse(partial_events[1])["choices"][0]["finish_reason"] == "length",
                      "streamed completed call hid a later truncation");
    return failures;
}

GenerationOutcome custom_call_outcome(const std::string& patch) {
    GenerationOutcome outcome;
    outcome.finish_reason = ninfer::FinishReason::StopToken;
    outcome.tool_calls.push_back(ninfer::GeneratedToolCall{
        .name = "apply_patch", .arguments_json = custom_tool_arguments_json(patch)});
    outcome.tool_calls.push_back(
        ninfer::GeneratedToolCall{.name = "shell", .arguments_json = R"({"cmd":"ls"})"});
    return outcome;
}

int test_custom_tool_responses() {
    int failures = 0;
    // A Codex-style patch: leading spaces, blank lines, quotes and the *** markers survive.
    const std::string patch = "*** Begin Patch\n*** Update File: a.py\n@@ def f():\n"
                              "-    x = \"1\"\n+    x = \"2\"\n \n+\n     return x\n*** End Patch";
    const std::unordered_set<std::string> custom_tools{"apply_patch"};
    const GenerationOutcome outcome = custom_call_outcome(patch);

    const Json aggregate =
        Json::parse(make_chat_completion_response(identity(), outcome, custom_tools));
    const Json& message = aggregate["choices"][0]["message"];
    const Json& custom  = message["tool_calls"][0];
    failures += check(
        aggregate["choices"][0]["finish_reason"] == "tool_calls" && custom.size() == 3 &&
            custom["type"] == "custom" && custom["id"].get<std::string>().starts_with("call_") &&
            custom["custom"].size() == 2 && custom["custom"]["name"] == "apply_patch" &&
            custom["custom"]["input"].get<std::string>() == patch && !custom.contains("function"),
        "an aggregate custom call is {id, type:custom, custom:{name, input}}");
    failures += check(message["tool_calls"][1]["type"] == "function" &&
                          message["tool_calls"][1]["function"]["arguments"] == R"({"cmd":"ls"})",
                      "a function call beside a custom call keeps the function shape");
    failures += check(Json::parse(make_chat_completion_response(
                          identity(), outcome))["choices"][0]["message"]["tool_calls"][0]["type"] ==
                          "function",
                      "without declared custom tools every call is a function call");

    OpenAIChatStream stream(identity(), false, false, false, custom_tools);
    (void)stream.start();
    const std::vector<std::string> events = stream.finish(outcome);
    std::string streamed_input;
    Json first_custom_delta;
    for (const std::string& event : events) {
        if (event == "data: [DONE]\n\n") { continue; }
        const Json payload = parse_sse(event);
        const Json& delta  = payload["choices"][0]["delta"];
        if (!delta.contains("tool_calls")) { continue; }
        for (const Json& call : delta["tool_calls"]) {
            if (call["index"] != 0) { continue; }
            if (first_custom_delta.is_null()) { first_custom_delta = call; }
            streamed_input += call["custom"]["input"].get<std::string>();
        }
    }
    failures +=
        check(first_custom_delta["index"] == 0 && first_custom_delta["type"] == "custom" &&
                  first_custom_delta["id"].get<std::string>().starts_with("call_") &&
                  first_custom_delta["custom"]["name"] == "apply_patch" &&
                  !first_custom_delta.contains("function"),
              "a streamed custom call delta is {index, id, type:custom, custom:{name, input}}");
    failures += check(streamed_input == patch &&
                          streamed_input == custom["custom"]["input"].get<std::string>(),
                      "streamed custom inputs concatenate to the aggregate input byte for byte");
    failures +=
        check(parse_sse(events[events.size() - 2])["choices"][0]["finish_reason"] == "tool_calls",
              "a streamed custom call finishes with tool_calls");

    // History round trip: the aggregate answer, sent back as assistant history, lowers to the
    // exact Engine call that produced it.
    Json history     = base_request();
    history["tools"] = Json::array({custom_tool(), function_tool("shell")});
    Json assistant   = message;
    assistant.erase("reasoning_content");
    history["messages"] =
        Json::array({Json{{"role", "user"}, {"content", "fix it"}}, assistant,
                     Json{{"role", "tool"}, {"tool_call_id", custom["id"]}, {"content", "Done!"}},
                     Json{{"role", "tool"},
                          {"tool_call_id", message["tool_calls"][1]["id"]},
                          {"content", "a"}}});
    const GenerationRequest replayed = parse(history).generation;
    const auto& calls                = replayed.messages.at(1).tool_calls;
    failures += check(calls.size() == 2 && calls[0].name == "apply_patch" &&
                          calls[0].id == custom["id"].get<std::string>() &&
                          calls[0].arguments_json == outcome.tool_calls[0].arguments_json &&
                          calls[1].arguments_json == outcome.tool_calls[1].arguments_json &&
                          replayed.has_tool_history(),
                      "custom call history lowers back to the Engine call it came from");
    const ninfer::PromptInput replayed_prompt = prompt(replayed);
    failures += check(replayed_prompt.messages.size() == 4 &&
                          replayed_prompt.messages[1].tool_calls.size() == 2 &&
                          replayed_prompt.messages[1].tool_calls[0].arguments_json ==
                              outcome.tool_calls[0].arguments_json,
                      "custom call history reaches PromptInput as the Engine call");

    auto history_error = [&](Json call) {
        Json invalid                         = history;
        invalid["messages"][1]["tool_calls"] = Json::array({std::move(call)});
        invalid["messages"].erase(3);
        return api_error([&] { (void)parse(invalid); });
    };
    ApiError error = history_error(
        Json{{"id", "c1"}, {"type", "custom"}, {"custom", Json{{"name", "apply_patch"}}}});
    failures += check(error.param == "messages", "a custom history call without input");
    error =
        history_error(Json{{"id", "c1"},
                           {"type", "custom"},
                           {"custom", Json{{"name", "apply_patch"}, {"input", Json::object()}}}});
    failures += check(error.param == "messages", "a custom history call with non-string input");
    error = history_error(Json{
        {"id", "c1"}, {"type", "custom"}, {"custom", Json{{"name", "shell"}, {"input", "ls"}}}});
    failures += check(error.code == "invalid_tool_history",
                      "a custom history call naming a declared function");
    error = history_error(Json{{"id", "c1"},
                               {"type", "function"},
                               {"function", Json{{"name", "apply_patch"}, {"arguments", "{}"}}}});
    failures += check(error.code == "invalid_tool_history",
                      "a function history call naming a declared custom tool");
    error = history_error(Json{{"id", "c1"}, {"type", "mcp"}});
    failures += check(error.code == "tool_type_not_supported",
                      "a history call of another type stays rejected");

    // Strict lowering guarantees exactly one string `input`; anything else is a server bug and
    // must not reach the client as a patch.
    GenerationOutcome broken            = outcome;
    broken.tool_calls[0].arguments_json = R"({"input":"x","extra":1})";
    failures +=
        check(throws_logic(
                  [&] { (void)make_chat_completion_response(identity(), broken, custom_tools); }),
              "an aggregate custom call without exactly one string input is an internal error");
    OpenAIChatStream broken_stream(identity(), false, false, false, custom_tools);
    (void)broken_stream.start();
    failures +=
        check(throws_logic([&] { (void)broken_stream.finish(broken); }),
              "a streamed custom call without exactly one string input is an internal error");
    return failures;
}

int test_stream_observations() {
    int failures = 0;
    OpenAIChatStream stream(identity(), true, true, true);
    const Json role = parse_sse(stream.start());
    failures += check(!role.contains("timings") && !role.contains("prompt_progress"),
                      "transport role chunk precedes Engine observations");

    stream.note_start(
        ninfer::GenerationStart{.prompt = {.prompt_tokens = 32}, .reused_prompt_tokens = 12});
    const Json initial = parse_sse(stream.initial_prompt_progress());
    failures +=
        check(initial["choices"][0]["delta"].empty() && initial["prompt_progress"]["total"] == 32 &&
                  initial["prompt_progress"]["cache"] == 12 &&
                  initial["prompt_progress"]["processed"] == 12 &&
                  initial["prompt_progress"]["time_ms"] == 0,
              "initial prompt progress begins at the admitted cache frontier");

    const Json middle = parse_sse(stream.prompt_progress(ninfer::PromptProgress{
        .total_prompt_tokens     = 32,
        .reused_prompt_tokens    = 12,
        .processed_prompt_tokens = 20,
        .elapsed_ns              = 57000000,
    }));
    failures += check(middle["prompt_progress"]["processed"] == 20 &&
                          middle["prompt_progress"]["time_ms"] == 57,
                      "prompt progress exposes a cumulative completed frontier");
    const Json complete = parse_sse(stream.prompt_progress(ninfer::PromptProgress{
        .total_prompt_tokens     = 32,
        .reused_prompt_tokens    = 12,
        .processed_prompt_tokens = 32,
        .elapsed_ns              = 100000000,
    }));
    failures +=
        check(complete["prompt_progress"]["processed"] == complete["prompt_progress"]["total"],
              "final prompt progress reaches the complete prompt");

    stream.note_timing(ninfer::GenerationTimingObservation{
        .generated_tokens = 1, .prompt_elapsed_ns = 110000000, .generation_elapsed_ns = 0});
    stream.note_timing(ninfer::GenerationTimingObservation{
        .generated_tokens      = 3,
        .prompt_elapsed_ns     = 110000000,
        .generation_elapsed_ns = 20000000,
    });
    const Json content = parse_sse(stream.content_delta("answer"));
    failures +=
        check(content["timings"]["prompt_n"] == 20 && content["timings"]["predicted_n"] == 3 &&
                  content["timings"]["predicted_per_second"] == 100.0,
              "visible output uses the latest independent commit observation");

    GenerationOutcome outcome = sample_outcome();
    outcome.reasoning.clear();
    const std::vector<std::string> terminal = stream.finish(outcome);
    failures += check(parse_sse(terminal[1])["timings"]["predicted_n"] == 7,
                      "terminal usage replaces live timing with exact final accounting");
    return failures;
}

int test_common_objects() {
    int failures      = 0;
    const ModelDescription text_only{.id = "qwen", .max_model_len = 240000, .vision = false};
    const Json models = Json::parse(make_models_list(text_only, 7));
    failures +=
        check(models["data"][0]["id"] == "qwen" && models["data"][0]["max_model_len"] == 240000 &&
                  models["data"][0]["context_window"] == 240000 &&
                  models["data"][0]["context_length"] == 240000,
              "models list advertises the configured context limit under every field name");
    failures += check(models["data"][0]["architecture"] ==
                          Json{{"input_modalities", Json::array({"text"})},
                               {"output_modalities", Json::array({"text"})}},
                      "a text-only server advertised media input");
    const Json model = Json::parse(make_model_object(text_only, 7));
    failures += check(model["max_model_len"] == 240000 && model["context_window"] == 240000 &&
                          model["context_length"] == 240000,
                      "model lookup advertises the configured context limit under every field name");

    const ModelDescription vision{.id = "qwen", .max_model_len = 240000, .vision = true};
    const Json vision_input = Json::array({"text", "image", "video"});
    failures += check(Json::parse(make_models_list(vision, 7))["data"][0]["architecture"]
                                                                  ["input_modalities"] ==
                              vision_input &&
                          Json::parse(make_model_object(vision, 7))["architecture"]
                                                                    ["input_modalities"] ==
                              vision_input,
                      "a --vision server did not advertise image and video input");
    const Json error = Json::parse(make_error_body(
        ApiError{.status = 400, .message = "bad", .param = "messages", .code = "invalid"}));
    failures += check(error["error"]["param"] == "messages" && error["error"]["code"] == "invalid",
                      "OpenAI common error shape remains stable");
    return failures;
}

int test_thinking_budget_extension() {
    Json body    = base_request();
    int failures = check(!parse(body).generation.thinking_budget,
                         "absent thinking_budget was not left unset");
    body["thinking_budget"] = nullptr;
    failures += check(!parse(body).generation.thinking_budget,
                      "null thinking_budget was not left unset");
    body["thinking_budget"] = 512;
    failures += check(parse(body).generation.thinking_budget == 512U,
                      "thinking_budget was not parsed");
    body["thinking_budget"] = 1;
    failures += check(parse(body).generation.thinking_budget == 1U,
                      "the smallest positive thinking_budget was rejected");
    for (const Json& invalid : {Json(0), Json(-5), Json("512"), Json(1.5), Json(true)}) {
        body["thinking_budget"] = invalid;
        failures += check(api_error([&] { (void)parse(body); }).param == "thinking_budget",
                          "invalid thinking_budget was accepted: " + invalid.dump());
    }
    return failures;
}

int test_graft_extension() {
    Json body       = base_request();
    int failures    = check(!parse(body).generation.graft, "absent graft was not left unset");
    body["graft"]   = nullptr;
    failures       += check(!parse(body).generation.graft, "null graft was not left unset");
    body["graft"]   = "";
    failures       += check(parse(body).generation.graft == "",
                            "empty graft did not explicitly select none");
    body["graft"]   = "product";
    failures       += check(parse(body).generation.graft == "product", "graft name was not parsed");
    body["graft"]   = 5;
    failures       += check(api_error([&] { (void)parse(body); }).param == "graft",
                            "non-string graft was accepted");
    return failures;
}

} // namespace

int main() {
    int failures = 0;
    failures += test_graft_extension();
    failures += test_thinking_budget_extension();
    failures += test_request_envelope_and_sampling();
    failures += test_standard_field_policy();
    failures += test_ignore_eos_with_constraints();
    failures += test_prompt_cache_boundaries();
    failures += test_constrained_decoding_extensions();
    failures += test_tools();
    failures += test_custom_tools();
    failures += test_messages_and_media();
    failures += test_reasoning_and_extensions();
    failures += test_stops_and_ranges();
    failures += test_aggregate_response();
    failures += test_stream_response();
    failures += test_custom_tool_responses();
    failures += test_stream_observations();
    failures += test_common_objects();
    if (failures == 0) { std::cout << "OpenAI Chat protocol tests passed\n"; }
    return failures == 0 ? 0 : 1;
}
