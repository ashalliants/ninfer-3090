#include "models/qwen3_5/frontend/tool_call_parser.h"

#include <nlohmann/json.hpp>

#include <initializer_list>
#include <iostream>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

using Json   = nlohmann::json;
namespace fi = ninfer::models::qwen3_5::frontend;

int fail(const std::string& message) {
    std::cerr << "FAIL: " << message << '\n';
    return 1;
}

int check(bool condition, const std::string& message) { return condition ? 0 : fail(message); }

std::string tool_definition(const std::string& tool_name, Json properties,
                            Json required = Json::array()) {
    Json parameters{{"type", "object"}, {"properties", std::move(properties)}};
    if (!required.empty()) { parameters["required"] = std::move(required); }
    return Json{{"type", "function"},
                {"function", Json{{"name", tool_name}, {"parameters", std::move(parameters)}}}}
        .dump();
}

std::shared_ptr<const fi::ToolCallOutputContract>
contract_from_definitions(const std::vector<std::string>& definitions) {
    return fi::build_tool_call_output_contract(definitions);
}

std::shared_ptr<const fi::ToolCallOutputContract> output_contract_for(const std::string& tool_name,
                                                                      Json properties) {
    const std::vector<std::string> definitions = {
        tool_definition(tool_name, std::move(properties))};
    return contract_from_definitions(definitions);
}

fi::ToolCallOutputContract contract_for(const std::string& tool_name, Json properties) {
    return *output_contract_for(tool_name, std::move(properties));
}

std::string
tool_call(std::string_view tool_name,
          std::initializer_list<std::pair<std::string_view, std::string_view>> parameters = {}) {
    std::string text = "<tool_call>\n<function=";
    text.append(tool_name);
    text += ">\n";
    for (const auto& [name, value] : parameters) {
        text += "<parameter=";
        text.append(name);
        text += ">\n";
        text.append(value);
        text += "\n</parameter>\n";
    }
    text += "</function>\n</tool_call>";
    return text;
}

int check_rejected(const std::string& text, const fi::ToolCallOutputContract& contract,
                   ninfer::ToolCallParseFallbackReason reason, std::string_view message) {
    const auto parsed = fi::parse_qwen_tool_call_output(text, 64, contract);
    return check(!parsed.is_tool_call_response && parsed.content == text &&
                     parsed.tool_calls.empty() && parsed.diagnostics.marker_seen &&
                     parsed.diagnostics.fallback_reason == reason,
                 std::string(message));
}

int check_parameter_schema_mismatch(const fi::ToolCallOutputContract& contract,
                                    std::string_view parameter_name, std::string_view value,
                                    std::string_view expected_json_value,
                                    std::string_view message) {
    const auto parsed = fi::parse_qwen_tool_call_output(
        tool_call("configure", {{parameter_name, value}}), 64, contract);
    const std::string expected_arguments = "{" + Json(std::string(parameter_name)).dump() + ":" +
                                           std::string(expected_json_value) + "}";
    return check(
        parsed.is_tool_call_response && parsed.content.empty() && parsed.tool_calls.size() == 1 &&
            parsed.tool_calls.front().arguments_json == expected_arguments &&
            parsed.diagnostics.marker_seen && parsed.diagnostics.structured_call_count == 1 &&
            parsed.diagnostics.schema_mismatch_arguments == 1 &&
            parsed.diagnostics.fallback_reason == ninfer::ToolCallParseFallbackReason::None,
        std::string(message));
}

int test_untyped_parameters() {
    const auto contract = contract_for("get_weather", Json::object());
    const auto parsed   = fi::parse_qwen_tool_call_output("Calling weather.\n"
                                                            "<tool_call>\n"
                                                            "<function=get_weather>\n"
                                                            "<parameter=city>\nParis\n</parameter>\n"
                                                            "<parameter=days>\n2\n</parameter>\n"
                                                            "</function>\n"
                                                            "</tool_call>",
                                                          64, contract);

    int failures = 0;
    failures += check(parsed.is_tool_call_response, "untyped call was not parsed");
    failures += check(parsed.content == "Calling weather.", "content prefix was not trimmed");
    failures += check(parsed.tool_calls.size() == 1, "untyped call count changed");
    if (parsed.tool_calls.size() != 1) { return failures; }
    failures += check(parsed.tool_calls.front().name == "get_weather", "function name changed");
    const Json args = Json::parse(parsed.tool_calls.front().arguments_json);
    failures += check(args.at("city") == "Paris", "untyped string inference changed");
    failures += check(args.at("days") == 2, "untyped JSON inference changed");
    return failures;
}

int test_multiple_calls() {
    const auto contract = contract_from_definitions(
        {tool_definition("first", Json::object()), tool_definition("second", Json::object())});
    const std::string text = tool_call("first", {{"payload", "{\"ok\":true,\"items\":[1,2]}"}}) +
                             "\n" + tool_call("second", {{"value", "plain text"}});
    const auto parsed = fi::parse_qwen_tool_call_output(text, 64, *contract);

    int failures = 0;
    failures += check(parsed.is_tool_call_response && parsed.tool_calls.size() == 2,
                      "multiple complete calls were not parsed");
    if (parsed.tool_calls.size() != 2) { return failures; }
    const Json first  = Json::parse(parsed.tool_calls[0].arguments_json);
    const Json second = Json::parse(parsed.tool_calls[1].arguments_json);
    failures +=
        check(first.at("payload").at("ok") == true && first.at("payload").at("items").at(1) == 2,
              "legacy object value changed");
    failures += check(second.at("value") == "plain text", "legacy plain text value changed");
    return failures;
}

int test_declared_strings_preserve_text() {
    const auto contract =
        contract_for("TaskUpdate",
                     Json{{"taskId", Json{{"type", "string"}}},
                          {"content", Json{{"type", "string"}}},
                          {"truthy", Json{{"type", "string"}}},
                          {"nullish", Json{{"type", "string"}}},
                          {"quoted", Json{{"type", "string"}}},
                          {"windows", Json{{"type", "string"}}},
                          {"string_or_number", Json{{"type", Json::array({"number", "string"})}}}});
    const auto parsed =
        fi::parse_qwen_tool_call_output("<tool_call>\n"
                                        "<function=TaskUpdate>\n"
                                        "<parameter=taskId>\n1\n</parameter>\n"
                                        "<parameter=content>\n  {\"x\":1}\n\n</parameter>\n"
                                        "<parameter=truthy>\ntrue\n</parameter>\n"
                                        "<parameter=nullish>\nnull\n</parameter>\n"
                                        "<parameter=quoted>\n\"literal\"\n</parameter>\n"
                                        "<parameter=windows>\r\n  value  \r\n</parameter>\n"
                                        "<parameter=string_or_number>\n7\n</parameter>\n"
                                        "</function>\n"
                                        "</tool_call>",
                                        128, contract);

    int failures = 0;
    failures += check(parsed.is_tool_call_response && parsed.tool_calls.size() == 1,
                      "declared string call was rejected");
    if (parsed.tool_calls.size() != 1) { return failures; }
    const Json args = Json::parse(parsed.tool_calls.front().arguments_json);
    failures += check(args.at("taskId") == "1", "numeric-shaped string was promoted");
    failures +=
        check(args.at("content") == "  {\"x\":1}\n", "string whitespace or content changed");
    failures += check(args.at("truthy") == "true" && args.at("nullish") == "null",
                      "boolean/null-shaped string was promoted");
    failures +=
        check(args.at("quoted") == "\"literal\"", "quoted string was reinterpreted as JSON");
    failures += check(args.at("windows") == "  value  ", "CRLF framing changed string content");
    failures +=
        check(args.at("string_or_number") == "7", "string-admitting union did not preserve text");
    return failures;
}

int test_string_values_preserve_embedded_tool_markup() {
    const auto contract       = contract_for("bash", Json{{"command", Json{{"type", "string"}}},
                                                          {"timeout", Json{{"type", "integer"}}}});
    const std::string command = "python3 - <<'PY'\n"
                                "import re\n"
                                "pattern = r'<parameter=edits>\\n(.*?)\\n</parameter>'\n"
                                "print(pattern)\n"
                                "PY";
    const std::string text    = tool_call("bash", {{"command", command}, {"timeout", "30"}});
    const auto parsed         = fi::parse_qwen_tool_call_output(text, 64, contract);

    int failures = 0;
    failures += check(parsed.is_tool_call_response && parsed.tool_calls.size() == 1,
                      "balanced parameter markup inside a string broke the tool call");
    if (parsed.tool_calls.size() != 1) { return failures; }
    const Json args = Json::parse(parsed.tool_calls.front().arguments_json);
    failures += check(args.at("command") == command,
                      "embedded parameter markup was removed from the string value");
    failures +=
        check(args.at("timeout") == 30, "sibling parameter after embedded markup was not parsed");

    const std::string nested_markup =
        "literal closes: </function> and </tool_call>\n"
        "<function=fake>body</function>\n"
        "<tool_call>body</tool_call>\n"
        "<parameter=outer>before<parameter=inner>value</parameter>after</parameter>";
    const std::string nested_text = tool_call("bash", {{"command", nested_markup}});
    const auto nested             = fi::parse_qwen_tool_call_output(nested_text, 64, contract);
    failures += check(nested.is_tool_call_response && nested.tool_calls.size() == 1,
                      "nested tool markup inside a string broke outer structure");
    if (nested.tool_calls.size() == 1) {
        const Json nested_args = Json::parse(nested.tool_calls.front().arguments_json);
        failures += check(nested_args.at("command") == nested_markup,
                          "nested function/tool/parameter markup was not preserved exactly");
    }
    return failures;
}

int test_parameter_delimiters_in_values() {
    using Reason        = ninfer::ToolCallParseFallbackReason;
    const auto contract = contract_for("bash", Json{{"command", Json{{"type", "string"}}},
                                                    {"timeout", Json{{"type", "integer"}}}});
    int failures        = 0;

    // A closer the format cannot continue from is text the value quotes: the call is kept whole,
    // whether the quote is followed by a sibling parameter or ends the call.
    for (const std::string command :
         {std::string("echo '</parameter>'"), std::string("echo '</parameter>' && ls\n</param>")}) {
        for (const bool sibling : {false, true}) {
            const std::string text =
                sibling ? tool_call("bash", {{"command", command}, {"timeout", "30"}})
                        : tool_call("bash", {{"command", command}});
            const auto parsed = fi::parse_qwen_tool_call_output(text, 64, contract);
            const std::string expected =
                "{\"command\":" + Json(command).dump() + (sibling ? ",\"timeout\":30}" : "}");
            failures += check(parsed.is_tool_call_response && parsed.content.empty() &&
                                  parsed.tool_calls.size() == 1 &&
                                  parsed.tool_calls.front().arguments_json == expected &&
                                  parsed.diagnostics.fallback_reason == Reason::None,
                              "a value quoting a parameter closer lost its call or its bytes");
        }
    }

    // Markup the format could continue from stays ambiguous, so those calls still fall back.
    failures += check_rejected(tool_call("bash", {{"command", "echo '<parameter=unterminated>'"}}),
                               contract, Reason::MalformedStructure,
                               "unbalanced nested parameter open was silently repaired");
    failures += check_rejected(tool_call("bash", {{"command", "echo '</parameter></function>'"}}),
                               contract, Reason::MalformedStructure,
                               "a closer followed by a function closer was guessed to be text");
    return failures;
}

int test_declared_json_types() {
    const auto contract = contract_for(
        "configure", Json{{"count", Json{{"type", "integer"}}},
                          {"total", Json{{"type", "number"}}},
                          {"ratio", Json{{"type", "number"}}},
                          {"enabled", Json{{"type", "boolean"}}},
                          {"payload", Json{{"type", "object"}}},
                          {"items", Json{{"type", "array"}}},
                          {"unset", Json{{"type", "null"}}},
                          {"optional", Json{{"type", Json::array({"integer", "null"})}}}});
    const std::string text = tool_call("configure", {{"count", "7"},
                                                     {"total", "8"},
                                                     {"ratio", "1.5"},
                                                     {"enabled", "true"},
                                                     {"payload", "{\"x\":1}"},
                                                     {"items", "[\"a\",2]"},
                                                     {"unset", "null"},
                                                     {"optional", "null"}});
    const auto parsed      = fi::parse_qwen_tool_call_output(text, 64, contract);

    int failures = 0;
    failures += check(parsed.is_tool_call_response && parsed.tool_calls.size() == 1,
                      "valid declared JSON values were rejected");
    if (parsed.tool_calls.size() != 1) { return failures; }
    const Json args = Json::parse(parsed.tool_calls.front().arguments_json);
    failures += check(args.at("count") == 7, "integer was not decoded");
    failures += check(args.at("total") == 8, "integer did not satisfy number");
    failures += check(args.at("ratio") == 1.5, "fractional number was not decoded");
    failures += check(args.at("enabled") == true, "JSON boolean was not decoded");
    failures += check(args.at("payload").is_object() && args.at("payload").at("x") == 1,
                      "object was not decoded");
    failures +=
        check(args.at("items").is_array() && args.at("items").at(1) == 2, "array was not decoded");
    failures += check(args.at("unset").is_null() && args.at("optional").is_null(),
                      "declared null was not decoded");
    return failures;
}

int test_boolean_boundary() {
    const auto contract =
        contract_for("configure", Json{{"lower_true", Json{{"type", "boolean"}}},
                                       {"title_true", Json{{"type", "boolean"}}},
                                       {"upper_true", Json{{"type", "boolean"}}},
                                       {"mixed_false", Json{{"type", "boolean"}}},
                                       {"spaced_true", Json{{"type", "boolean"}}},
                                       {"windows_false", Json{{"type", "boolean"}}}});
    const auto parsed =
        fi::parse_qwen_tool_call_output("<tool_call>\n"
                                        "<function=configure>\n"
                                        "<parameter=lower_true>\ntrue\n</parameter>\n"
                                        "<parameter=title_true>\nTrue\n</parameter>\n"
                                        "<parameter=upper_true>\nTRUE\n</parameter>\n"
                                        "<parameter=mixed_false>\nfAlSe\n</parameter>\n"
                                        "<parameter=spaced_true>\n \tTrUe \n</parameter>\n"
                                        "<parameter=windows_false>\r\nFaLsE\r\n</parameter>\n"
                                        "</function>\n"
                                        "</tool_call>",
                                        64, contract);

    int failures = 0;
    failures += check(parsed.is_tool_call_response && parsed.tool_calls.size() == 1,
                      "case-insensitive booleans were rejected");
    if (parsed.tool_calls.size() != 1) { return failures; }
    const Json args = Json::parse(parsed.tool_calls.front().arguments_json);
    failures += check(args.at("lower_true") == true && args.at("title_true") == true &&
                          args.at("upper_true") == true && args.at("spaced_true") == true,
                      "true variants were not canonicalized");
    failures += check(args.at("mixed_false") == false && args.at("windows_false") == false,
                      "false variants were not canonicalized");

    const auto one_flag = contract_for("configure", Json{{"flag", Json{{"type", "boolean"}}}});
    failures += check_parameter_schema_mismatch(one_flag, "flag", "1", "1",
                                                "integer boolean mismatch was not structured");
    failures += check_parameter_schema_mismatch(one_flag, "flag", "0", "0",
                                                "zero boolean mismatch was not structured");
    failures += check_parameter_schema_mismatch(one_flag, "flag", "\"true\"", "\"true\"",
                                                "string boolean mismatch was not structured");
    failures += check_parameter_schema_mismatch(one_flag, "flag", "yes", "\"yes\"",
                                                "plain boolean mismatch was not structured");
    failures += check_parameter_schema_mismatch(one_flag, "flag", "None", "\"None\"",
                                                "Python null mismatch was not structured");
    failures += check_parameter_schema_mismatch(one_flag, "flag", "null", "null",
                                                "null boolean mismatch was not structured");
    return failures;
}

int test_exact_integer_boundary() {
    const auto integer_contract =
        contract_for("configure", Json{{"decimal", Json{{"type", "integer"}}},
                                       {"exponent", Json{{"type", "integer"}}},
                                       {"scaled", Json{{"type", "integer"}}},
                                       {"negative_zero", Json{{"type", "integer"}}},
                                       {"large", Json{{"type", "integer"}}}});
    const std::string valid = tool_call("configure", {{"decimal", "7.0"},
                                                      {"exponent", "1e2"},
                                                      {"scaled", "100e-2"},
                                                      {"negative_zero", "-0.0"},
                                                      {"large", "9007199254740992.0"}});
    const auto parsed       = fi::parse_qwen_tool_call_output(valid, 64, integer_contract);

    int failures = 0;
    failures += check(parsed.is_tool_call_response && parsed.tool_calls.size() == 1,
                      "mathematically integral JSON numbers were rejected");
    if (parsed.tool_calls.size() == 1) {
        failures += check(parsed.tool_calls.front().arguments_json ==
                              "{\"decimal\":7.0,\"exponent\":1e2,\"scaled\":100e-2,"
                              "\"negative_zero\":-0.0,\"large\":9007199254740992.0}",
                          "integer JSON lexemes were rewritten");
    }

    const auto one_integer = contract_for("configure", Json{{"value", Json{{"type", "integer"}}}});
    failures += check_parameter_schema_mismatch(one_integer, "value", "7.5", "7.5",
                                                "fractional integer mismatch was not structured");
    failures += check_parameter_schema_mismatch(one_integer, "value", "1e-1", "1e-1",
                                                "fractional exponent mismatch was not structured");
    failures += check_parameter_schema_mismatch(
        one_integer, "value", "9007199254740992.5", "9007199254740992.5",
        "large fractional integer mismatch lost its exact lexeme");

    const auto one_number = contract_for("configure", Json{{"value", Json{{"type", "number"}}}});
    const std::string large_fraction = tool_call("configure", {{"value", "9007199254740992.5"}});
    const auto number_parsed = fi::parse_qwen_tool_call_output(large_fraction, 64, one_number);
    failures += check(number_parsed.is_tool_call_response && number_parsed.tool_calls.size() == 1 &&
                          number_parsed.tool_calls.front().arguments_json ==
                              "{\"value\":9007199254740992.5}",
                      "valid number was rejected or lost its original precision");
    return failures;
}

int test_composed_schema_types() {
    const auto contract = contract_for(
        "configure",
        Json{{"flag",
              Json{{"anyOf", Json::array({Json{{"type", "boolean"}}, Json{{"type", "null"}}})}}},
             {"unset",
              Json{{"oneOf", Json::array({Json{{"type", "null"}}, Json{{"type", "boolean"}}})}}},
             {"count",
              Json{{"anyOf", Json::array({Json{{"type", "integer"}}, Json{{"type", "null"}}})}}},
             {"nested",
              Json{{"anyOf", Json::array({Json{{"oneOf", Json::array({Json{{"type", "boolean"}},
                                                                      Json{{"type", "null"}}})}},
                                          Json{{"type", "integer"}}})}}},
             {"string_or_number",
              Json{{"oneOf", Json::array({Json{{"type", "string"}}, Json{{"type", "number"}}})}}}});
    const std::string text = tool_call("configure", {{"flag", "False"},
                                                     {"unset", "null"},
                                                     {"count", "7.0"},
                                                     {"nested", "TRUE"},
                                                     {"string_or_number", "7"}});
    const auto parsed      = fi::parse_qwen_tool_call_output(text, 64, contract);

    int failures = 0;
    failures += check(parsed.is_tool_call_response && parsed.tool_calls.size() == 1,
                      "explicit anyOf/oneOf primitive union was rejected");
    if (parsed.tool_calls.size() == 1) {
        const Json args = Json::parse(parsed.tool_calls.front().arguments_json);
        failures += check(args.at("flag") == false && args.at("unset").is_null(),
                          "nullable boolean composition was decoded incorrectly");
        failures += check(args.at("count") == 7.0 && args.at("nested") == true,
                          "nested primitive composition was decoded incorrectly");
        failures += check(args.at("string_or_number") == "7",
                          "string-admitting composition did not preserve text");
    }
    failures += check_parameter_schema_mismatch(
        contract, "count", "7.5", "7.5",
        "fractional anyOf integer/null mismatch was not structured");
    return failures;
}

int test_empty_declared_non_string_is_omitted() {
    const Json properties = {
        {"file_path", Json{{"type", "string"}}},
        {"new_string", Json{{"type", "string"}}},
        {"old_string", Json{{"type", "string"}}},
        {"replace_all", Json{{"type", "boolean"}}},
    };
    const std::string text = "I need one more check.\n\n"
                             "<tool_call>\n"
                             "<function=Edit>\n"
                             "<parameter=file_path>\n/tmp/probe.cpp\n</parameter>\n"
                             "<parameter=new_string>\n"
                             "std::map<std::uint32_t, int> counts;\n"
                             "</parameter>\n"
                             "<parameter=old_string>\nold line\n</parameter>\n"
                             "<parameter=replace_all>\n</parameter>\n"
                             "</function>\n"
                             "</tool_call>";

    const std::vector<std::string> definitions = {tool_definition(
        "Edit", properties, Json::array({"file_path", "new_string", "old_string"}))};
    const auto contract                        = contract_from_definitions(definitions);
    const auto parsed = fi::parse_qwen_tool_call_output(text, 128, *contract);

    int failures = 0;
    failures +=
        check(parsed.is_tool_call_response && parsed.content == "I need one more check." &&
                  parsed.tool_calls.size() == 1 && parsed.diagnostics.marker_seen &&
                  parsed.diagnostics.structured_call_count == 1 &&
                  parsed.diagnostics.empty_arguments_omitted == 1 &&
                  parsed.diagnostics.schema_mismatch_arguments == 0 &&
                  parsed.diagnostics.fallback_reason == ninfer::ToolCallParseFallbackReason::None,
              "empty optional boolean demoted a complete Edit call to text");
    if (parsed.tool_calls.size() == 1) {
        const Json args = Json::parse(parsed.tool_calls.front().arguments_json);
        failures += check(args.size() == 3 && args.at("file_path") == "/tmp/probe.cpp" &&
                              args.at("new_string") == "std::map<std::uint32_t, int> counts;" &&
                              args.at("old_string") == "old line" && !args.contains("replace_all"),
                          "empty optional boolean was not omitted from Edit arguments");
    }

    bool every_split_matches = true;
    for (std::size_t split = 0; split <= text.size(); ++split) {
        fi::ToolCallOutputDecoder decoder(contract, 128);
        std::string visible = decoder.feed(std::string_view(text).substr(0, split));
        visible += decoder.feed(std::string_view(text).substr(split));
        auto terminal = decoder.finish();
        if (visible != "I need one more check." || !terminal.content.empty() ||
            terminal.tool_calls.size() != 1 ||
            terminal.tool_calls.front().arguments_json !=
                parsed.tool_calls.front().arguments_json ||
            terminal.diagnostics != parsed.diagnostics) {
            every_split_matches = false;
            break;
        }
    }
    failures += check(every_split_matches,
                      "incremental Edit parsing depends on the transport chunk boundary");

    fi::ToolCallOutputDecoder bytewise(contract, 128);
    std::string bytewise_visible;
    for (const char byte : text) { bytewise_visible += bytewise.feed(std::string_view(&byte, 1)); }
    auto bytewise_terminal = bytewise.finish();
    failures +=
        check(bytewise_visible == "I need one more check." && bytewise_terminal.content.empty() &&
                  bytewise_terminal.tool_calls.size() == 1 &&
                  bytewise_terminal.diagnostics == parsed.diagnostics,
              "bytewise Edit parsing changed the terminal tool-call semantics");

    const auto string_contract =
        contract_for("configure", Json{{"label", Json{{"type", "string"}}}});
    const auto empty_string = fi::parse_qwen_tool_call_output(
        tool_call("configure", {{"label", ""}}), 64, string_contract);
    failures +=
        check(empty_string.is_tool_call_response && empty_string.tool_calls.size() == 1 &&
                  Json::parse(empty_string.tool_calls.front().arguments_json).at("label") == "",
              "empty declared string was incorrectly omitted");
    return failures;
}

int test_schema_mismatches_remain_structured() {
    const auto contract =
        contract_for("configure", Json{{"integer_value", Json{{"type", "integer"}}},
                                       {"number_value", Json{{"type", "number"}}},
                                       {"boolean_value", Json{{"type", "boolean"}}},
                                       {"object_value", Json{{"type", "object"}}},
                                       {"array_value", Json{{"type", "array"}}},
                                       {"null_value", Json{{"type", "null"}}}});

    int failures = 0;
    failures += check_parameter_schema_mismatch(contract, "number_value", "\"1\"", "\"1\"",
                                                "string number mismatch was not structured");
    failures += check_parameter_schema_mismatch(contract, "boolean_value", "[]", "[]",
                                                "array boolean mismatch was not structured");
    failures += check_parameter_schema_mismatch(contract, "object_value", "[]", "[]",
                                                "array object mismatch was not structured");
    failures += check_parameter_schema_mismatch(contract, "array_value", "{}", "{}",
                                                "object array mismatch was not structured");
    failures += check_parameter_schema_mismatch(contract, "null_value", "false", "false",
                                                "boolean null mismatch was not structured");
    failures += check_parameter_schema_mismatch(
        contract, "object_value", "{'x': True}", "\"{'x': True}\"",
        "Python object mismatch was not preserved for client validation");
    failures += check_parameter_schema_mismatch(
        contract, "array_value", "['a', None]", "\"['a', None]\"",
        "Python array mismatch was not preserved for client validation");
    return failures;
}

int test_unsupported_schema_uses_legacy_policy() {
    const auto contract = contract_for(
        "configure",
        Json{{"missing_type", Json::object()},
             {"alias", Json{{"type", "int"}}},
             {"invalid_type_array", Json{{"type", Json::array({"integer", "int"})}}},
             {"partial_anyof", Json{{"anyOf", Json::array({Json{{"type", "integer"}},
                                                           Json{{"enum", Json::array({1, 2})}}})}}},
             {"mixed_composition", Json{{"anyOf", Json::array({Json{{"type", "boolean"}}})},
                                        {"oneOf", Json::array({Json{{"type", "null"}}})}}}});
    const std::string text = tool_call("configure", {{"missing_type", "7"},
                                                     {"alias", "8"},
                                                     {"invalid_type_array", "9"},
                                                     {"partial_anyof", "7.5"},
                                                     {"mixed_composition", "True"},
                                                     {"undeclared", "{\"x\":1}"}});
    const auto parsed      = fi::parse_qwen_tool_call_output(text, 64, contract);

    int failures = 0;
    failures += check(parsed.is_tool_call_response && parsed.tool_calls.size() == 1 &&
                          parsed.diagnostics.schema_mismatch_arguments == 1,
                      "unsupported schema did not retain legacy policy");
    if (parsed.tool_calls.size() != 1) { return failures; }
    const Json args = Json::parse(parsed.tool_calls.front().arguments_json);
    failures += check(args.at("missing_type") == 7 && args.at("alias") == 8 &&
                          args.at("invalid_type_array") == 9,
                      "legacy numeric inference changed");
    failures += check(args.at("partial_anyof") == 7.5 && args.at("mixed_composition") == "True",
                      "unsupported composition was partially inferred");
    failures +=
        check(args.at("undeclared").at("x") == 1, "undeclared parameter legacy inference changed");
    return failures;
}

int test_strict_structure_and_active_tool_set() {
    const auto contract = contract_for("configure", Json{{"value", Json{{"type", "string"}}}});
    int failures        = 0;

    const std::string malformed = "<tool_call>\n<function=configure>\n";
    failures +=
        check_rejected(malformed, contract, ninfer::ToolCallParseFallbackReason::MalformedStructure,
                       "missing structural tags were accepted");

    const std::string suffix = tool_call("configure", {{"value", "x"}}) + "\nextra answer";
    failures +=
        check_rejected(suffix, contract, ninfer::ToolCallParseFallbackReason::TrailingContent,
                       "non-whitespace suffix was accepted");

    const std::string missing_parameter_close =
        "<tool_call>\n<function=configure>\n<parameter=value>\nx\n"
        "</function>\n</tool_call>";
    failures += check_rejected(missing_parameter_close, contract,
                               ninfer::ToolCallParseFallbackReason::MalformedStructure,
                               "missing parameter close was repaired");

    const std::string unknown_tool = tool_call("other", {{"value", "x"}});
    failures +=
        check_rejected(unknown_tool, contract, ninfer::ToolCallParseFallbackReason::UndeclaredTool,
                       "undeclared tool name was accepted");

    const std::string invalid_name = tool_call("bad.name", {{"value", "x"}});
    failures +=
        check_rejected(invalid_name, contract, ninfer::ToolCallParseFallbackReason::InvalidToolName,
                       "invalid function-name character was accepted");
    return failures;
}

int test_name_limits_and_non_strict_omissions() {
    const std::string name(128, 'a');
    const auto name_contract        = contract_for(name, Json::object());
    const std::string text          = tool_call(name);
    const auto anthropic            = fi::parse_qwen_tool_call_output(text, 128, name_contract);
    const auto openai               = fi::parse_qwen_tool_call_output(text, 64, name_contract);
    const std::string too_long_text = tool_call(std::string(129, 'a'));
    const auto too_long = fi::parse_qwen_tool_call_output(too_long_text, 128, name_contract);

    int failures = 0;
    failures += check(anthropic.is_tool_call_response && anthropic.tool_calls.size() == 1,
                      "128-character Anthropic tool name was rejected");
    failures += check(!openai.is_tool_call_response, "128-character OpenAI tool name was accepted");
    failures +=
        check(!too_long.is_tool_call_response, "129-character Anthropic tool name was accepted");

    const std::string definition = tool_definition(
        "optional", Json{{"value", Json{{"type", "string"}}}}, Json::array({"value"}));
    const std::vector<std::string> definitions = {definition};
    const auto contract                        = contract_from_definitions(definitions);
    const auto omitted = fi::parse_qwen_tool_call_output(tool_call("optional"), 64, *contract);
    failures += check(omitted.is_tool_call_response && omitted.tool_calls.size() == 1 &&
                          omitted.tool_calls.front().arguments_json == "{}",
                      "non-strict parser enforced required parameters");
    return failures;
}

int test_duplicate_tool_contracts_rejected() {
    const auto definition =
        tool_definition("configure", Json{{"value", Json{{"type", "integer"}}}});
    try {
        (void)contract_from_definitions({definition, definition});
    } catch (const ninfer::RequestError& error) {
        return check(error.kind() == ninfer::RequestErrorKind::InvalidToolConstraint,
                     "duplicate declaration error kind");
    }
    return check(false, "duplicate declarations were accepted");
}

int test_all_or_nothing_structural_commit() {
    const auto contract    = contract_for("configure", Json{{"flag", Json{{"type", "boolean"}}}});
    const std::string text = tool_call("configure", {{"flag", "true"}}) +
                             "\n<tool_call>\n<function=configure>\n<parameter=flag>\nfalse\n";
    return check_rejected(text, contract, ninfer::ToolCallParseFallbackReason::MalformedStructure,
                          "partially valid tool-call region was partially committed");
}

int test_incremental_valid_and_boolean() {
    fi::ToolCallOutputDecoder legacy(output_contract_for("get_weather", Json::object()), 64);
    std::string visible;
    visible += legacy.feed("Calling weather.  \n<tool_");
    visible += legacy.feed("call>\n<function=get_weather>");
    visible += legacy.feed("\n</function>\n</tool_call>");
    auto legacy_terminal = legacy.finish();
    visible += legacy_terminal.content;

    auto bool_contract =
        output_contract_for("configure", Json{{"enabled", Json{{"type", "boolean"}}}});
    fi::ToolCallOutputDecoder boolean(std::move(bool_contract), 64);
    std::string boolean_visible;
    boolean_visible += boolean.feed("<tool_call>\n<function=configure>\n<parameter=enabled>\nT");
    boolean_visible += boolean.feed("r");
    boolean_visible += boolean.feed("ue\n</parameter>\n</function>\n</tool_call>");
    auto boolean_terminal = boolean.finish();

    int failures = 0;
    failures += check(visible == "Calling weather." && legacy_terminal.tool_calls.size() == 1,
                      "incremental valid call was not committed");
    failures += check(boolean_visible.empty() && boolean_terminal.content.empty() &&
                          boolean_terminal.tool_calls.size() == 1,
                      "incremental boolean call leaked as content");
    if (boolean_terminal.tool_calls.size() == 1) {
        const Json args = Json::parse(boolean_terminal.tool_calls.front().arguments_json);
        failures += check(args.at("enabled") == true,
                          "split case-insensitive boolean was not canonicalized");
    }
    return failures;
}

int test_incremental_fallback_preserves_bytes() {
    const std::string original = "prefix  \n<tool_call>\n<function=broken>";
    fi::ToolCallOutputDecoder malformed(output_contract_for("broken", Json::object()), 64);
    std::string restored;
    restored += malformed.feed(original.substr(0, 10));
    restored += malformed.feed(original.substr(10));
    auto malformed_terminal = malformed.finish();
    restored += malformed_terminal.content;

    fi::ToolCallOutputDecoder ordinary(std::make_shared<fi::ToolCallOutputContract>(), 64);
    std::string ordinary_text;
    ordinary_text += ordinary.feed("ordinary text  ");
    ordinary_text += ordinary.finish().content;

    const std::string partial_original = "  <tool_x then <tool_";
    fi::ToolCallOutputDecoder partial(std::make_shared<fi::ToolCallOutputContract>(), 64);
    std::string partial_restored;
    partial_restored += partial.feed("  <too");
    partial_restored += partial.feed("l_x then <tool_");
    partial_restored += partial.finish().content;

    int failures = 0;
    failures += check(restored == original && malformed_terminal.diagnostics.marker_seen &&
                          malformed_terminal.diagnostics.fallback_reason ==
                              ninfer::ToolCallParseFallbackReason::MalformedStructure,
                      "malformed incremental call lost raw bytes or fallback diagnostics");
    failures += check(ordinary_text == "ordinary text  ",
                      "ordinary incremental output lost trailing whitespace");
    failures +=
        check(partial_restored == partial_original, "partial marker mismatch lost raw bytes");
    return failures;
}

int test_incremental_embedded_parameter_markup() {
    auto contract = output_contract_for("bash", Json{{"command", Json{{"type", "string"}}}});
    const std::string command = "pattern='<parameter=inner>value</parameter>'\n"
                                "printf '%s' \"$pattern\"";
    const std::string text    = tool_call("bash", {{"command", command}});

    fi::ToolCallOutputDecoder decoder(std::move(contract), 64);
    std::string visible;
    constexpr std::size_t kChunk = 7;
    for (std::size_t offset = 0; offset < text.size(); offset += kChunk) {
        visible += decoder.feed(std::string_view(text).substr(offset, kChunk));
    }
    auto terminal = decoder.finish();

    int failures = 0;
    failures +=
        check(visible.empty() && terminal.content.empty() && terminal.tool_calls.size() == 1,
              "chunked embedded parameter markup was not committed as a tool call");
    if (terminal.tool_calls.size() == 1) {
        const Json args = Json::parse(terminal.tool_calls.front().arguments_json);
        failures += check(args.at("command") == command,
                          "chunked embedded parameter markup changed string bytes");
    }
    return failures;
}

int test_free_tool_continuation() {
    const auto contract = output_contract_for("echo", Json{{"value", {{"type", "string"}}}});
    int failures        = 0;
    for (const auto& [prefix, suffix] :
         std::initializer_list<std::pair<std::string_view, std::string_view>>{
             {"Earlier <tool_ca", "broken"},
             {"Earlier <tool_ca", "ll>broken"},
             {"Earlier <tool_call>\n<function=echo>", "broken"}}) {
        fi::ToolCallOutputDecoder decoder(contract, 64);
        decoder.initialize_continuation(prefix);
        std::string visible = decoder.feed(suffix);
        visible += decoder.finish(ninfer::FinishReason::OutputLimit).content;
        failures += check(visible == suffix, "free continuation republished its prompt prefix");
    }
    const auto complete = tool_call("echo", {{"value", "hello"}});
    const auto split    = complete.find("hello") + 2;
    fi::ToolCallOutputDecoder decoder(contract, 64);
    decoder.initialize_continuation(std::string_view(complete).substr(0, split));
    (void)decoder.feed(std::string_view(complete).substr(split));
    const auto result = decoder.finish();
    failures += check(result.tool_calls.size() == 1 && result.content.empty() &&
                          result.tool_calls[0].arguments_json == R"({"value":"hello"})",
                      "free continuation lost the completed call");
    bool rejected = false;
    try {
        fi::ToolCallOutputDecoder completed(contract, 64);
        completed.initialize_continuation(complete);
    } catch (const ninfer::RequestError&) { rejected = true; }
    failures += check(rejected, "free continuation accepted an already published call");

    // A prefix that holds a quoted marker: the real call follows, and the held prefix bytes the
    // client already has are not republished as content.
    fi::ToolCallOutputDecoder quoted(contract, 64);
    quoted.initialize_continuation("Quote <tool_call>x then");
    std::string quoted_visible = quoted.feed(" more\n" + complete);
    const auto quoted_result   = quoted.finish();
    failures += check(quoted_visible.empty() && quoted_result.content == " more" &&
                          quoted_result.tool_calls.size() == 1 &&
                          quoted_result.tool_calls[0].arguments_json == R"({"value":"hello"})",
                      "free continuation republished a held quoted-marker prefix");
    return failures;
}

int test_quoted_marker_before_real_call() {
    const auto contract = contract_for("bash", Json{{"command", Json{{"type", "string"}}}});
    const std::string quoted =
        "<tool_call>\\n<function=shell>\\n<function=command>\\nprintf broken\\n</parameter>\\n"
        "</function>\\n</tool_call>";
    const std::string text = "explaining " + quoted + " then the real turn\n" +
                             tool_call("bash", {{"command", "echo ok"}});
    const auto parsed      = fi::parse_qwen_tool_call_output(text, 64, contract);

    return check(parsed.is_tool_call_response && parsed.tool_calls.size() == 1 &&
                     parsed.tool_calls.front().name == "bash" &&
                     parsed.tool_calls.front().arguments_json == R"({"command":"echo ok"})" &&
                     parsed.content == "explaining " + quoted + " then the real turn" &&
                     parsed.diagnostics.marker_seen &&
                     parsed.diagnostics.structured_call_count == 1 &&
                     parsed.diagnostics.fallback_reason == ninfer::ToolCallParseFallbackReason::None,
                 "a quoted marker before the real call demoted the turn or lost content");
}

int test_later_candidate_must_consume_the_end() {
    const auto contract = contract_for("bash", Json{{"command", Json{{"type", "string"}}}});
    const std::string quoted =
        "<tool_call>\\n<function=shell>\\n<parameter=command>\\nbroken\\n</parameter>\\n"
        "</function>\\n</tool_call>";
    const std::string text =
        quoted + "\n" + tool_call("bash", {{"command", "echo ok"}}) + "\nstill explaining";
    return check_rejected(text, contract, ninfer::ToolCallParseFallbackReason::MalformedStructure,
                          "a quoted marker before a non-terminal call was partially committed");
}

int test_incremental_quoted_marker_preserves_bytes() {
    auto contract = output_contract_for("bash", Json{{"command", Json{{"type", "string"}}}});
    const std::string quoted =
        "<tool_call>\\n<function=shell>\\n<function=command>\\nbroken\\n</parameter>\\n"
        "</function>\\n</tool_call>";
    const std::string text = "explaining " + quoted + " then the real turn\n" +
                             tool_call("bash", {{"command", "echo ok"}});

    fi::ToolCallOutputDecoder decoder(std::move(contract), 64);
    std::string visible;
    constexpr std::size_t kChunk = 5;
    for (std::size_t offset = 0; offset < text.size(); offset += kChunk) {
        visible += decoder.feed(std::string_view(text).substr(offset, kChunk));
    }
    auto terminal = decoder.finish();

    int failures = 0;
    failures += check(terminal.tool_calls.size() == 1 &&
                          terminal.tool_calls.front().arguments_json == R"({"command":"echo ok"})",
                      "incremental quoted marker hid the real tool call");
    failures += check(visible == "explaining" &&
                          visible + terminal.content == "explaining " + quoted + " then the real turn",
                      "incremental quoted marker lost or duplicated bytes");
    failures += check(terminal.diagnostics.marker_seen &&
                          terminal.diagnostics.structured_call_count == 1 &&
                          terminal.diagnostics.fallback_reason ==
                              ninfer::ToolCallParseFallbackReason::None,
                      "incremental quoted marker changed terminal diagnostics");
    return failures;
}

// A byte that breaks a marker prefix may itself start the next marker, and a constrained
// contract recognizes only the <tool_call> form its grammar can emit.
int test_streamed_marker_boundaries() {
    int failures         = 0;
    const auto free_tool = output_contract_for("echo", Json{{"value", {{"type", "string"}}}});
    {
        fi::ToolCallOutputDecoder decoder(free_tool, 64);
        std::string visible = decoder.feed("a<");
        visible += decoder.feed(tool_call("echo", {{"value", "hello"}}));
        const auto result = decoder.finish();
        failures += check(visible == "a<" && result.content.empty() &&
                              result.tool_calls.size() == 1 &&
                              result.tool_calls[0].arguments_json == R"({"value":"hello"})",
                          "a broken marker prefix swallowed the marker that followed it");
    }
    const auto constrained = fi::select_tool_call_contract(free_tool, ninfer::ToolChoice{});
    failures += check(constrained && constrained->constrained,
                      "default tool choice did not constrain tool calls");
    const std::string prose = "Quote <invoke name=\"echo\"> and <function_calls> as text.";
    fi::ToolCallOutputDecoder decoder(constrained, 64);
    std::string visible = decoder.feed(prose);
    const auto result   = decoder.finish();
    failures += check(visible == prose && result.content.empty() && result.tool_calls.empty(),
                      "constrained content spelling an agent-harness marker left the content");
    return failures;
}

int check_single_task_call(const fi::ParsedToolCallOutput& parsed, std::string_view content,
                           std::string_view arguments, std::string_view message) {
    return check(parsed.is_tool_call_response && parsed.content == content &&
                     parsed.tool_calls.size() == 1 &&
                     parsed.tool_calls.front().name == "TaskCreate" &&
                     parsed.tool_calls.front().arguments_json == arguments &&
                     parsed.diagnostics.fallback_reason == ninfer::ToolCallParseFallbackReason::None,
                 std::string(message));
}

int test_claude_code_xml_markup_variants() {
    const auto contract =
        contract_for("TaskCreate", Json{{"description", Json{{"type", "string"}}}});
    int failures = 0;

    failures += check_single_task_call(
        fi::parse_qwen_tool_call_output(
            "<tool_call>\n<function name=\"TaskCreate\">\n<parameter name=\"description\">\n"
            "Initial setup\n</parameter>\n</function>\n</tool_call>",
            128, contract),
        "", R"({"description":"Initial setup"})", "function name attribute syntax was not parsed");

    failures += check_single_task_call(
        fi::parse_qwen_tool_call_output(
            "<tool_call>\n<invoke name=\"TaskCreate\">\n<parameter name=\"description\">\n"
            "Create tasks\n</parameter>\n</invoke>\n</tool_call>",
            128, contract),
        "", R"({"description":"Create tasks"})", "invoke tag syntax was not parsed");

    failures += check_single_task_call(
        fi::parse_qwen_tool_call_output(
            "<function_calls>\n<invoke name=\"TaskCreate\">\n<parameter name=\"description\">\n"
            "Function calls container\n</parameter>\n</invoke>\n</function_calls>",
            128, contract),
        "", R"({"description":"Function calls container"})",
        "function_calls container syntax was not parsed");

    failures += check_single_task_call(
        fi::parse_qwen_tool_call_output(
            "Plan is ready:\n<invoke name=\"TaskCreate\">\n<param name=\"description\">\n"
            "Standalone invoke\n</param>\n</invoke>",
            128, contract),
        "Plan is ready:", R"({"description":"Standalone invoke"})",
        "standalone invoke after plan was not parsed");

    failures += check_single_task_call(
        fi::parse_qwen_tool_call_output(
            "<function='TaskCreate'>\n<parameter='description'>\nquoted\n</parameter>\n"
            "</function>",
            128, contract),
        "", R"({"description":"quoted"})", "quoted bare function header was not parsed");

    // A wrapper with no call, or one left open, is not a call.
    failures += check_rejected("<function_calls>\n</function_calls>", contract,
                               ninfer::ToolCallParseFallbackReason::MalformedStructure,
                               "an empty function_calls wrapper was accepted");
    failures += check_rejected(
        "<function_calls>\n<invoke name=\"TaskCreate\">\n</invoke>\n", contract,
        ninfer::ToolCallParseFallbackReason::MalformedStructure,
        "an unclosed function_calls wrapper was accepted");
    return failures;
}

int test_duplicate_parameter_keeps_last_value() {
    const auto contract = contract_for("configure", Json{{"value", Json{{"type", "string"}}}});
    const auto parsed   = fi::parse_qwen_tool_call_output(
        tool_call("configure", {{"value", "first"}, {"value", "second"}}), 64, contract);
    return check(parsed.is_tool_call_response && parsed.content.empty() &&
                     parsed.tool_calls.size() == 1 &&
                     parsed.tool_calls.front().arguments_json == R"({"value":"second"})" &&
                     parsed.diagnostics.duplicate_parameters_repaired == 1 &&
                     parsed.diagnostics.fallback_reason == ninfer::ToolCallParseFallbackReason::None,
                 "a duplicate parameter did not keep the last value with one recorded repair");
}

int test_duplicate_parameters_keep_last_value() {
    const auto contract = contract_for("configure", Json{{"value", Json{{"type", "string"}},
                                                          {"other", Json{{"type", "string"}}}}});
    int failures        = 0;

    // A repeated identical parameter is the common agent-harness case: the second write leaves the
    // value alone, and the repair is still counted.
    const auto identical = fi::parse_qwen_tool_call_output(
        tool_call("configure", {{"value", "first"}, {"value", "first"}}), 64, contract);
    failures += check(identical.is_tool_call_response && identical.tool_calls.size() == 1 &&
                          identical.tool_calls.front().arguments_json == R"({"value":"first"})" &&
                          identical.diagnostics.duplicate_parameters_repaired == 1,
                      "an identical duplicate parameter was not kept once and counted");

    // A conflicting repeat keeps the last value at the first key's position.
    const auto conflicting = fi::parse_qwen_tool_call_output(
        tool_call("configure", {{"value", "first"}, {"other", "x"}, {"value", "second"}}), 64,
        contract);
    failures += check(conflicting.is_tool_call_response && conflicting.tool_calls.size() == 1 &&
                          conflicting.tool_calls.front().arguments_json ==
                              R"({"value":"second","other":"x"})" &&
                          conflicting.diagnostics.duplicate_parameters_repaired == 1 &&
                          conflicting.diagnostics.fallback_reason ==
                              ninfer::ToolCallParseFallbackReason::None,
                      "a conflicting duplicate parameter did not keep the last value");
    return failures;
}

int test_attribute_token_boundary() {
    const auto contract =
        contract_for("TaskCreate", Json{{"description", Json{{"type", "string"}}}});
    const auto parsed = fi::parse_qwen_tool_call_output(
        "<tool_call>\n<function filename=\"x\" name=\"TaskCreate\">\n"
        "<parameter filename=\"ignored\" name=\"description\">\nCreate task\n</parameter>\n"
        "</function>\n</tool_call>",
        128, contract);
    return check_single_task_call(parsed, "", R"({"description":"Create task"})",
                                  "attribute token boundary failed to extract the name");
}

int test_mismatched_closing_tags_rejected() {
    const auto contract =
        contract_for("TaskCreate", Json{{"description", Json{{"type", "string"}}}});
    using Reason = ninfer::ToolCallParseFallbackReason;
    int failures = 0;
    failures += check_rejected(
        "<tool_call>\n<function name=\"TaskCreate\">\n<parameter name=\"description\">\n"
        "Value\n</parameter>\n</invoke>\n</tool_call>",
        contract, Reason::MalformedStructure,
        "function opening with invoke closing tag was accepted");
    failures += check_rejected(
        "<tool_call>\n<invoke name=\"TaskCreate\">\n<parameter name=\"description\">\n"
        "Value\n</parameter>\n</function>\n</tool_call>",
        contract, Reason::MalformedStructure,
        "invoke opening with function closing tag was accepted");
    failures += check_rejected(
        "<tool_call>\n<function name=\"TaskCreate\">\n<parameter name=\"description\">\n"
        "Value\n</param>\n</function>\n</tool_call>",
        contract, Reason::MalformedStructure,
        "parameter opening with param closing tag was accepted");
    failures += check_rejected(
        "<tool_call>\n<function name=\"TaskCreate\">\n<param name=\"description\">\n"
        "Value\n</parameter>\n</function>\n</tool_call>",
        contract, Reason::MalformedStructure,
        "param opening with parameter closing tag was accepted");
    failures += check_rejected(
        "<tool_call>\n<functionX name=\"TaskCreate\">\n<parameter name=\"description\">\n"
        "Value\n</parameter>\n</function>\n</tool_call>",
        contract, Reason::MalformedStructure,
        "function opener without a token boundary was accepted");
    failures += check_rejected(
        "<tool_call>\n<invokeX name=\"TaskCreate\">\n<parameter name=\"description\">\n"
        "Value\n</parameter>\n</invoke>\n</tool_call>",
        contract, Reason::MalformedStructure,
        "invoke opener without a token boundary was accepted");
    {
        // A value quoting a balanced pair of the other parameter tag family stays one value.
        const std::string quoted_short = "run <param name=\"x\">v</param> done";
        const auto long_outer = fi::parse_qwen_tool_call_output(
            "<tool_call>\n<function name=\"TaskCreate\">\n<parameter name=\"description\">\n" +
                quoted_short + "\n</parameter>\n</function>\n</tool_call>",
            64, contract);
        failures += check(long_outer.is_tool_call_response && long_outer.tool_calls.size() == 1,
                          "param pair quoted inside a parameter value was rejected");
        const std::string quoted_long = "run <parameter name=\"x\">v</parameter> done";
        const auto short_outer = fi::parse_qwen_tool_call_output(
            "<tool_call>\n<function name=\"TaskCreate\">\n<param name=\"description\">\n" +
                quoted_long + "\n</param>\n</function>\n</tool_call>",
            64, contract);
        failures += check(short_outer.is_tool_call_response && short_outer.tool_calls.size() == 1,
                          "parameter pair quoted inside a param value was rejected");
    }
    return failures;
}

int test_claude_code_plan_and_task_create_exact_repro() {
    const std::string task_create = tool_definition(
        "TaskCreate", Json{{"description", Json{{"type", "string"}}},
                           {"task_type", Json{{"type", "string"}}},
                           {"priority", Json{{"type", "integer"}}}});
    const std::string task_update =
        tool_definition("TaskUpdate", Json{{"taskId", Json{{"type", "string"}}},
                                           {"status", Json{{"type", "string"}}}});
    const auto contract = contract_from_definitions({task_create, task_update});

    const std::string plan =
        "I have analyzed the repository requirements. Here is the implementation plan:\n\n"
        "### Plan\n"
        "1. Inspect existing CUDA kernels in `src/ops/softmax_attention/`\n"
        "2. Add test coverage for long context attention splits\n"
        "3. Update frontend tool call decoder\n\n"
        "Let me create the first task in the tracking system now:";
    const std::string full_response = plan +
                                      "\n\n"
                                      "<tool_call>\n"
                                      "<function name=\"TaskCreate\">\n"
                                      "<parameter name=\"description\">\n"
                                      "Implement split-KV page-safety and bounded loops\n"
                                      "</parameter>\n"
                                      "<parameter name=\"task_type\">\n"
                                      "feature\n"
                                      "</parameter>\n"
                                      "<parameter name=\"priority\">\n"
                                      "1\n"
                                      "</parameter>\n"
                                      "</function>\n"
                                      "</tool_call>";
    return check_single_task_call(
        fi::parse_qwen_tool_call_output(full_response, 128, *contract), plan,
        R"({"description":"Implement split-KV page-safety and bounded loops",)"
        R"("task_type":"feature","priority":1})",
        "Claude Code plan + TaskCreate did not parse exactly");
}

// A call opener inside a Markdown code fence or inline code of the answer is a quoted example:
// it stays text, with no byte lost, while calls at the top level still fire. Only the text before
// an opener decides, so every streamed chunking agrees with the whole output.
int test_markdown_quoted_calls() {
    const std::vector<std::string> definitions = {
        tool_definition("write", Json{{"path", Json{{"type", "string"}}}})};
    const auto contract       = contract_from_definitions(definitions);
    const std::string call    = tool_call("write", {{"path", "/tmp/a.txt"}});
    const std::string args    = R"({"path":"/tmp/a.txt"})";
    // (arguments of every call, content) of one output.
    using Outcome = std::pair<std::vector<std::string>, std::string>;
    int failures  = 0;
    const auto outcome = [&](const std::string& text) {
        const auto whole = fi::parse_qwen_tool_call_output(text, 64, *contract);
        Outcome reference({}, whole.content);
        for (const auto& parsed_call : whole.tool_calls) {
            reference.first.push_back(parsed_call.arguments_json);
        }
        for (const std::size_t width : {std::size_t{1}, std::size_t{2}, std::size_t{3},
                                        std::size_t{5}, std::size_t{7}, std::size_t{64},
                                        text.size()}) {
            fi::ToolCallOutputDecoder decoder(contract, 64);
            Outcome streamed;
            for (std::size_t at = 0; at < text.size(); at += width) {
                streamed.second += decoder.feed(std::string_view(text).substr(at, width));
            }
            auto terminal = decoder.finish();
            streamed.second += terminal.content;
            for (const auto& parsed_call : terminal.tool_calls) {
                streamed.first.push_back(parsed_call.arguments_json);
            }
            if (streamed != reference) {
                failures += fail("a streamed chunking of width " + std::to_string(width) +
                                 " disagrees with the whole output: " + text.substr(0, 40));
            }
        }
        return reference;
    };
    const std::vector<std::string> quoted{
        "Format:\n```xml\n" + call + "\n```\nDone.", "Format:\n~~~\n" + call + "\n~~~\nDone.",
        "Use `" + call + "` like this.", "Use `x`, then `\n" + call + "`", "```\n" + call,
        "Text\n\n```python\nx = 1\n```\n```\n" + call};
    for (const std::string& body : quoted) {
        failures += check(outcome(body) == Outcome({}, body),
                          "a call quoted in Markdown code became a call or lost bytes: " +
                              body.substr(0, 30));
    }
    // (output, content before the call)
    const std::vector<std::pair<std::string, std::string>> real{
        {call, ""},
        {"Let me do it.\n\n" + call, "Let me do it."},
        {"Let me do it. " + call, "Let me do it."},
        {"```\ncode\n```\n" + call, "```\ncode\n```"},
        {"`a` and `b` " + call, "`a` and `b`"},
        {"~~~\nx\n~~~\n\n" + call, "~~~\nx\n~~~"}};
    for (const auto& [body, content] : real) {
        failures += check(outcome(body) == Outcome({args}, content),
                          "a top-level call after closed Markdown code did not fire exactly: " +
                              body.substr(0, 30));
    }
    failures += check(outcome(call + "\n" + call) == Outcome({args, args}, ""),
                      "two top-level calls did not both fire");
    // A quoted example before the real call stays text; the call still fires.
    const std::string example = "Example:\n```\n" + call + "\n```\nNow for real.";
    failures += check(outcome(example + "\n\n" + call) == Outcome({args}, example),
                      "a quoted example before a real call was not kept as text");
    return failures;
}

} // namespace

int main() {
    int failures = 0;
    failures += test_free_tool_continuation();
    failures += test_untyped_parameters();
    failures += test_multiple_calls();
    failures += test_declared_strings_preserve_text();
    failures += test_string_values_preserve_embedded_tool_markup();
    failures += test_parameter_delimiters_in_values();
    failures += test_declared_json_types();
    failures += test_boolean_boundary();
    failures += test_exact_integer_boundary();
    failures += test_composed_schema_types();
    failures += test_empty_declared_non_string_is_omitted();
    failures += test_schema_mismatches_remain_structured();
    failures += test_unsupported_schema_uses_legacy_policy();
    failures += test_strict_structure_and_active_tool_set();
    failures += test_name_limits_and_non_strict_omissions();
    failures += test_duplicate_tool_contracts_rejected();
    failures += test_all_or_nothing_structural_commit();
    failures += test_incremental_valid_and_boolean();
    failures += test_incremental_fallback_preserves_bytes();
    failures += test_incremental_embedded_parameter_markup();
    failures += test_quoted_marker_before_real_call();
    failures += test_later_candidate_must_consume_the_end();
    failures += test_incremental_quoted_marker_preserves_bytes();
    failures += test_streamed_marker_boundaries();
    failures += test_claude_code_xml_markup_variants();
    failures += test_duplicate_parameter_keeps_last_value();
    failures += test_duplicate_parameters_keep_last_value();
    failures += test_attribute_token_boundary();
    failures += test_mismatched_closing_tags_rejected();
    failures += test_claude_code_plan_and_task_create_exact_repro();
    failures += test_markdown_quoted_calls();
    if (failures == 0) { std::cout << "ok\n"; }
    return failures == 0 ? 0 : 1;
}
