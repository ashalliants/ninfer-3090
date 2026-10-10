#pragma once

#include "models/qwen3_5/frontend/tool_contract.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace ninfer::models::qwen3_5::frontend {

struct ParsedToolCallOutput {
    bool is_tool_call_response = false;
    std::string content;
    std::vector<GeneratedToolCall> tool_calls;
    ToolCallParseDiagnostics diagnostics;
};

// The Markdown code state of answer text read so far: an open ``` or ~~~ fence, and the
// backticks of the current paragraph. A tool-call marker of free output inside a fence or inline
// code is a quoted example and stays text. Only the text before a marker decides, so streamed
// and whole outputs agree.
class MarkdownCodeTracker {
public:
    void feed(std::string_view text);
    // Whether the next byte lies inside a code fence or inline code.
    [[nodiscard]] bool in_code() const noexcept;

private:
    void end_line();

    std::string line_;           // the current, unfinished line
    char fence_          = '\0'; // the open fence's character, '\0' when none is open
    std::uint32_t ticks_ = 0;    // backticks of the current paragraph's finished lines
};

// What the free parser repairs (`--tolerant-tool-calls`). No level keeps a call whose parameter
// value was cut: a value runs to its closing tag or the call is dropped.
enum class ToolCallRepair : std::uint8_t {
    // All-or-nothing per region.
    None,
    // Calls with all their closing tags are kept around broken markup: a suffix or a malformed
    // call after them is discarded, an unclosed `<function_calls>` wrapper after them is
    // accepted, a malformed function opener (a dropped or doubled `<`, a leaked `<|im_start|>`, a
    // dropped `function` keyword) or a missing `>` after the function name is repaired, and an
    // undeclared tool name stays structured. A turn cut by the output budget, the context, a
    // cancellation or a stop string gets this level, as constrained output keeps only its
    // completed calls.
    CompleteCalls,
    // Also keeps a final call whose closing tags are missing after a closed parameter: the model
    // ended the turn itself, on its stop token.
    UnclosedFinalCall,
};

// The repair level of free tool output whose turn ended for `reason`.
[[nodiscard]] ToolCallRepair tool_call_repair(bool tolerant, FinishReason reason) noexcept;

// Parses Qwen's XML-like tool-call format and the agent-harness variants of it. Free output is
// all-or-nothing per region: the first region (from a marker outside Markdown code, or a later
// `<tool_call>`) that parses to the end of the output becomes the structured turn and the text
// before it is content; otherwise the whole output is content. `code` is the Markdown code state
// of the text before `text`. `repair` relaxes "parses to the end" as ToolCallRepair describes.
[[nodiscard]] ParsedToolCallOutput
parse_qwen_tool_call_output(const std::string& text, std::size_t max_tool_name_length,
                            const ToolCallOutputContract& contract, MarkdownCodeTracker code = {},
                            ToolCallRepair repair = ToolCallRepair::None);

// Incrementally publishes bytes that are provably outside a possible terminal Qwen tool-call
// suffix. At terminal time, constrained output retains completed calls across interruptions;
// free output restores malformed regions verbatim.
class ToolCallOutputDecoder {
public:
    struct Terminal {
        std::string content;
        std::vector<GeneratedToolCall> tool_calls;
        ToolCallParseDiagnostics diagnostics;
    };

    // `tolerant` enables free-output repairs at finish, at the level its finish reason allows.
    // Constrained contracts never consult it.
    ToolCallOutputDecoder(std::shared_ptr<const ToolCallOutputContract> contract,
                          std::size_t max_tool_name_length, bool tolerant = false);

    [[nodiscard]] bool in_tool_region() const noexcept {
        return saw_tool_marker_ || !pending_tag_.empty();
    }
    [[nodiscard]] std::string feed(std::string_view text);
    void initialize_continuation(std::string_view prefix);
    // `open_reasoning` is the thinking of a turn that ended without closing it. For free tool
    // output, when the turn ended on its stop token and no content was fed, complete declared
    // calls that end that thinking become the structured turn.
    [[nodiscard]] Terminal finish(FinishReason reason             = FinishReason::StopToken,
                                  std::string_view open_reasoning = {});

private:
    std::shared_ptr<const ToolCallOutputContract> contract_;
    std::string trailing_whitespace_;
    std::string tool_region_;
    // Held bytes that are a prefix of some tool-call marker.
    std::string pending_tag_;
    // The code state of the text published so far, and of the text before the tool region (the
    // state its terminal parse starts from).
    MarkdownCodeTracker code_;
    MarkdownCodeTracker region_code_;
    std::size_t max_tool_name_length_        = 0;
    bool tolerant_                           = false;
    bool saw_tool_marker_                    = false;
    bool fed_content_                        = false;
    bool finished_                          = false;
    std::size_t continuation_withheld_bytes_ = 0;
};

} // namespace ninfer::models::qwen3_5::frontend
