#pragma once

// C++ port of vcc/projector.py from hermes-vcc-engine (Apache-2.0, commit 31ee4fe7). See NOTICE in
// this directory for the attribution and tools/bench/vcc_golden.py for the golden outputs this port
// reproduces byte for byte.
//
// A deterministic, rule-based summary of a span of chat messages: task, goal, failed calls, errors
// and their fixes, error names, files touched, commands run with their outcomes, and a session log
// whose [#N] tags are absolute message indices. No model call, no clock, no I/O.
//
// ChatMessage -> Hermes message dict (OpenAI chat format), the input the Python projector reads:
//   role        System/Developer/User/Assistant/Tool -> "system"/"developer"/"user"/"assistant"/
//               "tool". Only user, assistant and tool messages contribute.
//   content     a list with one part per MessagePart: Text -> {"type": "text"} (its text), Media
//               Image -> {"type": "image_url"} (reads "[image]"), Media Video -> {"type": "file"}
//               (reads "[file]"). The projector joins the parts with one space.
//   tool_calls  [{"id", "type": "function", "function": {"name", "arguments": arguments_json}}].
//   tool_call_id as is. A tool message has no name of its own; an unmatched result reads "tool".
//   _row_id     the absolute message index, so refs read "[#N]"; Python treats row 0 as missing,
//               which makes message 0 read "[t0]".
//   reasoning_content is not passed (the projector drops reasoning).
// Strings are UTF-8; invalid UTF-8 and lone-surrogate JSON escapes decode to U+FFFD. Every length
// and limit counts Unicode code points, as Python's str does.
//
// Not ported: the previous-summary merge (`previous`), focus terms, the session id and the
// is_real_user filter (every non-empty user message counts), which no caller here uses.

#include "ninfer/types.h"

#include <cstddef>
#include <span>
#include <string>

namespace ninfer::runtime::compaction {

// vcc.projector.Limits: per-item caps in code points, and section sizes.
struct ProjectorLimits {
    std::size_t user_chars      = 600;
    std::size_t assistant_chars = 400;
    std::size_t call_chars      = 160;
    std::size_t result_chars    = 160;
    std::size_t error_chars     = 600;
    std::size_t goal_chars      = 800;
    std::size_t task_chars      = 600;
    std::size_t max_files       = 40;
    std::size_t max_commands    = 25;
    std::size_t max_blocked     = 12;
    std::size_t max_fixes       = 12;
    std::size_t max_error_names = 20;
    // Share of the budget an inherited non-vcc summary may keep; only the (unported) merge reads it.
    double earlier_share = 0.15;
};

// messages[first, last) is the region to summarise; indices in [#N] tags are absolute message
// indices. The result is at most max(200, budget_chars) code points, as in Python.
[[nodiscard]] std::string project_transcript(std::span<const ChatMessage> messages, std::size_t first,
                                             std::size_t last, std::size_t budget_chars,
                                             const ProjectorLimits& limits = {});

} // namespace ninfer::runtime::compaction
