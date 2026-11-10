#include "runtime/engine/compaction/summary_strategy.h"

#include "runtime/engine/compaction/vcc_projector.h"

#include <nlohmann/json.hpp>

#include <cmath>
#include <string_view>

namespace ninfer::runtime::compaction {
namespace {

struct ToolRename {
    std::string_view client;
    std::string_view projector;
    // An argument key renamed to `to` when `to` is absent; empty for none.
    std::string_view from;
    std::string_view to;
};

// The table in summary_strategy.h.
constexpr ToolRename kToolRenames[] = {
    {"Bash", "terminal", {}, {}},
    {"BashOutput", "terminal", {}, {}},
    {"exec_command", "terminal", {}, {}},
    {"Read", "read_file", "filePath", "file_path"},
    {"read", "read_file", "filePath", "file_path"},
    {"Write", "write_file", "filePath", "file_path"},
    {"write", "write_file", "filePath", "file_path"},
    {"Edit", "edit_file", "filePath", "file_path"},
    {"MultiEdit", "edit_file", "filePath", "file_path"},
    {"edit", "edit_file", "filePath", "file_path"},
    {"NotebookEdit", "edit_file", "notebook_path", "path"},
    {"Grep", "search_files", {}, {}},
    {"Glob", "search_files", {}, {}},
    {"grep", "search_files", {}, {}},
    {"glob", "search_files", {}, {}},
    {"apply_patch", "patch", {}, {}},
};

std::string rename_key(const std::string& arguments, std::string_view from, std::string_view to) {
    if (from.empty()) { return arguments; }
    nlohmann::ordered_json parsed = nlohmann::ordered_json::parse(arguments, nullptr, false);
    if (parsed.is_discarded() || !parsed.is_object() || !parsed.contains(from) ||
        parsed.contains(to)) {
        return arguments;
    }
    nlohmann::ordered_json renamed = nlohmann::ordered_json::object();
    for (auto it = parsed.begin(); it != parsed.end(); ++it) {
        renamed[it.key() == from ? std::string(to) : it.key()] = it.value();
    }
    return renamed.dump();
}

void replace_all(std::string& text, std::string_view from, std::string_view to) {
    for (std::size_t at = text.find(from); at != std::string::npos; at = text.find(from, at)) {
        text.replace(at, from.size(), to);
        at += to.size();
    }
}

} // namespace

std::vector<ChatMessage> normalize_tool_calls(std::span<const ChatMessage> messages) {
    std::vector<ChatMessage> out(messages.begin(), messages.end());
    for (ChatMessage& message : out) {
        for (ToolCall& call : message.tool_calls) {
            for (const ToolRename& rename : kToolRenames) {
                if (call.name != rename.client) { continue; }
                call.name           = std::string(rename.projector);
                call.arguments_json = rename_key(call.arguments_json, rename.from, rename.to);
                break;
            }
        }
    }
    return out;
}

std::string strip_session_search(std::string projection) {
    replace_all(projection, "; read any of them in full with session_search(around_message_id=N).",
                "; the full messages were omitted from this conversation.");
    replace_all(projection, "; use session_search to read them.", ".");
    return projection;
}

std::size_t summary_budget_chars(std::uint32_t max_context, std::uint32_t summary_max_tokens) noexcept {
    const auto share =
        static_cast<std::uint64_t>(std::floor(0.06 * static_cast<double>(max_context)));
    const std::uint64_t tokens =
        std::max<std::uint64_t>(kSummaryMinimumTokens, std::min<std::uint64_t>(share, summary_max_tokens));
    return static_cast<std::size_t>(tokens * kSummaryCharsPerToken);
}

std::string build_summary_text(std::span<const ChatMessage> messages, std::size_t first,
                               std::size_t last, std::size_t budget_chars) {
    const std::vector<ChatMessage> normalized = normalize_tool_calls(messages);
    return summary_message_text(
        first, last, strip_session_search(project_transcript(normalized, first, last, budget_chars)));
}

std::pair<std::uint32_t, std::uint32_t> common_prefix_suffix(std::span<const TokenId> a,
                                                              std::span<const TokenId> b) noexcept {
    const std::size_t limit = std::min(a.size(), b.size());
    std::size_t prefix      = 0;
    while (prefix < limit && a[prefix] == b[prefix]) { ++prefix; }
    std::size_t suffix = 0;
    while (suffix < limit - prefix && a[a.size() - 1 - suffix] == b[b.size() - 1 - suffix]) {
        ++suffix;
    }
    return {static_cast<std::uint32_t>(prefix), static_cast<std::uint32_t>(suffix)};
}

} // namespace ninfer::runtime::compaction
