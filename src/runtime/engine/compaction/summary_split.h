#pragma once

// The summary strategy of lossy prefill (docs/maintainer/lossy-long-prefill.md, "Alternative
// strategy: deterministic conversation summary"): which messages of a long conversation are
// replaced by one summary message, and how that message is put in their place.

#include "ninfer/types.h"

#include <cstddef>
#include <span>
#include <string>
#include <string_view>

namespace ninfer::runtime::compaction {

// Every summary message the Engine writes starts with this text (summary_message_text).
inline constexpr std::string_view kSummaryMessagePrefix = "[Server-side summary of messages ";

// A User message whose first text part starts with kSummaryMessagePrefix: one the Engine wrote in
// place of earlier messages. It is never summarised again as an ordinary user turn.
[[nodiscard]] bool is_summary_message(const ChatMessage& message) noexcept;

// The middle of a conversation to summarise: messages [first, last). Empty if first >= last.
struct SummarySplit {
    std::size_t first = 0;
    std::size_t last  = 0;

    [[nodiscard]] bool empty() const noexcept { return first >= last; }
};

// Head: the leading System/Developer messages plus `protect_first` messages, extended through the
// last existing summary message (an earlier summary stays in the head, as it was rendered) and past
// any Tool messages (so the middle never starts with a result whose call is in the head).
// Tail: starts at the latest index that leaves at least `protect_last` messages and at least
// `min_tail_users` real user messages (User, not a summary), moved earlier so it never starts at a
// Tool message, which would separate an Assistant tool call from its results.
// The middle is what lies between; empty when head and tail meet.
[[nodiscard]] SummarySplit split_for_summary(std::span<const ChatMessage> messages,
                                             std::size_t protect_first  = 3,
                                             std::size_t protect_last   = 20,
                                             std::size_t min_tail_users = 1);

// "[Server-side summary of messages <first>-<last-1>; originals omitted]\n" + projection.
[[nodiscard]] std::string summary_message_text(std::size_t first, std::size_t last,
                                               std::string_view projection);

// `input` with messages [first, last) replaced by one User message whose text is `summary` (the
// whole message text, summary_message_text). Cache markers after the range move with the messages
// they follow; a marker inside it is dropped, and one right after it follows the summary message.
// Throws std::invalid_argument when the range is empty or past the end.
[[nodiscard]] PromptInput substitute_summary(PromptInput input, std::size_t first, std::size_t last,
                                             std::string summary);

} // namespace ninfer::runtime::compaction
