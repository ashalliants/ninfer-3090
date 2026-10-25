#include "runtime/engine/compaction/summary_split.h"

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace ninfer::runtime::compaction {

bool is_summary_message(const ChatMessage& message) noexcept {
    if (message.role != ChatRole::User) { return false; }
    for (const MessagePart& part : message.parts) {
        if (part.kind != MessagePartKind::Text) { return false; }
        return std::string_view(part.text).starts_with(kSummaryMessagePrefix);
    }
    return false;
}

namespace {

bool is_real_user(const ChatMessage& message) noexcept {
    return message.role == ChatRole::User && !is_summary_message(message);
}

} // namespace

SummarySplit split_for_summary(std::span<const ChatMessage> messages, std::size_t protect_first,
                               std::size_t protect_last, std::size_t min_tail_users) {
    const std::size_t n = messages.size();
    std::size_t head    = 0;
    while (head < n && (messages[head].role == ChatRole::System ||
                        messages[head].role == ChatRole::Developer)) {
        ++head;
    }
    head = std::min(n, head + protect_first);
    for (std::size_t i = n; i > head; --i) {
        if (is_summary_message(messages[i - 1])) {
            head = i;
            break;
        }
    }
    while (head < n && messages[head].role == ChatRole::Tool) { ++head; }

    if (n <= protect_last) { return {head, head}; }
    std::size_t tail  = n - protect_last;
    std::size_t users = 0;
    for (std::size_t i = tail; i < n; ++i) { users += is_real_user(messages[i]) ? 1U : 0U; }
    while (users < min_tail_users && tail > head) {
        --tail;
        users += is_real_user(messages[tail]) ? 1U : 0U;
    }
    while (tail > 0 && messages[tail].role == ChatRole::Tool) { --tail; }
    if (tail <= head) { return {head, head}; }
    return {head, tail};
}

std::string summary_message_text(std::size_t first, std::size_t last, std::string_view projection) {
    std::string text(kSummaryMessagePrefix);
    text += std::to_string(first);
    text += '-';
    text += std::to_string(last == 0 ? 0 : last - 1);
    text += "; originals omitted]\n";
    text += projection;
    return text;
}

PromptInput substitute_summary(PromptInput input, std::size_t first, std::size_t last,
                               std::string summary) {
    auto& messages = input.messages;
    if (first >= last || last > messages.size()) {
        throw std::invalid_argument("summary range is empty or past the end of the conversation");
    }
    ChatMessage message;
    message.role = ChatRole::User;
    message.parts.push_back(MessagePart{.kind = MessagePartKind::Text, .text = std::move(summary)});
    messages[first] = std::move(message);
    messages.erase(messages.begin() + static_cast<std::ptrdiff_t>(first + 1),
                   messages.begin() + static_cast<std::ptrdiff_t>(last));

    // Message counts after the range shrink by the removed messages; a boundary inside the range
    // has nothing left to mark.
    const auto removed = static_cast<std::uint32_t>(last - first - 1);
    std::vector<PromptCacheMarker> markers;
    markers.reserve(input.context_cache.markers.size());
    for (PromptCacheMarker marker : input.context_cache.markers) {
        const std::uint32_t count = marker.after_message_count;
        if (marker.location == PromptCacheMarkerLocation::MessageBoundary) {
            if (count > first && count < last) { continue; }
            if (count >= last) { marker.after_message_count = count - removed; }
        } else if (marker.location == PromptCacheMarkerLocation::MessagePartBoundary) {
            // A one-based count of the message the boundary lies in.
            if (count > first && count <= last) { continue; }
            if (count > last) { marker.after_message_count = count - removed; }
        }
        markers.push_back(marker);
    }
    input.context_cache.markers = std::move(markers);
    return input;
}

} // namespace ninfer::runtime::compaction
