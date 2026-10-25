#include "runtime/engine/compaction/summary_registry.h"

#include <stdexcept>
#include <string_view>
#include <utility>

namespace ninfer::runtime::compaction {
namespace {

constexpr std::uint64_t kFnvOffset = 14695981039346656037ULL;
constexpr std::uint64_t kFnvPrime  = 1099511628211ULL;

void mix(std::uint64_t& hash, std::string_view bytes) noexcept {
    for (const char c : bytes) {
        hash ^= static_cast<unsigned char>(c);
        hash *= kFnvPrime;
    }
    // A length terminator, so ("ab", "c") and ("a", "bc") differ.
    const std::uint64_t size = bytes.size();
    for (int shift = 0; shift < 64; shift += 8) {
        hash ^= (size >> shift) & 0xFFU;
        hash *= kFnvPrime;
    }
}

void mix(std::uint64_t& hash, std::uint64_t value) noexcept {
    for (int shift = 0; shift < 64; shift += 8) {
        hash ^= (value >> shift) & 0xFFU;
        hash *= kFnvPrime;
    }
}

bool same_media(const OwnedMedia& a, const OwnedMedia& b) noexcept {
    return a.kind == b.kind && a.bytes == b.bytes && a.media_type == b.media_type &&
           a.source_name == b.source_name && a.image_resize_policy == b.image_resize_policy;
}

bool same_message(const ChatMessage& a, const ChatMessage& b, bool compare_reasoning) noexcept {
    if (a.role != b.role || a.tool_call_id != b.tool_call_id || a.parts.size() != b.parts.size() ||
        a.tool_calls.size() != b.tool_calls.size()) {
        return false;
    }
    if (compare_reasoning && a.reasoning_content != b.reasoning_content) { return false; }
    for (std::size_t i = 0; i < a.parts.size(); ++i) {
        const MessagePart& x = a.parts[i];
        const MessagePart& y = b.parts[i];
        if (x.kind != y.kind || x.text != y.text) { return false; }
        if (x.kind == MessagePartKind::Media && !same_media(x.media, y.media)) { return false; }
    }
    for (std::size_t i = 0; i < a.tool_calls.size(); ++i) {
        const ToolCall& x = a.tool_calls[i];
        const ToolCall& y = b.tool_calls[i];
        if (x.id != y.id || x.name != y.name || x.arguments_json != y.arguments_json) {
            return false;
        }
    }
    return true;
}

std::size_t message_bytes(const ChatMessage& message) noexcept {
    std::size_t bytes = sizeof(ChatMessage) + message.reasoning_content.size() +
                        message.tool_call_id.size();
    for (const MessagePart& part : message.parts) {
        bytes += sizeof(MessagePart) + part.text.size() + part.media.bytes.size();
    }
    for (const ToolCall& call : message.tool_calls) {
        bytes += sizeof(ToolCall) + call.id.size() + call.name.size() + call.arguments_json.size();
    }
    return bytes;
}

std::uint64_t combine(std::uint64_t prefix, std::uint64_t message) noexcept {
    std::uint64_t hash = prefix;
    mix(hash, message);
    return hash;
}

} // namespace

std::uint64_t summary_message_digest(const ChatMessage& message) noexcept {
    std::uint64_t hash = kFnvOffset;
    mix(hash, static_cast<std::uint64_t>(message.role));
    mix(hash, message.tool_call_id);
    for (const MessagePart& part : message.parts) {
        mix(hash, static_cast<std::uint64_t>(part.kind));
        mix(hash, part.text);
        if (part.kind == MessagePartKind::Media) {
            mix(hash, std::string_view(reinterpret_cast<const char*>(part.media.bytes.data()),
                                       part.media.bytes.size()));
        }
    }
    for (const ToolCall& call : message.tool_calls) {
        mix(hash, call.id);
        mix(hash, call.name);
        mix(hash, call.arguments_json);
    }
    return hash;
}

SummaryRegistry::SummaryRegistry(std::size_t capacity, std::size_t max_bytes)
    : capacity_(capacity), max_bytes_(max_bytes) {
    if (capacity_ == 0) { throw std::invalid_argument("summary registry capacity must be positive"); }
}

void SummaryRegistry::record(std::span<const ChatMessage> original_prefix, std::size_t first,
                             std::string summary) {
    if (first >= original_prefix.size()) {
        throw std::invalid_argument("a recorded summary must replace at least one message");
    }
    Entry entry;
    entry.prefix.assign(original_prefix.begin(), original_prefix.end());
    entry.first   = first;
    entry.summary = std::move(summary);
    entry.digest  = kFnvOffset;
    entry.bytes   = entry.summary.size();
    for (const ChatMessage& message : entry.prefix) {
        entry.digest = combine(entry.digest, summary_message_digest(message));
        entry.bytes += message_bytes(message);
    }
    std::scoped_lock lock(mutex_);
    for (auto it = entries_.begin(); it != entries_.end(); ++it) {
        if (it->digest == entry.digest && it->first == entry.first &&
            it->prefix.size() == entry.prefix.size()) {
            bool same = true;
            for (std::size_t i = 0; same && i < entry.prefix.size(); ++i) {
                same = same_message(it->prefix[i], entry.prefix[i], i < first);
            }
            if (same) {
                bytes_ -= it->bytes;
                entries_.erase(it);
                break;
            }
        }
    }
    bytes_ += entry.bytes;
    entries_.push_front(std::move(entry));
    evict_locked();
}

void SummaryRegistry::evict_locked() {
    // The newest entry stays even when it alone exceeds the byte bound.
    while (entries_.size() > 1 && (entries_.size() > capacity_ || bytes_ > max_bytes_)) {
        bytes_ -= entries_.back().bytes;
        entries_.pop_back();
    }
}

std::optional<SummaryHit> SummaryRegistry::find(std::span<const ChatMessage> messages) const {
    std::vector<std::uint64_t> digests(messages.size() + 1);
    digests[0] = kFnvOffset;
    for (std::size_t i = 0; i < messages.size(); ++i) {
        digests[i + 1] = combine(digests[i], summary_message_digest(messages[i]));
    }
    std::scoped_lock lock(mutex_);
    auto best = entries_.end();
    for (auto it = entries_.begin(); it != entries_.end(); ++it) {
        const std::size_t length = it->prefix.size();
        if (length > messages.size() || digests[length] != it->digest) { continue; }
        if (best != entries_.end() && best->prefix.size() >= length) { continue; }
        bool same = true;
        for (std::size_t i = 0; same && i < length; ++i) {
            same = same_message(it->prefix[i], messages[i], i < it->first);
        }
        if (same) { best = it; }
    }
    if (best == entries_.end()) { return std::nullopt; }
    entries_.splice(entries_.begin(), entries_, best);
    const Entry& hit = entries_.front();
    return SummaryHit{.first = hit.first, .last = hit.prefix.size(), .summary = hit.summary};
}

std::size_t SummaryRegistry::size() const {
    std::scoped_lock lock(mutex_);
    return entries_.size();
}

} // namespace ninfer::runtime::compaction
