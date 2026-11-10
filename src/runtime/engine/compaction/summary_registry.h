#pragma once

// Continuity for the summary strategy (docs/maintainer/lossy-long-prefill.md, "Alternative
// strategy"): the conversation prefixes the Engine summarised and the summary each got, so a later
// turn of the same conversation renders the same summary (the same tokens) and hits the cache.

#include "ninfer/types.h"

#include <cstddef>
#include <cstdint>
#include <list>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace ninfer::runtime::compaction {

// A recorded summary: messages [first, last) of the prefix it was recorded for are replaced by one
// message with text `summary` (the whole message text).
struct SummaryHit {
    std::size_t first = 0;
    std::size_t last  = 0;
    std::string summary;
};

// Thread-safe, bounded: at most `capacity` entries and `max_bytes` of retained message text, the
// least recently used evicted first. A prefix is compared exactly (role, text parts, media bytes,
// tool calls, tool_call_id; reasoning_content too, except inside the summarised range, which the
// summary does not depend on). A digest of each prefix only shortlists the candidates.
class SummaryRegistry {
public:
    static constexpr std::size_t kDefaultCapacity = 256;
    static constexpr std::size_t kDefaultMaxBytes = std::size_t{256} << 20U;

    explicit SummaryRegistry(std::size_t capacity = kDefaultCapacity,
                             std::size_t max_bytes = kDefaultMaxBytes);

    // `original_prefix` is the conversation up to the end of the summarised range, as it stood when
    // it was summarised (it may itself start with an earlier substitution); messages
    // [first, original_prefix.size()) were replaced. Recording the same prefix again replaces it.
    void record(std::span<const ChatMessage> original_prefix, std::size_t first,
                std::string summary);

    // The longest recorded prefix `messages` starts with, or nullopt.
    [[nodiscard]] std::optional<SummaryHit> find(std::span<const ChatMessage> messages) const;

    [[nodiscard]] std::size_t size() const;

private:
    struct Entry {
        std::vector<ChatMessage> prefix;
        std::size_t first   = 0;
        std::uint64_t digest = 0;
        std::size_t bytes   = 0;
        std::string summary;
    };

    void evict_locked();

    std::size_t capacity_;
    std::size_t max_bytes_;
    mutable std::mutex mutex_;
    // Most recently used first.
    mutable std::list<Entry> entries_;
    std::size_t bytes_ = 0;
};

// The digest of one message the registry shortlists by (reasoning_content left out).
[[nodiscard]] std::uint64_t summary_message_digest(const ChatMessage& message) noexcept;

} // namespace ninfer::runtime::compaction
