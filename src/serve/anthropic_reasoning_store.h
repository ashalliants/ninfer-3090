#pragma once

// Process-local bounded store for reasoning that an Anthropic response hid with
// thinking.display="omitted". The block's signature is the response message id, so a client that
// replays the empty block lets the next prompt restore the exact reasoning text. Restoring it keeps
// the rendered prompt byte-identical to the tokens the Engine already cached. A miss (restart,
// eviction, foreign signature) only costs that cache reuse.

#include <cstddef>
#include <list>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>

namespace ninfer::serve {

class AnthropicReasoningStore {
public:
    AnthropicReasoningStore(std::size_t max_records, std::size_t max_bytes);

    // get() refreshes LRU recency.
    [[nodiscard]] std::optional<std::string> get(const std::string& signature);
    // Replaces an existing entry. Text larger than the whole byte budget is not stored.
    void put(const std::string& signature, std::string reasoning);

    [[nodiscard]] std::size_t size() const;
    [[nodiscard]] std::size_t bytes() const;

private:
    struct Entry {
        std::string reasoning;
        std::list<std::string>::iterator lru;
    };

    [[nodiscard]] static std::size_t entry_bytes(const std::string& signature,
                                                 const std::string& reasoning) noexcept;
    void erase_locked(std::unordered_map<std::string, Entry>::iterator entry);

    std::size_t max_records_ = 0;
    std::size_t max_bytes_   = 0;
    mutable std::mutex mutex_;
    std::unordered_map<std::string, Entry> records_;
    std::list<std::string> lru_;
    std::size_t current_bytes_ = 0;
};

} // namespace ninfer::serve
