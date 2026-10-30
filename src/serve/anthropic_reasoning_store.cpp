#include "serve/anthropic_reasoning_store.h"

#include <utility>

namespace ninfer::serve {

AnthropicReasoningStore::AnthropicReasoningStore(std::size_t max_records, std::size_t max_bytes)
    : max_records_(max_records), max_bytes_(max_bytes) {}

std::size_t AnthropicReasoningStore::entry_bytes(const std::string& signature,
                                                 const std::string& reasoning) noexcept {
    return sizeof(Entry) + signature.size() * 2U + reasoning.size();
}

std::shared_ptr<const std::string> AnthropicReasoningStore::get(const std::string& signature) {
    const std::lock_guard lock(mutex_);
    const auto found = records_.find(signature);
    if (found == records_.end()) { return nullptr; }
    lru_.splice(lru_.begin(), lru_, found->second.lru);
    return found->second.reasoning;
}

void AnthropicReasoningStore::put(const std::string& signature, std::string reasoning) {
    const std::size_t bytes = entry_bytes(signature, reasoning);
    if (max_records_ == 0 || bytes > max_bytes_) { return; }
    const std::lock_guard lock(mutex_);
    if (const auto existing = records_.find(signature); existing != records_.end()) {
        erase_locked(existing);
    }
    while (!lru_.empty() && (records_.size() >= max_records_ || current_bytes_ + bytes > max_bytes_)) {
        erase_locked(records_.find(lru_.back()));
    }
    lru_.push_front(signature);
    current_bytes_ += bytes;
    records_.emplace(signature,
                     Entry{std::make_shared<const std::string>(std::move(reasoning)), lru_.begin()});
}

std::size_t AnthropicReasoningStore::size() const {
    const std::lock_guard lock(mutex_);
    return records_.size();
}

std::size_t AnthropicReasoningStore::bytes() const {
    const std::lock_guard lock(mutex_);
    return current_bytes_;
}

void AnthropicReasoningStore::erase_locked(
    std::unordered_map<std::string, Entry>::iterator entry) {
    current_bytes_ -= entry_bytes(entry->first, *entry->second.reasoning);
    lru_.erase(entry->second.lru);
    records_.erase(entry);
}

} // namespace ninfer::serve
