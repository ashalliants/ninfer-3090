#pragma once

#include "models/qwen3_5/program/storage/graft_loader.h"

#include <cstdint>
#include <filesystem>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>

namespace ninfer::models::qwen3_5 {

class GraftRegistry {
public:
    /** Register a graft artifact file under `id`. Thread-safe; concurrent registrations allowed. */
    void register_graft(std::string id, std::filesystem::path path);

    /** Look up a graft by ID. Returns nullptr if not found or loading failed. */
    [[nodiscard]] LoadedGraft const* find(std::string_view id) const noexcept;

    /** Return number of registered grafts (for diagnostics). */
    [[nodiscard]] std::uint32_t count() const noexcept {
        std::lock_guard const lock(mutex_);
        return static_cast<std::uint32_t>(grafts_.size());
    }

    /** Check whether any grafts are registered. */
    [[nodiscard]] bool empty() const noexcept {
        std::lock_guard const lock(mutex_);
        return grafts_.empty();
    }

    /** Return maximum layer count across all registered grafts (global reservation size). */
    [[nodiscard]] std::uint32_t max_graft_layers() const noexcept {
        std::lock_guard const lock(mutex_);
        std::uint32_t result = 0;
        for (const auto& [id, loaded] : grafts_) {
            if (loaded.layers() > static_cast<std::int32_t>(result)) {
                result = static_cast<std::uint32_t>(loaded.layers());
            }
        }
        return result;
    }

private:
    mutable std::mutex mutex_;
    std::unordered_map<std::string, LoadedGraft> grafts_;
};

} // namespace ninfer::models::qwen3_5
