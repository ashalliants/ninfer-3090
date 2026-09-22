#pragma once

#include "models/qwen3_5/program/internal.h"
#include "models/qwen3_5/program/storage/graft_loader.h"
#include "core/device.h"

#include <memory>
#include <string>
#include <unordered_map>

// Engine-level graft storage.
// Holds LoadedGraft objects keyed by user-facing ID. Thread-safe for concurrent lookups;
// registration is single-threaded during engine construction.

namespace ninfer::models::qwen3_5 {

class GraftStorage {
public:
    void register_graft(std::string id, std::filesystem::path const& path);
    [[nodiscard]] LoadedGraft const* find(std::string_view id) const noexcept;

private:
    std::unordered_map<std::string, LoadedGraft> registry_;
};

} // namespace ninfer::models::qwen3_5
