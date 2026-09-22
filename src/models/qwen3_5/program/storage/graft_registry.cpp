#include "models/qwen3_5/program/storage/graft_registry.h"

namespace ninfer::models::qwen3_5 {

void GraftRegistry::register_graft(std::string id, std::filesystem::path path) {
    auto g  = GraftLoader::load(path, ""); // model_id validated against resident model later
    std::lock_guard const lock(mutex_);
    grafts_.try_emplace(std::move(id), std::move(g));
}

LoadedGraft const* GraftRegistry::find(std::string_view id) const noexcept {
    std::lock_guard const lock(mutex_);
    auto it = grafts_.find(static_cast<std::string>(id));
    return it != grafts_.end() ? &it->second : nullptr;
}

} // namespace ninfer::models::qwen3_5
