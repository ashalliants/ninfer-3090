#include "models/qwen3_5/program/storage/graft_storage.h"

namespace ninfer::models::qwen3_5 {

void GraftStorage::register_graft(std::string id, std::filesystem::path const& path) {
    registry_[std::move(id)] = GraftLoader::load(path, "");
}

LoadedGraft const* GraftStorage::find(std::string_view id) const noexcept {
    auto it = registry_.find(static_cast<std::string>(id));
    return it != registry_.end() ? &it->second : nullptr;
}

} // namespace ninfer::models::qwen3_5
