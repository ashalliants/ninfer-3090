// Adapted from Infernix a3edb450 src/models/qwen4_exp/program/expert_cache/expert_state.cpp (Apache-2.0).
// Modified for NInfer-3090: NInfer's namespace.

#include "models/qwen4_exp/program/expert_cache/expert_state.h"

#include <array>
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

namespace ninfer::models::qwen4_exp::expert_cache {
namespace {

// Layout (version 1): magic, version, identity length and bytes, key count, counts[keys], ranked
// count, ranked keys, then the magic again as a footer.
constexpr std::array<char, 8> kMagic = {'N', 'I', 'N', 'F', 'Q', '4', 'E', 'S'};
constexpr std::uint32_t kVersion     = 1;
constexpr std::uint32_t kMaxKeys     = 1U << 24; // a damaged key count must not allocate gigabytes

template <class T>
void put(std::ofstream& out, const T& value) {
    out.write(reinterpret_cast<const char*>(&value), sizeof(T));
}

template <class T>
bool get(std::ifstream& in, T& value) {
    in.read(reinterpret_cast<char*>(&value), sizeof(T));
    return static_cast<bool>(in);
}

} // namespace

std::string save_expert_state(const std::filesystem::path& path, std::string_view identity,
                              const SavedState& state) {
    std::filesystem::path temporary = path;
    temporary += ".tmp";
    {
        std::ofstream out(temporary, std::ios::binary | std::ios::trunc);
        if (!out) { return "cannot create " + temporary.string(); }
        out.write(kMagic.data(), kMagic.size());
        put(out, kVersion);
        put(out, static_cast<std::uint32_t>(identity.size()));
        out.write(identity.data(), static_cast<std::streamsize>(identity.size()));
        put(out, static_cast<std::uint32_t>(state.counts.size()));
        out.write(reinterpret_cast<const char*>(state.counts.data()),
                  static_cast<std::streamsize>(state.counts.size() * sizeof(std::uint32_t)));
        put(out, static_cast<std::uint32_t>(state.ranked.size()));
        out.write(reinterpret_cast<const char*>(state.ranked.data()),
                  static_cast<std::streamsize>(state.ranked.size() * sizeof(std::uint32_t)));
        out.write(kMagic.data(), kMagic.size());
        out.flush();
        if (!out) {
            out.close();
            std::error_code ignored;
            std::filesystem::remove(temporary, ignored);
            return "writing " + temporary.string() + " failed";
        }
    }
    std::error_code error;
    std::filesystem::rename(temporary, path, error);
    if (error) {
        std::filesystem::remove(temporary, error);
        return "cannot replace " + path.string();
    }
    return {};
}

ExpertStateRead read_expert_state(const std::filesystem::path& path) {
    ExpertStateRead out;
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        out.message = "no saved expert state at " + path.string() + " yet";
        return out;
    }
    std::array<char, 8> magic{};
    std::uint32_t version = 0, identity_bytes = 0, keys = 0, ranked = 0;
    in.read(magic.data(), magic.size());
    if (!in || magic != kMagic || !get(in, version) || version != kVersion) {
        out.message = "not an expert state file of this version";
        return out;
    }
    if (!get(in, identity_bytes) || identity_bytes > (1U << 16)) {
        out.message = "the expert state file is damaged";
        return out;
    }
    ExpertStateFile file;
    file.identity.assign(identity_bytes, '\0');
    in.read(file.identity.data(), identity_bytes);
    if (!in || !get(in, keys) || keys > kMaxKeys) {
        out.message = "the expert state file is damaged";
        return out;
    }
    SavedState& state = file.state;
    state.counts.resize(keys);
    in.read(reinterpret_cast<char*>(state.counts.data()), static_cast<std::streamsize>(keys * sizeof(std::uint32_t)));
    if (!in || !get(in, ranked) || ranked > keys) {
        out.message = "the expert state file is damaged";
        return out;
    }
    state.ranked.resize(ranked);
    in.read(reinterpret_cast<char*>(state.ranked.data()), static_cast<std::streamsize>(ranked * sizeof(std::uint32_t)));
    in.read(magic.data(), magic.size());
    if (!in || magic != kMagic) {
        out.message = "the expert state file is truncated";
        return out;
    }
    for (const std::uint32_t key : state.ranked) {
        if (key >= keys) {
            out.message = "the expert state file is damaged";
            return out;
        }
    }
    out.file = std::move(file);
    return out;
}

ExpertStateLoad load_expert_state(const std::filesystem::path& path, std::string_view identity, std::uint32_t keys) {
    ExpertStateLoad out;
    ExpertStateRead read = read_expert_state(path);
    if (!read.file) {
        out.message = std::move(read.message);
        return out;
    }
    if (read.file->identity != identity) {
        out.message = "the saved expert state belongs to another artifact";
        return out;
    }
    if (read.file->state.counts.size() != keys) {
        out.message = "the saved expert state has another expert count";
        return out;
    }
    out.state = std::move(read.file->state);
    return out;
}

std::string session_state_identity(const std::filesystem::path& artifact) {
    std::error_code error;
    const auto size = std::filesystem::file_size(artifact, error);
    const auto time = std::filesystem::last_write_time(artifact, error);
    return "artifact=" + std::filesystem::absolute(artifact).string() + ";size=" + std::to_string(error ? 0U : size) +
           ";mtime=" + std::to_string(error ? 0 : time.time_since_epoch().count());
}

std::string default_state_identity(std::string_view artifact_id_hex) {
    return "artifact_id=" + std::string(artifact_id_hex);
}

ExpertStateFiles resolve_expert_state_files(bool enabled, const std::filesystem::path& artifact,
                                            const std::filesystem::path& state_path,
                                            const std::filesystem::path& default_path,
                                            std::string_view artifact_id_hex) {
    ExpertStateFiles files;
    if (!enabled) { return files; }
    files.primary = state_path;
    if (files.primary.empty()) {
        files.primary = artifact;
        files.primary += ".expert-state";
    }
    files.primary_identity = session_state_identity(artifact);
    files.fallback         = default_path;
    if (files.fallback.empty()) {
        files.fallback = artifact;
        files.fallback += ".expert-state.default";
    }
    files.fallback_identity = default_state_identity(artifact_id_hex);
    // A default that is the session file would be written by the server: there is none then.
    if (files.fallback == files.primary) {
        files.fallback.clear();
        files.fallback_identity.clear();
    }
    return files;
}

WarmStartLoad load_warm_start(const ExpertStateFiles& files, std::uint32_t keys) {
    WarmStartLoad out;
    if (files.primary.empty()) {
        out.load.message = "the saved expert state is off";
        return out;
    }
    out.load = load_expert_state(files.primary, files.primary_identity, keys);
    if (out.load.state || files.fallback.empty()) { return out; }
    ExpertStateLoad fallback = load_expert_state(files.fallback, files.fallback_identity, keys);
    if (!fallback.state) {
        // A default that is simply not there is the normal case; only say why another one failed.
        if (std::filesystem::exists(files.fallback)) {
            out.load.message += "; the default state " + files.fallback.string() + " was not used: " + fallback.message;
        }
        return out;
    }
    out.load          = std::move(fallback);
    out.from_fallback = true;
    return out;
}

std::string export_expert_state(const std::filesystem::path& session, const std::filesystem::path& out,
                                std::string_view artifact_id_hex, std::string_view expected_session_identity) {
    if (artifact_id_hex.size() != 32 ||
        artifact_id_hex.find_first_not_of("0123456789abcdef") != std::string_view::npos) {
        return "the artifact id must be 32 lowercase hex digits";
    }
    std::error_code error;
    if (std::filesystem::equivalent(session, out, error)) { return "the output is the session file itself"; }
    const ExpertStateRead read = read_expert_state(session);
    if (!read.file) { return read.message; }
    if (!expected_session_identity.empty() && read.file->identity != expected_session_identity) {
        return "the session state was saved for another artifact (or another copy of this one: path, size or "
               "modification time differ); pass the artifact id instead to export it anyway";
    }
    return save_expert_state(out, default_state_identity(artifact_id_hex), read.file->state);
}

} // namespace ninfer::models::qwen4_exp::expert_cache
