#pragma once

// Adapted from Infernix a3edb450 src/models/qwen4_exp/program/expert_cache/expert_state.h (Apache-2.0).
// Modified for NInfer-3090: NInfer's namespace.

// The expert cache's saved state (design §19.3.5 S4b, warm start): every expert's LFRU count and the
// resident experts, highest score first, written at stop and between requests and read at the next
// start to fill the VRAM frames before the first request. A file is used only when its identity
// (the artifact and the key count) matches; anything else is ignored with a reason.
//
// Two kinds of file carry two identities. A session file, written by a running server beside the
// model, is tied to that copy of the artifact (path, size and modification time). A default file,
// shipped beside the model and read when no usable session file exists, is tied to the artifact id
// alone, so every copy of the same conversion accepts it; it is written only by
// export_expert_state, never by a server.

#include "models/qwen4_exp/program/expert_cache/expert_cache.h"

#include <filesystem>
#include <optional>
#include <string>
#include <cstdint>
#include <string_view>

namespace ninfer::models::qwen4_exp::expert_cache {

struct ExpertStateLoad {
    std::optional<SavedState> state;
    std::string message; // why nothing was loaded
};

// Writes `state` to `path` through `path.tmp` and a rename; returns an error message, empty on success.
[[nodiscard]] std::string save_expert_state(const std::filesystem::path& path, std::string_view identity,
                                            const SavedState& state);

// Reads a state saved for `identity` with `keys` keys.
[[nodiscard]] ExpertStateLoad load_expert_state(const std::filesystem::path& path, std::string_view identity,
                                                std::uint32_t keys);

// A state file as written: its identity and contents, with no identity or key count expected.
struct ExpertStateFile {
    std::string identity;
    SavedState state;
};
struct ExpertStateRead {
    std::optional<ExpertStateFile> file;
    std::string message; // why nothing was read
};
[[nodiscard]] ExpertStateRead read_expert_state(const std::filesystem::path& path);

// The identity of a session file: this copy of the artifact (its absolute path, size and
// modification time).
[[nodiscard]] std::string session_state_identity(const std::filesystem::path& artifact);

// The identity of a default file: the artifact id as artifact::artifact_id_hex writes it.
[[nodiscard]] std::string default_state_identity(std::string_view artifact_id_hex);

// Where a server reads and writes its expert state. `primary` is the session file (read at start,
// written at stop and between requests); `fallback` is the shipped default (read only, and only when
// the primary is missing or belongs to another artifact). All four are empty when the state is off
// (`--no-expert-state`): nothing is read, not even the default.
struct ExpertStateFiles {
    std::filesystem::path primary;
    std::string primary_identity;
    std::filesystem::path fallback;
    std::string fallback_identity;
};
// `state_path` and `default_path` empty select `<artifact>.expert-state` and
// `<artifact>.expert-state.default`.
[[nodiscard]] ExpertStateFiles resolve_expert_state_files(bool enabled, const std::filesystem::path& artifact,
                                                          const std::filesystem::path& state_path,
                                                          const std::filesystem::path& default_path,
                                                          std::string_view artifact_id_hex);

struct WarmStartLoad {
    ExpertStateLoad load;
    bool from_fallback = false; // load.state came from the default file
};
// The warm start's state: the primary file when it loads, else the fallback (when there is one).
// A fallback is read only after the primary failed to load.
[[nodiscard]] WarmStartLoad load_warm_start(const ExpertStateFiles& files, std::uint32_t keys);

// Rewrites the session file `session` as a default file for the artifact with id `artifact_id_hex`
// (32 lowercase hex digits), at `out`. When `expected_session_identity` is not empty the session
// file must carry it, so a state saved for another artifact is not exported by mistake. Returns an
// error message, empty on success.
[[nodiscard]] std::string export_expert_state(const std::filesystem::path& session, const std::filesystem::path& out,
                                              std::string_view artifact_id_hex,
                                              std::string_view expected_session_identity = {});

} // namespace ninfer::models::qwen4_exp::expert_cache
