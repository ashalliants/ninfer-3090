#pragma once

// Written for NInfer-3090.
//
// The folder `python3 -m tools.convert.dsv4_engram_volume GGUF DIR` writes: the Engram volume, the GGUF's
// token map and the manifest `engram.json` that ties them together. An artifact converted without
// `--ngram-out` records no volume and drops the token map and the multipliers, so the runtime binds this
// folder instead (`--engram-dir DIR`, or NINFER_DSV4_ENGRAM_DIR). `load_engram_sidecar` reads and checks it
// against the artifact's own config: the layers, n-gram size, heads, row width, pad token and bucket counts
// must be the ones the manifest was written with, the token map must have the checksum recorded, and the
// volume file must be the size and have the header (id included) the manifest names.

#include "models/dsv4/config.h"
#include "models/dsv4/execution/engram.h"

#include <filesystem>

namespace ninfer::models::dsv4 {

inline constexpr const char* kEngramManifestName = "engram.json";

struct EngramSidecar {
    std::filesystem::path volume_path;
    qwen4_exp::NgramTableConfig table;
    EngramHashParams params;
};

// Throws ArtifactError (naming the folder and what differs) when the folder is not a consistent Engram
// folder for `config`.
[[nodiscard]] EngramSidecar load_engram_sidecar(const std::filesystem::path& directory, const TextConfig& config);

} // namespace ninfer::models::dsv4
