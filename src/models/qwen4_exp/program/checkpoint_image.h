#pragma once

// Checkpoint images of the Qwen3.8-Flash-Next context cache (docs/maintainer/flash-next-context-cache.md):
// the byte form the Engine's context store keeps, so a conversation survives a restart.
//
// An image holds a set of checkpoints (one continuation's recovery points, deepest first, or one
// shared prefix alone) in the Qwen3.5 Program's container (storage/checkpoint_image_codec.h): the
// magic, this format's version, the model binding, a Flash-Next configuration section, the session,
// every exact identity once, a point table, then the payload: every KV page once, in the order the
// points (deepest first) first reach them, so successive images of one conversation share their
// leading pages byte for byte in the content-addressed store, then every state image once, and a
// trailing checksum. Pages and state images are stored compact (each plane's bytes back to back,
// none of a slab's alignment padding), so the payload does not depend on where the slabs were.
//
// Import validates everything (binding, configuration, checksum, table bounds, identities against
// their keys) before it takes a slab, and rejects any mismatch with std::invalid_argument; it returns
// nullopt when the tier has no room or cannot store the bytes.

#include "models/qwen4_exp/program/context_cache.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string_view>
#include <vector>

namespace ninfer::models::qwen4_exp {

// Where a checkpoint's bytes sit in its slabs, and what an image must match to mean the same thing.
struct CheckpointLayout {
    std::size_t slab_bytes = 0; // one page record
    // Per attention layer (the 12 QSA layers, then the drafter's): its five planes' offsets in a page
    // record and their bytes per page.
    std::vector<std::array<std::size_t, 5>> plane_offsets;
    std::vector<std::array<std::size_t, 5>> plane_bytes;
    // The state image in its slabs: the state block [0, state_bytes), the drafter's saved residual
    // at residual_offset (residual_bytes, 0 without the drafter) and the last position's logits at
    // logits_offset (logits_bytes).
    std::size_t state_bytes     = 0;
    std::size_t residual_offset = 0, residual_bytes = 0;
    std::size_t logits_offset   = 0, logits_bytes = 0;
    std::uint32_t image_slabs   = 0;
    bool vision                 = false;
    std::uint32_t identity_tag  = 0;
    // Bounds for import: the context (tokens) and the public token domain.
    std::uint32_t capacity = 0;
    std::int64_t vocabulary = 0;

    [[nodiscard]] std::size_t page_bytes() const noexcept;
    [[nodiscard]] std::size_t image_bytes() const noexcept { return state_bytes + residual_bytes + logits_bytes; }
};

// The image of `points` (1 to 64 distinct valid checkpoints; a SharedPrefix one only alone and
// without a session). Throws std::invalid_argument for a set that cannot be exported, and what the
// tier throws when it cannot read the slabs.
[[nodiscard]] qwen3_5::CheckpointImage export_checkpoint_image(const HostCheckpointStore& store, CheckpointTier& tier,
                                                               const CheckpointLayout& layout,
                                                               std::span<const qwen3_5::CheckpointHandle> points,
                                                               const std::optional<qwen3_5::PreparedSessionKey>& session,
                                                               std::string_view binding);

// The slab bytes an import of `image` takes. Throws std::invalid_argument for an image this layout
// and binding cannot import.
[[nodiscard]] std::size_t checkpoint_image_slab_bytes(const CheckpointLayout& layout, std::span<const std::uint8_t> image,
                                                      std::string_view binding);

// Imports `image` as checkpoints in `store` (deepest first). Throws std::invalid_argument as
// checkpoint_image_slab_bytes does; nullopt (nothing kept) when the store has too few free slabs or
// the tier cannot store the bytes.
[[nodiscard]] std::optional<qwen3_5::ImportedCheckpoints>
import_checkpoint_image(HostCheckpointStore& store, CheckpointTier& tier, const CheckpointLayout& layout,
                        std::span<const std::uint8_t> image, std::string_view binding);

} // namespace ninfer::models::qwen4_exp
