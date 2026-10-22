#pragma once

#include <algorithm>
#include <cstddef>
#include <optional>

namespace ninfer::runtime {

// Freeing Host bytes for a pause snapshot counts total free bytes, but the state image needs one
// contiguous run. When the bytes are free yet no run is big enough, the next round must free more.
// Returns how many bytes to ask the victim search to keep free next (clamped to the Host budget:
// freeing everything evictable is the last way to coalesce a run), or nullopt when nothing more can
// be freed or the image is larger than the whole budget.
[[nodiscard]] inline std::optional<std::size_t>
next_pause_host_target(std::size_t free_bytes, std::size_t unplaced_state_image_bytes,
                       std::size_t capacity_bytes) noexcept {
    if (unplaced_state_image_bytes == 0 || unplaced_state_image_bytes > capacity_bytes ||
        free_bytes >= capacity_bytes) {
        return std::nullopt;
    }
    return std::min(capacity_bytes, free_bytes + unplaced_state_image_bytes);
}

// Each round must free something new, and the loop stays bounded even if the layout keeps
// refusing: eviction costs cached history, so it must not run away.
inline constexpr std::size_t kMaxPauseHostRounds = 8;

} // namespace ninfer::runtime
