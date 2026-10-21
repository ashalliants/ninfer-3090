#pragma once

#include <cstddef>
#include <optional>

namespace ninfer::runtime {

// Freeing Host bytes for a pause snapshot counts total free bytes, but the state image needs one
// contiguous run. When the bytes are free yet no run is big enough, the next round must free more.
// Returns how many bytes to ask the victim search to keep free next, or nullopt when asking for
// more cannot help (the Host budget cannot hold that much).
[[nodiscard]] inline std::optional<std::size_t>
next_pause_host_target(std::size_t free_bytes, std::size_t unplaced_state_image_bytes,
                       std::size_t capacity_bytes) noexcept {
    if (unplaced_state_image_bytes == 0 || unplaced_state_image_bytes > capacity_bytes ||
        free_bytes > capacity_bytes - unplaced_state_image_bytes) {
        return std::nullopt;
    }
    return free_bytes + unplaced_state_image_bytes;
}

// Each round must free something new, and the loop stays bounded even if the layout keeps
// refusing: eviction costs cached history, so it must not run away.
inline constexpr std::size_t kMaxPauseHostRounds = 8;

} // namespace ninfer::runtime
