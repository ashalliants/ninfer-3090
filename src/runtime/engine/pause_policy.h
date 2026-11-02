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

// A pause snapshot may use what is free in the snapshot reserve, which the context cache's budget
// does not include; eviction only has to make up the rest.
[[nodiscard]] inline std::size_t pause_reserve_free(std::size_t reserve_bytes,
                                                    std::size_t reserve_occupied_bytes) noexcept {
    return reserve_bytes - std::min(reserve_bytes, reserve_occupied_bytes);
}

// The free budget bytes the victim search must reach for a snapshot of `snapshot_bytes` to fit.
// It can exceed the budget: later paused snapshots sit in the reserve, and releasing them is part
// of reaching it, so the search is allowed targets up to the budget plus the reserve.
[[nodiscard]] inline std::size_t pause_eviction_target(std::size_t snapshot_bytes,
                                                       std::size_t reserve_free) noexcept {
    return snapshot_bytes > reserve_free ? snapshot_bytes - reserve_free : 0;
}

// Whether the free budget bytes, plus the reserve bytes eviction has freed since the target was
// set (releasing a snapshot that sat in the reserve frees reserve bytes, not budget bytes), cover
// `target`.
[[nodiscard]] inline bool pause_target_reached(std::size_t free_budget_bytes,
                                               std::size_t reserve_free_at_target,
                                               std::size_t reserve_free_now,
                                               std::size_t target) noexcept {
    const std::size_t gained =
        reserve_free_now > reserve_free_at_target ? reserve_free_now - reserve_free_at_target : 0;
    return free_budget_bytes + gained >= target;
}

// Each round must free something new, and the loop stays bounded even if the layout keeps
// refusing: eviction costs cached history, so it must not run away.
inline constexpr std::size_t kMaxPauseHostRounds = 8;

} // namespace ninfer::runtime
