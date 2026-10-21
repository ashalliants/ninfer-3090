#pragma once

#include <cstdint>

namespace ninfer::runtime {

// KV pages of the two typed pools. Backend is the draft model's cache and is zero without one.
struct KvPageCounts {
    std::uint64_t main    = 0;
    std::uint64_t backend = 0;
};

// Whether a fresh prompt's whole KV need can ever be met next to the pages that resident requests
// currently hold. Admission only reserves the first prefill chunk, so a prompt that fails this
// starts, fills the pool, is paused and later recomputes its history. Pages held only by cached
// checkpoints are not counted in `pinned`: reclaim can evict them.
//
// `need` counts only pages not already pinned (a prefix shared with a resident is already in
// `pinned`). With no resident the prompt is always let in, which keeps the queue moving.
[[nodiscard]] constexpr bool prompt_fits_beside_residents(KvPageCounts need, KvPageCounts pinned,
                                                          KvPageCounts capacity,
                                                          bool residents_present) noexcept {
    if (!residents_present) { return true; }
    return pinned.main + need.main <= capacity.main &&
           pinned.backend + need.backend <= capacity.backend;
}

} // namespace ninfer::runtime
