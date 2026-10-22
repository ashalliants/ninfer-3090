#pragma once

#include <cstdint>

namespace ninfer::runtime {

// KV pages of the two typed pools. Backend is the draft model's cache and is zero without one.
struct KvPageCounts {
    std::uint64_t main    = 0;
    std::uint64_t backend = 0;
};

// Pages of a reused prefix that stay shared with their donor. An unaligned prefix's last page is
// partly filled: the new request copies it before appending, so that page is new and not shared.
[[nodiscard]] constexpr std::uint64_t shareable_prefix_pages(std::uint64_t prefix_frontier,
                                                             std::uint64_t page_size) noexcept {
    return prefix_frontier / page_size;
}

// Whether a fresh prompt's whole KV need can ever be met next to the pages that resident requests
// currently hold. Admission only reserves the first prefill chunk, so a prompt that fails this
// starts, fills the pool, is paused and later recomputes its history. Pages held only by cached
// checkpoints are not counted in `pinned`: reclaim can evict them.
//
// `need` counts only pages not already pinned (a prefix shared with a resident is already in
// `pinned`). `pinned` also includes pages the pool has promised to residents but not yet mapped, and
// `capacity` is what the pool can still hand out (pages lent away are excluded). With no resident the prompt is always let in, which keeps the queue moving.
[[nodiscard]] constexpr bool prompt_fits_beside_residents(KvPageCounts need, KvPageCounts pinned,
                                                          KvPageCounts capacity,
                                                          bool residents_present) noexcept {
    if (!residents_present) { return true; }
    return pinned.main + need.main <= capacity.main &&
           pinned.backend + need.backend <= capacity.backend;
}

} // namespace ninfer::runtime
