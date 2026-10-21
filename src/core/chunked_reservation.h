#pragma once

#include <cstddef>
#include <span>
#include <utility>

namespace ninfer {

// Reserves a destination for every item, splitting the list into consecutive chunks, each as large as
// the allocator can place. The Host arena hands out contiguous runs, so a request whose pages
// no longer fit in one run (the arena is fragmented even though enough bytes are free in total)
// still succeeds as several smaller reservations.
//
// `reserve(span)` returns an optional reservation for exactly those items. `accept(reservation&&,
// span)` takes ownership of each reservation as soon as it is made. Returns false when even a
// single item cannot be placed; the caller unwinds whatever `accept` already took.
template <class Item, class Reserve, class Accept>
[[nodiscard]] bool reserve_in_chunks(std::span<const Item> items, Reserve&& reserve,
                                     Accept&& accept) {
    std::size_t offset = 0;
    while (offset < items.size()) {
        const auto remaining = items.size() - offset;
        const auto chunk     = [&](std::size_t n) { return items.subspan(offset, n); };
        auto count           = remaining;
        // Halve until something fits, then search between that size and the last size that failed
        // for the largest one that still fits, so a run that holds 6 of 8 items is not left using
        // only 4. A reservation is released before a larger one is tried (the larger one may need
        // the very run it holds), so the size already known to fit is re-reserved on a miss.
        auto held          = reserve(chunk(count));
        std::size_t failed = remaining + 1;
        while (!held && count) {
            failed = count;
            count /= 2;
            if (count) { held = reserve(chunk(count)); }
        }
        if (!held) { return false; }
        auto high = failed - 1;
        while (count < high) {
            const auto mid = count + (high - count + 1) / 2;
            held.reset();
            auto larger = reserve(chunk(mid));
            if (larger) {
                held  = std::move(larger);
                count = mid;
            } else {
                high = mid - 1;
                held = reserve(chunk(count));
                if (!held) { return false; }
            }
        }
        accept(std::move(*held), chunk(count));
        offset += count;
    }
    return true;
}

} // namespace ninfer
