#pragma once

#include <cstddef>
#include <span>
#include <utility>

namespace ninfer {

// Reserves a destination for every item, splitting the list into the fewest consecutive chunks
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
        auto count = items.size() - offset;
        while (count) {
            auto reservation = reserve(items.subspan(offset, count));
            if (reservation) {
                accept(std::move(*reservation), items.subspan(offset, count));
                break;
            }
            count /= 2;
        }
        if (!count) { return false; }
        offset += count;
    }
    return true;
}

} // namespace ninfer
