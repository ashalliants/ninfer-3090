#include "core/expert_extents.h"

#include <algorithm>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>

namespace ninfer {

PackedExtents pack_expert_extents(std::span<const std::uint64_t> record_bytes, std::uint64_t first_offset) {
    if (first_offset % kExtentAlignment != 0) {
        throw std::invalid_argument("expert extents: first_offset is not a multiple of " +
                                    std::to_string(kExtentAlignment));
    }
    constexpr auto max_offset = static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max());
    PackedExtents out;
    out.extents.reserve(record_bytes.size());
    std::uint64_t at = first_offset;
    for (const std::uint64_t bytes : record_bytes) {
        if (bytes == 0) { throw std::invalid_argument("expert extents: a record has zero bytes"); }
        if (bytes > max_offset - kExtentAlignment) { throw std::overflow_error("expert extents: record too large"); }
        const std::uint64_t padded = padded_extent_bytes(bytes);
        if (at > max_offset || padded > max_offset - at) {
            throw std::overflow_error("expert extents: layout passes the largest file offset");
        }
        out.extents.push_back({at, bytes});
        at += padded;
    }
    out.end = at;
    return out;
}

ExpertExtentReader::ExpertExtentReader(const ReadOnlyFile& file, std::vector<ExpertExtent> extents)
    : file_(file), extents_(std::move(extents)) {
    const std::uint64_t file_bytes = file_.current_bytes();
    for (std::size_t i = 0; i < extents_.size(); ++i) {
        const auto& e = extents_[i];
        if (e.bytes == 0 || e.offset % kExtentAlignment != 0) {
            throw std::invalid_argument("expert extent " + std::to_string(i) + " is empty or misaligned");
        }
        // Before padding: a record larger than the file is refused here, so padding (which adds up
        // to 4095) cannot wrap and every padded size below is exact.
        if (e.bytes > file_bytes) {
            throw std::out_of_range("expert extent " + std::to_string(i) + " is larger than the file");
        }
        const std::uint64_t padded = padded_extent_bytes(e.bytes);
        if (e.offset > file_bytes || padded > file_bytes - e.offset) {
            throw std::out_of_range("expert extent " + std::to_string(i) + " ends past the file");
        }
    }
}

std::uint64_t ExpertExtentReader::ring_bytes(std::span<const std::uint32_t> wanted, std::size_t depth) const {
    std::uint64_t largest = 0;
    for (const std::uint32_t index : wanted) {
        largest = std::max(largest, padded_extent_bytes(extents_.at(index).bytes));
    }
    return largest * std::max<std::size_t>(depth, 1);
}

void ExpertExtentReader::read(std::span<const std::uint32_t> wanted, std::span<std::byte> ring,
                              const Consume& consume) const {
    for (const std::uint32_t index : wanted) {
        if (index >= extents_.size()) { throw std::out_of_range("expert extent index " + std::to_string(index)); }
    }
    if (wanted.empty()) { return; }
    // Positions in the request, grouped by padded size (stable: a group keeps the caller's order).
    std::vector<std::size_t> order(wanted.size());
    std::iota(order.begin(), order.end(), std::size_t{0});
    const auto padded_of = [&](std::size_t position) { return padded_extent_bytes(extents_[wanted[position]].bytes); };
    std::stable_sort(order.begin(), order.end(),
                     [&](std::size_t a, std::size_t b) { return padded_of(a) < padded_of(b); });

    if (reinterpret_cast<std::uintptr_t>(ring.data()) % kExtentAlignment != 0) {
        throw std::invalid_argument("expert extents: the ring is not " + std::to_string(kExtentAlignment) +
                                    "-byte aligned");
    }
    if (ring.size() < padded_of(order.back())) {
        throw std::invalid_argument("expert extents: the ring is smaller than one block of the largest extent");
    }
    std::vector<std::uint64_t> offsets;
    for (std::size_t first = 0; first < order.size();) {
        const std::uint64_t block = padded_of(order[first]);
        std::size_t last          = first;
        offsets.clear();
        while (last < order.size() && padded_of(order[last]) == block) {
            offsets.push_back(extents_[wanted[order[last]]].offset);
            ++last;
        }
        const std::span<std::byte> group_ring = ring.first(ring.size() / block * block);
        file_.read_direct_blocks(offsets, block, group_ring, [&](std::size_t i, std::span<const std::byte> bytes) {
            const std::size_t position = order[first + i];
            consume(position, bytes.first(extents_[wanted[position]].bytes));
        });
        first = last;
    }
}

} // namespace ninfer
