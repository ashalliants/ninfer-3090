#pragma once

// Where each routed expert's record sits in the artifact file, and a reader that fetches a set of them
// from the SSD with their device latency overlapped. This is the SSD tier's I/O layer
// (docs/maintainer/dsv4-flash-plan.md, sections 7 and 8, milestone 2): it knows offsets and sizes and
// nothing about models, quantisation or the GPU, so it is tested on any machine with a disk.
//
// An extent starts on a kExtentAlignment boundary and is read in whole blocks (unbuffered I/O), so the
// file must extend to the end of the last extent's padded size. Records of different sizes (per-layer
// quantisation) are read in groups of one padded size, because ReadOnlyFile::read_direct_blocks reads
// one block size per call; a model has a handful of sizes, so a request makes a handful of calls.

#include "core/read_only_file.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <span>
#include <vector>

namespace ninfer {

inline constexpr std::uint64_t kExtentAlignment = 4096; // what unbuffered I/O needs of offsets, sizes, buffers

struct ExpertExtent {
    std::uint64_t offset = 0; // a multiple of kExtentAlignment
    std::uint64_t bytes  = 0; // the record's size; the file holds it padded up to kExtentAlignment
};

[[nodiscard]] constexpr std::uint64_t padded_extent_bytes(std::uint64_t bytes) noexcept {
    return (bytes + kExtentAlignment - 1) / kExtentAlignment * kExtentAlignment;
}

struct PackedExtents {
    std::vector<ExpertExtent> extents; // parallel to the input sizes
    std::uint64_t end = 0;             // the file size the extents need (the last one's padded end)
};

// Lays records of the given sizes one after another from `first_offset` (a multiple of
// kExtentAlignment), each padded to the alignment. Throws std::invalid_argument for a zero-byte record or
// a misaligned first_offset, and std::overflow_error when the layout would pass the largest file offset.
[[nodiscard]] PackedExtents pack_expert_extents(std::span<const std::uint64_t> record_bytes,
                                                std::uint64_t first_offset = 0);

class ExpertExtentReader {
public:
    // Receives the position in the request (not the extent index) and the record's bytes, which are
    // valid only until it returns (the ring block is then reused). Runs on the calling thread.
    using Consume = std::function<void(std::size_t, std::span<const std::byte>)>;

    // `file` must outlive the reader. Throws std::invalid_argument for a misaligned or zero-byte
    // extent, and std::out_of_range for one whose padded end is past the file's current size.
    ExpertExtentReader(const ReadOnlyFile& file, std::vector<ExpertExtent> extents);

    [[nodiscard]] std::size_t size() const noexcept { return extents_.size(); }
    [[nodiscard]] const ExpertExtent& extent(std::size_t index) const { return extents_.at(index); }

    // The ring that reads `wanted` (extent indices) with `depth` reads in flight: depth blocks of
    // the largest padded size. Use it to size the pinned ring.
    [[nodiscard]] std::uint64_t ring_bytes(std::span<const std::uint32_t> wanted, std::size_t depth) const;

    // Reads every wanted extent into `ring` blocks and calls consume(i, record) for wanted[i], in
    // completion order within a size group (groups run smallest first). `ring` must be
    // kExtentAlignment-aligned and hold at least one block of the largest wanted padded size; more
    // blocks mean more reads in flight. A bad index throws std::out_of_range before any I/O, and a
    // failed or short read throws after the reads in flight finish.
    void read(std::span<const std::uint32_t> wanted, std::span<std::byte> ring, const Consume& consume) const;

private:
    const ReadOnlyFile& file_;
    std::vector<ExpertExtent> extents_;
};

} // namespace ninfer
