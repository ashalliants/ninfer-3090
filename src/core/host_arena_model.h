#pragma once

#include "core/chunked_reservation.h"
#include "core/host_context_arena.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <utility>
#include <vector>

namespace ninfer {

// A value copy of a HostContextArena's free runs that replays allocations without touching the
// arena: the same rounding, the same first-fit choice and the same merging of released runs, so
// a sequence of allocations succeeds here exactly when it would succeed there. Used to ask, before
// anything is evicted or moved, whether a placement that takes several allocations in turn (a
// pause snapshot: a state image, then KV pages split over runs) would fit. It models the arena's
// byte placement only, inside a SnapshotScope (the whole backing is usable); descriptor limits of
// the typed pools are not modelled.
class HostArenaModel {
public:
    struct Extent {
        std::size_t offset = 0;
        std::size_t bytes  = 0;
    };

    // Bytes taken from the model, given back when it is destroyed unless dismissed.
    class Allocation {
    public:
        Allocation(HostArenaModel& model, Extent extent) noexcept
            : model_(&model), extent_(extent) {}
        Allocation(Allocation&& other) noexcept
            : model_(std::exchange(other.model_, nullptr)), extent_(other.extent_) {}
        Allocation& operator=(Allocation&&) = delete;
        Allocation(const Allocation&)       = delete;
        Allocation& operator=(const Allocation&) = delete;
        ~Allocation() {
            if (model_) { model_->release(extent_); }
        }

        // Keeps the bytes taken for the rest of the placement.
        void dismiss() noexcept { model_ = nullptr; }

    private:
        HostArenaModel* model_;
        Extent extent_;
    };

    HostArenaModel(std::vector<Extent> free_runs, std::size_t minimum_allocation_bytes)
        : free_(std::move(free_runs)), minimum_(minimum_allocation_bytes) {}

    explicit HostArenaModel(const HostContextArena& arena)
        : minimum_(arena.minimum_allocation_bytes()) {
        for (const auto& extent : arena.free_extents()) {
            free_.push_back({extent.offset, extent.bytes});
        }
    }

    // The bytes an allocation of `bytes` takes, or 0 when the arena would refuse the size.
    [[nodiscard]] std::size_t allocation_bytes(std::size_t bytes) const noexcept {
        constexpr std::size_t alignment = HostContextArena::alignment;
        if (bytes == 0 || bytes > std::numeric_limits<std::size_t>::max() - (alignment - 1)) {
            return 0;
        }
        const std::size_t rounded = (bytes + alignment - 1) & ~(alignment - 1);
        return rounded < minimum_ ? 0 : rounded;
    }

    [[nodiscard]] std::optional<Allocation> allocate(std::size_t bytes) {
        const std::size_t rounded = allocation_bytes(bytes);
        if (rounded == 0) { return std::nullopt; }
        for (std::size_t index = 0; index < free_.size(); ++index) {
            if (free_[index].bytes < rounded) { continue; }
            const Extent taken{free_[index].offset, rounded};
            free_[index].offset += rounded;
            free_[index].bytes -= rounded;
            if (free_[index].bytes == 0) {
                free_.erase(free_.begin() + static_cast<std::ptrdiff_t>(index));
            }
            return Allocation(*this, taken);
        }
        return std::nullopt;
    }

private:
    void release(Extent extent) {
        const auto position = std::lower_bound(
            free_.begin(), free_.end(), extent.offset,
            [](const Extent& candidate, std::size_t offset) { return candidate.offset < offset; });
        auto inserted = free_.insert(position, extent);
        if (inserted != free_.begin()) {
            auto previous = inserted - 1;
            if (previous->offset + previous->bytes == inserted->offset) {
                previous->bytes += inserted->bytes;
                free_.erase(inserted);
                inserted = previous;
            }
        }
        const auto next = inserted + 1;
        if (next != free_.end() && inserted->offset + inserted->bytes == next->offset) {
            inserted->bytes += next->bytes;
            free_.erase(next);
        }
    }

    std::vector<Extent> free_;
    std::size_t minimum_ = 0;
};

// What a pause snapshot places in the Host arena: the state image (0 when it is already there),
// then each KV pool's pages missing from it, each page `stride` bytes.
struct PauseHostNeed {
    struct Pool {
        std::size_t pages  = 0;
        std::size_t stride = 0;
    };
    std::size_t image_bytes = 0;
    Pool text;
    Pool backend;
};

// Whether the pause snapshot places, replaying what the pause does in its order (ProgramImpl::
// prepare_backup): the image takes one run, then each pool's pages are placed in as few runs as fit
// (reserve_in_chunks), each reservation staying taken. `model` is a copy; the arena is untouched.
[[nodiscard]] inline bool pause_snapshot_places(HostArenaModel model, const PauseHostNeed& need) {
    if (need.image_bytes) {
        auto image = model.allocate(need.image_bytes);
        if (!image) { return false; }
        image->dismiss();
    }
    for (const auto& pool : {need.text, need.backend}) {
        if (pool.pages == 0) { continue; }
        // Only the number of pages matters to the placement, not which they are.
        const std::vector<std::uint32_t> pages(pool.pages);
        const bool placed = reserve_in_chunks<std::uint32_t>(
            std::span<const std::uint32_t>(pages),
            [&](std::span<const std::uint32_t> chunk) {
                return model.allocate(chunk.size() * pool.stride);
            },
            [](HostArenaModel::Allocation&& held, std::span<const std::uint32_t>) {
                held.dismiss();
            });
        if (!placed) { return false; }
    }
    return true;
}

} // namespace ninfer
