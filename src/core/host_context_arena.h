#pragma once

#include "core/arena.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

namespace ninfer {

class HostContextArena;

// A unique, immovable physical extent. Keeping this owner alive pins its bytes; views never
// allocate or charge a second copy. Transfer destinations remain reserved until publication.
class HostContextAllocation {
public:
    HostContextAllocation() noexcept = default;
    ~HostContextAllocation();
    HostContextAllocation(const HostContextAllocation&)            = delete;
    HostContextAllocation& operator=(const HostContextAllocation&) = delete;
    HostContextAllocation(HostContextAllocation&& other) noexcept;
    HostContextAllocation& operator=(HostContextAllocation&& other) noexcept;

    [[nodiscard]] bool valid() const noexcept { return owner_ != nullptr; }

    [[nodiscard]] std::size_t bytes() const noexcept { return bytes_; }

    [[nodiscard]] bool reserved() const noexcept { return valid() && reserved_; }

    [[nodiscard]] std::byte* data() const noexcept;
    void publish() noexcept;
    bool release() noexcept;

private:
    friend class HostContextArena;

    HostContextAllocation(HostContextArena& owner, std::size_t offset, std::size_t bytes,
                          bool reserved = true) noexcept
        : owner_(&owner), offset_(offset), bytes_(bytes), reserved_(reserved) {}

    void disarm() noexcept;

    HostContextArena* owner_ = nullptr;
    std::size_t offset_      = 0;
    std::size_t bytes_       = 0;
    bool reserved_           = false;
};

// One startup-fixed pinned backing shared by every optional Host context representation.
// minimum_allocation_bytes is the smallest supported complete geometry, not a quota split.
// It bounds metadata at startup; all allocations and split pieces must meet that minimum.
// The arena must outlive its allocations and the typed pools that own them.
class HostContextArena {
public:
    static constexpr std::size_t alignment = 256;

    HostContextArena(std::size_t capacity_bytes, std::size_t minimum_allocation_bytes);
    HostContextArena(const HostContextArena&)            = delete;
    HostContextArena& operator=(const HostContextArena&) = delete;
    HostContextArena(HostContextArena&&)                 = delete;
    HostContextArena& operator=(HostContextArena&&)      = delete;

    [[nodiscard]] std::size_t capacity_bytes() const noexcept { return capacity_bytes_; }

    [[nodiscard]] std::size_t minimum_allocation_bytes() const noexcept { return minimum_bytes_; }

    [[nodiscard]] std::size_t occupied_bytes() const noexcept { return occupied_bytes_; }

    [[nodiscard]] std::size_t reserved_bytes() const noexcept { return reserved_bytes_; }

    [[nodiscard]] std::size_t live_bytes() const noexcept {
        return occupied_bytes_ - reserved_bytes_;
    }

    [[nodiscard]] std::size_t free_bytes() const noexcept {
        return capacity_bytes_ - occupied_bytes_;
    }

    [[nodiscard]] std::size_t peak_occupied_bytes() const noexcept { return peak_occupied_bytes_; }

    [[nodiscard]] std::size_t allocation_count() const noexcept { return allocation_count_; }

    // How many separate free runs there are. Many small runs mean the backing is fragmented.
    [[nodiscard]] std::size_t free_extent_count() const noexcept { return free_extents_.size(); }

    // Allocations that found no extent, split by why: `geometry` when enough bytes were free in
    // total but no single run fit, `shortage` when the bytes were simply not there. Monotonic.
    // Inside a PlacementScope the misses are not counted; the scope counts one failure for the
    // whole placement if it fails.
    [[nodiscard]] std::uint64_t geometry_failures() const noexcept { return geometry_failures_; }

    [[nodiscard]] std::uint64_t shortage_failures() const noexcept { return shortage_failures_; }

    // A placement that tries several allocation sizes before it settles (reserve_in_chunks halves
    // a run that does not fit) is one placement, not one per size it tried: its misses are
    // expected and would otherwise swamp the failure counters. Call `failed` with the bytes the
    // whole placement needed when it could not be made at all.
    class [[nodiscard]] PlacementScope {
    public:
        explicit PlacementScope(HostContextArena& arena) noexcept : arena_(arena) {
            ++arena_.placement_depth_;
        }
        ~PlacementScope() { --arena_.placement_depth_; }
        PlacementScope(const PlacementScope&)            = delete;
        PlacementScope& operator=(const PlacementScope&) = delete;

        void failed(std::size_t bytes) noexcept { arena_.count_failure(bytes); }

    private:
        HostContextArena& arena_;
    };

    [[nodiscard]] bool can_allocate(std::size_t bytes) const noexcept;
    [[nodiscard]] std::optional<HostContextAllocation> allocate(std::size_t bytes) noexcept;
    [[nodiscard]] std::pair<HostContextAllocation, HostContextAllocation>
    split(HostContextAllocation&& allocation, std::size_t byte_offset);

private:
    friend class HostContextAllocation;

    struct FreeExtent {
        std::size_t offset = 0;
        std::size_t bytes  = 0;
    };

    [[nodiscard]] std::size_t allocation_bytes(std::size_t bytes) const noexcept;
    [[nodiscard]] std::optional<std::size_t> find_free_extent(std::size_t bytes) const noexcept;
    void insert_free_extent(FreeExtent extent) noexcept;
    void release(std::size_t offset, std::size_t bytes, bool reserved) noexcept;
    void count_failure(std::size_t rounded_bytes) noexcept;

    std::optional<PinnedHostBuffer> backing_;
    std::size_t capacity_bytes_      = 0;
    std::size_t minimum_bytes_       = 0;
    std::size_t occupied_bytes_      = 0;
    std::size_t reserved_bytes_      = 0;
    std::size_t peak_occupied_bytes_ = 0;
    std::size_t allocation_count_    = 0;
    std::uint64_t geometry_failures_ = 0;
    std::uint64_t shortage_failures_ = 0;
    std::size_t placement_depth_     = 0;
    std::vector<FreeExtent> free_extents_;
};

} // namespace ninfer
