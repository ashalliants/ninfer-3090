#include "core/host_context_arena.h"

#include <cuda_runtime_api.h>

#include <cstring>
#include <iostream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace {
int failures = 0;

void expect(bool condition, std::string_view message) {
    if (!condition) {
        ++failures;
        std::cerr << "FAIL: " << message << '\n';
    }
}

void test_reserve_publish_and_rollback() {
    ninfer::HostContextArena arena(1025, 1);
    expect(arena.capacity_bytes() == 1025 && arena.occupied_bytes() == 0,
           "idle fixed backing remains fully charged");
    auto source = arena.allocate(257);
    expect(source && source->bytes() == 512 && arena.reserved_bytes() == 512,
           "allocation includes alignment padding immediately");
    std::memset(source->data(), 0x51, source->bytes());
    source->publish();
    source->publish();
    expect(arena.live_bytes() == 512 && arena.reserved_bytes() == 0,
           "publication charges one unique live allocation");
    {
        auto target = arena.allocate(257);
        expect(target && target->data() != source->data() && arena.occupied_bytes() == 1024 &&
                   arena.live_bytes() == 512 && arena.reserved_bytes() == 512,
               "whole destination is reserved while the source remains resident");
        expect(!arena.allocate(1) && !arena.can_allocate(1),
               "unused trailing byte cannot satisfy aligned allocation");
    }
    expect(arena.occupied_bytes() == 512 && arena.reserved_bytes() == 0 &&
               arena.peak_occupied_bytes() == 1024,
           "cancelled destination rolls back occupancy without losing peak");
    expect(source->data()[0] == std::byte{0x51} && source->data()[511] == std::byte{0x51},
           "failed destination preserves the pinned source");
    auto moved = std::move(*source);
    expect(!source->valid() && arena.allocation_count() == 1,
           "moving ownership neither duplicates nor releases bytes");
    expect(moved.release() && !moved.release() && arena.occupied_bytes() == 0 &&
               arena.capacity_bytes() == 1025,
           "last owner release frees reusable bytes without hiding fixed backing");
    expect(!arena.allocate(0) && !arena.allocate(std::numeric_limits<std::size_t>::max()),
           "zero and overflowing requests fail without changing accounting");
}

void test_fragmentation_and_split() {
    ninfer::HostContextArena arena(2048, 256);
    auto full = arena.allocate(2048);
    full->publish();
    std::vector<ninfer::HostContextAllocation> pieces;
    pieces.reserve(8);
    while (full->bytes() > 256) {
        auto split = arena.split(std::move(*full), 256);
        pieces.push_back(std::move(split.first));
        *full = std::move(split.second);
    }
    pieces.push_back(std::move(*full));
    expect(arena.allocation_count() == 8 && arena.live_bytes() == 2048,
           "split reaches the smallest geometry without changing charged bytes");
    for (std::size_t index = 0; index < pieces.size(); index += 2) {
        (void)pieces[index].release();
    }
    expect(arena.free_bytes() == 1024 && !arena.can_allocate(512) && !arena.allocate(512),
           "actual contiguous extents, not total free bytes, govern availability");
    (void)pieces[1].release();
    auto merged = arena.allocate(768);
    expect(merged.has_value(), "adjacent released split pieces coalesce for another geometry");
    bool invalid_split_rejected = false;
    try {
        (void)arena.split(std::move(*merged), 1);
    } catch (const std::out_of_range&) { invalid_split_rejected = true; }
    expect(invalid_split_rejected && merged->valid() && merged->bytes() == 768,
           "invalid split leaves its source allocation owned");
    pieces.clear();
    merged.reset();
    expect(arena.occupied_bytes() == 0 && arena.allocation_count() == 0 && arena.can_allocate(2048),
           "all split pieces return to one complete reusable extent");
    ninfer::HostContextArena disabled(0, 256);
    expect(!disabled.allocate(256) && disabled.capacity_bytes() == 0,
           "Host zero has no backing or allocations");
}

void test_failures_are_counted_by_cause() {
    constexpr std::size_t unit = ninfer::HostContextArena::alignment;
    ninfer::HostContextArena arena(4 * unit, unit);
    expect(arena.geometry_failures() == 0 && arena.shortage_failures() == 0 &&
               arena.free_extent_count() == 1,
           "a fresh arena has one free run and no failures");
    auto first  = arena.allocate(unit);
    auto second = arena.allocate(unit);
    auto third  = arena.allocate(unit);
    auto fourth = arena.allocate(unit);
    expect(first && second && third && fourth && !arena.allocate(unit),
           "a full arena refuses another allocation");
    expect(arena.shortage_failures() == 1 && arena.geometry_failures() == 0,
           "no free bytes is a shortage, not a geometry failure");
    (void)first->release();
    (void)third->release();
    expect(arena.free_bytes() == 2 * unit && arena.free_extent_count() == 2 &&
               !arena.allocate(2 * unit),
           "two separated free units cannot satisfy a two-unit request");
    expect(arena.geometry_failures() == 1 && arena.shortage_failures() == 1,
           "enough total bytes but no single run is a geometry failure");
    expect(!arena.allocate(3 * unit) && arena.shortage_failures() == 2,
           "a request larger than all free bytes is a shortage");
    (void)arena.can_allocate(2 * unit);
    expect(arena.geometry_failures() == 1, "asking whether an allocation would fit never counts");
    expect(arena.allocate(unit) && arena.geometry_failures() == 1 && arena.shortage_failures() == 2,
           "a successful allocation changes no counter");
}

// A placement that tries smaller sizes after a miss (reserve_in_chunks) counts one failure for the
// whole placement, and none when it succeeds in pieces (#332: every probe used to count).
void test_placement_scope_counts_once() {
    constexpr std::size_t unit = ninfer::HostContextArena::alignment;
    ninfer::HostContextArena arena(4 * unit, unit);
    auto first  = arena.allocate(unit);
    auto second = arena.allocate(unit);
    auto third  = arena.allocate(unit);
    auto fourth = arena.allocate(unit);
    (void)first->release();
    (void)third->release();
    {
        ninfer::HostContextArena::PlacementScope placement(arena);
        expect(!arena.allocate(2 * unit), "two separated free units cannot hold two units");
        auto a = arena.allocate(unit);
        auto b = arena.allocate(unit);
        expect(a && b, "the placement fits once split into single units");
    }
    expect(arena.geometry_failures() == 0 && arena.shortage_failures() == 0,
           "a placement made in pieces counts no failure for the sizes it tried");
    {
        ninfer::HostContextArena::PlacementScope placement(arena);
        expect(!arena.allocate(2 * unit) && !arena.allocate(unit), "a full arena holds nothing");
        placement.failed(2 * unit);
    }
    expect(arena.shortage_failures() == 1 && arena.geometry_failures() == 0,
           "a placement that cannot be made counts one failure, by its cause");
    expect(!arena.allocate(unit) && arena.shortage_failures() == 2,
           "outside a placement every miss counts again");
}

// --host-pause-reserve-mib: a byte budget only pause snapshots may go past (#332).
void test_snapshot_reserve() {
    constexpr std::size_t unit = ninfer::HostContextArena::alignment;
    ninfer::HostContextArena arena(4 * unit, unit);
    arena.reserve_for_snapshots(unit + 1); // rounds up to two units
    expect(arena.snapshot_reserve_bytes() == 2 * unit && arena.general_capacity_bytes() == 2 * unit,
           "the reserve is whole alignment units taken from the general budget");
    auto a = arena.allocate(unit);
    auto b = arena.allocate(unit);
    expect(a && b && !arena.allocate(unit) && !arena.can_allocate(unit),
           "cached history cannot allocate past its budget into the reserve");
    expect(arena.shortage_failures() == 1 && arena.geometry_failures() == 0,
           "a full general budget is a shortage, whatever the reserve holds");
    std::optional<ninfer::HostContextAllocation> kept;
    {
        ninfer::HostContextArena::SnapshotScope snapshot(arena);
        expect(arena.can_allocate(2 * unit) && arena.placeable_bytes(unit) == 2 * unit,
               "a snapshot sees the reserve");
        kept = arena.allocate(unit);
        expect(kept && arena.occupied_bytes() == 3 * unit &&
                   arena.general_occupied_bytes() == 2 * unit,
               "a snapshot past the budget is counted in the reserve, not against the budget");
    }
    (void)a->release();
    expect(arena.occupied_bytes() == 2 * unit && arena.general_occupied_bytes() == 2 * unit &&
               !arena.allocate(unit),
           "a snapshot's copy left behind counts against the budget like cached history");
    (void)kept->release();
    expect(arena.allocate(unit).has_value(),
           "evicting the left-behind copy returns the room to the budget");
    expect(arena.placeable_bytes(3 * unit) == 0 && arena.placeable_bytes(2 * unit) == 2 * unit,
           "a free run holds only whole units");
    bool twice = false;
    try {
        arena.reserve_for_snapshots(unit);
    } catch (const std::logic_error&) { twice = true; }
    expect(twice, "the reserve is set once, before any allocation");
    bool whole = false;
    try {
        ninfer::HostContextArena other(4 * unit, unit);
        other.reserve_for_snapshots(4 * unit);
    } catch (const std::invalid_argument&) { whole = true; }
    expect(whole, "a reserve cannot take the whole backing");
}

} // namespace

int main() {
    int count               = 0;
    const cudaError_t error = cudaGetDeviceCount(&count);
    if (error == cudaErrorNoDevice || error == cudaErrorInsufficientDriver ||
        (error == cudaSuccess && count == 0)) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    try {
        test_reserve_publish_and_rollback();
        test_fragmentation_and_split();
        test_failures_are_counted_by_cause();
        test_placement_scope_counts_once();
        test_snapshot_reserve();
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return failures == 0 ? 0 : 1;
}
