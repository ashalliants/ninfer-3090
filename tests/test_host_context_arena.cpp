#include "core/host_context_arena.h"

#include <cuda_runtime_api.h>

#include <cstring>
#include <iostream>
#include <limits>
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

// Fixed-size state images (4 units) and short KV runs (1 unit) share one arena. Filled in
// alternation and then emptied of KV, the freed bytes are one hole when KV sits at the high end
// and four unusable 1-unit holes when everything is carved from the low end.
void test_placement_keeps_state_images_and_kv_runs_apart() {
    constexpr std::size_t unit  = ninfer::HostContextArena::alignment;
    constexpr std::size_t state = 4 * unit;
    const auto fill = [&](ninfer::HostContextArena& arena, ninfer::HostPlacement kv_placement,
                          std::vector<ninfer::HostContextAllocation>& states,
                          std::vector<ninfer::HostContextAllocation>& kv) {
        for (int index = 0; index < 4; ++index) {
            auto image = arena.allocate(state, ninfer::HostPlacement::Low);
            auto run   = arena.allocate(unit, kv_placement);
            expect(image && run, "alternating state image and KV run fit while the arena fills");
            if (!image || !run) { return; }
            image->publish();
            run->publish();
            states.push_back(std::move(*image));
            kv.push_back(std::move(*run));
        }
    };

    {
        ninfer::HostContextArena arena(20 * unit, unit);
        std::vector<ninfer::HostContextAllocation> states, kv;
        fill(arena, ninfer::HostPlacement::Low, states, kv);
        kv.clear();
        expect(arena.free_bytes() == state && arena.largest_free_extent_bytes() == unit &&
                   arena.free_extent_count() == 4,
               "single-ended placement leaves the freed KV bytes as four 1-unit holes");
        expect(!arena.can_allocate(state) && !arena.allocate(state, ninfer::HostPlacement::Low),
               "a state image fails although as many bytes are free as it needs");
    }
    {
        ninfer::HostContextArena arena(20 * unit, unit);
        std::vector<ninfer::HostContextAllocation> states, kv;
        fill(arena, ninfer::HostPlacement::High, states, kv);
        bool kv_above_states = true;
        for (const auto& run : kv) {
            for (const auto& image : states) { kv_above_states &= run.data() > image.data(); }
        }
        expect(kv_above_states, "KV runs sit above every state image");
        kv.clear();
        expect(arena.free_bytes() == state && arena.largest_free_extent_bytes() == state &&
                   arena.free_extent_count() == 1,
               "two-ended placement leaves the freed KV bytes as one extent");
        auto image = arena.allocate(state, ninfer::HostPlacement::Low);
        expect(image.has_value(), "a state image fits once the KV runs are gone");
    }
}

void test_failures_are_counted_by_cause() {
    constexpr std::size_t unit = ninfer::HostContextArena::alignment;
    ninfer::HostContextArena arena(4 * unit, unit);
    expect(arena.geometry_failures() == 0 && arena.shortage_failures() == 0 &&
               arena.free_extent_count() == 1,
           "a fresh arena has one free run and no failures");
    auto first  = arena.allocate(unit, ninfer::HostPlacement::Low);
    auto second = arena.allocate(unit, ninfer::HostPlacement::Low);
    auto third  = arena.allocate(unit, ninfer::HostPlacement::Low);
    auto fourth = arena.allocate(unit, ninfer::HostPlacement::Low);
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

void test_high_placement_takes_the_back_of_the_last_fitting_extent() {
    constexpr std::size_t unit = ninfer::HostContextArena::alignment;
    ninfer::HostContextArena arena(8 * unit, unit);
    auto low = arena.allocate(2 * unit, ninfer::HostPlacement::Low);
    auto mid = arena.allocate(2 * unit, ninfer::HostPlacement::Low);
    auto top = arena.allocate(2 * unit, ninfer::HostPlacement::High);
    expect(low && mid && top && mid->data() == low->data() + 2 * unit &&
               top->data() == low->data() + 6 * unit,
           "Low packs from the front and High packs from the back of the backing");
    const std::byte* const reused = mid->data();
    (void)mid->release();
    auto small = arena.allocate(unit, ninfer::HostPlacement::High);
    // Releasing `mid` joined its hole to the one above it: units 2-5 are one free extent.
    expect(small && small->data() == reused + 3 * unit,
           "High carves the tail of the free extent, not its front");
    auto filler = arena.allocate(unit, ninfer::HostPlacement::High);
    expect(filler && filler->data() == reused + 2 * unit && arena.free_bytes() == 2 * unit &&
               arena.largest_free_extent_bytes() == 2 * unit,
           "the remaining hole is still reported as one extent");
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
        test_placement_keeps_state_images_and_kv_runs_apart();
        test_high_placement_takes_the_back_of_the_last_fitting_extent();
        test_failures_are_counted_by_cause();
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return failures == 0 ? 0 : 1;
}
