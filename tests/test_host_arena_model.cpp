#include "core/host_arena_model.h"

#include <cstdint>
#include <iostream>
#include <optional>
#include <random>
#include <span>
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

using ninfer::HostArenaModel;
using ninfer::HostContextAllocation;
using ninfer::HostContextArena;
using ninfer::PauseHostNeed;

constexpr std::size_t kUnit = HostContextArena::alignment;

// What a pause does to the real arena, in the same order and with the same chunking, keeping what
// it reserves until the end. The model must agree with this on every layout.
bool place_on_arena(HostContextArena& arena, const PauseHostNeed& need) {
    std::vector<HostContextAllocation> kept;
    if (need.image_bytes) {
        auto image = arena.allocate(need.image_bytes);
        if (!image) { return false; }
        kept.push_back(std::move(*image));
    }
    for (const auto& pool : {need.text, need.backend}) {
        if (pool.pages == 0) { continue; }
        const std::vector<std::uint32_t> pages(pool.pages);
        const bool placed = ninfer::reserve_in_chunks<std::uint32_t>(
            std::span<const std::uint32_t>(pages),
            [&](std::span<const std::uint32_t> chunk) {
                return arena.allocate(chunk.size() * pool.stride);
            },
            [&](HostContextAllocation&& held, std::span<const std::uint32_t>) {
                kept.push_back(std::move(held));
            });
        if (!placed) { return false; }
    }
    return true;
}

// An arena of `units` allocation units whose free runs are exactly the unit ranges `free_units`
// (each a begin, length pair); every other unit is held by the returned allocations.
struct Layout {
    HostContextArena arena;
    std::vector<HostContextAllocation> held;

    Layout(std::size_t units, const std::vector<std::pair<std::size_t, std::size_t>>& free_units)
        : arena(units * kUnit, kUnit) {
        std::vector<std::optional<HostContextAllocation>> all;
        for (std::size_t i = 0; i < units; ++i) { all.push_back(arena.allocate(kUnit)); }
        std::vector<bool> release(units, false);
        for (const auto& [begin, length] : free_units) {
            for (std::size_t i = begin; i < begin + length; ++i) { release[i] = true; }
        }
        for (std::size_t i = 0; i < units; ++i) {
            if (!release[i]) { held.push_back(std::move(*all[i])); }
        }
        // The released ones are dropped with `all`.
    }
};

// The check this replaces: what the free runs take in units of the larger page stride bounds
// the pages (the image counted in too), so a run holding pages of the smaller pool only was lost.
bool old_bound_fits(const HostContextArena& arena, const PauseHostNeed& need) {
    const std::size_t bytes = need.image_bytes + need.text.pages * need.text.stride +
                              need.backend.pages * need.backend.stride;
    return bytes <= arena.placeable_bytes(std::max(need.text.stride, need.backend.stride));
}

void test_mixed_strides_fit_where_the_larger_stride_bound_refuses() {
    // Free runs of 4 and 2 units. One 3-unit text page and three 1-unit backend pages: the text
    // page takes the first run and the backend pages the remaining 1 + 2 units.
    Layout layout(16, {{0, 4}, {8, 2}});
    const PauseHostNeed need{.image_bytes = 0,
                             .text        = {.pages = 1, .stride = 3 * kUnit},
                             .backend     = {.pages = 3, .stride = kUnit}};
    expect(!old_bound_fits(layout.arena, need), "fixture: the larger-stride bound must refuse");
    expect(ninfer::pause_snapshot_places(HostArenaModel(layout.arena), need),
           "the model refused a snapshot that fits in pieces");
    expect(place_on_arena(layout.arena, need), "the arena refused what the model placed");
}

void test_state_image_and_pages_contend_for_the_same_run() {
    // Free runs of 3 and 3 units. The image takes a whole run, so the 4 one-unit pages left fit
    // only if they are split over what remains (3 units): they do not.
    Layout layout(16, {{0, 3}, {8, 3}});
    const PauseHostNeed too_big{.image_bytes = 3 * kUnit,
                                .text        = {.pages = 4, .stride = kUnit},
                                .backend     = {}};
    expect(!ninfer::pause_snapshot_places(HostArenaModel(layout.arena), too_big),
           "the model ignored the image taking a run");
    expect(!place_on_arena(layout.arena, too_big), "fixture: the arena must refuse too");
    const PauseHostNeed fits{.image_bytes = 3 * kUnit,
                             .text        = {.pages = 3, .stride = kUnit},
                             .backend     = {}};
    expect(ninfer::pause_snapshot_places(HostArenaModel(layout.arena), fits) &&
               place_on_arena(layout.arena, fits),
           "an image and pages that fill the free runs exactly were refused");
}

void test_a_model_is_a_copy_and_leaves_the_arena_alone() {
    Layout layout(16, {{0, 4}, {8, 2}});
    const std::size_t free_before = layout.arena.free_bytes();
    const std::size_t runs_before = layout.arena.free_extent_count();
    const PauseHostNeed need{.image_bytes = 2 * kUnit,
                             .text        = {.pages = 2, .stride = kUnit},
                             .backend     = {}};
    expect(ninfer::pause_snapshot_places(HostArenaModel(layout.arena), need), "fixture must fit");
    expect(layout.arena.free_bytes() == free_before &&
               layout.arena.free_extent_count() == runs_before,
           "asking whether a snapshot fits changed the arena");
}

void test_sizes_the_arena_refuses() {
    HostArenaModel model({{0, 4 * kUnit}}, 2 * kUnit);
    expect(model.allocation_bytes(0) == 0 && model.allocation_bytes(kUnit) == 0 &&
               model.allocation_bytes(kUnit + 1) == 2 * kUnit,
           "allocation sizes do not follow the arena's rounding and minimum");
    expect(!model.allocate(5 * kUnit), "an allocation larger than every run succeeded");
}

// Random layouts, random needs: the model and the arena agree on every one.
void test_model_agrees_with_the_arena() {
    std::mt19937 random(398);
    std::size_t fits = 0, refused = 0, rescued = 0;
    for (int trial = 0; trial < 4000; ++trial) {
        const std::size_t units = 8 + random() % 40;
        std::vector<std::pair<std::size_t, std::size_t>> free_units;
        for (std::size_t at = 0; at < units;) {
            const std::size_t gap = 1 + random() % 4;
            const std::size_t run = 1 + random() % 6;
            at += gap;
            if (at >= units) { break; }
            free_units.push_back({at, std::min(run, units - at)});
            at += run;
        }
        Layout layout(units, free_units);
        PauseHostNeed need;
        need.image_bytes   = random() % 3 ? 0 : (1 + random() % 4) * kUnit;
        need.text.pages    = random() % 7;
        need.text.stride   = (1 + random() % 4) * kUnit;
        need.backend.pages = random() % 3 ? 0 : random() % 7;
        need.backend.stride = (1 + random() % 4) * kUnit;
        const bool model = ninfer::pause_snapshot_places(HostArenaModel(layout.arena), need);
        const bool real  = place_on_arena(layout.arena, need);
        expect(model == real, "the model and the arena disagree on a layout");
        (model ? fits : refused)++;
        if (model && !old_bound_fits(layout.arena, need)) { ++rescued; }
        if (model != real) { return; }
    }
    expect(fits > 100 && refused > 100, "the random layouts did not exercise both outcomes");
    expect(rescued > 20, "no random layout showed a snapshot the larger-stride bound lost");
}
} // namespace

int main() {
    test_mixed_strides_fit_where_the_larger_stride_bound_refuses();
    test_state_image_and_pages_contend_for_the_same_run();
    test_a_model_is_a_copy_and_leaves_the_arena_alone();
    test_sizes_the_arena_refuses();
    test_model_agrees_with_the_arena();
    if (failures) { return 1; }
    std::cout << "ok\n";
    return 0;
}
