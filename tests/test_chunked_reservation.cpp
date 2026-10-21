#include "core/chunked_reservation.h"

#include <cstddef>
#include <iostream>
#include <numeric>
#include <optional>
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

// First-fit over free runs counted in pages, like HostContextArena: the free pages can add up to
// the request while no single run holds it.
struct FakeArena {
    std::vector<std::size_t> runs;
    std::size_t calls = 0;

    std::optional<int> reserve(std::size_t pages) {
        ++calls;
        for (auto& run : runs) {
            if (run >= pages) {
                run -= pages;
                return 1;
            }
        }
        return std::nullopt;
    }
};

struct Result {
    bool ok = false;
    std::vector<std::vector<int>> chunks;
};

Result run(FakeArena& arena, std::size_t count) {
    std::vector<int> items(count);
    std::iota(items.begin(), items.end(), 0);
    Result result;
    result.ok = ninfer::reserve_in_chunks<int>(
        items, [&](std::span<const int> chunk) { return arena.reserve(chunk.size()); },
        [&](int&&, std::span<const int> chunk) {
            result.chunks.emplace_back(chunk.begin(), chunk.end());
        });
    return result;
}

std::size_t placed(const Result& result) {
    std::size_t total = 0;
    for (const auto& chunk : result.chunks) { total += chunk.size(); }
    return total;
}

void test_one_run_is_enough() {
    FakeArena arena{{16}};
    const auto result = run(arena, 8);
    expect(result.ok && result.chunks.size() == 1 && result.chunks[0].size() == 8,
           "a single big enough run takes everything as one reservation");
    expect(arena.calls == 1, "no retries when the first attempt fits");
}

void test_fragmented_arena_splits() {
    // 9 pages free in total, but the largest run is 4: one contiguous request for 8 must fail.
    FakeArena single{{3, 2, 4}};
    expect(!single.reserve(8), "precondition: no run holds all 8 pages");
    FakeArena arena{{3, 2, 4}};
    const auto result = run(arena, 8);
    expect(result.ok, "fragmented arena still places every page");
    expect(placed(result) == 8 && result.chunks.size() > 1, "pages are split across runs");
    std::vector<int> flattened;
    for (const auto& chunk : result.chunks) {
        flattened.insert(flattened.end(), chunk.begin(), chunk.end());
    }
    std::vector<int> expected(8);
    std::iota(expected.begin(), expected.end(), 0);
    expect(flattened == expected, "chunks are consecutive and keep the page order");
}

void test_not_enough_total_space() {
    FakeArena arena{{3, 2}};
    const auto result = run(arena, 8);
    expect(!result.ok, "a genuine shortage still fails");
    expect(placed(result) < 8, "the caller sees which chunks were already taken, to unwind them");
}

void test_empty_request() {
    FakeArena arena{{}};
    const auto result = run(arena, 0);
    expect(result.ok && result.chunks.empty() && arena.calls == 0,
           "nothing to place needs no reservation");
}

void test_single_page_run() {
    FakeArena arena{{1, 1, 1}};
    const auto result = run(arena, 3);
    expect(result.ok && placed(result) == 3, "one-page runs each take one page");
}
} // namespace

int main() {
    test_one_run_is_enough();
    test_fragmented_arena_splits();
    test_not_enough_total_space();
    test_empty_request();
    test_single_page_run();
    if (failures) { return 1; }
    std::cout << "ok\n";
    return 0;
}
