#include "runtime/engine/context_cache/recovery_loss.h"

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <random>
#include <span>
#include <vector>

namespace {

int check(bool condition, const char* message) {
    if (condition) { return 0; }
    std::cerr << message << '\n';
    return 1;
}

std::uint64_t comparisons = 0;

// Counts equality tests: the planner's stall (issue #402) was a membership scan run once per
// survivor per removed checkpoint, so its cost shows up here.
struct Handle {
    std::uint32_t index      = 0;
    std::uint64_t generation = 0;

    friend bool operator==(Handle left, Handle right) noexcept {
        ++comparisons;
        return left.index == right.index && left.generation == right.generation;
    }
};

// A checkpoint is a session's prompt up to `frontier`. Sessions share their first `shared`
// tokens (a common system prompt) and nothing after it.
struct Point {
    std::uint32_t session  = 0;
    std::uint32_t frontier = 0;
    bool live              = true;
};

struct Cache {
    std::vector<Point> points;
    std::uint32_t shared         = 0;
    mutable std::uint64_t probes = 0;

    Handle add(std::uint32_t session, std::uint32_t frontier, bool live = true) {
        points.push_back({session, frontier, live});
        return {static_cast<std::uint32_t>(points.size() - 1), 1};
    }

    bool valid(Handle h) const {
        return h.index < points.size() && h.generation == 1 && points[h.index].live;
    }

    std::uint32_t frontier(Handle h) const { return points.at(h.index).frontier; }

    bool prefix(Handle earlier, Handle later) const {
        ++probes;
        const auto& a = points.at(earlier.index);
        const auto& b = points.at(later.index);
        return a.frontier <= b.frontier && (a.session == b.session || a.frontier <= shared);
    }

    std::uint64_t loss(std::span<const Handle> removed, std::span<const Handle> surviving) const {
        return ninfer::runtime::checkpoint_recovery_loss(
            removed, surviving, [&](Handle h) { return valid(h); },
            [&](Handle h) { return frontier(h); },
            [&](Handle a, Handle b) { return prefix(a, b); });
    }

    // The planner's algorithm before #402, kept verbatim as the reference.
    std::uint64_t reference(std::span<const Handle> removed,
                            std::span<const Handle> surviving) const {
        std::vector<Handle> ordered;
        for (const auto handle : removed) {
            if (valid(handle) &&
                std::find(ordered.begin(), ordered.end(), handle) == ordered.end()) {
                ordered.push_back(handle);
            }
        }
        std::sort(ordered.begin(), ordered.end(), [&](auto left, auto right) {
            const auto l = frontier(left), r = frontier(right);
            return l != r ? l > r : left.index < right.index;
        });

        struct LineageLoss {
            Handle deepest;
            std::uint32_t tokens;
        };

        std::vector<LineageLoss> lineages;
        for (const auto handle : ordered) {
            std::uint32_t fallback = 0;
            for (const auto retained : surviving) {
                if (!valid(retained) ||
                    std::find(ordered.begin(), ordered.end(), retained) != ordered.end()) {
                    continue;
                }
                if (frontier(retained) > fallback && prefix(retained, handle)) {
                    fallback = frontier(retained);
                }
            }
            const auto lost = frontier(handle) - fallback;
            const auto lineage =
                std::find_if(lineages.begin(), lineages.end(),
                             [&](const auto& item) { return prefix(handle, item.deepest); });
            if (lineage == lineages.end()) {
                lineages.push_back({handle, lost});
            } else {
                lineage->tokens = std::max(lineage->tokens, lost);
            }
        }
        std::uint64_t result = 0;
        for (const auto& lineage : lineages) { result += lineage.tokens; }
        return result;
    }
};

int test_lineages() {
    int failures = 0;
    Cache cache;
    cache.shared    = 0;
    const auto r100 = cache.add(0, 100);
    const auto r400 = cache.add(0, 400);
    const auto r900 = cache.add(0, 900);
    const auto f700 = cache.add(1, 700);
    const std::vector<Handle> all{r100, r400, r900, f700};
    failures += check(cache.loss(std::span(&r900, 1), all) == 500,
                      "loss is measured to the deepest surviving prefix");
    const std::vector<Handle> pair{r400, r900};
    failures += check(cache.loss(pair, all) == 800, "one lineage counts its deepest loss once");
    const std::vector<Handle> forks{r900, f700};
    failures += check(cache.loss(forks, all) == 500 + 700, "independent lineages add");
    failures += check(cache.loss(std::span(&r100, 1), {}) == 100, "no survivor loses everything");
    const std::vector<Handle> twice{r900, r900, r400};
    failures += check(cache.loss(twice, all) == 800, "a repeated handle counts once");
    const Handle stale{r400.index, 2};
    const std::vector<Handle> kept_stale{r100, stale};
    failures += check(cache.loss(std::span(&r900, 1), kept_stale) == 800,
                      "a stale handle is not a fallback");
    failures += check(cache.loss(std::span(&stale, 1), all) == 0, "a stale handle loses nothing");
    return failures;
}

// Random forests with shared prefixes, dead slots, stale handles, repeats and overlap between
// the removed and surviving sets: the result must match the old algorithm exactly.
int test_matches_reference() {
    std::mt19937 rng(402);
    for (int seed = 0; seed < 3000; ++seed) {
        Cache cache;
        cache.shared        = std::uniform_int_distribution<std::uint32_t>(0, 40)(rng);
        const auto sessions = std::uniform_int_distribution<std::uint32_t>(1, 6)(rng);
        const auto count    = std::uniform_int_distribution<int>(1, 40)(rng);
        std::vector<Handle> handles;
        for (int i = 0; i < count; ++i) {
            const auto session = std::uniform_int_distribution<std::uint32_t>(0, sessions - 1)(rng);
            const auto frontier = std::uniform_int_distribution<std::uint32_t>(0, 120)(rng);
            handles.push_back(cache.add(session, frontier, rng() % 8 != 0));
        }
        const auto pick = [&](int n) {
            std::vector<Handle> out;
            for (int i = 0; i < n; ++i) {
                auto h = handles[rng() % handles.size()];
                if (rng() % 16 == 0) { h.generation = 2; }
                out.push_back(h);
            }
            return out;
        };
        const auto removed   = pick(std::uniform_int_distribution<int>(0, count)(rng));
        const auto surviving = pick(std::uniform_int_distribution<int>(0, 2 * count)(rng));
        const auto expected  = cache.reference(removed, surviving);
        const auto actual    = cache.loss(removed, surviving);
        if (expected != actual) {
            std::cerr << "seed " << seed << ": expected " << expected << ", got " << actual << '\n';
            return 1;
        }
    }
    return 0;
}

// A full Host tier after a long agent session: 64 sessions of 64 checkpoints each, 2048 tokens
// apart (up to 128k), all retained. One reclaim action releases the tails of 16 sessions.
int test_full_host_tier_scales() {
    int failures = 0;
    Cache cache;
    cache.shared                      = 1500;
    constexpr std::uint32_t kSessions = 64, kDepth = 64, kStep = 2048;
    std::vector<Handle> surviving, removed;
    for (std::uint32_t s = 0; s < kSessions; ++s) {
        for (std::uint32_t d = 1; d <= kDepth; ++d) {
            const auto handle = cache.add(s, d * kStep);
            surviving.push_back(handle);
            if (s < 16 && d > kDepth / 2) { removed.push_back(handle); }
        }
    }
    comparisons     = 0;
    cache.probes    = 0;
    const auto loss = cache.loss(removed, surviving);
    failures += check(loss == 16ULL * (kDepth / 2) * kStep,
                      "each released tail falls back to its session's middle checkpoint");
    // The old algorithm did about |removed|^2 * |surviving| / 2 = 1.07e9 comparisons here.
    const std::uint64_t r = removed.size(), s = surviving.size();
    const std::uint64_t work = comparisons + cache.probes;
    std::cout << "full Host tier: " << comparisons << " comparisons, " << cache.probes
              << " prefix probes (removed " << r << ", surviving " << s << ")\n";
    failures += check(work <= 4 * (r + s) * 64,
                      "recovery loss is not near-linear in the retained checkpoints");
    return failures;
}

} // namespace

int main() {
    int failures = 0;
    failures += test_lineages();
    failures += test_matches_reference();
    failures += test_full_host_tier_scales();
    if (failures == 0) { std::cout << "ok\n"; }
    return failures == 0 ? 0 : 1;
}
