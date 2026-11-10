#pragma once

#include <algorithm>
#include <cstdint>
#include <span>
#include <utility>
#include <vector>

namespace ninfer::runtime {

// Prompt tokens to recompute when the `removed` checkpoints are released and a later request
// restarts from the deepest `surviving` checkpoint that is a prefix of each one. Removed
// checkpoints on one lineage (each a prefix of the deepest) count that lineage's largest loss
// once; independent lineages add.
//
// `valid(h)` says a handle is live, `frontier(h)` gives its depth in tokens, and
// `prefix(earlier, later)` says `earlier` is a prefix of `later`. `prefix` must be false
// whenever `earlier` is deeper than `later`.
//
// The reclaim planner calls this once per candidate action, against every checkpoint the cache
// retains. Survivors are therefore filtered once and searched deepest first, stopping at the
// first prefix: a full Host tier holds thousands of checkpoints, and a scan of every survivor
// per removed checkpoint (with a linear membership test inside it) stalled cold prefills for
// minutes (issue #402).
template <class Handle, class Valid, class Frontier, class Prefix>
[[nodiscard]] std::uint64_t
checkpoint_recovery_loss(std::span<const Handle> removed, std::span<const Handle> surviving,
                         Valid&& valid, Frontier&& frontier, Prefix&& prefix) {
    const auto by_index = [](const Handle& left, const Handle& right) {
        return left.index < right.index;
    };
    // Valid removed handles, unique, sorted by slot index for membership tests. Handles with
    // one index may still differ (owner, generation), so equality is checked within a run.
    std::vector<Handle> members;
    members.reserve(removed.size());
    for (const auto handle : removed) {
        if (valid(handle)) { members.push_back(handle); }
    }
    std::stable_sort(members.begin(), members.end(), by_index);
    {
        std::vector<Handle> unique;
        unique.reserve(members.size());
        for (auto run = members.begin(); run != members.end();) {
            const auto end   = std::upper_bound(run, members.end(), *run, by_index);
            const auto first = unique.size();
            for (auto it = run; it != end; ++it) {
                if (std::find(unique.begin() + static_cast<std::ptrdiff_t>(first), unique.end(),
                              *it) == unique.end()) {
                    unique.push_back(*it);
                }
            }
            run = end;
        }
        members = std::move(unique);
    }
    const auto is_member = [&](const Handle& handle) {
        const auto [first, last] =
            std::equal_range(members.begin(), members.end(), handle, by_index);
        return std::find(first, last, handle) != last;
    };

    struct Point {
        Handle handle;
        std::uint32_t frontier;
    };

    const auto deeper_first = [](const Point& left, const Point& right) {
        return left.frontier != right.frontier ? left.frontier > right.frontier
                                               : left.handle.index < right.handle.index;
    };

    std::vector<Point> ordered;
    ordered.reserve(members.size());
    for (const auto handle : members) {
        ordered.push_back({handle, static_cast<std::uint32_t>(frontier(handle))});
    }
    std::sort(ordered.begin(), ordered.end(), deeper_first);

    // A survivor at frontier 0 can never be a fallback (no tokens are kept), so it is skipped.
    std::vector<Point> fallbacks;
    fallbacks.reserve(surviving.size());
    for (const auto handle : surviving) {
        if (!valid(handle) || is_member(handle)) { continue; }
        const auto depth = static_cast<std::uint32_t>(frontier(handle));
        if (depth > 0) { fallbacks.push_back({handle, depth}); }
    }
    std::sort(fallbacks.begin(), fallbacks.end(), deeper_first);

    struct LineageLoss {
        Handle deepest;
        std::uint32_t tokens;
    };

    std::vector<LineageLoss> lineages;
    for (const auto& point : ordered) {
        // A survivor deeper than this checkpoint cannot be its prefix; the first prefix at or
        // below it is the deepest fallback.
        auto candidate =
            std::partition_point(fallbacks.begin(), fallbacks.end(),
                                 [&](const Point& p) { return p.frontier > point.frontier; });
        std::uint32_t fallback = 0;
        for (; candidate != fallbacks.end(); ++candidate) {
            if (prefix(candidate->handle, point.handle)) {
                fallback = candidate->frontier;
                break;
            }
        }
        const auto lost    = point.frontier - fallback;
        const auto lineage = std::find_if(lineages.begin(), lineages.end(), [&](const auto& item) {
            return prefix(point.handle, item.deepest);
        });
        if (lineage == lineages.end()) {
            lineages.push_back({point.handle, lost});
        } else {
            lineage->tokens = std::max(lineage->tokens, lost);
        }
    }
    std::uint64_t result = 0;
    for (const auto& lineage : lineages) { result += lineage.tokens; }
    return result;
}

} // namespace ninfer::runtime
