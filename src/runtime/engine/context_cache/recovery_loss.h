#pragma once

#include <algorithm>
#include <cstdint>
#include <functional>
#include <type_traits>
#include <span>
#include <utility>
#include <vector>

namespace ninfer::runtime {

// The survivors' half of checkpoint_recovery_loss, prepared once. The reclaim planner prices many
// candidate releases against one retained set, each against "every retained checkpoint except the
// candidate's own". Filtering and sorting that set per candidate made each query O(N log N) and the
// victim search over a full Host tier O(N^2 log N) on the Engine thread. A table is built from the
// whole retained set; a query skips its own removed checkpoints while it scans, which gives exactly
// the answer of checkpoint_recovery_loss(removed, retained less removed).
template <class Handle, class Valid, class Frontier, class Prefix> class RecoveryLossTable {
public:
    RecoveryLossTable(std::span<const Handle> retained, Valid valid, Frontier frontier,
                      Prefix prefix)
        : valid_(std::move(valid)), frontier_(std::move(frontier)), prefix_(std::move(prefix)) {
        fallbacks_.reserve(retained.size());
        // A survivor at frontier 0 can never be a fallback (no tokens are kept), so it is skipped.
        for (const auto handle : retained) {
            if (!valid_(handle)) { continue; }
            const auto depth = static_cast<std::uint32_t>(frontier_(handle));
            if (depth > 0) { fallbacks_.push_back({handle, depth}); }
        }
        std::sort(fallbacks_.begin(), fallbacks_.end(), deeper_first);
    }

    // Prompt tokens to recompute when `removed` is released from the retained set, with the
    // `excluded` checkpoints also gone (not survivors, but not counted as lost either).
    [[nodiscard]] std::uint64_t loss(std::span<const Handle> removed,
                                     std::span<const Handle> excluded = {}) const {
        // Valid removed handles, unique, sorted by slot index for membership tests. Handles with
        // one index may still differ (owner, generation), so equality is checked within a run.
        std::vector<Handle> members;
        members.reserve(removed.size());
        for (const auto handle : removed) {
            if (valid_(handle)) { members.push_back(handle); }
        }
        std::stable_sort(members.begin(), members.end(), by_index);
        {
            std::vector<Handle> unique;
            unique.reserve(members.size());
            for (auto run = members.begin(); run != members.end();) {
                const auto end   = std::upper_bound(run, members.end(), *run, by_index);
                const auto first = unique.size();
                for (auto it = run; it != end; ++it) {
                    if (std::find(unique.begin() + static_cast<std::ptrdiff_t>(first),
                                  unique.end(), *it) == unique.end()) {
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

        std::vector<Point> ordered;
        ordered.reserve(members.size());
        for (const auto handle : members) {
            ordered.push_back({handle, static_cast<std::uint32_t>(frontier_(handle))});
        }
        std::sort(ordered.begin(), ordered.end(), deeper_first);

        struct LineageLoss {
            Handle deepest;
            std::uint32_t tokens;
        };

        std::vector<LineageLoss> lineages;
        for (const auto& point : ordered) {
            // A survivor deeper than this checkpoint cannot be its prefix; the first prefix at or
            // below it is the deepest fallback. Removed checkpoints are not survivors.
            auto candidate =
                std::partition_point(fallbacks_.begin(), fallbacks_.end(),
                                     [&](const Point& p) { return p.frontier > point.frontier; });
            std::uint32_t fallback = 0;
            for (; candidate != fallbacks_.end(); ++candidate) {
                if (is_member(candidate->handle) ||
                    std::find(excluded.begin(), excluded.end(), candidate->handle) !=
                        excluded.end()) {
                    continue;
                }
                if (prefix_(candidate->handle, point.handle)) {
                    fallback = candidate->frontier;
                    break;
                }
            }
            const auto lost    = point.frontier - fallback;
            const auto lineage =
                std::find_if(lineages.begin(), lineages.end(), [&](const auto& item) {
                    return prefix_(point.handle, item.deepest);
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

private:
    struct Point {
        Handle handle;
        std::uint32_t frontier;
    };

    static bool by_index(const Handle& left, const Handle& right) {
        return left.index < right.index;
    }

    static bool deeper_first(const Point& left, const Point& right) {
        return left.frontier != right.frontier ? left.frontier > right.frontier
                                               : left.handle.index < right.handle.index;
    }

    Valid valid_;
    Frontier frontier_;
    Prefix prefix_;
    std::vector<Point> fallbacks_;
};

// Prompt tokens to recompute when the `removed` checkpoints are released and a later request
// restarts from the deepest `surviving` checkpoint that is a prefix of each one. Removed
// checkpoints on one lineage (each a prefix of the deepest) count that lineage's largest loss
// once; independent lineages add.
//
// `valid(h)` says a handle is live, `frontier(h)` gives its depth in tokens, and
// `prefix(earlier, later)` says `earlier` is a prefix of `later`. `prefix` must be false
// whenever `earlier` is deeper than `later`.
//
// Survivors are filtered once and searched deepest first, stopping at the first prefix: a full
// Host tier holds thousands of checkpoints, and a scan of every survivor per removed checkpoint
// (with a linear membership test inside it) stalled cold prefills for minutes (issue #402). A
// caller pricing many releases against one retained set builds a RecoveryLossTable instead.
template <class Handle, class Valid, class Frontier, class Prefix>
[[nodiscard]] std::uint64_t
checkpoint_recovery_loss(std::span<const Handle> removed, std::span<const Handle> surviving,
                         Valid&& valid, Frontier&& frontier, Prefix&& prefix) {
    // Removed handles listed among the survivors are skipped by the query, as before.
    const RecoveryLossTable<Handle, std::reference_wrapper<std::remove_reference_t<Valid>>,
                            std::reference_wrapper<std::remove_reference_t<Frontier>>,
                            std::reference_wrapper<std::remove_reference_t<Prefix>>>
        table(surviving, std::ref(valid), std::ref(frontier), std::ref(prefix));
    return table.loss(removed);
}

} // namespace ninfer::runtime
