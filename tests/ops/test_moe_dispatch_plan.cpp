// offloaded_moe::plan_dispatch, host only: the exact dispatch of k*T routed ids that the one-launch
// route and dispatch of decode-width calls (ops::moe_route_dispatch, T <= 8) writes on the device,
// and the oracle the GPU test checks those arrays against.
//   - plan_counts_offsets: per-expert counts and their exclusive scan, offsets[E] = the total;
//   - plan_jobs_ascending: the experts with entries, in ascending id whatever the ids' order;
//   - plan_entries_ascending: entries grouped by expert, each group in ascending entry index;
//   - plan_drops_out_of_range: an id outside [0, E) is not dispatched and not counted;
//   - plan_t8_k10: a decode call's 80 entries (T = 8, k = 10, E = 512), experts shared by columns.

#include "ops/offloaded_sparse_moe/dispatch_plan.h"

#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

namespace om = ninfer::ops::offloaded_moe;
using ids_t  = std::vector<std::int32_t>;

int failures = 0;

void check(bool ok, const std::string& what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what.c_str());
        ++failures;
    }
}

void plan_counts_offsets() {
    const ids_t ids = {3, 1, 3, 0};
    const auto plan = om::plan_dispatch(ids, 5);
    check(plan.counts == ids_t{1, 1, 0, 2, 0}, "plan_counts_offsets: counts");
    check(plan.offsets == ids_t{0, 1, 2, 2, 4, 4}, "plan_counts_offsets: offsets (exclusive scan, offsets[E] = 4)");
}

void plan_jobs_ascending() {
    const ids_t ids = {9, 2, 7, 2, 0};
    const auto plan = om::plan_dispatch(ids, 10);
    check(plan.jobs == ids_t{0, 2, 7, 9}, "plan_jobs_ascending: jobs are the used experts in ascending id");
    check(plan.counts[2] == 2 && plan.counts[1] == 0, "plan_jobs_ascending: counts");
}

void plan_entries_ascending() {
    const ids_t ids = {4, 1, 4, 1, 4};
    const auto plan = om::plan_dispatch(ids, 6);
    check(plan.entries == ids_t{1, 3, 0, 2, 4},
          "plan_entries_ascending: expert 1's entries, then expert 4's, each ascending");
    check(plan.jobs == ids_t{1, 4}, "plan_entries_ascending: jobs");
}

void plan_drops_out_of_range() {
    const ids_t ids = {-1, 2, 5, 0x7FFFFFFF, 2, 4};
    const auto plan = om::plan_dispatch(ids, 5);
    check(plan.counts == ids_t{0, 0, 2, 0, 1}, "plan_drops_out_of_range: only ids in [0, E) are counted");
    check(plan.offsets == ids_t{0, 0, 0, 2, 2, 3}, "plan_drops_out_of_range: offsets[E] counts the dispatched entries");
    check(plan.entries == ids_t{1, 4, 5}, "plan_drops_out_of_range: the dropped entries are not dispatched");
    check(plan.jobs == ids_t{2, 4}, "plan_drops_out_of_range: jobs");
    bool refused = false;
    try {
        (void)om::plan_dispatch(ids, 0);
    } catch (const std::invalid_argument&) { refused = true; }
    check(refused, "plan_dispatch accepted zero experts");
}

void plan_t8_k10() {
    constexpr int kTopK = 10, kColumns = 8, kExperts = 512;
    // Column t's slots: expert 7 first in every column, then distinct experts per column that repeat
    // across columns (slot s of column t is 64 + (s + 3 * t) % 40), and expert 500 in the even
    // columns' last slot.
    ids_t ids(static_cast<std::size_t>(kTopK) * kColumns);
    for (int t = 0; t < kColumns; ++t) {
        for (int s = 0; s < kTopK; ++s) {
            int e = s == 0 ? 7 : 64 + (s + 3 * t) % 40;
            if (s == kTopK - 1 && t % 2 == 0) { e = 500; }
            ids[static_cast<std::size_t>(t) * kTopK + s] = e;
        }
    }
    const auto plan = om::plan_dispatch(ids, kExperts);
    check(plan.counts.size() == kExperts && plan.offsets.size() == kExperts + 1, "plan_t8_k10: array sizes");
    check(plan.offsets[kExperts] == kTopK * kColumns && plan.entries.size() == kTopK * kColumns,
          "plan_t8_k10: all 80 entries dispatched");
    check(plan.counts[7] == kColumns && plan.counts[500] == kColumns / 2, "plan_t8_k10: shared experts' counts");
    const auto group = [&](int e) {
        return ids_t(plan.entries.begin() + plan.offsets[e], plan.entries.begin() + plan.offsets[e + 1]);
    };
    check(group(7) == ids_t{0, 10, 20, 30, 40, 50, 60, 70},
          "plan_t8_k10: expert 7 holds slot 0 of every column, ascending");
    check(group(500) == ids_t{9, 29, 49, 69}, "plan_t8_k10: expert 500 holds the even columns' last slot, ascending");
    // Every group against a brute-force filter of the ids in entry order; jobs ascending and exactly
    // the experts with entries.
    ids_t jobs;
    for (int e = 0; e < kExperts; ++e) {
        ids_t want;
        for (int i = 0; i < kTopK * kColumns; ++i) {
            if (ids[static_cast<std::size_t>(i)] == e) { want.push_back(i); }
        }
        check(static_cast<int>(want.size()) == plan.counts[e], "plan_t8_k10: count of expert " + std::to_string(e));
        check(group(e) == want, "plan_t8_k10: entries of expert " + std::to_string(e));
        if (!want.empty()) { jobs.push_back(e); }
    }
    check(plan.jobs == jobs, "plan_t8_k10: jobs");
    std::printf("plan_t8_k10: %zu jobs over %d entries\n", plan.jobs.size(), kTopK * kColumns);
}

} // namespace

int main() {
    plan_counts_offsets();
    plan_jobs_ascending();
    plan_entries_ascending();
    plan_drops_out_of_range();
    plan_t8_k10();
    if (failures != 0) {
        std::fprintf(stderr, "%d check(s) failed\n", failures);
        return 1;
    }
    std::printf("moe dispatch plan checks passed\n");
    return 0;
}
