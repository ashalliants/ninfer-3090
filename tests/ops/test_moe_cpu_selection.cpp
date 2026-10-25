// offloaded_moe::select_cpu_jobs, host only: which of a moe_experts call's non-resident experts the
// CPU miss service computes (the rule cpu_plan_kernel applies on the device), and the oracle the GPU
// test checks the published MissRequest against.
//   - fewest_columns_first: the narrowest jobs are taken first;
//   - ties_in_job_order: jobs of one width keep their job order (ascending expert id), resident
//     experts are skipped;
//   - width_cap_excludes: jobs wider than max_job_columns (or than 8) are never taken, but count as
//     misses for the divisor;
//   - divisor_zero_takes_all: divisor 0 takes every eligible miss up to the cap;
//   - divisor_three: M / 3 misses stay on the GPU (M - M / 3 go to the CPU);
//   - cap_limits: never more than the cap; inconsistent inputs are refused.
//
// And the prefill CPU share (cpu_share.h: cpu_share_take, select_prefill_cpu_jobs), the rule a call
// whose channel sets share_num / share_den applies instead of the divisor:
//   - share_matches_llround: at 3 / 5 the take is llround(0.6 * n) for every n <= 1024 (0.6 * n
//     never ends in .5), and at 1 / 2 halves round up;
//   - prefill_rows_cap: experts with more than max_rows rows are never taken but count toward n;
//   - prefill_sorted_by_rows_then_id: the fewest rows first, then the lower expert id, whatever the
//     job order;
//   - share_zero_takes_none: num 0 takes nothing; bad fractions are refused.

#include "ops/offloaded_sparse_moe/cpu_selection.h"
#include "ops/offloaded_sparse_moe/cpu_share.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

namespace om = ninfer::ops::offloaded_moe;
using ints   = std::vector<std::int32_t>;
using picks  = std::vector<int>;

int failures = 0;

void check(bool ok, const std::string& what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what.c_str());
        ++failures;
    }
}

// resident[e] for experts [0, n): those listed are resident.
std::vector<std::uint8_t> residency(int n, const std::vector<int>& resident) {
    std::vector<std::uint8_t> out(static_cast<std::size_t>(n), 0);
    for (const int e : resident) { out[static_cast<std::size_t>(e)] = 1; }
    return out;
}

void fewest_columns_first() {
    const ints experts = {1, 2, 3, 4}, columns = {3, 1, 2, 1};
    const auto none    = residency(8, {});
    check(om::select_cpu_jobs(experts, columns, none, 4, 0, 8) == picks{1, 3, 2, 0},
          "fewest_columns_first: one-column jobs, then two, then three");
    check(om::select_cpu_jobs(experts, columns, none, 3, 0, 8) == picks{1, 3, 2},
          "fewest_columns_first: the cap drops the widest");
}

void ties_in_job_order() {
    const ints experts = {5, 9, 11, 20}, columns = {2, 2, 2, 2};
    check(om::select_cpu_jobs(experts, columns, residency(32, {9}), 3, 0, 8) == picks{0, 2, 3},
          "ties_in_job_order: one width in job order, the resident expert skipped");
    check(om::select_cpu_jobs(experts, columns, residency(32, {}), 2, 0, 8) == picks{0, 1},
          "ties_in_job_order: the cap keeps the first jobs");
}

void width_cap_excludes() {
    // Widths 1, 4, 8, 2 and a wide job of 9 columns (the wide route): with max_job_columns 2 only the
    // 1- and 2-column jobs may go; every miss still counts toward the divisor.
    const ints experts = {0, 1, 2, 3, 4}, columns = {1, 4, 8, 2, 9};
    const auto none    = residency(5, {});
    check(om::select_cpu_jobs(experts, columns, none, 64, 0, 2) == picks{0, 3},
          "width_cap_excludes: only jobs of at most max_job_columns");
    check(om::select_cpu_jobs(experts, columns, none, 64, 0, 8) == picks{0, 3, 1, 2},
          "width_cap_excludes: a 9-column job is never taken");
    // M = 5 misses, 5 - 5 / 3 = 4 wanted, only two eligible.
    check(om::select_cpu_jobs(experts, columns, none, 64, 3, 2) == picks{0, 3},
          "width_cap_excludes: ineligible misses count toward the divisor");
    // Four 1-column jobs and the wide one: M = 5, so 5 - 5 / 3 = 4 go to the CPU (M = 4 would give 3).
    const ints narrow = {1, 1, 1, 1, 9};
    check(om::select_cpu_jobs(experts, narrow, none, 64, 3, 8) == picks{0, 1, 2, 3},
          "width_cap_excludes: want 5 - 5 / 3 = 4 of the four eligible");
}

void divisor_zero_takes_all() {
    const ints experts = {2, 4, 6, 8, 10}, columns = {1, 2, 1, 3, 1};
    check(om::select_cpu_jobs(experts, columns, residency(16, {}), 64, 0, 8) == picks{0, 2, 4, 1, 3},
          "divisor_zero_takes_all: every miss");
    check(om::select_cpu_jobs(experts, columns, residency(16, {4, 8}), 64, 0, 8) == picks{0, 2, 4},
          "divisor_zero_takes_all: every non-resident job");
}

void divisor_three() {
    for (const auto& [misses, want] : std::vector<std::pair<int, int>>{{1, 1}, {2, 2}, {3, 2}, {4, 3}, {7, 5}}) {
        ints experts, columns;
        for (int j = 0; j < misses; ++j) {
            experts.push_back(j);
            columns.push_back(1);
        }
        const auto got = om::select_cpu_jobs(experts, columns, residency(misses, {}), 64, 3, 8);
        check(static_cast<int>(got.size()) == want,
              "divisor_three: " + std::to_string(misses) + " misses send " + std::to_string(want) + " to the CPU");
    }
    // Resident jobs are not misses: 6 jobs, 3 resident, M = 3, want 2.
    const ints experts = {0, 1, 2, 3, 4, 5}, columns = {1, 1, 1, 1, 1, 1};
    check(om::select_cpu_jobs(experts, columns, residency(6, {0, 2, 4}), 64, 3, 8) == picks{1, 3},
          "divisor_three: resident jobs do not count");
}

void cap_limits() {
    const ints experts = {0, 1, 2, 3, 4, 5}, columns = {4, 3, 2, 1, 1, 2};
    const auto none    = residency(6, {});
    check(om::select_cpu_jobs(experts, columns, none, 2, 0, 8) == picks{3, 4}, "cap_limits: two of six");
    check(om::select_cpu_jobs(experts, columns, none, 0, 0, 8).empty(), "cap_limits: cap 0 takes none");
    check(om::select_cpu_jobs(experts, columns, none, 4, 3, 8) == picks{3, 4, 2, 5},
          "cap_limits: the divisor's 4 and the cap's 4 agree");
    check(om::select_cpu_jobs(experts, columns, none, 3, 3, 8) == picks{3, 4, 2}, "cap_limits: the cap wins");
    const auto refused = [](auto&& f) {
        try {
            f();
        } catch (const std::invalid_argument&) { return true; }
        return false;
    };
    check(refused([&] { (void)om::select_cpu_jobs(experts, ints{1}, none, 4, 3, 8); }),
          "cap_limits: mismatched spans are refused");
    check(refused([&] { (void)om::select_cpu_jobs(ints{9}, ints{1}, none, 4, 3, 8); }),
          "cap_limits: an expert outside resident is refused");
    check(refused([&] { (void)om::select_cpu_jobs(experts, columns, none, 4, 3, 9); }),
          "cap_limits: max_job_columns above 8 is refused");
    check(refused([&] { (void)om::select_cpu_jobs(experts, columns, none, -1, 3, 8); }),
          "cap_limits: a negative cap is refused");
}

void share_matches_llround() {
    for (int n = 0; n <= 1024; ++n) {
        const auto want = static_cast<int>(std::llround(0.6 * n));
        if (om::cpu_share_take(n, n, 3, 5) != want) {
            check(false, "share_matches_llround: n = " + std::to_string(n) + " takes " +
                             std::to_string(om::cpu_share_take(n, n, 3, 5)) + ", llround gives " +
                             std::to_string(want));
        }
    }
    // The candidates bound the take; a half rounds up (n = 5 at 1 / 2 takes 3).
    check(om::cpu_share_take(1024, 100, 3, 5) == 100, "share_matches_llround: the candidates bound the take");
    check(om::cpu_share_take(5, 5, 1, 2) == 3, "share_matches_llround: 2.5 rounds up");
    check(om::cpu_share_take(7, 7, 1, 1) == 7, "share_matches_llround: share 1 takes every candidate");
}

void prefill_rows_cap() {
    // Rows 1, 9, 8, 12, 2: experts 1 and 3 are too wide, n = 5 misses, take llround(3) = 3 of the
    // three candidates.
    const ints experts = {0, 1, 2, 3, 4}, rows = {1, 9, 8, 12, 2};
    const auto none    = residency(5, {});
    check(om::select_prefill_cpu_jobs(experts, rows, none, 3, 5, 8) == picks{0, 4, 2},
          "prefill_rows_cap: only experts of at most 8 rows, fewest first");
    check(om::select_prefill_cpu_jobs(experts, rows, none, 3, 5, 2) == picks{0, 4},
          "prefill_rows_cap: max_rows 2 leaves the 8-row expert");
    // n counts the wide misses: 10 misses (5 narrow, 5 wide) at 3 / 5 take 6, bounded to the 5 narrow.
    const ints many = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}, many_rows = {1, 20, 1, 20, 1, 20, 1, 20, 1, 20};
    check(om::select_prefill_cpu_jobs(many, many_rows, residency(10, {}), 3, 5, 8) == picks{0, 2, 4, 6, 8},
          "prefill_rows_cap: wide misses count toward n");
    // 10 narrow misses take 6; 2 of them resident: n = 8 takes 5.
    const ints ones(10, 1);
    check(om::select_prefill_cpu_jobs(many, ones, residency(10, {}), 3, 5, 8).size() == 6,
          "prefill_rows_cap: 10 misses take 6");
    check(om::select_prefill_cpu_jobs(many, ones, residency(10, {0, 5}), 3, 5, 8) == picks{1, 2, 3, 4, 6},
          "prefill_rows_cap: resident experts are not misses");
    check(om::select_prefill_cpu_jobs(many, ones, residency(10, {}), 3, 5, 8, 4).size() == 4,
          "prefill_rows_cap: the channel's cap bounds the take");
}

void prefill_sorted_by_rows_then_id() {
    // Jobs listed out of id order: the result is by (rows, expert id), not job order.
    const ints experts = {7, 3, 5, 1, 6}, rows = {2, 2, 1, 2, 1};
    check(om::select_prefill_cpu_jobs(experts, rows, residency(8, {}), 1, 1, 8) == picks{2, 4, 3, 1, 0},
          "prefill_sorted_by_rows_then_id: rows 1 (ids 5, 6), then rows 2 (ids 1, 3, 7)");
    // In dispatch order (ascending ids) the rule orders like the device's: fewest rows, then job order.
    const ints sorted = {1, 3, 5, 6, 7}, sorted_rows = {2, 2, 1, 1, 2};
    check(om::select_prefill_cpu_jobs(sorted, sorted_rows, residency(8, {}), 1, 1, 8) ==
              om::select_cpu_jobs(sorted, sorted_rows, residency(8, {}), 64, 0, 8),
          "prefill_sorted_by_rows_then_id: on ascending ids it orders like select_cpu_jobs");
    check(om::select_prefill_cpu_jobs(experts, rows, residency(8, {}), 3, 5, 8) == picks{2, 4, 3},
          "prefill_sorted_by_rows_then_id: 3 / 5 of 5 takes the first 3");
}

void share_zero_takes_none() {
    const ints experts = {0, 1, 2}, rows = {1, 1, 1};
    const auto none    = residency(3, {});
    check(om::select_prefill_cpu_jobs(experts, rows, none, 0, 5, 8).empty(), "share_zero_takes_none: 0 / 5");
    check(om::cpu_share_take(1024, 1024, 0, 1) == 0, "share_zero_takes_none: 0 / 1 of 1024");
    check(om::cpu_share_take(0, 0, 3, 5) == 0, "share_zero_takes_none: no misses");
    const auto refused = [](auto&& f) {
        try {
            f();
        } catch (const std::invalid_argument&) { return true; }
        return false;
    };
    check(refused([&] { (void)om::cpu_share_take(4, 4, 1, 0); }), "share_zero_takes_none: den 0 is refused");
    check(refused([&] { (void)om::cpu_share_take(4, 4, 6, 5); }), "share_zero_takes_none: a share above 1 is refused");
    check(refused([&] { (void)om::cpu_share_take(4, 4, -1, 5); }), "share_zero_takes_none: a negative num is refused");
    check(refused([&] { (void)om::cpu_share_take(4, 5, 3, 5); }),
          "share_zero_takes_none: more candidates than misses is refused");
    check(refused([&] { (void)om::select_prefill_cpu_jobs(experts, ints{1}, none, 3, 5, 8); }),
          "share_zero_takes_none: mismatched spans are refused");
    check(refused([&] { (void)om::select_prefill_cpu_jobs(experts, rows, none, 3, 5, 9); }),
          "share_zero_takes_none: max_rows above 8 is refused");
    check(refused([&] { (void)om::select_prefill_cpu_jobs(experts, rows, none, 3, 5, 8, -1); }),
          "share_zero_takes_none: a negative cap is refused");
}

} // namespace

int main() {
    fewest_columns_first();
    ties_in_job_order();
    width_cap_excludes();
    divisor_zero_takes_all();
    divisor_three();
    cap_limits();
    share_matches_llround();
    prefill_rows_cap();
    prefill_sorted_by_rows_then_id();
    share_zero_takes_none();
    if (failures != 0) {
        std::fprintf(stderr, "%d check(s) failed\n", failures);
        return 1;
    }
    std::printf("moe cpu selection checks passed\n");
    return 0;
}
