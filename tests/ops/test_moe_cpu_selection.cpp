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

#include "ops/offloaded_sparse_moe/cpu_selection.h"

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

} // namespace

int main() {
    fewest_columns_first();
    ties_in_job_order();
    width_cap_excludes();
    divisor_zero_takes_all();
    divisor_three();
    cap_limits();
    if (failures != 0) {
        std::fprintf(stderr, "%d check(s) failed\n", failures);
        return 1;
    }
    std::printf("moe cpu selection checks passed\n");
    return 0;
}
