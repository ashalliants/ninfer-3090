#pragma once

// Which of a call's routed experts the CPU miss service computes, as a pure host function: the rule
// cpu_plan_kernel (cuda/moe_layer.cu) applies on the device when a moe_experts call has a CPU
// channel (ops::MoeCpuChannel), and the oracle the GPU test checks the published MissRequest
// against.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace ninfer::ops::offloaded_moe {

// The widest job the CPU serves: offloaded_moe::kMaxColumns (moe_layer.cu asserts they agree; this
// header stays free of the CUDA headers so host-only builds can test it).
inline constexpr int kMaxCpuJobColumns = 8;
// The widest job a channel may hand the CPU (offloaded_moe::kMaxCpuSplitColumns): a job of w > 8
// columns takes cpu_job_slots(w) consecutive request slots of kMaxCpuJobColumns columns each (the
// last one the rest), which the CPU computes as separate jobs of the same expert. Every column's
// output is the same bits in any grouping, so splitting changes no result.
inline constexpr int kMaxCpuSplitJobColumns = 64;

// Request slots of a job of `columns` columns.
constexpr int cpu_job_slots(int columns) { return (columns + kMaxCpuJobColumns - 1) / kMaxCpuJobColumns; }

// The request slots of the chosen jobs (indices into the call's jobs, in request order) with
// job_columns[j] columns each: {job, first column of the slot within the job}, slot by slot.
struct CpuSlot {
    int job    = 0;
    int column = 0;
    bool operator==(const CpuSlot&) const = default;
};
inline std::vector<CpuSlot> cpu_request_slots(std::span<const int> chosen, std::span<const std::int32_t> job_columns) {
    std::vector<CpuSlot> out;
    for (const int j : chosen) {
        for (int c = 0; c < job_columns[static_cast<std::size_t>(j)]; c += kMaxCpuJobColumns) { out.push_back({j, c}); }
    }
    return out;
}

// The first chosen jobs of `ordered` (candidates in selection order) that fit: at most `want` jobs
// and `slot_cap` slots, stopping at the first that does not fit. Candidates come in non-decreasing
// width, so a job that does not fit leaves no room for any later one: this is the device's
// per-width rule (cpu_plan_kernel).
inline std::vector<int> take_cpu_slots(std::vector<int> ordered, std::span<const std::int32_t> job_columns, int want,
                                       int slot_cap) {
    int slots = 0;
    std::size_t n = 0;
    while (n < ordered.size() && static_cast<int>(n) < want) {
        const int need = cpu_job_slots(job_columns[static_cast<std::size_t>(ordered[n])]);
        if (slots + need > slot_cap) { break; }
        slots += need;
        ++n;
    }
    ordered.resize(n);
    return ordered;
}

/// The call's jobs are job_experts[j] (MoeDispatch::jobs, ascending expert id) with job_columns[j]
/// columns each; resident[e] != 0 when expert e has a device frame. Of the M non-resident jobs,
/// want = min(cap, divisor > 0 ? M - M / divisor : M) go to the CPU, chosen among those with at most
/// max_job_columns columns (up to kMaxCpuSplitJobColumns): the fewest-column ones first and, within a
/// width, in job order, while their slots (cpu_job_slots) fit in cap. Returns the chosen jobs'
/// indices into job_experts, in the order the request lists them (cpu_request_slots expands them).
inline std::vector<int> select_cpu_jobs(std::span<const std::int32_t> job_experts,
                                        std::span<const std::int32_t> job_columns,
                                        std::span<const std::uint8_t> resident, int cap, int divisor,
                                        int max_job_columns) {
    if (job_experts.size() != job_columns.size() || cap < 0 || divisor < 0 || max_job_columns < 1 ||
        max_job_columns > kMaxCpuSplitJobColumns) {
        throw std::invalid_argument("select_cpu_jobs: inconsistent jobs or options out of range");
    }
    int misses = 0;
    std::vector<int> eligible;
    for (std::size_t j = 0; j < job_experts.size(); ++j) {
        const std::int32_t expert = job_experts[j];
        if (expert < 0 || static_cast<std::size_t>(expert) >= resident.size() || job_columns[j] < 1) {
            throw std::invalid_argument("select_cpu_jobs: a job's expert or column count is out of range");
        }
        if (resident[static_cast<std::size_t>(expert)] != 0) { continue; }
        ++misses;
        if (job_columns[j] <= max_job_columns) { eligible.push_back(static_cast<int>(j)); }
    }
    const int want = std::min(cap, divisor > 0 ? misses - misses / divisor : misses);
    std::stable_sort(eligible.begin(), eligible.end(), [&](int a, int b) {
        return job_columns[static_cast<std::size_t>(a)] < job_columns[static_cast<std::size_t>(b)];
    });
    return take_cpu_slots(std::move(eligible), job_columns, want, cap);
}

} // namespace ninfer::ops::offloaded_moe
