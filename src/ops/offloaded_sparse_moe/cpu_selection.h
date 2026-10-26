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
#include <vector>

namespace ninfer::ops::offloaded_moe {

// The widest job the CPU serves: offloaded_moe::kMaxColumns (moe_layer.cu asserts they agree; this
// header stays free of the CUDA headers so host-only builds can test it).
inline constexpr int kMaxCpuJobColumns = 8;

/// The call's jobs are job_experts[j] (MoeDispatch::jobs, ascending expert id) with job_columns[j]
/// columns each; resident[e] != 0 when expert e has a device frame. Of the M non-resident jobs,
/// want = min(cap, divisor > 0 ? M - M / divisor : M) go to the CPU, chosen among those with at most
/// max_job_columns columns: the fewest-column ones first and, within a width, in job order. Returns
/// the chosen jobs' indices into job_experts, in the order the request lists them (its slots).
inline std::vector<int> select_cpu_jobs(std::span<const std::int32_t> job_experts,
                                        std::span<const std::int32_t> job_columns,
                                        std::span<const std::uint8_t> resident, int cap, int divisor,
                                        int max_job_columns) {
    if (job_experts.size() != job_columns.size() || cap < 0 || divisor < 0 || max_job_columns < 1 ||
        max_job_columns > kMaxCpuJobColumns) {
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
    if (static_cast<int>(eligible.size()) > want) { eligible.resize(static_cast<std::size_t>(want)); }
    return eligible;
}

} // namespace ninfer::ops::offloaded_moe
