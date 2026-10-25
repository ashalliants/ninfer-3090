#pragma once

// Adapted from Basalt 8d73a39 src/prefill/prefill.cpp (the prefill CPU share) (MIT, third_party/basalt/LICENSE).
// Modified for NInfer-3090: the share is an exact fraction num / den rounded half up in integers
// (Basalt: llround of a double), and the rule runs on the device (cpu_plan_kernel in
// cuda/moe_layer.cu) when a moe_experts call's channel sets share_num / share_den, so each call
// follows its own routing without a host round trip. These pure host functions state that rule and
// are the oracle the GPU test checks the published MissRequest against. Free of CUDA headers, so
// host-only builds test it.

#include "ops/offloaded_sparse_moe/cpu_selection.h"

#include <algorithm>
#include <climits>
#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <vector>

namespace ninfer::ops::offloaded_moe {

/// Of a call's `streamed_experts` non-resident experts, `candidates` of which the CPU may take, the
/// number it takes: min(candidates, round(n * num / den)) with halves rounded up, computed exactly as
/// (2 * num * n + den) / (2 * den). Requires 0 <= candidates <= n, den >= 1 and 0 <= num <= den.
inline int cpu_share_take(int streamed_experts, int candidates, int num, int den) {
    if (streamed_experts < 0 || candidates < 0 || candidates > streamed_experts || den < 1 || num < 0 || num > den) {
        throw std::invalid_argument("cpu_share_take: counts or share out of range");
    }
    const long long share = (2LL * num * streamed_experts + den) / (2LL * den);
    return static_cast<int>(std::min<long long>(candidates, share));
}

/// The prefill share's choice. The call's jobs are job_experts[j] with job_rows[j] rows (columns)
/// each; resident[e] != 0 when expert e has a device frame. Of the n non-resident jobs, those with
/// at most max_rows rows are the candidates, ordered by (rows, expert id); the first
/// min(cap, cpu_share_take(n, candidates, num, den)) go to the CPU. Returns their indices into
/// job_experts in that order (the request's slots). cap is the channel's max_jobs.
inline std::vector<int> select_prefill_cpu_jobs(std::span<const std::int32_t> job_experts,
                                                std::span<const std::int32_t> job_rows,
                                                std::span<const std::uint8_t> resident, int num, int den,
                                                int max_rows = kMaxCpuJobColumns, int cap = INT_MAX) {
    if (job_experts.size() != job_rows.size() || cap < 0 || max_rows < 1 || max_rows > kMaxCpuJobColumns) {
        throw std::invalid_argument("select_prefill_cpu_jobs: inconsistent jobs or options out of range");
    }
    int misses = 0;
    std::vector<int> candidates;
    for (std::size_t j = 0; j < job_experts.size(); ++j) {
        const std::int32_t expert = job_experts[j];
        if (expert < 0 || static_cast<std::size_t>(expert) >= resident.size() || job_rows[j] < 1) {
            throw std::invalid_argument("select_prefill_cpu_jobs: a job's expert or row count is out of range");
        }
        if (resident[static_cast<std::size_t>(expert)] != 0) { continue; }
        ++misses;
        if (job_rows[j] <= max_rows) { candidates.push_back(static_cast<int>(j)); }
    }
    std::sort(candidates.begin(), candidates.end(), [&](int a, int b) {
        const auto ra = job_rows[static_cast<std::size_t>(a)], rb = job_rows[static_cast<std::size_t>(b)];
        return ra != rb ? ra < rb : job_experts[static_cast<std::size_t>(a)] < job_experts[static_cast<std::size_t>(b)];
    });
    const int take = std::min(cap, cpu_share_take(misses, static_cast<int>(candidates.size()), num, den));
    candidates.resize(static_cast<std::size_t>(take));
    return candidates;
}

} // namespace ninfer::ops::offloaded_moe
