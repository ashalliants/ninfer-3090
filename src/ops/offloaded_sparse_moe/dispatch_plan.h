#pragma once

// The dispatch of a call's routed entries by expert, as a pure host function: what
// ops::moe_route_dispatch's one-launch route (decode-width calls) writes into MoeDispatch, element
// for element, and the oracle the GPU test checks it against. moe_dispatch writes the same counts,
// offsets and jobs, but its entries within an expert in an unspecified order.

#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <vector>

namespace ninfer::ops::offloaded_moe {

struct DispatchPlan {
    std::vector<std::int32_t> counts;  // [E]: dispatched entries of each expert
    std::vector<std::int32_t> offsets; // [E + 1]: exclusive scan of counts; offsets[E] = the total
    std::vector<std::int32_t> jobs;    // the experts with entries, ascending id (job_count = size)
    std::vector<std::int32_t> entries; // [offsets[E]]: entry indices grouped by expert, each ascending
};

/// ids: the k*T routed ids, column-major (entry t * k + slot is ids[t * k + slot], MoeRouting::ids).
/// An id outside [0, experts) is not dispatched (moe_dispatch drops it the same way; moe_route
/// writes one only for a column it cannot rank, such as an all-NaN column).
inline DispatchPlan plan_dispatch(std::span<const std::int32_t> ids, int experts) {
    if (experts <= 0) { throw std::invalid_argument("plan_dispatch needs at least one expert"); }
    const auto in_range = [&](std::int32_t id) { return id >= 0 && id < experts; };
    DispatchPlan plan;
    plan.counts.assign(static_cast<std::size_t>(experts), 0);
    for (const std::int32_t id : ids) {
        if (in_range(id)) { ++plan.counts[static_cast<std::size_t>(id)]; }
    }
    plan.offsets.assign(static_cast<std::size_t>(experts) + 1, 0);
    for (int e = 0; e < experts; ++e) {
        const auto i        = static_cast<std::size_t>(e);
        plan.offsets[i + 1] = plan.offsets[i] + plan.counts[i];
        if (plan.counts[i] > 0) { plan.jobs.push_back(e); }
    }
    plan.entries.assign(static_cast<std::size_t>(plan.offsets.back()), 0);
    std::vector<std::int32_t> cursor(plan.offsets.begin(), plan.offsets.end() - 1);
    for (std::size_t i = 0; i < ids.size(); ++i) {
        if (in_range(ids[i])) {
            plan.entries[static_cast<std::size_t>(cursor[static_cast<std::size_t>(ids[i])]++)] =
                static_cast<std::int32_t>(i);
        }
    }
    return plan;
}

} // namespace ninfer::ops::offloaded_moe
