#pragma once

#include <cstdint>
#include <optional>
#include <span>

namespace ninfer::runtime {

// Policy for letting a short request in ahead of a long prefill.
//
// A fresh request never preempts a resident one, so with every lane busy it waits for the lane's
// request to finish: behind a 100,000-token prompt that is about a minute, whatever the newcomer
// itself costs. A prefill is cheap to interrupt, though. It stops cleanly between steps and a
// paused request keeps its progress as a snapshot, so the Engine can pause the long prefill, serve
// the short request and resume the long one afterwards.
//
// This decides only whom to pause. The Engine decides whether a snapshot can be taken.
//
// A waiting request qualifies when it costs at most 1/kYieldShortestRatio of what the victim still
// has to do, so a request that is itself long never displaces another, and a victim qualifies only
// with enough left to be worth the snapshot. Each request can be paused at most
// kMaximumYieldsPerRequest times, so a stream of short requests cannot hold a long one back for
// ever.
inline constexpr std::uint64_t kYieldShortestRatio        = 4;
inline constexpr std::uint64_t kMaximumYieldsPerRequest = 4;

struct YieldCandidate {
    std::uint32_t lane           = 0;
    // Predicted time to finish the request's prefill. UINT64_MAX is the cost model saturating,
    // which says nothing, so such a candidate is never chosen.
    std::uint64_t remaining_ns   = 0;
    // Times the request has already been paused, for any reason.
    std::uint64_t pauses         = 0;
};

// The lane to pause for a request that needs `waiting_ns` of prefill, or nullopt. 0 for
// `min_remaining_ns` turns the policy off.
[[nodiscard]] inline std::optional<std::uint32_t>
choose_yield_victim(std::uint64_t min_remaining_ns, std::uint64_t waiting_ns,
                    std::span<const YieldCandidate> candidates) noexcept {
    if (min_remaining_ns == 0 || waiting_ns == UINT64_MAX) { return std::nullopt; }
    std::optional<YieldCandidate> best;
    for (const YieldCandidate& candidate : candidates) {
        if (candidate.pauses >= kMaximumYieldsPerRequest || candidate.remaining_ns == UINT64_MAX ||
            candidate.remaining_ns < min_remaining_ns ||
            waiting_ns > candidate.remaining_ns / kYieldShortestRatio) {
            continue;
        }
        if (!best || candidate.remaining_ns > best->remaining_ns ||
            (candidate.remaining_ns == best->remaining_ns && candidate.lane < best->lane)) {
            best = candidate;
        }
    }
    return best ? std::optional<std::uint32_t>(best->lane) : std::nullopt;
}

} // namespace ninfer::runtime
