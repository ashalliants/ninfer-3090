#pragma once

#include <cstdint>
#include <vector>

namespace ninfer::runtime {

// Prompt positions where a long prefill leaves a progress anchor: every `spacing` tokens, strictly
// inside the prompt (its end already carries the InputReplay point). A request paused mid-prefill
// that loses its snapshot then replays from the nearest anchor instead of from the root, so the
// recomputation is bounded by `spacing`. Zero spacing turns the feature off.
[[nodiscard]] inline std::vector<std::uint32_t>
progress_anchor_frontiers(std::uint32_t prompt_tokens, std::uint32_t spacing) {
    std::vector<std::uint32_t> frontiers;
    if (spacing == 0) { return frontiers; }
    for (std::uint64_t frontier = spacing; frontier < prompt_tokens; frontier += spacing) {
        frontiers.push_back(static_cast<std::uint32_t>(frontier));
    }
    return frontiers;
}

} // namespace ninfer::runtime
