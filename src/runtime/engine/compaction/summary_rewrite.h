#pragma once

// The summary strategy's record of one prompt it rewrote (summary_strategy.h), carried with the
// request so usage, prompt progress and the request log count the client's prompt.

#include "ninfer/types.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <optional>

namespace ninfer::runtime::compaction {

// A prompt whose messages were replaced by a summary, as usage and the request log report it.
struct SummaryRewrite {
    // The client's prompt, rendered; usage prompt_tokens.
    std::uint32_t original_tokens = 0;
    // The prompt as submitted.
    std::uint32_t prompt_tokens = 0;
    // Leading and trailing tokens the two have in common (the head, and the tail after the summary).
    std::uint32_t common_prefix = 0;
    std::uint32_t common_suffix = 0;
    // The newest summarised range, [first, last) in the client's message indices.
    std::array<std::uint32_t, 2> messages{};
    // Summarised by this request; false when a recorded summary was substituted (a later turn).
    bool fresh = false;

    // A frontier of the submitted prompt in the client's tokens: unchanged in the common head,
    // shifted by the length difference in the common tail, and the head's end inside the summary.
    // Monotone, and maps prompt_tokens to original_tokens.
    [[nodiscard]] std::uint32_t original_frontier(std::uint32_t frontier) const noexcept {
        if (frontier <= common_prefix) { return frontier; }
        if (frontier >= prompt_tokens - common_suffix) {
            const std::int64_t shifted = static_cast<std::int64_t>(frontier) + original_tokens -
                                         static_cast<std::int64_t>(prompt_tokens);
            return static_cast<std::uint32_t>(
                std::clamp<std::int64_t>(shifted, common_prefix, original_tokens));
        }
        return common_prefix;
    }
};

// What the summary strategy did for one prompt, carried with the request: the probe's answer
// (set when the prompt was long enough to be probed) and the rewrite (set when messages were
// replaced). `prompt_tokens` is the submitted prompt's length.
struct SummaryRecord {
    std::optional<LossySummaryProbe> probe;
    std::optional<SummaryRewrite> rewrite;
    std::uint32_t prompt_tokens = 0;
};

} // namespace ninfer::runtime::compaction
