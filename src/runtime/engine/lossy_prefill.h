#pragma once

// The Engine's side of lossy long-prompt prefill (docs/maintainer/lossy-long-prefill.md): when a
// request takes the path, and what its prefill is predicted to cost. The Program runs the phases
// (Program::enable_lossy); EngineCore follows them per request (RequestRecord::lossy).

#include "runtime/contract/resources.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <optional>

namespace ninfer::runtime {

struct LossyTriggerInput {
    // LossyPrefillOptions::min_tokens; 0 is off.
    std::uint32_t min_tokens = 0;
    // The plan's lossy().exact_end, the end of the product prompt; nullopt when the Program says
    // the request cannot take the path (no scout, media, a rewritten prompt, ...).
    std::optional<std::uint32_t> exact_end;
    std::uint32_t prompt_tokens = 0;
    bool has_media              = false;
    // The chosen reuse source: none (a cold start), or the role of the checkpoint it binds.
    std::optional<CheckpointRole> source_role;
    std::uint32_t reused_tokens = 0;
};

// M, the end of the exact prefix, when the request takes the lossy path; nullopt otherwise. The
// path runs only for a conversation that is not cached: the source is nothing or a shared
// (product-prompt) checkpoint, never a private conversation record, so a large new turn in a warm
// conversation always prefills in full. The threshold applies to U = [M, prompt end), what is left
// after the product prompt and the reused prefix.
[[nodiscard]] inline std::optional<std::uint32_t>
lossy_prefill_trigger(const LossyTriggerInput& input) noexcept {
    if (input.min_tokens == 0 || !input.exact_end || input.has_media) { return std::nullopt; }
    if (input.source_role && *input.source_role != CheckpointRole::SharedPrefix) {
        return std::nullopt;
    }
    const std::uint32_t exact_end = std::max(*input.exact_end, input.reused_tokens);
    if (exact_end >= input.prompt_tokens || input.prompt_tokens - exact_end <= input.min_tokens) {
        return std::nullopt;
    }
    return exact_end;
}

// Tokens of U a selection is expected to keep, for predicting the main pass before the scout has
// run.
[[nodiscard]] inline std::uint32_t lossy_expected_kept(std::uint32_t uncached_tokens,
                                                       float keep_fraction) noexcept {
    const double keep = std::clamp(static_cast<double>(keep_fraction), 0.0, 1.0);
    return static_cast<std::uint32_t>(std::ceil(keep * static_cast<double>(uncached_tokens)));
}

// `share` of `ns`, saturating like the cost model does.
[[nodiscard]] inline std::uint64_t scaled_ns(std::uint64_t ns, double share) noexcept {
    if (ns == UINT64_MAX) { return UINT64_MAX; }
    const double scaled = static_cast<double>(ns) * std::clamp(share, 0.0, 1.0);
    return scaled >= 1.8e19 ? UINT64_MAX : static_cast<std::uint64_t>(scaled);
}

[[nodiscard]] inline std::uint64_t saturating_add(std::uint64_t a, std::uint64_t b) noexcept {
    return a > UINT64_MAX - b ? UINT64_MAX : a + b;
}

} // namespace ninfer::runtime
