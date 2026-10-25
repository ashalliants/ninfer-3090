// The phases of a lossy long-prompt prefill on one Program lane (header only, links no kernel).
//
// docs/maintainer/lossy-long-prefill.md, "How it works". A lossy request prefills in three phases:
//
//   Exact  an ordinary prefill from the reuse frontier up to M, the end of the product prompt (the
//          last SharedStablePrefix marker at or after the reuse frontier). Captures and shared
//          publication happen as for any prefill. Steps never cross M.
//   Scout  layers below the filter layer over U = [M, prompt end), scoring every token of U. The
//          cursor stays at M; nothing is committed, captured or paused.
//   Main   after selection and compaction, an ordinary prefill of the kept tokens from M.
#pragma once

#include "models/qwen3_5/frontend/prepared_prompt.h"
#include "ninfer/types.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <span>
#include <vector>

namespace ninfer::models::qwen3_5::detail {

enum class LossyPhase : std::uint8_t { Exact, Scout, Main };

// What the lane's next prefill step runs.
enum class LossyStep : std::uint8_t {
    Ordinary,   // an ordinary prefill step (Exact before M, or Main)
    EnterScout, // Exact at M: open the scout and run its first chunk
    Scout,      // the next scout chunk
    Decline,    // Exact at M while another lane holds the one scout: give up the lossy path and
                // prefill all of U as an ordinary request (reported as skipped)
};

// `scout_busy`: another lane holds the Program's single scout session.
[[nodiscard]] constexpr LossyStep lossy_step(LossyPhase phase, std::uint32_t cursor,
                                             std::uint32_t exact_end,
                                             bool scout_busy = false) noexcept {
    switch (phase) {
    case LossyPhase::Exact:
        if (cursor < exact_end) { return LossyStep::Ordinary; }
        return scout_busy ? LossyStep::Decline : LossyStep::EnterScout;
    case LossyPhase::Scout: return LossyStep::Scout;
    case LossyPhase::Main: return LossyStep::Ordinary;
    }
    return LossyStep::Ordinary;
}

// The most tokens an ordinary step may process: an Exact step stops at M, so the scout begins at
// the committed frontier. No limit in Main.
[[nodiscard]] constexpr std::uint32_t lossy_step_limit(LossyPhase phase, std::uint32_t cursor,
                                                       std::uint32_t exact_end) noexcept {
    if (phase != LossyPhase::Exact) { return std::numeric_limits<std::uint32_t>::max(); }
    return exact_end > cursor ? exact_end - cursor : 0U;
}

// Between M and the first Main chunk the lane may not be paused or captured: the scout's own
// pages and the destination GDN slot it advanced are not a consistent state, and the capture
// groups past M still describe the uncompacted prompt.
[[nodiscard]] constexpr bool lossy_refuses_pause(LossyPhase phase) noexcept {
    return phase == LossyPhase::Scout;
}

[[nodiscard]] constexpr bool lossy_holds_captures(LossyPhase phase, std::uint32_t cursor,
                                                  std::uint32_t exact_end) noexcept {
    return phase != LossyPhase::Main && cursor >= exact_end;
}

// The planned end of the product prompt: the deepest SharedStablePrefix opportunity, 0 without one.
[[nodiscard]] inline std::uint32_t
lossy_planned_exact_end(std::span<const PreparedCacheOpportunity> opportunities) noexcept {
    std::uint32_t end = 0;
    for (const auto& opportunity : opportunities) {
        if (opportunity.kind == PromptCacheMarkerKind::SharedStablePrefix) {
            end = std::max(end, opportunity.frontier);
        }
    }
    return end;
}

// M for a bound request: the planned product-prompt end when it lies at or after the reuse
// frontier, else the reuse frontier itself.
[[nodiscard]] constexpr std::uint32_t lossy_exact_end(std::uint32_t planned_exact_end,
                                                      std::uint32_t reuse_frontier) noexcept {
    return std::max(planned_exact_end, reuse_frontier);
}

// One flag per token of `span`: nonzero where the id is in `special_ids` (the tokenizer's special
// and added tokens: role markers, tool-call delimiters, im_start/im_end), which selection always
// keeps. `special_ids` is sorted ascending.
[[nodiscard]] inline std::vector<std::uint8_t>
lossy_forced_tokens(std::span<const TokenId> span, std::span<const TokenId> special_ids) {
    std::vector<std::uint8_t> forced(span.size(), 0);
    for (std::size_t i = 0; i < span.size(); ++i) {
        forced[i] = std::binary_search(special_ids.begin(), special_ids.end(), span[i]) ? 1U : 0U;
    }
    return forced;
}

} // namespace ninfer::models::qwen3_5::detail
