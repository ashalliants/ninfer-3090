// Token selection for lossy long-prompt prefill (host only).
//
// Given one attention-mass score per uncached prompt token, pick the tokens to keep: the forced
// ones (explicit, head, recent) plus whole blocks in descending block score up to the budget.
#pragma once

#include "ninfer/types.h"

#include <cstdint>
#include <span>
#include <vector>

namespace ninfer::models::qwen3_5::detail {

struct SelectionInput {
    std::span<const float> scores;        // one per token of U, attention mass
    std::span<const std::uint8_t> forced; // one per token of U; nonzero = always keep
};

struct SelectionResult {
    std::vector<std::uint32_t> kept; // offsets into U, ascending, unique
    bool skipped = false;            // skip rule fired: kept == every offset
};

// Forced tokens are always kept, even past the budget. When their count exceeds
// skip_fraction * |U| everything is kept and `skipped` is set. Otherwise whole blocks are added in
// descending block score (ties: lower index first) until the budget ceil(keep_fraction * |U|) is
// reached; the last block may overshoot by less than one block.
[[nodiscard]] SelectionResult select_tokens(const SelectionInput& input, const LossyPrefillOptions& options);

} // namespace ninfer::models::qwen3_5::detail
