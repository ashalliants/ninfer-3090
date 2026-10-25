// Prompt compaction and gap maps for lossy long-prompt prefill (host only).
//
// After token selection keeps a subset of the uncached span U = [begin, original_end), the
// sequence's prompt is replaced by the compacted prompt: [0, begin) unchanged, the kept tokens of U
// with their original positions, then [original_end, size) unchanged. A GapMap records that
// mapping so a later prompt that starts with the same original tokens can be rewritten onto the
// compacted identity before ordinary prefix matching.
#pragma once

#include "models/qwen3_5/frontend/prepared_prompt.h"

#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <utility>
#include <vector>

namespace ninfer::models::qwen3_5::detail {

struct GapMap {
    std::uint32_t begin        = 0;            // M: first original index of U
    std::uint32_t original_end = 0;            // M + |U|
    std::array<std::uint64_t, 2> digest{};     // PrefixShortlistDigests::at(original_end) of the original
    std::vector<std::pair<std::uint32_t, std::uint32_t>> kept_runs; // (offset in U, length), ascending

    [[nodiscard]] std::uint32_t kept_tokens() const noexcept;
    [[nodiscard]] std::uint32_t dropped_tokens() const noexcept {
        return original_end - begin - kept_tokens();
    }
};

using GapMapSet = std::vector<std::shared_ptr<const GapMap>>; // lineage order

// U is [begin, original.token_ids.size()). kept_offsets are ascending unique offsets into U.
[[nodiscard]] GapMap make_gap_map(const PreparedPromptData& original, std::uint32_t begin,
                                  std::span<const std::uint32_t> kept_offsets);

// Kept prompt: [0, begin) unchanged, then the kept tokens of U, then [original_end, size)
// unchanged. token_ids, token_types and all three position axes are subset and keep their original
// values; rope_delta grows by the dropped count. Rewrite checkpoints, rewrite execution frontiers and
// cache opportunities are mapped through the index translation; a frontier is dropped when the
// token just before it was dropped (two original frontiers would otherwise share one compacted
// frontier). Vision token spans after U shift down; a span touching U is rejected. The n-gram index
// is reset. Requires external_prefix_tokens <= begin and at least original_end tokens. The digest is
// not checked: use apply_gap_map for prompts that are not known to match.
[[nodiscard]] PreparedPromptData compact_prompt(const PreparedPromptData& original, const GapMap& map);

// nullopt unless the prompt has at least original_end tokens and its digest over
// [0, original_end) equals map.digest; otherwise compact_prompt(prompt, map). Several maps of one
// lineage apply in order, each to the previous result.
[[nodiscard]] std::optional<PreparedPromptData> apply_gap_map(const PreparedPromptData& prompt,
                                                              const GapMap& map);

} // namespace ninfer::models::qwen3_5::detail
