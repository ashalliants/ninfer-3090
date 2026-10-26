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

// A run of tokens whose positions advance by one on every axis: token index + k sits at
// first[axis] + k. Text is a single run, so the original positions cost a few bytes.
struct PositionRun {
    std::uint32_t index = 0;
    std::array<std::int32_t, 3> first{};

    [[nodiscard]] friend bool operator==(const PositionRun&, const PositionRun&) noexcept = default;
};

struct GapMap {
    std::uint32_t begin        = 0;        // M: first original index of U
    std::uint32_t original_end = 0;        // M + |U|
    std::array<std::uint64_t, 2> digest{}; // gap_map_digest(original, original_end): a pre-check only
    std::vector<std::pair<std::uint32_t, std::uint32_t>> kept_runs; // (offset in U, length), ascending
    // The original prompt over [0, original_end), compared exactly before any rewrite.
    std::vector<TokenId> original_ids;
    std::vector<std::uint8_t> original_types;
    std::vector<PositionRun> original_positions; // ascending index, first run at 0

    [[nodiscard]] std::uint32_t kept_tokens() const noexcept;
    [[nodiscard]] std::uint32_t dropped_tokens() const noexcept {
        return original_end - begin - kept_tokens();
    }
};

using GapMapSet = std::vector<std::shared_ptr<const GapMap>>; // lineage order

// Digest of token ids, types and the three position axes over [0, end). It covers nothing else
// (no rewrite frontiers), so a follow-up serialization that only adds frontiers still matches.
[[nodiscard]] std::array<std::uint64_t, 2> gap_map_digest(const PreparedPromptData& prompt,
                                                          std::uint32_t end);

// U is [begin, original.token_ids.size()). kept_offsets are ascending unique offsets into U. The
// prompt must have no Vision items (lossy prefill is text only).
[[nodiscard]] GapMap make_gap_map(const PreparedPromptData& original, std::uint32_t begin,
                                  std::span<const std::uint32_t> kept_offsets);

// Where the prompt's generation opener (`<|im_start|>assistant\n<think>\n`) begins: the last
// `turn_start` with no `turn_end` after it, the open turn the model is asked to write. nullopt when
// either token is unknown, when the prompt has no open turn, or when it continues the final
// assistant message (continuation_content set): then the prompt has no generation opener.
[[nodiscard]] std::optional<std::uint32_t> generation_opener(const PreparedPromptData& prompt,
                                                             std::optional<TokenId> turn_start,
                                                             std::optional<TokenId> turn_end);

// The same map with U ending at max(last dropped token + 1, opener): through the end of the last
// message, which a follow-up turn reproduces verbatim, and not through the generation opener, which
// a follow-up serializes differently (its assistant turn replaces it). Without this no follow-up
// would ever match the map. It does not end at the last dropped token: the recent window is always
// kept, so that would make every prompt sharing the original through the last gap (the same
// document with another question, a regenerated last message, a parallel fork) inherit a selection
// scored with another prompt's query rows. Only part of the trailing kept run leaves U (and the
// stored original), and compaction copies it unchanged either way, so compact_prompt gives the
// same prompt and frontier translation the same frontiers. nullopt (or an opener at or past
// original_end) keeps the full span. `original` is the prompt the map was made from. A keep-all map
// (nothing dropped) is a logic error: a skipped selection makes no gap map.
[[nodiscard]] GapMap end_at_last_message(GapMap map, const PreparedPromptData& original,
                                         std::optional<std::uint32_t> opener);

// Kept prompt: [0, begin) unchanged, then the kept tokens of U, then [original_end, size)
// unchanged. token_ids, token_types and all three position axes are subset and keep their original
// values; rope_delta grows by the dropped count. Rewrite checkpoints, rewrite execution frontiers and
// cache opportunities are mapped through the index translation; a frontier is dropped when the
// token just before it was dropped (two original frontiers would otherwise share one compacted
// frontier). Vision token spans after U shift down; a span touching U is rejected. The n-gram index
// is reset. Requires external_prefix_tokens <= begin and at least original_end tokens. The digest is
// not checked: use apply_gap_map for prompts that are not known to match.
[[nodiscard]] PreparedPromptData compact_prompt(const PreparedPromptData& original, const GapMap& map);

// nullopt unless the prompt has at least original_end tokens, no Vision tokens before
// original_end, and token ids, types and all three position axes over [0, original_end) equal the
// map's stored original exactly (the digest is only a quick reject); otherwise
// compact_prompt(prompt, map). Also nullopt, rather than a throw, where compaction could not
// represent the result: an installed external prefix reaching past the map's begin, or a
// rope_delta that would leave int32. Rewrite frontiers do not take part in the match. Several maps
// of one lineage apply in order, each to the previous result.
[[nodiscard]] std::optional<PreparedPromptData> apply_gap_map(const PreparedPromptData& prompt,
                                                              const GapMap& map);

// The maps of a lineage a checkpoint at `frontier` holds kept tokens of (begin < frontier). A
// shallower checkpoint does not depend on them and does not keep them alive.
[[nodiscard]] inline GapMapSet gap_maps_through(const GapMapSet& lineage, std::uint32_t frontier) {
    GapMapSet out;
    for (const auto& map : lineage) {
        if (map && map->begin < frontier) { out.push_back(map); }
    }
    return out;
}

// The frontier in the original prompt that `frontier` of a prompt rewritten through `lineage`
// (in the order the maps were applied) stands for: a frontier at or below a map's begin is
// unchanged, one past its kept tokens moves up by the dropped count, and one inside them ends just
// after the original position of the last kept token it covers. Usage counts prompt tokens in
// the client's terms with it (docs/maintainer/lossy-long-prefill.md, "Capacity and reporting").
[[nodiscard]] std::uint32_t original_frontier(const GapMapSet& lineage, std::uint32_t frontier) noexcept;

struct GapMapRewrite {
    PreparedPromptData prompt;
    GapMapSet applied; // in the order they were applied: the prompt's lineage
};

// Rewrites a prompt with every map in `maps` that matches it, the one with the deepest
// original_end first, each later one tried on the previous result, until none matches. Each map
// applies at most once. Null entries are ignored. A prompt no map matches comes back unchanged.
[[nodiscard]] GapMapRewrite rewrite_with_gap_maps(PreparedPromptData prompt,
                                                  std::span<const std::shared_ptr<const GapMap>> maps);

} // namespace ninfer::models::qwen3_5::detail
