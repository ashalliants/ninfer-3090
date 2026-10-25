// Lossy long-prefill prompt compaction and gap maps (CPU only).
#include "models/qwen3_5/program/lossy/gap_map.h"
#include "models/qwen3_5/program/prefix_identity.h"

#include <cstdint>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using ninfer::TokenId;
using ninfer::models::qwen3_5::PreparedCacheOpportunity;
using ninfer::models::qwen3_5::PreparedPromptData;
using ninfer::models::qwen3_5::RewriteCheckpointKind;
using ninfer::models::qwen3_5::RewriteCheckpointSpec;
using ninfer::models::qwen3_5::detail::apply_gap_map;
using ninfer::models::qwen3_5::detail::compact_prompt;
using ninfer::models::qwen3_5::detail::gap_map_digest;
using ninfer::models::qwen3_5::detail::GapMap;
using ninfer::models::qwen3_5::detail::make_gap_map;
using ninfer::models::qwen3_5::detail::original_frontier;
using ninfer::models::qwen3_5::detail::GapMapSet;
using ninfer::models::qwen3_5::detail::PositionRun;
using ninfer::models::qwen3_5::detail::rewrite_with_gap_maps;
using ninfer::models::qwen3_5::detail::ResidentPrefixIdentity;

void require(bool ok, const std::string& what) {
    if (!ok) { throw std::runtime_error("FAILED: " + what); }
}

// A text prompt from explicit ids and positions (the same position on all three axes).
PreparedPromptData make_prompt(const std::vector<TokenId>& ids,
                               const std::vector<std::int32_t>& positions,
                               std::int32_t rope_delta) {
    PreparedPromptData p;
    p.token_ids = ids;
    p.token_types.resize(ids.size());
    for (std::size_t i = 0; i < ids.size(); ++i) {
        p.token_types[i] = static_cast<std::uint8_t>(ids[i] & 1);
    }
    for (int axis = 0; axis < 3; ++axis) {
        p.positions.insert(p.positions.end(), positions.begin(), positions.end());
    }
    p.rope_delta = rope_delta;
    return p;
}

// n text tokens with id 100 + i at position i.
PreparedPromptData text_prompt(std::size_t n) {
    std::vector<TokenId> ids(n);
    std::vector<std::int32_t> positions(n);
    for (std::size_t i = 0; i < n; ++i) {
        ids[i]       = static_cast<TokenId>(100 + i);
        positions[i] = static_cast<std::int32_t>(i);
    }
    return make_prompt(ids, positions, 0);
}

// Appends text tokens continuing the prompt's ids at slot + rope_delta positions.
PreparedPromptData extend(const PreparedPromptData& p, std::size_t extra, TokenId first_id) {
    std::vector<TokenId> ids = p.token_ids;
    std::vector<std::int32_t> positions(p.position_axis(0).begin(), p.position_axis(0).end());
    for (std::size_t i = 0; i < extra; ++i) {
        ids.push_back(static_cast<TokenId>(first_id + static_cast<TokenId>(i)));
        positions.push_back(static_cast<std::int32_t>(p.token_ids.size() + i) + p.rope_delta);
    }
    return make_prompt(ids, positions, p.rope_delta);
}

void require_axes(const PreparedPromptData& p, const std::vector<std::int32_t>& expected,
                  const std::string& what) {
    for (int axis = 0; axis < 3; ++axis) {
        const auto values = p.position_axis(axis);
        require(std::vector<std::int32_t>(values.begin(), values.end()) == expected,
                what + ": axis " + std::to_string(axis));
    }
}

std::vector<TokenId> ids_at(const PreparedPromptData& p, const std::vector<std::size_t>& index) {
    std::vector<TokenId> out;
    for (const std::size_t i : index) { out.push_back(p.token_ids[i]); }
    return out;
}

std::vector<std::uint8_t> types_at(const PreparedPromptData& p,
                                   const std::vector<std::size_t>& index) {
    std::vector<std::uint8_t> out;
    for (const std::size_t i : index) { out.push_back(p.token_types[i]); }
    return out;
}

const std::vector<std::uint32_t> kKept{0, 1, 5, 6, 7}; // U offsets; original indices 4,5,9,10,11
const std::vector<std::size_t> kKeptIndex{0, 1, 2, 3, 4, 5, 9, 10, 11};
const std::vector<std::int32_t> kKeptPositions{0, 1, 2, 3, 4, 5, 9, 10, 11};

void test_subsets_tokens_and_keeps_positions() {
    const PreparedPromptData original = text_prompt(20);
    const GapMap map                  = make_gap_map(original, 4, kKept);
    require(map.begin == 4 && map.original_end == 20, "map span");
    require(map.kept_tokens() == 5 && map.dropped_tokens() == 11, "map counts");
    require((map.kept_runs == std::vector<std::pair<std::uint32_t, std::uint32_t>>{{0, 2}, {5, 3}}),
            "map runs");
    require(map.digest == gap_map_digest(original, 20), "digest over [0, original_end)");
    require(map.original_ids == original.token_ids && map.original_types == original.token_types,
            "stored original ids and types");
    require((map.original_positions == std::vector<PositionRun>{PositionRun{0, {0, 0, 0}}}),
            "text positions stored as one run");

    const PreparedPromptData kept = compact_prompt(original, map);
    require(kept.token_ids.size() == 9, "kept length");
    require(kept.token_ids == ids_at(original, kKeptIndex), "kept ids");
    require(kept.token_types == types_at(original, kKeptIndex), "kept types");
    require_axes(kept, kKeptPositions, "kept positions");
    require(kept.rope_delta == 11, "rope_delta grows by the dropped count");
    require(kept.ngram_index.index == nullptr, "ngram index reset");
}

void test_tail_after_u_is_unchanged() {
    // U = [4, 16) of a 16-token prompt, applied to a 24-token prompt with the same first 16 tokens.
    const PreparedPromptData head = text_prompt(16);
    const GapMap map              = make_gap_map(head, 4, std::vector<std::uint32_t>{0, 1, 5, 6, 7});
    const PreparedPromptData full = text_prompt(24);
    const PreparedPromptData kept = compact_prompt(full, map);
    std::vector<std::size_t> index{0, 1, 2, 3, 4, 5, 9, 10, 11};
    for (std::size_t i = 16; i < 24; ++i) { index.push_back(i); }
    require(kept.token_ids == ids_at(full, index), "tail ids");
    require(kept.token_types == types_at(full, index), "tail types");
    std::vector<std::int32_t> positions(index.begin(), index.end());
    require_axes(kept, positions, "tail positions");
    require(kept.rope_delta == 7, "rope_delta");
    // Text after U sits at slot + rope_delta.
    for (std::size_t slot = 9; slot < kept.token_ids.size(); ++slot) {
        require(kept.position_axis(0)[slot] == static_cast<std::int32_t>(slot) + kept.rope_delta,
                "tail position is slot + rope_delta");
    }
}

void test_rewrite_checkpoint_mapped() {
    PreparedPromptData original = text_prompt(20);
    // Frontier 11 follows kept token 10 (compacted index 7); recovery 10 follows kept token 9.
    original.identity.rewrite_checkpoint =
        RewriteCheckpointSpec{RewriteCheckpointKind::TurnClosure, 11, 10};
    original.identity.rewrite_execution_frontiers = {3, 6, 10, 12, 20};
    original.context_cache.opportunities          = {
        PreparedCacheOpportunity{.frontier = 2, .input_order = 0},
        PreparedCacheOpportunity{.frontier = 8, .input_order = 1},
        PreparedCacheOpportunity{.frontier = 12, .input_order = 2},
    };
    const GapMap map              = make_gap_map(original, 4, kKept);
    const PreparedPromptData kept = compact_prompt(original, map);
    require(kept.identity.rewrite_checkpoint.has_value(), "checkpoint kept");
    require(kept.identity.rewrite_checkpoint->kind == RewriteCheckpointKind::TurnClosure, "kind");
    require(kept.identity.rewrite_checkpoint->frontier == 8, "frontier mapped");
    require(kept.identity.rewrite_checkpoint->recovery_frontier == 7, "recovery mapped");
    // 3 is before U; 6 follows kept 5; 10 follows kept 9; 12 follows kept 11; 20 follows dropped 19.
    require((kept.identity.rewrite_execution_frontiers == std::vector<std::uint32_t>{3, 6, 7, 9}),
            "execution frontiers mapped, dropped-span ones removed");
    require(kept.context_cache.opportunities.size() == 2, "opportunity in dropped span removed");
    require(kept.context_cache.opportunities[0].frontier == 2 &&
                kept.context_cache.opportunities[0].input_order == 0,
            "opportunity before U unchanged");
    require(kept.context_cache.opportunities[1].frontier == 9 &&
                kept.context_cache.opportunities[1].input_order == 2,
            "opportunity in U mapped");
}

void test_rewrite_checkpoint_in_dropped_span_is_dropped() {
    PreparedPromptData original = text_prompt(20);
    // Frontier 8 follows dropped token 7.
    original.identity.rewrite_checkpoint =
        RewriteCheckpointSpec{RewriteCheckpointKind::ResponseReplay, 8, 6};
    const GapMap map              = make_gap_map(original, 4, kKept);
    const PreparedPromptData kept = compact_prompt(original, map);
    require(!kept.identity.rewrite_checkpoint.has_value(), "checkpoint dropped");

    // Its recovery frontier alone in a dropped span also drops it.
    original.identity.rewrite_checkpoint =
        RewriteCheckpointSpec{RewriteCheckpointKind::ResponseReplay, 11, 8};
    require(!compact_prompt(original, map).identity.rewrite_checkpoint.has_value(),
            "checkpoint with dropped recovery frontier dropped");
}

void test_result_identity_differs_from_original() {
    const PreparedPromptData original = text_prompt(20);
    const GapMap map                  = make_gap_map(original, 4, kKept);
    const PreparedPromptData kept     = compact_prompt(original, map);
    ResidentPrefixIdentity identity;
    identity.assign(kept);
    require(!identity.matches(original, original.token_ids.size()), "no match at original length");
    require(!identity.matches(original, kept.token_ids.size()), "no match at kept length");
    require(identity.matches(original, 6), "shared prefix up to the first gap still matches");
    require(gap_map_digest(kept, 9) != map.digest, "kept digest differs");
}

void test_follow_up_prompt_is_rewritten() {
    const PreparedPromptData original = text_prompt(20);
    const GapMap map                  = make_gap_map(original, 4, kKept);
    const PreparedPromptData follow   = extend(original, 7, 500);
    const std::optional<PreparedPromptData> rewritten = apply_gap_map(follow, map);
    require(rewritten.has_value(), "rewritten");
    const PreparedPromptData compacted = compact_prompt(original, map);
    std::vector<TokenId> ids           = compacted.token_ids;
    std::vector<std::int32_t> positions = kKeptPositions;
    for (std::int32_t i = 0; i < 7; ++i) {
        ids.push_back(500 + i);
        positions.push_back(20 + i);
    }
    require(rewritten->token_ids == ids, "rewritten ids");
    require_axes(*rewritten, positions, "rewritten positions");
    require(rewritten->rope_delta == 11, "rewritten rope_delta");

    // The rewritten prompt matches the compacted original's identity at its full length.
    ResidentPrefixIdentity identity;
    identity.assign(compacted);
    require(identity.matches(*rewritten, compacted.token_ids.size()), "matches compacted identity");
}

void test_partial_match_is_not_rewritten() {
    const PreparedPromptData original = text_prompt(20);
    const GapMap map                  = make_gap_map(original, 4, kKept);
    PreparedPromptData differs        = extend(original, 7, 500);
    differs.token_ids[7]              = 9999; // a dropped token inside U
    require(!apply_gap_map(differs, map).has_value(), "differs inside U");
    PreparedPromptData differs_kept = extend(original, 7, 500);
    differs_kept.token_ids[19]      = 9999; // last token of U
    require(!apply_gap_map(differs_kept, map).has_value(), "differs at the end of U");
    PreparedPromptData moved = extend(original, 7, 500);
    moved.positions[10]      = 42; // same tokens, different position on axis 0
    require(!apply_gap_map(moved, map).has_value(), "differs in position");
    require(!apply_gap_map(text_prompt(19), map).has_value(), "shorter than original_end");
    require(apply_gap_map(original, map).has_value(), "exactly the original is rewritten");
}

void test_follow_up_with_extra_rewrite_frontier_is_rewritten() {
    const PreparedPromptData original = text_prompt(20);
    const GapMap map                  = make_gap_map(original, 4, kKept);
    PreparedPromptData follow         = extend(original, 7, 500);
    // The follow-up serialization marks frontiers the original prompt did not have.
    follow.identity.rewrite_execution_frontiers = {3, 12, 20, 24};
    const auto rewritten = apply_gap_map(follow, map);
    require(rewritten.has_value(), "rewritten despite extra rewrite frontiers");
    require(rewritten->token_ids.size() == 16, "rewritten length");
    // 3 is before U; 12 follows kept 11; 20 follows dropped 19; 24 is after U.
    require((rewritten->identity.rewrite_execution_frontiers == std::vector<std::uint32_t>{3, 9, 13}),
            "frontiers translated");
}

void test_digest_collision_is_not_enough() {
    const PreparedPromptData original = text_prompt(20);
    const PreparedPromptData follow   = extend(original, 7, 500);
    GapMap map                        = make_gap_map(original, 4, kKept);
    require(map.digest == gap_map_digest(follow, 20), "digest matches the follow-up");
    GapMap ids = map;
    ids.original_ids[7] = 9999; // a dropped token: the digest alone would let it through
    require(!apply_gap_map(follow, ids).has_value(), "stored id differs");
    GapMap types = map;
    types.original_types[10] ^= 1U;
    require(!apply_gap_map(follow, types).has_value(), "stored type differs");
    GapMap positions = map;
    positions.original_positions.push_back(PositionRun{6, {7, 6, 6}});
    require(!apply_gap_map(follow, positions).has_value(), "stored position differs");
    require(apply_gap_map(follow, map).has_value(), "unaltered map still applies");
}

void test_two_maps_apply_in_order() {
    const PreparedPromptData original = text_prompt(20);
    const GapMap first                = make_gap_map(original, 4, kKept);
    // The next turn arrives: 10 new tokens, rewritten by the first map, then scouted itself with
    // U = [9, 19) of the rewritten prompt.
    const PreparedPromptData turn2 = extend(original, 10, 500);
    const auto turn2_rewritten     = apply_gap_map(turn2, first);
    require(turn2_rewritten.has_value() && turn2_rewritten->token_ids.size() == 19, "turn 2");
    const GapMap second = make_gap_map(*turn2_rewritten, 9, std::vector<std::uint32_t>{0, 8, 9});
    require(second.begin == 9 && second.original_end == 19 && second.dropped_tokens() == 7,
            "second map span");

    // Turn 3 adds 3 more tokens to turn 2.
    const PreparedPromptData turn3 = extend(turn2, 3, 700);
    require(!apply_gap_map(turn3, second).has_value(), "second map does not apply first");
    const auto after_first = apply_gap_map(turn3, first);
    require(after_first.has_value(), "first map applies");
    const auto after_second = apply_gap_map(*after_first, second);
    require(after_second.has_value(), "second map applies to the first's result");
    require((after_second->token_ids ==
             std::vector<TokenId>{100, 101, 102, 103, 104, 105, 109, 110, 111, 500, 508, 509, 700,
                                  701, 702}),
            "ids after both maps");
    require_axes(*after_second, {0, 1, 2, 3, 4, 5, 9, 10, 11, 20, 28, 29, 30, 31, 32},
                 "positions after both maps");
    require(after_second->rope_delta == 18, "rope_delta after both maps");
    require(after_second->position_axis(0)[12] == 12 + after_second->rope_delta,
            "new text at slot + rope_delta");
}

void test_rejects_invalid_input() {
    const PreparedPromptData original = text_prompt(20);
    auto throws                       = [](auto&& fn) {
        try {
            fn();
        } catch (const std::invalid_argument&) { return true; }
        return false;
    };
    require(throws([&] { (void)make_gap_map(original, 21, std::vector<std::uint32_t>{}); }),
            "begin past the end");
    require(throws([&] { (void)make_gap_map(original, 4, std::vector<std::uint32_t>{1, 1}); }),
            "duplicate offsets");
    require(throws([&] { (void)make_gap_map(original, 4, std::vector<std::uint32_t>{3, 2}); }),
            "unsorted offsets");
    require(throws([&] { (void)make_gap_map(original, 4, std::vector<std::uint32_t>{16}); }),
            "offset outside U");
    const GapMap map = make_gap_map(original, 4, kKept);
    require(throws([&] { (void)compact_prompt(text_prompt(19), map); }), "prompt shorter than U");
    PreparedPromptData vision = original;
    vision.vision_items.emplace_back();
    vision.vision_items.back().token_spans.push_back({1, 2});
    require(throws([&] { (void)make_gap_map(vision, 4, kKept); }), "Vision prompt rejected");
    PreparedPromptData follow_vision = extend(original, 7, 500);
    follow_vision.vision_items        = vision.vision_items;
    require(!apply_gap_map(follow_vision, map).has_value(), "Vision before original_end: no rewrite");
    PreparedPromptData external = original;
    external.external_prefix_tokens = 5;
    require(throws([&] { (void)compact_prompt(external, map); }), "external prefix inside U");
    external.external_prefix_tokens = 4;
    require(compact_prompt(external, map).external_prefix_tokens == 4, "external prefix up to M");
}


// Where compaction would throw (an installed graft reaching into U, a rope_delta past int32), the
// map does not apply: planning must treat it as no rewrite, never fail the request.
void test_apply_returns_nullopt_where_compaction_would_throw() {
    const PreparedPromptData original = text_prompt(20);
    const GapMap map                  = make_gap_map(original, 4, kKept);
    PreparedPromptData external       = extend(original, 7, 500);
    external.external_prefix_tokens   = 5;
    require(!apply_gap_map(external, map).has_value(), "external prefix inside U: no rewrite");
    external.external_prefix_tokens = 4;
    require(apply_gap_map(external, map).has_value(), "external prefix up to M is rewritten");
    PreparedPromptData far = extend(original, 7, 500);
    far.rope_delta         = std::numeric_limits<std::int32_t>::max() - 3; // + 11 dropped overflows
    require(!apply_gap_map(far, map).has_value(), "rope_delta overflow: no rewrite");
}

std::shared_ptr<const GapMap> shared_map(GapMap map) {
    return std::make_shared<const GapMap>(std::move(map));
}

// Planning applies every live map that matches, deepest original_end first, each to the previous
// result, so a lineage's maps compose in any registry order and an unrelated map is ignored.
void test_rewrite_applies_lineage_in_any_order() {
    const PreparedPromptData original = text_prompt(20);
    const auto first                  = shared_map(make_gap_map(original, 4, kKept));
    const PreparedPromptData turn2    = extend(original, 10, 500);
    const auto turn2_rewritten        = apply_gap_map(turn2, *first);
    require(turn2_rewritten.has_value(), "turn 2 rewritten");
    const auto second =
        shared_map(make_gap_map(*turn2_rewritten, 9, std::vector<std::uint32_t>{0, 8, 9}));
    const auto unrelated = shared_map(make_gap_map(extend(text_prompt(4), 16, 900), 4, kKept));
    const PreparedPromptData turn3 = extend(turn2, 3, 700);

    const GapMapSet registry{unrelated, second, first};
    const auto rewritten = rewrite_with_gap_maps(turn3, registry);
    require(rewritten.applied.size() == 2 && rewritten.applied[0] == first &&
                rewritten.applied[1] == second,
            "both lineage maps applied in lineage order");
    const auto expected = apply_gap_map(*apply_gap_map(turn3, *first), *second);
    require(rewritten.prompt.token_ids == expected->token_ids &&
                rewritten.prompt.positions == expected->positions &&
                rewritten.prompt.rope_delta == expected->rope_delta,
            "rewrite equals the maps applied in order");

    PreparedPromptData other = text_prompt(25);
    other.token_ids[2]       = 9999; // before every map's begin
    const auto none          = rewrite_with_gap_maps(other, registry);
    require(none.applied.empty() && none.prompt.token_ids == other.token_ids,
            "a prompt no map matches is unchanged");
}

// Two maps that both match the raw prompt: the one covering more of it wins, and the other, which
// no longer matches the rewritten prompt, is not applied.
void test_rewrite_prefers_the_deepest_map() {
    const PreparedPromptData original = text_prompt(20);
    const auto shallow                = shared_map(make_gap_map(original, 4, kKept));
    const PreparedPromptData longer   = extend(original, 10, 500);
    const auto deep = shared_map(make_gap_map(longer, 4, std::vector<std::uint32_t>{0, 1, 25}));
    const PreparedPromptData prompt = extend(longer, 2, 800);
    const auto rewritten            = rewrite_with_gap_maps(prompt, GapMapSet{shallow, deep});
    require(rewritten.applied.size() == 1 && rewritten.applied[0] == deep, "deepest map applied");
    require(rewritten.prompt.token_ids == compact_prompt(prompt, *deep).token_ids,
            "deep compaction");
}

// A map that drops nothing rewrites a prompt to itself; it is applied once, not forever.
void test_rewrite_applies_each_map_once() {
    const PreparedPromptData original = text_prompt(8);
    std::vector<std::uint32_t> all{0, 1, 2, 3, 4, 5};
    const auto identity = shared_map(make_gap_map(original, 2, all));
    const auto rewritten = rewrite_with_gap_maps(extend(original, 3, 500), GapMapSet{identity});
    require(rewritten.applied.size() == 1, "identity map applied once");
    require(rewritten.prompt.token_ids.size() == 11, "identity map keeps every token");
}

// Usage reports frontiers of a rewritten prompt in the client's tokens.
void test_original_frontier_maps_through_the_lineage() {
    const PreparedPromptData original = text_prompt(20);
    const auto first = shared_map(make_gap_map(original, 4, kKept)); // 16 of U, 5 kept, 11 dropped
    const GapMapSet one{first};
    require(original_frontier({}, 7) == 7, "no lineage leaves a frontier alone");
    require(original_frontier(one, 3) == 3 && original_frontier(one, 4) == 4, "below begin");
    require(original_frontier(one, 5) == 5 && original_frontier(one, 6) == 6, "first kept run");
    require(original_frontier(one, 7) == 10, "inside the second kept run");
    require(original_frontier(one, 9) == 20, "past the kept tokens: + dropped");
    require(original_frontier(one, 12) == 23, "the tail after U");

    const PreparedPromptData turn2 = extend(original, 10, 500);
    const auto turn2_rewritten     = apply_gap_map(turn2, *first);
    require(turn2_rewritten.has_value(), "turn 2 rewrites");
    const auto second =
        shared_map(make_gap_map(*turn2_rewritten, 9, std::vector<std::uint32_t>{0, 8, 9}));
    const GapMapSet both{first, second};
    require(original_frontier(both, 15) == 33, "the end of turn 3 maps to its original length");
    require(original_frontier(both, 10) == 21, "the first kept token of the second map");
    require(original_frontier(both, 9) == 20, "the second map's begin through the first map");
}

} // namespace

int main() {
    try {
        test_subsets_tokens_and_keeps_positions();
        test_tail_after_u_is_unchanged();
        test_rewrite_checkpoint_mapped();
        test_rewrite_checkpoint_in_dropped_span_is_dropped();
        test_result_identity_differs_from_original();
        test_follow_up_prompt_is_rewritten();
        test_partial_match_is_not_rewritten();
        test_follow_up_with_extra_rewrite_frontier_is_rewritten();
        test_digest_collision_is_not_enough();
        test_two_maps_apply_in_order();
        test_rejects_invalid_input();
        test_apply_returns_nullopt_where_compaction_would_throw();
        test_rewrite_applies_lineage_in_any_order();
        test_rewrite_prefers_the_deepest_map();
        test_rewrite_applies_each_map_once();
        test_original_frontier_maps_through_the_lineage();
        std::cout << "lossy gap map tests passed\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}
