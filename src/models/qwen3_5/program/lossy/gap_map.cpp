#include "models/qwen3_5/program/lossy/gap_map.h"

#include <algorithm>
#include <bit>
#include <iterator>
#include <limits>
#include <optional>
#include <span>
#include <stdexcept>
#include <vector>

namespace ninfer::models::qwen3_5::detail {

namespace {

void check_shape(const PreparedPromptData& prompt) {
    const std::size_t tokens = prompt.token_ids.size();
    if (prompt.token_types.size() != tokens || prompt.positions.size() != 3U * tokens) {
        throw std::invalid_argument("gap map prompt has an invalid shape");
    }
    if (tokens > std::numeric_limits<std::uint32_t>::max()) {
        throw std::invalid_argument("gap map prompt exceeds uint32 tokens");
    }
}

void check_map(const GapMap& map) {
    if (map.begin > map.original_end) { throw std::invalid_argument("gap map span is inverted"); }
    const std::uint32_t span = map.original_end - map.begin;
    for (std::size_t i = 0; i < map.kept_runs.size(); ++i) {
        const auto [offset, length] = map.kept_runs[i];
        if (length == 0 || offset > span || length > span - offset) {
            throw std::invalid_argument("gap map kept run is outside U");
        }
        // Runs are ascending, disjoint and merged (a gap between neighbours), so the list is
        // canonical.
        if (i > 0 && offset <= map.kept_runs[i - 1].first + map.kept_runs[i - 1].second) {
            throw std::invalid_argument("gap map kept runs are not ascending separated runs");
        }
    }
}

// Translates original token frontiers (token counts) to compacted ones.
class FrontierMap {
public:
    explicit FrontierMap(const GapMap& map) : map_(map), dropped_(map.dropped_tokens()) {
        run_starts_.reserve(map.kept_runs.size());
        std::uint32_t kept = 0;
        for (const auto& [offset, length] : map.kept_runs) {
            run_starts_.push_back(kept);
            kept += length;
        }
    }

    // A frontier inside (begin, original_end] survives only if the token just before it was kept;
    // otherwise it would share its compacted frontier with an earlier original one.
    [[nodiscard]] std::optional<std::uint32_t> operator()(std::uint32_t frontier) const {
        if (frontier <= map_.begin) { return frontier; }
        if (frontier > map_.original_end) { return frontier - dropped_; }
        const std::uint32_t offset = frontier - 1U - map_.begin; // offset in U of the last token
        const auto run             = std::upper_bound(
            map_.kept_runs.begin(), map_.kept_runs.end(), offset,
            [](std::uint32_t value, const auto& kept_run) { return value < kept_run.first; });
        if (run == map_.kept_runs.begin()) { return std::nullopt; }
        const auto index = static_cast<std::size_t>(std::distance(map_.kept_runs.begin(), run) - 1);
        const auto& [run_offset, run_length] = map_.kept_runs[index];
        if (offset >= run_offset + run_length) { return std::nullopt; }
        return map_.begin + run_starts_[index] + (offset - run_offset) + 1U;
    }

private:
    const GapMap& map_;
    std::uint32_t dropped_ = 0;
    std::vector<std::uint32_t> run_starts_; // kept tokens before each run
};

template <typename T>
std::vector<T> gather(std::span<const T> values, std::span<const std::uint32_t> source) {
    std::vector<T> out;
    out.reserve(source.size());
    for (const std::uint32_t index : source) { out.push_back(values[index]); }
    return out;
}

constexpr std::array<std::uint64_t, 2> kDigestOffset{1469598103934665603ULL, 7809847782465536322ULL};
constexpr std::array<std::uint64_t, 2> kDigestPrime{1099511628211ULL, 14029467366897019727ULL};

void mix(std::array<std::uint64_t, 2>& digest, std::uint64_t value) noexcept {
    for (std::size_t lane = 0; lane < digest.size(); ++lane) {
        const std::uint64_t lane_value =
            lane == 0 ? value : std::rotl(value ^ 0x9e3779b97f4a7c15ULL, 29);
        for (std::uint32_t byte = 0; byte < 8; ++byte) {
            digest[lane] ^= static_cast<std::uint8_t>(lane_value >> (8U * byte));
            digest[lane] *= kDigestPrime[lane];
        }
    }
}

std::array<std::int32_t, 3> positions_at(const PreparedPromptData& prompt, std::size_t index) {
    const std::size_t tokens = prompt.token_ids.size();
    return {prompt.positions[index], prompt.positions[tokens + index],
            prompt.positions[2U * tokens + index]};
}

bool continues(const PositionRun& run, std::uint32_t index,
               const std::array<std::int32_t, 3>& positions) {
    const std::int64_t step = static_cast<std::int64_t>(index) - run.index;
    for (std::size_t axis = 0; axis < positions.size(); ++axis) {
        if (static_cast<std::int64_t>(run.first[axis]) + step != positions[axis]) { return false; }
    }
    return true;
}

std::vector<PositionRun> position_runs(const PreparedPromptData& prompt, std::uint32_t end) {
    std::vector<PositionRun> runs;
    for (std::uint32_t i = 0; i < end; ++i) {
        const auto positions = positions_at(prompt, i);
        if (runs.empty() || !continues(runs.back(), i, positions)) {
            runs.push_back(PositionRun{i, positions});
        }
    }
    return runs;
}

// Exact comparison of the prompt's ids, types and positions over [0, original_end) with the
// original stored in the map.
bool matches_original(const PreparedPromptData& prompt, const GapMap& map) {
    const std::uint32_t end = map.original_end;
    if (map.original_ids.size() != end || map.original_types.size() != end ||
        (end > 0 && (map.original_positions.empty() || map.original_positions.front().index != 0))) {
        throw std::invalid_argument("gap map stored original does not cover its span");
    }
    if (!std::equal(map.original_ids.begin(), map.original_ids.end(), prompt.token_ids.begin()) ||
        !std::equal(map.original_types.begin(), map.original_types.end(),
                    prompt.token_types.begin())) {
        return false;
    }
    std::size_t run = 0;
    for (std::uint32_t i = 0; i < end; ++i) {
        while (run + 1 < map.original_positions.size() && map.original_positions[run + 1].index <= i) {
            ++run;
        }
        if (!continues(map.original_positions[run], i, positions_at(prompt, i))) { return false; }
    }
    return true;
}

} // namespace

std::array<std::uint64_t, 2> gap_map_digest(const PreparedPromptData& prompt, std::uint32_t end) {
    check_shape(prompt);
    if (end > prompt.token_ids.size()) {
        throw std::invalid_argument("gap map digest frontier exceeds the prompt");
    }
    std::array<std::uint64_t, 2> digest = kDigestOffset;
    mix(digest, end);
    for (std::uint32_t i = 0; i < end; ++i) {
        mix(digest, static_cast<std::uint32_t>(prompt.token_ids[i]));
        mix(digest, prompt.token_types[i]);
        for (const std::int32_t position : positions_at(prompt, i)) {
            mix(digest, static_cast<std::uint32_t>(position));
        }
    }
    return digest;
}

std::uint32_t GapMap::kept_tokens() const noexcept {
    std::uint32_t kept = 0;
    for (const auto& run : kept_runs) { kept += run.second; }
    return kept;
}

GapMap make_gap_map(const PreparedPromptData& original, std::uint32_t begin,
                    std::span<const std::uint32_t> kept_offsets) {
    check_shape(original);
    const auto tokens = static_cast<std::uint32_t>(original.token_ids.size());
    if (begin > tokens) { throw std::invalid_argument("gap map begins past the prompt"); }
    if (!original.vision_items.empty()) {
        throw std::invalid_argument("gap maps cover text-only prompts");
    }
    GapMap map;
    map.begin        = begin;
    map.original_end = tokens;
    const std::uint32_t span = tokens - begin;
    for (std::size_t i = 0; i < kept_offsets.size(); ++i) {
        const std::uint32_t offset = kept_offsets[i];
        if (offset >= span || (i > 0 && offset <= kept_offsets[i - 1])) {
            throw std::invalid_argument("kept offsets must be ascending, unique and inside U");
        }
        if (!map.kept_runs.empty() &&
            map.kept_runs.back().first + map.kept_runs.back().second == offset) {
            ++map.kept_runs.back().second;
        } else {
            map.kept_runs.emplace_back(offset, 1U);
        }
    }
    map.digest         = gap_map_digest(original, map.original_end);
    map.original_ids   = original.token_ids;
    map.original_types = original.token_types;
    map.original_positions = position_runs(original, map.original_end);
    return map;
}

PreparedPromptData compact_prompt(const PreparedPromptData& original, const GapMap& map) {
    check_shape(original);
    check_map(map);
    const auto tokens = static_cast<std::uint32_t>(original.token_ids.size());
    if (tokens < map.original_end) {
        throw std::invalid_argument("prompt is shorter than the gap map's span");
    }
    if (original.external_prefix_tokens > map.begin) {
        throw std::invalid_argument("external prefix extends into the gap map's span");
    }
    const std::uint32_t dropped = map.dropped_tokens();
    const std::int64_t rope_delta =
        static_cast<std::int64_t>(original.rope_delta) + static_cast<std::int64_t>(dropped);
    if (rope_delta > std::numeric_limits<std::int32_t>::max()) {
        throw std::overflow_error("compacted rope_delta exceeds int32");
    }

    std::vector<std::uint32_t> source;
    source.reserve(tokens - dropped);
    for (std::uint32_t i = 0; i < map.begin; ++i) { source.push_back(i); }
    for (const auto& [offset, length] : map.kept_runs) {
        for (std::uint32_t i = 0; i < length; ++i) { source.push_back(map.begin + offset + i); }
    }
    for (std::uint32_t i = map.original_end; i < tokens; ++i) { source.push_back(i); }

    PreparedPromptData out = original; // the copy carries no n-gram index
    out.ngram_index.index.reset();
    out.token_ids   = gather<TokenId>(original.token_ids, source);
    out.token_types = gather<std::uint8_t>(original.token_types, source);
    out.positions.clear();
    out.positions.reserve(3U * source.size());
    for (int axis = 0; axis < 3; ++axis) {
        const std::vector<std::int32_t> values =
            gather<std::int32_t>(original.position_axis(axis), source);
        out.positions.insert(out.positions.end(), values.begin(), values.end());
    }
    out.rope_delta = static_cast<std::int32_t>(rope_delta);

    for (VisionItem& item : out.vision_items) {
        for (TokenSpan& span : item.token_spans) {
            if (span.begin + span.count <= map.begin) { continue; }
            if (span.begin < map.original_end) {
                throw std::invalid_argument("Vision tokens inside the gap map's span");
            }
            span.begin -= dropped;
        }
    }

    const FrontierMap translate(map);
    if (const auto& checkpoint = original.identity.rewrite_checkpoint) {
        const auto frontier = translate(checkpoint->frontier);
        const auto recovery = translate(checkpoint->recovery_frontier);
        if (frontier && recovery) {
            out.identity.rewrite_checkpoint =
                RewriteCheckpointSpec{checkpoint->kind, *frontier, *recovery};
        } else {
            out.identity.rewrite_checkpoint.reset();
        }
    }
    out.identity.rewrite_execution_frontiers.clear();
    for (const std::uint32_t frontier : original.identity.rewrite_execution_frontiers) {
        if (const auto mapped = translate(frontier)) {
            out.identity.rewrite_execution_frontiers.push_back(*mapped);
        }
    }
    out.context_cache.opportunities.clear();
    for (PreparedCacheOpportunity opportunity : original.context_cache.opportunities) {
        if (const auto mapped = translate(opportunity.frontier)) {
            opportunity.frontier = *mapped;
            out.context_cache.opportunities.push_back(opportunity);
        }
    }
    return out;
}

std::optional<PreparedPromptData> apply_gap_map(const PreparedPromptData& prompt,
                                                const GapMap& map) {
    check_shape(prompt);
    if (prompt.token_ids.size() < map.original_end) { return std::nullopt; }
    // The original was text only; Vision content is not part of the stored original.
    for (const VisionItem& item : prompt.vision_items) {
        for (const TokenSpan& span : item.token_spans) {
            if (span.begin < map.original_end) { return std::nullopt; }
        }
    }
    if (gap_map_digest(prompt, map.original_end) != map.digest) { return std::nullopt; }
    if (!matches_original(prompt, map)) { return std::nullopt; }
    // What compact_prompt would reject for this prompt (not for the map) is no rewrite.
    if (prompt.external_prefix_tokens > map.begin ||
        static_cast<std::int64_t>(prompt.rope_delta) + map.dropped_tokens() >
            std::numeric_limits<std::int32_t>::max()) {
        return std::nullopt;
    }
    return compact_prompt(prompt, map);
}

GapMapRewrite rewrite_with_gap_maps(PreparedPromptData prompt,
                                    std::span<const std::shared_ptr<const GapMap>> maps) {
    std::vector<const std::shared_ptr<const GapMap>*> order;
    order.reserve(maps.size());
    for (const auto& map : maps) {
        if (map) { order.push_back(&map); }
    }
    std::stable_sort(order.begin(), order.end(), [](const auto* a, const auto* b) {
        return (*a)->original_end > (*b)->original_end;
    });
    GapMapRewrite out{.prompt = std::move(prompt), .applied = {}};
    std::vector<bool> used(order.size(), false);
    for (bool changed = true; changed;) {
        changed = false;
        for (std::size_t i = 0; i < order.size(); ++i) {
            if (used[i]) { continue; }
            auto rewritten = apply_gap_map(out.prompt, **order[i]);
            if (!rewritten) { continue; }
            used[i]    = true;
            out.prompt = std::move(*rewritten);
            out.applied.push_back(*order[i]);
            changed = true;
            break; // the deepest remaining map is tried first on the new prompt
        }
    }
    return out;
}

} // namespace ninfer::models::qwen3_5::detail
