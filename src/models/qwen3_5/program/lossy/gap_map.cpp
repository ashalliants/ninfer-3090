#include "models/qwen3_5/program/lossy/gap_map.h"

#include "models/qwen3_5/program/prefix_identity.h"

#include <algorithm>
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

} // namespace

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
    PrefixShortlistDigests digests;
    digests.assign(original);
    map.digest = digests.at(map.original_end);
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
    PrefixShortlistDigests digests;
    digests.assign(prompt);
    if (digests.at(map.original_end) != map.digest) { return std::nullopt; }
    return compact_prompt(prompt, map);
}

} // namespace ninfer::models::qwen3_5::detail
