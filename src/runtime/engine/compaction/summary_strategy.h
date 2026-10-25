#pragma once

// The summary strategy of lossy prefill (`--lossy-prefill-strategy summary`,
// docs/maintainer/lossy-long-prefill.md, "Alternative strategy: deterministic conversation
// summary"). For a cold long prompt, Engine::prepare replaces the middle of the conversation with
// one deterministic summary message (vcc_projector.h) and prepares that as an ordinary exact
// prompt; SummaryRegistry makes later turns render the same summary, so they hit the cache.
//
// prepare_for_strategy is the whole decision, written against a small set of operations so a test
// can drive it with a fake frontend; Engine::prepare supplies the real ones.

#include "ninfer/types.h"
#include "runtime/engine/compaction/summary_registry.h"
#include "runtime/engine/compaction/summary_rewrite.h"
#include "runtime/engine/compaction/summary_split.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace ninfer::runtime::compaction {

// The projector knows Hermes-style tool names only, case-sensitively. A copy of the messages with
// our clients' tool calls renamed to them, so their commands and files reach "Commands Run" and
// "Relevant Files" (the projector itself stays the byte-exact port):
//
//   client tool name                         projector name   argument keys
//   Bash, BashOutput, exec_command           terminal         command / cmd as sent
//   Read, read                               read_file        file_path (filePath -> file_path)
//   Write, write                             write_file       file_path (filePath -> file_path)
//   Edit, MultiEdit, edit                    edit_file        file_path (filePath -> file_path)
//   NotebookEdit                             edit_file        notebook_path -> path
//   Grep, Glob, grep, glob                   search_files     pattern, path as sent
//   apply_patch                              patch            as sent
//
// A key is renamed only when the target key is absent; argument order is kept. Any other name,
// and arguments that are not a JSON object, pass through unchanged.
[[nodiscard]] std::vector<ChatMessage> normalize_tool_calls(std::span<const ChatMessage> messages);

// The projection tells the model to call session_search, which our clients do not have. Replaces
// those instructions with plain statements that the originals were omitted. Deterministic.
[[nodiscard]] std::string strip_session_search(std::string projection);

// The summary's length budget in projector characters: min(0.06 x max_context, max_tokens)
// tokens, at least 1,500, at kSummaryCharsPerToken characters per token.
inline constexpr std::size_t kSummaryCharsPerToken   = 4;
inline constexpr std::uint32_t kSummaryMinimumTokens = 1500;
[[nodiscard]] std::size_t summary_budget_chars(std::uint32_t max_context,
                                               std::uint32_t summary_max_tokens) noexcept;

// The summary message text for messages [first, last): the header, then the projection of the
// normalized messages with the session_search instructions replaced.
[[nodiscard]] std::string build_summary_text(std::span<const ChatMessage> messages, std::size_t first,
                                             std::size_t last, std::size_t budget_chars);

// Leading and trailing tokens two prompts share, at most min(size) in total.
[[nodiscard]] std::pair<std::uint32_t, std::uint32_t>
common_prefix_suffix(std::span<const TokenId> a, std::span<const TokenId> b) noexcept;

struct SummaryStrategyConfig {
    // LossyPrefillOptions::min_tokens: the part after the product prompt must exceed it.
    std::uint32_t min_tokens = 0;
    std::size_t budget_chars = 0;
    std::size_t protect_first  = 3;
    std::size_t protect_last   = 20;
    std::size_t min_tail_users = 1;
};

[[nodiscard]] inline bool summary_strategy_active(const LossyPrefillOptions& options) noexcept {
    return options.enabled() && options.strategy == LossyStrategy::Summary;
}

template <class Prepared>
struct SummaryPreparation {
    Prepared prepared;
    std::optional<SummaryRewrite> rewrite;
    // The coldness probe's answer; unset when the prompt was not long enough to be probed.
    std::optional<LossySummaryProbe> probe;
};

// At most this many recorded summaries are applied in a row (each later one was recorded on a
// conversation that already held the earlier ones).
inline constexpr int kMaximumSummaryChain = 8;

// The summary strategy for one prompt. `ops` provides:
//   Prepared prepare(PromptInput)                      the Frontend's prepare
//   std::uint32_t tokens(const Prepared&)              its prompt tokens
//   std::uint32_t exact_end(const Prepared&)           the product prompt's end (shared marker)
//   bool eligible(const Prepared&)                     text only, no installed graft
//   LossySummaryProbe probe_reuse(const Prepared&)     Cold unless a private record would serve it
//                                                      (Warm) or the probe timed out (Timeout)
//   std::span<const TokenId> token_ids(const Prepared&)
//
// 1. Recorded summaries the messages start with are substituted, oldest first.
// 2. The result is prepared. A cold, eligible prompt with more than min_tokens after the product
//    prompt is split (split_for_summary, told which messages are substituted summaries); a
//    non-empty middle is replaced by its summary and prepared again, kept only when it is
//    shorter, and recorded.
// 3. When anything was replaced, the client's original prompt is prepared too (once), so usage
//    reports its length.
template <class Ops>
[[nodiscard]] auto prepare_summarized(PromptInput input, const SummaryStrategyConfig& config,
                                      SummaryRegistry& registry, Ops& ops)
    -> SummaryPreparation<decltype(ops.prepare(std::move(input)))> {
    using Prepared = decltype(ops.prepare(std::move(input)));
    // The client's index of each current message (a summary message: the first it replaced), and
    // which current messages are summaries this function substituted.
    std::vector<std::size_t> origin(input.messages.size());
    std::iota(origin.begin(), origin.end(), std::size_t{0});
    std::vector<std::uint8_t> substituted(input.messages.size(), 0);
    const auto client_range = [&](std::size_t first, std::size_t last) {
        return std::array<std::uint32_t, 2>{static_cast<std::uint32_t>(origin[first]),
                                            static_cast<std::uint32_t>(origin[last - 1] + 1)};
    };
    const auto replace_origin = [&](std::size_t first, std::size_t last) {
        origin.erase(origin.begin() + static_cast<std::ptrdiff_t>(first + 1),
                     origin.begin() + static_cast<std::ptrdiff_t>(last));
        substituted.erase(substituted.begin() + static_cast<std::ptrdiff_t>(first + 1),
                          substituted.begin() + static_cast<std::ptrdiff_t>(last));
        substituted[first] = 1;
    };

    std::optional<PromptInput> original;
    PromptInput current = std::move(input);
    std::optional<std::array<std::uint32_t, 2>> range;
    for (int chain = 0; chain < kMaximumSummaryChain; ++chain) {
        auto hit = registry.find(current.messages);
        if (!hit || hit->first >= hit->last || hit->last > current.messages.size()) { break; }
        if (!original) { original = current; }
        range = client_range(hit->first, hit->last);
        replace_origin(hit->first, hit->last);
        current = substitute_summary(std::move(current), hit->first, hit->last,
                                     std::move(hit->summary));
    }

    Prepared prepared = ops.prepare(current);
    std::optional<Prepared> original_prepared;
    std::optional<LossySummaryProbe> probe;
    bool fresh = false;
    const std::uint32_t tokens    = ops.tokens(prepared);
    const std::uint32_t exact_end = ops.exact_end(prepared);
    if (config.min_tokens > 0 && ops.eligible(prepared) && tokens > exact_end &&
        tokens - exact_end > config.min_tokens) {
        probe = ops.probe_reuse(prepared);
    }
    if (probe == LossySummaryProbe::Cold) {
        const SummarySplit split =
            split_for_summary(current.messages, config.protect_first, config.protect_last,
                              config.min_tail_users, substituted);
        if (!split.empty()) {
            std::string text =
                build_summary_text(current.messages, split.first, split.last, config.budget_chars);
            PromptInput candidate = substitute_summary(current, split.first, split.last, text);
            Prepared next         = ops.prepare(candidate);
            if (ops.tokens(next) < tokens) {
                registry.record(std::span<const ChatMessage>(current.messages).first(split.last),
                                split.first, std::move(text));
                if (original) {
                    original_prepared.emplace(ops.prepare(std::move(*original)));
                } else {
                    original_prepared.emplace(std::move(prepared));
                }
                range = client_range(split.first, split.last);
                prepared = std::move(next);
                fresh    = true;
            }
        }
    }
    if (!range) { return {std::move(prepared), std::nullopt, probe}; }
    if (!original_prepared) { original_prepared.emplace(ops.prepare(std::move(*original))); }

    SummaryRewrite rewrite;
    rewrite.original_tokens = ops.tokens(*original_prepared);
    rewrite.prompt_tokens   = ops.tokens(prepared);
    const auto [prefix, suffix] =
        common_prefix_suffix(ops.token_ids(*original_prepared), ops.token_ids(prepared));
    rewrite.common_prefix = prefix;
    rewrite.common_suffix = suffix;
    rewrite.messages      = *range;
    rewrite.fresh         = fresh;
    return {std::move(prepared), rewrite, probe};
}

// Engine::prepare's entry point: the summary strategy when it is on, otherwise one ordinary
// prepare (the select strategy, or lossy prefill off): no probe, no registry.
template <class Ops>
[[nodiscard]] auto prepare_for_strategy(PromptInput input, const LossyPrefillOptions& options,
                                        const SummaryStrategyConfig& config,
                                        SummaryRegistry& registry, Ops& ops)
    -> SummaryPreparation<decltype(ops.prepare(std::move(input)))> {
    if (!summary_strategy_active(options)) {
        return {ops.prepare(std::move(input)), std::nullopt, std::nullopt};
    }
    return prepare_summarized(std::move(input), config, registry, ops);
}

} // namespace ninfer::runtime::compaction
