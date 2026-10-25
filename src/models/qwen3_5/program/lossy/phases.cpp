// The lossy-prefill phases of a Program lane (docs/maintainer/lossy-long-prefill.md, "How it
// works"; lossy_flow.h): enable_lossy, the Scout steps of advance_prefill, and the selection and
// compaction that end the scout.
#include "models/qwen3_5/program/program_impl.h"
#include "models/qwen3_5/program/context_work.h"
#include "models/qwen3_5/program/lossy/selection.h"
#include "models/qwen3_5/program/text_positions.h"

#include "core/device.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <memory>
#include <span>
#include <stdexcept>
#include <utility>
#include <variant>
#include <vector>

namespace ninfer::models::qwen3_5::detail {

bool ProgramImpl::enable_lossy(SequenceHandle handle) {
    if (!valid_sequence(handle) || !scout_buffers || !lossy_options.enabled()) { return false; }
    if (parameters.text.split_execution() ||
        std::holds_alternative<MoeConfig>(parameters.model.config().text.ffn)) {
        return false;
    }
    const auto lane = ContractAccess::lane(handle).value;
    auto& request   = requests[lane];
    if (!request.base || !request.base->lossy || request.lossy ||
        request.lifecycle != Lifecycle::Prefilling || !request.prefill || request.recovery ||
        request.capture_pending) {
        return false;
    }
    const auto& staged = *request.prefill;
    if (staged.vision || staged.prompt.has_media() || staged.cursor != staged.base ||
        staged.slice.index != 0) {
        return false;
    }
    const std::uint32_t exact_end = lossy_exact_end(request.base->lossy->exact_end, staged.base);
    if (exact_end >= staged.prompt_tokens) { return false; }
    request.lossy = RequestControl::Lossy{.phase = LossyPhase::Exact, .exact_end = exact_end};
    return true;
}

bool ProgramImpl::abort_lossy_scout(SequenceHandle handle) {
    if (!valid_sequence(handle)) { return false; }
    const auto lane = ContractAccess::lane(handle).value;
    auto& request   = requests[lane];
    if (!request.lossy || request.lossy->phase != LossyPhase::Scout || !scout_session ||
        scout_session->lane != lane || !request.prefill) {
        return false;
    }
    // Every scout step synchronized and settled its unit, so nothing still writes the scout's
    // pages. end_scout puts back the GDN states and the growth reservation and releases every page
    // past M; the lane is then the Prefilling lane at M the scout started from.
    end_scout(handle);
    // Cannot happen: a scout begins only with the cursor at M (advance_lossy_scout checks it) and
    // its steps never move the cursor. A lane found elsewhere is corrupt, and the throw sends the
    // Engine to its worker recovery (clear every lane, verify the baseline), which is the right
    // answer to that, like end_scout's own mapping check.
    if (request.prefill->cursor != request.lossy->exact_end) {
        throw std::logic_error("an aborted lossy scout left its lane away from M");
    }
    // Main without a selection: the ordinary prefill of all of U. Captures past M apply again.
    request.lossy->phase = LossyPhase::Main;
    return true;
}

double ProgramImpl::lossy_scout_share() const noexcept {
    if (!scout_buffers) { return 0.0; }
    const auto layers = parameters.model.config().text.layer_types.size();
    if (layers == 0) { return 0.0; }
    return std::min(1.0, static_cast<double>(scout_buffers->filter_layer + 1U) /
                             static_cast<double>(layers));
}

bool ProgramImpl::lane_refuses_pause(std::uint32_t lane) const noexcept {
    const auto& lossy = requests[lane].lossy;
    return lossy && lossy_refuses_pause(lossy->phase);
}

bool ProgramImpl::lane_holds_captures(std::uint32_t lane) const noexcept {
    const auto& request = requests[lane];
    return request.lossy && request.prefill &&
           lossy_holds_captures(request.lossy->phase, request.prefill->cursor,
                                request.lossy->exact_end);
}

PrefillProgress ProgramImpl::advance_lossy_scout(std::uint32_t lane,
                                                 runtime::ExecutionTiming* failed_timing,
                                                 std::uint32_t max_tokens) {
    runtime::ExecutionTimingRecorder timing(runtime::ExecutionTimingPhase::Submit, failed_timing);
    auto& request = requests[lane];
    if (request.lifecycle != Lifecycle::Prefilling || !request.prefill || !request.lossy) {
        throw std::logic_error("lossy scout step requires a prefilling lossy request");
    }
    if (request.capture_pending) {
        throw std::logic_error("prefill cannot advance while a capture is pending");
    }
    auto& staged       = *request.prefill;
    auto& lossy        = *request.lossy;
    const auto handle  = sequence_handle(lane);
    const auto started = Clock::now();
    try {
        if (lossy.phase == LossyPhase::Exact) {
            if (staged.cursor != lossy.exact_end) {
                throw std::logic_error("lossy scout must begin at M");
            }
            // The scout is new work past the reuse frontier. A reservation made for capture
            // groups past M describes the uncompacted prompt and is never used.
            request.recovery.reset();
            request.capture_reservation.reset();
            begin_scout(handle);
            lossy.phase = LossyPhase::Scout;
        }
        const std::uint32_t remaining = scout_session->prompt_tokens - scout_session->cursor;
        timing.pause();
        const ScoutStep step = advance_scout(handle, max_tokens);
        timing.include(step.timing);
        timing.resume_post();
        if (!step.reserved) {
            throw std::logic_error("lossy scout chunk exceeds its reserved prefill unit");
        }
        const double seconds = std::chrono::duration<double>(Clock::now() - started).count();
        lossy.scout_seconds += seconds;
        staged.elapsed_seconds += seconds;

        PrefillProgress progress;
        progress.scouted_tokens = step.processed_tokens;
        progress.cap_applied =
            max_tokens != 0 && max_tokens < std::min(prefill_chunk, remaining) &&
            step.processed_tokens == max_tokens;
        if (step.complete) { finish_lossy_scout(lane, progress); }
        progress.summary = runtime::BeginSummary{.prompt_tokens        = staged.prompt_tokens,
                                                 .reused_prompt_tokens = staged.base,
                                                 .prefix_reuse_path    = staged.reuse};
        progress.timing  = timing.finish();
        return progress;
    } catch (...) {
        timing.begin_wait();
        try {
            device.synchronize();
        } catch (...) {}
        timing.end_wait();
        clear_execution_failure_lanes(std::span<const std::uint32_t>(&lane, 1));
        throw;
    }
}

void ProgramImpl::finish_lossy_scout(std::uint32_t lane, PrefillProgress& progress) {
    auto& request              = requests[lane];
    auto& staged               = *request.prefill;
    auto& lossy                = *request.lossy;
    const std::uint32_t begin  = lossy.exact_end;
    const std::uint32_t span   = staged.prompt_tokens - begin;
    const std::vector<float> scores = scout_scores();
    // Restores the state the main pass reads and releases every page past M.
    end_scout(sequence_handle(lane));
    if (scores.size() != span) { throw std::logic_error("scout scored the wrong span"); }

    const auto u = std::span<const TokenId>(staged.prompt.token_ids).subspan(begin, span);
    const std::vector<std::uint8_t> forced = lossy_forced_tokens(u, lossy_special_tokens);
    const SelectionResult selection = select_tokens({.scores = scores, .forced = forced},
                                                    lossy_options);
    LossyOutcome outcome{.exact_end       = begin,
                         .uncached_tokens = span,
                         .kept_tokens     = static_cast<std::uint32_t>(selection.kept.size()),
                         .skipped         = selection.skipped || selection.kept.size() == span,
                         .scout_seconds   = lossy.scout_seconds};
    if (outcome.skipped) {
        // Nothing is dropped: Main is the ordinary prefill of all of U, with no gap map.
        outcome.kept_tokens = span;
    } else {
        auto map = std::make_shared<const GapMap>(make_gap_map(staged.prompt, begin, selection.kept));
        install_compacted_prompt(lane, compact_prompt(staged.prompt, *map), std::move(map));
        progress.compacted_plan =
            RequestBasePlan(std::const_pointer_cast<RequestBasePlanImpl>(request.base));
    }
    lossy.phase                      = LossyPhase::Main;
    progress.compacted_prompt_tokens = staged.prompt_tokens;
    progress.lossy                   = outcome;
}

void ProgramImpl::install_compacted_prompt(std::uint32_t lane, PreparedPromptData&& compacted,
                                           std::shared_ptr<const GapMap> map) {
    auto& request  = requests[lane];
    auto& staged   = *request.prefill;
    auto& sequence = active_sequence(lane);
    // Task 4: text prefill derives RoPE from sequence.rope_delta; the staged prompt's must agree.
    if (sequence.rope_delta != staged.prompt.rope_delta ||
        sequence.ledger.size() != staged.prompt_tokens || staged.cursor != map->begin ||
        sequence.text_kv_valid != staged.cursor ||
        text_kv_addresses->committed_frontier(sequence.kv->text) != staged.cursor) {
        throw std::logic_error("lossy compaction does not match the lane at M");
    }
    const RequestBasePlanImpl& original = *request.base;
    auto rebased                        = std::make_shared<RequestBasePlanImpl>();
    rebased->summary                    = original.summary;
    rebased->summary.prompt_tokens      = static_cast<std::uint32_t>(compacted.token_ids.size());
    rebased->context_cache              = compacted.context_cache;
    rebased->sampling                   = original.sampling;
    rebased->allow_prefix_reuse         = original.allow_prefix_reuse;
    rebased->gap_maps                   = original.gap_maps;
    rebased->gap_maps.push_back(map);
    finish_request_plan(*rebased, std::move(compacted));
    const PreparedPromptData& prompt = *rebased->prompt;

    // The sequence's identity is the kept token list; the original prompt reaches it through the
    // gap map (plan_request).
    sequence.ledger = prompt.token_ids;
    sequence.prefix_identity.assign(prompt);
    sequence.prefix_digests = rebased->prefix_digests;
    sequence.rope_delta     = prompt.rope_delta;
    sequence.gap_maps.push_back(map);
    staged.prompt        = prompt;
    staged.prompt_tokens = rebased->summary.prompt_tokens;
    const std::span<const std::int32_t> axis0(staged.prompt.positions.data(),
                                              staged.prompt.token_ids.size());
    staged.regular_text_positions =
        regular_text_positions(axis0, staged.cursor, staged.prompt.rope_delta);
    request.base = std::move(rebased);
    request.capture_reservation.reset();
    initialize_captures(lane, staged.cursor, staged.prompt_tokens);
    gap_map_registry.push_back(map);
}

} // namespace ninfer::models::qwen3_5::detail
