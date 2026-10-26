#include "core/device.h"
#include "models/qwen3_5/execution/parameters.h"
#include "models/qwen3_5/frontend/frontend.h"
#include "models/qwen3_5/frontend/prepared_prompt.h"
#include "models/qwen3_5/load.h"
#include "models/qwen3_5/program/program.h"
#include "models/qwen3_5/program/context_work.h"
#include "models/qwen3_5/program/lossy/gap_map.h"
#include "models/qwen3_5/program/lossy/selection.h"
#include "models/qwen3_5/program/program_impl.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstddef>
#include <cstdlib>
#include <functional>
#include <iterator>
#include <memory>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

using namespace ninfer;
namespace qwen = models::qwen3_5;

constexpr std::uint32_t kCapacity      = 512;
constexpr std::uint32_t kChunk         = 128;
constexpr std::uint32_t kPromptTokens  = 384;
constexpr std::uint32_t kPauseFrontier = 256;

void require(bool condition, std::string_view message) {
    if (!condition) { throw std::runtime_error(std::string(message)); }
}

bool transferred(const qwen::ContextProgress& progress, runtime::ContextTransferDirection direction,
                 std::optional<runtime::ContextResourceClass> resource = std::nullopt) {
    return std::any_of(progress.transfers.begin(), progress.transfers.end(),
                       [&](const auto& transfer) {
                           return transfer.direction == direction &&
                                  (!resource || transfer.resource == *resource) &&
                                  transfer.work.payload_bytes != 0;
                       });
}

std::uint32_t transferred_pages(const qwen::ContextProgress& progress,
                                runtime::ContextTransferDirection direction,
                                runtime::ContextResourceClass resource) {
    std::uint32_t pages = 0;
    for (const auto& transfer : progress.transfers) {
        if (transfer.resource == resource && transfer.direction == direction) {
            pages += transfer.page_count;
        }
    }
    return pages;
}

const qwen::ContextDemotion*
demotion_with_sources(std::span<const qwen::ContextDemotion> candidates,
                      std::span<const qwen::CheckpointHandle> sources) {
    const auto selected =
        std::find_if(candidates.begin(), candidates.end(), [&](const auto& quote) {
            return quote.sources.size() == sources.size() &&
                   std::all_of(sources.begin(), sources.end(), [&](auto source) {
                       return std::find(quote.sources.begin(), quote.sources.end(), source) !=
                              quote.sources.end();
                   });
        });
    return selected == candidates.end() ? nullptr : &*selected;
}

qwen::ContextDemotionBatch demotion_batch(const qwen::ContextReclaimPlan& plan,
                                          std::span<const qwen::ContextDemotion> ordered,
                                          runtime::ContextResourceUsage shortage) {
    auto batch = plan.begin_kv_batch(shortage);
    for (const auto& quote : ordered) {
        if (!batch.append(quote)) { break; }
    }
    return batch;
}

bool batch_covers(const qwen::ContextDemotionBatch& batch,
                  std::span<const qwen::ContextRelease> releases,
                  std::span<const qwen::CheckpointHandle> sources) {
    const auto selected = std::find_if(releases.begin(), releases.end(), [&](const auto& quote) {
        return quote.sources.size() == sources.size() &&
               std::all_of(sources.begin(), sources.end(), [&](auto source) {
                   return std::find(quote.sources.begin(), quote.sources.end(), source) !=
                          quote.sources.end();
               });
    });
    return selected != releases.end() && batch.covers(*selected);
}

class Fixture {
public:
    Fixture(DeviceContext& device, qwen::Program& program, qwen::Frontend& frontend)
        : device_(device), program_(program), frontend_(frontend),
          empty_(program.physical_usage()) {}

    qwen::RequestBasePlan request(std::uint32_t tokens, bool history = false,
                                  bool allow_prefix_identity = true) {
        std::vector<TokenId> ids(tokens, 198);
        ids.front() = 1000;
        runtime::ResolvedExecutionOptions options;
        options.requested_output_tokens    = 8;
        options.allow_prefix_reuse         = history;
        options.sampling.presence_penalty  = 0.5F;
        options.sampling.frequency_penalty = 0.25F;
        return program_.plan_request(
            frontend_.prepare_tokens(std::move(ids), allow_prefix_identity), options);
    }

    void synchronize() {
        device_.synchronize();
        CUDA_CHECK(cudaStreamSynchronize(device_.transfer_stream));
    }

    qwen::ContextProgress settle(bool cancelled = false) {
        std::atomic<bool> flag{cancelled};
        for (unsigned step = 0; step < 8; ++step) {
            // Wait on actual submitted streams; do not guess whether DMA has completed from
            // elapsed time or delay cancellation until a convenient scheduling outcome.
            synchronize();
            auto progress = program_.poll_context({&flag});
            if (progress.complete) {
                require(!program_.has_context_transaction(), "completed context remains pending");
                return progress;
            }
        }
        throw std::runtime_error("bounded native context transaction did not complete");
    }

    qwen::SourceCandidate root(const qwen::RequestBasePlan& base) {
        auto source = program_.inspect_source(base, std::nullopt);
        require(source.has_value(), "fresh request has no root source");
        return std::move(*source);
    }

    qwen::SequenceHandle
    bind(const qwen::RequestBasePlan& base, std::uint32_t lane = 0,
         qwen::ResumeState* resume           = nullptr,
         qwen::ExecutionUnitKind resume_kind = qwen::ExecutionUnitKind::Decode) {
        const bool needs_replay = resume && !resume->has_snapshot();
        require(static_cast<bool>(program_.start_binding(
                    base, {lane}, root(base), resume, resume_kind,
                    resume_kind == qwen::ExecutionUnitKind::Prefill ? 0 : 1)),
                "request binding could not reserve its first unit");
        auto progress = settle();
        require(progress.published && progress.sequence.has_value(),
                "binding did not publish a lane");
        if (needs_replay) {
            require(progress.replaying, "snapshot-free binding did not enter replay");
        }
        return *progress.sequence;
    }

    void reserve(qwen::SequenceHandle sequence, qwen::ExecutionUnitKind kind,
                 std::uint32_t tokens = 0) {
        const std::array<qwen::ExecutionUnit, 1> units{{{sequence, kind, tokens}}};
        require(static_cast<bool>(program_.reserve_units(units)),
                "finite native unit was not reserved");
    }

    void prefill_prefix(qwen::SequenceHandle sequence, std::uint32_t tokens) {
        std::uint32_t processed = 0;
        while (processed < tokens) {
            reserve(sequence, qwen::ExecutionUnitKind::Prefill);
            auto step = program_.advance_prefill(sequence);
            require(!step.complete && !step.pending, "partial prefill sampled before its target");
            require(step.processed_prompt_tokens != 0, "partial prefill made no progress");
            processed += step.processed_prompt_tokens;
            if (step.capture_ready) { program_.skip_capture(sequence); }
        }
        require(processed == tokens, "partial prefill crossed its requested frontier");
    }

    // Main KV restored under a lane's first pass, reported with its prefill progress.
    std::vector<runtime::ContextTransferObservation> landed_;

    [[nodiscard]] bool landed_from_host() const {
        return std::any_of(landed_.begin(), landed_.end(), [](const auto& transfer) {
            return transfer.direction == runtime::ContextTransferDirection::HostToDevice &&
                   transfer.resource == runtime::ContextResourceClass::MainKV &&
                   transfer.work.payload_bytes != 0 && transfer.elapsed_ns != 0;
        });
    }

    void finish_prefill(qwen::SequenceHandle sequence, std::uint32_t remaining, bool terminal) {
        std::uint32_t processed = 0;
        for (unsigned iteration = 0; iteration < remaining / kChunk + 4; ++iteration) {
            const bool recovering = program_.recovery_pending(sequence);
            reserve(sequence, qwen::ExecutionUnitKind::Prefill);
            auto step = program_.advance_prefill(sequence);
            landed_.insert(landed_.end(), step.landed_transfers.begin(),
                           step.landed_transfers.end());
            processed += step.processed_prompt_tokens;
            if (step.capture_ready) { program_.skip_capture(sequence); }
            if (!step.complete) {
                require(step.processed_prompt_tokens != 0 || step.capture_ready,
                        "prefill made no observable progress");
                continue;
            }
            require(step.pending && step.pending->row_count() == 1,
                    "completed prefill did not return one pending row");
            require(!recovering || program_.recovery_pending(sequence),
                    "uncommitted prefill output released recovery protection");
            require(processed == remaining, "resumed prefill repeated or skipped committed input");
            const std::array<runtime::CommitDecision, 1> decision{
                {{.accepted_tokens = 1, .terminal = terminal}}};
            auto result = program_.commit(std::move(*step.pending), decision);
            require(result.row_count == 1 && result.rows[0].disposition ==
                                                 (terminal ? runtime::CommitDisposition::Finishable
                                                           : runtime::CommitDisposition::Active),
                    "prefill commit returned the wrong request phase");
            if (result.capture_ready[0]) { program_.skip_capture(sequence); }
            return;
        }
        throw std::runtime_error("prefill exceeded its bounded chunk count");
    }

    void decode_and_finish(qwen::SequenceHandle sequence) {
        reserve(sequence, qwen::ExecutionUnitKind::Decode, 1);
        const std::array<qwen::SequenceHandle, 1> members{sequence};
        const std::array<runtime::RoundBudget, 1> budgets{{{.generated_tokens_remaining = 1}}};
        auto pending = program_.decode(members, budgets);
        const std::array<runtime::CommitDecision, 1> decision{
            {{.accepted_tokens = 1, .terminal = true}}};
        auto committed = program_.commit(std::move(pending), decision);
        require(committed.rows[0].disposition == runtime::CommitDisposition::Finishable,
                "request could not decode after recovery");
        auto finished = program_.finish(sequence);
        require(finished.status == runtime::ConsumeStatus::Consumed && !finished.checkpoint,
                "history-disabled request did not release its execution resources");
    }

    qwen::CheckpointHandle checkpoint(const qwen::RequestBasePlan& base) {
        const auto sequence = bind(base);
        finish_prefill(sequence, base.summary().prompt_tokens, true);
        auto finished = program_.finish(sequence);
        require(finished.status == runtime::ConsumeStatus::Consumed && finished.checkpoint,
                "seed request did not publish its complete checkpoint");
        return *finished.checkpoint;
    }

    void expect_usage(const qwen::PhysicalUsageSnapshot& expected, std::string_view label) {
        const auto actual = program_.physical_usage();
        require(actual.occupied == expected.occupied &&
                    actual.host_reserved_bytes == expected.host_reserved_bytes &&
                    actual.host_state_slots == expected.host_state_slots &&
                    actual.host_kv_bytes == expected.host_kv_bytes,
                label);
    }

    void expect_empty(std::string_view label) {
        require(!program_.has_context_transaction(), "cleanup left a context operation pending");
        expect_usage(empty_, label);
    }

    void next_request() {
        auto base           = request(kChunk);
        const auto sequence = bind(base);
        finish_prefill(sequence, kChunk, false);
        decode_and_finish(sequence);
        expect_empty("next request did not return typed stores to the initial baseline");
    }

    struct MarkerRequest {
        qwen::RequestBasePlan base;
        std::vector<std::uint32_t> frontiers;
    };

    MarkerRequest marker_request() {
        // Frontiers come from the real artifact tokenizer and public frontend markers.
        PromptInput input;
        ChatMessage system;
        system.role = ChatRole::System;
        std::string instructions;
        for (unsigned word = 0; word < 20; ++word) { instructions += "a "; }
        system.parts.push_back({.kind = MessagePartKind::Text, .text = instructions});
        input.messages.push_back(std::move(system));
        ChatMessage user;
        user.role = ChatRole::User;
        std::string content;
        for (unsigned word = 0; word < 350; ++word) { content += "b "; }
        user.parts.push_back({.kind = MessagePartKind::Text, .text = content});
        input.messages.push_back(std::move(user));
        input.context_cache.allow_engine_automatic_shared_prefixes = false;
        // End before the space: byte-BPE may join that space to the following word.
        for (const auto bytes : {13U, 21U, 29U}) {
            input.context_cache.markers.push_back(
                {.kind                      = PromptCacheMarkerKind::SharedStablePrefix,
                 .evidence                  = SharedCandidateEvidence::ExplicitBoundary,
                 .location                  = PromptCacheMarkerLocation::LeadingInstructionBoundary,
                 .leading_instruction_bytes = bytes});
        }
        auto prompt          = frontend_.prepare(std::move(input));
        const auto& prepared = qwen::PreparedPromptAccess::view(prompt);
        std::vector<std::uint32_t> frontiers;
        for (const auto& candidate : prepared.context_cache.opportunities) {
            frontiers.push_back(candidate.frontier);
        }
        std::sort(frontiers.begin(), frontiers.end());
        require(frontiers.size() == 3 && frontiers.front() != 0 && frontiers.back() < kChunk &&
                    std::adjacent_find(frontiers.begin(), frontiers.end()) == frontiers.end(),
                "marker fixture did not create three distinct boundaries in one chunk");
        require(prepared.token_ids.size() > frontiers.back() + 2 * kChunk,
                "marker fixture has insufficient suffix to exercise ordinary chunks");
        runtime::ResolvedExecutionOptions options;
        options.requested_output_tokens = 8;
        options.allow_prefix_reuse      = true;
        return {program_.plan_request(std::move(prompt), options), std::move(frontiers)};
    }

    qwen::CheckpointHandle capture(qwen::SequenceHandle sequence, std::uint32_t frontier,
                                   runtime::CheckpointRole role) {
        require(program_.capture_is_input(sequence) ==
                    (role == runtime::CheckpointRole::InputReplay),
                "capture offer confused an input recovery point with a shared prefix");
        const auto before = program_.physical_usage();
        require(program_.start_capture(sequence),
                "offered semantic capture has no reserved destination");
        const auto progress = settle();
        require(progress.published && progress.captured_checkpoints.size() == 1,
                "single-role capture did not publish exactly one recovery point");
        const auto checkpoint = progress.captured_checkpoints.front();
        const auto summary    = program_.checkpoint_summary(checkpoint);
        require(summary.frontier == frontier && summary.role == role,
                "capture publication changed its selected frontier or role");
        if (role == runtime::CheckpointRole::InputReplay) {
            const auto after = program_.physical_usage();
            require(progress.operations.partial_tail_cow_pages == 0 &&
                        after.occupied.main_kv_pages == before.occupied.main_kv_pages &&
                        after.occupied.backend_kv_pages == before.occupied.backend_kv_pages &&
                        !transferred(progress, runtime::ContextTransferDirection::DeviceToDevice,
                                     runtime::ContextResourceClass::MainKV) &&
                        !transferred(progress, runtime::ContextTransferDirection::DeviceToDevice,
                                     runtime::ContextResourceClass::BackendKV),
                    "internal input recovery point copied or separately allocated KV history");
        }
        return checkpoint;
    }

    void capture_opportunities() {
        auto marked          = marker_request();
        const auto sequence  = bind(marked.base);
        std::uint32_t cursor = 0;
        std::vector<qwen::CheckpointHandle> shared;
        for (const auto frontier : marked.frontiers) {
            reserve(sequence, qwen::ExecutionUnitKind::Prefill);
            const auto step = program_.advance_prefill(sequence);
            require(step.processed_prompt_tokens == frontier - cursor && step.capture_ready &&
                        !step.complete && !step.pending,
                    "Native skipped an explicitly marked prefix boundary");
            cursor = frontier;
            shared.push_back(capture(sequence, frontier, runtime::CheckpointRole::SharedPrefix));
        }
        reserve(sequence, qwen::ExecutionUnitKind::Prefill);
        const auto ordinary = program_.advance_prefill(sequence);
        require(ordinary.processed_prompt_tokens == kChunk && !ordinary.capture_ready &&
                    !ordinary.complete && !ordinary.pending,
                "ordinary prefill chunk created an unrequested recovery point");
        require(program_.abort(sequence).status == runtime::ConsumeStatus::Consumed,
                "marker fixture did not release its active reader");
        for (const auto point : shared) {
            require(program_.release_checkpoint(point), "explicit prefix retained a reader");
        }
        expect_empty("explicit prefix captures leaked typed resources");

        for (const bool history : {true, false}) {
            auto raw          = request(kPromptTokens, history);
            const auto active = bind(raw);
            for (unsigned chunk = 0; chunk < 2; ++chunk) {
                reserve(active, qwen::ExecutionUnitKind::Prefill);
                const auto step = program_.advance_prefill(active);
                require(step.processed_prompt_tokens == kChunk && !step.capture_ready &&
                            !step.complete && !step.pending,
                        "raw prefill chunk offered an unrequested recovery point");
            }
            require(program_.abort(active).status == runtime::ConsumeStatus::Consumed,
                    "raw chunk fixture did not release its lane");
            expect_empty("ordinary chunks retained optional history");
        }
    }

    void input_replay_lifecycle() {
        // The non-page-aligned P shares its directory and tail with the active continuation.
        constexpr std::uint32_t prompt_tokens = 2 * kChunk + 1;
        auto base                             = request(prompt_tokens, true);
        const auto active                     = bind(base);
        prefill_prefix(active, 2 * kChunk);
        reserve(active, qwen::ExecutionUnitKind::Prefill);
        auto begin = program_.advance_prefill(active);
        require(begin.complete && begin.pending && begin.processed_prompt_tokens == 1,
                "input replay fixture did not reach its full input boundary");
        const std::array<runtime::CommitDecision, 1> accepted{{{.accepted_tokens = 1}}};
        const auto committed = program_.commit(std::move(*begin.pending), accepted);
        require(committed.capture_ready[0], "full raw input did not offer its replay point");
        const auto input = capture(active, prompt_tokens, runtime::CheckpointRole::InputReplay);

        reserve(active, qwen::ExecutionUnitKind::Decode, 1);
        const std::array<qwen::SequenceHandle, 1> members{active};
        const std::array<runtime::RoundBudget, 1> budgets{{{.generated_tokens_remaining = 1}}};
        auto pending = program_.decode(members, budgets);
        const std::array<runtime::CommitDecision, 1> terminal{
            {{.accepted_tokens = 1, .terminal = true}}};
        (void)program_.commit(std::move(pending), terminal);
        const auto finished = program_.finish(active);
        require(finished.checkpoint && program_.valid_checkpoint(input),
                "finishing E invalidated its retained input recovery point");
        require(program_.release_checkpoint(*finished.checkpoint),
                "input replay fixture could not retire the replaced endpoint");

        const std::array<qwen::CheckpointHandle, 1> points{input};
        const auto candidate = program_.inspect_source(base, input, true, points);
        require(candidate && candidate->reused_tokens == prompt_tokens && candidate->move_history &&
                    !candidate->move_state && candidate->private_points.size() == 1,
                "repeated input did not retain P while adopting its private KV history");
        require(static_cast<bool>(program_.start_binding(base, {1}, *candidate)),
                "input replay could not reserve its writer");
        const auto bound = settle();
        require(bound.sequence &&
                    bound.private_points == std::vector(points.begin(), points.end()) &&
                    bound.retired_checkpoints.empty() && bound.operations.state_forks == 1 &&
                    bound.operations.partial_tail_cow_pages == 0,
                "input replay fork lost its carried point or copied a private KV tail");
        finish_prefill(*bound.sequence, 0, true);
        const auto repeated = program_.finish(*bound.sequence);
        require(repeated.checkpoint && program_.valid_checkpoint(input) &&
                    program_.inspect_source(base, input).has_value(),
                "carried input recovery point did not survive request completion");
        require(program_.release_checkpoint(*repeated.checkpoint) &&
                    program_.release_checkpoint(input),
                "carried input point retained a completed request reader");
        expect_empty("input recovery lifecycle leaked state or KV references");
    }

    void active_input_zero_growth_binding() {
        for (const std::uint32_t prompt_tokens : {64U, 65U}) {
            auto base           = request(prompt_tokens, true);
            const auto original = bind(base);
            reserve(original, qwen::ExecutionUnitKind::Prefill);
            auto begin = program_.advance_prefill(original);
            require(begin.complete && begin.pending,
                    "active input fixture did not reach its full input boundary");
            const std::array<runtime::CommitDecision, 1> accepted{{{.accepted_tokens = 1}}};
            const auto committed = program_.commit(std::move(*begin.pending), accepted);
            require(committed.capture_ready[0], "active input fixture did not offer P");
            const auto input =
                capture(original, prompt_tokens, runtime::CheckpointRole::InputReplay);

            // Exact-input reuse needs no Main page beyond the exported prefix. The original
            // request still owns the source writer while the second lane obtains its own view.
            const auto candidate = program_.inspect_source(base, input);
            require(candidate && candidate->reused_tokens == prompt_tokens &&
                        static_cast<bool>(program_.start_binding(base, {1}, *candidate)),
                    "active input could not reserve a second request without Main growth");
            synchronize();
            auto copying = program_.poll_context({});
            if (!copying.complete) {
                require(copying.advanced && !copying.sequence && program_.context_blocks(original),
                        "pending active prefix export did not block its source writer");
            }
            const auto bound = copying.complete ? std::move(copying) : settle();
            require(bound.sequence && !program_.context_blocks(original) &&
                        transferred(bound, runtime::ContextTransferDirection::DeviceToDevice,
                                    runtime::ContextResourceClass::MainKV) == (prompt_tokens == 65),
                    "active prefix binding did not settle its required tail copy and writer block");
            finish_prefill(*bound.sequence, 0, false);

            // Both writers must remain usable after the view publication, including growth at
            // the page-aligned boundary and append within the separately copied partial tail.
            for (const auto sequence : {original, *bound.sequence}) {
                reserve(sequence, qwen::ExecutionUnitKind::Decode, 1);
                const std::array<qwen::SequenceHandle, 1> members{sequence};
                const std::array<runtime::RoundBudget, 1> budgets{
                    {{.generated_tokens_remaining = 1}}};
                auto pending = program_.decode(members, budgets);
                const std::array<runtime::CommitDecision, 1> terminal{
                    {{.accepted_tokens = 1, .terminal = true}}};
                const auto decoded = program_.commit(std::move(pending), terminal);
                require(decoded.rows[0].disposition == runtime::CommitDisposition::Finishable,
                        "a writer could not progress after exporting its active prefix");
                const auto finished = program_.finish(sequence);
                require(finished.checkpoint && program_.release_checkpoint(*finished.checkpoint) &&
                            program_.valid_checkpoint(input),
                        "independent completion invalidated the retained input point");
            }
            require(program_.release_checkpoint(input),
                    "active prefix binding retained an independent writer reference");
            expect_empty("zero-growth active input binding leaked physical resources");
        }
    }

    void replay_carries_independent_input() {
        constexpr std::uint32_t prompt_tokens = 65;
        auto base                             = request(prompt_tokens, true);
        // Equal input tokens, independently computed histories. B is a candidate for replay;
        // A's retained P must continue to carry A's own state and KV together.
        const auto source_b = checkpoint(base);
        const auto active_a = bind(base);
        reserve(active_a, qwen::ExecutionUnitKind::Prefill);
        auto begin = program_.advance_prefill(active_a);
        require(begin.complete && begin.pending, "independent-history fixture did not begin A");
        const std::array<runtime::CommitDecision, 1> accepted{{{.accepted_tokens = 1}}};
        const auto committed = program_.commit(std::move(*begin.pending), accepted);
        require(committed.capture_ready[0], "independent-history fixture did not offer P_A");
        const auto input_a = capture(active_a, prompt_tokens, runtime::CheckpointRole::InputReplay);
        const std::array<qwen::CheckpointHandle, 1> points_a{input_a};
        const std::array<qwen::CheckpointHandle, 1> points_b{source_b};
        const std::array<qwen::CheckpointHandle, 2> independent{input_a, source_b};
        const auto input_resources   = program_.checkpoint_footprint(points_a);
        const auto check_independent = [&] {
            require(program_.valid_checkpoint(input_a) && program_.valid_checkpoint(source_b),
                    "recovery dropped an independently owned checkpoint");
            const auto a        = program_.checkpoint_footprint(points_a);
            const auto b        = program_.checkpoint_footprint(points_b);
            const auto together = program_.checkpoint_footprint(independent);
            require(a == input_resources &&
                        together.main_kv_pages == a.main_kv_pages + b.main_kv_pages &&
                        together.backend_kv_pages == a.backend_kv_pages + b.backend_kv_pages &&
                        together.state_slots == a.state_slots + b.state_slots &&
                        together.host_bytes == a.host_bytes + b.host_bytes,
                    "recovery spliced independently computed input state and KV histories");
        };

        reserve(active_a, qwen::ExecutionUnitKind::Decode, 1);
        const std::array<qwen::SequenceHandle, 1> members{active_a};
        const std::array<runtime::RoundBudget, 1> budgets{{{.generated_tokens_remaining = 1}}};
        auto pending = program_.decode(members, budgets);
        (void)program_.commit(std::move(pending), accepted);
        require(program_.start_pause(active_a, false),
                "independent-history fixture could not pause A for replay");
        auto paused = settle();
        require(paused.paused && !paused.paused->has_snapshot() &&
                    paused.paused->frontier() == prompt_tokens + 1,
                "independent-history Replay pause lost A's committed target");
        check_independent();
        auto source = program_.inspect_source(base, source_b, false, points_a);
        require(source.has_value(), "independent replay source did not match");
        source->take_private   = true; // A owns these points even though B supplies the source.
        const auto before_bind = program_.physical_usage();
        require(source &&
                    static_cast<bool>(program_.start_binding(base, {1}, *source, &*paused.paused)),
                "A could not select independently computed B as its replay source");
        auto rebound = settle();
        paused.paused.reset();
        require(rebound.sequence && rebound.replaying &&
                    rebound.private_points == std::vector<qwen::CheckpointHandle>{input_a} &&
                    program_.physical_usage().occupied.state_slots ==
                        before_bind.occupied.state_slots + 1,
                "Replay did not transfer A's original point with exactly one new active writer");
        check_independent();
        reserve(*rebound.sequence, qwen::ExecutionUnitKind::Replay);
        const auto replayed = program_.advance_replay(*rebound.sequence);
        require(replayed.complete && replayed.processed_tokens == 1 && !replayed.capture_ready,
                "cross-source Replay rebuilt more than A's committed suffix");
        require(program_.start_pause(*rebound.sequence, true),
                "cross-source request could not snapshot its recovered execution");
        auto snapshot = settle();
        require(snapshot.paused && snapshot.paused->has_snapshot(),
                "cross-source request lost its pause snapshot");
        check_independent();
        auto snapshot_source = root(base);
        snapshot_source.private_points.assign(points_a.begin(), points_a.end());
        snapshot_source.take_private = true;
        require(static_cast<bool>(
                    program_.start_binding(base, {0}, snapshot_source, &*snapshot.paused)),
                "cross-source snapshot could not restore with its independent input point");
        auto restored = settle();
        snapshot.paused.reset();
        require(restored.sequence && !restored.replaying &&
                    restored.private_points == std::vector<qwen::CheckpointHandle>{input_a},
                "snapshot restore cloned or replaced A's owned input point");
        check_independent();

        // B's resumed writer remains active: P_A must still own a separately writable history.
        const auto input_source = program_.inspect_source(base, input_a, true, points_a);
        require(input_source && input_source->move_history &&
                    static_cast<bool>(program_.start_binding(base, {1}, *input_source)),
                "independent input point cannot bind while the recovered B history is active");
        const auto input_bound = settle();
        require(input_bound.sequence &&
                    input_bound.private_points == std::vector<qwen::CheckpointHandle>{input_a},
                "independent input binding lost its retained recovery point");
        finish_prefill(*input_bound.sequence, 0, true);
        const auto finished = program_.finish(*input_bound.sequence);
        require(finished.checkpoint &&
                    program_.abort(*restored.sequence).status == runtime::ConsumeStatus::Consumed &&
                    program_.release_checkpoint(*finished.checkpoint) &&
                    program_.release_checkpoint(input_a) && program_.release_checkpoint(source_b),
                "cross-source recovery did not release its independent history owners");
        expect_empty("cross-source Replay and Snapshot retained physical references");
    }

    void retained_input_host_suffix() {
        constexpr std::uint32_t prompt_tokens = 65;
        auto base                             = request(prompt_tokens, true);
        const auto active                     = bind(base);
        reserve(active, qwen::ExecutionUnitKind::Prefill);
        auto begin = program_.advance_prefill(active);
        require(begin.complete && begin.pending && begin.pending->tokens().size() == 1,
                "Host suffix fixture did not produce one Begin token");
        std::vector<TokenId> endpoint_input(prompt_tokens, 198);
        endpoint_input.front() = 1000;
        endpoint_input.push_back(begin.pending->tokens().front());
        const std::array<runtime::CommitDecision, 1> accepted{{{.accepted_tokens = 1}}};
        const auto committed = program_.commit(std::move(*begin.pending), accepted);
        require(committed.capture_ready[0], "Host suffix fixture did not offer P");
        const auto input = capture(active, prompt_tokens, runtime::CheckpointRole::InputReplay);

        // Extend the same history across complete KV pages, retaining the earlier input point.
        const std::array<qwen::SequenceHandle, 1> members{active};
        std::array<TokenId, kChunk> forced;
        forced.fill(198);
        const std::array<std::optional<std::uint32_t>, 1> splits{std::nullopt};
        reserve(active, qwen::ExecutionUnitKind::Control, forced.size());
        (void)program_.append_forced_tokens(members, forced, forced.size(), splits);
        endpoint_input.insert(endpoint_input.end(), forced.begin(), forced.end());
        reserve(active, qwen::ExecutionUnitKind::Decode, 1);
        const std::array<runtime::RoundBudget, 1> budgets{{{.generated_tokens_remaining = 1}}};
        auto pending = program_.decode(members, budgets);
        const std::array<runtime::CommitDecision, 1> terminal{
            {{.accepted_tokens = 1, .terminal = true}}};
        (void)program_.commit(std::move(pending), terminal);
        const auto finished = program_.finish(active);
        require(finished.checkpoint && program_.valid_checkpoint(input),
                "Host suffix fixture lost its endpoint or retained input");
        const auto endpoint = *finished.checkpoint;
        const std::array<qwen::CheckpointHandle, 2> history{input, endpoint};
        for (const bool backend : {false, true}) {
            const auto usage = program_.physical_usage().occupied;
            runtime::ContextResourceUsage shortage;
            if (backend) {
                if (!usage.backend_kv_pages) { continue; }
                shortage.backend_kv_pages = usage.backend_kv_pages;
            } else {
                shortage.main_kv_pages = usage.main_kv_pages;
            }
            auto candidates = program_.plan_reclaim(history, {}, shortage).demotions;
            require(!candidates.empty(), "shared history has no Host demotion candidate");
            while (!candidates.empty()) {
                const auto& quote = candidates.front();
                require(quote.host_bytes > 0 && program_.start_demote(quote),
                        "shared history could not transfer its selected KV pages to Host");
                require(settle().published, "shared history Host transfer did not complete");
                candidates = program_.plan_reclaim(history, {}, shortage).demotions;
            }
        }

        runtime::ResolvedExecutionOptions options;
        options.requested_output_tokens = 8;
        options.allow_prefix_reuse      = true;
        auto endpoint_base =
            program_.plan_request(frontend_.prepare_tokens(std::move(endpoint_input)), options);
        const auto reader = program_.inspect_source(endpoint_base, endpoint);
        require(reader && reader->reused_tokens == endpoint_base.summary().prompt_tokens,
                "retained endpoint does not match its committed input");
        const auto before_reader = program_.physical_usage();
        require(static_cast<bool>(program_.start_binding(endpoint_base, {1}, *reader)),
                "Host endpoint could not reserve an independent reader");
        const std::array<qwen::CheckpointHandle, 1> endpoint_set{endpoint};
        require(program_.host_bytes_released(endpoint_set) == 0 &&
                    !program_.release_checkpoint(endpoint),
                "pending reader allowed its complete Host suffix to be reclaimed");
        require(!settle(true).published, "Host suffix reader did not cancel");
        expect_usage(before_reader, "cancelled Host reader changed retained history ownership");

        const auto expected_release = program_.host_bytes_released(endpoint_set);
        const auto before_release   = program_.physical_usage().occupied.host_bytes;
        require(expected_release > 0 && program_.release_checkpoint(endpoint),
                "retiring E did not expose its unneeded Host suffix");
        const auto after_release = program_.physical_usage().occupied.host_bytes;
        require(before_release >= after_release &&
                    before_release - after_release == expected_release &&
                    program_.valid_checkpoint(input),
                "Host suffix release disagrees with its quote or invalidates retained P");

        const std::array<qwen::CheckpointHandle, 1> points{input};
        const auto retained = program_.inspect_source(base, input, true, points);
        require(retained && retained->reused_tokens == prompt_tokens &&
                    static_cast<bool>(program_.start_binding(base, {0}, *retained)),
                "retained P could not restore after its deeper endpoint was retired");
        const auto restored = settle();
        require(restored.sequence &&
                    restored.private_points == std::vector<qwen::CheckpointHandle>{input},
                "restoring the trimmed history lost its retained input point");
        finish_prefill(*restored.sequence, 0, true);
        const auto repeated = program_.finish(*restored.sequence);
        require(repeated.checkpoint && program_.release_checkpoint(*repeated.checkpoint) &&
                    program_.release_checkpoint(input),
                "restored input history retained completed execution references");
        expect_empty("retained input Host suffix fixture leaked physical resources");
    }

    void private_move() {
        require(program_.physical_usage().capacity.state_slots == 1,
                "private Move fixture must have exactly one physical state slot");
        auto base            = request(65, true);
        const auto source    = checkpoint(base);
        const auto candidate = program_.inspect_source(base, source, true);
        require(candidate && candidate->move_state && candidate->move_history,
                "exclusive continuation was not eligible for Move");
        require(static_cast<bool>(program_.start_binding(base, {0}, *candidate)),
                "exclusive continuation required a second state slot");
        const auto bound = settle();
        require(bound.sequence && !program_.valid_checkpoint(source) &&
                    bound.retired_checkpoints == std::vector<qwen::CheckpointHandle>{source} &&
                    bound.operations.state_moves == 1 && bound.operations.state_forks == 0 &&
                    bound.operations.partial_tail_cow_pages == 0 &&
                    !transferred(bound, runtime::ContextTransferDirection::DeviceToDevice),
                "private Move copied resources or failed to retire its consumed endpoint");
        finish_prefill(*bound.sequence, 0, true);
        const auto finished = program_.finish(*bound.sequence);
        require(finished.checkpoint && program_.release_checkpoint(*finished.checkpoint),
                "private Move did not publish a releasable successor");
        expect_empty("private Move retained its consumed owner");
    }

    void capture_retry_after_reclaim() {
        require(program_.physical_usage().capacity.state_slots == 2 &&
                    program_.physical_usage().capacity.host_bytes == 0,
                "capture retry fixture requires one active and one optional State slot");
        auto seed           = request(64, true);
        const auto inactive = checkpoint(seed);
        auto marked         = marker_request();
        const auto active   = bind(marked.base);
        reserve(active, qwen::ExecutionUnitKind::Prefill);
        const auto full = program_.physical_usage();
        require(full.occupied.state_slots == full.capacity.state_slots,
                "capture retry fixture did not fill its State capacity");
        for (unsigned attempt = 0; attempt < 2; ++attempt) {
            const auto prepared = program_.prepare_capture(active);
            require(prepared && !prepared->reserved &&
                        prepared->frontier == marked.frontiers.front() &&
                        prepared->shortage == runtime::ContextResourceUsage{.state_slots = 1} &&
                        prepared->host_bytes > 0,
                    "failed capture preparation lost its semantic boundary or physical shortage");
            expect_usage(full, "failed capture preparation retained a physical reservation");
        }
        require(program_.release_checkpoint(inactive),
                "capture retry fixture could not reclaim its idle checkpoint");
        const auto prepared = program_.prepare_capture(active);
        require(prepared && prepared->reserved && prepared->frontier == marked.frontiers.front() &&
                    prepared->shortage == runtime::ContextResourceUsage{},
                "capture preparation did not retry the original boundary after reclamation");
        const auto step = program_.advance_prefill(active);
        require(step.capture_ready && !step.complete && !step.pending &&
                    step.processed_prompt_tokens == marked.frontiers.front(),
                "retried capture advanced beyond its reserved semantic boundary");
        const auto saved =
            capture(active, marked.frontiers.front(), runtime::CheckpointRole::SharedPrefix);
        require(program_.abort(active).status == runtime::ConsumeStatus::Consumed &&
                    program_.release_checkpoint(saved),
                "retried capture did not release its active writer and saved prefix");
        expect_empty("capture retry retained physical ownership");
    }

    void capture_without_destination() {
        require(program_.physical_usage().capacity.state_slots == 1 &&
                    program_.physical_usage().capacity.host_bytes == 0,
                "capture pressure fixture has an optional StateImage destination");
        auto marked         = marker_request();
        const auto sequence = bind(marked.base);
        for (unsigned chunk = 0; chunk < 2; ++chunk) {
            reserve(sequence, qwen::ExecutionUnitKind::Prefill);
            const auto step = program_.advance_prefill(sequence);
            require(step.processed_prompt_tokens == kChunk && !step.complete && !step.capture_ready,
                    "unavailable optional prefix changed the native prefill chunk boundary");
        }
        require(program_.abort(sequence).status == runtime::ConsumeStatus::Consumed,
                "capture pressure fixture did not release its sole state slot");
        expect_empty("unavailable optional capture retained typed resources");
        next_request();
    }

    void cancel_gpu_binding() {
        auto base = request(65, true); // A partial page makes the second binding perform real COW.
        const auto source   = checkpoint(base);
        const auto baseline = program_.physical_usage();
        for (const bool submit_copy : {false, true}) {
            auto candidate = program_.inspect_source(base, source);
            require(candidate && candidate->reused_tokens == 65, "seed checkpoint is not reusable");
            require(static_cast<bool>(program_.start_binding(base, {0}, *candidate)),
                    "checkpoint binding did not reserve its destinations");
            std::optional<qwen::ContextProgress> completed;
            if (submit_copy) {
                synchronize();
                auto progress = program_.poll_context({});
                if (progress.complete) { completed.emplace(std::move(progress)); }
            }
            const auto cancelled = completed ? std::move(*completed) : settle(true);
            if (completed) {
                require(cancelled.published && cancelled.sequence &&
                            program_.abort(*cancelled.sequence).status ==
                                runtime::ConsumeStatus::Consumed,
                        "cancellation lost a binding which completed before it arrived");
            } else {
                require(!cancelled.published && !cancelled.sequence,
                        "cancelled binding published a live sequence");
            }
            if (submit_copy) {
                require(transferred(cancelled, runtime::ContextTransferDirection::DeviceToDevice,
                                    runtime::ContextResourceClass::MainKV),
                        "post-copy cancellation did not exercise a real KV tail copy");
            }
            expect_usage(baseline, "cancelled binding retained a lane, state or KV reservation");
            require(program_.checkpoint_summary(source).frontier == 65,
                    "binding cancellation destroyed its source checkpoint");
        }
        require(program_.release_checkpoint(source), "cancelled binding left its source pinned");
        expect_empty("binding cancellation leaked typed resources");
        next_request();
    }

    void raw_identity_opt_out() {
        auto reusable     = request(65, true);
        const auto source = checkpoint(reusable);
        require(program_.inspect_source(reusable, source).has_value(),
                "raw identity opt-out fixture has no reusable checkpoint");
        auto disabled = request(65, true, false);
        require(!program_.inspect_source(disabled, source),
                "raw identity opt-out still accepts a matching checkpoint");
        require(program_.inspect_source(disabled, std::nullopt).has_value() &&
                    !disabled.summary().publish_continuation,
                "raw identity opt-out must preserve root execution without publishing history");
        require(program_.release_checkpoint(source),
                "raw identity fixture retained its checkpoint");
        expect_empty("raw identity opt-out leaked typed resources");
    }

    void cancel_host_restore() {
        auto base         = request(65, true);
        const auto source = checkpoint(base);
        const std::array<qwen::CheckpointHandle, 1> allowed{source};
        for (const auto resource :
             {runtime::ContextResourceClass::State, runtime::ContextResourceClass::MainKV,
              runtime::ContextResourceClass::BackendKV}) {
            if (resource == runtime::ContextResourceClass::BackendKV &&
                program_.physical_usage().occupied.backend_kv_pages == 0) {
                continue;
            }
            const auto usage = program_.physical_usage().occupied;
            runtime::ContextResourceUsage shortage;
            switch (resource) {
            case runtime::ContextResourceClass::State:
                shortage.state_slots = usage.state_slots;
                break;
            case runtime::ContextResourceClass::MainKV:
                shortage.main_kv_pages = usage.main_kv_pages;
                break;
            case runtime::ContextResourceClass::BackendKV:
                shortage.backend_kv_pages = usage.backend_kv_pages;
                break;
            }
            const auto candidates = program_.plan_reclaim(allowed, {}, shortage).demotions;
            const auto* quote     = demotion_with_sources(candidates, allowed);
            require(quote && quote->host_bytes != 0 && program_.start_demote(*quote),
                    "source could not demote its selected resources for restore");
            const auto demoted = settle();
            require(
                demoted.published &&
                    transferred(demoted, runtime::ContextTransferDirection::DeviceToHost, resource),
                "demotion did not copy its typed source to Host");
        }
        const auto baseline = program_.physical_usage();
        auto candidate      = program_.inspect_source(base, source);
        require(candidate.has_value(), "Host checkpoint is not a valid recovery source");
        require(static_cast<bool>(program_.start_binding(base, {1}, *candidate)),
                "Host restore did not reserve its destinations");
        // start_binding has submitted the prelude H2D (State). Main KV lands layer by layer only
        // after the prelude, so a binding cancelled first never submits it. Cancellation stays
        // asserted while poll settles the prelude.
        const auto cancelled = settle(true);
        require(!cancelled.published && !cancelled.sequence &&
                    transferred(cancelled, runtime::ContextTransferDirection::HostToDevice,
                                runtime::ContextResourceClass::State) &&
                    !transferred(cancelled, runtime::ContextTransferDirection::HostToDevice,
                                 runtime::ContextResourceClass::MainKV),
                "restore cancellation did not settle its submitted State copy alone");
        expect_usage(baseline, "cancelled H2D restore retained unpublished destinations");

        // A published binding whose Main KV is still landing, released before any pass reads it:
        // the release waits for the copies, then every resource returns.
        candidate = program_.inspect_source(base, source);
        require(candidate.has_value() &&
                    static_cast<bool>(program_.start_binding(base, {1}, *candidate)),
                "Host restore could not be bound again after a cancellation");
        const auto landing = settle();
        require(landing.published && landing.sequence.has_value(),
                "Host restore did not publish before its Main KV landed");
        require(program_.abort(*landing.sequence).status == runtime::ConsumeStatus::Consumed,
                "a landing restore's lane could not be released before its first pass");
        for (const auto point : landing.private_points) {
            if (program_.valid_checkpoint(point)) { (void)program_.release_checkpoint(point); }
        }
        // The binding may have consumed (moved) its source.
        if (program_.valid_checkpoint(source)) {
            require(program_.release_checkpoint(source),
                    "released landing restore retained source leases");
        }
        expect_empty("a released landing restore leaked typed resources");
        next_request();
    }

    void cancel_and_destroy_paused_prefill() {
        for (const bool cancel_transfer : {true, false}) {
            auto base           = request(kPromptTokens);
            const auto sequence = bind(base);
            prefill_prefix(sequence, kPauseFrontier);
            require(program_.start_pause(sequence, true), "prefill pause was not accepted");
            auto paused = settle(cancel_transfer);
            require(transferred(paused, runtime::ContextTransferDirection::DeviceToHost,
                                runtime::ContextResourceClass::State) &&
                        transferred(paused, runtime::ContextTransferDirection::DeviceToHost,
                                    runtime::ContextResourceClass::MainKV),
                    "pause did not submit its complete State/KV snapshot");
            if (cancel_transfer) {
                require(!paused.published && !paused.paused,
                        "cancelled pause published a durable snapshot");
            } else {
                require(paused.published && paused.paused && paused.paused->has_snapshot() &&
                            paused.paused->frontier() == kPauseFrontier,
                        "prefill pause lost its exact recovery frontier");
                // Engine-wide active cleanup must not steal externally owned snapshot content.
                program_.fail_all_cleanup();
                require(paused.paused->has_snapshot(), "active cleanup revoked an owned snapshot");
                paused.paused.reset();
            }
            expect_empty("cancelled or destroyed paused prefill retained physical resources");
            next_request();
        }
    }

    void replay_pause_and_resume() {
        auto base           = request(kPromptTokens);
        const auto original = bind(base);
        prefill_prefix(original, kPauseFrontier);
        require(program_.start_pause(original, false), "replay-only pause was not accepted");
        auto first = settle();
        require(first.paused && !first.paused->has_snapshot() &&
                    first.paused->frontier() == kPauseFrontier,
                "snapshot-free prefill pause lost its replay target");
        expect_empty("snapshot-free pause did not release its execution resources");

        const auto replay = bind(base, 1, &*first.paused, qwen::ExecutionUnitKind::Prefill);
        first.paused.reset();
        reserve(replay, qwen::ExecutionUnitKind::Replay);
        const auto partial = program_.advance_replay(replay);
        require(partial.processed_tokens == kChunk && !partial.complete,
                "replay did not yield after its first finite chunk");
        require(program_.start_pause(replay, false),
                "partially replayed request could not pause again");
        auto second = settle();
        require(second.paused && !second.paused->has_snapshot() &&
                    second.paused->frontier() == kPauseFrontier,
                "repeated replay pause replaced the original target with its partial cursor");
        expect_empty("repeated replay pause retained execution resources");

        const auto resumed = bind(base, 0, &*second.paused, qwen::ExecutionUnitKind::Prefill);
        second.paused.reset();
        require(program_.abort(original).status == runtime::ConsumeStatus::InvariantMismatch &&
                    program_.abort(replay).status == runtime::ConsumeStatus::InvariantMismatch,
                "a stale pre-pause sequence handle consumed a newly bound lane");
        std::uint32_t replayed = 0;
        bool complete          = false;
        for (unsigned step = 0; step < kPauseFrontier / kChunk + 1; ++step) {
            reserve(resumed, qwen::ExecutionUnitKind::Replay);
            const auto progress = program_.advance_replay(resumed);
            replayed += progress.processed_tokens;
            if (progress.complete) {
                complete = true;
                break;
            }
        }
        require(complete && replayed == kPauseFrontier,
                "second replay did not recover the entire original target");
        finish_prefill(resumed, kPromptTokens - kPauseFrontier, false);
        decode_and_finish(resumed);
        expect_empty("replayed request retained physical resources after finishing");
        next_request();

        // Cancellation owns the paused request by destroying its ResumeState; no request output
        // session or resumed lane is needed merely to reclaim a snapshot-free replay record.
        const auto to_cancel = bind(base);
        prefill_prefix(to_cancel, kPauseFrontier);
        require(program_.start_pause(to_cancel, false), "cancel fixture could not pause prefill");
        auto saved = settle();
        require(saved.paused.has_value(), "cancel fixture lost its paused prefill record");
        const auto rebuilding = bind(base, 1, &*saved.paused, qwen::ExecutionUnitKind::Prefill);
        saved.paused.reset();
        reserve(rebuilding, qwen::ExecutionUnitKind::Replay);
        require(!program_.advance_replay(rebuilding).complete,
                "cancel fixture replay completed early");
        require(program_.start_pause(rebuilding, false), "cancel fixture could not pause replay");
        auto cancelled = settle();
        require(cancelled.paused && cancelled.paused->frontier() == kPauseFrontier,
                "paused replay cancellation lost its request target");
        cancelled.paused.reset();
        expect_empty("destroyed paused replay retained typed resources");
        next_request();
    }

    void retired_host_image_preserves_input() {
        auto base         = request(65, true);
        const auto active = bind(base);
        reserve(active, qwen::ExecutionUnitKind::Prefill);
        auto begin = program_.advance_prefill(active);
        require(begin.pending.has_value(), "Host retirement fixture did not begin");
        const std::array<runtime::CommitDecision, 1> accepted{{{.accepted_tokens = 1}}};
        require(program_.commit(std::move(*begin.pending), accepted).capture_ready[0],
                "Host retirement fixture did not offer its input point");
        const auto input = capture(active, 65, runtime::CheckpointRole::InputReplay);
        reserve(active, qwen::ExecutionUnitKind::Decode, 1);
        const std::array<qwen::SequenceHandle, 1> members{active};
        const std::array<runtime::RoundBudget, 1> budgets{{{.generated_tokens_remaining = 1}}};
        auto output = program_.decode(members, budgets);
        const std::array<runtime::CommitDecision, 1> terminal{
            {{.accepted_tokens = 1, .terminal = true}}};
        (void)program_.commit(std::move(output), terminal);
        const auto finished = program_.finish(active);
        require(finished.checkpoint.has_value(), "Host retirement fixture lost its endpoint");
        const auto endpoint      = *finished.checkpoint;
        const auto offload_state = [&](qwen::CheckpointHandle point) {
            const std::array handles{point};
            const auto plans = program_.plan_reclaim(handles, {}, {.state_slots = 1});
            const auto found =
                std::find_if(plans.demotions.begin(), plans.demotions.end(),
                             [](const auto& p) { return p.released.state_slots == 1; });
            require(found != plans.demotions.end() && program_.start_demote(*found),
                    "Host retirement fixture could not offload its State image");
            const auto bytes = found->host_bytes;
            require(settle().published, "Host retirement State transfer failed");
            return bytes;
        };
        const auto image_bytes = offload_state(endpoint);
        require(image_bytes != 0, "Host retirement fixture has no image allocation");
        std::vector<qwen::CheckpointHandle> fillers;
        auto filler = request(32, true);
        while (program_.physical_usage().capacity.host_bytes -
                   program_.physical_usage().occupied.host_bytes >=
               image_bytes) {
            fillers.push_back(checkpoint(filler));
            (void)offload_state(fillers.back());
        }
        while (program_.physical_usage().occupied.state_slots <
               program_.physical_usage().capacity.state_slots) {
            fillers.push_back(checkpoint(filler));
        }
        const std::array points{input};
        const std::array retired{endpoint};
        auto source = program_.inspect_source(base, input, true, points, retired);
        require(source && source->move_state,
                "Host retirement fixture did not require consuming its input before retirement");
        const auto binding = program_.start_binding(base, {0}, *source);
        require(binding && !binding.consumed_source,
                "binding discarded input despite the Host destination freed by retirement");
        const auto bound = settle();
        require(bound.sequence && program_.valid_checkpoint(input) &&
                    bound.private_points == std::vector(points.begin(), points.end()) &&
                    transferred(bound, runtime::ContextTransferDirection::DeviceToHost,
                                runtime::ContextResourceClass::State),
                "retired Host space did not preserve the complete input recovery image");
        require(program_.abort(*bound.sequence).status == runtime::ConsumeStatus::Consumed &&
                    program_.release_checkpoint(input),
                "Host retirement fixture could not release its binding and input");
        for (const auto point : fillers) {
            require(program_.release_checkpoint(point), "Host retirement filler leaked ownership");
        }
        expect_empty("Host retirement preservation leaked resources");
    }

    void rewind_binding_retirement() {
        require(program_.physical_usage().capacity.main_kv_pages == 8,
                "rewind fixture requires the eight-page Main pool");
        for (const bool releases_suffix : {false, true}) {
            constexpr std::uint32_t prefix = 65;
            auto original                  = request(prefix, true);
            const auto active              = bind(original);
            reserve(active, qwen::ExecutionUnitKind::Prefill);
            auto begin = program_.advance_prefill(active);
            require(begin.pending.has_value(), "rewind fixture did not complete input");
            const std::array<runtime::CommitDecision, 1> accepted{{{.accepted_tokens = 1}}};
            require(program_.commit(std::move(*begin.pending), accepted).capture_ready[0],
                    "rewind fixture did not offer input recovery");
            const auto input = capture(active, prefix, runtime::CheckpointRole::InputReplay);
            const std::array<qwen::SequenceHandle, 1> members{active};
            if (releases_suffix) {
                std::array<TokenId, kChunk> forced;
                forced.fill(198);
                const std::array<std::optional<std::uint32_t>, 1> splits{std::nullopt};
                reserve(active, qwen::ExecutionUnitKind::Control, forced.size());
                (void)program_.append_forced_tokens(members, forced, forced.size(), splits);
            }
            reserve(active, qwen::ExecutionUnitKind::Decode, 1);
            const std::array<runtime::RoundBudget, 1> budgets{{{.generated_tokens_remaining = 1}}};
            auto output = program_.decode(members, budgets);
            const std::array<runtime::CommitDecision, 1> terminal{
                {{.accepted_tokens = 1, .terminal = true}}};
            (void)program_.commit(std::move(output), terminal);
            auto finished = program_.finish(active);
            require(finished.checkpoint.has_value(), "rewind fixture lost its deeper endpoint");
            const auto endpoint = *finished.checkpoint;
            auto competing      = request(kPromptTokens);
            const auto blocker  = bind(competing, 1);
            if (releases_suffix) {
                prefill_prefix(blocker, 2 * kChunk);
            } else {
                finish_prefill(blocker, kPromptTokens, false);
            }
            require(program_.physical_usage().occupied.main_kv_pages == 8,
                    "rewind fixture did not fill its Main pool");
            auto rewritten = request(kCapacity, true);
            const std::array<qwen::CheckpointHandle, 1> retirement{endpoint};
            auto choice = program_.inspect_source(rewritten, input, true, {}, retirement);
            require(choice && choice->move_history && choice->move_state &&
                        program_.valid_checkpoint(input) && program_.valid_checkpoint(endpoint),
                    "rewind evaluation mutated history or missed the authorized Move");
            const auto before = program_.physical_usage();
            auto binding      = program_.start_binding(rewritten, {0}, *choice);
            if (!releases_suffix) {
                require(!binding && binding.source_valid && binding.shortage.main_kv_pages == 2 &&
                            !program_.has_context_transaction() &&
                            program_.valid_checkpoint(endpoint) && program_.valid_checkpoint(input),
                        "failed rewind retired a still-useful checkpoint");
                expect_usage(before, "failed rewind changed live resources");
                require(program_.abort(blocker).status == runtime::ConsumeStatus::Consumed,
                        "rewind blocker could not release its resources");
                binding = program_.start_binding(rewritten, {0}, *choice);
            }
            require(binding &&
                        binding.retired_points ==
                            std::vector(retirement.begin(), retirement.end()) &&
                        !program_.valid_checkpoint(endpoint),
                    "binding failed to fund its reservation from the authorized retired suffix");
            const auto bound = settle();
            require(bound.sequence && bound.operations.state_moves == 1 &&
                        bound.operations.state_forks == 0 && !program_.valid_checkpoint(input),
                    "rewind ownership handoff needlessly forked or retained its consumed source");
            require(program_.abort(*bound.sequence).status == runtime::ConsumeStatus::Consumed,
                    "rewind binding could not release its writer");
            if (releases_suffix) {
                require(program_.abort(blocker).status == runtime::ConsumeStatus::Consumed,
                        "rewind suffix fixture leaked its blocking request");
            }
            expect_empty("rewind binding leaked retired history or unit reservations");
        }
    }

    void fork_exceeds_physical_capacity() {
        auto prefix       = request(kCapacity - 1, true);
        const auto point  = checkpoint(prefix);
        auto full         = request(kCapacity, true);
        const auto source = program_.inspect_source(full, point);
        require(source && !source->move_history, "fork capacity fixture consumed its source");
        const auto before  = program_.physical_usage();
        const auto blocked = program_.start_binding(full, {0}, *source);
        require(!blocked && blocked.source_valid && !blocked.capacity_possible &&
                    blocked.shortage.main_kv_pages == 1 && program_.valid_checkpoint(point),
                "a full-pool fork waited for capacity that its extra tail can never obtain");
        expect_usage(before, "impossible fork changed its immutable source");
        require(program_.release_checkpoint(point), "fork capacity fixture leaked its source");
        const auto cold = bind(full);
        finish_prefill(cold, kCapacity, true);
        const auto finished = program_.finish(cold);
        require(finished.checkpoint && program_.release_checkpoint(*finished.checkpoint),
                "the same request could not finish through the feasible root route");
        expect_empty("fork capacity fallback leaked physical resources");
    }

    void recovery_capacity_and_progress() {
        require(program_.physical_usage().capacity.main_kv_pages == 8,
                "recovery fixture requires the bounded eight-page Main pool");
        for (const bool snapshot : {false, true}) {
            auto base           = request(kPromptTokens);
            const auto original = bind(base);
            prefill_prefix(original, kPauseFrontier);
            require(program_.start_pause(original, snapshot), "recovery fixture could not pause");
            auto paused = settle();
            require(paused.paused && paused.paused->has_snapshot() == snapshot,
                    "recovery fixture did not retain the selected recovery mode");

            auto competing_base = request(kPauseFrontier);
            const auto blocker  = bind(competing_base, 1);
            finish_prefill(blocker, kPauseFrontier, false);
            const auto before  = program_.physical_usage();
            const auto refused = program_.start_binding(base, {0}, root(base), &*paused.paused,
                                                        qwen::ExecutionUnitKind::Prefill, 0);
            require(!refused && refused.shortage.main_kv_pages == 2 &&
                        !program_.has_context_transaction() &&
                        paused.paused->has_snapshot() == snapshot,
                    "recovery admitted its first chunk without capacity for restored progress");
            expect_usage(before, "failed recovery changed live resource ownership");
            require(program_.abort(blocker).status == runtime::ConsumeStatus::Consumed,
                    "recovery blocker failed to release its resources");

            const auto recovered = bind(base, 0, &*paused.paused, qwen::ExecutionUnitKind::Prefill);
            paused.paused.reset();
            require(program_.recovery_pending(recovered),
                    "resumed binding lost its recovery permit");
            const auto competitor = bind(competing_base, 1);
            prefill_prefix(competitor, kChunk);
            const std::array<qwen::ExecutionUnit, 1> competitor_unit{
                {{competitor, qwen::ExecutionUnitKind::Prefill, 0}}};
            const auto check_protected = [&] {
                const auto growth = program_.reserve_units(competitor_unit);
                require(!growth && growth.shortage.main_kv_pages == 2 &&
                            program_.recovery_pending(recovered),
                        "another request consumed capacity promised to complete recovery");
            };
            check_protected();
            if (!snapshot) {
                std::uint32_t replayed = 0;
                bool complete          = false;
                for (unsigned step = 0; step < kPauseFrontier / kChunk + 1; ++step) {
                    reserve(recovered, qwen::ExecutionUnitKind::Replay);
                    const auto progress = program_.advance_replay(recovered);
                    replayed += progress.processed_tokens;
                    check_protected();
                    if (progress.complete) {
                        complete = true;
                        break;
                    }
                }
                require(complete && replayed == kPauseFrontier,
                        "recovery failed to reconstruct its saved frontier");
            }
            finish_prefill(recovered, kPromptTokens - kPauseFrontier, false);
            require(!program_.recovery_pending(recovered),
                    "committed new prefill output did not release recovery protection");
            require(program_.abort(competitor).status == runtime::ConsumeStatus::Consumed,
                    "recovery competitor failed to release resources");
            decode_and_finish(recovered);
            expect_empty("completed recovery retained protected resources");

            const auto partial = bind(base);
            prefill_prefix(partial, kChunk);
            require(program_.start_pause(partial, snapshot), "partial recovery could not pause");
            auto saved           = settle();
            const auto continued = bind(base, 0, &*saved.paused, qwen::ExecutionUnitKind::Prefill);
            saved.paused.reset();
            if (!snapshot) {
                reserve(continued, qwen::ExecutionUnitKind::Replay);
                require(program_.advance_replay(continued).complete,
                        "partial recovery did not reconstruct its single saved chunk");
            }
            require(program_.recovery_pending(continued),
                    "reconstructing old progress prematurely released recovery protection");
            prefill_prefix(continued, kChunk);
            require(!program_.recovery_pending(continued),
                    "new partial prefill did not release recovery protection");
            require(program_.abort(continued).status == runtime::ConsumeStatus::Consumed,
                    "partial recovery did not release its writer");
            expect_empty("partial recovery retained resource reservations");
            next_request();
        }
    }

    void restore_full_pool_snapshot() {
        constexpr std::uint32_t frontier = 500;
        auto base                        = request(frontier);
        const auto sequence              = bind(base);
        finish_prefill(sequence, frontier, false);
        require(program_.physical_usage().occupied.main_kv_pages == 8,
                "full-pool snapshot fixture did not occupy all eight Main pages");
        require(program_.start_pause(sequence, true),
                "full-pool request could not save its snapshot");
        auto paused = settle();
        require(paused.paused && paused.paused->has_snapshot() &&
                    paused.paused->frontier() == frontier,
                "full-pool request fell back to replay instead of saving its snapshot");
        {
            require(
                static_cast<bool>(program_.start_binding(base, {1}, root(base), &*paused.paused)),
                "own snapshot restore required a ninth Main page");
            const auto cancelled = settle(true);
            require(!cancelled.published && !cancelled.sequence && paused.paused->has_snapshot() &&
                        paused.paused->frontier() == frontier,
                    "cancelled own restore consumed its original paused snapshot");
            require(program_.physical_usage().occupied.main_kv_pages <= 8 &&
                        program_.physical_usage().host_reserved_bytes == 0,
                    "cancelled own restore leaked destination reservations");
        }
        const auto resumed = bind(base, 0, &*paused.paused);
        paused.paused.reset();
        decode_and_finish(resumed);
        expect_empty("full-pool snapshot restore retained its transferred checkpoint ownership");
        next_request();
    }

    void independent_physical_demotions() {
        auto first_base   = request(64, true);
        auto second_base  = request(128, true);
        const auto first  = checkpoint(first_base);
        const auto second = checkpoint(second_base);
        const std::array<qwen::CheckpointHandle, 2> allowed{first, second};
        const auto before     = program_.physical_usage().occupied;
        const auto candidates = program_.plan_reclaim(allowed, {}, {.main_kv_pages = 3}).demotions;
        const std::array<qwen::CheckpointHandle, 1> first_owner{first}, second_owner{second};
        const auto* first_quote  = demotion_with_sources(candidates, first_owner);
        const auto* second_quote = demotion_with_sources(candidates, second_owner);
        require(first_quote && first_quote->released.main_kv_pages == 1 && second_quote &&
                    second_quote->released.main_kv_pages == 2 && first_quote->host_bytes > 0 &&
                    program_.start_demote(*first_quote),
                "independent physical owners did not offer independently executable demotions");
        require(settle().published &&
                    program_.physical_usage().occupied.main_kv_pages + 1 == before.main_kv_pages &&
                    program_.physical_usage().occupied.host_bytes == first_quote->host_bytes,
                "demotion changed resources outside its selected physical owner");
        require(second_quote->host_bytes > 0 && program_.start_demote(*second_quote),
                "demoting one independent owner invalidated the other physical action");
        require(settle().published && program_.physical_usage().occupied.main_kv_pages == 0 &&
                    program_.physical_usage().occupied.host_bytes ==
                        first_quote->host_bytes + second_quote->host_bytes,
                "independent demotions did not account for their separate Host allocations");
        require(program_.release_checkpoint(first) && program_.release_checkpoint(second),
                "independent demotions retained a completed checkpoint reference");
        expect_empty("independent demotions leaked physical resources");
    }

    void demotion_holder_changes() {
        for (const std::uint32_t source_tokens : {64U, 65U}) {
            auto seed         = request(source_tokens, true);
            const auto source = checkpoint(seed);
            const std::array source_set{source};
            const auto full_plan = program_.plan_reclaim(source_set, {}, {.main_kv_pages = 2});
            const auto* full     = demotion_with_sources(full_plan.demotions, source_set);
            const auto tail_plan = program_.plan_reclaim(source_set, {}, {.main_kv_pages = 1});
            const auto* tail     = demotion_with_sources(tail_plan.demotions, source_set);
            require(full && tail, "holder-change fixture has no original demotion quote");

            auto extension       = request(source_tokens + 64, true);
            const auto candidate = program_.inspect_source(extension, source);
            require(candidate &&
                        static_cast<bool>(program_.start_binding(extension, {0}, *candidate)),
                    "holder-change fixture could not fork its retained prefix");
            const auto bound = settle();
            require(bound.sequence.has_value(), "holder-change prefix fork did not publish");
            finish_prefill(*bound.sequence, 64, true);
            const auto finished = program_.finish(*bound.sequence);
            require(finished.checkpoint.has_value(), "holder-change descendant was not retained");
            const auto descendant = *finished.checkpoint;
            const auto before     = program_.physical_usage();
            require(!program_.start_demote(*full) && !program_.has_context_transaction(),
                    "an old quote ignored a newly retained holder of its shared full page");
            expect_usage(before, "stale-holder rejection changed resources before validation");

            if (source_tokens % 64) {
                // The descendant's COW tail has another handle at the same logical position.
                // It changes the full page's holder set but does not hold the original tail.
                require(program_.start_demote(*tail) && settle().published &&
                            program_.physical_usage().occupied.main_kv_pages + 1 ==
                                before.occupied.main_kv_pages,
                        "COW at the same position invalidated an independent original tail");
            }
            const std::array family{source, descendant};
            const auto releases = program_.plan_releases(family, {}, {.main_kv_pages = 1});
            const auto family_release =
                std::find_if(releases.begin(), releases.end(), [&](const auto& quote) {
                    return quote.sources.size() == 2 &&
                           std::find(quote.sources.begin(), quote.sources.end(), source) !=
                               quote.sources.end() &&
                           std::find(quote.sources.begin(), quote.sources.end(), descendant) !=
                               quote.sources.end();
                });
            require(family_release != releases.end() &&
                        family_release->released.main_kv_pages ==
                            program_.physical_usage().occupied.main_kv_pages -
                                empty_.occupied.main_kv_pages,
                    "family release did not combine unique suffixes with their shared prefix");
            require(program_.release_checkpoint(source) && program_.release_checkpoint(descendant),
                    "holder-change fixture retained a finished checkpoint");
            expect_empty("holder-change validation leaked physical resources");
        }
    }

    void batched_physical_demotions() {
        auto first_base   = request(64, true);
        auto second_base  = request(192, true);
        const auto first  = checkpoint(first_base);
        const auto second = checkpoint(second_base);
        const std::array<qwen::CheckpointHandle, 2> allowed{first, second};
        const std::array<qwen::CheckpointHandle, 1> first_owner{first}, second_owner{second};
        const auto main_plan     = program_.plan_reclaim(allowed, {}, {.main_kv_pages = 4});
        const auto& candidates   = main_plan.demotions;
        const auto* first_quote  = demotion_with_sources(candidates, first_owner);
        const auto* second_quote = demotion_with_sources(candidates, second_owner);
        require(first_quote && second_quote && first_quote->released.main_kv_pages == 1 &&
                    second_quote->released.main_kv_pages == 3,
                "batch fixture did not produce independent one-page and three-page actions");
        const std::array ordered{*first_quote, *second_quote};
        const auto complete_batch = demotion_batch(main_plan, ordered, {.main_kv_pages = 4});
        const auto complete       = complete_batch.finish();
        const auto clipped = demotion_batch(main_plan, ordered, {.main_kv_pages = 2}).finish();
        require(complete && complete->released.main_kv_pages == 4 && clipped &&
                    clipped->released.main_kv_pages == 2 && clipped->sources.size() == 2,
                "batch did not clip the last physical action to the requested shortage");
        const std::array clipped_only{*clipped};
        require(batch_covers(complete_batch, main_plan.releases, first_owner) &&
                    batch_covers(complete_batch, main_plan.releases, second_owner) &&
                    !batch_covers(demotion_batch(main_plan, clipped_only, {.main_kv_pages = 4}),
                                  main_plan.releases, second_owner),
                "matching owners concealed deletion pages absent from the demotion batch");
        const auto backend_pages = program_.physical_usage().occupied.backend_kv_pages;
        if (backend_pages) {
            const auto backend_plan =
                program_.plan_reclaim(allowed, {}, {.backend_kv_pages = backend_pages});
            const auto backend_batch = demotion_batch(backend_plan, backend_plan.demotions,
                                                      {.backend_kv_pages = backend_pages});
            require(batch_covers(backend_batch, backend_plan.releases, first_owner) &&
                        batch_covers(backend_batch, backend_plan.releases, second_owner),
                    "complete Backend demotions did not cover their actual deletion pages");
        }
        const auto baseline = program_.physical_usage();
        require(program_.start_demote(*clipped) && program_.has_context_transaction(),
                "independent page actions did not start one asynchronous demotion");
        const auto cancelled = settle(true);
        require(!cancelled.published && cancelled.operations.pressure_spill_pages == 2 &&
                    transferred_pages(cancelled, runtime::ContextTransferDirection::DeviceToHost,
                                      runtime::ContextResourceClass::MainKV) == 2,
                "cancelled batch did not settle its two submitted Main page copies");
        expect_usage(baseline,
                     "cancelled batch leaked reservations or dropped uncopied Device pages");

        require(program_.start_demote(*clipped),
                "cancelled batch could not reacquire its complete destination set");
        const auto demoted = settle();
        require(demoted.published && demoted.operations.pressure_spill_pages == 2 &&
                    transferred_pages(demoted, runtime::ContextTransferDirection::DeviceToHost,
                                      runtime::ContextResourceClass::MainKV) == 2 &&
                    program_.physical_usage().occupied.main_kv_pages + 2 ==
                        baseline.occupied.main_kv_pages &&
                    program_.physical_usage().occupied.host_bytes ==
                        baseline.occupied.host_bytes + clipped->host_bytes &&
                    program_.checkpoint_matches(first, first_base) &&
                    program_.checkpoint_matches(second, second_base),
                "one batch did not publish exactly two Main pages while preserving both histories");
        const auto after = program_.physical_usage();
        require(!program_.start_demote(*complete) && !program_.has_context_transaction(),
                "stale batch applied an action after one of its Device sources changed");
        expect_usage(after, "rejected stale batch partially allocated or released resources");
        const auto remaining = program_.plan_reclaim(allowed, {}, {.main_kv_pages = 4});
        const auto remaining_batch =
            demotion_batch(remaining, remaining.demotions, {.main_kv_pages = 4});
        require(batch_covers(remaining_batch, remaining.releases, second_owner) &&
                    !batch_covers(remaining_batch, remaining.releases, first_owner),
                "coverage counted Host-only pages as a Device deletion release");
        require(program_.release_checkpoint(first) && program_.release_checkpoint(second),
                "batched demotion retained a completed checkpoint reference");
        expect_empty("batched demotion leaked physical resources");
    }

    void physical_facts() {
        auto seed         = request(64, true);
        const auto source = checkpoint(seed);
        const std::array<qwen::CheckpointHandle, 1> source_set{source};
        const auto state_candidates =
            program_.plan_reclaim(source_set, {}, {.state_slots = 1}).demotions;
        const auto main_candidates =
            program_.plan_reclaim(source_set, {}, {.main_kv_pages = 1}).demotions;
        const auto* state_quote = demotion_with_sources(state_candidates, source_set);
        const auto* main_quote  = demotion_with_sources(main_candidates, source_set);
        require(state_quote && state_quote->host_bytes > 0 && main_quote &&
                    main_quote->host_bytes > 0 && main_quote->released.main_kv_pages == 1,
                "device-only checkpoint did not report its selected Host allocation demand");
        const auto main_bytes = main_quote->host_bytes;
        require(program_.start_demote(*main_quote),
                "shared-prefix fact fixture could not demote its selected Main page");
        require(settle().published, "shared-prefix fact fixture lost its demotion");
        require(program_.physical_usage().occupied.host_bytes == main_bytes,
                "demotion quote disagrees with its actual Host allocation");
        require(program_.host_bytes_released(source_set) == main_bytes,
                "unique checkpoint Host bytes were not fully releasable");

        auto extension = request(128, true);
        auto candidate = program_.inspect_source(extension, source);
        require(candidate && static_cast<bool>(program_.start_binding(extension, {1}, *candidate)),
                "shared-prefix fact fixture could not bind its descendant");
        auto bound = settle();
        require(bound.sequence.has_value(), "shared-prefix descendant did not publish");
        const auto sequence = *bound.sequence;
        const auto borrowed = program_.checkpoint_summary(source);
        require(program_.checkpoint_footprint(source_set).main_kv_pages == 1 &&
                    borrowed.evictable_resources.main_kv_pages == 0 &&
                    borrowed.evictable_resources.state_slots == 0,
                "active borrowed State/KV was exposed as an evictable Device replica");
        require(program_.plan_reclaim(source_set, {}, {.state_slots = 1}).demotions.empty() &&
                    program_.plan_reclaim(source_set, {}, {.main_kv_pages = 1}).demotions.empty(),
                "active borrowed content was exposed as a demotion candidate");
        require(program_.release_redundant_host(source_set) == 0,
                "Host replica cleanup ignored an excluded source");
        require(program_.host_bytes_released(source_set) == 0,
                "deleting one shared prefix claimed the descendant's Host page");

        finish_prefill(sequence, 64, true);
        auto finished = program_.finish(sequence);
        require(finished.checkpoint.has_value(), "descendant did not retain a complete checkpoint");
        const auto descendant = *finished.checkpoint;
        const std::array<qwen::CheckpointHandle, 2> family{source, descendant};
        const std::array<qwen::CheckpointHandle, 3> repeated{source, descendant, source};
        require(program_.host_bytes_released(source_set) == 0 &&
                    program_.host_bytes_released(family) == main_bytes &&
                    program_.host_bytes_released(repeated) == main_bytes,
                "fixed-set Host accounting double-counted shared or duplicate references");
        const auto main_releases = program_.plan_releases(family, {}, {.main_kv_pages = 1});
        const auto expected_main =
            program_.physical_usage().occupied.main_kv_pages - empty_.occupied.main_kv_pages;
        require(std::any_of(main_releases.begin(), main_releases.end(),
                            [&](const auto& quote) {
                                return quote.sources.size() == family.size() &&
                                       std::all_of(family.begin(), family.end(),
                                                   [&](auto handle) {
                                                       return std::find(quote.sources.begin(),
                                                                        quote.sources.end(),
                                                                        handle) !=
                                                              quote.sources.end();
                                                   }) &&
                                       quote.released.main_kv_pages == expected_main;
                            }),
                "Main release quote did not account the last physical aliases exactly");
        const auto family_plan        = program_.plan_reclaim(family, {}, {.main_kv_pages = 2});
        const auto& family_candidates = family_plan.demotions;
        const auto* shared_page       = demotion_with_sources(family_candidates, family);
        const std::array<qwen::CheckpointHandle, 1> descendant_set{descendant};
        const auto* private_tail = demotion_with_sources(family_candidates, descendant_set);
        require(program_.plan_reclaim(source_set, {}, {.main_kv_pages = 1}).demotions.empty() &&
                    shared_page && shared_page->released.main_kv_pages == 1 &&
                    shared_page->host_bytes == 0 && private_tail &&
                    private_tail->released.main_kv_pages == 1 && private_tail->host_bytes > 0,
                "shared Main page was reclaimable without every retained alias in allowed");
        const std::array tail_only{*private_tail};
        const auto family_batch =
            demotion_batch(family_plan, family_candidates, {.main_kv_pages = 2});
        const auto tail_batch = demotion_batch(family_plan, tail_only, {.main_kv_pages = 2});
        require(batch_covers(family_batch, family_plan.releases, family) &&
                    batch_covers(tail_batch, family_plan.releases, descendant_set) &&
                    !batch_covers(tail_batch, family_plan.releases, family) &&
                    !batch_covers(family_batch, family_plan.releases, source_set),
                "demotion coverage confused shared holders with pages actually freed by deletion");
        const auto source_frontier     = program_.checkpoint_metadata(source).frontier;
        const auto descendant_frontier = program_.checkpoint_metadata(descendant).frontier;
        require(family_plan.recovery_loss(descendant_set, source_set) ==
                        descendant_frontier - source_frontier &&
                    family_plan.recovery_loss(family, {}) == descendant_frontier &&
                    family_plan.recovery_loss(descendant_set, source_set) ==
                        descendant_frontier - source_frontier,
                "reused prefix facts confused different surviving recovery sets");
        const auto unprotected_suffix =
            program_.plan_reclaim(family, source_set, {.main_kv_pages = 1}).demotions;
        require(std::all_of(unprotected_suffix.begin(), unprotected_suffix.end(),
                            [&](const auto& quote) {
                                return quote.sources ==
                                           std::vector<qwen::CheckpointHandle>{descendant} &&
                                       quote.host_bytes > 0;
                            }),
                "demotion exposed an excluded Device/Host prefix through its descendant alias");
        const auto baseline = program_.physical_usage();
        candidate           = program_.inspect_source(extension, source);
        require(candidate && static_cast<bool>(program_.start_binding(extension, {0}, *candidate)),
                "pinned replica fixture could not start binding");
        require(program_.release_redundant_host({}) == 0 &&
                    program_.plan_reclaim(family, {}, {.main_kv_pages = 1}).demotions.empty() &&
                    program_.plan_reclaim(family, {}, {.state_slots = 1}).demotions.empty() &&
                    program_.plan_releases(family, {}, {.main_kv_pages = 1}).empty() &&
                    program_.plan_releases(family, {}, {.backend_kv_pages = 1}).empty() &&
                    program_.host_bytes_released(family) == 0,
                "replica cleanup or fixed-set release ignored a pending reader's record pin");
        require(!settle(true).published, "pinned replica fixture did not cancel binding");
        expect_usage(baseline, "pinned replica checks changed physical ownership");

        // The first action shares both holders; the second shares only the descendant.
        // Their physical pages are distinct, and only the private tail needs a new Host copy.
        const std::array overlapping_holders{*shared_page, *private_tail};
        const auto mixed_plan = program_.plan_reclaim(family, {}, {.main_kv_pages = 2});
        const auto mixed_batch =
            demotion_batch(mixed_plan, overlapping_holders, {.main_kv_pages = 2}).finish();
        require(mixed_batch && mixed_batch->released.main_kv_pages == 2 &&
                    mixed_batch->sources.size() == family.size() &&
                    mixed_batch->host_bytes == private_tail->host_bytes &&
                    program_.start_demote(*mixed_batch),
                "overlapping holders prevented batching distinct shared and private pages");
        const auto mixed_cancelled = settle(true);
        require(
            !mixed_cancelled.published && mixed_cancelled.operations.pressure_spill_pages == 1 &&
                transferred_pages(mixed_cancelled, runtime::ContextTransferDirection::DeviceToHost,
                                  runtime::ContextResourceClass::MainKV) == 1 &&
                program_.physical_usage().occupied.main_kv_pages + 1 ==
                    baseline.occupied.main_kv_pages &&
                program_.physical_usage().occupied.host_bytes == baseline.occupied.host_bytes &&
                program_.physical_usage().host_reserved_bytes == baseline.host_reserved_bytes &&
                program_.checkpoint_matches(source, seed) &&
                program_.checkpoint_matches(descendant, extension),
            "mixed batch cancellation lost a complete history or retained its new Host copy");
        candidate = program_.inspect_source(extension, source);
        require(candidate && static_cast<bool>(program_.start_binding(extension, {0}, *candidate)),
                "cancelled mixed batch lost its retained Host prefix");
        // Main KV lands layer by layer after the binding and is reported with the first pass;
        // releasing the lane before any pass waits for the copies.
        const auto restored = settle();
        require(restored.sequence &&
                    !transferred(restored, runtime::ContextTransferDirection::HostToDevice,
                                 runtime::ContextResourceClass::MainKV) &&
                    program_.abort(*restored.sequence).status == runtime::ConsumeStatus::Consumed,
                "cancelled mixed batch could not restore its released Device replica");
        expect_usage(baseline, "mixed batch restore retained an extra writer or destination");

        const auto release_and_check = [&](qwen::CheckpointHandle handle) {
            const std::array one{handle};
            const auto quoted_pages = [&](runtime::ContextResourceUsage shortage, bool main_pool) {
                const auto quotes    = program_.plan_releases(one, {}, shortage);
                std::uint32_t result = 0;
                for (const auto& quote : quotes) {
                    require(quote.sources == std::vector<qwen::CheckpointHandle>{handle},
                            "single-checkpoint release included another history");
                    result = std::max(result, main_pool ? quote.released.main_kv_pages
                                                        : quote.released.backend_kv_pages);
                }
                return result;
            };
            const auto main_pages    = quoted_pages({.main_kv_pages = 1}, true);
            const auto backend_pages = quoted_pages({.backend_kv_pages = 1}, false);
            const auto before        = program_.physical_usage().occupied;
            require(program_.release_checkpoint(handle), "quoted checkpoint could not be released");
            const auto after = program_.physical_usage().occupied;
            require(before.main_kv_pages == after.main_kv_pages + main_pages &&
                        before.backend_kv_pages == after.backend_kv_pages + backend_pages,
                    "typed release quantities disagree with actual physical page release");
        };
        // MTP's 63-column Backend prefix has a copied tail in its descendant. Only Main
        // induces the shared family action above; each deletion still has an exact quote.
        release_and_check(source);
        require(program_.physical_usage().occupied.host_bytes == main_bytes,
                "releasing one ancestor prematurely freed shared Host storage");
        const std::array<qwen::CheckpointHandle, 1> remaining{descendant};
        require(program_.host_bytes_released(remaining) == main_bytes,
                "last descendant did not account the uniquely retained Host page");
        release_and_check(descendant);
        expect_empty("fixed-set Host accounting fixture leaked physical resources");

        auto larger               = request(2 * 64 + 1, true);
        const auto partial_source = checkpoint(larger);
        const std::array<qwen::CheckpointHandle, 1> partial_set{partial_source};
        const auto before_partial = program_.physical_usage();
        const auto partial_candidates =
            program_.plan_reclaim(partial_set, {}, {.main_kv_pages = 1}).demotions;
        const auto* partial_quote = demotion_with_sources(partial_candidates, partial_set);
        require(partial_quote && partial_quote->released.main_kv_pages == 1 &&
                    program_.start_demote(*partial_quote),
                "one-page shortage did not produce a bounded demotion");
        const auto partial_progress = settle();
        const auto after_partial    = program_.physical_usage();
        require(partial_progress.published &&
                    after_partial.occupied.main_kv_pages + 1 ==
                        before_partial.occupied.main_kv_pages &&
                    after_partial.occupied.host_bytes == partial_quote->host_bytes &&
                    program_.checkpoint_matches(partial_source, larger),
                "partial demotion moved excess KV or invalidated its complete recovery point");
        const auto partial_candidate = program_.inspect_source(larger, partial_source);
        require(partial_candidate &&
                    std::count_if(
                        partial_candidate->transfers.begin(), partial_candidate->transfers.end(),
                        [](const auto& transfer) {
                            return transfer.resource == runtime::ContextResourceClass::MainKV &&
                                   transfer.direction ==
                                       runtime::ContextTransferDirection::HostToDevice &&
                                   transfer.page_count == 1;
                        }) == 1,
                "mixed Device/Host recovery did not request exactly its missing Main page");
        require(program_.release_checkpoint(partial_source),
                "partially demoted recovery point did not release its references");
        expect_empty("partial demotion retained physical resources");

        auto pause_base     = request(2 * kChunk);
        const auto to_pause = bind(pause_base);
        prefill_prefix(to_pause, kChunk);
        const auto pause_bytes = program_.pause_host_bytes(to_pause);
        require(pause_bytes && *pause_bytes > 0 && program_.start_pause(to_pause, true),
                "snapshot fact fixture could not reserve its Host demand");
        auto paused = settle();
        require(paused.paused && paused.paused->snapshot_handle(),
                "snapshot fact fixture did not preserve its checkpoint identity");
        const auto resources = program_.snapshot_resources(*paused.paused);
        require(resources.state_slots == 0 && resources.main_kv_pages == 0 &&
                    resources.backend_kv_pages == 0 && resources.host_bytes == *pause_bytes &&
                    resources.host_bytes == program_.physical_usage().occupied.host_bytes,
                "paused snapshot typed resources disagree with the actual saved replicas");
        require(program_.revoke_snapshot(*paused.paused) && !paused.paused->snapshot_handle() &&
                    program_.snapshot_resources(*paused.paused) == runtime::ContextResourceUsage{},
                "revoked snapshot retained an identity or reclaimable resource claim");
        paused.paused.reset();
        expect_empty("snapshot fact fixture leaked physical resources");
    }

    void demote_to_host(qwen::CheckpointHandle source) {
        const std::array<qwen::CheckpointHandle, 1> allowed{source};
        for (const auto resource :
             {runtime::ContextResourceClass::State, runtime::ContextResourceClass::MainKV,
              runtime::ContextResourceClass::BackendKV}) {
            const auto usage = program_.checkpoint_footprint(allowed);
            runtime::ContextResourceUsage shortage;
            switch (resource) {
            case runtime::ContextResourceClass::State:
                shortage.state_slots = usage.state_slots;
                break;
            case runtime::ContextResourceClass::MainKV:
                shortage.main_kv_pages = usage.main_kv_pages;
                break;
            case runtime::ContextResourceClass::BackendKV:
                shortage.backend_kv_pages = usage.backend_kv_pages;
                break;
            }
            if (!shortage.state_slots && !shortage.main_kv_pages && !shortage.backend_kv_pages) {
                continue;
            }
            const auto candidates = program_.plan_reclaim(allowed, {}, shortage).demotions;
            const auto* quote     = demotion_with_sources(candidates, allowed);
            require(quote && program_.start_demote(*quote),
                    "image fixture could not demote its checkpoint to Host");
            require(settle().published, "image fixture demotion did not complete");
        }
        const auto left = program_.checkpoint_footprint(allowed);
        require(!left.state_slots && !left.main_kv_pages && !left.backend_kv_pages,
                "image fixture checkpoint kept Device replicas after demotion");
    }

    // A checkpoint image is the exact Host representation of its points: exporting a Device
    // checkpoint, the same checkpoint after Host demotion, and the checkpoint rebuilt from the image
    // gives identical bytes, KV pages and StateImages included. The rebuilt point is Host-resident,
    // findable under the original key, and resumes the request it came from.
    void checkpoint_image_round_trip() {
        constexpr std::string_view kBinding = "native-transactions-image";
        auto base             = request(kPromptTokens, true);
        const auto original   = checkpoint(base);
        const std::array originals{original};
        const auto from_device = program_.export_checkpoints(originals, std::nullopt, kBinding);
        const auto frontier    = program_.checkpoint_metadata(original).frontier;
        require(from_device.tokens == frontier && from_device.keys.size() == 1 &&
                    from_device.keys.front() == program_.checkpoint_key(original, frontier) &&
                    from_device.prefix_digests.size() == frontier + 1U &&
                    !from_device.regions.empty(),
                "exported image does not describe its endpoint");
        std::size_t payload = 0;
        for (const auto& region : from_device.regions) { payload += region.length; }
        require(payload > from_device.bytes.size() / 2, "image regions miss its payload");
        // One point names one StateImage, the last region.
        std::cout << "checkpoint image: " << from_device.bytes.size() << " bytes for " << frontier
                  << " tokens, StateImage " << from_device.regions.back().length << " bytes\n";

        demote_to_host(original);
        const auto from_host = program_.export_checkpoints(originals, std::nullopt, kBinding);
        require(from_host.bytes == from_device.bytes,
                "Device and Host replicas of one checkpoint exported different bytes");

        qwen::PreparedSessionKey session;
        session.size = 7;
        std::copy_n("restore", 7, session.bytes.begin());
        const auto keyed = program_.export_checkpoints(originals, session, kBinding);
        require(program_.release_checkpoint(original), "image fixture could not release source");
        expect_empty("released image source leaked typed resources");

        bool rejected = false;
        try {
            (void)program_.import_checkpoints(keyed.bytes, "another-model");
        } catch (const std::invalid_argument&) { rejected = true; }
        require(rejected, "an image was imported under a different model binding");
        auto damaged = keyed.bytes;
        damaged[keyed.regions.front().offset + 17] ^= 0x5a;
        rejected = false;
        try {
            (void)program_.import_checkpoints(damaged, kBinding);
        } catch (const std::invalid_argument&) { rejected = true; }
        require(rejected, "a damaged image was imported");
        // Another format version, or a configuration this Program does not have (here the
        // speculative backend, the second configuration field after the magic, version and
        // binding), is refused for that reason, before the checksum is consulted.
        const auto rejection = [&](std::size_t offset) {
            auto altered = keyed.bytes;
            altered[offset] ^= 0x01;
            try {
                (void)program_.import_checkpoints(altered, kBinding);
            } catch (const std::invalid_argument& error) { return std::string(error.what()); }
            return std::string();
        };
        constexpr std::size_t kVersionOffset = 8;
        const std::size_t config_offset =
            kVersionOffset + 2 * sizeof(std::uint32_t) + kBinding.size();
        require(rejection(kVersionOffset).find("version") != std::string::npos,
                "an image of another format version was not refused as such");
        require(rejection(config_offset + sizeof(std::uint32_t)).find("configuration") !=
                    std::string::npos,
                "an image of another speculative configuration was not refused as such");
        expect_empty("rejected imports changed typed resources");

        const auto needed = program_.checkpoint_image_host_bytes(keyed.bytes, kBinding);
        auto imported     = program_.import_checkpoints(keyed.bytes, kBinding);
        require(imported && imported->points.size() == 1 && imported->session == session,
                "image did not import its single point and session key");
        const auto restored = imported->points.front();
        const std::array restored_points{restored};
        const auto footprint = program_.checkpoint_footprint(restored_points);
        require(program_.checkpoint_metadata(restored).frontier == frontier &&
                    program_.checkpoint_key(restored, frontier) == from_device.keys.front() &&
                    footprint.state_slots == 0 && footprint.main_kv_pages == 0 &&
                    footprint.backend_kv_pages == 0 && footprint.host_bytes != 0 &&
                    program_.physical_usage().occupied.host_bytes <= needed,
                "imported checkpoint is not a Host-resident copy of the original");
        const auto again = program_.export_checkpoints(restored_points, session, kBinding);
        require(again.bytes == keyed.bytes, "re-exporting an imported checkpoint changed bytes");

        // The restored point resumes a continuation of its prompt where the original stopped.
        auto next      = request(kPromptTokens + kChunk, true);
        auto candidate = program_.inspect_source(next, restored);
        require(candidate && candidate->reused_tokens == frontier,
                "imported checkpoint is not a recovery source for its continuation");
        require(static_cast<bool>(program_.start_binding(next, {0}, *candidate)),
                "imported checkpoint could not bind");
        const auto bound = settle();
        require(bound.published && bound.sequence &&
                    !transferred(bound, runtime::ContextTransferDirection::HostToDevice,
                                 runtime::ContextResourceClass::MainKV),
                "imported checkpoint binding waited for its Main KV restore");
        landed_.clear();
        finish_prefill(*bound.sequence, next.summary().prompt_tokens - frontier, true);
        require(landed_from_host(), "imported checkpoint did not restore Main KV from Host");
        auto finished = program_.finish(*bound.sequence);
        require(finished.status == runtime::ConsumeStatus::Consumed,
                "request resumed from an imported checkpoint did not finish");
        if (finished.checkpoint) { require(program_.release_checkpoint(*finished.checkpoint), ""); }
        if (program_.valid_checkpoint(restored)) {
            require(program_.release_checkpoint(restored), "imported checkpoint stayed leased");
        }
        expect_empty("imported checkpoint leaked typed resources");
    }

    // A public shared prefix travels as an image of its own (no session, no private point beside
    // it) and comes back as a shared prefix: same bytes on re-export, same key, still a source.
    void shared_prefix_image_round_trip() {
        constexpr std::string_view kBinding = "native-transactions-image";
        auto marked         = marker_request();
        const auto sequence = bind(marked.base);
        reserve(sequence, qwen::ExecutionUnitKind::Prefill);
        const auto step = program_.advance_prefill(sequence);
        require(step.capture_ready, "shared image fixture did not reach its first marker");
        const auto frontier = marked.frontiers.front();
        const auto shared   = capture(sequence, frontier, runtime::CheckpointRole::SharedPrefix);
        require(program_.abort(sequence).status == runtime::ConsumeStatus::Consumed,
                "shared image fixture did not release its lane");
        const std::array points{shared};
        qwen::PreparedSessionKey session;
        session.size = 1;
        session.bytes[0] = 's';
        bool rejected = false;
        try {
            (void)program_.export_checkpoints(points, session, kBinding);
        } catch (const std::invalid_argument&) { rejected = true; }
        require(rejected, "a shared prefix was exported with a conversation's session key");
        const auto image = program_.export_checkpoints(points, std::nullopt, kBinding);
        require(image.keys.size() == 1 && image.keys.front().frontier == frontier,
                "shared prefix image does not describe its endpoint");
        require(program_.release_checkpoint(shared), "shared image source retained a reader");
        expect_empty("released shared image source leaked typed resources");
        auto imported = program_.import_checkpoints(image.bytes, kBinding);
        require(imported && imported->points.size() == 1 && !imported->session,
                "shared prefix image did not import as one point without a session");
        const auto restored = imported->points.front();
        require(program_.checkpoint_metadata(restored).role == runtime::CheckpointRole::SharedPrefix &&
                    program_.checkpoint_key(restored, frontier) == image.keys.front(),
                "imported shared prefix lost its role or key");
        const std::array restored_points{restored};
        require(program_.export_checkpoints(restored_points, std::nullopt, kBinding).bytes ==
                    image.bytes,
                "re-exporting an imported shared prefix changed bytes");
        auto again = marker_request();
        require(static_cast<bool>(program_.inspect_source(again.base, restored)),
                "imported shared prefix is not a source for a prompt that starts with it");
        require(program_.release_checkpoint(restored), "imported shared prefix stayed leased");
        expect_empty("imported shared prefix leaked typed resources");
    }

    void grammar_row_failure() {
        struct Masks final : runtime::TokenMaskProvider {
            void uploaded(std::size_t, std::size_t) noexcept override {}

            bool constrained(std::size_t row) const noexcept override { return row == 0; }

            std::uint32_t fill(std::size_t, std::span<const TokenId> drafts,
                               std::span<std::uint32_t> words) override {
                std::fill(words.begin(), words.end(), 0);
                const auto stride = words.size() / (drafts.size() + 1);
                for (std::size_t col = 0; col <= drafts.size(); ++col) words[col * stride] = 1;
                return 1; // The first predicted position is a real dead end.
            }
        } masks;

        auto base        = request(32);
        const auto first = bind(base, 0);
        finish_prefill(first, 32, false);
        const auto second = bind(base, 1);
        finish_prefill(second, 32, false);
        const std::array<qwen::SequenceHandle, 2> members{first, second};
        const std::array<qwen::ExecutionUnit, 2> units{
            {{first, qwen::ExecutionUnitKind::Decode, 1},
             {second, qwen::ExecutionUnitKind::Decode, 1}}};
        require(static_cast<bool>(program_.reserve_units(units)), "grammar mixed unit reservation");
        const std::array<runtime::RoundBudget, 2> budgets{{{1}, {1}}};
        auto pending = program_.decode(members, budgets, nullptr, &masks);
        require(pending.constraint_failed(0) && !pending.constraint_failed(1),
                "grammar failure lost its row");
        const std::array<runtime::CommitDecision, 2> decisions{
            {{.terminal = true, .failed = true}, {.accepted_tokens = 1}}};
        auto committed = program_.commit(std::move(pending), decisions);
        require(committed.rows[0].disposition == runtime::CommitDisposition::FailedReleased &&
                    committed.rows[1].disposition == runtime::CommitDisposition::Active,
                "grammar failure contaminated a healthy row");
        require(program_.abort(second).status == runtime::ConsumeStatus::Consumed,
                "healthy row was released by another row's grammar error");
        expect_empty("grammar row failure leaked physical resources");
    }


private:
    DeviceContext& device_;
    qwen::Program& program_;
    qwen::Frontend& frontend_;
    qwen::PhysicalUsageSnapshot empty_;
};

void shared_capture_alignment(DeviceContext& device, const qwen::execution::Parameters& parameters,
                              qwen::Frontend& frontend, EngineOptions options) {
    options.max_context                       = 128;
    options.kv_capacity                       = KvCapacityPolicy::explicit_capacity(256);
    options.max_concurrency                   = 2;
    options.context_cache.enabled             = true;
    options.context_cache.device_state_slots  = 2;
    options.context_cache.host_capacity_bytes = 0;
    auto planner = qwen::detail::make_sequence_planner_impl(parameters, device, options);
    auto plan    = qwen::detail::finalize_sequence_plan_impl(std::move(planner), 4);
    qwen::detail::ProgramImpl program(parameters, *plan, device, {});
    const auto empty = program.physical_usage().occupied;

    const auto settle = [&](bool cancelled = false) {
        std::atomic<bool> flag{cancelled};
        for (unsigned step = 0; step < 8; ++step) {
            device.synchronize();
            CUDA_CHECK(cudaStreamSynchronize(device.transfer_stream));
            auto progress = program.poll_context({&flag});
            if (progress.complete) { return progress; }
        }
        throw std::runtime_error("aligned capture transaction did not settle");
    };
    const auto advance = [&](qwen::SequenceHandle sequence) {
        const std::array<qwen::ExecutionUnit, 1> units{
            {{sequence, qwen::ExecutionUnitKind::Prefill}}};
        require(static_cast<bool>(program.reserve_units(units)),
                "aligned capture could not reserve its prefill unit");
        auto step = program.advance_prefill(sequence, nullptr, nullptr);
        if (step.pending) {
            const std::array<runtime::CommitDecision, 1> accepted{{{.accepted_tokens = 1}}};
            const auto committed = program.commit(std::move(*step.pending), accepted, {}, nullptr);
            step.capture_ready   = committed.capture_ready[0];
        }
        return step;
    };
    for (const std::uint32_t frontier : {64U, 65U}) {
        for (const bool shared_input : {false, true}) {
            // Exercise the Native contract with exact token positions. Frontend marker parsing
            // is covered separately; these boundaries must not depend on a chat template.
            const auto prompt_tokens = frontier + (shared_input ? 0U : 16U);
            auto prepared            = qwen::PreparedPromptAccess::take(
                frontend.prepare_tokens(std::vector<TokenId>(prompt_tokens, 198)));
            prepared.context_cache.opportunities.push_back(
                {.kind     = PromptCacheMarkerKind::SharedStablePrefix,
                 .evidence = SharedCandidateEvidence::ExplicitBoundary,
                 .frontier = frontier});
            runtime::ResolvedExecutionOptions request;
            request.requested_output_tokens = 8;
            request.allow_prefix_reuse      = true;
            auto base                       = program.plan_request(std::move(prepared), request);
            for (const bool cancel_capture : {true, false}) {
                const auto root = program.inspect_source(base, std::nullopt);
                require(root &&
                            static_cast<bool>(program.start_binding(
                                base, {0}, *root, nullptr, qwen::ExecutionUnitKind::Prefill, 0)),
                        "aligned capture could not bind its source writer");
                const auto bound = settle();
                require(bound.sequence.has_value(), "aligned capture source did not publish");
                const auto original = *bound.sequence;
                const auto step     = advance(original);
                require(step.processed_prompt_tokens == frontier && step.capture_ready &&
                            step.complete == shared_input,
                        "aligned Shared capture was skipped or crossed its frontier");
                require(program.start_capture(original),
                        "aligned Shared capture could not reserve its destination");
                const auto captured = settle(cancel_capture);
                if (cancel_capture) {
                    require(!captured.published && captured.captured_checkpoints.empty(),
                            "cancelled aligned capture published a checkpoint");
                } else {
                    require(captured.published &&
                                captured.captured_checkpoints.size() == (shared_input ? 2U : 1U),
                            "aligned capture dropped its Shared or InputReplay role");
                    std::optional<qwen::CheckpointHandle> shared;
                    for (const auto point : captured.captured_checkpoints) {
                        const auto summary = program.checkpoint_summary(point);
                        require(summary.frontier == frontier,
                                "aligned capture published the wrong frontier");
                        if (summary.role == runtime::CheckpointRole::SharedPrefix) {
                            shared = point;
                        } else {
                            require(shared_input &&
                                        summary.role == runtime::CheckpointRole::InputReplay,
                                    "aligned capture published an unexpected role");
                        }
                    }
                    const bool main_copy = frontier % 64 != 0;
                    const auto backend_frontier =
                        frontier -
                        (options.speculative.backend == SpeculativeBackend::Mtp ? 1U : 0U);
                    const bool backend_copy =
                        (options.speculative.backend == SpeculativeBackend::Mtp ||
                         options.speculative.backend == SpeculativeBackend::DFlash) &&
                        backend_frontier % 64 != 0;
                    require(
                        shared &&
                            captured.operations.partial_tail_cow_pages ==
                                static_cast<unsigned>(main_copy) +
                                    static_cast<unsigned>(backend_copy) &&
                            transferred(captured, runtime::ContextTransferDirection::DeviceToDevice,
                                        runtime::ContextResourceClass::MainKV) == main_copy &&
                            transferred(captured, runtime::ContextTransferDirection::DeviceToDevice,
                                        runtime::ContextResourceClass::BackendKV) == backend_copy,
                        "capture did not independently copy the required Main/backend tails");
                    require(
                        program.checkpoint_footprint(captured.captured_checkpoints).state_slots ==
                            1,
                        "same-frontier capture allocated a State per logical role");

                    // A real independent writer must adopt the saved prefix and compute only
                    // its suffix while the original writer and both logical roles remain live.
                    const auto source = program.inspect_source(base, *shared);
                    require(
                        source && source->reused_tokens == frontier &&
                            static_cast<bool>(program.start_binding(
                                base, {1}, *source, nullptr, qwen::ExecutionUnitKind::Prefill, 0)),
                        "captured aligned Shared prefix could not bind another request");
                    const auto reused = settle();
                    require(reused.sequence.has_value(), "Shared prefix binding did not publish");
                    const auto suffix = advance(*reused.sequence);
                    require(suffix.complete &&
                                suffix.processed_prompt_tokens == prompt_tokens - frontier,
                            "Shared prefix binding repeated or skipped the saved input");
                    if (suffix.capture_ready) { program.skip_capture(*reused.sequence); }
                    require(program.abort(*reused.sequence).status ==
                                runtime::ConsumeStatus::Consumed,
                            "Shared prefix reader retained execution resources");
                }
                require(program.abort(original).status == runtime::ConsumeStatus::Consumed,
                        "aligned capture retained its source writer");
                for (const auto point : captured.captured_checkpoints) {
                    require(program.release_checkpoint(point),
                            "aligned capture retained a checkpoint reader");
                }
                require(program.physical_usage().occupied == empty,
                        "aligned capture or cancellation leaked physical resources");
            }
        }
    }
}

// Progress anchors reach the planned capture groups as extra LongAnchor frontiers, and the option
// is refused where nothing could keep them (context cache off).
void progress_anchor_planning(DeviceContext& device, const qwen::execution::Parameters& parameters,
                              qwen::Frontend& frontend, EngineOptions options) {
    options.max_context                       = 512;
    options.kv_capacity                       = KvCapacityPolicy::explicit_capacity(512);
    options.max_concurrency                   = 1;
    options.prefill_chunk                     = kChunk;
    options.progress_anchor_tokens            = 2 * kChunk;
    options.context_cache.enabled             = true;
    options.context_cache.device_state_slots  = 2;
    options.context_cache.host_capacity_bytes = 0;
    auto planner = qwen::detail::make_sequence_planner_impl(parameters, device, options);
    // The explicit capacity fixes the page count; the page size depends on the KV storage.
    const std::uint32_t pages = planner->curve.minimum_main_page_groups;
    auto plan                 = qwen::detail::finalize_sequence_plan_impl(std::move(planner), pages);
    qwen::detail::ProgramImpl program(parameters, *plan, device, {});
    runtime::ResolvedExecutionOptions request;
    request.requested_output_tokens = 8;
    request.allow_prefix_reuse      = true;
    const auto frontiers_for = [&](std::uint32_t prompt_tokens) {
        auto prepared = qwen::PreparedPromptAccess::take(
            frontend.prepare_tokens(std::vector<TokenId>(prompt_tokens, 198)));
        return program.plan_request(std::move(prepared), request).capture_frontiers();
    };
    require(frontiers_for(500) == std::vector<std::uint32_t>({256, 500}),
            "progress anchors were not planned at every spacing inside the prompt");
    require(frontiers_for(256) == std::vector<std::uint32_t>({256}),
            "a progress anchor duplicated the InputReplay frontier");

    options.context_cache.enabled = false;
    bool rejected                 = false;
    try {
        (void)qwen::detail::make_sequence_planner_impl(parameters, device, options);
    } catch (const std::invalid_argument&) { rejected = true; }
    require(rejected, "progress anchors were accepted without the context cache");
}

// A cap on a prefill step shortens that step and nothing else: the prompt is still computed in
// full, in order, whatever sizes the steps take.
void capped_prefill_steps(DeviceContext& device, const qwen::execution::Parameters& parameters,
                          qwen::Frontend& frontend, EngineOptions options) {
    constexpr std::uint32_t kStepChunk = 2 * kChunk;
    constexpr std::uint32_t kPrompt    = 3 * kChunk;
    options.max_context                       = kCapacity;
    options.kv_capacity                       = KvCapacityPolicy::explicit_capacity(kCapacity);
    options.prefill_chunk                     = kStepChunk;
    options.max_concurrency                   = 1;
    options.context_cache.enabled             = false;
    options.context_cache.device_state_slots  = 0;
    options.context_cache.host_capacity_bytes = 0;
    auto planner = qwen::detail::make_sequence_planner_impl(parameters, device, options);
    // The explicit capacity fixes the page count; the page size depends on the KV storage.
    const std::uint32_t pages = planner->curve.minimum_main_page_groups;
    auto plan                 = qwen::detail::finalize_sequence_plan_impl(std::move(planner), pages);
    qwen::detail::ProgramImpl program(parameters, *plan, device, {});
    const auto empty = program.physical_usage().occupied;

    const auto settle = [&]() {
        std::atomic<bool> flag{false};
        for (unsigned step = 0; step < 8; ++step) {
            device.synchronize();
            CUDA_CHECK(cudaStreamSynchronize(device.transfer_stream));
            auto progress = program.poll_context({&flag});
            if (progress.complete) { return progress; }
        }
        throw std::runtime_error("capped prefill transaction did not settle");
    };
    const auto step = [&](qwen::SequenceHandle sequence, std::uint32_t max_tokens) {
        const std::array<qwen::ExecutionUnit, 1> units{
            {{sequence, qwen::ExecutionUnitKind::Prefill}}};
        require(static_cast<bool>(program.reserve_units(units)),
                "capped prefill could not reserve its unit");
        return program.advance_prefill(sequence, nullptr, nullptr, max_tokens);
    };

    // Each schedule of steps must end at the prompt's end after the same total, however it is cut.
    const std::vector<std::vector<std::uint32_t>> schedules{
        {0, 0}, {kChunk, 0}, {kChunk, kChunk, kChunk}, {kChunk, kChunk, 0}, {4 * kChunk, 0}};
    for (const auto& caps : schedules) {
        runtime::ResolvedExecutionOptions request;
        request.requested_output_tokens = 8;
        auto base = program.plan_request(
            qwen::PreparedPromptAccess::take(
                frontend.prepare_tokens(std::vector<TokenId>(kPrompt, 198))),
            request);
        const auto root = program.inspect_source(base, std::nullopt);
        require(root.has_value(), "capped prefill request has no root source");
        require(static_cast<bool>(program.start_binding(
                    base, {0}, *root, nullptr, qwen::ExecutionUnitKind::Prefill, 0)),
                "capped prefill could not bind its request");
        const auto bound = settle();
        require(bound.sequence.has_value(), "capped prefill binding did not publish");
        const auto sequence = *bound.sequence;

        std::uint32_t done = 0;
        for (std::size_t index = 0; index < caps.size(); ++index) {
            const std::uint32_t cap  = caps[index];
            const std::uint32_t left = kPrompt - done;
            const std::uint32_t want = std::min({left, kStepChunk, cap == 0 ? kStepChunk : cap});
            auto result              = step(sequence, cap);
            require(result.processed_prompt_tokens == want,
                    "a prefill step ignored its cap or ran past its chunk");
            done += result.processed_prompt_tokens;
            require(result.complete == (done == kPrompt),
                    "a capped prefill reported completion before the prompt's end");
            if (result.pending) {
                const std::array<runtime::CommitDecision, 1> accepted{{{.accepted_tokens = 1}}};
                (void)program.commit(std::move(*result.pending), accepted, {}, nullptr);
            }
        }
        require(done == kPrompt, "a schedule of capped steps did not cover the prompt");
        require(program.abort(sequence).status == runtime::ConsumeStatus::Consumed,
                "capped prefill retained its execution resources");
        require(program.physical_usage().occupied == empty,
                "capped prefill leaked physical resources");
    }
}

// A chunk run in layer slices does the same work as one run whole: the slices cut the layer loop
// and park the residual stream in between, and neither changes a value. So the first sampled token
// is identical, a lane part-way through a chunk refuses to pause, and aborting it releases
// everything.
void sliced_prefill_matches_whole(DeviceContext& device, const qwen::execution::Parameters& parameters,
                                  qwen::Frontend& frontend, EngineOptions options) {
    constexpr std::uint32_t kPrompt = 2 * kChunk + 40;
    options.max_context                       = kCapacity;
    options.kv_capacity                       = KvCapacityPolicy::explicit_capacity(kCapacity);
    options.prefill_chunk                     = kChunk;
    options.max_concurrency                   = 1;
    options.context_cache.enabled             = false;
    options.context_cache.device_state_slots  = 0;
    options.context_cache.host_capacity_bytes = 0;
    auto planner = qwen::detail::make_sequence_planner_impl(parameters, device, options);
    // The explicit capacity fixes the page count; the page size depends on the KV storage.
    const std::uint32_t pages = planner->curve.minimum_main_page_groups;
    auto plan                 = qwen::detail::finalize_sequence_plan_impl(std::move(planner), pages);
    qwen::detail::ProgramImpl program(parameters, *plan, device, {});
    const auto empty = program.physical_usage().occupied;

    const auto settle = [&]() {
        std::atomic<bool> flag{false};
        for (unsigned step = 0; step < 8; ++step) {
            device.synchronize();
            CUDA_CHECK(cudaStreamSynchronize(device.transfer_stream));
            auto progress = program.poll_context({&flag});
            if (progress.complete) { return progress; }
        }
        throw std::runtime_error("sliced prefill transaction did not settle");
    };
    std::vector<TokenId> ids(kPrompt);
    for (std::uint32_t i = 0; i < kPrompt; ++i) { ids[i] = 100 + static_cast<TokenId>((i * 37) % 900); }

    struct Outcome {
        TokenId first = -1;
        std::uint32_t partial_calls = 0;
        std::uint32_t processed     = 0;
    };
    const auto run = [&](std::uint32_t slices, bool abort_part_way) {
        runtime::ResolvedExecutionOptions request;
        request.requested_output_tokens = 8;
        auto base = program.plan_request(
            qwen::PreparedPromptAccess::take(frontend.prepare_tokens(ids)), request);
        const auto root = program.inspect_source(base, std::nullopt);
        require(root.has_value(), "sliced prefill request has no root source");
        require(static_cast<bool>(program.start_binding(
                    base, {0}, *root, nullptr, qwen::ExecutionUnitKind::Prefill, 0)),
                "sliced prefill could not bind its request");
        const auto bound = settle();
        require(bound.sequence.has_value(), "sliced prefill binding did not publish");
        const auto sequence = *bound.sequence;

        Outcome outcome;
        for (unsigned call = 0; call < 64 && outcome.processed < kPrompt; ++call) {
            const std::array<qwen::ExecutionUnit, 1> units{
                {{sequence, qwen::ExecutionUnitKind::Prefill}}};
            require(static_cast<bool>(program.reserve_units(units)),
                    "sliced prefill could not reserve its unit");
            auto step = program.advance_prefill(sequence, nullptr, nullptr, 0, slices);
            if (step.partial) {
                ++outcome.partial_calls;
                require(step.processed_prompt_tokens == 0 && !step.complete && !step.pending,
                        "a partial slice reported progress");
                // A lane part-way through a chunk has no consistent frontier to pause at.
                require(!program.start_pause(sequence, true, nullptr),
                        "a lane part-way through a sliced chunk could be paused");
                if (abort_part_way) {
                    require(program.abort(sequence).status == runtime::ConsumeStatus::Consumed,
                            "a lane part-way through a sliced chunk could not be aborted");
                    require(program.physical_usage().occupied == empty,
                            "aborting a sliced chunk leaked physical resources");
                    return outcome;
                }
                continue;
            }
            outcome.processed += step.processed_prompt_tokens;
            if (step.pending) {
                outcome.first = step.pending->tokens().front();
                const std::array<runtime::CommitDecision, 1> accepted{{{.accepted_tokens = 1}}};
                (void)program.commit(std::move(*step.pending), accepted, {}, nullptr);
            }
        }
        require(outcome.processed == kPrompt, "sliced prefill did not cover the prompt");
        require(program.abort(sequence).status == runtime::ConsumeStatus::Consumed,
                "sliced prefill retained its execution resources");
        require(program.physical_usage().occupied == empty, "sliced prefill leaked physical resources");
        return outcome;
    };

    const Outcome whole = run(0, false);
    require(whole.partial_calls == 0 && whole.first >= 0, "the unsliced reference run is wrong");
    for (const std::uint32_t slices : {2U, 3U, 5U}) {
        const Outcome sliced = run(slices, false);
        // Three chunks (128, 128, 40 tokens), each with slices-1 partial calls.
        require(sliced.partial_calls == 3 * (slices - 1),
                "a chunk did not run in the requested number of slices");
        require(sliced.first == whole.first,
                "a sliced prefill sampled a different first token than the same chunks run whole");
    }
    (void)run(4, true);
}

// A sibling decodes between the slices of another lane's chunk. The sliced lane's residual waits in
// prefill_hidden while the sibling's decode round uses the same workspace and device stream, so this
// is the resume-after-a-sibling's-decode path: the sliced lane must still sample the token an
// unsliced run samples, and the sibling must decode the tokens it decodes with nobody beside it.
void sliced_prefill_beside_a_decoding_lane(DeviceContext& device,
                                           const qwen::execution::Parameters& parameters,
                                           qwen::Frontend& frontend, EngineOptions options) {
    constexpr std::uint32_t kLongPrompt  = 2 * kChunk + 40;
    // 62 tokens: the sibling's third decode round needs a new KV page (64 tokens per page), so it
    // takes a page from the pool while the sliced lane's chunk is part-way through.
    constexpr std::uint32_t kShortPrompt = 62;
    constexpr unsigned kRounds           = 12;
    options.max_context                       = kCapacity;
    options.kv_capacity                       = KvCapacityPolicy::explicit_capacity(kCapacity);
    options.prefill_chunk                     = kChunk;
    options.max_concurrency                   = 2;
    options.context_cache.enabled             = false;
    options.context_cache.device_state_slots  = 0;
    options.context_cache.host_capacity_bytes = 0;
    auto planner = qwen::detail::make_sequence_planner_impl(parameters, device, options);
    const std::uint32_t pages = planner->curve.minimum_main_page_groups;
    auto plan                 = qwen::detail::finalize_sequence_plan_impl(std::move(planner), pages);
    qwen::detail::ProgramImpl program(parameters, *plan, device, {});
    const auto empty = program.physical_usage().occupied;

    const auto settle = [&]() {
        std::atomic<bool> flag{false};
        for (unsigned step = 0; step < 8; ++step) {
            device.synchronize();
            CUDA_CHECK(cudaStreamSynchronize(device.transfer_stream));
            auto progress = program.poll_context({&flag});
            if (progress.complete) { return progress; }
        }
        throw std::runtime_error("two-lane sliced prefill transaction did not settle");
    };
    const auto make_ids = [](std::uint32_t count, std::uint32_t salt) {
        std::vector<TokenId> ids(count);
        for (std::uint32_t i = 0; i < count; ++i) {
            ids[i] = 100 + static_cast<TokenId>((i * 37 + salt) % 900);
        }
        return ids;
    };
    const auto bind = [&](const std::vector<TokenId>& ids, std::uint32_t lane) {
        runtime::ResolvedExecutionOptions request;
        request.requested_output_tokens = 64;
        auto base = program.plan_request(
            qwen::PreparedPromptAccess::take(frontend.prepare_tokens(ids)), request);
        const auto root = program.inspect_source(base, std::nullopt);
        require(root.has_value(), "two-lane request has no root source");
        require(static_cast<bool>(program.start_binding(
                    base, {lane}, *root, nullptr, qwen::ExecutionUnitKind::Prefill, 0)),
                "two-lane request could not bind");
        const auto bound = settle();
        require(bound.sequence.has_value(), "two-lane binding did not publish");
        return *bound.sequence;
    };
    // Runs `sequence`'s prefill to its first token, whole or in `slices`; `between` runs after every
    // partial call (the sibling's decode round). Returns the first sampled token.
    const auto prefill = [&](qwen::SequenceHandle sequence, std::uint32_t prompt_tokens,
                             std::uint32_t slices, const std::function<void()>& between) {
        TokenId first = -1;
        std::uint32_t processed = 0;
        for (unsigned call = 0; call < 64 && processed < prompt_tokens; ++call) {
            const std::array<qwen::ExecutionUnit, 1> units{
                {{sequence, qwen::ExecutionUnitKind::Prefill}}};
            require(static_cast<bool>(program.reserve_units(units)),
                    "two-lane prefill could not reserve its unit");
            auto step = program.advance_prefill(sequence, nullptr, nullptr, 0, slices);
            if (step.partial) {
                between();
                continue;
            }
            processed += step.processed_prompt_tokens;
            if (step.pending) {
                first = step.pending->tokens().front();
                const std::array<runtime::CommitDecision, 1> accepted{{{.accepted_tokens = 1}}};
                (void)program.commit(std::move(*step.pending), accepted, {}, nullptr);
            }
        }
        require(processed == prompt_tokens && first >= 0, "two-lane prefill did not finish");
        return first;
    };
    // One decode round for `sequence`; returns the token it produced.
    const auto decode_round = [&](qwen::SequenceHandle sequence) {
        const std::array<qwen::ExecutionUnit, 1> units{
            {{sequence, qwen::ExecutionUnitKind::Decode, 64}}}; // 64: remaining output budget
        require(static_cast<bool>(program.reserve_units(units)),
                "the sibling could not reserve a decode unit");
        const std::array<qwen::SequenceHandle, 1> rows{sequence};
        const std::array<runtime::RoundBudget, 1> budgets{{{.generated_tokens_remaining = 64}}};
        auto pending = program.decode(rows, budgets, nullptr, nullptr);
        const TokenId token = pending.tokens().front();
        const std::array<runtime::CommitDecision, 1> accepted{{{.accepted_tokens = 1}}};
        (void)program.commit(std::move(pending), accepted, {}, nullptr);
        return token;
    };

    const auto long_ids  = make_ids(kLongPrompt, 0);
    const auto short_ids = make_ids(kShortPrompt, 7);
    struct Outcome {
        TokenId long_first = -1;
        std::vector<TokenId> sibling;
    };
    // `sliced` > 0: the long lane runs in slices and the sibling decodes after every partial call.
    // Otherwise the long lane runs whole and the sibling decodes alone afterwards.
    const auto run = [&](std::uint32_t slices, bool with_long) {
        Outcome outcome;
        const auto sibling = bind(short_ids, 1);
        (void)prefill(sibling, kShortPrompt, 0, [] {});
        std::optional<qwen::SequenceHandle> long_lane;
        if (with_long) { long_lane = bind(long_ids, 0); }
        const auto one_round = [&] {
            if (outcome.sibling.size() < kRounds) { outcome.sibling.push_back(decode_round(sibling)); }
        };
        if (with_long) { outcome.long_first = prefill(*long_lane, kLongPrompt, slices, one_round); }
        while (outcome.sibling.size() < kRounds) { one_round(); }
        if (long_lane) {
            require(program.abort(*long_lane).status == runtime::ConsumeStatus::Consumed,
                    "the long lane retained its resources");
        }
        require(program.abort(sibling).status == runtime::ConsumeStatus::Consumed,
                "the sibling retained its resources");
        require(program.physical_usage().occupied == empty,
                "the two-lane scenario leaked physical resources");
        return outcome;
    };

    const Outcome alone = run(0, false);
    const Outcome whole = run(0, true);
    require(alone.sibling == whole.sibling,
            "a sibling decoded different tokens when a whole chunk ran beside it");
    for (const std::uint32_t slices : {2U, 3U, 4U}) {
        const Outcome sliced = run(slices, true);
        require(sliced.long_first == whole.long_first,
                "a sliced prefill beside a decoding lane sampled a different first token");
        require(sliced.sibling == alone.sibling,
                "a sibling decoded different tokens between another lane's slices");
    }
}

// The admission fit check on real KV pools (--admission-fit-check): whether a fresh prompt's whole
// KV need can ever fit beside the pages the resident requests hold. Pure arithmetic is covered by
// the CPU scheduler test; this checks the Program's own accounting of mapped pages, live reservations
// and the pool's usable pages against a real allocator.
void prompt_fits_beside_residents_on_real_pools(DeviceContext& device,
                                                const qwen::execution::Parameters& parameters,
                                                qwen::Frontend& frontend, EngineOptions options) {
    // 15 pages of 64 tokens: a 500-token prompt needs 8, so two of them cannot share the pool.
    constexpr std::uint32_t kPoolTokens = 15 * 64;
    options.max_context                       = kCapacity;
    options.kv_capacity                       = KvCapacityPolicy::explicit_capacity(kPoolTokens);
    options.prefill_chunk                     = kChunk;
    options.max_concurrency                   = 2;
    options.context_cache.enabled             = false;
    options.context_cache.device_state_slots  = 0;
    options.context_cache.host_capacity_bytes = 0;
    auto planner = qwen::detail::make_sequence_planner_impl(parameters, device, options);
    // Not the curve minimum (8 pages, one request's worth): the pool this test needs is the 15 pages
    // of the explicit capacity, which the curve allows between one request and max_concurrency of them.
    constexpr std::uint32_t pages = kPoolTokens / 64;
    require(pages >= planner->curve.minimum_main_page_groups &&
                pages <= planner->curve.maximum_main_page_groups,
            "the fit test's pool is outside the capacity curve");
    auto plan                 = qwen::detail::finalize_sequence_plan_impl(std::move(planner), pages);
    qwen::detail::ProgramImpl program(parameters, *plan, device, {});
    const auto empty = program.physical_usage().occupied;

    const auto settle = [&]() {
        std::atomic<bool> flag{false};
        for (unsigned step = 0; step < 8; ++step) {
            device.synchronize();
            CUDA_CHECK(cudaStreamSynchronize(device.transfer_stream));
            auto progress = program.poll_context({&flag});
            if (progress.complete) { return progress; }
        }
        throw std::runtime_error("admission fit transaction did not settle");
    };
    const auto make_ids = [](std::uint32_t count, std::uint32_t salt) {
        std::vector<TokenId> ids(count);
        for (std::uint32_t i = 0; i < count; ++i) {
            ids[i] = 100 + static_cast<TokenId>((i * 37 + salt) % 900);
        }
        return ids;
    };
    const auto plan_request = [&](const std::vector<TokenId>& ids) {
        runtime::ResolvedExecutionOptions request;
        request.requested_output_tokens = 8;
        return program.plan_request(qwen::PreparedPromptAccess::take(frontend.prepare_tokens(ids)),
                                    request);
    };
    // Binds `ids` on `lane` and prefills `calls` chunks of it (by default all of them), so it holds
    // its pages as a resident.
    const auto make_resident = [&](const std::vector<TokenId>& ids, std::uint32_t lane,
                                   std::optional<unsigned> calls = std::nullopt) {
        auto base       = plan_request(ids);
        const auto root = program.inspect_source(base, std::nullopt);
        require(root.has_value(), "resident request has no root source");
        require(static_cast<bool>(program.start_binding(
                    base, {lane}, *root, nullptr, qwen::ExecutionUnitKind::Prefill, 0)),
                "resident request could not bind");
        const auto bound = settle();
        require(bound.sequence.has_value(), "resident binding did not publish");
        const auto sequence = *bound.sequence;
        std::uint32_t processed = 0;
        for (unsigned call = 0; call < calls.value_or(64) && processed < ids.size(); ++call) {
            const std::array<qwen::ExecutionUnit, 1> units{
                {{sequence, qwen::ExecutionUnitKind::Prefill}}};
            require(static_cast<bool>(program.reserve_units(units)),
                    "resident could not reserve its unit");
            auto step = program.advance_prefill(sequence, nullptr, nullptr, 0);
            processed += step.processed_prompt_tokens;
            if (step.pending) {
                const std::array<runtime::CommitDecision, 1> accepted{{{.accepted_tokens = 1}}};
                (void)program.commit(std::move(*step.pending), accepted, {}, nullptr);
            }
        }
        require(calls ? processed == std::size_t{*calls} * kChunk : processed == ids.size(),
                calls ? "resident prefill did not reach its chunk" : "resident prefill did not finish");
        return sequence;
    };
    const auto fits = [&](const std::vector<TokenId>& ids) {
        auto base       = plan_request(ids);
        const auto root = program.inspect_source(base, std::nullopt);
        require(root.has_value(), "candidate request has no root source");
        return program.prompt_fits_beside_residents(base, *root);
    };

    const auto newcomer = make_ids(500, 11);
    // Nothing resident: always admitted.
    require(fits(newcomer), "a fresh prompt did not fit an empty pool");

    // A small resident leaves room (2 + 8 of 15 pages).
    const auto small = make_resident(make_ids(100, 3), 0);
    require(fits(newcomer), "a prompt was deferred beside a resident that leaves room");
    require(program.abort(small).status == runtime::ConsumeStatus::Consumed,
            "the small resident retained its resources");

    // A large resident does not (8 + 8 of 15 pages): the newcomer is deferred ...
    const auto large = make_resident(make_ids(500, 5), 0);
    require(!fits(newcomer), "a prompt that cannot fit beside the resident was admitted");
    // ... and fits again once the resident's pages are released.
    require(program.abort(large).status == runtime::ConsumeStatus::Consumed,
            "the large resident retained its resources");
    require(fits(newcomer), "a prompt was still deferred after the resident released its pages");

    // A large resident one chunk into its prefill holds 2 pages but will need 8 (#332): the rest
    // of its prompt counts, so the newcomer is still deferred.
    const auto starting = make_resident(make_ids(500, 5), 0, 1);
    require(!fits(newcomer),
            "a prompt was admitted beside a resident that has not yet reached its prompt end");
    require(program.abort(starting).status == runtime::ConsumeStatus::Consumed,
            "the starting resident retained its resources");
    require(program.physical_usage().occupied == empty, "the fit check leaked physical resources");
}

void replay_sampling_counts(DeviceContext& device, const qwen::execution::Parameters& parameters,
                            qwen::Frontend& frontend, EngineOptions options) {
    options.max_context                       = 128;
    options.kv_capacity                       = KvCapacityPolicy::explicit_capacity(128);
    options.max_concurrency                   = 2;
    options.context_cache.enabled             = false;
    options.context_cache.device_state_slots  = 0;
    options.context_cache.host_capacity_bytes = 0;
    auto planner = qwen::detail::make_sequence_planner_impl(parameters, device, options);
    auto plan    = qwen::detail::finalize_sequence_plan_impl(std::move(planner), 2);
    qwen::detail::ProgramImpl program(parameters, *plan, device, {});

    // This exact oracle counts committed output events, independently of the execution route.
    // The prompt token appears many times but contributes no occurrences of its own.
    const std::vector<TokenId> prompt(32, 200);
    auto prepared = frontend.prepare_tokens(prompt);
    runtime::ResolvedExecutionOptions request;
    request.requested_output_tokens    = 8;
    request.allow_prefix_reuse         = false;
    request.sampling.presence_penalty  = 0.5F;
    request.sampling.frequency_penalty = 0.25F;
    auto base =
        program.plan_request(qwen::PreparedPromptAccess::take(std::move(prepared)), request);

    const auto settle = [&]() {
        for (unsigned step = 0; step < 8; ++step) {
            device.synchronize();
            CUDA_CHECK(cudaStreamSynchronize(device.transfer_stream));
            auto progress = program.poll_context({});
            if (progress.complete) { return progress; }
        }
        throw std::runtime_error("sampling-count recovery transaction did not complete");
    };
    const auto bind = [&](std::uint32_t lane, qwen::ResumeState* resume) {
        auto source = program.inspect_source(base, std::nullopt);
        require(source && static_cast<bool>(program.start_binding(
                              base, {lane}, *source, resume, qwen::ExecutionUnitKind::Decode, 1)),
                "sampling-count fixture could not bind its lane");
        auto progress = settle();
        require(progress.published && progress.sequence,
                "sampling-count fixture did not publish a sequence");
        require(!resume || progress.replaying, "sampling-count resume did not enter replay");
        return *progress.sequence;
    };
    const auto reserve = [&](qwen::SequenceHandle sequence, qwen::ExecutionUnitKind kind,
                             std::uint32_t tokens = 0) {
        const std::array<qwen::ExecutionUnit, 1> units{{{sequence, kind, tokens}}};
        require(static_cast<bool>(program.reserve_units(units)),
                "sampling-count fixture could not reserve a finite unit");
    };

    const auto initial = bind(0, nullptr);
    reserve(initial, qwen::ExecutionUnitKind::Prefill);
    auto begin = program.advance_prefill(initial, nullptr, nullptr);
    require(begin.complete && begin.pending && begin.pending->tokens().size() == 1,
            "sampling-count fixture did not produce exactly one Begin token");
    const TokenId first = begin.pending->tokens().front();
    const std::array<runtime::CommitDecision, 1> accepted{{{.accepted_tokens = 1}}};
    const auto committed = program.commit(std::move(*begin.pending), accepted,
                                          runtime::CommitObservation::AllRows, nullptr);
    require(committed.rows[0].disposition == runtime::CommitDisposition::Active,
            "sampling-count fixture did not commit Begin");

    const std::array<TokenId, 3> forced{198, 198, 199};
    const std::array<qwen::SequenceHandle, 1> members{initial};
    const std::array<std::optional<std::uint32_t>, 1> splits{2U};
    reserve(initial, qwen::ExecutionUnitKind::Control, forced.size());
    (void)program.append_forced_tokens(members, forced, forced.size(), splits, nullptr);

    const auto vocabulary =
        static_cast<std::int32_t>(parameters.model.resources().public_token_count);
    std::vector<std::int32_t> expected_counts(static_cast<std::size_t>(vocabulary), 0);
    ++expected_counts.at(static_cast<std::size_t>(first));
    for (const TokenId token : forced) { ++expected_counts.at(static_cast<std::size_t>(token)); }
    std::vector<TokenId> expected_ledger = prompt;
    expected_ledger.push_back(first);
    expected_ledger.insert(expected_ledger.end(), forced.begin(), forced.end());
    const std::uint32_t expected_frontier = static_cast<std::uint32_t>(expected_ledger.size() - 1U);
    const std::array<std::uint32_t, 1> expected_splits{
        static_cast<std::uint32_t>(prompt.size() + 1U + *splits.front())};

    const auto check = [&](std::uint32_t lane, std::string_view phase) {
        std::vector<std::int32_t> counts(expected_counts.size());
        const auto source =
            program.token_counts.slice(1, static_cast<std::int32_t>(lane), 1).view({vocabulary});
        device.synchronize();
        CUDA_CHECK(cudaMemcpy(counts.data(), source.data, source.bytes(), cudaMemcpyDeviceToHost));
        require(counts == expected_counts,
                std::string(phase) + ": sampling counts differ from committed output occurrences");
        const auto& sequence     = program.sequences[lane];
        const auto actual_splits = sequence.prefix_identity.execution_frontiers();
        require(sequence.ledger == expected_ledger &&
                    sequence.execution_frontier == expected_frontier &&
                    sequence.ledger_frontier == expected_frontier + 1U &&
                    sequence.prefix_identity.size() == expected_ledger.size() &&
                    actual_splits.size() == expected_splits.size() &&
                    std::equal(actual_splits.begin(), actual_splits.end(), expected_splits.begin()),
                std::string(phase) +
                    ": replay changed committed ledger, E/S or forced execution split");
    };
    check(0, "after forced control");
    require(program.start_pause(initial, false, nullptr), "sampling-count fixture could not pause");
    auto paused = settle();
    require(paused.paused && !paused.paused->has_snapshot() &&
                paused.paused->frontier() == expected_frontier,
            "sampling-count fixture did not retain its committed replay target");
    const auto resumed = bind(1, &*paused.paused);
    paused.paused.reset();
    check(1, "after lane rebinding");
    bool recovered         = false;
    std::uint32_t replayed = 0;
    for (unsigned step = 0; step < 3; ++step) {
        reserve(resumed, qwen::ExecutionUnitKind::Replay);
        const auto progress = program.advance_replay(resumed, nullptr);
        replayed += progress.processed_tokens;
        check(1, "after replay chunk");
        if (progress.complete) {
            recovered = true;
            break;
        }
    }
    require(recovered && replayed == expected_frontier,
            "sampling-count fixture did not finish replaying its execution frontier");
    require(program.abort(resumed).status == runtime::ConsumeStatus::Consumed,
            "sampling-count fixture could not release its restored lane");
    const auto released = program.physical_usage();
    require(released.occupied == runtime::ContextResourceUsage{} &&
                released.host_reserved_bytes == 0,
            "sampling-count fixture leaked typed resources");
}

// ---- Explicit RoPE positions in text prefill (lossy long-prefill, Task 4) ----

constexpr std::uint32_t kLongCapacity = 4096;
constexpr std::uint32_t kLongChunk    = 512;
constexpr std::uint32_t kLongPrompt   = 3000;
constexpr std::int32_t kRopeOffset    = 17;
// rk4v4 is the production KV storage; int8 is the extra one.
constexpr std::array<KvCacheStorage, 2> kPositionStorages{
    KvCacheStorage::RotatedLloyd4KeyInt4Value, KvCacheStorage::Int8Group64};

std::vector<TokenId> long_prompt_ids(std::uint32_t count) {
    std::vector<TokenId> ids(count);
    for (std::uint32_t i = 0; i < count; ++i) {
        ids[i] = 100 + static_cast<TokenId>((i * 37 + i / 64) % 900);
    }
    return ids;
}

// Every position axis set to positions[i], the shape a text prompt carries.
void set_text_positions(qwen::PreparedPromptData& prompt, const std::vector<std::int32_t>& axis,
                        std::int32_t rope_delta) {
    const std::size_t tokens = prompt.token_ids.size();
    require(axis.size() == tokens, "position fixture does not cover its prompt");
    prompt.positions.resize(3 * tokens);
    for (std::size_t a = 0; a < 3; ++a) {
        std::copy(axis.begin(), axis.end(), prompt.positions.begin() + a * tokens);
    }
    prompt.rope_delta = rope_delta;
}

void append_device_bytes(std::vector<std::byte>& bytes, const Tensor& tensor) {
    require(tensor.is_contiguous(), "a compared device region is not contiguous");
    const auto offset = bytes.size();
    bytes.resize(offset + tensor.bytes());
    CUDA_CHECK(
        cudaMemcpy(bytes.data() + offset, tensor.data, tensor.bytes(), cudaMemcpyDeviceToHost));
}

// The page slices of one KV plane: one per page (page-major) or one per page and head.
template <class Visit>
void visit_page_slices(const Tensor& plane, PagedKVPlaneOrder order, std::int32_t page,
                       Visit&& visit) {
    if (order == PagedKVPlaneOrder::PageMajor) {
        visit(plane.slice(3, page, 1));
        return;
    }
    for (std::int32_t head = 0; head < plane.ne[3]; ++head) {
        visit(plane.slice(3, head, 1).slice(2, page, 1));
    }
}

// A page's unused tail columns keep whatever an earlier run left there, so every run starts from
// zeroed planes and the comparison can take whole pages.
void zero_text_kv(qwen::detail::ProgramImpl& program, DeviceContext& device) {
    const auto& pool  = program.decoder->text_kv.page_pool();
    const auto order  = pool.geometry().device_plane_order;
    for (std::size_t index = 0; index < pool.plane_count(); ++index) {
        const Tensor& plane = pool.plane(index);
        const std::int32_t pages =
            order == PagedKVPlaneOrder::PageMajor ? plane.ne[3] : plane.ne[2];
        for (std::int32_t page = 0; page < pages; ++page) {
            visit_page_slices(plane, order, page, [&](const Tensor& slice) {
                require(slice.is_contiguous(), "a KV page slice is not contiguous");
                CUDA_CHECK(cudaMemsetAsync(slice.data, 0, slice.bytes(), device.stream));
            });
        }
    }
    device.synchronize();
}

// Text KV of a lane's first `tokens` positions, gathered page by page through its block table.
std::vector<std::byte> text_kv_bytes(qwen::detail::ProgramImpl& program, std::uint32_t lane,
                                     std::uint32_t tokens) {
    auto& cache          = program.decoder->text_kv;
    const auto& pool     = cache.page_pool();
    const auto order     = pool.geometry().device_plane_order;
    const auto page_size = pool.geometry().page_tokens;
    const auto row       = program.text_kv_addresses->bound_row(program.sequences[lane].kv->text);
    const auto table     = cache.execution_tables().matrix().slice(1, row, 1);
    std::vector<std::int32_t> physical((tokens + page_size - 1) / page_size);
    CUDA_CHECK(cudaMemcpy(physical.data(), table.data, physical.size() * sizeof(std::int32_t),
                          cudaMemcpyDeviceToHost));
    std::vector<std::byte> bytes;
    for (std::size_t index = 0; index < pool.plane_count(); ++index) {
        for (const std::int32_t page : physical) {
            require(page >= 0, "a prompt KV page is not mapped");
            visit_page_slices(pool.plane(index), order, page,
                              [&](const Tensor& slice) { append_device_bytes(bytes, slice); });
        }
    }
    return bytes;
}

struct PrefillImage {
    TokenId first = -1;
    std::vector<std::byte> logits;
    std::vector<std::byte> kv;
    std::vector<std::byte> state;
};

struct PositionsProgram {
    std::unique_ptr<qwen::detail::SequencePlanImpl> plan;
    std::unique_ptr<qwen::detail::ProgramImpl> program;
};

PositionsProgram make_positions_program(DeviceContext& device,
                                        const qwen::execution::Parameters& parameters,
                                        EngineOptions options, KvCacheStorage storage,
                                        std::uint32_t lanes) {
    options.max_context                       = kLongCapacity;
    options.kv_capacity                       = KvCapacityPolicy::explicit_capacity(kLongCapacity);
    options.prefill_chunk                     = kLongChunk;
    options.max_concurrency                   = lanes;
    options.kv_cache                          = storage;
    options.context_cache.enabled             = false;
    options.context_cache.device_state_slots  = 0;
    options.context_cache.host_capacity_bytes = 0;
    auto planner = qwen::detail::make_sequence_planner_impl(parameters, device, options);
    const std::uint32_t pages = lanes * planner->curve.minimum_main_page_groups;
    PositionsProgram out;
    out.plan    = qwen::detail::finalize_sequence_plan_impl(std::move(planner), pages);
    out.program = std::make_unique<qwen::detail::ProgramImpl>(parameters, *out.plan, device,
                                                              StartupObserver{});
    return out;
}

qwen::ContextProgress settle_positions(qwen::detail::ProgramImpl& program, DeviceContext& device) {
    std::atomic<bool> flag{false};
    for (unsigned step = 0; step < 8; ++step) {
        device.synchronize();
        CUDA_CHECK(cudaStreamSynchronize(device.transfer_stream));
        auto progress = program.poll_context({&flag});
        if (progress.complete) { return progress; }
    }
    throw std::runtime_error("explicit-position transaction did not settle");
}

// Binds `prompt` to lane 0 of an idle Program, prefills it (whole, or in `slices` layer slices) and
// returns the last-token logits, the prompt's text KV and the lane's GDN state image as they stand
// when the prefill finishes, then releases the lane.
PrefillImage prefill_image(qwen::detail::ProgramImpl& program, DeviceContext& device,
                           qwen::PreparedPromptData prompt, std::uint32_t slices) {
    const auto empty           = program.physical_usage().occupied;
    const auto prompt_tokens   = static_cast<std::uint32_t>(prompt.token_ids.size());
    zero_text_kv(program, device);
    runtime::ResolvedExecutionOptions request;
    request.requested_output_tokens = 8;
    request.allow_prefix_reuse      = false;
    auto base       = program.plan_request(std::move(prompt), request);
    const auto root = program.inspect_source(base, std::nullopt);
    require(root.has_value(), "explicit-position request has no root source");
    require(static_cast<bool>(program.start_binding(base, {0}, *root, nullptr,
                                                    qwen::ExecutionUnitKind::Prefill, 0)),
            "explicit-position request could not bind");
    const auto bound = settle_positions(program, device);
    require(bound.sequence.has_value(), "explicit-position binding did not publish");
    const auto sequence = *bound.sequence;

    PrefillImage image;
    std::uint32_t processed = 0;
    for (unsigned call = 0; call < 256 && processed < prompt_tokens; ++call) {
        const std::array<qwen::ExecutionUnit, 1> units{
            {{sequence, qwen::ExecutionUnitKind::Prefill}}};
        require(static_cast<bool>(program.reserve_units(units)),
                "explicit-position prefill could not reserve its unit");
        auto step = program.advance_prefill(sequence, nullptr, nullptr, 0, slices);
        if (step.partial) { continue; }
        processed += step.processed_prompt_tokens;
        if (!step.pending) { continue; }
        device.synchronize();
        image.first = step.pending->tokens().front();
        append_device_bytes(image.logits, program.io.logits.slice(1, 0, 1));
        image.kv = text_kv_bytes(program, 0, prompt_tokens);
        const auto& layout = program.state_images->host_layout();
        image.state.resize(layout.image_bytes);
        const auto selectors = program.state_selectors(program.sequences[0]);
        program.state_images->copy_to_host(selectors.destination,
                                           qwen::HostStateImageView{image.state.data(), &layout},
                                           RankStreams(device.stream));
        device.synchronize();
        const std::array<runtime::CommitDecision, 1> accepted{{{.accepted_tokens = 1}}};
        (void)program.commit(std::move(*step.pending), accepted, {}, nullptr);
    }
    require(processed == prompt_tokens && image.first >= 0,
            "explicit-position prefill did not finish its prompt");
    require(program.abort(sequence).status == runtime::ConsumeStatus::Consumed,
            "explicit-position prefill retained its execution resources");
    require(program.physical_usage().occupied == empty,
            "explicit-position prefill leaked physical resources");
    return image;
}

void require_same_image(const PrefillImage& actual, const PrefillImage& expected,
                        std::string_view what) {
    require(actual.first == expected.first, std::string(what) + ": first token differs");
    require(actual.logits == expected.logits, std::string(what) + ": last-token logits differ");
    require(actual.kv == expected.kv, std::string(what) + ": text KV differs");
    require(actual.state == expected.state, std::string(what) + ": GDN state differs");
}

// Positions i + 17 given explicitly (the prompt's rope_delta stays 0, so prefill cannot derive them
// from the KV slot) must equal the slot + rope_delta path with rope_delta 17, bit for bit, whole
// and in four layer slices.
void explicit_positions_match_rope_delta(DeviceContext& device,
                                         const qwen::execution::Parameters& parameters,
                                         qwen::Frontend& frontend, EngineOptions options,
                                         bool with_slices) {
    const auto ids = long_prompt_ids(kLongPrompt);
    std::vector<std::int32_t> offset(kLongPrompt);
    for (std::uint32_t i = 0; i < kLongPrompt; ++i) {
        offset[i] = static_cast<std::int32_t>(i) + kRopeOffset;
    }
    for (const KvCacheStorage storage : kPositionStorages) {
        auto fixture = make_positions_program(device, parameters, options, storage, 1);
        auto delta_prompt = qwen::PreparedPromptAccess::take(frontend.prepare_tokens(ids));
        set_text_positions(delta_prompt, offset, kRopeOffset);
        auto explicit_prompt = qwen::PreparedPromptAccess::take(frontend.prepare_tokens(ids));
        set_text_positions(explicit_prompt, offset, 0);

        const PrefillImage reference = prefill_image(*fixture.program, device, delta_prompt, 0);
        const PrefillImage explicit_run =
            prefill_image(*fixture.program, device, explicit_prompt, with_slices ? 4U : 0U);
        require_same_image(explicit_run, reference,
                           with_slices ? "explicit positions in four layer slices"
                                       : "explicit positions");
        if (!with_slices) {
            // A guard against a vacuous pass: the slot positions (delta 0) are a different input.
            auto slot_prompt = qwen::PreparedPromptAccess::take(frontend.prepare_tokens(ids));
            const PrefillImage slot =
                prefill_image(*fixture.program, device, std::move(slot_prompt), 0);
            require(slot.logits != reference.logits,
                    "positions i + 17 produced the logits of positions i");
        }
    }
}

void explicit_positions_equal_rope_delta_path(DeviceContext& device,
                                              const qwen::execution::Parameters& parameters,
                                              qwen::Frontend& frontend, EngineOptions options) {
    explicit_positions_match_rope_delta(device, parameters, frontend, options, false);
}

void explicit_positions_with_layer_slices(DeviceContext& device,
                                          const qwen::execution::Parameters& parameters,
                                          qwen::Frontend& frontend, EngineOptions options) {
    explicit_positions_match_rope_delta(device, parameters, frontend, options, true);
}

// A gapped prompt (kept tokens at their original positions, rope_delta raised by the dropped count)
// paused with no snapshot is rebuilt by replay from its ledger. The replay must prefill with the
// stored positions: the greedy continuation after the resume equals the run that never paused. The
// first prefill runs in capped 200-token steps and replay in 512-token chunks, so a path that took
// RoPE from the chunk layout (slot + a delta reset on the first chunk) would not agree with itself.
void replay_preserves_gapped_positions(DeviceContext& device,
                                       const qwen::execution::Parameters& parameters,
                                       qwen::Frontend& frontend, EngineOptions options) {
    constexpr std::uint32_t kGeneratedTokens = 32;
    constexpr std::uint32_t kPauseAfter      = 8;
    constexpr std::uint32_t kGapBegin        = 256;
    constexpr std::uint32_t kPrefillStepCap  = 200;
    const auto original = qwen::PreparedPromptAccess::take(
        frontend.prepare_tokens(long_prompt_ids(kLongPrompt)));
    // Keep every other 64-token block of U = [256, 3000) and its last 512 tokens.
    std::vector<std::uint32_t> kept;
    const std::uint32_t span = kLongPrompt - kGapBegin;
    for (std::uint32_t offset = 0; offset < span; ++offset) {
        if ((offset / 64) % 2 == 0 || offset + 512 >= span) { kept.push_back(offset); }
    }
    const auto map = qwen::detail::make_gap_map(original, kGapBegin, kept);
    const auto gapped = qwen::detail::compact_prompt(original, map);
    require(gapped.token_ids.size() == kGapBegin + kept.size() &&
                gapped.rope_delta == static_cast<std::int32_t>(span - kept.size()) &&
                gapped.positions[gapped.token_ids.size() - 1] ==
                    static_cast<std::int32_t>(kLongPrompt - 1),
            "the gapped prompt fixture is not the compacted original");
    const auto gapped_tokens = static_cast<std::uint32_t>(gapped.token_ids.size());

    for (const KvCacheStorage storage : kPositionStorages) {
        auto fixture   = make_positions_program(device, parameters, options, storage, 2);
        auto& program  = *fixture.program;
        const auto empty = program.physical_usage().occupied;
        runtime::ResolvedExecutionOptions request;
        request.requested_output_tokens = 64;
        request.allow_prefix_reuse      = false;

        const auto reserve = [&](qwen::SequenceHandle sequence, qwen::ExecutionUnitKind kind,
                                 std::uint32_t tokens = 0) {
            const std::array<qwen::ExecutionUnit, 1> units{{{sequence, kind, tokens}}};
            require(static_cast<bool>(program.reserve_units(units)),
                    "gapped replay could not reserve a unit");
        };
        const auto bind = [&](qwen::RequestBasePlan& base, std::uint32_t lane,
                              qwen::ResumeState* resume) {
            auto source = program.inspect_source(base, std::nullopt);
            require(source && static_cast<bool>(
                                  program.start_binding(base, {lane}, *source, resume,
                                                        qwen::ExecutionUnitKind::Decode, 1)),
                    "gapped replay could not bind its lane");
            auto progress = settle_positions(program, device);
            require(progress.published && progress.sequence, "gapped replay did not publish");
            require(!resume || progress.replaying, "gapped resume did not enter replay");
            return *progress.sequence;
        };
        const auto prefill = [&](qwen::SequenceHandle sequence) {
            std::uint32_t processed = 0;
            for (unsigned call = 0; call < 64 && processed < gapped_tokens; ++call) {
                // A capped step that ends mid-chunk returns without a token; keep going.
                reserve(sequence, qwen::ExecutionUnitKind::Prefill);
                auto step = program.advance_prefill(sequence, nullptr, nullptr, kPrefillStepCap);
                processed += step.processed_prompt_tokens;
                if (step.pending) {
                    const TokenId first = step.pending->tokens().front();
                    const std::array<runtime::CommitDecision, 1> accepted{{{.accepted_tokens = 1}}};
                    (void)program.commit(std::move(*step.pending), accepted, {}, nullptr);
                    require(processed == gapped_tokens, "gapped prefill sampled early");
                    return first;
                }
            }
            throw std::runtime_error("gapped prefill did not finish its prompt");
        };
        const auto decode_round = [&](qwen::SequenceHandle sequence) {
            reserve(sequence, qwen::ExecutionUnitKind::Decode, 64);
            const std::array<qwen::SequenceHandle, 1> rows{sequence};
            const std::array<runtime::RoundBudget, 1> budgets{{{.generated_tokens_remaining = 64}}};
            auto pending        = program.decode(rows, budgets, nullptr, nullptr);
            const TokenId token = pending.tokens().front();
            const std::array<runtime::CommitDecision, 1> accepted{{{.accepted_tokens = 1}}};
            (void)program.commit(std::move(pending), accepted, {}, nullptr);
            return token;
        };

        // The unpaused reference.
        std::vector<TokenId> unpaused;
        {
            auto base = program.plan_request(qwen::PreparedPromptData(gapped), request);
            const auto sequence = bind(base, 0, nullptr);
            unpaused.push_back(prefill(sequence));
            while (unpaused.size() < kGeneratedTokens) {
                unpaused.push_back(decode_round(sequence));
            }
            require(program.abort(sequence).status == runtime::ConsumeStatus::Consumed,
                    "the unpaused gapped run retained its resources");
        }

        std::vector<TokenId> resumed_tokens;
        {
            auto base = program.plan_request(qwen::PreparedPromptData(gapped), request);
            const auto initial = bind(base, 0, nullptr);
            resumed_tokens.push_back(prefill(initial));
            while (resumed_tokens.size() < kPauseAfter) {
                resumed_tokens.push_back(decode_round(initial));
            }
            require(program.start_pause(initial, false, nullptr), "gapped run could not pause");
            auto paused = settle_positions(program, device);
            require(paused.paused && !paused.paused->has_snapshot(),
                    "gapped pause kept a snapshot, so nothing replays");
            const auto resumed = bind(base, 1, &*paused.paused);
            paused.paused.reset();
            bool recovered = false;
            for (unsigned step = 0; step < 64 && !recovered; ++step) {
                reserve(resumed, qwen::ExecutionUnitKind::Replay);
                recovered = program.advance_replay(resumed, nullptr).complete;
            }
            require(recovered, "gapped replay did not finish");
            while (resumed_tokens.size() < kGeneratedTokens) {
                resumed_tokens.push_back(decode_round(resumed));
            }
            require(program.abort(resumed).status == runtime::ConsumeStatus::Consumed,
                    "the resumed gapped run retained its resources");
        }
        require(program.physical_usage().occupied == empty, "gapped replay leaked resources");
        require(resumed_tokens == unpaused,
                "a gapped prompt replayed after a pause decoded different greedy tokens");
    }
}

// ---- Lossy-prefill scout chunks (Task 6) ----

constexpr std::uint32_t kScoutCapacity = 36864;
constexpr std::uint32_t kScoutChunk    = 4096;
constexpr std::uint32_t kScoutPrefix   = 2048;
constexpr std::uint32_t kScoutSpan     = 12288;
constexpr std::uint32_t kScoutLayer    = 15;
constexpr std::uint32_t kScoutRows     = 64;

struct ScoutFixture {
    std::unique_ptr<qwen::detail::SequencePlanImpl> plan;
    std::unique_ptr<qwen::detail::ProgramImpl> program;
};

// One rk4v4 Program with the scout's scratch (lossy prefill on), two lanes and a context cache for
// the shared-prefix binding.
ScoutFixture make_scout_program(DeviceContext& device, const qwen::execution::Parameters& parameters,
                                EngineOptions options) {
    options.max_context                       = kScoutCapacity;
    options.kv_capacity                       = KvCapacityPolicy::explicit_capacity(kScoutCapacity);
    options.prefill_chunk                     = kScoutChunk;
    options.max_concurrency                   = 2;
    options.kv_cache                          = KvCacheStorage::RotatedLloyd4KeyInt4Value;
    options.context_cache.enabled             = true;
    options.context_cache.device_state_slots  = 2;
    options.context_cache.host_capacity_bytes = 0;
    options.lossy_prefill.min_tokens          = 1;
    options.lossy_prefill.filter_layer        = kScoutLayer;
    options.lossy_prefill.query_rows          = kScoutRows;
    auto planner = qwen::detail::make_sequence_planner_impl(parameters, device, options);
    const std::uint32_t pages = planner->curve.minimum_main_page_groups;
    ScoutFixture out;
    out.plan    = qwen::detail::finalize_sequence_plan_impl(std::move(planner), pages);
    out.program = std::make_unique<qwen::detail::ProgramImpl>(parameters, *out.plan, device,
                                                              StartupObserver{});
    require(out.program->scout_buffers.has_value(), "lossy prefill planned no scout scratch");
    return out;
}

// What the main pass starts from: the GDN state image in the slot the lane reads, the text KV of
// the cached prefix and the free page count.
struct ScoutTrace {
    std::vector<std::byte> state;
    std::vector<std::byte> kv;
    std::uint32_t free_pages   = 0;
    std::uint32_t mapped_pages = 0;
};

ScoutTrace scout_trace(qwen::detail::ProgramImpl& program, DeviceContext& device,
                       std::uint32_t lane, std::uint32_t prefix) {
    device.synchronize();
    ScoutTrace trace;
    const auto& layout = program.state_images->host_layout();
    trace.state.resize(layout.image_bytes);
    const auto selectors = program.state_selectors(program.sequences[lane]);
    program.state_images->copy_to_host(selectors.source,
                                       qwen::HostStateImageView{trace.state.data(), &layout},
                                       RankStreams(device.stream));
    device.synchronize();
    trace.kv           = text_kv_bytes(program, lane, prefix);
    trace.free_pages   = program.text_kv_pages->physical_pool().available_pages();
    trace.mapped_pages = program.text_kv_addresses->mapped_pages(program.sequences[lane].kv->text);
    return trace;
}

void require_no_trace(const ScoutTrace& after, const ScoutTrace& before, std::string_view what) {
    require(after.state == before.state, std::string(what) + ": GDN state changed");
    require(after.kv == before.kv, std::string(what) + ": cached-prefix KV changed");
    require(after.free_pages == before.free_pages, std::string(what) + ": KV pages leaked");
    require(after.mapped_pages == before.mapped_pages, std::string(what) + ": KV mapping changed");
}

class ScoutLane {
public:
    ScoutLane(qwen::detail::ProgramImpl& program, DeviceContext& device)
        : program_(program), device_(device) {}

    // A fresh request for `ids`, bound on `lane` with no reuse and prefilled to `prefix` tokens.
    qwen::SequenceHandle private_prefix(const std::vector<TokenId>& ids, std::uint32_t lane,
                                        std::uint32_t prefix, qwen::Frontend& frontend) {
        runtime::ResolvedExecutionOptions request;
        request.requested_output_tokens = 8;
        request.allow_prefix_reuse      = false;
        auto base       = program_.plan_request(
            qwen::PreparedPromptAccess::take(frontend.prepare_tokens(ids)), request);
        const auto root = program_.inspect_source(base, std::nullopt);
        const auto sequence = bind(base, lane, *root);
        if (prefix != 0) {
            reserve(sequence);
            const auto step = program_.advance_prefill(sequence, nullptr, nullptr, prefix);
            require(step.processed_prompt_tokens == prefix && !step.pending,
                    "the cached prefix did not stop at its frontier");
        }
        return sequence;
    }

    qwen::SequenceHandle bind(qwen::RequestBasePlan& base, std::uint32_t lane,
                              const qwen::SourceCandidate& source) {
        require(static_cast<bool>(program_.start_binding(base, {lane}, source, nullptr,
                                                         qwen::ExecutionUnitKind::Prefill, 0)),
                "scout request could not bind");
        const auto bound = settle_positions(program_, device_);
        require(bound.sequence.has_value(), "scout binding did not publish");
        return *bound.sequence;
    }

    void reserve(qwen::SequenceHandle sequence) {
        const std::array<qwen::ExecutionUnit, 1> units{
            {{sequence, qwen::ExecutionUnitKind::Prefill}}};
        require(static_cast<bool>(program_.reserve_units(units)),
                "scout test could not reserve a prefill unit");
    }

    // Scouts the lane's uncached span in steps of at most `step` tokens (stopping after `stop`
    // chunks when nonzero) and returns the scores (empty when stopped early). Ends the scout.
    std::vector<float> scout(qwen::SequenceHandle sequence, std::uint32_t step,
                             std::uint32_t stop = 0) {
        program_.begin_scout(sequence);
        require(program_.scout_active(sequence), "begin_scout did not open a scout");
        std::vector<float> scores;
        for (std::uint32_t chunk = 0; chunk < 1024; ++chunk) {
            if (stop != 0 && chunk == stop) { break; }
            const auto result = program_.advance_scout(sequence, step);
            require(result.reserved && result.processed_tokens != 0,
                    "a scout chunk found too few KV pages");
            if (result.complete) {
                scores = program_.scout_scores();
                break;
            }
        }
        program_.end_scout(sequence);
        require(!program_.scout_active(sequence), "end_scout left the scout open");
        return scores;
    }

    // Prefills the rest of the prompt and returns the first token and the last-token logits.
    std::pair<TokenId, std::vector<std::byte>> finish(qwen::SequenceHandle sequence) {
        for (unsigned call = 0; call < 64; ++call) {
            reserve(sequence);
            auto step = program_.advance_prefill(sequence, nullptr, nullptr);
            if (step.capture_ready) { program_.skip_capture(sequence); }
            if (!step.pending) { continue; }
            device_.synchronize();
            std::vector<std::byte> logits;
            append_device_bytes(logits, program_.io.logits.slice(1, 0, 1));
            const TokenId first = step.pending->tokens().front();
            const std::array<runtime::CommitDecision, 1> accepted{{{.accepted_tokens = 1}}};
            const auto committed = program_.commit(std::move(*step.pending), accepted, {}, nullptr);
            if (committed.capture_ready[0]) { program_.skip_capture(sequence); }
            return {first, std::move(logits)};
        }
        throw std::runtime_error("scout test prefill did not finish its prompt");
    }

    void abort(qwen::SequenceHandle sequence) {
        require(program_.abort(sequence).status == runtime::ConsumeStatus::Consumed,
                "scout lane retained its execution resources");
    }

private:
    qwen::detail::ProgramImpl& program_;
    DeviceContext& device_;
};

void require_scores_sane(const qwen::detail::ProgramImpl& program,
                         const std::vector<float>& scores, std::uint32_t span) {
    require(scores.size() == span, "scout scored the wrong number of keys");
    double sum = 0.0;
    for (const float score : scores) {
        require(std::isfinite(score) && score >= 0.0F, "scout score is not a finite mass");
        sum += score;
    }
    // Each query row and head distributes a unit of mass over the keys.
    const double expected =
        static_cast<double>(kScoutRows) *
        static_cast<double>(program.parameters.model.config().text.attention->num_attention_heads);
    require(std::abs(sum - expected) <= 1e-2 * expected, "scout scores do not sum to rows x heads");
}

// Scouting 12,288 tokens after a 2,048-token cached prefix leaves the prefix's KV, the GDN state
// the main pass reads and the free page count exactly as they were, and the ordinary prefill that
// follows equals one that never scouted, bit for bit. Run for a private prefix (the lane writes
// its state in place, so the scout saves and restores it) and a shared prefix (a fork is pending:
// the scout writes the destination and the main pass reads the untouched checkpoint).
void scout_leaves_no_trace(DeviceContext& device, const qwen::execution::Parameters& parameters,
                           qwen::Frontend& frontend, EngineOptions options) {
    auto fixture  = make_scout_program(device, parameters, options);
    auto& program = *fixture.program;
    ScoutLane lanes(program, device);
    const auto empty = program.physical_usage().occupied;
    const auto ids   = long_prompt_ids(kScoutPrefix + kScoutSpan);

    // Private prefix.
    {
        const auto reference_sequence = lanes.private_prefix(ids, 0, kScoutPrefix, frontend);
        const auto reference          = lanes.finish(reference_sequence);
        lanes.abort(reference_sequence);

        const auto sequence = lanes.private_prefix(ids, 0, kScoutPrefix, frontend);
        const auto selectors = program.state_selectors(program.sequences[0]);
        require(selectors.source == selectors.destination,
                "a private prefix writes its state in place (the scout must save and restore it)");
        const ScoutTrace before = scout_trace(program, device, 0, kScoutPrefix);
        require_scores_sane(program, lanes.scout(sequence, 0), kScoutSpan);
        require_no_trace(scout_trace(program, device, 0, kScoutPrefix), before, "private scout");
        const auto after = lanes.finish(sequence);
        require(after.first == reference.first && after.second == reference.second,
                "a prefill after a private scout differs from one without it");
        lanes.abort(sequence);
    }
    require(program.physical_usage().occupied == empty, "private scout leaked resources");

    // Shared prefix.
    {
        auto prepared = qwen::PreparedPromptAccess::take(frontend.prepare_tokens(ids));
        prepared.context_cache.opportunities.push_back(
            {.kind     = PromptCacheMarkerKind::SharedStablePrefix,
             .evidence = SharedCandidateEvidence::ExplicitBoundary,
             .frontier = kScoutPrefix});
        runtime::ResolvedExecutionOptions request;
        request.requested_output_tokens = 8;
        request.allow_prefix_reuse      = true;
        auto base         = program.plan_request(std::move(prepared), request);
        const auto root   = program.inspect_source(base, std::nullopt);
        const auto writer = lanes.bind(base, 0, *root);
        lanes.reserve(writer);
        const auto step = program.advance_prefill(writer, nullptr, nullptr);
        require(step.processed_prompt_tokens == kScoutPrefix && step.capture_ready,
                "the shared prefix writer did not stop at its capture");
        require(program.start_capture(writer), "the shared prefix could not be captured");
        const auto captured = settle_positions(program, device);
        std::optional<qwen::CheckpointHandle> shared;
        for (const auto point : captured.captured_checkpoints) {
            if (program.checkpoint_summary(point).role == runtime::CheckpointRole::SharedPrefix) {
                shared = point;
            }
        }
        require(shared.has_value(), "the shared prefix was not captured");
        lanes.abort(writer);

        const auto source = program.inspect_source(base, *shared);
        require(source && source->reused_tokens == kScoutPrefix,
                "the shared prefix does not cover its frontier");
        const auto reference_sequence = lanes.bind(base, 1, *source);
        const auto reference          = lanes.finish(reference_sequence);
        lanes.abort(reference_sequence);

        const auto sequence = lanes.bind(base, 1, *source);
        const auto selectors = program.state_selectors(program.sequences[1]);
        require(selectors.source != selectors.destination,
                "a shared-prefix binding has no pending fork");
        const ScoutTrace before = scout_trace(program, device, 1, kScoutPrefix);
        require_scores_sane(program, lanes.scout(sequence, 0), kScoutSpan);
        require_no_trace(scout_trace(program, device, 1, kScoutPrefix), before, "shared scout");
        const auto after = lanes.finish(sequence);
        require(after.first == reference.first && after.second == reference.second,
                "a prefill after a shared-prefix scout differs from one without it");
        lanes.abort(sequence);
        for (const auto point : captured.captured_checkpoints) {
            require(program.release_checkpoint(point), "the shared prefix kept a reader");
        }
    }
    require(program.physical_usage().occupied == empty, "shared scout leaked resources");
}

// A scout ended after its second chunk (an abort) leaves the same no trace, gives back every page
// it took, and a lane aborted part-way through a scout takes the scout with it.
void scout_abort_leaves_no_trace(DeviceContext& device,
                                 const qwen::execution::Parameters& parameters,
                                 qwen::Frontend& frontend, EngineOptions options) {
    auto fixture  = make_scout_program(device, parameters, options);
    auto& program = *fixture.program;
    ScoutLane lanes(program, device);
    const auto empty = program.physical_usage().occupied;
    const auto ids   = long_prompt_ids(kScoutPrefix + kScoutSpan);

    const auto sequence     = lanes.private_prefix(ids, 0, kScoutPrefix, frontend);
    const ScoutTrace before = scout_trace(program, device, 0, kScoutPrefix);
    const auto scores       = lanes.scout(sequence, 0, 2);
    require(scores.empty(), "a scout ended after two chunks reported scores");
    require_no_trace(scout_trace(program, device, 0, kScoutPrefix), before, "aborted scout");

    // Abort the lane itself in the middle of a second scout.
    program.begin_scout(sequence);
    const auto step = program.advance_scout(sequence, 0);
    require(step.reserved && !step.complete, "the second scout's first chunk did not run");
    lanes.abort(sequence);
    require(!program.scout_session.has_value(), "an aborted lane kept its scout");
    require(program.physical_usage().occupied == empty, "an aborted scout leaked resources");
}

// The scores do not depend on how the span is cut into chunks: 1,024- and 4,096-token scout
// chunks keep the same tokens (Jaccard >= 0.99) and agree per token within 30% relative.
void scout_scores_chunk_invariant(DeviceContext& device,
                                  const qwen::execution::Parameters& parameters,
                                  qwen::Frontend& frontend, EngineOptions options) {
    auto fixture  = make_scout_program(device, parameters, options);
    auto& program = *fixture.program;
    ScoutLane lanes(program, device);
    const auto ids   = long_prompt_ids(kScoutPrefix + kScoutSpan);
    const auto run   = [&](std::uint32_t step) {
        const auto sequence = lanes.private_prefix(ids, 0, kScoutPrefix, frontend);
        auto scores         = lanes.scout(sequence, step);
        lanes.abort(sequence);
        require_scores_sane(program, scores, kScoutSpan);
        return scores;
    };
    const auto narrow = run(1024);
    const auto wide   = run(4096);
    // Chunk invariance is a ranking property. Each chunk size is bit-reproducible, but the two
    // tile the BF16 GEMMs and GDN chunks differently, and softmax mass amplifies that rounding
    // exponentially: measured on the 3090 (2026-10-25), 65% of tokens differ by more than 1%
    // (the first chunk as much as the rest, so it is not a chunk-boundary carry), at most 17%,
    // while the total mass is equal and the kept sets are identical (Jaccard 1.0). So the kept
    // set is the gate; the per-token bound only catches gross errors (a chunk scored at the
    // wrong positions or with stale keys).
    require(narrow.size() == wide.size(), "scout scores differ in length across chunk sizes");
    double mass = 0.0;
    for (const float v : narrow) {
        mass += std::abs(v);
    }
    const double floor_abs = 1e-6 * mass;
    double max_rel         = 0.0;
    for (std::size_t i = 0; i < narrow.size(); ++i) {
        const double scale = std::max(std::abs(narrow[i]), std::abs(wide[i]));
        const double diff  = std::abs(double(narrow[i]) - double(wide[i]));
        if (scale > floor_abs) {
            max_rel = std::max(max_rel, diff / scale);
        }
        require(diff <= 0.3 * scale + floor_abs,
                "scout score " + std::to_string(i) + " depends on the chunk size: " +
                    std::to_string(narrow[i]) + " vs " + std::to_string(wide[i]));
    }
    const LossyPrefillOptions lossy{};
    const std::vector<std::uint8_t> forced(narrow.size(), 0);
    const auto keep_narrow = models::qwen3_5::detail::select_tokens({narrow, forced}, lossy);
    const auto keep_wide   = models::qwen3_5::detail::select_tokens({wide, forced}, lossy);
    std::vector<std::uint32_t> both;
    std::set_intersection(keep_narrow.kept.begin(), keep_narrow.kept.end(), keep_wide.kept.begin(),
                          keep_wide.kept.end(), std::back_inserter(both));
    const double uni = double(keep_narrow.kept.size() + keep_wide.kept.size() - both.size());
    const double jaccard = uni == 0.0 ? 1.0 : double(both.size()) / uni;
    std::cout << "scout chunk invariance: max relative difference " << max_rel
              << ", kept-set Jaccard " << jaccard << " (" << keep_narrow.kept.size() << " vs "
              << keep_wide.kept.size() << " kept)\n";
    require(jaccard >= 0.99, "scout kept sets depend on the chunk size: Jaccard " +
                                 std::to_string(jaccard));
}

// In a 32K-token haystack with a pass key sentence at 50% depth and the question at the end, the
// pass key's tokens score in the top 5% of the prompt.
void scout_finds_needle(DeviceContext& device, const qwen::execution::Parameters& parameters,
                        qwen::Frontend& frontend, EngineOptions options) {
    constexpr std::uint32_t kHaystack = 32768;
    const auto filler = frontend.tokenize_text(
        " The grass is green. The sky is blue. The sun is yellow. Here we go. There and back "
        "again.");
    const auto lead     = frontend.tokenize_text(" The pass key is ");
    const auto key      = frontend.tokenize_text("71432");
    const auto trail    = frontend.tokenize_text(". Remember it. 71432 is the pass key.");
    const auto question = frontend.tokenize_text(
        " What is the pass key? Answer with the number only. The pass key is");
    require(!filler.empty() && !key.empty(), "the haystack fixture did not tokenize");
    std::vector<TokenId> ids;
    const auto fill_to = [&](std::size_t target) {
        while (ids.size() + filler.size() <= target) {
            ids.insert(ids.end(), filler.begin(), filler.end());
        }
    };
    fill_to(kHaystack / 2);
    ids.insert(ids.end(), lead.begin(), lead.end());
    const std::size_t key_begin = ids.size();
    ids.insert(ids.end(), key.begin(), key.end());
    ids.insert(ids.end(), trail.begin(), trail.end());
    fill_to(kHaystack - question.size());
    ids.insert(ids.end(), question.begin(), question.end());

    auto fixture  = make_scout_program(device, parameters, options);
    auto& program = *fixture.program;
    ScoutLane lanes(program, device);
    const auto sequence = lanes.private_prefix(ids, 0, 0, frontend);
    const auto scores   = lanes.scout(sequence, 0);
    lanes.abort(sequence);
    require(scores.size() == ids.size(), "the needle scout did not score its whole prompt");

    std::vector<float> sorted = scores;
    std::sort(sorted.begin(), sorted.end(), std::greater<>());
    const float cutoff = sorted[sorted.size() / 20];
    for (std::size_t i = key_begin; i < key_begin + key.size(); ++i) {
        require(scores[i] > cutoff, "pass key token " + std::to_string(i - key_begin) +
                                        " is outside the top 5% of scout scores");
    }
}

// ---- Lossy prefill phases (Task 7) ----

constexpr std::uint32_t kLossyCapacity = 45056;
constexpr std::uint32_t kLossyPrompt   = 40960;
constexpr std::uint32_t kLossyMarker   = 2048; // M: the product prompt's shared marker

// One rk4v4 Program with the lossy path on (threshold 1), two lanes and a context cache.
ScoutFixture make_lossy_program(DeviceContext& device, const qwen::execution::Parameters& parameters,
                                EngineOptions options, std::uint32_t capacity, float keep) {
    options.max_context                       = capacity;
    options.kv_capacity                       = KvCapacityPolicy::explicit_capacity(capacity);
    options.prefill_chunk                     = kScoutChunk;
    options.max_concurrency                   = 2;
    options.kv_cache                          = KvCacheStorage::RotatedLloyd4KeyInt4Value;
    options.use_cuda_graph                    = false;
    options.context_cache.enabled             = true;
    options.context_cache.device_state_slots  = 4;
    options.context_cache.host_capacity_bytes = 0;
    options.lossy_prefill.min_tokens          = 1;
    options.lossy_prefill.filter_layer        = kScoutLayer;
    options.lossy_prefill.query_rows          = kScoutRows;
    options.lossy_prefill.keep_fraction       = keep;
    // Startup refuses a keep share above the skip share (keep-all runs keep 1.0).
    options.lossy_prefill.skip_fraction = std::max(keep, options.lossy_prefill.skip_fraction);
    auto planner = qwen::detail::make_sequence_planner_impl(parameters, device, options);
    const std::uint32_t pages = 2 * planner->curve.minimum_main_page_groups;
    ScoutFixture out;
    out.plan    = qwen::detail::finalize_sequence_plan_impl(std::move(planner), pages);
    out.program = std::make_unique<qwen::detail::ProgramImpl>(parameters, *out.plan, device,
                                                              StartupObserver{});
    require(out.program->scout_buffers.has_value(), "lossy prefill planned no scout scratch");
    return out;
}

// A text prompt of `ids` with a SharedStablePrefix marker at kLossyMarker, planned with prefix reuse.
qwen::RequestBasePlan lossy_request(qwen::detail::ProgramImpl& program, qwen::Frontend& frontend,
                                    std::vector<TokenId> ids, std::uint32_t output_tokens = 64) {
    auto prepared = qwen::PreparedPromptAccess::take(frontend.prepare_tokens(std::move(ids)));
    prepared.context_cache.opportunities.push_back(
        {.kind     = PromptCacheMarkerKind::SharedStablePrefix,
         .evidence = SharedCandidateEvidence::ExplicitBoundary,
         .frontier = kLossyMarker});
    // M is the product prompt's end (lossy_planned_exact_end); an explicit marker alone no
    // longer sets it, so the raw-token prompt declares its product prompt ending at the marker.
    prepared.context_cache.product_prompt_end = kLossyMarker;
    runtime::ResolvedExecutionOptions request;
    request.requested_output_tokens = output_tokens;
    request.allow_prefix_reuse      = true;
    return program.plan_request(std::move(prepared), request);
}

struct LossyRun {
    qwen::SequenceHandle sequence;
    TokenId first = -1;
    std::vector<std::byte> logits;
    std::vector<std::byte> kv;
    std::vector<std::byte> state;
    std::uint32_t scouted = 0;
    std::uint32_t prompt_tokens = 0; // what the lane prefilled (compacted when lossy)
    std::optional<qwen::LossyOutcome> outcome;
    // Main KV pages the lane maps right after the step that ended the scout.
    std::optional<std::uint32_t> pages_after_scout;
    std::vector<qwen::CheckpointHandle> shared; // the exact capture at M
};

// Binds `base` on `lane` (from `source`, the root when empty), asks for the lossy path when
// `enable` is set (requiring enable_lossy to answer *enable), and prefills to the first token. The SharedPrefix capture at M is taken; every other capture is
// skipped. The first token is committed as terminal when `terminal`, else as an ordinary token.
LossyRun lossy_prefill(qwen::detail::ProgramImpl& program, DeviceContext& device,
                       qwen::RequestBasePlan& base, std::uint32_t lane,
                       std::optional<bool> enable, bool terminal,
                       std::optional<qwen::SourceCandidate> source = std::nullopt) {
    ScoutLane lanes(program, device);
    if (!source) { source = program.inspect_source(base, std::nullopt); }
    require(source.has_value(), "lossy test request has no source");
    LossyRun run;
    run.sequence = lanes.bind(base, lane, *source);
    if (enable) {
        require(program.enable_lossy(run.sequence) == *enable,
                "enable_lossy did not answer as expected");
    }
    run.prompt_tokens = base.summary().prompt_tokens;
    for (unsigned call = 0; call < 512; ++call) {
        lanes.reserve(run.sequence);
        auto step = program.advance_prefill(run.sequence, nullptr, nullptr);
        run.scouted += step.scouted_tokens;
        require(step.scouted_tokens == 0 || step.processed_prompt_tokens == 0,
                "a scout step committed prompt tokens");
        if (step.compacted_prompt_tokens) {
            require(!run.outcome && step.lossy.has_value(), "the scout ended twice");
            run.outcome       = step.lossy;
            run.prompt_tokens = *step.compacted_prompt_tokens;
            run.pages_after_scout = program.text_kv_addresses->mapped_pages(
                program.sequences[lane].kv->text);
        }
        if (step.capture_ready) {
            if (program.sequences[lane].text_kv_valid == kLossyMarker &&
                program.start_capture(run.sequence)) {
                const auto captured = settle_positions(program, device);
                for (const auto point : captured.captured_checkpoints) {
                    run.shared.push_back(point);
                }
            } else {
                program.skip_capture(run.sequence);
            }
        }
        if (!step.pending) { continue; }
        device.synchronize();
        run.first = step.pending->tokens().front();
        append_device_bytes(run.logits, program.io.logits.slice(1, 0, 1));
        run.kv = text_kv_bytes(program, lane, run.prompt_tokens);
        const auto& layout = program.state_images->host_layout();
        run.state.resize(layout.image_bytes);
        const auto selectors = program.state_selectors(program.sequences[lane]);
        program.state_images->copy_to_host(selectors.destination,
                                           qwen::HostStateImageView{run.state.data(), &layout},
                                           RankStreams(device.stream));
        device.synchronize();
        const std::array<runtime::CommitDecision, 1> accepted{
            {{.accepted_tokens = 1, .terminal = terminal}}};
        const auto committed = program.commit(std::move(*step.pending), accepted, {}, nullptr);
        if (committed.capture_ready[0]) { program.skip_capture(run.sequence); }
        return run;
    }
    throw std::runtime_error("lossy test prefill did not finish its prompt");
}

std::vector<TokenId> greedy_tokens(qwen::detail::ProgramImpl& program, qwen::SequenceHandle sequence,
                                   std::uint32_t count) {
    std::vector<TokenId> tokens;
    for (std::uint32_t i = 0; i < count; ++i) {
        const std::array<qwen::ExecutionUnit, 1> units{
            {{sequence, qwen::ExecutionUnitKind::Decode, count - i}}};
        require(static_cast<bool>(program.reserve_units(units)), "greedy decode could not reserve");
        const std::array<qwen::SequenceHandle, 1> rows{sequence};
        const std::array<runtime::RoundBudget, 1> budgets{
            {{.generated_tokens_remaining = count - i}}};
        auto pending = program.decode(rows, budgets, nullptr, nullptr);
        tokens.push_back(pending.tokens().front());
        const std::array<runtime::CommitDecision, 1> accepted{{{.accepted_tokens = 1}}};
        (void)program.commit(std::move(pending), accepted, {}, nullptr);
    }
    return tokens;
}

// LossyKeepAllMatchesOrdinaryPrefill: keep 1.0 keeps all of U (the skip rule), so after the scout
// the request is an ordinary prefill from M. The first token, its logits, the prompt's KV, the GDN
// state and 64 greedy tokens equal a request on the same Program that never took the lossy path,
// bit for bit. Both stop at the capture at M, so their chunks are cut the same way.
void lossy_keep_all_matches_ordinary_prefill(DeviceContext& device,
                                             const qwen::execution::Parameters& parameters,
                                             qwen::Frontend& frontend, EngineOptions options) {
    auto fixture  = make_lossy_program(device, parameters, options, kScoutCapacity, 1.0F);
    auto& program = *fixture.program;
    const auto empty = program.physical_usage().occupied;
    const auto ids   = long_prompt_ids(kScoutPrefix + kScoutSpan);
    const auto run   = [&](bool lossy) {
        zero_text_kv(program, device);
        auto base = lossy_request(program, frontend, ids);
        require(base.lossy().has_value() && base.lossy()->exact_end == kLossyMarker,
                "the plan does not offer the lossy path at the marker");
        auto result = lossy_prefill(program, device, base, 0,
                                    lossy ? std::optional<bool>(true) : std::nullopt, false);
        // The first token and 63 more: 64 greedy tokens.
        auto tokens = greedy_tokens(program, result.sequence, 63);
        tokens.insert(tokens.begin(), result.first);
        require(program.abort(result.sequence).status == runtime::ConsumeStatus::Consumed,
                "keep-all lane retained its execution resources");
        for (const auto point : result.shared) {
            require(program.release_checkpoint(point), "the shared prefix kept a reader");
        }
        return std::pair{std::move(result), std::move(tokens)};
    };
    const auto reference = run(false);
    const auto lossy     = run(true);
    require(reference.first.scouted == 0 && !reference.first.outcome, "the reference scouted");
    require(lossy.first.scouted == kScoutPrefix + kScoutSpan - kLossyMarker,
            "the lossy request did not scout all of U");
    require(lossy.first.outcome && lossy.first.outcome->skipped &&
                lossy.first.outcome->kept_tokens == lossy.first.outcome->uncached_tokens,
            "keep 1.0 did not keep all of U");
    require(lossy.first.prompt_tokens == reference.first.prompt_tokens,
            "keep 1.0 changed the prompt length");
    require(lossy.first.first == reference.first.first, "keep-all: first token differs");
    require(lossy.first.logits == reference.first.logits, "keep-all: last-token logits differ");
    require(lossy.first.kv == reference.first.kv, "keep-all: prompt KV differs");
    require(lossy.first.state == reference.first.state, "keep-all: GDN state differs");
    require(lossy.second == reference.second, "keep-all: greedy tokens differ");
    require(program.physical_usage().occupied == empty, "keep-all runs leaked resources");
}

// lossy_releases_pages_above_kept_frontier: a 40K prompt at keep 0.2. During the scout the lane
// maps pages past M; the step that ends it leaves only M's pages, and once the first token is
// committed the lane holds the pages of M + kept + 1 tokens, not of the original prompt.
// LossyFlow.ExactRequestNeverBindsSelectedState: the same prompt planned before the lossy run (so
// not rewritten) matches the exact shared capture at M but none of the selected checkpoints.
// Follow-up: the original prompt + the first token + a new tail plans onto the kept identity
// (shorter prompt_tokens, original reported_prompt_tokens, no lossy path), reuses the selected
// continuation and binds it; once every holder is gone the map dies and the same prompt plans as
// itself again.
void lossy_selected_lineage(DeviceContext& device, const qwen::execution::Parameters& parameters,
                            qwen::Frontend& frontend, EngineOptions options) {
    auto fixture  = make_lossy_program(device, parameters, options, kLossyCapacity, 0.2F);
    auto& program = *fixture.program;
    const auto empty = program.physical_usage().occupied;
    const auto ids   = long_prompt_ids(kLossyPrompt);

    auto exact = lossy_request(program, frontend, ids); // planned before any gap map exists
    auto base  = lossy_request(program, frontend, ids);
    require(base.summary().prompt_tokens == kLossyPrompt &&
                base.summary().reported_prompt_tokens == kLossyPrompt,
            "a cold prompt was rewritten");
    auto run = lossy_prefill(program, device, base, 0, true, true);
    require(run.scouted != 0, "the lossy request did not scout");
    require(run.outcome && !run.outcome->skipped, "keep 0.2 kept everything");
    const auto& outcome = *run.outcome;
    require(outcome.exact_end == kLossyMarker && outcome.uncached_tokens == kLossyPrompt - kLossyMarker,
            "the scout did not cover [M, end)");
    require(run.scouted == outcome.uncached_tokens, "the scout steps did not add up to U");
    require(run.prompt_tokens == kLossyMarker + outcome.kept_tokens,
            "the compacted prompt is not C + kept");
    require(run.pages_after_scout == qwen::detail::kv_pages_for_frontier(kLossyMarker),
            "the scout's pages above M were not released");
    const auto& sequence = program.sequences[0];
    require(sequence.ledger.size() == run.prompt_tokens + 1 && sequence.gap_maps.size() == 1,
            "the lane's ledger is not the kept token list");
    require(sequence.rope_delta == static_cast<std::int32_t>(kLossyPrompt - run.prompt_tokens),
            "generation does not continue at the original prompt length");
    const auto resident = program.text_kv_addresses->mapped_pages(sequence.kv->text);
    require(resident <= qwen::detail::kv_pages_for_frontier(run.prompt_tokens + 1),
            "resident pages after prefill exceed kv_pages_for_frontier(M + kept + 1)");
    require(run.shared.size() == 1, "the exact shared prefix at M was not captured");

    auto finished = program.finish(run.sequence);
    require(finished.status == runtime::ConsumeStatus::Consumed && finished.checkpoint,
            "the lossy request published no continuation");
    const auto selected = *finished.checkpoint;
    require(program.checkpoint(selected).frontier == run.prompt_tokens &&
                program.checkpoint(selected).gap_maps.size() == 1,
            "the selected continuation does not carry its gap map");

    // Exactness: the unrewritten prompt reaches the exact capture at M and nothing selected.
    require(program.checkpoint_matches(run.shared.front(), exact),
            "the exact shared prefix no longer matches the original prompt");
    require(!program.checkpoint_matches(selected, exact) &&
                !program.inspect_source(exact, selected).has_value(),
            "an exact request can bind the selected continuation");

    // Follow-up turn.
    std::vector<TokenId> follow = ids;
    follow.push_back(run.first);
    const auto tail = long_prompt_ids(96);
    follow.insert(follow.end(), tail.begin(), tail.end());
    auto next = lossy_request(program, frontend, follow, 8);
    require(next.summary().reported_prompt_tokens == follow.size() &&
                next.summary().prompt_tokens == run.prompt_tokens + 1 + tail.size(),
            "the follow-up was not rewritten onto the kept identity");
    require(!next.lossy().has_value(), "a rewritten follow-up offers the lossy path again");
    require(program.checkpoint_matches(selected, next), "the follow-up does not match the selection");
    const auto reuse = program.inspect_source(next, selected);
    require(reuse && reuse->reused_tokens == run.prompt_tokens,
            "the follow-up does not reuse the selected continuation");
    auto second = lossy_prefill(program, device, next, 1, false, true, reuse);
    require(second.scouted == 0 && second.first >= 0, "the follow-up scouted or did not finish");
    require(program.sequences[1].gap_maps.size() == 1, "the follow-up lost its lineage");
    auto second_finished = program.finish(second.sequence);
    require(second_finished.status == runtime::ConsumeStatus::Consumed,
            "the follow-up kept its execution resources");

    // Alias removal: with every checkpoint gone nothing holds the map.
    std::vector<qwen::CheckpointHandle> held = run.shared;
    held.push_back(selected);
    if (second_finished.checkpoint) { held.push_back(*second_finished.checkpoint); }
    for (const auto point : held) {
        if (program.valid_checkpoint(point)) {
            require(program.release_checkpoint(point), "a lineage checkpoint kept a reader");
        }
    }
    for (std::size_t index = 0; index < program.checkpoints.size(); ++index) {
        const auto& slot = program.checkpoints[index];
        if (slot.value && !slot.value->gap_maps.empty()) {
            require(program.release_checkpoint({&program, static_cast<std::uint32_t>(index),
                                                slot.generation}),
                    "a lineage capture kept a reader");
        }
    }
    { const auto plan_holds_the_map = std::move(next); } // a live plan holds its lineage too
    auto again = lossy_request(program, frontend, follow, 8);
    require(again.summary().prompt_tokens == follow.size(),
            "a prompt was rewritten by a map no checkpoint holds");
    require(program.physical_usage().occupied == empty, "the lossy lineage leaked resources");
}

// The scout window (between M and the first Main chunk) on a two-lane Program:
// - after the first scout step the lane refuses pause and every capture gate;
// - a second lossy request reaching M while the first scouts declines the lossy path (reported as
//   skipped and declined), prefills all of its prompt and is not cleared;
// - aborting the scouting lane through advance_prefill + abort leaves the shared checkpoint at M
//   (its GDN state and the prefix KV a new reader binds) exactly as before the scout, and every
//   resource returns once the checkpoints are released.
void lossy_scout_window(DeviceContext& device, const qwen::execution::Parameters& parameters,
                        qwen::Frontend& frontend, EngineOptions options) {
    auto fixture  = make_lossy_program(device, parameters, options, kScoutCapacity, 0.2F);
    auto& program = *fixture.program;
    ScoutLane lanes(program, device);
    const auto empty = program.physical_usage().occupied;
    const auto ids   = long_prompt_ids(kScoutPrefix + kScoutSpan);

    auto base           = lossy_request(program, frontend, ids);
    const auto root     = program.inspect_source(base, std::nullopt);
    const auto sequence = lanes.bind(base, 0, *root);
    require(program.enable_lossy(sequence), "the first lossy request was refused");
    std::vector<qwen::CheckpointHandle> shared;
    std::optional<ScoutTrace> before;
    for (unsigned call = 0; call < 64 && !program.scout_active(sequence); ++call) {
        if (program.sequences[0].text_kv_valid == kLossyMarker && !before &&
            !program.requests[0].capture_pending) {
            before = scout_trace(program, device, 0, kLossyMarker);
        }
        lanes.reserve(sequence);
        auto step = program.advance_prefill(sequence, nullptr, nullptr);
        require(!step.pending, "the lossy request finished before scouting");
        if (step.capture_ready) {
            require(program.start_capture(sequence), "the shared prefix at M was not captured");
            const auto captured = settle_positions(program, device);
            shared.insert(shared.end(), captured.captured_checkpoints.begin(),
                          captured.captured_checkpoints.end());
        }
        if (step.scouted_tokens != 0) {
            require(step.processed_prompt_tokens == 0, "a scout step committed prompt tokens");
        }
    }
    require(program.scout_active(sequence) && before && shared.size() == 1,
            "the lossy request did not reach its scout after an exact capture at M");

    // The window refuses pause and capture.
    require(!program.start_pause(sequence, true, nullptr, false), "a scouting lane paused");
    require(!program.start_pause(sequence, true, nullptr, true), "a scouting lane paused (yield)");
    lanes.reserve(sequence);
    require(!program.prepare_capture(sequence).has_value(), "a scouting lane prepared a capture");
    require(!program.start_capture(sequence), "a scouting lane started a capture");
    require(!program.capture_is_input(sequence), "a scouting lane reports an input capture");

    // A second lossy request reaching M declines and prefills normally.
    {
        auto other = lossy_request(program, frontend, long_prompt_ids(kScoutPrefix + kScoutSpan + 128));
        // Bound from the shared checkpoint, it is at M at once: its first step meets the busy scout.
        const auto at_m = program.inspect_source(other, shared.front());
        require(at_m && at_m->reused_tokens == kLossyMarker, "the second request does not reuse M");
        auto run = lossy_prefill(program, device, other, 1, true, false, at_m);
        require(run.outcome && run.outcome->declined && run.outcome->skipped &&
                    run.outcome->kept_tokens == run.outcome->uncached_tokens,
                "the second lossy request did not decline the busy scout");
        require(run.scouted == 0 && run.first >= 0 &&
                    run.prompt_tokens == kScoutPrefix + kScoutSpan + 128,
                "the declined request did not prefill all of its prompt");
        require(program.scout_active(sequence), "the declined request disturbed the scout");
        require(program.abort(run.sequence).status == runtime::ConsumeStatus::Consumed,
                "the declined lane retained its execution resources");
        for (const auto point : run.shared) {
            require(program.release_checkpoint(point), "the declined lane's capture kept a reader");
        }
    }

    // Abort mid-scout, then look at what a new reader of the checkpoint at M finds.
    lanes.reserve(sequence);
    const auto step = program.advance_prefill(sequence, nullptr, nullptr);
    require(step.scouted_tokens != 0 && !step.compacted_prompt_tokens,
            "the second scout step did not run mid-scout");
    lanes.abort(sequence);
    require(!program.scout_session.has_value(), "an aborted lane kept its scout");
    const auto source = program.inspect_source(base, shared.front());
    require(source && source->reused_tokens == kLossyMarker, "the checkpoint at M stopped matching");
    const auto reader       = lanes.bind(base, 0, *source);
    const ScoutTrace after  = scout_trace(program, device, 0, kLossyMarker);
    require(after.state == before->state, "aborted scout: the GDN state at M changed");
    require(after.kv == before->kv, "aborted scout: the prefix KV at M changed");
    lanes.abort(reader);
    for (const auto point : shared) {
        require(program.release_checkpoint(point), "the shared prefix kept a reader");
    }
    require(program.physical_usage().occupied == empty, "the aborted scout leaked resources");
}

// Program::abort_lossy_scout, the Engine's way to pause a scouting lane under memory pressure:
// after one scout step it ends the scout with the lane back at M exactly as the scout found it (GDN
// state and prefix KV bit-identical, no page past M kept), the request continues as an ordinary
// prefill, and the lane can then be paused.
void lossy_scout_abort_then_pause(DeviceContext& device, const qwen::execution::Parameters& parameters,
                                  qwen::Frontend& frontend, EngineOptions options) {
    auto fixture  = make_lossy_program(device, parameters, options, kScoutCapacity, 0.2F);
    auto& program = *fixture.program;
    ScoutLane lanes(program, device);
    const auto empty = program.physical_usage().occupied;
    auto base           = lossy_request(program, frontend, long_prompt_ids(kScoutPrefix + kScoutSpan));
    const auto root     = program.inspect_source(base, std::nullopt);
    const auto sequence = lanes.bind(base, 0, *root);
    require(program.enable_lossy(sequence), "the lossy request was refused");
    require(!program.abort_lossy_scout(sequence), "abort_lossy_scout ended a scout that never began");
    std::vector<qwen::CheckpointHandle> shared;
    std::optional<ScoutTrace> at_m;
    for (unsigned call = 0; call < 64 && !program.scout_active(sequence); ++call) {
        lanes.reserve(sequence);
        auto step = program.advance_prefill(sequence, nullptr, nullptr);
        require(!step.pending, "the lossy request finished before scouting");
        if (step.capture_ready) {
            require(program.start_capture(sequence), "the shared prefix at M was not captured");
            const auto captured = settle_positions(program, device);
            shared.insert(shared.end(), captured.captured_checkpoints.begin(),
                          captured.captured_checkpoints.end());
        }
        // The lane at M, settled, just before the step that opens the scout.
        if (program.sequences[0].text_kv_valid == kLossyMarker && !program.scout_active(sequence) &&
            !program.requests[0].capture_pending) {
            at_m = scout_trace(program, device, 0, kLossyMarker);
        }
    }
    require(program.scout_active(sequence) && at_m, "the lossy request did not reach its scout");
    require(!program.start_pause(sequence, false, nullptr, false), "a scouting lane paused");

    require(program.abort_lossy_scout(sequence), "abort_lossy_scout refused a scouting lane");
    require(!program.scout_active(sequence) && !program.scout_session.has_value(),
            "abort_lossy_scout left the scout open");
    require_no_trace(scout_trace(program, device, 0, kLossyMarker), *at_m, "aborted scout at M");
    require(program.sequences[0].text_kv_valid == kLossyMarker &&
                program.requests[0].prefill->cursor == kLossyMarker,
            "the aborted scout moved the lane away from M");
    require(!program.abort_lossy_scout(sequence), "a second abort_lossy_scout ended a scout");

    // Pausable again; the pause saves the lane at M.
    require(program.start_pause(sequence, true, nullptr, false),
            "the lane did not pause after its scout ended");
    auto paused = settle_positions(program, device);
    require(paused.published && paused.paused && paused.paused->frontier() == kLossyMarker,
            "the pause after an aborted scout did not save the lane at M");
    paused.paused.reset();
    for (const auto point : shared) {
        require(program.release_checkpoint(point), "the shared prefix kept a reader");
    }
    require(program.physical_usage().occupied == empty, "the aborted scout and pause leaked resources");
}

SpeculativeBackend selected_backend(std::string_view name) {
    if (name == "none") { return SpeculativeBackend::None; }
    if (name == "mtp") { return SpeculativeBackend::Mtp; }
    if (name == "dflash") { return SpeculativeBackend::DFlash; }
    if (name == "dflash2") { return SpeculativeBackend::DFlash2; }
    throw std::invalid_argument("backend must be none, mtp, dflash or dflash2");
}

} // namespace

int main(int argc, char** argv) {
    const char* artifact = std::getenv("NINFER_TEST_ARTIFACT");
    if (!artifact || !*artifact) {
        std::cout << "SKIP: set NINFER_TEST_ARTIFACT to an explicit .ninfer artifact\n";
        return 77;
    }
    try {
        require(argc <= 2,
                "usage: ninfer_qwen3_5_native_transactions_test [none|mtp|dflash|dflash2]");
        const auto backend = selected_backend(argc == 2 ? argv[1] : "none");
        DeviceContext device;
        models::LoadOptions selected;
        selected.speculative = backend;
        // The masked-draft backends propose through the optimized head, as the launchers run them.
        const bool masked_draft =
            backend == SpeculativeBackend::DFlash || backend == SpeculativeBackend::DFlash2;
        if (masked_draft) { selected.proposal_head = ProposalHead::Optimized; }
        auto model = qwen::load_model(artifact, selected, device);
        qwen::execution::Parameters parameters(*model);
        auto frontend = qwen::make_frontend(model->resources(),
                                            {.vision_enabled = false, .max_context = kCapacity});
        EngineOptions options;
        options.max_context              = kCapacity;
        options.kv_capacity              = KvCapacityPolicy::explicit_capacity(2 * kCapacity);
        options.prefill_chunk            = kChunk;
        options.max_concurrency          = 2;
        options.kv_cache                 = KvCacheStorage::Fp8E4M3Row256;
        options.use_cuda_graph           = false;
        options.speculative.backend      = backend;
        options.speculative.draft_tokens = backend == SpeculativeBackend::None ? 0U : 3U;
        if (masked_draft) { options.speculative.proposal_head = ProposalHead::Optimized; }
        options.context_cache.device_state_slots  = 2;
        options.context_cache.host_capacity_bytes = 512ULL * 1024 * 1024;
        auto planner = qwen::make_sequence_planner(parameters, device, options);
        auto plan    = std::move(planner).finalize(16);
        auto program = qwen::create_program(parameters, std::move(plan), device, {});
        {
            Fixture fixture(device, *program, frontend);
            fixture.capture_opportunities();
            fixture.input_replay_lifecycle();
            fixture.active_input_zero_growth_binding();
            fixture.replay_carries_independent_input();
            fixture.retained_input_host_suffix();
            fixture.retired_host_image_preserves_input();
            fixture.raw_identity_opt_out();
            fixture.cancel_gpu_binding();
            fixture.cancel_host_restore();
            fixture.cancel_and_destroy_paused_prefill();
            fixture.replay_pause_and_resume();
            fixture.independent_physical_demotions();
            fixture.demotion_holder_changes();
            fixture.batched_physical_demotions();
            fixture.physical_facts();
            fixture.grammar_row_failure();
            fixture.checkpoint_image_round_trip();
            fixture.shared_prefix_image_round_trip();
        }
        program.reset();
        options.kv_capacity    = KvCapacityPolicy::explicit_capacity(kCapacity);
        auto full_pool_planner = qwen::make_sequence_planner(parameters, device, options);
        auto full_pool_plan    = std::move(full_pool_planner).finalize(8);
        auto full_pool = qwen::create_program(parameters, std::move(full_pool_plan), device, {});
        {
            Fixture full_pool_fixture(device, *full_pool, frontend);
            full_pool_fixture.rewind_binding_retirement();
            full_pool_fixture.fork_exceeds_physical_capacity();
            full_pool_fixture.recovery_capacity_and_progress();
            full_pool_fixture.restore_full_pool_snapshot();
        }
        full_pool.reset();
        options.max_context                       = kCapacity;
        options.max_concurrency                   = 1;
        options.context_cache.device_state_slots  = 0;
        options.context_cache.host_capacity_bytes = 0;
        auto capture_planner = qwen::make_sequence_planner(parameters, device, options);
        auto capture_plan    = std::move(capture_planner).finalize(8);
        auto no_capture_destination =
            qwen::create_program(parameters, std::move(capture_plan), device, {});
        {
            Fixture no_capture_fixture(device, *no_capture_destination, frontend);
            no_capture_fixture.capture_without_destination();
            no_capture_fixture.private_move();
        }
        no_capture_destination.reset();
        options.context_cache.device_state_slots = 1;
        auto capture_retry_planner = qwen::make_sequence_planner(parameters, device, options);
        auto capture_retry_plan    = std::move(capture_retry_planner).finalize(8);
        auto capture_retry =
            qwen::create_program(parameters, std::move(capture_retry_plan), device, {});
        {
            Fixture capture_retry_fixture(device, *capture_retry, frontend);
            capture_retry_fixture.capture_retry_after_reclaim();
        }
        capture_retry.reset();
        shared_capture_alignment(device, parameters, frontend, options);
        progress_anchor_planning(device, parameters, frontend, options);
        prompt_fits_beside_residents_on_real_pools(device, parameters, frontend, options);
        capped_prefill_steps(device, parameters, frontend, options);
        sliced_prefill_matches_whole(device, parameters, frontend, options);
        sliced_prefill_beside_a_decoding_lane(device, parameters, frontend, options);
        replay_sampling_counts(device, parameters, frontend, options);
        // A 3,000-token prompt needs a Frontend with a longer context than the fixtures above.
        auto long_frontend = qwen::make_frontend(
            model->resources(), {.vision_enabled = false, .max_context = kLongCapacity});
        explicit_positions_equal_rope_delta_path(device, parameters, long_frontend, options);
        explicit_positions_with_layer_slices(device, parameters, long_frontend, options);
        replay_preserves_gapped_positions(device, parameters, long_frontend, options);
        // The scout tests need a 32K-token Frontend.
        auto scout_frontend = qwen::make_frontend(
            model->resources(), {.vision_enabled = false, .max_context = kScoutCapacity});
        scout_leaves_no_trace(device, parameters, scout_frontend, options);
        scout_abort_leaves_no_trace(device, parameters, scout_frontend, options);
        scout_scores_chunk_invariant(device, parameters, scout_frontend, options);
        scout_finds_needle(device, parameters, scout_frontend, options);
        lossy_keep_all_matches_ordinary_prefill(device, parameters, scout_frontend, options);
        lossy_scout_window(device, parameters, scout_frontend, options);
        lossy_scout_abort_then_pause(device, parameters, scout_frontend, options);
        // The 40K lossy prompts need a Frontend with a longer context.
        auto lossy_frontend = qwen::make_frontend(
            model->resources(), {.vision_enabled = false, .max_context = kLossyCapacity});
        lossy_selected_lineage(device, parameters, lossy_frontend, options);
        std::cout << "OK native transaction cancellation, paused ownership and replay recovery\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL native transactions: " << error.what() << '\n';
        return 1;
    }
}
