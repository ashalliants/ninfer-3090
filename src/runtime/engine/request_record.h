#pragma once

#include "core/nvtx.h"
#include "ninfer/types.h"
#include "runtime/contract/execution.h"
#include "runtime/engine/generation_budget.h"
#include "runtime/engine/context_cache/types.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <exception>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace ninfer::runtime {

enum class RequestEngineHostPhase : std::uint8_t {
    Boundary,
    CommitOutput,
    Maintenance,
};

struct RequestHostTiming {
    std::uint64_t queue_wait_ns                   = 0;
    std::uint64_t engine_boundary_exposed_ns      = 0;
    std::uint64_t program_submit_exposed_ns       = 0;
    std::uint64_t program_post_exposed_ns         = 0;
    std::uint64_t engine_commit_output_exposed_ns = 0;
    std::uint64_t engine_maintenance_exposed_ns   = 0;
    std::uint64_t device_wait_exposed_ns          = 0;
    std::uint64_t constraint_draft_wait_exposed_ns = 0;
    std::uint64_t decode_host_exposed_ns          = 0;
    std::uint64_t decode_device_wait_exposed_ns   = 0;
    std::uint64_t prefill_units                   = 0;
    std::uint64_t decode_rounds                   = 0;
    std::uint64_t control_units                   = 0;

    void expose_engine(RequestEngineHostPhase phase, std::uint64_t elapsed_ns,
                       bool decode_member) noexcept {
        switch (phase) {
        case RequestEngineHostPhase::Boundary:
            engine_boundary_exposed_ns += elapsed_ns;
            break;
        case RequestEngineHostPhase::CommitOutput:
            engine_commit_output_exposed_ns += elapsed_ns;
            break;
        case RequestEngineHostPhase::Maintenance:
            engine_maintenance_exposed_ns += elapsed_ns;
            break;
        }
        if (decode_member) { decode_host_exposed_ns += elapsed_ns; }
    }

    void expose_program(ExecutionTiming timing, bool decode_member) noexcept {
        program_submit_exposed_ns += timing.submit_host_ns;
        program_post_exposed_ns += timing.post_host_ns;
        device_wait_exposed_ns += timing.device_wait_ns;
        constraint_draft_wait_exposed_ns += timing.constraint_draft_wait_ns;
        if (decode_member) {
            decode_host_exposed_ns += timing.host_ns();
            decode_device_wait_exposed_ns += timing.device_wait_ns;
        }
    }

    [[nodiscard]] GenerationEngineTiming public_snapshot() const noexcept {
        constexpr double kNanosecondsToSeconds = 1.0e-9;
        return GenerationEngineTiming{
            .queue_wait_seconds = static_cast<double>(queue_wait_ns) * kNanosecondsToSeconds,
            .engine_boundary_exposed_seconds =
                static_cast<double>(engine_boundary_exposed_ns) * kNanosecondsToSeconds,
            .program_submit_exposed_seconds =
                static_cast<double>(program_submit_exposed_ns) * kNanosecondsToSeconds,
            .program_post_exposed_seconds =
                static_cast<double>(program_post_exposed_ns) * kNanosecondsToSeconds,
            .engine_commit_output_exposed_seconds =
                static_cast<double>(engine_commit_output_exposed_ns) * kNanosecondsToSeconds,
            .engine_maintenance_exposed_seconds =
                static_cast<double>(engine_maintenance_exposed_ns) * kNanosecondsToSeconds,
            .device_wait_exposed_seconds =
                static_cast<double>(device_wait_exposed_ns) * kNanosecondsToSeconds,
            .constraint_draft_wait_exposed_seconds =
                static_cast<double>(constraint_draft_wait_exposed_ns) * kNanosecondsToSeconds,
            .decode_host_exposed_seconds =
                static_cast<double>(decode_host_exposed_ns) * kNanosecondsToSeconds,
            .decode_device_wait_exposed_seconds =
                static_cast<double>(decode_device_wait_exposed_ns) * kNanosecondsToSeconds,
            .prefill_units = prefill_units,
            .decode_rounds = decode_rounds,
            .control_units = control_units,
        };
    }
};

enum class EngineRequestState : std::uint8_t {
    Waiting,
    Materializing,
    Prefill,
    Replay,
    Pausing,
    Paused,
    DecodeReady,
    ControlReady,
    ModelFinished,
};

template <class ModelContract>
struct RequestRecord {
    using Clock          = std::chrono::steady_clock;
    using PreparedPrompt = typename ModelContract::PreparedPrompt;
    using OutputSession  = typename ModelContract::OutputSession;
    using BasePlan       = typename ModelContract::RequestBasePlan;
    using SequenceHandle = typename ModelContract::SequenceHandle;
    using ResumeState    = typename ModelContract::ResumeState;
    using Checkpoint     = typename ModelContract::CheckpointHandle;
    using StreamEvent = std::variant<GenerationTimingObservation, GenerationFirstTokenObservation,
                                     OutputDelta, std::unique_ptr<GenerationSchedulingObservation>>;

    RequestRecord(std::uint64_t request_identity, std::uint64_t publication_sequence,
                  PreparedPrompt input, OutputSession output_session, PromptSummary summary,
                  double frontend_seconds, ResolvedRequestOptions request_options,
                  OutputConsumerMode output_consumer, GenerationObservationOptions observation,
                  Clock::time_point limit, Clock::time_point submit_time)
        : generation_range(nvtx::Name::Generate, nvtx::Category::Runtime, request_identity),
          id(request_identity), publication_order(publication_sequence), prompt(std::move(input)),
          output(std::move(output_session)), prompt_summary(std::move(summary)),
          prepare_seconds(frontend_seconds), options(std::move(request_options)),
          consumer_mode(output_consumer), observation(std::move(observation)), deadline(limit),
          submitted(submit_time) {}

    RequestRecord(const RequestRecord&)            = delete;
    RequestRecord& operator=(const RequestRecord&) = delete;

    [[nodiscard]] bool is_prefilling() const noexcept {
        return model_state == EngineRequestState::Prefill;
    }

    [[nodiscard]] bool is_replaying() const noexcept {
        return model_state == EngineRequestState::Replay;
    }

    [[nodiscard]] bool is_materializing() const noexcept {
        return model_state == EngineRequestState::Materializing;
    }

    [[nodiscard]] bool is_decode_ready() const noexcept {
        return model_state == EngineRequestState::DecodeReady;
    }

    [[nodiscard]] bool is_control_ready() const noexcept {
        return model_state == EngineRequestState::ControlReady;
    }

    [[nodiscard]] bool is_model_finished() const noexcept {
        return model_state == EngineRequestState::ModelFinished;
    }

    // The process range follows Request ownership across submit, worker, and consumer threads.
    nvtx::ScopedAsyncRange generation_range;
    const std::uint64_t id;
    const std::uint64_t publication_order;
    PreparedPrompt prompt;
    OutputSession output;
    PromptSummary prompt_summary;
    double prepare_seconds = 0.0;
    ResolvedRequestOptions options;
    const OutputConsumerMode consumer_mode;
    const GenerationObservationOptions observation;
    Clock::time_point deadline;
    Clock::time_point submitted;
    std::optional<Clock::time_point> admitted_at;
    std::optional<Clock::time_point> first_token;
    std::optional<Clock::time_point> last_token;
    bool queue_wait_recorded = false;
    std::optional<GenerationBudget> budget;
    std::optional<BeginSummary> admitted_begin;
    std::optional<BeginSummary> begin;
    std::vector<TokenId> generated;
    std::string content;
    std::string reasoning;
    std::optional<LaneId> lane;
    std::optional<SequenceHandle> sequence;
    std::atomic<bool> cancelled{false};
    EngineRequestState model_state        = EngineRequestState::Waiting;
    bool capture_pending                  = false;
    EngineRequestState post_capture_state = EngineRequestState::Prefill;
    std::optional<FinishReason> terminal_reason;

    std::optional<BasePlan> base_plan;
    std::optional<ResumeState> suspended;
    ContinuationOwnerToken continuation_owner = 0;
    // Shared checkpoints this request published at its progress anchors, oldest first. Only the
    // newest few are kept (EngineCore::trim_progress_anchors).
    std::vector<Checkpoint> progress_anchors;
    std::uint64_t device_to_host_bytes        = 0;
    std::uint64_t host_to_device_bytes        = 0;
    EngineRequestState resume_phase           = EngineRequestState::Prefill;
    // The last pause interrupted a replay (resume_phase is then still the phase from before the
    // replay's own pause), so RuntimeStats::lanes reports the pausing lane as prefilling.
    bool paused_while_replaying               = false;
    std::uint32_t admission_bypasses          = 0;
    // Times a younger waiter resuming a prefix held on the device was admitted ahead of this one
    // (EngineOptions::admission_affinity; Scheduler::kMaxAffinityOvertakes bounds it).
    std::uint32_t overtakes                   = 0;
    // The affinity answer for this waiter, computed once per admission_generation (0: never).
    std::uint64_t affinity_generation         = 0;
    bool affine                               = false;
    std::uint64_t admission_generation        = 0;
    bool admission_observed                   = false;
    GenerationAdmissionStats admission;
    std::optional<Clock::time_point> source_wait_started;
    // Context store hydration: the store is consulted once per request. While `hydrating`, a host
    // worker reads the stored image and the request is not offered for admission.
    bool store_probed                   = false;
    // With a bucket: the store found nothing, so its listing is being refreshed (once per request)
    // in case another engine stored this session since; the store is then consulted again.
    bool store_lookup                   = false;
    bool store_looked_up                = false;
    bool hydrating                      = false;
    bool hydrated                       = false;
    std::uint32_t hydration_frontier    = 0;
    std::uint32_t hydration_reuse_before = 0;
    Clock::time_point hydration_started{};
    Clock::time_point hydration_deadline{};

    // Single flight: while set, a resident sibling is prefilling this request's prefix towards a
    // checkpoint it will publish, and the request is not offered for admission. One wait at most.
    bool awaiting_sibling  = false;
    bool sibling_wait_used = false;
    // Counted once: admission held this request back because its whole prompt could not fit
    // beside the resident requests (EngineOptions::admission_fit_check).
    bool fit_deferred = false;

    // A prefill chunk is part-way through its layers (PrefillScheduleOptions::layer_slices): the
    // lane must be advanced again before anything else claims the Program's parking buffer, and
    // cannot be paused until it finishes. `slice_elapsed_ns` and `slice_prefix` describe the step
    // for the pacing model, which sees the step whole.
    bool slice_in_flight                   = false;
    std::uint64_t slice_elapsed_ns         = 0;
    std::uint64_t slice_wall_ns            = 0;
    std::uint32_t slice_planned_tokens     = 0; // the chunk's planned width before a stall cap // worker wall time of the slices so far
    std::uint32_t slice_prefix             = 0;
    bool recovery_pending                  = false;
    std::uint64_t committed_decode_tokens  = 0;
    GenerationRecoveryRoute recovery_route = GenerationRecoveryRoute::None;
    std::uint64_t preemption_count         = 0;
    std::uint64_t replay_restores          = 0;
    std::uint64_t snapshot_restores        = 0;
    std::uint64_t replayed_tokens          = 0;
    std::uint64_t paused_ns                = 0;
    std::optional<Clock::time_point> paused_at;
    std::uint32_t computed_prompt_tokens = 0;
    GenerationTimings generation_timings;
    RequestHostTiming host_timing;
    std::uint64_t initial_binding_ns = 0;
    GenerationWorkTiming prefill_work;
    GenerationWorkTiming replay_work;
    std::array<std::array<GenerationTransferTiming, 3>, 3> context_transfers{};
    std::optional<GenerationFirstOutputTiming> first_output_timing;
    SpeculativeStats speculative_stats;

    std::mutex mutex;
    std::condition_variable cv;
    std::optional<GenerationStart> stream_start;
    std::optional<PromptProgress> stream_progress;
    std::vector<StreamEvent> events;
    GenerationResult result;
    std::exception_ptr error;
    bool response_done     = false;
    bool consumer_released = false;
    bool capacity_released = false;
};

} // namespace ninfer::runtime
