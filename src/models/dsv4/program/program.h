#pragma once

// Written for NInfer-3090.
//
// The DeepSeek-V4.1-Flash Program: the smallest implementation of the execution contract EngineCore
// drives (the surface of models/qwen4_exp/program/program.h) over execution/forward.h. One lane, one
// request at a time (a second request waits in the Engine's queue), the whole request's extent reserved
// at admission, the forward run synchronously, sampling on the host over the FP32 logits.
//
// Not here, each refused or reported as absent the way qwen4_exp's Program does for what it lacks: the
// context cache and every checkpoint (prefix reuse, the context store), pausing, speculation, vision,
// prompt grafts, log-probabilities, the decode graph. docs/maintainer/dsv4-flash-plan.md, milestones 5-7.

#include "models/dsv4/execution/forward.h"
#include "models/dsv4/model.h"
#include "models/dsv4/program/host_sampling.h"
#include "models/qwen3_5/frontend/frontend.h"
#include "models/qwen3_5/frontend/graft.h"
#include "models/qwen3_5/frontend/output_session.h"
#include "models/qwen3_5/frontend/prepared_prompt.h"
#include "models/qwen3_5/program/program.h"
#include "ninfer/types.h"
#include "runtime/contract/execution.h"
#include "runtime/contract/request.h"
#include "runtime/contract/resources.h"
#include "runtime/contract/timing.h"

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <random>
#include <span>
#include <string_view>
#include <utility>
#include <vector>

namespace ninfer {
struct DeviceContext;
}

namespace ninfer::models::dsv4 {

using PreparedPrompt      = qwen3_5::PreparedPrompt;
using Frontend            = qwen3_5::Frontend;
using OutputSession       = qwen3_5::OutputSession;
using PublishedOutput     = qwen3_5::PublishedOutput;
using PreparedSessionKey  = qwen3_5::PreparedSessionKey;
using CheckpointHandle    = qwen3_5::CheckpointHandle;
using CheckpointMetadata  = qwen3_5::CheckpointMetadata;
using CheckpointSummary   = qwen3_5::CheckpointSummary;
using CheckpointImage     = qwen3_5::CheckpointImage;
using ImportedCheckpoints = qwen3_5::ImportedCheckpoints;
using SourceCandidate     = qwen3_5::SourceCandidate;
using BindingReservation  = qwen3_5::BindingReservation;
using CapturePreparation  = qwen3_5::CapturePreparation;
using ContextOperationKind = qwen3_5::ContextOperationKind;
using ExecutionUnitKind   = qwen3_5::ExecutionUnitKind;

class Program;

class RequestBasePlan {
public:
    RequestBasePlan(RequestBasePlan&&) noexcept;
    RequestBasePlan& operator=(RequestBasePlan&&) noexcept;
    ~RequestBasePlan();
    RequestBasePlan(const RequestBasePlan&)            = delete;
    RequestBasePlan& operator=(const RequestBasePlan&) = delete;

    [[nodiscard]] const runtime::RequestPlanSummary& summary() const noexcept;
    [[nodiscard]] const qwen3_5::PreparedContextCache& context_cache() const noexcept;
    // No context cache: no capture frontiers, no shared prefixes.
    [[nodiscard]] std::vector<std::uint32_t> capture_frontiers() const { return {}; }
    [[nodiscard]] std::vector<std::uint32_t> shared_prefix_frontiers() const { return {}; }
    [[nodiscard]] std::optional<qwen3_5::PrefixShortlistKey> prefix_shortlist_key(std::uint32_t) const noexcept {
        return std::nullopt;
    }

    struct Impl;

private:
    explicit RequestBasePlan(std::shared_ptr<Impl> impl) noexcept;
    std::shared_ptr<Impl> impl_;
    friend class Program;
};

class SequenceHandle {
public:
    SequenceHandle() noexcept                                       = default;
    SequenceHandle(const SequenceHandle&) noexcept                  = default;
    SequenceHandle& operator=(const SequenceHandle&) noexcept       = default;
    friend bool operator==(SequenceHandle, SequenceHandle) noexcept = default;

private:
    const void* owner_ = nullptr;
    std::uint64_t epoch_ = 0;
    friend class Program;
};

struct ContextDemotion {
    std::vector<CheckpointHandle> sources;
    std::size_t host_bytes     = 0;
    std::size_t host_placeable = 0;
    runtime::ContextResourceUsage released;
};

struct ContextRelease {
    std::vector<CheckpointHandle> sources;
    runtime::ContextResourceUsage released;
};

class ContextDemotionBatch {
public:
    [[nodiscard]] bool append(const ContextDemotion&) { return false; }
    [[nodiscard]] bool covers(const ContextRelease&) const { return false; }
    [[nodiscard]] std::optional<ContextDemotion> finish() const { return std::nullopt; }
};

struct ContextReclaimPlan {
    std::vector<ContextDemotion> demotions;
    std::vector<ContextRelease> releases;

    [[nodiscard]] ContextDemotionBatch begin_kv_batch(runtime::ContextResourceUsage) const { return {}; }
    [[nodiscard]] std::uint64_t recovery_loss(std::span<const CheckpointHandle>, std::span<const CheckpointHandle>) const {
        return 0;
    }
};

// A paused request. This Program never pauses; the type exists for the contract.
class ResumeState {
public:
    ResumeState() noexcept = default;
    [[nodiscard]] bool has_snapshot() const noexcept { return false; }
    [[nodiscard]] std::optional<CheckpointHandle> snapshot_handle() const noexcept { return std::nullopt; }
    [[nodiscard]] std::uint32_t frontier() const noexcept { return 0; }
};

struct ExecutionUnit {
    SequenceHandle sequence;
    ExecutionUnitKind kind = ExecutionUnitKind::Decode;
    // Decode: remaining output budget. Control: exact forced span. Other kinds: zero.
    std::uint32_t tokens = 0;
};

// The token one model unit sampled, awaiting the Engine's commit decision.
class PendingBatch {
public:
    PendingBatch() noexcept = default;
    PendingBatch(PendingBatch&& other) noexcept
        : owner_(std::exchange(other.owner_, nullptr)), transaction_(std::exchange(other.transaction_, 0)),
          token_(other.token_), timing_(std::exchange(other.timing_, {})),
          constraint_failed_(other.constraint_failed_) {}
    PendingBatch& operator=(PendingBatch&&)      = delete;
    PendingBatch(const PendingBatch&)            = delete;
    PendingBatch& operator=(const PendingBatch&) = delete;

    [[nodiscard]] std::size_t row_count() const noexcept { return owner_ != nullptr ? 1 : 0; }
    [[nodiscard]] std::span<const TokenId> tokens() const noexcept { return {&token_, owner_ != nullptr ? std::size_t{1} : 0}; }
    // No speculation: one token per row.
    [[nodiscard]] std::span<const std::int32_t> row_counts() const noexcept { return {}; }
    [[nodiscard]] std::uint32_t row_stride() const noexcept { return 1; }
    [[nodiscard]] bool constraint_failed(std::size_t) const noexcept { return constraint_failed_; }
    [[nodiscard]] bool token_domain_failed(std::size_t) const noexcept { return false; }
    [[nodiscard]] runtime::ExecutionTiming execution_timing() const noexcept { return timing_; }

private:
    const void* owner_         = nullptr;
    std::uint64_t transaction_ = 0;
    TokenId token_             = 0;
    runtime::ExecutionTiming timing_;
    bool constraint_failed_ = false;
    friend class Program;
};

struct PrefillProgress {
    runtime::BeginSummary summary;
    std::uint32_t processed_prompt_tokens = 0;
    bool complete                         = false;
    bool partial                          = false;
    bool cap_applied                      = false;
    runtime::ExecutionTiming timing;
    std::optional<PendingBatch> pending;
    bool capture_ready = false;
    std::vector<runtime::ContextTransferObservation> landed_transfers;
};

struct CommitRowResult {
    runtime::CommitDisposition disposition = runtime::CommitDisposition::Active;
    GenerationTimings timings;
    SpeculativeStats speculative;

    struct SpeculativeCounters {
        std::uint64_t rounds          = 0;
        std::uint64_t drafted_tokens  = 0;
        std::uint64_t accepted_tokens = 0;
        std::uint64_t fallback_steps  = 0;
        std::uint64_t ngram_rounds          = 0;
        std::uint64_t ngram_drafted_tokens  = 0;
        std::uint64_t ngram_accepted_tokens = 0;
        std::uint64_t relaxed_kept          = 0;
    } speculative_counters;
};

struct CommitResult {
    std::array<CommitRowResult, kMaximumConcurrency> rows{};
    std::array<bool, kMaximumConcurrency> capture_ready{};
    std::size_t row_count = 0;
    runtime::ExecutionTiming timing;
};

struct DiscardResult {
    runtime::ConsumeStatus status = runtime::ConsumeStatus::InvariantMismatch;
    std::size_t row_count         = 0;
};

struct FinishResult {
    runtime::ConsumeStatus status = runtime::ConsumeStatus::InvariantMismatch;
    GenerationTimings timings;
    SpeculativeStats speculative;
    std::optional<CheckpointHandle> checkpoint;
};

struct AbortResult {
    runtime::ConsumeStatus status = runtime::ConsumeStatus::InvariantMismatch;
    GenerationTimings timings;
    SpeculativeStats speculative;
};

struct ReplayProgress {
    bool capture_ready             = false;
    std::uint32_t processed_tokens = 0;
    bool complete                  = false;
    runtime::ExecutionTiming timing;
};

struct ContextProgress {
    ContextOperationKind kind = ContextOperationKind::Bind;
    bool advanced             = false;
    bool complete             = false;
    bool published            = false;
    std::optional<SequenceHandle> sequence;
    std::optional<ResumeState> paused;
    bool replaying = false;
    std::vector<CheckpointHandle> private_points;
    std::vector<CheckpointHandle> retired_checkpoints;
    std::vector<CheckpointHandle> captured_checkpoints;
    std::vector<runtime::ContextTransferObservation> transfers;
    runtime::ContextOperationCounts operations;
    std::optional<GenerationTimings> request_timings;
    SpeculativeStats request_speculative;
};

struct PhysicalUsageSnapshot {
    runtime::ContextResourceUsage occupied;
    runtime::ContextResourceUsage capacity;
    std::size_t host_reserved_bytes      = 0;
    std::size_t host_peak_occupied_bytes = 0;
    std::uint32_t host_state_slots       = 0;
    std::size_t host_kv_bytes            = 0;
    std::size_t host_free_extents        = 0;
    std::uint64_t host_geometry_failures = 0;
    std::uint64_t host_shortage_failures = 0;
    std::size_t host_pause_reserve_bytes          = 0;
    std::size_t host_pause_reserve_occupied_bytes = 0;
};

struct ProgramOptions {
    std::uint32_t max_context   = 0;
    // The most prompt tokens one forward call takes (Forward's max_chunk).
    std::uint32_t prefill_chunk = 512;
    // The public vocabulary: sampled tokens and grammar masks cover [0, token_domain).
    std::uint32_t token_domain = 0;
};

class Program {
public:
    // `forward` must outlive the Program. `device` supplies the stream the forward runs on.
    Program(Forward& forward, DeviceContext& device, ProgramOptions options);
    ~Program() noexcept;
    Program(const Program&)            = delete;
    Program& operator=(const Program&) = delete;

    [[nodiscard]] RequestBasePlan plan_request(PreparedPrompt&& prompt, const runtime::ResolvedExecutionOptions& options);
    [[nodiscard]] CheckpointHandle install_external_checkpoint(const qwen3_5::PromptGraft&);
    [[nodiscard]] std::optional<SourceCandidate>
    inspect_source(const RequestBasePlan& base, std::optional<CheckpointHandle> checkpoint, bool consume_source = false,
                   std::span<const CheckpointHandle> private_points = {},
                   std::span<const CheckpointHandle> retired_points = {}) const;
    [[nodiscard]] qwen3_5::PrefixShortlistKey checkpoint_key(CheckpointHandle, std::uint32_t) const;
    [[nodiscard]] runtime::ContextResourceUsage checkpoint_footprint(std::span<const CheckpointHandle>) const { return {}; }
    [[nodiscard]] CheckpointSummary checkpoint_summary(CheckpointHandle) const;
    [[nodiscard]] CheckpointMetadata checkpoint_metadata(CheckpointHandle) const;
    [[nodiscard]] bool checkpoint_matches(CheckpointHandle, const RequestBasePlan&) const { return false; }
    [[nodiscard]] runtime::PrefillWork prefill_work(const RequestBasePlan& base, std::uint32_t begin, std::uint32_t end) const;
    [[nodiscard]] std::uint32_t checkpoint_recovery_frontier(CheckpointHandle, const RequestBasePlan&, std::uint32_t) const {
        return 0;
    }
    [[nodiscard]] std::uint64_t checkpoint_recovery_loss(std::span<const CheckpointHandle>, std::span<const CheckpointHandle>) const {
        return 0;
    }
    [[nodiscard]] bool valid_checkpoint(CheckpointHandle) const noexcept { return false; }
    [[nodiscard]] bool release_checkpoint(CheckpointHandle) noexcept { return false; }
    [[nodiscard]] bool revoke_snapshot(ResumeState&) noexcept { return false; }
    [[nodiscard]] runtime::ContextResourceUsage snapshot_resources(const ResumeState&) const { return {}; }
    [[nodiscard]] std::size_t host_bytes_released(std::span<const CheckpointHandle>) const { return 0; }
    [[nodiscard]] std::optional<std::size_t> pause_host_bytes(SequenceHandle) const { return std::nullopt; }
    [[nodiscard]] std::size_t pause_state_image_unplaced(SequenceHandle) const { return 0; }
    [[nodiscard]] bool pause_snapshot_fits(SequenceHandle) const { return false; }
    [[nodiscard]] std::size_t release_redundant_host(std::span<const CheckpointHandle>, std::optional<SequenceHandle> = std::nullopt) {
        return 0;
    }
    [[nodiscard]] runtime::ResourceReservation reserve_units(std::span<const ExecutionUnit> units);
    [[nodiscard]] bool reclaim_capture_reservation(runtime::ContextResourceUsage) { return false; }
    void release_units(std::span<const SequenceHandle>) noexcept {}
    [[nodiscard]] BindingReservation start_binding(const RequestBasePlan& base, runtime::LaneId lane, const SourceCandidate& source,
                                                   ResumeState* resume = nullptr,
                                                   ExecutionUnitKind resume_kind = ExecutionUnitKind::Decode,
                                                   std::uint32_t resume_tokens   = 1);
    [[nodiscard]] bool prompt_fits_beside_residents(const RequestBasePlan&, const SourceCandidate&) const { return true; }
    [[nodiscard]] bool start_capture(SequenceHandle) { return false; }
    [[nodiscard]] bool capture_is_input(SequenceHandle) const { return false; }
    [[nodiscard]] std::optional<CapturePreparation> prepare_capture(SequenceHandle) { return std::nullopt; }
    void skip_capture(SequenceHandle) {}
    [[nodiscard]] ContextReclaimPlan plan_reclaim(std::span<const CheckpointHandle>, std::span<const CheckpointHandle>,
                                                  runtime::ContextResourceUsage) const {
        return {};
    }
    [[nodiscard]] std::vector<ContextRelease> plan_releases(std::span<const CheckpointHandle>, std::span<const CheckpointHandle>,
                                                            runtime::ContextResourceUsage) const {
        return {};
    }
    [[nodiscard]] bool start_demote(const ContextDemotion&) { return false; }
    [[nodiscard]] bool start_pause(SequenceHandle, bool, runtime::ExecutionTiming* = nullptr, bool = false) { return false; }
    [[nodiscard]] ContextProgress poll_context(runtime::CancellationFlagView cancellation);
    [[nodiscard]] bool has_context_transaction() const noexcept { return binding_; }
    [[nodiscard]] bool context_blocks(SequenceHandle) const noexcept { return false; }
    [[nodiscard]] bool recovery_pending(SequenceHandle) const noexcept { return false; }
    [[nodiscard]] runtime::ResourceReservation reserve_vision_window(SequenceHandle) { return {.reserved = true}; }
    [[nodiscard]] bool vision_pending(SequenceHandle) const noexcept { return false; }
    bool poll_vision() { return false; }
    bool drain_vision_window() { return false; }
    [[nodiscard]] PrefillProgress advance_prefill(SequenceHandle sequence, runtime::ExecutionTiming* failed_timing = nullptr,
                                                  runtime::TokenMaskProvider* masks = nullptr, std::uint32_t max_tokens = 0,
                                                  std::uint32_t layer_slices = 0);
    [[nodiscard]] ReplayProgress advance_replay(SequenceHandle, runtime::ExecutionTiming* = nullptr);
    [[nodiscard]] PendingBatch decode(std::span<const SequenceHandle> sequences, std::span<const runtime::RoundBudget> budgets,
                                      runtime::ExecutionTiming* failed_timing = nullptr,
                                      runtime::TokenMaskProvider* masks       = nullptr);
    [[nodiscard]] runtime::ExecutionTiming
    append_forced_tokens(std::span<const SequenceHandle> sequences, std::span<const TokenId> row_major_tokens,
                         std::uint32_t row_stride, std::span<const std::optional<std::uint32_t>> prefix_execution_splits,
                         runtime::ExecutionTiming* failed_timing = nullptr);
    [[nodiscard]] CommitResult commit(PendingBatch&& pending, std::span<const runtime::CommitDecision> decisions,
                                      runtime::CommitObservation observation  = runtime::CommitObservation::AllRows,
                                      runtime::ExecutionTiming* failed_timing = nullptr);
    [[nodiscard]] DiscardResult abort_pending(PendingBatch&& pending) noexcept;
    [[nodiscard]] FinishResult finish(SequenceHandle sequence) noexcept;
    [[nodiscard]] AbortResult abort(SequenceHandle sequence) noexcept;
    void fail_all_cleanup() noexcept;
    [[nodiscard]] PhysicalUsageSnapshot physical_usage() const noexcept;
    [[nodiscard]] MemorySummary memory_summary() const noexcept;
    void reset_memory_peaks() noexcept {}
    [[nodiscard]] CheckpointImage export_checkpoints(std::span<const CheckpointHandle>, const std::optional<PreparedSessionKey>&,
                                                     std::string_view, std::vector<std::uint8_t> = {}) const;
    [[nodiscard]] std::size_t checkpoint_image_host_bytes(std::span<const std::uint8_t>, std::string_view) const;
    [[nodiscard]] std::optional<ImportedCheckpoints> import_checkpoints(std::span<const std::uint8_t>, std::string_view);

private:
    using Clock = std::chrono::steady_clock;
    enum class Phase : std::uint8_t { Free, Binding, Prefill, Pending, Decode, Finishable };

    void require_sequence(SequenceHandle sequence) const;
    [[nodiscard]] SequenceHandle handle() const noexcept;
    [[nodiscard]] bool valid(SequenceHandle sequence) const noexcept;
    void release_lane() noexcept;
    // Runs history[from, to) through the forward from position `from` and waits for it; the last
    // position's logits are then in `logits_`.
    void run_forward(std::size_t from, std::size_t to);
    // Downloads the logits and picks the next token under the lane's sampling and the grammar mask.
    TokenId sample_next(runtime::TokenMaskProvider* masks, bool& constraint_failed);
    [[nodiscard]] GenerationTimings timings() const;

    Forward& forward_;
    DeviceContext& device_;
    ProgramOptions options_;

    // The single lane.
    struct Lane {
        Phase phase = Phase::Free;
        std::shared_ptr<RequestBasePlan::Impl> base;
        std::vector<TokenId> history; // the prompt, then every committed token
        std::uint32_t state_tokens  = 0; // history tokens the forward has consumed
        std::uint32_t prompt_tokens = 0;
        std::uint32_t extent        = 0;
        TokenCounts generated;
        std::mt19937_64 rng;
        runtime::BeginSummary begin;
        Clock::time_point admitted;
        double first_token_seconds = 0.0;
        double prefill_seconds     = 0.0;
        double decode_seconds      = 0.0;
    } lane_;
    std::uint64_t epoch_       = 0;
    std::uint64_t transaction_ = 0;
    std::uint64_t pending_     = 0; // the open pending batch's transaction, or 0
    TokenId pending_token_     = 0;
    bool binding_              = false;
    const Tensor* logits_ = nullptr;
    std::vector<float> host_logits_;
    std::vector<std::uint32_t> mask_words_;
};

// The common EngineCore is instantiated once for this model.
struct RuntimeTypes {
    using Frontend          = dsv4::Frontend;
    using PreparedPrompt    = dsv4::PreparedPrompt;
    using OutputSession     = dsv4::OutputSession;
    using PublishedOutput   = dsv4::PublishedOutput;
    using RequestBasePlan   = dsv4::RequestBasePlan;
    using SequenceHandle    = dsv4::SequenceHandle;
    using CheckpointHandle  = dsv4::CheckpointHandle;
    using CheckpointSummary = dsv4::CheckpointSummary;
    using SourceCandidate   = dsv4::SourceCandidate;
    using ResumeState       = dsv4::ResumeState;
    using ExecutionUnit     = dsv4::ExecutionUnit;
    using ExecutionUnitKind = dsv4::ExecutionUnitKind;
    using ContextProgress   = dsv4::ContextProgress;
    using PendingBatch      = dsv4::PendingBatch;
    using PrefillProgress   = dsv4::PrefillProgress;
    using ReplayProgress    = dsv4::ReplayProgress;
    using CommitResult      = dsv4::CommitResult;
    using DiscardResult     = dsv4::DiscardResult;
    using FinishResult      = dsv4::FinishResult;
    using AbortResult       = dsv4::AbortResult;
    using Program           = dsv4::Program;
    using CacheSessionKey   = dsv4::PreparedSessionKey;
    using CheckpointImage   = dsv4::CheckpointImage;
};

} // namespace ninfer::models::dsv4
