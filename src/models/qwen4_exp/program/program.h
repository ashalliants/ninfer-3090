#pragma once

// Adapted from Infernix a3edb450 src/models/qwen4_exp/program/program.h (Apache-2.0).
// Modified for NInfer-3090: implements NInfer's Engine execution contract (the one EngineCore,
// Scheduler and ResourceManager drive for Qwen3.5) instead of Infernix's hybrid-cache contract:
// prefix reuse through NInfer's context cache with checkpoints held only in a pinned Host tier
// (docs/maintainer/flash-next-context-cache.md), one lane, MTP speculation (Infernix's drafter,
// verification and commit), image input (Infernix's vision program: the tower encodes in frames
// lent by the expert cache at the prompt's first prefill unit, the handoff stays lent until the
// prompt's last call), no n-gram copy drafts, CPU-served misses or SSD tier.
//
// The Qwen4Exp Program: the Engine-facing execution of Qwen3.8-Flash-Next. It owns the lane, its
// paged INT8 KV (an elastic VMM pool), recurrent state, sampling state and the one pending model
// unit, the VRAM expert cache (LFRU frames, warm-start state) and the decode CUDA graph. Prompts and
// outputs use the Qwen3.5 frontend (the same tokenizer and chat template).

#include "models/qwen3_5/frontend/frontend.h"
#include "models/qwen3_5/frontend/output_session.h"
#include "models/qwen3_5/frontend/prepared_prompt.h"
// The passive data types of the Engine's execution contract (checkpoint handles, source
// candidates, images) are declared with the Qwen3.5 Program; this Program uses them unchanged.
#include "models/qwen3_5/frontend/graft.h"
#include "models/qwen3_5/program/program.h"
#include "models/qwen4_exp/config.h"
#include "models/qwen4_exp/memory_plan.h"
#include "ninfer/types.h"
#include "runtime/contract/execution.h"
#include "runtime/contract/request.h"
#include "runtime/contract/resources.h"
#include "runtime/contract/timing.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace ninfer {
struct DeviceContext;
}

namespace ninfer::models::qwen4_exp {

namespace execution {
class Parameters;
}

namespace detail {
class ProgramImpl;
struct BasePlanImpl;
struct ResumeStateImpl;
struct ContractAccess;
} // namespace detail

using PreparedPrompt     = qwen3_5::PreparedPrompt;
using Frontend           = qwen3_5::Frontend;
using OutputSession      = qwen3_5::OutputSession;
using PublishedOutput    = qwen3_5::PublishedOutput;
using PreparedSessionKey = qwen3_5::PreparedSessionKey;

class RequestBasePlan {
public:
    RequestBasePlan(RequestBasePlan&&) noexcept;
    RequestBasePlan& operator=(RequestBasePlan&&) noexcept;
    ~RequestBasePlan();
    RequestBasePlan(const RequestBasePlan&)            = delete;
    RequestBasePlan& operator=(const RequestBasePlan&) = delete;

    [[nodiscard]] const runtime::RequestPlanSummary& summary() const noexcept;
    [[nodiscard]] const qwen3_5::PreparedContextCache& context_cache() const noexcept;
    // The prompt frontiers this request captures (ascending), those that publish a SharedPrefix
    // checkpoint, and the shortlist key of any prompt frontier. Empty and nullopt when the request
    // does not use the context cache.
    [[nodiscard]] std::vector<std::uint32_t> capture_frontiers() const;
    [[nodiscard]] std::vector<std::uint32_t> shared_prefix_frontiers() const;
    [[nodiscard]] std::optional<qwen3_5::PrefixShortlistKey> prefix_shortlist_key(std::uint32_t frontier) const noexcept;

private:
    explicit RequestBasePlan(std::shared_ptr<detail::BasePlanImpl> impl) noexcept;
    std::shared_ptr<detail::BasePlanImpl> impl_;
    friend class detail::ProgramImpl;
};

class SequenceHandle {
public:
    SequenceHandle() noexcept                                       = default;
    SequenceHandle(const SequenceHandle&) noexcept                  = default;
    SequenceHandle& operator=(const SequenceHandle&) noexcept       = default;
    friend bool operator==(SequenceHandle, SequenceHandle) noexcept = default;

private:
    const void* owner_ = nullptr;
    runtime::LaneId lane_{};
    std::uint64_t epoch_ = 0;

    friend struct detail::ContractAccess;
};

// The context cache's types (the Qwen3.5 Program's passive data). Every checkpoint lives in the
// Host tier only, so reclaim releases Host bytes and never demotes.
using CheckpointHandle   = qwen3_5::CheckpointHandle;
using CheckpointMetadata = qwen3_5::CheckpointMetadata;
using CheckpointSummary  = qwen3_5::CheckpointSummary;
using CheckpointImage    = qwen3_5::CheckpointImage;
using ImportedCheckpoints = qwen3_5::ImportedCheckpoints;
using SourceCandidate    = qwen3_5::SourceCandidate;
using BindingReservation = qwen3_5::BindingReservation;
using CapturePreparation = qwen3_5::CapturePreparation;
using ContextOperationKind = qwen3_5::ContextOperationKind;
using ExecutionUnitKind  = qwen3_5::ExecutionUnitKind;

struct ContextDemotion {
    std::vector<CheckpointHandle> sources;
    std::size_t host_bytes = 0;
    runtime::ContextResourceUsage released;
};

struct ContextRelease {
    std::vector<CheckpointHandle> sources;
    runtime::ContextResourceUsage released;
};

// Nothing is ever demoted (no checkpoint is on the device): a batch has nothing to append, cover
// or finish.
class ContextDemotionBatch {
public:
    [[nodiscard]] bool append(const ContextDemotion&) { return false; }
    [[nodiscard]] bool covers(const ContextRelease&) const { return false; }
    [[nodiscard]] std::optional<ContextDemotion> finish() const { return std::nullopt; }
};

// Read-only facts for one synchronous evaluation: Host release actions only.
struct ContextReclaimPlan {
    std::vector<ContextDemotion> demotions;
    std::vector<ContextRelease> releases;

    [[nodiscard]] ContextDemotionBatch begin_kv_batch(runtime::ContextResourceUsage) const { return {}; }
    [[nodiscard]] std::uint64_t recovery_loss(std::span<const CheckpointHandle> removed,
                                              std::span<const CheckpointHandle> surviving) const;

private:
    const detail::ProgramImpl* owner_ = nullptr;
    friend class detail::ProgramImpl;
};

// A paused request. This Program does not pause (it has one lane and reserves a request's whole
// extent at admission), so a ResumeState only exists to satisfy the contract's types.
class ResumeState {
public:
    ResumeState(ResumeState&&) noexcept;
    ResumeState& operator=(ResumeState&&) noexcept;
    ~ResumeState();
    ResumeState(const ResumeState&)            = delete;
    ResumeState& operator=(const ResumeState&) = delete;
    [[nodiscard]] bool has_snapshot() const noexcept { return false; }
    [[nodiscard]] std::optional<CheckpointHandle> snapshot_handle() const noexcept { return std::nullopt; }
    [[nodiscard]] std::uint32_t frontier() const noexcept;

private:
    explicit ResumeState(std::unique_ptr<detail::ResumeStateImpl>) noexcept;
    std::unique_ptr<detail::ResumeStateImpl> impl_;
    friend class detail::ProgramImpl;
};

struct ExecutionUnit {
    SequenceHandle sequence;
    ExecutionUnitKind kind = ExecutionUnitKind::Decode;
    // Decode: remaining output budget. Control: exact forced span. Other kinds: zero.
    std::uint32_t tokens = 0;
};

// The sampled tokens of one model unit, awaiting the Engine's commit decisions. Its token span
// points into Program storage that stays valid until commit or abort_pending.
class PendingBatch {
public:
    PendingBatch() noexcept = default;
    ~PendingBatch()         = default;
    PendingBatch(PendingBatch&& other) noexcept
        : owner_(std::exchange(other.owner_, nullptr)), transaction_(std::exchange(other.transaction_, 0)),
          rows_(other.rows_), row_count_(std::exchange(other.row_count_, 0)),
          tokens_(std::exchange(other.tokens_, {})), timing_(std::exchange(other.timing_, {})),
          constraint_failed_(other.constraint_failed_), counts_(other.counts_),
          stride_(std::exchange(other.stride_, 1U)), speculative_(std::exchange(other.speculative_, false)) {}
    PendingBatch& operator=(PendingBatch&&)      = delete;
    PendingBatch(const PendingBatch&)            = delete;
    PendingBatch& operator=(const PendingBatch&) = delete;

    [[nodiscard]] std::size_t row_count() const noexcept { return row_count_; }
    [[nodiscard]] std::span<const TokenId> tokens() const noexcept { return tokens_; }
    // A verification round licenses each row's accepted drafts and its correction or bonus token
    // (tokens row_stride apart); any other round one token per row.
    [[nodiscard]] std::span<const std::int32_t> row_counts() const noexcept {
        return speculative_ ? std::span<const std::int32_t>(counts_.data(), row_count_) : std::span<const std::int32_t>{};
    }
    [[nodiscard]] std::uint32_t row_stride() const noexcept { return stride_; }
    [[nodiscard]] bool constraint_failed(std::size_t row) const { return constraint_failed_.at(row); }
    [[nodiscard]] runtime::ExecutionTiming execution_timing() const noexcept { return timing_; }

private:
    const void* owner_         = nullptr;
    std::uint64_t transaction_ = 0;
    std::array<SequenceHandle, kMaximumConcurrency> rows_{};
    std::size_t row_count_ = 0;
    std::span<const TokenId> tokens_;
    runtime::ExecutionTiming timing_;
    std::array<bool, kMaximumConcurrency> constraint_failed_{};
    std::array<std::int32_t, kMaximumConcurrency> counts_{};
    std::uint32_t stride_ = 1;
    bool speculative_     = false;

    friend struct detail::ContractAccess;
};

struct PrefillProgress {
    runtime::BeginSummary summary;
    std::uint32_t processed_prompt_tokens = 0;
    bool complete                         = false;
    runtime::ExecutionTiming timing;
    std::optional<PendingBatch> pending;
    bool capture_ready = false;
    // The Engine records Main KV restored under the first prefill pass. This Program's restore
    // lands within the binding (reported by poll_context), so the list stays empty.
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
        // The Engine's n-gram copy counters; Flash-Next drafts no n-gram copies, so they stay 0.
        std::uint64_t ngram_rounds          = 0;
        std::uint64_t ngram_drafted_tokens  = 0;
        std::uint64_t ngram_accepted_tokens = 0;
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
    // The Flash-Next cache is a slab pool with no extents to fragment; these stay zero.
    std::size_t host_free_extents          = 0;
    std::uint64_t host_geometry_failures   = 0;
    std::uint64_t host_shortage_failures   = 0;
};

// Device memory a Program allocates besides its expert frames, from the options and the model's
// configuration alone, so startup checks it before the weights are read. The constructor
// allocates exactly these sizes.
struct ProgramDevicePlan {
    std::uint32_t kv_pages = 0;
    std::uint64_t kv        = 0; // KV pages and block table backed at startup (the elastic pool's base)
    std::uint64_t kv_max    = 0; // the KV at max_context; above `kv`, the elastic pool takes it from the frames
    std::uint64_t workspace = 0; // the forward and sampling workspace arena
    std::uint64_t staging   = 0; // the expert miss staging slots
    std::uint64_t state     = 0; // GDN, PLE and QSA tail state
    std::uint64_t io        = 0; // per-call inputs, logits and sampling arrays
    std::uint64_t residency = 0; // the expert frame tables and route log
    // MTP speculation: verification records, tail copies, the drafter's residuals and round words.
    std::uint64_t speculation = 0;
    std::uint64_t expert_record_stride = 0; // one expert frame
    std::uint32_t max_frames           = 0; // the most frames the expert cache uses
    std::uint32_t graph_bound          = 0; // CUDA graph executables the Program may instantiate (counts both halves of the split decode)

    [[nodiscard]] std::uint64_t fixed_bytes() const noexcept {
        return kv + workspace + staging + state + io + residency + speculation;
    }
};

struct ProgramOptions {
    int device                    = 0;
    std::uint32_t max_context     = 0;
    std::uint32_t max_concurrency = 1;
    std::uint32_t prefill_chunk   = 1024;
    bool use_cuda_graph           = true;
    std::filesystem::path ngram_volume;
    // The expert cache's saved state: read at startup to fill the frames (warm start), written at
    // shutdown and, between requests, every 10 minutes. Empty: no warm start. The identity names
    // the artifact the state belongs to.
    std::filesystem::path expert_state;
    std::string expert_state_identity;
    // Device memory left free for the display and other programs once the fixed allocations and
    // the expert frames are made; empty selects it from the display state.
    std::optional<std::uint64_t> vram_headroom;
    // At most this many bytes of expert frames (measurement: a small cache that must evict).
    std::optional<std::uint64_t> expert_cache_bytes;
    bool expert_cache = true;
    // The runtime VRAM monitor that shrinks and grows the expert cache at round boundaries as other
    // programs take and release device memory, and how long free memory must stay above the
    // cache's target before it grows.
    bool vram_monitor              = true;
    double vram_grow_delay_seconds = 30.0;
    // Elastic KV: the tokens of KV backed at startup (0: Infernix's 32,768). Lower values grow the
    // pool from the expert frames at admission and return it at the request's end.
    std::uint32_t kv_base_tokens = 0;
    // The host-to-device link rate in bytes per second; 0 measures it at startup.
    double link_bytes_per_second = 0.0;
    // MTP speculative decoding: the most drafts one round verifies (1..7; 0 disables). The model
    // must have loaded its MTP drafter. Each round drafts the count its acceptance estimate
    // favours, up to this bound (Infernix's draft-length policy).
    std::uint32_t mtp_draft_tokens = 0;
    // The draft-length policy's cost of one more verified column, relative to a plain round
    // (Infernix measured 0.38 on the RTX 5090), and a measurement mode that always drafts the most.
    double mtp_width_cost = 0.38;
    bool mtp_fixed_drafts = false;
    // Image input: the model loaded its vision tower. Prompts with media encode it into frames lent
    // by the expert cache (its io words are planned only when set).
    bool vision = false;
    // The context cache's pinned Host tier (checkpoints for prefix reuse), sized by the startup RAM
    // ledger; 0: no prefix reuse.
    std::uint64_t context_cache_bytes = 0;
};

class Program {
public:
    Program(const execution::Parameters& parameters, DeviceContext& device, ProgramOptions options);
    ~Program() noexcept;
    Program(const Program&)            = delete;
    Program& operator=(const Program&) = delete;
    Program(Program&&)                 = delete;
    Program& operator=(Program&&)      = delete;

    // The device plan of a Program with these options over a model with this configuration;
    // validates the options.
    [[nodiscard]] static ProgramDevicePlan plan_device(const ProgramOptions& options, const TextConfig& config,
                                                       std::uint32_t public_tokens,
                                                       std::span<const std::uint64_t> record_strides);
    [[nodiscard]] const ProgramDevicePlan& device_plan() const noexcept;
    // How the expert frames were sized at construction.
    [[nodiscard]] const VramSizing& vram_sizing() const noexcept;
    // Expert-cache statistics since startup: routed (layer, expert) uses, hits, promotions, frames.
    struct ExpertCacheCounters {
        std::uint64_t routed = 0, hits = 0, promotions = 0, seeded = 0;
        std::uint32_t frames = 0;
    };
    [[nodiscard]] ExpertCacheCounters expert_cache_counters() const noexcept;

    [[nodiscard]] RequestBasePlan plan_request(PreparedPrompt&& prompt,
                                               const runtime::ResolvedExecutionOptions& options);
    [[nodiscard]] CheckpointHandle install_external_checkpoint(const qwen3_5::PromptGraft& graft);
    [[nodiscard]] std::optional<SourceCandidate>
    inspect_source(const RequestBasePlan& base, std::optional<CheckpointHandle> checkpoint,
                   bool consume_source                              = false,
                   std::span<const CheckpointHandle> private_points = {},
                   std::span<const CheckpointHandle> retired_points = {}) const;
    [[nodiscard]] qwen3_5::PrefixShortlistKey checkpoint_key(CheckpointHandle, std::uint32_t frontier) const;
    [[nodiscard]] runtime::ContextResourceUsage checkpoint_footprint(std::span<const CheckpointHandle>) const;
    [[nodiscard]] CheckpointSummary checkpoint_summary(CheckpointHandle checkpoint) const;
    [[nodiscard]] CheckpointMetadata checkpoint_metadata(CheckpointHandle checkpoint) const;
    [[nodiscard]] bool checkpoint_matches(CheckpointHandle checkpoint, const RequestBasePlan& base) const;
    // Prefill cost features of prompt positions [begin, end): tokens and the calls they take.
    [[nodiscard]] runtime::PrefillWork prefill_work(const RequestBasePlan& base, std::uint32_t begin,
                                                    std::uint32_t end) const;
    [[nodiscard]] std::uint32_t checkpoint_recovery_frontier(CheckpointHandle retained, const RequestBasePlan& base,
                                                             std::uint32_t target) const;
    [[nodiscard]] std::uint64_t checkpoint_recovery_loss(std::span<const CheckpointHandle> removed,
                                                         std::span<const CheckpointHandle> surviving) const;
    [[nodiscard]] bool valid_checkpoint(CheckpointHandle checkpoint) const noexcept;
    [[nodiscard]] bool release_checkpoint(CheckpointHandle checkpoint) noexcept;
    // No pause, so no snapshot.
    [[nodiscard]] bool revoke_snapshot(ResumeState&) noexcept { return false; }
    [[nodiscard]] runtime::ContextResourceUsage snapshot_resources(const ResumeState&) const { return {}; }
    [[nodiscard]] std::size_t host_bytes_released(std::span<const CheckpointHandle> checkpoints) const;
    [[nodiscard]] std::optional<std::size_t> pause_host_bytes(SequenceHandle) const { return std::nullopt; }
    // Drops the lane's references to Host pages no checkpoint holds (a capture then copies them
    // again), except for `pending_backup`'s lane. Returns the bytes freed.
    [[nodiscard]] std::size_t release_redundant_host(std::span<const CheckpointHandle> excluded,
                                                     std::optional<SequenceHandle> pending_backup = std::nullopt);
    [[nodiscard]] runtime::ResourceReservation reserve_units(std::span<const ExecutionUnit> units);
    [[nodiscard]] bool reclaim_capture_reservation(runtime::ContextResourceUsage shortage);
    void release_units(std::span<const SequenceHandle> sequences) noexcept;
    [[nodiscard]] BindingReservation start_binding(const RequestBasePlan& base, runtime::LaneId lane,
                                                   const SourceCandidate& source, ResumeState* resume = nullptr,
                                                   ExecutionUnitKind resume_kind = ExecutionUnitKind::Decode,
                                                   std::uint32_t resume_tokens   = 1);
    [[nodiscard]] bool start_capture(SequenceHandle sequence);
    [[nodiscard]] bool capture_is_input(SequenceHandle sequence) const;
    [[nodiscard]] std::optional<CapturePreparation> prepare_capture(SequenceHandle sequence);
    void skip_capture(SequenceHandle sequence);
    [[nodiscard]] ContextReclaimPlan plan_reclaim(std::span<const CheckpointHandle> allowed,
                                                  std::span<const CheckpointHandle> excluded,
                                                  runtime::ContextResourceUsage shortage) const;
    [[nodiscard]] std::vector<ContextRelease> plan_releases(std::span<const CheckpointHandle> allowed,
                                                            std::span<const CheckpointHandle> excluded,
                                                            runtime::ContextResourceUsage shortage) const;
    [[nodiscard]] bool start_demote(const ContextDemotion&) { return false; }
    // One lane and a whole-extent reservation at admission: there is never a younger resident to
    // pause, so pausing is refused.
    [[nodiscard]] bool start_pause(SequenceHandle, bool, runtime::ExecutionTiming* = nullptr) { return false; }
    [[nodiscard]] ContextProgress poll_context(runtime::CancellationFlagView cancellation);
    [[nodiscard]] bool has_context_transaction() const noexcept;
    [[nodiscard]] bool context_blocks(SequenceHandle) const noexcept { return false; }
    [[nodiscard]] bool recovery_pending(SequenceHandle) const noexcept { return false; }
    // A prompt's images encode inside its first prefill unit (one lane: nothing runs beside it), so
    // the Engine's Vision window calls are no-ops.
    [[nodiscard]] runtime::ResourceReservation reserve_vision_window(SequenceHandle) { return {.reserved = true}; }
    [[nodiscard]] bool vision_pending(SequenceHandle) const noexcept { return false; }
    bool poll_vision() { return false; }
    bool drain_vision_window() { return false; }
    [[nodiscard]] PrefillProgress advance_prefill(SequenceHandle sequence,
                                                  runtime::ExecutionTiming* failed_timing = nullptr,
                                                  runtime::TokenMaskProvider* masks       = nullptr);
    [[nodiscard]] ReplayProgress advance_replay(SequenceHandle sequence,
                                                runtime::ExecutionTiming* failed_timing = nullptr);
    [[nodiscard]] PendingBatch decode(std::span<const SequenceHandle> sequences,
                                      std::span<const runtime::RoundBudget> budgets,
                                      runtime::ExecutionTiming* failed_timing = nullptr,
                                      runtime::TokenMaskProvider* masks       = nullptr);
    [[nodiscard]] runtime::ExecutionTiming
    append_forced_tokens(std::span<const SequenceHandle> sequences, std::span<const TokenId> row_major_tokens,
                         std::uint32_t row_stride,
                         std::span<const std::optional<std::uint32_t>> prefix_execution_splits,
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
    // The context store (checkpoint images on disk) is not built for this model: the Engine refuses
    // --context-store for it, so these are never reached.
    [[nodiscard]] CheckpointImage export_checkpoints(std::span<const CheckpointHandle>,
                                                     const std::optional<PreparedSessionKey>&,
                                                     std::string_view binding) const;
    [[nodiscard]] std::size_t checkpoint_image_host_bytes(std::span<const std::uint8_t>, std::string_view) const;
    [[nodiscard]] std::optional<ImportedCheckpoints> import_checkpoints(std::span<const std::uint8_t>,
                                                                        std::string_view);

private:
    std::unique_ptr<detail::ProgramImpl> impl_;
};

// The common EngineCore is instantiated once for this model.
struct RuntimeTypes {
    using Frontend          = qwen4_exp::Frontend;
    using PreparedPrompt    = qwen4_exp::PreparedPrompt;
    using OutputSession     = qwen4_exp::OutputSession;
    using PublishedOutput   = qwen4_exp::PublishedOutput;
    using RequestBasePlan   = qwen4_exp::RequestBasePlan;
    using SequenceHandle    = qwen4_exp::SequenceHandle;
    using CheckpointHandle  = qwen4_exp::CheckpointHandle;
    using CheckpointSummary = qwen4_exp::CheckpointSummary;
    using SourceCandidate   = qwen4_exp::SourceCandidate;
    using ResumeState       = qwen4_exp::ResumeState;
    using ExecutionUnit     = qwen4_exp::ExecutionUnit;
    using ExecutionUnitKind = qwen4_exp::ExecutionUnitKind;
    using ContextProgress   = qwen4_exp::ContextProgress;
    using PendingBatch      = qwen4_exp::PendingBatch;
    using PrefillProgress   = qwen4_exp::PrefillProgress;
    using ReplayProgress    = qwen4_exp::ReplayProgress;
    using CommitResult      = qwen4_exp::CommitResult;
    using DiscardResult     = qwen4_exp::DiscardResult;
    using FinishResult      = qwen4_exp::FinishResult;
    using AbortResult       = qwen4_exp::AbortResult;
    using Program           = qwen4_exp::Program;
    using CacheSessionKey   = qwen4_exp::PreparedSessionKey;
    using CheckpointImage   = qwen4_exp::CheckpointImage;
};

} // namespace ninfer::models::qwen4_exp
