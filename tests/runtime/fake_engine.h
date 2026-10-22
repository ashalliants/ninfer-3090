#pragma once

// A GPU-free stand-in for the model side of EngineCore, so the Engine's scheduling can be tested on
// a host with no card: which prefill step runs when, how decode rounds interleave, when a prefill
// is paused for a newcomer and when it is restored.
//
// What it is: the contract EngineCore is a template over (Instance::ModelContract), implemented
// by a deterministic simulation. A "prompt" is a token count, a prefill step consumes tokens and
// reports a cost from a fixed cost model, a decode round yields one token per lane, a pause saves
// the cursor and a restore resumes from it. Every call is appended to an event log, and a hook runs
// on the worker thread before each execution unit so a test can hold the Engine at a known point.
//
// What it is not: the real Program. It has no memory, no checkpoints (the context cache is left
// empty), no speculation and no media; its pause always saves a snapshot, restores always
// succeed and a context transaction takes exactly one poll. It checks the Engine's logic and the
// order of its calls, not GPU behaviour.

#include "core/device_snapshot.h"
#include "ninfer/types.h"
#include "runtime/contract/execution.h"
#include "runtime/contract/resources.h"
#include "runtime/engine/engine_core.h"
#include "runtime/engine/kv_capacity.h"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <memory>
#include <set>
#include <string_view>
#include <mutex>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace ninfer::runtime::fake {

struct Device {
    void bind_to_current_thread() const {}
    [[nodiscard]] cudaError_t synchronize_status() const { return cudaSuccess; }
};

// ---- contract types ------------------------------------------------------------------------

// The shortlist key the store code reads: the digests identify the prompt, here by its id alone.
struct Key {
    std::array<std::uint64_t, 2> digests{};
    std::uint32_t frontier     = 0;
    std::uint32_t identity_tag = 0;
    friend bool operator==(const Key&, const Key&) noexcept = default;
};

struct PreparedPrompt {
    std::uint32_t tokens = 0;
    std::uint64_t id     = 0;
};

struct SequenceHandle {
    std::uint32_t lane  = 0;
    std::uint64_t epoch = 0;
    friend bool operator==(SequenceHandle, SequenceHandle) noexcept = default;
};

struct CheckpointHandle {
    std::uint32_t index      = 0;
    std::uint64_t generation = 0;
    friend bool operator==(CheckpointHandle, CheckpointHandle) noexcept = default;
};

struct CheckpointMetadata {
    std::uint32_t frontier = 0;
    CheckpointRole role    = CheckpointRole::Continuation;
    bool leased            = false;
};

struct CheckpointSummary {
    Key key;
    std::uint32_t frontier = 0;
    CheckpointRole role    = CheckpointRole::Continuation;
    bool leased            = false;
    ContextResourceUsage evictable_resources;
};

struct CacheSessionKey {
    friend bool operator==(const CacheSessionKey&, const CacheSessionKey&) noexcept = default;
};

struct CacheAdvice {
    bool update_session_index = false;
    std::optional<CacheSessionKey> session_key;
};

class RequestBasePlan {
public:
    RequestBasePlan() = default;
    RequestBasePlan(std::uint64_t id, std::uint32_t prompt_tokens, std::uint32_t output_tokens) {
        id_                         = id;
        summary_.prompt_tokens      = prompt_tokens;
        summary_.requested_output_tokens = output_tokens;
        summary_.effective_output_tokens = output_tokens;
        summary_.effective_limit_reason  = FinishReason::OutputLimit;
        summary_.publish_continuation    = false;
    }

    [[nodiscard]] const RequestPlanSummary& summary() const noexcept { return summary_; }
    [[nodiscard]] const CacheAdvice& context_cache() const noexcept { return advice_; }
    [[nodiscard]] std::vector<std::uint32_t> capture_frontiers() const { return {}; }
    [[nodiscard]] std::vector<std::uint32_t> shared_prefix_frontiers() const { return {}; }
    [[nodiscard]] std::optional<Key> prefix_shortlist_key(std::uint32_t frontier) const noexcept {
        if (frontier > summary_.prompt_tokens) { return std::nullopt; }
        return Key{{id_, 0}, frontier, 0};
    }
    [[nodiscard]] std::uint64_t id() const noexcept { return id_; }

private:
    std::uint64_t id_ = 0;
    RequestPlanSummary summary_;
    CacheAdvice advice_;
};

struct SourceCandidate {
    std::optional<CheckpointHandle> checkpoint;
    std::uint32_t reused_tokens = 0;
    PrefillWork remaining_work;
    std::vector<ContextTransferRequirement> transfers;
    bool consume_source = false;
    bool take_private   = false;
    bool move_state     = false;
    bool move_history   = false;
    bool split_state    = false;
    bool backup_state   = false;
    std::vector<CheckpointHandle> private_points;
    std::vector<CheckpointHandle> retired_points;
};

struct BindingReservation {
    bool reserved          = false;
    bool source_valid      = true;
    bool capacity_possible = true;
    ContextResourceUsage shortage;
    std::vector<CheckpointHandle> retired_points;
    std::optional<CheckpointHandle> consumed_source;

    explicit operator bool() const noexcept { return reserved; }
};

// What a paused request keeps. The fake always saves the cursor, so a restore never recomputes.
class ResumeState {
public:
    ResumeState() = default;
    ResumeState(std::uint32_t frontier, std::uint32_t generated, std::optional<CheckpointHandle> snapshot)
        : frontier_(frontier), generated_(generated), snapshot_(snapshot) {}
    ResumeState(ResumeState&&) noexcept            = default;
    ResumeState& operator=(ResumeState&&) noexcept = default;

    [[nodiscard]] bool has_snapshot() const noexcept { return snapshot_.has_value(); }
    [[nodiscard]] std::optional<CheckpointHandle> snapshot_handle() const noexcept { return snapshot_; }
    [[nodiscard]] std::uint32_t frontier() const noexcept { return frontier_; }
    [[nodiscard]] std::uint32_t generated() const noexcept { return generated_; }

private:
    std::uint32_t frontier_  = 0;
    std::uint32_t generated_ = 0;
    std::optional<CheckpointHandle> snapshot_;
};

enum class ExecutionUnitKind : std::uint8_t { Prefill, Replay, Decode, Control, Normalize };

struct ExecutionUnit {
    SequenceHandle sequence;
    ExecutionUnitKind kind = ExecutionUnitKind::Decode;
    std::uint32_t tokens   = 0;
};

class PendingBatch {
public:
    PendingBatch() = default;
    PendingBatch(PendingBatch&&) noexcept            = default;
    PendingBatch& operator=(PendingBatch&&) noexcept = default;
    PendingBatch(const PendingBatch&)                = delete;
    PendingBatch& operator=(const PendingBatch&)     = delete;

    [[nodiscard]] std::size_t row_count() const noexcept { return rows_.size(); }
    [[nodiscard]] std::span<const TokenId> tokens() const noexcept { return tokens_; }
    [[nodiscard]] std::span<const std::int32_t> row_counts() const noexcept { return {}; }
    [[nodiscard]] std::uint32_t row_stride() const noexcept { return 1; }
    [[nodiscard]] bool constraint_failed(std::size_t) const { return false; }
    [[nodiscard]] ExecutionTiming execution_timing() const noexcept { return timing_; }

    std::vector<SequenceHandle> rows_;
    std::vector<TokenId> tokens_;
    ExecutionTiming timing_;
};

struct PrefillProgress {
    BeginSummary summary;
    std::uint32_t processed_prompt_tokens = 0;
    bool complete                         = false;
    bool cap_applied                      = false;
    ExecutionTiming timing;
    std::optional<PendingBatch> pending;
    bool capture_ready = false;
    std::vector<ContextTransferObservation> landed_transfers;
};

struct ReplayProgress {
    bool capture_ready             = false;
    std::uint32_t processed_tokens = 0;
    bool complete                  = false;
    ExecutionTiming timing;
};

struct CommitRowResult {
    CommitDisposition disposition = CommitDisposition::Active;
    GenerationTimings timings;
    SpeculativeStats speculative;
    struct SpeculativeCounters {
        std::uint64_t rounds                = 0;
        std::uint64_t drafted_tokens        = 0;
        std::uint64_t accepted_tokens       = 0;
        std::uint64_t fallback_steps        = 0;
        std::uint64_t ngram_rounds          = 0;
        std::uint64_t ngram_drafted_tokens  = 0;
        std::uint64_t ngram_accepted_tokens = 0;
    } speculative_counters;
};

struct CommitResult {
    std::array<CommitRowResult, kMaximumConcurrency> rows{};
    std::array<bool, kMaximumConcurrency> capture_ready{};
    std::size_t row_count = 0;
    ExecutionTiming timing;
};

struct DiscardResult {
    ConsumeStatus status  = ConsumeStatus::InvariantMismatch;
    std::size_t row_count = 0;
};

struct FinishResult {
    ConsumeStatus status = ConsumeStatus::InvariantMismatch;
    GenerationTimings timings;
    SpeculativeStats speculative;
    std::optional<CheckpointHandle> checkpoint;
};

struct AbortResult {
    ConsumeStatus status = ConsumeStatus::InvariantMismatch;
    GenerationTimings timings;
    SpeculativeStats speculative;
};

struct CapturePreparation {
    std::uint32_t frontier = 0;
    bool reserved          = false;
    ContextResourceUsage shortage;
    std::size_t host_bytes = 0;
};

enum class ContextOperationKind : std::uint8_t { Bind, Capture, Demote, Pause };

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
    std::vector<ContextTransferObservation> transfers;
    ContextOperationCounts operations;
    std::optional<GenerationTimings> request_timings;
    SpeculativeStats request_speculative;
};

struct CheckpointImage {
    struct Region {
        std::uint64_t offset = 0;
        std::uint64_t length = 0;
    };
    std::vector<std::uint8_t> bytes;
    std::vector<Region> regions;
    std::vector<Key> keys;
    std::vector<std::array<std::uint64_t, 2>> prefix_digests;
    std::uint32_t tokens = 0;
};

struct ImportedCheckpoints {
    std::vector<CheckpointHandle> points;
    std::optional<CacheSessionKey> session;
};

struct PhysicalUsageSnapshot {
    ContextResourceUsage occupied;
    ContextResourceUsage capacity;
    std::size_t host_reserved_bytes        = 0;
    std::size_t host_peak_occupied_bytes   = 0;
    std::uint32_t host_state_slots         = 0;
    std::size_t host_kv_bytes              = 0;
    std::size_t host_free_extents          = 0;
    std::uint64_t host_geometry_failures   = 0;
    std::uint64_t host_shortage_failures   = 0;
};

// ---- output ----------------------------------------------------------------------------------

class PublishedOutput {
public:
    using iterator       = std::array<OutputDelta, 2>::iterator;
    using const_iterator = std::array<OutputDelta, 2>::const_iterator;

    [[nodiscard]] bool empty() const noexcept { return size_ == 0; }
    [[nodiscard]] std::size_t size() const noexcept { return size_; }
    [[nodiscard]] iterator begin() noexcept { return values_.begin(); }
    [[nodiscard]] const_iterator begin() const noexcept { return values_.begin(); }
    [[nodiscard]] iterator end() noexcept { return values_.begin() + size_; }
    [[nodiscard]] const_iterator end() const noexcept { return values_.begin() + size_; }
    [[nodiscard]] OutputDelta& back() noexcept { return values_[size_ - 1]; }
    [[nodiscard]] const OutputDelta& back() const noexcept { return values_[size_ - 1]; }
    void clear() noexcept { size_ = 0; }
    void push_back(OutputDelta value) { values_[size_++] = std::move(value); }

private:
    std::array<OutputDelta, 2> values_{};
    std::size_t size_ = 0;
};

// Accepts every token the budget allows and finishes when the budget runs out.
class OutputSession {
public:
    [[nodiscard]] OutputDecision preview_model(std::span<const TokenId> tokens,
                                               std::uint32_t total_budget_remaining,
                                               FinishReason limit_reason) {
        const auto accepted = static_cast<std::uint32_t>(
            std::min<std::size_t>(tokens.size(), total_budget_remaining));
        OutputDecision decision;
        decision.accepted_tokens = std::max<std::uint32_t>(accepted, 1U);
        if (total_budget_remaining <= tokens.size()) { decision.finish_reason = limit_reason; }
        preview_tokens_ = decision.accepted_tokens;
        preview_final_  = decision.finish_reason;
        return decision;
    }
    [[nodiscard]] std::uint32_t model_token_budget_remaining(std::uint32_t total) const noexcept {
        return total;
    }
    [[nodiscard]] std::span<const TokenId> pending_control_tokens() const noexcept { return {}; }
    [[nodiscard]] OutputDecision preview_control(std::span<const TokenId>, std::uint32_t) {
        return {};
    }
    [[nodiscard]] OutputDecision preview_terminal(FinishReason reason) {
        OutputDecision decision;
        decision.finish_reason = reason;
        preview_final_         = reason;
        preview_tokens_        = 0;
        return decision;
    }
    [[nodiscard]] PublishedOutput commit_preview() {
        PublishedOutput out;
        total_tokens_ += preview_tokens_;
        if (preview_tokens_ != 0 || preview_final_ != FinishReason::None) {
            OutputDelta delta;
            delta.text = std::string(preview_tokens_, 'x');
            out.push_back(std::move(delta));
        }
        preview_tokens_ = 0;
        preview_final_  = FinishReason::None;
        return out;
    }
    void discard_preview() {
        preview_tokens_ = 0;
        preview_final_  = FinishReason::None;
    }
    [[nodiscard]] bool constrained() const noexcept { return false; }
    void observe_constraint(bool, double) noexcept {}
    void constraint_uploaded(std::size_t) noexcept {}
    [[nodiscard]] std::optional<ConstraintObservation> constraint_observation() const {
        return std::nullopt;
    }
    [[nodiscard]] std::uint32_t grammar_masks(std::span<const TokenId>, std::span<std::uint32_t>) {
        return 0;
    }
    [[nodiscard]] std::vector<GeneratedToolCall> take_tool_calls() noexcept { return {}; }
    [[nodiscard]] ToolCallParseDiagnostics tool_call_parse_diagnostics() const noexcept { return {}; }
    [[nodiscard]] std::uint32_t reasoning_tokens() const noexcept { return 0; }
    [[nodiscard]] ThinkingBudgetStats thinking_stats() const noexcept { return {}; }
    [[nodiscard]] std::optional<std::string> matched_stop_string() const { return std::nullopt; }

private:
    std::uint32_t preview_tokens_ = 0;
    std::uint32_t total_tokens_   = 0;
    FinishReason preview_final_   = FinishReason::None;
};

struct Frontend {
    [[nodiscard]] std::uint32_t thinking_control_token_count() const noexcept { return 0; }
    template <class... Options>
    [[nodiscard]] OutputSession make_output_session(const PreparedPrompt&, const Options&...) const {
        return {};
    }
};

// ---- the simulation --------------------------------------------------------------------------

struct Event {
    enum class Kind : std::uint8_t {
        Bind,
        Restore,
        Prefill,
        Decode,
        PauseStart,
        PauseDone,
        Finish,
        Abort,
    } kind;
    std::uint32_t lane       = 0;
    std::uint64_t request    = 0; // the plan's id
    std::uint32_t tokens     = 0; // prefill: processed; decode: rows
    std::uint32_t cap        = 0; // prefill: the max_tokens it was given
    std::uint32_t position   = 0; // prefill: cursor before the step
    bool snapshot            = false;
};

struct Cost {
    // Virtual time a prefill step reports: fixed + per_token * tokens.
    std::uint64_t prefill_fixed_ns     = 10'000'000;
    std::uint64_t prefill_per_token_ns = 100'000;
    // Real time a decode round or a prefill step spends, so wall-clock accounting sees them.
    std::chrono::microseconds decode_sleep{1000};
    std::chrono::microseconds prefill_sleep{0};
};

class Program {
public:
    explicit Program(std::uint32_t prefill_chunk, Cost cost = {}) : chunk_(prefill_chunk), cost_(cost) {}

    // ---- test control ----
    // Runs on the worker thread before an execution unit; a test blocks in it to hold the Engine.
    std::function<void(const Event&)> before_unit;

    [[nodiscard]] std::vector<Event> events() const {
        std::lock_guard lock(log_mutex_);
        return log_;
    }

    // ---- planning ----
    // The admission fit check (--admission-fit-check) asks the Program. The fake pool never fills, so
    // the answer is whatever a test sets (default: it fits).
    [[nodiscard]] bool prompt_fits_beside_residents(const RequestBasePlan&,
                                                    const SourceCandidate&) const noexcept {
        return fits_.load(std::memory_order_acquire);
    }
    void set_fits(bool fits) noexcept { fits_.store(fits, std::memory_order_release); }
    [[nodiscard]] RequestBasePlan plan_request(PreparedPrompt&& prompt,
                                               const ResolvedExecutionOptions& options) {
        if (prompt.tokens == 0) { throw std::invalid_argument("prompt must contain tokens"); }
        return RequestBasePlan(prompt.id, prompt.tokens, options.requested_output_tokens);
    }

    [[nodiscard]] PrefillWork prefill_work(const RequestBasePlan&, std::uint32_t begin,
                                           std::uint32_t end) const {
        return make_prefill_work(begin, end > begin ? end - begin : 0, 0, 0, chunk_);
    }

    [[nodiscard]] std::optional<SourceCandidate>
    inspect_source(const RequestBasePlan& base, std::optional<CheckpointHandle> checkpoint,
                   bool = false, std::span<const CheckpointHandle> = {},
                   std::span<const CheckpointHandle> = {}) const {
        if (checkpoint) { return std::nullopt; }
        SourceCandidate source;
        source.remaining_work = prefill_work(base, 0, base.summary().prompt_tokens);
        return source;
    }

    // ---- checkpoints: the fake keeps none ----
    [[nodiscard]] Key checkpoint_key(CheckpointHandle, std::uint32_t) const { return {}; }
    [[nodiscard]] ContextResourceUsage checkpoint_footprint(std::span<const CheckpointHandle>) const {
        return {};
    }
    [[nodiscard]] CheckpointSummary checkpoint_summary(CheckpointHandle) const { return {}; }
    [[nodiscard]] CheckpointMetadata checkpoint_metadata(CheckpointHandle) const { return {}; }
    [[nodiscard]] bool checkpoint_matches(CheckpointHandle, const RequestBasePlan&) const {
        return false;
    }
    [[nodiscard]] std::uint32_t checkpoint_recovery_frontier(CheckpointHandle,
                                                             const RequestBasePlan&,
                                                             std::uint32_t) const {
        return 0;
    }
    [[nodiscard]] std::uint64_t checkpoint_recovery_loss(std::span<const CheckpointHandle>,
                                                         std::span<const CheckpointHandle>) const {
        return 0;
    }
    [[nodiscard]] bool valid_checkpoint(CheckpointHandle handle) const noexcept {
        return handle.generation != 0 && snapshots_.count(handle.index) != 0;
    }
    [[nodiscard]] bool release_checkpoint(CheckpointHandle) noexcept { return false; }
    [[nodiscard]] bool revoke_snapshot(ResumeState&) noexcept { return false; }
    [[nodiscard]] std::size_t pause_state_image_unplaced(SequenceHandle) const { return 0; }
    [[nodiscard]] ContextResourceUsage snapshot_resources(const ResumeState&) const { return {}; }
    [[nodiscard]] std::size_t host_bytes_released(std::span<const CheckpointHandle>) const {
        return 0;
    }
    [[nodiscard]] std::optional<std::size_t> pause_host_bytes(SequenceHandle) const {
        return std::size_t{1} << 20;
    }
    [[nodiscard]] std::size_t release_redundant_host(std::span<const CheckpointHandle>,
                                                     std::optional<SequenceHandle> = std::nullopt) {
        return 0;
    }

    struct Demotion {
        CheckpointHandle handle;
        std::size_t host_bytes = 0;
        std::vector<CheckpointHandle> sources;
        ContextResourceUsage released;
    };
    struct Release {
        std::vector<CheckpointHandle> sources;
        ContextResourceUsage released;
    };
    struct Batch {
        bool append(const Demotion&) { return false; }
        [[nodiscard]] bool covers(const Release&) const { return false; }
        [[nodiscard]] std::optional<Demotion> finish() const { return std::nullopt; }
    };
    struct ReclaimPlan {
        std::vector<Demotion> demotions;
        std::vector<Release> releases;
        [[nodiscard]] Batch begin_kv_batch(ContextResourceUsage) const { return {}; }
        [[nodiscard]] std::uint64_t recovery_loss(std::span<const CheckpointHandle>,
                                                  std::span<const CheckpointHandle>) const {
            return 0;
        }
    };
    [[nodiscard]] ReclaimPlan plan_reclaim(std::span<const CheckpointHandle>,
                                           std::span<const CheckpointHandle>,
                                           ContextResourceUsage) const {
        return {};
    }
    [[nodiscard]] std::vector<Release> plan_releases(std::span<const CheckpointHandle>,
                                                     std::span<const CheckpointHandle>,
                                                     ContextResourceUsage) const {
        return {};
    }
    [[nodiscard]] bool start_demote(const Demotion&) { return false; }

    // ---- resources ----
    [[nodiscard]] ResourceReservation reserve_units(std::span<const ExecutionUnit>) {
        return {.reserved = true};
    }
    void release_units(std::span<const SequenceHandle>) noexcept {}
    [[nodiscard]] bool reclaim_capture_reservation(ContextResourceUsage) { return false; }
    [[nodiscard]] ResourceReservation reserve_vision_window(SequenceHandle) { return {.reserved = true}; }
    [[nodiscard]] bool vision_pending(SequenceHandle) const noexcept { return false; }
    bool poll_vision() { return false; }
    bool drain_vision_window() { return false; }
    [[nodiscard]] bool context_blocks(SequenceHandle) const noexcept { return false; }

    [[nodiscard]] PhysicalUsageSnapshot physical_usage() const noexcept {
        PhysicalUsageSnapshot usage;
        usage.capacity = {.state_slots      = 64,
                          .main_kv_pages    = 1U << 20,
                          .backend_kv_pages = 0,
                          .host_bytes       = std::size_t{1} << 40};
        for (const auto& lane : lanes_) {
            if (lane.bound) { ++usage.occupied.state_slots; }
        }
        return usage;
    }
    [[nodiscard]] MemorySummary memory_summary() const noexcept { return {}; }
    void reset_memory_peaks() noexcept {}
    [[nodiscard]] std::uint64_t persistent_capacity_bytes() const noexcept { return 0; }

    // ---- context transactions: one poll each ----
    [[nodiscard]] BindingReservation start_binding(const RequestBasePlan& base, LaneId lane,
                                                   const SourceCandidate&,
                                                   ResumeState* resume = nullptr,
                                                   ExecutionUnitKind = ExecutionUnitKind::Decode,
                                                   std::uint32_t = 1) {
        if (transaction_ || lane.value >= lanes_.size() || lanes_[lane.value].bound) {
            return {};
        }
        Transaction tx;
        tx.kind = ContextOperationKind::Bind;
        tx.lane = lane.value;
        tx.plan = base.id();
        tx.prompt_tokens = base.summary().prompt_tokens;
        if (resume != nullptr) {
            tx.cursor    = resume->frontier();
            tx.generated = resume->generated();
            tx.resumed   = true;
            if (auto handle = resume->snapshot_handle()) { snapshots_.erase(handle->index); }
        }
        transaction_ = tx;
        return {.reserved = true};
    }
    [[nodiscard]] bool start_pause(SequenceHandle sequence, bool save_snapshot, ExecutionTiming* = nullptr,
                                   bool require_snapshot = false) {
        (void)require_snapshot; // the fake always has room for the snapshot
        if (transaction_ || !valid(sequence)) { return false; }
        Transaction tx;
        tx.kind     = ContextOperationKind::Pause;
        tx.lane     = sequence.lane;
        tx.snapshot = save_snapshot;
        transaction_ = tx;
        record({Event::Kind::PauseStart, sequence.lane, lanes_[sequence.lane].plan, 0, 0,
                lanes_[sequence.lane].cursor, save_snapshot});
        return true;
    }
    [[nodiscard]] bool start_capture(SequenceHandle) { return false; }
    [[nodiscard]] bool capture_is_input(SequenceHandle) const { return false; }
    [[nodiscard]] std::optional<CapturePreparation> prepare_capture(SequenceHandle) {
        return std::nullopt;
    }
    void skip_capture(SequenceHandle) {}
    [[nodiscard]] bool has_context_transaction() const noexcept { return transaction_.has_value(); }
    [[nodiscard]] bool recovery_pending(SequenceHandle) const noexcept { return false; }

    [[nodiscard]] ContextProgress poll_context(CancellationFlagView) {
        ContextProgress progress;
        if (!transaction_) {
            progress.complete = true;
            return progress;
        }
        const Transaction tx = *transaction_;
        transaction_.reset();
        progress.kind      = tx.kind;
        progress.advanced  = true;
        progress.complete  = true;
        progress.published = true;
        if (tx.kind == ContextOperationKind::Bind) {
            Lane& lane      = lanes_[tx.lane];
            lane            = Lane{};
            lane.bound      = true;
            lane.epoch      = ++epoch_;
            lane.plan       = tx.plan;
            lane.prompt     = tx.prompt_tokens;
            lane.cursor     = tx.cursor;
            lane.generated  = tx.generated;
            lane.decoding   = tx.cursor >= tx.prompt_tokens && tx.generated != 0;
            progress.sequence = SequenceHandle{tx.lane, lane.epoch};
            record({tx.resumed ? Event::Kind::Restore : Event::Kind::Bind, tx.lane, tx.plan, 0, 0,
                    tx.cursor, false});
        } else if (tx.kind == ContextOperationKind::Pause) {
            Lane& lane = lanes_[tx.lane];
            std::optional<CheckpointHandle> snapshot;
            if (tx.snapshot) {
                snapshot = CheckpointHandle{next_snapshot_, 1};
                snapshots_.insert(next_snapshot_++);
            }
            progress.paused.emplace(lane.cursor, lane.generated, snapshot);
            record({Event::Kind::PauseDone, tx.lane, lane.plan, 0, 0, lane.cursor, tx.snapshot});
            lane = Lane{};
        }
        return progress;
    }

    // ---- execution ----
    [[nodiscard]] PrefillProgress advance_prefill(SequenceHandle sequence,
                                                  ExecutionTiming* = nullptr,
                                                  TokenMaskProvider* = nullptr,
                                                  std::uint32_t max_tokens = 0) {
        Lane& lane = checked(sequence);
        const std::uint32_t remaining = lane.prompt - lane.cursor;
        const std::uint32_t uncapped  = std::min(chunk_, remaining);
        std::uint32_t step            = uncapped;
        if (max_tokens != 0) { step = std::min(step, max_tokens); }
        Event event{Event::Kind::Prefill, sequence.lane, lane.plan, step, max_tokens, lane.cursor, false};
        hook(event);
        if (cost_.prefill_sleep.count() != 0) { std::this_thread::sleep_for(cost_.prefill_sleep); }
        PrefillProgress progress;
        progress.summary = {.prompt_tokens = lane.prompt};
        progress.processed_prompt_tokens = step;
        progress.cap_applied             = step < uncapped; // the fake has no capture frontiers
        progress.timing.device_wait_ns = cost_.prefill_fixed_ns + cost_.prefill_per_token_ns * step;
        lane.cursor += step;
        record(event);
        if (lane.cursor >= lane.prompt) {
            progress.complete = true;
            PendingBatch batch;
            batch.rows_.push_back(sequence);
            batch.tokens_.push_back(7);
            progress.pending = std::move(batch);
            lane.pending     = true;
        }
        return progress;
    }
    [[nodiscard]] ReplayProgress advance_replay(SequenceHandle, ExecutionTiming* = nullptr) {
        throw std::logic_error("the fake never replays: its pause always saves a snapshot");
    }
    [[nodiscard]] PendingBatch decode(std::span<const SequenceHandle> sequences,
                                      std::span<const RoundBudget>, ExecutionTiming* = nullptr,
                                      TokenMaskProvider* = nullptr) {
        Event event{Event::Kind::Decode, sequences.empty() ? 0U : sequences.front().lane,
                    sequences.empty() ? 0U : lanes_[sequences.front().lane].plan,
                    static_cast<std::uint32_t>(sequences.size()), 0, 0, false};
        hook(event);
        if (cost_.decode_sleep.count() != 0) { std::this_thread::sleep_for(cost_.decode_sleep); }
        PendingBatch batch;
        for (const auto sequence : sequences) {
            checked(sequence).pending = true;
            batch.rows_.push_back(sequence);
            batch.tokens_.push_back(7);
        }
        record(event);
        return batch;
    }
    [[nodiscard]] ExecutionTiming append_forced_tokens(std::span<const SequenceHandle>,
                                                       std::span<const TokenId>, std::uint32_t,
                                                       std::span<const std::optional<std::uint32_t>>,
                                                       ExecutionTiming* = nullptr) {
        throw std::logic_error("the fake has no forced tokens");
    }
    [[nodiscard]] CommitResult commit(PendingBatch&& pending,
                                      std::span<const CommitDecision> decisions,
                                      CommitObservation = CommitObservation::AllRows,
                                      ExecutionTiming* = nullptr) {
        CommitResult result;
        result.row_count = pending.rows_.size();
        for (std::size_t row = 0; row < pending.rows_.size(); ++row) {
            Lane& lane   = checked(pending.rows_[row]);
            lane.pending = false;
            const auto& decision = decisions[row];
            CommitDisposition disposition = CommitDisposition::Active;
            if (decision.failed) {
                disposition = CommitDisposition::FailedReleased;
            } else if (decision.cancelled) {
                disposition = CommitDisposition::CancelledReleased;
            } else if (decision.terminal) {
                disposition = CommitDisposition::Finishable;
            }
            lane.generated += decision.accepted_tokens;
            lane.decoding = true;
            result.rows[row].disposition = disposition;
        }
        return result;
    }
    [[nodiscard]] DiscardResult abort_pending(PendingBatch&& pending) noexcept {
        for (const auto sequence : pending.rows_) {
            if (valid(sequence)) { lanes_[sequence.lane].pending = false; }
        }
        return {ConsumeStatus::Consumed, pending.rows_.size()};
    }
    [[nodiscard]] FinishResult finish(SequenceHandle sequence) noexcept {
        if (!valid(sequence)) { return {}; }
        record({Event::Kind::Finish, sequence.lane, lanes_[sequence.lane].plan, 0, 0, 0, false});
        lanes_[sequence.lane] = Lane{};
        FinishResult result;
        result.status = ConsumeStatus::Consumed;
        return result;
    }
    [[nodiscard]] AbortResult abort(SequenceHandle sequence) noexcept {
        if (!valid(sequence)) { return {}; }
        record({Event::Kind::Abort, sequence.lane, lanes_[sequence.lane].plan, 0, 0,
                lanes_[sequence.lane].cursor, false});
        lanes_[sequence.lane] = Lane{};
        AbortResult result;
        result.status = ConsumeStatus::Consumed;
        return result;
    }
    void fail_all_cleanup() noexcept {
        for (auto& lane : lanes_) { lane = Lane{}; }
        transaction_.reset();
    }

    // ---- checkpoint images (the store is off in these tests) ----
    [[nodiscard]] CheckpointImage export_checkpoints(std::span<const CheckpointHandle>,
                                                     const std::optional<CacheSessionKey>&,
                                                     std::string_view) const {
        return {};
    }
    [[nodiscard]] std::size_t checkpoint_image_host_bytes(std::span<const std::uint8_t>,
                                                          std::string_view) const {
        return 0;
    }
    [[nodiscard]] std::optional<ImportedCheckpoints>
    import_checkpoints(std::span<const std::uint8_t>, std::string_view) {
        return std::nullopt;
    }

private:
    struct Lane {
        bool bound              = false;
        std::uint64_t epoch     = 0;
        std::uint64_t plan      = 0;
        std::uint32_t prompt    = 0;
        std::uint32_t cursor    = 0;
        std::uint32_t generated = 0;
        bool decoding           = false;
        bool pending            = false;
    };
    struct Transaction {
        ContextOperationKind kind = ContextOperationKind::Bind;
        std::uint32_t lane        = 0;
        std::uint64_t plan        = 0;
        std::uint32_t prompt_tokens = 0;
        std::uint32_t cursor      = 0;
        std::uint32_t generated   = 0;
        bool resumed              = false;
        bool snapshot             = false;
    };

    [[nodiscard]] bool valid(SequenceHandle sequence) const noexcept {
        return sequence.lane < lanes_.size() && lanes_[sequence.lane].bound &&
               lanes_[sequence.lane].epoch == sequence.epoch;
    }
    Lane& checked(SequenceHandle sequence) {
        if (!valid(sequence)) { throw std::logic_error("fake program: stale sequence handle"); }
        return lanes_[sequence.lane];
    }
    void hook(const Event& event) {
        if (before_unit) { before_unit(event); }
    }
    void record(Event event) {
        std::lock_guard lock(log_mutex_);
        log_.push_back(event);
    }

    std::uint32_t chunk_;
    Cost cost_;
    std::array<Lane, kMaximumConcurrency> lanes_{};
    std::atomic<bool> fits_{true};
    std::optional<Transaction> transaction_;
    std::uint64_t epoch_ = 0;
    std::uint32_t next_snapshot_ = 1;
    std::set<std::uint32_t> snapshots_;
    mutable std::mutex log_mutex_;
    std::vector<Event> log_;
};

struct ModelContract {
    using Frontend          = fake::Frontend;
    using PreparedPrompt    = fake::PreparedPrompt;
    using OutputSession     = fake::OutputSession;
    using PublishedOutput   = fake::PublishedOutput;
    using RequestBasePlan   = fake::RequestBasePlan;
    using SequenceHandle    = fake::SequenceHandle;
    using CheckpointHandle  = fake::CheckpointHandle;
    using CheckpointSummary = fake::CheckpointSummary;
    using SourceCandidate   = fake::SourceCandidate;
    using ResumeState       = fake::ResumeState;
    using ExecutionUnit     = fake::ExecutionUnit;
    using ExecutionUnitKind = fake::ExecutionUnitKind;
    using ContextProgress   = fake::ContextProgress;
    using PendingBatch      = fake::PendingBatch;
    using PrefillProgress   = fake::PrefillProgress;
    using ReplayProgress    = fake::ReplayProgress;
    using CommitResult      = fake::CommitResult;
    using DiscardResult     = fake::DiscardResult;
    using FinishResult      = fake::FinishResult;
    using AbortResult       = fake::AbortResult;
    using Program           = fake::Program;
    using CacheSessionKey   = fake::CacheSessionKey;
    using CheckpointImage   = fake::CheckpointImage;
};

struct Instance {
    using ModelContract = fake::ModelContract;
    using Device        = fake::Device;

    fake::Frontend frontend;
    KvCapacityResolution kv_capacity_resolution;
    std::unique_ptr<fake::Program> program;
    std::vector<CheckpointHandle> external_checkpoints;

    explicit Instance(std::uint32_t prefill_chunk, Cost cost = {})
        : program(std::make_unique<fake::Program>(prefill_chunk, cost)) {}
    void install_external_checkpoints() noexcept {}
};

} // namespace ninfer::runtime::fake
