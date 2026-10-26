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
// What it is not: the real Program. It has no memory, no speculation and no media, and its only
// checkpoints are shared prefixes a test installs (add_shared_prefix); its pause always saves a
// snapshot, restores always succeed and a context transaction takes exactly one poll. It checks
// the Engine's logic and the order of its calls, not GPU behaviour.
//
// Lossy prefill (Program::enable_lossy, docs/maintainer/lossy-long-prefill.md) is modelled the
// way the Engine sees it: a plan with lossy() prefills exactly to M, then scout steps report
// scouted_tokens with nothing processed, and the step that ends the scout reports the compacted
// length (M + keep share of U) and a compacted plan. While a lane scouts, start_pause refuses;
// abort_lossy_scout ends the scout at M, and so does abort. A pause in Exact carries the lossy
// state to the restore (CarriedLossy), as the real Program's saved control does.

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
#include <cmath>
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
    // The plan's lossy().exact_end: set for a prompt the lossy path may take.
    std::optional<std::uint32_t> lossy_exact_end;
    // The first `shared_tokens` tokens are the shared prefix `shared_group` (0: none), which an
    // installed add_shared_prefix checkpoint of that group serves.
    std::uint64_t shared_group  = 0;
    std::uint32_t shared_tokens = 0;
};

struct LossyPlan {
    std::uint32_t exact_end = 0;
};

struct LossyOutcome {
    std::uint32_t exact_end       = 0;
    std::uint32_t uncached_tokens = 0;
    std::uint32_t kept_tokens     = 0;
    bool skipped                  = false;
    double scout_seconds          = 0.0;
    bool declined                 = false;
    // The fake keeps the first kept_tokens of U: one run.
    std::vector<std::array<std::uint32_t, 2>> kept_runs;
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
    RequestBasePlan(const PreparedPrompt& prompt, std::uint32_t output_tokens)
        : RequestBasePlan(prompt.id, prompt.tokens, output_tokens) {
        if (prompt.lossy_exact_end) { lossy_ = LossyPlan{*prompt.lossy_exact_end}; }
        shared_group_  = prompt.shared_group;
        shared_tokens_ = prompt.shared_tokens;
    }
    RequestBasePlan(std::uint64_t id, std::uint32_t prompt_tokens, std::uint32_t output_tokens) {
        id_                         = id;
        summary_.prompt_tokens      = prompt_tokens;
        summary_.reported_prompt_tokens = prompt_tokens;
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
        if (shared_group_ != 0 && frontier <= shared_tokens_) {
            return Key{{shared_group_, 1}, frontier, 0};
        }
        return Key{{id_, 0}, frontier, 0};
    }
    [[nodiscard]] std::uint64_t id() const noexcept { return id_; }
    [[nodiscard]] std::optional<LossyPlan> lossy() const noexcept { return lossy_; }
    [[nodiscard]] std::uint64_t shared_group() const noexcept { return shared_group_; }
    [[nodiscard]] std::uint32_t shared_tokens() const noexcept { return shared_tokens_; }
    // One gap, as the real gap map counts it: past `gap_begin_` and all `gap_kept_` kept tokens a
    // frontier stands for `gap_dropped_` more of the client's tokens (the fake keeps the first
    // tokens of U, so a frontier among them is unchanged).
    [[nodiscard]] std::uint32_t reported_frontier(std::uint32_t frontier) const noexcept {
        return gap_dropped_ != 0 && frontier > gap_begin_ && frontier >= gap_begin_ + gap_kept_
                   ? frontier + gap_dropped_
                   : frontier;
    }

    // The plan the fake reports after a selection kept `kept` of the tokens after `begin`.
    [[nodiscard]] RequestBasePlan compacted(std::uint32_t begin, std::uint32_t kept) const {
        RequestBasePlan out             = *this;
        const std::uint32_t dropped     = summary_.prompt_tokens - begin - kept;
        out.summary_.prompt_tokens      = begin + kept;
        out.lossy_.reset();
        out.gap_begin_   = begin;
        out.gap_kept_    = kept;
        out.gap_dropped_ = dropped;
        return out;
    }

private:
    std::uint64_t id_ = 0;
    RequestPlanSummary summary_;
    CacheAdvice advice_;
    std::optional<LossyPlan> lossy_;
    std::uint64_t shared_group_  = 0;
    std::uint32_t shared_tokens_ = 0;
    std::uint32_t gap_begin_     = 0;
    std::uint32_t gap_kept_      = 0;
    std::uint32_t gap_dropped_   = 0;
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
// A paused lane's lossy state, carried to its restore like the real Program's saved RequestControl
// (pause.cpp keeps the whole control, install_binding moves it back): a request paused in Exact
// goes on to M and scouts after its restore. A scout cannot pause; Main needs nothing carried (the
// Engine rebinds the compacted plan).
struct CarriedLossy {
    std::uint32_t exact_end = 0;
};

class ResumeState {
public:
    ResumeState() = default;
    ResumeState(std::uint32_t frontier, std::uint32_t generated, std::optional<CheckpointHandle> snapshot,
                std::uint32_t base = 0, std::optional<CarriedLossy> lossy = std::nullopt)
        : frontier_(frontier), generated_(generated), base_(base), snapshot_(snapshot), lossy_(lossy) {}
    ResumeState(ResumeState&&) noexcept            = default;
    ResumeState& operator=(ResumeState&&) noexcept = default;

    [[nodiscard]] bool has_snapshot() const noexcept { return snapshot_.has_value(); }
    [[nodiscard]] std::optional<CheckpointHandle> snapshot_handle() const noexcept { return snapshot_; }
    [[nodiscard]] std::uint32_t frontier() const noexcept { return frontier_; }
    [[nodiscard]] std::uint32_t generated() const noexcept { return generated_; }
    [[nodiscard]] std::uint32_t base() const noexcept { return base_; }
    [[nodiscard]] std::optional<CarriedLossy> lossy() const noexcept { return lossy_; }

private:
    std::uint32_t frontier_  = 0;
    std::uint32_t generated_ = 0;
    std::uint32_t base_      = 0; // the admitted reuse frontier, which the Begin summary keeps
    std::optional<CheckpointHandle> snapshot_;
    std::optional<CarriedLossy> lossy_;
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
    std::uint32_t scouted_tokens          = 0;
    std::optional<std::uint32_t> compacted_prompt_tokens;
    std::optional<RequestBasePlan> compacted_plan;
    std::optional<LossyOutcome> lossy;
    bool complete                         = false;
    bool partial                          = false;
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
    std::size_t host_pause_reserve_bytes          = 0;
    std::size_t host_pause_reserve_occupied_bytes = 0;
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
        EnableLossy, // enable_lossy took the request; position: M
        Scout,       // a scout step: tokens scouted, position: M + already scouted
        ScoutEnd,    // a scout ended early and its lane was restored to M (abort, abort_lossy_scout)
    } kind;
    std::uint32_t lane       = 0;
    std::uint64_t request    = 0; // the plan's id
    std::uint32_t tokens     = 0; // prefill: processed; decode: rows
    std::uint32_t cap        = 0; // prefill: the max_tokens it was given
    std::uint32_t position   = 0; // prefill: cursor before the step
    bool snapshot            = false;
    std::uint32_t slice      = 0;     // prefill: which layer slice of its chunk this call ran
    bool partial             = false; // prefill: more slices of this chunk remain
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

    // Installs a SharedPrefix checkpoint over the first `frontier` tokens of group `group`, for
    // Instance::external_checkpoints (the Engine pins those as shared prefixes at construction).
    [[nodiscard]] CheckpointHandle add_shared_prefix(std::uint64_t group, std::uint32_t frontier) {
        const CheckpointHandle handle{kSharedBase + static_cast<std::uint32_t>(shared_.size()), 1};
        shared_.push_back({group, frontier});
        return handle;
    }
    // Lossy prefill: the kept share of U a selection reports (1: keeps everything, so skipped),
    // whether enable_lossy takes anything at all, and the scout's cost share.
    void set_lossy_keep(double keep) noexcept { lossy_keep_ = keep; }
    void set_lossy_supported(bool supported) noexcept { lossy_supported_ = supported; }
    // While a lane scouts and another is bound, a Decode unit finds no KV pages: the memory
    // pressure under which the Engine has to pause a scouting lane.
    void set_decode_shortage_while_scouting(bool on) noexcept { decode_shortage_while_scouting_ = on; }
    [[nodiscard]] std::uint32_t scout_pause_refusals() const noexcept {
        return scout_pause_refusals_.load(std::memory_order_acquire);
    }
    [[nodiscard]] bool scouting() const noexcept { return scout_lane_.has_value(); }

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
        return RequestBasePlan(prompt, options.requested_output_tokens);
    }

    [[nodiscard]] PrefillWork prefill_work(const RequestBasePlan&, std::uint32_t begin,
                                           std::uint32_t end) const {
        return make_prefill_work(begin, end > begin ? end - begin : 0, 0, 0, chunk_);
    }

    [[nodiscard]] std::optional<SourceCandidate>
    inspect_source(const RequestBasePlan& base, std::optional<CheckpointHandle> checkpoint,
                   bool = false, std::span<const CheckpointHandle> = {},
                   std::span<const CheckpointHandle> = {}) const {
        SourceCandidate source;
        if (checkpoint) {
            const auto* shared = shared_prefix(*checkpoint);
            if (shared == nullptr || !serves(*shared, base)) { return std::nullopt; }
            source.checkpoint    = checkpoint;
            source.reused_tokens = shared->frontier;
        }
        source.remaining_work =
            prefill_work(base, source.reused_tokens, base.summary().prompt_tokens);
        return source;
    }

    // ---- checkpoints: only installed shared prefixes (add_shared_prefix) ----
    [[nodiscard]] Key checkpoint_key(CheckpointHandle handle, std::uint32_t frontier) const {
        if (const auto* shared = shared_prefix(handle)) {
            return Key{{shared->group, 1}, frontier, 0};
        }
        return {};
    }
    [[nodiscard]] ContextResourceUsage checkpoint_footprint(std::span<const CheckpointHandle>) const {
        return {};
    }
    [[nodiscard]] CheckpointSummary checkpoint_summary(CheckpointHandle handle) const {
        if (const auto* shared = shared_prefix(handle)) {
            return {.key      = checkpoint_key(handle, shared->frontier),
                    .frontier = shared->frontier,
                    .role     = CheckpointRole::SharedPrefix};
        }
        return {};
    }
    [[nodiscard]] CheckpointMetadata checkpoint_metadata(CheckpointHandle handle) const {
        if (const auto* shared = shared_prefix(handle)) {
            return {.frontier = shared->frontier, .role = CheckpointRole::SharedPrefix};
        }
        return {};
    }
    [[nodiscard]] bool checkpoint_matches(CheckpointHandle handle, const RequestBasePlan& base) const {
        const auto* shared = shared_prefix(handle);
        return shared != nullptr && serves(*shared, base);
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
        return handle.generation != 0 &&
               (snapshots_.count(handle.index) != 0 || shared_prefix(handle) != nullptr);
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
    // Whether a snapshot fits the Host as things stand (default: it does). A test sets it false to
    // stand for a Host full of cached history.
    [[nodiscard]] bool pause_snapshot_fits(SequenceHandle) const noexcept {
        return snapshot_fits_.load(std::memory_order_acquire);
    }
    void set_snapshot_fits(bool fits) noexcept { snapshot_fits_.store(fits, std::memory_order_release); }
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
    [[nodiscard]] ResourceReservation reserve_units(std::span<const ExecutionUnit> units) {
        if (decode_shortage_while_scouting_ && scout_lane_ &&
            std::count_if(lanes_.begin(), lanes_.end(), [](const Lane& l) { return l.bound; }) > 1 &&
            std::any_of(units.begin(), units.end(), [](const ExecutionUnit& unit) {
                return unit.kind == ExecutionUnitKind::Decode;
            })) {
            return {.reserved = false, .shortage = {.main_kv_pages = 1}};
        }
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
            // A scout's KV pages are the lane's own until the scout ends (one per token here).
            usage.occupied.main_kv_pages += lane.scouted;
        }
        return usage;
    }
    [[nodiscard]] MemorySummary memory_summary() const noexcept { return {}; }
    void reset_memory_peaks() noexcept {}
    [[nodiscard]] std::uint64_t persistent_capacity_bytes() const noexcept { return 0; }

    // ---- context transactions: one poll each ----
    [[nodiscard]] BindingReservation start_binding(const RequestBasePlan& base, LaneId lane,
                                                   const SourceCandidate& source,
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
        tx.base_plan     = base;
        tx.cursor        = source.reused_tokens;
        tx.base          = source.reused_tokens;
        if (resume != nullptr) {
            tx.cursor    = resume->frontier();
            tx.generated = resume->generated();
            tx.base      = resume->base();
            tx.resumed   = true;
            tx.lossy     = resume->lossy();
            if (auto handle = resume->snapshot_handle()) { snapshots_.erase(handle->index); }
        }
        transaction_ = tx;
        return {.reserved = true};
    }
    [[nodiscard]] bool start_pause(SequenceHandle sequence, bool save_snapshot, ExecutionTiming* = nullptr,
                                   bool require_snapshot = false) {
        (void)require_snapshot; // the fake always has room for the snapshot
        if (transaction_ || !valid(sequence)) { return false; }
        // Like the real Program: a scout keeps uncommitted pages, so there is no frontier to save.
        if (scout_lane_ == sequence.lane) {
            scout_pause_refusals_.fetch_add(1, std::memory_order_acq_rel);
            return false;
        }
        // Like the real Program: a chunk part-way through its layers has no frontier to save.
        if (lanes_[sequence.lane].slice_count != 0) { return false; }
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
            lane.base_plan  = tx.base_plan;
            lane.prompt     = tx.prompt_tokens;
            lane.base       = tx.base;
            lane.resumed    = tx.resumed;
            lane.cursor     = tx.cursor;
            lane.generated  = tx.generated;
            lane.decoding   = tx.cursor >= tx.prompt_tokens && tx.generated != 0;
            if (tx.lossy) {
                lane.lossy = Lossy{.phase = LossyPhase::Exact, .exact_end = tx.lossy->exact_end};
            }
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
            std::optional<CarriedLossy> carried;
            if (lane.lossy && lane.lossy->phase == LossyPhase::Exact) {
                carried = CarriedLossy{.exact_end = lane.lossy->exact_end};
            }
            progress.paused.emplace(lane.cursor, lane.generated, snapshot, lane.base, carried);
            record({Event::Kind::PauseDone, tx.lane, lane.plan, 0, 0, lane.cursor, tx.snapshot});
            lane = Lane{};
        }
        return progress;
    }

    // ---- execution ----
    [[nodiscard]] PrefillProgress advance_prefill(SequenceHandle sequence,
                                                  ExecutionTiming* = nullptr,
                                                  TokenMaskProvider* = nullptr,
                                                  std::uint32_t max_tokens   = 0,
                                                  std::uint32_t layer_slices = 0) {
        Lane& lane = checked(sequence);
        // A chunk begun in slices is finished at the size it began, as the real Program does.
        const bool resuming = lane.slice_count != 0;
        std::uint32_t step  = lane.slice_tokens;
        bool cap_narrowed   = false;
        std::optional<LossyOutcome> declined;
        std::uint32_t exact_limit = UINT32_MAX;
        if (!resuming && lane.lossy && lane.lossy->phase != LossyPhase::Main) {
            auto& lossy = *lane.lossy;
            if (lane.cursor < lossy.exact_end) {
                exact_limit = lossy.exact_end - lane.cursor; // Exact stops at M
            } else if (lossy.phase == LossyPhase::Exact && scout_lane_ &&
                       *scout_lane_ != sequence.lane) {
                // The one scout is taken: prefill all of U as an ordinary request.
                lossy.phase = LossyPhase::Main;
                const std::uint32_t span = lane.prompt - lossy.exact_end;
                declined = LossyOutcome{.exact_end       = lossy.exact_end,
                                        .uncached_tokens = span,
                                        .kept_tokens     = span,
                                        .skipped         = true,
                                        .declined        = true};
            } else {
                return scout_step(sequence, lane, max_tokens);
            }
        }
        if (!resuming) {
            const std::uint32_t remaining = std::min(lane.prompt - lane.cursor, exact_limit);
            step                          = std::min(chunk_, remaining);
            const std::uint32_t uncapped = step;
            if (max_tokens != 0) { step = std::min(step, max_tokens); }
            cap_narrowed = step < uncapped; // the fake has no capture frontiers
            if (layer_slices > 1) {
                lane.slice_count  = layer_slices;
                lane.slice_index  = 0;
                lane.slice_tokens = step;
                lane.slice_cap_applied = cap_narrowed;
            }
        }
        const bool sliced = lane.slice_count != 0;
        const bool last   = !sliced || lane.slice_index + 1 >= lane.slice_count;
        Event event{Event::Kind::Prefill, sequence.lane, lane.plan, last ? step : 0U, max_tokens,
                    lane.cursor, false, lane.slice_index, !last};
        hook(event);
        if (cost_.prefill_sleep.count() != 0) { std::this_thread::sleep_for(cost_.prefill_sleep); }
        PrefillProgress progress;
        progress.summary = summary(lane);
        if (declined) {
            progress.compacted_prompt_tokens = lane.prompt;
            progress.lossy                   = declined;
        }
        const std::uint64_t step_ns = cost_.prefill_fixed_ns + cost_.prefill_per_token_ns * step;
        progress.timing.device_wait_ns = sliced ? step_ns / lane.slice_count : step_ns;
        if (!last) {
            // Part of the chunk ran: nothing is processed and the cursor stays.
            ++lane.slice_index;
            progress.partial = true;
            record(event);
            return progress;
        }
        lane.slice_count = 0;
        lane.slice_index = 0;
        progress.processed_prompt_tokens = step;
        progress.cap_applied             = resuming ? lane.slice_cap_applied : cap_narrowed;
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
        // A cleared lane drops its scout and the scout's pages with it.
        if (scout_lane_ == sequence.lane) { end_scout(sequence.lane); }
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
        scout_lane_.reset();
    }

    // ---- lossy prefill ----
    [[nodiscard]] bool enable_lossy(SequenceHandle sequence) {
        if (!valid(sequence)) { return false; }
        Lane& lane       = lanes_[sequence.lane];
        const auto plan  = lane.base_plan.lossy();
        if (!lossy_supported_ || !plan || lane.lossy || lane.resumed || lane.cursor != lane.base) {
            return false;
        }
        const std::uint32_t exact_end = std::max(plan->exact_end, lane.base);
        if (exact_end >= lane.prompt) { return false; }
        lane.lossy = Lossy{.phase = LossyPhase::Exact, .exact_end = exact_end};
        record({Event::Kind::EnableLossy, sequence.lane, lane.plan, 0, 0, exact_end, false});
        return true;
    }
    [[nodiscard]] bool abort_lossy_scout(SequenceHandle sequence) {
        if (!valid(sequence) || scout_lane_ != sequence.lane) { return false; }
        end_scout(sequence.lane);
        lanes_[sequence.lane].lossy->phase = LossyPhase::Main;
        return true;
    }
    [[nodiscard]] double lossy_scout_share() const noexcept { return lossy_supported_ ? 0.25 : 0.0; }

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
    static constexpr std::uint32_t kSharedBase = 1'000'000;
    enum class LossyPhase : std::uint8_t { Exact, Scout, Main };
    struct Lossy {
        LossyPhase phase        = LossyPhase::Exact;
        std::uint32_t exact_end = 0;
        double scout_seconds    = 0.0;
    };
    struct SharedPrefix {
        std::uint64_t group    = 0;
        std::uint32_t frontier = 0;
    };
    struct Lane {
        bool bound              = false;
        std::uint64_t epoch     = 0;
        std::uint64_t plan      = 0;
        std::uint32_t prompt    = 0;
        std::uint32_t cursor    = 0;
        std::uint32_t generated = 0;
        bool decoding           = false;
        bool pending            = false;
        std::uint32_t slice_count  = 0; // slices of the chunk in progress; 0: not part-way
        std::uint32_t slice_index  = 0;
        std::uint32_t slice_tokens = 0;
        bool slice_cap_applied = false;
        RequestBasePlan base_plan;
        std::uint32_t base   = 0; // the reuse frontier the binding started from
        bool resumed         = false;
        std::optional<Lossy> lossy;
        std::uint32_t scouted = 0; // tokens of U the running scout has covered
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
        RequestBasePlan base_plan;
        std::uint32_t base        = 0;
        std::optional<CarriedLossy> lossy;
    };

    [[nodiscard]] const SharedPrefix* shared_prefix(CheckpointHandle handle) const noexcept {
        if (handle.generation != 1 || handle.index < kSharedBase ||
            handle.index - kSharedBase >= shared_.size()) {
            return nullptr;
        }
        return &shared_[handle.index - kSharedBase];
    }
    [[nodiscard]] static bool serves(const SharedPrefix& shared, const RequestBasePlan& base) noexcept {
        return base.shared_group() == shared.group && shared.frontier <= base.shared_tokens() &&
               shared.frontier < base.summary().prompt_tokens;
    }
    [[nodiscard]] static BeginSummary summary(const Lane& lane) noexcept {
        return {.prompt_tokens        = lane.prompt,
                .reused_prompt_tokens = lane.base,
                .prefix_reuse_path = lane.base ? PrefixReusePath::Checkpoint : PrefixReusePath::Root};
    }
    void end_scout(std::uint32_t lane) noexcept {
        record({Event::Kind::ScoutEnd, lane, lanes_[lane].plan, lanes_[lane].scouted, 0,
                lanes_[lane].cursor, false});
        lanes_[lane].scouted = 0;
        scout_lane_.reset();
    }

    // One scout step at M: scores up to a chunk of U and processes nothing. The step that covers
    // the rest of U selects: the prompt becomes M + the kept share of U (unchanged when it keeps
    // everything), and the plan follows.
    PrefillProgress scout_step(SequenceHandle sequence, Lane& lane, std::uint32_t max_tokens) {
        auto& lossy = *lane.lossy;
        if (lossy.phase == LossyPhase::Exact) {
            lossy.phase = LossyPhase::Scout;
            scout_lane_ = sequence.lane;
        }
        const std::uint32_t span      = lane.prompt - lossy.exact_end;
        const std::uint32_t uncapped  = std::min(chunk_, span - lane.scouted);
        const std::uint32_t step      = max_tokens != 0 ? std::min(uncapped, max_tokens) : uncapped;
        Event event{Event::Kind::Scout, sequence.lane, lane.plan, step, max_tokens,
                    lossy.exact_end + lane.scouted, false};
        hook(event);
        if (cost_.prefill_sleep.count() != 0) { std::this_thread::sleep_for(cost_.prefill_sleep); }
        PrefillProgress progress;
        progress.scouted_tokens = step;
        progress.cap_applied    = step < uncapped;
        const auto step_ns      = static_cast<std::uint64_t>(
            static_cast<double>(cost_.prefill_fixed_ns + cost_.prefill_per_token_ns * step) *
            lossy_scout_share());
        progress.timing.device_wait_ns = step_ns;
        lossy.scout_seconds += static_cast<double>(step_ns) * 1e-9;
        lane.scouted += step;
        record(event);
        if (lane.scouted == span) {
            const auto kept = static_cast<std::uint32_t>(std::clamp<double>(
                std::ceil(lossy_keep_ * static_cast<double>(span)), 1.0, static_cast<double>(span)));
            LossyOutcome outcome{.exact_end       = lossy.exact_end,
                                 .uncached_tokens = span,
                                 .kept_tokens     = kept,
                                 .skipped         = kept == span,
                                 .scout_seconds   = lossy.scout_seconds,
                                 .kept_runs       = {{0U, kept}}};
            if (kept != span) {
                lane.base_plan          = lane.base_plan.compacted(lossy.exact_end, kept);
                lane.prompt             = lossy.exact_end + kept;
                progress.compacted_plan = lane.base_plan;
            }
            lossy.phase  = LossyPhase::Main;
            lane.scouted = 0;
            scout_lane_.reset();
            progress.compacted_prompt_tokens = lane.prompt;
            progress.lossy                   = outcome;
        }
        progress.summary = summary(lane);
        return progress;
    }

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
    std::atomic<bool> snapshot_fits_{true};
    std::uint32_t next_snapshot_ = 1;
    std::set<std::uint32_t> snapshots_;
    std::vector<SharedPrefix> shared_;
    std::optional<std::uint32_t> scout_lane_;
    double lossy_keep_                   = 0.2;
    bool lossy_supported_                = true;
    bool decode_shortage_while_scouting_ = false;
    std::atomic<std::uint32_t> scout_pause_refusals_{0};
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
