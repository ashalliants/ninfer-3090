#pragma once

// Adapted from Infernix a3edb450 src/models/qwen4_exp/program/program.h (Apache-2.0).
// Modified for NInfer-3090: implements NInfer's Engine execution contract (the one EngineCore,
// Scheduler and ResourceManager drive for Qwen3.5) instead of Infernix's hybrid-cache contract:
// prefix reuse through NInfer's context cache with checkpoints in a pinned Host tier or an NVMe tier
// behind a RAM cache, persisted by the Engine's context store
// (docs/maintainer/flash-next-context-cache.md), one lane, MTP speculation (Infernix's drafter,
// verification and commit), image input (Infernix's vision program: the tower encodes in frames
// lent by the expert cache at the prompt's first prefill unit, the handoff stays lent until the
// prompt's last call), CPU-served misses in verification rounds only (opt-in), no n-gram copy
// drafts or SSD expert tier.
//
// The Qwen4Exp Program: the Engine-facing execution of Qwen3.8-Flash-Next. It owns the lane, its
// paged INT8 KV (an elastic VMM pool), recurrent state, sampling state and the one pending model
// unit, the VRAM expert cache (LFRU frames, warm-start state) and the decode CUDA graph. Prompts and
// outputs use the Qwen3.5 frontend (the same tokenizer and chat template).

#include "core/gdn_replay_records.h"
#include "models/qwen3_5/frontend/frontend.h"
#include "models/qwen3_5/frontend/output_session.h"
#include "models/qwen3_5/frontend/prepared_prompt.h"
// The passive data types of the Engine's execution contract (checkpoint handles, source
// candidates, images) are declared with the Qwen3.5 Program; this Program uses them unchanged.
#include "models/qwen3_5/frontend/graft.h"
#include "models/qwen3_5/program/program.h"
#include "models/qwen4_exp/config.h"
#include "models/qwen4_exp/memory_plan.h"
#include "models/qwen4_exp/program/expert_cache/expert_cache.h"
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
    // Never set: this Program does not slice its prefill (the Engine reads it for any model).
    bool partial = false;
    bool cap_applied                      = false; // this Program ignores the cap
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
    // --shared-expert-stream: the static workspace's room for the shared expert's side-stream carve
    // at its widest call (plan_shared_expert_workspace; 0 when off), allocated with `workspace`.
    std::uint64_t shared_expert_workspace = 0;
    // --mtp-constrained: a verification round's grammar masks, one per draft and the bonus position
    // (plan_mtp_constrained_masks; 0 when off or without MTP).
    std::uint64_t mtp_constrained_masks = 0;
    // --mtp-joint-acceptance: the drafts' probabilities, one FP32 word per draft
    // (plan_mtp_draft_probs; 0 when off or without MTP).
    std::uint64_t mtp_draft_probs = 0;
    // --hot-embedding-rows: the device copy of the token embedding's first rows
    // (plan_hot_embedding(...).hot; 0 when off). The whole table is in the pinned block then.
    std::uint64_t hot_embedding = 0;
    // --cpu-verify-misses or --prefill-cpu-share: the CPU miss service's device counter of published
    // requests (plan_cpu_misses(...).device; 0 when both are off). Its channel and the calls' error
    // word are pinned host memory (.pinned), charged to the RAM ledger.
    std::uint64_t cpu_misses = 0;
    // --hc-int8: the int8 copies of every mixer's hyper-connection matrices (plan_hc_int8; 0 when
    // off), beside the BF16 weights, which stay loaded.
    std::uint64_t hc_int8_bytes = 0;
    // --ngram-flag-wait: the device counter of the round graphs' flag waits
    // (plan_ngram_flag_wait(...).device; 0 when off). The flag and the error word are pinned host
    // memory (.pinned), charged to the RAM ledger.
    std::uint64_t ngram_flag_wait = 0;
    std::uint64_t expert_record_stride = 0; // one expert frame
    std::uint32_t max_frames           = 0; // the most frames the expert cache uses
    std::uint32_t graph_bound          = 0; // CUDA graph executables the Program may instantiate (counts both halves of the split decode)

    [[nodiscard]] std::uint64_t fixed_bytes() const noexcept {
        return kv + workspace + staging + state + io + residency + speculation + shared_expert_workspace +
               mtp_constrained_masks + mtp_draft_probs + hot_embedding + cpu_misses + hc_int8_bytes +
               ngram_flag_wait;
    }
};

struct ProgramOptions {
    int device                    = 0;
    std::uint32_t max_context     = 0;
    std::uint32_t max_concurrency = 1;
    std::uint32_t prefill_chunk   = 1024;
    bool use_cuda_graph           = true;
    std::filesystem::path ngram_volume;
    // The n-gram volume's keep-alive period in milliseconds (0: off); see FlashNextOptions.
    std::uint32_t ngram_keepalive_ms = 40;
    // The expert cache's saved state: read at startup to fill the frames (warm start), written at
    // shutdown and, between requests, every 10 minutes. Empty: no warm start. The identity names
    // the artifact the state belongs to.
    std::filesystem::path expert_state;
    std::string expert_state_identity;
    // The default state shipped beside the model: read (never written) only when `expert_state`
    // is set and could not be used, so a fresh install starts warm. Its identity is the artifact
    // id, which every copy of the conversion shares. Empty: none.
    std::filesystem::path default_expert_state;
    std::string default_expert_state_identity;
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
    // A constrained row speculates too: the verification takes one grammar mask per position from
    // the round's drafts (FlashNextOptions::mtp_constrained). Off: a constrained row decodes plain.
    bool mtp_constrained = false;
    // Basalt's joint-acceptance draft length (FlashNextOptions::mtp_joint_acceptance): every round
    // drafts the ceiling (mtp_draft_tokens, or less near the request's end) with each draft's
    // probability, then verifies joint_acceptance_window of them (draft_policy.h) under the lane's
    // JointAcceptanceThreshold. It replaces the acceptance-estimate policy; with mtp_fixed_drafts
    // it is refused. Off: the drafter graphs, the drafts and the window are as before.
    bool mtp_joint_acceptance = false;
    // EngineOptions::relaxed_drafting is set: a sampled verification whose request has a threshold
    // runs the relaxed acceptance instance (relaxed drafting; changes the output distribution).
    // Off: every verification runs the exact instance, as before.
    bool relaxed_drafting = false;
    // FlashNextOptions::ngram_prefetch: each MTP round reads the anchor column's n-gram rows into
    // the volume's host cache while the draft graph runs, so verification finds them cached. Only
    // the host work moves; the rows, the drafts and the output are unchanged.
    bool ngram_prefetch = false;
    // Image input: the model loaded its vision tower. Prompts with media encode it into frames lent
    // by the expert cache (its io words are planned only when set).
    bool vision = false;
    // The context cache's pinned Host tier (checkpoints for prefix reuse), sized by the startup RAM
    // ledger. Without an NVMe tier it holds every checkpoint (0: no prefix reuse); with one it is
    // the NVMe tier's RAM cache (0: disk-only reuse).
    std::uint64_t context_cache_bytes = 0;
    // The context cache's NVMe tier (program/nvme_tier.h): a scratch file of this many bytes at
    // context_nvme_path holds every checkpoint, created at startup and removed at exit; 0: none.
    // Its pinned bounce buffer (both halves) is context_bounce_bytes.
    std::uint64_t context_nvme_bytes = 0;
    std::filesystem::path context_nvme_path;
    std::uint64_t context_bounce_bytes = 128ULL << 20;
    // Each MoE block's shared expert on a high-priority side stream beside the routed experts
    // (execution::ForwardOptions::shared_expert_stream); the same outputs, bit for bit.
    bool shared_expert_stream = false;
    // FlashNextOptions::hot_embedding_rows: the model loaded its token embedding into the pinned
    // block (plan_load with hot_embedding), and the Program copies its first rows to the device; the
    // forward reads the rest from the pinned block (execution::ForwardOptions::hot_embedding_rows).
    // 0: the embedding is in device memory.
    std::uint32_t hot_embedding_rows = 0;
    // FlashNextOptions::gdn_head_map: the GDN recurrence reads the key heads in place through the
    // tiled head map (execution::ForwardOptions::gdn_head_map) instead of tiled copies; the GDN
    // replay records then hold the key heads only (gdn_records_spec). The same outputs, bit for bit.
    bool gdn_head_map = false;
    // FlashNextOptions::moe_precast_input: each routed-expert call casts its input to A8 once
    // (execution::ForwardOptions::moe_precast_input); the workspace terms include its planes.
    bool moe_precast_input = false;
    // FlashNextOptions::moe_gate_up_registers: the routed experts' gate/up rows from registers
    // (execution::ForwardOptions::moe_gate_up_registers); requires moe_precast_input. No memory.
    bool moe_gate_up_registers = false;
    // FlashNextOptions::moe_fused_route: decode-width routed-expert calls route and dispatch in one
    // launch (execution::ForwardOptions::moe_fused_route). No memory.
    bool moe_fused_route = false;
    // FlashNextOptions::cpu_verify_misses: a CPU miss service (ops::offloaded_moe::CpuMissService)
    // over the expert banks, running for the Program's life; each MTP verification's routed experts
    // hand it some of their misses (execution::ForwardExperts::cpu), computed by its worker team
    // while the GPU runs the other experts and the shared expert. The same outputs, bit for bit. A
    // round whose CPU leg times out throws (its request ends with an error, nothing of the round is
    // committed); once the service thread has failed, rounds decode plain.
    // Requires MTP (mtp_draft_tokens > 0). cpu_verify_workers: the team's threads, the service thread
    // included (1..64).
    bool cpu_verify_misses           = false;
    std::uint32_t cpu_verify_workers = 6;
    // FlashNextOptions::prefill_cpu_share (num / den, num 0: off): the same service (started for
    // either option) takes that share of each layer's misses in prefill and forced-token calls
    // (execution::ForwardExperts::prefill_cpu, ForwardBatch::cpu_share), the fewest-column ones of
    // at most 8 columns first, while the GPU stages and runs the rest. A streamed call instead leaves
    // that share of each layer's experts, the lowest LFRU use counts, out of the expert stream
    // (ExpertStream::Share), and the CPU takes every one of those misses of at most 8 columns. The
    // same outputs, bit for bit. A call whose CPU leg times out throws (its request ends with an error);
    // once the service thread has failed, prefill stays on the GPU. 0 <= num <= den, 1 <= den <= 32767.
    std::uint32_t prefill_cpu_share_num = 0;
    std::uint32_t prefill_cpu_share_den = 1;
    // FlashNextOptions::gdn_async_commit: a verification commit the row continues from folds its
    // GDN columns on the Program's fold stream (gdn_fold_on_side_stream); every later device use of
    // the state (the next round's front, any other entry point) waits for it. Requires MTP
    // (mtp_draft_tokens > 0). The same state, bit for bit. No device memory (a stream and two events).
    bool gdn_async_commit = false;
    // FlashNextOptions::hc_int8: the Program makes int8 copies of every mixer's hyper-connection
    // matrices at construction (execution::HcInt8Copies, ProgramDevicePlan::hc_int8_bytes) and the
    // forward's mixer calls of up to 16 columns read them (execution::ForwardOptions::hc_int8;
    // calls of more than 16 columns keep the BF16 weights). Changes the numerics; no workspace.
    bool hc_int8 = false;
    // FlashNextOptions::hc_fused_read: every decode-width routed-expert block's input is read by one
    // ops::hc_fused_read (execution::ForwardOptions::hc_fused_read: the previous inject, the int8
    // mixer, the router, routing, dispatch and the A8 cast of x). Requires hc_int8. Bitwise the
    // outputs of hc_int8 alone; the workspace terms add Forward::hc_fused_read_workspace_bytes.
    bool hc_fused_read = false;
    // FlashNextOptions::ngram_flag_wait: a single-row plain round and an MTP verification run as one
    // graph (ngram_rows_in_graph): the front, wait_flag_sequence (core/host_flag_wait.h) on the
    // Program's mapped flag, the rows' copy from the pinned block, the back. The host reads the rows
    // into the pinned block after launching the graph and raises the flag. Bitwise the two-part
    // round. A wait that times out (the host did not raise the flag within kWaitFlagTimeoutNs) fails
    // the round: its request ends with an error and nothing of it is committed. Memory:
    // plan_ngram_flag_wait.
    bool ngram_flag_wait = false;
    // FlashNextOptions::ggml_decode_mma: the forward's GGML dense Linears of at most 8 columns run
    // the int8 tensor-core decode route (execution::ForwardOptions::ggml_decode_mma). Changes the
    // numerics (summation order); no memory, no workspace.
    bool ggml_decode_mma = false;
    // FlashNextOptions::partial_ram_residency: the model's text banks were left in the artifact
    // (plan_load with partial). The Program reads `partial_vram_keys` (hottest first) into frames,
    // where they live only in VRAM, and every other expert into `partial_slots[l]` pinned RAM slots of
    // layer l (plan_partial_ram), from `artifact_path`. Promotions keep their RAM copy; evicting an
    // expert held only in VRAM writes it back first (ExpertResidency::PartialSetup). The frame pool
    // keeps its startup size (no VRAM monitor), the whole KV is mapped at startup (no frames taken
    // for it) and no frames are lent (prefill runs static calls, no expert stream). The same outputs,
    // bit for bit. Refused when the frames cannot hold every VRAM key.
    bool partial_ram_residency = false;
    std::vector<std::uint32_t> partial_slots;
    std::vector<std::uint32_t> partial_vram_keys;
    std::filesystem::path artifact_path;
    // The saved expert state the startup already read for the placement (its counts seed the LFRU);
    // empty: the Program reads `expert_state` itself.
    std::shared_ptr<const expert_cache::SavedState> partial_state;
};

// --ngram-flag-wait: whether a round runs as one graph that waits for its n-gram rows (true) or in
// two parts around the host's row read, as off. Only with the option and CUDA graphs on, and once
// the round's two parts have run eagerly (`eager_done`: their kernels' one-time setup is done, and
// a round never spins on the device while the host could still block on it), and not when launches
// are serialized (`launch_blocking`: CUDA_LAUNCH_BLOCKING=1 at startup, where every one-graph round
// would time out, since the host raises the flag only after the launch returns).
[[nodiscard]] bool ngram_rows_in_graph(const ProgramOptions& options, bool eager_done,
                                       bool launch_blocking = false) noexcept;

// The memory of --ngram-flag-wait: `device` is ProgramDevicePlan::ngram_flag_wait (the waits'
// 4-byte sequence counter), `pinned` the mapped flag and error word (one 64-byte line each). Both 0
// without ngram_flag_wait.
struct NgramFlagPlan {
    std::uint64_t device = 0;
    std::uint64_t pinned = 0;
};
[[nodiscard]] NgramFlagPlan plan_ngram_flag_wait(const ProgramOptions& options) noexcept;

// --gdn-async-commit: whether a verification commit runs its GDN replay fold on the fold stream
// (true) or on the device stream as before. Only a commit the row continues from (`terminal`
// false) goes to the fold stream: a terminal one (the end of the request, or of its budget) folds
// on the device stream, so the state is final before the Engine sees the round.
[[nodiscard]] bool gdn_fold_on_side_stream(const ProgramOptions& options, bool terminal) noexcept;

// ProgramDevicePlan::hc_int8_bytes of these options: execution::HcInt8Copies::bytes of the mixers
// the Program binds, from the configuration (each layer's attn_hc and mlp_hc with their injection
// rows, the final mixer without, and with MTP the drafter's two block mixers and final mixer: 97,
// or 100 with MTP, on Flash-Next); 0 without hc_int8.
[[nodiscard]] std::uint64_t plan_hc_int8(const ProgramOptions& options, const TextConfig& config);

// --hc-fused-read's share of the forward workspace for calls of up to `columns` columns
// (execution::Forward::hc_fused_read_workspace_bytes), part of ProgramDevicePlan::workspace (the
// static arena and the drafter step's term); 0 without hc_fused_read. The wide prefill arena takes
// none (its calls are wider than the fused read).
[[nodiscard]] std::size_t plan_hc_fused_read_workspace(const ProgramOptions& options, const TextConfig& config,
                                                       std::int32_t columns);

// The CPU miss service of --cpu-verify-misses and --prefill-cpu-share (one service, started for
// either) and its memory. Its channel buffers fit the larger use: max_jobs CPU-served experts per
// layer call and calls of up to max_columns columns. The verification channel (0 without
// cpu_verify_misses or MTP): verify_jobs experts per call (the service's default 4), calls of up to
// 1 + mtp_draft_tokens columns, M / pcie_divisor misses left on the GPU. The prefill channel (0
// without prefill_cpu_share): up to offloaded_moe::kMaxCpuJobs experts per call (every miss may go),
// calls of up to the wider of the static prefill chunk and a streamed call's 4,096 columns (the
// chunk alone with partial_ram_residency, which streams nothing; at most kMaxCpuCallColumns and the
// context; a wider call stays on the GPU), share_num / share_den of the
// misses. `device` is ProgramDevicePlan::cpu_misses (the service's 4-byte request counter); `pinned`
// its mapped channel (the request, the x columns and the outputs, the done and heartbeat words) and
// the Program's error word; `pageable` the worker team's buffers. All 0 when neither use is on.
struct CpuMissPlan {
    int workers = 0, max_jobs = 0, max_columns = 0;
    int verify_jobs = 0, verify_columns = 0, pcie_divisor = 0;
    int prefill_jobs = 0, prefill_columns = 0, share_num = 0, share_den = 0;
    std::uint64_t device   = 0;
    std::uint64_t pinned   = 0;
    std::uint64_t pageable = 0; // the worker team's buffers and the service's job list
};
[[nodiscard]] CpuMissPlan plan_cpu_misses(const ProgramOptions& options);

// ProgramDevicePlan::shared_expert_workspace of these options: the side-stream carve
// (execution::Forward::shared_expert_workspace_bytes) of the static workspace's widest call, a
// prefill chunk, a round over every lane or an MTP verification; 0 without shared_expert_stream.
[[nodiscard]] std::uint64_t plan_shared_expert_workspace(const ProgramOptions& options, const TextConfig& config);
// The widest call on the static workspace the term above is planned for.
[[nodiscard]] std::uint32_t shared_expert_columns(const ProgramOptions& options);
// The startup check of --shared-expert-stream: the loaded weights' shared expert Linears take `need`
// bytes of scratch at calls of up to `columns` columns (execution::Forward::
// max_shared_expert_scratch_bytes), against the `planned` bytes; refuses (std::invalid_argument
// naming the flag) when they do not fit.
void require_shared_expert_workspace(std::uint64_t need, std::uint64_t planned, std::uint32_t columns);
// ProgramDevicePlan::mtp_constrained_masks of these options over a tokenizer of `public_tokens`:
// 1 + mtp_draft_tokens masks of ceil(public_tokens / 32) words; 0 without mtp_constrained or MTP.
[[nodiscard]] std::uint64_t plan_mtp_constrained_masks(const ProgramOptions& options, std::uint32_t public_tokens);
// ProgramDevicePlan::mtp_draft_probs of these options: one FP32 word per draft (mtp_draft_tokens);
// 0 without mtp_joint_acceptance or MTP.
[[nodiscard]] std::uint64_t plan_mtp_draft_probs(const ProgramOptions& options);
// The GDN replay records of one W-column verification: every GDN layer, one record row. Their key
// records hold one key head per value head (the copy route), or with gdn_head_map the model's key
// heads read through the tiled map (spec.head_map Tiled).
[[nodiscard]] GdnReplayRecordSpec gdn_records_spec(const TextConfig& config, std::int32_t width, bool gdn_head_map);
// The GDN replay records' share of ProgramDevicePlan::speculation: the records of the widest
// verification (1 + mtp_draft_tokens columns, gdn_records_spec), 256-byte aligned; 0 without MTP.
[[nodiscard]] std::uint64_t plan_gdn_record_bytes(const ProgramOptions& options, const TextConfig& config);

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
    // --gdn-async-commit: the GDN replay folds every Program in this process has run on its fold
    // stream (a test's proof that the option's route ran).
    [[nodiscard]] static std::uint64_t process_async_gdn_folds() noexcept;
    // --ngram-flag-wait: the plain rounds, and the MTP verifications, every Program in this process
    // has run as one graph waiting for their n-gram rows (a test's proof that the option's route ran).
    [[nodiscard]] static std::uint64_t process_ngram_flag_rounds() noexcept;
    [[nodiscard]] static std::uint64_t process_ngram_flag_verifications() noexcept;

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
    [[nodiscard]] std::size_t pause_state_image_unplaced(SequenceHandle) const { return 0; }
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
    [[nodiscard]] bool prompt_fits_beside_residents(const RequestBasePlan&, const SourceCandidate&) const {
        return true; // one lane reserves its whole extent at admission
    }
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
    [[nodiscard]] bool start_pause(SequenceHandle, bool, runtime::ExecutionTiming* = nullptr,
                                   bool = false) { return false; }
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
    // `max_tokens` and `layer_slices` are accepted so the Engine can pass them to any model; this
    // model sizes its own calls (the router decides how wide each is) and ignores both.
    [[nodiscard]] PrefillProgress advance_prefill(SequenceHandle sequence,
                                                  runtime::ExecutionTiming* failed_timing = nullptr,
                                                  runtime::TokenMaskProvider* masks       = nullptr,
                                                  std::uint32_t max_tokens                = 0,
                                                  std::uint32_t layer_slices              = 0);
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
    // The context store's checkpoint images (program/checkpoint_image.h). Export and import require
    // no open context transaction or pending unit; an image binds to this Program's page and state
    // layout and to the caller's model binding, and import rejects any other with
    // std::invalid_argument. Import returns nullopt when the cache has no room for it now.
    [[nodiscard]] CheckpointImage export_checkpoints(std::span<const CheckpointHandle> points,
                                                     const std::optional<PreparedSessionKey>& session,
                                                     std::string_view binding) const;
    [[nodiscard]] std::size_t checkpoint_image_host_bytes(std::span<const std::uint8_t> image, std::string_view binding) const;
    [[nodiscard]] std::optional<ImportedCheckpoints> import_checkpoints(std::span<const std::uint8_t> image,
                                                                        std::string_view binding);

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
