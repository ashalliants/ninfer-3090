// Written for NInfer-3090. See program.h.

#include "models/dsv4/program/program.h"

#include "core/device.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <stdexcept>
#include <string>

namespace ninfer::models::dsv4 {

namespace {

// Accounting unit for the KV pages the Engine's ledgers count (the forward owns the real layout).
constexpr std::uint32_t kPageTokens = 256;

void cuda_ok(cudaError_t error, const char* what) {
    if (error != cudaSuccess) { throw std::runtime_error(std::string("dsv4 program: ") + what + ": " + cudaGetErrorString(error)); }
}

double seconds_between(std::chrono::steady_clock::time_point a, std::chrono::steady_clock::time_point b) {
    return std::chrono::duration<double>(b - a).count();
}

std::uint64_t elapsed_ns(std::chrono::steady_clock::time_point since) {
    return static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - since).count());
}

} // namespace

struct RequestBasePlan::Impl {
    runtime::RequestPlanSummary summary;
    qwen3_5::PreparedContextCache context_cache;
    ResolvedSamplingParameters sampling;
    std::vector<TokenId> prompt;
};

RequestBasePlan::RequestBasePlan(std::shared_ptr<Impl> impl) noexcept : impl_(std::move(impl)) {}
RequestBasePlan::RequestBasePlan(RequestBasePlan&&) noexcept            = default;
RequestBasePlan& RequestBasePlan::operator=(RequestBasePlan&&) noexcept = default;
RequestBasePlan::~RequestBasePlan()                                     = default;
const runtime::RequestPlanSummary& RequestBasePlan::summary() const noexcept { return impl_->summary; }
const qwen3_5::PreparedContextCache& RequestBasePlan::context_cache() const noexcept { return impl_->context_cache; }

Program::Program(Forward& forward, DeviceContext& device, ProgramOptions options)
    : forward_(forward), device_(device), options_(options) {
    if (options_.max_context == 0 || options_.prefill_chunk == 0 || options_.token_domain == 0) {
        throw std::invalid_argument("dsv4 program: max_context, prefill_chunk and token_domain must be nonzero");
    }
    host_logits_.resize(options_.token_domain);
    mask_words_.resize((options_.token_domain + 31U) / 32U);
}

Program::~Program() noexcept {
    // Nothing of the forward's work may still be reading the buffers the Engine frees after this.
    (void)cudaStreamSynchronize(device_.stream);
}

// ---------------------------------------------------------------------------------------- admission
RequestBasePlan Program::plan_request(PreparedPrompt&& prompt, const runtime::ResolvedExecutionOptions& options) {
    auto data = qwen3_5::PreparedPromptAccess::take(std::move(prompt));
    if (data.token_ids.empty()) { throw std::invalid_argument("prompt must contain tokens"); }
    if (data.token_ids.size() > options_.max_context) { throw std::invalid_argument("prompt exceeds configured context capacity"); }
    if (data.has_media()) { throw std::invalid_argument("DeepSeek-V4.1-Flash takes text only in this release"); }
    if (data.external_prefix_tokens != 0 || data.steering >= 0 || data.lora >= 0) {
        throw std::invalid_argument("DeepSeek-V4.1-Flash has no prompt grafts, steering vectors or LoRA adapters");
    }
    for (const TokenId id : data.token_ids) {
        if (id < 0 || static_cast<std::uint32_t>(id) >= options_.token_domain) {
            throw std::invalid_argument("prompt contains token outside the public token domain");
        }
    }
    auto base                              = std::make_shared<RequestBasePlan::Impl>();
    const auto n                           = static_cast<std::uint32_t>(data.token_ids.size());
    base->summary.prompt_tokens            = n;
    base->summary.reported_prompt_tokens   = n;
    base->summary.requested_output_tokens  = options.requested_output_tokens;
    const std::uint32_t capacity_output    = options_.max_context - n + 1U;
    base->summary.effective_output_tokens  = std::min(options.requested_output_tokens, capacity_output);
    base->summary.effective_limit_reason   = options.requested_output_tokens <= capacity_output ? FinishReason::OutputLimit
                                                                                                 : FinishReason::ContextCapacity;
    base->summary.publish_continuation     = false; // no context cache
    base->context_cache                    = data.context_cache;
    base->sampling                         = options.sampling;
    base->prompt                           = std::move(data.token_ids);
    return RequestBasePlan(std::move(base));
}

CheckpointHandle Program::install_external_checkpoint(const qwen3_5::PromptGraft&) {
    throw std::invalid_argument("DeepSeek-V4.1-Flash has no prompt grafts (they need the context cache)");
}

std::optional<SourceCandidate> Program::inspect_source(const RequestBasePlan& base, std::optional<CheckpointHandle> checkpoint, bool,
                                                       std::span<const CheckpointHandle>, std::span<const CheckpointHandle>) const {
    if (checkpoint) { return std::nullopt; } // no checkpoint exists
    SourceCandidate out;
    out.remaining_work = prefill_work(base, 0, base.summary().prompt_tokens);
    return out;
}

qwen3_5::PrefixShortlistKey Program::checkpoint_key(CheckpointHandle, std::uint32_t) const {
    throw std::logic_error("DeepSeek-V4.1-Flash has no context cache");
}
CheckpointSummary Program::checkpoint_summary(CheckpointHandle) const { throw std::logic_error("DeepSeek-V4.1-Flash has no context cache"); }
CheckpointMetadata Program::checkpoint_metadata(CheckpointHandle) const { throw std::logic_error("DeepSeek-V4.1-Flash has no context cache"); }

runtime::PrefillWork Program::prefill_work(const RequestBasePlan&, std::uint32_t begin, std::uint32_t end) const {
    return runtime::make_prefill_work(begin, end > begin ? end - begin : 0, 0, 0, options_.prefill_chunk);
}

BindingReservation Program::start_binding(const RequestBasePlan& plan, runtime::LaneId lane, const SourceCandidate& source,
                                          ResumeState* resume, ExecutionUnitKind, std::uint32_t) {
    BindingReservation out;
    if (resume != nullptr || source.checkpoint || source.reused_tokens != 0) {
        out.source_valid = false; // nothing is ever paused or cached here
        return out;
    }
    if (lane.value != 0 || lane_.phase != Phase::Free || binding_ || pending_ != 0) {
        throw std::logic_error("DeepSeek-V4.1-Flash: binding on a busy lane (it serves one request at a time)");
    }
    const auto& base = *plan.impl_;
    lane_            = Lane{};
    lane_.phase      = Phase::Binding;
    lane_.base       = plan.impl_;
    lane_.extent     = std::min(options_.max_context, base.summary.prompt_tokens + base.summary.effective_output_tokens);
    lane_.admitted   = Clock::now();
    ++epoch_;
    binding_     = true;
    out.reserved = true;
    return out;
}

ContextProgress Program::poll_context(runtime::CancellationFlagView cancellation) {
    ContextProgress out;
    out.kind = ContextOperationKind::Bind;
    if (!binding_) { return out; }
    binding_     = false;
    out.advanced = true;
    out.complete = true;
    if (cancellation.requested()) {
        release_lane();
        return out; // not published: the Engine completes the request as cancelled
    }
    const auto& base   = *lane_.base;
    lane_.history      = base.prompt;
    lane_.prompt_tokens = base.summary.prompt_tokens;
    lane_.state_tokens = 0;
    lane_.rng.seed(base.sampling.seed != 0 ? base.sampling.seed : entropy_seed());
    lane_.begin = {.prompt_tokens = lane_.prompt_tokens, .reused_prompt_tokens = 0, .prefix_reuse_path = PrefixReusePath::Root};
    lane_.phase = Phase::Prefill;
    out.published = true;
    out.sequence  = handle();
    return out;
}

runtime::ResourceReservation Program::reserve_units(std::span<const ExecutionUnit> units) {
    for (const auto& unit : units) {
        require_sequence(unit.sequence);
        if (unit.kind == ExecutionUnitKind::Replay) { throw std::logic_error("DeepSeek-V4.1-Flash does not pause, so it never replays"); }
    }
    return {.reserved = true}; // the whole extent belongs to the one lane
}

// ---------------------------------------------------------------------------------------- execution
SequenceHandle Program::handle() const noexcept {
    SequenceHandle out;
    out.owner_ = this;
    out.epoch_ = epoch_;
    return out;
}

bool Program::valid(SequenceHandle sequence) const noexcept {
    return sequence.owner_ == this && sequence.epoch_ == epoch_ && lane_.phase != Phase::Free;
}

void Program::require_sequence(SequenceHandle sequence) const {
    if (!valid(sequence)) { throw std::logic_error("DeepSeek-V4.1-Flash: unknown or released sequence"); }
}

void Program::release_lane() noexcept {
    lane_ = Lane{};
    ++epoch_; // handles to the released request go stale
    binding_ = false;
}

void Program::run_forward(std::size_t from, std::size_t to) {
    std::size_t at = from;
    while (at < to) {
        const std::size_t width = std::min<std::size_t>(to - at, options_.prefill_chunk);
        logits_ = &forward_.run(std::span<const std::int32_t>(lane_.history.data() + at, width), static_cast<std::int32_t>(at));
        at += width;
    }
    cuda_ok(cudaStreamSynchronize(device_.stream), "forward");
    lane_.state_tokens = static_cast<std::uint32_t>(to);
}

TokenId Program::sample_next(runtime::TokenMaskProvider* masks, bool& constraint_failed, bool& numerical_failure) {
    // The forward's logits are the last call's (run() returns the same device tensor every time).
    if (logits_ == nullptr) { throw std::logic_error("dsv4 program: sampling before any forward call"); }
    cuda_ok(cudaMemcpyAsync(host_logits_.data(), logits_->data, host_logits_.size() * sizeof(float), cudaMemcpyDeviceToHost, device_.stream),
            "logits download");
    cuda_ok(cudaStreamSynchronize(device_.stream), "logits sync");
    std::span<const std::uint32_t> mask;
    constraint_failed = false;
    numerical_failure = false;
    if (masks != nullptr && masks->constrained(0)) {
        const std::uint32_t dead = masks->fill(0, {}, std::span<std::uint32_t>(mask_words_));
        masks->uploaded(0, mask_words_.size() * 4);
        constraint_failed = (dead & 1U) != 0;
        mask              = mask_words_;
    }
    const TokenId token = sample_host(host_logits_, lane_.base->sampling, lane_.generated, mask, lane_.rng);
    if (token == kNonFiniteLogit) {
        numerical_failure = true; // the Engine ends the request with a numerical failure
        return 0;
    }
    if (token < 0) {
        // Nothing licensed. The grammar's own dead-end bit (already in constraint_failed) says the mask left no
        // legal token; otherwise a legal token existed and every licensed logit was -infinity: a numerical fault.
        if (!constraint_failed) { numerical_failure = true; }
        return 0;
    }
    return token;
}

PrefillProgress Program::advance_prefill(SequenceHandle sequence, runtime::ExecutionTiming* failed_timing,
                                         runtime::TokenMaskProvider* masks, std::uint32_t, std::uint32_t) {
    const auto start = Clock::now();
    require_sequence(sequence);
    if (lane_.phase != Phase::Prefill || pending_ != 0) { throw std::logic_error("DeepSeek-V4.1-Flash: prefill on a lane that is not prefilling"); }
    PrefillProgress out;
    out.summary = lane_.begin;
    try {
        const std::uint32_t remaining = lane_.prompt_tokens - lane_.state_tokens;
        const std::uint32_t width     = std::min(remaining, options_.prefill_chunk);
        out.processed_prompt_tokens   = width;
        out.complete                  = width == remaining;
        const auto wait               = Clock::now();
        run_forward(lane_.state_tokens, lane_.state_tokens + width);
        out.timing.device_wait_ns = elapsed_ns(wait);
        if (out.complete) {
            bool failed = false, numerical = false;
            pending_token_    = sample_next(masks, failed, numerical);
            pending_          = ++transaction_;
            lane_.phase       = Phase::Pending;
            PendingBatch batch;
            batch.owner_              = this;
            batch.transaction_        = pending_;
            batch.token_              = pending_token_;
            batch.constraint_failed_  = failed;
            batch.token_domain_failed_ = numerical;
            batch.timing_             = out.timing;
            out.pending.emplace(std::move(batch));
        }
        out.timing.submit_host_ns = elapsed_ns(start) - std::min(elapsed_ns(start), out.timing.device_wait_ns);
        lane_.prefill_seconds += seconds_between(start, Clock::now());
        return out;
    } catch (...) {
        if (failed_timing != nullptr) { failed_timing->submit_host_ns += elapsed_ns(start); }
        release_lane();
        throw;
    }
}

ReplayProgress Program::advance_replay(SequenceHandle, runtime::ExecutionTiming*) {
    throw std::logic_error("DeepSeek-V4.1-Flash does not pause, so it never replays");
}

PendingBatch Program::decode(std::span<const SequenceHandle> sequences, std::span<const runtime::RoundBudget> budgets,
                             runtime::ExecutionTiming* failed_timing, runtime::TokenMaskProvider* masks) {
    const auto start = Clock::now();
    if (sequences.size() != 1 || budgets.size() != 1) { throw std::invalid_argument("DeepSeek-V4.1-Flash: a decode round has one row (one lane)"); }
    require_sequence(sequences[0]);
    if (lane_.phase != Phase::Decode || pending_ != 0 || lane_.history.size() != lane_.state_tokens + 1ULL) {
        throw std::logic_error("DeepSeek-V4.1-Flash: decode row is not at a committed frontier");
    }
    try {
        runtime::ExecutionTiming timing;
        const auto wait = Clock::now();
        run_forward(lane_.state_tokens, lane_.state_tokens + 1); // the last committed token is the input
        timing.device_wait_ns = elapsed_ns(wait);
        bool failed = false, numerical = false;
        pending_token_        = sample_next(masks, failed, numerical);
        pending_              = ++transaction_;
        lane_.phase           = Phase::Pending;
        timing.submit_host_ns = elapsed_ns(start) - std::min(elapsed_ns(start), timing.device_wait_ns);
        lane_.decode_seconds += seconds_between(start, Clock::now());
        PendingBatch batch;
        batch.owner_             = this;
        batch.transaction_       = pending_;
        batch.token_             = pending_token_;
        batch.constraint_failed_ = failed;
        batch.token_domain_failed_ = numerical;
        batch.timing_            = timing;
        return batch;
    } catch (...) {
        if (failed_timing != nullptr) { failed_timing->submit_host_ns += elapsed_ns(start); }
        release_lane();
        throw;
    }
}

runtime::ExecutionTiming Program::append_forced_tokens(std::span<const SequenceHandle> sequences, std::span<const TokenId> tokens,
                                                       std::uint32_t stride, std::span<const std::optional<std::uint32_t>>,
                                                       runtime::ExecutionTiming* failed_timing) {
    const auto start = Clock::now();
    if (sequences.size() != 1 || stride == 0 || tokens.size() != stride) {
        throw std::invalid_argument("DeepSeek-V4.1-Flash: forced tokens do not match their row");
    }
    require_sequence(sequences[0]);
    if (lane_.phase != Phase::Decode || pending_ != 0 || lane_.history.size() != lane_.state_tokens + 1ULL ||
        lane_.state_tokens + stride > options_.max_context) {
        throw std::logic_error("DeepSeek-V4.1-Flash: forced tokens on a lane that is not at a committed frontier");
    }
    for (const TokenId token : tokens) {
        if (token < 0 || static_cast<std::uint32_t>(token) >= options_.token_domain) { throw std::invalid_argument("forced token outside the domain"); }
    }
    try {
        // The model reads the pending input token and every forced token but the last, which becomes the
        // next decode input.
        lane_.history.insert(lane_.history.end(), tokens.begin(), tokens.end());
        for (const TokenId token : tokens) { ++lane_.generated[token]; }
        run_forward(lane_.state_tokens, lane_.state_tokens + stride);
        runtime::ExecutionTiming timing;
        timing.submit_host_ns = elapsed_ns(start);
        lane_.decode_seconds += seconds_between(start, Clock::now());
        return timing;
    } catch (...) {
        if (failed_timing != nullptr) { failed_timing->submit_host_ns += elapsed_ns(start); }
        release_lane();
        throw;
    }
}

GenerationTimings Program::timings() const {
    GenerationTimings out;
    out.prefill_seconds     = lane_.prefill_seconds;
    out.decode_seconds      = lane_.decode_seconds;
    out.first_token_seconds = lane_.first_token_seconds;
    return out;
}

CommitResult Program::commit(PendingBatch&& pending, std::span<const runtime::CommitDecision> decisions, runtime::CommitObservation,
                             runtime::ExecutionTiming*) {
    const auto start = Clock::now();
    if (pending.owner_ != this || pending.transaction_ != pending_ || pending_ == 0) {
        throw std::logic_error("DeepSeek-V4.1-Flash: commit of a foreign or stale pending batch");
    }
    if (decisions.size() != 1) { throw std::logic_error("DeepSeek-V4.1-Flash: commit decisions are not row aligned"); }
    const auto& d                = decisions[0];
    const bool constraint_failed = pending.constraint_failed_ || pending.token_domain_failed_;
    pending.owner_               = nullptr;
    pending_                     = 0;
    CommitResult out;
    out.row_count = 1;
    if (d.cancelled || d.failed) {
        if ((d.cancelled && d.failed) || (!d.cancelled && d.failed != constraint_failed) || d.accepted_tokens != 0 || !d.terminal) {
            release_lane();
            throw std::logic_error("DeepSeek-V4.1-Flash: pending decision is invalid");
        }
        out.rows[0].timings     = timings();
        out.rows[0].disposition = d.failed ? runtime::CommitDisposition::FailedReleased : runtime::CommitDisposition::CancelledReleased;
        release_lane();
    } else {
        if (d.accepted_tokens != 1) {
            release_lane();
            throw std::logic_error("DeepSeek-V4.1-Flash: a row commits exactly one token");
        }
        if (lane_.history.size() == lane_.prompt_tokens) { lane_.first_token_seconds = seconds_between(lane_.admitted, Clock::now()); }
        lane_.history.push_back(pending_token_);
        ++lane_.generated[pending_token_];
        lane_.phase             = d.terminal ? Phase::Finishable : Phase::Decode;
        out.rows[0].disposition = d.terminal ? runtime::CommitDisposition::Finishable : runtime::CommitDisposition::Active;
        out.rows[0].timings     = timings();
    }
    out.timing.post_host_ns = elapsed_ns(start);
    return out;
}

DiscardResult Program::abort_pending(PendingBatch&& pending) noexcept {
    DiscardResult out;
    if (pending.owner_ != this || pending.transaction_ != pending_) { return out; }
    pending.owner_ = nullptr;
    pending_       = 0;
    out.row_count  = 1;
    release_lane();
    out.status = runtime::ConsumeStatus::Consumed;
    return out;
}

FinishResult Program::finish(SequenceHandle sequence) noexcept {
    FinishResult out;
    if (!valid(sequence) || lane_.phase != Phase::Finishable || pending_ != 0) { return out; }
    out.timings = timings();
    release_lane();
    out.status = runtime::ConsumeStatus::Consumed;
    return out;
}

AbortResult Program::abort(SequenceHandle sequence) noexcept {
    AbortResult out;
    if (!valid(sequence) || pending_ != 0) { return out; }
    out.timings = timings();
    (void)cudaStreamSynchronize(device_.stream);
    release_lane();
    out.status = runtime::ConsumeStatus::Consumed;
    return out;
}

void Program::fail_all_cleanup() noexcept {
    (void)cudaStreamSynchronize(device_.stream);
    pending_ = 0;
    release_lane();
}

PhysicalUsageSnapshot Program::physical_usage() const noexcept {
    PhysicalUsageSnapshot out;
    out.capacity.state_slots   = 1;
    out.capacity.main_kv_pages = (options_.max_context + kPageTokens - 1) / kPageTokens;
    if (lane_.phase != Phase::Free) {
        out.occupied.state_slots   = 1;
        out.occupied.main_kv_pages = (std::max(lane_.extent, 1U) + kPageTokens - 1) / kPageTokens;
    }
    return out;
}

MemorySummary Program::memory_summary() const noexcept {
    MemorySummary out;
    out.device                      = device_.device;
    out.max_context                 = options_.max_context;
    out.kv_capacity                 = options_.max_context;
    out.kv_capacity_page_groups     = (options_.max_context + kPageTokens - 1) / kPageTokens;
    out.kv_capacity_max_page_groups = out.kv_capacity_page_groups;
    out.kv_cache                    = KvCacheStorage::BFloat16;
    out.kv_payload_bytes            = forward_.device_bytes();
    return out;
}

CheckpointImage Program::export_checkpoints(std::span<const CheckpointHandle>, const std::optional<PreparedSessionKey>&, std::string_view,
                                            std::vector<std::uint8_t>) const {
    throw std::logic_error("DeepSeek-V4.1-Flash has no context store");
}
std::size_t Program::checkpoint_image_host_bytes(std::span<const std::uint8_t>, std::string_view) const {
    throw std::invalid_argument("DeepSeek-V4.1-Flash has no context store");
}
std::optional<ImportedCheckpoints> Program::import_checkpoints(std::span<const std::uint8_t>, std::string_view) {
    throw std::invalid_argument("DeepSeek-V4.1-Flash has no context store");
}

} // namespace ninfer::models::dsv4
