#pragma once

// Small fixed-capacity request execution for every backend.

#include "core/device.h"
#include "core/device_snapshot.h"
#include "core/nvtx.h"
#include "ninfer/types.h"
#include "runtime/contract/execution.h"
#include "runtime/contract/resources.h"
#include "runtime/engine/lossy_prefill.h"
#include "runtime/engine/prefill_pacing.h"
#include "runtime/engine/prefill_yield.h"
#include "runtime/engine/request_record.h"
#include "runtime/engine/context_cache/resource_manager.h"
#include "runtime/engine/context_store/context_store.h"
#include "runtime/engine/pause_policy.h"
#include "runtime/engine/scheduler.h"
#include "runtime/engine/effective_thinking_budget.h"
#include "runtime/engine/generation_budget.h"
#include "runtime/engine/worker_fault.h"
#include "runtime/engine/worker_recovery.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <exception>
#include <functional>
#include <future>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

namespace ninfer::runtime {

template <class Instance>
class EngineCore {

public:
    using ModelContract      = typename Instance::ModelContract;
    // The device the worker thread binds and synchronizes. Instances name it so a test can supply
    // one that needs no GPU; production instances use DeviceContext.
    using Device             = typename Instance::Device;
    using ExecutionUnit      = typename ModelContract::ExecutionUnit;
    using UnitKind           = typename ModelContract::ExecutionUnitKind;
    using Checkpoint         = typename ModelContract::CheckpointHandle;
    using SequenceHandle     = typename ModelContract::SequenceHandle;
    using PendingBatch       = typename ModelContract::PendingBatch;
    using PreparedPrompt     = typename ModelContract::PreparedPrompt;
    using PublishedOutput    = typename ModelContract::PublishedOutput;
    using Request            = RequestRecord<ModelContract>;
    using Scheduling         = Scheduler<Request>;
    using RoundMembership    = typename Scheduling::RoundMembership;
    using ControlMembership  = typename Scheduling::ControlMembership;
    using ResourceManagement = ResourceManager<ModelContract>;
    using Clock              = std::chrono::steady_clock;

    // Model suspend is implemented by the instance that owns suspendable device memory (the
    // Qwen3.5 ModelInstance). For any other, enabling it is refused at construction.
    static constexpr bool kInstanceSuspendable =
        requires(Instance& instance, Device& device, DeviceSnapshot& snapshot) {
            instance.suspend(device, snapshot, &snapshot);
        };

    // Lossy long-prompt prefill (lossy_prefill.h) needs a Program that runs it (the Qwen3.5 one);
    // for any other, the option is refused at startup and the Engine never takes the path.
    using ProgramType = typename ModelContract::Program;
    using BasePlan    = typename ModelContract::RequestBasePlan;
    static constexpr bool kLossyCapable =
        requires(ProgramType& program, const ProgramType& view, const BasePlan& plan,
                 SequenceHandle sequence, typename ModelContract::PrefillProgress& progress) {
            plan.lossy();
            program.enable_lossy(sequence);
            program.abort_lossy_scout(sequence);
            view.lossy_scout_share();
            progress.scouted_tokens;
            progress.compacted_prompt_tokens;
            progress.compacted_plan;
            progress.lossy;
        };

    // A frontier of the plan's prompt in the client's prompt tokens: what usage and prompt
    // progress report. Only a Program that rewrites prompts (lossy prefill) can make them differ.
    [[nodiscard]] static std::uint32_t reported_frontier(const BasePlan& plan,
                                                         std::uint32_t frontier) noexcept {
        if constexpr (requires { plan.reported_frontier(frontier); }) {
            return plan.reported_frontier(frontier);
        } else {
            return frontier;
        }
    }

    class RoundMasks final : public TokenMaskProvider {
    public:
        std::array<decltype(&std::declval<Request&>().output), kMaximumConcurrency> outputs{};

        bool constrained(std::size_t row) const noexcept override {
            return outputs[row] && outputs[row]->constrained();
        }

        std::uint32_t fill(std::size_t row, std::span<const TokenId> drafts,
                           std::span<std::uint32_t> words) override {
            return outputs[row]->grammar_masks(drafts, words);
        }

        void uploaded(std::size_t row, std::size_t bytes) noexcept override {
            outputs[row]->constraint_uploaded(bytes);
        }
    };

    EngineCore(Instance& instance, Device& device, const EngineOptions& options,
               ContextMachineCostModel context_cost)
        : instance_(instance), device_(device), max_context_(options.max_context),
          max_concurrency_(options.max_concurrency),
          admission_fit_check_(options.admission_fit_check),
          max_outstanding_(static_cast<std::size_t>(options.max_concurrency) +
                           options.max_pending_requests),
          pending_timeout_(std::chrono::milliseconds(options.pending_timeout_ms)),
          prefill_chunk_(std::min(options.prefill_chunk, options.max_context)),
          pacing_(PrefillPacingOptions{
              .stall_budget_ms      = options.prefill_schedule.stall_budget_ms,
              .decode_share_percent = options.prefill_schedule.decode_share_percent}),
          yield_min_remaining_ns_(
              static_cast<std::uint64_t>(options.prefill_schedule.yield_remaining_ms) * 1'000'000ULL),
          layer_slices_(options.prefill_schedule.layer_slices),
          lossy_options_(options.lossy_prefill),
          progress_anchor_tokens_(options.progress_anchors_shared ? options.progress_anchor_tokens : 0),
          fault_listener_(options.fault_listener),
          resources_(options.context_cache.enabled, std::move(context_cost)) {
        if (max_concurrency_ == 0 || max_concurrency_ > kMaximumConcurrency ||
            options.max_pending_requests == 0 || pending_timeout_.count() <= 0) {
            throw std::invalid_argument("Engine core bounds are invalid");
        }
        suspend_options_ = options.suspend;
        if (options.suspend.enabled) {
            if constexpr (kInstanceSuspendable) {
                if (!instance_.suspendable()) {
                    throw std::logic_error(
                        "model suspend is enabled but the instance is not suspendable");
                }
                const bool pinned = options.suspend.snapshot_memory == SuspendSnapshotMemory::Pinned;
                const auto memory = pinned ? DeviceSnapshot::Memory::Pinned
                                           : DeviceSnapshot::Memory::Pageable;
                state_snapshot_.emplace(
                    memory, pinned ? instance_.program->persistent_capacity_bytes() : 0);
                if (options.suspend.weights == SuspendWeightSource::Host) {
                    weight_snapshot_.emplace(memory,
                                             pinned ? instance_.weight_host_copy_bytes() : 0);
                }
                auto_resume_.store(options.suspend.auto_resume, std::memory_order_release);
            } else {
                throw std::invalid_argument("this model does not support model suspend");
            }
        }
        publish_residency();
        restore_external_sources();
        paused_.reserve(max_outstanding_);
        std::promise<void> startup;
        std::future<void> started = startup.get_future();
        worker_                   = std::thread([this, startup = std::move(startup)]() mutable {
            try {
                device_.bind_to_current_thread();
                // Nothing has been admitted yet: this is what a verified recovery returns to.
                quiescent_usage_ = instance_.program->physical_usage();
                publish_runtime_stats();
                startup.set_value();
            } catch (...) {
                startup.set_exception(std::current_exception());
                return;
            }
            worker_loop();
        });
        try {
            started.get();
        } catch (...) {
            if (worker_.joinable()) { worker_.join(); }
            throw;
        }
    }

    ~EngineCore() noexcept { stop_worker(); }

    // Stops the worker, failing whatever is still waiting, and leaves the Engine alive so that a
    // thread woken by that can finish its callbacks (Submission::wait ends in release_consumer)
    // before the Engine is destroyed. Idempotent; the destructor calls it.
    void stop_worker() noexcept {
        {
            std::lock_guard lock(queue_mutex_);
            stopping_ = true;
        }
        queue_cv_.notify_all();
        if (worker_.joinable()) { worker_.join(); }
        stop_hydration_reader();
    }

    EngineCore(const EngineCore&)            = delete;
    EngineCore& operator=(const EngineCore&) = delete;

    class Submission {
    public:
        Submission() noexcept = default;

        ~Submission() { reset(); }

        Submission(Submission&& other) noexcept
            : owner_(std::exchange(other.owner_, nullptr)), request_(std::move(other.request_)),
              effective_thinking_budget_(other.effective_thinking_budget_) {}

        Submission& operator=(Submission&& other) noexcept {
            if (this != &other) {
                reset();
                owner_                     = std::exchange(other.owner_, nullptr);
                request_                   = std::move(other.request_);
                effective_thinking_budget_ = other.effective_thinking_budget_;
            }
            return *this;
        }

        Submission(const Submission&)            = delete;
        Submission& operator=(const Submission&) = delete;

        GenerationResult wait(OutputSink* sink, const CancellationView& cancellation) {
            if (owner_ == nullptr || request_ == nullptr) {
                throw std::logic_error("concurrent submission is empty");
            }
            const bool streaming = request_->consumer_mode == OutputConsumerMode::Streaming;
            if (streaming != (sink != nullptr)) {
                throw std::invalid_argument(
                    "GenerationHandle wait sink does not match its submitted consumer mode");
            }
            EngineCore* owner = std::exchange(owner_, nullptr);
            return owner->wait_for_request(std::exchange(request_, nullptr), sink, cancellation);
        }

        // Fixed when the request was submitted; never reads the session the worker mutates.
        [[nodiscard]] std::optional<std::uint32_t> effective_thinking_budget() const noexcept {
            return effective_thinking_budget_;
        }

    private:
        Submission(EngineCore& owner, std::shared_ptr<Request> request,
                   std::optional<std::uint32_t> effective_thinking_budget) noexcept
            : owner_(&owner), request_(std::move(request)),
              effective_thinking_budget_(effective_thinking_budget) {}

        void reset() noexcept {
            if (owner_ != nullptr && request_ != nullptr) {
                owner_->abandon_request(std::move(request_));
            }
            owner_ = nullptr;
        }

        EngineCore* owner_ = nullptr;
        std::shared_ptr<Request> request_;
        std::optional<std::uint32_t> effective_thinking_budget_;

        friend class EngineCore;
    };

    Submission submit(PreparedPrompt prompt, PromptSummary prompt_summary, double prepare_seconds,
                      ResolvedRequestOptions options, OutputConsumerMode consumer_mode,
                      GenerationObservationOptions observation,
                      Clock::time_point pending_deadline = {},
                      std::optional<compaction::SummaryRecord> summary = std::nullopt) {
        const Clock::time_point submitted = Clock::now();
        if (pending_deadline == Clock::time_point{}) {
            pending_deadline = submitted + pending_timeout_;
        }
        if (submitted >= pending_deadline) {
            throw RequestError(RequestErrorKind::QueueTimeout,
                               "inference request expired before submission");
        }

        std::uint64_t request_id        = 0;
        std::uint64_t publication_order = 0;
        {
            std::lock_guard lock(queue_mutex_);
            if (stopping_ || failed_) {
                throw RequestError(RequestErrorKind::Unavailable,
                                   "inference engine is unavailable");
            }
            if (suspended_.load(std::memory_order_acquire) &&
                !auto_resume_.load(std::memory_order_acquire)) {
                throw RequestError(RequestErrorKind::Unavailable, "model is suspended");
            }
            if (outstanding_ >= max_outstanding_) {
                throw RequestError(RequestErrorKind::Overloaded, "inference request queue is full");
            }
            if (next_request_id_ == 0 || next_publication_order_ == 0) {
                throw std::overflow_error("request identity space exhausted");
            }
            ++outstanding_;
            request_id        = next_request_id_++;
            publication_order = next_publication_order_++;
        }

        std::shared_ptr<Request> request;
        std::optional<std::uint32_t> effective_budget;
        try {
            // A thinking cap the context cannot honour together with its early-close suffix is
            // lowered to what fits (or loses early close) instead of failing the request.
            apply_effective_thinking_budget(
                options.execution.thinking,
                effective_output_capacity(options.execution.requested_output_tokens, max_context_,
                                          prompt_summary.prompt_tokens),
                instance_.frontend.thinking_control_token_count());
            const auto constraint_started = observation.phase_timings ? Clock::now() : submitted;
            auto output                   = instance_.frontend.make_output_session(
                prompt, options.stop, options.output, options.execution.thinking,
                options.constraint, options.tool_choice);
            if (Clock::now() >= pending_deadline) {
                throw RequestError(RequestErrorKind::QueueTimeout,
                                   "inference request expired during grammar preparation");
            }
            const auto ready = Clock::now();
            output.observe_constraint(
                observation.phase_timings,
                std::chrono::duration<double>(ready - constraint_started).count());
            prepare_seconds += std::chrono::duration<double>(ready - submitted).count();
            request = std::make_shared<Request>(request_id, publication_order, std::move(prompt),
                                                std::move(output), prompt_summary, prepare_seconds,
                                                std::move(options), consumer_mode,
                                                std::move(observation), pending_deadline, ready);
            effective_budget         = request->output.thinking_stats().effective_budget;
            request->summary         = std::move(summary);
        } catch (...) {
            release_reserved_capacity();
            throw;
        }

        {
            std::lock_guard lock(queue_mutex_);
            if (stopping_ || failed_) {
                --outstanding_;
                throw RequestError(RequestErrorKind::Unavailable,
                                   "inference engine is unavailable");
            }
            pending_.push_back(request);
        }
        request_admission_check();
        queue_cv_.notify_one();
        return Submission(*this, std::move(request), effective_budget);
    }

    [[nodiscard]] MemorySummary memory_summary() const {
        std::scoped_lock lock(execution_mutex_);
        MemorySummary out                      = instance_.program->memory_summary();
        const KvCapacityResolution& resolution = instance_.kv_capacity_resolution;
        out.kv_capacity_mode                   = resolution.mode;
        out.kv_capacity_page_groups            = resolution.main_page_groups;
        out.kv_capacity_max_page_groups        = resolution.maximum_main_page_groups;
        out.minimum_runtime_reservation_bytes  = resolution.minimum_runtime_reservation_bytes;
        out.kv_capacity_increment_bytes        = resolution.bytes_per_additional_main_page_group;
        out.runtime_reservation_bytes          = resolution.runtime_reservation_bytes;
        out.available_after_weights_bytes      = resolution.available_after_weights_bytes;
        out.available_after_startup_bytes      = resolution.available_after_startup_bytes;
        out.kv_capacity_headroom_bytes         = resolution.automatic_headroom_bytes;
        out.planned_slack_bytes                = resolution.planned_slack_bytes;
        out.stage_runtime_reservation_bytes    = resolution.extra_rank_reservation_bytes;
        out.kv_capacity_binding_stage          = resolution.binding_rank;
        out.requested_max_context              = resolution.requested_context;
        return out;
    }

    [[nodiscard]] RuntimeStats runtime_stats() const {
        RuntimeStats stats;
        {
            std::lock_guard lock(stats_mutex_);
            stats = published_stats_;
        }
        stats.lossy_prefill_summary_probe_timeouts = probe_timeouts_.load(std::memory_order_relaxed);
        return stats;
    }

    // The summary strategy's reuse probe, from a caller thread (Engine::prepare): Warm when a
    // private (conversation) record, or a shared checkpoint deeper than `exact_end`, this prompt
    // starts with would be reused, from the cache or from
    // the context store (deeper than `exact_end`, the product prompt), Cold otherwise. `prompt` is
    // a copy the probe plans and drops. The probe is posted to the worker, which answers it between
    // cycles (serve_reuse_probes); an answer not given within `wait` is Timeout, counted in
    // lossy_prefill_summary_probe_timeouts and treated as warm by the caller. A stopping or failed
    // Engine answers Warm at once.
    [[nodiscard]] LossySummaryProbe has_private_reuse(PreparedPrompt prompt,
                                                      const ResolvedExecutionOptions& execution,
                                                      std::uint32_t exact_end,
                                                      std::chrono::milliseconds wait) {
        auto probe       = std::make_shared<ReuseProbe>();
        probe->prompt    = std::move(prompt);
        probe->execution = execution;
        probe->exact_end = exact_end;
        {
            std::lock_guard lock(queue_mutex_);
            if (stopping_ || failed_) { return LossySummaryProbe::Warm; }
            probes_.push_back(probe);
            probes_pending_.store(true, std::memory_order_release);
        }
        queue_cv_.notify_all();
        std::unique_lock lock(probe->mutex);
        if (!probe->cv.wait_for(lock, wait, [&] { return probe->warm.has_value(); })) {
            probe->abandoned = true;
            probe_timeouts_.fetch_add(1, std::memory_order_relaxed);
            return LossySummaryProbe::Timeout;
        }
        return *probe->warm ? LossySummaryProbe::Warm : LossySummaryProbe::Cold;
    }

    // An Engine-wide failure: every queued request failed, every new one is refused, and the
    // Engine never recovers. A stop or a held suspend is not a failure.
    [[nodiscard]] bool has_failed() const {
        std::lock_guard lock(queue_mutex_);
        return failed_;
    }

    [[nodiscard]] bool is_available() const {
        std::lock_guard lock(queue_mutex_);
        // A suspended model that resumes on demand still serves; one held suspended does not.
        const bool held_suspended = suspended_.load(std::memory_order_acquire) &&
                                    !auto_resume_.load(std::memory_order_acquire);
        return !stopping_ && !failed_ && !held_suspended;
    }

    // Model residency (EngineOptions::suspend). A suspend runs between units while the Engine is
    // idle -- nothing queued, admitted, paused or running, no context transaction -- and refuses
    // (Overloaded) otherwise; it never cancels work. While suspended nothing executes: a queued
    // generation request resumes the model first when auto_resume is set and fails as Unavailable
    // when it is not. A failed resume leaves the model suspended with its state intact, fails the
    // requests that were waiting on it and is retried by the next request or an explicit resume.
    // Adapted from NInfer-all 0dcb1da9 src/runtime/engine/engine_core.h (Apache-2.0).
    ResidencyStatus suspend(std::optional<bool> auto_resume) {
        require_suspend_enabled();
        std::scoped_lock lock(execution_mutex_);
        device_.bind_to_current_thread();
        if (suspended_.load(std::memory_order_acquire)) {
            auto_resume_.store(auto_resume.value_or(suspend_options_.auto_resume),
                               std::memory_order_release);
            publish_residency();
            return residency();
        }
        if constexpr (kInstanceSuspendable) {
            {
                std::lock_guard queue_lock(queue_mutex_);
                // outstanding_ counts a request from the moment submit() reserves it, so one
                // still preparing its prompt (not yet in pending_) also keeps the engine busy.
                bool busy = outstanding_ != 0 || !pending_.empty() || materializing_.has_value() ||
                            !paused_.empty() || context_owner_ != nullptr ||
                            !sibling_waits_.empty() || admission_decision_.has_value() ||
                            instance_.program->has_context_transaction();
                for (std::uint32_t lane = 0; lane < max_concurrency_; ++lane) {
                    busy = busy || slots_[lane] != nullptr;
                }
                if (busy) {
                    throw RequestError(RequestErrorKind::Overloaded,
                                       "model suspend requires an idle engine");
                }
                // Set before the queue lock is released: a request submitted from here on waits
                // for the resume (or is refused) instead of racing the release of the memory it
                // would use.
                auto_resume_.store(auto_resume.value_or(suspend_options_.auto_resume),
                                   std::memory_order_release);
                suspended_.store(true, std::memory_order_release);
            }
            try {
                last_suspend_ = instance_.suspend(device_, *state_snapshot_,
                                                  weight_snapshot_ ? &*weight_snapshot_ : nullptr);
            } catch (const ResidencyRollbackError& error) {
                // Released memory could not be put back: stay suspended (state is held in the
                // snapshots) so the next request or an explicit resume retries the restore.
                last_residency_error_ = error.what();
                publish_residency();
                throw;
            } catch (...) {
                suspended_.store(false, std::memory_order_release);
                queue_cv_.notify_all();
                publish_residency();
                throw;
            }
            ++suspend_count_;
            publish_residency();
            publish_runtime_stats();
        }
        return residency();
    }

    ResidencyStatus resume() {
        require_suspend_enabled();
        std::scoped_lock lock(execution_mutex_);
        device_.bind_to_current_thread();
        resume_locked();
        return residency();
    }

    [[nodiscard]] ResidencyStatus residency() const {
        std::lock_guard lock(stats_mutex_);
        return published_residency_;
    }

    void reset_memory_peaks() noexcept {
        try {
            std::scoped_lock lock(execution_mutex_);
            instance_.program->reset_memory_peaks();
        } catch (...) {}
    }

    using CheckpointImage = typename ModelContract::CheckpointImage;

    // The durable context store (owned by the Engine, which outlives this core). `sink` queues one
    // image for the Engine's writer and must not block; it returns whether the image was accepted.
    // `ready` reports an idle writer; `queued` whether an image id is still waiting for (or in) the
    // writer; `failures` counts queued writes that later failed, and `failed_writes` hands over the
    // ids of those not yet taken: each such continuation the store does not hold is written again.
    // Failed writes the writer reports: their ids, and whether some were lost (then every
    // continuation the store does not hold counts as failed).
    struct FailedStoreWrites {
        std::vector<std::string> ids;
        bool lost = false;
    };

    struct ContextStoreHooks {
        ContextStore* store = nullptr;
        std::string binding;
        std::function<bool(CheckpointImage&&)> sink;
        // Write-on-evict: queues the image of a continuation the cache is about to lose. May take
        // more than `sink` (it is the last chance to keep the conversation); must not block.
        std::function<bool(CheckpointImage&&)> evict_sink;
        // Queues the image of one public shared prefix (a system or tools prompt); stored as a
        // shared image. Must not block.
        std::function<bool(CheckpointImage&&)> shared_sink;
        std::function<bool()> ready;
        std::function<bool(const std::string&)> queued;
        std::function<std::uint64_t()> failures;
        std::function<FailedStoreWrites()> failed_writes;
        // A buffer for the next image's bytes: the writer hands back the last written image's,
        // whose pages are already mapped, so copying an image out does not fault them in again.
        std::function<std::vector<std::uint8_t>()> export_buffer;
        // A retained continuation unused this long is written in the background; zero disables it.
        std::chrono::milliseconds idle_persist{0};
        // The longest a request waits for its stored session to be read back.
        std::chrono::milliseconds hydration_budget{0};
        // The store keeps a copy in a bucket: a request that finds nothing has the listing
        // refreshed once before it prefills, and a session only the bucket holds is downloaded
        // when the download fits in the request's wait.
        bool remote = false;
    };

    void set_context_store(ContextStoreHooks hooks) {
        std::scoped_lock lock(execution_mutex_);
        store_                = std::move(hooks);
        store_removals_seen_  = store_removal_count();
        store_failures_seen_  = store_.failures ? store_.failures() : 0;
        if (store_.store && !hydration_reader_.joinable()) {
            hydration_reader_ = std::thread([this] { hydration_reader_loop(hydration_queue_); });
        }
        if (store_.store && store_.remote && !lookup_reader_.joinable()) {
            lookup_reader_ = std::thread([this] { hydration_reader_loop(lookup_queue_); });
        }
        typename ResourceManagement::Durability durability;
        if (store_.store) {
            durability.stored_bytes =
                [this](ContinuationOwnerToken token) -> std::optional<std::uint64_t> {
                const auto found = owner_marks_.find(token);
                if (found == owner_marks_.end() ||
                    !holds(found->second, resources_.points(token))) {
                    return std::nullopt;
                }
                return found->second.stored_bytes;
            };
        }
        if (store_.store) {
            durability.shared_stored_bytes =
                [this](Checkpoint handle) -> std::optional<std::uint64_t> {
                for (const auto& [stored, bytes] : shared_stored_) {
                    if (stored == handle) { return bytes; }
                }
                return std::nullopt;
            };
        }
        if (store_.evict_sink) {
            durability.releasing = [this](std::span<const typename ResourceManagement::RetainedOwner>
                                              owners) { write_on_evict(owners); };
        }
        resources_.set_durability(std::move(durability));
        publish_store_read_rate();
    }

    // Prefill time the cost model expects for a prompt of `tokens` from the root, for ranking
    // stored images by the recomputation they save.
    [[nodiscard]] std::uint64_t prefill_estimate_ns(std::uint32_t tokens) const noexcept {
        return resources_.prefill_ns(make_prefill_work(0, tokens, 0, 0, 0));
    }

    // Prefill time the cost model expects for a request still in the queue: every chunk the prompt
    // will run and the Vision work of its media, so the backlog is not lower than the work that runs.
    // A long text prompt the lossy path would take (cold, so from the root) is priced as its scout
    // and the main pass over the share of it a selection keeps.
    [[nodiscard]] std::uint64_t queued_prefill_ns(const PromptSummary& summary) const noexcept {
        const double share = lossy_scout_share();
        if (share > 0.0 && select_lossy() && !summary.has_media &&
            summary.prompt_tokens > lossy_options_.min_tokens) {
            const auto kept = lossy_expected_kept(summary.prompt_tokens, lossy_options_.keep_fraction);
            return saturating_add(
                scaled_ns(resources_.prefill_ns(
                              make_prefill_work(0, summary.prompt_tokens, 0, 0, prefill_chunk_)),
                          share),
                resources_.prefill_ns(make_prefill_work(0, kept, 0, 0, prefill_chunk_)));
        }
        return resources_.prefill_ns(make_prefill_work(0, summary.prompt_tokens, summary.vision_items,
                                                       summary.vision_patches, prefill_chunk_));
    }

    // Host context bytes free now, for choosing which stored images to restore at start-up.
    [[nodiscard]] std::size_t free_host_bytes() const {
        std::scoped_lock lock(execution_mutex_);
        const auto usage = instance_.program->physical_usage();
        return usage.capacity.host_bytes - usage.occupied.host_bytes;
    }

    // Restores one stored image into the empty cache at start-up as a Host-resident continuation.
    // Never evicts anything. False when it does not fit now or does not belong to this Engine.
    // `read_ns`, when known, is how long the store took to read it: with the import time it is one
    // observation of the store's read rate.
    bool restore_image(std::span<const std::uint8_t> image, std::uint64_t read_ns = 0) {
        std::scoped_lock lock(execution_mutex_);
        device_.bind_to_current_thread();
        auto& program = *instance_.program;
        if (program.has_context_transaction() || suspended_.load(std::memory_order_acquire)) {
            return false;
        }
        const auto started = Clock::now();
        try {
            const auto needed = program.checkpoint_image_host_bytes(image, store_.binding);
            const auto usage  = program.physical_usage();
            if (needed > usage.capacity.host_bytes - usage.occupied.host_bytes) { return false; }
            auto imported = program.import_checkpoints(image, store_.binding);
            if (!imported || !adopt_imported(*imported, image.size())) { return false; }
        } catch (const std::invalid_argument&) { return false; }
        if (read_ns) {
            observe_store_read(image.size(),
                               read_ns + static_cast<std::uint64_t>(
                                             std::chrono::duration_cast<std::chrono::nanoseconds>(
                                                 Clock::now() - started)
                                                 .count()));
        }
        publish_runtime_stats();
        return true;
    }

    // Hands every inactive retained continuation the store does not hold in its current state to
    // `write`, most recently used first, until `deadline`. For shutdown: `write` may block.
    std::uint32_t persist_all(const std::function<bool(CheckpointImage&&, bool shared)>& write,
                              Clock::time_point deadline) {
        std::scoped_lock lock(execution_mutex_);
        device_.bind_to_current_thread();
        auto& program = *instance_.program;
        if (suspended_.load(std::memory_order_acquire)) {
            // Persisting copies device state out, so a model that resumes on demand comes back
            // for it; one held suspended (or whose resume fails) loses only what the device
            // alone held.
            if (!auto_resume_.load(std::memory_order_acquire)) { return 0; }
            try {
                resume_locked();
            } catch (...) {
                return 0;
            }
        }
        if (program.has_context_transaction()) { return 0; }
        forget_failed_store_writes();
        forget_removed_store_images();
        std::uint32_t written = 0;
        // Shared prefixes first: each one serves every conversation that starts with it.
        for (const auto handle : unstored_shared(program)) {
            if (Clock::now() >= deadline) { break; }
            try {
                auto image       = program.export_checkpoints({&handle, 1}, std::nullopt,
                                                              store_.binding, export_buffer());
                const auto bytes = image.bytes.size();
                if (write(std::move(image), true)) {
                    shared_stored_.emplace_back(handle, bytes);
                    ++written;
                }
            } catch (...) {}
        }
        // A shared write can make the store evict older images: check their marks again so the
        // owners pass does not count them as durable.
        forget_failed_store_writes();
        forget_removed_store_images();
        auto owners = resources_.retained_owners(program);
        std::sort(owners.begin(), owners.end(), [&](const auto& a, const auto& b) {
            return mark(a.token).last_used > mark(b.token).last_used;
        });
        for (const auto& owner : owners) {
            if (Clock::now() >= deadline) { break; }
            // The write before this one ran synchronously and can have made the store evict this
            // owner's image; with the worker stopped nothing else would check its mark.
            forget_failed_store_writes();
            forget_removed_store_images();
            if (owner.active || settled(mark(owner.token), owner.points)) { continue; }
            try {
                auto image        = program.export_checkpoints(owner.points, owner.session,
                                                               store_.binding, export_buffer());
                const auto bytes  = image.bytes.size();
                if (write(std::move(image), false)) {
                    record_stored(owner.token, owner.points, bytes);
                    ++written;
                }
            } catch (...) {}
        }
        return written;
    }

private:
    enum class HostWorkClass : std::uint8_t {
        Decode,
        Prefill,
        Control,
    };

    using EngineHostPhase = RequestEngineHostPhase;

    [[nodiscard]] static nvtx::Name phase_range_name(EngineHostPhase phase) noexcept;

    struct ActiveExposure {
        std::shared_ptr<Request> request;
        std::uint32_t lane = 0;
    };

    struct ActiveExposureSet {
        std::array<ActiveExposure, kMaximumConcurrency> entries{};
        std::size_t size = 0;
    };

    struct HostPhaseMeasurement {
        Clock::time_point started;
        std::uint64_t accounted_before = 0;
        ActiveExposureSet exposed;
    };

    [[nodiscard]] static std::uint64_t elapsed_ns(Clock::time_point started,
                                                  Clock::time_point finished) noexcept;

    [[nodiscard]] ActiveExposureSet active_exposure_set() const;

    [[nodiscard]] HostPhaseMeasurement begin_host_phase() const;

    void set_host_work_class(HostWorkClass work_class,
                             std::span<const std::uint32_t> decode_lanes = {}) noexcept;

    [[nodiscard]] bool current_decode_contains(std::uint32_t lane) const noexcept;

    void add_class_host_time(std::uint64_t host_ns, std::uint64_t device_wait_ns) noexcept;

    void expose_engine_phase(const ActiveExposureSet& exposed, EngineHostPhase phase,
                             std::uint64_t elapsed) noexcept;

    void finish_engine_phase(const HostPhaseMeasurement& measurement,
                             EngineHostPhase phase) noexcept;

    void record_program_timing(runtime::ExecutionTiming timing,
                               const ActiveExposureSet& exposed) noexcept;

    void finish_program_call(const HostPhaseMeasurement& measurement,
                             runtime::ExecutionTiming timing) noexcept;

    void record_detail(std::uint64_t RuntimeHostWorkStats::*elapsed_member,
                       std::uint64_t RuntimeHostWorkStats::*invocation_member,
                       Clock::time_point started) noexcept;

    class EnginePhaseScope {
    public:
        EnginePhaseScope(EngineCore& owner, EngineHostPhase phase)
            : owner_(owner), phase_(phase), measurement_(owner.begin_host_phase()) {
            range_.emplace(phase_range_name(phase), nvtx::Category::Runtime);
        }

        ~EnginePhaseScope() { finish(); }

        EnginePhaseScope(const EnginePhaseScope&)            = delete;
        EnginePhaseScope& operator=(const EnginePhaseScope&) = delete;

        void pause_range() noexcept { range_.reset(); }

        void resume_range() noexcept {
            if (active_ && !range_) {
                range_.emplace(phase_range_name(phase_), nvtx::Category::Runtime);
            }
        }

        void checkpoint() noexcept {
            if (!active_) { return; }
            owner_.finish_engine_phase(measurement_, phase_);
            measurement_ = owner_.begin_host_phase();
        }

        void finish() noexcept {
            if (!active_) { return; }
            range_.reset();
            owner_.finish_engine_phase(measurement_, phase_);
            active_ = false;
        }

    private:
        EngineCore& owner_;
        EngineHostPhase phase_;
        HostPhaseMeasurement measurement_;
        std::optional<nvtx::ScopedRange> range_;
        bool active_ = true;
    };

    class ProgramCallScope {
    public:
        explicit ProgramCallScope(EngineCore& owner)
            : owner_(owner), measurement_(owner.begin_host_phase()) {}

        ~ProgramCallScope() noexcept { finish(failed_timing_); }

        ProgramCallScope(const ProgramCallScope&)            = delete;
        ProgramCallScope& operator=(const ProgramCallScope&) = delete;

        [[nodiscard]] runtime::ExecutionTiming& failed_timing() noexcept { return failed_timing_; }

        void finish(runtime::ExecutionTiming timing) noexcept {
            if (!active_) { return; }
            owner_.finish_program_call(measurement_, timing);
            active_ = false;
        }

    private:
        EngineCore& owner_;
        HostPhaseMeasurement measurement_;
        runtime::ExecutionTiming failed_timing_;
        bool active_ = true;
    };

    void publish_runtime_stats();

    GenerationResult wait_for_request(std::shared_ptr<Request> request, OutputSink* sink,
                                      const CancellationView& cancellation) {
        struct ConsumerGuard {
            EngineCore* owner;
            std::shared_ptr<Request> request;

            ~ConsumerGuard() { owner->release_consumer(request); }
        } guard{this, request};

        std::exception_ptr caller_error;
        bool scheduling_failed   = false;
        const auto fail_consumer = [&] {
            if (caller_error == nullptr) { caller_error = std::current_exception(); }
            request->cancelled.store(true, std::memory_order_release);
            request_admission_check();
            queue_cv_.notify_one();
        };
        std::optional<GenerationStart> start;
        std::optional<PromptProgress> progress;
        std::vector<typename Request::StreamEvent> events;
        for (;;) {
            start.reset();
            progress.reset();
            events.clear();
            bool done = false;
            {
                std::unique_lock lock(request->mutex);
                request->cv.wait_for(lock, std::chrono::milliseconds(10), [&] {
                    return request->response_done || request->stream_start.has_value() ||
                           request->stream_progress.has_value() || !request->events.empty();
                });
                start = std::move(request->stream_start);
                request->stream_start.reset();
                progress = std::move(request->stream_progress);
                request->stream_progress.reset();
                events.swap(request->events);
                done = request->response_done;
            }

            if (caller_error == nullptr && sink != nullptr) {
                try {
                    if (start) { sink->start(std::move(*start)); }
                    if (progress) { sink->progress(std::move(*progress)); }
                } catch (...) { fail_consumer(); }
            }
            for (auto& event : events) {
                if (const auto* scheduling =
                        std::get_if<std::unique_ptr<GenerationSchedulingObservation>>(&event)) {
                    if (scheduling_failed) { continue; }
                    try {
                        request->observation.scheduling(**scheduling);
                    } catch (...) {
                        scheduling_failed = true;
                        fail_consumer();
                    }
                } else if (const auto* first_token =
                               std::get_if<GenerationFirstTokenObservation>(&event)) {
                    if (caller_error == nullptr) {
                        try {
                            request->observation.first_token(*first_token);
                        } catch (...) { fail_consumer(); }
                    }
                } else if (caller_error == nullptr && sink != nullptr) {
                    try {
                        if (auto* timing = std::get_if<GenerationTimingObservation>(&event)) {
                            sink->timing(std::move(*timing));
                        } else {
                            sink->publish(std::move(std::get<OutputDelta>(event)));
                        }
                    } catch (...) { fail_consumer(); }
                }
            }

            if (caller_error == nullptr) {
                try {
                    if (cancellation.requested()) {
                        request->cancelled.store(true, std::memory_order_release);
                        request_admission_check();
                        queue_cv_.notify_one();
                    }
                } catch (...) { fail_consumer(); }
            }
            if (!done) { continue; }

            if (caller_error != nullptr) { std::rethrow_exception(caller_error); }
            std::lock_guard lock(request->mutex);
            if (request->error != nullptr) { std::rethrow_exception(request->error); }
            return std::move(request->result);
        }
    }

    // Coalesces admission-visible queue/resource changes. Ordinary prefill/decode progress does not
    // re-arm a temporarily blocked inspection.
    void request_admission_check() noexcept {
        admission_check_pending_.store(true, std::memory_order_release);
    }

    [[nodiscard]] bool consume_admission_check() noexcept {
        return admission_check_pending_.exchange(false, std::memory_order_acq_rel);
    }

    struct MaterializingRequest {
        std::shared_ptr<Request> request;
        LaneId destination;
        BeginSummary summary;
        bool resumed = false;
        typename ResourceManagement::SourceChoice source;
    };

    struct AdmissionDecision {
        std::shared_ptr<Request> request;
        bool restoring = false;
        std::vector<typename ResourceManagement::SourceChoice> sources;
        std::size_t current = 0;
        typename ResourceManagement::ReclaimCursor reclaim;
    };

    struct CaptureDecision {
        SequenceHandle sequence;
        std::uint32_t frontier = 0;
        typename ResourceManagement::ReclaimCursor reclaim;
    };

    [[nodiscard]] std::optional<GenerationTimingObservation>
    record_committed_output(const std::shared_ptr<Request>& request,
                            std::uint32_t accepted_tokens) {
        if (accepted_tokens == 0) { return std::nullopt; }
        cumulative_stats_.generated_tokens += accepted_tokens;
        const bool observe_wall =
            request->observation.phase_timings || request->observation.live_timings;
        const bool need_now         = !request->first_token || observe_wall;
        const Clock::time_point now = need_now ? Clock::now() : Clock::time_point{};
        if (!request->first_token) {
            request->first_token = now;
            if (request->observation.first_token) {
                GenerationFirstTokenObservation observation{
                    .prepare_seconds = request->prepare_seconds,
                    .elapsed_since_submit_seconds =
                        std::chrono::duration<double>(now - request->submitted).count(),
                    .queue_wait_seconds = request->admitted_at
                                              ? std::chrono::duration<double>(
                                                    *request->admitted_at - request->submitted)
                                                    .count()
                                              : 0.0,
                };
                {
                    std::lock_guard lock(request->mutex);
                    request->events.emplace_back(observation);
                }
                request->cv.notify_one();
            }
        }
        if (!observe_wall) { return std::nullopt; }
        if (!request->admitted_at || !request->first_token) {
            throw std::logic_error("committed output has no observed admission boundary");
        }
        request->last_token = now;
        if (!request->observation.live_timings) { return std::nullopt; }
        if (request->generated.size() > std::numeric_limits<std::uint32_t>::max()) {
            throw std::overflow_error("generated token count exceeds observation domain");
        }
        return GenerationTimingObservation{
            .generated_tokens      = static_cast<std::uint32_t>(request->generated.size()),
            .prompt_elapsed_ns     = elapsed_ns(*request->admitted_at, *request->first_token),
            .generation_elapsed_ns = elapsed_ns(*request->first_token, *request->last_token),
        };
    }

    void record_first_output(const std::shared_ptr<Request>& request);
    static void record_execution_work(GenerationWorkTiming& work, ExecutionTiming timing) noexcept;

    void append_output(const std::shared_ptr<Request>& request, PublishedOutput output,
                       std::optional<GenerationTimingObservation> timing = std::nullopt,
                       EnginePhaseScope* phase                           = nullptr) {
        if (output.empty() && !timing) { return; }
        if (!request->first_output_timing &&
            std::any_of(output.begin(), output.end(),
                        [](const auto& delta) { return !delta.text.empty(); })) {
            if (phase) { phase->checkpoint(); }
            record_first_output(request);
        }
        const bool streaming = request->consumer_mode == OutputConsumerMode::Streaming;
        {
            std::lock_guard lock(request->mutex);
            if (streaming && timing) { request->events.emplace_back(std::move(*timing)); }
            for (OutputDelta& delta : output) {
                std::string& full = delta.channel == OutputChannel::Reasoning ? request->reasoning
                                                                              : request->content;
                full += delta.text;
                if (streaming) { request->events.emplace_back(std::move(delta)); }
            }
        }
        if (streaming) { request->cv.notify_one(); }
    }

    void publish_prompt_progress(const std::shared_ptr<Request>& request) {
        if (!request->observation.prompt_progress) { return; }
        if (!request->admitted_begin || !request->admitted_at) {
            throw std::logic_error("prompt progress has no admitted request boundary");
        }
        const BeginSummary& begin = *request->admitted_begin;
        if (begin.reused_prompt_tokens > begin.prompt_tokens ||
            request->computed_prompt_tokens > begin.prompt_tokens - begin.reused_prompt_tokens) {
            throw std::logic_error("prompt progress exceeds the admitted prompt frontier");
        }
        // In the client's prompt tokens: a lossy prompt's dropped tokens count as processed once the
        // selection has skipped them, and a prompt rewritten onto a lossy lineage counts its
        // original length (docs/maintainer/lossy-long-prefill.md, "Capacity and reporting").
        const std::uint32_t total  = reported_prompt_tokens(*request);
        const std::uint32_t reused = reported_frontier(*request, begin.reused_prompt_tokens);
        const PromptProgress progress{
            .total_prompt_tokens  = total,
            .reused_prompt_tokens = reused,
            .processed_prompt_tokens =
                std::min(total, reported_frontier(*request, begin.reused_prompt_tokens +
                                                                request->computed_prompt_tokens)),
            .elapsed_ns = elapsed_ns(*request->admitted_at, Clock::now()),
        };
        {
            std::lock_guard lock(request->mutex);
            if (request->response_done) { return; }
            request->stream_progress = progress;
        }
        request->cv.notify_one();
    }

    void publish_generation_start(const std::shared_ptr<Request>& request, BeginSummary begin) {
        if (request->admitted_begin) {
            throw std::logic_error("request admission published generation start twice");
        }
        request->admitted_begin = begin;
        if (request->observation.phase_timings || request->observation.live_timings ||
            request->observation.prompt_progress) {
            if (!request->admitted_at) {
                throw std::logic_error("generation start has no binding timestamp");
            }
        }
        if (request->consumer_mode != OutputConsumerMode::Streaming) { return; }
        {
            std::lock_guard lock(request->mutex);
            if (request->stream_start || request->response_done) {
                throw std::logic_error("streaming request has an invalid generation-start state");
            }
            request->stream_start = GenerationStart{
                .prompt               = request->prompt_summary,
                .reused_prompt_tokens = reported_frontier(*request, begin.reused_prompt_tokens),
            };
        }
        request->cv.notify_one();
    }

    void release_reserved_capacity() noexcept {
        std::lock_guard lock(queue_mutex_);
        if (outstanding_ != 0) { --outstanding_; }
    }

    void release_consumer(const std::shared_ptr<Request>& request) noexcept {
        bool release = false;
        {
            std::lock_guard lock(request->mutex);
            request->consumer_released = true;
            if (request->response_done && !request->capacity_released) {
                request->capacity_released = true;
                release                    = true;
            }
        }
        if (release) { release_reserved_capacity(); }
    }

    void abandon_request(std::shared_ptr<Request> request) noexcept {
        request->cancelled.store(true, std::memory_order_release);
        request_admission_check();
        queue_cv_.notify_one();
        release_consumer(request);
    }

    bool mark_completed(const std::shared_ptr<Request>& request) noexcept {
        bool release = false;
        {
            std::lock_guard lock(request->mutex);
            if (request->consumer_released && !request->capacity_released) {
                request->capacity_released = true;
                release                    = true;
            }
        }
        return release;
    }

    void release_planning_state(const std::shared_ptr<Request>& request) noexcept {
        if (admission_decision_ && admission_decision_->request == request) {
            admission_decision_.reset();
        }
        if (!request->admitted_at) {
            request->admission.revoked_checkpoints = resources_.source_revocations(request->id);
            finish_source_wait(request, Clock::now());
        }
        resources_.release_source(*instance_.program, request->id);
        if (request->continuation_owner) {
            resources_.abandon(*instance_.program, request->continuation_owner);
            request->continuation_owner = 0;
        }
        request->suspended.reset();
        request->base_plan.reset();
    }

    void complete_error(const std::shared_ptr<Request>& request, std::exception_ptr error) {
        // A request on the lossy path that fails before its outcome counts as aborted, so the
        // outcome counters add up to the requests that took the path.
        lossy_aborted(*request);
        if (request->preemption_count) {
            observe_scheduling(request, GenerationSchedulingTransition::Terminal);
        }
        release_planning_state(request);
        request->prompt      = {};
        request->model_state = EngineRequestState::ModelFinished;
        request->sequence.reset();
        request->lane.reset();
        request->budget.reset();
        request->terminal_reason.reset();
        {
            std::lock_guard lock(request->mutex);
            if (request->response_done) { return; }
            request->error         = std::move(error);
            request->response_done = true;
        }
        if (mark_completed(request)) { release_reserved_capacity(); }
        request->cv.notify_one();
    }

    void complete_success(const std::shared_ptr<Request>& request, FinishReason reason) {
        HostPhaseMeasurement completion = begin_host_phase();
        // Read before release_planning_state drops the plan.
        const std::optional<std::uint32_t> reported_reused =
            request->begin && request->base_plan
                ? std::optional(reported_frontier(*request, request->begin->reused_prompt_tokens))
                : std::nullopt;
        if (request->preemption_count) {
            observe_scheduling(request, GenerationSchedulingTransition::Terminal);
        }
        double prompt_wall_seconds     = 0.0;
        double generation_wall_seconds = 0.0;
        if (request->observation.phase_timings && request->first_token) {
            if (!request->admitted_at || !request->last_token) {
                throw std::logic_error("observed request completed without stable timing bounds");
            }
            prompt_wall_seconds =
                std::chrono::duration<double>(*request->first_token - *request->admitted_at)
                    .count();
            generation_wall_seconds =
                std::chrono::duration<double>(*request->last_token - *request->first_token).count();
        }
        release_planning_state(request);
        request->prompt      = {};
        request->model_state = EngineRequestState::ModelFinished;
        if (!request->queue_wait_recorded) {
            request->host_timing.queue_wait_ns = elapsed_ns(request->submitted, Clock::now());
            request->queue_wait_recorded       = true;
        }
        GenerationResult result;
        result.engine_request_id            = request->id;
        result.admission                    = request->admission;
        result.computed_prefill_tokens      = request->computed_prompt_tokens;
        result.scheduling.preemptions       = request->preemption_count;
        result.scheduling.replay_restores   = request->replay_restores;
        result.scheduling.snapshot_restores = request->snapshot_restores;
        result.scheduling.replayed_tokens   = request->replayed_tokens;
        if (request->paused_at) {
            request->paused_ns += elapsed_ns(*request->paused_at, Clock::now());
            request->paused_at.reset();
        }
        result.scheduling.paused_ns            = request->paused_ns;
        result.scheduling.device_to_host_bytes = request->device_to_host_bytes;
        result.scheduling.host_to_device_bytes = request->host_to_device_bytes;
        result.prompt                          = request->prompt_summary;
        result.generated_token_ids             = std::move(request->generated);
        result.content                         = std::move(request->content);
        result.reasoning                       = std::move(request->reasoning);
        result.constraint                      = request->output.constraint_observation();
        result.tool_calls                      = request->output.take_tool_calls();
        result.tool_call_parse                 = request->output.tool_call_parse_diagnostics();
        result.reasoning_tokens                = request->output.reasoning_tokens();
        result.finish_reason                   = reason;
        result.matched_stop_string             = request->output.matched_stop_string();
        if (request->begin) {
            result.reused_prompt_tokens =
                reported_reused.value_or(request->begin->reused_prompt_tokens);
            result.prefix_reuse_path = request->begin->prefix_reuse_path;
        }
        if (request->lossy) {
            result.lossy_prefill = request->lossy->report;
        } else if (request->summary) {
            result.lossy_prefill = summary_report(*request->summary);
        }
        result.timings                 = request->generation_timings;
        result.timings.prepare_seconds = request->prepare_seconds;
        result.speculative             = std::move(request->speculative_stats);
        result.thinking                = request->output.thinking_stats();
        if (request->first_token) {
            result.timings.first_token_seconds =
                request->prepare_seconds +
                std::chrono::duration<double>(*request->first_token - request->submitted).count();
        }
        result.timings.prompt_wall_seconds     = prompt_wall_seconds;
        result.timings.generation_wall_seconds = generation_wall_seconds;
        result.timings.total_seconds =
            request->prepare_seconds +
            std::chrono::duration<double>(Clock::now() - request->submitted).count();
        request->sequence.reset();
        request->lane.reset();
        request->budget.reset();
        request->terminal_reason.reset();
        finish_engine_phase(completion, EngineHostPhase::CommitOutput);
        result.engine_timing       = request->host_timing.public_snapshot();
        result.first_output_timing = request->first_output_timing;
        {
            std::lock_guard lock(request->mutex);
            if (request->response_done) { return; }
            request->result        = std::move(result);
            request->response_done = true;
        }
        if (mark_completed(request)) { release_reserved_capacity(); }
        request->cv.notify_one();
        // Only a published, uncancelled result shows the Engine serving again; a cancellation,
        // or a completion that threw before publication, leaves the recovery streak standing.
        if (reason != FinishReason::Cancelled) { recovery_streak_.record_success(); }
    }

    void complete_cancelled(const std::shared_ptr<Request>& request) {
        (void)request->output.preview_terminal(FinishReason::Cancelled);
        append_output(request, request->output.commit_preview());
        complete_success(request, FinishReason::Cancelled);
    }

    void complete_detached_cancelled(const std::shared_ptr<Request>& request) {
        try {
            complete_cancelled(request);
        } catch (...) {
            const std::exception_ptr error = std::current_exception();
            complete_error(request, error);
            throw;
        }
    }

    void remove_completed_slot(std::uint32_t lane) {
        capture_decisions_[lane].reset();
        slots_[lane].reset();
        request_admission_check();
        scheduler_.capacity_released();
    }

    [[nodiscard]] std::array<bool, kMaximumConcurrency> snapshot_cancellations() const noexcept {
        std::array<bool, kMaximumConcurrency> cancelled{};
        for (std::uint32_t lane = 0; lane < max_concurrency_; ++lane) {
            if (slots_[lane] != nullptr) {
                cancelled[lane] = slots_[lane]->cancelled.load(std::memory_order_acquire);
            }
        }
        return cancelled;
    }

    bool settle_terminal_requests(HostPhaseMeasurement& boundary) {
        if (instance_.program->has_context_transaction()) { return false; }

        bool changed = false;
        for (;;) {
            std::optional<std::uint32_t> selected;
            for (std::uint32_t lane = 0; lane < max_concurrency_; ++lane) {
                const auto& request = slots_[lane];
                if (request == nullptr || !request->terminal_reason) { continue; }
                if (!selected ||
                    request->publication_order < slots_[*selected]->publication_order) {
                    selected = lane;
                }
            }
            if (!selected) { break; }

            const std::uint32_t lane = *selected;
            const auto request       = slots_[lane];
            if (!request->is_model_finished() || request->capture_pending || !request->sequence ||
                !request->lane || request->lane->value != lane) {
                throw std::logic_error("terminal-pending request has invalid ownership");
            }
            const FinishReason reason = *request->terminal_reason;
            auto finished             = instance_.program->finish(*request->sequence);
            if (finished.status != ConsumeStatus::Consumed) {
                throw std::logic_error("terminal native sequence could not finish");
            }
            if (finished.checkpoint) {
                resources_.publish(*instance_.program, request->continuation_owner,
                                   *finished.checkpoint);
            }
            if (request->continuation_owner && store_.store) {
                mark(request->continuation_owner).last_used = Clock::now();
            }
            resources_.finish(*instance_.program, request->continuation_owner, *request->base_plan,
                              request->publication_order);
            request->continuation_owner = 0;
            request->generation_timings = finished.timings;
            request->speculative_stats  = std::move(finished.speculative);
            request->terminal_reason.reset();

            finish_engine_phase(boundary, EngineHostPhase::Boundary);
            complete_success(request, reason);
            remove_completed_slot(lane);
            boundary = begin_host_phase();
            changed  = true;
        }
        if (changed) { publish_runtime_stats(); }
        return changed;
    }

    void cancel_active_requests(const std::array<bool, kMaximumConcurrency>& cancelled_at_boundary,
                                HostPhaseMeasurement& boundary) {
        if (instance_.program->has_context_transaction()) { return; }
        bool changed = false;
        for (std::uint32_t lane = 0; lane < max_concurrency_; ++lane) {
            const auto& request = slots_[lane];
            if (request == nullptr || !cancelled_at_boundary[lane]) { continue; }
            if (request->capture_pending) { continue; }
            if (!request->sequence || !request->lane || request->lane->value != lane) {
                throw std::logic_error("active cancellation has no sequence binding");
            }
            (void)request->output.preview_terminal(FinishReason::Cancelled);
            if (request->is_prefilling()) {
                ++cumulative_stats_.cancelled_prefills;
                cumulative_stats_.cancelled_prefill_computed_tokens +=
                    request->computed_prompt_tokens;
            }
            // A scout ends with its lane: the Program drops it and its pages in abort.
            lossy_aborted(*request);
            auto aborted                = instance_.program->abort(*request->sequence);
            request->generation_timings = aborted.timings;
            request->speculative_stats  = std::move(aborted.speculative);
            finish_engine_phase(boundary, EngineHostPhase::Boundary);
            append_output(request, request->output.commit_preview());
            complete_success(request, FinishReason::Cancelled);
            remove_completed_slot(lane);
            boundary = begin_host_phase();
            changed  = true;
        }
        if (changed) { publish_runtime_stats(); }
    }

    [[nodiscard]] bool expire_pending_requests() {
        std::vector<std::shared_ptr<Request>> cancelled;
        std::vector<std::shared_ptr<Request>> expired;
        bool have_pending = false;
        {
            std::lock_guard lock(queue_mutex_);
            const auto now = Clock::now();
            for (auto it = pending_.begin(); it != pending_.end();) {
                if ((*it)->cancelled.load(std::memory_order_acquire)) {
                    cancelled.push_back(*it);
                    it = pending_.erase(it);
                } else if (now >= (*it)->deadline) {
                    expired.push_back(*it);
                    it = pending_.erase(it);
                } else {
                    ++it;
                }
            }
            have_pending = !pending_.empty();
        }
        // Requests that left the queue without being admitted, and how long they had waited. The
        // reason is the decision that removed them, not a later re-read of the flag or clock.
        {
            const auto now = Clock::now();
            cumulative_stats_.waiting_cancelled_requests += cancelled.size();
            cumulative_stats_.waiting_expired_requests += expired.size();
            for (const auto& request : cancelled) {
                cumulative_stats_.waiting_abandoned_seconds +=
                    std::chrono::duration<double>(now - request->submitted).count();
            }
            for (const auto& request : expired) {
                cumulative_stats_.waiting_abandoned_seconds +=
                    std::chrono::duration<double>(now - request->submitted).count();
            }
        }
        try {
            for (const auto& request : cancelled) { complete_detached_cancelled(request); }
            for (const auto& request : expired) {
                complete_error(request,
                               std::make_exception_ptr(RequestError(
                                   RequestErrorKind::QueueTimeout,
                                   "inference request expired while waiting for admission")));
            }
        } catch (...) {
            const std::exception_ptr error = std::current_exception();
            for (const auto& request : cancelled) { complete_error(request, error); }
            for (const auto& request : expired) { complete_error(request, error); }
            throw;
        }
        if (!cancelled.empty() || !expired.empty()) {
            request_admission_check();
            publish_runtime_stats();
        }
        return have_pending;
    }

    void commit_pending(PendingBatch&& pending, std::span<const std::uint32_t> lane_indices,
                        bool decode_round,
                        const std::array<bool, kMaximumConcurrency>& cancelled_at_unit_start) {
        EnginePhaseScope phase(*this, EngineHostPhase::CommitOutput);
        const std::size_t row_count = lane_indices.size();
        if (row_count == 0 || row_count != pending.row_count() || pending.row_stride() == 0 ||
            (!pending.row_counts().empty() && pending.row_counts().size() != row_count) ||
            pending.tokens().size() < static_cast<std::size_t>(pending.row_stride()) * row_count) {
            (void)instance_.program->abort_pending(std::move(pending));
            throw std::logic_error("pending batch returned an invalid ragged layout");
        }

        std::array<CommitDecision, kMaximumConcurrency> decisions{};
        std::array<FinishReason, kMaximumConcurrency> finish_reasons{};
        std::array<ContinuationAction, kMaximumConcurrency> continuations{};
        std::array<std::size_t, kMaximumConcurrency> generated_sizes{};
        std::array<bool, kMaximumConcurrency> cancelled{};
        std::array<bool, kMaximumConcurrency> failed{};
        bool generated_staged = false;
        std::array<std::shared_ptr<Request>, kMaximumConcurrency> terminal_requests{};
        std::array<std::uint32_t, kMaximumConcurrency> terminal_lanes{};
        std::array<FinishReason, kMaximumConcurrency> terminal_reasons{};
        std::size_t terminal_count    = 0;
        const auto rollback_generated = [&]() {
            for (std::size_t row = 0; row < row_count; ++row) {
                const auto& request = slots_[lane_indices[row]];
                if (request != nullptr) { request->output.discard_preview(); }
                if (generated_staged && request != nullptr &&
                    request->generated.size() >= generated_sizes[row]) {
                    request->generated.resize(generated_sizes[row]);
                }
            }
            generated_staged = false;
        };
        try {
            for (std::size_t row = 0; row < row_count; ++row) {
                const std::uint32_t lane = lane_indices[row];
                const auto& request      = slots_[lane];
                if (request == nullptr || !request->sequence || !request->lane ||
                    request->lane->value != lane || !request->budget) {
                    throw std::logic_error("pending row has no active Engine request");
                }
                if (decode_round) { ++request->host_timing.decode_rounds; }
                cancelled[row] = cancelled_at_unit_start[lane];
                failed[row]    = !cancelled[row] && pending.constraint_failed(row);
                const std::int32_t raw_count =
                    pending.row_counts().empty() ? 1 : pending.row_counts()[row];
                if (raw_count <= 0 || raw_count > static_cast<std::int32_t>(pending.row_stride())) {
                    throw std::logic_error("pending row has an invalid licensed extent");
                }
                const std::uint32_t count = static_cast<std::uint32_t>(raw_count);
                const auto row_tokens     = pending.tokens().subspan(row * pending.row_stride(),
                                                                     static_cast<std::size_t>(count));
                generated_sizes[row]      = request->generated.size();
                if (failed[row]) {
                    decisions[row] = CommitDecision{.terminal = true, .failed = true};
                    continue;
                }
                if (cancelled[row]) {
                    (void)request->output.preview_terminal(FinishReason::Cancelled);
                    decisions[row] = CommitDecision{
                        .accepted_tokens = 0,
                        .terminal        = true,
                        .cancelled       = true,
                    };
                    finish_reasons[row] = FinishReason::Cancelled;
                    continue;
                }
                const OutputDecision decision = request->output.preview_model(
                    row_tokens, request->budget->remaining(), request->budget->limit_reason());
                if (decision.accepted_tokens == 0 || decision.accepted_tokens > count ||
                    (!decision.finished() && decision.accepted_tokens != count) ||
                    (decision.finished() && decision.continuation != ContinuationAction::Decode) ||
                    (decision.prefix_execution_split_after &&
                     (*decision.prefix_execution_split_after == 0 ||
                      *decision.prefix_execution_split_after > decision.accepted_tokens))) {
                    throw std::logic_error("output policy returned an invalid licensed prefix");
                }
                decisions[row] = CommitDecision{
                    .accepted_tokens              = decision.accepted_tokens,
                    .terminal                     = decision.finished(),
                    .cancelled                    = false,
                    .prefix_execution_split_after = decision.prefix_execution_split_after,
                };
                finish_reasons[row] = decision.finish_reason;
                continuations[row]  = decision.continuation;
            }
            generated_staged = true;
            for (std::size_t row = 0; row < row_count; ++row) {
                const std::uint32_t accepted = decisions[row].accepted_tokens;
                if (accepted == 0) { continue; }
                const auto& request = slots_[lane_indices[row]];
                if (request->generated.size() > request->generated.capacity() ||
                    accepted > request->generated.capacity() - request->generated.size()) {
                    throw std::logic_error("admission did not reserve generated-token capacity");
                }
                const auto first = pending.tokens().begin() +
                                   static_cast<std::ptrdiff_t>(row * pending.row_stride());
                request->generated.insert(request->generated.end(), first,
                                          first + static_cast<std::ptrdiff_t>(accepted));
            }
        } catch (...) {
            const std::exception_ptr error = std::current_exception();
            rollback_generated();
            const auto discarded = instance_.program->abort_pending(std::move(pending));
            if (discarded.status != ConsumeStatus::Consumed) {
                throw std::logic_error("failed pending batch could not be discarded");
            }
            std::rethrow_exception(error);
        }

        std::optional<typename ModelContract::CommitResult> committed_storage;
        try {
            phase.pause_range();
            ProgramCallScope program_call(*this);
            auto committed = instance_.program->commit(
                std::move(pending), std::span<const CommitDecision>(decisions.data(), row_count),
                CommitObservation::ReleasedRowsOnly, &program_call.failed_timing());
            program_call.finish(committed.timing);
            // The commit samples and settles the unit's output, so its time belongs to the unit:
            // a decode round, or the prefill unit that emitted the first token.
            (decode_round ? cumulative_stats_.decode_seconds_total
                          : cumulative_stats_.prefill_seconds_total) +=
                static_cast<double>(committed.timing.elapsed_ns()) * 1e-9;
            committed_storage.emplace(std::move(committed));
            phase.resume_range();
        } catch (...) {
            rollback_generated();
            throw;
        }
        generated_staged = false;
        auto& committed  = *committed_storage;
        if (committed.row_count != row_count) {
            throw std::logic_error("Runtime commit result is not row aligned");
        }
        for (std::size_t row = 0; row < row_count; ++row) {
            const CommitDisposition expected =
                failed[row]               ? CommitDisposition::FailedReleased
                : cancelled[row]          ? CommitDisposition::CancelledReleased
                : decisions[row].terminal ? CommitDisposition::Finishable
                                          : CommitDisposition::Active;
            if (committed.rows[row].disposition != expected) {
                throw std::logic_error("Runtime commit row disposition is invalid");
            }
            if (committed.capture_ready[row] &&
                (decode_round || expected != CommitDisposition::Active)) {
                throw std::logic_error("Runtime exposed a capture outside a committed Begin row");
            }
        }
        const bool terminal_in_batch = std::any_of(
            decisions.begin(), decisions.begin() + static_cast<std::ptrdiff_t>(row_count),
            [](const CommitDecision& decision) { return decision.terminal; });

        for (std::size_t row = 0; row < row_count; ++row) {
            const auto& request  = slots_[lane_indices[row]];
            auto& observed       = request->speculative_stats;
            const auto& counters = committed.rows[row].speculative_counters;
            cumulative_stats_.speculative_rounds += counters.rounds - observed.rounds;
            cumulative_stats_.speculative_draft_tokens +=
                counters.drafted_tokens - observed.drafted_tokens;
            cumulative_stats_.speculative_accepted_tokens +=
                counters.accepted_tokens - observed.accepted_tokens;
            cumulative_stats_.speculative_fallback_steps +=
                counters.fallback_steps - observed.fallback_steps;
            cumulative_stats_.speculative_ngram_rounds +=
                counters.ngram_rounds - observed.ngram_rounds;
            cumulative_stats_.speculative_ngram_draft_tokens +=
                counters.ngram_drafted_tokens - observed.ngram_drafted_tokens;
            cumulative_stats_.speculative_ngram_accepted_tokens +=
                counters.ngram_accepted_tokens - observed.ngram_accepted_tokens;
            observed.rounds                = counters.rounds;
            observed.drafted_tokens        = counters.drafted_tokens;
            observed.accepted_tokens       = counters.accepted_tokens;
            observed.fallback_steps        = counters.fallback_steps;
            observed.ngram_rounds          = counters.ngram_rounds;
            observed.ngram_drafted_tokens  = counters.ngram_drafted_tokens;
            observed.ngram_accepted_tokens = counters.ngram_accepted_tokens;
            if (cancelled[row] || failed[row]) {
                request->generation_timings = committed.rows[row].timings;
                request->speculative_stats  = std::move(committed.rows[row].speculative);
            }
        }

        if (decode_round) {
            ++cumulative_stats_.decode_rounds;
            cumulative_stats_.decode_row_rounds += row_count;
            for (std::size_t row = 0; row < row_count; ++row) {
                if (!cancelled[row]) {
                    cumulative_stats_.committed_decode_tokens += decisions[row].accepted_tokens;
                    slots_[lane_indices[row]]->committed_decode_tokens +=
                        decisions[row].accepted_tokens;
                }
            }
        }

        try {
            for (std::size_t row = 0; row < row_count; ++row) {
                const std::uint32_t lane     = lane_indices[row];
                const auto& request          = slots_[lane];
                const std::uint32_t accepted = decisions[row].accepted_tokens;
                if (failed[row]) {
                    complete_error(request, std::make_exception_ptr(
                                                RequestError(RequestErrorKind::ConstraintDeadEnd,
                                                             "grammar has no valid next token")));
                    remove_completed_slot(lane);
                    continue;
                }
                if (!cancelled[row]) { request->budget->commit(accepted); }
                update_recovery(request, cancelled[row]);
                auto published = request->output.commit_preview();
                auto timing    = record_committed_output(request, accepted);
                append_output(request, std::move(published), std::move(timing), &phase);
                if (decisions[row].terminal) {
                    if (cancelled[row]) {
                        terminal_requests[terminal_count] = request;
                        terminal_lanes[terminal_count]    = lane;
                        terminal_reasons[terminal_count]  = finish_reasons[row];
                        ++terminal_count;
                    } else {
                        request->model_state     = EngineRequestState::ModelFinished;
                        request->terminal_reason = finish_reasons[row];
                    }
                } else if (committed.capture_ready[row]) {
                    if (!request->is_prefilling()) {
                        throw std::logic_error("prompt-frontier capture lost its prefill owner");
                    }
                    const EngineRequestState post_capture_state =
                        continuations[row] == ContinuationAction::ApplyTargetControl
                            ? EngineRequestState::ControlReady
                            : EngineRequestState::DecodeReady;
                    if (terminal_in_batch) {
                        instance_.program->skip_capture(*request->sequence);
                        request->model_state = post_capture_state;
                    } else {
                        reserve_active_capture(request, post_capture_state);
                    }
                    committed.capture_ready[row] = false;
                    if (!request->capture_pending) {
                        request->model_state =
                            continuations[row] == ContinuationAction::ApplyTargetControl
                                ? EngineRequestState::ControlReady
                                : EngineRequestState::DecodeReady;
                    }
                } else {
                    request->model_state =
                        continuations[row] == ContinuationAction::ApplyTargetControl
                            ? EngineRequestState::ControlReady
                            : EngineRequestState::DecodeReady;
                }
            }
        } catch (...) {
            phase.finish();
            for (std::size_t index = 0; index < terminal_count; ++index) {
                complete_success(terminal_requests[index], terminal_reasons[index]);
                remove_completed_slot(terminal_lanes[index]);
            }
            throw;
        }
        phase.finish();
        for (std::size_t index = 0; index < terminal_count; ++index) {
            complete_success(terminal_requests[index], terminal_reasons[index]);
            remove_completed_slot(terminal_lanes[index]);
        }
    }

    // Shared progress anchors (EngineOptions::progress_anchors_shared) exist so that work done on a
    // long prompt survives a cancellation, not to cache every stride of it: a request keeps its
    // two newest unreused anchors and releases an older one the moment a newer one lands. An
    // anchor something has asked for since stays, as it has proven itself like any other prefix.
    void trim_progress_anchors(const std::shared_ptr<Request>& request,
                               std::span<const Checkpoint> captured) {
        constexpr std::size_t kAnchorsKept = 2;
        if (progress_anchor_tokens_ == 0 || !request->base_plan) { return; }
        const std::uint32_t prompt_tokens = request->base_plan->summary().prompt_tokens;
        for (const auto point : captured) {
            if (!instance_.program->valid_checkpoint(point)) { continue; }
            const auto summary = instance_.program->checkpoint_metadata(point);
            if (summary.role == CheckpointRole::SharedPrefix && summary.frontier < prompt_tokens &&
                summary.frontier % progress_anchor_tokens_ == 0) {
                request->progress_anchors.push_back(point);
                if (store_.shared_sink) {
                    std::erase_if(progress_anchor_points_, [&](const auto handle) {
                        return !instance_.program->valid_checkpoint(handle);
                    });
                    progress_anchor_points_.push_back(point);
                }
            }
        }
        while (request->progress_anchors.size() > kAnchorsKept) {
            const Checkpoint stale = request->progress_anchors.front();
            request->progress_anchors.erase(request->progress_anchors.begin());
            (void)resources_.retire_unreused_shared(*instance_.program, stale);
        }
    }

    void reserve_active_capture(const std::shared_ptr<Request>& request, EngineRequestState next) {
        request->model_state        = next;
        request->capture_pending    = true;
        request->post_capture_state = next;
        if (instance_.program->has_context_transaction()) { return; }
        bool started = instance_.program->start_capture(*request->sequence);
        if (!started) {
            instance_.program->skip_capture(*request->sequence);
            request->capture_pending = false;
            return;
        }
        context_owner_ = request;
    }

    bool prepare_semantic_capture(const std::shared_ptr<Request>& request) {
        auto& decision = capture_decisions_[request->lane->value];
        if (decision && decision->sequence != *request->sequence) { decision.reset(); }
        if (instance_.program->capture_is_input(*request->sequence) && !decision) {
            (void)resources_.recycle_input(*instance_.program, request->continuation_owner);
        }
        while (const auto capture = instance_.program->prepare_capture(*request->sequence)) {
            if (capture->reserved) {
                decision.reset();
                return true;
            }
            if (instance_.program->has_context_transaction()) { return false; }
            const auto admission = resources_.capture_admission(
                *instance_.program, request->continuation_owner, *request->base_plan,
                capture->frontier,
                instance_.program->capture_is_input(*request->sequence) &&
                    !capture->shortage.main_kv_pages && !capture->shortage.backend_kv_pages);
            if (!decision || decision->frontier != capture->frontier) {
                decision.emplace(
                    CaptureDecision{*request->sequence, capture->frontier,
                                    resources_.begin_reclaim(*instance_.program, admission)});
            }
            const auto usage = instance_.program->physical_usage();
            if (capture->host_bytes && capture->host_bytes <= usage.capacity.host_bytes) {
                if (capture->host_bytes > usage.capacity.host_bytes - usage.occupied.host_bytes) {
                    (void)instance_.program->release_redundant_host({}, request->sequence);
                }
                if (const auto victims = resources_.host_victims(
                        *instance_.program, capture->host_bytes, std::nullopt, {}, {}, admission,
                        &decision->reclaim)) {
                    resources_.commit_host_victims(*instance_.program, *victims, decision->reclaim);
                    if (!victims->empty()) { scheduler_.capacity_released(); }
                    if (const auto retry = instance_.program->prepare_capture(*request->sequence);
                        retry && retry->reserved) {
                        decision.reset();
                        return true;
                    }
                }
            }
            if (capture->shortage.main_kv_pages && instance_.program->drain_vision_window()) {
                continue;
            }
            const auto progress =
                resources_.reclaim(*instance_.program, capture->shortage, {}, decision->reclaim);
            if (progress == ReclaimProgress::Transferring) { return false; }
            if (progress == ReclaimProgress::Changed) { continue; }
            instance_.program->skip_capture(*request->sequence);
            decision.reset();
        }
        decision.reset();
        return true;
    }

    void
    resolve_prefill_progress(const std::shared_ptr<Request>& request,
                             typename ModelContract::PrefillProgress&& progress,
                             const std::array<bool, kMaximumConcurrency>& cancelled_at_unit_start,
                             std::uint64_t step_elapsed_ns = 0) {
        EnginePhaseScope phase(*this, EngineHostPhase::CommitOutput);
        ++cumulative_stats_.host_work.prefill_units;
        ++request->host_timing.prefill_units;
        cumulative_stats_.computed_prefill_tokens += progress.processed_prompt_tokens;
        if (!request->admitted_begin) {
            throw std::logic_error("prefill progress has no committed admission summary");
        }
        observe_lossy_step(request, progress, step_elapsed_ns);
        const BeginSummary& begin = *request->admitted_begin;
        if (begin.reused_prompt_tokens > begin.prompt_tokens) {
            throw std::logic_error("admitted prefix exceeds its prompt");
        }
        const std::uint32_t suffix_tokens = begin.prompt_tokens - begin.reused_prompt_tokens;
        if (request->computed_prompt_tokens > suffix_tokens ||
            progress.processed_prompt_tokens > suffix_tokens - request->computed_prompt_tokens) {
            throw std::logic_error("prefill unit exceeded the admitted prompt suffix");
        }
        const auto computed_begin = begin.reused_prompt_tokens + request->computed_prompt_tokens;
        request->computed_prompt_tokens += progress.processed_prompt_tokens;
        if (!request->cancelled.load(std::memory_order_acquire)) {
            resources_.observe_committed(*request->base_plan, request->id, computed_begin,
                                         computed_begin + progress.processed_prompt_tokens);
        }
        if (progress.complete && request->computed_prompt_tokens != suffix_tokens) {
            throw std::logic_error("completed prefill did not reach the admitted prompt frontier");
        }
        if (progress.processed_prompt_tokens != 0) { publish_prompt_progress(request); }
        if (progress.capture_ready) {
            if (progress.complete || progress.pending) {
                throw std::logic_error("prefill capture offer overlaps prompt completion");
            }
            reserve_active_capture(request, EngineRequestState::Prefill);
            return;
        }
        if (!progress.complete) { return; }
        if (!request->lane || !progress.pending) {
            throw std::logic_error("completed prefill has no lane or pending token");
        }
        if (!request->admitted_begin || progress.summary != *request->admitted_begin) {
            throw std::logic_error("runtime Begin summary differs from committed admission");
        }
        finish_lossy_prefill(*request);
        const std::uint32_t lane = request->lane->value;
        request->begin           = progress.summary;
        const std::array<std::uint32_t, 1> lanes{lane};
        phase.finish();
        commit_pending(std::move(*progress.pending), lanes, false, cancelled_at_unit_start);
        progress.pending.reset();
    }

    // `decode_ran` is whether this cycle ran a decode round: only then does the step owe it a share.
    void run_prefill_step(std::uint32_t lane,
                          const std::array<bool, kMaximumConcurrency>& cancelled_at_unit_start,
                          bool decode_ran) {
        nvtx::ScopedRange prefill_range(nvtx::Name::Prefill, nvtx::Category::Prefill);
        EnginePhaseScope setup(*this, EngineHostPhase::CommitOutput);
        const auto request = slots_[lane];
        if (request == nullptr || !request->is_prefilling() || request->capture_pending) {
            throw std::logic_error("staged prefill lane has invalid request state");
        }
        if (!request->sequence) {
            throw std::logic_error("prefill request has no sequence handle");
        }
        setup.finish();
        RoundMasks masks;
        masks.outputs[0] = &request->output;
        auto* provider   = request->output.constrained() ? &masks : nullptr;
        // While another lane decodes, a step that would run for longer than the stall budget is
        // cut down to what the cost model predicts fits it (PrefillScheduleOptions).
        const bool pacing_on     = pacing_.options().active();
        const bool any_neighbour = (pacing_on || layer_slices_ > 1) && other_lane_streaming(lane);
        const bool streaming_neighbour = pacing_on && any_neighbour;
        // A chunk that has begun in slices keeps going whether or not the neighbour still decodes:
        // the Program holds its residual until the last slice. A new one starts only for a neighbour.
        const std::uint32_t slices =
            layer_slices_ > 1 && (request->slice_in_flight || any_neighbour) ? layer_slices_ : 0U;
        // A slice that follows a decode round is a boundary the neighbour decoded across.
        if (request->slice_in_flight && decode_ran) { ++cumulative_stats_.prefill_slice_boundaries; }
        std::uint32_t planned_tokens = 0;
        std::uint32_t step_cap       = 0;
        std::uint32_t step_prefix    = 0;
        bool          have_position  = false;
        // The position is read whenever pacing is on, so an uncontended step still calibrates the
        // cost model; only the cap itself waits for a neighbour.
        if (pacing_on && !request->slice_in_flight) {
            if (const auto position = sibling_cursor(*request)) {
                const std::uint32_t prompt_tokens = request->base_plan->summary().prompt_tokens;
                have_position  = true;
                step_prefix    = *position;
                planned_tokens = std::min(prefill_chunk_, prompt_tokens - std::min(prompt_tokens, *position));
                if (streaming_neighbour) {
                    step_cap = pacing_.step_cap(resources_.costs(), step_prefix, planned_tokens, true);
                }
            }
        }
        ProgramCallScope program_call(*this);
        const Clock::time_point step_started = Clock::now();
        auto progress = instance_.program->advance_prefill(
            *request->sequence, &program_call.failed_timing(), provider, step_cap, slices);
        program_call.finish(progress.timing);
        const std::uint64_t step_wall_ns = elapsed_ns(step_started, Clock::now());
        if (!progress.landed_transfers.empty()) {
            typename ModelContract::ContextProgress landed;
            landed.transfers = std::move(progress.landed_transfers);
            record_context_work(landed, request);
        }
        if (progress.partial) {
            // Some layers ran. The step is not done: nothing is resolved, the lane stays first in
            // line for the next prefill turn, and the cycle's decode round runs before it.
            if (!request->slice_in_flight) {
                request->slice_in_flight = true;
                request->slice_elapsed_ns = 0;
                request->slice_planned_tokens = planned_tokens;
                request->slice_wall_ns    = 0;
                request->slice_prefix     = sibling_cursor(*request).value_or(0U);
                ++cumulative_stats_.sliced_prefill_steps;
            }
            request->slice_elapsed_ns += progress.timing.elapsed_ns();
            request->slice_wall_ns += step_wall_ns;
            cumulative_stats_.prefill_seconds_total +=
                static_cast<double>(progress.timing.elapsed_ns()) * 1e-9;
            if (!request->first_output_timing) {
                record_execution_work(request->prefill_work, progress.timing);
            }
            consume_armed_worker_failure();
            publish_runtime_stats();
            return;
        }
        // The step's whole cost, for the pacing model: the earlier slices plus this one.
        std::uint64_t step_elapsed_ns = progress.timing.elapsed_ns();
        std::uint64_t step_wall_total = step_wall_ns;
        if (request->slice_in_flight) {
            step_elapsed_ns += request->slice_elapsed_ns;
            step_wall_total += request->slice_wall_ns;
            step_prefix    = request->slice_prefix;
            planned_tokens = request->slice_planned_tokens;
            have_position = pacing_on;
            request->slice_in_flight = false;
            // The lane can be paused again. Whatever was refused while it was part-way (a short
            // request that wanted its lane) is looked at again, because nothing else would.
            request_admission_check();
        }
        // A media step ignores the cap and is priced by a text-only model, so it must not teach the
        // correction factor; a step cut short by a capture or rewrite frontier did not run at the
        // cap either. The cap counts as honored only when the step ran exactly that wide.
        if (have_position && progress.processed_prompt_tokens != 0 &&
            !request->prompt_summary.has_media) {
            pacing_.prefill_step_done(
                predicted_prefill_step_ns(resources_.costs(), step_prefix,
                                          progress.processed_prompt_tokens),
                step_elapsed_ns, decode_ran, step_wall_total);
            if (progress.cap_applied) {
                ++cumulative_stats_.paced_prefill_steps;
                cumulative_stats_.paced_prefill_deferred_tokens +=
                    planned_tokens - progress.processed_prompt_tokens;
            }
        }
        cumulative_stats_.prefill_seconds_total +=
            static_cast<double>(progress.timing.elapsed_ns()) * 1e-9;
        if (!request->first_output_timing) {
            record_execution_work(request->prefill_work, progress.timing);
        }
        consume_armed_worker_failure();
        resolve_prefill_progress(request, std::move(progress), cancelled_at_unit_start,
                                 step_elapsed_ns);
        publish_runtime_stats();
    }

    // ---- lossy prefill (lossy_prefill.h, docs/maintainer/lossy-long-prefill.md) ----

    [[nodiscard]] double lossy_scout_share() const noexcept {
        if constexpr (kLossyCapable) {
            return select_lossy() ? instance_.program->lossy_scout_share() : 0.0;
        } else {
            return 0.0;
        }
    }

    // The token (select) strategy runs in the core; the summary strategy rewrote the prompt in
    // Engine::prepare, so the core never scouts for it.
    [[nodiscard]] bool select_lossy() const noexcept {
        return lossy_options_.enabled() && lossy_options_.strategy == LossyStrategy::Select;
    }

    // The client's prompt length: what prompt progress counts to.
    [[nodiscard]] static std::uint32_t reported_prompt_tokens(const Request& request) noexcept {
        if (request.summary && request.summary->rewrite) {
            return request.summary->rewrite->original_tokens;
        }
        const auto& summary = request.base_plan->summary();
        return summary.reported_prompt_tokens != 0 ? summary.reported_prompt_tokens
                                                   : summary.prompt_tokens;
    }

    // A frontier of the request's prompt in the client's tokens: through the plan's lossy lineage,
    // then through a summary rewrite.
    [[nodiscard]] static std::uint32_t reported_frontier(const Request& request,
                                                         std::uint32_t frontier) noexcept {
        const std::uint32_t planned =
            request.base_plan ? reported_frontier(*request.base_plan, frontier) : frontier;
        return request.summary && request.summary->rewrite
                   ? request.summary->rewrite->original_frontier(planned)
                   : planned;
    }

    // The request log's report for a prompt the summary strategy rewrote (completed) or probed
    // and left whole (skipped); the outcome counters count the same outcomes.
    [[nodiscard]] static LossyPrefillReport summary_report(const compaction::SummaryRecord& record) {
        LossyPrefillReport report{.outcome         = LossyPrefillOutcome::Skipped,
                                  .uncached_tokens = record.prompt_tokens,
                                  .kept_tokens     = record.prompt_tokens,
                                  .strategy        = LossyStrategy::Summary,
                                  .summary_probe   = record.probe};
        if (const auto& rewrite = record.rewrite) {
            report.outcome             = LossyPrefillOutcome::Completed;
            report.uncached_tokens     = rewrite->original_tokens;
            report.kept_tokens         = rewrite->prompt_tokens;
            report.summarized_messages = rewrite->messages;
            report.summary_reused      = !rewrite->fresh;
        }
        return report;
    }

    // At admission, once the source is chosen: whether this request takes the lossy path
    // (lossy_prefill_trigger). The Program is told once the binding publishes.
    //
    // Accepted v1 limitation: a plan is made once, when the request is first considered for
    // admission (ensure_base_plan), and only a plan made after a lossy prefill finished is
    // rewritten onto its kept lineage. A follow-up planned earlier (one already queued behind it)
    // cannot match the selected continuation: it reuses at most the exact product prompt and,
    // past the threshold, takes the lossy path (a scout) of its own.
    void plan_lossy_prefill(const std::shared_ptr<Request>& request,
                            const typename ModelContract::SourceCandidate& source) {
        request->lossy.reset();
        if constexpr (kLossyCapable) {
            if (!select_lossy()) { return; }
            const auto& plan  = *request->base_plan;
            const auto lossy  = plan.lossy();
            std::optional<CheckpointRole> role;
            if (source.checkpoint) {
                role = instance_.program->checkpoint_metadata(*source.checkpoint).role;
            }
            const auto exact_end = lossy_prefill_trigger({
                .min_tokens    = lossy_options_.min_tokens,
                .exact_end     = lossy ? std::optional(lossy->exact_end) : std::nullopt,
                .prompt_tokens = plan.summary().prompt_tokens,
                .has_media     = request->prompt_summary.has_media,
                .source_role   = role,
                .reused_tokens = source.reused_tokens,
            });
            if (exact_end) { request->lossy = LossyPrefillState{.exact_end = *exact_end}; }
        } else {
            (void)source;
        }
    }

    // After the binding published and before any step.
    void start_lossy_prefill(const std::shared_ptr<Request>& request) {
        if (!request->lossy || request->lossy->phase != LossyPrefillState::Phase::Planned) {
            return;
        }
        if constexpr (kLossyCapable) {
            if (instance_.program->enable_lossy(*request->sequence)) {
                request->lossy->phase = LossyPrefillState::Phase::Exact;
                return;
            }
        }
        request->lossy.reset();
    }

    [[nodiscard]] LossyPrefillReport lossy_report(const Request& request) const noexcept {
        const auto& state = *request.lossy;
        const std::uint32_t prompt = request.base_plan->summary().prompt_tokens;
        return LossyPrefillReport{
            .outcome         = LossyPrefillOutcome::Aborted,
            .uncached_tokens = prompt > state.exact_end ? prompt - state.exact_end : 0U,
            .filter_layer    = lossy_options_.filter_layer,
        };
    }

    // A step's lossy reports: scout progress, and once, the outcome and the prompt's new length.
    void observe_lossy_step(const std::shared_ptr<Request>& request,
                            typename ModelContract::PrefillProgress& progress,
                            std::uint64_t step_elapsed_ns) {
        if constexpr (kLossyCapable) {
            using Phase = LossyPrefillState::Phase;
            if (progress.scouted_tokens != 0) {
                if (!request->lossy || !request->lossy->before_outcome() ||
                    progress.processed_prompt_tokens != 0) {
                    throw std::logic_error("a scout step for a request not on the lossy path");
                }
                request->lossy->phase = Phase::Scout;
                request->lossy->scouted += progress.scouted_tokens;
                cumulative_stats_.lossy_prefill_scouted_tokens += progress.scouted_tokens;
            }
            if (progress.compacted_prompt_tokens) {
                if (!request->lossy || !request->lossy->before_outcome() || !progress.lossy) {
                    throw std::logic_error("a prompt-length change for a request not on the lossy path");
                }
                auto& state        = *request->lossy;
                BeginSummary& begin = *request->admitted_begin;
                const std::uint32_t length = *progress.compacted_prompt_tokens;
                if (length < begin.reused_prompt_tokens + request->computed_prompt_tokens ||
                    length > begin.prompt_tokens) {
                    throw std::logic_error("a lossy prefill's prompt length is out of range");
                }
                LossyPrefillReport report = lossy_report(*request);
                const auto& outcome       = *progress.lossy;
                report.uncached_tokens    = outcome.uncached_tokens;
                report.kept_tokens        = outcome.kept_tokens;
                report.scout_ms           = outcome.scout_seconds * 1000.0;
                report.kept_runs          = outcome.kept_runs;
                report.outcome            = outcome.declined ? LossyPrefillOutcome::Declined
                                            : outcome.skipped ? LossyPrefillOutcome::Skipped
                                                              : LossyPrefillOutcome::Completed;
                // The Program's plan describes the request from now on: a re-admission after a
                // pause, checkpoint matching, captures and accounting all read the Engine's.
                if (progress.compacted_plan) {
                    *request->base_plan = std::move(*progress.compacted_plan);
                    if (request->lane) { capture_decisions_[request->lane->value].reset(); }
                }
                if (request->base_plan->summary().prompt_tokens != length) {
                    throw std::logic_error("a lossy prefill changed its length without its plan");
                }
                begin.prompt_tokens = length;
                // Declined and skipped are counted now (both "skipped"); a compacted prompt once its
                // kept tokens are prefilled.
                if (report.outcome != LossyPrefillOutcome::Completed) {
                    ++cumulative_stats_.lossy_prefills_skipped;
                }
                if (report.outcome != LossyPrefillOutcome::Declined) {
                    cumulative_stats_.lossy_prefill_kept_tokens += report.kept_tokens;
                }
                state.report = report;
                state.phase  = Phase::Main;
                // A short request waiting to yield past this lane could not pause it while it
                // scouted; it can now, and nothing else would look again.
                request_admission_check();
            }
            if (progress.processed_prompt_tokens != 0 && request->lossy &&
                request->lossy->phase == Phase::Main) {
                request->lossy->main_ns += step_elapsed_ns;
            }
        } else {
            (void)request;
            (void)progress;
            (void)step_elapsed_ns;
        }
    }

    // The prompt is prefilled: a compacted request is counted completed.
    void finish_lossy_prefill(Request& request) {
        if (!request.lossy) { return; }
        auto& state = *request.lossy;
        using Phase = LossyPrefillState::Phase;
        if (state.phase == Phase::Main && state.report) {
            state.report->main_ms = static_cast<double>(state.main_ns) * 1e-6;
            if (state.report->outcome == LossyPrefillOutcome::Completed) {
                ++cumulative_stats_.lossy_prefills_completed;
            }
            state.phase = Phase::Ended;
        } else if (state.before_outcome() && state.phase != Phase::Planned) {
            // The path did not run to its outcome (the lane was rebuilt without it).
            lossy_aborted(request);
        }
    }

    // A request on the lossy path that ends, or leaves the path, before its outcome counts as
    // aborted: cancelled, failed, or its scout ended for memory pressure. One already counted, or
    // not yet bound, is left alone. noexcept: it only updates counters and the report.
    void lossy_aborted(Request& request) noexcept {
        if (!request.lossy) { return; }
        auto& state = *request.lossy;
        using Phase = LossyPrefillState::Phase;
        const bool open = state.phase == Phase::Exact || state.phase == Phase::Scout ||
                          (state.phase == Phase::Main && state.report &&
                           state.report->outcome == LossyPrefillOutcome::Completed);
        if (!open) { return; }
        if (!state.report && request.base_plan) { state.report = lossy_report(request); }
        if (!state.report) { state.report = LossyPrefillReport{.filter_layer = lossy_options_.filter_layer}; }
        state.report->outcome = LossyPrefillOutcome::Aborted;
        state.report->main_ms = static_cast<double>(state.main_ns) * 1e-6;
        state.phase           = Phase::Ended;
        ++cumulative_stats_.lossy_prefills_aborted;
    }

    // Where the Engine has to pause a scouting lane for memory (start_pause refuses while the scout
    // runs, and waiting for it could wait on the very pages the scout holds): the scout ends at M,
    // the request goes on as an ordinary full prefill from there, and the pause proceeds. Its
    // restore resumes at M from the snapshot. False when the lane is not scouting.
    bool end_lossy_scout_for_pause(const std::shared_ptr<Request>& request) {
        if constexpr (kLossyCapable) {
            if (!request->lossy || !request->lossy->scouting() ||
                !instance_.program->abort_lossy_scout(*request->sequence)) {
                return false;
            }
            lossy_aborted(*request);
            // The scout's pages past M are free again.
            scheduler_.capacity_released();
            request_admission_check();
            return true;
        } else {
            (void)request;
            return false;
        }
    }

    // What the lossy path still costs from `cursor`: the exact part to M, the scout over what is
    // left of U at its share of a full prefill (the layers below the filter layer), and the main
    // pass over the share of U a selection keeps.
    [[nodiscard]] std::uint64_t lossy_remaining_ns(const BasePlan& plan, std::uint32_t cursor,
                                                   const LossyPrefillState& state) const {
        const std::uint32_t prompt = plan.summary().prompt_tokens;
        const std::uint32_t m      = std::min(state.exact_end, prompt);
        std::uint64_t total        = 0;
        if (cursor < m) { total = resources_.prefill_ns(instance_.program->prefill_work(plan, cursor, m)); }
        const std::uint32_t scouted = std::min(prompt - m, state.scouted);
        total = saturating_add(
            total, scaled_ns(resources_.prefill_ns(
                                 instance_.program->prefill_work(plan, m + scouted, prompt)),
                             lossy_scout_share()));
        const std::uint32_t kept = lossy_expected_kept(prompt - m, lossy_options_.keep_fraction);
        return saturating_add(
            total, resources_.prefill_ns(instance_.program->prefill_work(plan, m, m + kept)));
    }

    void ensure_base_plan(const std::shared_ptr<Request>& request) {
        if (!request->base_plan) {
            request->base_plan.emplace(instance_.program->plan_request(std::move(request->prompt),
                                                                       request->options.execution));
            const auto& summary = request->base_plan->summary();
            std::uint32_t output_tokens = summary.effective_output_tokens;
            FinishReason limit_reason   = summary.effective_limit_reason;
            // A prompt the summary strategy shortened may not generate past where the client's own
            // prompt would reach max_context, as the Program's cap does for a prompt it rewrote;
            // when that cap binds, the request ends for context capacity, not its output limit.
            if (request->summary && request->summary->rewrite &&
                request->summary->rewrite->original_tokens <= max_context_) {
                const std::uint32_t capacity =
                    max_context_ - request->summary->rewrite->original_tokens + 1U;
                if (capacity < output_tokens) {
                    output_tokens = capacity;
                    limit_reason  = FinishReason::ContextCapacity;
                }
            }
            request->budget.emplace(output_tokens, limit_reason);
            request->generated.reserve(summary.effective_output_tokens);
        }
    }

    [[nodiscard]] std::optional<std::uint32_t> free_lane() const noexcept {
        for (std::uint32_t lane = 0; lane < max_concurrency_; ++lane) {
            if (!slots_[lane] && (!materializing_ || materializing_->destination.value != lane)) {
                return lane;
            }
        }
        return std::nullopt;
    }

    [[nodiscard]] bool resident_empty() const noexcept {
        return !materializing_ &&
               std::all_of(slots_.begin(), slots_.end(), [](const auto& p) { return !p; });
    }

    [[nodiscard]] bool has_older_resident(std::uint64_t ticket) const noexcept {
        return std::any_of(slots_.begin(), slots_.end(), [ticket](const auto& request) {
            return request && request->id < ticket;
        });
    }

    void record_context_work(const typename ModelContract::ContextProgress& progress,
                             const std::shared_ptr<Request>& owner);

    bool progress_context_transaction(HostPhaseMeasurement& boundary) {
        if (!instance_.program->has_context_transaction()) { return false; }
        const auto owner = materializing_ ? materializing_->request : context_owner_;
        const auto cancellation =
            owner ? CancellationFlagView{&owner->cancelled} : CancellationFlagView{};
        auto progress = instance_.program->poll_context(cancellation);
        if (!progress.complete) { return progress.advanced; }
        record_context_work(progress, owner);
        if (owner && progress.request_timings) {
            owner->generation_timings = *progress.request_timings;
            owner->speculative_stats  = std::move(progress.request_speculative);
        }
        if (materializing_) {
            const auto control  = std::move(*materializing_);
            const auto& request = control.request;
            if (!progress.published) {
                complete_detached_cancelled(request);
                materializing_.reset();
                scheduler_.capacity_released();
                request_admission_check();
                return true;
            }
            auto carried = progress.private_points;
            if (control.resumed) {
                for (const auto point : resources_.points(request->continuation_owner)) {
                    if (instance_.program->valid_checkpoint(point) &&
                        std::find(carried.begin(), carried.end(), point) == carried.end()) {
                        carried.push_back(point);
                    }
                }
            }
            request->continuation_owner =
                resources_.adopt(*instance_.program, control.source, *request->base_plan,
                                 request->publication_order, carried, progress.retired_checkpoints);
            resources_.release_source(*instance_.program, request->id);
            request->lane     = control.destination;
            request->sequence = progress.sequence;
            if (!request->sequence) {
                throw std::logic_error("binding completed without a native sequence");
            }
            request->model_state = progress.replaying ? EngineRequestState::Replay
                                   : control.resumed  ? request->resume_phase
                                                      : EngineRequestState::Prefill;
            if (control.resumed) {
                if (!progress.replaying) {
                    ++cumulative_stats_.snapshot_restores;
                    ++request->snapshot_restores;
                }
                request->suspended.reset();
                request->recovery_pending = instance_.program->recovery_pending(*request->sequence);
                if (request->paused_at) {
                    request->paused_ns += elapsed_ns(*request->paused_at, Clock::now());
                }
                request->paused_at.reset();
                // A request paused in Exact (before M) keeps its lossy path: the Program's saved
                // control carries its phase and M to the restored lane (pause.cpp, install_binding),
                // which goes on to M and scouts. Nothing is enabled again here; enable_lossy refuses
                // a lane already on the path. A scout cannot pause, and Main runs on the compacted
                // plan this rebinding used.
                observe_scheduling(request, GenerationSchedulingTransition::Restored);
            } else {
                request->initial_binding_ns = elapsed_ns(*request->admitted_at, Clock::now());
                if (control.summary.reused_prompt_tokens) {
                    ++cumulative_stats_.checkpoint_selections;
                } else {
                    ++cumulative_stats_.root_selections;
                }
                // In the client's tokens, as usage reports them; computed_prefill_tokens counts
                // what the Program actually prefilled.
                cumulative_stats_.reused_prompt_tokens +=
                    reported_frontier(*request, control.summary.reused_prompt_tokens);
                cumulative_stats_.prompt_tokens += reported_prompt_tokens(*request);
                if (const auto& summary = request->summary; summary && summary->rewrite) {
                    ++(summary->rewrite->fresh ? cumulative_stats_.lossy_prefill_summaries_new
                                               : cumulative_stats_.lossy_prefill_summaries_reused);
                    ++cumulative_stats_.lossy_prefills_completed;
                } else if (summary) {
                    ++cumulative_stats_.lossy_prefills_skipped;
                }
                cumulative_stats_.last_selected_frontier_tokens =
                    control.summary.reused_prompt_tokens;
                // Generation start was published when the first binding obtained its resources.
                start_lossy_prefill(request);
            }
            finish_engine_phase(boundary, EngineHostPhase::Boundary);
            slots_[control.destination.value] = request;
            materializing_.reset();
            boundary = begin_host_phase();
        } else if (context_owner_) {
            const auto request = context_owner_;
            if (request->model_state == EngineRequestState::Pausing) {
                const auto lane = request->lane->value;
                if (!progress.published) {
                    finish_engine_phase(boundary, EngineHostPhase::Boundary);
                    slots_[lane].reset();
                    request->lane.reset();
                    request->sequence.reset();
                    context_owner_.reset();
                    complete_detached_cancelled(request);
                    scheduler_.capacity_released();
                    request_admission_check();
                    boundary = begin_host_phase();
                    return true;
                }
                request->suspended      = std::move(progress.paused);
                request->model_state    = EngineRequestState::Paused;
                request->recovery_route = request->suspended->has_snapshot()
                                              ? GenerationRecoveryRoute::Snapshot
                                              : GenerationRecoveryRoute::Replay;
                observe_scheduling(request, GenerationSchedulingTransition::Paused);
                paused_.push_back(request);
                std::sort(paused_.begin(), paused_.end(),
                          [](const auto& a, const auto& b) { return a->id < b->id; });
                // Yielding a borrower may unblock an older suspended request. The newly
                // paused request itself waits for a later event, avoiding immediate churn.
                if (paused_.front() != request) { scheduler_.capacity_released(); }
                finish_engine_phase(boundary, EngineHostPhase::Boundary);
                slots_[lane].reset();
                request->lane.reset();
                request->sequence.reset();
                request_admission_check();
                boundary = begin_host_phase();
            } else {
                if (progress.published) {
                    ++cumulative_stats_.active_captures_completed;
                } else {
                    ++cumulative_stats_.active_captures_aborted;
                }
                request->capture_pending = false;
                request->model_state     = request->post_capture_state;
                for (const auto point : progress.captured_checkpoints) {
                    resources_.publish(*instance_.program, request->continuation_owner, point);
                }
                trim_progress_anchors(request, progress.captured_checkpoints);
            }
            context_owner_.reset();
        }
        if (progress.kind == decltype(progress.kind)::Demote) { scheduler_.capacity_released(); }
        request_admission_check();
        return true;
    }

    bool reclaim(ContextResourceUsage shortage, bool allow_pause,
                 std::span<const Checkpoint> excluded               = {},
                 typename ResourceManagement::ReclaimCursor* cursor = nullptr,
                 std::optional<std::uint64_t> priority_ticket       = std::nullopt) {
        if (!shortage.state_slots && !shortage.main_kv_pages && !shortage.backend_kv_pages &&
            !shortage.host_bytes) {
            return false;
        }
        // An overlay Vision window holds Main KV pages only until its encode finishes. Waiting for
        // it is bounded and loses nothing, so it comes before evicting or pausing anything.
        if (shortage.main_kv_pages && instance_.program->drain_vision_window()) { return true; }
        if (instance_.program->reclaim_capture_reservation(shortage)) {
            request_admission_check();
            return true;
        }
        const auto progress =
            cursor ? resources_.reclaim(*instance_.program, shortage, excluded, *cursor)
                   : resources_.reclaim(*instance_.program, shortage, excluded);
        if (progress != ReclaimProgress::Blocked) {
            if (progress == ReclaimProgress::Changed) {
                scheduler_.capacity_released();
                request_admission_check();
            }
            return true;
        }
        if (cursor && cursor->rights.purpose != ReclaimPurpose::Execution) { return false; }
        for (auto it = paused_.rbegin(); it != paused_.rend(); ++it) {
            auto& request = **it;
            if (!request.suspended || !request.suspended->has_snapshot()) { continue; }
            const auto handle = *request.suspended->snapshot_handle();
            if (std::find(excluded.begin(), excluded.end(), handle) != excluded.end()) { continue; }
            const auto resources = instance_.program->snapshot_resources(*request.suspended);
            if (((shortage.state_slots && resources.state_slots) ||
                 (shortage.main_kv_pages && resources.main_kv_pages) ||
                 (shortage.backend_kv_pages && resources.backend_kv_pages) ||
                 (shortage.host_bytes && resources.host_bytes)) &&
                instance_.program->revoke_snapshot(*request.suspended)) {
                request.recovery_route = GenerationRecoveryRoute::Replay;
                observe_scheduling(*it, GenerationSchedulingTransition::SnapshotRevoked);
                if (admission_decision_ && admission_decision_->request.get() == &request) {
                    admission_decision_.reset();
                }
                request_admission_check();
                return true;
            }
        }
        return allow_pause && pause_reclaim_victim(priority_ticket);
    }

    bool pause_reclaim_victim(std::optional<std::uint64_t> priority_ticket) {
        const auto victim = scheduler_.youngest_victim(slots_, max_concurrency_, priority_ticket);
        if (!victim) { return false; }
        return pause_resident(*victim);
    }

    // `require_snapshot` refuses the pause unless the request's progress can be saved. A pause that
    // cannot save it makes the request recompute everything it had done, which is acceptable under
    // resource pressure and not for a pause that only serves the queue.
    bool pause_resident(std::uint32_t lane, bool require_snapshot = false) {
        if (instance_.program->has_context_transaction()) { return false; }
        if (slots_[lane] && slots_[lane]->slice_in_flight) { return false; }
        // An admission that pauses a resident has touched another request: a failure from here
        // on is no longer confined to the request being admitted.
        admission_subject_.reset();
        auto request       = slots_[lane];
        if (request && request->lossy && request->lossy->scouting()) {
            // start_pause refuses a scouting lane. A pause that only serves the queue waits for
            // the main pass; one for memory ends the scout, and the request continues in full.
            if (require_snapshot || !end_lossy_scout_for_pause(request)) { return false; }
        }
        bool save_snapshot = false;
        if (require_snapshot) {
            // A pause that only serves the queue is not worth evicting other requests' history, so
            // it saves a snapshot only where the image already fits: nothing is released, revoked or
            // marked for replay before the answer is known.
            save_snapshot = instance_.program->pause_host_bytes(*request->sequence).has_value() &&
                            instance_.program->pause_state_image_unplaced(*request->sequence) == 0;
        } else if (const auto bytes = instance_.program->pause_host_bytes(*request->sequence)) {
            const auto usage = instance_.program->physical_usage();
            if (*bytes <= usage.capacity.host_bytes &&
                *bytes > usage.capacity.host_bytes - usage.occupied.host_bytes) {
                (void)instance_.program->release_redundant_host({}, request->sequence);
            }
            std::vector<Checkpoint> later;
            for (auto it = paused_.rbegin(); it != paused_.rend(); ++it) {
                if ((*it)->id > request->id && (*it)->suspended &&
                    (*it)->suspended->has_snapshot()) {
                    later.push_back(*(*it)->suspended->snapshot_handle());
                }
            }
            // The victim search counts free bytes, but the state image needs one contiguous run:
            // when the freed bytes do not form one, ask for another image's worth and try again.
            std::size_t target = *bytes;
            for (std::size_t round = 0; round < kMaxPauseHostRounds; ++round) {
                const auto victims =
                    resources_.host_victims(*instance_.program, target, std::nullopt, later);
                if (!victims) { break; }
                for (const auto handle : *victims) {
                    const auto usage = instance_.program->physical_usage();
                    if (usage.capacity.host_bytes - usage.occupied.host_bytes >= target) { break; }
                    if (resources_.release_victim(*instance_.program, handle)) { continue; }
                    const auto owner =
                        std::find_if(paused_.begin(), paused_.end(), [&](const auto& other) {
                            return other->suspended &&
                                   other->suspended->snapshot_handle() == handle;
                        });
                    if (owner == paused_.end() ||
                        !instance_.program->revoke_snapshot(*(*owner)->suspended)) {
                        throw std::logic_error("pause Host victim changed after preflight");
                    }
                    (*owner)->recovery_route = GenerationRecoveryRoute::Replay;
                    observe_scheduling(*owner, GenerationSchedulingTransition::SnapshotRevoked);
                }
                const auto unplaced =
                    instance_.program->pause_state_image_unplaced(*request->sequence);
                if (unplaced == 0) {
                    save_snapshot = true;
                    break;
                }
                const auto usage = instance_.program->physical_usage();
                const auto next  = next_pause_host_target(
                    usage.capacity.host_bytes - usage.occupied.host_bytes, unplaced,
                    usage.capacity.host_bytes);
                if (!next) { break; }
                target = *next;
            }
        }
        if (require_snapshot && !save_snapshot) { return false; }
        ProgramCallScope call(*this);
        const auto pause_started = Clock::now();
        if (!instance_.program->start_pause(*request->sequence, save_snapshot,
                                            &call.failed_timing(), require_snapshot)) {
            return false;
        }
        call.finish(call.failed_timing());
        if (!request->is_replaying()) { request->resume_phase = request->model_state; }
        request->model_state = EngineRequestState::Pausing;
        request->paused_at   = pause_started;
        ++request->preemption_count;
        ++cumulative_stats_.preemptions;
        request->recovery_route = GenerationRecoveryRoute::None;
        observe_scheduling(request, GenerationSchedulingTransition::PauseStarted, pause_started);
        context_owner_ = request;
        capture_decisions_[lane].reset();
        scheduler_.preempted();
        return true;
    }

    void finish_source_wait(const std::shared_ptr<Request>& request,
                            Clock::time_point now) noexcept {
        if (!request->source_wait_started) { return; }
        request->admission.source_wait_seconds +=
            static_cast<double>(elapsed_ns(*request->source_wait_started, now)) * 1.0e-9;
        request->source_wait_started.reset();
    }

    // A wait must save at least this much prefill over the best source available now.
    static constexpr std::uint32_t kSiblingWaitMinimumTokens = 1024;

    [[nodiscard]] bool slice_in_flight_any() const noexcept {
        return std::any_of(slots_.begin(), slots_.begin() + max_concurrency_,
                           [](const auto& request) { return request && request->slice_in_flight; });
    }

    // Whether a lane other than `lane` is generating: its next token waits behind this lane's
    // prefill step.
    [[nodiscard]] bool other_lane_streaming(std::uint32_t lane) const noexcept {
        for (std::uint32_t other = 0; other < max_concurrency_; ++other) {
            const auto& request = slots_[other];
            if (other != lane && request && !request->cancelled.load(std::memory_order_acquire) &&
                (request->is_decode_ready() || request->is_control_ready())) {
                return true;
            }
        }
        return false;
    }

    // Prompt position the resident sibling has committed, or nullopt when it is not prefilling.
    [[nodiscard]] std::optional<std::uint32_t> sibling_cursor(const Request& sibling) const {
        if (!sibling.base_plan || sibling.cancelled.load(std::memory_order_acquire)) {
            return std::nullopt;
        }
        std::uint32_t cursor = 0;
        if (materializing_ && materializing_->request.get() == &sibling) {
            if (materializing_->resumed) { return std::nullopt; }
            cursor = materializing_->summary.reused_prompt_tokens;
        } else if (sibling.is_prefilling() && sibling.admitted_begin && sibling.lane) {
            // A replaying lane keeps its earlier computed_prompt_tokens while replay advances a
            // separate cursor, so only a lane in fresh prefill has a trustworthy position.
            cursor = sibling.admitted_begin->reused_prompt_tokens + sibling.computed_prompt_tokens;
        } else {
            return std::nullopt;
        }
        return cursor;
    }

    // Every request occupying a lane or the materializing slot; sibling_cursor filters to those
    // actually prefilling.
    [[nodiscard]] std::vector<std::shared_ptr<Request>> resident_siblings() const {
        std::vector<std::shared_ptr<Request>> out;
        if (materializing_ && !materializing_->resumed) { out.push_back(materializing_->request); }
        for (std::uint32_t lane = 0; lane < max_concurrency_; ++lane) {
            if (slots_[lane]) { out.push_back(slots_[lane]); }
        }
        return out;
    }

    // Single flight for concurrent siblings: a fresh request whose prompt shares a prefix with a
    // resident request still prefilling it, past a capture that resident has planned but not yet
    // published, waits for that checkpoint instead of prefilling the same tokens on another lane.
    // The wait is bounded by the predicted remaining prefill and is taken at most once.
    bool await_sibling(const std::shared_ptr<Request>& request,
                       const std::vector<typename ResourceManagement::SourceChoice>& sources) {
        if (request->sibling_wait_used || max_concurrency_ < 2 ||
            !request->base_plan->summary().publish_continuation) {
            return false;
        }
        const auto& base   = *request->base_plan;
        const auto prompt  = base.summary().prompt_tokens;
        std::uint32_t best = 0;
        for (const auto& choice : sources) { best = std::max(best, choice.source.reused_tokens); }
        std::optional<SiblingWait> chosen;
        for (const auto& sibling : resident_siblings()) {
            const auto cursor = sibling_cursor(*sibling);
            if (!cursor || !sibling->base_plan->summary().publish_continuation) { continue; }
            const auto& other = *sibling->base_plan;
            // Longest common prompt prefix, by digest bisection.
            std::uint32_t low = 0, high = std::min(prompt, other.summary().prompt_tokens);
            while (low < high) {
                const auto middle = low + (high - low + 1U) / 2U;
                const auto mine   = base.prefix_shortlist_key(middle);
                const auto theirs = other.prefix_shortlist_key(middle);
                if (mine && theirs && *mine == *theirs) {
                    low = middle;
                } else {
                    high = middle - 1U;
                }
            }
            // Only a SharedPrefix capture can be waited for: it is exposed to every request as soon
            // as it is published, while an InputReplay or LongAnchor point belongs to its owner and
            // ResourceManager::candidates() offers it only once that owner is no longer active, which
            // the sibling is not after publishing it. Deepest first, so a frontier too costly to
            // wait for falls back to a shallower one that still saves enough.
            auto frontiers = other.shared_prefix_frontiers();
            std::sort(frontiers.begin(), frontiers.end(),
                      [](std::uint32_t a, std::uint32_t b) { return a > b; });
            for (const auto frontier : frontiers) {
                if (frontier <= *cursor || frontier > low || frontier <= best ||
                    frontier < best + kSiblingWaitMinimumTokens) {
                    continue;
                }
                // The sibling's remaining prefill to the checkpoint, doubled because other lanes'
                // units interleave with it, bounds the wait. Compared against half the limit rather
                // than doubled first: the cost model returns UINT64_MAX when it saturates, and
                // doubling that would wrap into a short wait.
                const auto work              = instance_.program->prefill_work(other, *cursor, frontier);
                const std::uint64_t cost     = resources_.prefill_ns(work);
                const std::uint64_t limit_ns = static_cast<std::uint64_t>(
                    std::chrono::duration_cast<std::chrono::nanoseconds>(pending_timeout_ / 2).count());
                if (cost > limit_ns / 2) { continue; }
                if (!chosen || frontier > chosen->frontier) {
                    chosen = SiblingWait{request, sibling->id, frontier,
                                         Clock::now() + std::chrono::nanoseconds(2 * cost) +
                                             std::chrono::seconds(1)};
                }
                break;
            }
        }
        if (!chosen) { return false; }
        request->awaiting_sibling  = true;
        request->sibling_wait_used = true;
        if (!request->source_wait_started) { request->source_wait_started = Clock::now(); }
        sibling_waits_.push_back(std::move(*chosen));
        return true;
    }

    // Releases a waiter once its sibling has published or passed the awaited checkpoint, left
    // the lanes, or overran the predicted wait. The waiter is then offered for admission again.
    void process_sibling_waits() {
        if (sibling_waits_.empty()) { return; }
        const auto now      = Clock::now();
        const auto siblings = resident_siblings();
        std::erase_if(sibling_waits_, [&](SiblingWait& wait) {
            const auto found   = std::find_if(siblings.begin(), siblings.end(),
                                              [&](const auto& r) { return r->id == wait.sibling; });
            const auto cursor  = found == siblings.end() ? std::nullopt : sibling_cursor(**found);
            const bool pending = cursor && now < wait.deadline &&
                                 !wait.waiter->cancelled.load(std::memory_order_acquire) &&
                                 (*cursor < wait.frontier || (*found)->capture_pending);
            if (pending) { return false; }
            wait.waiter->awaiting_sibling     = false;
            wait.waiter->admission_generation = 0;
            request_admission_check();
            return true;
        });
    }

    void retain_admission_source(const std::shared_ptr<Request>& request,
                                 const typename ResourceManagement::SourceChoice& choice,
                                 bool restoring) {
        resources_.retain_source(*instance_.program, request->id, choice);
        if (!restoring && choice.source.reused_tokens && !request->source_wait_started) {
            request->source_wait_started = Clock::now();
        } else if (!restoring && !choice.source.reused_tokens) {
            finish_source_wait(request, Clock::now());
        }
    }

    // How long a paused prefill waits for the request it yielded to before it may be restored
    // anyway, in case that request cannot be admitted.
    static constexpr std::chrono::seconds kYieldHold{10};

    // With every lane busy, a request at the head of the queue that costs far less than what a
    // resident prefill still has to do pauses that prefill (prefill_yield.h). Returns whether a
    // pause started.
    bool yield_long_prefill_for_head() {
        std::shared_ptr<Request> head;
        {
            std::lock_guard lock(queue_mutex_);
            if (pending_.empty()) { return false; }
            head = pending_.front();
        }
        if (head->id == yield_beneficiary_ || head->hydrating || head->awaiting_sibling ||
            head->cancelled.load(std::memory_order_acquire)) {
            return false;
        }
        std::array<YieldCandidate, kMaximumConcurrency> field{};
        std::size_t count = 0;
        for (std::uint32_t lane = 0; lane < max_concurrency_; ++lane) {
            const auto& resident = slots_[lane];
            // A scouting lane cannot pause (its scout keeps uncommitted pages). The head waits
            // until the main pass starts; the step that ends the scout looks again.
            if (!resident || !resident->is_prefilling() || resident->capture_pending ||
                resident->slice_in_flight || resident->prompt_summary.has_media ||
                !resident->base_plan || (resident->lossy && resident->lossy->scouting())) {
                continue;
            }
            if (const auto cursor = sibling_cursor(*resident)) {
                field[count++] = {lane, prefill_remaining_ns(*resident, *cursor),
                                  resident->preemption_count};
            }
        }
        if (count == 0) { return false; }
        // Planning the head is what admission does next; a failure here belongs to it alone.
        admission_subject_ = head;
        ensure_base_plan(head);
        const std::uint32_t prompt = head->base_plan->summary().prompt_tokens;
        const std::uint64_t waiting_ns =
            resources_.prefill_ns(instance_.program->prefill_work(*head->base_plan, 0, prompt));
        const auto victim = choose_yield_victim(yield_min_remaining_ns_, waiting_ns,
                                                std::span<const YieldCandidate>(field.data(), count));
        if (!victim || !pause_resident(*victim, true)) { return false; }
        ++cumulative_stats_.prefill_yields;
        yield_beneficiary_ = head->id;
        yield_hold_        = true;
        yield_hold_until_  = Clock::now() + kYieldHold;
        return true;
    }

    [[nodiscard]] std::uint64_t prefill_remaining_ns(const Request& request,
                                                     std::uint32_t cursor) const {
        const std::uint32_t prompt = request.base_plan->summary().prompt_tokens;
        if (cursor >= prompt) { return 0; }
        if (request.lossy && request.lossy->before_outcome() && lossy_scout_share() > 0.0) {
            return lossy_remaining_ns(*request.base_plan, cursor, *request.lossy);
        }
        return resources_.prefill_ns(instance_.program->prefill_work(*request.base_plan, cursor, prompt));
    }

    // A prefill paused for a newcomer is restored behind it, not into the lane it just freed:
    // restoring is otherwise the first thing an empty set of lanes does. The hold ends when the
    // newcomer leaves the queue, or after kYieldHold if it never gets in.
    [[nodiscard]] bool restore_held_for_yield() {
        if (!yield_hold_) { return false; }
        bool waiting = false;
        {
            std::lock_guard lock(queue_mutex_);
            waiting = std::any_of(pending_.begin(), pending_.end(), [&](const auto& request) {
                return request->id == yield_beneficiary_ &&
                       !request->cancelled.load(std::memory_order_acquire);
            });
        }
        if (!waiting || Clock::now() >= yield_hold_until_) {
            yield_hold_ = false;
            return false;
        }
        return true;
    }

    bool try_admit_one(bool restoring) {
        if (instance_.program->has_context_transaction()) { return false; }
        const auto lane = free_lane();
        if (!lane && !restoring && yield_min_remaining_ns_ != 0 && yield_long_prefill_for_head()) {
            return true;
        }
        if (!lane && restoring) {
            if (!paused_.empty() && !has_older_resident(paused_.front()->id)) {
                if (const auto victim =
                        scheduler_.youngest_victim(slots_, max_concurrency_, paused_.front()->id)) {
                    return pause_resident(*victim);
                }
            }
            return false;
        }
        if (admission_decision_ &&
            (admission_decision_->restoring != restoring ||
             (restoring && (paused_.empty() || paused_.front() != admission_decision_->request)))) {
            admission_decision_.reset();
        }
        std::vector<std::shared_ptr<Request>> candidates;
        if (admission_decision_) {
            candidates.push_back(admission_decision_->request);
        } else if (restoring) {
            if (!paused_.empty()) { candidates.push_back(paused_.front()); }
        } else {
            std::lock_guard lock(queue_mutex_);
            candidates = scheduler_.fresh_candidates(pending_, max_concurrency_);
        }
        for (const auto& request : candidates) {
            // Until admission touches a resident lane (pause_resident clears it), a failure from
            // here on belongs to this request alone; see contain_admission_failure.
            admission_subject_ = request;
            const ReclaimRights rights{restoring ? ReclaimPurpose::Execution
                                                 : ReclaimPurpose::FreshAdmission,
                                       request->id};
            if (!admission_decision_) {
                if (!restoring) {
                    scheduler_.admission_candidate_checked(request->id);
                    // Its stored session is being read back; offered again once that settles.
                    if (request->hydrating || request->awaiting_sibling) { continue; }
                    if (request->admission_generation == admission_generation_) { continue; }
                    request->admission_generation = admission_generation_;
                    consume_armed_planning_failure();
                }
                ensure_base_plan(request);
                const auto retained = resources_.retained_source(request->id);
                const auto revoked  = resources_.source_revocations(request->id);
                if (!restoring && revoked > request->admission.revoked_checkpoints) {
                    request->admission.revoked_checkpoints = revoked;
                    if (!retained) {
                        request->admission.fallback_reason = AdmissionFallbackReason::SourceRevoked;
                    }
                }
                std::vector<typename ResourceManagement::SourceChoice> sources;
                if (restoring && request->suspended->has_snapshot()) {
                    auto& choice                 = sources.emplace_back();
                    choice.resume_owner          = request->continuation_owner
                                                       ? std::optional(request->continuation_owner)
                                                       : std::nullopt;
                    choice.source.private_points = resources_.points(request->continuation_owner);
                    choice.source.take_private =
                        resources_.can_transfer_private(request->continuation_owner, request->id);
                } else {
                    sources = resources_.candidates(*instance_.program, *request->base_plan,
                                                    restoring ? request->suspended->frontier()
                                                              : UINT32_MAX,
                                                    restoring && request->continuation_owner
                                                        ? std::optional(request->continuation_owner)
                                                        : std::nullopt,
                                                    request->id, request->publication_order);
                }
                if (!restoring && start_hydration(request, sources)) { continue; }
                if (!restoring && await_sibling(request, sources)) { continue; }
                if (!restoring && !sources.empty()) {
                    const auto preferred = sources.front().source.reused_tokens;
                    if (!request->admission_observed) {
                        request->admission.preferred_reused_tokens = preferred;
                        request->admission_observed                = true;
                    } else if (retained && preferred < retained->source.reused_tokens) {
                        request->admission.fallback_reason = AdmissionFallbackReason::CostChanged;
                    }
                }
                admission_decision_.emplace(AdmissionDecision{
                    request, restoring, std::move(sources), 0,
                    resources_.begin_reclaim(*instance_.program, std::nullopt, rights)});
            }
            auto& decision = *admission_decision_;
            ContextResourceUsage shortage;
            bool deferred = false;
            std::optional<std::size_t> first_unfit;
            while (decision.current < decision.sources.size()) {
                auto& choice            = decision.sources[decision.current];
                const bool own_snapshot = restoring && request->suspended->has_snapshot();
                if (!own_snapshot &&
                    !resources_.prepare_source(*instance_.program, *request->base_plan, choice,
                                               request->id)) {
                    if (!restoring) {
                        request->admission.fallback_reason =
                            resources_.source_revocations(request->id) &&
                                    !resources_.retained_source(request->id)
                                ? AdmissionFallbackReason::SourceRevoked
                                : AdmissionFallbackReason::SourceInvalid;
                    }
                    ++decision.current;
                    continue;
                }
                if (!lane) {
                    retain_admission_source(request, choice, restoring);
                    deferred = true;
                    break;
                }
                // A new selection replaces its old waiting reference before capacity is
                // checked. The replacement is retained before any old point is released.
                if (resources_.has_source_record(request->id)) {
                    retain_admission_source(request, choice, restoring);
                }
                if (!restoring && admission_fit_check_ && !resident_empty() &&
                    !instance_.program->prompt_fits_beside_residents(*request->base_plan,
                                                                    choice.source)) {
                    // Starting would fill the pool beside older requests, get this one paused and
                    // recompute its history later. Try the remaining sources (a cheaper reuse or a
                    // root start may fit); queue it only when none does.
                    if (!first_unfit) { first_unfit = decision.current; }
                    ++decision.current;
                    continue;
                }
                auto binding_started = Clock::now();
                const auto bind      = [&](const auto& source) {
                    return instance_.program->start_binding(
                        *request->base_plan, LaneId{*lane}, source,
                        restoring ? &*request->suspended : nullptr,
                        request->resume_phase == EngineRequestState::Prefill ? UnitKind::Prefill
                             : request->resume_phase == EngineRequestState::ControlReady
                                 ? UnitKind::Control
                                 : UnitKind::Decode,
                        request->resume_phase == EngineRequestState::Prefill ? 0U
                             : request->resume_phase == EngineRequestState::ControlReady
                                 ? static_cast<std::uint32_t>(
                                  request->output.pending_control_tokens().size())
                                 : 1U);
                };
                auto reservation = bind(choice.source);
                if (!reservation && !reservation.source_valid) {
                    if (!restoring) {
                        request->admission.fallback_reason =
                            resources_.source_revocations(request->id) &&
                                    !resources_.retained_source(request->id)
                                ? AdmissionFallbackReason::SourceRevoked
                                : AdmissionFallbackReason::SourceInvalid;
                    }
                    ++decision.current;
                    continue;
                }
                if (!reservation) {
                    shortage = reservation.shortage;
                    if (!own_snapshot) { retain_admission_source(request, choice, restoring); }
                    const auto protected_source = own_snapshot
                                                      ? request->suspended->snapshot_handle()
                                                      : choice.source.checkpoint;
                    const auto excluded         = protected_source
                                                      ? std::span<const Checkpoint>(&*protected_source, 1)
                                                      : std::span<const Checkpoint>{};
                    if (reclaim(shortage, restoring && !has_older_resident(request->id), excluded,
                                &decision.reclaim,
                                restoring ? std::optional(request->id) : std::nullopt)) {
                        if (instance_.program->has_context_transaction()) { return true; }
                        continue;
                    }
                    // Another waiting reference may be the only obstacle to a Move.
                    // Grant its revocation only when the resulting complete bind succeeds.
                    if (!own_snapshot) {
                        auto transferable = choice;
                        if (resources_.prepare_source(*instance_.program, *request->base_plan,
                                                      transferable, request->id, rights) &&
                            (transferable.source.consume_source != choice.source.consume_source ||
                             transferable.source.retired_points != choice.source.retired_points)) {
                            const auto transfer_started = Clock::now();
                            auto attempt                = bind(transferable.source);
                            if (attempt.source_valid) {
                                reservation.capacity_possible |= attempt.capacity_possible;
                            }
                            if (attempt) {
                                binding_started = transfer_started;
                                reservation     = std::move(attempt);
                                choice          = std::move(transferable);
                            }
                        }
                    }
                    if (!reservation) {
                        // Resident progress can release/unlock capacity. A root chunk's
                        // smaller initial allocation is not a reason to discard this source.
                        if (reservation.capacity_possible &&
                            (!resident_empty() || (!restoring && !paused_.empty()))) {
                            deferred = true;
                            break;
                        }
                        if (!restoring) {
                            request->admission.fallback_reason =
                                reservation.capacity_possible
                                    ? AdmissionFallbackReason::IsolatedCapacity
                                    : AdmissionFallbackReason::CapacityLimit;
                        }
                        ++decision.current;
                        continue;
                    }
                }
                resources_.binding_started(request->id, reservation.retired_points,
                                           reservation.consumed_source);
                const auto& source = choice.source;
                if (!restoring) {
                    plan_lossy_prefill(request, source);
                    const auto& planned = request->base_plan->summary();
                    // A prompt rewritten onto a lossy-prefilled lineage (the selection alias).
                    if (planned.reported_prompt_tokens > planned.prompt_tokens) {
                        ++cumulative_stats_.lossy_prefill_alias_hits;
                    }
                }
                const BeginSummary begin{
                    .prompt_tokens        = request->base_plan->summary().prompt_tokens,
                    .reused_prompt_tokens = source.reused_tokens,
                    .prefix_reuse_path =
                        source.reused_tokens ? PrefixReusePath::Checkpoint : PrefixReusePath::Root};
                if (restoring) {
                    std::erase(paused_, request);
                    observe_scheduling(request, GenerationSchedulingTransition::RestoreStarted,
                                       binding_started);
                } else {
                    std::lock_guard lock(queue_mutex_);
                    scheduler_.admitted(pending_, *request, max_concurrency_);
                    std::erase(pending_, request);
                }
                request->model_state = EngineRequestState::Materializing;
                materializing_.emplace(
                    MaterializingRequest{request, LaneId{*lane}, begin, restoring, choice});
                admission_decision_.reset();
                if (!restoring) {
                    settle_hydration(request, source.reused_tokens);
                    finish_source_wait(request, binding_started);
                    request->admission.revoked_checkpoints =
                        resources_.source_revocations(request->id);
                    request->admitted_at = binding_started;
                    request->host_timing.queue_wait_ns =
                        elapsed_ns(request->submitted, *request->admitted_at);
                    request->queue_wait_recorded = true;
                    publish_generation_start(request, begin);
                }
                return true;
            }
            if (!deferred && first_unfit) {
                // No source fits beside the residents; keep the preferred one and wait.
                retain_admission_source(request, decision.sources[*first_unfit], restoring);
                if (!request->fit_deferred) {
                    request->fit_deferred = true;
                    ++cumulative_stats_.admission_fit_deferrals;
                }
                deferred = true;
            }
            admission_decision_.reset();
            if (deferred) {
                if (restoring) {
                    scheduler_.restoration_blocked();
                    return false;
                }
                continue;
            }
            if (admission_check_pending_.load(std::memory_order_acquire)) { return true; }
            if (resident_empty()) {
                throw std::logic_error("isolated request cannot reserve its first legal unit after "
                                       "context reclamation: state=" +
                                       std::to_string(shortage.state_slots) +
                                       ", main_kv=" + std::to_string(shortage.main_kv_pages) +
                                       ", backend_kv=" + std::to_string(shortage.backend_kv_pages) +
                                       ", host_bytes=" + std::to_string(shortage.host_bytes));
            }
            if (restoring) {
                scheduler_.restoration_blocked();
                return false;
            }
        }
        return false;
    }

    void observe_scheduling(const std::shared_ptr<Request>& request,
                            GenerationSchedulingTransition transition, Clock::time_point now = {}) {
        if (!request->observation.scheduling) { return; }
        if (now == Clock::time_point{}) { now = Clock::now(); }
        const GenerationSchedulingObservation observation{
            .transition              = transition,
            .route                   = request->recovery_route,
            .engine_request_id       = request->id,
            .steady_ns               = elapsed_ns(Clock::time_point{}, now),
            .elapsed_ns              = elapsed_ns(request->submitted, now),
            .preemption_index        = request->preemption_count,
            .global_prefill_tokens   = cumulative_stats_.computed_prefill_tokens,
            .global_decode_tokens    = cumulative_stats_.committed_decode_tokens,
            .global_replayed_tokens  = cumulative_stats_.replayed_tokens,
            .request_prefill_tokens  = request->computed_prompt_tokens,
            .request_decode_tokens   = request->committed_decode_tokens,
            .request_replayed_tokens = request->replayed_tokens,
        };
        {
            std::lock_guard lock(request->mutex);
            request->events.emplace_back(
                std::make_unique<GenerationSchedulingObservation>(observation));
        }
        request->cv.notify_one();
    }

    void update_recovery(const std::shared_ptr<Request>& request, bool released = false) {
        if (!request->recovery_pending) { return; }
        request->recovery_pending = !released && request->sequence &&
                                    instance_.program->recovery_pending(*request->sequence);
        if (!request->recovery_pending) {
            if (!released) {
                observe_scheduling(request, GenerationSchedulingTransition::RecoveryComplete);
            }
            scheduler_.capacity_released();
            request_admission_check();
        }
    }

    // The optional Vision requirement of the licensed prefill unit (overlay residency). It is
    // reserved after every required unit of the round, so it can only take pages nobody executing
    // needs, and a shortage is answered by reclaiming cached content alone: never by revoking a
    // paused snapshot or pausing a resident. When that does not suffice the unit encodes in a
    // window of its own.
    void reserve_vision_window(const Request& request) {
        std::optional<typename ResourceManagement::ReclaimCursor> cursor;
        for (;;) {
            const auto result = instance_.program->reserve_vision_window(*request.sequence);
            if (result || !result.shortage.main_kv_pages ||
                instance_.program->has_context_transaction()) {
                return;
            }
            if (!cursor) { cursor.emplace(resources_.begin_reclaim(*instance_.program)); }
            const auto progress =
                resources_.reclaim(*instance_.program, result.shortage, {}, *cursor);
            if (progress != ReclaimProgress::Changed) { return; }
            scheduler_.capacity_released();
            request_admission_check();
        }
    }

    enum class ReservationScope { Round, Prefill };

    void reserve_resident_units(ReservationScope scope = ReservationScope::Round) {
        // Finished overlay Vision encodes give their Main KV pages back before anything reserves.
        (void)instance_.program->poll_vision();
        runnable_units_.fill(false);
        std::optional<std::uint32_t> recovering;
        std::optional<std::uint32_t> oldest;
        for (std::uint32_t lane = 0; lane < max_concurrency_; ++lane) {
            const auto& request = slots_[lane];
            if (!request || !request->sequence || request->is_model_finished()) { continue; }
            if (!oldest || request->id < slots_[*oldest]->id) { oldest = lane; }
            if (request->recovery_pending) {
                if (recovering) { throw std::logic_error("multiple protected recoveries"); }
                recovering = lane;
            }
        }
        const auto prefill = recovering && (slots_[*recovering]->is_prefilling() ||
                                            slots_[*recovering]->is_replaying())
                                 ? recovering
                                 : scheduler_.next_prefill(slots_, max_concurrency_);
        std::vector<std::uint32_t> candidates;
        for (std::uint32_t lane = 0; lane < max_concurrency_; ++lane) {
            const auto& request = slots_[lane];
            if (!request || request->capture_pending || !request->sequence ||
                request->cancelled.load(std::memory_order_acquire) ||
                instance_.program->context_blocks(*request->sequence)) {
                continue;
            }
            if (prefill == lane || (scope == ReservationScope::Round &&
                                    (request->is_control_ready() || request->is_decode_ready()))) {
                candidates.push_back(lane);
            }
        }
        std::sort(candidates.begin(), candidates.end(), [&](auto a, auto b) {
            if (a == b) { return false; }
            if (recovering == a || recovering == b) { return recovering == a; }
            return slots_[a]->id < slots_[b]->id;
        });
        std::optional<std::uint32_t> blocked;
        for (const auto lane : candidates) {
            const auto request = slots_[lane];
            // Reclaiming for an earlier row may already have paused this candidate.
            if (!request || !request->sequence || request->capture_pending ||
                request->cancelled.load(std::memory_order_acquire) ||
                instance_.program->context_blocks(*request->sequence) ||
                !(request->is_control_ready() || request->is_decode_ready() ||
                  request->is_prefilling() || request->is_replaying())) {
                continue;
            }
            const ExecutionUnit unit{
                *request->sequence,
                request->is_control_ready()  ? UnitKind::Control
                : request->is_decode_ready() ? UnitKind::Decode
                : request->is_replaying()    ? UnitKind::Replay
                                             : UnitKind::Prefill,
                request->is_control_ready()
                    ? static_cast<std::uint32_t>(request->output.pending_control_tokens().size())
                : request->is_decode_ready()
                    ? (request->recovery_pending ? 1U
                                                 : request->output.model_token_budget_remaining(
                                                       request->budget->remaining()))
                    : 0U};
            std::optional<typename ResourceManagement::ReclaimCursor> cursor;
            for (;;) {
                const auto result = instance_.program->reserve_units({&unit, 1});
                if (result) {
                    runnable_units_[lane] = true;
                    break;
                }
                if (instance_.program->has_context_transaction()) { break; }
                if (!cursor) { cursor.emplace(resources_.begin_reclaim(*instance_.program)); }
                if (reclaim(result.shortage, false, {}, &*cursor)) {
                    if (!instance_.program->has_context_transaction()) { continue; }
                    break;
                }
                if (recovering == lane) {
                    throw std::logic_error("protected recovery exceeded its Native reservation");
                }
                if (!recovering && oldest == lane) {
                    // Only the oldest required unit can displace younger residents. Other
                    // failed rows wait or yield their own lane; they cannot block ready rows.
                    // The same shortage has already exhausted cache and paused snapshots.
                    if (pause_reclaim_victim(request->id)) { break; }
                    // A lane part-way through a sliced chunk cannot be paused. The chunk finishes
                    // within a few cycles; this unit is tried again then.
                    if (slice_in_flight_any()) { break; }
                    throw std::logic_error("oldest resident cannot obtain its legal unit");
                }
                if (oldest != lane && (!blocked || request->id > slots_[*blocked]->id)) {
                    blocked = lane;
                }
                break;
            }
        }
        if (blocked && !instance_.program->has_context_transaction()) {
            (void)pause_resident(*blocked);
        }
        if (prefill && runnable_units_[*prefill] && !prepare_semantic_capture(slots_[*prefill])) {
            runnable_units_[*prefill] = false;
        }
        if (prefill && runnable_units_[*prefill] && slots_[*prefill]->is_prefilling()) {
            reserve_vision_window(*slots_[*prefill]);
        }
        // A pause may have selected an already licensed younger row. Its permit belongs
        // to Native cleanup, and it must not enter this cycle's compact execution batch.
        for (std::uint32_t lane = 0; lane < max_concurrency_; ++lane) {
            if (!slots_[lane] || slots_[lane]->model_state == EngineRequestState::Pausing ||
                (slots_[lane]->sequence &&
                 instance_.program->context_blocks(*slots_[lane]->sequence))) {
                runnable_units_[lane] = false;
            }
        }
    }

    void run_decode_round(const RoundMembership& membership,
                          const std::array<bool, kMaximumConcurrency>& cancelled_at_unit_start) {
        nvtx::ScopedRange decode_range(nvtx::Name::Decode, nvtx::Category::Decode,
                                       static_cast<std::uint64_t>(membership.size));
        ProgramCallScope program_call(*this);
        RoundMasks masks;
        bool constrained = false;
        for (std::size_t row = 0; row < membership.size; ++row) {
            masks.outputs[row] = &slots_[membership.lane_span()[row]]->output;
            constrained |= masks.outputs[row]->constrained();
        }
        auto pending = instance_.program->decode(
            membership.sequence_span(), membership.budget_span(), &program_call.failed_timing(),
            constrained ? &masks : nullptr);
        program_call.finish(pending.execution_timing());
        cumulative_stats_.decode_seconds_total +=
            static_cast<double>(pending.execution_timing().elapsed_ns()) * 1e-9;
        consume_armed_decode_failure();
        commit_pending(std::move(pending), membership.lane_span(), true, cancelled_at_unit_start);
        publish_runtime_stats();
    }

    void run_control_batch(const ControlMembership& membership) {
        nvtx::ScopedRange control_range(nvtx::Name::ControlBatch, nvtx::Category::Control,
                                        static_cast<std::uint64_t>(membership.size));
        EnginePhaseScope phase(*this, EngineHostPhase::CommitOutput);
        if (membership.empty() || membership.row_stride == 0 ||
            membership.tokens.size() !=
                static_cast<std::size_t>(membership.row_stride) * membership.size) {
            throw std::logic_error("thinking control membership is invalid");
        }

        std::array<std::size_t, kMaximumConcurrency> generated_sizes{};
        std::array<std::optional<std::uint32_t>, kMaximumConcurrency> prefix_execution_splits{};
        bool generated_staged         = false;
        const auto rollback_generated = [&]() {
            if (!generated_staged) { return; }
            for (std::size_t row = 0; row < membership.size; ++row) {
                const auto& request = slots_[membership.lanes[row]];
                if (request != nullptr) { request->output.discard_preview(); }
                if (request != nullptr && request->generated.size() >= generated_sizes[row]) {
                    request->generated.resize(generated_sizes[row]);
                }
            }
            generated_staged = false;
        };

        for (std::size_t row = 0; row < membership.size; ++row) {
            const auto& request = slots_[membership.lanes[row]];
            if (request == nullptr) {
                throw std::logic_error("thinking control membership lost its request");
            }
            generated_sizes[row] = request->generated.size();
        }
        generated_staged = true;
        try {
            for (std::size_t row = 0; row < membership.size; ++row) {
                const std::uint32_t lane = membership.lanes[row];
                const auto& request      = slots_[lane];
                if (request == nullptr || !request->is_control_ready() ||
                    request->capture_pending || !request->budget || !request->sequence ||
                    !request->lane || request->lane->value != lane) {
                    throw std::logic_error("thinking control row lost its active request");
                }
                const std::span<const TokenId> tokens =
                    std::span<const TokenId>(membership.tokens)
                        .subspan(row * membership.row_stride, membership.row_stride);
                const OutputDecision decision =
                    request->output.preview_control(tokens, request->budget->remaining());
                if (decision.accepted_tokens != membership.row_stride || decision.finished() ||
                    decision.continuation != ContinuationAction::Decode ||
                    (decision.prefix_execution_split_after &&
                     (*decision.prefix_execution_split_after == 0 ||
                      *decision.prefix_execution_split_after > decision.accepted_tokens))) {
                    throw std::logic_error("target control preview returned an invalid decision");
                }
                prefix_execution_splits[row] = decision.prefix_execution_split_after;
                if (request->generated.size() > request->generated.capacity() ||
                    tokens.size() > request->generated.capacity() - request->generated.size()) {
                    throw std::logic_error(
                        "admission did not reserve thinking-control token capacity");
                }
                request->generated.insert(request->generated.end(), tokens.begin(), tokens.end());
            }
            phase.pause_range();
            ProgramCallScope program_call(*this);
            const runtime::ExecutionTiming timing = instance_.program->append_forced_tokens(
                membership.sequence_span(), membership.tokens, membership.row_stride,
                std::span<const std::optional<std::uint32_t>>(prefix_execution_splits.data(),
                                                              membership.size),
                &program_call.failed_timing());
            program_call.finish(timing);
            // Forced control tokens are counted as committed decode tokens below, so their
            // execution time belongs to the decode total as well.
            cumulative_stats_.decode_seconds_total +=
                static_cast<double>(timing.elapsed_ns()) * 1e-9;
            phase.resume_range();
        } catch (...) {
            rollback_generated();
            throw;
        }
        generated_staged = false;

        ++cumulative_stats_.host_work.control_units;
        for (const std::uint32_t lane : membership.lane_span()) {
            ++slots_[lane]->host_timing.control_units;
            slots_[lane]->committed_decode_tokens += membership.row_stride;
        }
        cumulative_stats_.committed_decode_tokens += membership.size * membership.row_stride;

        for (std::size_t row = 0; row < membership.size; ++row) {
            const std::uint32_t lane = membership.lanes[row];
            const auto& request      = slots_[lane];
            request->budget->commit(membership.row_stride);
            auto timing = record_committed_output(request, membership.row_stride);
            append_output(request, request->output.commit_preview(), std::move(timing), &phase);
            update_recovery(request);
            request->model_state = EngineRequestState::DecodeReady;
        }
        publish_runtime_stats();
    }

    // The latch: every request fails and the Engine serves nothing more. Taken when recovery is
    // refused (device fault, failure streak, or a cleanup that cannot be verified) and at shutdown.
    // The worker holds execution_mutex_ across the failing operation and this cleanup, so no
    // Program introspection can observe a partially cleared physical state.
    // With a `fault`, the ids of the queued and paused requests being failed are recorded into
    // it, the queued ones from the very set swapped out under the lock, so a request enqueued
    // concurrently is either in the event or was rejected by failed_ -- never failed but
    // unreported.
    void fail_all_locked(std::exception_ptr error, EngineFaultEvent* fault = nullptr) noexcept {
        std::deque<std::shared_ptr<Request>> pending;
        {
            std::lock_guard lock(queue_mutex_);
            failed_ = true;
            pending.swap(pending_);
        }
        if (fault != nullptr) {
            try {
                for (const auto& request : pending) {
                    fault->queued_request_ids.push_back(request->id);
                }
                for (const auto& request : paused_) {
                    fault->queued_request_ids.push_back(request->id);
                }
            } catch (...) {}
        }
        scheduler_ = Scheduling{};
        // A latched engine never runs process_sibling_waits() again. Each waiter is in `pending`
        // (or `paused_`) and is completed with the error below, which finishes its source wait; the
        // list only has to stop holding the requests alive.
        for (auto& wait : sibling_waits_) {
            wait.waiter->awaiting_sibling     = false;
            wait.waiter->admission_generation = 0;
        }
        sibling_waits_.clear();
        const std::shared_ptr<Request> materializing_request =
            materializing_ ? materializing_->request : nullptr;
        // Native retires all transfer readers before any external checkpoint owner is destroyed.
        instance_.program->fail_all_cleanup();
        materializing_.reset();
        admission_decision_.reset();
        for (auto& decision : capture_decisions_) { decision.reset(); }
        context_owner_.reset();
        for (auto& request : paused_) {
            request->suspended.reset();
            complete_error(request, error);
        }
        paused_.clear();
        resources_.release_all(*instance_.program);
        try {
            restore_external_sources();
        } catch (...) {}
        for (std::uint32_t lane = 0; lane < max_concurrency_; ++lane) {
            if (slots_[lane] != nullptr) {
                complete_error(slots_[lane], error);
                slots_[lane].reset();
            }
        }
        if (materializing_request != nullptr) { complete_error(materializing_request, error); }
        for (const auto& request : pending) { complete_error(request, error); }
        publish_runtime_stats();
    }

    // Installed external context (direct prompt grafts) must stay bound to its pinned checkpoints
    // whenever requests can be admitted. Reinstalls any graft the Program no longer holds and
    // re-advertises every one as a pinned shared prefix. Runs at construction, after the context
    // cache is cleared, and is the hook a worker recovery calls once it has re-established an idle,
    // empty physical baseline. Requires an idle Program.
    void restore_external_sources() {
        instance_.install_external_checkpoints();
        for (const auto handle : instance_.external_checkpoints) {
            resources_.pin_shared(*instance_.program, handle);
        }
    }

    // On refusal `refusal` names why, for the operator log. Assigning it allocates and the callers
    // are noexcept: a failed allocation leaves the refusal empty rather than terminating the
    // process before the latch is published.
    static void set_refusal(std::string& refusal, std::string_view reason,
                            std::string_view detail = {}) noexcept {
        try {
            refusal.assign(reason);
            if (!detail.empty()) {
                refusal += ": ";
                refusal += detail;
            }
        } catch (...) { refusal.clear(); }
    }

    // Worker-only. Drains the device and reports whether it is usable; on a fault `detail` names
    // the CUDA error. Both recovery and the latch free physical state that work issued before the
    // failure may still read, so every failure path runs this first.
    [[nodiscard]] bool device_healthy(std::string& detail) noexcept {
        if (consume_armed_device_fault()) {
            set_refusal(detail, "injected device fault");
            return false;
        }
        const cudaError_t status = device_.synchronize_status();
        if (status == cudaSuccess) { return true; }
        set_refusal(detail, cudaGetErrorName(status), cudaGetErrorString(status));
        return false;
    }

    // Worker-only. Ends the Program's open context transaction through its own cancellation path
    // (the one a cancelled owner takes), after the device has been drained. True when none is left.
    [[nodiscard]] bool abort_context_transaction() {
        const std::atomic<bool> cancelled{true};
        for (int poll = 0; poll < 4 && instance_.program->has_context_transaction(); ++poll) {
            auto progress = instance_.program->poll_context(CancellationFlagView{&cancelled});
            if (progress.complete) { record_context_work(progress, nullptr); }
        }
        return !instance_.program->has_context_transaction();
    }

    // A throw while admitting one waiting or paused request -- planning it, choosing and
    // preparing its source, reserving or starting its binding -- before admission touched any
    // resident lane belongs to that request alone. Fail it, end the binding it may have started,
    // release what it held in the context cache, and leave every other request running, queued
    // or paused. Not a recovery: the streak and RuntimeStats::engine_recoveries are untouched.
    // Returns false, leaving the failure to recover_locked, when the subject is unknown, the
    // device is not healthy (then `device_fault` is set) or the containment itself fails.
    [[nodiscard]] bool contain_admission_failure(std::exception_ptr error, bool restoring,
                                                 std::optional<std::string>& device_fault) noexcept {
        const std::shared_ptr<Request> request = std::exchange(admission_subject_, nullptr);
        if (!request) { return false; }
        std::string detail;
        if (!device_healthy(detail)) {
            device_fault = std::move(detail);
            return false;
        }
        try {
            admission_decision_.reset();
            const bool materializing = materializing_ && materializing_->request == request;
            // Admission opens no transaction while another is open, so one now open and owned by
            // no one (neither materializing nor a pausing/capturing resident) was started by this
            // admission: its binding, or a demotion its reclaim began. Both end harmlessly.
            if (materializing ||
                (instance_.program->has_context_transaction() && !materializing_ &&
                 !context_owner_)) {
                if (!abort_context_transaction()) { return false; }
            }
            if (materializing) {
                materializing_.reset();
                scheduler_.capacity_released();
            } else if (materializing_) {
                return false;
            }
            if (restoring) {
                std::erase(paused_, request);
            } else {
                std::lock_guard lock(queue_mutex_);
                std::erase(pending_, request);
            }
            complete_error(request, error);
            request_admission_check();
            publish_runtime_stats();
        } catch (...) { return false; }
        EngineFaultEvent fault;
        try {
            fault.message   = exception_text(error);
            fault.unit      = "admission";
            fault.contained = true;
            fault.request_ids.push_back(request->id);
        } catch (...) {}
        notify_fault(fault);
        return true;
    }

    // A host-side failure in the worker (an invariant or capacity error from the context cache, a
    // planner, or a unit's host bookkeeping) fails only the requests it could have corrupted --
    // the running lanes, including one pausing or capturing, and the one materializing -- and
    // keeps the queue and the paused requests. The Program ends its open context transaction,
    // pending batch and every lane (fail_all_cleanup, after the caller drained the device); the
    // failed requests release their sources and continuations. The reusable context cache is
    // cleared, and each paused request gives up its snapshot and continuation points and will
    // restore by replay, so nothing optional survives that could hide a leak: the Program's
    // physical usage must then equal the quiescent baseline taken after startup exactly. Any
    // cleanup error or difference refuses the recovery, and the caller latches. The caller has
    // already checked the device and the streak (recovery_refusal).
    [[nodiscard]] bool recover_locked(std::exception_ptr error, std::string& refusal,
                                      Clock::time_point failed_at) noexcept {
        try {
            std::vector<std::shared_ptr<Request>> affected;
            for (std::uint32_t lane = 0; lane < max_concurrency_; ++lane) {
                if (slots_[lane] != nullptr) { affected.push_back(slots_[lane]); }
            }
            if (materializing_ && materializing_->request != nullptr) {
                affected.push_back(materializing_->request);
            }
            instance_.program->fail_all_cleanup();
            // Until every affected request is completed, slots_ and materializing_ keep naming
            // them, so a refusal's latch still delivers the error to each.
            for (const auto& request : affected) { complete_error(request, error); }
            materializing_.reset();
            context_owner_.reset();
            admission_decision_.reset();
            admission_subject_.reset();
            for (auto& decision : capture_decisions_) { decision.reset(); }
            runnable_units_.fill(false);
            for (auto& slot : slots_) { slot.reset(); }
            for (const auto& request : paused_) {
                if (request->suspended && request->suspended->has_snapshot()) {
                    if (!instance_.program->revoke_snapshot(*request->suspended)) {
                        throw std::logic_error("a paused snapshot could not be released");
                    }
                    request->recovery_route = GenerationRecoveryRoute::Replay;
                    observe_scheduling(request, GenerationSchedulingTransition::SnapshotRevoked);
                }
                // Released with the whole cache below; the restore adopts a new owner.
                request->continuation_owner = 0;
            }
            // Owners, shared prefixes and the sources queued requests retained.
            resources_.release_all(*instance_.program);
            // Every continuation is gone; what the store holds stays valid and is read back on
            // demand (hydration), so only the marks of the released owners are dropped.
            owner_marks_.clear();
            scheduler_.capacity_released();
        } catch (...) {
            set_refusal(refusal, "recovery cleanup failed", exception_text(std::current_exception()));
            return false;
        }
        const auto usage = instance_.program->physical_usage();
        if (instance_.program->has_context_transaction()) {
            set_refusal(refusal, "recovery left a context transaction open");
            return false;
        }
        if (!quiescent_usage_ || !returned_to_baseline(usage, *quiescent_usage_)) {
            std::string difference;
            try {
                if (quiescent_usage_) {
                    difference = describe_usage_difference(usage, *quiescent_usage_);
                }
            } catch (...) {}
            set_refusal(refusal,
                        "physical usage did not return to the startup baseline (now/baseline)",
                        difference);
            return false;
        }
        // The cache is empty and verified: put back what startup made available to every request.
        try {
            restore_external_sources();
        } catch (...) {
            set_refusal(refusal, "external sources could not be restored",
                        exception_text(std::current_exception()));
            return false;
        }
        pacing_.reset_decode_debt();
        recovery_streak_.record_recovery(failed_at);
        ++cumulative_stats_.engine_recoveries;
        request_admission_check();
        return true;
    }

    // Worker-only, before cleanup clears them: the requests a failure is delivered to.
    void collect_fault_requests(EngineFaultEvent& event) const {
        for (std::uint32_t lane = 0; lane < max_concurrency_; ++lane) {
            if (slots_[lane] != nullptr) {
                event.request_ids.push_back(slots_[lane]->id);
                event.lanes.push_back(lane);
            }
        }
        if (materializing_ && materializing_->request != nullptr) {
            event.request_ids.push_back(materializing_->request->id);
        }
    }

    void notify_fault(const EngineFaultEvent& event) const noexcept {
        if (!fault_listener_) { return; }
        try {
            fault_listener_(event);
        } catch (...) {}
    }

    // One admission attempt; a failure confined to its request is contained, any other rethrown.
    void admit_or_contain(bool restoring, std::optional<std::string>& device_fault) {
        admission_subject_.reset();
        // A queued write that failed since the last scan left its marks standing; clear them
        // before admission prices eviction, or the failed image is counted as durable. Images the
        // store removed on its own while the worker slept are forgotten the same way.
        (void)forget_failed_store_writes();
        forget_removed_store_images();
        try {
            (void)try_admit_one(restoring);
        } catch (...) {
            if (!contain_admission_failure(std::current_exception(), restoring, device_fault)) {
                throw;
            }
        }
        admission_subject_.reset();
    }

    // ---- Durable context store ----------------------------------------------------------------

    struct OwnerMark {
        // The points of the image last written (or queued) for this continuation and its size.
        // Checkpoint contents never change under a handle, and handles carry a generation, so a
        // continuation whose current points are all among them is held by the store as it is now.
        std::vector<Checkpoint> stored;
        std::uint64_t stored_bytes = 0;
        // The store id of that image.
        std::string image_id;
        // The points of an image the store removed on its own (space, age, damage, or superseded).
        // A continuation still at exactly these points is not written again: the store chose to
        // let it go, and writing it back would only evict another image, which would then be
        // written back in turn.
        std::vector<Checkpoint> dropped;
        Clock::time_point last_used{};
    };

    [[nodiscard]] static bool covers(const std::vector<Checkpoint>& set,
                                     std::span<const Checkpoint> points) {
        return !set.empty() && !points.empty() &&
               std::all_of(points.begin(), points.end(), [&](const auto point) {
                   return std::find(set.begin(), set.end(), point) != set.end();
               });
    }

    [[nodiscard]] static bool holds(const OwnerMark& entry, std::span<const Checkpoint> points) {
        return covers(entry.stored, points);
    }

    // Not to be written: the store holds the continuation as it is now, or let it go as it is now.
    [[nodiscard]] static bool settled(const OwnerMark& entry, std::span<const Checkpoint> points) {
        return holds(entry, points) || covers(entry.dropped, points);
    }

    // The store id of the image of `points`, as ContextStore::image_id names it: the deepest point's
    // key (an image lists its points deepest first). Empty when there is no such key; a mark with
    // no id is found missing at the next check and treated like an image the store let go.
    template <class Program>
    [[nodiscard]] static std::string store_image_id(const Program& program,
                                                    std::span<const Checkpoint> points,
                                                    bool shared) noexcept {
        try {
            if (points.empty()) { return {}; }
            Checkpoint deepest     = points.front();
            std::uint32_t frontier = program.checkpoint_metadata(deepest).frontier;
            for (const auto point : points) {
                const auto depth = program.checkpoint_metadata(point).frontier;
                if (depth > frontier) {
                    deepest  = point;
                    frontier = depth;
                }
            }
            return ContextStore::image_id(program.checkpoint_key(deepest, frontier).digests,
                                          frontier, shared);
        } catch (...) { return {}; }
    }

    void record_stored(ContinuationOwnerToken token, std::span<const Checkpoint> points,
                       std::uint64_t bytes) {
        auto& entry        = mark(token);
        entry.stored       = std::vector<Checkpoint>(points.begin(), points.end());
        entry.stored_bytes = bytes;
        entry.image_id     = store_image_id(*instance_.program, points, false);
        entry.dropped.clear();
    }

    OwnerMark& mark(ContinuationOwnerToken token) {
        auto [found, inserted] = owner_marks_.try_emplace(token);
        if (inserted) { found->second.last_used = Clock::now(); }
        return found->second;
    }


    // Images the store removed on its own (space, age, damage, or superseded by a longer image).
    // When it grows, each mark is checked against the store: a mark whose image is gone is
    // dropped, and the others stand. Clearing every mark instead had every idle conversation
    // written again after any turn of any other (each write supersedes the conversation's previous
    // image), and each rewrite evicted another image from a full store, which cleared them again.
    [[nodiscard]] std::uint64_t store_removal_count() const noexcept {
        if (store_.store == nullptr) { return 0; }
        try {
            const auto stats = store_.store->stats();
            return stats.evicted_for_space + stats.expired + stats.corrupt_removed +
                   stats.remote_removed + stats.superseded;
        } catch (...) { return store_removals_seen_; }
    }

    void forget_removed_store_images() noexcept {
        const auto removed = store_removal_count();
        if (removed == store_removals_seen_) { return; }
        store_removals_seen_ = removed;
        drop_removed_store_marks();
    }

    // A write that was accepted into the queue and then failed left its continuation marked. The
    // writer names the images that failed: each is written again unless the store holds it after
    // all, even if a removal scan meanwhile took it for an image the store let go.
    bool forget_failed_store_writes() noexcept {
        if (!store_.failures) { return false; }
        const auto failed = store_.failures();
        FailedStoreWrites reported;
        try {
            if (store_.failed_writes) {
                reported = store_.failed_writes();
            } else if (failed != store_failures_seen_) {
                reported.lost = true; // no ids to go by
            }
        } catch (...) { reported.lost = failed != store_failures_seen_; }
        if (failed == store_failures_seen_ && reported.ids.empty() && !reported.lost) {
            return false;
        }
        store_failures_seen_ = failed;
        retry_failed_store_writes(reported);
        return true;
    }

    // The buffer to export the next image into (empty when the writer has none to hand back).
    [[nodiscard]] std::vector<std::uint8_t> export_buffer() noexcept {
        try {
            if (store_.export_buffer) { return store_.export_buffer(); }
        } catch (...) {}
        return {};
    }

    // Whether the store holds the image `id` for this Engine's model and configuration, or the
    // Engine's writer still has it to write. A store that cannot answer is taken to hold it, so a
    // failing lookup never sets off rewrites.
    [[nodiscard]] bool store_has_image(const std::string& id) const noexcept {
        if (id.empty()) { return false; }
        try {
            if (store_.queued && store_.queued(id)) { return true; }
            const auto info = store_.store->find(id);
            return info && info->binding == store_.binding;
        } catch (...) { return true; }
    }

    // After the store removed images on its own: a mark whose image is gone is dropped (written
    // again only once its continuation has changed, OwnerMark::dropped); the others stand.
    void drop_removed_store_marks() noexcept {
        try {
            auto& program = *instance_.program;
            for (auto& [token, entry] : owner_marks_) {
                if (entry.stored.empty() || store_has_image(entry.image_id)) { continue; }
                entry.dropped = std::move(entry.stored);
                entry.stored.clear();
                entry.stored_bytes = 0;
            }
            std::erase_if(shared_stored_, [&](const auto& stored) {
                if (!program.valid_checkpoint(stored.first)) { return true; }
                const Checkpoint handle = stored.first;
                if (store_has_image(store_image_id(program, {&handle, 1}, true))) { return false; }
                shared_dropped_.push_back(handle);
                return true;
            });
        } catch (...) {} // marks not yet checked stand: a stale one costs a hit, never an answer
    }

    // After queued writes failed: their continuations are no longer stored, and not dropped
    // either, so the next scan writes them again. With ids lost, every continuation whose image the
    // store does not hold is treated so.
    void retry_failed_store_writes(const FailedStoreWrites& reported) noexcept {
        try {
            const auto& ids   = reported.ids;
            const auto failed = [&](const std::string& id) {
                return !id.empty() &&
                       (reported.lost || std::find(ids.begin(), ids.end(), id) != ids.end()) &&
                       !store_has_image(id);
            };
            auto& program = *instance_.program;
            for (auto& [token, entry] : owner_marks_) {
                if (!failed(entry.image_id)) { continue; }
                entry.stored.clear();
                entry.stored_bytes = 0;
                entry.dropped.clear();
            }
            const auto shared_failed = [&](Checkpoint handle) {
                return program.valid_checkpoint(handle) &&
                       failed(store_image_id(program, {&handle, 1}, true));
            };
            std::erase_if(shared_stored_,
                          [&](const auto& stored) { return shared_failed(stored.first); });
            std::erase_if(shared_dropped_, shared_failed);
        } catch (...) {}
    }

    // Writes at most one idle retained continuation to the store. Called by the worker between
    // units, never while a request waits for admission, binds, prefills or replays, so keeping
    // the store current does not delay a request already waiting; at least a second apart.
    void persist_idle_session() noexcept {
        if (!store_.sink) { return; }
        const auto now = Clock::now();
        if (now < next_persist_scan_) { return; }
        next_persist_scan_ = now + std::chrono::seconds(1);
        try {
            // Marks are reconciled even with idle writes off: eviction pricing and
            // write-on-evict read them whether or not this worker ever writes.
            const bool failing = forget_failed_store_writes();
            forget_removed_store_images();
            // With idle writes off the scan still runs to the pruning below: write-on-evict adds a
            // mark for every owner it is offered, and nothing else ever erases them.
            const bool idle_writes = store_.idle_persist.count() > 0;
            if (idle_writes && failing) {
                // A failing disk: do not export deep sessions over and over.
                next_persist_scan_ = now + std::chrono::seconds(30);
                return;
            }
            auto& program = *instance_.program;
            if (materializing_ || context_owner_ || program.has_context_transaction() ||
                (store_.ready && !store_.ready())) {
                return;
            }
            {
                std::lock_guard lock(queue_mutex_);
                if (!pending_.empty()) { return; }
            }
            for (std::uint32_t lane = 0; lane < max_concurrency_; ++lane) {
                const auto& request = slots_[lane];
                if (request && (request->is_prefilling() || request->is_replaying() ||
                                request->is_materializing() || request->capture_pending)) {
                    return;
                }
            }
            const auto owners = resources_.retained_owners(program);
            std::erase_if(owner_marks_, [&](const auto& entry) {
                return std::none_of(owners.begin(), owners.end(),
                                    [&](const auto& owner) { return owner.token == entry.first; });
            });
            if (!idle_writes) { return; }
            const typename ResourceManagement::RetainedOwner* chosen = nullptr;
            std::uint32_t chosen_depth                                = 0;
            for (const auto& owner : owners) {
                if (owner.active) { continue; }
                auto& entry = mark(owner.token);
                if (settled(entry, owner.points) ||
                    now - entry.last_used < store_.idle_persist) {
                    continue;
                }
                std::uint32_t depth = 0;
                for (const auto point : owner.points) {
                    depth = std::max(depth, program.checkpoint_metadata(point).frontier);
                }
                // The deepest first: it is the most expensive to lose.
                if (chosen && depth <= chosen_depth) { continue; }
                chosen       = &owner;
                chosen_depth = depth;
            }
            if (!chosen) { return; }
            const auto started = Clock::now();
            auto image       = program.export_checkpoints(chosen->points, chosen->session,
                                                              store_.binding, export_buffer());
            cumulative_stats_.context_store_idle_export_seconds +=
                std::chrono::duration<double>(Clock::now() - started).count();
            const auto bytes = image.bytes.size();
            if (store_.sink(std::move(image))) {
                record_stored(chosen->token, chosen->points, bytes);
                ++cumulative_stats_.context_store_idle_writes;
            } else {
                // The write queue is full: try again later, not on every unit.
                next_persist_scan_ = now + std::chrono::seconds(4);
            }
        } catch (...) { next_persist_scan_ = now + std::chrono::seconds(4); }
    }

    // Extra reclaim rounds a hydration import may take when free Host bytes are split too finely
    // for the image (each round frees as much again as the image needs).
    static constexpr int kHydrationImportRounds = 3;

    // Catalogues an imported image: a shared prefix as a public entry, private points as one
    // inactive continuation. Either way the store holds it as it now is.
    template <class Imported>
    bool adopt_imported(const Imported& imported, std::uint64_t bytes) {
        auto& program = *instance_.program;
        if (imported.points.size() == 1 &&
            program.checkpoint_metadata(imported.points.front()).role ==
                CheckpointRole::SharedPrefix) {
            if (!resources_.adopt_restored_shared(program, imported.points.front())) {
                return false;
            }
            shared_stored_.emplace_back(imported.points.front(), bytes);
            return true;
        }
        const auto token = resources_.adopt_restored(program, imported.points, imported.session);
        if (!token) { return false; }
        mark(token).last_used = Clock::now();
        record_stored(token, imported.points, bytes);
        return true;
    }

    // Public shared prefixes the store does not hold yet; drops marks of released ones. Not those
    // the store let go (written once, then removed on its own) nor progress anchors: an anchor
    // insures one conversation's long prefill against a cancellation, and its retry finds it in
    // memory; the conversation's own image holds the same pages once the request ends.
    template <class Program>
    [[nodiscard]] std::vector<Checkpoint> unstored_shared(Program& program) {
        std::erase_if(shared_stored_,
                      [&](const auto& entry) { return !program.valid_checkpoint(entry.first); });
        const auto released = [&](const auto handle) { return !program.valid_checkpoint(handle); };
        std::erase_if(shared_dropped_, released);
        std::erase_if(progress_anchor_points_, released);
        auto shared = resources_.retained_shared(program);
        std::erase_if(shared, [&](const auto handle) {
            const auto among = [&](const auto& set) {
                return std::find(set.begin(), set.end(), handle) != set.end();
            };
            return std::any_of(shared_stored_.begin(), shared_stored_.end(),
                               [&](const auto& entry) { return entry.first == handle; }) ||
                   among(shared_dropped_) || among(progress_anchor_points_);
        });
        return shared;
    }

    // The size of a shared image in the store with this prefix's exact key, if there is one.
    template <class Program>
    [[nodiscard]] std::optional<std::uint64_t> stored_shared_image(Program& program,
                                                                   Checkpoint handle) const {
        const auto frontier = program.checkpoint_metadata(handle).frontier;
        const auto key      = program.checkpoint_key(handle, frontier);
        for (const ContextStore::Info& info : store_.store->list()) {
            if (!info.shared || info.binding != store_.binding || info.checkpoints.empty()) {
                continue;
            }
            const auto& stored = info.checkpoints.front();
            if (stored.frontier == frontier && stored.digests == key.digests &&
                stored.identity_tag == key.identity_tag) {
                return info.image_bytes;
            }
        }
        return std::nullopt;
    }

    // Writes a public shared prefix as soon as it is published, busy or not: one per second at a
    // boundary without a context transaction, while the writer is idle. A system or tools prompt
    // is captured once and then serves every tenant that sends it, so it is worth a short
    // Engine-worker copy (counted in context_store_shared_export_seconds) even under load.
    void persist_shared_prefix() noexcept {
        if (!store_.shared_sink) { return; }
        const auto now = Clock::now();
        if (now < next_shared_scan_) { return; }
        next_shared_scan_ = now + std::chrono::seconds(1);
        try {
            auto& program = *instance_.program;
            if (program.has_context_transaction() || (store_.ready && !store_.ready())) { return; }
            bool failed = false;
            for (const auto handle : unstored_shared(program)) {
                // A prefix captured again after an eviction has the key of the image the store
                // already holds: that image serves it, so it is not exported a second time.
                if (const auto bytes = stored_shared_image(program, handle)) {
                    shared_stored_.emplace_back(handle, *bytes);
                    continue;
                }
                // One prefix that cannot be exported (the Program refuses it, or a transient
                // failure) must not keep the others from being tried: each candidate is isolated,
                // as in the shutdown flush. One export per scan.
                std::optional<CheckpointImage> image;
                try {
                    const auto started = Clock::now();
                    image.emplace(program.export_checkpoints({&handle, 1}, std::nullopt,
                                                             store_.binding, export_buffer()));
                    cumulative_stats_.context_store_shared_export_seconds +=
                        std::chrono::duration<double>(Clock::now() - started).count();
                } catch (...) {
                    failed = true;
                    continue;
                }
                // The sink ends the scan whatever it answers: a rejection (full queue, shutdown)
                // or a throw backs off like a failed scan instead of exporting another prefix.
                const auto bytes = image->bytes.size();
                bool accepted    = false;
                try { accepted = store_.shared_sink(std::move(*image)); } catch (...) {}
                if (accepted) {
                    shared_stored_.emplace_back(handle, bytes);
                    ++cumulative_stats_.context_store_shared_writes;
                } else {
                    next_shared_scan_ = now + std::chrono::seconds(4);
                }
                return;
            }
            // Every candidate failed: back off instead of retrying them each second.
            if (failed) { next_shared_scan_ = now + std::chrono::seconds(4); }
        } catch (...) { next_shared_scan_ = now + std::chrono::seconds(4); }
    }

    // Write-on-evict: a retained continuation the cache is about to lose a point of, and that the
    // store does not hold as it is now, is exported and queued before the release. The export is
    // a step of the Engine worker (a D2H/H2H copy of the image, measured in
    // context_store_eviction_export_seconds); the queued image is caller memory, so the
    // conversation survives until it is written even though its Host copy is released now. A
    // full queue, an open context transaction or a failed export leaves today's behaviour: the
    // points are released and the next turn prefills.
    void write_on_evict(std::span<const typename ResourceManagement::RetainedOwner> owners) noexcept {
        if (!store_.evict_sink) { return; }
        auto& program = *instance_.program;
        if (program.has_context_transaction()) { return; }
        (void)forget_failed_store_writes();
        forget_removed_store_images();
        for (const auto& owner : owners) {
            try {
                if (settled(mark(owner.token), owner.points)) { continue; }
                const auto started = Clock::now();
                auto image = program.export_checkpoints(owner.points, owner.session, store_.binding, export_buffer());
                const auto bytes = image.bytes.size();
                cumulative_stats_.context_store_eviction_export_seconds +=
                    std::chrono::duration<double>(Clock::now() - started).count();
                if (store_.evict_sink(std::move(image))) {
                    record_stored(owner.token, owner.points, bytes);
                    ++cumulative_stats_.context_store_eviction_writes;
                }
            } catch (...) {}
        }
    }

    struct HydrationTask {
        std::shared_ptr<Request> request;
        std::string id;
        std::optional<std::vector<std::uint8_t>> bytes;
        // Wall time of the store read, for the cost model's read rate.
        std::uint64_t read_ns = 0;
        // What the read downloaded from the bucket first, and how long the download alone took.
        ContextStore::LoadReport fetch;
        // Not a read: refresh the store's listing of the bucket, then consult the store again.
        bool lookup = false;
    };

    // A request that would prefill for less than this does not wait for a bucket listing.
    static constexpr auto kStoreLookupMinimumPrefill = std::chrono::seconds(2);
    // The longest a request waits for the listing.
    static constexpr auto kStoreLookupBudget = std::chrono::seconds(5);

    // Time to fetch `bytes` of missing chunks from the bucket, at the rate the last fetches
    // achieved. A fetch is download, verification and installation together (what a waiting
    // request waits for), so that is what the rate measures, not the network alone.
    [[nodiscard]] std::uint64_t remote_fetch_ns(std::uint64_t bytes) const noexcept {
        const double ns = static_cast<double>(bytes) * remote_ns_per_byte_;
        return ns >= static_cast<double>(UINT64_MAX) ? UINT64_MAX : static_cast<std::uint64_t>(ns);
    }

    // Before admission: when the store holds a checkpoint of this very prompt deeper than anything
    // the cache offers, and reading it back, copying it to the device and prefilling the rest is
    // cheaper than the best source the cache offers (the cost model prices all three), read it
    // back on the host reader while the request waits. True when a read was started; the request
    // is then not offered for admission until the read settles or its deadline passes.
    bool start_hydration(const std::shared_ptr<Request>& request,
                         std::span<const typename ResourceManagement::SourceChoice> sources) {
        if (store_.store == nullptr || request->store_probed || !request->base_plan ||
            !request->options.execution.allow_prefix_reuse ||
            store_.hydration_budget.count() <= 0) {
            return false;
        }
        request->store_probed = true;
        std::uint32_t resident = 0;
        // Sources are ranked cheapest first; the root is always among them.
        std::uint64_t best_cost = UINT64_MAX;
        for (const auto& choice : sources) {
            resident  = std::max(resident, choice.source.reused_tokens);
            best_cost = std::min(best_cost, resources_.source_cost_ns(choice.source));
        }
        const auto& base         = *request->base_plan;
        const auto prompt_tokens = base.summary().prompt_tokens;
        // The prefill this request faces from what is resident.
        const auto full_prefill =
            resources_.prefill_ns(instance_.program->prefill_work(base, resident, prompt_tokens));
        // A download from the bucket is not weighed against the prefill it saves: it only makes
        // this request wait, while a prefill holds the GPU every other lane shares. It has to fit
        // in the wait, though (at the rate downloads have achieved), or it would time out anyway.
        // The wait is the read's deadline below: the budget, or less when the request has less
        // left before its own deadline.
        const auto now     = Clock::now();
        const auto left    = std::max<Clock::duration>(request->deadline - now, Clock::duration::zero());
        const auto wait    = std::min<Clock::duration>(store_.hydration_budget, left);
        const auto wait_ns = static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(wait).count());
        std::string best_id;
        std::uint32_t best_frontier = resident;
        try {
            // The binding does not include the context length: an image deeper than this Engine's
            // context could never be restored, and reading it would only cost time
            // (for_each_stored_match leaves it out).
            for_each_stored_match(base, [&](const ContextStore::Info& info,
                                            const ContextStore::CheckpointKey& key) {
                if (info.fetch_bytes > 0 && remote_fetch_ns(info.fetch_bytes) >= wait_ns) { return; }
                const auto restore = resources_.store_restore_ns(info.image_bytes);
                if (restore >= best_cost || key.frontier <= resident) { return; }
                const auto prefill = resources_.prefill_ns(
                    instance_.program->prefill_work(base, key.frontier, prompt_tokens));
                const auto cost = prefill > UINT64_MAX - restore ? UINT64_MAX : restore + prefill;
                if (cost >= best_cost) { return; }
                best_id       = info.id;
                best_frontier = key.frontier;
                best_cost     = cost;
            });
        } catch (...) { return false; }
        if (best_id.empty()) {
            // Nothing here, but another engine may have stored this session since the bucket was
            // last listed (the conversation moved here a moment ago). A request with a long
            // prefill ahead lists it once and looks again; a short one is not worth the wait.
            if (!store_.remote || request->store_looked_up ||
                full_prefill < static_cast<std::uint64_t>(
                                   std::chrono::nanoseconds(kStoreLookupMinimumPrefill).count())) {
                return false;
            }
            request->store_looked_up    = true;
            request->store_lookup       = true;
            request->hydrating          = true;
            request->hydration_started  = now;
            request->hydration_deadline = std::min(request->deadline, now + kStoreLookupBudget);
            {
                std::lock_guard lock(hydration_mutex_);
                HydrationTask task;
                task.request = request;
                task.lookup  = true;
                lookup_queue_.push_back(std::move(task));
            }
            // Two threads wait on this condition variable, each for its own queue.
            hydration_cv_.notify_all();
            return true;
        }
        request->hydrating               = true;
        request->hydration_frontier      = best_frontier;
        request->hydration_reuse_before  = resident;
        request->hydration_started       = now;
        request->hydration_deadline      = std::min(request->deadline, now + store_.hydration_budget);
        {
            std::lock_guard lock(hydration_mutex_);
            HydrationTask task;
            task.request = request;
            task.id      = std::move(best_id);
            hydration_queue_.push_back(std::move(task));
        }
        hydration_cv_.notify_all();
        return true;
    }

    // A bucket lookup settled (or ran out of time): the request consults the store again, now
    // with whatever the listing found, and is offered for admission as usual.
    void finish_store_lookup(const std::shared_ptr<Request>& request) {
        request->store_lookup = false;
        request->hydrating    = false;
        request->store_probed = false;
        cumulative_stats_.context_store_lookup_seconds +=
            std::chrono::duration<double>(Clock::now() - request->hydration_started).count();
        ++cumulative_stats_.context_store_lookups;
        request->admission_generation = 0;
        request_admission_check();
    }

    // Runs the tasks of one queue: reads on the hydration reader, bucket lookups on their own
    // thread, so a lookup stalled on the network never holds up a read (or the reverse). The
    // waiting request is released at its deadline either way (process_hydrations).
    void hydration_reader_loop(std::deque<HydrationTask>& queue) noexcept {
        for (;;) {
            HydrationTask task;
            {
                std::unique_lock lock(hydration_mutex_);
                hydration_cv_.wait(lock, [&] { return hydration_stop_ || !queue.empty(); });
                if (hydration_stop_) { return; }
                task = std::move(queue.front());
                queue.pop_front();
            }
            if (const auto started = Clock::now(); started < task.request->hydration_deadline) {
                try {
                    if (task.lookup) {
                        // The bucket as of when the request looked: a listing taken since (for
                        // another request) serves it too.
                        (void)store_.store->refresh_remote_since(task.request->hydration_started,
                                                                 task.request->hydration_deadline);
                    } else {
                        task.bytes = store_.store->load(task.id, false, &task.fetch);
                    }
                } catch (...) { task.bytes.reset(); }
                task.read_ns = static_cast<std::uint64_t>(
                    std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - started)
                        .count());
            }
            {
                std::lock_guard lock(hydration_mutex_);
                hydration_done_.push_back(std::move(task));
            }
            queue_cv_.notify_all();
        }
    }

    void stop_hydration_reader() noexcept {
        {
            std::lock_guard lock(hydration_mutex_);
            hydration_stop_ = true;
        }
        hydration_cv_.notify_all();
        if (hydration_reader_.joinable()) { hydration_reader_.join(); }
        if (lookup_reader_.joinable()) { lookup_reader_.join(); }
    }

    void observe_store_read(std::uint64_t image_bytes, std::uint64_t nanoseconds) noexcept {
        resources_.observe_store_read(image_bytes, nanoseconds);
        publish_store_read_rate();
    }

    void publish_store_read_rate() noexcept {
        const auto per_byte_q32 = resources_.costs().store_read.ns_per_byte_q32;
        cumulative_stats_.context_store_read_bytes_per_second =
            per_byte_q32 == 0 ? 0.0
                              : 1e9 * static_cast<double>(kContextCostQ32One) /
                                    static_cast<double>(per_byte_q32);
        // The rate may change while no request runs (store set up, a start-up restore), when
        // nothing else republishes: update the published copy too. execution_mutex_ (held by the
        // callers) before stats_mutex_ is the existing order.
        std::lock_guard lock(stats_mutex_);
        published_stats_.context_store_read_bytes_per_second =
            cumulative_stats_.context_store_read_bytes_per_second;
    }

    void finish_hydration(const std::shared_ptr<Request>& request, bool failed) {
        request->hydrating = false;
        cumulative_stats_.context_store_hydration_seconds +=
            std::chrono::duration<double>(Clock::now() - request->hydration_started).count();
        if (failed) { ++cumulative_stats_.context_store_hydration_failures; }
        request->admission_generation = 0;
        request_admission_check();
    }

    // Imports a read-back image for a waiting request as Host-resident checkpoints, making room in
    // the Host tier under optional-write rights (never revoking another request's waiting source),
    // and protects the restored source for the request until it is admitted.
    bool import_for_request(const std::shared_ptr<Request>& request,
                            std::span<const std::uint8_t> image, const std::string& id) {
        auto& program     = *instance_.program;
        const auto needed = program.checkpoint_image_host_bytes(image, store_.binding);
        auto usage        = program.physical_usage();
        if (needed > usage.capacity.host_bytes) { return false; }
        if (needed > usage.capacity.host_bytes - usage.occupied.host_bytes) {
            (void)program.release_redundant_host({});
            usage = program.physical_usage();
        }
        auto cursor = resources_.begin_reclaim(
            program, std::nullopt, ReclaimRights{ReclaimPurpose::OptionalWrite, request->id});
        if (needed > usage.capacity.host_bytes - usage.occupied.host_bytes) {
            const auto victims = resources_.host_victims(program, needed, std::nullopt, {}, {},
                                                         std::nullopt, &cursor);
            if (!victims) { return false; }
            resources_.commit_host_victims(program, *victims, cursor);
            if (!victims->empty()) { scheduler_.capacity_released(); }
        }
        auto imported = program.import_checkpoints(image, store_.binding);
        // Free bytes are not free geometry: each StateImage needs one contiguous Host extent, and
        // under pressure the free space is often split below that. Give up the next-ranked Host
        // contents (the same optional-write rights, never another request's waiting source) a
        // round at a time until the image fits or nothing more may go.
        for (int round = 0; !imported && round < kHydrationImportRounds; ++round) {
            if (resources_.reclaim(program, ContextResourceUsage{.host_bytes = needed}, {},
                                   cursor) != ReclaimProgress::Changed) {
                break;
            }
            scheduler_.capacity_released();
            imported = program.import_checkpoints(image, store_.binding);
        }
        if (!imported || !adopt_imported(*imported, image.size())) { return false; }
        auto sources = resources_.candidates(program, *request->base_plan, UINT32_MAX, std::nullopt,
                                             request->id, request->publication_order);
        const auto restored = std::find_if(sources.begin(), sources.end(), [&](const auto& choice) {
            return choice.source.checkpoint &&
                   std::find(imported->points.begin(), imported->points.end(),
                             *choice.source.checkpoint) != imported->points.end();
        });
        if (restored == sources.end()) { return false; }
        retain_admission_source(request, *restored, false);
        request->hydrated = true;
        try {
            store_.store->touch(id);
        } catch (...) {}
        return true;
    }

    // True when the cache now offers this request a source at least as deep as the stored image it
    // was waiting for.
    [[nodiscard]] bool cache_covers_hydration(const std::shared_ptr<Request>& request) {
        const auto sources = resources_.candidates(*instance_.program, *request->base_plan,
                                                   UINT32_MAX, std::nullopt, request->id,
                                                   request->publication_order);
        return std::any_of(sources.begin(), sources.end(), [&](const auto& choice) {
            return choice.source.reused_tokens >= request->hydration_frontier;
        });
    }

    // Settles finished reads and expired waits. Imports wait for a boundary without a context
    // transaction, since they may give up cached checkpoints.
    void process_hydrations() {
        if (store_.store == nullptr) { return; }
        auto& program = *instance_.program;
        std::deque<HydrationTask> done;
        {
            std::lock_guard lock(hydration_mutex_);
            if (program.has_context_transaction()) {
                // Settle only the reads nobody waits for any longer.
                for (auto it = hydration_done_.begin(); it != hydration_done_.end();) {
                    if (!it->request->hydrating) {
                        it = hydration_done_.erase(it);
                    } else {
                        ++it;
                    }
                }
            } else {
                done.swap(hydration_done_);
            }
        }
        for (auto& task : done) {
            const auto& request = task.request;
            // A completed fetch measures the fetch rate whatever then becomes of the request
            // (gone, already covered, or an import without room).
            if (task.bytes && task.fetch.fetched_bytes > 0) {
                observe_remote_fetch(task.fetch.fetched_bytes, task.fetch.fetch_ns);
            }
            if (!request->hydrating) { continue; } // gave up at its deadline
            bool waiting = false;
            {
                std::lock_guard lock(queue_mutex_);
                waiting = std::find(pending_.begin(), pending_.end(), request) != pending_.end();
            }
            if (!waiting) {
                // Cancelled or failed meanwhile: settle nothing for it, lookup or read.
                request->hydrating    = false;
                request->store_lookup = false;
                continue;
            }
            if (task.lookup) {
                finish_store_lookup(request);
                continue;
            }
            if (task.bytes && cache_covers_hydration(request)) {
                // Another waiter's read-back of the same image (or anything else) has already
                // brought this prefix in: importing it again would catalogue a second copy.
                finish_hydration(request, false);
                continue;
            }
            bool imported = false;
            if (task.bytes) {
                const auto started = Clock::now();
                try {
                    imported = import_for_request(request, *task.bytes, task.id);
                } catch (const std::invalid_argument&) { imported = false; }
                if (imported) {
                    // The fetch taught the fetch rate above; the rest of the read (the local
                    // files read, verified and reassembled) and the import teach the disk's.
                    observe_store_read(task.bytes->size(),
                                       task.read_ns - std::min(task.read_ns, task.fetch.fetch_ns) +
                                           static_cast<std::uint64_t>(
                                               std::chrono::duration_cast<std::chrono::nanoseconds>(
                                                   Clock::now() - started)
                                                   .count()));
                }
            }
            finish_hydration(request, !imported);
        }
        const auto now = Clock::now();
        std::vector<std::shared_ptr<Request>> expired;
        {
            std::lock_guard lock(queue_mutex_);
            for (const auto& request : pending_) {
                if (request->hydrating && now >= request->hydration_deadline) {
                    expired.push_back(request);
                }
            }
        }
        for (const auto& request : expired) {
            if (request->store_lookup) {
                finish_store_lookup(request);
            } else {
                finish_hydration(request, true);
            }
        }
    }

    // Moves the download rate towards what one download achieved (half the way: it follows a
    // changed link within a few downloads without swinging on one slow one).
    void observe_remote_fetch(std::uint64_t bytes, std::uint64_t nanoseconds) noexcept {
        if (bytes == 0 || nanoseconds == 0) { return; }
        const double observed = static_cast<double>(nanoseconds) / static_cast<double>(bytes);
        remote_ns_per_byte_   = 0.5 * remote_ns_per_byte_ + 0.5 * observed;
        cumulative_stats_.context_store_remote_fetch_bytes_per_second = 1e9 / remote_ns_per_byte_;
    }

    // Counts a read-back once its request is admitted: only if the admission resumes from (at
    // least) the stored checkpoint, with the tokens it actually gained.
    void settle_hydration(const std::shared_ptr<Request>& request, std::uint32_t reused) noexcept {
        if (!request->hydrated) { return; }
        request->hydrated = false;
        if (reused >= request->hydration_frontier && reused > request->hydration_reuse_before) {
            ++cumulative_stats_.context_store_hydrations;
            cumulative_stats_.context_store_hydrated_tokens +=
                reused - request->hydration_reuse_before;
        } else {
            ++cumulative_stats_.context_store_hydration_failures;
        }
    }

    void require_suspend_enabled() const {
        if (!state_snapshot_) {
            throw std::invalid_argument(
                "model suspend is disabled; start the Engine with suspend enabled");
        }
    }

    // Under the execution mutex.
    void resume_locked() {
        if (!suspended_.load(std::memory_order_acquire)) { return; }
        if constexpr (kInstanceSuspendable) {
            try {
                last_resume_ = instance_.resume(device_, *state_snapshot_,
                                                weight_snapshot_ ? &*weight_snapshot_ : nullptr);
            } catch (const std::exception& error) {
                last_residency_error_ = error.what();
                publish_residency();
                throw;
            }
            last_residency_error_.clear();
            ++resume_count_;
            suspended_.store(false, std::memory_order_release);
            publish_residency();
            publish_runtime_stats();
            queue_cv_.notify_all();
        }
    }

    // Under the execution mutex, while suspended, from the worker: work was queued.
    void serve_suspended_queue_locked() {
        bool queued = false;
        {
            std::lock_guard lock(queue_mutex_);
            queued = !pending_.empty();
        }
        if (!queued) { return; }
        if (auto_resume_.load(std::memory_order_acquire)) {
            try {
                resume_locked();
                return;
            } catch (const std::exception& error) {
                fail_pending_locked(std::make_exception_ptr(RequestError(
                    RequestErrorKind::Unavailable,
                    std::string("model resume failed: ") + error.what())));
                return;
            }
        }
        fail_pending_locked(std::make_exception_ptr(
            RequestError(RequestErrorKind::Unavailable, "model is suspended")));
    }

    // Ends every queued request with `error`; nothing is admitted or running when this is called,
    // so no Program state is touched (the device memory may be released).
    void fail_pending_locked(const std::exception_ptr& error) noexcept {
        std::vector<std::shared_ptr<Request>> failed;
        {
            std::lock_guard lock(queue_mutex_);
            failed.assign(pending_.begin(), pending_.end());
            pending_.clear();
        }
        for (const auto& request : failed) { complete_error(request, error); }
        if (!failed.empty()) { publish_runtime_stats(); }
    }

    void publish_residency() {
        ResidencyStatus status;
        status.enabled     = state_snapshot_.has_value();
        status.state       = suspended_.load(std::memory_order_acquire) ? ModelResidency::Suspended
                                                                        : ModelResidency::Resident;
        status.auto_resume = auto_resume_.load(std::memory_order_acquire);
        if constexpr (kInstanceSuspendable) {
            if (status.enabled) {
                status.releasable_device_bytes = instance_.releasable_device_bytes();
                status.host_snapshot_bytes =
                    state_snapshot_->bytes() + (weight_snapshot_ ? weight_snapshot_->bytes() : 0);
                status.host_reserved_bytes =
                    state_snapshot_->host_capacity_bytes() +
                    (weight_snapshot_ ? weight_snapshot_->host_capacity_bytes() : 0);
            }
        }
        status.suspend_count = suspend_count_;
        status.resume_count  = resume_count_;
        status.last_suspend  = last_suspend_;
        status.last_resume   = last_resume_;
        status.last_error    = last_residency_error_;
        std::lock_guard lock(stats_mutex_);
        published_residency_ = std::move(status);
    }

    // ---- the summary strategy's reuse probes (has_private_reuse) ----

    struct ReuseProbe {
        PreparedPrompt prompt{};
        ResolvedExecutionOptions execution;
        std::uint32_t exact_end = 0;
        std::mutex mutex;
        std::condition_variable cv;
        std::optional<bool> warm;
        bool abandoned = false;
    };

    // Visits every checkpoint key of a context-store image on this binding that this Engine could
    // restore (at most max_context) and whose prefix the plan's prompt starts with, as admission's
    // hydration compares them (prefix shortlist key and identity tag). Images only the bucket holds
    // are included once a listing has registered them.
    template <class Visit>
    void for_each_stored_match(const BasePlan& base, Visit&& visit) {
        const auto prompt_tokens = base.summary().prompt_tokens;
        for (const ContextStore::Info& info : store_.store->list()) {
            if (info.binding != store_.binding || info.tokens > max_context_) { continue; }
            for (const ContextStore::CheckpointKey& key : info.checkpoints) {
                if (key.frontier > prompt_tokens) { continue; }
                const auto mine = base.prefix_shortlist_key(key.frontier);
                if (mine && mine->digests == key.digests && mine->identity_tag == key.identity_tag) {
                    visit(info, key);
                }
            }
        }
    }

    // On the worker, under the execution lock: the probe's answer (true: warm).
    [[nodiscard]] bool private_reuse_locked(ReuseProbe& probe) {
        const auto plan = instance_.program->plan_request(std::move(probe.prompt), probe.execution);
        // Shared matches past the product prompt are the conversation too (lossy_prefill_trigger).
        if (resources_.has_conversation_match(*instance_.program, plan, probe.exact_end)) {
            return true;
        }
        if (store_.store == nullptr) { return false; }
        bool stored = false;
        for_each_stored_match(plan, [&](const ContextStore::Info&, const ContextStore::CheckpointKey& key) {
            stored = stored || key.frontier > probe.exact_end;
        });
        return stored;
    }

    // Answers the queued probes; `warm_only` (suspended or stopping) answers warm without looking.
    void serve_reuse_probes(bool warm_only) noexcept {
        // Every worker cycle calls this; without a posted probe (always, unless the summary
        // strategy is on) it returns without touching the queue lock. A probe posted after the
        // exchange sets the flag again and is answered on the next cycle.
        if (!probes_pending_.exchange(false, std::memory_order_acq_rel)) { return; }
        std::deque<std::shared_ptr<ReuseProbe>> probes;
        {
            std::lock_guard lock(queue_mutex_);
            probes.swap(probes_);
        }
        for (const auto& probe : probes) {
            {
                std::lock_guard lock(probe->mutex);
                if (probe->abandoned) { continue; }
            }
            bool warm = true;
            if (!warm_only) {
                try {
                    warm = private_reuse_locked(*probe);
                } catch (...) { warm = true; }
            }
            {
                std::lock_guard lock(probe->mutex);
                probe->warm = warm;
            }
            probe->cv.notify_all();
        }
    }

    void worker_loop() noexcept {
        for (;;) {
            {
                std::unique_lock lock(queue_mutex_);
                if (!stopping_ && pending_.empty() && probes_.empty() && resident_empty() &&
                    paused_.empty() && !instance_.program->has_context_transaction()) {
                    const auto woken = [&] {
                        return stopping_ || !pending_.empty() || !probes_.empty();
                    };
                    if (store_.sink && store_.idle_persist.count() > 0) {
                        // Wake up now and then to write idle continuations to the store.
                        queue_cv_.wait_for(lock, std::chrono::seconds(1), woken);
                    } else {
                        queue_cv_.wait(lock, woken);
                    }
                }
                if (stopping_) {
                    lock.unlock();
                    std::scoped_lock execution_lock(execution_mutex_);
                    serve_reuse_probes(true);
                    const auto error = std::make_exception_ptr(RequestError(
                        RequestErrorKind::Unavailable, "inference engine is shutting down"));
                    if (suspended_.load(std::memory_order_acquire)) {
                        // Nothing is admitted and the device memory is released: only the queue
                        // has anything to fail, and the Program is not touched.
                        {
                            std::lock_guard queue_lock(queue_mutex_);
                            failed_ = true;
                        }
                        fail_pending_locked(error);
                        serve_reuse_probes(true);
                        return;
                    }
                    fail_all_locked(error);
                    serve_reuse_probes(true);
                    return;
                }
            }
            std::unique_lock execution_lock(execution_mutex_);
            if (suspended_.load(std::memory_order_acquire)) {
                // Nothing executes while suspended. The wait above returned because work was
                // queued: resume for it, or refuse it when the model is held suspended. A probe
                // does not resume the model: it is answered warm.
                serve_reuse_probes(true);
                serve_suspended_queue_locked();
                continue;
            }
            // Between cycles: the summary strategy's probes, before this cycle changes the cache.
            serve_reuse_probes(false);
            bool executed = false;
            // The execution unit in progress, for the fault event.
            const char* unit = "boundary";
            // Set when a failed admission containment already found the device faulted.
            std::optional<std::string> device_fault;
            try {
                set_host_work_class(HostWorkClass::Control);
                auto boundary = begin_host_phase();
                (void)expire_pending_requests();
                executed = progress_context_transaction(boundary);
                process_hydrations();
                process_sibling_waits();
                if (!instance_.program->has_context_transaction()) {
                    for (const auto& request : slots_) {
                        if (request && request->capture_pending) {
                            reserve_active_capture(request, request->post_capture_state);
                            break;
                        }
                    }
                }
                (void)settle_terminal_requests(boundary);
                cancel_active_requests(snapshot_cancellations(), boundary);
                for (auto it = paused_.begin();
                     !instance_.program->has_context_transaction() && it != paused_.end();) {
                    const auto request = *it;
                    if (!request->cancelled.load(std::memory_order_acquire)) {
                        ++it;
                        continue;
                    }
                    request->suspended.reset();
                    it = paused_.erase(it);
                    complete_detached_cancelled(request);
                    scheduler_.capacity_released();
                }
                const bool recovering =
                    std::any_of(slots_.begin(), slots_.end(),
                                [](const auto& r) { return r && r->recovery_pending; }) ||
                    (materializing_ && materializing_->resumed);
                if (paused_.empty() && !context_owner_ && !recovering) {
                    scheduler_.paused_queue_empty();
                }
                if (!recovering &&
                    ((admission_decision_ && admission_decision_->restoring) ||
                     scheduler_.should_restore(!paused_.empty() && !restore_held_for_yield(),
                                               resident_empty()))) {
                    unit = "admission";
                    admit_or_contain(true, device_fault);
                    unit = "boundary";
                    executed |= progress_context_transaction(boundary);
                }
                reserve_resident_units();
                // A failed restore leaves spare resources available to fresh requests.
                // An in-flight restoration/reclaim keeps its own candidate until settled.
                if (!admission_decision_ || !admission_decision_->restoring) {
                    if (!admission_decision_ && consume_admission_check()) {
                        ++admission_generation_;
                        scheduler_.begin_admission_scan();
                    }
                    if (admission_decision_ || scheduler_.admission_scan_pending()) {
                        unit = "admission";
                        admit_or_contain(false, device_fault);
                        unit = "boundary";
                    }
                }
                if (instance_.program->has_context_transaction()) {
                    executed |= progress_context_transaction(boundary);
                    if (!instance_.program->has_context_transaction()) { reserve_resident_units(); }
                }
                const auto cancelled = snapshot_cancellations();
                auto controls = scheduler_.build_control_membership(slots_, max_concurrency_);
                ControlMembership selected_control;
                selected_control.row_stride = controls.row_stride;
                // Forced tokens run through the prefill path and write the buffer a sliced chunk
                // parks its residual in, so they wait until that chunk is done.
                const bool hold_controls = slice_in_flight_any();
                for (std::size_t row = 0; row < controls.size; ++row) {
                    if (hold_controls || !runnable_units_[controls.lanes[row]] ||
                        slots_[controls.lanes[row]]->cancelled.load(std::memory_order_acquire)) {
                        continue;
                    }
                    selected_control.lanes[selected_control.size]       = controls.lanes[row];
                    selected_control.sequences[selected_control.size++] = controls.sequences[row];
                    const auto begin = controls.tokens.begin() + row * controls.row_stride;
                    selected_control.tokens.insert(selected_control.tokens.end(), begin,
                                                   begin + controls.row_stride);
                }
                controls = std::move(selected_control);
                finish_engine_phase(boundary, EngineHostPhase::Boundary);
                if (!controls.empty()) {
                    set_host_work_class(HostWorkClass::Control);
                    unit = "control";
                    run_control_batch(controls);
                    executed = true;
                }
                // Control rows become decode-ready next cycle, after reserving their new unit.
                auto decode = scheduler_.build_round_membership(slots_, max_concurrency_);
                {
                    RoundMembership selected;
                    for (std::size_t row = 0; row < decode.size; ++row) {
                        if (!runnable_units_[decode.lanes[row]] ||
                            slots_[decode.lanes[row]]->cancelled.load(std::memory_order_acquire)) {
                            continue;
                        }
                        if (std::find(controls.lane_span().begin(), controls.lane_span().end(),
                                      decode.lanes[row]) != controls.lane_span().end()) {
                            continue;
                        }
                        selected.lanes[selected.size]     = decode.lanes[row];
                        selected.sequences[selected.size] = decode.sequences[row];
                        selected.budgets[selected.size++] = decode.budgets[row];
                    }
                    decode = selected;
                }
                bool decode_ran = false;
                if (!decode.empty()) {
                    set_host_work_class(HostWorkClass::Decode, decode.lane_span());
                    unit = "decode";
                    const auto decode_started = Clock::now();
                    run_decode_round(decode, cancelled);
                    pacing_.decode_round_done(elapsed_ns(decode_started, Clock::now()));
                    executed   = true;
                    decode_ran = true;
                }
                if (instance_.program->has_context_transaction()) {
                    set_host_work_class(HostWorkClass::Control);
                    unit                   = "boundary";
                    auto progress_boundary = begin_host_phase();
                    executed |= progress_context_transaction(progress_boundary);
                    if (!instance_.program->has_context_transaction()) {
                        // Control/Decode permits were consumed. Only the still-unexecuted
                        // Prefill/Replay turn can receive a new permit in this round.
                        reserve_resident_units(ReservationScope::Prefill);
                    }
                    finish_engine_phase(progress_boundary, EngineHostPhase::Boundary);
                }
                auto prefill_slots = slots_;
                for (std::uint32_t lane = 0; lane < max_concurrency_; ++lane) {
                    // A lane whose media item still encodes in an overlay Vision window keeps its
                    // permit and yields the prefill turn; the other lanes run meanwhile.
                    if (!runnable_units_[lane] ||
                        (prefill_slots[lane] &&
                         (prefill_slots[lane]->cancelled.load(std::memory_order_acquire) ||
                          (prefill_slots[lane]->sequence &&
                           instance_.program->vision_pending(*prefill_slots[lane]->sequence))))) {
                        prefill_slots[lane].reset();
                    }
                }
                auto prefill_lane = scheduler_.next_prefill(prefill_slots, max_concurrency_);
                // A cycle without a decode round owes nothing, whether or not a prefill lane is
                // selectable: stale debt must not delay an unrelated prompt later.
                if (!decode_ran) { pacing_.reset_decode_debt(); }
                // A decode share holds the prefill turn back while the rounds since the last step
                // have not yet used their share of its time.
                if (prefill_lane && pacing_.defer_prefill(decode_ran)) {
                    ++cumulative_stats_.deferred_prefill_turns;
                    prefill_lane.reset();
                }
                if (const auto lane = prefill_lane) {
                    const auto request = slots_[*lane];
                    set_host_work_class(HostWorkClass::Prefill);
                    unit = "prefill";
                    if (request->is_replaying()) {
                        ProgramCallScope call(*this);
                        const Clock::time_point replay_started = Clock::now();
                        auto progress = instance_.program->advance_replay(*request->sequence,
                                                                          &call.failed_timing());
                        call.finish(progress.timing);
                        pacing_.prefill_step_done(0, progress.timing.elapsed_ns(), decode_ran,
                                                  elapsed_ns(replay_started, Clock::now()));
                        if (!request->first_output_timing) {
                            record_execution_work(request->replay_work, progress.timing);
                        }
                        request->replayed_tokens += progress.processed_tokens;
                        cumulative_stats_.replayed_tokens += progress.processed_tokens;
                        if (progress.capture_ready) {
                            reserve_active_capture(request, EngineRequestState::Replay);
                        }
                        if (progress.complete) {
                            ++cumulative_stats_.replay_restores;
                            ++request->replay_restores;
                            request->model_state = request->resume_phase;
                            observe_scheduling(request,
                                               GenerationSchedulingTransition::ReplayComplete);
                        }
                    } else {
                        run_prefill_step(*lane, snapshot_cancellations(), decode_ran);
                    }
                    scheduler_.prefill_executed(*lane, max_concurrency_);
                    update_recovery(request);
                    executed = true;
                }
                unit = "boundary";
                persist_shared_prefix();
                persist_idle_session();
                publish_runtime_stats();
            } catch (...) {
                const std::exception_ptr error = std::current_exception();
                const Clock::time_point failed_at = Clock::now();
                EngineFaultEvent fault;
                try {
                    fault.message = exception_text(error);
                    fault.unit    = unit;
                    collect_fault_requests(fault);
                } catch (...) {}
                std::string detail;
                if (device_fault) {
                    detail = std::move(*device_fault);
                }
                const bool device_ok = !device_fault && device_healthy(detail);
                std::string refusal;
                bool recovered = false;
                if (const char* reason = recovery_refusal(device_ok, recovery_streak_, failed_at)) {
                    set_refusal(refusal, reason, detail);
                } else {
                    recovered = recover_locked(error, refusal, failed_at);
                }
                if (!recovered) { fail_all_locked(error, &fault); }
                try {
                    fault.latched      = !recovered;
                    fault.latch_reason = refusal;
                    fault.consecutive_failures =
                        recovery_streak_.count() + (recovered ? 0U : 1U);
                    fault.maximum_consecutive_failures = recovery_streak_.maximum();
                } catch (...) {}
                try {
                    publish_runtime_stats();
                } catch (...) {}
                // After the latch took effect, so is_available() (and /health) already report it.
                notify_fault(fault);
                if (!recovered) {
                    // Probes queued before the latch: answered warm, not left to time out.
                    serve_reuse_probes(true);
                    return;
                }
                continue;
            }
            execution_lock.unlock();
            if (!executed) {
                std::unique_lock lock(queue_mutex_);
                queue_cv_.wait_for(lock, std::chrono::milliseconds(1));
            }
        }
    }

    Instance& instance_;
    Device& device_;
    const std::uint32_t max_context_;
    const std::uint32_t max_concurrency_;
    const bool admission_fit_check_;
    const std::size_t max_outstanding_;
    const std::chrono::milliseconds pending_timeout_;
    const std::uint32_t prefill_chunk_;
    PrefillPacing pacing_;
    // Zero: a long prefill is never paused for a waiting request.
    const std::uint64_t yield_min_remaining_ns_;
    // Layer slices per prefill chunk while a neighbour decodes; 0 or 1: whole chunks.
    const std::uint32_t layer_slices_;
    // Lossy long-prompt prefill; off unless min_tokens is set.
    const LossyPrefillOptions lossy_options_;
    // The request a prefill was last paused for, so it cannot pause a second one for the same
    // newcomer, and the hold that keeps the paused prefill from being restored ahead of it.
    std::uint64_t yield_beneficiary_ = 0;
    bool yield_hold_                 = false;
    Clock::time_point yield_hold_until_{};
    const std::uint32_t progress_anchor_tokens_;
    const std::function<void(const EngineFaultEvent&)> fault_listener_;
    ResourceManagement resources_;

    mutable std::mutex execution_mutex_;
    // Summary-strategy reuse probes waiting for the worker (guarded by queue_mutex_).
    std::deque<std::shared_ptr<ReuseProbe>> probes_;
    // Set (under queue_mutex_) with every posted probe; serve_reuse_probes clears it.
    std::atomic<bool> probes_pending_{false};
    std::atomic<std::uint64_t> probe_timeouts_{0};
    mutable std::mutex queue_mutex_;
    mutable std::mutex stats_mutex_;
    std::condition_variable queue_cv_;
    std::deque<std::shared_ptr<Request>> pending_;
    std::size_t outstanding_              = 0;
    std::uint64_t next_request_id_        = 1;
    std::uint64_t next_publication_order_ = 1;
    std::array<std::shared_ptr<Request>, kMaximumConcurrency> slots_{};
    std::optional<MaterializingRequest> materializing_;

    struct SiblingWait {
        std::shared_ptr<Request> waiter;
        std::uint64_t sibling;
        std::uint32_t frontier;
        Clock::time_point deadline;
    };

    std::vector<SiblingWait> sibling_waits_;
    std::optional<AdmissionDecision> admission_decision_;
    std::uint64_t admission_generation_ = 1;
    std::array<std::optional<CaptureDecision>, kMaximumConcurrency> capture_decisions_;
    std::shared_ptr<Request> context_owner_;
    std::vector<std::shared_ptr<Request>> paused_;
    Scheduling scheduler_;
    std::array<bool, kMaximumConcurrency> runnable_units_{};
    // Worker-only: the request an admission attempt is deciding for, while a failure would be
    // confined to it.
    std::shared_ptr<Request> admission_subject_;
    // Worker-only: Program physical usage once startup finished, which recovery must restore.
    std::optional<decltype(std::declval<Instance&>().program->physical_usage())> quiescent_usage_;
    // Worker-only: recoveries since the last request completed successfully. Failures closer
    // together than the window count toward the latch; see RecoveryStreak.
    static constexpr std::uint32_t kMaximumConsecutiveRecoveries = 3;
    static constexpr std::chrono::seconds kRecoveryHealthyWindow{30};
    RecoveryStreak recovery_streak_{kMaximumConsecutiveRecoveries, kRecoveryHealthyWindow};
    std::atomic<bool> admission_check_pending_{false};
    std::uint64_t worker_accounted_elapsed_ns_ = 0;
    HostWorkClass current_host_work_class_     = HostWorkClass::Control;
    std::array<std::uint32_t, kMaximumConcurrency> current_decode_lanes_{};
    std::size_t current_decode_lane_count_ = 0;
    RuntimeStats cumulative_stats_;
    RuntimeStats published_stats_;
    bool stopping_ = false;
    bool failed_   = false;

    // Durable context store state; the worker owns everything but the hydration queues.
    ContextStoreHooks store_;
    std::unordered_map<ContinuationOwnerToken, OwnerMark> owner_marks_;
    // Public shared prefixes written (or queued) to the store, with their image sizes.
    std::vector<std::pair<Checkpoint, std::uint64_t>> shared_stored_;
    // Shared prefixes whose image the store removed on its own: not written again.
    std::vector<Checkpoint> shared_dropped_;
    // Shared progress anchors (trim_progress_anchors): never written to the store.
    std::vector<Checkpoint> progress_anchor_points_;
    Clock::time_point next_shared_scan_{};
    std::uint64_t store_removals_seen_ = 0;
    std::uint64_t store_failures_seen_ = 0;
    Clock::time_point next_persist_scan_{};
    std::mutex hydration_mutex_;
    std::condition_variable hydration_cv_;
    std::deque<HydrationTask> hydration_queue_;
    std::deque<HydrationTask> lookup_queue_; // bucket lookups, run on lookup_reader_
    std::deque<HydrationTask> hydration_done_;
    // Nanoseconds per byte downloaded from the bucket: assumed 100 MB/s until a download says
    // otherwise. Engine worker only.
    double remote_ns_per_byte_ = 10.0;
    bool hydration_stop_ = false;
    std::thread hydration_reader_;
    std::thread lookup_reader_;

    // Model residency. The snapshots exist only when suspend is enabled and are touched under the
    // execution mutex; the flags are also read by submit and is_available without it.
    ModelSuspendOptions suspend_options_;
    std::optional<DeviceSnapshot> state_snapshot_;
    std::optional<DeviceSnapshot> weight_snapshot_;
    std::atomic<bool> suspended_{false};
    std::atomic<bool> auto_resume_{true};
    std::uint64_t suspend_count_ = 0;
    std::uint64_t resume_count_  = 0;
    std::optional<ResidencyTransition> last_suspend_;
    std::optional<ResidencyTransition> last_resume_;
    std::string last_residency_error_;
    // Guarded by stats_mutex_.
    ResidencyStatus published_residency_;

    std::thread worker_;
};

} // namespace ninfer::runtime

#include "runtime/engine/engine_metrics.inl"
