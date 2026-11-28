#pragma once

// Written for NInfer-3090. The Engine's core for an Inkling-2 artifact (docs/maintainer/inkling2-model.md,
// "Serving"): the model, its frontend, and one worker thread. With max_concurrency 1 it runs requests one at a time,
// in submission order (DSpark when configured); with max_concurrency N > 1 the text Program holds N slots and the
// worker decodes every running request as a column of one batched forward, prefilling new ones a chunk at a time
// between decode steps ("Batched decode"). Output is sampled on the host from each request's logits column.

#include "core/device.h"
#include "models/inkling2/frontend/constraint.h"
#include "models/inkling2/frontend/frontend.h"
#include "models/inkling2/model.h"
#include "ninfer/types.h"
#include "runtime/contract/request.h"

#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <exception>
#include <filesystem>
#include <memory>
#include <mutex>
#include <optional>
#include <thread>
#include <vector>

namespace ninfer::models::inkling2 {
struct ProgramLimits;
class DenseWeightsQ8;
struct DeviceLora;
struct DeviceAblation;
} // namespace ninfer::models::inkling2

namespace ninfer::runtime {

class Inkling2Core {
public:
    // What a request may ask of Inkling-2; throws std::invalid_argument. The Engine runs it for every request, also
    // those it answers without the core (no output tokens).
    void check_request(const ResolvedRequestOptions& options) const;

    struct Job;

    // What GenerationHandle holds: wait() streams the job's events to the sink and returns its result.
    class Submission {
    public:
        explicit Submission(std::shared_ptr<Job> job) : job_(std::move(job)) {}
        ~Submission();
        Submission(Submission&&) noexcept            = default;
        Submission& operator=(Submission&&) noexcept = default;
        Submission(const Submission&)                = delete;
        Submission& operator=(const Submission&)     = delete;
        GenerationResult wait(OutputSink* sink, const CancellationView& cancellation);
        [[nodiscard]] std::optional<std::uint32_t> effective_thinking_budget() const noexcept;

    private:
        std::shared_ptr<Job> job_;
    };

    // Loads the artifact and starts the worker. `options` is the Engine's, normalized.
    Inkling2Core(const EngineOptions& options, DeviceContext& device);
    ~Inkling2Core();
    Inkling2Core(const Inkling2Core&)            = delete;
    Inkling2Core& operator=(const Inkling2Core&) = delete;

    [[nodiscard]] const models::inkling2::Frontend& frontend() const noexcept { return *frontend_; }
    [[nodiscard]] LoadSummary load_summary() const { return load_; }
    // The context one request may reach (the Engine's max_context; with max_concurrency > 1 it may be lower than
    // asked: the slots' rings were fitted to the device) and the requests run at once.
    [[nodiscard]] std::uint32_t max_context() const noexcept { return options_.max_context; }
    [[nodiscard]] std::uint32_t max_concurrency() const noexcept { return static_cast<std::uint32_t>(slots_); }

    Submission submit(models::inkling2::PreparedPrompt prompt, PromptSummary summary, double prepare_seconds,
                      ResolvedRequestOptions options, OutputConsumerMode consumer_mode,
                      GenerationObservationOptions observation, std::chrono::steady_clock::time_point pending_deadline);

    [[nodiscard]] MemorySummary memory_summary() const;
    [[nodiscard]] RuntimeStats runtime_stats() const;
    [[nodiscard]] bool is_available() const noexcept { return !failed_; }
    [[nodiscard]] bool has_failed() const noexcept { return failed_; }
    ResidencyStatus suspend(std::optional<bool> auto_resume);
    ResidencyStatus resume();
    [[nodiscard]] ResidencyStatus residency() const { return {}; }
    void reset_memory_peaks() noexcept {}

private:
    void worker();
    void fail_queued_locked(const char* why); // with mutex_ held
    void run(Job& job);
    // max_concurrency > 1: the scheduler (worker_batched) and one request's steps in its slot.
    struct Active;
    void worker_batched();
    bool start(Active& a);                       // admission to the first prefill chunk; false when already finished
    bool prefill_step(Active& a);                // one chunk; false when finished (cancelled)
    bool commit(Active& a, TokenId next, bool injected); // a chosen token; false when the request is finished
    void finish(Active& a);                      // the result and the job's end, after its last token
    void end_job(Job& job, std::exception_ptr error, std::uint32_t lane); // marks it done; reports a fault
    void release_stream() noexcept; // the programs, then the stream they run on

    EngineOptions options_;
    DeviceContext& device_;
    LoadSummary load_;
    std::unique_ptr<models::inkling2::Model> model_;
    std::unique_ptr<models::inkling2::Frontend> frontend_;
    // Output constraints' grammars over the vocabulary, built on the first constrained request.
    std::unique_ptr<models::inkling2::ConstraintCompiler> constraints_;
    cudaStream_t stream_ = nullptr;
    // The prefix-reuse snapshots' device-to-host copies (TextProgram::save_async), beside the forwards on stream_.
    cudaStream_t copy_stream_ = nullptr;
    struct Programs;
    void build_programs(models::inkling2::ProgramLimits limits, std::unique_ptr<models::inkling2::DenseWeightsQ8>& dense,
                        const models::inkling2::VisionConfig* vision, bool dspark);
    std::unique_ptr<Programs> programs_;
    // The opt-in adapters (models/inkling2/program/adapters.h): options_.loras and options_.grafts in order, which a
    // prompt's lora and graft index, and their digests (prefix reuse keys a snapshot on the pair a prompt ran with).
    std::vector<std::unique_ptr<models::inkling2::DeviceLora>> loras_;
    std::vector<std::unique_ptr<models::inkling2::DeviceAblation>> grafts_;
    std::vector<std::uint64_t> lora_digests_, graft_digests_;
    std::int32_t chunk_ = 256; // prompt tokens per forward, chosen at startup
    bool drafter_head_copy_ = true; // the drafter's 4-bit copy of the head, unless memory is short
    std::uint32_t drafter_window_ = 0; // the drafter's global window (0: the whole context), chosen at startup
    std::int32_t slots_ = 1;           // max_concurrency: the text Program's slots
    std::uint32_t requested_context_ = 0; // max_context as asked, before fitting the slots
    std::size_t snapshot_host_bytes_ = 0; // the prefix-reuse snapshots' pinned host memory (build_programs)
    std::size_t free_after_weights_  = 0; // device memory free before the programs (measured with several slots)

    mutable std::mutex mutex_;
    std::condition_variable wake_;
    std::deque<std::shared_ptr<Job>> queue_;
    bool stop_    = false;
    std::uint32_t running_ = 0; // requests the worker has taken from the queue and not finished
    std::atomic<bool> failed_{false};
    std::atomic<std::uint64_t> next_id_{1};
    std::uint32_t consecutive_failures_ = 0; // the worker's streak of faulted requests (EngineFaultEvent)
    // runtime_stats(): the monotonic counters and the one lane's gauges, which the worker updates as it runs a
    // request. Taken alone or after mutex_, never before it.
    mutable std::mutex stats_mutex_;
    RuntimeStats totals_;   // the counters; its gauges and lanes are filled in by runtime_stats()
    std::array<RuntimeLaneStats, kMaximumConcurrency> lanes_{}; // per slot; Idle between requests
    std::array<std::chrono::steady_clock::time_point, kMaximumConcurrency> lane_first_token_{};
    std::thread thread_;
};

// True when the artifact's text component is Inkling-2.
[[nodiscard]] bool artifact_is_inkling2(const std::filesystem::path& artifact) noexcept;

} // namespace ninfer::runtime
