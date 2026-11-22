#pragma once

// Written for NInfer-3090. The Engine's core for an Inkling-2 artifact (docs/maintainer/inkling2-model.md,
// "Serving"): the model, its frontend, and one worker thread that runs requests one at a time, in submission
// order, each from position 0 through the correctness-first text Program (no prefix reuse, no batching). Output
// is sampled on the host from the last column's logits.

#include "core/device.h"
#include "models/inkling2/frontend/frontend.h"
#include "models/inkling2/model.h"
#include "ninfer/types.h"
#include "runtime/contract/request.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
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
    void release_stream() noexcept; // the programs, then the stream they run on

    EngineOptions options_;
    DeviceContext& device_;
    LoadSummary load_;
    std::unique_ptr<models::inkling2::Model> model_;
    std::unique_ptr<models::inkling2::Frontend> frontend_;
    cudaStream_t stream_ = nullptr;
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
    bool drafter_head_copy_ = true; // the drafter's Q8 copy of the head, unless memory is short
    std::uint32_t drafter_window_ = 0; // the drafter's global window (0: the whole context), chosen at startup

    mutable std::mutex mutex_;
    std::condition_variable wake_;
    std::deque<std::shared_ptr<Job>> queue_;
    bool stop_    = false;
    bool running_ = false; // the worker has a request
    std::atomic<bool> failed_{false};
    std::atomic<std::uint64_t> next_id_{1};
    std::uint32_t consecutive_failures_ = 0; // the worker's streak of faulted requests (EngineFaultEvent)
    // runtime_stats(): the monotonic counters and the one lane's gauges, which the worker updates as it runs a
    // request. Taken alone or after mutex_, never before it.
    mutable std::mutex stats_mutex_;
    RuntimeStats totals_;   // the counters; its gauges and lanes are filled in by runtime_stats()
    RuntimeLaneStats lane_; // Idle between requests
    std::chrono::steady_clock::time_point lane_first_token_{};
    std::thread thread_;
};

// True when the artifact's text component is Inkling-2.
[[nodiscard]] bool artifact_is_inkling2(const std::filesystem::path& artifact) noexcept;

} // namespace ninfer::runtime
