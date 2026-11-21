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

namespace ninfer::runtime {

class Inkling2Core {
public:
    struct Job;

    // What GenerationHandle holds: wait() streams the job's events to the sink and returns its result.
    class Submission {
    public:
        explicit Submission(std::shared_ptr<Job> job) : job_(std::move(job)) {}
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
    void run(Job& job);

    EngineOptions options_;
    DeviceContext& device_;
    LoadSummary load_;
    std::unique_ptr<models::inkling2::Model> model_;
    std::unique_ptr<models::inkling2::Frontend> frontend_;
    cudaStream_t stream_ = nullptr;
    struct Programs;
    std::unique_ptr<Programs> programs_;

    mutable std::mutex mutex_;
    std::condition_variable wake_;
    std::deque<std::shared_ptr<Job>> queue_;
    bool stop_ = false;
    std::atomic<bool> failed_{false};
    std::atomic<std::uint64_t> next_id_{1};
    std::atomic<std::uint64_t> prompt_tokens_{0}, generated_tokens_{0};
    std::thread thread_;
};

// True when the artifact's text component is Inkling-2.
[[nodiscard]] bool artifact_is_inkling2(const std::filesystem::path& artifact) noexcept;

} // namespace ninfer::runtime
