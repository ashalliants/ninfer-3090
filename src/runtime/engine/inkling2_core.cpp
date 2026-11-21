// Written for NInfer-3090. See inkling2_core.h.

#include "runtime/engine/inkling2_core.h"

#include "artifact/framing.h"
#include "artifact/reader.h"
#include "core/arena.h"
#include "models/inkling2/program/text_program.h"
#include "models/inkling2/program/vision_program.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

namespace ninfer::runtime {
namespace {

namespace ink = models::inkling2;
using Clock   = std::chrono::steady_clock;

double seconds_between(Clock::time_point a, Clock::time_point b) { return std::chrono::duration<double>(b - a).count(); }

void check_cuda(cudaError_t status, const char* what) {
    if (status != cudaSuccess) { throw std::runtime_error(std::string("inkling2 core: ") + what + ": " + cudaGetErrorString(status)); }
}

constexpr std::int32_t kChunk = 256; // prompt tokens per forward
// <|end_message|><|message_model|><|content_text|>: what a thinking budget injects to close the reasoning.
constexpr std::uint32_t kCloseTokens = 3;
// <|message_model|><|content_thinking|>: the model opens its own turn (the template has no generation prompt).
constexpr std::uint32_t kOpenTokens = 2;

// One token from the last column's logits (OpenAI presence and frequency penalties over the generated tokens, then
// temperature, top-k, min-p and top-p). Temperature 0 is argmax, the lowest id among equals.
TokenId sample(std::vector<float>& logits, const ResolvedSamplingParameters& p,
               const std::unordered_map<TokenId, std::uint32_t>& counts, std::mt19937_64& rng) {
    if (p.presence_penalty != 0.0F || p.frequency_penalty != 0.0F) {
        for (const auto& [id, n] : counts) {
            logits[static_cast<std::size_t>(id)] -= p.presence_penalty + p.frequency_penalty * static_cast<float>(n);
        }
    }
    const auto better = [&](std::int32_t a, std::int32_t b) {
        const float x = logits[static_cast<std::size_t>(a)], y = logits[static_cast<std::size_t>(b)];
        if (std::isnan(y)) { return !std::isnan(x) || a < b; }
        return x > y || (x == y && a < b);
    };
    if (p.temperature <= 0.0F) {
        std::int32_t best = 0;
        for (std::int32_t v = 1; v < static_cast<std::int32_t>(logits.size()); ++v) {
            if (better(v, best)) { best = v; }
        }
        return best;
    }
    const std::size_t k = static_cast<std::size_t>(std::clamp<std::int32_t>(p.top_k, 1, static_cast<std::int32_t>(logits.size())));
    std::vector<std::int32_t> ids(logits.size());
    std::iota(ids.begin(), ids.end(), 0);
    std::partial_sort(ids.begin(), ids.begin() + static_cast<std::ptrdiff_t>(k), ids.end(), better);
    ids.resize(k);
    std::vector<double> prob(k);
    const double top = logits[static_cast<std::size_t>(ids[0])];
    double sum       = 0.0;
    for (std::size_t i = 0; i < k; ++i) {
        prob[i] = std::exp((logits[static_cast<std::size_t>(ids[i])] - top) / p.temperature);
        sum += prob[i];
    }
    for (double& q : prob) { q /= sum; }
    std::size_t keep = k;
    if (p.min_p > 0.0F) {
        while (keep > 1 && prob[keep - 1] < p.min_p * prob[0]) { --keep; }
    }
    if (p.top_p < 1.0F) {
        double mass = 0.0;
        for (std::size_t i = 0; i < keep; ++i) {
            mass += prob[i];
            if (mass >= p.top_p) {
                keep = i + 1;
                break;
            }
        }
    }
    const double total = std::accumulate(prob.begin(), prob.begin() + static_cast<std::ptrdiff_t>(keep), 0.0);
    double u           = std::uniform_real_distribution<double>(0.0, total)(rng);
    for (std::size_t i = 0; i < keep; ++i) {
        if ((u -= prob[i]) < 0.0) { return ids[i]; }
    }
    return ids[keep - 1];
}

} // namespace

struct Inkling2Core::Programs {
    ink::TextProgram text;
    std::unique_ptr<ink::VisionEncoder> vision;
    Programs(const ink::Model& model, ink::ProgramLimits limits, cudaStream_t stream) : text(model, std::move(limits), stream) {}
};

struct Inkling2Core::Job {
    std::uint64_t id = 0;
    ink::PreparedPrompt prompt;
    PromptSummary summary;
    double prepare_seconds = 0.0;
    ResolvedRequestOptions options;
    OutputConsumerMode mode = OutputConsumerMode::Aggregate;
    GenerationObservationOptions observation;
    Clock::time_point submitted, deadline;

    using Event = std::variant<GenerationStart, PromptProgress, GenerationTimingObservation, OutputDelta,
                               GenerationFirstTokenObservation>;
    std::mutex mutex;
    std::condition_variable changed;
    std::deque<Event> events;
    bool done = false;
    GenerationResult result;
    std::exception_ptr error;
    std::atomic<bool> cancelled{false};

    void post(Event event) {
        {
            std::lock_guard lock(mutex);
            events.push_back(std::move(event));
        }
        changed.notify_all();
    }
};

GenerationResult Inkling2Core::Submission::wait(OutputSink* sink, const CancellationView& cancellation) {
    if (!job_) { throw std::logic_error("inkling2 submission already waited"); }
    const std::shared_ptr<Job> job = std::move(job_);
    if ((job->mode == OutputConsumerMode::Streaming) != (sink != nullptr)) {
        throw std::invalid_argument("GenerationHandle wait sink does not match its submitted consumer mode");
    }
    for (;;) {
        if (cancellation.requested()) { job->cancelled = true; }
        std::deque<Job::Event> events;
        bool done = false;
        {
            std::unique_lock lock(job->mutex);
            job->changed.wait_for(lock, std::chrono::milliseconds(10), [&] { return job->done || !job->events.empty(); });
            events.swap(job->events);
            done = job->done;
        }
        // A consumer callback that throws ends the request: the worker stops at its next token.
        try {
        for (Job::Event& event : events) {
            std::visit(
                [&](auto& e) {
                    using E = std::decay_t<decltype(e)>;
                    if constexpr (std::is_same_v<E, GenerationFirstTokenObservation>) {
                        if (job->observation.first_token) { job->observation.first_token(e); }
                    } else if (sink != nullptr) {
                        if constexpr (std::is_same_v<E, GenerationStart>) { sink->start(std::move(e)); }
                        if constexpr (std::is_same_v<E, PromptProgress>) { sink->progress(e); }
                        if constexpr (std::is_same_v<E, GenerationTimingObservation>) { sink->timing(e); }
                        if constexpr (std::is_same_v<E, OutputDelta>) { sink->publish(std::move(e)); }
                    }
                },
                event);
        }
        } catch (...) {
            job->cancelled = true;
            throw;
        }
        if (done) {
            if (job->error) { std::rethrow_exception(job->error); }
            return std::move(job->result);
        }
    }
}

std::optional<std::uint32_t> Inkling2Core::Submission::effective_thinking_budget() const noexcept {
    return job_ ? job_->options.execution.thinking.effective_budget : std::nullopt;
}

Inkling2Core::Submission::~Submission() {
    // Dropped without wait(): the job is abandoned (skipped if still queued, stopped at its next step if running).
    if (job_) { job_->cancelled = true; }
}

Inkling2Core::Inkling2Core(const EngineOptions& options, DeviceContext& device) : options_(options), device_(device) {
    if (options_.purpose != EnginePurpose::Generation) { throw std::invalid_argument("Inkling-2 serves generation only"); }
    if (options_.devices.size() > 1) { throw std::invalid_argument("Inkling-2 runs on one device"); }
    const auto start = Clock::now();
    const artifact::Reader reader(options_.artifact_path);
    model_ = ink::load_model(reader, device_, &options_.startup_observer);
    if (options_.enable_vision && model_->vision_config() == nullptr) {
        throw std::invalid_argument("--vision: this Inkling-2 artifact has no vision tower (convert it with --mmproj)");
    }
    const ink::VisionConfig* vision = options_.enable_vision ? model_->vision_config() : nullptr;
    ink::FrontendOptions frontend_options{.max_context = options_.max_context, .vision = vision};
    frontend_ = std::make_unique<ink::Frontend>(model_->resources(), frontend_options);
    check_cuda(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking), "stream");
    ink::ProgramLimits limits;
    limits.max_context = static_cast<std::int32_t>(options_.max_context);
    limits.max_chunk   = kChunk;
    programs_          = std::make_unique<Programs>(*model_, limits, stream_);
    if (vision != nullptr) { programs_->vision = std::make_unique<ink::VisionEncoder>(*model_, stream_); }
    const auto& stats         = model_->storage_stats();
    load_.architecture        = "inkling2";
    load_.model_name          = "Inkling-2-Mini";
    load_.artifact_id         = artifact::artifact_id_hex(reader.artifact_id());
    load_.weight_formats      = {"bf16", "ggml_q8_0", "ggml_rec_nvfp4_nvfp4", "fp32"};
    load_.prefill_signature   = "inkling2-fp32";
    load_.takes_video         = false;
    load_.load_seconds        = seconds_between(start, Clock::now());
    load_.upload_seconds       = stats.upload_seconds;
    load_.artifact_bytes_read  = stats.read_bytes;
    load_.host_to_device_bytes = stats.h2d_bytes;
    load_.peak_staging_bytes   = stats.peak_staging_bytes;
    load_.device_object_count  = stats.device_object_count;
    load_.host_object_count    = stats.host_object_count;
    thread_ = std::thread([this] { worker(); });
}

Inkling2Core::~Inkling2Core() {
    {
        std::lock_guard lock(mutex_);
        stop_ = true;
    }
    wake_.notify_all();
    if (thread_.joinable()) { thread_.join(); }
    programs_.reset();
    if (stream_ != nullptr) { cudaStreamDestroy(stream_); }
}

Inkling2Core::Submission Inkling2Core::submit(ink::PreparedPrompt prompt, PromptSummary summary, double prepare_seconds,
                                              ResolvedRequestOptions options, OutputConsumerMode consumer_mode,
                                              GenerationObservationOptions observation, Clock::time_point pending_deadline) {
    if (options.constraint) { throw std::invalid_argument("Inkling-2 serves no output constraints yet"); }
    if (options.tool_choice.mode == ToolChoiceMode::Required || options.tool_choice.allowed_names) {
        throw std::invalid_argument("Inkling-2 serves tool_choice auto or none only");
    }
    if (options.output.raw || options.stop.publish_stop_token) {
        throw std::invalid_argument("Inkling-2 publishes decoded text only: no raw output or published stop token");
    }
    if (options.execution.thinking.loop != ReasoningLoopAction::Off) {
        throw std::invalid_argument("Inkling-2 has no reasoning-loop guard: run without --reasoning-loop");
    }
    // allow_prefix_reuse only permits reuse: with no context cache every request runs from position 0, which honours it.
    const auto submitted = Clock::now();
    if (pending_deadline == Clock::time_point{}) {
        pending_deadline = submitted + std::chrono::milliseconds(options_.pending_timeout_ms);
    }
    if (submitted >= pending_deadline) {
        throw RequestError(RequestErrorKind::QueueTimeout, "inference request expired before submission");
    }
    // A thinking budget counts reasoning tokens after the model's kOpenTokens turn opening, closes the reasoning
    // with kCloseTokens control tokens, and the answer needs at least one more: the budget takes effect only where
    // all of those fit in the output limit (the effective budget).
    if (const auto budget = options.execution.thinking.budget) {
        constexpr std::uint32_t frame = kOpenTokens + kCloseTokens + 1;
        const std::uint32_t limit     = options.execution.requested_output_tokens;
        options.execution.thinking.effective_budget =
            limit > frame ? std::optional(std::min(*budget, limit - frame)) : std::nullopt;
    }
    auto job             = std::make_shared<Job>();
    job->id              = next_id_++;
    job->prompt          = std::move(prompt);
    job->summary         = summary;
    job->prepare_seconds = prepare_seconds;
    job->options         = std::move(options);
    job->mode            = consumer_mode;
    job->observation     = std::move(observation);
    job->submitted       = submitted;
    job->deadline        = pending_deadline;
    {
        std::lock_guard lock(mutex_);
        if (failed_) { throw RequestError(RequestErrorKind::Unavailable, "the Inkling-2 Engine has failed"); }
        // One running request and max_pending_requests waiting, as the Engine core admits with one lane.
        if (queue_.size() + (running_ ? 1U : 0U) >= std::size_t{options_.max_pending_requests} + 1U) {
            throw RequestError(RequestErrorKind::Overloaded, "the Inkling-2 queue is full");
        }
        queue_.push_back(job);
    }
    wake_.notify_all();
    return Submission(std::move(job));
}

void Inkling2Core::fail_queued_locked(const char* why) {
    for (const auto& waiting : queue_) {
        {
            std::lock_guard job_lock(waiting->mutex);
            waiting->error = std::make_exception_ptr(RequestError(RequestErrorKind::Unavailable, why));
            waiting->done  = true;
        }
        waiting->changed.notify_all();
    }
    queue_.clear();
}

void Inkling2Core::worker() {
    device_.bind_to_current_thread();
    for (;;) {
        std::shared_ptr<Job> job;
        {
            std::unique_lock lock(mutex_);
            wake_.wait(lock, [&] { return stop_ || !queue_.empty(); });
            if (stop_ || failed_) {
                // The Engine is going, or the device failed: what still waits is not run.
                fail_queued_locked(stop_ ? "the Inkling-2 Engine is shutting down" : "the Inkling-2 Engine has failed");
                if (stop_) { return; }
                continue;
            }
            job = std::move(queue_.front());
            queue_.pop_front();
            running_ = true;
        }
        try {
            run(*job);
        } catch (...) {
            job->error = std::current_exception();
            // A CUDA failure leaves the device state unknown: no later request runs, queued ones fail now.
            if (cudaGetLastError() != cudaSuccess || cudaStreamQuery(stream_) == cudaErrorIllegalAddress) {
                std::lock_guard lock(mutex_);
                failed_ = true;
                fail_queued_locked("the Inkling-2 Engine has failed");
            }
        }
        {
            std::lock_guard lock(job->mutex);
            job->done = true;
        }
        job->changed.notify_all();
        {
            std::lock_guard lock(mutex_);
            running_ = false;
        }
    }
}

void Inkling2Core::run(Job& job) {
    const auto began      = Clock::now();
    GenerationResult& out = job.result;
    out.engine_request_id = job.id;
    out.prompt            = job.summary;
    out.timings.prepare_seconds        = job.prepare_seconds;
    out.thinking.requested_budget      = job.options.execution.thinking.budget;
    out.thinking.effective_budget      = job.options.execution.thinking.effective_budget;
    out.engine_timing.queue_wait_seconds = seconds_between(job.submitted, began);
    if (job.deadline != Clock::time_point{} && began > job.deadline) {
        throw RequestError(RequestErrorKind::QueueTimeout, "the request waited past its deadline");
    }
    const bool streaming = job.mode == OutputConsumerMode::Streaming;
    // A stream starts exactly once, also when it was cancelled while queued.
    if (streaming) { job.post(GenerationStart{.prompt = job.summary}); }
    if (job.cancelled) {
        out.finish_reason = FinishReason::Cancelled;
        return;
    }
    const auto& prompt   = job.prompt;
    const auto& text_cfg = model_->config();
    const std::int32_t n = static_cast<std::int32_t>(prompt.tokens.size());

    // Images: each encoding copied out, since a chunk may need the end of one and the start of the next.
    const auto vision_start = Clock::now();
    std::vector<DeviceBuffer> encoded;
    for (const ink::PromptImage& image : prompt.images) {
        const Tensor emb = programs_->vision->encode(image.image);
        encoded.emplace_back(emb.bytes());
        check_cuda(cudaMemcpyAsync(encoded.back().p, emb.data, emb.bytes(), cudaMemcpyDeviceToDevice, stream_), "image copy");
    }
    if (!prompt.images.empty()) {
        check_cuda(cudaStreamSynchronize(stream_), "vision");
        out.timings.vision_seconds = seconds_between(vision_start, Clock::now());
    }

    // Prefill from position 0, in chunks.
    auto& program = programs_->text;
    program.reset();
    const auto prefill_start = Clock::now();
    const std::size_t hidden = text_cfg.hidden_size;
    Tensor logits;
    for (std::int32_t s = 0; s < n; s += kChunk) {
        if (job.cancelled) {
            out.finish_reason = FinishReason::Cancelled;
            return;
        }
        const std::int32_t len = std::min(kChunk, n - s);
        std::vector<ink::InputEmbeddings> inputs;
        for (std::size_t i = 0; i < prompt.images.size(); ++i) {
            const std::int32_t first = prompt.images[i].column, count = prompt.images[i].image.tokens();
            const std::int32_t lo = std::max(s, first), hi = std::min(s + len, first + count);
            if (lo >= hi) { continue; }
            const Tensor all(encoded[i].p, DType::FP32, {static_cast<std::int32_t>(hidden), count});
            inputs.push_back({lo - s, all.slice(1, lo - first, hi - lo)});
        }
        logits = program.forward(std::span(prompt.tokens).subspan(static_cast<std::size_t>(s), static_cast<std::size_t>(len)), inputs);
        if (job.observation.prompt_progress && streaming) {
            check_cuda(cudaStreamSynchronize(stream_), "prefill");
            job.post(PromptProgress{.total_prompt_tokens     = static_cast<std::uint32_t>(n),
                                    .processed_prompt_tokens = static_cast<std::uint32_t>(s + len),
                                    .elapsed_ns = static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                                                                 Clock::now() - prefill_start).count())});
        }
    }
    out.computed_prefill_tokens = static_cast<std::uint32_t>(n);
    prompt_tokens_ += static_cast<std::uint64_t>(n);

    // Decode.
    const auto& exec  = job.options.execution;
    const auto& stop  = job.options.stop;
    std::vector<TokenId> stop_ids = stop.token_ids;
    if (stop.include_model_defaults) { stop_ids.push_back(ink::token::kEndSampling); }
    ink::OutputDecoder decoder(*frontend_, stop.strings, job.options.tool_choice.mode != ToolChoiceMode::None);
    std::mt19937_64 rng(exec.sampling.seed);
    std::unordered_map<TokenId, std::uint32_t> counts;
    std::vector<float> host(text_cfg.unpadded_vocab_size);
    std::deque<TokenId> forced; // a thinking budget's close: end the reasoning, open the answer
    const auto decode_start = Clock::now();
    Clock::time_point first_token{};
    std::uint32_t thinking_tokens = 0;
    const auto emit = [&](std::vector<OutputDelta> deltas) {
        if (!streaming) { return; }
        for (OutputDelta& d : deltas) { job.post(std::move(d)); }
    };
    out.finish_reason = FinishReason::OutputLimit;
    for (;;) {
        if (job.cancelled) {
            out.finish_reason = FinishReason::Cancelled;
            break;
        }
        TokenId next = 0;
        bool injected = false;
        if (!forced.empty()) {
            next     = forced.front();
            injected = true;
            forced.pop_front();
        } else {
            const Tensor last = logits.slice(1, logits.ne[1] - 1, 1);
            check_cuda(cudaMemcpyAsync(host.data(), last.data, host.size() * 4, cudaMemcpyDeviceToHost, stream_), "logits");
            check_cuda(cudaStreamSynchronize(stream_), "logits");
            next = sample(host, exec.sampling, counts, rng);
        }
        if (first_token == Clock::time_point{}) {
            first_token = Clock::now();
            out.timings.prefill_seconds     = seconds_between(prefill_start, first_token);
            out.timings.first_token_seconds = seconds_between(job.submitted, first_token) + job.prepare_seconds;
            job.post(GenerationFirstTokenObservation{.prepare_seconds              = job.prepare_seconds,
                                                     .elapsed_since_submit_seconds = seconds_between(job.submitted, first_token),
                                                     .queue_wait_seconds           = out.engine_timing.queue_wait_seconds});
        }
        const bool is_stop = !injected && std::find(stop_ids.begin(), stop_ids.end(), next) != stop_ids.end();
        out.generated_token_ids.push_back(next);
        ++counts[next];
        ++generated_tokens_;
        if (is_stop) {
            out.finish_reason = FinishReason::StopToken;
            break;
        }
        const bool reasoning_before = decoder.in_reasoning();
        auto step = decoder.push(next);
        emit(std::move(step.deltas));
        if (reasoning_before && !injected) { ++thinking_tokens; }
        if (step.stop) {
            out.finish_reason       = FinishReason::StopString;
            out.matched_stop_string = decoder.matched_stop();
            break;
        }
        // The close also comes early when only it and one answer token still fit (a longer opening than kOpenTokens).
        const bool no_room = out.generated_token_ids.size() + kCloseTokens + 1 >= exec.requested_output_tokens;
        if (exec.thinking.effective_budget && decoder.in_reasoning() && forced.empty() && !out.thinking.applied &&
            (thinking_tokens >= *exec.thinking.effective_budget || no_room) &&
            program.position() + 2 + kCloseTokens < static_cast<std::int64_t>(options_.max_context)) {
            forced                    = {ink::token::kEndMessage, ink::token::kMessageModel, ink::token::kContentText};
            out.thinking.applied      = true;
            out.thinking.injected_tokens += kCloseTokens;
        }
        if (job.observation.live_timings && streaming) {
            const auto now = Clock::now();
            job.post(GenerationTimingObservation{
                .generated_tokens      = static_cast<std::uint32_t>(out.generated_token_ids.size()),
                .prompt_elapsed_ns     = static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(first_token - prefill_start).count()),
                .generation_elapsed_ns = static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(now - first_token).count())});
        }
        if (out.generated_token_ids.size() >= exec.requested_output_tokens) { break; }
        if (program.position() + 1 >= static_cast<std::int64_t>(options_.max_context)) {
            out.finish_reason = FinishReason::ContextCapacity;
            break;
        }
        const std::int32_t one[1] = {next};
        logits = program.forward(one);
    }
    emit(decoder.finish());
    if (decoder.matched_stop() && out.finish_reason != FinishReason::StopString) {
        // A stop string inside an unfinished call's text, which finish() publishes as content.
        out.finish_reason       = FinishReason::StopString;
        out.matched_stop_string = decoder.matched_stop();
    }
    out.content                 = decoder.content();
    out.reasoning               = decoder.reasoning();
    out.tool_calls              = decoder.tool_calls();
    out.tool_call_parse         = decoder.tool_call_parse();
    out.reasoning_tokens        = decoder.reasoning_tokens();
    out.thinking.model_thinking_tokens = thinking_tokens;
    const auto end              = Clock::now();
    out.timings.decode_seconds  = first_token == Clock::time_point{} ? 0.0 : seconds_between(first_token, end);
    out.timings.prompt_wall_seconds     = first_token == Clock::time_point{} ? 0.0 : seconds_between(vision_start, first_token);
    out.timings.generation_wall_seconds = out.timings.decode_seconds;
    out.timings.total_seconds   = seconds_between(job.submitted, end) + job.prepare_seconds;
}

bool artifact_is_inkling2(const std::filesystem::path& artifact) noexcept {
    try {
        const artifact::Reader reader(artifact);
        return models::inkling2::is_inkling2(reader);
    } catch (...) { return false; }
}

MemorySummary Inkling2Core::memory_summary() const {
    MemorySummary m;
    m.device                = options_.device;
    m.max_context           = options_.max_context;
    m.requested_max_context = options_.max_context;
    m.kv_capacity           = options_.max_context;
    return m;
}

RuntimeStats Inkling2Core::runtime_stats() const {
    RuntimeStats s;
    s.prompt_tokens    = prompt_tokens_;
    s.generated_tokens = generated_tokens_;
    return s;
}

ResidencyStatus Inkling2Core::suspend(std::optional<bool>) {
    throw std::invalid_argument("model suspend is not available for Inkling-2");
}

ResidencyStatus Inkling2Core::resume() { throw std::invalid_argument("model suspend is not available for Inkling-2"); }

} // namespace ninfer::runtime
