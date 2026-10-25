// EngineCore's scheduling, run against the GPU-free simulation in fake_engine.h.
//
// These tests exercise the real worker loop, admission, pause/restore and stats code with a fake
// model, so they check what the Engine does and in what order, not what a GPU would. The fake's
// cost model and the Engine's are the same numbers, so a prefill step of N tokens "takes"
// fixed + N * per_token on both sides.

#include "runtime/fake_engine.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <filesystem>
#include <future>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <thread>

namespace {

using namespace ninfer;
using namespace ninfer::runtime;
using Clock = std::chrono::steady_clock;
using fake::Event;

constexpr std::uint64_t kFixedNs    = 10'000'000; // 10 ms per prefill step
constexpr std::uint64_t kPerTokenNs = 100'000;    // 0.1 ms per token

void require(bool condition, const std::string& message) {
    if (!condition) { throw std::runtime_error(message); }
}

using Result = std::shared_future<GenerationResult>;

struct Settings {
    std::uint32_t chunk          = 1024;
    std::uint32_t lanes          = 2;
    std::uint32_t stall_budget   = 0;
    std::uint32_t decode_share   = 0;
    std::uint32_t yield_ms       = 0;
    std::chrono::microseconds decode_sleep{1000};
    // Real time a prefill step takes, for a test that must submit while one runs.
    std::chrono::microseconds prefill_sleep{0};
    std::uint32_t layer_slices = 0;
    bool admission_fit_check = false;
    std::uint32_t pending_timeout_ms = 60'000;
    // Lossy prefill: LossyPrefillOptions::min_tokens (0: off).
    std::uint32_t lossy_min_tokens = 0;
    bool context_cache             = false;
    // Shared prefixes installed before the Engine starts: (group, frontier).
    std::vector<std::pair<std::uint64_t, std::uint32_t>> shared_prefixes;
};

class Rig {
public:
    explicit Rig(Settings settings)
        : settings_(settings),
          instance_(settings.chunk, [&] {
              fake::Cost cost;
              cost.prefill_fixed_ns     = kFixedNs;
              cost.prefill_per_token_ns = kPerTokenNs;
              cost.decode_sleep         = settings.decode_sleep;
              cost.prefill_sleep        = settings.prefill_sleep;
              return cost;
          }()) {
        EngineOptions options;
        options.max_context           = 1U << 20;
        options.max_concurrency       = settings.lanes;
        options.max_pending_requests  = 8;
        options.pending_timeout_ms    = settings.pending_timeout_ms;
        options.prefill_chunk         = settings.chunk;
        options.context_cache.enabled = settings.context_cache;
        options.admission_fit_check   = settings.admission_fit_check;
        options.lossy_prefill.min_tokens = settings.lossy_min_tokens;
        for (const auto& [group, frontier] : settings.shared_prefixes) {
            instance_.external_checkpoints.push_back(
                instance_.program->add_shared_prefix(group, frontier));
        }
        options.prefill_schedule      = {.stall_budget_ms      = settings.stall_budget,
                                         .yield_remaining_ms   = settings.yield_ms,
                                         .decode_share_percent = settings.decode_share,
                                         .layer_slices         = settings.layer_slices};
        ContextMachineCostModel costs;
        costs.prefill.chunk_ns     = kFixedNs;
        costs.prefill.token_ns_q32 = kPerTokenNs << 32U;
        core_ = std::make_unique<EngineCore<fake::Instance>>(instance_, device_, options, costs);
    }

    ~Rig() {
        // Release a test that failed while holding the worker, then stop the Engine, which fails
        // whatever is still waiting so the waiter threads can end.
        release();
        // Stop the worker first, with the Engine still alive: it fails whatever is waiting, and a
        // waiter that has just been woken still calls back into the Engine (release_consumer).
        // Only after every waiter has ended is the Engine destroyed.
        core_->stop_worker();
        for (std::thread& waiter : waiters_) { waiter.join(); }
        core_.reset();
    }

    // Starts a request and returns its result future. `id` names it in the event log.
    Result run(std::uint64_t id, std::uint32_t prompt_tokens, std::uint32_t output_tokens,
               std::shared_ptr<std::atomic<bool>> cancel = nullptr) {
        return run_prompt(fake::PreparedPrompt{prompt_tokens, id}, output_tokens, std::move(cancel));
    }

    // `sink`, when given, streams the request (with prompt progress); it must outlive the Rig.
    Result run_prompt(fake::PreparedPrompt prompt, std::uint32_t output_tokens,
                      std::shared_ptr<std::atomic<bool>> cancel = nullptr,
                      OutputSink* sink                         = nullptr) {
        ResolvedRequestOptions request;
        request.execution.requested_output_tokens = output_tokens;
        PromptSummary summary;
        summary.prompt_tokens = prompt.tokens;
        GenerationObservationOptions observation;
        observation.prompt_progress = sink != nullptr;
        auto submission = std::make_shared<EngineCore<fake::Instance>::Submission>(core_->submit(
            prompt, summary, 0.0, std::move(request),
            sink != nullptr ? OutputConsumerMode::Streaming : OutputConsumerMode::Aggregate,
            observation));
        // A promise, not std::async: a future from async blocks in its destructor, which would
        // deadlock a failing test that still holds the worker.
        auto promise = std::make_shared<std::promise<GenerationResult>>();
        Result result = promise->get_future().share();
        waiters_.emplace_back([submission, promise, cancel, sink] {
            try {
                CancellationView view;
                if (cancel) {
                    view = CancellationView([cancel] { return cancel->load(std::memory_order_acquire); });
                }
                promise->set_value(submission->wait(sink, view));
            } catch (...) { promise->set_exception(std::current_exception()); }
        });
        return result;
    }

    // Gives the Engine a context store backed by a bucket. The store must outlive the Rig.
    void attach_store(ContextStore& store, std::chrono::milliseconds hydration_budget) {
        core_->set_context_store({.store            = &store,
                                  .binding          = "fake-binding",
                                  .hydration_budget = hydration_budget,
                                  .remote           = true});
    }

    fake::Program& program() { return *instance_.program; }
    EngineCore<fake::Instance>& core() { return *core_; }

    [[nodiscard]] std::vector<Event> events() const { return instance_.program->events(); }

    [[nodiscard]] std::vector<Event> events_of(std::uint64_t id, Event::Kind kind) const {
        std::vector<Event> out;
        for (const Event& event : events()) {
            if (event.request == id && event.kind == kind) { out.push_back(event); }
        }
        return out;
    }

    [[nodiscard]] std::size_t count(Event::Kind kind) const {
        std::size_t n = 0;
        for (const Event& event : events()) { n += event.kind == kind; }
        return n;
    }

    // Holds the worker thread inside the first unit for which `when` is true until release().
    void hold_at(std::function<bool(const Event&)> when) {
        std::lock_guard lock(gate_.mutex);
        gate_.when    = std::move(when);
        gate_.held    = false;
        gate_.release = false;
        program().before_unit = [this](const Event& event) {
            std::unique_lock lock(gate_.mutex);
            if (!gate_.when || gate_.held || !gate_.when(event)) { return; }
            gate_.held = true;
            gate_.cv.notify_all();
            gate_.cv.wait(lock, [&] { return gate_.release; });
        };
    }

    bool wait_held(std::chrono::milliseconds limit = std::chrono::seconds(20)) {
        std::unique_lock lock(gate_.mutex);
        return gate_.cv.wait_for(lock, limit, [&] { return gate_.held; });
    }

    void release() {
        std::lock_guard lock(gate_.mutex);
        gate_.release = true;
        gate_.cv.notify_all();
    }

    bool wait_for(const std::function<bool()>& condition,
                  std::chrono::milliseconds limit = std::chrono::seconds(20)) {
        const auto deadline = Clock::now() + limit;
        while (Clock::now() < deadline) {
            if (condition()) { return true; }
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
        }
        return condition();
    }

    [[nodiscard]] std::string dump() const {
        std::ostringstream out;
        const char* names[] = {"Bind",   "Restore",     "Prefill", "Decode",  "PauseStart",
                               "PauseDone", "Finish",   "Abort",   "EnableLossy", "Scout",
                               "ScoutEnd"};
        for (const Event& event : events()) {
            out << "  " << names[static_cast<int>(event.kind)] << " lane=" << event.lane
                << " req=" << event.request << " tokens=" << event.tokens << " cap=" << event.cap
                << " pos=" << event.position << '\n';
        }
        return out.str();
    }

private:
    struct Gate {
        std::mutex mutex;
        std::condition_variable cv;
        std::function<bool(const Event&)> when;
        bool held    = false;
        bool release = false;
    };

    Settings settings_;
    std::vector<std::thread> waiters_;
    fake::Instance instance_;
    fake::Device device_;
    Gate gate_;
    std::unique_ptr<EngineCore<fake::Instance>> core_;
};

// Waits for a request and reports whether it finished. A request that ended with an error throws,
// so a scenario cannot pass on its scheduling events while the request itself failed.
[[nodiscard]] bool finished(const Result& result,
                            std::chrono::milliseconds limit = std::chrono::seconds(30)) {
    if (result.wait_for(limit) != std::future_status::ready) { return false; }
    try {
        (void)result.get();
    } catch (const std::exception& error) {
        throw std::runtime_error(std::string("a request ended with an error: ") + error.what());
    }
    return true;
}

// Waits up to `limit` for a request that may legitimately still be running; if it did end, an
// error fails the scenario like finished() does.
void drain(const Result& result, std::chrono::milliseconds limit) { (void)finished(result, limit); }

// ---- tests -----------------------------------------------------------------------------------

void test_a_request_runs_to_completion() {
    Rig rig({.chunk = 256, .lanes = 1});
    auto result = rig.run(1, 1000, 3);
    require(finished(result), "the request did not finish\n" + rig.dump());
    (void)result.get();
    const auto prefill = rig.events_of(1, Event::Kind::Prefill);
    require(prefill.size() == 4 && prefill[0].tokens == 256 && prefill[3].tokens == 232 &&
                prefill[3].position == 768,
            "a 1000-token prompt did not prefill as 256+256+256+232\n" + rig.dump());
    require(std::all_of(prefill.begin(), prefill.end(), [](const Event& e) { return e.cap == 0; }),
            "a step was capped with every pacing option off\n" + rig.dump());
    // The first token comes from the last prefill step, so 3 output tokens need 2 decode rounds.
    require(rig.count(Event::Kind::Decode) == 2, "unexpected number of decode rounds\n" + rig.dump());
    require(rig.count(Event::Kind::Finish) == 1 && rig.count(Event::Kind::Bind) == 1,
            "the request did not bind and finish once\n" + rig.dump());
}

// Long output keeps a lane decoding while another request prefills.
constexpr std::uint32_t kLongOutput = 600;

bool decoding(Rig& rig, std::uint32_t rounds) {
    return rig.wait_for([&] { return rig.count(Event::Kind::Decode) >= rounds; });
}

void test_stall_budget_caps_steps_next_to_a_decoding_lane() {
    // 10 ms + 0.1 ms/token: an 80 ms budget fits 700 tokens, so steps run as 640.
    Rig rig({.chunk = 1024, .lanes = 2, .stall_budget = 80});
    auto stream = rig.run(1, 64, kLongOutput);
    require(decoding(rig, 3), "the neighbour never started decoding\n" + rig.dump());
    auto prompt = rig.run(2, 4096, 1);
    require(finished(prompt), "the long prompt did not finish\n" + rig.dump());
    (void)prompt.get();
    const auto steps = rig.events_of(2, Event::Kind::Prefill);
    std::uint32_t total = 0, capped = 0;
    for (const Event& step : steps) {
        total += step.tokens;
        if (step.cap != 0) {
            ++capped;
            require(step.cap == 640 && step.tokens <= 640,
                    "a capped step was not the largest multiple of 128 that fits\n" + rig.dump());
        }
    }
    require(total == 4096, "the capped steps did not cover the prompt\n" + rig.dump());
    require(capped >= 5, "the prompt was not paced next to a decoding lane\n" + rig.dump());
    require(rig.core().runtime_stats().paced_prefill_steps >= capped &&
                rig.core().runtime_stats().paced_prefill_deferred_tokens > 0,
            "capped steps were not counted");
    drain(stream, std::chrono::seconds(30));
}

void test_stall_budget_leaves_a_lone_prefill_alone() {
    Rig rig({.chunk = 1024, .lanes = 2, .stall_budget = 80});
    auto prompt = rig.run(1, 4096, 1);
    require(finished(prompt), "the prompt did not finish\n" + rig.dump());
    (void)prompt.get();
    for (const Event& step : rig.events_of(1, Event::Kind::Prefill)) {
        require(step.cap == 0 && step.tokens == 1024,
                "a prefill with nobody decoding was cut\n" + rig.dump());
    }
    require(rig.core().runtime_stats().paced_prefill_steps == 0, "a step was counted as paced");
}

void test_stall_budget_off_never_caps() {
    Rig rig({.chunk = 1024, .lanes = 2});
    auto stream = rig.run(1, 64, kLongOutput);
    require(decoding(rig, 3), "the neighbour never started decoding\n" + rig.dump());
    auto prompt = rig.run(2, 4096, 1);
    require(finished(prompt), "the long prompt did not finish\n" + rig.dump());
    (void)prompt.get();
    for (const Event& step : rig.events_of(2, Event::Kind::Prefill)) {
        require(step.cap == 0 && step.tokens == 1024, "a step was cut with the budget off\n" + rig.dump());
    }
    drain(stream, std::chrono::seconds(30));
}

// Decode rounds that ran between consecutive prefill steps of request `id`.
std::vector<std::size_t> decode_rounds_between_steps(const Rig& rig, std::uint64_t id) {
    std::vector<std::size_t> gaps;
    std::optional<std::size_t> since;
    for (const Event& event : rig.events()) {
        if (event.kind == Event::Kind::Prefill && event.request == id) {
            if (since) { gaps.push_back(*since); }
            since = 0;
        } else if (event.kind == Event::Kind::Decode && since) {
            ++*since;
        }
    }
    return gaps;
}

void test_decode_share_gives_the_stream_a_burst_per_step() {
    // A 1024-token step "takes" 112 ms; at 50% the Engine owes decode 112 ms, a 1 ms round at a time.
    // The debt follows the worker's wall clock (as the decode repayment does), so the fake step
    // really sleeps that long.
    Rig shared({.chunk = 1024, .lanes = 2, .decode_share = 50,
                .prefill_sleep = std::chrono::microseconds(112'000)});
    auto stream = shared.run(1, 64, 2000);
    require(decoding(shared, 3), "the neighbour never started decoding\n" + shared.dump());
    auto prompt = shared.run(2, 4096, 1);
    require(finished(prompt), "the long prompt did not finish\n" + shared.dump());
    (void)prompt.get();
    const auto gaps = decode_rounds_between_steps(shared, 2);
    require(gaps.size() == 3, "expected four prefill steps\n" + shared.dump());
    for (const std::size_t rounds : gaps) {
        require(rounds >= 60, "a 50% share ran only " + std::to_string(rounds) +
                                   " decode rounds between steps\n" + shared.dump());
    }
    require(shared.core().runtime_stats().deferred_prefill_turns > 150,
            "deferred turns were not counted");

    Rig plain({.chunk = 1024, .lanes = 2});
    auto plain_stream = plain.run(1, 64, 2000);
    require(decoding(plain, 3), "the neighbour never started decoding\n" + plain.dump());
    auto plain_prompt = plain.run(2, 4096, 1);
    require(finished(plain_prompt), "the long prompt did not finish\n" + plain.dump());
    (void)plain_prompt.get();
    for (const std::size_t rounds : decode_rounds_between_steps(plain, 2)) {
        require(rounds <= 2, "with no share a stream got " + std::to_string(rounds) +
                                 " rounds between steps\n" + plain.dump());
    }
    drain(stream, std::chrono::seconds(60));
    drain(plain_stream, std::chrono::seconds(60));
}

void test_decode_share_does_nothing_without_a_decoding_lane() {
    Rig rig({.chunk = 1024, .lanes = 2, .decode_share = 50});
    auto prompt = rig.run(1, 4096, 1);
    require(finished(prompt), "the prompt did not finish\n" + rig.dump());
    (void)prompt.get();
    require(rig.events_of(1, Event::Kind::Prefill).size() == 4, "unexpected prefill steps");
    require(rig.core().runtime_stats().deferred_prefill_turns == 0,
            "a prefill with nobody decoding was held back");
}

// A long prompt holds the only lane. Returns once the worker is held inside its second step.
void start_long_prompt(Rig& rig, Result& long_result) {
    rig.hold_at([](const Event& e) {
        return e.kind == Event::Kind::Prefill && e.request == 1 && e.position == 1024;
    });
    long_result = rig.run(1, 8192, 1);
    require(rig.wait_held(), "the long prompt never reached its second step\n" + rig.dump());
}

void test_without_yield_a_short_request_waits_for_the_long_prompt() {
    Rig rig({.chunk = 1024, .lanes = 1});
    Result long_result;
    start_long_prompt(rig, long_result);
    auto short_result = rig.run(2, 100, 1);
    rig.release();
    require(finished(long_result) && finished(short_result), "a request did not finish\n" + rig.dump());
    const auto events = rig.events();
    std::size_t long_finish = 0, short_bind = 0;
    for (std::size_t i = 0; i < events.size(); ++i) {
        if (events[i].kind == Event::Kind::Finish && events[i].request == 1) { long_finish = i; }
        if (events[i].kind == Event::Kind::Bind && events[i].request == 2) { short_bind = i; }
    }
    require(short_bind > long_finish, "the short request got in before the long prompt finished\n" + rig.dump());
    require(rig.count(Event::Kind::PauseStart) == 0, "a prefill was paused with yielding off");
}

void test_a_short_request_pauses_a_long_prefill_and_it_resumes() {
    Rig rig({.chunk = 1024, .lanes = 1, .yield_ms = 100});
    Result long_result;
    start_long_prompt(rig, long_result);
    auto short_result = rig.run(2, 100, 1);
    rig.release();
    require(finished(short_result), "the short request did not finish\n" + rig.dump());
    require(finished(long_result), "the long prompt did not finish\n" + rig.dump());

    const auto events = rig.events();
    auto index_of = [&](Event::Kind kind, std::uint64_t id) -> std::optional<std::size_t> {
        for (std::size_t i = 0; i < events.size(); ++i) {
            if (events[i].kind == kind && events[i].request == id) { return i; }
        }
        return std::nullopt;
    };
    const auto pause_start = index_of(Event::Kind::PauseStart, 1);
    const auto pause_done  = index_of(Event::Kind::PauseDone, 1);
    const auto short_bind  = index_of(Event::Kind::Bind, 2);
    const auto short_end   = index_of(Event::Kind::Finish, 2);
    const auto restore     = index_of(Event::Kind::Restore, 1);
    const auto long_end    = index_of(Event::Kind::Finish, 1);
    require(pause_start && pause_done && short_bind && short_end && restore && long_end,
            "the pause, the short request or the restore is missing from the log\n" + rig.dump());
    require(*pause_start < *pause_done && *pause_done < *short_bind && *short_bind < *short_end,
            "the short request did not run in the lane the pause freed\n" + rig.dump());
    require(*short_end < *restore,
            "the long prefill was restored before the short request finished\n" + rig.dump());
    require(*restore < *long_end, "the long prefill never finished after its restore\n" + rig.dump());
    require(events[*pause_start].snapshot && events[*pause_done].snapshot,
            "the pause did not save the prefill's progress\n" + rig.dump());

    // Progress survived: the prompt was computed once, in order, with no step repeated.
    std::uint32_t expected = 0;
    for (const Event& step : rig.events_of(1, Event::Kind::Prefill)) {
        require(step.position == expected, "a prefill step repeated or skipped tokens\n" + rig.dump());
        expected += step.tokens;
    }
    require(expected == 8192, "the long prompt was not computed in full\n" + rig.dump());
    require(rig.core().runtime_stats().prefill_yields == 1, "the yield was not counted");
}

void test_a_long_newcomer_does_not_displace_a_long_prefill() {
    Rig rig({.chunk = 1024, .lanes = 1, .yield_ms = 100});
    Result long_result;
    start_long_prompt(rig, long_result);
    auto other_long = rig.run(2, 8192, 1);
    rig.release();
    require(finished(long_result) && finished(other_long), "a request did not finish\n" + rig.dump());
    require(rig.count(Event::Kind::PauseStart) == 0,
            "a request as long as the resident prefill paused it\n" + rig.dump());
}

void test_a_prefill_with_little_left_is_not_paused() {
    // After the second step of 4 the prompt has ~2 steps (~225 ms) left; require more than that.
    Rig rig({.chunk = 1024, .lanes = 1, .yield_ms = 5000});
    Result long_result;
    start_long_prompt(rig, long_result);
    auto short_result = rig.run(2, 100, 1);
    rig.release();
    require(finished(long_result) && finished(short_result), "a request did not finish\n" + rig.dump());
    require(rig.count(Event::Kind::PauseStart) == 0,
            "a prefill with less than the threshold left was paused\n" + rig.dump());
}

// --admission-fit-check on the real worker loop: a newcomer the Program says cannot fit beside the
// running request is queued (counted once, however many cycles it waits) and starts as soon as the
// resident leaves; one that fits starts at once, and without the flag the answer is never asked.
void test_a_prompt_that_does_not_fit_beside_a_resident_waits_for_it() {
    Rig rig({.chunk = 1024, .lanes = 2, .admission_fit_check = true});
    rig.program().set_fits(false);
    Result long_result;
    start_long_prompt(rig, long_result);
    auto short_result = rig.run(2, 100, 1);
    rig.release();
    require(finished(long_result) && finished(short_result), "a request did not finish\n" + rig.dump());
    const auto events = rig.events();
    std::size_t long_finish = 0, short_bind = 0;
    for (std::size_t i = 0; i < events.size(); ++i) {
        if (events[i].kind == Event::Kind::Finish && events[i].request == 1) { long_finish = i; }
        if (events[i].kind == Event::Kind::Bind && events[i].request == 2) { short_bind = i; }
    }
    require(short_bind > long_finish,
            "a prompt that did not fit started beside the resident\n" + rig.dump());
    require(rig.core().runtime_stats().admission_fit_deferrals == 1,
            "the deferral was not counted exactly once: " +
                std::to_string(rig.core().runtime_stats().admission_fit_deferrals));
}

void test_a_prompt_that_fits_beside_a_resident_starts_at_once() {
    Rig rig({.chunk = 1024, .lanes = 2, .admission_fit_check = true});
    Result long_result;
    start_long_prompt(rig, long_result);
    auto short_result = rig.run(2, 100, 1);
    rig.release();
    require(finished(long_result) && finished(short_result), "a request did not finish\n" + rig.dump());
    const auto events = rig.events();
    std::size_t long_finish = 0, short_bind = 0;
    for (std::size_t i = 0; i < events.size(); ++i) {
        if (events[i].kind == Event::Kind::Finish && events[i].request == 1) { long_finish = i; }
        if (events[i].kind == Event::Kind::Bind && events[i].request == 2) { short_bind = i; }
    }
    require(short_bind < long_finish, "a prompt that fits waited for the resident\n" + rig.dump());
    require(rig.core().runtime_stats().admission_fit_deferrals == 0, "a fitting prompt was deferred");
}

void test_the_fit_answer_is_not_asked_without_the_flag() {
    Rig rig({.chunk = 1024, .lanes = 2});
    rig.program().set_fits(false);
    Result long_result;
    start_long_prompt(rig, long_result);
    auto short_result = rig.run(2, 100, 1);
    rig.release();
    require(finished(long_result) && finished(short_result), "a request did not finish\n" + rig.dump());
    require(rig.core().runtime_stats().admission_fit_deferrals == 0,
            "the fit check ran with the flag off");
}

void test_backlog_gauge_tracks_queued_and_resident_prefill() {
    Rig rig({.chunk = 1024, .lanes = 1, .prefill_sleep = std::chrono::microseconds(10'000)});
    // The gauge as published before each prefill step: stats are published at unit boundaries.
    std::vector<double> samples;
    std::mutex samples_mutex;
    rig.program().before_unit = [&](const Event& event) {
        if (event.kind != Event::Kind::Prefill) { return; }
        const double backlog = rig.core().runtime_stats().prefill_backlog_seconds;
        std::lock_guard lock(samples_mutex);
        samples.push_back(backlog);
    };
    auto resident = rig.run(1, 8192, 1);
    auto queued   = rig.run(2, 4096, 1);
    require(finished(resident) && finished(queued), "a request did not finish\n" + rig.dump());
    // With the second request queued, the 8 steps of the resident prompt (8 x 112 ms) and its own
    // 4,096 tokens (410 ms) are ahead of a newcomer: about 1.3 s at the most. It shrinks as the
    // resident prompt advances.
    std::lock_guard lock(samples_mutex);
    require(samples.size() >= 8, "too few prefill steps sampled");
    const double peak = *std::max_element(samples.begin(), samples.end());
    std::string listing;
    for (const double sample : samples) { listing += " " + std::to_string(sample); }
    require(peak > 0.8 && peak < 2.0, "backlog peaked at " + std::to_string(peak) + " s:" + listing);
    require(samples[2] > samples[6], "the backlog did not shrink as the resident prompt advanced");
    require(rig.wait_for([&] { return rig.core().runtime_stats().prefill_backlog_seconds == 0.0; }),
            "the backlog did not return to zero when idle");
}

// ---- layer slices ----------------------------------------------------------------------------

// Prefill events of request `id`, in order.
std::vector<Event> prefill_events(const Rig& rig, std::uint64_t id) {
    return rig.events_of(id, Event::Kind::Prefill);
}

void test_layer_slices_run_a_neighbour_between_slices() {
    Rig rig({.chunk = 1024, .lanes = 2, .layer_slices = 4});
    auto stream = rig.run(1, 64, kLongOutput);
    require(decoding(rig, 3), "the neighbour never started decoding\n" + rig.dump());
    auto prompt = rig.run(2, 4096, 1);
    require(finished(prompt), "the long prompt did not finish\n" + rig.dump());
    (void)prompt.get();

    const auto steps = prefill_events(rig, 2);
    require(steps.size() == 16, "four chunks of four slices were expected\n" + rig.dump());
    for (std::size_t i = 0; i < steps.size(); ++i) {
        const bool last = i % 4 == 3;
        require(steps[i].slice == i % 4 && steps[i].partial == !last,
                "slice order or partial flag is wrong\n" + rig.dump());
        require(steps[i].tokens == (last ? 1024U : 0U) && steps[i].position == (i / 4) * 1024U,
                "a slice processed tokens, or a chunk started at the wrong place\n" + rig.dump());
    }
    // Between consecutive slices the neighbour got a decode round.
    const auto gaps = decode_rounds_between_steps(rig, 2);
    require(gaps.size() == 15, "unexpected number of gaps");
    for (const std::size_t rounds : gaps) {
        require(rounds >= 1, "two slices ran back to back with the neighbour waiting\n" + rig.dump());
    }
    const auto stats = rig.core().runtime_stats();
    require(stats.sliced_prefill_steps == 4 && stats.prefill_slice_boundaries == 12,
            "sliced steps or boundaries were miscounted");
    (void)stream.wait_for(std::chrono::seconds(30));
}

// A chunk that the stall cap narrowed and that then runs in slices still counts as one paced step with
// its deferred tokens: the cap is applied on the first call and reported by the last.
void test_a_capped_sliced_chunk_counts_as_paced_once() {
    // 10 ms + 0.1 ms/token: an 80 ms budget fits 700 tokens, so each 1024-token chunk runs as 640.
    Rig rig({.chunk = 1024, .lanes = 2, .stall_budget = 80, .layer_slices = 4});
    auto stream = rig.run(1, 64, kLongOutput);
    require(decoding(rig, 3), "the neighbour never started decoding\n" + rig.dump());
    auto prompt = rig.run(2, 4096, 1);
    require(finished(prompt), "the long prompt did not finish\n" + rig.dump());
    const auto stats = rig.core().runtime_stats();
    // Chunks of a 4,096-token prompt at 640 a step: planned 1024 (x5, 384 deferred each), then 896
    // (256 deferred), then a 256-token tail the cap does not narrow. All seven run in slices.
    require(stats.sliced_prefill_steps == 7, "the chunks were not all sliced: " +
                                                 std::to_string(stats.sliced_prefill_steps));
    require(stats.paced_prefill_steps == 6,
            "a capped sliced chunk was not counted as paced exactly once: " +
                std::to_string(stats.paced_prefill_steps));
    require(stats.paced_prefill_deferred_tokens == 5U * 384U + 256U,
            "the deferred tokens of the capped sliced chunks are wrong (an underflow, or lost): " +
                std::to_string(stats.paced_prefill_deferred_tokens));
    drain(stream, std::chrono::seconds(30));
}

void test_layer_slices_need_a_neighbour() {
    Rig rig({.chunk = 1024, .lanes = 2, .layer_slices = 4});
    auto prompt = rig.run(1, 4096, 1);
    require(finished(prompt), "the prompt did not finish\n" + rig.dump());
    (void)prompt.get();
    for (const Event& step : prefill_events(rig, 1)) {
        require(!step.partial && step.tokens == 1024, "a lone prefill was sliced\n" + rig.dump());
    }
    require(rig.core().runtime_stats().sliced_prefill_steps == 0, "a step was counted as sliced");
}

void test_a_sliced_chunk_is_not_interleaved_with_another_prefill() {
    // Two prompts prefill next to a stream: a chunk that has begun finishes before the other
    // lane's chunk starts, because the Program has one parking buffer for the residual.
    Rig rig({.chunk = 1024, .lanes = 3, .layer_slices = 4});
    auto stream = rig.run(1, 64, kLongOutput);
    require(decoding(rig, 3), "the neighbour never started decoding\n" + rig.dump());
    auto first  = rig.run(2, 4096, 1);
    auto second = rig.run(3, 4096, 1);
    require(finished(first) && finished(second), "a prompt did not finish\n" + rig.dump());
    std::optional<std::uint64_t> open;
    for (const Event& event : rig.events()) {
        if (event.kind != Event::Kind::Prefill || event.request == 1) { continue; }
        if (open) {
            require(event.request == *open, "another lane's chunk started inside a sliced chunk\n" +
                                                rig.dump());
        }
        open = event.partial ? std::optional<std::uint64_t>(event.request) : std::nullopt;
    }
    require(!open, "a sliced chunk never finished");
    (void)stream.wait_for(std::chrono::seconds(30));
}

void test_a_pause_waits_for_a_sliced_chunk() {
    // A short request arrives while the long prefill is part-way through a chunk. It cannot pause
    // the prefill there; it does at the next chunk boundary.
    Rig rig({.chunk = 1024, .lanes = 2, .yield_ms = 100, .layer_slices = 4});
    auto stream = rig.run(1, 64, kLongOutput);
    require(decoding(rig, 3), "the neighbour never started decoding\n" + rig.dump());
    rig.hold_at([](const Event& e) {
        return e.kind == Event::Kind::Prefill && e.request == 2 && e.position == 2048 && e.slice == 1;
    });
    auto long_result = rig.run(2, 16384, 1);
    require(rig.wait_held(), "the long prompt never reached its third chunk\n" + rig.dump());
    auto short_result = rig.run(3, 100, 1);
    rig.release();
    require(finished(short_result), "the short request did not finish\n" + rig.dump());
    require(finished(long_result), "the long prompt did not finish\n" + rig.dump());

    require(rig.count(Event::Kind::PauseStart) == 1, "the long prefill was not paused once\n" + rig.dump());
    bool mid_chunk = false;
    for (const Event& event : rig.events()) {
        if (event.kind == Event::Kind::Prefill && event.request == 2) { mid_chunk = event.partial; }
        require(!(event.kind == Event::Kind::PauseStart && mid_chunk),
                "a prefill was paused part-way through a chunk\n" + rig.dump());
    }
    (void)stream.wait_for(std::chrono::seconds(30));
}

void test_cancelling_a_sliced_chunk_leaves_the_engine_working() {
    Rig rig({.chunk = 1024, .lanes = 2, .layer_slices = 4});
    auto stream = rig.run(1, 64, kLongOutput);
    require(decoding(rig, 3), "the neighbour never started decoding\n" + rig.dump());
    rig.hold_at([](const Event& e) {
        return e.kind == Event::Kind::Prefill && e.request == 2 && e.position == 1024 && e.slice == 1;
    });
    auto cancel      = std::make_shared<std::atomic<bool>>(false);
    auto cancelled   = rig.run(2, 8192, 1, cancel);
    require(rig.wait_held(), "the long prompt never reached its second chunk\n" + rig.dump());
    cancel->store(true, std::memory_order_release);
    rig.release();
    require(finished(cancelled), "the cancelled request did not finish\n" + rig.dump());
    require(rig.count(Event::Kind::Abort) >= 1, "the cancelled lane was not released\n" + rig.dump());
    // The Engine still works: a new prompt prefills (in slices again) and finishes.
    auto next = rig.run(3, 2048, 1);
    require(finished(next), "a request after the cancellation did not finish\n" + rig.dump());
    (void)stream.wait_for(std::chrono::seconds(30));
}

// ---- context store with a bucket -------------------------------------------------------------
//
// The Engine's request-time path against a real ContextStore and an in-memory bucket: a request
// that finds nothing has the bucket listed once, then is offered what the listing found. The fake
// imports nothing, so a fetched session is not restored and the request prefills; these tests
// check what the Engine asks of the store, not the restore.

// An in-memory bucket. While `hold` is set every call waits, as a stalled link would.
class MemoryBucket final : public ObjectStore {
public:
    void put(const std::string& key, std::span<const std::uint8_t> bytes) override {
        gate();
        std::scoped_lock lock(mutex_);
        objects_[key] = {std::vector<std::uint8_t>(bytes.begin(), bytes.end()), clock_};
    }
    std::optional<std::vector<std::uint8_t>> get(const std::string& key) override {
        gate();
        std::scoped_lock lock(mutex_);
        const auto found = objects_.find(key);
        if (found == objects_.end()) { return std::nullopt; }
        return found->second.first;
    }
    bool exists(const std::string& key) override {
        gate();
        std::scoped_lock lock(mutex_);
        return objects_.contains(key);
    }
    std::vector<ObjectInfo> list(const std::string& prefix) override {
        gate();
        std::scoped_lock lock(mutex_);
        ++lists;
        std::vector<ObjectInfo> out;
        for (const auto& [key, value] : objects_) {
            if (key.starts_with(prefix)) { out.push_back({key, value.first.size(), value.second}); }
        }
        return out;
    }
    bool touch(const std::string& key) override {
        std::scoped_lock lock(mutex_);
        return objects_.contains(key);
    }
    void remove(const std::string& key) override {
        std::scoped_lock lock(mutex_);
        objects_.erase(key);
    }
    void interrupt() noexcept override { set_hold(false); }

    void set_hold(bool value) {
        {
            std::scoped_lock lock(mutex_);
            hold_ = value;
        }
        cv_.notify_all();
    }
    [[nodiscard]] std::int64_t clock() const noexcept { return clock_; }

    std::atomic<int> lists{0};

private:
    void gate() {
        std::unique_lock lock(mutex_);
        cv_.wait(lock, [&] { return !hold_; });
    }
    std::mutex mutex_;
    std::condition_variable cv_;
    std::map<std::string, std::pair<std::vector<std::uint8_t>, std::int64_t>> objects_;
    std::int64_t clock_ = 1'000'000'000'000;
    bool hold_          = false;
};

class StoreDirectory {
public:
    StoreDirectory() {
        static std::atomic<int> counter{0};
        path_ = std::filesystem::temp_directory_path() /
                ("ninfer_engine_store_test_" + std::to_string(counter.fetch_add(1)) + "_" +
                 std::to_string(Clock::now().time_since_epoch().count() & 0xffffff));
        std::filesystem::remove_all(path_);
        std::filesystem::create_directories(path_);
    }
    ~StoreDirectory() {
        std::error_code ignored;
        std::filesystem::remove_all(path_, ignored);
    }
    [[nodiscard]] const std::filesystem::path& path() const noexcept { return path_; }

private:
    std::filesystem::path path_;
};

ContextStore::Options bucket_store_options(const std::filesystem::path& directory,
                                           const std::shared_ptr<MemoryBucket>& bucket) {
    ContextStore::Options options;
    options.directory      = directory;
    options.chunk_bytes    = 4096;
    options.remote         = bucket;
    options.remote_prefix  = "ninfer/";
    options.remote_refresh = std::chrono::seconds(3600); // only the Engine's lookups list it
    options.clock          = [bucket] { return bucket->clock(); };
    return options;
}

bool store_drained(ContextStore& store) {
    return store.flush_remote(Clock::now() + std::chrono::seconds(10));
}

// Another engine stores a session of prompt `id` (the fake's shortlist key is {id, 0} at every
// frontier) up to `frontier`, with `bytes` of state, and uploads it.
void store_session_elsewhere(const std::shared_ptr<MemoryBucket>& bucket, std::uint64_t id,
                             std::uint32_t frontier, std::size_t bytes) {
    StoreDirectory directory;
    ContextStore other(bucket_store_options(directory.path(), bucket));
    ContextStore::Description description;
    description.id      = "5e55" + std::to_string(id);
    description.binding = "fake-binding";
    description.tokens  = frontier;
    description.checkpoints.push_back({.frontier = frontier, .digests = {id, 0}, .identity_tag = 0});
    std::vector<std::uint8_t> image(64 + bytes);
    std::uint32_t state = static_cast<std::uint32_t>(id) * 2654435761U + 1U;
    for (auto& byte : image) {
        state = state * 1664525U + 1013904223U;
        byte  = static_cast<std::uint8_t>(state >> 24U);
    }
    (void)other.put(description, image, std::array{ContextStore::Region{64, bytes}});
    require(store_drained(other), "the other engine's upload did not finish");
}

// 10 ms + 0.1 ms per token: a 40,000-token prompt faces 4 s of prefill, past the 2 s a request
// needs ahead of it before it waits for a listing.
constexpr std::uint32_t kLongPrompt = 40'000;

void test_a_request_that_finds_nothing_lists_the_bucket_once_and_fetches() {
    auto bucket = std::make_shared<MemoryBucket>();
    StoreDirectory directory;
    ContextStore store(bucket_store_options(directory.path(), bucket));
    require(store_drained(store), "the start-up refresh did not finish");
    // Uploaded after this engine's last listing: only a lookup can find it.
    store_session_elsewhere(bucket, 7, 30'000, 64 * 1024);
    const int lists = bucket->lists;
    {
        Rig rig({.chunk = 1024, .lanes = 1});
        rig.attach_store(store, std::chrono::seconds(2));
        auto result = rig.run(7, kLongPrompt, 1);
        require(finished(result), "the request did not finish\n" + rig.dump());
        const auto stats = rig.core().runtime_stats();
        require(stats.context_store_lookups == 1,
                "the request did not look the bucket up exactly once: " +
                    std::to_string(stats.context_store_lookups));
        require(bucket->lists == lists + 1, "the lookup did not list the bucket exactly once");
        require(store.stats().remote_download_bytes == 64 * 1024,
                "the session the lookup found was not fetched: " +
                    std::to_string(store.stats().remote_download_bytes) + " bytes");
        require(rig.wait_for([&] {
                    return rig.core().runtime_stats().context_store_remote_fetch_bytes_per_second > 0;
                }),
                "the fetch did not teach the fetch rate");
        // The fake imports nothing: the fetched session is a failed hydration, and it prefilled.
        require(rig.core().runtime_stats().context_store_hydration_failures == 1,
                "the fetched session was not offered for import");
        // A second request for the same prompt finds the session without a listing.
        auto again = rig.run(7, kLongPrompt, 1);
        require(finished(again), "the second request did not finish\n" + rig.dump());
        require(rig.core().runtime_stats().context_store_lookups == 1 && bucket->lists == lists + 1,
                "a request that found a session listed the bucket again");
    }
}

void test_a_short_prefill_does_not_wait_for_a_listing() {
    auto bucket = std::make_shared<MemoryBucket>();
    StoreDirectory directory;
    ContextStore store(bucket_store_options(directory.path(), bucket));
    require(store_drained(store), "the start-up refresh did not finish");
    const int lists = bucket->lists;
    {
        Rig rig({.chunk = 1024, .lanes = 1});
        rig.attach_store(store, std::chrono::seconds(2));
        auto result = rig.run(8, 4000, 1); // 0.41 s of prefill
        require(finished(result), "the request did not finish\n" + rig.dump());
        require(rig.core().runtime_stats().context_store_lookups == 0 && bucket->lists == lists,
                "a short prefill waited for a bucket listing");
    }
}

void test_a_fetch_that_cannot_fit_the_wait_is_not_started() {
    auto bucket = std::make_shared<MemoryBucket>();
    StoreDirectory directory;
    ContextStore store(bucket_store_options(directory.path(), bucket));
    require(store_drained(store), "the start-up refresh did not finish");
    // 4 MiB at the assumed 100 MB/s is 42 ms, past a 20 ms wait.
    store_session_elsewhere(bucket, 9, 30'000, 4 << 20);
    {
        Rig rig({.chunk = 1024, .lanes = 1});
        rig.attach_store(store, std::chrono::milliseconds(20));
        auto result = rig.run(9, kLongPrompt, 1);
        require(finished(result), "the request did not finish\n" + rig.dump());
        require(rig.core().runtime_stats().context_store_lookups == 1, "the request did not look");
        const auto info = store.find("5e559");
        require(info && !info->local, "the lookup did not register the session");
        require(store.stats().remote_downloads == 0,
                "a fetch that could not fit the request's wait was started");
    }
}

// The wait a fetch must fit is the request's own: a request with less left before its deadline than
// the hydration budget does not start a fetch it would be released from before it ends.
void test_a_fetch_must_fit_the_requests_remaining_deadline() {
    auto bucket = std::make_shared<MemoryBucket>();
    StoreDirectory directory;
    ContextStore store(bucket_store_options(directory.path(), bucket));
    require(store_drained(store), "the start-up refresh did not finish");
    // 16 MiB at the assumed 100 MB/s is 168 ms: inside the 10 s budget, past the 100 ms deadline.
    store_session_elsewhere(bucket, 11, 30'000, 16 << 20);
    (void)store.refresh_remote(); // already registered: no lookup
    {
        Rig rig({.chunk = 1024, .lanes = 1, .pending_timeout_ms = 100});
        rig.attach_store(store, std::chrono::seconds(10));
        auto result = rig.run(11, kLongPrompt, 1);
        require(finished(result), "the request did not finish\n" + rig.dump());
        require(store.stats().remote_downloads == 0,
                "a fetch longer than the request's remaining deadline was started");
    }
}

void test_a_stalled_listing_releases_the_request_at_its_budget() {
    auto bucket = std::make_shared<MemoryBucket>();
    StoreDirectory directory;
    ContextStore store(bucket_store_options(directory.path(), bucket));
    require(store_drained(store), "the start-up refresh did not finish");
    {
        Rig rig({.chunk = 1024, .lanes = 1});
        rig.attach_store(store, std::chrono::seconds(2));
        bucket->set_hold(true);
        const auto started = Clock::now();
        auto result        = rig.run(10, kLongPrompt, 1);
        const bool done    = finished(result, std::chrono::seconds(15));
        const auto waited  = Clock::now() - started;
        // Let the lookup thread's listing end before the Engine joins it.
        bucket->set_hold(false);
        require(done, "a stalled listing held the request past its budget\n" + rig.dump());
        require(waited >= std::chrono::milliseconds(4500) && waited < std::chrono::seconds(10),
                "the request was not released at the 5 s lookup budget: " +
                    std::to_string(std::chrono::duration_cast<std::chrono::milliseconds>(waited).count()) +
                    " ms");
        require(rig.core().runtime_stats().context_store_lookups == 1, "the lookup was not counted");
    }
}


// ---- lossy prefill (docs/maintainer/lossy-long-prefill.md) ----------------------------------

static_assert(EngineCore<fake::Instance>::kLossyCapable,
              "the fake must model the lossy Program contract the Engine uses");

constexpr std::uint32_t kLossyChunk  = 4096;
constexpr std::uint32_t kMarker      = 2048;  // the product prompt: a shared marker
constexpr std::uint32_t kLossyPrompt = 40000; // U = 37,952 after the marker
constexpr std::uint32_t kLossyKept   = 7591;  // ceil(0.2 * 37,952)

fake::PreparedPrompt lossy_prompt(std::uint64_t id, std::uint32_t tokens = kLossyPrompt,
                                  std::uint64_t shared_group = 0) {
    fake::PreparedPrompt prompt{tokens, id};
    prompt.lossy_exact_end = kMarker;
    if (shared_group != 0) {
        prompt.shared_group  = shared_group;
        prompt.shared_tokens = kMarker;
    }
    return prompt;
}

std::uint32_t total_tokens(const std::vector<Event>& events) {
    std::uint32_t total = 0;
    for (const Event& event : events) { total += event.tokens; }
    return total;
}

std::optional<std::size_t> last_index(const std::vector<Event>& events, Event::Kind kind,
                                      std::uint64_t id) {
    std::optional<std::size_t> found;
    for (std::size_t i = 0; i < events.size(); ++i) {
        if (events[i].kind == kind && events[i].request == id) { found = i; }
    }
    return found;
}

std::optional<std::size_t> first_index(const std::vector<Event>& events, Event::Kind kind,
                                       std::uint64_t id) {
    for (std::size_t i = 0; i < events.size(); ++i) {
        if (events[i].kind == kind && events[i].request == id) { return i; }
    }
    return std::nullopt;
}

// The trigger itself: never for a private conversation record, the threshold on what follows M.
void test_lossy_trigger_rule() {
    const LossyTriggerInput cold{.min_tokens    = 30000,
                                 .exact_end     = kMarker,
                                 .prompt_tokens = kMarker + 31000};
    require(lossy_prefill_trigger(cold) == kMarker, "a cold prompt past the threshold did not trigger");
    auto shared          = cold;
    shared.source_role   = CheckpointRole::SharedPrefix;
    shared.reused_tokens = kMarker;
    require(lossy_prefill_trigger(shared) == kMarker, "a shared-prefix source did not trigger");
    for (const auto role : {CheckpointRole::Continuation, CheckpointRole::InputReplay,
                            CheckpointRole::LongAnchor}) {
        auto warm          = shared;
        warm.source_role   = role;
        warm.reused_tokens = kMarker + 1000;
        require(!lossy_prefill_trigger(warm), "a private conversation record triggered the path");
    }
    auto short_rest          = cold;
    short_rest.prompt_tokens = kMarker + 29000;
    require(!lossy_prefill_trigger(short_rest), "29K after the marker triggered a 30K threshold");
    auto exact          = cold;
    exact.prompt_tokens = kMarker + 30000;
    require(!lossy_prefill_trigger(exact), "the threshold is exclusive");
    auto deeper          = shared;
    deeper.reused_tokens = 5000; // M is the deeper of the marker and the reuse frontier
    deeper.prompt_tokens = 5000 + 30001;
    require(lossy_prefill_trigger(deeper) == 5000u, "M did not follow the reuse frontier");
    auto off       = cold;
    off.min_tokens = 0;
    require(!lossy_prefill_trigger(off), "min_tokens 0 is not off");
    auto media      = cold;
    media.has_media = true;
    require(!lossy_prefill_trigger(media), "a prompt with media triggered the path");
    auto unplanned      = cold;
    unplanned.exact_end = std::nullopt;
    require(!lossy_prefill_trigger(unplanned), "a plan without lossy() triggered the path");
}

// LossyTriggersOnlyWhenCold on the worker loop: a shared product-prompt source with 31K after
// the marker takes the path, 29K does not, and a plan without lossy() never does.
void test_lossy_triggers_only_when_cold() {
    test_lossy_trigger_rule();
    Rig rig({.chunk = kLossyChunk, .lanes = 1, .lossy_min_tokens = 30000, .context_cache = true,
             .shared_prefixes = {{7, kMarker}}});
    auto taken = rig.run_prompt(lossy_prompt(1, kMarker + 31000, 7), 1);
    require(finished(taken), "the 31K request did not finish\n" + rig.dump());
    require(taken.get().reused_prompt_tokens == kMarker,
            "the request did not bind the shared prefix\n" + rig.dump());
    const auto enabled = rig.events_of(1, Event::Kind::EnableLossy);
    require(enabled.size() == 1 && enabled[0].position == kMarker,
            "a shared source with 31K after the marker did not take the path at M\n" + rig.dump());

    auto below = rig.run_prompt(lossy_prompt(2, kMarker + 29000, 7), 1);
    require(finished(below), "the 29K request did not finish\n" + rig.dump());
    require(rig.events_of(2, Event::Kind::EnableLossy).empty() &&
                rig.events_of(2, Event::Kind::Scout).empty(),
            "29K after the marker took the path\n" + rig.dump());

    auto unplanned = rig.run_prompt(fake::PreparedPrompt{kMarker + 31000, 3}, 1);
    require(finished(unplanned), "the unplanned request did not finish\n" + rig.dump());
    require(rig.events_of(3, Event::Kind::EnableLossy).empty(),
            "a plan without lossy() took the path\n" + rig.dump());
    const auto stats = rig.core().runtime_stats();
    require(stats.lossy_prefills_completed == 1 && stats.lossy_prefills_skipped == 0,
            "only the 31K request should have been counted");

    Rig off({.chunk = kLossyChunk, .lanes = 1});
    auto plain = off.run_prompt(lossy_prompt(4), 1);
    require(finished(plain), "the request with the option off did not finish\n" + off.dump());
    require(off.events_of(4, Event::Kind::EnableLossy).empty(),
            "the path ran with --lossy-prefill-min-tokens 0\n" + off.dump());
}

// LossyPromptLengthChangeAccepted: the step that ends the scout shrinks the prompt; the Engine
// takes the new length and plan, and the prefill finishes on it without a frontier check firing.
void test_lossy_prompt_length_change_accepted() {
    Rig rig({.chunk = kLossyChunk, .lanes = 1, .lossy_min_tokens = 30000});
    auto result = rig.run_prompt(lossy_prompt(1), 2);
    require(finished(result), "the lossy request did not finish\n" + rig.dump());
    const auto& generation = result.get();
    const auto prefill     = rig.events_of(1, Event::Kind::Prefill);
    const auto scout       = rig.events_of(1, Event::Kind::Scout);
    require(total_tokens(scout) == kLossyPrompt - kMarker && scout.front().position == kMarker,
            "the scout did not cover U from M\n" + rig.dump());
    require(total_tokens(prefill) == kMarker + kLossyKept,
            "the prefill did not cover the exact part and the kept tokens\n" + rig.dump());
    require(prefill.front().tokens == kMarker && prefill.back().position + prefill.back().tokens ==
                                                     kMarker + kLossyKept,
            "the exact part did not stop at M or the main pass overran\n" + rig.dump());
    require(generation.computed_prefill_tokens == kMarker + kLossyKept,
            "computed prefill tokens are not the kept count");
    require(generation.prompt.prompt_tokens == kLossyPrompt, "the prompt length reported changed");
}

// LossyCancelDuringScout: the request finishes cancelled, the scout ends with its lane and no page
// is left behind.
void test_lossy_cancel_during_scout() {
    Rig rig({.chunk = kLossyChunk, .lanes = 1, .lossy_min_tokens = 30000});
    rig.hold_at([](const Event& e) {
        return e.kind == Event::Kind::Scout && e.request == 1 && e.position == kMarker + kLossyChunk;
    });
    auto cancel = std::make_shared<std::atomic<bool>>(false);
    auto result = rig.run_prompt(lossy_prompt(1), 4, cancel);
    require(rig.wait_held(), "the scout never reached its second step\n" + rig.dump());
    cancel->store(true, std::memory_order_release);
    std::this_thread::sleep_for(std::chrono::milliseconds(50)); // the waiter forwards the flag
    rig.release();
    require(finished(result), "the cancelled request did not finish\n" + rig.dump());
    require(result.get().finish_reason == FinishReason::Cancelled, "the request was not cancelled");
    const auto events = rig.events();
    const auto ended  = first_index(events, Event::Kind::ScoutEnd, 1);
    const auto abort  = first_index(events, Event::Kind::Abort, 1);
    require(ended && abort && *ended < *abort, "the fake did not report the scout restored\n" + rig.dump());
    require(rig.events_of(1, Event::Kind::Scout).size() < 10, "the scout ran to its end\n" + rig.dump());
    require(rig.wait_for([&] {
                const auto stats = rig.core().runtime_stats();
                return stats.device_main_kv_occupied_pages == 0 &&
                       stats.device_state_occupied_slots == 0;
            }),
            "pages or a state slot leaked after the cancelled scout");
    const auto stats = rig.core().runtime_stats();
    require(stats.lossy_prefills_aborted == 1 && stats.lossy_prefills_completed == 0,
            "the cancelled scout was not counted aborted");
    const auto& report = result.get().lossy_prefill;
    require(report && report->outcome == LossyPrefillOutcome::Aborted &&
                report->uncached_tokens == kLossyPrompt - kMarker,
            "the cancelled request's lossy report is wrong");
}

// LossyYieldRefusedDuringScout: a short request at the head cannot pause a scouting lane (the
// Engine does not even ask); it waits until the main pass starts and then yields as today.
void test_lossy_yield_refused_during_scout() {
    Rig rig({.chunk = kLossyChunk, .lanes = 1, .yield_ms = 100, .lossy_min_tokens = 30000});
    rig.hold_at([](const Event& e) {
        return e.kind == Event::Kind::Scout && e.request == 1 && e.position == kMarker;
    });
    auto long_result = rig.run_prompt(lossy_prompt(1), 1);
    require(rig.wait_held(), "the scout never started\n" + rig.dump());
    auto short_result = rig.run(2, 100, 1);
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    rig.release();
    require(finished(short_result) && finished(long_result), "a request did not finish\n" + rig.dump());
    const auto events      = rig.events();
    const auto last_scout  = last_index(events, Event::Kind::Scout, 1);
    const auto pause_start = first_index(events, Event::Kind::PauseStart, 1);
    const auto short_bind  = first_index(events, Event::Kind::Bind, 2);
    const auto long_end    = first_index(events, Event::Kind::Finish, 1);
    require(rig.program().scout_pause_refusals() == 0,
            "the Engine asked to pause a scouting lane\n" + rig.dump());
    require(last_scout && pause_start && *pause_start > *last_scout,
            "the head did not yield once the main pass started\n" + rig.dump());
    require(short_bind && long_end && *short_bind < *long_end,
            "the short request waited for the whole lossy prefill\n" + rig.dump());
    require(rig.core().runtime_stats().prefill_yields == 1, "the yield was not counted");
    // Restored on its compacted plan: the main pass resumed where it stopped and ended at M + kept.
    const auto prefill = rig.events_of(1, Event::Kind::Prefill);
    require(total_tokens(prefill) == kMarker + kLossyKept,
            "the restored main pass did not continue on the compacted prompt\n" + rig.dump());
}

// LossyPreemptionDuringScout: memory pressure needs the scouting lane's pages. The pause is never
// refused into a deadlock: the scout ends at M, the request is paused, restored, and prefills U in
// full.
void test_lossy_preemption_during_scout() {
    Rig rig({.chunk = kLossyChunk, .lanes = 2, .lossy_min_tokens = 30000});
    rig.program().set_decode_shortage_while_scouting(true);
    auto stream = rig.run(1, 64, kLongOutput);
    require(decoding(rig, 3), "the neighbour never started decoding\n" + rig.dump());
    auto lossy = rig.run_prompt(lossy_prompt(2), 1);
    require(finished(lossy) && finished(stream), "a request did not finish\n" + rig.dump());
    const auto events = rig.events();
    const auto ended  = first_index(events, Event::Kind::ScoutEnd, 2);
    const auto pause  = first_index(events, Event::Kind::PauseStart, 2);
    const auto back   = first_index(events, Event::Kind::Restore, 2);
    require(ended && pause && back && *ended < *pause && *pause < *back,
            "the scout did not end before the pause, or the request was not restored\n" + rig.dump());
    require(rig.program().scout_pause_refusals() == 0,
            "the Engine asked to pause a scouting lane\n" + rig.dump());
    require(total_tokens(rig.events_of(2, Event::Kind::Prefill)) == kLossyPrompt,
            "after the aborted scout the prompt was not prefilled in full\n" + rig.dump());
    const auto stats = rig.core().runtime_stats();
    require(stats.lossy_prefills_aborted == 1 && stats.lossy_prefills_completed == 0 &&
                stats.preemptions >= 1,
            "the aborted scout was not counted");
}

class ProgressSink final : public OutputSink {
public:
    void start(GenerationStart start) override {
        std::lock_guard lock(mutex);
        starts.push_back(start);
    }
    void progress(PromptProgress progress) override {
        std::lock_guard lock(mutex);
        updates.push_back(progress);
    }
    void timing(GenerationTimingObservation) override {}
    void publish(OutputDelta) override {}

    std::mutex mutex;
    std::vector<GenerationStart> starts;
    std::vector<PromptProgress> updates;
};

// LossyUsageReportsOriginal: usage and prompt progress count the client's prompt, not the kept
// one; the cached tokens are the exact reused prefix.
void test_lossy_usage_reports_original() {
    Rig rig({.chunk = kLossyChunk, .lanes = 1, .lossy_min_tokens = 30000, .context_cache = true,
             .shared_prefixes = {{7, kMarker}}});
    ProgressSink sink;
    auto result = rig.run_prompt(lossy_prompt(1, kLossyPrompt, 7), 2, nullptr, &sink);
    require(finished(result), "the lossy request did not finish\n" + rig.dump());
    const auto& generation = result.get();
    require(generation.prompt.prompt_tokens == kLossyPrompt,
            "usage prompt_tokens is not the original length");
    require(generation.reused_prompt_tokens == kMarker, "cached tokens are not the exact prefix");
    require(generation.computed_prefill_tokens == kLossyKept, "the kept tokens were not computed");
    std::lock_guard lock(sink.mutex);
    require(sink.starts.size() == 1 && sink.starts[0].reused_prompt_tokens == kMarker &&
                sink.starts[0].prompt.prompt_tokens == kLossyPrompt,
            "the stream start does not count the client's prompt");
    require(!sink.updates.empty(), "no prompt progress was streamed");
    std::uint32_t last = 0;
    for (const auto& update : sink.updates) {
        require(update.total_prompt_tokens == kLossyPrompt && update.reused_prompt_tokens == kMarker &&
                    update.processed_prompt_tokens >= last &&
                    update.processed_prompt_tokens <= kLossyPrompt,
                "prompt progress left the client's prompt: total " +
                    std::to_string(update.total_prompt_tokens) + ", processed " +
                    std::to_string(update.processed_prompt_tokens));
        last = update.processed_prompt_tokens;
    }
    require(last == kLossyPrompt, "prompt progress did not reach the original length: " +
                                      std::to_string(last));
}

// LossyCountersAndRequestLog: one completed, one skipped (the selection kept everything) and one
// declined (the scout was taken), with their tokens and the request's report.
void test_lossy_counters_and_request_log() {
    Rig rig({.chunk = kLossyChunk, .lanes = 1, .lossy_min_tokens = 30000});
    auto completed = rig.run_prompt(lossy_prompt(1), 1);
    require(finished(completed), "the lossy request did not finish\n" + rig.dump());
    const auto report = completed.get().lossy_prefill;
    require(report && report->outcome == LossyPrefillOutcome::Completed &&
                report->uncached_tokens == kLossyPrompt - kMarker &&
                report->kept_tokens == kLossyKept && report->filter_layer == 15 &&
                report->scout_ms > 0.0 && report->main_ms > 0.0,
            "the completed request's lossy report is wrong");
    rig.program().set_lossy_keep(1.0);
    auto skipped = rig.run_prompt(lossy_prompt(2), 1);
    require(finished(skipped), "the skipped request did not finish\n" + rig.dump());
    const auto skipped_report = skipped.get().lossy_prefill;
    require(skipped_report && skipped_report->outcome == LossyPrefillOutcome::Skipped &&
                skipped_report->kept_tokens == kLossyPrompt - kMarker,
            "the skipped request's lossy report is wrong");
    require(total_tokens(rig.events_of(2, Event::Kind::Prefill)) == kLossyPrompt,
            "a selection that kept everything did not prefill in full");
    auto stats = rig.core().runtime_stats();
    require(stats.lossy_prefills_completed == 1 && stats.lossy_prefills_skipped == 1 &&
                stats.lossy_prefills_aborted == 0,
            "the outcomes were not counted");
    require(stats.lossy_prefill_scouted_tokens == 2ULL * (kLossyPrompt - kMarker) &&
                stats.lossy_prefill_kept_tokens == kLossyKept + (kLossyPrompt - kMarker),
            "scouted or kept tokens were not counted: " +
                std::to_string(stats.lossy_prefill_scouted_tokens) + " / " +
                std::to_string(stats.lossy_prefill_kept_tokens));
    require(stats.lossy_prefill_alias_hits == 0, "an alias hit was counted without a rewrite");

    // Two lanes, one scout: the second request reaches M while the first scouts and declines.
    Rig two({.chunk = kLossyChunk, .lanes = 2, .lossy_min_tokens = 30000});
    two.hold_at([](const Event& e) {
        return e.kind == Event::Kind::Scout && e.request == 1 && e.position == kMarker + kLossyChunk;
    });
    auto first = two.run_prompt(lossy_prompt(1), 1);
    require(two.wait_held(), "the first scout never reached its second step\n" + two.dump());
    auto second = two.run_prompt(lossy_prompt(2), 1);
    // Let the second request bind (one binding a cycle) while the first scout is held.
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    two.release();
    require(finished(first) && finished(second), "a request did not finish\n" + two.dump());
    const auto declined = second.get().lossy_prefill;
    require(declined && declined->outcome == LossyPrefillOutcome::Declined &&
                declined->kept_tokens == kLossyPrompt - kMarker,
            "the second request did not decline the busy scout\n" + two.dump());
    require(two.events_of(2, Event::Kind::Scout).empty() &&
                total_tokens(two.events_of(2, Event::Kind::Prefill)) == kLossyPrompt,
            "a declined request scouted or did not prefill in full\n" + two.dump());
    const auto two_stats = two.core().runtime_stats();
    require(two_stats.lossy_prefills_completed == 1 && two_stats.lossy_prefills_skipped == 1 &&
                two_stats.lossy_prefill_scouted_tokens == kLossyPrompt - kMarker &&
                two_stats.lossy_prefill_kept_tokens == kLossyKept,
            "a declined request was not counted skipped, or its tokens were counted as scouted");
}

// The backlog prices a lossy prefill as its scout and the main pass over the kept share.
void test_lossy_backlog_predicts_scout_and_main() {
    Rig rig({.chunk = kLossyChunk, .lanes = 1, .lossy_min_tokens = 30000});
    std::optional<double> at_scout;
    std::mutex sample_mutex;
    rig.program().before_unit = [&](const Event& event) {
        if (event.kind != Event::Kind::Scout || event.position != kMarker) { return; }
        std::lock_guard lock(sample_mutex);
        if (!at_scout) { at_scout = rig.core().runtime_stats().prefill_backlog_seconds; }
    };
    auto result = rig.run_prompt(lossy_prompt(1), 1);
    require(finished(result), "the lossy request did not finish\n" + rig.dump());
    // At M: the scout over U at its quarter share, (10 x 10 ms + 37,952 x 0.1 ms) / 4 = 974 ms, and
    // the main pass over 7,591 kept tokens, 2 x 10 ms + 759 ms. A full prefill of U would be 3.9 s.
    std::lock_guard lock(sample_mutex);
    require(at_scout.has_value(), "the backlog was not sampled at the scout");
    require(*at_scout > 1.6 && *at_scout < 1.9,
            "the backlog at M predicted " + std::to_string(*at_scout) + " s");
}

} // namespace

int main() {
    int failures = 0;
    const auto run = [&](const char* name, void (*test)()) {
        try {
            test();
            std::cout << "ok   " << name << std::endl;
        } catch (const std::exception& error) {
            std::cerr << "FAIL " << name << ": " << error.what() << '\n';
            ++failures;
        }
    };
    run("a request runs to completion", test_a_request_runs_to_completion);
    run("stall budget caps steps next to a decoding lane",
        test_stall_budget_caps_steps_next_to_a_decoding_lane);
    run("stall budget leaves a lone prefill alone", test_stall_budget_leaves_a_lone_prefill_alone);
    run("stall budget off never caps", test_stall_budget_off_never_caps);
    run("decode share gives the stream a burst per step",
        test_decode_share_gives_the_stream_a_burst_per_step);
    run("decode share does nothing without a decoding lane",
        test_decode_share_does_nothing_without_a_decoding_lane);
    run("without yield a short request waits",
        test_without_yield_a_short_request_waits_for_the_long_prompt);
    run("a short request pauses a long prefill and it resumes",
        test_a_short_request_pauses_a_long_prefill_and_it_resumes);
    run("a prompt that does not fit beside a resident waits for it",
        test_a_prompt_that_does_not_fit_beside_a_resident_waits_for_it);
    run("a prompt that fits beside a resident starts at once",
        test_a_prompt_that_fits_beside_a_resident_starts_at_once);
    run("the fit answer is not asked without the flag",
        test_the_fit_answer_is_not_asked_without_the_flag);
    run("a long newcomer does not displace a long prefill",
        test_a_long_newcomer_does_not_displace_a_long_prefill);
    run("a prefill with little left is not paused", test_a_prefill_with_little_left_is_not_paused);
    run("backlog gauge tracks queued and resident prefill",
        test_backlog_gauge_tracks_queued_and_resident_prefill);
    run("layer slices run a neighbour between slices",
        test_layer_slices_run_a_neighbour_between_slices);
    run("a capped sliced chunk counts as paced once",
        test_a_capped_sliced_chunk_counts_as_paced_once);
    run("a capped sliced chunk counts as paced once",
        test_a_capped_sliced_chunk_counts_as_paced_once);
    run("layer slices need a neighbour", test_layer_slices_need_a_neighbour);
    run("a sliced chunk is not interleaved with another prefill",
        test_a_sliced_chunk_is_not_interleaved_with_another_prefill);
    run("a pause waits for a sliced chunk", test_a_pause_waits_for_a_sliced_chunk);
    run("cancelling a sliced chunk leaves the engine working",
        test_cancelling_a_sliced_chunk_leaves_the_engine_working);
    run("a request that finds nothing lists the bucket once and fetches",
        test_a_request_that_finds_nothing_lists_the_bucket_once_and_fetches);
    run("a short prefill does not wait for a listing", test_a_short_prefill_does_not_wait_for_a_listing);
    run("a fetch that cannot fit the wait is not started",
        test_a_fetch_that_cannot_fit_the_wait_is_not_started);
    run("a fetch must fit the request's remaining deadline",
        test_a_fetch_must_fit_the_requests_remaining_deadline);
    run("a stalled listing releases the request at its budget",
        test_a_stalled_listing_releases_the_request_at_its_budget);
    run("LossyTriggersOnlyWhenCold", test_lossy_triggers_only_when_cold);
    run("LossyPromptLengthChangeAccepted", test_lossy_prompt_length_change_accepted);
    run("LossyCancelDuringScout", test_lossy_cancel_during_scout);
    run("LossyYieldRefusedDuringScout", test_lossy_yield_refused_during_scout);
    run("LossyPreemptionDuringScout", test_lossy_preemption_during_scout);
    run("LossyUsageReportsOriginal", test_lossy_usage_reports_original);
    run("LossyCountersAndRequestLog", test_lossy_counters_and_request_log);
    run("LossyBacklogPredictsScoutAndMain", test_lossy_backlog_predicts_scout_and_main);
    return failures == 0 ? 0 : 1;
}
