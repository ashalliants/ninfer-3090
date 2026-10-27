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
    PrefillOrder prefill_order = PrefillOrder::Rotate;
    bool admission_fit_check = false;
    std::uint32_t pending_timeout_ms = 60'000;
    // Lossy prefill: LossyPrefillOptions::min_tokens (0: off).
    std::uint32_t lossy_min_tokens = 0;
    LossyStrategy lossy_strategy   = LossyStrategy::Select;
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
        options.lossy_prefill.strategy   = settings.lossy_strategy;
        // The fake Program keeps 0.2 of U (fake::Program::set_lossy_keep); the Engine prices the
        // main pass with the same share, so the backlog checks stay independent of the default.
        options.lossy_prefill.keep_fraction = 0.2F;
        for (const auto& [group, frontier] : settings.shared_prefixes) {
            instance_.external_checkpoints.push_back(
                instance_.program->add_shared_prefix(group, frontier));
        }
        options.prefill_schedule      = {.stall_budget_ms      = settings.stall_budget,
                                         .yield_remaining_ms   = settings.yield_ms,
                                         .decode_share_percent = settings.decode_share,
                                         .layer_slices         = settings.layer_slices,
                                         .order                = settings.prefill_order};
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
    // `record`: what the summary strategy did for the prompt (Engine::prepare reports the client's
    // length in the summary when it rewrote it, as here).
    Result run_prompt(fake::PreparedPrompt prompt, std::uint32_t output_tokens,
                      std::shared_ptr<std::atomic<bool>> cancel = nullptr,
                      OutputSink* sink                         = nullptr,
                      std::optional<compaction::SummaryRecord> record = std::nullopt) {
        ResolvedRequestOptions request;
        request.execution.requested_output_tokens = output_tokens;
        PromptSummary summary;
        summary.prompt_tokens =
            record && record->rewrite ? record->rewrite->original_tokens : prompt.tokens;
        GenerationObservationOptions observation;
        observation.prompt_progress = sink != nullptr;
        auto submission = std::make_shared<EngineCore<fake::Instance>::Submission>(core_->submit(
            prompt, summary, 0.0, std::move(request),
            sink != nullptr ? OutputConsumerMode::Streaming : OutputConsumerMode::Aggregate,
            observation, {}, record));
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

    // Gives the Engine a context store backed by a bucket. The store must outlive the Rig. Each
    // failed hydration is recorded in hydration_events().
    void attach_store(ContextStore& store, std::chrono::milliseconds hydration_budget) {
        core_->set_context_store({.store              = &store,
                                  .binding            = "fake-binding",
                                  .hydration_budget   = hydration_budget,
                                  .remote             = true,
                                  .hydration_listener = [this](const ContextStoreHydrationEvent& event) {
                                      std::lock_guard lock(events_mutex_);
                                      hydration_events_.push_back(event);
                                  }});
    }
    [[nodiscard]] std::vector<ContextStoreHydrationEvent> hydration_events() const {
        std::lock_guard lock(events_mutex_);
        return hydration_events_;
    }

    // Gives the Engine a context store it writes idle conversations to, through `write` (which
    // stores the image and returns whether it took it). The store must outlive the Rig.
    // `failures` and `failed_writes`, when given, stand for the writer's failure count and its
    // reports of the writes that failed.
    void attach_store_writer(ContextStore& store, std::function<bool(fake::CheckpointImage&&)> write,
                             std::chrono::milliseconds idle,
                             std::function<std::uint64_t()> failures             = {},
                             std::function<EngineCore<fake::Instance>::FailedStoreWrites()> failed_writes = {}) {
        core_->set_context_store({.store        = &store,
                                  .binding      = "fake-binding",
                                  .sink         = std::move(write),
                                  .failures     = std::move(failures),
                                  .failed_writes = std::move(failed_writes),
                                  .idle_persist = idle});
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
    mutable std::mutex events_mutex_;
    std::vector<ContextStoreHydrationEvent> hydration_events_;
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

// A Host full of cached history has no room for the long prefill's snapshot. Before #332's
// follow-up the yield checked only the state image, so it could pause, lose the snapshot and make
// the long prompt recompute everything; now it does not pause at all.
void test_no_yield_when_the_snapshot_does_not_fit() {
    Rig rig({.chunk = 1024, .lanes = 1, .yield_ms = 100});
    rig.program().set_snapshot_fits(false);
    Result long_result;
    start_long_prompt(rig, long_result);
    auto short_result = rig.run(2, 100, 1);
    rig.release();
    require(finished(long_result) && finished(short_result), "a request did not finish\n" + rig.dump());
    require(rig.count(Event::Kind::PauseStart) == 0,
            "a prefill yielded although its snapshot had no room\n" + rig.dump());
    require(rig.core().runtime_stats().prefill_yields == 0, "a yield was counted");
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

// Two long prompts in two lanes: rotation interleaves their steps, so both finish late; shortest
// runs the earlier one (the earlier deadline, same cost) to the end before the later one starts.
void test_rotation_interleaves_two_long_prefills() {
    Rig rig({.chunk = 1024, .lanes = 2});
    Result first;
    start_long_prompt(rig, first);
    auto second = rig.run(2, 8192, 1);
    rig.release();
    require(finished(first) && finished(second), "a request did not finish\n" + rig.dump());
    const auto events = rig.events();
    const auto second_first = first_index(events, Event::Kind::Prefill, 2);
    const auto first_last   = last_index(events, Event::Kind::Prefill, 1);
    require(second_first && first_last && *second_first < *first_last,
            "rotation did not interleave the two prefills\n" + rig.dump());
}

void test_shortest_runs_one_long_prefill_at_a_time() {
    Rig rig({.chunk = 1024, .lanes = 2, .prefill_order = PrefillOrder::Shortest});
    Result first;
    start_long_prompt(rig, first);
    auto second = rig.run(2, 8192, 1);
    rig.release();
    require(finished(first) && finished(second), "a request did not finish\n" + rig.dump());
    const auto events = rig.events();
    const auto first_last   = last_index(events, Event::Kind::Prefill, 1);
    const auto second_first = first_index(events, Event::Kind::Prefill, 2);
    require(first_last && second_first && *first_last < *second_first,
            "the later of two equal prompts prefilled before the earlier one finished\n" +
                rig.dump());
}

void test_shortest_lets_a_short_prompt_overtake_a_long_one() {
    Rig rig({.chunk = 1024, .lanes = 2, .prefill_order = PrefillOrder::Shortest});
    Result long_result;
    start_long_prompt(rig, long_result);
    auto short_result = rig.run(2, 100, 1);
    rig.release();
    require(finished(long_result) && finished(short_result), "a request did not finish\n" + rig.dump());
    const auto events = rig.events();
    std::optional<std::size_t> long_third_step;
    for (std::size_t i = 0; i < events.size(); ++i) {
        if (events[i].kind == Event::Kind::Prefill && events[i].request == 1 &&
            events[i].position == 2048) {
            long_third_step = i;
        }
    }
    const auto short_last = last_index(events, Event::Kind::Prefill, 2);
    require(long_third_step && short_last && *short_last < *long_third_step,
            "the short prompt waited for the long prompt's next step\n" + rig.dump());
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
        if (key.find("/chunks/") != std::string::npos) {
            ++chunk_gets;
            std::this_thread::sleep_for(chunk_delay.load());
        } else if (key.find("/manifests/") != std::string::npos) {
            ++manifest_gets;
        }
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
    std::atomic<int> manifest_gets{0};
    std::atomic<int> chunk_gets{0};
    // How long each chunk download takes, as on a slow link.
    std::atomic<std::chrono::milliseconds> chunk_delay{std::chrono::milliseconds(0)};

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

// The id the Engine stores a session of prompt `id` up to `frontier` under (the fake's shortlist key
// is {id, 0} at every frontier).
std::string session_id(std::uint64_t id, std::uint32_t frontier) {
    return context_store_image_id({id, 0}, frontier, false);
}

// Another engine stores a session of prompt `id` up to `frontier`, with `bytes` of state, and
// uploads it.
void store_session_elsewhere(const std::shared_ptr<MemoryBucket>& bucket, std::uint64_t id,
                             std::uint32_t frontier, std::size_t bytes) {
    StoreDirectory directory;
    ContextStore other(bucket_store_options(directory.path(), bucket));
    ContextStore::Description description;
    description.id      = session_id(id, frontier);
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
        require(stats.context_store_lookup_hits == 1, "the lookup that found the session was not a hit");
        // The fake imports nothing: the fetched session is a failed hydration, and it prefilled.
        require(rig.core().runtime_stats().context_store_hydration_failures == 1,
                "the fetched session was not offered for import");
        // Said why, once, with the session it was.
        const auto failed = rig.core().runtime_stats().context_store_hydration_failures_by_reason;
        require(failed[static_cast<std::size_t>(ContextStoreHydrationFailure::HostGeometry)] == 1,
                "the failed import was not counted under its reason");
        const auto events = rig.hydration_events();
        require(events.size() == 1 && events[0].id == session_id(7, 30'000) &&
                    events[0].reason == ContextStoreHydrationFailure::HostGeometry &&
                    events[0].bytes == 64 + 64 * 1024 && events[0].fetch_bytes == 64 * 1024 &&
                    events[0].frontier == 30'000,
                "the failed hydration was not reported with its session");
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
        const auto info = store.find(session_id(9, 30'000));
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
        require(rig.core().runtime_stats().context_store_lookup_hits == 0,
                "a lookup that found nothing was counted as a hit");
    }
}

// A lookup reads only the manifests of sessions ending on the request's own prompt: sessions other
// engines uploaded for other conversations are left to the background refresh, and a request whose
// session is not there is released as soon as the listing shows it.
void test_a_lookup_reads_only_the_requests_own_manifests() {
    auto bucket = std::make_shared<MemoryBucket>();
    StoreDirectory directory;
    ContextStore store(bucket_store_options(directory.path(), bucket));
    require(store_drained(store), "the start-up refresh did not finish");
    store_session_elsewhere(bucket, 12, 30'000, 4096);
    for (std::uint64_t other = 20; other < 26; ++other) {
        store_session_elsewhere(bucket, other, 30'000, 4096);
    }
    // Another conversation's session on a frontier this prompt also reaches.
    store_session_elsewhere(bucket, 26, 1'000, 4096);
    const int manifests = bucket->manifest_gets;
    {
        Rig rig({.chunk = 1024, .lanes = 1});
        rig.attach_store(store, std::chrono::seconds(2));
        auto result = rig.run(12, kLongPrompt, 1);
        require(finished(result), "the request did not finish\n" + rig.dump());
        const auto stats = rig.core().runtime_stats();
        require(stats.context_store_lookups == 1 && stats.context_store_lookup_hits == 1,
                "the lookup did not find the request's session");
        require(bucket->manifest_gets == manifests + 1 && store.find(session_id(12, 30'000)),
                "the lookup read manifests of other conversations: " +
                    std::to_string(bucket->manifest_gets - manifests));
        require(!store.find(session_id(20, 30'000)) && !store.find(session_id(26, 1'000)),
                "the lookup registered sessions of other conversations");

        // Nothing of this prompt is in the bucket: no manifest is read.
        auto missing = rig.run(13, kLongPrompt, 1);
        require(finished(missing), "the second request did not finish\n" + rig.dump());
        const auto after = rig.core().runtime_stats();
        require(after.context_store_lookups == 2 && after.context_store_lookup_hits == 1 &&
                    bucket->manifest_gets == manifests + 1,
                "a lookup that matched nothing read manifests or counted a hit");
    }
    // The background refresh still registers everything else.
    require(store.refresh_remote() == 7, "the refresh did not register what the lookups left");
}

// A download that cannot finish in the request's wait stops at its deadline (a miss, not damage):
// the request prefills, and the reader is free for the next request instead of downloading the
// rest of a session nobody waits for.
void test_a_slow_download_stops_at_the_requests_deadline() {
    auto bucket = std::make_shared<MemoryBucket>();
    StoreDirectory directory;
    ContextStore store(bucket_store_options(directory.path(), bucket));
    require(store_drained(store), "the start-up refresh did not finish");
    // 64 chunks of 4 KiB; each takes 100 ms, eight at a time: 0.8 s, past a 200 ms wait (the
    // assumed rate prices it at a few ms, so the read is started).
    store_session_elsewhere(bucket, 14, 30'000, 64 * 4096);
    (void)store.refresh_remote(); // registered: no lookup
    bucket->chunk_delay = std::chrono::milliseconds(100);
    {
        Rig rig({.chunk = 1024, .lanes = 1});
        rig.attach_store(store, std::chrono::milliseconds(200));
        auto result = rig.run(14, kLongPrompt, 1);
        require(finished(result), "the request did not finish\n" + rig.dump());
        const auto stats = rig.core().runtime_stats();
        require(stats.context_store_hydration_failures == 1 &&
                    stats.context_store_hydration_failures_by_reason[static_cast<std::size_t>(
                        ContextStoreHydrationFailure::Deadline)] == 1,
                "the expired read was not a deadline failure");
        // Long enough for the whole download, had it gone on.
        std::this_thread::sleep_for(std::chrono::milliseconds(1500));
        require(store.stats().remote_downloads < 64,
                "the download went on past the request's deadline: " +
                    std::to_string(store.stats().remote_downloads) + " chunks");
        const auto info = store.find(session_id(14, 30'000));
        require(info && !info->local && store.stats().corrupt_removed == 0,
                "a download stopped at its deadline was treated as damage");
    }
}

// A read that completes after the request's wait ended, while the worker was busy past the deadline,
// is not imported when the worker gets to it: the request was given that long, no more.
void test_a_read_finished_after_the_deadline_is_not_imported() {
    auto bucket = std::make_shared<MemoryBucket>();
    StoreDirectory directory;
    ContextStore store(bucket_store_options(directory.path(), bucket));
    require(store_drained(store), "the start-up refresh did not finish");
    // One chunk, which starts inside the 200 ms wait and arrives at 300 ms.
    store_session_elsewhere(bucket, 16, 30'000, 4096);
    (void)store.refresh_remote();
    bucket->chunk_delay = std::chrono::milliseconds(300);
    {
        Rig rig({.chunk = 1024, .lanes = 2});
        rig.attach_store(store, std::chrono::milliseconds(200));
        // The worker stalls in one unit from when the download starts until well after it ends,
        // as behind a long prefill step.
        std::atomic<bool> stalled{false};
        rig.program().before_unit = [&](const Event&) {
            if (bucket->chunk_gets > 0 && !stalled.exchange(true)) {
                std::this_thread::sleep_for(std::chrono::milliseconds(600));
            }
        };
        auto busy   = rig.run(1, 4000, 2000); // keeps the worker running units
        auto result = rig.run(16, kLongPrompt, 1);
        require(finished(result), "the request did not finish\n" + rig.dump());
        require(stalled, "the worker never stalled during the download");
        const auto failed = rig.core().runtime_stats().context_store_hydration_failures_by_reason;
        require(failed[static_cast<std::size_t>(ContextStoreHydrationFailure::Deadline)] == 1 &&
                    failed[static_cast<std::size_t>(ContextStoreHydrationFailure::HostGeometry)] == 0,
                "a read that finished after its deadline was imported");
        drain(busy, std::chrono::seconds(30));
        rig.program().before_unit = nullptr;
    }
}

// A failure the import did not foresee fails only the hydration: the request prefills, and the
// Engine does not treat it as a worker fault.
void test_an_import_that_throws_fails_only_the_hydration() {
    auto bucket = std::make_shared<MemoryBucket>();
    StoreDirectory directory;
    ContextStore store(bucket_store_options(directory.path(), bucket));
    require(store_drained(store), "the start-up refresh did not finish");
    store_session_elsewhere(bucket, 15, 30'000, 4096);
    (void)store.refresh_remote();
    {
        Rig rig({.chunk = 1024, .lanes = 1});
        rig.attach_store(store, std::chrono::seconds(2));
        rig.program().import_throws = true;
        auto result = rig.run(15, kLongPrompt, 1);
        require(finished(result), "the request did not finish\n" + rig.dump());
        const auto stats = rig.core().runtime_stats();
        require(stats.engine_recoveries == 0, "a failed import was handled as a worker fault");
        require(stats.context_store_hydration_failures == 1 &&
                    stats.context_store_hydration_failures_by_reason[static_cast<std::size_t>(
                        ContextStoreHydrationFailure::Exception)] == 1,
                "the failed import was not counted as an exception");
        const auto events = rig.hydration_events();
        require(events.size() == 1 && events[0].reason == ContextStoreHydrationFailure::Exception,
                "the failed import was not reported");
    }
}


// ---- idle writes ---------------------------------------------------------------------------

// Stores a fake image as the Engine's writer does, under the id the Engine expects.
void put_image(ContextStore& store, const fake::CheckpointImage& image) {
    ContextStore::Description description;
    const auto& key     = image.keys.front();
    description.id      = ContextStore::image_id(key.digests, key.frontier, false);
    description.binding = "fake-binding";
    description.tokens  = image.tokens;
    description.checkpoints.push_back(
        {.frontier = key.frontier, .digests = key.digests, .identity_tag = key.identity_tag});
    std::vector<ContextStore::Region> regions;
    for (const auto& region : image.regions) { regions.push_back({region.offset, region.length}); }
    (void)store.put(description, image.bytes, regions);
}

// Another conversation's second turn, whose image supersedes its first: a removal from the store
// that has nothing to do with the sessions the Engine holds.
void supersede_elsewhere(ContextStore& store) {
    ContextStore::Description first;
    first.id      = "e1";
    first.binding = "fake-binding";
    first.tokens  = 100;
    first.checkpoints.push_back({.frontier = 100, .digests = {77, 7}, .identity_tag = 0});
    const std::vector<std::uint8_t> image(64 + 4096, 1);
    (void)store.put(first, image, std::array{ContextStore::Region{64, 4096}});
    std::vector<std::array<std::uint64_t, 2>> digests(201, {77, 0});
    digests[100] = {77, 7};
    ContextStore::Description second = first;
    second.id                         = "e2";
    second.tokens                     = 200;
    second.checkpoints                = {{.frontier = 200, .digests = {77, 8}, .identity_tag = 0}};
    second.prefix_digests             = digests;
    (void)store.put(second, image, std::array{ContextStore::Region{64, 4096}});
    require(!store.find("e1") && store.stats().superseded == 1, "the second turn did not supersede");
}

// Restores the image of conversation `group` from the store into the Engine's cache, where it is a
// retained continuation the store holds as it is.
void restore_stored(Rig& rig, ContextStore& store, std::uint64_t group) {
    const auto image = rig.program().image_of(group, 4096);
    const auto id    = ContextStore::image_id(image.keys.front().digests, 4096, false);
    const auto bytes = store.load(id);
    require(bytes.has_value(), "the stored session could not be read");
    require(rig.core().restore_image(*bytes), "the stored session was not restored");
}

// The store removing an image of another conversation (here superseded by its next turn, which
// happens on every turn of every conversation) does not make the Engine write the sessions it
// holds again: they are unchanged, and the store still holds them.
void test_an_unrelated_removal_does_not_rewrite_stored_sessions() {
    StoreDirectory directory;
    ContextStore::Options options;
    options.directory   = directory.path();
    options.chunk_bytes = 4096;
    ContextStore store(options);
    Rig rig({.chunk = 1024, .lanes = 1, .context_cache = true});
    std::atomic<int> writes{0};
    rig.attach_store_writer(
        store,
        [&](fake::CheckpointImage&& image) {
            put_image(store, image);
            ++writes;
            return true;
        },
        std::chrono::milliseconds(50));
    for (const std::uint64_t group : {1, 2}) {
        put_image(store, rig.program().image_of(group, 4096));
        restore_stored(rig, store, group);
    }
    auto busy = rig.run(9, 64, 3000); // keeps the worker cycling for about 3 s
    supersede_elsewhere(store);
    std::this_thread::sleep_for(std::chrono::milliseconds(2500)); // two idle scans or more
    require(writes == 0 && rig.program().exports() == 0,
            "an unrelated removal had unchanged stored sessions written again: " +
                std::to_string(writes.load()) + " writes, " +
                std::to_string(rig.program().exports()) + " exports");
    require(finished(busy), "the request did not finish\n" + rig.dump());
}

// A store with room for one of two sessions keeps the newer and evicts the older. The older one is
// not written back: that would evict the newer, which would then be written back, and so on.
void test_a_session_the_store_evicted_is_not_written_back() {
    StoreDirectory directory;
    ContextStore::Options options;
    options.directory   = directory.path();
    options.chunk_bytes = 4096;
    options.max_bytes   = 96 * 1024; // one 64 KiB image and its manifest, not two
    ContextStore store(options);
    Rig rig({.chunk = 1024, .lanes = 1, .context_cache = true});
    std::atomic<int> writes{0};
    rig.attach_store_writer(
        store,
        [&](fake::CheckpointImage&& image) {
            put_image(store, image);
            ++writes;
            return true;
        },
        std::chrono::milliseconds(50));
    put_image(store, rig.program().image_of(1, 4096));
    restore_stored(rig, store, 1);
    put_image(store, rig.program().image_of(2, 4096));
    require(store.stats().evicted_for_space == 1 && store.stats().images == 1,
            "the second session did not evict the first");
    restore_stored(rig, store, 2);
    auto busy = rig.run(9, 64, 3000);
    std::this_thread::sleep_for(std::chrono::milliseconds(2500));
    require(writes == 0,
            "a session the store evicted was written back: " + std::to_string(writes.load()) +
                " writes, " + std::to_string(store.stats().evicted_for_space) + " evictions");
    require(finished(busy), "the request did not finish\n" + rig.dump());
}

// The writer's failure reports, as the Engine's writer thread makes them.
struct FailedWrites {
    std::mutex mutex;
    std::uint64_t count = 0;
    std::vector<std::string> ids;
    void fail(const std::string& id) {
        std::lock_guard lock(mutex);
        ++count;
        ids.push_back(id);
    }
    std::uint64_t failures() {
        std::lock_guard lock(mutex);
        return count;
    }
    bool lost = false;
    void lose() {
        std::lock_guard lock(mutex);
        ++count;
        lost = true;
    }
    EngineCore<fake::Instance>::FailedStoreWrites take() {
        std::lock_guard lock(mutex);
        return {std::exchange(ids, {}), std::exchange(lost, false)};
    }
};

// A queued write that failed is written again, even when a removal scan saw its image missing first
// and took it for one the store let go (the failure report lands after the scan).
void test_a_failed_write_is_retried_after_a_removal_scan_dropped_it() {
    StoreDirectory directory;
    ContextStore::Options options;
    options.directory   = directory.path();
    options.chunk_bytes = 4096;
    ContextStore store(options);
    Rig rig({.chunk = 1024, .lanes = 1, .context_cache = true});
    std::atomic<int> writes{0};
    FailedWrites failed;
    rig.attach_store_writer(
        store,
        [&](fake::CheckpointImage&& image) {
            put_image(store, image);
            ++writes;
            return true;
        },
        std::chrono::milliseconds(50), [&] { return failed.failures(); },
        [&] { return failed.take(); });
    const auto image = rig.program().image_of(1, 4096);
    const auto id    = ContextStore::image_id(image.keys.front().digests, 4096, false);
    put_image(store, image);
    restore_stored(rig, store, 1);
    // The write "failed": its image is not in the store. A removal elsewhere is scanned first.
    require(store.erase(id), "the image could not be erased");
    supersede_elsewhere(store);
    // The shutdown flush writes whatever is not settled, with no idle wait or failure back-off.
    int flushed = 0;
    const auto flush = [&] {
        return rig.core().persist_all(
            [&](fake::CheckpointImage&& written, bool) {
                put_image(store, written);
                ++flushed;
                return true;
            },
            Clock::now() + std::chrono::seconds(5));
    };
    (void)flush();
    require(flushed == 0, "a session taken for one the store let go was written");
    failed.fail(id);
    (void)flush();
    require(flushed == 1, "the failed write was not written again: " + std::to_string(flushed) +
                              " writes");
    require(store.find(id).has_value(), "the retried write did not store the session");
}

// When the writer lost some failure reports (more failed than it keeps), every continuation whose
// image the store does not hold is written again, not trusted as stored.
// A flush writes every unsaved continuation even with idle writes off, and nothing more.
void test_a_flush_writes_unsaved_sessions_with_idle_writes_off() {
    StoreDirectory directory;
    ContextStore::Options options;
    options.directory   = directory.path();
    options.chunk_bytes = 4096;
    ContextStore store(options);
    Rig rig({.chunk = 1024, .lanes = 1, .context_cache = true});
    std::atomic<int> writes{0};
    FailedWrites failed;
    rig.attach_store_writer(
        store,
        [&](fake::CheckpointImage&& image) {
            put_image(store, image);
            ++writes;
            return true;
        },
        std::chrono::milliseconds(0), [&] { return failed.failures(); },
        [&] { return failed.take(); });
    for (const std::uint64_t group : {1, 2}) {
        put_image(store, rig.program().image_of(group, 4096));
        restore_stored(rig, store, group);
    }
    // Session 1's write "failed": it is unsaved; session 2 is stored.
    const auto id =
        ContextStore::image_id(rig.program().image_of(1, 4096).keys.front().digests, 4096, false);
    require(store.erase(id), "the image could not be erased");
    failed.fail(id);
    require(rig.core().unsaved_store_sessions() == 1, "the failed write did not leave one unsaved");
    std::this_thread::sleep_for(std::chrono::milliseconds(1500));
    require(writes == 0, "a session was written with idle writes off and no flush");
    rig.core().set_store_flush(true);
    const bool flushed = rig.wait_for([&] { return rig.core().unsaved_store_sessions() == 0; },
                                      std::chrono::seconds(5));
    rig.core().set_store_flush(false);
    require(flushed && writes == 1 && store.find(id).has_value(),
            "the flush did not write exactly the unsaved session: " + std::to_string(writes.load()) +
                " writes");
}

void test_lost_failure_reports_retry_every_missing_session() {
    StoreDirectory directory;
    ContextStore::Options options;
    options.directory   = directory.path();
    options.chunk_bytes = 4096;
    ContextStore store(options);
    Rig rig({.chunk = 1024, .lanes = 1, .context_cache = true});
    FailedWrites failed;
    rig.attach_store_writer(
        store, [&](fake::CheckpointImage&& image) { put_image(store, image); return true; },
        std::chrono::milliseconds(50), [&] { return failed.failures(); },
        [&] { return failed.take(); });
    for (const std::uint64_t group : {1, 2}) {
        put_image(store, rig.program().image_of(group, 4096));
        restore_stored(rig, store, group);
    }
    // Session 1's write "failed" and its report was lost; session 2 is still stored.
    const auto lost_id =
        ContextStore::image_id(rig.program().image_of(1, 4096).keys.front().digests, 4096, false);
    require(store.erase(lost_id), "the image could not be erased");
    failed.lose();
    std::vector<std::string> flushed;
    (void)rig.core().persist_all(
        [&](fake::CheckpointImage&& written, bool) {
            flushed.push_back(ContextStore::image_id(written.keys.front().digests,
                                                     written.keys.front().frontier, false));
            put_image(store, written);
            return true;
        },
        Clock::now() + std::chrono::seconds(5));
    require(flushed == std::vector<std::string>{lost_id},
            "lost failure reports did not have exactly the missing session written again: " +
                std::to_string(flushed.size()) + " writes");
}

// An export copies into the buffer the writer hands back (an earlier image's, already mapped),
// not a fresh one.
void test_an_export_reuses_the_writers_spare_buffer() {
    StoreDirectory directory;
    ContextStore::Options options;
    options.directory   = directory.path();
    options.chunk_bytes = 4096;
    ContextStore store(options);
    Rig rig({.chunk = 1024, .lanes = 1, .context_cache = true});
    FailedWrites failed;
    std::vector<std::uint8_t> spare(1 << 20);
    const std::uint8_t* const spare_data = spare.data();
    rig.core().set_context_store({.store         = &store,
                                  .binding       = "fake-binding",
                                  .failures      = [&] { return failed.failures(); },
                                  .failed_writes = [&] { return failed.take(); },
                                  .export_buffer = [&] { return std::exchange(spare, {}); }});
    const auto image = rig.program().image_of(1, 4096);
    const auto id    = ContextStore::image_id(image.keys.front().digests, 4096, false);
    put_image(store, image);
    restore_stored(rig, store, 1);
    require(store.erase(id), "the image could not be erased");
    failed.fail(id);
    int flushed = 0;
    (void)rig.core().persist_all(
        [&](fake::CheckpointImage&& written, bool) {
            require(written.bytes.data() == spare_data, "the image was not written into the spare buffer");
            ++flushed;
            return true;
        },
        Clock::now() + std::chrono::seconds(5));
    const auto buffers = rig.program().export_buffers();
    require(flushed == 1 && buffers.size() == 1 && buffers.front() == spare_data,
            "the export was not given the writer's spare buffer");
}

// An image of the same id written under another model or configuration does not count as this
// conversation stored: when this Engine's write of it fails, it is written again.
void test_a_foreign_image_of_the_same_id_does_not_count_as_stored() {
    StoreDirectory directory;
    ContextStore::Options options;
    options.directory   = directory.path();
    options.chunk_bytes = 4096;
    ContextStore store(options);
    Rig rig({.chunk = 1024, .lanes = 1, .context_cache = true});
    std::atomic<int> writes{0};
    FailedWrites failed;
    rig.attach_store_writer(
        store,
        [&](fake::CheckpointImage&& image) {
            put_image(store, image);
            ++writes;
            return true;
        },
        std::chrono::milliseconds(50), [&] { return failed.failures(); },
        [&] { return failed.take(); });
    const auto image = rig.program().image_of(1, 4096);
    const auto id    = ContextStore::image_id(image.keys.front().digests, 4096, false);
    put_image(store, image);
    restore_stored(rig, store, 1);
    // Another configuration's image now holds the id, and this Engine's write of it failed.
    ContextStore::Description foreign;
    foreign.id      = id;
    foreign.binding = "another-binding";
    foreign.tokens  = 4096;
    foreign.checkpoints.push_back({.frontier = 4096, .digests = image.keys.front().digests});
    (void)store.put(foreign, image.bytes, std::array{ContextStore::Region{fake::Program::kImageHeaderBytes, 64 * 1024}});
    failed.fail(id);
    auto busy = rig.run(9, 64, 4000);
    require(rig.wait_for([&] { return writes.load() == 1; }, std::chrono::seconds(3)),
            "a foreign image of the same id counted as the conversation stored: " +
                std::to_string(writes.load()) + " writes");
    require(store.find(id) && store.find(id)->binding == "fake-binding",
            "the retried write did not replace the foreign image");
    require(finished(busy), "the request did not finish\n" + rig.dump());
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
    // A shared checkpoint past the product prompt is the conversation (an earlier turn's
    // automatic or cache_control boundary): cached, so no lossy path however long the new turn.
    auto deeper          = shared;
    deeper.reused_tokens = kMarker + 1;
    deeper.prompt_tokens = kMarker + 1 + 60000;
    require(!lossy_prefill_trigger(deeper), "a shared source past the product prompt triggered");
    auto below_marker          = shared;
    below_marker.reused_tokens = kMarker / 2; // a shallower shared prefix: still cold
    require(lossy_prefill_trigger(below_marker) == kMarker,
            "a shared source inside the product prompt did not trigger at the product end");
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

// LossySharedSourcePastProductPrompt: turn N published a shared checkpoint at the end of its own
// last message (OpenAI's automatic boundary, Anthropic cache_control). Turn N+1 binding it is a warm
// conversation: a large new turn prefills in full. The same prompt bound only at the product end
// takes the path. The summary strategy's probe answers warm for the deeper shared match.
void test_lossy_shared_source_past_product_prompt() {
    constexpr std::uint32_t kTurn = kMarker + 10000; // turn N's shared boundary
    Rig rig({.chunk = kLossyChunk, .lanes = 1, .lossy_min_tokens = 30000, .context_cache = true,
             .shared_prefixes = {{7, kMarker}, {8, kTurn}}});
    fake::PreparedPrompt warm = lossy_prompt(1, kTurn + 31000, 8);
    warm.shared_tokens        = kTurn;
    auto warm_result          = rig.run_prompt(std::move(warm), 1);
    require(finished(warm_result), "the warm request did not finish\n" + rig.dump());
    require(warm_result.get().reused_prompt_tokens == kTurn,
            "the request did not bind turn N's shared checkpoint\n" + rig.dump());
    require(rig.events_of(1, Event::Kind::EnableLossy).empty() &&
                rig.events_of(1, Event::Kind::Scout).empty() && !warm_result.get().lossy_prefill,
            "a shared source past the product prompt took the lossy path\n" + rig.dump());
    require(total_tokens(rig.events_of(1, Event::Kind::Prefill)) == 31000,
            "the new turn was not prefilled in full\n" + rig.dump());

    auto cold = rig.run_prompt(lossy_prompt(2, kTurn + 31000, 7), 1);
    require(finished(cold), "the cold request did not finish\n" + rig.dump());
    const auto enabled = rig.events_of(2, Event::Kind::EnableLossy);
    require(cold.get().reused_prompt_tokens == kMarker && enabled.size() == 1 &&
                enabled[0].position == kMarker,
            "a shared source at the product end did not take the path at M\n" + rig.dump());

    Rig summary({.chunk = kLossyChunk, .lanes = 1, .lossy_min_tokens = 30000,
                 .lossy_strategy = LossyStrategy::Summary, .context_cache = true,
                 .shared_prefixes = {{7, kMarker}, {8, kTurn}}});
    ResolvedExecutionOptions execution;
    execution.requested_output_tokens = 1;
    fake::PreparedPrompt deep = lossy_prompt(3, kTurn + 31000, 8);
    deep.shared_tokens        = kTurn;
    require(summary.core().has_private_reuse(deep, execution, kMarker,
                                             std::chrono::milliseconds(250)) ==
                LossySummaryProbe::Warm,
            "a shared match past the product prompt was not probed warm");
    require(summary.core().has_private_reuse(lossy_prompt(4, kTurn + 31000, 7), execution, kMarker,
                                             std::chrono::milliseconds(250)) ==
                LossySummaryProbe::Cold,
            "a shared match at the product end was not probed cold");
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

// LossyResumesAfterExactPause: a short request at the head yields the long one while it is still
// prefilling the product prompt (Exact, before M). Restored, it finishes Exact at M, scouts U and
// prefills the kept tokens: the pause does not drop the lossy path. The Program carries the path
// in its saved control, so the Engine enables it once, at the first binding.
void test_lossy_resumes_after_exact_pause() {
    constexpr std::uint32_t kDeepMarker = 3 * kLossyChunk; // Exact takes three chunks
    constexpr std::uint32_t kPrompt     = kDeepMarker + 31000;
    constexpr std::uint32_t kKept       = 6200;            // ceil(0.2 * 31,000)
    Rig rig({.chunk = kLossyChunk, .lanes = 1, .yield_ms = 100, .lossy_min_tokens = 30000});
    rig.hold_at([](const Event& e) {
        return e.kind == Event::Kind::Prefill && e.request == 1 && e.position == kLossyChunk;
    });
    fake::PreparedPrompt prompt = lossy_prompt(1, kPrompt);
    prompt.lossy_exact_end      = kDeepMarker;
    auto long_result = rig.run_prompt(std::move(prompt), 1);
    require(rig.wait_held(), "the exact phase never reached its second chunk\n" + rig.dump());
    auto short_result = rig.run(2, 100, 1);
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    rig.release();
    require(finished(short_result) && finished(long_result), "a request did not finish\n" + rig.dump());
    const auto events      = rig.events();
    const auto pause_start = first_index(events, Event::Kind::PauseStart, 1);
    const auto restore     = first_index(events, Event::Kind::Restore, 1);
    const auto first_scout = first_index(events, Event::Kind::Scout, 1);
    require(pause_start && restore && first_scout && *pause_start < *restore &&
                *restore < *first_scout,
            "the long request was not paused in Exact and restored before its scout\n" + rig.dump());
    require(events[*pause_start].position < kDeepMarker, "the pause was not before M\n" + rig.dump());
    require(rig.events_of(1, Event::Kind::EnableLossy).size() == 1,
            "the path was not enabled exactly once\n" + rig.dump());
    const auto scout = rig.events_of(1, Event::Kind::Scout);
    require(total_tokens(scout) == kPrompt - kDeepMarker && scout.front().position == kDeepMarker,
            "the restored request did not scout U from M\n" + rig.dump());
    require(total_tokens(rig.events_of(1, Event::Kind::Prefill)) == kDeepMarker + kKept,
            "the restored request did not prefill M plus the kept tokens\n" + rig.dump());
    const auto& report = long_result.get().lossy_prefill;
    require(report && report->outcome == LossyPrefillOutcome::Completed &&
                report->uncached_tokens == kPrompt - kDeepMarker && report->kept_tokens == kKept,
            "the resumed lossy prefill was not reported completed");
    const auto stats = rig.core().runtime_stats();
    require(stats.lossy_prefills_completed == 1 && stats.lossy_prefills_aborted == 0 &&
                stats.prefill_yields == 1,
            "the resumed lossy prefill was not counted completed");
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

// SummaryRewriteReportsOriginal: a prompt the summary strategy rewrote runs as an ordinary prompt
// (no scout), and usage, prompt progress, the request log and the counters count the client's.
void test_summary_rewrite_reports_original() {
    Rig rig({.chunk = kLossyChunk, .lanes = 1, .lossy_min_tokens = 30000,
             .lossy_strategy = LossyStrategy::Summary});
    const compaction::SummaryRewrite rewrite{.original_tokens = 40000,
                                             .prompt_tokens   = 9000,
                                             .common_prefix   = 2000,
                                             .common_suffix   = 6000,
                                             .messages        = {4, 180},
                                             .fresh           = true};
    const auto record = [](compaction::SummaryRewrite r) {
        return compaction::SummaryRecord{.probe = LossySummaryProbe::Cold, .rewrite = r,
                                         .prompt_tokens = r.prompt_tokens};
    };
    ProgressSink sink;
    auto result = rig.run_prompt(lossy_prompt(1, 9000), 2, nullptr, &sink, record(rewrite));
    require(finished(result), "the summarised request did not finish\n" + rig.dump());
    const auto& generation = result.get();
    require(generation.prompt.prompt_tokens == 40000, "usage prompt_tokens is not the client's");
    require(generation.computed_prefill_tokens == 9000, "the submitted prompt was not prefilled");
    require(rig.events_of(1, Event::Kind::Scout).empty() &&
                rig.events_of(1, Event::Kind::EnableLossy).empty(),
            "a summarised prompt took the token path\n" + rig.dump());
    const auto& report = generation.lossy_prefill;
    require(report && report->strategy == LossyStrategy::Summary &&
                report->outcome == LossyPrefillOutcome::Completed &&
                report->uncached_tokens == 40000 && report->kept_tokens == 9000 &&
                report->summarized_messages ==
                    std::optional(std::array<std::uint32_t, 2>{4, 180}) &&
                !report->summary_reused && report->summary_probe == LossySummaryProbe::Cold,
            "the summary report is wrong");
    {
        std::lock_guard lock(sink.mutex);
        require(sink.starts.size() == 1 && sink.starts[0].prompt.prompt_tokens == 40000,
                "the stream start does not count the client's prompt");
        std::uint32_t last = 0;
        for (const auto& update : sink.updates) {
            require(update.total_prompt_tokens == 40000 && update.processed_prompt_tokens >= last &&
                        update.processed_prompt_tokens <= 40000,
                    "prompt progress left the client's prompt");
            last = update.processed_prompt_tokens;
        }
        require(last == 40000, "prompt progress did not reach the client's length: " +
                                   std::to_string(last));
    }
    auto reused       = rewrite;
    reused.fresh      = false;
    auto second       = rig.run_prompt(lossy_prompt(2, 9000), 1, nullptr, nullptr, record(reused));
    require(finished(second), "the reused-summary request did not finish\n" + rig.dump());
    require(second.get().lossy_prefill && second.get().lossy_prefill->summary_reused,
            "a substituted summary was not reported as reused");
    // A long prompt the probe could not settle is left whole: skipped, with the probe's answer.
    auto timed = rig.run_prompt(lossy_prompt(3, kLossyPrompt), 1, nullptr, nullptr,
                                compaction::SummaryRecord{.probe         = LossySummaryProbe::Timeout,
                                                          .prompt_tokens = kLossyPrompt});
    require(finished(timed), "the probed request did not finish\n" + rig.dump());
    const auto& skipped = timed.get().lossy_prefill;
    require(skipped && skipped->strategy == LossyStrategy::Summary &&
                skipped->outcome == LossyPrefillOutcome::Skipped &&
                skipped->summary_probe == LossySummaryProbe::Timeout &&
                skipped->uncached_tokens == kLossyPrompt && skipped->kept_tokens == kLossyPrompt &&
                !skipped->summarized_messages && timed.get().prompt.prompt_tokens == kLossyPrompt,
            "a probed, unsummarised prompt's report is wrong");
    const auto stats = rig.core().runtime_stats();
    require(stats.lossy_prefill_summaries_new == 1 && stats.lossy_prefill_summaries_reused == 1 &&
                stats.lossy_prefills_completed == 2 && stats.lossy_prefills_skipped == 1 &&
                stats.lossy_prefill_scouted_tokens == 0 && stats.prompt_tokens == 80000 + kLossyPrompt,
            "the summary counters must match the logged outcomes");
}

// SummaryOutputCappedByOriginal: a shortened prompt may not generate past where the client's own
// prompt would reach max_context.
void test_summary_output_capped_by_original() {
    Rig rig({.chunk = kLossyChunk, .lanes = 1, .lossy_min_tokens = 30000,
             .lossy_strategy = LossyStrategy::Summary});
    const std::uint32_t max_context = 1U << 20; // the Rig's
    const auto rewrite_with = [](std::uint32_t original) {
        return compaction::SummaryRewrite{.original_tokens = original,
                                          .prompt_tokens   = 9000,
                                          .messages        = {4, 180},
                                          .fresh           = true};
    };
    const compaction::SummaryRewrite rewrite = rewrite_with(max_context - 3);
    auto result = rig.run_prompt(lossy_prompt(1, 9000), 64, nullptr, nullptr,
                                 compaction::SummaryRecord{.probe = LossySummaryProbe::Cold,
                                                           .rewrite = rewrite,
                                                           .prompt_tokens = 9000});
    require(finished(result), "the request did not finish\n" + rig.dump());
    require(result.get().generated_token_ids.size() == 4,
            "the output was not capped at max_context - original + 1: " +
                std::to_string(result.get().generated_token_ids.size()));
    require(result.get().finish_reason == FinishReason::ContextCapacity,
            "a request stopped by the summary's context cap must end for context capacity");
    // A cap that does not bind leaves the request's own output limit.
    auto roomy = rig.run_prompt(lossy_prompt(2, 9000), 8, nullptr, nullptr,
                                compaction::SummaryRecord{.probe = LossySummaryProbe::Cold,
                                                          .rewrite = rewrite_with(40000),
                                                          .prompt_tokens = 9000});
    require(finished(roomy), "the request did not finish\n" + rig.dump());
    require(roomy.get().generated_token_ids.size() == 8 &&
                roomy.get().finish_reason == FinishReason::OutputLimit,
            "a non-binding summary cap changed the finish reason");
}

// SummaryStrategyDisablesTokenPath: under --lossy-prefill-strategy summary the core never scouts,
// even for a cold long prompt the summary did not rewrite (a busy probe counts as warm).
void test_summary_strategy_disables_token_path() {
    Rig rig({.chunk = kLossyChunk, .lanes = 1, .lossy_min_tokens = 30000,
             .lossy_strategy = LossyStrategy::Summary});
    auto result = rig.run_prompt(lossy_prompt(1), 1);
    require(finished(result), "the request did not finish\n" + rig.dump());
    require(rig.events_of(1, Event::Kind::EnableLossy).empty() &&
                rig.events_of(1, Event::Kind::Scout).empty() && !result.get().lossy_prefill,
            "the summary strategy ran the token path\n" + rig.dump());
    require(total_tokens(rig.events_of(1, Event::Kind::Prefill)) == kLossyPrompt,
            "the prompt was not prefilled in full");
}

// PrivateReuseProbe: the summary strategy's probe from a caller thread. An idle Engine with only a
// shared product prompt cached answers cold; a worker busy past the wait answers warm, in bounded
// time.
void test_private_reuse_probe() {
    Rig rig({.chunk = kLossyChunk, .lanes = 1, .lossy_min_tokens = 30000,
             .lossy_strategy = LossyStrategy::Summary, .context_cache = true,
             .shared_prefixes = {{7, kMarker}}});
    ResolvedExecutionOptions execution;
    execution.requested_output_tokens = 1;
    require(rig.core().has_private_reuse(lossy_prompt(1, kLossyPrompt, 7), execution, kMarker,
                                         std::chrono::milliseconds(250)) == LossySummaryProbe::Cold,
            "a prompt only the shared prefix serves was not probed cold");
    rig.hold_at([](const Event& e) { return e.kind == Event::Kind::Prefill && e.request == 2; });
    auto busy = rig.run_prompt(lossy_prompt(2, 8192), 1);
    require(rig.wait_held(), "the worker was not held\n" + rig.dump());
    const auto started = Clock::now();
    const auto answer  = rig.core().has_private_reuse(lossy_prompt(3, kLossyPrompt, 7), execution,
                                                      kMarker, std::chrono::milliseconds(50));
    const auto waited  = Clock::now() - started;
    rig.release();
    require(answer == LossySummaryProbe::Timeout,
            "a probe the busy worker could not answer was not a timeout");
    require(waited >= std::chrono::milliseconds(45) && waited < std::chrono::seconds(2),
            "the probe did not wait its bound");
    require(rig.core().runtime_stats().lossy_prefill_summary_probe_timeouts == 1,
            "the timeout was not counted");
    require(finished(busy), "the held request did not finish\n" + rig.dump());
    // Between cycles of a running request, the worker answers: a long decode leaves gaps.
    auto decoding = rig.run_prompt(lossy_prompt(4, 2048), 200);
    require(rig.wait_for([&] { return !rig.events_of(4, Event::Kind::Decode).empty(); }),
            "the decode did not start\n" + rig.dump());
    require(rig.core().has_private_reuse(lossy_prompt(5, kLossyPrompt, 7), execution, kMarker,
                                         std::chrono::milliseconds(250)) == LossySummaryProbe::Cold,
            "a decoding worker did not answer the probe between cycles");
    require(finished(decoding), "the decoding request did not finish\n" + rig.dump());
}

// StoredSessionProbedWarm: a conversation only the context store holds (admission would hydrate
// it) is warm, so it is not summarised; one the store does not hold, or holds only within the
// product prompt, is cold.
void test_stored_session_probed_warm() {
    auto bucket = std::make_shared<MemoryBucket>();
    StoreDirectory directory;
    ContextStore store(bucket_store_options(directory.path(), bucket));
    require(store_drained(store), "the start-up refresh did not finish");
    const auto put = [&](std::uint64_t id, std::uint32_t frontier) {
        ContextStore::Description description;
        description.id      = "5e57" + std::to_string(id);
        description.binding = "fake-binding";
        description.tokens  = frontier;
        description.checkpoints.push_back({.frontier = frontier, .digests = {id, 0}, .identity_tag = 0});
        std::vector<std::uint8_t> image(64 + 4096, static_cast<std::uint8_t>(id));
        (void)store.put(description, image, std::array{ContextStore::Region{64, 4096}});
        require(store.find(description.id).has_value(), "the session was not stored");
    };
    put(7, 30'000);
    put(9, 1'000);
    Rig rig({.chunk = kLossyChunk, .lanes = 1, .lossy_min_tokens = 30000,
             .lossy_strategy = LossyStrategy::Summary});
    rig.attach_store(store, std::chrono::seconds(2));
    ResolvedExecutionOptions execution;
    execution.requested_output_tokens = 1;
    const auto probe = [&](std::uint64_t id) {
        return rig.core().has_private_reuse(fake::PreparedPrompt{kLossyPrompt, id}, execution, kMarker,
                                            std::chrono::milliseconds(250));
    };
    require(probe(7) == LossySummaryProbe::Warm, "a session the store holds was probed cold");
    require(probe(8) == LossySummaryProbe::Cold, "a session the store does not hold was probed warm");
    require(probe(9) == LossySummaryProbe::Cold,
            "a stored prefix within the product prompt made the prompt warm");
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
                report->kept_tokens == kLossyKept && report->filter_layer == 7 &&
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

// A request on the lossy path that fails before its outcome is counted aborted, so the outcomes
// add up to the requests that took the path.
void test_lossy_failure_counts_aborted() {
    Rig rig({.chunk = kLossyChunk, .lanes = 1, .lossy_min_tokens = 30000});
    rig.hold_at([](const Event& e) {
        return e.kind == Event::Kind::Scout && e.request == 1 && e.position == kMarker + kLossyChunk;
    });
    auto result = rig.run_prompt(lossy_prompt(1), 1);
    require(rig.wait_held(), "the scout never reached its second step\n" + rig.dump());
    arm_worker_failures(1); // the held scout step throws once it has run
    rig.release();
    bool failed = false;
    try {
        (void)finished(result);
    } catch (const std::exception&) { failed = true; }
    arm_worker_failures(0);
    require(failed, "the injected failure did not fail the request\n" + rig.dump());
    const auto stats = rig.core().runtime_stats();
    require(stats.lossy_prefills_aborted == 1 && stats.lossy_prefills_completed == 0 &&
                stats.lossy_prefills_skipped == 0,
            "a lossy request that failed was not counted aborted");
    // The Engine recovered: the next lossy request runs to its outcome.
    auto next = rig.run_prompt(lossy_prompt(2), 1);
    require(finished(next), "the Engine did not serve after the failure\n" + rig.dump());
    require(rig.core().runtime_stats().lossy_prefills_completed == 1,
            "the request after the failure was not counted");
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
    run("no yield when the snapshot does not fit", test_no_yield_when_the_snapshot_does_not_fit);
    run("rotation interleaves two long prefills", test_rotation_interleaves_two_long_prefills);
    run("shortest runs one long prefill at a time", test_shortest_runs_one_long_prefill_at_a_time);
    run("shortest lets a short prompt overtake a long one",
        test_shortest_lets_a_short_prompt_overtake_a_long_one);
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
    run("a lookup reads only the request's own manifests",
        test_a_lookup_reads_only_the_requests_own_manifests);
    run("a slow download stops at the request's deadline",
        test_a_slow_download_stops_at_the_requests_deadline);
    run("a read finished after the deadline is not imported",
        test_a_read_finished_after_the_deadline_is_not_imported);
    run("an import that throws fails only the hydration",
        test_an_import_that_throws_fails_only_the_hydration);
    run("an unrelated removal does not rewrite stored sessions",
        test_an_unrelated_removal_does_not_rewrite_stored_sessions);
    run("a session the store evicted is not written back",
        test_a_session_the_store_evicted_is_not_written_back);
    run("a failed write is retried after a removal scan dropped it",
        test_a_failed_write_is_retried_after_a_removal_scan_dropped_it);
    run("a flush writes unsaved sessions with idle writes off",
        test_a_flush_writes_unsaved_sessions_with_idle_writes_off);
    run("lost failure reports retry every missing session",
        test_lost_failure_reports_retry_every_missing_session);
    run("an export reuses the writer's spare buffer", test_an_export_reuses_the_writers_spare_buffer);
    run("a foreign image of the same id does not count as stored",
        test_a_foreign_image_of_the_same_id_does_not_count_as_stored);
    run("LossyTriggersOnlyWhenCold", test_lossy_triggers_only_when_cold);
    run("LossyPromptLengthChangeAccepted", test_lossy_prompt_length_change_accepted);
    run("LossyCancelDuringScout", test_lossy_cancel_during_scout);
    run("LossyYieldRefusedDuringScout", test_lossy_yield_refused_during_scout);
    run("LossyResumesAfterExactPause", test_lossy_resumes_after_exact_pause);
    run("LossySharedSourcePastProductPrompt", test_lossy_shared_source_past_product_prompt);
    run("LossyPreemptionDuringScout", test_lossy_preemption_during_scout);
    run("LossyUsageReportsOriginal", test_lossy_usage_reports_original);
    run("LossyCountersAndRequestLog", test_lossy_counters_and_request_log);
    run("LossyBacklogPredictsScoutAndMain", test_lossy_backlog_predicts_scout_and_main);
    run("LossyFailureCountsAborted", test_lossy_failure_counts_aborted);
    run("SummaryRewriteReportsOriginal", test_summary_rewrite_reports_original);
    run("SummaryStrategyDisablesTokenPath", test_summary_strategy_disables_token_path);
    run("PrivateReuseProbe", test_private_reuse_probe);
    run("StoredSessionProbedWarm", test_stored_session_probed_warm);
    run("SummaryOutputCappedByOriginal", test_summary_output_capped_by_original);
    return failures == 0 ? 0 : 1;
}
