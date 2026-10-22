// EngineCore's scheduling, run against the GPU-free simulation in fake_engine.h.
//
// These tests exercise the real worker loop, admission, pause/restore and stats code with a fake
// model, so they check what the Engine does and in what order, not what a GPU would. The fake's
// cost model and the Engine's are the same numbers, so a prefill step of N tokens "takes"
// fixed + N * per_token on both sides.

#include "runtime/fake_engine.h"

#include <chrono>
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

struct Settings {
    std::uint32_t chunk          = 1024;
    std::uint32_t lanes          = 2;
    std::uint32_t stall_budget   = 0;
    std::uint32_t decode_share   = 0;
    std::uint32_t yield_ms       = 0;
    std::chrono::microseconds decode_sleep{1000};
    // Real time a prefill step takes, for a test that must submit while one runs.
    std::chrono::microseconds prefill_sleep{0};
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
        options.pending_timeout_ms    = 60'000;
        options.prefill_chunk         = settings.chunk;
        options.context_cache.enabled = false;
        options.prefill_schedule      = {.stall_budget_ms      = settings.stall_budget,
                                         .yield_remaining_ms   = settings.yield_ms,
                                         .decode_share_percent = settings.decode_share};
        ContextMachineCostModel costs;
        costs.prefill.chunk_ns     = kFixedNs;
        costs.prefill.token_ns_q32 = kPerTokenNs << 32U;
        core_ = std::make_unique<EngineCore<fake::Instance>>(instance_, device_, options, costs);
    }

    ~Rig() {
        // Release a test that failed while holding the worker, then stop the Engine, which fails
        // whatever is still waiting so the waiter threads can end.
        release();
        core_.reset();
        for (std::thread& waiter : waiters_) { waiter.join(); }
    }

    // Starts a request and returns its result future. `id` names it in the event log.
    std::future<GenerationResult> run(std::uint64_t id, std::uint32_t prompt_tokens,
                                      std::uint32_t output_tokens) {
        ResolvedRequestOptions request;
        request.execution.requested_output_tokens = output_tokens;
        PromptSummary summary;
        summary.prompt_tokens = prompt_tokens;
        auto submission = std::make_shared<EngineCore<fake::Instance>::Submission>(core_->submit(
            fake::PreparedPrompt{prompt_tokens, id}, summary, 0.0, std::move(request),
            OutputConsumerMode::Aggregate, GenerationObservationOptions{}));
        // A promise, not std::async: a future from async blocks in its destructor, which would
        // deadlock a failing test that still holds the worker.
        auto promise = std::make_shared<std::promise<GenerationResult>>();
        auto result  = promise->get_future();
        waiters_.emplace_back([submission, promise] {
            try {
                promise->set_value(submission->wait(nullptr, CancellationView{}));
            } catch (...) { promise->set_exception(std::current_exception()); }
        });
        return result;
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
        const char* names[] = {"Bind", "Restore", "Prefill", "Decode", "PauseStart", "PauseDone",
                               "Finish", "Abort"};
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

[[nodiscard]] bool finished(std::future<GenerationResult>& result,
                            std::chrono::milliseconds limit = std::chrono::seconds(30)) {
    return result.wait_for(limit) == std::future_status::ready;
}

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
    (void)stream.wait_for(std::chrono::seconds(30));
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
    (void)stream.wait_for(std::chrono::seconds(30));
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
    (void)stream.wait_for(std::chrono::seconds(60));
    (void)plain_stream.wait_for(std::chrono::seconds(60));
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
void start_long_prompt(Rig& rig, std::future<GenerationResult>& long_result) {
    rig.hold_at([](const Event& e) {
        return e.kind == Event::Kind::Prefill && e.request == 1 && e.position == 1024;
    });
    long_result = rig.run(1, 8192, 1);
    require(rig.wait_held(), "the long prompt never reached its second step\n" + rig.dump());
}

void test_without_yield_a_short_request_waits_for_the_long_prompt() {
    Rig rig({.chunk = 1024, .lanes = 1});
    std::future<GenerationResult> long_result;
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
    std::future<GenerationResult> long_result;
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
    std::future<GenerationResult> long_result;
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
    std::future<GenerationResult> long_result;
    start_long_prompt(rig, long_result);
    auto short_result = rig.run(2, 100, 1);
    rig.release();
    require(finished(long_result) && finished(short_result), "a request did not finish\n" + rig.dump());
    require(rig.count(Event::Kind::PauseStart) == 0,
            "a prefill with less than the threshold left was paused\n" + rig.dump());
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
    run("a long newcomer does not displace a long prefill",
        test_a_long_newcomer_does_not_displace_a_long_prefill);
    run("a prefill with little left is not paused", test_a_prefill_with_little_left_is_not_paused);
    run("backlog gauge tracks queued and resident prefill",
        test_backlog_gauge_tracks_queued_and_resident_prefill);
    return failures == 0 ? 0 : 1;
}
