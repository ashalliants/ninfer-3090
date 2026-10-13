// Qwen3.8-Flash-Next with two lanes (--max-concurrency 2) through the public Engine on the real
// artifact. Skips (77) unless NINFER_TEST_QWEN4EXP_ARTIFACT names a Qwen4Exp artifact (its n-gram
// volume beside it).
//
// Exactness mode (--exact): two lanes decode in the same rounds (plain batched rounds; MTP drafts only
// when one lane decodes), and every request's greedy ids must equal the same request run alone on
// a one-lane Engine. One-lane Engine first (the references: three prompts of different lengths, the
// longest deep inside the QSA budget, and the longest prompt plus a suffix), then one two-lane Engine:
//   pairs         every two of the three prompts at once (both lanes decode in shared rounds)
//   staggered     the longest prompt first, a short one submitted while it decodes (its prefill calls
//                 interleave with the other lane's decode rounds, then both decode together)
//   alone         one request on the two-lane Engine
//   cache         (--cache) the longest prompt cached, then its suffix prompt (resumed from the
//                 checkpoint in one lane) beside a short prompt; then the same pair with one request
//                 cancelled after a few tokens: the other lane's ids must be unchanged
//                 The cached prompt is 8,192 tokens, a prefill call boundary: a resume where the cold
//                 prefill has no call boundary may differ in greedy ids (test_engine_real.cpp, cache_check).
//
//   ninfer_qwen4_exp_lanes_real_test --exact [--mtp K] [--generate N] [--cache] [--max-context N]
//
// Bench mode (--bench): one Engine with --concurrency C lanes; JOBS (token files, or LENGTH for a
// synthetic prompt of that many tokens) are submitted at once, at most C in flight at a time. Reports
// each request's prefill and decode speed and the makespan.
//
//   ninfer_qwen4_exp_lanes_real_test --bench [--concurrency C] [--mtp K] [--generate N] JOBS...
//
// A synthetic prompt is a fixed pseudo-random sequence of ordinary token ids (exactness needs
// deterministic ids, not language), different per length.

#include "ninfer/engine.h"
#include "ninfer/types.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#ifdef _WIN32
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    include <windows.h>
#endif

using namespace ninfer;

namespace {

double available_ram_gib() {
#ifdef _WIN32
    MEMORYSTATUSEX status{};
    status.dwLength = sizeof(status);
    if (GlobalMemoryStatusEx(&status)) { return static_cast<double>(status.ullAvailPhys) / (1ULL << 30); }
#endif
    return 0.0;
}

// Between Engines: waits up to a minute for the host RAM of a destroyed Engine to come back.
void wait_for_ram(double before) {
    const auto start = std::chrono::steady_clock::now();
    double now       = available_ram_gib();
    while (now < before - 1.0 && std::chrono::steady_clock::now() - start < std::chrono::seconds(60)) {
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        now = available_ram_gib();
    }
    std::printf("available RAM %.1f GiB (%.1f before the Engine)\n", now, before);
}

std::vector<TokenId> synthetic(std::size_t length, std::uint32_t seed) {
    std::vector<TokenId> out;
    std::uint64_t state = 0x9E3779B97F4A7C15ULL ^ (static_cast<std::uint64_t>(seed) * 0xBF58476D1CE4E5B9ULL);
    for (std::size_t i = 0; i < length; ++i) {
        state = state * 6364136223846793005ULL + 1442695040888963407ULL;
        out.push_back(static_cast<TokenId>(2'000 + (state >> 40) % 9'000));
    }
    return out;
}

std::vector<TokenId> parse_job(const std::string& job, std::uint32_t seed) {
    if (job.empty() || job.find_first_not_of("0123456789") == std::string::npos) {
        return synthetic(std::stoul(job), seed);
    }
    std::string text = job;
    if (text.front() == '@') {
        std::ifstream file(text.substr(1));
        if (!file) { throw std::invalid_argument("cannot read " + text.substr(1)); }
        text.assign(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
    }
    for (char& ch : text) {
        if (ch == ',' || ch == '\n' || ch == '\r' || ch == '\t') { ch = ' '; }
    }
    std::vector<TokenId> out;
    std::stringstream stream(text);
    TokenId id = 0;
    while (stream >> id) { out.push_back(id); }
    return out;
}

EngineOptions base_options(const std::string& artifact, std::uint32_t max_context, std::uint32_t lanes,
                           std::uint32_t mtp, bool cache) {
    EngineOptions options;
    options.artifact_path                = artifact;
    options.max_context                  = max_context;
    options.kv_capacity                  = KvCapacityPolicy::explicit_capacity(max_context);
    options.max_concurrency              = lanes;
    options.prefill_chunk                = 1024;
    options.kv_cache                     = KvCacheStorage::Int8Group64;
    options.context_cache.enabled        = cache;
    options.context_cache.host_capacity_bytes = cache ? std::optional<std::size_t>(2ULL << 30) : std::optional<std::size_t>(0);
    options.pending_timeout_ms           = 3'600'000;
    options.flash_next.save_expert_state = false;
    if (mtp > 0) {
        options.speculative.backend      = SpeculativeBackend::Mtp;
        options.speculative.draft_tokens = mtp;
    }
    return options;
}

struct Run {
    std::vector<TokenId> ids;
    GenerationTimings timings;
    SpeculativeStats speculative;
    std::uint32_t reused = 0;
    double submitted = 0, finished = 0; // seconds from the sitting's start
};

class CancelSink final : public OutputSink {
public:
    explicit CancelSink(std::uint32_t after) : after_(after) {}
    void start(GenerationStart) override {}
    void progress(PromptProgress) override {}
    void publish(OutputDelta) override {}
    void timing(GenerationTimingObservation t) override {
        if (after_ != 0 && t.generated_tokens >= after_) { cancel_ = true; }
    }
    [[nodiscard]] bool cancelled() const { return cancel_.load(); }

private:
    std::uint32_t after_;
    std::atomic<bool> cancel_{false};
};

using Clock = std::chrono::steady_clock;

double since(Clock::time_point origin) { return std::chrono::duration<double>(Clock::now() - origin).count(); }

// One greedy request, to `tokens` generated tokens with the model's stops off; optionally cancelled
// after `cancel_after` committed tokens.
Run generate(Engine& engine, const std::vector<TokenId>& prompt, std::uint32_t tokens, bool reuse, Clock::time_point origin,
             std::uint32_t cancel_after = 0) {
    RequestOptions request;
    request.execution.requested_output_tokens = tokens;
    request.execution.allow_prefix_reuse      = reuse;
    request.execution.sampling.temperature    = 0.0F;
    request.stop.include_model_defaults       = false;
    request.output.raw                        = true;
    Run out;
    out.submitted = since(origin);
    CancelSink sink(cancel_after);
    auto handle = engine.submit(engine.prepare_tokens(prompt, reuse), request, OutputConsumerMode::Streaming,
                                GenerationObservationOptions{.live_timings = true});
    auto result = handle.wait(&sink, CancellationView([&sink] { return sink.cancelled(); }));
    out.ids         = std::move(result.generated_token_ids);
    out.timings     = result.timings;
    out.speculative = result.speculative;
    out.reused      = result.reused_prompt_tokens;
    out.finished    = since(origin);
    return out;
}

bool same(const std::string& label, const std::vector<TokenId>& reference, const std::vector<TokenId>& ids) {
    if (reference == ids) {
        std::printf("ok: %s: %zu ids identical\n", label.c_str(), ids.size());
        return true;
    }
    std::size_t first = 0;
    while (first < reference.size() && first < ids.size() && reference[first] == ids[first]) { ++first; }
    std::printf("FAIL: %s: %zu ids against %zu, first difference at generated token %zu\n", label.c_str(), ids.size(),
                reference.size(), first);
    return false;
}

void print_run(const std::string& label, const Run& run, std::size_t prompt) {
    const double decode = run.ids.size() > 1 && run.timings.decode_seconds > 0
                              ? static_cast<double>(run.ids.size() - 1) / run.timings.decode_seconds
                              : 0.0;
    std::printf("%s: %zu prompt tokens (%u reused), %zu generated; submitted at %.2f s, first token %.2f s, finished "
                "%.2f s; prefill %.2f s, decode %.2f s (%.1f tok/s)\n",
                label.c_str(), prompt, run.reused, run.ids.size(), run.submitted, run.timings.first_token_seconds,
                run.finished, run.timings.prefill_seconds, run.timings.decode_seconds, decode);
}

struct Options {
    std::uint32_t max_context = 16'384, tokens = 96, mtp = 0, lanes = 2;
    bool cache = false;
    bool resume_alone = false; // --resume-alone: the cache resume runs without a second request beside it
    std::vector<std::string> jobs;
};

Options parse(const std::vector<std::string>& args) {
    Options o;
    for (std::size_t i = 0; i < args.size(); ++i) {
        const auto value = [&] {
            if (i + 1 >= args.size()) { throw std::invalid_argument(args[i] + " needs a value"); }
            return static_cast<std::uint32_t>(std::stoul(args[++i]));
        };
        if (args[i] == "--max-context") {
            o.max_context = value();
        } else if (args[i] == "--generate") {
            o.tokens = value();
        } else if (args[i] == "--mtp") {
            o.mtp = value();
        } else if (args[i] == "--concurrency") {
            o.lanes = value();
        } else if (args[i] == "--resume-alone") {
            o.resume_alone = true;
        } else if (args[i] == "--cache") {
            o.cache = true;
        } else {
            o.jobs.push_back(args[i]);
        }
    }
    return o;
}

int exact(const std::string& artifact, const Options& o) {
    // Three prompts of different lengths (the last is deep in the QSA budget) and the last with a
    // suffix; the references are one-lane runs of each.
    const std::vector<TokenId> short_prompt  = synthetic(700, 1);
    const std::vector<TokenId> middle_prompt = synthetic(3'000, 2);
    const std::vector<TokenId> long_prompt   = synthetic(8'192, 3);
    std::vector<TokenId> long_suffix         = long_prompt;
    {
        const auto suffix = synthetic(600, 4);
        long_suffix.insert(long_suffix.end(), suffix.begin(), suffix.end());
    }
    const std::vector<const std::vector<TokenId>*> prompts = {&short_prompt, &middle_prompt, &long_prompt};
    const char* names[]                                    = {"short(700)", "middle(3000)", "long(8192)"};
    const auto origin                                      = Clock::now();
    const double ram_before                                = available_ram_gib();
    std::vector<Run> reference;
    Run reference_suffix;
    {
        Engine engine(base_options(artifact, o.max_context, 1, o.mtp, false));
        for (std::size_t i = 0; i < prompts.size(); ++i) {
            reference.push_back(generate(engine, *prompts[i], o.tokens, false, origin));
            print_run(std::string("one lane, alone: ") + names[i], reference.back(), prompts[i]->size());
        }
        reference_suffix = generate(engine, long_suffix, o.tokens, false, origin);
        print_run("one lane, alone: long(8192)+600", reference_suffix, long_suffix.size());
    }
    wait_for_ram(ram_before);
    bool ok = true;
    {
        Engine engine(base_options(artifact, o.max_context, o.lanes, o.mtp, o.cache));
        const auto pair = [&](std::size_t a, std::size_t b, int stagger_ms, const char* label) {
            Run ra, rb;
            std::thread first([&] { ra = generate(engine, *prompts[a], o.tokens, false, origin); });
            std::this_thread::sleep_for(std::chrono::milliseconds(stagger_ms));
            std::thread second([&] { rb = generate(engine, *prompts[b], o.tokens, false, origin); });
            first.join();
            second.join();
            print_run(std::string(label) + " " + names[a], ra, prompts[a]->size());
            print_run(std::string(label) + " " + names[b], rb, prompts[b]->size());
            // The two requests overlapped in time, or this did not test shared rounds.
            const bool overlapped = ra.submitted < rb.finished && rb.submitted < ra.finished;
            if (!overlapped) {
                std::printf("FAIL: %s: the requests did not overlap\n", label);
                ok = false;
            }
            ok = same(std::string(label) + " " + names[a], reference[a].ids, ra.ids) && ok;
            ok = same(std::string(label) + " " + names[b], reference[b].ids, rb.ids) && ok;
        };
        if (o.lanes >= 2) {
            pair(0, 1, 0, "pair");
            pair(1, 2, 0, "pair");
            pair(2, 0, 0, "pair");
            // The long prompt decodes while the short one prefills (its calls interleave with the
            // long one's decode rounds), then both decode together.
            pair(2, 0, 1500, "staggered");
            pair(1, 0, 400, "staggered");
        }
        for (std::size_t i = 0; i < prompts.size(); ++i) {
            const Run r = generate(engine, *prompts[i], o.tokens, false, origin);
            print_run(std::string("two lanes, alone: ") + names[i], r, prompts[i]->size());
            ok = same(std::string("alone ") + names[i], reference[i].ids, r.ids) && ok;
        }
        if (o.cache) {
            // The long prompt cached, then its suffix prompt resumes from the checkpoint beside a short prompt.
            const Run warm = generate(engine, long_prompt, o.tokens, true, origin);
            print_run("cache: long(8192) cold, captures", warm, long_prompt.size());
            ok = same("cache long(8192)", reference[2].ids, warm.ids) && ok;
            Run resumed, other;
            if (o.resume_alone || o.lanes < 2) {
                resumed = generate(engine, long_suffix, o.tokens, true, origin);
            } else {
                std::thread first([&] { resumed = generate(engine, long_suffix, o.tokens, true, origin); });
                std::thread second([&] { other = generate(engine, short_prompt, o.tokens, false, origin); });
                first.join();
                second.join();
            }
            print_run("cache: long(8192)+600 resumed", resumed, long_suffix.size());
            ok = same("cache resumed long+600", reference_suffix.ids, resumed.ids) && ok;
            if (!other.ids.empty()) {
                print_run("cache: short(700) beside it", other, short_prompt.size());
                ok = same("cache short beside the resume", reference[0].ids, other.ids) && ok;
            }
            if (resumed.reused != long_prompt.size()) {
                std::printf("FAIL: the resume reused %u prompt tokens, expected %zu\n", resumed.reused, long_prompt.size());
                ok = false;
            }
            // One lane cancelled after a few tokens: the other's ids are unchanged.
            if (o.lanes < 2) { return ok ? 0 : 1; }
            Run cancelled, survivor;
            std::thread third([&] { cancelled = generate(engine, middle_prompt, o.tokens, false, origin, 8); });
            std::thread fourth([&] { survivor = generate(engine, short_prompt, o.tokens, false, origin); });
            third.join();
            fourth.join();
            std::printf("cancelled lane stopped after %zu tokens\n", cancelled.ids.size());
            ok = same("cancel: the other lane", reference[0].ids, survivor.ids) && ok;
            if (cancelled.ids.size() >= o.tokens) {
                std::printf("FAIL: the cancelled request ran to its end\n");
                ok = false;
            }
            // A request after the cancel: the lane is clean.
            const Run again = generate(engine, middle_prompt, o.tokens, false, origin);
            ok              = same("after the cancel: middle(3000)", reference[1].ids, again.ids) && ok;
        }
        const MemorySummary memory = engine.memory_summary();
        std::printf("memory: KV %.0f MiB backed of %u tokens\n", static_cast<double>(memory.kv_payload_bytes) / (1 << 20),
                    memory.kv_capacity);
    }
    std::printf(ok ? "qwen4_exp lanes exactness passed\n" : "FAIL: qwen4_exp lanes exactness\n");
    return ok ? 0 : 1;
}

int bench(const std::string& artifact, const Options& o) {
    if (o.jobs.empty()) { throw std::invalid_argument("--bench needs jobs"); }
    std::vector<std::vector<TokenId>> prompts;
    for (std::size_t i = 0; i < o.jobs.size(); ++i) { prompts.push_back(parse_job(o.jobs[i], static_cast<std::uint32_t>(10 + i))); }
    const auto load = Clock::now();
    Engine engine(base_options(artifact, o.max_context, o.lanes, o.mtp, o.cache));
    std::printf("engine loaded in %.1f s (%u lanes)\n", since(load), o.lanes);
    const auto origin = Clock::now();
    std::vector<Run> runs(prompts.size());
    std::atomic<std::size_t> next{0};
    std::vector<std::thread> workers;
    for (std::uint32_t w = 0; w < o.lanes; ++w) {
        workers.emplace_back([&] {
            for (;;) {
                const std::size_t i = next.fetch_add(1);
                if (i >= prompts.size()) { return; }
                runs[i] = generate(engine, prompts[i], o.tokens, o.cache, origin);
            }
        });
    }
    for (auto& w : workers) { w.join(); }
    const double makespan = since(origin);
    std::size_t generated = 0;
    for (std::size_t i = 0; i < runs.size(); ++i) {
        print_run("job " + std::to_string(i), runs[i], prompts[i].size());
        generated += runs[i].ids.size();
    }
    std::printf("makespan %.2f s for %zu generated tokens (%.1f tok/s overall)\n", makespan, generated,
                static_cast<double>(generated) / makespan);
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    const char* artifact_env = std::getenv("NINFER_TEST_QWEN4EXP_ARTIFACT");
    if (artifact_env == nullptr) {
        std::printf("SKIP: set NINFER_TEST_QWEN4EXP_ARTIFACT to a Qwen3.8-Flash-Next artifact\n");
        return 77;
    }
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    try {
        std::vector<std::string> args(argv + 1, argv + argc);
        if (args.empty()) {
            std::printf("SKIP: pass --exact or --bench\n");
            return 77;
        }
        const std::string mode = args.front();
        const Options options  = parse(std::vector<std::string>(args.begin() + 1, args.end()));
        if (mode == "--exact") { return exact(artifact_env, options); }
        if (mode == "--bench") { return bench(artifact_env, options); }
        throw std::invalid_argument("unknown mode " + mode);
    } catch (const std::exception& error) {
        std::printf("FAIL: %s\n", error.what());
        return 1;
    }
}
