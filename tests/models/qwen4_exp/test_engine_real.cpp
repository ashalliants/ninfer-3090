// Qwen3.8-Flash-Next through the public Engine on the real artifact (milestone M1). Skips (77)
// unless NINFER_TEST_QWEN4EXP_ARTIFACT names a Qwen4Exp artifact (its n-gram volume beside it).
//
// Default mode (the M1 gates):
//   1. Placement invariance end to end: greedy ids of one request are identical with the expert
//      cache cold (first request of a fresh Engine), warm (the same request again), warm-started
//      from the saved state of the first Engine (a second Engine), and small (a 1 GiB cache that
//      must evict every round, in a third Engine whose KV pool starts at 64 tokens and grows from the
//      expert frames, and which decodes without the CUDA graph, so graph replay = eager as well).
//   2. The RAM ledger refuses cleanly: an Engine asked to keep more RAM free than the machine has
//      fails before anything is pinned, with a message that says how much is missing.
//   3. The elastic KV pool's grow/shrink cycle on a full expert cache (kv_cycle below): three
//      short/long alternations stay within the OS video-memory budget at every sample, and greedy
//      ids repeat. --kv-cycle runs only this check.
//
//   ninfer_qwen4_exp_engine_real_test [TOKENS] [--generate N] [--max-context N] [--kv-cycle]
//
// Bench mode: one Engine, then each job in order (a prompt and greedy output tokens with model
// stops disabled, so every request runs to N). Reports prefill and decode tok/s per job; the
// Program logs each request's expert-cache hit rate.
//
//   ninfer_qwen4_exp_engine_real_test --bench [--max-context N] [--generate N] [--no-graph]
//       [--expert-cache-mib N] [--no-expert-state] [--split N] [--print-ids] [--vision] [--prefill-chunk N]
//       [--mtp K [--mtp-fixed] [--width-cost X]] TOKENS...
//
// --vision loads the artifact's vision tower (an Engine started with --vision; the jobs stay text).
//
// --mtp K decodes with MTP speculation (the artifact must hold the drafter), verifying up to K
// drafts per round (--mtp-fixed: always K); each job also prints its acceptance.
//
// MTP exactness mode: greedy output with MTP speculation must equal plain greedy output. One plain
// Engine decodes every job, then one MTP Engine per depth K; the ids must be identical.
//
//   ninfer_qwen4_exp_engine_real_test --mtp-exact K[,K...] [--max-context N] [--generate N]
//       [--mtp-fixed] TOKENS...
//
// --split N streams each request with live timings and also reports decode tok/s over generated
// tokens 2..N and N+1..end (decode right after a prefill, while the expert cache refills).
// --print-ids prints each job's generated ids (a needle check decodes them).
//
// Context cache mode (cache_check below): a conversation through an Engine with the context cache,
// every request's greedy ids against the same prompt run cold, and the prompt tokens reused.
//
//   ninfer_qwen4_exp_engine_real_test --cache [--mtp K] [--host-mib N] [--nvme-mib N] [--generate N]
//       [--max-context N] [--evict] [--store DIR]
//   --evict: four distinct prompts twice round through a small Host tier. --nvme-mib N: the NVMe tier
//   (--host-mib 0 with it: every checkpoint on disk). --store DIR: restart survival through the
//   context store (restart_check below).
//
// TOKENS is @FILE (ids separated by commas or whitespace) or a comma-separated id list.

#include "core/vram_budget.h"
#include "ninfer/engine.h"
#include "ninfer/types.h"

#include <cuda_runtime.h>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef _WIN32
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    include <windows.h>
#endif
#include <thread>

#ifndef NINFER_SOURCE_DIR
#    define NINFER_SOURCE_DIR "."
#endif

using namespace ninfer;

namespace {

std::vector<TokenId> parse_tokens(const std::string& argument) {
    std::string text = argument;
    if (!text.empty() && text.front() == '@') {
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

double available_ram_gib() {
#ifdef _WIN32
    MEMORYSTATUSEX status{};
    status.dwLength = sizeof(status);
    if (GlobalMemoryStatusEx(&status)) { return static_cast<double>(status.ullAvailPhys) / (1ULL << 30); }
#endif
    return 0.0;
}

// Between Engines: says how much host RAM came back after one was destroyed, waiting up to a minute
// for it to return near `before`.
void wait_for_ram(double before) {
    const auto start = std::chrono::steady_clock::now();
    double now       = available_ram_gib();
    while (now < before - 1.0 && std::chrono::steady_clock::now() - start < std::chrono::seconds(60)) {
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        now = available_ram_gib();
    }
    std::printf("available RAM %.1f GiB (%.1f before the Engine) after %.1f s\n", now, before,
                std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count());
}

EngineOptions base_options(const std::string& artifact, std::uint32_t max_context) {
    EngineOptions options;
    options.artifact_path         = artifact;
    options.max_context           = max_context;
    options.kv_capacity           = KvCapacityPolicy::explicit_capacity(max_context);
    options.max_concurrency       = 1;
    options.prefill_chunk         = 1024;
    options.kv_cache              = KvCacheStorage::Int8Group64;
    options.context_cache.enabled = false;
    options.pending_timeout_ms    = 3'600'000;
    return options;
}

struct Run {
    std::vector<TokenId> ids;
    GenerationTimings timings;
    SpeculativeStats speculative;
    std::uint32_t reused = 0; // prompt tokens the context cache supplied
    double wall = 0;
    // With a split: the time from the first generated token to token `split` and to the last one.
    std::uint32_t split = 0;
    double to_split = 0, to_last = 0;
};

// Records the Engine's live timing observations (one per committed token) of a streamed request.
class SplitSink final : public OutputSink {
public:
    explicit SplitSink(std::uint32_t split) : split_(split) {}
    void start(GenerationStart) override {}
    void progress(PromptProgress) override {}
    void publish(OutputDelta) override {}
    void timing(GenerationTimingObservation t) override {
        const double s = static_cast<double>(t.generation_elapsed_ns) * 1e-9;
        if (t.generated_tokens <= split_) { to_split = s; }
        to_last = s;
    }
    double to_split = 0, to_last = 0;

private:
    std::uint32_t split_;
};

Run generate(Engine& engine, const std::vector<TokenId>& prompt, std::uint32_t tokens, std::uint32_t split = 0) {
    RequestOptions request;
    request.execution.requested_output_tokens = tokens;
    request.execution.allow_prefix_reuse      = false;
    request.execution.sampling.temperature    = 0.0F;
    request.stop.include_model_defaults       = false;
    request.output.raw                        = true;
    const auto start = std::chrono::steady_clock::now();
    GenerationResult result;
    SplitSink sink(split);
    if (split == 0) {
        result = engine.generate(engine.prepare_tokens(prompt, false), request);
    } else {
        auto handle = engine.submit(engine.prepare_tokens(prompt, false), request, OutputConsumerMode::Streaming,
                                    GenerationObservationOptions{.live_timings = true});
        result      = handle.wait(&sink);
    }
    Run out;
    out.split    = split;
    out.to_split = sink.to_split;
    out.to_last  = sink.to_last;
    out.ids         = std::move(result.generated_token_ids);
    out.timings     = result.timings;
    out.speculative = result.speculative;
    out.reused      = result.reused_prompt_tokens;
    out.wall    = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    return out;
}

// Cancels its request once `after` tokens have been committed (0: never).
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

// A greedy request that may reuse the context cache, optionally cancelled after `cancel_after`
// committed tokens.
Run generate_cached(Engine& engine, const std::vector<TokenId>& prompt, std::uint32_t tokens,
                    std::uint32_t cancel_after = 0) {
    RequestOptions request;
    request.execution.requested_output_tokens = tokens;
    request.execution.allow_prefix_reuse      = true;
    request.execution.sampling.temperature    = 0.0F;
    request.stop.include_model_defaults       = false;
    request.output.raw                        = true;
    const auto start = std::chrono::steady_clock::now();
    CancelSink sink(cancel_after);
    // A reusable prompt identity: the context cache keys it.
    auto handle  = engine.submit(engine.prepare_tokens(prompt, true), request, OutputConsumerMode::Streaming,
                                 GenerationObservationOptions{.live_timings = true});
    auto result  = handle.wait(&sink, CancellationView([&sink] { return sink.cancelled(); }));
    Run out;
    out.ids         = std::move(result.generated_token_ids);
    out.timings     = result.timings;
    out.speculative = result.speculative;
    out.reused      = result.reused_prompt_tokens;
    out.wall        = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    return out;
}

void print_speculation(const char* label, const SpeculativeStats& s) {
    if (!s.enabled) { return; }
    std::string per;
    for (std::size_t j = 0; j < s.accepted_per_position.size(); ++j) {
        char item[48];
        std::snprintf(item, sizeof(item), "%s%.3f", j ? " " : "",
                      s.rounds ? static_cast<double>(s.accepted_per_position[j]) / static_cast<double>(s.rounds) : 0.0);
        per += item;
    }
    std::printf("%s: mtp rounds %llu, plain %llu, drafted %llu, accepted %llu (%.1f%%), %.3f tokens per round; "
                "accepted at position 1.. [%s]\n",
                label, static_cast<unsigned long long>(s.rounds), static_cast<unsigned long long>(s.fallback_steps),
                static_cast<unsigned long long>(s.drafted_tokens), static_cast<unsigned long long>(s.accepted_tokens),
                s.drafted_tokens ? 100.0 * static_cast<double>(s.accepted_tokens) / static_cast<double>(s.drafted_tokens) : 0.0,
                s.rounds ? 1.0 + static_cast<double>(s.accepted_tokens) / static_cast<double>(s.rounds) : 0.0, per.c_str());
}

void print_run(const char* label, const Run& run, std::size_t prompt) {
    const double decode = run.ids.size() > 1 && run.timings.decode_seconds > 0
                              ? static_cast<double>(run.ids.size() - 1) / run.timings.decode_seconds
                              : 0.0;
    // Wall-clock decode: N - 1 tokens over the request's wall time after its prefill, which counts
    // every host step between rounds (commit, the expert cache's update, n-gram reads).
    const double after_prefill = run.wall - run.timings.prefill_seconds;
    const double wall_decode   = run.ids.size() > 1 && after_prefill > 0
                                     ? static_cast<double>(run.ids.size() - 1) / after_prefill
                                     : 0.0;
    // FNV-1a over the generated ids: equal hashes across runs of one prompt mean equal outputs.
    std::uint64_t hash = 1469598103934665603ULL;
    for (const TokenId id : run.ids) {
        hash ^= static_cast<std::uint32_t>(id);
        hash *= 1099511628211ULL;
    }
    std::printf("%s: %zu prompt tokens, %zu generated; prefill %.2f s (%.1f tok/s), decode %.2f s (%.2f tok/s), "
                "after prefill %.2f s (%.2f tok/s), wall %.2f s, ids %016llx\n",
                label, prompt, run.ids.size(), run.timings.prefill_seconds,
                run.timings.prefill_seconds > 0 ? static_cast<double>(prompt) / run.timings.prefill_seconds : 0.0,
                run.timings.decode_seconds, decode, after_prefill, wall_decode, run.wall,
                static_cast<unsigned long long>(hash));
    if (run.split > 1 && run.ids.size() > run.split && run.to_split > 0 && run.to_last > run.to_split) {
        std::printf("%s: split at %u: tokens 2-%u %.2f tok/s, tokens %u-%zu %.2f tok/s\n", label, run.split,
                    run.split, static_cast<double>(run.split - 1) / run.to_split, run.split + 1, run.ids.size(),
                    static_cast<double>(run.ids.size() - run.split) / (run.to_last - run.to_split));
    }
    std::fflush(stdout);
}

bool same(const char* label, const std::vector<TokenId>& reference, const std::vector<TokenId>& ids);

// Reads this process's video memory (free memory and, under WDDM, its OS budget and usage) every
// millisecond on its own thread, as the Program's VRAM monitor does, and records the largest usage
// over the budget.
class VramSampler {
public:
    VramSampler() : source_(open_vram_budget_source(0)), thread_([this] { run(); }) {}
    ~VramSampler() {
        stop_ = true;
        thread_.join();
    }
    VramSampler(const VramSampler&)            = delete;
    VramSampler& operator=(const VramSampler&) = delete;
    // Bytes of usage over the budget at the worst sample so far (0: never over), and the samples.
    [[nodiscard]] std::uint64_t worst_over() const { return worst_.load(); }
    [[nodiscard]] std::uint64_t samples() const { return samples_.load(); }
    [[nodiscard]] bool has_budget() const { return has_budget_.load(); }
    // This process's device memory now (the OS budget's usage; free memory's complement without one).
    [[nodiscard]] static std::uint64_t usage() {
        const VramSnapshot s = open_vram_budget_source(0)->query(); // its own source: not shared with the thread
        return s.has_budget ? s.local_usage : s.device_total - s.device_free;
    }

private:
    void run() {
        if (cudaSetDevice(0) != cudaSuccess) { return; }
        while (!stop_.load()) {
            VramSnapshot s;
            try {
                s = source_->query();
            } catch (...) {
                // Like the production monitor: a failed read is no sample, and sampling stops.
                has_budget_ = false;
                return;
            }
            has_budget_ = s.has_budget;
            if (s.has_budget && s.local_usage > s.local_budget) {
                const std::uint64_t over = s.local_usage - s.local_budget;
                if (over > worst_.load()) { worst_ = over; }
            }
            ++samples_;
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }
    std::unique_ptr<VramBudgetSource> source_;
    std::atomic<bool> stop_{false}, has_budget_{false};
    std::atomic<std::uint64_t> worst_{0}, samples_{0};
    std::thread thread_;
};

// The elastic KV pool's grow/shrink cycle on a full expert cache (the production sizing: VRAM filled
// to the headroom): a short and a long prompt alternate, so the pool grows from its 1,024-token base
// by ~0.5 GiB, more than the free memory, and shrinks back at the next admission, three times. The
// growth must take its memory from the frames before it maps it: the process stays within its OS
// budget at every sample (the VRAM sampler reads free memory throughout, as the monitor does, which
// is what turned the earlier overcommit into cuMemSetAccess CUDA_ERROR_NOT_READY), no request fails,
// greedy ids repeat exactly, and every frame a growth took comes back after the shrink.
bool kv_cycle(const std::string& artifact, const std::vector<TokenId>& prompt, const std::vector<TokenId>& long_part) {
    std::vector<TokenId> long_prompt;
    while (long_prompt.size() < 40'000) { long_prompt.insert(long_prompt.end(), long_part.begin(), long_part.end()); }
    long_prompt.resize(40'000);
    EngineOptions options                = base_options(artifact, 65'536);
    options.flash_next.save_expert_state = false;
    options.flash_next.kv_base_tokens    = 1024;
    Engine engine(options);
    VramSampler sampler;
    bool ok = true;
    std::vector<TokenId> short_ids, long_ids;
    std::uint64_t first_usage = 0, last_usage = 0;
    for (int cycle = 0; cycle < 3; ++cycle) {
        const Run small = generate(engine, prompt, 8);
        print_run("kv cycle, short prompt", small, prompt.size());
        // The pool has shrunk back and the frames have taken their memory again.
        (cycle == 0 ? first_usage : last_usage) = sampler.usage();
        const Run large = generate(engine, long_prompt, 8);
        print_run("kv cycle, long prompt (KV grown from the frames)", large, long_prompt.size());
        if (cycle == 0) {
            short_ids = small.ids;
            long_ids  = large.ids;
            ok        = ok && short_ids.size() == 8 && long_ids.size() == 8;
        } else {
            ok = same("kv cycle, short prompt again", short_ids, small.ids) && ok;
            ok = same("kv cycle, long prompt again", long_ids, large.ids) && ok;
        }
    }
    const std::uint64_t over = sampler.worst_over();
    // No sample at all means the sampler failed: fail. Without an OS budget (not WDDM) the over-budget
    // check cannot run: say so explicitly rather than reporting it as verified.
    const bool sampled = sampler.samples() > 0;
    if (!sampled) { std::printf("FAIL: kv cycle: the VRAM sampler produced no samples\n"); }
    if (sampled && !sampler.has_budget()) {
        std::printf("SKIP: kv cycle: no OS budget telemetry on this platform; the over-budget check did not run\n");
    }
    std::printf("%s: kv cycle: %llu VRAM samples, OS budget %s, worst usage over the budget %llu MiB\n",
                over == 0 ? "ok" : "FAIL", static_cast<unsigned long long>(sampler.samples()),
                sampler.has_budget() ? "read" : "unavailable", static_cast<unsigned long long>(over >> 20));
    // Every frame a growth took comes back: the device holds as much after the third shrink as after
    // the first (a leak loses a 64 MiB frame chunk per cycle).
    const bool kept = last_usage + (32ULL << 20) >= first_usage;
    std::printf("%s: kv cycle: device usage after the short prompt %llu MiB (first cycle), %llu MiB (third)\n",
                kept ? "ok" : "FAIL", static_cast<unsigned long long>(first_usage >> 20),
                static_cast<unsigned long long>(last_usage >> 20));
    return ok && sampled && over == 0 && kept;
}

bool same(const char* label, const std::vector<TokenId>& reference, const std::vector<TokenId>& ids) {
    if (ids == reference) {
        std::printf("%s: %zu ids identical\n", label, ids.size());
        return true;
    }
    std::size_t first = 0;
    while (first < ids.size() && first < reference.size() && ids[first] == reference[first]) { ++first; }
    std::printf("FAIL: %s differs from the cold run at generated token %zu\n", label, first);
    return false;
}

int bench(const std::string& artifact, const std::vector<std::string>& args) {
    std::uint32_t max_context = 33'792, tokens = 512, split = 0, mtp = 0, prefill_chunk = 1024;
    bool graph = true, state = true, print_ids = false, mtp_fixed = false, vision = false;
    double width_cost = -1.0;
    std::optional<std::uint64_t> cache;
    std::vector<std::string> jobs;
    for (std::size_t i = 0; i < args.size(); ++i) {
        const std::string& a = args[i];
        const auto value     = [&] {
            if (i + 1 >= args.size()) { throw std::invalid_argument(a + " needs a value"); }
            return std::stoull(args[++i]);
        };
        if (a == "--max-context") {
            max_context = static_cast<std::uint32_t>(value());
        } else if (a == "--generate") {
            tokens = static_cast<std::uint32_t>(value());
        } else if (a == "--expert-cache-mib") {
            cache = value() << 20U;
        } else if (a == "--split") {
            split = static_cast<std::uint32_t>(value());
        } else if (a == "--no-graph") {
            graph = false;
        } else if (a == "--no-expert-state") {
            state = false;
        } else if (a == "--print-ids") {
            print_ids = true;
        } else if (a == "--mtp") {
            mtp = static_cast<std::uint32_t>(value());
        } else if (a == "--mtp-fixed") {
            mtp_fixed = true;
        } else if (a == "--vision") {
            vision = true;
        } else if (a == "--width-cost" && i + 1 < args.size()) {
            width_cost = std::stod(args[++i]);
        } else if (a == "--prefill-chunk") {
            prefill_chunk = static_cast<std::uint32_t>(value());
        } else {
            jobs.push_back(a);
        }
    }
    EngineOptions options                  = base_options(artifact, max_context);
    options.prefill_chunk                  = prefill_chunk;
    options.use_cuda_graph                 = graph;
    options.flash_next.expert_cache_bytes  = cache;
    options.flash_next.save_expert_state   = state;
    // Image input enabled (the artifact's vision tower loaded): text jobs run as a --vision server's.
    options.enable_vision = vision;
    if (mtp > 0) {
        options.speculative.backend         = SpeculativeBackend::Mtp;
        options.speculative.draft_tokens    = mtp;
        options.flash_next.mtp_fixed_drafts = mtp_fixed;
        if (width_cost >= 0.0) { options.flash_next.mtp_width_cost = width_cost; }
    }
    const auto load = std::chrono::steady_clock::now();
    Engine engine(options);
    std::printf("engine loaded in %.1f s\n", std::chrono::duration<double>(std::chrono::steady_clock::now() - load).count());
    for (const std::string& job : jobs) {
        const auto prompt = parse_tokens(job);
        const Run run     = generate(engine, prompt, tokens, split);
        print_run(job.c_str(), run, prompt.size());
        print_speculation(job.c_str(), run.speculative);
        if (print_ids) {
            std::string ids;
            for (const TokenId id : run.ids) { ids += (ids.empty() ? "" : ",") + std::to_string(id); }
            std::printf("%s: ids: %s\n", job.c_str(), ids.c_str());
        }
    }
    return 0;
}

// Greedy MTP output equals plain greedy output: speculation is lossless. Plain first (one Engine),
// then one Engine per depth; every job's ids must be identical, and each MTP run must have
// verified drafts (so acceptance, rejection and the commit paths all ran).
int mtp_exact(const std::string& artifact, const std::vector<std::string>& args) {
    std::uint32_t max_context = 33'792, tokens = 256;
    bool fixed = false;
    std::vector<std::uint32_t> depths;
    std::vector<std::string> jobs;
    for (std::size_t i = 0; i < args.size(); ++i) {
        const std::string& a = args[i];
        if (a == "--max-context" && i + 1 < args.size()) {
            max_context = static_cast<std::uint32_t>(std::stoul(args[++i]));
        } else if (a == "--generate" && i + 1 < args.size()) {
            tokens = static_cast<std::uint32_t>(std::stoul(args[++i]));
        } else if (a == "--mtp-fixed") {
            fixed = true;
        } else if (depths.empty() && a.find_first_not_of("0123456789,") == std::string::npos) {
            std::stringstream list(a);
            std::string item;
            while (std::getline(list, item, ',')) { depths.push_back(static_cast<std::uint32_t>(std::stoul(item))); }
        } else {
            jobs.push_back(a);
        }
    }
    if (depths.empty() || jobs.empty()) { throw std::invalid_argument("--mtp-exact needs depths and token files"); }
    std::vector<std::vector<TokenId>> plain;
    const double ram_before = available_ram_gib();
    {
        EngineOptions options                = base_options(artifact, max_context);
        options.flash_next.save_expert_state = false;
        Engine engine(options);
        for (const std::string& job : jobs) {
            const auto prompt = parse_tokens(job);
            const Run run     = generate(engine, prompt, tokens);
            print_run(("plain " + job).c_str(), run, prompt.size());
            plain.push_back(run.ids);
        }
    }
    bool ok = true;
    for (const std::uint32_t depth : depths) {
        wait_for_ram(ram_before);
        EngineOptions options                = base_options(artifact, max_context);
        options.flash_next.save_expert_state = false;
        options.speculative.backend          = SpeculativeBackend::Mtp;
        options.speculative.draft_tokens     = depth;
        options.flash_next.mtp_fixed_drafts  = fixed;
        Engine engine(options);
        for (std::size_t j = 0; j < jobs.size(); ++j) {
            const auto prompt       = parse_tokens(jobs[j]);
            const Run run           = generate(engine, prompt, tokens);
            const std::string label = "mtp K=" + std::to_string(depth) + " " + jobs[j];
            print_run(label.c_str(), run, prompt.size());
            print_speculation(label.c_str(), run.speculative);
            ok = same(label.c_str(), plain[j], run.ids) && ok;
            if (run.speculative.rounds == 0 || run.speculative.accepted_tokens == run.speculative.drafted_tokens) {
                std::printf("FAIL: %s did not exercise both acceptance and rejection\n", label.c_str());
                ok = false;
            }
        }
    }
    std::printf(ok ? "qwen4_exp mtp exactness passed\n" : "FAIL: qwen4_exp mtp exactness\n");
    return ok ? 0 : 1;
}

// The context cache (docs/maintainer/flash-next-context-cache.md) against cold runs. One Engine with
// the cache runs a conversation; then one Engine without it runs the same prompts cold. Each cached
// request's greedy ids are compared with its cold run, and its reused tokens with what the cache
// should have supplied. P is 4,096 tokens, one prefill call, and S 600 (a wide call too), so a
// resume at |P| runs exactly the calls the cold prompt runs:
//   a  P                 cold, captures the prompt's end and the endpoint          must equal cold
//   b  P again           exact hit: sampled from the stored logits (reused = |P|)  must equal cold
//   c  P + S             resumes at |P| (the prompt's checkpoint)                  must equal cold
//   d  P + S + c + T     resumes from c's endpoint (|P + S + c| - 1)               reported
//   e  A, then A + S     A = 5,120 tokens: a resume where cold prefill has no call boundary, reported
//   f  P + S, cancelled after 4 tokens (after its capture); then P + S again from the root (the
//      Engine abandons a cancelled request's continuation)                         must equal cold
// The prefill's numerics depend on where its calls start (greedy ids differ between
// --prefill-chunk 64, 96, 100 and 1024 on a 300-token prompt, master 04901788), so a resume where
// cold prefill has no call boundary, and one from decode-computed state (d), may differ from the
// cold run; those are reported with the first differing token, not failed. --mtp K runs both
// Engines with MTP.
//   ninfer_qwen4_exp_engine_real_test --cache [--mtp K] [--host-mib N] [--nvme-mib N] [--generate N]
//       (--nvme-mib: the NVMe tier, the Host tier its RAM cache; --host-mib 0 with it: disk only)
int cache_check(const std::string& artifact, const std::vector<std::string>& args,
                const std::vector<TokenId>& long_prompt, const std::vector<TokenId>& suffix,
                const std::vector<TokenId>& tail);

// Restart survival through the context store (--cache ... --store DIR, an empty directory):
//   E1 (the configured tiers, the store): a: P cold, c: P + S (resumes at |P|); shut down, which
//      writes the conversation to the store.
//   E2 (same): restores it at start-up; c': P + S again must be an exact hit (reused |P + S|) with
//      c's ids; x: P + S + U resumes at |P + S| from the restored checkpoint.
//   Then one byte of each of the store's largest files is damaged.
//   E3 (the RAM tier alone, the damaged store): restores nothing; a, c and x again, now from RAM: a
//      and c must give E1's ids, and x must give E2's (restored from the store, and from NVMe when E2
//      had --host-mib 0 --nvme-mib N, against restored from RAM).
template <class Options>
int restart_check(const Options& options, const std::filesystem::path& store, const std::vector<TokenId>& long_prompt,
                  const std::vector<TokenId>& suffix, std::uint32_t tokens) {
    std::error_code error;
    if (std::filesystem::exists(store, error) && !std::filesystem::is_empty(store, error)) {
        throw std::invalid_argument("--store needs an empty (or new) directory: " + store.string());
    }
    const auto concat = [](std::initializer_list<const std::vector<TokenId>*> parts) {
        std::vector<TokenId> out;
        for (const auto* part : parts) { out.insert(out.end(), part->begin(), part->end()); }
        return out;
    };
    const std::vector<TokenId> P(long_prompt.begin(), long_prompt.begin() + 4'096);
    const std::vector<TokenId> PS = concat({&P, &suffix});
    const std::vector<TokenId> U(long_prompt.begin() + 4'096, long_prompt.begin() + 4'096 + 700);
    const std::vector<TokenId> PSU = concat({&PS, &U});
    const auto stored = [&](bool tiers) {
        EngineOptions o                    = options(true);
        o.context_store.directory          = store;
        o.context_store.idle_persist       = std::chrono::seconds(0); // written at shutdown only
        if (!tiers) {
            o.context_cache.host_capacity_bytes = std::nullopt;
            o.flash_next.context_nvme_bytes     = 0;
        }
        return o;
    };
    const auto step = [&](Engine& engine, const char* label, const std::vector<TokenId>& prompt, std::uint32_t reuse) {
        Run r = generate_cached(engine, prompt, tokens);
        print_run(label, r, prompt.size());
        std::printf("%s: reused %u of %zu prompt tokens (expected %u), first token %.3f s\n", label, r.reused, prompt.size(),
                    reuse, r.timings.first_token_seconds);
        return r;
    };
    bool ok                 = true;
    const auto expect_reuse = [&](const char* label, const Run& r, std::uint32_t reuse) {
        if (r.reused != reuse) {
            std::printf("FAIL: %s reused %u prompt tokens, expected %u\n", label, r.reused, reuse);
            ok = false;
        }
    };
    const double ram_before = available_ram_gib();
    Run a1, c1, c2, x2;
    {
        Engine engine(stored(true));
        a1 = step(engine, "E1 a: P (cold)", P, 0);
        c1 = step(engine, "E1 c: P + S", PS, static_cast<std::uint32_t>(P.size()));
        expect_reuse("E1 c", c1, static_cast<std::uint32_t>(P.size()));
    }
    std::vector<std::filesystem::path> largest; // every file of the largest size
    std::uintmax_t largest_bytes = 0;
    const auto scan_store        = [&](const char* when) {
        std::size_t files = 0;
        largest.clear();
        largest_bytes = 0;
        for (const auto& entry : std::filesystem::recursive_directory_iterator(store)) {
            if (!entry.is_regular_file()) { continue; }
            ++files;
            if (entry.file_size() > largest_bytes) {
                largest_bytes = entry.file_size();
                largest.clear();
            }
            if (entry.file_size() == largest_bytes) { largest.push_back(entry.path()); }
        }
        std::printf("store after %s: %zu files, the largest %.1f MiB\n", when, files,
                    static_cast<double>(largest_bytes) / (1 << 20));
        return files;
    };
    if (scan_store("E1") == 0) {
        std::printf("FAIL: shutdown wrote nothing to the store\n");
        return 1;
    }
    wait_for_ram(ram_before);
    {
        const auto start = std::chrono::steady_clock::now();
        Engine engine(stored(true));
        const auto stats = engine.runtime_stats();
        std::printf("E2: started in %.1f s, restored %llu session(s), %.0f MiB, in %.3f s\n",
                    std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count(),
                    static_cast<unsigned long long>(stats.context_store_restored),
                    static_cast<double>(stats.context_store_restored_bytes) / (1 << 20), stats.context_store_restore_seconds);
        if (stats.context_store_restored == 0) {
            std::printf("FAIL: E2 restored nothing from the store\n");
            ok = false;
        }
        c2 = step(engine, "E2 c': P + S again (restored)", PS, static_cast<std::uint32_t>(PS.size()));
        expect_reuse("E2 c'", c2, static_cast<std::uint32_t>(PS.size()));
        ok = same("E2 c' against E1 c", c1.ids, c2.ids) && ok;
        x2 = step(engine, "E2 x: P + S + U (resumes at |P + S|)", PSU, static_cast<std::uint32_t>(PS.size()));
        expect_reuse("E2 x", x2, static_cast<std::uint32_t>(PS.size()));
    }
    // E2's shutdown wrote the image again (the conversation grew) and the store keeps the old chunks, so
    // one "largest file" may belong to no live image. A 280 MiB image holds several full-size chunks:
    // damage one byte in every file of the largest size, so the live image is certain to be hit.
    scan_store("E2");
    for (const auto& path : largest) {
        // One damaged byte: the store must refuse the image (a checksum per chunk), never restore it.
        std::fstream file(path, std::ios::in | std::ios::out | std::ios::binary);
        file.seekg(static_cast<std::streamoff>(largest_bytes / 2));
        char byte = 0;
        file.read(&byte, 1);
        byte = static_cast<char>(byte ^ 0x5A);
        file.seekp(static_cast<std::streamoff>(largest_bytes / 2));
        file.write(&byte, 1);
    }
    wait_for_ram(ram_before);
    {
        Engine engine(stored(false));
        const auto stats = engine.runtime_stats();
        std::printf("E3 (RAM tier, damaged store): restored %llu session(s)\n",
                    static_cast<unsigned long long>(stats.context_store_restored));
        if (stats.context_store_restored != 0) {
            std::printf("FAIL: a damaged store image was restored\n");
            ok = false;
        }
        const Run a3 = step(engine, "E3 a: P (cold)", P, 0);
        expect_reuse("E3 a", a3, 0);
        ok = same("E3 a against E1 a", a1.ids, a3.ids) && ok;
        const Run c3 = step(engine, "E3 c: P + S", PS, static_cast<std::uint32_t>(P.size()));
        ok = same("E3 c against E1 c", c1.ids, c3.ids) && ok;
        const Run x3 = step(engine, "E3 x: P + S + U (from RAM)", PSU, static_cast<std::uint32_t>(PS.size()));
        expect_reuse("E3 x", x3, static_cast<std::uint32_t>(PS.size()));
        ok = same("E2 x (restored from the store) against E3 x (from RAM)", x3.ids, x2.ids) && ok;
    }
    std::printf(ok ? "qwen4_exp context store restart checks passed\n" : "FAIL: qwen4_exp context store restart checks\n");
    return ok ? 0 : 1;
}

int cache_check(const std::string& artifact, const std::vector<std::string>& args,
                const std::vector<TokenId>& long_prompt, const std::vector<TokenId>& suffix,
                const std::vector<TokenId>& tail) {
    std::uint32_t tokens = 32, mtp = 0, max_context = 16'384;
    std::optional<std::uint64_t> host;
    std::uint64_t nvme = 0;
    std::filesystem::path store;
    bool evict = false;
    for (std::size_t i = 0; i < args.size(); ++i) {
        if (args[i] == "--evict") {
            evict = true;
            continue;
        }
        if (args[i] == "--store" && i + 1 < args.size()) {
            store = args[++i];
            continue;
        }
        const auto value = [&] {
            if (i + 1 >= args.size()) { throw std::invalid_argument(args[i] + " needs a value"); }
            return std::stoull(args[++i]);
        };
        if (args[i] == "--mtp") {
            mtp = static_cast<std::uint32_t>(value());
        } else if (args[i] == "--host-mib") {
            host = value() << 20U;
        } else if (args[i] == "--nvme-mib") {
            nvme = value() << 20U;
        } else if (args[i] == "--generate") {
            tokens = static_cast<std::uint32_t>(value());
        } else if (args[i] == "--max-context") {
            max_context = static_cast<std::uint32_t>(value());
        } else {
            throw std::invalid_argument("unknown --cache argument " + args[i]);
        }
    }
    const auto options = [&](bool cache) {
        EngineOptions o                      = base_options(artifact, max_context);
        o.flash_next.save_expert_state       = false;
        o.context_cache.enabled              = cache;
        o.context_cache.host_capacity_bytes  = cache ? host : std::optional<std::size_t>(0);
        o.flash_next.context_nvme_bytes      = cache ? nvme : 0;
        if (mtp > 0) {
            o.speculative.backend      = SpeculativeBackend::Mtp;
            o.speculative.draft_tokens = mtp;
        }
        return o;
    };
    if (!store.empty()) { return restart_check(options, store, long_prompt, suffix, tokens); }
    const auto concat = [](std::initializer_list<const std::vector<TokenId>*> parts) {
        std::vector<TokenId> out;
        for (const auto* part : parts) { out.insert(out.end(), part->begin(), part->end()); }
        return out;
    };
    if (long_prompt.size() < 5'120) { throw std::invalid_argument("the cache check needs 5,120 prompt tokens"); }
    const std::vector<TokenId> P(long_prompt.begin(), long_prompt.begin() + 4'096);
    const std::vector<TokenId> PS = concat({&P, &suffix});
    const std::vector<TokenId> A(long_prompt.begin(), long_prompt.begin() + 5'120);
    const std::vector<TokenId> AS = concat({&A, &suffix});
    std::vector<std::vector<TokenId>> prompts;
    std::vector<std::string> labels;
    std::vector<Run> cached;
    std::vector<std::uint32_t> expected;
    std::vector<bool> exact;
    const double ram_before = available_ram_gib();
    bool ok                 = true;
    if (evict) {
        // Eviction under a small Host tier (--host-mib 512: about three 4K-token checkpoints with
        // their endpoints): four distinct 4,096-token prompts, twice round. Every request must match
        // its cold run (a resume here is an exact hit or none), whatever the cache kept.
        std::vector<std::vector<TokenId>> distinct;
        for (std::size_t k = 0; k < 4; ++k) {
            std::vector<TokenId> x;
            for (std::size_t i = 0; i < 4'096; ++i) { x.push_back(long_prompt[(i + k * 1'024) % long_prompt.size()]); }
            distinct.push_back(std::move(x));
        }
        std::vector<Run> runs;
        {
            Engine engine(options(true));
            for (int round = 0; round < 2; ++round) {
                for (std::size_t k = 0; k < distinct.size(); ++k) {
                    const std::string label = "evict round " + std::to_string(round) + " prompt " + std::to_string(k);
                    runs.push_back(generate_cached(engine, distinct[k], tokens));
                    print_run(label.c_str(), runs.back(), distinct[k].size());
                    std::printf("%s: reused %u\n", label.c_str(), runs.back().reused);
                }
            }
            const MemorySummary memory = engine.memory_summary();
            std::printf("Host tier: %.0f of %.0f MiB occupied, %u checkpoints\n",
                        static_cast<double>(memory.host_context_occupied_bytes) / (1 << 20),
                        static_cast<double>(memory.host_context_capacity_bytes) / (1 << 20),
                        memory.host_state_occupied_slots);
            ok = memory.host_context_occupied_bytes <= memory.host_context_capacity_bytes && ok;
        }
        wait_for_ram(ram_before);
        Engine engine(options(false));
        for (std::size_t k = 0; k < distinct.size(); ++k) {
            const Run cold = generate(engine, distinct[k], tokens);
            for (int round = 0; round < 2; ++round) {
                const std::string label = "evict round " + std::to_string(round) + " prompt " + std::to_string(k);
                ok = same(label.c_str(), cold.ids, runs[static_cast<std::size_t>(round) * distinct.size() + k].ids) && ok;
            }
        }
        std::printf(ok ? "qwen4_exp context cache eviction checks passed\n" : "FAIL: qwen4_exp context cache eviction\n");
        return ok ? 0 : 1;
    }
    {
        Engine engine(options(true));
        const auto run = [&](const char* label, const std::vector<TokenId>& prompt, std::uint32_t reuse, bool must,
                             std::uint32_t cancel_after = 0) {
            Run r = generate_cached(engine, prompt, tokens, cancel_after);
            print_run(label, r, prompt.size());
            print_speculation(label, r.speculative);
            std::printf("%s: reused %u of %zu prompt tokens (expected %u), first token %.3f s, prefill %.3f s\n", label,
                        r.reused, prompt.size(), reuse, r.timings.first_token_seconds, r.timings.prefill_seconds);
            if (cancel_after == 0) {
                prompts.push_back(prompt);
                labels.push_back(label);
                cached.push_back(r);
                expected.push_back(reuse);
                exact.push_back(must);
            }
            return r;
        };
        const Run a = run("a: P (cold, captures)", P, 0, true);
        (void)run("b: P again (exact hit)", P, static_cast<std::uint32_t>(P.size()), true);
        const Run c = run("c: P + S (resume at P)", PS, static_cast<std::uint32_t>(P.size()), true);
        const std::vector<TokenId> Q = concat({&PS, &c.ids, &tail});
        (void)run("d: P + S + c + T (resume from c's endpoint)", Q, static_cast<std::uint32_t>(PS.size() + c.ids.size() - 1),
                  false);
        (void)run("e1: A (5,120 tokens)", A, 0, true);
        (void)run("e2: A + S (resume at 5,120)", AS, static_cast<std::uint32_t>(A.size()), false);
        // A cancelled request's continuation is abandoned by the Engine (its checkpoints, captured
        // before the cancel, are released), so the same prompt starts from the root again.
        const Run f = run("f1: P + S cancelled after 4 tokens", PS, 0, false, 4);
        std::printf("f1: %zu tokens before the cancel took effect\n", f.ids.size());
        (void)run("f2: P + S after the cancelled request", PS, 0, true);
        (void)a;
        const MemorySummary memory = engine.memory_summary();
        std::printf("Host tier: %.0f of %.0f MiB occupied, %.0f reserved, %u checkpoints, %.0f MiB of KV pages\n",
                    static_cast<double>(memory.host_context_occupied_bytes) / (1 << 20),
                    static_cast<double>(memory.host_context_capacity_bytes) / (1 << 20),
                    static_cast<double>(memory.host_context_reserved_bytes) / (1 << 20), memory.host_state_occupied_slots,
                    static_cast<double>(memory.host_kv_occupied_bytes) / (1 << 20));
        if (memory.host_context_capacity_bytes == 0 || memory.host_context_reserved_bytes != 0 ||
            memory.host_context_occupied_bytes > memory.host_context_capacity_bytes) {
            std::printf("FAIL: the Host tier's accounting is inconsistent between requests\n");
            ok = false;
        }
    }
    wait_for_ram(ram_before);
    {
        Engine engine(options(false));
        for (std::size_t i = 0; i < prompts.size(); ++i) {
            const Run cold = generate(engine, prompts[i], tokens);
            const std::string label = std::string("cold ") + labels[i];
            print_run(label.c_str(), cold, prompts[i].size());
            if (exact[i]) {
                ok = same(labels[i].c_str(), cold.ids, cached[i].ids) && ok;
            } else {
                std::size_t first = 0;
                while (first < cold.ids.size() && first < cached[i].ids.size() && cold.ids[first] == cached[i].ids[first]) {
                    ++first;
                }
                std::printf("REPORT: %s: %s\n", labels[i].c_str(),
                            cold.ids == cached[i].ids
                                ? "ids identical to the cold run"
                                : ("differs from the cold run from generated token " + std::to_string(first)).c_str());
            }
            if (cached[i].reused != expected[i]) {
                std::printf("FAIL: %s reused %u prompt tokens, expected %u\n", labels[i].c_str(), cached[i].reused,
                            expected[i]);
                ok = false;
            }
        }
    }
    std::printf(ok ? "qwen4_exp context cache checks passed\n" : "FAIL: qwen4_exp context cache checks\n");
    return ok ? 0 : 1;
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
        const std::string artifact = artifact_env;
        std::vector<std::string> args(argv + 1, argv + argc);
        if (!args.empty() && args.front() == "--bench") {
            return bench(artifact, std::vector<std::string>(args.begin() + 1, args.end()));
        }
        if (!args.empty() && args.front() == "--mtp-exact") {
            return mtp_exact(artifact, std::vector<std::string>(args.begin() + 1, args.end()));
        }
        std::string tokens_arg = "@" + std::string(NINFER_SOURCE_DIR) + "/tests/fixtures/qwen4_exp/ppl_tokens-chat.ids";
        std::uint32_t generate_tokens = 48, max_context = 8192, prompt_tokens = 300;
        bool kv_cycle_only = false, cache_mode = false;
        std::vector<std::string> cache_args;
        if (!args.empty() && args.front() == "--cache") {
            cache_mode = true;
            cache_args.assign(args.begin() + 1, args.end());
            args.clear();
        }
        for (std::size_t i = 0; i < args.size(); ++i) {
            if (args[i] == "--kv-cycle") {
                kv_cycle_only = true;
            } else if (args[i] == "--generate" && i + 1 < args.size()) {
                generate_tokens = static_cast<std::uint32_t>(std::stoul(args[++i]));
            } else if (args[i] == "--max-context" && i + 1 < args.size()) {
                max_context = static_cast<std::uint32_t>(std::stoul(args[++i]));
            } else if (args[i] == "--prompt" && i + 1 < args.size()) {
                prompt_tokens = static_cast<std::uint32_t>(std::stoul(args[++i]));
            } else {
                tokens_arg = args[i];
            }
        }
        auto prompt = parse_tokens(tokens_arg);
        if (prompt.size() > prompt_tokens) { prompt.resize(prompt_tokens); }
        if (prompt.empty()) { throw std::invalid_argument("empty prompt"); }
        // A long prompt (every fixture text, ~5,100 tokens): past QSA's 2,048-token selection
        // budget, so attention drops blocks, and past one KV chunk per plane, so the small-cache
        // Engine's 64-token KV base must grow from the expert frames.
        std::vector<TokenId> long_prompt;
        for (const char* name : {"ppl_tokens-code", "ppl_tokens-doc", "ppl_tokens-chat", "infernix_texts-code",
                                 "infernix_texts-doc", "infernix_texts-chat"}) {
            const auto part = parse_tokens("@" + std::string(NINFER_SOURCE_DIR) + "/tests/fixtures/qwen4_exp/" + name + ".ids");
            long_prompt.insert(long_prompt.end(), part.begin(), part.end());
        }
        if (cache_mode) {
            const auto fixture = [&](const char* name, std::size_t count) {
                auto ids = parse_tokens("@" + std::string(NINFER_SOURCE_DIR) + "/tests/fixtures/qwen4_exp/" + name + ".ids");
                ids.resize(std::min(ids.size(), count));
                return ids;
            };
            return cache_check(artifact, cache_args, long_prompt, fixture("ppl_tokens-doc", 600),
                               fixture("ppl_tokens-code", 200));
        }
        if (kv_cycle_only) {
            const bool cycled = kv_cycle(artifact, prompt, long_prompt);
            std::printf(cycled ? "qwen4_exp kv cycle passed\n" : "FAIL: qwen4_exp kv cycle\n");
            return cycled ? 0 : 1;
        }
        const std::uint32_t long_tokens = 24;
        std::vector<TokenId> long_cold;
        const auto state_path = std::filesystem::temp_directory_path() / "ninfer-qwen4exp-test.expert-state";
        std::filesystem::remove(state_path);
        bool ok = true;

        // 2 first: the refusal costs no load. More reserve than the machine has must refuse before
        // anything is pinned, naming the shortfall.
        {
            EngineOptions options                 = base_options(artifact, max_context);
            options.flash_next.ram_headroom_bytes = 1ULL << 40; // 1 TiB
            try {
                Engine refused(options);
                std::printf("FAIL: an Engine keeping 1 TiB of RAM free started\n");
                ok = false;
            } catch (const std::exception& error) {
                const std::string what = error.what();
                const bool clean = what.find("cannot start") != std::string::npos && what.find("short") != std::string::npos;
                std::printf("%s: RAM ledger refusal: %s\n", clean ? "ok" : "FAIL", what.c_str());
                ok = ok && clean;
            }
        }

        const double ram_before = available_ram_gib();
        std::vector<TokenId> cold;
        {
            EngineOptions options                = base_options(artifact, max_context);
            options.flash_next.expert_state_path = state_path;
            Engine engine(options);
            const Run first = generate(engine, prompt, generate_tokens);
            print_run("cold cache", first, prompt.size());
            cold = first.ids;
            ok   = ok && cold.size() == generate_tokens;
            const Run again = generate(engine, prompt, generate_tokens);
            print_run("warm cache (same Engine)", again, prompt.size());
            ok = same("warm cache (same Engine)", cold, again.ids) && ok;
            const Run long_run = generate(engine, long_prompt, long_tokens);
            print_run("long prompt", long_run, long_prompt.size());
            long_cold = long_run.ids;
            ok        = ok && long_cold.size() == long_tokens;
        } // the Engine saves the expert state at shutdown
        wait_for_ram(ram_before);
        if (!std::filesystem::exists(state_path)) {
            std::printf("FAIL: the expert state was not saved at shutdown\n");
            ok = false;
        }
        {
            EngineOptions options                = base_options(artifact, max_context);
            options.flash_next.expert_state_path = state_path;
            Engine engine(options);
            const Run run = generate(engine, prompt, generate_tokens);
            print_run("warm start (saved state)", run, prompt.size());
            ok = same("warm start (saved state)", cold, run.ids) && ok;
        }
        wait_for_ram(ram_before);
        {
            EngineOptions options                 = base_options(artifact, max_context);
            options.flash_next.save_expert_state  = false;
            options.flash_next.expert_cache_bytes = 1ULL << 30;
            options.flash_next.kv_base_tokens     = 64;
            options.use_cuda_graph                = false;
            Engine engine(options);
            const Run run = generate(engine, prompt, generate_tokens);
            print_run("small cache (1 GiB, evicting), elastic KV, no CUDA graph", run, prompt.size());
            ok = same("small cache (1 GiB, evicting), elastic KV, no CUDA graph", cold, run.ids) && ok;
            const Run graphless = generate(engine, prompt, generate_tokens);
            ok = same("small cache, second request", cold, graphless.ids) && ok;
            const Run long_run = generate(engine, long_prompt, long_tokens);
            print_run("long prompt, small cache, KV grown", long_run, long_prompt.size());
            ok = same("long prompt, small cache, KV grown", long_cold, long_run.ids) && ok;
        }
        wait_for_ram(ram_before);
        ok = kv_cycle(artifact, prompt, long_prompt) && ok;
        std::filesystem::remove(state_path);
        std::printf(ok ? "qwen4_exp engine checks passed\n" : "FAIL: qwen4_exp engine checks\n");
        return ok ? 0 : 1;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL: %s\n", error.what());
        return 1;
    }
}
