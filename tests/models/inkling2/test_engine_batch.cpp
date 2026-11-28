// Written for NInfer-3090.
//
// Batched decode through the public Engine (Inkling2Core::worker_batched, max_concurrency > 1) on the real artifact:
// - requests submitted together (staggered, more than the slots, so some wait) give each the output it gives alone
//   on a one-request Engine: greedy ones token for token, a seeded sampled one too, a thinking budget's close and a
//   stop string included;
// - streaming gives the aggregate's text; cancellation ends one request and the others finish;
// - prefix reuse keeps one snapshot per slot: a repeated long prompt resumes (in its slot) and decodes what it did cold;
// - the runtime stats report the slots as lanes, and the counters add up;
// - with max_concurrency 8 at a context that cannot fit 8 times, the Engine fits a smaller context per slot and
//   reports it (options().max_context and the load note).
// Skips without NINFER_TEST_INKLING2_ARTIFACT. NINFER_INKLING2_BATCH_FIT=0 skips the fitting case (a slow startup).

#include "ninfer/engine.h"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <future>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

int g_failures = 0;
void check(bool ok, const std::string& what) {
    if (!ok) {
        std::fprintf(stderr, "FAILED: %s\n", what.c_str());
        ++g_failures;
    }
}

ninfer::PromptInput user_turn(const std::string& text) {
    ninfer::PromptInput input;
    input.messages.push_back({.role = ninfer::ChatRole::User, .parts = {{.kind = ninfer::MessagePartKind::Text, .text = text}}});
    return input;
}

ninfer::RequestOptions greedy(std::uint32_t tokens) {
    ninfer::RequestOptions options;
    options.execution.sampling.temperature    = 0.0F;
    options.execution.requested_output_tokens = tokens;
    return options;
}

struct Case {
    std::string prompt;
    ninfer::RequestOptions options;
};

class Collect final : public ninfer::OutputSink {
public:
    void start(ninfer::GenerationStart) override {}
    void progress(ninfer::PromptProgress) override {}
    void timing(ninfer::GenerationTimingObservation) override {}
    void publish(ninfer::OutputDelta d) override { (d.channel == ninfer::OutputChannel::Content ? content : reasoning) += d.text; }
    std::string content, reasoning;
};

std::vector<Case> cases() {
    std::vector<Case> out;
    const char* questions[] = {"Name three primary colours.", "What is 17 times 23? Answer briefly.",
                               "Write a haiku about the sea.", "List the planets of the solar system.",
                               "Explain what a hash map is in two sentences.", "Translate 'good morning' into French and German."};
    for (std::size_t i = 0; i < std::size(questions); ++i) { out.push_back({questions[i], greedy(static_cast<std::uint32_t>(40 + 12 * i))}); }
    // A long prompt (prefix reuse: past one prefill chunk, as reuse is chunk-grained), a seeded sample, a thinking
    // budget and a stop string.
    std::string facts = "Here is a list of facts.";
    for (int i = 0; i < 180; ++i) { facts += " Fact " + std::to_string(i) + ": the number " + std::to_string(i * 7) + " is noted."; }
    out.push_back({facts + " What is fact 3?", greedy(32)});
    auto sampled                           = greedy(48);
    sampled.execution.sampling.temperature = 1.0F;
    sampled.execution.sampling.seed        = 1234;
    out.push_back({"Tell me a short story about a robot.", sampled});
    auto budget                      = greedy(40);
    budget.execution.thinking.budget = 6;
    out.push_back({"Why is the sky blue?", budget});
    auto stop = greedy(64);
    stop.stop.strings.push_back({.text = "colour", .channel = ninfer::OutputChannel::Reasoning});
    out.push_back({"Name three primary colours.", stop});
    return out;
}

ninfer::EngineOptions base_options(const char* artifact) {
    ninfer::EngineOptions options;
    options.artifact_path = artifact;
    options.max_context   = 4096;
    return options;
}

} // namespace

int main() {
    const char* artifact = std::getenv("NINFER_TEST_INKLING2_ARTIFACT");
    if (artifact == nullptr) {
        std::printf("SKIP: set NINFER_TEST_INKLING2_ARTIFACT\n");
        return 77;
    }
    try {
        const auto all = cases();
        // Each alone, one request at a time.
        std::vector<ninfer::GenerationResult> alone;
        {
            ninfer::Engine engine(base_options(artifact));
            for (const Case& c : all) {
                auto o                         = c.options;
                o.execution.allow_prefix_reuse = false;
                alone.push_back(engine.generate(engine.prepare(user_turn(c.prompt)), o));
            }
        }
        std::printf("solo runs done\n");
        {
            // Adapters are one Program-wide, per-request selection that batched decode cannot apply per column: a LoRA or
            // a graft with --max-concurrency > 1 is refused before anything loads, naming both options.
            const auto refused = [&](const ninfer::EngineOptions& o) {
                try {
                    ninfer::Engine engine(o);
                } catch (const std::invalid_argument& e) {
                    const std::string what = e.what();
                    return what.find("--lora") != std::string::npos && what.find("--graft") != std::string::npos &&
                           what.find("--max-concurrency 1") != std::string::npos;
                }
                return false;
            };
            auto with_lora = base_options(artifact), with_graft = base_options(artifact);
            with_lora.max_concurrency = with_graft.max_concurrency = 2;
            with_lora.loras.push_back({"l", "missing-adapter"});
            with_graft.grafts.push_back({"g", "missing.safetensors"});
            check(refused(with_lora) && refused(with_graft), "a LoRA or a graft with --max-concurrency 2 is refused");
        }
        {
            auto options            = base_options(artifact);
            options.max_concurrency = 4;
            ninfer::Engine engine(options);
            check(engine.options().max_concurrency == 4 && engine.options().max_context == 4096 &&
                      engine.options().context_cache.device_state_slots == 4U,
                  "four slots of 4096 positions, one snapshot each");
            check(engine.options().kv_capacity.explicit_tokens == 4U * 4096U &&
                      engine.memory_summary().kv_capacity == engine.options().kv_capacity.explicit_tokens,
                  "the shared KV capacity is four rings of 4096");
            // All of them at once (a few milliseconds apart), more than the slots: the rest wait.
            std::vector<ninfer::GenerationHandle> handles;
            const auto t0 = Clock::now();
            for (const Case& c : all) {
                auto o                         = c.options;
                o.execution.allow_prefix_reuse = false;
                handles.push_back(engine.submit(engine.prepare(user_turn(c.prompt)), o, ninfer::OutputConsumerMode::Aggregate, {}));
                std::this_thread::sleep_for(std::chrono::milliseconds(30));
            }
            std::size_t tokens = 0;
            for (std::size_t i = 0; i < handles.size(); ++i) {
                const auto r = handles[i].wait(nullptr);
                tokens += r.generated_token_ids.size();
                check(r.generated_token_ids == alone[i].generated_token_ids && r.content == alone[i].content &&
                          r.reasoning == alone[i].reasoning && r.finish_reason == alone[i].finish_reason,
                      "request " + std::to_string(i) + " batched gives its solo output (" + std::to_string(r.generated_token_ids.size()) +
                          " against " + std::to_string(alone[i].generated_token_ids.size()) + " tokens)");
            }
            const double s = std::chrono::duration<double>(Clock::now() - t0).count();
            std::printf("%zu requests at 4 slots: %zu tokens in %.2f s (%.1f tok/s with prefill)\n", all.size(), tokens, s, tokens / s);
            const auto stats = engine.runtime_stats();
            check(stats.lane_count == 4 && stats.running_requests == 0 && stats.decode_row_rounds > stats.decode_rounds,
                  "four lanes, idle after, and decode rounds of several rows");

            // Streaming beside an aggregate request, and one cancelled while the others run.
            Collect sink;
            auto streamed  = engine.submit(engine.prepare(user_turn(all[0].prompt)), all[0].options, ninfer::OutputConsumerMode::Streaming, {});
            auto other     = engine.submit(engine.prepare(user_turn(all[1].prompt)), all[1].options, ninfer::OutputConsumerMode::Aggregate, {});
            std::atomic<bool> cancel{false};
            auto cancelled = std::async(std::launch::async, [&] {
                auto h = engine.submit(engine.prepare(user_turn(all[2].prompt)), greedy(2000), ninfer::OutputConsumerMode::Aggregate, {});
                return h.wait(nullptr, ninfer::CancellationView([&] { return cancel.load(); }));
            });
            const auto rs = streamed.wait(&sink);
            const auto ro = other.wait(nullptr);
            cancel        = true;
            const auto rc = cancelled.get();
            check(rs.generated_token_ids == alone[0].generated_token_ids && sink.content == rs.content && sink.reasoning == rs.reasoning,
                  "streaming batched gives the aggregate's tokens and text");
            check(ro.generated_token_ids == alone[1].generated_token_ids, "a request beside a stream gives its solo output");
            check(rc.finish_reason == ninfer::FinishReason::Cancelled && rc.generated_token_ids.size() < 2000, "cancellation in a batch");

            // Prefix reuse: the long prompt leaves a snapshot in its slot and resumes from it.
            const Case& longest = all[6];
            const auto cold     = engine.generate(engine.prepare(user_turn(longest.prompt)), longest.options);
            const auto again    = engine.generate(engine.prepare(user_turn(longest.prompt)), longest.options);
            check(again.reused_prompt_tokens > 0 && again.reused_prompt_tokens % 256 == 0 && again.generated_token_ids == alone[6].generated_token_ids &&
                      cold.generated_token_ids == alone[6].generated_token_ids,
                  "a repeated prompt resumes from its slot's snapshot (" + std::to_string(again.reused_prompt_tokens) + " reused) and decodes the same");
        }
        if (std::getenv("NINFER_INKLING2_BATCH_FIT") == nullptr || std::string(std::getenv("NINFER_INKLING2_BATCH_FIT")) != "0") {
            // A context floor under which the fit refuses to start: one step under 262144 does not fit 8 times.
            {
                auto options              = base_options(artifact);
                options.max_context       = 262144;
                options.kv_capacity       = ninfer::KvCapacityPolicy::explicit_capacity(262144);
                options.max_concurrency   = 8;
                options.fit_context_floor = 262144 - 256;
                bool refused              = false;
                try {
                    ninfer::Engine engine(options);
                } catch (const std::exception& e) { refused = std::string(e.what()).find("floor") != std::string::npos; }
                check(refused, "8 slots refuse a context floor that does not fit");
            }
            auto options              = base_options(artifact);
            options.max_context       = 262144;
            options.kv_capacity       = ninfer::KvCapacityPolicy::explicit_capacity(262144);
            options.max_concurrency   = 8;
            options.fit_context_floor = 4096;
            ninfer::Engine engine(options);
            const auto fitted = engine.options().max_context;
            std::string notes;
            for (const auto& n : engine.load_summary().notes) { notes += n + "\n"; }
            std::printf("8 slots asked 262144: %u positions each\n%s", fitted, notes.c_str());
            check(fitted < 262144 && fitted >= 4096 && fitted % 256 == 0 && notes.find("fitted") != std::string::npos,
                  "8 slots fit a smaller context each and report it");
            check(engine.options().kv_capacity.explicit_tokens == 8U * fitted && engine.memory_summary().kv_capacity == 8U * fitted,
                  "the fitted shared KV capacity is eight rings of the fitted context");
            const auto r = engine.generate(engine.prepare(user_turn(all[0].prompt)), all[0].options);
            check(r.generated_token_ids == alone[0].generated_token_ids, "the fitted Engine decodes the solo output");
        }
        if (std::getenv("NINFER_INKLING2_BATCH_FIT") == nullptr || std::string(std::getenv("NINFER_INKLING2_BATCH_FIT")) != "0") {
            // One slot with --fit-context, DSpark (its whole-context window), the vision tower resident and BF16 experts
            // (--mxfp8-experts off): the full 262,144 does not fit on a 24 or 32 GB card this way (the V100's largest is
            // 238,592; with the MXFP8 experts a V100 build holds by default it fits whole), so the search runs, with the
            // drafter sized at each candidate. The fit is a multiple of 1024 and the Engine decodes the solo output.
            const auto one_slot = [&](std::uint32_t floor) {
                auto options                                = base_options(artifact);
                options.max_context                         = 262144;
                options.kv_capacity                         = ninfer::KvCapacityPolicy::explicit_capacity(262144);
                options.fit_context_floor                   = floor;
                options.enable_vision                       = true;
                options.speculative.backend                 = ninfer::SpeculativeBackend::DSpark;
                options.speculative.draft_tokens            = 3;
                options.speculative.dspark_window           = 0;
                options.speculative.dspark_window_explicit  = true;
                options.experts_mxfp8                       = ninfer::ExpertsMxfp8::Off;
                return options;
            };
            {
                bool refused = false;
                try {
                    ninfer::Engine engine(one_slot(262144));
                } catch (const std::exception& e) { refused = std::string(e.what()).find("floor") != std::string::npos; }
                check(refused, "one slot refuses a context floor that does not fit");
            }
            ninfer::Engine engine(one_slot(16384));
            const auto fitted = engine.options().max_context;
            std::string notes;
            for (const auto& n : engine.load_summary().notes) { notes += n + "\n"; }
            std::printf("one slot, DSpark and vision, asked 262144: %u positions\n%s", fitted, notes.c_str());
            check(fitted < 262144 && fitted >= 16384 && fitted % 1024 == 0 && notes.find("fitted to the device") != std::string::npos &&
                      notes.find("drafter window whole context") != std::string::npos,
                  "one slot fits a multiple of 1024 with the drafter's whole-context window and reports it");
            const auto r = engine.generate(engine.prepare(user_turn(all[0].prompt)), all[0].options);
            check(r.generated_token_ids == alone[0].generated_token_ids, "the fitted DSpark Engine decodes the solo output");
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL: %s\n", e.what());
        return 1;
    }
    if (g_failures != 0) {
        std::fprintf(stderr, "FAIL: %d checks\n", g_failures);
        return 1;
    }
    std::printf("PASS\n");
    return 0;
}
