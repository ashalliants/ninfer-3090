// Written for NInfer-3090 (Inkling-2-Mini bring-up).
//
// The public Engine on an Inkling-2 artifact (runtime/engine/inkling2_core.h):
// - greedy output is the same through an aggregate and a streaming consumer, and from one request to the next;
//   the streamed deltas add up to the result's reasoning and content;
// - a fixed seed repeats a sampled output;
// - the output limit, a stop string and cancellation end a request with their finish reasons;
// - the vision test's image prompt (test1.jpeg) has llama.cpp's prompt length; its greedy continuation against
//   llama.cpp's is reported (the JPEG decoders differ, see the frontend test).
// Skips without NINFER_TEST_INKLING2_ARTIFACT; the image prompt also needs NINFER_INKLING2_VISION.

#include "ninfer/engine.h"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

namespace fs = std::filesystem;
using Clock  = std::chrono::steady_clock;

int g_failures = 0;
void check(bool ok, const std::string& what) {
    if (!ok) {
        std::fprintf(stderr, "FAILED: %s\n", what.c_str());
        ++g_failures;
    }
}

ninfer::PromptInput user_turn(std::vector<ninfer::MessagePart> parts) {
    ninfer::PromptInput input;
    input.messages.push_back({.role = ninfer::ChatRole::User, .parts = std::move(parts)});
    return input;
}

ninfer::MessagePart text(std::string value) { return {.kind = ninfer::MessagePartKind::Text, .text = std::move(value)}; }

ninfer::RequestOptions greedy(std::uint32_t tokens) {
    ninfer::RequestOptions options;
    options.execution.sampling.temperature = 0.0F;
    options.execution.requested_output_tokens = tokens;
    return options;
}

class Collect final : public ninfer::OutputSink {
public:
    void start(ninfer::GenerationStart s) override { started = true, prompt_tokens = s.prompt.prompt_tokens; }
    void progress(ninfer::PromptProgress p) override { processed = p.processed_prompt_tokens; }
    void timing(ninfer::GenerationTimingObservation) override {}
    void publish(ninfer::OutputDelta d) override {
        (d.channel == ninfer::OutputChannel::Content ? content : reasoning) += d.text;
        ++deltas;
    }
    bool started = false;
    std::uint32_t prompt_tokens = 0, processed = 0, deltas = 0;
    std::string content, reasoning;
};

// The prefill chunk the Engine picked, from its load note ("prefill | chunk N ...").
std::uint32_t chunk_of(const ninfer::Engine& engine) {
    for (const std::string& note : engine.load_summary().notes) {
        if (note.rfind("prefill | chunk ", 0) == 0) { return static_cast<std::uint32_t>(std::stoul(note.substr(16))); }
    }
    throw std::runtime_error("no prefill chunk note");
}

// A prompt of facts at least `tokens` long as the Engine counts it.
std::string facts_until(const ninfer::Engine& engine, std::uint32_t tokens) {
    std::string context = "Here is a list of facts.";
    for (int i = 0; engine.prepare(user_turn({text(context)})).summary().prompt_tokens < tokens; ++i) {
        context += " Fact " + std::to_string(i) + ": the number " + std::to_string(i * 7) + " is noted.";
    }
    return context;
}

std::string ids_text(const std::vector<ninfer::TokenId>& ids) {
    std::string s;
    for (const auto id : ids) { s += std::to_string(id) + " "; }
    return s;
}

} // namespace

int main() {
    const char* artifact_path = std::getenv("NINFER_TEST_INKLING2_ARTIFACT");
    if (artifact_path == nullptr) {
        std::printf("SKIP: set NINFER_TEST_INKLING2_ARTIFACT\n");
        return 77;
    }
    try {
        ninfer::EngineOptions options;
        options.artifact_path = artifact_path;
        options.max_context   = 4096;
        options.enable_vision = true;
        options.prefill_chunk_auto = true; // the widest chunk that fits, as the CLI and serve run without --prefill-chunk
        options.max_concurrency = 4; // Inkling-2 runs one: the Engine reports what it runs
        {
            ninfer::EngineOptions unbounded   = options;
            unbounded.max_pending_requests    = 0;
            try {
                ninfer::Engine refused(unbounded);
                check(false, "the shared option bounds are checked");
            } catch (const std::invalid_argument&) {}
        }
        {
            // What Inkling-2 cannot do is refused at construction, not left to fail later.
            ninfer::EngineOptions suspended = options;
            suspended.suspend.enabled       = true;
            try {
                ninfer::Engine refused(suspended);
                check(false, "model suspend is refused");
            } catch (const std::invalid_argument&) {}
        }
        {
            // An explicit --kv-dtype bf16 (or any kv_cache form) is refused: Inkling-2's rings are FP16 or FP32.
            ninfer::EngineOptions bf16 = options;
            bf16.kv_cache_explicit     = true;
            try {
                ninfer::Engine refused(bf16);
                check(false, "an explicit --kv-dtype bf16 is refused");
            } catch (const std::invalid_argument&) {}
        }
        for (const auto set : {+[](ninfer::ContextCacheOptions& c) { c.auto_host_cache = true; },
                               +[](ninfer::ContextCacheOptions& c) { c.device_state_slots = 2U; },
                               +[](ninfer::ContextCacheOptions& c) { c.host_capacity_bytes = std::size_t{1} << 30; }}) {
            ninfer::EngineOptions tiered = options;
            set(tiered.context_cache);
            try {
                ninfer::Engine refused(tiered);
                check(false, "a host-tier context cache option is refused");
            } catch (const std::invalid_argument&) {}
        }
        for (const auto set : {+[](ninfer::EngineOptions& o) { o.speculative.backend = ninfer::SpeculativeBackend::Mtp; },
                               +[](ninfer::EngineOptions& o) { o.speculative.draft_tokens = 3; },
                               +[](ninfer::EngineOptions& o) { o.relaxed_drafting = 0.5F; },
                               +[](ninfer::EngineOptions& o) { o.draft_context_rows = 64; }}) {
            ninfer::EngineOptions speculative = options;
            set(speculative);
            try {
                ninfer::Engine refused(speculative);
                check(false, "a speculative option is refused");
            } catch (const std::invalid_argument&) {}
        }
        {
            // Prefix reuse off: no snapshot, so no extra device state is reported. With FP32 rings and --prefill-math
            // unset, the prefill is the exact FP32 one too.
            ninfer::EngineOptions uncached = options;
            uncached.context_cache.enabled = false;
            uncached.kv_rings              = ninfer::KvRings::FP32;
            ninfer::Engine engine(uncached);
            check(!engine.options().context_cache.enabled && engine.options().context_cache.device_state_slots == 0U,
                  "no extra device state without prefix reuse");
            check(engine.load_summary().prefill_signature == "inkling2-fp32",
                  "--kv-dtype fp32 alone runs the exact FP32 prefill: " + engine.load_summary().prefill_signature);
        }
        {
            // Prefix reuse at --prefill-chunk 512 under the tensor-core default: every cut and snapshot is a multiple of
            // the chunk, so a resumed prompt runs a cold one's forward calls from the resume point on. A 512-token
            // prompt leaves no snapshot (a 256 one would resume [256,768) where a cold prefill runs [0,512)
            // [512,768)); a 513-token one leaves 512, from which a 1024-token prompt sharing the prefix resumes.
            ninfer::EngineOptions chunked   = options;
            chunked.enable_vision           = false;
            chunked.prefill_chunk           = 512;
            chunked.prefill_chunk_auto      = false; // a direct caller's chunk runs as set
            ninfer::Engine engine(chunked);
            check(chunk_of(engine) == 512, "--prefill-chunk 512 is the chunk that runs");
            const std::vector<ninfer::TokenId> base = engine.tokenize_text(facts_until(engine, 1100));
            check(base.size() >= 1024, "the facts tokenize to 1024 tokens or more");
            const auto prompt_of = [&](std::uint32_t tokens) {
                return std::vector<ninfer::TokenId>(base.begin(), base.begin() + std::min<std::size_t>(tokens, base.size()));
            };
            const auto run = [&](std::uint32_t tokens, bool reuse) {
                auto o                         = greedy(24);
                o.execution.allow_prefix_reuse = reuse;
                return engine.generate(engine.prepare_tokens(prompt_of(tokens)), o);
            };
            const auto cold = run(1024, false);
            (void)run(512, true);
            const auto after_512 = run(1024, true);
            check(after_512.reused_prompt_tokens == 0, "a 512-token prompt at chunk 512 leaves no snapshot: " +
                                                           std::to_string(after_512.reused_prompt_tokens) + " reused");
            (void)run(513, true);
            const auto resumed = run(1024, true);
            check(resumed.reused_prompt_tokens == 512 && resumed.computed_prefill_tokens == 512,
                  "a 1024-token prompt resumes from the 513-token prompt's snapshot at 512: " +
                      std::to_string(resumed.reused_prompt_tokens) + " reused");
            check(after_512.generated_token_ids == cold.generated_token_ids && resumed.generated_token_ids == cold.generated_token_ids,
                  "at chunk 512 a resumed prompt decodes what a cold one does");
        }
        const auto t0         = Clock::now();
        ninfer::Engine engine(options);
        std::printf("engine ready in %.1f s (%s)\n", std::chrono::duration<double>(Clock::now() - t0).count(),
                    engine.load_summary().architecture.c_str());
        check(engine.load_summary().architecture == "inkling2", "the Engine loads Inkling-2");
        check(engine.options().context_cache.enabled && engine.options().context_cache.device_state_slots == 1U &&
                  !engine.load_summary().takes_video && engine.options().max_concurrency == 1,
              "a one-snapshot context cache, no video and one request at a time is reported");
        check(engine.options().context_cache.host_capacity_bytes == std::optional<std::size_t>{0},
              "a resolved zero-byte Host tier is reported");

        const auto question = [] { return user_turn({text("Name three primary colours.")}); };
        // Aggregate, twice, and streaming.
        const ninfer::RuntimeStats before = engine.runtime_stats();
        const auto a = engine.generate(engine.prepare(question()), greedy(64));
        const auto b = engine.generate(engine.prepare(question()), greedy(64));
        Collect sink;
        ninfer::GenerationObservationOptions observe;
        observe.prompt_progress = true;
        auto handle             = engine.submit(engine.prepare(question()), greedy(64), ninfer::OutputConsumerMode::Streaming, observe);
        const auto c            = handle.wait(&sink);
        {
            // The runtime counters are the three requests' sums, and the lane is idle between requests.
            const ninfer::RuntimeStats after = engine.runtime_stats();
            std::uint64_t prompt = 0, generated = 0, computed = 0;
            for (const auto* r : {&a, &b, &c}) {
                prompt += r->prompt.prompt_tokens;
                generated += r->generated_token_ids.size();
                computed += r->computed_prefill_tokens;
            }
            check(after.prompt_tokens - before.prompt_tokens == prompt &&
                      after.generated_tokens - before.generated_tokens == generated &&
                      after.computed_prefill_tokens - before.computed_prefill_tokens == computed &&
                      after.committed_decode_tokens - before.committed_decode_tokens == generated - 3,
                  "the runtime token counters add up the requests");
            check(after.decode_rounds - before.decode_rounds == after.decode_row_rounds - before.decode_row_rounds &&
                      after.decode_rounds > before.decode_rounds && after.decode_seconds_total > before.decode_seconds_total &&
                      after.prefill_seconds_total > before.prefill_seconds_total,
                  "decode rounds and prefill and decode time are counted");
            check(after.lane_count == 1 && after.running_requests == 0 && after.waiting_requests == 0 &&
                      after.lanes[0].state == ninfer::RuntimeLaneStats::State::Idle,
                  "an idle Engine reports one idle lane and no waiting requests");
        }
        std::printf("greedy: %zu tokens, finish %d, reasoning %zu bytes, content %zu bytes, %.1f tok/s\n",
                    a.generated_token_ids.size(), static_cast<int>(a.finish_reason), a.reasoning.size(), a.content.size(),
                    (a.generated_token_ids.size() - 1) / std::max(a.timings.decode_seconds, 1e-9));
        check(a.generated_token_ids == b.generated_token_ids, "greedy repeats");
        check(a.generated_token_ids == c.generated_token_ids, "streaming gives the aggregate's tokens");
        check(sink.started && sink.prompt_tokens == a.prompt.prompt_tokens && sink.processed == a.prompt.prompt_tokens,
              "the stream starts and reports its prompt");
        check(sink.content == c.content && sink.reasoning == c.reasoning && sink.deltas > 0, "the deltas add up to the result");
        check(a.content == c.content && a.reasoning == c.reasoning, "streaming gives the aggregate's text");
        check(a.finish_reason == ninfer::FinishReason::OutputLimit || a.finish_reason == ninfer::FinishReason::StopToken,
              "greedy ends at the limit or the turn's end");
        check(!a.reasoning.empty(), "the model reasons first");

        // A fixed seed repeats a sampled output.
        auto sampled                           = greedy(32);
        sampled.execution.sampling.temperature = 1.0F;
        sampled.execution.sampling.seed        = 1234;
        const auto s1 = engine.generate(engine.prepare(question()), sampled);
        const auto s2 = engine.generate(engine.prepare(question()), sampled);
        check(s1.generated_token_ids == s2.generated_token_ids, "a seed repeats its sample");

        // The limit, a stop string, cancellation.
        const auto one = engine.generate(engine.prepare(question()), greedy(1));
        check(one.generated_token_ids.size() == 1 && one.finish_reason == ninfer::FinishReason::OutputLimit, "a one-token limit");
        auto stop = greedy(64);
        // The first word of the greedy reasoning is a stop string on that channel.
        const std::string first_word = a.reasoning.substr(0, a.reasoning.find(' '));
        stop.stop.strings.push_back({.text = first_word, .channel = ninfer::OutputChannel::Reasoning});
        const auto stopped = engine.generate(engine.prepare(question()), stop);
        check(stopped.finish_reason == ninfer::FinishReason::StopString && stopped.reasoning.empty() &&
                  stopped.matched_stop_string == first_word,
              "a stop string ends the output before it");
        const auto cancelled =
            engine.generate(engine.prepare(question()), greedy(64), nullptr, ninfer::CancellationView([] { return true; }));
        check(cancelled.finish_reason == ninfer::FinishReason::Cancelled, "cancellation");

        // A thinking budget closes the reasoning only where the close and one answer token fit the limit.
        {
            auto budget                          = greedy(8);
            budget.execution.thinking.budget     = 2;
            const auto closed                    = engine.generate(engine.prepare(question()), budget);
            check(closed.generated_token_ids.size() <= 8 && closed.thinking.applied && closed.thinking.effective_budget == 2u,
                  "a thinking budget within the limit");
            auto tight                      = greedy(4);
            tight.execution.thinking.budget = 1;
            {
                // The model's own turn opening counts against the limit too: 2 opening + 1 + 3 close + 1 answer.
                auto seven                      = greedy(7);
                seven.execution.thinking.budget = 3;
                const auto r                    = engine.generate(engine.prepare(question()), seven);
                // The close is the 4th to 6th token, and the 7th is the model's.
                check(r.generated_token_ids.size() == 7 && r.thinking.applied && r.thinking.effective_budget == 1u &&
                          r.generated_token_ids[5] == 200004,
                      "a thinking budget leaves room for an answer token: " + ids_text(r.generated_token_ids));
            }
            const auto open                 = engine.generate(engine.prepare(question()), tight);
            check(open.generated_token_ids.size() <= 4 && !open.thinking.applied && !open.thinking.effective_budget,
                  "no close past the limit");
        }
        {
            // At the context's end: a prompt of max_context - 1 tokens yields max_context - prompt + 1 = 2 tokens.
            const auto filler = engine.tokenize_text(" the");
            std::vector<ninfer::TokenId> full(static_cast<std::size_t>(options.max_context - 1), filler.at(0));
            const auto edge = engine.generate(engine.prepare_tokens(full), greedy(10));
            check(edge.generated_token_ids.size() == 2 && edge.finish_reason == ninfer::FinishReason::ContextCapacity,
                  "the last context position still yields a token: " + std::to_string(edge.generated_token_ids.size()));
        }
        check(!ninfer::artifact_takes_video(artifact_path), "discovery reads that Inkling-2 takes no video");
        {
            // Counting reports a prompt's length also when it leaves no room for a reply (generation refuses it).
            std::string text;
            for (std::uint32_t i = 0; i < options.max_context; ++i) { text += " the"; }
            const auto counted = engine.count_tokens(user_turn({{.kind = ninfer::MessagePartKind::Text, .text = text}}));
            check(counted > options.max_context, "a prompt past max_context is counted: " + std::to_string(counted));
            try {
                (void)engine.prepare(user_turn({{.kind = ninfer::MessagePartKind::Text, .text = text}}));
                check(false, "a prompt past max_context is refused for generation");
            } catch (const ninfer::RequestError&) {}
        }
        try {
            auto bad = greedy(4);
            bad.stop.token_ids.push_back(1'000'000'000);
            (void)engine.generate(engine.prepare(question()), bad);
            check(false, "a stop token outside the vocabulary is refused");
        } catch (const std::invalid_argument&) {}
        {
            // parallel_tool_calls false: the first complete call ends the request.
            ninfer::PromptInput input = user_turn({{.kind = ninfer::MessagePartKind::Text,
                                                    .text = "Get the current weather in Paris and in Tokyo. Call the tool once per city."}});
            input.options.enable_thinking = false;
            input.options.tool_jsons.push_back(
                R"({"type":"function","function":{"name":"get_weather","description":"Current weather for a city",)"
                R"("parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}})");
            auto tools = greedy(200);
            const auto free_calls = engine.generate(engine.prepare(input), tools);
            tools.tool_choice.parallel = false;
            const auto one_call = engine.generate(engine.prepare(input), tools);
            std::printf("tool calls: %zu with parallel calls, %zu without\n", free_calls.tool_calls.size(), one_call.tool_calls.size());
            check(one_call.tool_calls.size() == 1 && one_call.finish_reason == ninfer::FinishReason::StopToken,
                  "parallel false ends at the first call: " + std::to_string(one_call.tool_calls.size()));
        }
        {
            // A wrong sink is refused, and the request it was for is cancelled, not run to its limit.
            auto handle  = engine.submit(engine.prepare(question()), greedy(2000), ninfer::OutputConsumerMode::Aggregate, {});
            const auto t = Clock::now();
            Collect wrong;
            try {
                (void)handle.wait(&wrong);
                check(false, "a sink for an aggregate request is refused");
            } catch (const std::invalid_argument&) {}
            const auto next = engine.generate(engine.prepare(question()), greedy(4));
            check(next.generated_token_ids.size() == 4 && std::chrono::duration<double>(Clock::now() - t).count() < 10.0,
                  "a refused sink's request is cancelled");
        }
        // A dropped handle is abandoned; the next request runs.
        {
            const auto t = Clock::now();
            (void)engine.submit(engine.prepare(question()), greedy(2000), ninfer::OutputConsumerMode::Aggregate, {});
            const auto next = engine.generate(engine.prepare(question()), greedy(4));
            const double s  = std::chrono::duration<double>(Clock::now() - t).count();
            check(next.generated_token_ids.size() == 4 && s < 10.0, "a dropped request does not run to its limit");
        }
        try {
            (void)engine.prepare_tokens({});
            check(false, "an empty token prompt is refused");
        } catch (const std::invalid_argument&) {}
        try {
            auto looped                         = greedy(8);
            looped.execution.thinking.loop      = ninfer::ReasoningLoopAction::Conclude;
            (void)engine.generate(engine.prepare(question()), looped);
            check(false, "a reasoning-loop guard is refused");
        } catch (const std::invalid_argument&) {}
        try {
            // Also where the Engine answers without the core.
            auto none                         = greedy(0);
            none.execution.thinking.loop      = ninfer::ReasoningLoopAction::Conclude;
            (void)engine.generate(engine.prepare(question()), none);
            check(false, "a zero-output request is checked too");
        } catch (const std::invalid_argument&) {}
        for (int which = 0; which < 2; ++which) {
            try {
                auto raw = greedy(8);
                (which == 0 ? raw.output.raw : raw.stop.publish_stop_token) = true;
                (void)engine.generate(engine.prepare(question()), raw);
                check(false, "raw output and a published stop token are refused");
            } catch (const std::invalid_argument&) {}
        }
        {
            // A consumer that throws cancels its request, and the Engine serves the next one.
            struct Throws final : ninfer::OutputSink {
                void start(ninfer::GenerationStart) override {}
                void progress(ninfer::PromptProgress) override {}
                void timing(ninfer::GenerationTimingObservation) override {}
                void publish(ninfer::OutputDelta) override { throw std::runtime_error("consumer gone"); }
            } throws;
            auto handle = engine.submit(engine.prepare(question()), greedy(2000), ninfer::OutputConsumerMode::Streaming, {});
            const auto t = Clock::now();
            try {
                (void)handle.wait(&throws);
                check(false, "the throwing consumer's error reaches its caller");
            } catch (const std::runtime_error&) {}
            const auto next = engine.generate(engine.prepare(question()), greedy(4));
            check(next.generated_token_ids.size() == 4 && std::chrono::duration<double>(Clock::now() - t).count() < 10.0,
                  "a failed consumer's request stops instead of decoding 2000 tokens");
        }

        // The image prompt.
        if (const char* root = std::getenv("NINFER_INKLING2_VISION"); root != nullptr && fs::exists(fs::path(root) / "test1.jpeg")) {
            std::ifstream in(fs::path(root) / "test1.jpeg", std::ios::binary);
            ninfer::MessagePart image{.kind = ninfer::MessagePartKind::Media};
            image.media.kind  = ninfer::MediaKind::Image;
            image.media.bytes = {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
            const auto r = engine.generate(engine.prepare(user_turn({image, text("Describe this image in one paragraph.")})), greedy(48));
            check(r.prompt.prompt_tokens == 234 && r.prompt.vision_items == 1, "the image prompt has llama.cpp's 234 tokens");
            std::ifstream want_in(fs::path(root) / "test1.llama.ids");
            std::string line;
            std::getline(want_in, line);
            std::istringstream ids(line);
            std::vector<ninfer::TokenId> want;
            for (ninfer::TokenId id = 0; ids >> id;) { want.push_back(id); }
            std::size_t match = 0;
            while (match < want.size() && match < r.generated_token_ids.size() && want[match] == r.generated_token_ids[match]) { ++match; }
            std::printf("image: vision %.3f s, prefill %.3f s; greedy matches llama.cpp's for %zu of %zu tokens (reported)\n",
                        r.timings.vision_seconds, r.timings.prefill_seconds, match, want.size());
            std::printf("  ids: %s\n  reasoning: %s\n", ids_text(r.generated_token_ids).c_str(), r.reasoning.c_str());
            // An image prompt past one chunk resumes with the image in the reused prefix, which is then not encoded.
            std::string longer = "Describe this image in one paragraph, then list every object in it.";
            // Past one whole chunk: reuse is chunk-grained.
            while (engine.tokenize_text(longer).size() + r.prompt.prompt_tokens <= chunk_of(engine) + 32) { // the image's 234 and more
                longer += " Mention colours, shapes, text and layout, and be precise.";
            }
            auto no_reuse                         = greedy(16);
            no_reuse.execution.allow_prefix_reuse = false;
            const auto cold  = engine.generate(engine.prepare(user_turn({image, text(longer)})), no_reuse);
            const auto again = engine.generate(engine.prepare(user_turn({image, text(longer)})), greedy(16));
            check(cold.prompt.prompt_tokens > chunk_of(engine) && again.reused_prompt_tokens >= chunk_of(engine) &&
                      again.reused_prompt_tokens % chunk_of(engine) == 0 && again.timings.vision_seconds == 0.0 &&
                      again.generated_token_ids == cold.generated_token_ids,
                  "an image prompt resumes from its snapshot: " + std::to_string(again.reused_prompt_tokens) + " reused, vision " +
                      std::to_string(again.timings.vision_seconds) + " s, same output " +
                      std::to_string(again.generated_token_ids == cold.generated_token_ids));
        }
        // Prefix reuse: a prompt that repeats the last one's first whole chunks resumes from their snapshot and
        // decodes what a prefill from 0 does; a request that forbids reuse, or an unrelated prompt, does not.
        {
            const std::uint32_t chunk = chunk_of(engine); // reuse is chunk-grained: the prompt passes one whole chunk
            const std::string context = facts_until(engine, chunk + 160);
            const auto ask = [&](const std::string& tail, bool reuse) {
                auto o                              = greedy(24);
                o.execution.allow_prefix_reuse = reuse;
                return engine.generate(engine.prepare(user_turn({text(context + tail)})), o);
            };
            (void)engine.generate(engine.prepare(question()), greedy(4)); // any snapshot is another prompt's
            const auto cold  = ask(" What is fact 3?", false); // takes none, leaves one
            const auto again = ask(" What is fact 3?", true);
            check(cold.reused_prompt_tokens == 0 && cold.prompt.prompt_tokens > chunk, "a request that forbids reuse reuses nothing");
            check(again.reused_prompt_tokens > 0 && again.reused_prompt_tokens % chunk == 0 &&
                      again.reused_prompt_tokens < again.prompt.prompt_tokens &&
                      again.computed_prefill_tokens == again.prompt.prompt_tokens - again.reused_prompt_tokens,
                  "the same prompt resumes from its snapshot: " + std::to_string(again.reused_prompt_tokens));
            check(again.generated_token_ids == cold.generated_token_ids, "a resumed prompt decodes what a cold one does");
            const auto other = ask(" Which fact is about 42?", true);
            const auto other_cold = ask(" Which fact is about 42?", false);
            check(other.reused_prompt_tokens > 0 && other.generated_token_ids == other_cold.generated_token_ids,
                  "a prompt sharing the first chunks resumes and decodes what a cold one does");
            const auto unrelated = engine.generate(engine.prepare(question()), greedy(8));
            check(unrelated.reused_prompt_tokens == 0, "an unrelated prompt reuses nothing");
            check(ask(" What is fact 3?", true).reused_prompt_tokens == 0, "an unrelated prompt replaced the snapshot");
            std::printf("prefix reuse: %u of %u prompt tokens reused; prefill %.3f s cold, %.3f s resumed\n", again.reused_prompt_tokens,
                        again.prompt.prompt_tokens, cold.timings.prefill_seconds, again.timings.prefill_seconds);
        }
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL: %s\n", error.what());
        return 1;
    }
    if (g_failures != 0) {
        std::fprintf(stderr, "FAIL: %d checks\n", g_failures);
        return 1;
    }
    std::printf("PASS\n");
    return 0;
}
