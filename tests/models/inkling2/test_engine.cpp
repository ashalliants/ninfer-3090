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
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <sstream>
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
        {
            // What Inkling-2 cannot do is refused at construction, not left to fail later.
            ninfer::EngineOptions suspended = options;
            suspended.suspend.enabled       = true;
            try {
                ninfer::Engine refused(suspended);
                check(false, "model suspend is refused");
            } catch (const std::invalid_argument&) {}
        }
        const auto t0         = Clock::now();
        ninfer::Engine engine(options);
        std::printf("engine ready in %.1f s (%s)\n", std::chrono::duration<double>(Clock::now() - t0).count(),
                    engine.load_summary().architecture.c_str());
        check(engine.load_summary().architecture == "inkling2", "the Engine loads Inkling-2");
        check(!engine.options().context_cache.enabled && !engine.load_summary().takes_video,
              "no context cache and no video is reported");

        const auto question = [] { return user_turn({text("Name three primary colours.")}); };
        // Aggregate, twice, and streaming.
        const auto a = engine.generate(engine.prepare(question()), greedy(64));
        const auto b = engine.generate(engine.prepare(question()), greedy(64));
        Collect sink;
        ninfer::GenerationObservationOptions observe;
        observe.prompt_progress = true;
        auto handle             = engine.submit(engine.prepare(question()), greedy(64), ninfer::OutputConsumerMode::Streaming, observe);
        const auto c            = handle.wait(&sink);
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
