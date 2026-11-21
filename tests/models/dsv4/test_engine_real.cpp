// Written for NInfer-3090.
//
// DeepSeek-V4.1-Flash through the public Engine on the real artifact (needs NINFER_TEST_DSV4_ARTIFACT;
// skips with 77 otherwise). Experts are read from the artifact file, so a cold run takes minutes
// (about 7.5 s a token on a cold cache over a slow drive). NINFER_TEST_DSV4_EXPERT_CACHE_MIB sets the
// pinned expert cache (default: the Engine's 8 GiB); NINFER_TEST_DSV4_MAX_CONTEXT the context (default 2048).
//
//   1. a chat request 'What is the capital of France?' with greedy decoding and 24 tokens streams text
//      that contains 'Paris' and finishes on the output limit;
//   2. a second request on the same Engine (the lane is reused) samples with temperature and top-p;
//   3. a request that is cancelled after its first streamed text ends as cancelled and the Engine
//      still serves the next one.

#include "ninfer/engine.h"

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <string>

using namespace ninfer;

namespace {

int failures = 0;

void check(bool ok, const char* what) {
    std::printf("%s: %s\n", ok ? "ok  " : "FAIL", what);
    if (!ok) { ++failures; }
}

class CollectingSink final : public OutputSink {
public:
    void start(GenerationStart start) override { prompt_tokens = start.prompt.prompt_tokens; ++starts; }
    void progress(PromptProgress) override {}
    void timing(GenerationTimingObservation) override {}
    void publish(OutputDelta delta) override {
        text += delta.text;
        ++deltas;
        if (cancel_after_first != nullptr && !delta.text.empty()) { cancel_after_first->store(true); }
    }
    std::string text;
    std::uint32_t prompt_tokens = 0;
    int starts                  = 0;
    int deltas                  = 0;
    std::atomic<bool>* cancel_after_first = nullptr;
};

PromptInput question(const std::string& text) {
    PromptInput input;
    ChatMessage message;
    message.role = ChatRole::User;
    message.parts.push_back({.kind = MessagePartKind::Text, .text = text});
    input.messages.push_back(std::move(message));
    input.options.enable_thinking = false;
    return input;
}

RequestOptions request(std::uint32_t tokens, bool greedy) {
    RequestOptions options;
    options.execution.requested_output_tokens = tokens;
    options.execution.allow_prefix_reuse      = false;
    if (greedy) {
        options.execution.sampling.temperature = 0.0F;
    } else {
        options.execution.sampling.temperature = 0.8F;
        options.execution.sampling.top_p       = 0.9F;
        options.execution.sampling.seed        = 1234;
    }
    return options;
}

} // namespace

int main() {
    std::setvbuf(stdout, nullptr, _IOLBF, 0);
    const char* artifact = std::getenv("NINFER_TEST_DSV4_ARTIFACT");
    if (artifact == nullptr) {
        std::printf("SKIP: NINFER_TEST_DSV4_ARTIFACT is not set\n");
        return 77;
    }
    EngineOptions options;
    options.artifact_path = artifact;
    options.max_context   = 2048;
    options.kv_capacity   = KvCapacityPolicy::explicit_capacity(2048);
    if (const char* env = std::getenv("NINFER_TEST_DSV4_MAX_CONTEXT")) {
        options.max_context = static_cast<std::uint32_t>(std::atoi(env));
        options.kv_capacity = KvCapacityPolicy::explicit_capacity(options.max_context);
    }
    if (const char* env = std::getenv("NINFER_TEST_DSV4_EXPERT_CACHE_MIB")) {
        options.flash_next.expert_cache_bytes = static_cast<std::uint64_t>(std::atoll(env)) << 20;
    }
    options.prefill_chunk = 256;

    Engine engine(options);
    check(engine.load_summary().architecture == "deepseek41", "the load summary names the architecture");

    {
        CollectingSink sink;
        auto handle = engine.submit(engine.prepare(question("What is the capital of France?")), request(24, true),
                                    OutputConsumerMode::Streaming);
        const GenerationResult result = handle.wait(&sink);
        std::printf("greedy: prompt %u tokens, %zu generated, finish %d: '%s'\n", result.prompt.prompt_tokens,
                    result.generated_token_ids.size(), static_cast<int>(result.finish_reason), result.content.c_str());
        check(sink.starts == 1, "the stream began once");
        check(sink.deltas > 0, "text streamed token by token");
        check(sink.text == result.content, "the streamed text is the result's content");
        check(result.content.find("Paris") != std::string::npos, "the answer names Paris");
        check(result.generated_token_ids.size() <= 24 && !result.generated_token_ids.empty(), "at most 24 tokens");
        check(result.finish_reason == FinishReason::OutputLimit || result.finish_reason == FinishReason::StopToken,
              "finished on the limit or the end-of-turn token");
    }
    {
        // The lane is free again: a sampled request on the same Engine.
        CollectingSink sink;
        const GenerationResult result = engine.generate(engine.prepare(question("Name one famous painting.")), request(8, false), &sink);
        std::printf("sampled: %zu generated: '%s'\n", result.generated_token_ids.size(), result.content.c_str());
        check(!result.generated_token_ids.empty(), "sampled decoding produced tokens");
    }
    {
        std::atomic<bool> cancelled{false};
        CollectingSink sink;
        sink.cancel_after_first = &cancelled;
        const CancellationView view([&cancelled] { return cancelled.load(); });
        bool threw = false;
        try {
            (void)engine.generate(engine.prepare(question("Count from one to fifty.")), request(40, true), &sink, view);
        } catch (const std::exception& error) {
            threw = true;
            std::printf("cancelled request ended: %s\n", error.what());
        }
        std::printf("cancelled: streamed %d deltas, threw %d\n", sink.deltas, threw ? 1 : 0);
        check(sink.deltas >= 1 && sink.deltas < 40, "cancelling stopped the stream early");
        const GenerationResult after = engine.generate(engine.prepare(question("Say hello.")), request(4, true));
        check(!after.generated_token_ids.empty(), "the Engine serves the request after a cancelled one");
    }
    std::printf(failures == 0 ? "PASS\n" : "FAILED\n");
    return failures == 0 ? 0 : 1;
}
