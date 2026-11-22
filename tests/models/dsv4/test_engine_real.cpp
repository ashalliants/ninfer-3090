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
//      still serves the next one;
//   4. prefix reuse: a two-turn conversation (prompt, reply, new question) with reuse on skips the prefill of
//      everything the first turn computed and answers exactly as the same turn from scratch (reuse off) does; the
//      first turn's prompt sent again restores the checkpoint before its last token (the lane's state was
//      rewound) and gives the first answer again. Prints each request's prefill tokens and time to first token.

#include "ninfer/engine.h"

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <utility>
#include <vector>

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

PromptInput conversation(const std::vector<std::pair<ChatRole, std::string>>& turns) {
    PromptInput input;
    for (const auto& [role, text] : turns) {
        ChatMessage message;
        message.role = role;
        message.parts.push_back({.kind = MessagePartKind::Text, .text = text});
        input.messages.push_back(std::move(message));
    }
    input.options.enable_thinking = false;
    return input;
}

RequestOptions request(std::uint32_t tokens, bool greedy, bool reuse = false) {
    RequestOptions options;
    options.execution.requested_output_tokens = tokens;
    options.execution.allow_prefix_reuse      = reuse;
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
    {
        // Prefix reuse. A long first message makes the skipped prefill count.
        std::string passage = "Read this note. ";
        for (int i = 0; i < 12; ++i) {
            passage += "The harbour town of Veldmark holds a market every Tuesday, where fishermen sell herring, smoked eel and salt cod "
                       "beside bakers with rye loaves and farmers with late apples. ";
        }
        passage += "Now answer briefly: what day is the market?";
        const auto show = [](const char* label, const GenerationResult& r) {
            std::printf("%s: prompt %u tokens, reused %u, prefilled %u, time to first token %.2f s, %zu generated: '%s'\n", label,
                        r.prompt.prompt_tokens, r.reused_prompt_tokens, r.computed_prefill_tokens, r.timings.first_token_seconds,
                        r.generated_token_ids.size(), r.content.c_str());
        };
        const GenerationResult turn1 = engine.generate(engine.prepare(conversation({{ChatRole::User, passage}})), request(24, true, true));
        show("turn 1 (reuse on)", turn1);
        check(turn1.reused_prompt_tokens == 0 && turn1.computed_prefill_tokens == turn1.prompt.prompt_tokens, "the first turn prefills its whole prompt");
        check(!turn1.content.empty(), "the first turn answered");

        const std::vector<std::pair<ChatRole, std::string>> second{{ChatRole::User, passage}, {ChatRole::Assistant, turn1.content},
                                                                   {ChatRole::User, "And which day would it be if it moved one day later?"}};
        const GenerationResult turn2 = engine.generate(engine.prepare(conversation(second)), request(24, true, true));
        show("turn 2 (reuse on)", turn2);
        check(turn2.reused_prompt_tokens + 1 >= turn1.prompt.prompt_tokens, "the second turn reuses the first turn's prompt");
        check(turn2.prefix_reuse_path == PrefixReusePath::Checkpoint, "the reuse is reported as a prefix hit");
        check(turn2.reused_prompt_tokens + turn2.computed_prefill_tokens == turn2.prompt.prompt_tokens, "reused and prefilled tokens add up to the prompt");

        // The first turn's prompt again (the lane now holds the second turn): the last token runs, everything before
        // it comes from the checkpoint the first turn left.
        const GenerationResult retry = engine.generate(engine.prepare(conversation({{ChatRole::User, passage}})), request(24, true, true));
        show("turn 1 again (reuse on)", retry);
        check(retry.reused_prompt_tokens + 1 == retry.prompt.prompt_tokens, "a retry reuses all but its last token (the checkpoint)");
        check(retry.generated_token_ids == turn1.generated_token_ids, "a retry gives the first answer again");

        const GenerationResult scratch = engine.generate(engine.prepare(conversation(second)), request(24, true, false));
        show("turn 2 (reuse off)", scratch);
        check(scratch.reused_prompt_tokens == 0 && scratch.computed_prefill_tokens == scratch.prompt.prompt_tokens, "a request that opts out prefills everything");
        check(scratch.generated_token_ids == turn2.generated_token_ids, "the second turn answers as the same turn from scratch does");
        check(turn2.computed_prefill_tokens < scratch.computed_prefill_tokens, "reuse prefilled fewer tokens");

        // A request that opted out leaves nothing behind: the next one that allows reuse starts from scratch.
        const GenerationResult after_opt_out = engine.generate(engine.prepare(conversation(second)), request(2, true, true));
        show("turn 2 after an opt-out request (reuse on)", after_opt_out);
        check(after_opt_out.reused_prompt_tokens == 0, "an opt-out request's state is not reused");
    }
    std::printf(failures == 0 ? "PASS\n" : "FAILED\n");
    return failures == 0 ? 0 : 1;
}
