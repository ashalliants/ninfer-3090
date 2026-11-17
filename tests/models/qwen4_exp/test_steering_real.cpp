// Qwen3.8-Flash-Next refusal steering (--graft NAME=PATH) through the public Engine on the real
// artifact. Skips (77) unless NINFER_TEST_QWEN4EXP_ARTIFACT names a Qwen4Exp artifact (its n-gram
// volume beside it) and NINFER_TEST_STEERING_VECTOR names a GGUF control vector for it.
//
//   ninfer_qwen4_exp_steering_real_test [--generate N] [--mtp K] [--max-context N]
//
// One two-lane Engine holds the vector as "red". Greedy ids of a fixed chat prompt:
//   effect        a request selecting "red" generates different ids from one selecting none
//   isolation     two requests decoding in the same rounds, one selecting "red" and one none, each
//                 generate exactly the ids it generates alone (the selection is per row)
//   cache         the steered request run twice reuses its own cached prompt and repeats its ids;
//                 the unsteered request then run on the same prompt repeats ITS ids, so it did not
//                 resume from the steered request's checkpoints
//   opt-out       "graft": "" selects none, as an unstated graft does
//   refusal       an unknown name is refused

#include "ninfer/engine.h"
#include "ninfer/types.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <thread>
#include <vector>

using namespace ninfer;

namespace {

struct Options {
    std::uint32_t max_context = 4'096, tokens = 48, mtp = 0;
};

Options parse(const std::vector<std::string>& args) {
    Options o;
    for (std::size_t i = 0; i < args.size(); ++i) {
        const auto value = [&] {
            if (i + 1 >= args.size()) { throw std::invalid_argument(args[i] + " needs a value"); }
            return static_cast<std::uint32_t>(std::stoul(args[++i]));
        };
        if (args[i] == "--generate") {
            o.tokens = value();
        } else if (args[i] == "--mtp") {
            o.mtp = value();
        } else if (args[i] == "--max-context") {
            o.max_context = value();
        } else {
            throw std::invalid_argument("unknown option " + args[i]);
        }
    }
    return o;
}

EngineOptions engine_options(const std::string& artifact, const std::string& vector,
                             const Options& o) {
    EngineOptions options;
    options.artifact_path                     = artifact;
    options.max_context                       = o.max_context;
    options.kv_capacity                       = KvCapacityPolicy::explicit_capacity(o.max_context);
    options.max_concurrency                   = 2;
    options.prefill_chunk                     = 1024;
    options.kv_cache                          = KvCacheStorage::Int8Group64;
    options.context_cache.enabled             = true;
    options.context_cache.host_capacity_bytes = std::optional<std::size_t>(1ULL << 30);
    options.pending_timeout_ms                = 3'600'000;
    options.flash_next.save_expert_state      = false;
    options.flash_next.allow_unlocked_host_memory = true;
    // WSL (32 GB) cannot pin every expert: the experts stream from the artifact instead.
    if (std::getenv("NINFER_TEST_PARTIAL_RAM") != nullptr) {
        options.flash_next.partial_ram_residency = true;
    }
    if (const char* headroom = std::getenv("NINFER_TEST_VRAM_HEADROOM_MIB")) {
        options.flash_next.vram_headroom_bytes = std::stoull(headroom) << 20;
    }
    if (const char* headroom = std::getenv("NINFER_TEST_RAM_HEADROOM_MIB")) {
        options.flash_next.ram_headroom_bytes = std::stoull(headroom) << 20;
    }
    options.grafts = {GraftSource{.name = "red", .path = vector}};
    if (o.mtp > 0) {
        options.speculative.backend             = SpeculativeBackend::Mtp;
        options.speculative.draft_tokens        = o.mtp;
        options.flash_next.mtp_joint_acceptance = true;
        options.flash_next.mtp_rows_verify      = true;
    }
    return options;
}

struct Run {
    std::vector<TokenId> ids;
    std::uint32_t reused = 0;
};

// A chat prompt long enough to span several cache pages, selecting `graft` (nullopt: states none).
PromptInput chat(const std::string& question, std::optional<std::string> graft) {
    std::string body;
    for (int i = 0; i < 40; ++i) {
        body += "Paragraph " + std::to_string(i) +
                ": the library opens at nine and closes at five on weekdays. ";
    }
    ChatMessage message;
    message.role = ChatRole::User;
    message.parts.push_back(
        MessagePart{.kind = MessagePartKind::Text, .text = body + question, .media = {}});
    PromptInput input;
    input.messages.push_back(std::move(message));
    if (graft) { input.options.graft = *graft; }
    return input;
}

Run generate(Engine& engine, PromptInput input, std::uint32_t tokens, bool reuse) {
    RequestOptions request;
    request.execution.requested_output_tokens = tokens;
    request.execution.allow_prefix_reuse      = reuse;
    request.execution.sampling.temperature    = 0.0F;
    request.stop.include_model_defaults       = false;
    request.output.raw                        = true;
    auto handle = engine.submit(engine.prepare(std::move(input)), request,
                                OutputConsumerMode::Aggregate, GenerationObservationOptions{});
    auto result = handle.wait();
    return {std::move(result.generated_token_ids), result.reused_prompt_tokens};
}

int failures = 0;

void expect(bool condition, const std::string& what) {
    std::printf("%s: %s\n", condition ? "ok" : "FAIL", what.c_str());
    if (!condition) { ++failures; }
}

} // namespace

int main(int argc, char** argv) {
    const char* artifact = std::getenv("NINFER_TEST_QWEN4EXP_ARTIFACT");
    const char* vector   = std::getenv("NINFER_TEST_STEERING_VECTOR");
    if (artifact == nullptr || *artifact == '\0' || vector == nullptr || *vector == '\0') {
        std::printf("skip: NINFER_TEST_QWEN4EXP_ARTIFACT and NINFER_TEST_STEERING_VECTOR are not "
                    "both set\n");
        return 77;
    }
    try {
        const Options o = parse(std::vector<std::string>(argv + 1, argv + argc));
        Engine engine(engine_options(artifact, vector, o));
        const std::string question =
            "Please explain in two sentences what you would do if you were the librarian.";

        // The ids of each selection with the cache off: the references.
        const Run plain = generate(engine, chat(question, std::nullopt), o.tokens, false);
        const Run red   = generate(engine, chat(question, "red"), o.tokens, false);
        expect(!plain.ids.empty() && !red.ids.empty(), "both selections generate tokens");
        expect(plain.ids != red.ids,
               "selecting the vector changes the greedy ids (the steering has an effect)");
        const Run optout = generate(engine, chat(question, ""), o.tokens, false);
        expect(optout.ids == plain.ids, "an empty graft selects none, as an unstated graft does");

        // Isolation: one steered and one unsteered request decode in the same rounds.
        {
            Run a, b;
            std::thread first(
                [&] { a = generate(engine, chat(question, std::nullopt), o.tokens, false); });
            std::thread second(
                [&] { b = generate(engine, chat(question, "red"), o.tokens, false); });
            first.join();
            second.join();
            expect(a.ids == plain.ids,
                   "unsteered request beside a steered one repeats its solo ids");
            expect(b.ids == red.ids,
                   "steered request beside an unsteered one repeats its solo ids");
        }

        // Cache identity: steered twice (the second resumes from the first's checkpoints), then
        // unsteered on the same prompt, which must not resume from them.
        const Run red_first  = generate(engine, chat(question, "red"), o.tokens, true);
        const Run red_second = generate(engine, chat(question, "red"), o.tokens, true);
        expect(red_second.reused > 0, "a steered request reuses its own cached prompt");
        expect(red_first.ids == red.ids && red_second.ids == red.ids,
               "cached steered requests repeat the uncached steered ids");
        const Run plain_after = generate(engine, chat(question, std::nullopt), o.tokens, true);
        expect(plain_after.ids == plain.ids,
               "an unsteered request after a steered one repeats the unsteered ids");
        const Run plain_again = generate(engine, chat(question, std::nullopt), o.tokens, true);
        expect(plain_again.ids == plain.ids && plain_again.reused > 0,
               "an unsteered request reuses its own cached prompt");
        const Run red_after = generate(engine, chat(question, "red"), o.tokens, true);
        expect(red_after.ids == red.ids && red_after.reused > 0,
               "a steered request after an unsteered one still reuses its own cache");

        bool refused = false;
        try {
            (void)engine.prepare(chat(question, "green"));
        } catch (const std::invalid_argument&) { refused = true; }
        expect(refused, "a graft name that is not loaded is refused");
    } catch (const std::exception& error) {
        std::printf("FAIL: %s\n", error.what());
        return 1;
    }
    std::printf("%s steering_real\n", failures == 0 ? "OK" : "FAIL");
    return failures == 0 ? 0 : 1;
}
