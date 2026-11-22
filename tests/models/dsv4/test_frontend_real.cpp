// Written for NInfer-3090.
//
// The chat frontend on the real DeepSeek-V4.1-Flash artifact's resources: CPU only, no GPU lock, reads
// just the artifact's resource objects (needs NINFER_TEST_DSV4_ARTIFACT; skips with 77 otherwise).
// Checks that the Qwen3.5 frontend accepts the V4 tokenizer config and template, renders a one-turn
// chat to a prompt that starts with the BOS token, and carries the model's end-of-sentence token as a stop.

#include "artifact/reader.h"
#include "models/dsv4/model.h"
#include "models/qwen3_5/frontend/frontend.h"
#include "models/qwen3_5/frontend/prepared_prompt.h"
#include "models/registry.h"

#include <cstdio>
#include <cstdlib>
#include <string>

namespace q35 = ninfer::models::qwen3_5;

int main() {
    const char* path = std::getenv("NINFER_TEST_DSV4_ARTIFACT");
    if (path == nullptr) {
        std::printf("SKIP: NINFER_TEST_DSV4_ARTIFACT is not set\n");
        return 77;
    }
    ninfer::artifact::Reader reader(path);
    auto plan = ninfer::models::dsv4::plan_load(reader, true);
    const auto frontend = q35::make_frontend(plan.resources(), {.architecture = ninfer::models::Architecture::Dsv4, .vision_enabled = false, .max_context = 2048});
    ninfer::PromptInput input;
    ninfer::ChatMessage message;
    message.role = ninfer::ChatRole::User;
    message.parts.push_back({.kind = ninfer::MessagePartKind::Text, .text = "What is the capital of France?"});
    input.messages.push_back(std::move(message));
    input.options.enable_thinking = false;
    const auto prompt = frontend.prepare(input);
    const auto& ids   = q35::PreparedPromptAccess::view(prompt).token_ids;
    std::string rendered;
    for (const auto id : ids) { rendered += "[" + frontend.token_bytes(id) + "]"; }
    std::printf("prompt (%zu tokens): %s\n", ids.size(), rendered.c_str());
    std::printf("stop tokens:");
    for (const auto id : frontend.default_stop_policy().token_ids) { std::printf(" %d(%s)", id, frontend.token_bytes(id).c_str()); }
    std::printf("\n");
    bool two_turn_ok = true;
    {
        // Two-turn: the second prompt against the first prompt + reply.
        for (const bool thinking : {false, true}) {
            auto turn = [&](bool second) {
                ninfer::PromptInput in;
                ninfer::ChatMessage u;
                u.role = ninfer::ChatRole::User;
                u.parts.push_back({.kind = ninfer::MessagePartKind::Text, .text = "What is the capital of France?"});
                in.messages.push_back(u);
                if (second) {
                    ninfer::ChatMessage a;
                    a.role = ninfer::ChatRole::Assistant;
                    a.parts.push_back({.kind = ninfer::MessagePartKind::Text, .text = "The capital of France is Paris."});
                    if (thinking) { a.reasoning_content = "Simple question."; }
                    in.messages.push_back(a);
                    ninfer::ChatMessage u2;
                    u2.role = ninfer::ChatRole::User;
                    u2.parts.push_back({.kind = ninfer::MessagePartKind::Text, .text = "And of Germany?"});
                    in.messages.push_back(u2);
                }
                in.options.enable_thinking = thinking;
                const auto prompt = frontend.prepare(in);
                return q35::PreparedPromptAccess::view(prompt).token_ids;
            };
            const auto first = turn(false);
            const auto second = turn(true);
            std::string r1, r2;
            for (const auto id : first) { r1 += "[" + frontend.token_bytes(id) + "]"; }
            for (const auto id : second) { r2 += "[" + frontend.token_bytes(id) + "]"; }
            std::size_t common = 0;
            while (common < first.size() && common < second.size() && first[common] == second[common]) { ++common; }
            std::printf("thinking=%d\nturn1 (%zu): %s\nturn2 (%zu): %s\ncommon prefix %zu\n", thinking ? 1 : 0, first.size(), r1.c_str(), second.size(), r2.c_str(), common);
            // Without thinking the follow-up prompt starts with the whole first prompt and its reply; with thinking
            // the first prompt ends in `<think>`, the follow-up drops the reasoning, so they agree up to the token
            // before it (the Program's prefix reuse relies on both: program/prefix_reuse.h).
            const std::size_t expected = thinking ? first.size() - 1 : first.size();
            if (common != expected) { two_turn_ok = false; }
        }
    }
    const bool ok = !ids.empty() && ids.front() == 0 && !frontend.default_stop_policy().token_ids.empty() && two_turn_ok;
    std::printf(ok ? "PASS\n" : "FAIL: the prompt must start with the BOS (id 0), a stop token must exist and a follow-up turn must extend the first prompt\n");
    return ok ? 0 : 1;
}
