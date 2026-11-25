// Written for NInfer-3090.
//
// Output constraints on Inkling-2 through the public Engine (models/inkling2/frontend/constraint.h) on the real
// artifact:
// - a json_schema answer parses and has the schema's properties and types; a json_object one parses as an object; a
//   choice answer is one of the choices; each ends at the model's own end (StopToken) with a complete observation;
// - the reasoning before the answer is free text (a thinking budget's close included), and the answer is the content;
// - unconstrained requests give the same tokens before and after constrained ones, as on a fresh Engine (no state
//   carries over, and the unconstrained path is unchanged);
// - batched decode (max_concurrency 4): constrained and unconstrained requests together, each its solo output;
// - --spec dspark: a constrained greedy request gives the plain Engine's tokens (the drafts only speed it up);
// - a constraint with a custom stop string is refused.
// Skips without NINFER_TEST_INKLING2_ARTIFACT. NINFER_INKLING2_CONSTRAINT_DSPARK=0 skips the DSpark case.

#include "ninfer/engine.h"

#include <nlohmann/json.hpp>

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace {

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
    options.execution.allow_prefix_reuse      = false;
    return options;
}

constexpr const char* kGuestSchema =
    R"({"type":"object","properties":{"name":{"type":"string"},"room":{"type":"integer"},"request":{"type":"string"}},)"
    R"("required":["name","room","request"],"additionalProperties":false})";
constexpr const char* kGuestPrompt =
    "A hotel guest called Maria in room 214 asks for extra towels. Give her name, room number and request as JSON.";

struct Constrained {
    std::string prompt;
    ninfer::RequestOptions options;
    std::string what;
};

std::vector<Constrained> constrained_cases() {
    std::vector<Constrained> out;
    auto schema       = greedy(1024);
    schema.constraint = ninfer::OutputConstraint::json_schema(kGuestSchema);
    out.push_back({kGuestPrompt, schema, "json_schema"});
    auto object                           = greedy(768);
    object.constraint                     = ninfer::OutputConstraint::json_object();
    object.execution.sampling.temperature = 0.8F;
    object.execution.sampling.seed        = 99;
    object.execution.thinking.budget      = 48; // the injected close leads into the constrained answer
    out.push_back({"Describe a hotel room as a JSON object with a few fields.", object, "json_object, sampled, budget"});
    auto choice       = greedy(512);
    choice.constraint = ninfer::OutputConstraint::choice({"yes", "no"});
    out.push_back({"Is Paris the capital of France? Answer yes or no.", choice, "choice"});
    return out;
}

void check_answer(const Constrained& c, const ninfer::GenerationResult& r) {
    const std::string tag = c.what + ": ";
    check(r.finish_reason == ninfer::FinishReason::StopToken, tag + "ends at the model's own end");
    check(r.constraint.has_value() && r.constraint->complete && r.constraint->branch == ninfer::ConstraintOutputBranch::Content,
          tag + "the observation reports a complete answer");
    check(r.tool_calls.empty(), tag + "no tool call");
    if (c.options.constraint->kind == ninfer::OutputConstraintKind::Choice) {
        check(r.content == "yes" || r.content == "no", tag + "one of the choices, got '" + r.content + "'");
        return;
    }
    nlohmann::json j;
    try {
        j = nlohmann::json::parse(r.content);
    } catch (const std::exception& e) {
        check(false, tag + "the content parses as JSON: " + e.what() + " in '" + r.content + "'");
        return;
    }
    check(j.is_object(), tag + "an object");
    if (c.options.constraint->kind == ninfer::OutputConstraintKind::JsonSchema) {
        check(j.size() == 3 && j.contains("name") && j["name"].is_string() && j.contains("room") &&
                  j["room"].is_number_integer() && j.contains("request") && j["request"].is_string(),
              tag + "the schema's properties and types, got " + r.content);
    }
    std::printf("%s -> %s (reasoning %zu chars)\n", c.what.c_str(), r.content.c_str(), r.reasoning.size());
}

ninfer::EngineOptions base_options(const char* artifact) {
    ninfer::EngineOptions options;
    options.artifact_path = artifact;
    options.max_context   = 4096;
    return options;
}

const std::vector<std::string> kPlain = {"Name three primary colours.", "Write a haiku about a hotel lobby."};

} // namespace

int main() {
    const char* artifact = std::getenv("NINFER_TEST_INKLING2_ARTIFACT");
    if (artifact == nullptr) {
        std::printf("SKIP: set NINFER_TEST_INKLING2_ARTIFACT\n");
        return 77;
    }
    try {
        const auto cases = constrained_cases();
        std::vector<std::vector<ninfer::TokenId>> plain_fresh, constrained_solo;
        {
            ninfer::Engine engine(base_options(artifact));
            for (const auto& p : kPlain) { plain_fresh.push_back(engine.generate(engine.prepare(user_turn(p)), greedy(96)).generated_token_ids); }
            for (const auto& c : cases) {
                const auto r = engine.generate(engine.prepare(user_turn(c.prompt)), c.options);
                check_answer(c, r);
                constrained_solo.push_back(r.generated_token_ids);
            }
            for (std::size_t i = 0; i < kPlain.size(); ++i) {
                const auto again = engine.generate(engine.prepare(user_turn(kPlain[i])), greedy(96)).generated_token_ids;
                check(again == plain_fresh[i], "an unconstrained request after constrained ones decodes as before: " + kPlain[i]);
            }
            // As the Qwen models refuse it: a constrained answer ends at the model's own end.
            bool refused = false;
            try {
                auto o       = cases[0].options;
                o.stop.strings.push_back({.text = "}", .channel = ninfer::OutputChannel::Content});
                (void)engine.generate(engine.prepare(user_turn(kGuestPrompt)), o);
            } catch (const ninfer::RequestError&) { refused = true; }
            check(refused, "a constraint with a custom stop string is refused");
        }
        std::printf("single-lane cases done\n");
        {
            auto options            = base_options(artifact);
            options.max_concurrency = 4;
            ninfer::Engine engine(options);
            std::vector<ninfer::GenerationHandle> handles;
            for (const auto& c : cases) {
                handles.push_back(engine.submit(engine.prepare(user_turn(c.prompt)), c.options, ninfer::OutputConsumerMode::Aggregate, {}));
            }
            for (const auto& p : kPlain) {
                handles.push_back(engine.submit(engine.prepare(user_turn(p)), greedy(96), ninfer::OutputConsumerMode::Aggregate, {}));
            }
            for (std::size_t i = 0; i < handles.size(); ++i) {
                const auto r = handles[i].wait(nullptr);
                if (i < cases.size()) {
                    check_answer(cases[i], r);
                    check(r.generated_token_ids == constrained_solo[i], "batched: " + cases[i].what + " gives its solo tokens");
                } else {
                    check(r.generated_token_ids == plain_fresh[i - cases.size()], "batched: an unconstrained request gives its solo tokens");
                }
            }
        }
        std::printf("batched cases done\n");
        const char* dspark = std::getenv("NINFER_INKLING2_CONSTRAINT_DSPARK");
        if (dspark == nullptr || std::string(dspark) != "0") {
            auto options                     = base_options(artifact);
            options.speculative.backend      = ninfer::SpeculativeBackend::DSpark;
            options.speculative.draft_tokens = 3;
            ninfer::Engine engine(options);
            for (std::size_t i = 0; i < cases.size(); ++i) {
                if (cases[i].options.execution.sampling.temperature > 0.0F) { continue; } // sampled: covered by the seed tests
                const auto r = engine.generate(engine.prepare(user_turn(cases[i].prompt)), cases[i].options);
                check_answer(cases[i], r);
                check(r.generated_token_ids == constrained_solo[i], "dspark: " + cases[i].what + " gives the plain Engine's tokens");
            }
            std::printf("dspark cases done\n");
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAILED: %s\n", e.what());
        return 1;
    }
    if (g_failures != 0) {
        std::fprintf(stderr, "%d check(s) failed\n", g_failures);
        return 1;
    }
    std::printf("inkling2 engine constraints: all passed\n");
    return 0;
}
