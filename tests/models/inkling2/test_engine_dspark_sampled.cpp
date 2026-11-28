// Written for NInfer-3090 (Inkling-2-Mini).
//
// --spec dspark on sampled requests, and its draft-count options, through the public Engine on real chat prompts:
// - a sampled request (temperature, top-k, top-p, min-p, presence and frequency penalties) yields, token for token,
//   what a plain Engine samples under the same seed: every token is still drawn from the target's logits with the
//   request's next uniform, and the drafts (the drafter's draws under those uniforms) only save forwards;
// - the same holds, and greedy output stays plain greedy's, with --draft-min-p and --dspark-adaptive;
// - sampled requests draft and accept drafts (rounds run);
// - reported, not judged: tok/s and drafts accepted per configuration.
// Skips without NINFER_TEST_INKLING2_ARTIFACT. NINFER_INKLING2_SPEC_TOKENS: output tokens per request (default 128).
// NINFER_TEST_INKLING2_DFLASH2 (the DFlash2 release directory) runs the drafted variants with --spec dflash2 --draft-model.

#include "ninfer/engine.h"
#include "product/speculative_options.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
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
    input.messages.push_back(
        {.role = ninfer::ChatRole::User, .parts = {{.kind = ninfer::MessagePartKind::Text, .text = text}}});
    return input;
}

const std::vector<std::string> kPrompts = {
    "Write a Python function that returns the n-th Fibonacci number iteratively, with a docstring and type hints.",
    "Summarize the causes of the French Revolution in five bullet points.",
    "Write a short story (about 150 words) about a lighthouse keeper who finds a message in a bottle.",
    "Convert this JSON to YAML: {\"name\": \"app\", \"version\": \"1.2.0\", \"deps\": {\"react\": \"18\", \"vite\": \"5\"}}",
};

struct Config {
    const char* name;
    float temperature;
    std::int32_t top_k;
    float top_p, min_p, presence, frequency;
    std::uint64_t seed;
    std::uint32_t thinking_budget; // 0: none
};

const std::vector<Config> kConfigs = {
    {"greedy", 0.0F, 20, 1.0F, 0.0F, 0.0F, 0.0F, 0, 0},
    {"t0.6 top-p 0.95", 0.6F, 20, 0.95F, 0.0F, 0.0F, 0.0F, 11, 0},
    {"t0.8", 0.8F, 20, 1.0F, 0.0F, 0.0F, 0.0F, 7, 0},
    {"t1.0 top-k 10 min-p 0.05", 1.0F, 10, 1.0F, 0.05F, 0.0F, 0.0F, 3, 0},
    {"t0.7 top-p 0.9 presence 1.5", 0.7F, 20, 0.9F, 0.0F, 1.5F, 0.0F, 5, 0},
    {"t1.0 frequency 0.5 budget 12", 1.0F, 20, 1.0F, 0.0F, 0.0F, 0.5F, 9, 12},
};

ninfer::RequestOptions request(const Config& c, std::uint32_t tokens) {
    ninfer::RequestOptions o;
    auto& s                     = o.execution.sampling;
    s.temperature               = c.temperature;
    s.top_k                     = c.top_k;
    s.top_p                     = c.top_p;
    s.min_p                     = c.min_p;
    s.presence_penalty          = c.presence;
    s.frequency_penalty         = c.frequency;
    s.seed                      = c.seed;
    o.execution.requested_output_tokens = tokens;
    if (c.thinking_budget != 0) { o.execution.thinking.budget = c.thinking_budget; }
    return o;
}

struct Totals {
    std::vector<std::vector<ninfer::TokenId>> tokens; // per config, then prompt
    std::vector<double> seconds, generated, rounds, drafted, accepted; // per config
};

Totals run_all(ninfer::Engine& engine, std::uint32_t tokens) {
    Totals t;
    for (const Config& c : kConfigs) {
        double seconds = 0, generated = 0, rounds = 0, drafted = 0, accepted = 0;
        for (const std::string& prompt : kPrompts) {
            const auto r = engine.generate(engine.prepare(user_turn(prompt)), request(c, tokens));
            t.tokens.push_back(r.generated_token_ids);
            seconds += r.timings.decode_seconds;
            generated += static_cast<double>(r.generated_token_ids.size()) - 1;
            rounds += static_cast<double>(r.speculative.rounds);
            drafted += static_cast<double>(r.speculative.drafted_tokens);
            accepted += static_cast<double>(r.speculative.accepted_tokens);
        }
        t.seconds.push_back(seconds), t.generated.push_back(generated), t.rounds.push_back(rounds);
        t.drafted.push_back(drafted), t.accepted.push_back(accepted);
    }
    return t;
}

void report(const char* engine, const Totals& t) {
    for (std::size_t i = 0; i < kConfigs.size(); ++i) {
        std::printf("  %-14s %-30s %6.1f tok/s  %4.0f rounds  %5.1f%% of %4.0f drafts accepted\n", engine, kConfigs[i].name,
                    t.generated[i] / std::max(t.seconds[i], 1e-9), t.rounds[i], t.drafted[i] > 0 ? 100.0 * t.accepted[i] / t.drafted[i] : 0.0,
                    t.drafted[i]);
    }
}

} // namespace

int main() {
    const char* artifact_path = std::getenv("NINFER_TEST_INKLING2_ARTIFACT");
    if (artifact_path == nullptr) {
        std::printf("SKIP: set NINFER_TEST_INKLING2_ARTIFACT\n");
        return 77;
    }
    const std::uint32_t tokens = std::getenv("NINFER_INKLING2_SPEC_TOKENS") ? std::atoi(std::getenv("NINFER_INKLING2_SPEC_TOKENS")) : 128;
    try {
        ninfer::EngineOptions options;
        options.artifact_path = artifact_path;
        options.max_context   = 4096;
        Totals plain;
        {
            ninfer::Engine engine(options);
            plain = run_all(engine, tokens);
            report("plain", plain);
        }
        struct Variant {
            const char* name;
            std::uint32_t drafts;
            float min_p;
            bool adaptive;
        };
        for (const Variant& v : {Variant{"dspark 3", 3, 0.0F, false}, Variant{"min-p, adaptive", 5, 0.1F, true}}) {
            auto spec                        = options;
            spec.speculative.backend         = ninfer::SpeculativeBackend::DSpark;
            if (const char* dir = std::getenv("NINFER_TEST_INKLING2_DFLASH2"); dir != nullptr && *dir != '\0') {
                spec.speculative.backend     = ninfer::SpeculativeBackend::DFlash2;
                spec.speculative.draft_model = dir;
            }
            spec.speculative.draft_tokens    = v.drafts;
            spec.speculative.draft_min_p     = v.min_p;
            spec.speculative.dspark_adaptive = v.adaptive;
            ninfer::product::apply_speculative_defaults(spec.speculative);
            ninfer::Engine engine(spec);
            const Totals got = run_all(engine, tokens);
            report(v.name, got);
            for (std::size_t i = 0; i < got.tokens.size(); ++i) {
                const auto& x = got.tokens[i];
                const auto& y = plain.tokens[i];
                std::size_t at = 0;
                while (at < x.size() && at < y.size() && x[at] == y[at]) { ++at; }
                check(x == y, std::string(v.name) + ", " + kConfigs[i / kPrompts.size()].name + ", prompt " +
                                  std::to_string(i % kPrompts.size()) + ": the tokens differ from plain at " + std::to_string(at) +
                                  " (" + std::to_string(x.size()) + " against " + std::to_string(y.size()) + ")");
            }
            for (std::size_t c = 0; c < kConfigs.size(); ++c) {
                check(got.rounds[c] > 0 && got.accepted[c] > 0,
                      std::string(v.name) + ", " + kConfigs[c].name + ": rounds run and drafts are accepted");
            }
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
