// Written for NInfer-3090 (Inkling-2-Mini bring-up).
//
// --spec dspark through the public Engine (runtime/engine/inkling2_core.h) on real chat prompts:
// - greedy output equals a plain Engine's, token for token, for every prompt, also under a thinking budget and a
//   stop string (each token still goes through the plain loop; a round only saves forwards);
// - a short output budget yields plain greedy's first tokens, and no round drafts more than the budget can emit;
// - a sampled request speculates, repeats under a fixed seed and samples what a plain Engine does (the
//   configurations are ninfer_inkling2_engine_dspark_sampled_test's);
// - relaxed drafting, which DSpark does not implement, and draft counts outside 1..7 are refused;
// - --draft-min-p and --dspark-adaptive yield plain greedy's tokens too;
// - reported, not judged: rounds, drafts accepted, plain fallback steps, and decode tok/s of both Engines.
// Skips without NINFER_TEST_INKLING2_ARTIFACT. NINFER_INKLING2_SPEC_DRAFTS: drafts per round (default 3, the serving default);
// NINFER_INKLING2_SPEC_TOKENS: output tokens per prompt (default 256).

#include "ninfer/engine.h"
#include "product/speculative_options.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

int g_failures = 0;
std::vector<std::vector<ninfer::TokenId>> g_plain_tokens; // the plain Engine's greedy tokens per prompt
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

ninfer::RequestOptions greedy(std::uint32_t tokens) {
    ninfer::RequestOptions options;
    options.execution.sampling.temperature    = 0.0F;
    options.execution.requested_output_tokens = tokens;
    return options;
}

// Chat prompts of different kinds: code, a list, prose, arithmetic, and code read back.
const std::vector<std::string> kPrompts = {
    "Write a Python function that returns the n-th Fibonacci number iteratively, with a docstring and type hints.",
    "List the planets of the solar system in order from the Sun, one per line, with one fact about each.",
    "Write a short story (about 150 words) about a lighthouse keeper who finds a message in a bottle.",
    "What is 17 * 23 + 4 * 19? Show the steps.",
    "Explain what this C code prints and why:\n\n#include <stdio.h>\nint main(void) {\n    int a[5] = {1, 2, 3, 4, 5};\n"
    "    int s = 0;\n    for (int i = 0; i < 5; ++i) s += a[i] * a[i];\n    printf(\"%d\\n\", s);\n    return 0;\n}",
};

struct Run {
    std::vector<std::vector<ninfer::TokenId>> tokens;
    double decode_seconds = 0;
    std::size_t generated = 0;
    std::uint64_t rounds = 0, drafted = 0, accepted = 0, fallback = 0;
};

Run run_all(ninfer::Engine& engine, std::uint32_t tokens) {
    Run run;
    for (const std::string& prompt : kPrompts) {
        const auto r = engine.generate(engine.prepare(user_turn(prompt)), greedy(tokens));
        run.tokens.push_back(r.generated_token_ids);
        run.decode_seconds += r.timings.decode_seconds;
        run.generated += r.generated_token_ids.size();
        run.rounds += r.speculative.rounds, run.drafted += r.speculative.drafted_tokens;
        run.accepted += r.speculative.accepted_tokens, run.fallback += r.speculative.fallback_steps;
        std::printf("  %4zu tokens, %6.1f tok/s, %3llu rounds, %.2f accepted per round, %4llu plain steps: %.40s\n",
                    r.generated_token_ids.size(), (r.generated_token_ids.size() - 1) / std::max(r.timings.decode_seconds, 1e-9),
                    static_cast<unsigned long long>(r.speculative.rounds),
                    r.speculative.rounds ? double(r.speculative.accepted_tokens) / r.speculative.rounds : 0.0,
                    static_cast<unsigned long long>(r.speculative.fallback_steps), prompt.c_str());
    }
    return run;
}

} // namespace

int main() {
    const char* artifact_path = std::getenv("NINFER_TEST_INKLING2_ARTIFACT");
    if (artifact_path == nullptr) {
        std::printf("SKIP: set NINFER_TEST_INKLING2_ARTIFACT\n");
        return 77;
    }
    const std::uint32_t drafts = std::getenv("NINFER_INKLING2_SPEC_DRAFTS") ? std::atoi(std::getenv("NINFER_INKLING2_SPEC_DRAFTS")) : 3;
    const std::uint32_t tokens = std::getenv("NINFER_INKLING2_SPEC_TOKENS") ? std::atoi(std::getenv("NINFER_INKLING2_SPEC_TOKENS")) : 256;
    std::size_t bf16_weight_bytes = 0; // a plain BF16 Engine's weights on the device
    try {
        ninfer::EngineOptions options;
        options.artifact_path = artifact_path;
        options.max_context   = 4096;
        options.prefill_chunk_auto = true; // the widest chunk that fits, as the CLI and serve run without --prefill-chunk
        Run plain;
        std::vector<ninfer::TokenId> plain_budget, plain_stop;
        // Tight thinking budgets on every prompt: the forced close lands at many points of DSpark's rounds.
        constexpr std::uint32_t kBudgets[] = {1, 3, 6, 12};
        const auto budgeted = [](std::uint32_t budget) {
            auto o                      = greedy(32);
            o.execution.thinking.budget = budget;
            return o;
        };
        std::vector<std::vector<ninfer::TokenId>> plain_budgets;
        ninfer::RequestOptions sampled;
        sampled.execution.sampling.temperature    = 0.8F;
        sampled.execution.sampling.seed           = 7;
        sampled.execution.requested_output_tokens = 32;
        std::vector<ninfer::TokenId> plain_sampled;
        {
            ninfer::Engine engine(options);
            std::printf("plain:\n");
            plain = run_all(engine, tokens);
            g_plain_tokens = plain.tokens;
            bf16_weight_bytes = engine.memory_summary().weights.capacity_bytes;
            plain_sampled = engine.generate(engine.prepare(user_turn(kPrompts[2])), sampled).generated_token_ids;
            auto budget                      = greedy(96);
            budget.execution.thinking.budget = 24;
            plain_budget = engine.generate(engine.prepare(user_turn(kPrompts[0])), budget).generated_token_ids;
            for (const auto& prompt : kPrompts) {
                for (const std::uint32_t b : kBudgets) {
                    plain_budgets.push_back(engine.generate(engine.prepare(user_turn(prompt)), budgeted(b)).generated_token_ids);
                }
            }
            auto stop = greedy(256);
            stop.stop.strings.push_back({.text = "def ", .channel = ninfer::OutputChannel::Content});
            plain_stop = engine.generate(engine.prepare(user_turn(kPrompts[0])), stop).generated_token_ids;
        }
        options.speculative.backend      = ninfer::SpeculativeBackend::DSpark;
        options.speculative.draft_tokens = drafts;
        // As ninfer serve normalizes it (an LM-head proposal for every backend), which DSpark has no use for.
        ninfer::product::apply_speculative_defaults(options.speculative);
        {
            auto relaxed             = options;
            relaxed.relaxed_drafting = 0.5F;
            bool refused             = false;
            try {
                ninfer::Engine refused_engine(relaxed);
            } catch (const std::invalid_argument&) { refused = true; }
            check(refused, "DSpark with relaxed drafting is refused");
        }
        for (const std::uint32_t bad : {0U, 8U, 0xFFFFFFFFU}) {
            auto wide                        = options;
            wide.speculative.draft_tokens    = bad;
            bool refused                     = false;
            try {
                ninfer::Engine refused_engine(wide);
            } catch (const std::invalid_argument&) { refused = true; }
            check(refused, "DSpark with " + std::to_string(bad) + " drafts is refused");
        }
        ninfer::Engine engine(options);
        std::printf("dspark, %u drafts per round:\n", drafts);
        const Run spec = run_all(engine, tokens);
        {
            const auto stats = engine.runtime_stats();
            check(stats.speculative_rounds == spec.rounds && stats.speculative_draft_tokens == spec.drafted &&
                      stats.speculative_accepted_tokens == spec.accepted && stats.speculative_fallback_steps == spec.fallback,
                  "the Engine's cumulative DSpark counters sum the requests'");
        }
        for (std::size_t i = 0; i < kPrompts.size(); ++i) {
            const auto& x = spec.tokens[i];
            const auto& y = plain.tokens[i];
            std::size_t at = 0;
            while (at < x.size() && at < y.size() && x[at] == y[at]) { ++at; }
            check(x == y, "prompt " + std::to_string(i) + ": the speculative tokens differ from plain greedy at " + std::to_string(at) +
                              " (" + std::to_string(x.size()) + " against " + std::to_string(y.size()) + " tokens)");
        }
        // A round drafts fewer than `drafts` only when the output budget is that close.
        check(spec.rounds > 0 && spec.drafted <= spec.rounds * drafts && spec.accepted <= spec.drafted,
              "rounds ran and are counted");
        for (std::uint32_t limit = 1; limit <= drafts + 3; ++limit) {
            // The first token comes from the prefill and the second from a plain step, so a round starts after two
            // tokens at the earliest and drafts at most limit - 3 (its n drafts yield n + 1 tokens).
            const auto r = engine.generate(engine.prepare(user_turn(kPrompts[0])), greedy(limit));
            const std::vector<ninfer::TokenId> want(plain.tokens[0].begin(), plain.tokens[0].begin() + std::min<std::size_t>(limit, plain.tokens[0].size()));
            check(r.generated_token_ids == want, "an output budget of " + std::to_string(limit) + " yields plain greedy's first tokens");
            check(r.speculative.drafted_tokens <= r.speculative.rounds * (limit > 3 ? limit - 3 : 0) && r.speculative.accepted_tokens <= r.speculative.drafted_tokens,
                  "an output budget of " + std::to_string(limit) + " drafts no more than it can emit");
        }
        auto budget                      = greedy(96);
        budget.execution.thinking.budget = 24;
        const auto b = engine.generate(engine.prepare(user_turn(kPrompts[0])), budget);
        check(b.generated_token_ids == plain_budget && b.thinking.applied, "a thinking budget closes at the same token");
        {
            // The injected close is a plain step, never a consumed draft: every forwarded token is a round's anchor, an
            // accepted draft or a plain step, and the output is plain greedy's.
            std::size_t i = 0;
            for (const auto& prompt : kPrompts) {
                for (const std::uint32_t budget_tokens : kBudgets) {
                    const auto r     = engine.generate(engine.prepare(user_turn(prompt)), budgeted(budget_tokens));
                    const auto& spec = r.speculative;
                    const std::string what = "thinking budget " + std::to_string(budget_tokens) + " on prompt " +
                                             std::to_string(i / std::size(kBudgets));
                    check(r.generated_token_ids == plain_budgets[i++], what + " yields plain greedy's tokens");
                    check(r.generated_token_ids.size() - 1 == spec.rounds + spec.accepted_tokens + spec.fallback_steps,
                          what + ": the forwarded tokens are anchors, accepted drafts and plain steps");
                }
            }
        }
        auto stop = greedy(256);
        stop.stop.strings.push_back({.text = "def ", .channel = ninfer::OutputChannel::Content});
        const auto s = engine.generate(engine.prepare(user_turn(kPrompts[0])), stop);
        check(s.generated_token_ids == plain_stop, "a stop string ends at the same token");
        {
            // Prefix reuse with the drafter: a resumed prompt decodes what a cold one does (the round gating is timed, so
            // the acceptance is reported; test_dspark checks a restored drafter drafts the same).
            // Reuse is chunk-grained: the prompt passes one whole prefill chunk (the Engine's load note names it).
            std::uint32_t chunk = 0;
            for (const std::string& note : engine.load_summary().notes) {
                if (note.rfind("prefill | chunk ", 0) == 0) { chunk = static_cast<std::uint32_t>(std::stoul(note.substr(16))); }
            }
            check(chunk > 0, "the Engine reports its prefill chunk");
            std::string context = "Here is a list of facts.";
            for (int i = 0; engine.tokenize_text(context).size() <= chunk + 64; ++i) {
                context += " Fact " + std::to_string(i) + ": the number " + std::to_string(i * 7) + " is noted.";
            }
            context += " Write the facts about even numbers as a table.";
            auto cold_options                         = greedy(64);
            cold_options.execution.allow_prefix_reuse = false;
            const auto cold  = engine.generate(engine.prepare(user_turn(context)), cold_options);
            const auto again = engine.generate(engine.prepare(user_turn(context)), greedy(64));
            check(again.reused_prompt_tokens >= chunk && again.reused_prompt_tokens % chunk == 0 &&
                      again.generated_token_ids == cold.generated_token_ids,
                  "a resumed prompt decodes as a cold one does: " + std::to_string(again.reused_prompt_tokens) + " reused");
            std::printf("prefix reuse: %u reused; accepted %llu of %llu drafts resumed, %llu of %llu cold\n", again.reused_prompt_tokens,
                        static_cast<unsigned long long>(again.speculative.accepted_tokens),
                        static_cast<unsigned long long>(again.speculative.drafted_tokens),
                        static_cast<unsigned long long>(cold.speculative.accepted_tokens),
                        static_cast<unsigned long long>(cold.speculative.drafted_tokens));
        }
        const auto s1 = engine.generate(engine.prepare(user_turn(kPrompts[2])), sampled);
        const auto s2 = engine.generate(engine.prepare(user_turn(kPrompts[2])), sampled);
        check(s1.speculative.rounds > 0 && s1.generated_token_ids == s2.generated_token_ids, "a sampled request speculates and repeats");
        check(s1.speculative.backend == ninfer::SpeculativeBackend::DSpark && s1.speculative.draft_window == drafts,
              "a sampled request reports the configured backend");
        // A sampled request's output is the plain Engine's under the same seed, and a greedy request after it drafts
        // and yields plain greedy's tokens.
        check(s1.generated_token_ids == plain_sampled, "a sampled request samples what a plain Engine does");
        const auto after = engine.generate(engine.prepare(user_turn(kPrompts[0])), greedy(tokens));
        check(after.generated_token_ids == plain.tokens[0] && after.speculative.rounds > 0,
              "a greedy request after a sampled one drafts and yields plain greedy's tokens");
        const double plain_rate = (plain.generated - kPrompts.size()) / plain.decode_seconds;
        const double spec_rate  = (spec.generated - kPrompts.size()) / spec.decode_seconds;
        std::printf("plain %.1f tok/s, dspark %.1f tok/s (%.2fx); %llu rounds, %.2f drafts accepted per round, %llu plain steps\n",
                    plain_rate, spec_rate, spec_rate / plain_rate, static_cast<unsigned long long>(spec.rounds),
                    spec.rounds ? double(spec.accepted) / spec.rounds : 0.0, static_cast<unsigned long long>(spec.fallback));
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL: %s\n", error.what());
        return 1;
    }
    try { // --draft-min-p and --dspark-adaptive (the serving defaults) change only how many drafts a round verifies
        ninfer::EngineOptions options;
        options.artifact_path               = artifact_path;
        options.max_context                 = 4096;
        options.speculative.backend         = ninfer::SpeculativeBackend::DSpark;
        options.speculative.draft_tokens    = 5;
        options.speculative.draft_min_p     = 0.5F;
        options.speculative.dspark_adaptive = true;
        ninfer::product::apply_speculative_defaults(options.speculative);
        for (const float bad : {-0.1F, 1.0F}) {
            auto refused_options                    = options;
            refused_options.speculative.draft_min_p = bad;
            bool refused                            = false;
            try {
                ninfer::Engine refused_engine(refused_options);
            } catch (const std::invalid_argument&) { refused = true; }
            check(refused, "--draft-min-p " + std::to_string(bad) + " is refused");
        }
        ninfer::Engine engine(options);
        std::printf("dspark, up to 5 drafts, --draft-min-p 0.5, adaptive:\n");
        const Run adaptive = run_all(engine, tokens);
        check(adaptive.tokens == g_plain_tokens, "--draft-min-p and --dspark-adaptive yield plain greedy's tokens");
        check(adaptive.rounds > 0 && adaptive.accepted > 0 && adaptive.drafted <= adaptive.rounds * 5, "adaptive rounds run and are counted");
        std::printf("adaptive: %llu rounds, %.2f drafts accepted per round, %.1f tok/s\n", static_cast<unsigned long long>(adaptive.rounds),
                    adaptive.rounds ? double(adaptive.accepted) / adaptive.rounds : 0.0,
                    (adaptive.generated - kPrompts.size()) / adaptive.decode_seconds);
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL: %s\n", error.what());
        return 1;
    }
    try { // after the Engines above are gone: one Engine at a time fits the card
        // --dense-q8 head: the drafter proposes from the target's re-encoded head, so DSpark still yields the plain
        // Engine's greedy tokens under that head and accepts drafts; the re-encoding is reported with the weights.
        ninfer::EngineOptions q8_plain;
        q8_plain.artifact_path = artifact_path;
        q8_plain.max_context   = 4096;
        q8_plain.dense_q8      = ninfer::DenseQ8::Head;
        q8_plain.prefill_chunk_auto = true;
        auto q8_spec           = q8_plain;
        q8_spec.speculative.backend      = ninfer::SpeculativeBackend::DSpark;
        q8_spec.speculative.draft_tokens = drafts;
        ninfer::product::apply_speculative_defaults(q8_spec.speculative);
        Run want;
        {
            ninfer::Engine engine(q8_plain);
            want = run_all(engine, tokens);
            check(engine.memory_summary().weights.capacity_bytes > bf16_weight_bytes,
                  "the --dense-q8 head re-encoding is counted in the weights");
        }
        ninfer::Engine engine(q8_spec);
        const Run got = run_all(engine, tokens);
        check(got.tokens == want.tokens, "--dense-q8 head: DSpark yields the plain Engine's greedy tokens");
        check(got.rounds > 0 && got.accepted > 0, "--dense-q8 head: the drafts from the re-encoded head are accepted");
        std::printf("--dense-q8 head: %llu rounds, %.2f drafts accepted per round\n", static_cast<unsigned long long>(got.rounds),
                    got.rounds ? double(got.accepted) / got.rounds : 0.0);
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
