#include "ninfer/engine.h"
#include "lossy_fixture.h"

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

// Speculative decoding after a lossy long-prompt prefill (docs/maintainer/lossy-long-prefill.md):
// the draft state and the decode positions must follow the compacted lineage, so greedy tokens with
// MTP or DFlash2 must equal those of the same lossy prefill decoded without speculation.
// NINFER_LOSSY_SPEC_REAL selects the backend: dflash2 (default) or mtp.
namespace {
constexpr std::uint32_t kContext = 45056;
constexpr std::uint32_t kOutputs = 128;

void require(bool condition, const std::string& message) {
    if (!condition) { throw std::runtime_error(message); }
}

enum class Spec { DFlash2, Mtp };

Spec selected_spec() {
    const char* value           = std::getenv("NINFER_LOSSY_SPEC_REAL");
    const std::string_view name = value ? value : "dflash2";
    if (name == "dflash2") { return Spec::DFlash2; }
    if (name == "mtp") { return Spec::Mtp; }
    throw std::runtime_error("NINFER_LOSSY_SPEC_REAL must be dflash2 or mtp");
}

// rk4v4 as in production, the lossy option on, and speculation only when asked.
ninfer::EngineOptions lossy_options(const char* artifact, const Spec* spec) {
    ninfer::EngineOptions options;
    options.artifact_path                     = artifact;
    options.max_context                       = kContext;
    options.kv_capacity                       = ninfer::KvCapacityPolicy::explicit_capacity(kContext);
    options.prefill_chunk                     = 4096;
    options.kv_cache                          = ninfer::KvCacheStorage::RotatedLloyd4KeyInt4Value;
    options.max_concurrency                   = 1;
    options.max_pending_requests              = 1;
    options.context_cache.device_state_slots  = 4;
    options.context_cache.host_capacity_bytes = 0;
    options.lossy_prefill.min_tokens          = 30000;
    options.speculative.backend               = ninfer::SpeculativeBackend::None;
    if (spec && *spec == Spec::Mtp) {
        options.speculative.backend       = ninfer::SpeculativeBackend::Mtp;
        options.speculative.draft_tokens  = 3;
        options.speculative.proposal_head = ninfer::ProposalHead::Optimized;
    } else if (spec) {
        options.speculative.backend      = ninfer::SpeculativeBackend::DFlash2;
        options.speculative.draft_tokens = 7;
        options.gdn_state_fp16           = true;
    }
    return options;
}

struct Run {
    ninfer::GenerationResult result;
    std::uint32_t prompt_tokens = 0;
};

// One cold 40K request on a fresh engine; the lossy path must have been taken (kept < uncached,
// counters moved), so the comparison is not vacuous.
Run run_lossy(const char* artifact, const Spec* spec, const char* label) {
    ninfer::Engine engine(lossy_options(artifact, spec));
    auto input = ninfer::test::lossy_fixture_40k(engine, label);
    ninfer::RequestOptions request;
    request.execution.requested_output_tokens = kOutputs;
    request.execution.sampling.temperature    = 0.0F;
    request.execution.sampling.seed           = 0;
    request.execution.allow_prefix_reuse      = true;
    request.stop.include_model_defaults       = false;

    auto prompt                     = engine.prepare(input);
    const std::uint32_t tokens      = prompt.summary().prompt_tokens;
    const auto before               = engine.runtime_stats();
    auto result                     = engine.generate(std::move(prompt), request);
    const auto after                = engine.runtime_stats();
    const std::string where         = std::string(label) + ": ";
    require(result.lossy_prefill &&
                result.lossy_prefill->outcome == ninfer::LossyPrefillOutcome::Completed,
            where + "the lossy prefill path was not taken");
    require(result.lossy_prefill->kept_tokens < result.lossy_prefill->uncached_tokens &&
                result.lossy_prefill->uncached_tokens <= tokens &&
                result.prompt.prompt_tokens == tokens &&
                result.computed_prefill_tokens + result.reused_prompt_tokens < tokens &&
                after.lossy_prefills_completed == before.lossy_prefills_completed + 1,
            where + "lossy accounting does not show a thinned prompt");
    require(result.generated_token_ids.size() == kOutputs &&
                result.finish_reason == ninfer::FinishReason::OutputLimit,
            where + "did not generate the requested " + std::to_string(kOutputs) + " tokens");
    std::cout << label << ": prompt=" << tokens << " kept=" << result.lossy_prefill->kept_tokens
              << " uncached=" << result.lossy_prefill->uncached_tokens
              << " computed=" << result.computed_prefill_tokens << '\n';
    return {std::move(result), tokens};
}

int exercise(const char* artifact, Spec spec) {
    const char* name = spec == Spec::Mtp ? "mtp" : "dflash2";
    // The plain engine is destroyed before the speculative one is built: one 3090 holds one.
    const Run plain = run_lossy(artifact, nullptr, "plain lossy");
    const Run fast  = run_lossy(artifact, &spec, name);
    const auto expected_backend =
        spec == Spec::Mtp ? ninfer::SpeculativeBackend::Mtp : ninfer::SpeculativeBackend::DFlash2;
    const auto& stats = fast.result.speculative;
    require(stats.backend == expected_backend && stats.rounds != 0,
            std::string(name) + " generation bypassed speculation after the lossy prefill");
    require(stats.accepted_tokens != 0,
            std::string(name) + " accepted no proposal after the lossy prefill");
    require(plain.prompt_tokens == fast.prompt_tokens, "the two engines prepared different prompts");
    require(plain.result.lossy_prefill->kept_tokens == fast.result.lossy_prefill->kept_tokens,
            "the two engines kept different token counts, so the prefills are not comparable");
    const auto& a = plain.result.generated_token_ids;
    const auto& b = fast.result.generated_token_ids;
    for (std::size_t i = 0; i < a.size() && i < b.size(); ++i) {
        require(a[i] == b[i], std::string(name) + " diverged from plain decode at token " +
                                  std::to_string(i) + " (" + std::to_string(b[i]) + " vs " +
                                  std::to_string(a[i]) + ")");
    }
    std::cout << "ok " << name << " matches plain decode over " << a.size()
              << " tokens, accepted=" << stats.accepted_tokens << "/" << stats.drafted_tokens
              << " rounds=" << stats.rounds << " fallback=" << stats.fallback_steps << '\n';
    return 0;
}
} // namespace

int main() {
    const char* artifact = std::getenv("NINFER_TEST_ARTIFACT");
    if (!artifact || !*artifact) {
        std::cout << "skip: NINFER_TEST_ARTIFACT is not set\n";
        return 77;
    }
    try {
        return exercise(artifact, selected_spec());
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
