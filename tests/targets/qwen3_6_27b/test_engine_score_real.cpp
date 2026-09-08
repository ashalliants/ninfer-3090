#include "targets/guarded_main.h"
#include "ninfer/engine.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

int run_score_checks() {
    const char* artifact = std::getenv("NINFER_QWEN3_6_27B_WEIGHTS");
    if (artifact == nullptr || *artifact == '\0') {
        std::cout << "SKIP: NINFER_QWEN3_6_27B_WEIGHTS is not set\n";
        return 77;
    }

    ninfer::EngineOptions options;
    options.artifact_path = artifact;
    options.purpose       = ninfer::EnginePurpose::CausalScoring;
    options.max_context   = 2048;
    options.kv_cache      = ninfer::KvCacheStorage::Fp8E4M3Row256;
    ninfer::Engine engine(options);
    const auto& effective = engine.options();
    if (effective.max_concurrency != 1 || effective.prefill_chunk != 1024 ||
        effective.kv_capacity.mode != ninfer::KvCapacityMode::Explicit ||
        effective.kv_capacity.explicit_tokens != effective.max_context ||
        effective.context_cache.enabled ||
        effective.speculative.backend != ninfer::SpeculativeBackend::None ||
        effective.kv_cache != ninfer::KvCacheStorage::Fp8E4M3Row256) {
        std::cerr << "causal scoring options were not normalized correctly\n";
        return 1;
    }

    std::string text;
    const std::string paragraph =
        "NInfer scores each target token from the preceding hidden state. "
        "Every evaluation window owns fresh state and a fresh KV address space.\n";
    std::vector<ninfer::TokenId> tokens;
    while (tokens.size() < 1537) {
        text += paragraph;
        tokens = engine.tokenize_text(text);
    }
    tokens.resize(1537);

    const std::vector<float> all      = engine.score_tokens(tokens, 1);
    const std::vector<float> suffix   = engine.score_tokens(tokens, 513);
    const std::vector<float> repeated = engine.score_tokens(tokens, 513);
    if (all.size() != 1536 || suffix.size() != 1024 || repeated.size() != suffix.size()) {
        std::cerr << "causal scoring returned an invalid result shape\n";
        return 1;
    }
    float maximum_overlap_error = 0.0F;
    float maximum_repeat_error  = 0.0F;
    std::size_t repeat_differences = 0;
    std::size_t first_repeat_index = suffix.size();
    for (std::size_t i = 0; i < suffix.size(); ++i) {
        if (!std::isfinite(all[i + 512]) || !std::isfinite(suffix[i])) {
            std::cerr << "causal scoring returned a non-finite logprob\n";
            return 1;
        }
        maximum_overlap_error = std::max(maximum_overlap_error, std::abs(all[i + 512] - suffix[i]));
        if (suffix[i] != repeated[i]) {
            ++repeat_differences;
            if (first_repeat_index == suffix.size()) { first_repeat_index = i; }
            maximum_repeat_error =
                std::max(maximum_repeat_error, std::abs(suffix[i] - repeated[i]));
        }
    }
    // Two windows scored identically must agree, but "disagree" has two very different causes and
    // the old message asserted the worse one without measuring. State/KV leaking between windows
    // moves logprobs by a visible amount; the reduction order varying between two runs of the same
    // fp8 kernel moves them in the last bits. Print enough to tell them apart -- how many elements
    // moved, by how much, and where the first one is -- instead of one sentence naming a cause.
    if (repeat_differences != 0) {
        std::cerr << "a repeated score window did not reproduce: " << repeat_differences << " of "
                  << suffix.size() << " logprobs differ, worst |delta|=" << maximum_repeat_error
                  << " first at index " << first_repeat_index << " (suffix="
                  << suffix[first_repeat_index] << " repeated=" << repeated[first_repeat_index]
                  << ")\n";
        return 1;
    }
    if (maximum_overlap_error > 0.25F) {
        std::cerr << "overlapping target suffix changed by " << maximum_overlap_error << '\n';
        return 1;
    }
    std::cout << "OK causal_score_real max_overlap_error=" << maximum_overlap_error << '\n';
    return 0;
}

NINFER_GUARDED_TEST_MAIN(run_score_checks)
