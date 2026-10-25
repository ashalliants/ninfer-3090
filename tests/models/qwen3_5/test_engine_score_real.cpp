#include "guarded_main.h"
#include "ninfer/engine.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

// The target_logprobs Op's FP32 criterion (tests/ops/test_target_logprobs.cpp).
constexpr double kRelativeL2      = 2.0e-5;
constexpr double kGrossAbsolute   = 2.0e-4;
constexpr std::uint32_t kDecodeWindow = 4;

using ninfer::ScoringOptions;

ScoringOptions decode(int window) {
    return ScoringOptions{.phase = ScoringOptions::Phase::Decode, .window = window};
}

// Compares `actual` with `reference` under the target_logprobs criterion and prints the measured
// distance, so a failure on the GPU box says by how much.
int within_criterion(const char* label, const std::vector<float>& actual,
                     const std::vector<float>& reference) {
    if (actual.size() != reference.size() || actual.empty()) {
        std::cerr << label << ": result sizes differ\n";
        return 1;
    }
    double difference = 0.0;
    double norm       = 0.0;
    double worst      = 0.0;
    double total      = 0.0;
    std::size_t where = 0;
    for (std::size_t i = 0; i < actual.size(); ++i) {
        if (!std::isfinite(actual[i])) {
            std::cerr << label << ": non-finite log probability at " << i << '\n';
            return 1;
        }
        const double delta = static_cast<double>(actual[i]) - static_cast<double>(reference[i]);
        difference += delta * delta;
        norm += static_cast<double>(reference[i]) * static_cast<double>(reference[i]);
        total += std::abs(delta);
        if (std::abs(delta) > worst) {
            worst = std::abs(delta);
            where = i;
        }
    }
    const double relative = norm > 0.0 ? std::sqrt(difference / norm) : std::sqrt(difference);
    std::cout << label << ": relative L2 " << relative << ", max |delta| " << worst << " at row "
              << where << ", mean |delta| " << total / static_cast<double>(actual.size()) << '\n';
    if (relative > kRelativeL2 || worst > kGrossAbsolute) {
        std::cerr << label << ": outside the target_logprobs criterion\n";
        return 1;
    }
    return 0;
}

struct RecordingSink final : ninfer::ScoreRowSink {
    std::vector<std::int64_t> positions;
    std::vector<double> target_logprobs;
    std::size_t vocabulary = 0;
    bool finite            = true;
    const std::vector<ninfer::TokenId>* tokens = nullptr;

    void row(std::int64_t position, std::span<const float> logits) override {
        positions.push_back(position);
        if (vocabulary == 0) { vocabulary = logits.size(); }
        if (logits.size() != vocabulary || position < 0 ||
            static_cast<std::size_t>(position) >= tokens->size()) {
            finite = false;
            return;
        }
        // FP64 log-softmax at the target, the target_logprobs oracle.
        double maximum = -INFINITY;
        for (const float value : logits) {
            if (!std::isfinite(value)) { finite = false; }
            maximum = std::max(maximum, static_cast<double>(value));
        }
        double sum = 0.0;
        for (const float value : logits) { sum += std::exp(static_cast<double>(value) - maximum); }
        const auto target = static_cast<std::size_t>((*tokens)[static_cast<std::size_t>(position)]);
        const double lse = maximum + std::log(sum);
        target_logprobs.push_back(target < logits.size() ? static_cast<double>(logits[target]) - lse
                                                         : NAN);
    }
};

template <class Function>
int expect_invalid(const char* label, Function&& function) {
    try {
        (void)function();
    } catch (const std::invalid_argument&) { return 0; } catch (const std::exception& error) {
        std::cerr << label << ": expected invalid_argument, got " << error.what() << '\n';
        return 1;
    }
    std::cerr << label << ": expected invalid_argument\n";
    return 1;
}

// T=1 (the ordinary decode step) and T=4 (the verify forward) against prefill scoring.
int decode_phase_matches_prefill(ninfer::Engine& engine, const std::vector<ninfer::TokenId>& tokens,
                                 std::uint32_t first_target) {
    const std::vector<float> prefill = engine.score_tokens(tokens, first_target);
    int failures                     = 0;
    for (const int window : {1, static_cast<int>(kDecodeWindow)}) {
        const std::vector<float> decoded =
            engine.score_tokens(tokens, first_target, decode(window), nullptr);
        const std::string label = "decode_phase_matches_prefill T=" + std::to_string(window);
        failures += within_criterion(label.c_str(), decoded, prefill);
    }
    return failures;
}

// Every row of a T=1 run against a T=4 run, and a partial last window (T=3 over 256 targets).
int decode_windows_agree(ninfer::Engine& engine, const std::vector<ninfer::TokenId>& tokens,
                         std::uint32_t first_target) {
    const std::vector<float> one  = engine.score_tokens(tokens, first_target, decode(1), nullptr);
    const std::vector<float> four = engine.score_tokens(tokens, first_target, decode(4), nullptr);
    const std::vector<float> three =
        engine.score_tokens(tokens, first_target, decode(3), nullptr);
    const std::vector<float> again = engine.score_tokens(tokens, first_target, decode(4), nullptr);
    int failures = within_criterion("decode_windows_agree T=4 vs T=1", four, one);
    failures += within_criterion("decode_windows_agree T=3 vs T=1", three, one);
    if (again != four) {
        std::cerr << "decode_windows_agree: a repeated T=4 window inherited prior State/KV\n";
        ++failures;
    }
    return failures;
}

// The sink sees each target position in [first_target, n) exactly once, in order, and its logits
// give the log-probabilities score_tokens returns.
int row_sink_sees_every_target(ninfer::Engine& engine, const std::vector<ninfer::TokenId>& tokens,
                               std::uint32_t first_target) {
    int failures = 0;
    for (const int window : {1, 3}) {
        RecordingSink sink;
        sink.tokens = &tokens;
        const std::vector<float> logprobs =
            engine.score_tokens(tokens, first_target, decode(window), &sink);
        const std::string label = "row_sink_sees_every_target T=" + std::to_string(window);
        std::vector<std::int64_t> expected;
        for (std::size_t i = first_target; i < tokens.size(); ++i) {
            expected.push_back(static_cast<std::int64_t>(i));
        }
        if (sink.positions != expected) {
            std::cerr << label << ": the sink did not see each target exactly once, in order\n";
            ++failures;
            continue;
        }
        if (!sink.finite || sink.vocabulary == 0) {
            std::cerr << label << ": the sink received invalid logits\n";
            ++failures;
            continue;
        }
        std::vector<float> from_sink(sink.target_logprobs.begin(), sink.target_logprobs.end());
        failures += within_criterion((label + " sink logits").c_str(), from_sink, logprobs);
    }
    return failures;
}

int run() {
    const char* artifact = std::getenv("NINFER_TEST_ARTIFACT");
    if (artifact == nullptr || *artifact == '\0') {
        std::cout << "SKIP: NINFER_TEST_ARTIFACT is not set\n";
        return 77;
    }

    ninfer::EngineOptions options;
    options.artifact_path         = artifact;
    options.purpose               = ninfer::EnginePurpose::CausalScoring;
    options.max_context           = 2048;
    options.kv_cache              = ninfer::KvCacheStorage::Fp8E4M3Row256;
    options.scoring_decode_window = kDecodeWindow;
    ninfer::Engine engine(options);
    const auto& effective = engine.options();
    if (effective.max_concurrency != 1 || effective.prefill_chunk != 1024 ||
        effective.kv_capacity.mode != ninfer::KvCapacityMode::Explicit ||
        effective.kv_capacity.explicit_tokens != effective.max_context ||
        effective.context_cache.enabled ||
        effective.speculative.backend != ninfer::SpeculativeBackend::None ||
        effective.kv_cache != ninfer::KvCacheStorage::Fp8E4M3Row256 ||
        effective.scoring_decode_window != kDecodeWindow) {
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
    for (const float value : all) {
        if (!std::isfinite(value) || value > 0.0F) {
            std::cerr << "causal scoring returned an invalid log probability\n";
            return 1;
        }
    }
    for (std::size_t i = 0; i < suffix.size(); ++i) {
        if (!std::isfinite(suffix[i]) || suffix[i] > 0.0F) {
            std::cerr << "causal scoring returned a non-finite logprob\n";
            return 1;
        }
        if (suffix[i] != repeated[i]) {
            std::cerr << "a repeated score window inherited prior State/KV\n";
            return 1;
        }
    }

    // Decode scoring: a 769-token window whose first 512 targets are the prompt half, the last 256
    // decoded (64 windows of T=4, 86 of T=3 with a one-column tail).
    const std::vector<ninfer::TokenId> half(tokens.begin(), tokens.begin() + 769);
    constexpr std::uint32_t kFirstTarget = 513;
    int failures                         = 0;
    failures += decode_phase_matches_prefill(engine, half, kFirstTarget);
    failures += decode_windows_agree(engine, half, kFirstTarget);
    failures += row_sink_sees_every_target(engine, half, kFirstTarget);
    failures += expect_invalid("a Decode window wider than the planned one", [&] {
        return engine.score_tokens(half, kFirstTarget, decode(kDecodeWindow + 1), nullptr);
    });
    failures += expect_invalid("a row sink with the Prefill phase", [&] {
        RecordingSink sink;
        sink.tokens = &half;
        return engine.score_tokens(half, kFirstTarget, ScoringOptions{}, &sink);
    });
    if (failures != 0) { return 1; }
    std::cout << "OK causal_score_real\n";
    return 0;
}

} // namespace

NINFER_GUARDED_TEST_MAIN(run)
