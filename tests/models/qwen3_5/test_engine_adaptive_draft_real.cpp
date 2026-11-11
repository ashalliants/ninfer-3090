#include "ninfer/engine.h"
#include "kv_cache_storage.h"

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

// Adaptive DFlash2 draft window through the Engine: both captured round widths run, on CUDA Graph
// replay or eagerly (argv[1]), alone and beside a second lane, and a new request starts on the
// configured window however the previous one ended. Needs the real artifact (NINFER_TEST_ARTIFACT).
namespace {
constexpr std::uint32_t kWide   = 7;
constexpr std::uint32_t kNarrow = 3;

void require(bool condition, const char* message) {
    if (!condition) { throw std::runtime_error(message); }
}

ninfer::RequestOptions request(std::uint32_t outputs, bool hot) {
    ninfer::RequestOptions options;
    options.execution.requested_output_tokens = outputs;
    options.execution.sampling.seed           = 7;
    options.stop.include_model_defaults       = false;
    if (hot) {
        // Sampling at a high temperature makes most drafts miss, which is the workload the narrow
        // window exists for; greedy decoding of a counting prompt accepts nearly all of them.
        options.execution.sampling.temperature = 1.8F;
        options.execution.sampling.top_k       = 200;
    } else {
        options.execution.sampling.temperature = 0.0F;
    }
    return options;
}

void valid(const ninfer::GenerationResult& result, std::size_t outputs) {
    const auto& stats = result.speculative;
    require(result.generated_token_ids.size() == outputs &&
                result.finish_reason == ninfer::FinishReason::OutputLimit,
            "adaptive DFlash2 did not honor the requested output budget");
    require(stats.backend == ninfer::SpeculativeBackend::DFlash2 && stats.rounds != 0,
            "generation bypassed DFlash2");
    require(stats.adaptive_draft_tokens == kNarrow, "statistics do not report the narrow window");
    require(stats.adaptive_low_rounds <= stats.rounds, "more narrow rounds than rounds");
}
} // namespace

int main(int argc, char** argv) {
    const char* artifact = std::getenv("NINFER_TEST_ARTIFACT");
    if (!artifact || !*artifact) {
        std::cout << "skip: NINFER_TEST_ARTIFACT is not set\n";
        return 77;
    }
    try {
        const bool graph = argc > 1 ? std::stoi(argv[1]) != 0 : true;
        ninfer::EngineOptions options;
        options.artifact_path                    = artifact;
        options.max_context                      = 2304;
        options.kv_capacity                      = ninfer::KvCapacityPolicy::explicit_capacity(2304 * 2);
        options.prefill_chunk                    = 2304;
        options.max_concurrency                  = 2;
        options.use_cuda_graph                   = graph;
        options.kv_cache                         = ninfer::test::parse_kv_cache_storage("int8");
        options.context_cache.device_state_slots = 1;
        options.speculative.backend              = ninfer::SpeculativeBackend::DFlash2;
        options.speculative.draft_tokens         = kWide;
        options.speculative.proposal_head        = ninfer::ProposalHead::Optimized;
        options.speculative.adaptive_draft_tokens = kNarrow;
        ninfer::Engine engine(options);
        const auto prompt = engine.tokenize_text("Count from one to twenty: one, two, three,");

        // One lane, low acceptance: the first round runs the wide window and the policy then moves
        // to the narrow one, so both captured widths ran.
        const auto hot = engine.generate(engine.prepare_tokens(prompt), request(256, true));
        valid(hot, 256);
        require(hot.speculative.adaptive_low_rounds != 0, "the narrow window never ran");
        require(hot.speculative.adaptive_low_rounds < hot.speculative.rounds,
                "the wide window never ran");

        // The next request starts on the configured window even while the previous one's dwell on
        // the narrow one is unexpired: three tokens fit in one round, which must be the wide one.
        const auto fresh = engine.generate(engine.prepare_tokens(prompt), request(3, false));
        valid(fresh, 3);
        require(fresh.speculative.adaptive_low_rounds == 0,
                "a new request did not start on the configured window");

        // Two lanes share every round and therefore one window: both finish with statistics in
        // range.
        std::vector<ninfer::GenerationHandle> handles;
        handles.push_back(engine.submit(engine.prepare_tokens(prompt), request(96, true)));
        handles.push_back(engine.submit(engine.prepare_tokens(prompt), request(96, false)));
        for (auto& handle : handles) { valid(handle.wait(), 96); }

        std::cout << "ok graph=" << graph << " narrow_rounds=" << hot.speculative.adaptive_low_rounds
                  << "/" << hot.speculative.rounds << '\n';
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}
