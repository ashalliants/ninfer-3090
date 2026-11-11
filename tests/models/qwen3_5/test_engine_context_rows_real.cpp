// Draft-head context rows (--draft-context-rows) on a real artifact: greedy output must be the
// same ids with the option off and on, one request at a time, and the run reports both acceptance
// rates. Drafts only, so a difference in output is a bug; the acceptance change is the
// measurement, not a pass condition.
//
// The row set is one union across every lane (context_rows.h: "one fixed-capacity set shared by
// every lane"). This test originally also compared two concurrent requests the same way, and found
// (RTX 3090, Vast, 2026-11-10) that a second request still changes the first's greedy output even
// though one request at a time stays bit for bit identical: the real mechanism is not yet
// understood. Startup now refuses --draft-context-rows above one lane
// (planning/startup.cpp:validate_target_options) until that is fixed or the table becomes
// per-lane; this test instead confirms the refusal.
//
// NINFER_TEST_ARTIFACT names the Qwen3.8-27B or Qwen3.6-35B-A3B artifact; NINFER_CONTEXT_ROWS_SPEC
// selects mtp (default), dflash or dflash2.
#include "guarded_main.h"
#include "ninfer/engine.h"

#include <array>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

namespace {

ninfer::SpeculativeBackend backend() {
    const char* value = std::getenv("NINFER_CONTEXT_ROWS_SPEC");
    const std::string_view spec = value == nullptr ? "mtp" : value;
    if (spec == "dflash") { return ninfer::SpeculativeBackend::DFlash; }
    if (spec == "dflash2") { return ninfer::SpeculativeBackend::DFlash2; }
    return ninfer::SpeculativeBackend::Mtp;
}

ninfer::EngineOptions engine_options(const char* artifact, std::uint32_t context_rows,
                                     std::uint32_t max_concurrency) {
    ninfer::EngineOptions options;
    options.artifact_path             = artifact;
    options.max_context               = 4096;
    options.kv_capacity               = ninfer::KvCapacityPolicy::explicit_capacity(4096 * max_concurrency);
    options.prefill_chunk             = 1024;
    options.max_concurrency           = max_concurrency;
    options.kv_cache                  = ninfer::KvCacheStorage::Int8Group64;
    options.speculative.backend       = backend();
    options.speculative.draft_tokens  = backend() == ninfer::SpeculativeBackend::Mtp ? 3 : 7;
    options.speculative.proposal_head = ninfer::ProposalHead::Optimized;
    options.use_cuda_graph            = true;
    options.draft_context_rows        = context_rows;
    return options;
}

ninfer::RequestOptions greedy(std::uint32_t outputs) {
    ninfer::RequestOptions options;
    options.execution.requested_output_tokens = outputs;
    options.execution.sampling.temperature    = 0.0F;
    options.execution.sampling.seed           = 0;
    options.execution.allow_prefix_reuse      = false;
    options.stop.include_model_defaults       = false;
    return options;
}

// Prompts whose continuations repeat rare names and identifiers they introduce.
const std::vector<std::string>& prompts() {
    static const std::vector<std::string> value{
        "The expedition was led by Xochiquetzal Abernathy-Okonkwo and her deputy Thorvaldur "
        "Quisumbing. In the logbook, every entry is signed by Xochiquetzal Abernathy-Okonkwo or "
        "Thorvaldur Quisumbing. Write the next five logbook entries, each signed by name.\n",
        "struct FrobnicatorQuuxConfig { int zyzzyvaThreshold; float quibbleFactorKappa; };\n"
        "// Write a C function `void frobnicateQuuxWithZyzzyva(FrobnicatorQuuxConfig* cfg)` that "
        "reads cfg->zyzzyvaThreshold and cfg->quibbleFactorKappa four times each:\n",
    };
    return value;
}

struct Totals {
    std::uint64_t drafted  = 0;
    std::uint64_t accepted = 0;
    void add(const ninfer::GenerationResult& result) {
        drafted += result.speculative.drafted_tokens;
        accepted += result.speculative.accepted_tokens;
    }
    [[nodiscard]] double rate() const {
        return drafted == 0 ? 0.0 : static_cast<double>(accepted) / static_cast<double>(drafted);
    }
};

struct Run {
    std::vector<std::vector<ninfer::TokenId>> sequential;
    std::vector<std::vector<ninfer::TokenId>> concurrent; // empty when max_concurrency == 1
    Totals totals;
};

Run run(const char* artifact, std::uint32_t context_rows, std::uint32_t max_concurrency) {
    ninfer::Engine engine(engine_options(artifact, context_rows, max_concurrency));
    Run out;
    for (const std::string& text : prompts()) {
        const auto result =
            engine.generate(engine.prepare_tokens(engine.tokenize_text(text)), greedy(192));
        out.sequential.push_back(result.generated_token_ids);
        out.totals.add(result);
    }
    if (max_concurrency < 2) { return out; }
    // Both prompts at once: one lane's ids are context rows for the other's drafts too. Only
    // reachable with context_rows == 0: startup refuses this combination above one lane otherwise.
    auto first  = engine.submit(engine.prepare_tokens(engine.tokenize_text(prompts()[0])),
                                greedy(192));
    auto second = engine.submit(engine.prepare_tokens(engine.tokenize_text(prompts()[1])),
                                greedy(192));
    std::array<ninfer::GenerationResult, 2> results;
    std::exception_ptr error;
    std::thread waiter([&] {
        try {
            results[1] = second.wait();
        } catch (...) { error = std::current_exception(); }
    });
    results[0] = first.wait();
    waiter.join();
    if (error) { std::rethrow_exception(error); }
    for (const auto& result : results) {
        out.concurrent.push_back(result.generated_token_ids);
        out.totals.add(result);
    }
    return out;
}

int run_test() {
    const char* artifact = std::getenv("NINFER_TEST_ARTIFACT");
    if (artifact == nullptr || *artifact == '\0') {
        std::cout << "skip: NINFER_TEST_ARTIFACT is not set\n";
        return 77;
    }
    const Run off = run(artifact, 0, 2);
    // --draft-context-rows is refused above one lane (planning/startup.cpp): only sequential
    // generation is reachable for "on", so it is compared against off's sequential results only.
    const Run on = run(artifact, 512, 1);
    int failures = 0;
    if (on.sequential != off.sequential) {
        std::cerr << "greedy output changed with --draft-context-rows 512 (one request at a time)\n";
        ++failures;
    }
    bool refused = false;
    try {
        ninfer::Engine rejected(engine_options(artifact, 512, 2));
        (void)rejected;
    } catch (const std::invalid_argument& error) {
        refused = std::string_view(error.what()).find("--draft-context-rows") != std::string_view::npos;
    }
    if (!refused) {
        std::cerr << "--draft-context-rows above one lane was not refused at startup\n";
        ++failures;
    }
    std::cout << "acceptance: off " << off.totals.accepted << "/" << off.totals.drafted << " ("
              << off.totals.rate() * 100.0 << "%), on " << on.totals.accepted << "/"
              << on.totals.drafted << " (" << on.totals.rate() * 100.0 << "%)\n";
    if (failures == 0) { std::cout << "ok\n"; }
    return failures == 0 ? 0 : 1;
}

} // namespace

NINFER_GUARDED_TEST_MAIN(run_test)
