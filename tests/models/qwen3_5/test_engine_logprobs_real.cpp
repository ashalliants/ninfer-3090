// Token log-probabilities on the real artifact, under every decode backend.
//
// For each of ordinary decode, MTP and DFlash2 with n-gram copy rounds, at greedy sampling:
//
//   alignment   GenerationResult::token_logprobs has one entry per generated token, in step with
//               generated_token_ids, and each entry's token is the generated token
//   greedy      a greedy token is the model's most likely one, so at every position the sampled
//               entry equals the first alternative (same token, same logprob). This is what ties a
//               speculative round's verify column to its licensed token: a misplaced column would
//               show another position's distribution
//   shape       alternatives are best first, finite, and their probabilities sum to at most one
//   unchanged   asking for logprobs does not change the generated tokens
//   speculative the speculative modes ran rounds, accepted drafts and also rejected some (a
//               truncated round commits only a prefix of its columns), and the n-gram mode ran copy
//               rounds (whose width differs from the draft model's), so the mapping was exercised
//               at more than one verify width
//   lanes       a request that asks and one that does not run side by side, and each equals its
//               solo run
//
// Arguments: [kv-dtype (default rk4v4)].
#include "ninfer/engine.h"
#include "kv_cache_storage.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace {

void require(bool condition, const std::string& message) {
    if (!condition) { throw std::runtime_error(message); }
}

ninfer::ChatMessage message(ninfer::ChatRole role, std::string text) {
    ninfer::ChatMessage result;
    result.role = role;
    result.parts.push_back(ninfer::MessagePart{
        .kind = ninfer::MessagePartKind::Text, .text = std::move(text), .media = {}});
    return result;
}

ninfer::PromptInput prompt(std::string text) {
    ninfer::PromptInput input;
    input.options.enable_thinking = false;
    input.messages.push_back(message(ninfer::ChatRole::User, std::move(text)));
    return input;
}

// Numbered lines the model copies: the n-gram mode proposes copies from them, at its copy width.
std::string repeated_text() {
    std::ostringstream out;
    out << "Repeat this list exactly, with no other words:\n";
    for (int i = 1; i <= 12; ++i) {
        out << "item " << i << ": the pump reads sensor " << i * 3 << "\n";
    }
    return out.str();
}

constexpr std::uint32_t kBudget = 96;

ninfer::RequestOptions greedy(std::optional<std::uint32_t> logprobs_top_k) {
    ninfer::RequestOptions options;
    options.execution.requested_output_tokens = kBudget;
    options.execution.sampling.temperature    = 0.0F;
    options.execution.allow_prefix_reuse      = false;
    options.execution.logprobs_top_k          = logprobs_top_k;
    return options;
}

enum class Mode { Ordinary, Mtp, DFlash2Ngram };

const char* name(Mode mode) {
    switch (mode) {
    case Mode::Ordinary:
        return "ordinary";
    case Mode::Mtp:
        return "mtp";
    case Mode::DFlash2Ngram:
        return "dflash2+ngram";
    }
    return "";
}

ninfer::EngineOptions engine_options(const char* artifact, ninfer::KvCacheStorage kv, Mode mode,
                                     std::uint32_t lanes) {
    ninfer::EngineOptions options;
    options.artifact_path   = artifact;
    options.max_context     = 8192;
    options.kv_capacity     = ninfer::KvCapacityPolicy::explicit_capacity(8192);
    options.prefill_chunk   = 2048;
    options.max_concurrency = lanes;
    options.kv_cache        = kv;
    if (mode == Mode::Mtp) {
        options.speculative.backend       = ninfer::SpeculativeBackend::Mtp;
        options.speculative.draft_tokens  = 3;
        options.speculative.proposal_head = ninfer::ProposalHead::Optimized;
    } else if (mode == Mode::DFlash2Ngram) {
        options.speculative.backend            = ninfer::SpeculativeBackend::DFlash2;
        options.speculative.draft_tokens       = 7;
        options.speculative.proposal_head      = ninfer::ProposalHead::Optimized;
        options.speculative.ngram_draft_tokens = 15;
    }
    return options;
}

void check_entries(const ninfer::GenerationResult& result, std::uint32_t top,
                   const std::string& label) {
    const auto& ids = result.generated_token_ids;
    require(!ids.empty(), label + ": nothing was generated");
    require(result.token_logprobs.size() == ids.size(),
            label + ": " + std::to_string(result.token_logprobs.size()) + " logprob entries for " +
                std::to_string(ids.size()) + " tokens");
    for (std::size_t index = 0; index < ids.size(); ++index) {
        const ninfer::TokenLogprob& entry = result.token_logprobs[index];
        const std::string where           = label + ": token " + std::to_string(index);
        require(entry.sampled.token == ids[index], where + " is not the entry's token");
        require(entry.top.size() == top, where + " has the wrong number of alternatives");
        require(std::isfinite(entry.sampled.logprob) && entry.sampled.logprob <= 1.0e-4F,
                where + " has a logprob outside (-inf, 0]");
        require(!entry.sampled.bytes.empty() || index + 1 == ids.size(), where + " has no bytes");
        double mass = 0.0;
        for (std::size_t rank = 0; rank < entry.top.size(); ++rank) {
            require(std::isfinite(entry.top[rank].logprob),
                    where + " has a non-finite alternative");
            if (rank > 0) {
                require(entry.top[rank - 1].logprob >= entry.top[rank].logprob,
                        where + " has alternatives out of order");
            }
            mass += std::exp(static_cast<double>(entry.top[rank].logprob));
        }
        require(mass <= 1.0 + 1.0e-3, where + " has alternatives whose probabilities exceed one");
        // Greedy sampling takes the most likely token, so the sampled entry is the first
        // alternative. A verify column that belonged to another position would break this.
        require(entry.top[0].token == entry.sampled.token &&
                    entry.top[0].logprob == entry.sampled.logprob &&
                    entry.top[0].bytes == entry.sampled.bytes,
                where + " is not the model's most likely token under greedy sampling");
    }
}

void check_mode(const char* artifact, ninfer::KvCacheStorage kv, Mode mode) {
    const std::string label = name(mode);
    ninfer::Engine engine(engine_options(artifact, kv, mode, 2));
    const std::vector<ninfer::PromptInput> prompts = {
        prompt("Write a short numbered list of five fruits, one per line."),
        prompt("Count from 1 to 20 separated by commas."), prompt(repeated_text())};
    constexpr std::uint32_t kTop = 3;

    std::vector<std::vector<ninfer::TokenId>> plain_ids;
    std::uint64_t rounds = 0, drafted = 0, accepted = 0, copy_rounds = 0;
    for (std::size_t index = 0; index < prompts.size(); ++index) {
        const std::string where = label + " prompt " + std::to_string(index);
        const auto plain = engine.generate(engine.prepare(prompts[index]), greedy(std::nullopt));
        require(plain.token_logprobs.empty(), where + ": logprobs appeared without being asked");
        const auto asked = engine.generate(engine.prepare(prompts[index]), greedy(kTop));
        require(asked.generated_token_ids == plain.generated_token_ids,
                where + ": asking for logprobs changed the output");
        check_entries(asked, kTop, where);
        plain_ids.push_back(plain.generated_token_ids);
        rounds += asked.speculative.rounds;
        drafted += asked.speculative.drafted_tokens;
        accepted += asked.speculative.accepted_tokens;
        copy_rounds += asked.speculative.ngram_rounds;
        std::cout << where << ": " << asked.generated_token_ids.size() << " tokens, rounds "
                  << asked.speculative.rounds << ", accepted " << asked.speculative.accepted_tokens
                  << ", copy rounds " << asked.speculative.ngram_rounds << '\n';
    }
    if (mode != Mode::Ordinary) {
        require(rounds > 0 && accepted > 0, label + ": no speculative round accepted a draft");
        // A round that keeps only part of its drafts commits a prefix of its licensed columns:
        // the truncation the row/stride mapping has to get right.
        require(drafted > accepted,
                label + ": every draft was accepted, so no round was truncated");
    }
    if (mode == Mode::DFlash2Ngram) {
        require(copy_rounds > 0, label + ": no copy round ran, so only one verify width was tried");
    }

    // A request that asks and one that does not, at once.
    ninfer::GenerationResult asking, quiet;
    std::exception_ptr failure;
    std::thread other([&] {
        try {
            quiet = engine.generate(engine.prepare(prompts[1]), greedy(std::nullopt));
        } catch (...) { failure = std::current_exception(); }
    });
    try {
        asking = engine.generate(engine.prepare(prompts[0]), greedy(kTop));
    } catch (...) {
        other.join();
        throw;
    }
    other.join();
    if (failure) { std::rethrow_exception(failure); }
    check_entries(asking, kTop, label + " with a neighbour");
    require(asking.generated_token_ids == plain_ids[0],
            label + ": a logprob request beside another differs from its solo run");
    require(quiet.token_logprobs.empty() && quiet.generated_token_ids == plain_ids[1],
            label + ": the request beside a logprob request was disturbed");
}

} // namespace

int main(int argc, char** argv) {
    const char* artifact = std::getenv("NINFER_TEST_ARTIFACT");
    if (!artifact || !*artifact) {
        std::cout << "skip: NINFER_TEST_ARTIFACT is not set\n";
        return 77;
    }
    try {
        const auto kv = ninfer::test::parse_kv_cache_storage(argc > 1 ? argv[1] : "rk4v4");
        for (const Mode mode : {Mode::Ordinary, Mode::Mtp, Mode::DFlash2Ngram}) {
            check_mode(artifact, kv, mode);
        }
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    std::cout << "ok\n";
    return 0;
}
