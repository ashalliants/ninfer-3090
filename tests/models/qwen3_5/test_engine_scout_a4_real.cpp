#include "ninfer/engine.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

// scout_a4_kept_overlap: the int4 scout's gate on its ranking (docs/maintainer/lossy-long-prefill.md,
// "Int4 scout GEMMs"). The same 32K and 100K haystack prompts, each with a pass key at half depth,
// are scouted once with the int8 scout (the integer-A8 prefill routes, --prefill-a8) and once with
// --lossy-prefill-scout-a4. The blocks each chose by score (the forced head and recent blocks left
// out, ruling R11: both scouts keep those identically) are compared by Jaccard index against the
// spec gate of 0.9, and both must keep every token of the pass key. Ruling R16: the int4 scout
// missed the gate on the 3090 (0.84, 2026-10-25) and stays opt-in, so the gate is reported, not
// asserted; the test fails only below a 0.5 sanity floor (a broken int4 route, not a weaker one). The Jaccard over all kept blocks is
// printed for reference. rk4v4 KV as in production.
//
// The prompts are raw tokens (prepare_tokens) built from separately tokenized pieces, so the pass
// key's offsets are known exactly; with no product prompt the whole prompt is the filtered part.
namespace {

constexpr std::uint32_t kContext       = 110592;
constexpr double kJaccardGate          = 0.9; // the spec gate: reported only (R16)
constexpr double kJaccardFloor         = 0.5; // asserted: below it the int4 route is broken
constexpr const char* kPassKey         = "68213";
constexpr std::uint32_t kTargets[]     = {32768, 100000};

void require(bool condition, const std::string& message) {
    if (!condition) { throw std::runtime_error(message); }
}

ninfer::EngineOptions scout_options(const char* artifact, bool a4) {
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
    options.prefill_a8                        = true; // the int8 scout is the A8 routes
    options.lossy_prefill.min_tokens          = 16384;
    options.lossy_prefill.scout_a4            = a4;
    options.speculative.backend               = ninfer::SpeculativeBackend::None;
    return options;
}

struct Haystack {
    std::vector<ninfer::TokenId> tokens;
    // The pass key's digits: tokens [key_begin, key_end) of the prompt.
    std::uint32_t key_begin = 0;
    std::uint32_t key_end   = 0;
};

void append(std::vector<ninfer::TokenId>& out, const std::vector<ninfer::TokenId>& piece) {
    out.insert(out.end(), piece.begin(), piece.end());
}

// Inventory records (varied numbers, so no two blocks look alike) up to about `target` tokens, the
// pass key at half depth and the question last.
Haystack build_haystack(const ninfer::Engine& engine, std::uint32_t target) {
    Haystack h;
    append(h.tokens, engine.tokenize_text("You answer questions about the inventory records.\n"));
    const auto question =
        engine.tokenize_text("\nWhat is the secret pass key? Answer with the number only.\n");
    bool placed = false;
    for (int i = 0; h.tokens.size() + question.size() < target; ++i) {
        if (!placed && h.tokens.size() >= target / 2) {
            append(h.tokens, engine.tokenize_text("The secret pass key is "));
            h.key_begin = static_cast<std::uint32_t>(h.tokens.size());
            append(h.tokens, engine.tokenize_text(kPassKey));
            h.key_end = static_cast<std::uint32_t>(h.tokens.size());
            append(h.tokens, engine.tokenize_text(". Remember it: it is the pass key.\n"));
            placed = true;
        }
        append(h.tokens, engine.tokenize_text(
                             "Record " + std::to_string(i) + ": shelf " +
                             std::to_string((i * 37) % 101) + " holds " +
                             std::to_string((i * 53) % 997) + " units of part " +
                             std::to_string(1000 + i) + ".\n"));
    }
    append(h.tokens, question);
    require(placed && h.key_end > h.key_begin, "the pass key was not placed");
    return h;
}

struct Scouted {
    std::uint32_t uncached = 0;
    std::uint32_t exact_end = 0;
    std::uint32_t kept     = 0;
    double scout_ms        = 0.0;
    std::vector<std::array<std::uint32_t, 2>> runs;
};

Scouted scout(ninfer::Engine& engine, const Haystack& h, const std::string& label) {
    ninfer::RequestOptions request;
    request.execution.requested_output_tokens = 1;
    request.execution.sampling.temperature    = 0.0F;
    request.execution.allow_prefix_reuse      = false;
    request.stop.include_model_defaults       = false;
    auto prompt          = engine.prepare_tokens(h.tokens);
    const auto tokens    = prompt.summary().prompt_tokens;
    const auto result    = engine.generate(std::move(prompt), request);
    require(tokens == h.tokens.size(), label + ": the prepared prompt changed length");
    require(result.lossy_prefill &&
                result.lossy_prefill->outcome == ninfer::LossyPrefillOutcome::Completed,
            label + ": the lossy prefill path was not taken (or kept everything)");
    const auto& report = *result.lossy_prefill;
    require(report.uncached_tokens <= tokens && report.kept_tokens < report.uncached_tokens,
            label + ": the lossy accounting does not show a thinned prompt");
    std::uint32_t covered = 0;
    for (const auto& run : report.kept_runs) {
        require(run[0] < run[1] && run[1] <= report.uncached_tokens, label + ": a bad kept run");
        covered += run[1] - run[0];
    }
    require(covered == report.kept_tokens, label + ": the kept runs do not add up to kept_tokens");
    return {report.uncached_tokens, tokens - report.uncached_tokens, report.kept_tokens,
            report.scout_ms, report.kept_runs};
}

// Blocks of `block` tokens (the last one clipped to the span) that the runs keep whole. With
// `scored_only`, the blocks the head and recent rules force are left out, so only the blocks the
// scores chose are compared.
std::set<std::uint32_t> kept_blocks(const Scouted& s, std::uint32_t block, bool scored_only,
                                    std::uint32_t head, std::uint32_t recent) {
    std::set<std::uint32_t> out;
    for (const auto& run : s.runs) {
        for (std::uint32_t b = (run[0] + block - 1) / block; b * block < run[1]; ++b) {
            const std::uint32_t lo = b * block;
            const std::uint32_t hi = std::min(lo + block, s.uncached);
            if (hi > run[1]) { break; }
            if (scored_only && (lo < head || hi + recent > s.uncached)) { continue; }
            out.insert(b);
        }
    }
    return out;
}

// Nothing on either side measures nothing: nullopt, which the gate treats as a failure.
std::optional<double> jaccard(const std::set<std::uint32_t>& a, const std::set<std::uint32_t>& b) {
    std::size_t both = 0;
    for (const auto v : a) { both += b.count(v); }
    const std::size_t either = a.size() + b.size() - both;
    if (either == 0) { return std::nullopt; }
    return static_cast<double>(both) / static_cast<double>(either);
}

std::string shown(std::optional<double> v) { return v ? std::to_string(*v) : "not measured"; }

bool keeps(const Scouted& s, std::uint32_t begin, std::uint32_t end) {
    for (std::uint32_t t = begin; t < end; ++t) {
        const std::uint32_t offset = t - s.exact_end;
        const bool inside = std::any_of(s.runs.begin(), s.runs.end(), [&](const auto& run) {
            return run[0] <= offset && offset < run[1];
        });
        if (!inside) { return false; }
    }
    return true;
}

int exercise(const char* artifact) {
    std::vector<Haystack> prompts;
    std::vector<Scouted> int8;
    std::vector<Scouted> int4;
    ninfer::LossyPrefillOptions shape;
    {
        // One engine at a time: one 3090 holds one.
        ninfer::Engine engine(scout_options(artifact, false));
        shape = engine.options().lossy_prefill;
        for (const auto target : kTargets) { prompts.push_back(build_haystack(engine, target)); }
        for (const auto& h : prompts) {
            int8.push_back(scout(engine, h, "int8 " + std::to_string(h.tokens.size())));
        }
    }
    {
        ninfer::Engine engine(scout_options(artifact, true));
        for (const auto& h : prompts) {
            int4.push_back(scout(engine, h, "int4 " + std::to_string(h.tokens.size())));
        }
    }
    int failures  = 0;
    bool gate_met = true;
    for (std::size_t i = 0; i < prompts.size(); ++i) {
        const Haystack& h   = prompts[i];
        const Scouted& a    = int8[i];
        const Scouted& b    = int4[i];
        const std::string n = std::to_string(h.tokens.size()) + " tokens";
        require(a.uncached == b.uncached && a.exact_end == b.exact_end,
                n + ": the two scouts filtered different spans");
        require(h.key_begin >= a.exact_end, n + ": the pass key is outside the filtered part");
        const std::optional<double> all = jaccard(
            kept_blocks(a, shape.block_tokens, false, shape.head_tokens, shape.recent_tokens),
            kept_blocks(b, shape.block_tokens, false, shape.head_tokens, shape.recent_tokens));
        const std::optional<double> scored = jaccard(
            kept_blocks(a, shape.block_tokens, true, shape.head_tokens, shape.recent_tokens),
            kept_blocks(b, shape.block_tokens, true, shape.head_tokens, shape.recent_tokens));
        const bool key8 = keeps(a, h.key_begin, h.key_end);
        const bool key4 = keeps(b, h.key_begin, h.key_end);
        std::cout << n << ": kept int8=" << a.kept << " int4=" << b.kept << " of " << a.uncached
                  << "; kept-block Jaccard over score-chosen blocks " << shown(scored)
                  << " (gated >= " << kJaccardGate << "), all kept blocks " << shown(all)
                  << " (reference); pass key kept int8="
                  << key8 << " int4=" << key4 << "; scout int8 " << a.scout_ms << " ms, int4 "
                  << b.scout_ms << " ms (" << a.scout_ms / b.scout_ms << "x)\n";
        if (!scored) {
            std::cerr << n << ": neither scout kept a score-chosen block, so the overlap is not "
                         "measured\n";
            ++failures;
        } else {
            if (!(*scored >= kJaccardGate)) { gate_met = false; }
            if (!(*scored > kJaccardFloor)) {
                std::cerr << n << ": score-chosen kept-block Jaccard " << *scored
                          << " is at or below the sanity floor " << kJaccardFloor << '\n';
                ++failures;
            }
        }
        if (!key8 || !key4) {
            std::cerr << n << ": the pass key was dropped by the "
                      << (!key8 && !key4 ? "int8 and int4" : !key8 ? "int8" : "int4")
                      << " scout\n";
            ++failures;
        }
    }
    std::cout << (gate_met ? "GATE MET" : "GATE NOT MET (int4 scout stays opt-in)")
              << ": score-chosen kept-block Jaccard >= " << kJaccardGate << '\n';
    std::cout << (failures == 0 ? "OK" : "FAIL") << " scout_a4_kept_overlap\n";
    return failures == 0 ? 0 : 1;
}

} // namespace

int main() {
    const char* artifact = std::getenv("NINFER_TEST_ARTIFACT");
    if (!artifact || !*artifact) {
        std::cout << "skip: NINFER_TEST_ARTIFACT is not set\n";
        return 77;
    }
    try {
        return exercise(artifact);
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
