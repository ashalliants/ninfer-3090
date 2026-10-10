// N-gram copy rounds beside DFlash2 on the real artifact (step 3, PR 3.3).
//
//   copy        "return this file exactly": with the file in the prompt, and only as a `cat -n`
//               numbered tool result (so only its de-numbered source can match), the output equals
//               non-speculative greedy output token for token, with CUDA Graphs on and off, and
//               copies were accepted
//   lifecycle   a stop token inside a copy round, an output budget ending mid-copy and a
//               cancellation mid-copy each leave a prefix of that output; the Engine then
//               reproduces it whole
//   restore     a follow-up turn bound from its retained session copies again and matches the same
//               turn prefilled from scratch (the context store does not take DFlash backends)
//   constrained a JSON-schema answer copying a JSON blob from a tool result validates, copies were
//               verified, and the blob's number, which the schema makes a string, never leaks
//
// Arguments: [kv-dtype (default rk4v4)]. Preemption of a copying lane needs two lanes, which n-gram
// copies do not run yet; it is covered when they do.
#include "ninfer/engine.h"
#include "kv_cache_storage.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

void require(bool condition, const std::string& message) {
    if (!condition) { throw std::runtime_error(message); }
}

// A realistic source file with enough distinct lines that the model has to copy it.
std::string source_file() {
    std::ostringstream out;
    out << "#include \"pump/controller.h\"\n\n#include <algorithm>\n#include <cstdint>\n\n"
        << "namespace plant::pump {\n\n";
    const char* names[] = {"inlet",  "outlet", "bypass", "drain",  "relief",
                           "intake", "filter", "return", "header", "spare"};
    for (int i = 0; i < 10; ++i) {
        out << "// Reads the " << names[i] << " sensor and clamps it to its rated range.\n"
            << "double read_" << names[i] << "_pressure(const Controller& controller) {\n"
            << "    const double raw = controller.sensor(" << i * 3 + 1 << ").sample();\n"
            << "    return std::clamp(raw * " << 0.25 * (i + 1) << ", " << 10 * i << ".0, "
            << 400 + 35 * i << ".0);\n"
            << "}\n\n";
    }
    out << "} // namespace plant::pump\n";
    return out.str();
}

// The `cat -n` layout an agent's Read tool returns.
std::string numbered(const std::string& text) {
    std::ostringstream out;
    std::istringstream in(text);
    std::string line;
    int number = 0;
    while (std::getline(in, line)) {
        std::string label = std::to_string(++number);
        out << std::string(6 - label.size(), ' ') << label << '\t' << line << '\n';
    }
    return out.str();
}

ninfer::ChatMessage message(ninfer::ChatRole role, std::string text) {
    ninfer::ChatMessage result;
    result.role = role;
    result.parts.push_back(
        ninfer::MessagePart{.kind = ninfer::MessagePartKind::Text, .text = std::move(text), .media = {}});
    return result;
}

constexpr const char* kReturnFile =
    "Return the file exactly as it is: no commentary, no code fences and no line numbers.";

ninfer::PromptInput copy_prompt(bool numbered_tool_result) {
    ninfer::PromptInput input;
    input.options.enable_thinking = false;
    if (numbered_tool_result) {
        input.messages.push_back(
            message(ninfer::ChatRole::User, std::string("Read src/pump/pressure.cpp. ") + kReturnFile));
        input.messages.push_back(message(ninfer::ChatRole::Tool, numbered(source_file())));
    } else {
        input.messages.push_back(message(ninfer::ChatRole::User,
                                         "Here is src/pump/pressure.cpp:\n\n" + source_file() +
                                             "\n" + kReturnFile));
    }
    return input;
}

ninfer::RequestOptions greedy(std::uint32_t tokens, bool reuse = false) {
    ninfer::RequestOptions options;
    options.execution.requested_output_tokens = tokens;
    options.execution.sampling.temperature    = 0.0F;
    options.execution.allow_prefix_reuse      = reuse;
    return options;
}

ninfer::EngineOptions engine_options(const char* artifact, ninfer::KvCacheStorage kv, bool ngram,
                                     bool graph) {
    ninfer::EngineOptions options;
    options.artifact_path   = artifact;
    options.max_context     = 8192;
    options.kv_capacity     = ninfer::KvCapacityPolicy::explicit_capacity(8192);
    options.prefill_chunk   = 2048;
    options.max_concurrency = 1;
    options.kv_cache        = kv;
    options.use_cuda_graph  = graph;
    if (ngram) {
        options.speculative.backend            = ninfer::SpeculativeBackend::DFlash2;
        options.speculative.draft_tokens       = 7;
        options.speculative.proposal_head      = ninfer::ProposalHead::Optimized;
        options.speculative.ngram_draft_tokens = 15;
    }
    return options;
}

std::size_t first_divergence(const std::vector<ninfer::TokenId>& a,
                             const std::vector<ninfer::TokenId>& b) {
    std::size_t index = 0;
    while (index < a.size() && index < b.size() && a[index] == b[index]) { ++index; }
    return index;
}

void require_same(const ninfer::GenerationResult& got, const std::vector<ninfer::TokenId>& expected,
                  const std::string& label) {
    const std::size_t at = first_divergence(got.generated_token_ids, expected);
    require(at == expected.size() && got.generated_token_ids.size() == expected.size(),
            label + ": output diverges from non-speculative greedy at token " + std::to_string(at) +
                " of " + std::to_string(expected.size()));
}

void require_prefix(const ninfer::GenerationResult& got,
                    const std::vector<ninfer::TokenId>& expected, const std::string& label) {
    require(got.generated_token_ids.size() <= expected.size() &&
                first_divergence(got.generated_token_ids, expected) ==
                    got.generated_token_ids.size(),
            label + ": output is not a prefix of the uninterrupted output");
}

// Requests cancellation once some content has been published.
class CancelAfter final : public ninfer::OutputSink {
public:
    explicit CancelAfter(std::size_t bytes) : bytes_(bytes) {}
    void start(ninfer::GenerationStart) override {}
    void progress(ninfer::PromptProgress) override {}
    void timing(ninfer::GenerationTimingObservation) override {}
    void publish(ninfer::OutputDelta delta) override {
        seen_ += delta.text.size();
        if (seen_ >= bytes_) { requested_ = true; }
    }
    [[nodiscard]] bool requested() const { return requested_.load(); }

private:
    std::size_t bytes_ = 0;
    std::size_t seen_  = 0;
    std::atomic<bool> requested_{false};
};

struct Reference {
    std::vector<ninfer::TokenId> plain;
    std::vector<ninfer::TokenId> numbered;
};

Reference reference_outputs(const char* artifact, ninfer::KvCacheStorage kv, std::uint32_t budget) {
    ninfer::Engine engine(engine_options(artifact, kv, false, true));
    Reference reference;
    reference.plain = engine.generate(engine.prepare(copy_prompt(false)), greedy(budget))
                          .generated_token_ids;
    reference.numbered = engine.generate(engine.prepare(copy_prompt(true)), greedy(budget))
                             .generated_token_ids;
    return reference;
}

void check_copies(ninfer::Engine& engine, const Reference& reference, std::uint32_t budget,
                  const std::string& label) {
    for (const bool numbered_source : {false, true}) {
        const std::string name = label + (numbered_source ? " numbered" : " plain");
        const auto& expected   = numbered_source ? reference.numbered : reference.plain;
        const auto result =
            engine.generate(engine.prepare(copy_prompt(numbered_source)), greedy(budget));
        require_same(result, expected, name);
        require(result.speculative.ngram_draft_tokens == 15 &&
                    result.speculative.ngram_accepted_tokens > 0 &&
                    result.speculative.ngram_rounds <= result.speculative.rounds &&
                    result.speculative.ngram_accepted_tokens <=
                        result.speculative.accepted_tokens,
                name + ": no copy was accepted, or the copy counters are not a subset");
        std::cout << name << ": " << expected.size() << " tokens, rounds "
                  << result.speculative.rounds << ", copy rounds "
                  << result.speculative.ngram_rounds << ", copies accepted "
                  << result.speculative.ngram_accepted_tokens << '/'
                  << result.speculative.ngram_drafted_tokens << '\n';
    }
}

void check_lifecycle(ninfer::Engine& engine, const Reference& reference, std::uint32_t budget) {
    const auto& expected = reference.plain;
    require(expected.size() > 80, "reference copy is too short for the lifecycle cases");
    // A stop token deep inside the copy: the first occurrence of a token not seen before it.
    std::optional<std::size_t> stop_at;
    for (std::size_t i = 60; i < expected.size() && !stop_at; ++i) {
        if (std::find(expected.begin(), expected.begin() + static_cast<std::ptrdiff_t>(i),
                      expected[i]) == expected.begin() + static_cast<std::ptrdiff_t>(i)) {
            stop_at = i;
        }
    }
    require(stop_at.has_value(), "no unique token to stop at");
    auto stopping = greedy(budget);
    stopping.stop.token_ids.push_back(expected[*stop_at]);
    const auto stopped = engine.generate(engine.prepare(copy_prompt(false)), stopping);
    require(stopped.finish_reason == ninfer::FinishReason::StopToken &&
                stopped.generated_token_ids.size() == *stop_at + 1,
            "a stop token inside a copy did not end the output at that token");
    require_prefix(stopped, expected, "stop token");

    for (const std::uint32_t limit : {37U, 61U}) {
        const auto limited = engine.generate(engine.prepare(copy_prompt(false)), greedy(limit));
        require(limited.finish_reason == ninfer::FinishReason::OutputLimit &&
                    limited.generated_token_ids.size() == limit,
                "an output budget ending mid-copy was not honored");
        require_prefix(limited, expected, "output budget " + std::to_string(limit));
    }

    CancelAfter sink(120);
    const auto cancelled =
        engine.generate(engine.prepare(copy_prompt(false)), greedy(budget), &sink,
                        ninfer::CancellationView([&sink] { return sink.requested(); }));
    require(cancelled.finish_reason == ninfer::FinishReason::Cancelled,
            "cancellation mid-copy did not cancel");
    require_prefix(cancelled, expected, "cancellation");
    const auto after = engine.generate(engine.prepare(copy_prompt(false)), greedy(budget));
    require_same(after, expected, "after cancellation");
    std::cout << "lifecycle: stop at " << *stop_at << ", cancelled after "
              << cancelled.generated_token_ids.size() << " tokens\n";
}

ninfer::PromptInput follow_up(const std::string& reply) {
    ninfer::PromptInput input = copy_prompt(false);
    input.messages.push_back(message(ninfer::ChatRole::Assistant, reply));
    input.messages.push_back(message(
        ninfer::ChatRole::User, "Now return the same file again, exactly as before."));
    input.context_cache.allow_engine_automatic_shared_prefixes = false;
    return input;
}

// The context store does not take the DFlash backends, so a restored session here is one the
// context cache retained from the previous turn: the follow-up turn binds from that checkpoint
// instead of prefilling, and its copies come from the prompt's index all the same.
void check_restore(ninfer::Engine& engine, std::uint32_t budget) {
    ninfer::PromptInput first = copy_prompt(false);
    first.context_cache.allow_engine_automatic_shared_prefixes = false;
    const std::string reply = engine.generate(engine.prepare(first), greedy(budget, true)).content;
    const auto restored = engine.generate(engine.prepare(follow_up(reply)), greedy(budget, true));
    require(restored.reused_prompt_tokens > 0,
            "the follow-up turn did not restore its retained session");
    require(restored.speculative.ngram_accepted_tokens > 0, "the restored turn accepted no copies");
    const auto fresh = engine.generate(engine.prepare(follow_up(reply)), greedy(budget));
    require(fresh.reused_prompt_tokens == 0, "the control turn reused a session");
    require_same(restored, fresh.generated_token_ids, "restored copy");
    std::cout << "restore: reused " << restored.reused_prompt_tokens << " tokens, copies accepted "
              << restored.speculative.ngram_accepted_tokens << '\n';
}

void check_constrained(ninfer::Engine& engine) {
    const std::string blob =
        R"({"name": "inlet pump", "pressure": 101325, "unit": "Pa", "serial": "PX-4471-B", )"
        R"("location": "north header, bay 7", "maintenance": "replace the shaft seal and )"
        R"(check the impeller clearance against the drawing, then flush the suction line, )"
        R"(torque the casing bolts in a star pattern to the value on the nameplate, refit the )"
        R"(coupling guard, run the pump for ten minutes at half speed while watching the seal for )"
        R"(drips, and record the discharge pressure, the motor current and the bearing )"
        R"(temperature in the maintenance log before handing the unit back to operations"})";
    ninfer::PromptInput input;
    input.options.enable_thinking = false;
    input.messages.push_back(message(
        ninfer::ChatRole::User,
        "Read the pump record and return it as JSON with the same keys and values."));
    input.messages.push_back(message(ninfer::ChatRole::Tool, blob));
    auto options = greedy(512);
    // The schema makes "pressure" a string, so copying the record's bare number is illegal there.
    options.constraint = ninfer::OutputConstraint::json_schema(nlohmann::json{
        {"type", "object"},
        {"properties",
         {{"name", {{"type", "string"}}},
          {"pressure", {{"type", "string"}}},
          {"unit", {{"type", "string"}}},
          {"serial", {{"type", "string"}}},
          {"location", {{"type", "string"}}},
          {"maintenance", {{"type", "string"}}}}},
        {"required", {"name", "pressure", "unit", "serial", "location", "maintenance"}},
        {"additionalProperties", false}}
                                                                     .dump());
    const auto result = engine.generate(engine.prepare(input), options);
    const auto parsed = nlohmann::json::parse(result.content);
    require(parsed.at("pressure").is_string() && parsed.at("serial") == "PX-4471-B" &&
                parsed.at("maintenance").get<std::string>().find("impeller") != std::string::npos,
            "the constrained copy did not validate or did not reproduce the record");
    require(result.speculative.ngram_rounds > 0,
            "the constrained answer verified no copies: " + result.content);
    std::cout << "constrained: copy rounds " << result.speculative.ngram_rounds
              << ", copies accepted " << result.speculative.ngram_accepted_tokens << '/'
              << result.speculative.ngram_drafted_tokens << ", pressure "
              << parsed.at("pressure").dump() << '\n';
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
        constexpr std::uint32_t kBudget = 1536;
        const Reference reference       = reference_outputs(artifact, kv, kBudget);
        require(reference.plain.size() > 400 && reference.numbered.size() > 400,
                "the non-speculative reference did not copy the file");
        for (const bool graph : {true, false}) {
            ninfer::Engine engine(engine_options(artifact, kv, true, graph));
            check_copies(engine, reference, kBudget, graph ? "graphs" : "eager");
            if (graph) {
                check_lifecycle(engine, reference, kBudget);
                check_constrained(engine);
                check_restore(engine, kBudget);
            }
        }
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    std::cout << "ok\n";
    return 0;
}
