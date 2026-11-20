// Control vectors on the real dense model (GPU). NINFER_TEST_ARTIFACT is a Qwen3.8-27B artifact;
// skipped (77) without it.
//
//   zero      A llama.cpp (additive) vector of zeros on layers 10-58, written here for the 27B
//             geometry: a request selecting it must produce the base model's tokens bit for bit
//             (the kernel adds exactly zero), and a base request on an Engine holding it must
//             match an Engine with no vectors (unselected rows are untouched).
//   real      NINFER_TEST_CONTROL_VECTOR=<GGUF> (Qwen3.8-27B-refusal-cvec-GLP-49-L10-58-a1): the
//             steered greedy output differs from the base; it is the same with CUDA graphs on or
//             off; under MTP, and beside a concurrent base request in the same rounds, it is the same
//             up to a near tie (see same_up_to_tie); steered and base requests never share cached
//             context.
#include "ninfer/engine.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>

namespace {

namespace fs = std::filesystem;

ninfer::EngineOptions options(const char* artifact, std::vector<ninfer::GraftSource> grafts,
                              bool mtp = false, bool graphs = true, std::uint32_t lanes = 1) {
    ninfer::EngineOptions o;
    o.artifact_path                    = artifact;
    o.max_context                      = 2048;
    o.kv_capacity                      = ninfer::KvCapacityPolicy::explicit_capacity(2048);
    o.prefill_chunk                    = 512;
    o.enable_vision                    = false;
    o.max_concurrency                  = lanes;
    o.max_pending_requests             = lanes;
    o.use_cuda_graph                   = graphs;
    o.context_cache.device_state_slots = 4;
    o.grafts                           = std::move(grafts);
    if (mtp) {
        o.speculative.backend       = ninfer::SpeculativeBackend::Mtp;
        o.speculative.draft_tokens  = 3;
        o.speculative.proposal_head = ninfer::ProposalHead::Optimized;
    }
    return o;
}

ninfer::PromptInput prompt(const std::string& text, const std::string& graft) {
    ninfer::PromptInput input;
    input.options.enable_thinking = false;
    input.options.graft           = graft;
    ninfer::ChatMessage user;
    user.role = ninfer::ChatRole::User;
    user.parts.push_back(ninfer::MessagePart{.kind = ninfer::MessagePartKind::Text, .text = text, .media = {}});
    input.messages.push_back(std::move(user));
    return input;
}

ninfer::RequestOptions greedy(std::uint32_t tokens, bool reuse = false) {
    ninfer::RequestOptions request;
    request.execution.requested_output_tokens = tokens;
    request.execution.sampling.temperature    = 0.0F;
    request.execution.allow_prefix_reuse      = reuse;
    request.stop.include_model_defaults       = false;
    return request;
}

std::vector<ninfer::TokenId> run(ninfer::Engine& engine, const std::string& text, const std::string& graft,
                                 std::uint32_t tokens = 32) {
    return engine.generate(engine.prepare(prompt(text, graft)), greedy(tokens)).generated_token_ids;
}

int check(bool condition, const std::string& message) {
    if (condition) { return 0; }
    std::cerr << "FAIL: " << message << '\n';
    return 1;
}

template <typename T>
void pod(std::ofstream& out, T value) {
    out.write(reinterpret_cast<const char*>(&value), sizeof(T));
}

void text(std::ofstream& out, const std::string& s) {
    pod<std::uint64_t>(out, s.size());
    out.write(s.data(), static_cast<std::streamsize>(s.size()));
}

// A llama.cpp control vector of zeros for Qwen3.8-27B (hidden 5120) on layers 10-58.
void write_zero_vector(const fs::path& path) {
    constexpr std::uint64_t kHidden = 5120;
    std::ofstream out(path, std::ios::binary);
    out.write("GGUF", 4);
    pod<std::uint32_t>(out, 3);
    pod<std::uint64_t>(out, 49);
    pod<std::uint64_t>(out, 2);
    text(out, "general.architecture");
    pod<std::uint32_t>(out, 8);
    text(out, "controlvector");
    text(out, "controlvector.model_hint");
    pod<std::uint32_t>(out, 8);
    text(out, "qwen3_5");
    std::uint64_t offset = 0;
    for (int layer = 10; layer <= 58; ++layer) {
        text(out, "direction." + std::to_string(layer));
        pod<std::uint32_t>(out, 1);
        pod<std::uint64_t>(out, kHidden);
        pod<std::uint32_t>(out, 0);
        pod<std::uint64_t>(out, offset);
        offset += kHidden * 4;
    }
    while (out.tellp() % 32 != 0) { out.put('\0'); }
    const std::vector<float> zeros(kHidden * 49, 0.0f);
    out.write(reinterpret_cast<const char*>(zeros.data()), static_cast<std::streamsize>(zeros.size() * 4));
}

const std::string kPrompt = "Explain in two sentences how a refrigerator keeps food cold.";

// MTP verifies a round's columns in one multi-column pass, whose arithmetic differs slightly from a
// one-column decode step (a base request's token log-probabilities move by up to ~0.09 nats on a
// V100). A greedy argmax whose top two logits sit within that, or tie outright in BF16 (0.125 apart
// at this magnitude), may flip. So a steered MTP run must match the plain run token for token up to
// its first difference, and that difference must fall where the plain run's top two were within
// kNearTie nats. A steering bug (a skipped layer, a wrong row) moves log-probabilities far more.
constexpr float kNearTie = 0.25F;

bool same_up_to_tie(const ninfer::GenerationResult& reference, const std::vector<ninfer::TokenId>& other,
                    const std::string& what) {
    const auto& tokens = reference.generated_token_ids;
    std::size_t i      = 0;
    while (i < tokens.size() && i < other.size() && tokens[i] == other[i]) { ++i; }
    if (i == tokens.size() && i == other.size()) { return true; }
    if (i == tokens.size() || i == other.size()) {
        // One output is a strict prefix of the other: a run that stopped early (or ran on), not a
        // flipped near tie.
        std::cerr << what << ": " << other.size() << " tokens against the plain run's " << tokens.size()
                  << ", the same up to the shorter\n";
        return false;
    }
    if (i >= reference.token_logprobs.size() || reference.token_logprobs[i].top.size() < 2) {
        std::cerr << what << ": differs at token " << i << " with no alternatives to judge it by\n";
        return false;
    }
    const auto& top    = reference.token_logprobs[i].top;
    const float margin = top[0].logprob - top[1].logprob;
    std::cout << what << ": first differs at token " << i << ", where the plain run's top two are "
              << margin << " nats apart\n";
    return margin <= kNearTie;
}

int zero_scenario(const char* artifact) {
    const fs::path path = fs::temp_directory_path() / ("ninfer_cvec_zero_" + std::to_string(std::random_device{}()) + ".gguf");
    write_zero_vector(path);
    int failures = 0;
    std::vector<ninfer::TokenId> plain;
    {
        ninfer::Engine engine(options(artifact, {}));
        plain = run(engine, kPrompt, "");
    }
    {
        ninfer::Engine engine(options(artifact, {{.name = "zero", .path = path}}));
        failures += check(run(engine, kPrompt, "") == plain, "a base request changed when a vector was loaded");
        failures += check(run(engine, kPrompt, "zero") == plain, "a zero additive vector changed the tokens");
        bool unknown = false;
        try {
            (void)engine.prepare(prompt(kPrompt, "missing"));
        } catch (const std::exception&) { unknown = true; }
        failures += check(unknown, "an unknown vector name was accepted");
    }
    fs::remove(path);
    return failures;
}

int real_scenario(const char* artifact, const char* vector) {
    const std::vector<ninfer::GraftSource> grafts{{.name = "uncensored", .path = vector}};
    int failures = 0;
    std::vector<ninfer::TokenId> base, steered;
    {
        ninfer::Engine engine(options(artifact, grafts));
        base    = run(engine, kPrompt, "");
        steered = run(engine, kPrompt, "uncensored");
        failures += check(base != steered, "the vector did not change the greedy output");

        // Steered and base requests never share cached context; the vector shares with itself.
        const auto request = greedy(4, true);
        (void)engine.generate(engine.prepare(prompt(kPrompt + " Be brief.", "")), request);
        const auto cross = engine.generate(engine.prepare(prompt(kPrompt + " Be brief.", "uncensored")), request);
        failures += check(cross.reused_prompt_tokens == 0, "a steered request reused the base model's context");
        const auto same = engine.generate(engine.prepare(prompt(kPrompt + " Be brief.", "uncensored")), request);
        failures += check(same.reused_prompt_tokens != 0, "a steered request did not reuse its own context");
    }
    {
        ninfer::Engine engine(options(artifact, grafts, false, false));
        failures += check(run(engine, kPrompt, "uncensored") == steered, "steered tokens changed without CUDA graphs");
        failures += check(run(engine, kPrompt, "") == base, "base tokens changed without CUDA graphs");
    }
    ninfer::GenerationResult reference;
    {
        ninfer::Engine engine(options(artifact, grafts, false, true));
        auto request                     = greedy(32);
        request.execution.logprobs_top_k = 2;
        reference = engine.generate(engine.prepare(prompt(kPrompt, "uncensored")), request);
        failures += check(reference.generated_token_ids == steered,
                          "asking for log-probabilities changed the steered tokens");
    }
    {
        ninfer::Engine engine(options(artifact, grafts, true));
        failures += check(same_up_to_tie(reference, run(engine, kPrompt, "uncensored"), "steered under MTP"),
                          "steered tokens changed under MTP away from a near tie");
        failures += check(run(engine, kPrompt, "") == base, "base tokens changed under MTP");
    }
    {
        ninfer::Engine engine(options(artifact, grafts, true, true, 2));
        auto first  = engine.submit(engine.prepare(prompt(kPrompt, "uncensored")), greedy(32));
        auto second = engine.submit(engine.prepare(prompt(kPrompt, "")), greedy(32));
        failures += check(same_up_to_tie(reference, first.wait().generated_token_ids,
                                         "steered beside a base request (MTP)"),
                          "steered tokens changed beside a concurrent base request away from a near tie");
        failures += check(second.wait().generated_token_ids == base,
                          "base tokens changed beside a concurrent steered request");
    }
    return failures;
}

} // namespace

int main() {
    const char* artifact = std::getenv("NINFER_TEST_ARTIFACT");
    if (!artifact || !*artifact) {
        std::cout << "skip: NINFER_TEST_ARTIFACT is not set\n";
        return 77;
    }
    int failures = 0;
    try {
        failures += zero_scenario(artifact);
        if (const char* vector = std::getenv("NINFER_TEST_CONTROL_VECTOR")) {
            failures += real_scenario(artifact, vector);
        } else {
            std::cout << "NINFER_TEST_CONTROL_VECTOR is not set: the real-vector scenario is skipped\n";
        }
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
    if (failures == 0) { std::cout << "control vector real: all checks passed\n"; }
    return failures == 0 ? 0 : 1;
}
