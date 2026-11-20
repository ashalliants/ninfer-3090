// LoRA adapters on the real dense model (GPU). NINFER_TEST_ARTIFACT is a Qwen3.8-27B artifact;
// skipped (77) without it.
//
//   zero      An attention-only adapter with B = 0, written here for the 27B geometry: a request
//             selecting it must produce the base model's tokens bit for bit (the adapter's update
//             is exactly zero), and a base request on an Engine holding it must match an Engine with
//             no adapters (base rounds keep their graphs and arithmetic).
//   real      NINFER_TEST_LORA=<PEFT adapter dir> (Qwen3.8-27B-pi-adapted): the adapter changes
//             the output (a greedy token, or a token's log-probability by >= 0.08 nats); adapted
//             rounds run as CUDA graphs that read each row's adapter slot from device memory, so
//             the adapted tokens and log-probabilities are bit-identical with graphs on or off,
//             ordinary and under MTP; the greedy output is the same under MTP, beside a concurrent
//             base request in the same rounds, and under a second name for the same adapter (slot
//             1) beside the first (one captured graph serving several selections); the base and
//             the adapter never share cached context. NINFER_TEST_LORA_REFERENCE=<json> also compares with
//             {"prompt": "...", "tokens": [ids]}: the greedy continuation a PEFT/transformers run
//             of base + adapter produced for the same rendered prompt (thinking off).
#include "ninfer/engine.h"

#include <nlohmann/json.hpp>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <tuple>
#include <algorithm>
#include <string>
#include <vector>

#include <unistd.h>

namespace {

namespace fs = std::filesystem;
using Json   = nlohmann::json;

ninfer::EngineOptions options(const char* artifact, std::vector<ninfer::LoraSource> loras,
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
    o.loras                            = std::move(loras);
    if (mtp) {
        o.speculative.backend       = ninfer::SpeculativeBackend::Mtp;
        o.speculative.draft_tokens  = 3;
        o.speculative.proposal_head = ninfer::ProposalHead::Optimized;
    }
    return o;
}

ninfer::PromptInput prompt(const std::string& text, const std::string& lora) {
    ninfer::PromptInput input;
    input.options.enable_thinking = false;
    input.options.lora            = lora;
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

std::vector<ninfer::TokenId> run(ninfer::Engine& engine, const std::string& text, const std::string& lora,
                                 std::uint32_t tokens = 24) {
    return engine.generate(engine.prepare(prompt(text, lora)), greedy(tokens)).generated_token_ids;
}

const std::string kPrompt = "Write a short Python function that returns the n-th Fibonacci number.";

// Greedy tokens with each one's log-probability: a graph and an eager round must agree to the bit.
struct Scored {
    std::vector<ninfer::TokenId> tokens;
    std::vector<float> logprobs;
};

Scored run_scored(ninfer::Engine& engine, const std::string& text, const std::string& lora,
                  std::uint32_t tokens = 48) {
    auto request                     = greedy(tokens);
    request.execution.logprobs_top_k = 0;
    const auto result                = engine.generate(engine.prepare(prompt(text, lora)), request);
    Scored out{result.generated_token_ids, {}};
    for (const auto& entry : result.token_logprobs) { out.logprobs.push_back(entry.sampled.logprob); }
    return out;
}

bool starts_with(const std::vector<ninfer::TokenId>& tokens, const std::vector<ninfer::TokenId>& prefix) {
    return tokens.size() >= prefix.size() && std::equal(prefix.begin(), prefix.end(), tokens.begin());
}

bool identical(const Scored& a, const Scored& b) {
    if (a.tokens != b.tokens || a.logprobs.size() != b.logprobs.size()) { return false; }
    for (std::size_t i = 0; i < a.logprobs.size(); ++i) {
        if (std::memcmp(&a.logprobs[i], &b.logprobs[i], sizeof(float)) != 0) {
            std::cerr << "log-probability " << i << " differs: " << a.logprobs[i] << " vs " << b.logprobs[i]
                      << '\n';
            return false;
        }
    }
    return true;
}

// Whether selecting the adapter changes what the model computes. The greedy tokens alone are too
// weak a signal: on short prompts the pi adapter moves the chosen tokens' log-probabilities by
// tenths of a nat without flipping an argmax (Fibonacci prompt, 64 tokens, on a V100: the same 64
// tokens, the largest shift 0.31 nats). So the adapter passes when it changes a greedy token,
// or when it moves a shared-prefix token's log-probability by at least kAdapterShift. An adapter
// that is never applied moves nothing (an unadapted pass is the base pass bit for bit); one whose
// FFN update is lost only shows the unfused FFN route's rounding, which is at most 0.04 nats on
// the same prompt (measured with the pi adapter's B matrices zeroed).
constexpr float kAdapterShift = 0.08F;

bool adapter_changes_output(ninfer::Engine& engine) {
    auto request                              = greedy(64);
    request.execution.logprobs_top_k          = 0;
    const auto base    = engine.generate(engine.prepare(prompt(kPrompt, "")), request);
    const auto adapted = engine.generate(engine.prepare(prompt(kPrompt, "pi")), request);
    const std::size_t n = std::min(base.token_logprobs.size(), adapted.token_logprobs.size());
    float shift         = 0.0F;
    std::size_t shared  = 0;
    for (; shared < n && base.generated_token_ids[shared] == adapted.generated_token_ids[shared]; ++shared) {
        shift = std::max(shift, std::abs(base.token_logprobs[shared].sampled.logprob -
                                         adapted.token_logprobs[shared].sampled.logprob));
    }
    const bool diverged = base.generated_token_ids != adapted.generated_token_ids;
    std::cout << "adapter effect: " << (diverged ? "tokens differ from " + std::to_string(shared)
                                                 : std::string("same tokens"))
              << ", largest log-probability shift on the " << shared << " shared tokens " << shift
              << " nats\n";
    return diverged || shift >= kAdapterShift;
}

int check(bool condition, const std::string& message) {
    if (condition) { return 0; }
    std::cerr << "FAIL: " << message << '\n';
    return 1;
}

// A Qwen3.8-27B adapter on the full-attention layers 3 and 7 with random A and zero B, rank 8.
void write_zero_adapter(const fs::path& dir) {
    fs::create_directories(dir);
    std::ofstream(dir / "adapter_config.json")
        << Json{{"peft_type", "LORA"}, {"r", 8}, {"lora_alpha", 16}, {"bias", "none"}}.dump();
    struct Entry {
        std::string name;
        std::uint64_t rows, cols;
        bool zero;
    };
    std::vector<Entry> entries;
    for (const int layer : {3, 7}) {
        const std::string base = "base_model.model.model.language_model.layers." + std::to_string(layer) + ".self_attn.";
        for (const auto& [module, out, in] : std::vector<std::tuple<std::string, std::uint64_t, std::uint64_t>>{
                 {"q_proj", 12288, 5120}, {"k_proj", 1024, 5120}, {"v_proj", 1024, 5120}, {"o_proj", 5120, 6144}}) {
            entries.push_back({base + module + ".lora_A.weight", 8, in, false});
            entries.push_back({base + module + ".lora_B.weight", out, 8, true});
        }
    }
    Json header = Json::object();
    std::uint64_t offset = 0;
    for (const Entry& e : entries) {
        const std::uint64_t bytes = e.rows * e.cols * 4;
        header[e.name] = {{"dtype", "F32"}, {"shape", {e.rows, e.cols}}, {"data_offsets", {offset, offset + bytes}}};
        offset += bytes;
    }
    std::string text = header.dump();
    while (text.size() % 8 != 0) { text.push_back(' '); }
    std::ofstream out(dir / "adapter_model.safetensors", std::ios::binary);
    const std::uint64_t size = text.size();
    out.write(reinterpret_cast<const char*>(&size), 8);
    out.write(text.data(), static_cast<std::streamsize>(text.size()));
    std::mt19937 rng(1);
    std::uniform_real_distribution<float> uniform(-0.05f, 0.05f);
    for (const Entry& e : entries) {
        std::vector<float> values(e.rows * e.cols, 0.0f);
        if (!e.zero) { for (float& v : values) { v = uniform(rng); } }
        out.write(reinterpret_cast<const char*>(values.data()), static_cast<std::streamsize>(values.size() * 4));
    }
}

int zero_scenario(const char* artifact) {
    const fs::path dir = fs::temp_directory_path() / ("ninfer_lora_zero_" + std::to_string(::getpid()));
    write_zero_adapter(dir);
    int failures = 0;
    std::vector<ninfer::TokenId> plain;
    {
        ninfer::Engine engine(options(artifact, {}));
        plain = run(engine, kPrompt, "");
    }
    {
        ninfer::Engine engine(options(artifact, {{.name = "zero", .path = dir}}));
        failures += check(run(engine, kPrompt, "") == plain, "a base request changed when an adapter was loaded");
        failures += check(run(engine, kPrompt, "zero") == plain, "a zero-B adapter changed the tokens");
        bool unknown = false;
        try {
            (void)engine.prepare(prompt(kPrompt, "missing"));
        } catch (const std::exception&) { unknown = true; }
        failures += check(unknown, "an unknown adapter name was accepted");
    }
    fs::remove_all(dir);
    return failures;
}

int real_scenario(const char* artifact, const char* adapter) {
    const std::vector<ninfer::LoraSource> loras{{.name = "pi", .path = adapter}};
    int failures = 0;
    std::vector<ninfer::TokenId> base, adapted;
    {
        ninfer::Engine engine(options(artifact, loras));
        base    = run(engine, kPrompt, "");
        adapted = run(engine, kPrompt, "pi");
        failures += check(adapter_changes_output(engine), "the adapter did not change the model's output");

        // The base and the adapter never share cached context; the adapter shares with itself.
        const auto request = greedy(4, true);
        (void)engine.generate(engine.prepare(prompt(kPrompt + " Use recursion.", "")), request);
        const auto cross = engine.generate(engine.prepare(prompt(kPrompt + " Use recursion.", "pi")), request);
        failures += check(cross.reused_prompt_tokens == 0, "an adapted request reused the base model's context");
        const auto same = engine.generate(engine.prepare(prompt(kPrompt + " Use recursion.", "pi")), request);
        failures += check(same.reused_prompt_tokens != 0, "an adapted request did not reuse its own context");

        if (const char* reference = std::getenv("NINFER_TEST_LORA_REFERENCE")) {
            const Json expected = Json::parse(std::ifstream(reference));
            const auto tokens   = expected.at("tokens").get<std::vector<ninfer::TokenId>>();
            const auto ours     = run(engine, expected.at("prompt").get<std::string>(), "pi",
                                      static_cast<std::uint32_t>(tokens.size()));
            std::size_t agree = 0;
            while (agree < tokens.size() && agree < ours.size() && tokens[agree] == ours[agree]) { ++agree; }
            std::cout << "reference agreement: " << agree << " / " << tokens.size() << " greedy tokens\n";
            failures += check(agree >= std::min<std::size_t>(tokens.size(), 8),
                              "the adapted output diverges from the PEFT reference within 8 tokens");
        }
    }
    // Adapted graphs against eager adapted rounds, to the bit: ordinary decode, then MTP verify.
    for (const bool mtp : {false, true}) {
        const std::string mode = mtp ? " under MTP" : "";
        Scored graphed, eager, graphed_base, eager_base;
        {
            ninfer::Engine engine(options(artifact, loras, mtp, true));
            graphed      = run_scored(engine, kPrompt, "pi");
            graphed_base = run_scored(engine, kPrompt, "");
            // The 24-token runs above are the prefixes of these 48-token ones.
            failures += check(starts_with(graphed.tokens, adapted), "adapted tokens changed" + mode);
            failures += check(starts_with(graphed_base.tokens, base), "base tokens changed" + mode);
        }
        {
            ninfer::Engine engine(options(artifact, loras, mtp, false));
            eager      = run_scored(engine, kPrompt, "pi");
            eager_base = run_scored(engine, kPrompt, "");
        }
        failures += check(identical(graphed, eager),
                          "adapted rounds under CUDA graphs differ from eager adapted rounds" + mode);
        failures += check(identical(graphed_base, eager_base),
                          "base rounds under CUDA graphs differ from eager base rounds" + mode);
    }
    {
        ninfer::Engine engine(options(artifact, loras, true, true, 2));
        auto first  = engine.submit(engine.prepare(prompt(kPrompt, "pi")), greedy(24));
        auto second = engine.submit(engine.prepare(prompt(kPrompt, "")), greedy(24));
        failures += check(first.wait().generated_token_ids == adapted,
                          "adapted tokens changed beside a concurrent base request");
        failures += check(second.wait().generated_token_ids == base,
                          "base tokens changed beside a concurrent adapted request");
    }
    {
        // The same adapter in slots 0 and 1: every selection of a two-row round ([0, 1], [1, -1],
        // [1] alone) replays the one adapted graph of its batch size and must give pi's tokens.
        const std::vector<ninfer::LoraSource> twice{{.name = "pi", .path = adapter},
                                                    {.name = "pi2", .path = adapter}};
        ninfer::Engine engine(options(artifact, twice, false, true, 2));
        failures += check(run(engine, kPrompt, "pi2") == adapted, "the adapter in slot 1 changed its tokens");
        auto first  = engine.submit(engine.prepare(prompt(kPrompt, "pi")), greedy(24));
        auto second = engine.submit(engine.prepare(prompt(kPrompt, "pi2")), greedy(24));
        failures += check(first.wait().generated_token_ids == adapted,
                          "slot 0 changed its tokens beside slot 1 in the same rounds");
        failures += check(second.wait().generated_token_ids == adapted,
                          "slot 1 changed its tokens beside slot 0 in the same rounds");
        auto third  = engine.submit(engine.prepare(prompt(kPrompt, "pi2")), greedy(24));
        auto fourth = engine.submit(engine.prepare(prompt(kPrompt, "")), greedy(24));
        failures += check(third.wait().generated_token_ids == adapted,
                          "slot 1 changed its tokens beside a base request");
        failures += check(fourth.wait().generated_token_ids == base,
                          "base tokens changed beside slot 1");
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
        if (const char* adapter = std::getenv("NINFER_TEST_LORA")) {
            failures += real_scenario(artifact, adapter);
        } else {
            std::cout << "NINFER_TEST_LORA is not set: the real-adapter scenario is skipped\n";
        }
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
    if (failures == 0) { std::cout << "lora real: all checks passed\n"; }
    return failures == 0 ? 0 : 1;
}
