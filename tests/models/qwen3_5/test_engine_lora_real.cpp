// LoRA adapters on the real dense model (GPU). NINFER_TEST_ARTIFACT is a Qwen3.8-27B artifact;
// skipped (77) without it.
//
//   zero      An attention-only adapter with B = 0, written here for the 27B geometry: a request
//             selecting it must produce the base model's tokens bit for bit (the adapter's update
//             is exactly zero), and a base request on an Engine holding it must match an Engine with
//             no adapters (base rounds keep their graphs and arithmetic).
//   real      NINFER_TEST_LORA=<PEFT adapter dir> (Qwen3.8-27B-pi-adapted): the adapted greedy
//             output differs from the base; it is the same with CUDA graphs on or off, under MTP,
//             and beside a concurrent base request in the same rounds; the base and the adapter
//             never share cached context. NINFER_TEST_LORA_REFERENCE=<json> also compares with
//             {"prompt": "...", "tokens": [ids]}: the greedy continuation a PEFT/transformers run
//             of base + adapter produced for the same rendered prompt (thinking off).
#include "ninfer/engine.h"

#include <nlohmann/json.hpp>

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

const std::string kPrompt = "Write a short Python function that returns the n-th Fibonacci number.";

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
        failures += check(base != adapted, "the adapter did not change the greedy output");

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
    {
        ninfer::Engine engine(options(artifact, loras, false, false));
        failures += check(run(engine, kPrompt, "pi") == adapted, "adapted tokens changed without CUDA graphs");
        failures += check(run(engine, kPrompt, "") == base, "base tokens changed without CUDA graphs");
    }
    {
        ninfer::Engine engine(options(artifact, loras, true));
        failures += check(run(engine, kPrompt, "pi") == adapted, "adapted tokens changed under MTP");
        failures += check(run(engine, kPrompt, "") == base, "base tokens changed under MTP");
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
