// Written for NInfer-3090.
//
// Records the routed-expert access sequence of DeepSeek-V4.1-Flash on real prompts, for the offline policy
// comparison in test_expert_trace_replay.cpp. Needs NINFER_TEST_DSV4_ARTIFACT, NINFER_DSV4_EXPERT_TRACE (the
// file to write) and NINFER_DSV4_PROMPTS (a text file, one prompt per line, space-separated token ids); skips
// (77) otherwise. Each prompt is prefilled in chunks of 64 and continued greedily for NINFER_DSV4_NEW tokens
// (default 24); the tiers stay warm across prompts, as in a serving session. NINFER_DSV4_RAM_GB (default 4)
// and NINFER_DSV4_VRAM_GB (default 0: the routing does not depend on the tiers) size them. Prints the
// time per token and the tier hit counts of each prompt.

#include "core/device.h"
#include "models/dsv4/execution/forward.h"
#include "models/dsv4/model.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace dsv4 = ninfer::models::dsv4;

namespace {

double env_number(const char* name, double fallback) {
    const char* env = std::getenv(name);
    return env != nullptr && env[0] != '\0' ? std::atof(env) : fallback;
}

double seconds_since(std::chrono::steady_clock::time_point t) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t).count();
}

} // namespace

int main() {
    std::setvbuf(stdout, nullptr, _IOLBF, 0);
    const char* path    = std::getenv("NINFER_TEST_DSV4_ARTIFACT");
    const char* trace   = std::getenv("NINFER_DSV4_EXPERT_TRACE");
    const char* prompts = std::getenv("NINFER_DSV4_PROMPTS");
    if (path == nullptr || trace == nullptr || prompts == nullptr) {
        std::printf("SKIP: NINFER_TEST_DSV4_ARTIFACT, NINFER_DSV4_EXPERT_TRACE and NINFER_DSV4_PROMPTS are not all set\n");
        return 77;
    }
    std::vector<std::vector<std::int32_t>> lines;
    {
        std::ifstream in(prompts);
        for (std::string line; std::getline(in, line);) {
            std::istringstream words(line);
            std::vector<std::int32_t> ids;
            for (std::int32_t id; words >> id;) { ids.push_back(id); }
            if (!ids.empty()) { lines.push_back(std::move(ids)); }
        }
    }
    if (lines.empty()) {
        std::fprintf(stderr, "FAIL: no prompts in %s\n", prompts);
        return 1;
    }
    const int new_tokens         = static_cast<int>(env_number("NINFER_DSV4_NEW", 24));
    constexpr std::size_t kChunk = 64;

    ninfer::artifact::Reader reader(path);
    auto plan        = dsv4::plan_load(reader, true);
    const auto banks = plan.file_banks();
    ninfer::DeviceContext device(0);
    auto model = dsv4::materialize_model(std::move(plan), device);

    dsv4::FileExpertOptions options;
    options.cache_bytes        = static_cast<std::uint64_t>(env_number("NINFER_DSV4_RAM_GB", 4) * double(1ULL << 30));
    options.device_cache_bytes = static_cast<std::uint64_t>(env_number("NINFER_DSV4_VRAM_GB", 0) * double(1ULL << 30));
    options.read_threads       = static_cast<std::uint32_t>(env_number("NINFER_DSV4_READ_THREADS", 1));
    dsv4::FileExpertSource experts(reader, banks, model->config().experts, options);
    cudaStream_t stream = nullptr;
    dsv4::ForwardOptions forward_options;
    forward_options.limits = {.max_context = 2048, .max_chunk = kChunk};
    if (const char* env = std::getenv("NINFER_DSV4_ENGRAM_DIR"); env != nullptr && env[0] != '\0') { forward_options.engram_dir = env; }
    dsv4::Forward forward(*model, experts, forward_options, stream);

    int failures = 0;
    for (std::size_t p = 0; p < lines.size(); ++p) {
        const auto& prompt = lines[p];
        experts.reset_stats();
        auto t0 = std::chrono::steady_clock::now();
        for (std::size_t at = 0; at < prompt.size(); at += kChunk) {
            const std::size_t n = std::min(kChunk, prompt.size() - at);
            forward.run(std::span<const std::int32_t>(prompt).subspan(at, n), static_cast<std::int32_t>(at));
        }
        cudaStreamSynchronize(stream);
        const double prefill = seconds_since(t0);
        std::int32_t next = forward.argmax(), position = static_cast<std::int32_t>(prompt.size());
        std::vector<std::int32_t> generated;
        t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < new_tokens; ++i) {
            generated.push_back(next);
            forward.run(std::span<const std::int32_t>(&next, 1), position++);
            next = forward.argmax();
        }
        const double decode = seconds_since(t0);
        const auto& s       = experts.stats();
        std::printf("prompt %zu (%zu tokens): prefill %.1f s, decode %.0f ms/token; experts: %llu VRAM, %llu RAM, %llu file reads\n  ids:", p, prompt.size(), prefill,
                    1000.0 * decode / std::max(new_tokens, 1), static_cast<unsigned long long>(s.vram_hits), static_cast<unsigned long long>(s.ram_hits),
                    static_cast<unsigned long long>(s.misses));
        for (const auto id : generated) { std::printf(" %d", id); }
        std::printf("\n");
        if (generated.empty()) { ++failures; }
    }
    if (failures == 0) { std::printf("PASS\n"); }
    return failures == 0 ? 0 : 1;
}
