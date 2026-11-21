// Written for NInfer-3090.
//
// The whole DeepSeek-V4.1-Flash forward pass on the real artifact, experts served from the artifact file
// (needs NINFER_TEST_DSV4_ARTIFACT; skips otherwise). There is no whole-model oracle at this size, so the
// checks are the ones that catch wiring faults: finite logits, the same logits whatever the chunking of
// the prompt, a decode step equal to the same token run as part of a longer chunk, and greedy decoding
// that continues a prompt with the model's own words (NINFER_DSV4_TOKENS: space-separated ids, default
// "The capital of France is" with a BOS; NINFER_DSV4_NEW: tokens to generate, default 8; prints them).
// The expert tiers are set by NINFER_DSV4_RAM_GB (pinned RAM tier, default 8), NINFER_DSV4_VRAM_GB (device frame
// pool, default 12; 0 has none), NINFER_DSV4_READ_THREADS (parallel preads, default 1: on a hard disk concurrent reads slow it down about 4x,
// on a SATA SSD 8 to 16 threads read 1.5x faster) and NINFER_DSV4_ZC
// (zero-copy misses of the Op, default 0). Prints the time per token and the tier hit counts of each phase
// (cold decode, a second decode of the same text with the caches warm, and a fixed-routing probe).

#include "core/device.h"
#include "models/dsv4/execution/forward.h"
#include "models/dsv4/model.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <sstream>
#include <string>
#include <vector>

namespace dsv4 = ninfer::models::dsv4;

namespace {

int failures = 0;

void check(bool ok, const char* what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++failures;
    }
}

std::vector<float> download(const ninfer::Tensor& logits) {
    std::vector<float> out(static_cast<std::size_t>(logits.ne[0]));
    cudaMemcpy(out.data(), logits.data, out.size() * 4, cudaMemcpyDeviceToHost);
    return out;
}

std::vector<int> top(const std::vector<float>& v, int k) {
    std::vector<int> idx(v.size());
    for (std::size_t i = 0; i < idx.size(); ++i) { idx[i] = static_cast<int>(i); }
    std::partial_sort(idx.begin(), idx.begin() + k, idx.end(), [&](int a, int b) { return v[a] > v[b]; });
    idx.resize(k);
    return idx;
}

double env_number(const char* name, double fallback) {
    const char* env = std::getenv(name);
    return env != nullptr ? std::atof(env) : fallback;
}

void print_tiers(const char* label, const dsv4::FileExpertSource::Stats& s, double seconds, int tokens) {
    const double routed = static_cast<double>(s.vram_hits + s.ram_hits + s.misses);
    std::printf("%s: %.1f ms/token; experts: %llu VRAM hits, %llu RAM hits, %llu file reads (%.1f%% / %.1f%% / %.1f%%), %llu promotions, "
                "%.2f GB read, read %.2f s, ids wait %.2f s\n",
                label, 1000.0 * seconds / std::max(tokens, 1), static_cast<unsigned long long>(s.vram_hits),
                static_cast<unsigned long long>(s.ram_hits), static_cast<unsigned long long>(s.misses),
                100.0 * static_cast<double>(s.vram_hits) / std::max(routed, 1.0), 100.0 * static_cast<double>(s.ram_hits) / std::max(routed, 1.0),
                100.0 * static_cast<double>(s.misses) / std::max(routed, 1.0), static_cast<unsigned long long>(s.promotions),
                static_cast<double>(s.bytes_read) / 1e9, s.read_seconds, s.sync_seconds);
}

double seconds_since(std::chrono::steady_clock::time_point t) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t).count();
}

} // namespace

int main() {
    std::setvbuf(stdout, nullptr, _IOLBF, 0);
    const char* path = std::getenv("NINFER_TEST_DSV4_ARTIFACT");
    if (path == nullptr) {
        std::printf("SKIP: NINFER_TEST_DSV4_ARTIFACT is not set\n");
        return 77; // the SKIP_RETURN_CODE the registration declares
    }
    std::vector<std::int32_t> prompt{0, 671, 6102, 294, 8760, 344};
    if (const char* env = std::getenv("NINFER_DSV4_TOKENS")) {
        prompt.clear();
        std::istringstream in(env);
        for (std::int32_t id; in >> id;) { prompt.push_back(id); }
    }
    int new_tokens = 8;
    if (const char* env = std::getenv("NINFER_DSV4_NEW")) { new_tokens = std::atoi(env); }

    ninfer::artifact::Reader reader(path);
    auto plan = dsv4::plan_load(reader, true);
    const auto banks = plan.file_banks();
    ninfer::DeviceContext device(0);
    auto t0    = std::chrono::steady_clock::now();
    auto model = dsv4::materialize_model(std::move(plan), device);
    std::printf("loaded the dense weights in %.1f s\n", seconds_since(t0));

    dsv4::FileExpertOptions options;
    options.cache_bytes        = static_cast<std::uint64_t>(env_number("NINFER_DSV4_RAM_GB", 8) * double(1ULL << 30));
    options.device_cache_bytes = static_cast<std::uint64_t>(env_number("NINFER_DSV4_VRAM_GB", 12) * double(1ULL << 30));
    options.read_threads       = static_cast<std::uint32_t>(env_number("NINFER_DSV4_READ_THREADS", 1));
    options.zero_copy_misses   = static_cast<std::int32_t>(env_number("NINFER_DSV4_ZC", 0));
    dsv4::FileExpertSource experts(reader, banks, model->config().experts, options);
    cudaStream_t stream = nullptr;
    dsv4::ForwardOptions forward_options;
    forward_options.limits = {.max_context = 2048, .max_chunk = 64};
    dsv4::Forward forward(*model, experts, forward_options, stream);
    std::printf("forward state and workspace: %.2f GB; expert tiers: %u RAM slots, %u device frames, %u read threads\n",
                static_cast<double>(forward.device_bytes()) / 1e9, experts.cache_slots(), experts.device_frames(), options.read_threads);

    // Whole prompt in one chunk.
    t0 = std::chrono::steady_clock::now();
    const auto& logits = forward.run(prompt, 0);
    cudaStreamSynchronize(stream);
    std::printf("prefill of %zu tokens: %.1f s (cold expert cache)\n", prompt.size(), seconds_since(t0));
    print_tiers("cold prefill", experts.stats(), seconds_since(t0), static_cast<int>(prompt.size()));
    experts.reset_stats();
    const std::vector<float> whole = download(logits);
    bool finite = true;
    for (const float v : whole) { finite = finite && std::isfinite(v); }
    check(finite, "the logits are finite");
    std::printf("top tokens:");
    for (const int id : top(whole, 5)) { std::printf(" %d(%.2f)", id, whole[id]); }
    std::printf("\n");

    // The same prompt in two chunks must give the same logits (the state carries across the boundary).
    if (prompt.size() >= 3) {
        const std::size_t cut = prompt.size() / 2;
        forward.run(std::span<const std::int32_t>(prompt).first(cut), 0);
        const auto& split = forward.run(std::span<const std::int32_t>(prompt).subspan(cut), static_cast<std::int32_t>(cut));
        cudaStreamSynchronize(stream);
        const std::vector<float> two = download(split);
        double num = 0, den = 0;
        for (std::size_t i = 0; i < two.size(); ++i) {
            num += (two[i] - whole[i]) * (two[i] - whole[i]);
            den += whole[i] * whole[i];
        }
        std::printf("chunked vs whole logits: relative L2 %.5f\n", std::sqrt(num / den));
        check(std::sqrt(num / den) < 0.05, "chunked prefill matches the single chunk");
    }

    // Greedy continuation, one token per step.
    forward.run(prompt, 0);
    std::vector<std::int32_t> generated;
    std::int32_t position = static_cast<std::int32_t>(prompt.size());
    std::int32_t next     = forward.argmax();
    experts.reset_stats();
    t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < new_tokens; ++i) {
        generated.push_back(next);
        forward.run(std::span<const std::int32_t>(&next, 1), position++);
        next = forward.argmax();
    }
    const double decode_seconds = seconds_since(t0);
    std::printf("generated ids:");
    for (const auto id : generated) { std::printf(" %d", id); }
    // Decode must equal prefill: running prompt + generated[:-1] as ONE chunk gives, at its last position, the
    // token greedy decoding produced last. (Starting at position 0 begins a new sequence over the earlier state.)
    if (generated.size() >= 2) {
        std::vector<std::int32_t> all(prompt);
        all.insert(all.end(), generated.begin(), generated.end() - 1);
        if (all.size() <= forward_options.limits.max_chunk) {
            forward.run(all, 0);
            const std::int32_t again = forward.argmax();
            std::printf("decode equals prefill: %d vs %d\n", static_cast<int>(again), static_cast<int>(generated.back()));
            check(again == generated.back(), "the last greedy token equals the prefill's argmax over the same tokens");
            // Rebuild the decode state the loop left, so the warm probe below runs as before.
            forward.run(prompt, 0);
            for (std::size_t i = 0; i + 1 < generated.size(); ++i) {
                forward.run(std::span<const std::int32_t>(&generated[i], 1), static_cast<std::int32_t>(prompt.size() + i));
            }
            next     = generated.back();
            position = static_cast<std::int32_t>(prompt.size() + generated.size() - 1);
        }
    }
    if (std::getenv("NINFER_DSV4_TOKENS") == nullptr && generated.size() >= 2) {
        check(generated[0] == 11111 && generated[1] == 16, "the default prompt continues with ' Paris' and '.'");
    }
    std::printf("\ndecode: %d tokens in %.2f s (%.2f s/token)\n", new_tokens, decode_seconds, decode_seconds / std::max(new_tokens, 1));
    print_tiers("decode after the prefill", experts.stats(), decode_seconds, new_tokens);
    const std::vector<std::int32_t> first_pass = generated;
    if (!std::getenv("NINFER_DSV4_TOKENS") && new_tokens >= 4) {
        const std::int32_t expected[4] = {11111, 16, 4588, 455};
        check(std::equal(expected, expected + 4, generated.begin()), "greedy ids continue 'The capital of France is' as before");
    }

    // The same prompt and continuation again with the caches warm: identical tokens, and the speed a
    // repeated workload sees.
    for (int pass = 0; pass < 3; ++pass) {
        experts.reset_stats();
        forward.run(prompt, 0);
        cudaStreamSynchronize(stream);
        std::int32_t pos = static_cast<std::int32_t>(prompt.size());
        std::int32_t tok = forward.argmax();
        std::vector<std::int32_t> again;
        const auto start = std::chrono::steady_clock::now();
        for (int i = 0; i < new_tokens; ++i) {
            again.push_back(tok);
            forward.run(std::span<const std::int32_t>(&tok, 1), pos++);
            tok = forward.argmax();
        }
        print_tiers(pass == 0 ? "warm decode, pass 1 (same text again)" : pass == 1 ? "warm decode, pass 2" : "warm decode, pass 3", experts.stats(), seconds_since(start), new_tokens);
        check(again == first_pass, "the second decode of the same text gives the same ids");
        next     = tok;
        position = pos;
    }

    // Warm-cache decode speed: the same token fed again for several steps routes to the experts just used,
    // so the steps are served from the cache and the time is the compute and the per-layer host sync.
    {
        const int warm_steps = 8;
        for (int i = 0; i < 8; ++i) { forward.run(std::span<const std::int32_t>(&next, 1), position++); } // fill the cache
        experts.reset_stats();
        cudaStreamSynchronize(stream);
        const auto warm_start = std::chrono::steady_clock::now();
        for (int i = 0; i < warm_steps; ++i) { forward.run(std::span<const std::int32_t>(&next, 1), position++); }
        cudaStreamSynchronize(stream);
        const double warm = seconds_since(warm_start);
        const auto& w = experts.stats();
        print_tiers("probe, same token repeated (routing drifts)", w, warm, warm_steps);
    }
    std::printf("%s", forward.profile_report().c_str());
    if (failures == 0) { std::printf("PASS\n"); }
    return failures == 0 ? 0 : 1;
}
