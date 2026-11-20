// Written for NInfer-3090.
//
// The whole DeepSeek-V4.1-Flash forward pass on the real artifact, experts served from the artifact file
// (needs NINFER_TEST_DSV4_ARTIFACT; skips otherwise). There is no whole-model oracle at this size, so the
// checks are the ones that catch wiring faults: finite logits, the same logits whatever the chunking of
// the prompt, a decode step equal to the same token run as part of a longer chunk, and greedy decoding
// that continues a prompt with the model's own words (NINFER_DSV4_TOKENS: space-separated ids, default
// "The capital of France is" with a BOS; NINFER_DSV4_NEW: tokens to generate, default 8; prints them).

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

double seconds_since(std::chrono::steady_clock::time_point t) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t).count();
}

} // namespace

int main() {
    std::setvbuf(stdout, nullptr, _IOLBF, 0);
    const char* path = std::getenv("NINFER_TEST_DSV4_ARTIFACT");
    if (path == nullptr) {
        std::printf("SKIP: NINFER_TEST_DSV4_ARTIFACT is not set\n");
        return 0;
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
    options.cache_bytes = 8ULL << 30;
    dsv4::FileExpertSource experts(reader, banks, model->config().experts, options);
    cudaStream_t stream = nullptr;
    dsv4::ForwardOptions forward_options;
    forward_options.limits = {.max_context = 2048, .max_chunk = 64};
    dsv4::Forward forward(*model, experts, forward_options, stream);
    std::printf("forward state and workspace: %.2f GB\n", static_cast<double>(forward.device_bytes()) / 1e9);

    // Whole prompt in one chunk.
    t0 = std::chrono::steady_clock::now();
    const auto& logits = forward.run(prompt, 0);
    cudaStreamSynchronize(stream);
    std::printf("prefill of %zu tokens: %.1f s (cold expert cache)\n", prompt.size(), seconds_since(t0));
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
    t0                    = std::chrono::steady_clock::now();
    for (int i = 0; i < new_tokens; ++i) {
        generated.push_back(next);
        forward.run(std::span<const std::int32_t>(&next, 1), position++);
        next = forward.argmax();
    }
    const double decode_seconds = seconds_since(t0);
    std::printf("generated ids:");
    for (const auto id : generated) { std::printf(" %d", id); }
    std::printf("\ndecode: %d tokens in %.2f s (%.2f s/token)\n", new_tokens, decode_seconds, decode_seconds / std::max(new_tokens, 1));
    const auto& stats = experts.stats();
    std::printf("expert cache: %llu hits, %llu misses, %.1f GB read\n", static_cast<unsigned long long>(stats.hits),
                static_cast<unsigned long long>(stats.misses), static_cast<double>(stats.bytes_read) / 1e9);

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
        std::printf("warm decode probe: %d steps in %.3f s (%.1f ms/token), %llu hits, %llu misses\n", warm_steps, warm,
                    1000.0 * warm / warm_steps, static_cast<unsigned long long>(w.hits), static_cast<unsigned long long>(w.misses));
    }
    if (failures == 0) { std::printf("PASS\n"); }
    return failures == 0 ? 0 : 1;
}
