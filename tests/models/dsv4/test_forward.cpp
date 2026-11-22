// Written for NInfer-3090.
//
// The whole DeepSeek-V4.1-Flash forward pass on the real artifact, experts served from the artifact file
// (needs NINFER_TEST_DSV4_ARTIFACT; skips otherwise). There is no whole-model oracle at this size, so the
// checks are the ones that catch wiring faults: finite logits, the same logits whatever the chunking of
// the prompt, a decode step equal to the same token run as part of a longer chunk, and greedy decoding
// that continues a prompt with the model's own words (NINFER_DSV4_TOKENS: space-separated ids, default
// "The capital of France is" with a BOS; NINFER_DSV4_NEW: tokens to generate, default 8; prints them).
// The expert tiers are set by NINFER_DSV4_RAM_GB (pinned RAM tier, default 8), NINFER_DSV4_VRAM_GB (device frame
// pool, default: what is free once the forward exists less 2 GiB, as the engine sizes it; 0 has none), NINFER_DSV4_READ_THREADS (parallel preads, default 1: on a hard disk concurrent reads slow it down about 4x,
// on a SATA SSD 8 to 16 threads read 1.5x faster) and NINFER_DSV4_ZC
// (zero-copy misses of the Op, default 0). Prints the time per token and the tier hit counts of each phase
// (cold decode, a second decode of the same text with the caches warm, and a fixed-routing probe).
// NINFER_DSV4_ENGRAM_DIR: the Engram folder (tools/convert/dsv4_engram_volume.py); layers 1 and 14 then inject, and the
// default prompt is only required to continue with ' Paris' (the exact ids of the uninjected run differ). Chunked
// prefill and decode then also check that the hash history carries across calls.

#include "core/device.h"
#include "models/dsv4/execution/forward.h"
#include "models/dsv4/model.h"
#include "runtime/engine/dsv4_plan.h"

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
    // NINFER_DSV4_VRAM_GB unset: the engine's sizing, what is free once the forward exists less a headroom (below).
    const bool vram_from_free = std::getenv("NINFER_DSV4_VRAM_GB") == nullptr;
    options.device_cache_bytes = static_cast<std::uint64_t>(env_number("NINFER_DSV4_VRAM_GB", 0) * double(1ULL << 30));
    options.read_threads       = static_cast<std::uint32_t>(env_number("NINFER_DSV4_READ_THREADS", 1));
    options.zero_copy_misses   = static_cast<std::int32_t>(env_number("NINFER_DSV4_ZC", 0));
    dsv4::FileExpertSource experts(reader, banks, model->config().experts, options);
    cudaStream_t stream = nullptr;
    dsv4::ForwardOptions forward_options;
    forward_options.limits = {.max_context = 2048, .max_chunk = 64};
    if (const char* env = std::getenv("NINFER_DSV4_ENGRAM_DIR"); env != nullptr && env[0] != '\0') { forward_options.engram_dir = env; }
    dsv4::Forward forward(*model, experts, forward_options, stream);
    if (vram_from_free) {
        std::size_t free_bytes = 0, total_bytes = 0;
        cudaMemGetInfo(&free_bytes, &total_bytes);
        experts.set_device_cache_bytes(ninfer::runtime::dsv4_expert_vram_bytes(free_bytes));
    }
    std::printf("Engram injection: %s\n", forward.engram_applied() ? "applied" : "not applied (reduced quality)");
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
    for (const int id : top(whole, 5)) { std::printf(" %d(%.4f)", id, whole[id]); }
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
    std::size_t free_before = 0, total_before = 0;
    cudaMemGetInfo(&free_before, &total_before);
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
        check(generated[0] == 11111, "the default prompt continues with ' Paris'");
        if (!forward.engram_applied()) { check(generated[1] == 16, "the uninjected default prompt continues with ' Paris' and '.'"); }
    }
    {
        std::size_t free_after = 0, total = 0;
        cudaMemGetInfo(&free_after, &total);
        std::printf("device memory taken since before the first decode step (the decode graphs): %.1f MiB\n", (double(free_before) - double(free_after)) / 1048576.0);
    }
    std::printf("\ndecode: %d tokens in %.2f s (%.2f s/token)\n", new_tokens, decode_seconds, decode_seconds / std::max(new_tokens, 1));
    print_tiers("decode after the prefill", experts.stats(), decode_seconds, new_tokens);
    const std::vector<std::int32_t> first_pass = generated;
    if (!std::getenv("NINFER_DSV4_TOKENS") && new_tokens >= 4 && !forward.engram_applied()) {
        const std::int32_t expected[4] = {11111, 16, 4588, 455};
        check(std::equal(expected, expected + 4, generated.begin()), "greedy ids continue 'The capital of France is' as before");
    }

    // The same prompt and continuation again with the caches warm: identical tokens, and the speed a
    // repeated workload sees.
    for (int pass = 0; pass < 5; ++pass) {
        if (pass == 4) { forward.reset_profile(); } // the report below covers the last, fully warm pass
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
        print_tiers(pass == 0 ? "warm decode, pass 1 (same text again)" : pass == 1 ? "warm decode, pass 2" : pass == 2 ? "warm decode, pass 3" : pass == 3 ? "warm decode, pass 4" : "warm decode, pass 5", experts.stats(), seconds_since(start), new_tokens);
        check(again == first_pass, "the second decode of the same text gives the same ids");
        if (pass == 4) { std::printf("%s", forward.profile_report().c_str()); }
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

    // Checkpoints (prefix reuse): the state of an earlier position, captured and restored, equals having run to
    // that position. A 200-token sequence is run to a position past the first compressed row of the longest
    // ratio and off every compressor's group boundary, with the checkpoint taken there; a different continuation
    // then overwrites the state after it; the restore and the rest of the sequence must give the logits (and the
    // greedy tokens after them) of the same chunks run from scratch.
    {
        const std::size_t n = static_cast<std::size_t>(env_number("NINFER_DSV4_CKPT_TOKENS", 200)), k = 137;
        std::vector<std::int32_t> pool(prompt.begin(), prompt.end());
        for (const std::int32_t id : {1500, 2210, 3071, 4096, 5120, 6144, 7777, 8888, 9999, 12345, 20001, 30303}) { pool.push_back(id); }
        std::vector<std::int32_t> seq(n), other(70);
        std::uint32_t lcg = 12345;
        for (auto& t : seq) { lcg = lcg * 1664525U + 1013904223U; t = pool[(lcg >> 16) % pool.size()]; }
        for (auto& t : other) { lcg = lcg * 1664525U + 1013904223U; t = pool[(lcg >> 16) % pool.size()]; }
        const std::size_t chunk = forward_options.limits.max_chunk;
        // Runs tokens[from, to) in chunks; returns the last chunk's logits (the last position's).
        const auto run_range = [&](const std::vector<std::int32_t>& tokens, std::size_t from, std::size_t to) {
            const ninfer::Tensor* last = nullptr;
            for (std::size_t at = from; at < to;) {
                const std::size_t width = std::min(chunk, to - at);
                last = &forward.run(std::span<const std::int32_t>(tokens).subspan(at, width), static_cast<std::int32_t>(at));
                at += width;
            }
            cudaStreamSynchronize(stream);
            return download(*last);
        };
        const auto greedy = [&](int count) {
            std::vector<std::int32_t> ids;
            std::int32_t pos = forward.position(), tok = forward.argmax();
            for (int i = 0; i < count; ++i) {
                ids.push_back(tok);
                forward.run(std::span<const std::int32_t>(&tok, 1), pos++);
                tok = forward.argmax();
            }
            return ids;
        };
        const auto difference = [](const std::vector<float>& a, const std::vector<float>& b) {
            double num = 0, den = 0, peak = 0;
            for (std::size_t i = 0; i < a.size(); ++i) {
                num += (a[i] - b[i]) * (a[i] - b[i]);
                den += a[i] * a[i];
                peak = std::max(peak, std::fabs(double(a[i]) - double(b[i])));
            }
            return std::pair<double, double>(std::sqrt(num / std::max(den, 1e-30)), peak);
        };
        // From scratch, the chunks split at the checkpoint position.
        run_range(seq, 0, k);
        const std::vector<float> scratch = run_range(seq, k, n);
        const std::vector<std::int32_t> scratch_ids = greedy(4);

        // Capture at k, overwrite what follows with a different continuation, restore, finish the sequence.
        run_range(seq, 0, k);
        ninfer::models::dsv4::ForwardCheckpoint checkpoint;
        forward.capture(checkpoint);
        std::vector<std::int32_t> diverged(seq.begin(), seq.begin() + static_cast<std::ptrdiff_t>(k));
        diverged.insert(diverged.end(), other.begin(), other.end());
        run_range(diverged, k, diverged.size());
        check(forward.position() == static_cast<std::int32_t>(diverged.size()), "the diverging continuation advanced the state");
        forward.restore(checkpoint);
        check(forward.position() == static_cast<std::int32_t>(k), "restore returns the sequence to the captured position");
        const std::vector<float> restored = run_range(seq, k, n);
        const std::vector<std::int32_t> restored_ids = greedy(4);
        const auto [rel, peak] = difference(scratch, restored);
        std::printf("checkpoint of %zu positions: %.2f MiB on the device; restored vs scratch logits: relative L2 %.3g, max abs %.3g\n", k,
                    static_cast<double>(checkpoint.device_bytes()) / 1048576.0, rel, peak);
        check(rel == 0.0, "the restored state gives the very logits of a run from scratch");
        check(restored_ids == scratch_ids, "greedy tokens after a restore equal those of a run from scratch");

        // A second restore of the same checkpoint (a retry of the same prompt) works the same.
        forward.restore(checkpoint);
        const std::vector<float> again = run_range(seq, k, n);
        check(difference(scratch, again).second == 0.0, "a checkpoint restores more than once");
    }
    if (failures == 0) { std::printf("PASS\n"); }
    return failures == 0 ? 0 : 1;
}
