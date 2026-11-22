// Written for NInfer-3090.
//
// Greedy speculative decoding of Inkling-2-Mini with its DSpark drafter (src/models/inkling2/program/
// dspark_program.h) against plain greedy decoding, on the real artifact. Every round drafts block_size - 1 tokens
// after the anchor, verifies [anchor, drafts] in one target forward, keeps the anchor and the drafts the target's
// argmax agrees with (TextProgram::keep cuts the rest back), and commits their taps to the drafter.
//
// Criterion: the speculative tokens equal the plain greedy tokens, every one (a verify chunk's logits are the same
// bits as one token at a time, and keep() restores the state exactly, so any difference is a bug). Reported, not
// judged: drafts accepted per round and the speed of both.
//
// Environment: NINFER_TEST_INKLING2_ARTIFACT (an artifact with component dspark), NINFER_INKLING2_GOLDEN (the
// golden output directory: its p0, p1, ... tokens.json are the prompts); skips (77) without them.
// NINFER_INKLING2_SPEC_TOKENS: tokens to generate per prompt (default 64).

#include "artifact/reader.h"
#include "artifact/schema.h"
#include "core/device.h"
#include "core/tensor.h"
#include "models/inkling2/model.h"
#include "models/inkling2/program/dspark_program.h"
#include "models/inkling2/program/text_program.h"
#include "ninfer/ops/dspark_fp32.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

namespace ink = ninfer::models::inkling2;
namespace fs  = std::filesystem;

int failures = 0;

using Clock = std::chrono::steady_clock;
double since(Clock::time_point t) { return std::chrono::duration<double>(Clock::now() - t).count(); }

// The greedy token of every column of `logits` (FP32 [vocab, T] on the device).
std::vector<std::int32_t> greedy(const ninfer::Tensor& logits, std::int32_t valid, ninfer::DeviceBuffer& out, cudaStream_t stream) {
    ninfer::Tensor ids(out.p, ninfer::DType::I32, {logits.ne[1]});
    ninfer::ops::argmax_fp32(logits, valid, ids, stream);
    std::vector<std::int32_t> host(static_cast<std::size_t>(logits.ne[1]));
    if (cudaMemcpyAsync(host.data(), out.p, host.size() * 4, cudaMemcpyDeviceToHost, stream) != cudaSuccess ||
        cudaStreamSynchronize(stream) != cudaSuccess) {
        throw std::runtime_error("argmax download failed");
    }
    return host;
}

std::string join(const std::vector<std::int32_t>& v, std::size_t n) {
    std::string s;
    for (std::size_t i = 0; i < std::min(n, v.size()); ++i) { s += (i ? " " : "") + std::to_string(v[i]); }
    return s;
}

} // namespace

int main() {
    const char* artifact_path = std::getenv("NINFER_TEST_INKLING2_ARTIFACT");
    const char* golden_root   = std::getenv("NINFER_INKLING2_GOLDEN");
    if (artifact_path == nullptr || golden_root == nullptr) {
        std::printf("SKIP: set NINFER_TEST_INKLING2_ARTIFACT and NINFER_INKLING2_GOLDEN\n");
        return 77;
    }
    const int n_tokens = std::getenv("NINFER_INKLING2_SPEC_TOKENS") ? std::atoi(std::getenv("NINFER_INKLING2_SPEC_TOKENS")) : 128;
    // NINFER_INKLING2_SPEC_DRAFTS: drafts per round (default 3, as `--spec dspark` serves; `all`: block_size - 1).
    const bool adaptive    = std::getenv("NINFER_INKLING2_SPEC_ADAPTIVE") != nullptr;
    const double threshold = adaptive ? std::atof(std::getenv("NINFER_INKLING2_SPEC_ADAPTIVE")) : 0.0;
    const double ema_weight = std::getenv("NINFER_INKLING2_SPEC_EMA") ? std::atof(std::getenv("NINFER_INKLING2_SPEC_EMA")) : 0.3;
    const char* drafts_env = std::getenv("NINFER_INKLING2_SPEC_DRAFTS");
    const int max_drafts   = drafts_env == nullptr ? 3 : std::string_view(drafts_env) == "all" ? 1 << 30 : std::atoi(drafts_env);
    try {
        ninfer::artifact::Reader reader(artifact_path);
        ninfer::DeviceContext device(0);
        auto model = ink::load_model(reader, device);
        if (model->dspark_config() == nullptr) { throw std::runtime_error("the artifact has no DSpark drafter"); }
        const auto& dc   = *model->dspark_config();
        const auto vocab = static_cast<std::int32_t>(model->config().unpadded_vocab_size);
        const auto block = static_cast<std::int32_t>(dc.block_size);
        cudaStream_t stream = nullptr;
        if (cudaStreamCreate(&stream) != cudaSuccess) { throw std::runtime_error("cannot create a stream"); }
        constexpr std::int32_t kContext = 4096, kChunk = 64;
        ink::ProgramLimits plain_limits{kContext, kChunk};
        plain_limits.decode_graph = std::getenv("NINFER_INKLING2_PLAIN_EAGER") == nullptr;
        ink::TextProgram plain(*model, plain_limits, stream);
        ink::ProgramLimits limits{kContext, kChunk};
        limits.tap_visits   = dc.target_visits;
        limits.max_rollback = block;
        ink::TextProgram target(*model, limits, stream);
        const auto p0 = Clock::now();
        ink::DSparkProgram drafter(*model, kContext, kChunk, stream);
        std::printf("drafter ready in %.2f s (taps at visits %s)\n", since(p0),
                    join(std::vector<std::int32_t>(dc.target_visits.begin(), dc.target_visits.end()), 8).c_str());
        ninfer::DeviceBuffer ids(64 * 4);
        // Plain steps' taps, held until the drafter next runs (one extend for the stretch).
        const auto hidden = static_cast<std::int32_t>(model->config().hidden_size);
        std::vector<ninfer::DeviceBuffer> held_memory;
        std::vector<ninfer::Tensor> held;
        for (std::size_t i = 0; i < dc.target_visits.size(); ++i) {
            held_memory.emplace_back(static_cast<std::size_t>(hidden) * kChunk * 4);
            held.emplace_back(held_memory.back().p, ninfer::DType::FP32, std::initializer_list<std::int32_t>{hidden, kChunk});
        }

        double plain_seconds = 0, spec_seconds = 0, draft_seconds = 0, verify_seconds = 0;
        long rounds = 0, accepted = 0, generated = 0;
        for (int index = 0;; ++index) {
            const fs::path dir = fs::path(golden_root) / ("p" + std::to_string(index));
            if (!fs::exists(dir / "tokens.json")) { break; }
            std::ifstream meta_file(dir / "tokens.json");
            std::stringstream text;
            text << meta_file.rdbuf();
            const auto prompt = ninfer::artifact::Json::parse(text.str()).at("tokens").get<std::vector<std::int32_t>>();
            if (static_cast<std::int32_t>(prompt.size()) > kChunk) { throw std::runtime_error("prompt longer than one chunk"); }

            // Plain greedy.
            plain.reset();
            std::vector<std::int32_t> want{greedy(plain.forward(prompt), vocab, ids, stream).back()};
            auto t = Clock::now();
            while (static_cast<int>(want.size()) < n_tokens) {
                const std::int32_t one[] = {want.back()};
                want.push_back(greedy(plain.forward(one), vocab, ids, stream).back());
            }
            const double prompt_plain = since(t);
            plain_seconds += prompt_plain;

            // Speculative greedy.
            target.reset();
            drafter.reset();
            std::vector<std::int32_t> got{greedy(target.forward(prompt), vocab, ids, stream).back()};
            {
                std::vector<ninfer::Tensor> taps;
                for (std::size_t i = 0; i < dc.target_visits.size(); ++i) { taps.push_back(target.tap(i)); }
                drafter.extend(taps, static_cast<std::int32_t>(prompt.size()));
            }
            long prompt_rounds = 0, prompt_accepted = 0, plain_steps = 0;
            // Adaptive (NINFER_INKLING2_SPEC_ADAPTIVE): an EMA of drafts accepted per round; below the break-even the
            // next rounds are plain steps (the drafter still takes their taps), with a speculative probe every 8.
            double ema = 1.0;
            int since_probe = 0, interval = 8, held_n = 0;
            t = Clock::now();
            while (static_cast<int>(got.size()) < n_tokens) {
                const std::int32_t anchor = got.back();
                if (adaptive && ema < threshold && ++since_probe < interval && held_n < kChunk) {
                    const std::int32_t one[] = {anchor};
                    const auto next          = greedy(target.forward(one), vocab, ids, stream).back();
                    for (std::size_t i = 0; i < dc.target_visits.size(); ++i) {
                        const ninfer::Tensor tap = target.tap(i);
                        if (cudaMemcpyAsync(static_cast<float*>(held[i].data) + static_cast<std::size_t>(held_n) * hidden, tap.data,
                                            static_cast<std::size_t>(hidden) * 4, cudaMemcpyDeviceToDevice, stream) != cudaSuccess) {
                            throw std::runtime_error("tap hold failed");
                        }
                    }
                    ++held_n;
                    got.push_back(next);
                    ++plain_steps;
                    continue;
                }
                if (held_n > 0) {
                    drafter.extend(held, held_n);
                    held_n = 0;
                }
                since_probe               = 0;
                const auto d0             = Clock::now();
                const auto drafts         = drafter.draft(anchor, std::min(max_drafts, block - 1));
                draft_seconds += since(d0);
                std::vector<std::int32_t> verify{anchor};
                verify.insert(verify.end(), drafts.begin(), drafts.end());
                const auto v0     = Clock::now();
                const auto choice = greedy(target.forward(verify), vocab, ids, stream);
                verify_seconds += since(v0);
                std::int32_t n = 0; // drafts accepted
                while (n < static_cast<std::int32_t>(drafts.size()) && drafts[n] == choice[n]) { ++n; }
                target.keep(n + 1);
                std::vector<ninfer::Tensor> taps;
                for (std::size_t i = 0; i < dc.target_visits.size(); ++i) { taps.push_back(target.tap(i)); }
                drafter.extend(taps, n + 1);
                got.insert(got.end(), drafts.begin(), drafts.begin() + n);
                got.push_back(choice[n]);
                ++prompt_rounds;
                prompt_accepted += n;
                ema = (1 - ema_weight) * ema + ema_weight * n;
                // A probe that fails doubles the wait before the next (to 64 plain steps); a good round resets it.
                interval = ema < threshold ? std::min(2 * interval, 64) : 8;
            }
            const double prompt_spec = since(t);
            spec_seconds += prompt_spec;
            rounds += prompt_rounds;
            accepted += prompt_accepted;
            generated += n_tokens - 1;
            bool same = true;
            for (int i = 0; i < n_tokens; ++i) { same = same && got[i] == want[i]; }
            std::printf("p%d (%zu prompt tokens): %.1f vs %.1f tok/s; ", index, prompt.size(), (n_tokens - 1) / prompt_spec,
                        (n_tokens - 1) / prompt_plain);
            std::printf("%ld rounds, %.2f drafts accepted per round, %ld plain steps; %s\n",
                        prompt_rounds, double(prompt_accepted) / std::max(prompt_rounds, 1L), plain_steps,
                        same ? "same tokens as plain greedy" : "DIFFERENT tokens");
            if (!same) {
                std::fprintf(stderr, "FAIL: p%d speculative tokens differ\n  plain: %s\n  spec:  %s\n", index,
                             join(want, n_tokens).c_str(), join(got, n_tokens).c_str());
                ++failures;
            }
        }
        if (rounds == 0) { throw std::runtime_error("no prompts"); }
        {
            // Snapshots (prefix reuse): restored after more tokens went in, the target and the drafter continue as at
            // the save: the same next tokens and the same drafts.
            std::ifstream meta_file(fs::path(golden_root) / "p0" / "tokens.json");
            std::stringstream text;
            text << meta_file.rdbuf();
            const auto prompt = ninfer::artifact::Json::parse(text.str()).at("tokens").get<std::vector<std::int32_t>>();
            target.reset();
            drafter.reset();
            const std::int32_t anchor = greedy(target.forward(prompt), vocab, ids, stream).back();
            std::vector<ninfer::Tensor> taps;
            for (std::size_t i = 0; i < dc.target_visits.size(); ++i) { taps.push_back(target.tap(i)); }
            drafter.extend(taps, static_cast<std::int32_t>(prompt.size()));
            ink::TextProgram::Snapshot target_snapshot{ninfer::DeviceBuffer(target.snapshot_bytes())};
            ink::DSparkProgram::Snapshot drafter_snapshot{ninfer::DeviceBuffer(drafter.snapshot_bytes())};
            target.save(target_snapshot);
            drafter.save(drafter_snapshot);
            const auto drafts_before = drafter.draft(anchor, block - 1);
            const std::int32_t one[] = {anchor};
            const auto next_before   = greedy(target.forward(one), vocab, ids, stream).back();
            // More tokens into both, then back.
            const auto noise = std::vector<std::int32_t>(prompt.begin(), prompt.begin() + std::min<std::size_t>(prompt.size(), 16));
            (void)target.forward(noise);
            taps.clear();
            for (std::size_t i = 0; i < dc.target_visits.size(); ++i) { taps.push_back(target.tap(i)); }
            drafter.extend(taps, static_cast<std::int32_t>(noise.size()));
            target.restore(target_snapshot);
            drafter.restore(drafter_snapshot);
            const auto drafts_after = drafter.draft(anchor, block - 1);
            const auto next_after   = greedy(target.forward(one), vocab, ids, stream).back();
            const bool same = drafts_after == drafts_before && next_after == next_before && target.position() == static_cast<std::int64_t>(prompt.size()) + 1;
            std::printf("snapshot restore: %s\n", same ? "the same drafts and next token" : "DIFFERENT");
            if (!same) { ++failures; }
        }
        std::printf("plain greedy: %.2f tok/s; speculative: %.2f tok/s (%.2f drafts accepted per round, %.1f%% of the time drafting)\n",
                    generated / plain_seconds, generated / spec_seconds, double(accepted) / rounds, 100 * draft_seconds / spec_seconds);
        std::printf("per round: draft %.2f ms, verify %.2f ms; plain step %.2f ms\n", 1e3 * draft_seconds / rounds,
                    1e3 * verify_seconds / rounds, 1e3 * plain_seconds / generated);
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL: %s\n", error.what());
        return 1;
    }
    std::printf(failures ? "FAILED\n" : "PASS\n");
    return failures ? 1 : 0;
}
