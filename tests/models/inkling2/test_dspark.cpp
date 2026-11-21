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

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
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
    const int n_tokens = std::getenv("NINFER_INKLING2_SPEC_TOKENS") ? std::atoi(std::getenv("NINFER_INKLING2_SPEC_TOKENS")) : 64;
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
        ink::TextProgram plain(*model, {kContext, kChunk}, stream);
        ink::ProgramLimits limits{kContext, kChunk};
        limits.tap_visits   = dc.target_visits;
        limits.max_rollback = block;
        ink::TextProgram target(*model, limits, stream);
        const auto p0 = Clock::now();
        ink::DSparkProgram drafter(*model, kContext, kChunk, stream);
        std::printf("drafter ready in %.2f s (taps at visits %s)\n", since(p0),
                    join(std::vector<std::int32_t>(dc.target_visits.begin(), dc.target_visits.end()), 8).c_str());
        ninfer::DeviceBuffer ids(64 * 4);

        double plain_seconds = 0, spec_seconds = 0, draft_seconds = 0;
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
            plain_seconds += since(t);

            // Speculative greedy.
            target.reset();
            drafter.reset();
            std::vector<std::int32_t> got{greedy(target.forward(prompt), vocab, ids, stream).back()};
            {
                std::vector<ninfer::Tensor> taps;
                for (std::size_t i = 0; i < dc.target_visits.size(); ++i) { taps.push_back(target.tap(i)); }
                drafter.extend(taps, static_cast<std::int32_t>(prompt.size()));
            }
            long prompt_rounds = 0, prompt_accepted = 0;
            t = Clock::now();
            while (static_cast<int>(got.size()) < n_tokens) {
                const std::int32_t anchor = got.back();
                const auto d0             = Clock::now();
                const auto drafts         = drafter.draft(anchor);
                draft_seconds += since(d0);
                std::vector<std::int32_t> verify{anchor};
                verify.insert(verify.end(), drafts.begin(), drafts.end());
                const auto choice = greedy(target.forward(verify), vocab, ids, stream);
                std::int32_t n = 0; // drafts accepted
                while (n < block - 1 && drafts[n] == choice[n]) { ++n; }
                target.keep(n + 1);
                std::vector<ninfer::Tensor> taps;
                for (std::size_t i = 0; i < dc.target_visits.size(); ++i) { taps.push_back(target.tap(i)); }
                drafter.extend(taps, n + 1);
                got.insert(got.end(), drafts.begin(), drafts.begin() + n);
                got.push_back(choice[n]);
                ++prompt_rounds;
                prompt_accepted += n;
            }
            spec_seconds += since(t);
            rounds += prompt_rounds;
            accepted += prompt_accepted;
            generated += n_tokens - 1;
            bool same = true;
            for (int i = 0; i < n_tokens; ++i) { same = same && got[i] == want[i]; }
            std::printf("p%d (%zu prompt tokens): %ld rounds, %.2f drafts accepted per round; %s\n", index, prompt.size(), prompt_rounds,
                        double(prompt_accepted) / prompt_rounds, same ? "same tokens as plain greedy" : "DIFFERENT tokens");
            if (!same) {
                std::fprintf(stderr, "FAIL: p%d speculative tokens differ\n  plain: %s\n  spec:  %s\n", index,
                             join(want, n_tokens).c_str(), join(got, n_tokens).c_str());
                ++failures;
            }
        }
        if (rounds == 0) { throw std::runtime_error("no prompts"); }
        std::printf("plain greedy: %.2f tok/s; speculative: %.2f tok/s (%.2f drafts accepted per round, %.1f%% of the time drafting)\n",
                    generated / plain_seconds, generated / spec_seconds, double(accepted) / rounds, 100 * draft_seconds / spec_seconds);
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL: %s\n", error.what());
        return 1;
    }
    std::printf(failures ? "FAILED\n" : "PASS\n");
    return failures ? 1 : 0;
}
