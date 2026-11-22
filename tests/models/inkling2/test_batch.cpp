// Written for NInfer-3090.
//
// Batched decode of Inkling-2-Mini (TextProgram::forward_batch) against each request decoded alone, on the real
// artifact. Every prompt first runs alone through a one-slot Program (the Engine's single-request path: chunked
// prefill, then the single-token decode graph), greedy, keeping each step's logits. Then the same prompts run through
// one Program of kMaxBatchColumns slots with at most B in flight at once (B = 2, 4, 8): requests arrive staggered (one
// every few steps, waiting for a free slot), each prefills alone in its slot between decode steps, and every step
// decodes all the requests in flight as the columns of one forward_batch. They finish at different lengths, so the
// batches shrink and grow and the slots are reused.
//
// Criterion: every request's prefill logits and every decode step's logits column are byte-equal to its solo run's
// (so are its tokens). The prompt lengths put decode steps on both sides of the split attention's threshold
// (ops::kSplitMinKeys) and past the sliding window.
//
// Environment: NINFER_TEST_INKLING2_ARTIFACT; skips (77) without it. NINFER_INKLING2_BATCH_TOKENS: tokens per
// request (default 48). NINFER_INKLING2_KV=fp32: FP32 rings (default FP16, the Engine's). NINFER_INKLING2_BATCH_BENCH=1
// also times full batches of B = 1, 2, 4, 8 (64 steps each after a warm-up step, at a context of about 1,000);
// NINFER_INKLING2_BATCH_WIDTH=B times only that B and skips the comparison (for a profiler).

#include "artifact/reader.h"
#include "core/device.h"
#include "core/tensor.h"
#include "models/inkling2/model.h"
#include "models/inkling2/program/text_program.h"

#include <cuda_profiler_api.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

namespace ink = ninfer::models::inkling2;
using Clock   = std::chrono::steady_clock;

int failures = 0;

void check(bool ok, const std::string& what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what.c_str());
        ++failures;
    }
}

// Column t's first `vocab` logits.
std::vector<float> column(const ninfer::Tensor& logits, std::int32_t t, std::int32_t vocab, cudaStream_t stream) {
    std::vector<float> out(static_cast<std::size_t>(vocab));
    if (cudaMemcpyAsync(out.data(), static_cast<const char*>(logits.data) + static_cast<std::size_t>(t) * logits.nb[1], out.size() * 4,
                        cudaMemcpyDeviceToHost, stream) != cudaSuccess ||
        cudaStreamSynchronize(stream) != cudaSuccess) {
        throw std::runtime_error("logits download failed");
    }
    return out;
}

std::int32_t argmax(const std::vector<float>& x) {
    return static_cast<std::int32_t>(std::max_element(x.begin(), x.end()) - x.begin()); // the lowest id among equals
}

struct Prompt {
    std::vector<std::int32_t> tokens;
    std::int32_t decode = 0;
};

// The logits after the prompt (the first token's), then after each forwarded generated token.
struct Run {
    std::vector<std::vector<float>> logits;
    std::vector<std::int32_t> tokens;
};

void prefill(ink::TextProgram& program, const Prompt& p, std::int32_t chunk, std::vector<float>& last, std::int32_t vocab,
             cudaStream_t stream) {
    ninfer::Tensor logits;
    const auto n = static_cast<std::int32_t>(p.tokens.size());
    for (std::int32_t s = 0; s < n; s += chunk) {
        const std::int32_t len = std::min(chunk, n - s);
        logits = program.forward(std::span(p.tokens).subspan(static_cast<std::size_t>(s), static_cast<std::size_t>(len)));
    }
    last = column(logits, logits.ne[1] - 1, vocab, stream);
}

} // namespace

int main() {
    const char* artifact_path = std::getenv("NINFER_TEST_INKLING2_ARTIFACT");
    if (artifact_path == nullptr) {
        std::printf("SKIP: set NINFER_TEST_INKLING2_ARTIFACT\n");
        return 77;
    }
    const int n_tokens = std::getenv("NINFER_INKLING2_BATCH_TOKENS") ? std::atoi(std::getenv("NINFER_INKLING2_BATCH_TOKENS")) : 48;
    const bool fp32    = std::getenv("NINFER_INKLING2_KV") != nullptr && std::string(std::getenv("NINFER_INKLING2_KV")) == "fp32";
    const int only     = std::getenv("NINFER_INKLING2_BATCH_WIDTH") ? std::atoi(std::getenv("NINFER_INKLING2_BATCH_WIDTH")) : 0;
    const bool bench   = std::getenv("NINFER_INKLING2_BATCH_BENCH") != nullptr || only > 0;
    try {
        ninfer::artifact::Reader reader(artifact_path);
        ninfer::DeviceContext device(0);
        auto model = ink::load_model(reader, device);
        const auto& c    = model->config();
        const auto vocab = static_cast<std::int32_t>(c.unpadded_vocab_size);
        cudaStream_t stream = nullptr;
        if (cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) != cudaSuccess) { throw std::runtime_error("cannot create a stream"); }

        // The Engine's limits (FP16 rings, tensor-core prefill, split attention, last-column prefill logits).
        constexpr std::int32_t kChunk = 256;
        ink::ProgramLimits limits{4096, kChunk};
        limits.kv_fp16     = !fp32;
        limits.last_logits = true;
        limits.prefill_tc  = ink::ProgramLimits::TensorCoreParts::all();
        std::printf("KV rings %s; %d tokens a request\n", fp32 ? "FP32" : "FP16", n_tokens);

        // Prompts across the split threshold (512 keys) and the sliding window (1024), finishing at different lengths.
        const std::int32_t lengths[] = {37, 300, 470, 505, 700, 1030, 1500, 2100};
        std::vector<Prompt> prompts;
        std::uint32_t seed = 12345;
        for (std::size_t i = 0; i < std::size(lengths); ++i) {
            Prompt p;
            for (std::int32_t j = 0; j < lengths[i]; ++j) {
                seed = seed * 1664525U + 1013904223U;
                p.tokens.push_back(static_cast<std::int32_t>(1000 + (seed >> 8) % 120000));
            }
            p.decode = std::max(4, n_tokens - static_cast<std::int32_t>(i) * 3);
            prompts.push_back(std::move(p));
        }

        // Each request alone, through the single-sequence path.
        std::vector<Run> solo(prompts.size());
        if (only == 0) {
            ink::TextProgram program(*model, limits, stream);
            for (std::size_t r = 0; r < prompts.size(); ++r) {
                program.reset();
                std::vector<float> last;
                prefill(program, prompts[r], kChunk, last, vocab, stream);
                for (std::int32_t k = 0;; ++k) {
                    solo[r].logits.push_back(last);
                    solo[r].tokens.push_back(argmax(last));
                    if (k + 1 == prompts[r].decode) { break; }
                    const std::int32_t one[] = {solo[r].tokens.back()};
                    last = column(program.forward(one), 0, vocab, stream);
                }
            }
        }
        std::printf("solo runs done\n");

        ink::ProgramLimits batch_limits = limits;
        batch_limits.slots = ink::kMaxBatchColumns;
        ink::TextProgram program(*model, batch_limits, stream);
        for (const std::int32_t in_flight : {2, 4, 8}) {
            if (only > 0) { break; }
            struct Active {
                std::size_t request;
                std::int32_t slot;
                std::int32_t done = 0; // tokens generated
                std::int32_t next = 0; // the token to forward
            };
            std::vector<Active> active;
            std::deque<std::size_t> waiting;
            std::vector<bool> slot_used(ink::kMaxBatchColumns, false);
            std::size_t arrived = 0, finished = 0;
            std::int32_t columns_max = 0, step = 0, mismatched = 0;
            const auto compare = [&](std::size_t r, std::int32_t k, const std::vector<float>& got) {
                const auto& want = solo[r].logits[static_cast<std::size_t>(k)];
                if (std::memcmp(want.data(), got.data(), got.size() * 4) != 0) {
                    if (mismatched++ < 8) {
                        check(false, "B " + std::to_string(in_flight) + ": request " + std::to_string(r) + " logits of token " +
                                         std::to_string(k) + " differ from its solo run");
                    }
                }
            };
            while (finished < prompts.size()) {
                // One arrival every 3 steps; a waiting request takes the lowest free slot and prefills alone.
                if (step % 3 == 0 && arrived < prompts.size()) { waiting.push_back(arrived++); }
                while (!waiting.empty() && static_cast<std::int32_t>(active.size()) < in_flight) {
                    const std::size_t r = waiting.front();
                    waiting.pop_front();
                    const auto slot = static_cast<std::int32_t>(std::find(slot_used.begin(), slot_used.end(), false) - slot_used.begin());
                    slot_used[static_cast<std::size_t>(slot)] = true;
                    program.select(slot);
                    program.reset();
                    std::vector<float> last;
                    prefill(program, prompts[r], kChunk, last, vocab, stream);
                    compare(r, 0, last);
                    Active a{r, slot, 1, argmax(last)};
                    if (a.done == prompts[r].decode) {
                        slot_used[static_cast<std::size_t>(slot)] = false;
                        ++finished;
                    } else {
                        active.push_back(a);
                    }
                }
                ++step;
                if (active.empty()) { continue; }
                std::vector<std::int32_t> slots, tokens;
                for (const Active& a : active) {
                    slots.push_back(a.slot);
                    tokens.push_back(a.next);
                }
                columns_max = std::max(columns_max, static_cast<std::int32_t>(active.size()));
                const ninfer::Tensor logits = program.forward_batch(slots, tokens);
                for (std::size_t i = 0; i < active.size(); ++i) {
                    Active& a = active[i];
                    const auto got = column(logits, static_cast<std::int32_t>(i), vocab, stream);
                    compare(a.request, a.done, got);
                    a.next = argmax(got);
                    ++a.done;
                }
                for (std::size_t i = active.size(); i-- > 0;) {
                    if (active[i].done == prompts[active[i].request].decode) {
                        slot_used[static_cast<std::size_t>(active[i].slot)] = false;
                        ++finished;
                        active.erase(active.begin() + static_cast<std::ptrdiff_t>(i));
                    }
                }
            }
            check(columns_max == in_flight, "B " + std::to_string(in_flight) + ": the batch never reached its width");
            std::printf("B %d (%zu requests, %d steps, widest batch %d): %s\n", in_flight, prompts.size(), step, columns_max,
                        mismatched == 0 ? "every logit byte-equal to the solo runs" : (std::to_string(mismatched) + " columns differ").c_str());
        }

        if (only == 0) {
            // A tapped one-token forward() in a multi-slot Program (the DSpark target's plain step) keeps its taps:
            // it runs the single-sequence path for its slot, with the same logits and taps as a one-slot Program.
            ink::ProgramLimits tapped = limits;
            tapped.tap_visits         = {2, c.num_layer_visits / 2};
            ink::ProgramLimits tapped_slots = tapped;
            tapped_slots.slots              = 2;
            ink::TextProgram single(*model, tapped, stream), multi(*model, tapped_slots, stream);
            multi.select(1);
            std::vector<float> last_single, last_multi;
            prefill(single, prompts[1], kChunk, last_single, vocab, stream);
            prefill(multi, prompts[1], kChunk, last_multi, vocab, stream);
            const std::int32_t one[] = {argmax(last_single)};
            const auto ls = column(single.forward(one), 0, vocab, stream), lm = column(multi.forward(one), 0, vocab, stream);
            bool same = ls == lm;
            for (std::size_t i = 0; same && i < tapped.tap_visits.size(); ++i) {
                const ninfer::Tensor ts = single.tap(i), tm = multi.tap(i);
                same = ts.ne[1] == 1 && tm.ne[1] == 1 && column(ts, 0, ts.ne[0], stream) == column(tm, 0, tm.ne[0], stream);
            }
            check(same, "a tapped one-token forward in a multi-slot Program keeps the single-slot logits and taps");
        }

        if (bench) {
            // Full batches at about 1,000 positions each: aggregate tok/s per B. Each column decodes a different token
            // (one shared token would route every column to the same experts, which flatters the wider batches).
            Prompt p = prompts[5];
            std::vector<float> last;
            for (std::int32_t s = 0; s < ink::kMaxBatchColumns; ++s) {
                program.select(s);
                program.reset();
                prefill(program, p, kChunk, last, vocab, stream);
            }
            for (const std::int32_t b : {1, 2, 4, 8}) {
                if (only > 0 && b != only) { continue; }
                std::vector<std::int32_t> slots(static_cast<std::size_t>(b)), tokens(static_cast<std::size_t>(b), argmax(last));
                for (std::int32_t i = 0; i < b; ++i) {
                    slots[static_cast<std::size_t>(i)] = i;
                    if (i > 0) { tokens[static_cast<std::size_t>(i)] = p.tokens[(static_cast<std::size_t>(i) * 97) % p.tokens.size()]; }
                }
                (void)program.forward_batch(slots, tokens);
                cudaStreamSynchronize(stream);
                const auto t0 = Clock::now();
                constexpr int kSteps = 64;
                cudaProfilerStart();
                for (int k = 0; k < kSteps; ++k) { (void)program.forward_batch(slots, tokens); }
                cudaStreamSynchronize(stream);
                cudaProfilerStop();
                const double s = std::chrono::duration<double>(Clock::now() - t0).count();
                std::printf("bench B %d: %.2f ms a step, %.1f tok/s aggregate, %.1f a request\n", b, 1e3 * s / kSteps, b * kSteps / s,
                            kSteps / s);
            }
            // The single-sequence graph at the same context, for reference.
            if (only > 0) {
                cudaStreamDestroy(stream);
                std::printf("PASS\n");
                return 0;
            }
            ink::TextProgram one(*model, limits, stream);
            prefill(one, p, kChunk, last, vocab, stream);
            const std::int32_t tok[] = {argmax(last)};
            (void)one.forward(tok);
            cudaStreamSynchronize(stream);
            const auto t0 = Clock::now();
            for (int k = 0; k < 64; ++k) { (void)one.forward(tok); }
            cudaStreamSynchronize(stream);
            const double s = std::chrono::duration<double>(Clock::now() - t0).count();
            std::printf("bench single-sequence graph: %.2f ms a step, %.1f tok/s\n", 1e3 * s / 64, 64 / s);
        }
        cudaStreamDestroy(stream);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL: %s\n", e.what());
        return 1;
    }
    if (failures == 0) { std::printf("PASS\n"); }
    return failures == 0 ? 0 : 1;
}
