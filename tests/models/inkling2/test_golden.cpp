// Written for NInfer-3090.
//
// Inkling-2-Mini end to end on the real artifact against the CPU llama.cpp golden (tools/inkling2/llama_golden.cpp
// on UD-Q8_K_XL, docs/maintainer/inkling2-model.md "Validation"): for every golden prompt, every layer visit's
// attn_out, ffn_inp and l_out and the final norm (relative L2 error), and the logits (KLD of the engine's
// distribution from llama.cpp's per position, and the argmax). The prompt runs whole, then again in chunks of 3
// (state across calls), which must agree with the whole run. Then a greedy continuation from the first prompt,
// timed per decode step.
//
// Needs NINFER_TEST_INKLING2_ARTIFACT (.../inkling2-mini-q8.ninfer) and NINFER_INKLING2_GOLDEN (the driver's
// output directory, holding p0, p1, ...); skips (77) without them. Optional NINFER_INKLING2_DECODE (tokens to
// generate, default 32) and NINFER_INKLING2_DUMP: a directory that receives the engine's own logits, traces and
// routing per prompt in the golden's layout, for tools/inkling2/compare_golden.py (the engine against the FP64
// NumPy reference, optionally with the engine's routing forced: --routes).
//
// Acceptance (correctness first): the argmax agrees at every position; per-position KLD at most 0.05 and its mean
// over a prompt at most 0.01; chunked logits within 1e-3 of the whole run's. The spec's bar of 0.002 is for prompts
// without near-tied routing: on the chat prompt (p2) the engine is 2.1e-2 from llama.cpp at worst, the FP64 NumPy
// reference 2.6e-2, and the engine 3.4e-5 from that reference once both use the same experts
// (docs/maintainer/inkling2-model.md "Validation", item 5).

#include "artifact/reader.h"
#include "artifact/schema.h"
#include "core/device.h"
#include "models/inkling2/model.h"
#include "models/inkling2/program/text_program.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <span>
#include <sstream>
#include <string>
#include <vector>

namespace {

namespace ink = ninfer::models::inkling2;
namespace fs  = std::filesystem;

int failures = 0;

void check(bool ok, const std::string& what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what.c_str());
        ++failures;
    }
}

std::vector<float> read_floats(const fs::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) { throw std::runtime_error("cannot read " + path.string()); }
    in.seekg(0, std::ios::end);
    std::vector<float> out(static_cast<std::size_t>(in.tellg()) / sizeof(float));
    in.seekg(0);
    in.read(reinterpret_cast<char*>(out.data()), static_cast<std::streamsize>(out.size() * sizeof(float)));
    return out;
}

void write_file(const fs::path& path, const void* data, std::size_t bytes) {
    std::ofstream out(path, std::ios::binary);
    out.write(static_cast<const char*>(data), static_cast<std::streamsize>(bytes));
    out.close();
    if (!out) { throw std::runtime_error("cannot write " + path.string()); }
}

std::vector<float> to_host(const ninfer::Tensor& t) {
    std::vector<float> out(static_cast<std::size_t>(t.ne[0]) * t.ne[1]);
    if (cudaMemcpy2D(out.data(), t.ne[0] * sizeof(float), t.data, t.nb[1], t.ne[0] * sizeof(float), t.ne[1], cudaMemcpyDeviceToHost) !=
        cudaSuccess) {
        throw std::runtime_error("device copy failed");
    }
    return out;
}

double rel_error(const std::vector<float>& ours, const std::vector<float>& theirs) {
    double num = 0, den = 0;
    for (std::size_t i = 0; i < ours.size(); ++i) {
        const double d = double(ours[i]) - double(theirs[i]);
        num += d * d;
        den += double(theirs[i]) * double(theirs[i]);
    }
    return std::sqrt(num / den);
}

// Logits of the first `vocab` rows of each column of a [rows, T] device tensor.
std::vector<std::vector<float>> logit_columns(const ninfer::Tensor& logits, std::int32_t vocab) {
    const auto all = to_host(logits);
    std::vector<std::vector<float>> out;
    for (std::int32_t t = 0; t < logits.ne[1]; ++t) {
        out.emplace_back(all.begin() + static_cast<std::ptrdiff_t>(t) * logits.ne[0],
                         all.begin() + static_cast<std::ptrdiff_t>(t) * logits.ne[0] + vocab);
    }
    return out;
}

std::vector<double> log_softmax(const float* x, std::size_t n) {
    const double top = *std::max_element(x, x + n);
    double sum = 0;
    for (std::size_t i = 0; i < n; ++i) { sum += std::exp(double(x[i]) - top); }
    std::vector<double> out(n);
    for (std::size_t i = 0; i < n; ++i) { out[i] = double(x[i]) - top - std::log(sum); }
    return out;
}

// KL(golden || ours).
double kld(const float* golden, const float* ours, std::size_t n) {
    const auto lp = log_softmax(golden, n), lq = log_softmax(ours, n);
    double sum = 0;
    for (std::size_t i = 0; i < n; ++i) { sum += std::exp(lp[i]) * (lp[i] - lq[i]); }
    return sum;
}

std::size_t argmax(const float* x, std::size_t n) { return static_cast<std::size_t>(std::max_element(x, x + n) - x); }

} // namespace

int main() {
    const char* artifact_path = std::getenv("NINFER_TEST_INKLING2_ARTIFACT");
    const char* golden_root   = std::getenv("NINFER_INKLING2_GOLDEN");
    if (artifact_path == nullptr || golden_root == nullptr) {
        std::printf("SKIP: set NINFER_TEST_INKLING2_ARTIFACT and NINFER_INKLING2_GOLDEN\n");
        return 77;
    }
    const char* dump_root = std::getenv("NINFER_INKLING2_DUMP");
    const int decode_tokens = std::getenv("NINFER_INKLING2_DECODE") ? std::atoi(std::getenv("NINFER_INKLING2_DECODE")) : 32;
    // NINFER_INKLING2_PREFILL=N also times an N-token prefill in chunks of 256 (the golden prompts' tokens repeated).
    const int prefill_tokens = std::getenv("NINFER_INKLING2_PREFILL") ? std::atoi(std::getenv("NINFER_INKLING2_PREFILL")) : 0;
    try {
        ninfer::artifact::Reader reader(artifact_path);
        ninfer::DeviceContext device(0);
        const auto t0 = std::chrono::steady_clock::now();
        auto model    = ink::load_model(reader, device);
        std::printf("loaded in %.1f s\n", std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
        const auto& c = model->config();
        const auto vocab = static_cast<std::int32_t>(c.unpadded_vocab_size);
        cudaStream_t stream = nullptr;
        if (cudaStreamCreate(&stream) != cudaSuccess) { throw std::runtime_error("cannot create a stream"); }
        // NINFER_INKLING2_DENSE=head|all runs the main and eager programs over the 8-bit dense re-encoding of the head
        // or of every dense projection (the KLDs below are then those of the lossy form against llama.cpp's BF16).
        const std::string dense_env = std::getenv("NINFER_INKLING2_DENSE") ? std::getenv("NINFER_INKLING2_DENSE") : "";
        std::unique_ptr<ink::DenseWeightsQ8> dense_q8;
        if (dense_env == "all" || dense_env == "head") {
            dense_q8 = std::make_unique<ink::DenseWeightsQ8>(*model, dense_env == "all" ? ink::DenseForm::Q8All : ink::DenseForm::Q8Head, stream);
        }
        const ink::DenseWeightsQ8* dense = dense_q8.get();
        ink::ProgramLimits main_limits{4096, 64};
        main_limits.dense = dense;
        ink::TextProgram program(*model, main_limits, stream);
        if (dense != nullptr) { std::printf("dense projections: 8-bit re-encoding (%s)\n", dense_env.c_str()); }

        // NINFER_INKLING2_DENSE_KLD=<file of token ids>: the 8-bit dense forms against the BF16 one over that text,
        // teacher-forced in chunks of 256: KLD(BF16 || Q8) per position and the argmax agreement.
        if (const char* kld_ids = std::getenv("NINFER_INKLING2_DENSE_KLD"); kld_ids != nullptr) {
            std::ifstream in(kld_ids);
            std::vector<std::int32_t> text_ids;
            for (std::int32_t id; in >> id;) { text_ids.push_back(id); }
            constexpr std::int32_t kChunk = 256;
            const auto n = static_cast<std::int32_t>(text_ids.size());
            ink::ProgramLimits limits{n + kChunk, kChunk};
            ink::TextProgram exact(*model, limits, stream);
            std::unique_ptr<ink::DenseWeightsQ8> all;
            if (dense == nullptr) { all = std::make_unique<ink::DenseWeightsQ8>(*model, ink::DenseForm::Q8All, stream); }
            limits.dense = dense != nullptr ? dense : all.get();
            ink::TextProgram eight(*model, limits, stream);
            std::vector<double> klds;
            int agree = 0;
            for (std::int32_t s = 0; s < n; s += kChunk) {
                const std::int32_t len = std::min(kChunk, n - s);
                const auto ids = std::span(text_ids).subspan(static_cast<std::size_t>(s), static_cast<std::size_t>(len));
                const auto a = logit_columns(exact.forward(ids), vocab);
                const auto b = logit_columns(eight.forward(ids), vocab);
                for (std::int32_t t = 0; t < len; ++t) {
                    klds.push_back(kld(a[t].data(), b[t].data(), vocab));
                    agree += argmax(a[t].data(), vocab) == argmax(b[t].data(), vocab);
                }
            }
            std::vector<double> sorted = klds;
            std::sort(sorted.begin(), sorted.end());
            double mean = 0;
            for (double k : klds) { mean += k; }
            mean /= static_cast<double>(klds.size());
            std::printf("dense Q8 (%s) vs BF16 over %d tokens: KLD mean %.3e median %.3e p99 %.3e max %.3e; argmax agrees %d/%d (%.2f%%)\n",
                        limits.dense->form() == ink::DenseForm::Q8All ? "all" : "head", n, mean, sorted[sorted.size() / 2], sorted[sorted.size() * 99 / 100], sorted.back(), agree, n, 100.0 * agree / n);
        }

        // Measurements over a text of token ids (each runs only that, not the golden comparison):
        // NINFER_INKLING2_ATTENTION_KLD=<ids>: the split decode attention against the single block, teacher-forced
        //   through 8-token calls (the split form only differs in calls of at most 8 tokens) over [0, 2048),
        //   [16384, 16896) and [29696, 30208), 256 at a time (not compared) elsewhere; KLD and argmax agreement per
        //   context range.
        // NINFER_INKLING2_DECODE_BENCH=<ids>: prefill (chunks of 256) and decode (64 single-token graph steps after
        //   one untimed step) speed at contexts 1024, 16384 and the whole text less 65 (NINFER_INKLING2_BENCH_CONTEXT:
        //   only that one); NINFER_INKLING2_SINGLE_BLOCK=1 measures the single-block attention.
        const auto read_ids = [](const char* path) {
            std::ifstream in(path);
            std::vector<std::int32_t> ids;
            for (std::int32_t id; in >> id;) { ids.push_back(id); }
            return ids;
        };
        // KLD and argmax per range [0, 1024), [1024, 4096), [4096, 16384), [16384, n) of the compared positions.
        const auto report_ranges = [](const char* label, const std::vector<double>& klds, const std::vector<int>& agrees,
                                      const std::vector<char>& compared) {
            const auto n = static_cast<std::int32_t>(klds.size());
            const std::int32_t edges[] = {0, 1024, 4096, 16384, n};
            for (std::size_t e = 0; e + 1 < std::size(edges) && edges[e] < n; ++e) {
                std::vector<double> part;
                int agree = 0;
                double mean = 0;
                for (std::int32_t i = edges[e]; i < std::min(edges[e + 1], n); ++i) {
                    if (compared[static_cast<std::size_t>(i)] == 0) { continue; }
                    part.push_back(klds[static_cast<std::size_t>(i)]);
                    mean += part.back(), agree += agrees[static_cast<std::size_t>(i)];
                }
                if (part.empty()) { continue; }
                std::sort(part.begin(), part.end());
                const auto count = static_cast<int>(part.size());
                std::printf("%s, positions [%d, %d): KLD mean %.3e median %.3e p99 %.3e max %.3e; argmax agrees %d/%d (%.2f%%)\n", label,
                            edges[e], std::min(edges[e + 1], n), mean / count, part[part.size() / 2], part[part.size() * 99 / 100],
                            part.back(), agree, count, 100.0 * agree / count);
            }
        };
        if (const char* kld_ids = std::getenv("NINFER_INKLING2_ATTENTION_KLD"); kld_ids != nullptr) {
            const auto text_ids = read_ids(kld_ids);
            constexpr std::int32_t kChunk = 256;
            const auto n = static_cast<std::int32_t>(text_ids.size());
            ink::ProgramLimits limits{n + kChunk, kChunk};
            limits.decode_graph = false;
            limits.split_attention = false;
            ink::TextProgram single(*model, limits, stream);
            limits.split_attention = true;
            ink::TextProgram split(*model, limits, stream);
            const auto narrow = [](std::int32_t at) { return at < 2048 || (at >= 16384 && at < 16896) || (at >= 29696 && at < 30208); };
            std::vector<double> klds;
            std::vector<int> agrees;
            std::vector<char> compared;
            for (std::int32_t s = 0; s < n;) {
                const std::int32_t len = std::min(narrow(s) ? 8 : kChunk, n - s);
                const auto ids = std::span(text_ids).subspan(static_cast<std::size_t>(s), static_cast<std::size_t>(len));
                const auto a = logit_columns(single.forward(ids), vocab);
                const auto b = logit_columns(split.forward(ids), vocab);
                for (std::int32_t t = 0; t < len; ++t) {
                    klds.push_back(kld(a[t].data(), b[t].data(), vocab));
                    agrees.push_back(argmax(a[t].data(), vocab) == argmax(b[t].data(), vocab) ? 1 : 0);
                    compared.push_back(narrow(s) ? 1 : 0);
                }
                s += len;
            }
            report_ranges("split attention vs single block", klds, agrees, compared);
            return 0;
        }
        if (const char* bench_ids = std::getenv("NINFER_INKLING2_DECODE_BENCH"); bench_ids != nullptr) {
            const auto text_ids = read_ids(bench_ids);
            constexpr std::int32_t kChunk = 256, kSteps = 64;
            const auto n = static_cast<std::int32_t>(text_ids.size());
            ink::ProgramLimits limits{n + kChunk, kChunk};
            limits.split_attention = std::getenv("NINFER_INKLING2_SINGLE_BLOCK") == nullptr;
            const char* form = limits.split_attention ? "split attention" : "single-block attention";
            ink::TextProgram bench(*model, limits, stream);
            const std::int32_t only = std::getenv("NINFER_INKLING2_BENCH_CONTEXT") ? std::atoi(std::getenv("NINFER_INKLING2_BENCH_CONTEXT")) : 0;
            for (const std::int32_t context : {1024, 16384, n - kSteps - 1}) {
                if (context > n - kSteps - 1 || (only > 0 && context != only)) { continue; }
                bench.reset();
                cudaStreamSynchronize(stream);
                const auto p0 = std::chrono::steady_clock::now();
                for (std::int32_t s = 0; s < context; s += kChunk) {
                    const std::int32_t len = std::min(kChunk, context - s);
                    bench.forward(std::span(text_ids).subspan(static_cast<std::size_t>(s), static_cast<std::size_t>(len)));
                }
                cudaStreamSynchronize(stream);
                const auto p1 = std::chrono::steady_clock::now();
                // One untimed step first: the decode graph is captured on the first single-token forward.
                bench.forward(std::span(text_ids).subspan(static_cast<std::size_t>(context), 1));
                cudaStreamSynchronize(stream);
                const auto p1d = std::chrono::steady_clock::now();
                for (std::int32_t i = 1; i <= kSteps; ++i) { bench.forward(std::span(text_ids).subspan(static_cast<std::size_t>(context + i), 1)); }
                cudaStreamSynchronize(stream);
                const auto p2 = std::chrono::steady_clock::now();
                std::printf("%s, context %d: prefill %.1f tok/s, decode %.2f tok/s\n", form, context,
                            context / std::chrono::duration<double>(p1 - p0).count(), kSteps / std::chrono::duration<double>(p2 - p1d).count());
            }
            return 0;
        }

        std::vector<std::int32_t> first_prompt;
        for (int index = 0;; ++index) {
            const fs::path dir = fs::path(golden_root) / ("p" + std::to_string(index));
            if (!fs::exists(dir / "tokens.json")) { break; }
            std::ifstream meta_file(dir / "tokens.json");
            std::stringstream text;
            text << meta_file.rdbuf();
            const auto meta = ninfer::artifact::Json::parse(text.str());
            const auto tokens = meta.at("tokens").get<std::vector<std::int32_t>>();
            const auto n = static_cast<std::int32_t>(tokens.size());
            if (index == 0) { first_prompt = tokens; }

            // Whole prompt with traces.
            std::map<std::string, double> worst; // per trace kind
            std::map<std::string, std::string> worst_at;
            double max_rel = 0;
            const fs::path dump = dump_root != nullptr ? fs::path(dump_root) / dir.filename() : fs::path();
            if (!dump.empty()) {
                fs::create_directories(dump);
                std::ofstream(dump / "tokens.json") << text.str();
            }
            ink::TextProgram::Trace trace = [&](const std::string& name, const ninfer::Tensor& value) {
                if (value.dtype == ninfer::DType::I32) {
                    if (!dump.empty()) {
                        std::vector<std::int32_t> ids(static_cast<std::size_t>(value.numel()));
                        cudaMemcpy(ids.data(), value.data, ids.size() * 4, cudaMemcpyDeviceToHost);
                        write_file(dump / (name + ".i32"), ids.data(), ids.size() * 4);
                    }
                    return;
                }
                if (!dump.empty()) {
                    const auto ours = to_host(value);
                    write_file(dump / (name + ".f32"), ours.data(), ours.size() * 4);
                }
                const fs::path file = dir / (name + ".f32");
                if (!fs::exists(file)) { return; } // llama.cpp names a block-starting visit's attention output ffn_inp
                const auto theirs = read_floats(file);
                const auto ours   = to_host(value);
                check(theirs.size() == ours.size(), name + ": golden size differs");
                if (theirs.size() != ours.size()) { return; }
                const double e   = rel_error(ours, theirs);
                const auto kind  = name.substr(0, name.find('-'));
                if (e > worst[kind]) {
                    worst[kind]    = e;
                    worst_at[kind] = name;
                }
                max_rel = std::max(max_rel, e);
            };
            program.reset();
            const auto whole = logit_columns(program.forward(tokens, &trace), vocab);
            if (!dump.empty()) {
                std::vector<float> flat;
                for (const auto& column : whole) { flat.insert(flat.end(), column.begin(), column.end()); }
                write_file(dump / "logits.f32", flat.data(), flat.size() * 4);
            }
            const auto golden = read_floats(dir / "logits.f32");
            check(golden.size() == static_cast<std::size_t>(n) * vocab, "golden logits size");
            double max_kld = 0, sum_kld = 0;
            int agree = 0;
            std::string argmaxes;
            for (std::int32_t t = 0; t < n; ++t) {
                const float* g = golden.data() + static_cast<std::size_t>(t) * vocab;
                const double k = kld(g, whole[t].data(), vocab);
                max_kld = std::max(max_kld, k);
                sum_kld += k;
                const auto a = argmax(whole[t].data(), vocab), b = argmax(g, vocab);
                agree += a == b;
                argmaxes += (t ? " " : "") + std::to_string(a) + (a == b ? "" : "!=" + std::to_string(b));
            }
            std::printf("p%d (%d tokens): max per-visit rel error %.3e", index, n, max_rel);
            for (const auto& [kind, e] : worst) { std::printf("; %s %.3e (%s)", kind.c_str(), e, worst_at[kind].c_str()); }
            std::printf("\n    KLD mean %.2e max %.2e; argmax %d/%d [%s]\n", sum_kld / n, max_kld, agree, n, argmaxes.c_str());
            check(agree == n, "p" + std::to_string(index) + ": argmax differs from llama.cpp");
            check(max_kld <= 0.05, "p" + std::to_string(index) + ": KLD above 0.05");
            check(sum_kld / n <= 0.01, "p" + std::to_string(index) + ": mean KLD above 0.01");

            // The same prompt in chunks of 3.
            program.reset();
            double chunk_diff = 0;
            for (std::int32_t s = 0; s < n; s += 3) {
                const std::int32_t len = std::min(3, n - s);
                const auto part = logit_columns(program.forward(std::span(tokens).subspan(s, len)), vocab);
                for (std::int32_t t = 0; t < len; ++t) {
                    for (std::int32_t i = 0; i < vocab; ++i) {
                        chunk_diff = std::max(chunk_diff, double(std::fabs(part[t][i] - whole[s + t][i])));
                    }
                }
            }
            std::printf("    chunks of 3 vs whole: max |logit difference| %.2e\n", chunk_diff);
            check(chunk_diff <= 1e-3, "p" + std::to_string(index) + ": chunked logits differ from the whole run");
        }
        check(!first_prompt.empty(), "no golden prompts found");

        if (prefill_tokens > 0 && !first_prompt.empty()) {
            constexpr std::int32_t kChunk = 256;
            ink::TextProgram long_program(*model, {prefill_tokens + kChunk, kChunk}, stream);
            std::vector<std::int32_t> tokens;
            while (static_cast<int>(tokens.size()) < prefill_tokens) {
                tokens.insert(tokens.end(), first_prompt.begin(), first_prompt.end());
            }
            tokens.resize(static_cast<std::size_t>(prefill_tokens));
            for (int pass = 0; pass < 2; ++pass) { // the first pass warms the kernels
                long_program.reset();
                cudaStreamSynchronize(stream);
                const auto p0 = std::chrono::steady_clock::now();
                for (int s = 0; s < prefill_tokens; s += kChunk) {
                    const int len = std::min<int>(kChunk, prefill_tokens - s);
                    (void)long_program.forward(std::span(tokens).subspan(static_cast<std::size_t>(s), static_cast<std::size_t>(len)));
                }
                cudaStreamSynchronize(stream);
                const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - p0).count();
                if (pass == 1) {
                    std::printf("prefill: %d tokens in chunks of %d in %.2f s = %.1f tok/s\n", prefill_tokens, kChunk, seconds,
                                prefill_tokens / seconds);
                }
            }
        }

        // Greedy continuation of the first prompt.
        if (!first_prompt.empty() && decode_tokens > 0) {
            program.reset();
            auto logits = logit_columns(program.forward(first_prompt), vocab);
            std::int32_t next = static_cast<std::int32_t>(argmax(logits.back().data(), vocab));
            std::string ids = std::to_string(next);
            std::vector<float> decoded; // every step's logits, for NINFER_INKLING2_DUMP (bit-exact comparisons of decode)
            cudaDeviceSynchronize();
            const auto d0 = std::chrono::steady_clock::now();
            for (int i = 1; i < decode_tokens; ++i) {
                const std::int32_t one[] = {next};
                logits = logit_columns(program.forward(one), vocab);
                next   = static_cast<std::int32_t>(argmax(logits.back().data(), vocab));
                ids += " " + std::to_string(next);
                if (dump_root != nullptr) { decoded.insert(decoded.end(), logits.back().begin(), logits.back().end()); }
            }
            const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - d0).count();
            if (dump_root != nullptr) { write_file(fs::path(dump_root) / "decode_logits.f32", decoded.data(), decoded.size() * 4); }
            std::printf("greedy continuation of p0: %s\n", ids.c_str());
            std::printf("decode: %d steps in %.2f s = %.2f tok/s (FP32 correctness kernels, logits copied to the host each step)\n",
                        decode_tokens - 1, seconds, (decode_tokens - 1) / seconds);

            // The decode graph against eager single-token calls: the same tokens give the same bits.
            std::vector<std::int32_t> steps;
            {
                std::istringstream in(ids);
                for (std::int32_t id; in >> id;) { steps.push_back(id); }
            }
            const std::size_t compare_steps = std::min<std::size_t>(steps.size(), 8);
            auto replay = [&](ink::TextProgram& p) {
                std::vector<std::vector<float>> out;
                p.reset();
                (void)p.forward(first_prompt);
                for (std::size_t i = 0; i < compare_steps; ++i) {
                    const std::int32_t one[] = {steps[i]};
                    out.push_back(logit_columns(p.forward(one), vocab).back());
                }
                return out;
            };
            const auto graphed = replay(program);
            ink::ProgramLimits eager_limits{4096, 64};
            eager_limits.dense        = dense;
            eager_limits.decode_graph = false;
            ink::TextProgram eager(*model, eager_limits, stream);
            const auto plain = replay(eager);
            float worst = 0.0F;
            for (std::size_t i = 0; i < compare_steps; ++i) {
                for (std::int32_t j = 0; j < vocab; ++j) { worst = std::max(worst, std::fabs(graphed[i][j] - plain[i][j])); }
            }
            std::printf("decode graph vs eager over %zu steps: max |logit difference| %.2e\n", compare_steps, worst);
            if (worst != 0.0F) {
                std::fprintf(stderr, "FAIL: the decode graph differs from eager calls\n");
                ++failures;
            }
        }
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL: %s\n", error.what());
        return 1;
    }
    if (failures) {
        std::fprintf(stderr, "%d failure(s)\n", failures);
        return 1;
    }
    std::printf("PASS\n");
    return 0;
}
