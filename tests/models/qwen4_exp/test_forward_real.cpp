// Adapted from Infernix a3edb450 tests/models/qwen4_exp/test_forward_real.cpp (Apache-2.0; the
// --dump-logits and --blocks taps of 45f20fb9).
// Modified for NInfer-3090: one sequence per harness (NInfer's single-sequence QSA Ops); NInfer's
// paged INT8 KV built directly; every expert in the pinned host bank, served by the GPU narrow route
// through staging slots (or zero-copy); no CPU-served columns; a timed decode mode (--decode).
//
// Qwen4Exp forward on the real converted artifact. Skips (77) unless NINFER_TEST_QWEN4EXP_ARTIFACT
// names a Qwen4Exp artifact; the n-gram volume is NINFER_TEST_QWEN4EXP_NGRAM or <artifact>.ngram.
//
//   ninfer_qwen4_exp_forward_real_test TOKENS [--logits OUT.bin] [--residuals OUT.bin]
//       [--routes OUT.bin] [--blocks OUT.bin] [--decode N] [--staging SLOTS]
//   ninfer_qwen4_exp_forward_real_test TOKENS --dump-logits OUT.bin [--chunk N] [TOKENS ...]...
//
// TOKENS is a comma-separated id list, or @FILE holding ids separated by commas or whitespace; the
// options after a TOKENS apply to it (--staging applies to the whole run).
//
// With --dump-logits the text is scored teacher-forced: it is prefilled in chunks of N (default 256)
// with FP32 logits at every position, written in Strata's --dump-logits layout (int32 vocabulary,
// int32 rows, then one FP32 row per position), and the perplexity is printed. Several texts are
// scored in order with one model load.
//
// Otherwise the test prefills every token as one chunk (the logits of the last position, and the
// optional taps: --residuals BF16 [layers][T][S*H], --routes I32 [layers][k][T], --blocks BF16
// [layers][mixer in, mixer out, MoE in, MoE out][T][H]); then, in a fresh state, it prefills all but
// the last N tokens (N = --decode, default 1) and decodes them one call per token, and checks that
// the last decode step's top token equals the single prefill's.

#include "artifact/reader.h"
#include "core/arena.h"
#include "core/device.h"
#include "core/paged_kv_cache.h"
#include "models/qwen4_exp/execution/forward.h"
#include "models/qwen4_exp/execution/parameters.h"
#include "models/qwen4_exp/frontend/ngram_hash.h"
#include "models/qwen4_exp/model.h"
#include "models/qwen4_exp/program/ngram_volume.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <memory>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <psapi.h>
#endif

using namespace ninfer;
namespace q4 = ninfer::models::qwen4_exp;
namespace ex = ninfer::models::qwen4_exp::execution;

namespace {

std::vector<std::int32_t> parse_tokens(const std::string& argument) {
    std::string text = argument;
    if (!text.empty() && text.front() == '@') {
        std::ifstream file(text.substr(1));
        if (!file) { throw std::invalid_argument("cannot read " + text.substr(1)); }
        text.assign(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
    }
    for (char& ch : text) {
        if (ch == ',' || ch == '\n' || ch == '\r' || ch == '\t') { ch = ' '; }
    }
    std::vector<std::int32_t> out;
    std::stringstream stream(text);
    std::int32_t id = 0;
    while (stream >> id) { out.push_back(id); }
    return out;
}

double gib(std::uint64_t bytes) { return static_cast<double>(bytes) / 1073741824.0; }

// Host memory after each stage: the pinned expert banks are the bulk.
void print_memory(const char* stage) {
#ifdef _WIN32
    PROCESS_MEMORY_COUNTERS_EX counters{};
    MEMORYSTATUSEX status{};
    status.dwLength = sizeof(status);
    if (K32GetProcessMemoryInfo(GetCurrentProcess(), reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&counters),
                                sizeof(counters)) &&
        GlobalMemoryStatusEx(&status)) {
        std::printf("[memory] %s: private %.2f GiB, working set %.2f GiB, system available %.2f GiB\n", stage,
                    gib(counters.PrivateUsage), gib(counters.WorkingSetSize), gib(status.ullAvailPhys));
        return;
    }
#endif
    std::printf("[memory] %s\n", stage);
}

double seconds_since(std::chrono::steady_clock::time_point t) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t).count();
}

// The one sequence's recurrent state, paged INT8 KV for `context` positions, and the expert
// sources: no resident experts, misses staged through `staging_slots` device slots.
struct Harness {
    Harness(const ex::Parameters& parameters, DeviceContext& device, std::int32_t context, std::int32_t columns,
            std::int32_t staging_slots)
        : config(parameters.model.config()) {
        const auto& c = config;
        const auto gdn_layers = static_cast<std::size_t>(c.gdn_layers);
        const std::size_t conv_bytes = std::size_t(c.gdn.conv_channels()) * (c.gdn.conv_kernel - 1) * 2;
        const std::size_t recurrent_bytes =
            std::size_t(c.gdn.key_head_dim) * c.gdn.value_head_dim * c.gdn.value_heads * 4;
        const std::size_t ple_bytes = std::size_t(c.residual_width()) * c.ple.conv_span() * 2;
        const std::int32_t di = static_cast<std::int32_t>(c.qsa.index_head_dim);
        const std::int32_t r  = static_cast<std::int32_t>(c.qsa.compress_ratio);
        const std::size_t tail_bytes = std::size_t(di) * (r - 1) * 2;
        state_backing = DeviceBuffer(gdn_layers * (conv_bytes + recurrent_bytes) + ple_bytes +
                                     c.attention_layers * tail_bytes + 4096);
        state_backing.fill(0);
        auto* p = static_cast<std::byte*>(state_backing.p);
        ex::ForwardState state;
        for (std::size_t l = 0; l < gdn_layers; ++l) {
            state.gdn_conv.emplace_back(p, DType::BF16,
                                        std::initializer_list<std::int32_t>{static_cast<std::int32_t>(c.gdn.conv_channels()),
                                                                            static_cast<std::int32_t>(c.gdn.conv_kernel - 1)});
            p += conv_bytes;
            state.gdn_recurrent.emplace_back(p, DType::FP32,
                                             std::initializer_list<std::int32_t>{static_cast<std::int32_t>(c.gdn.key_head_dim),
                                                                                 static_cast<std::int32_t>(c.gdn.value_head_dim),
                                                                                 static_cast<std::int32_t>(c.gdn.value_heads)});
            p += recurrent_bytes;
        }
        state.ple_conv = Tensor(p, DType::BF16, {static_cast<std::int32_t>(c.residual_width()),
                                                 static_cast<std::int32_t>(c.ple.conv_span()), 1});
        p += ple_bytes;
        for (std::uint32_t l = 0; l < c.attention_layers; ++l) {
            state.qsa_tails.emplace_back(p, DType::BF16, std::initializer_list<std::int32_t>{di, r - 1});
            p += tail_bytes;
        }

        // Int8Group64 K/V [256, 64, Hkv, pages] with FP16 scales [4, 64, Hkv, pages], and the pooled
        // index keys [Di / R, 64, 1, pages], through one identity block table.
        const std::int32_t pages = (context + kPagedKVPageSize - 1) / kPagedKVPageSize;
        const std::int32_t D = static_cast<std::int32_t>(c.attention.head_dim), Hkv = static_cast<std::int32_t>(c.attention.kv_heads);
        const std::size_t codes  = std::size_t(D) * kPagedKVPageSize * Hkv * pages;
        const std::size_t scales = std::size_t(D / 64) * kPagedKVPageSize * Hkv * pages * 2;
        const std::size_t pooled = std::size_t(di / r) * kPagedKVPageSize * pages * 2;
        const std::size_t layer_bytes = 2 * codes + 2 * scales + pooled;
        kv_backing = DeviceBuffer(c.attention_layers * layer_bytes + pages * sizeof(std::int32_t) + 4096);
        kv_backing.fill(0);
        std::vector<std::int32_t> table(static_cast<std::size_t>(pages));
        std::iota(table.begin(), table.end(), 0);
        auto* kvp = static_cast<std::byte*>(kv_backing.p);
        ex::ForwardKV kv;
        kv.block_table = Tensor(kvp, DType::I32, {pages});
        kv_backing.copy_from_host(table.data(), table.size() * sizeof(std::int32_t));
        kvp += (pages * sizeof(std::int32_t) + 255) / 256 * 256;
        kv.max_visible_keys = static_cast<std::uint32_t>(context);
        for (std::uint32_t l = 0; l < c.attention_layers; ++l) {
            PagedKVLayerView view;
            view.k_pages = Tensor(kvp, DType::I8, {D, kPagedKVPageSize, Hkv, pages});
            kvp += codes;
            view.v_pages = Tensor(kvp, DType::I8, {D, kPagedKVPageSize, Hkv, pages});
            kvp += codes;
            view.k_scale_pages = Tensor(kvp, DType::FP16, {D / 64, kPagedKVPageSize, Hkv, pages});
            kvp += scales;
            view.v_scale_pages = Tensor(kvp, DType::FP16, {D / 64, kPagedKVPageSize, Hkv, pages});
            kvp += scales;
            view.block_table  = kv.block_table;
            view.head_dim     = D;
            view.num_kv_heads = Hkv;
            view.storage      = KvCacheStorage::Int8Group64;
            kv.layers.push_back(view);
            kv.pooled.emplace_back(kvp, DType::BF16, std::initializer_list<std::int32_t>{di / r, kPagedKVPageSize, 1, pages});
            kvp += pooled;
        }

        frames_backing = DeviceBuffer(sizeof(std::int32_t) * c.moe.experts * c.num_hidden_layers);
        frames_backing.fill(0xFF); // -1: every expert served from the pinned host bank
        ex::ForwardExperts experts;
        for (std::uint32_t l = 0; l < c.num_hidden_layers; ++l) {
            experts.frames.push_back(static_cast<const std::int32_t*>(frames_backing.p) + l * c.moe.experts);
        }
        std::uint64_t stride = 0;
        for (const auto& bank : parameters.model.expert_banks()) { stride = std::max(stride, bank.record_stride); }
        if (staging_slots > 0) {
            staging = DeviceBuffer(static_cast<std::size_t>(staging_slots) * stride);
            experts.staging_base  = static_cast<std::uint8_t*>(staging.p);
            experts.staging_slots = staging_slots;
        }
        work = std::make_unique<WorkspaceArena>(
            ex::Forward::workspace_bytes(c, columns, static_cast<std::uint32_t>(context), device.execution_view()));
        forward = std::make_unique<ex::Forward>(parameters, device, *work, std::move(state), std::move(kv),
                                                std::move(experts));
    }

    const q4::TextConfig& config;
    DeviceBuffer state_backing, kv_backing, frames_backing, staging;
    std::unique_ptr<WorkspaceArena> work;
    std::unique_ptr<ex::Forward> forward;
};

// Device copies of one call's inputs: positions [first, first + width) of `tokens`.
struct Call {
    DeviceBuffer buffer;
    ex::ForwardBatch batch;
};

Call make_call(const q4::TextConfig& c, const q4::NgramVolume& volume, const std::vector<std::int32_t>& tokens,
               std::int32_t first, std::int32_t width, bool every_column) {
    const q4::NgramHash hash(c.ple.ngram, 0);
    const std::size_t row_bytes = c.ple.table.row_bytes, heads = hash.heads();
    std::vector<std::int32_t> padded(c.ple.ngram.ngram_size - 1, c.eos_token_id);
    padded.insert(padded.end(), tokens.begin(), tokens.begin() + first + width);
    std::vector<std::uint32_t> row_ids(static_cast<std::size_t>(width) * heads);
    hash.row_ids(padded, static_cast<std::size_t>(width), row_ids.data());
    std::vector<std::byte> ngram(row_ids.size() * row_bytes);
    volume.read_rows(row_ids, ngram);
    std::vector<std::int32_t> ids(tokens.begin() + first, tokens.begin() + first + width), positions(width);
    std::iota(positions.begin(), positions.end(), first);
    // Text tokens: the three RoPE axes equal the position (axis-major [T, 3]).
    std::vector<std::int32_t> rope(3 * static_cast<std::size_t>(width));
    for (int a = 0; a < 3; ++a) {
        for (std::int32_t t = 0; t < width; ++t) { rope[a * width + t] = first + t; }
    }
    const std::int32_t block = first / static_cast<std::int32_t>(c.qsa.compress_ratio) *
                               static_cast<std::int32_t>(c.qsa.compress_ratio);
    const std::vector<std::int32_t> block_rope{block, block, block};
    std::vector<std::int32_t> last;
    if (every_column) {
        last.resize(width);
        std::iota(last.begin(), last.end(), 0);
    } else {
        last.push_back(width - 1);
    }
    Call call;
    const auto padded_bytes = [](std::size_t n) { return (n + 255) / 256 * 256; };
    const std::size_t i32   = sizeof(std::int32_t);
    call.buffer = DeviceBuffer(padded_bytes(ids.size() * i32) * 2 + padded_bytes(rope.size() * i32) +
                               padded_bytes(3 * i32) + padded_bytes(last.size() * i32) + padded_bytes(ngram.size()));
    auto* base         = static_cast<std::byte*>(call.buffer.p);
    std::size_t offset = 0;
    auto put           = [&](const void* data, std::size_t n) {
        call.buffer.copy_from_host(data, n, offset);
        void* p = base + offset;
        offset += padded_bytes(n);
        return p;
    };
    call.batch.ids              = Tensor(put(ids.data(), ids.size() * i32), DType::I32, {width});
    call.batch.positions        = Tensor(put(positions.data(), positions.size() * i32), DType::I32, {width});
    call.batch.rope_positions   = Tensor(put(rope.data(), rope.size() * i32), DType::I32, {width, 3});
    call.batch.block_start_rope = Tensor(put(block_rope.data(), 3 * i32), DType::I32, {3});
    call.batch.logit_columns =
        Tensor(put(last.data(), last.size() * i32), DType::I32, {static_cast<std::int32_t>(last.size())});
    call.batch.ngram_rows = Tensor(put(ngram.data(), ngram.size()), DType::U8,
                                   {static_cast<std::int32_t>(row_bytes), static_cast<std::int32_t>(heads), width});
    return call;
}

std::vector<float> to_float(const DeviceBuffer& logits, std::size_t count) {
    std::vector<float> out(count);
    logits.copy_to_host(out.data(), count * sizeof(float));
    return out;
}

std::vector<int> top(const float* v, std::size_t n, int k) {
    std::vector<int> index(n);
    std::iota(index.begin(), index.end(), 0);
    std::partial_sort(index.begin(), index.begin() + k, index.end(), [&](int a, int b) { return v[a] > v[b]; });
    index.resize(k);
    return index;
}

double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v.empty() ? 0.0 : v[v.size() / 2];
}

} // namespace

int main(int argc, char** argv) {
    const char* artifact_env = std::getenv("NINFER_TEST_QWEN4EXP_ARTIFACT");
    if (artifact_env == nullptr || argc < 2) {
        std::printf("SKIP: set NINFER_TEST_QWEN4EXP_ARTIFACT and pass a token list\n");
        return 77;
    }
    struct Job {
        std::vector<std::int32_t> tokens;
        std::string logits_path, residuals_path, routes_path, blocks_path, dump_path;
        std::int32_t chunk  = 256;
        std::int32_t decode = 1;
    };
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    try {
        std::vector<Job> jobs;
        std::int32_t staging_slots = 64;
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg.rfind("--", 0) != 0) {
                jobs.push_back(Job{parse_tokens(arg)});
                if (jobs.back().tokens.size() < 2) { throw std::invalid_argument("pass at least two tokens"); }
                continue;
            }
            if (i + 1 >= argc) { throw std::invalid_argument(arg + " needs a value"); }
            const std::string value = argv[++i];
            if (arg == "--staging") {
                staging_slots = std::stoi(value);
                continue;
            }
            if (jobs.empty()) { throw std::invalid_argument(arg + " needs a token list before it"); }
            Job& job = jobs.back();
            if (arg == "--logits") {
                job.logits_path = value;
            } else if (arg == "--residuals") {
                job.residuals_path = value;
            } else if (arg == "--routes") {
                job.routes_path = value;
            } else if (arg == "--blocks") {
                job.blocks_path = value;
            } else if (arg == "--dump-logits") {
                job.dump_path = value;
            } else if (arg == "--chunk") {
                job.chunk = std::stoi(value);
                if (job.chunk <= 0) { throw std::invalid_argument("--chunk must be positive"); }
            } else if (arg == "--decode") {
                job.decode = std::stoi(value);
            } else {
                throw std::invalid_argument("unknown option " + arg);
            }
        }
        if (jobs.empty()) { throw std::invalid_argument("pass a token list"); }
        const bool scoring = !jobs.front().dump_path.empty();
        for (const Job& job : jobs) {
            if (job.dump_path.empty() == scoring || (jobs.size() > 1 && job.dump_path.empty())) {
                throw std::invalid_argument("several token lists are scored only, each with its own --dump-logits");
            }
        }
        print_memory("start");
        DeviceContext device(0);
        auto t0 = std::chrono::steady_clock::now();
        artifact::Reader reader(artifact_env);
        if (!q4::is_qwen4_exp(reader)) {
            std::printf("SKIP: %s is not a Qwen4Exp artifact\n", artifact_env);
            return 77;
        }
        const auto model = q4::load_model(reader, device);
        device.synchronize();
        const auto& stats = model->storage_stats();
        std::printf("loaded in %.1f s: %.2f GiB device, %.2f GiB pinned host\n", seconds_since(t0),
                    gib(stats.device_capacity_bytes), gib(stats.pinned_bytes));
        print_memory("materialized");
        const ex::Parameters parameters(*model);
        const auto& c         = model->config();
        const char* ngram_env = std::getenv("NINFER_TEST_QWEN4EXP_NGRAM");
        q4::NgramVolume volume(ngram_env != nullptr ? std::filesystem::path(ngram_env)
                                                    : q4::default_ngram_volume(artifact_env),
                               c.ple.table);
        // Strata's keep-alive defaults: a read after 100 ms of idleness, for a minute after a request.
        volume.set_keepalive(std::chrono::milliseconds(100), std::chrono::milliseconds(60000));
        const std::size_t vocab = c.vocab_size;
        const auto H            = static_cast<std::int32_t>(c.hidden_size);
        const auto blocks       = static_cast<std::size_t>(c.num_hidden_layers);
        std::printf("staging slots: %d (0: zero-copy)\n", staging_slots);
        if (scoring) {
            for (const Job& job : jobs) {
                const auto& tokens         = job.tokens;
                const auto n               = static_cast<std::int32_t>(tokens.size());
                const std::int32_t context = std::max<std::int32_t>(256, (n + 63) / 64 * 64);
                const std::int32_t chunk   = std::min(job.chunk, n);
                Harness harness(parameters, device, context, chunk, staging_slots);
                DeviceBuffer logits(vocab * chunk * sizeof(float));
                std::ofstream out(job.dump_path, std::ios::binary);
                const std::int32_t header[2] = {static_cast<std::int32_t>(vocab), n};
                out.write(reinterpret_cast<const char*>(header), sizeof(header));
                double nll = 0.0;
                std::int32_t scored = 0, same_top1 = 0;
                const auto volume_before = volume.counters();
                t0 = std::chrono::steady_clock::now();
                for (std::int32_t first = 0; first < n; first += chunk) {
                    const std::int32_t width = std::min(chunk, n - first);
                    auto call                = make_call(c, volume, tokens, first, width, true);
                    Tensor chunk_logits(logits.p, DType::FP32, {static_cast<std::int32_t>(vocab), width});
                    harness.forward->run(call.batch, chunk_logits);
                    device.synchronize();
                    const auto rows = to_float(logits, vocab * width);
                    out.write(reinterpret_cast<const char*>(rows.data()),
                              static_cast<std::streamsize>(rows.size() * sizeof(float)));
                    for (std::int32_t i = 0; i < width && first + i + 1 < n; ++i) {
                        const float* row = rows.data() + static_cast<std::size_t>(i) * vocab;
                        const float peak = *std::max_element(row, row + vocab);
                        double sum       = 0.0;
                        for (std::size_t v = 0; v < vocab; ++v) { sum += std::exp(double(row[v]) - peak); }
                        const std::int32_t next = tokens[first + i + 1];
                        nll -= double(row[next]) - peak - std::log(sum);
                        same_top1 += std::max_element(row, row + vocab) - row == next;
                        ++scored;
                    }
                }
                if (!out) { throw std::runtime_error("cannot write " + job.dump_path); }
                const double seconds = seconds_since(t0);
                const auto volume_after = volume.counters();
                std::printf("%s: scored %d positions in %.1f s (%.1f tok/s in chunks of %d): mean NLL %.5f nats, "
                            "perplexity %.4f, top-1 = next %.1f%%; n-gram rows %llu, cache hits %llu, block reads %llu "
                            "in %.2f s\n",
                            job.dump_path.c_str(), scored, seconds, n / seconds, chunk, nll / scored,
                            std::exp(nll / scored), 100.0 * same_top1 / scored,
                            static_cast<unsigned long long>(volume_after.rows - volume_before.rows),
                            static_cast<unsigned long long>(volume_after.hits - volume_before.hits),
                            static_cast<unsigned long long>(volume_after.reads - volume_before.reads),
                            (volume_after.read_ns - volume_before.read_ns) * 1e-9);
            }
            print_memory("scored");
            return 0;
        }

        const Job& job             = jobs.front();
        const auto& tokens         = job.tokens;
        const auto n               = static_cast<std::int32_t>(tokens.size());
        const std::int32_t context = std::max<std::int32_t>(256, (n + 63) / 64 * 64);
        if (job.decode < 1 || job.decode >= n) { throw std::invalid_argument("--decode must be in [1, tokens)"); }
        DeviceBuffer logits(vocab * sizeof(float));
        std::vector<float> prefill;
        {
            // 1. Every token as one chunk: logits of the last position and the taps.
            Harness harness(parameters, device, context, n, staging_slots);
            auto full = make_call(c, volume, tokens, 0, n, false);
            Tensor full_logits(logits.p, DType::FP32, {static_cast<std::int32_t>(vocab), 1});
            const auto W = static_cast<std::int32_t>(c.residual_width());
            const auto K = static_cast<std::int32_t>(c.moe.top_k);
            const std::size_t residual_bytes = static_cast<std::size_t>(W) * n * 2;
            const std::size_t route_bytes    = static_cast<std::size_t>(K) * n * 4;
            const std::size_t io_bytes       = static_cast<std::size_t>(H) * n * 2;
            const bool tapped = !job.residuals_path.empty() || !job.routes_path.empty() || !job.blocks_path.empty();
            DeviceBuffer taps(tapped ? blocks * (residual_bytes + route_bytes + 4 * io_bytes) : 256);
            std::vector<Tensor> residual_taps, route_taps, io_taps[4];
            auto* base                  = static_cast<std::byte*>(taps.p);
            const std::size_t io_offset = blocks * (residual_bytes + route_bytes);
            if (tapped) {
                for (std::size_t b = 0; b < blocks; ++b) {
                    residual_taps.emplace_back(base + b * residual_bytes, DType::BF16, std::initializer_list<std::int32_t>{W, n});
                    route_taps.emplace_back(base + blocks * residual_bytes + b * route_bytes, DType::I32,
                                            std::initializer_list<std::int32_t>{K, n});
                    for (std::size_t kind = 0; kind < 4; ++kind) {
                        io_taps[kind].emplace_back(base + io_offset + (b * 4 + kind) * io_bytes, DType::BF16,
                                                   std::initializer_list<std::int32_t>{H, n});
                    }
                }
            }
            ex::ForwardTap tap{&residual_taps, &route_taps, &io_taps[0], &io_taps[1], &io_taps[2], &io_taps[3]};
            t0 = std::chrono::steady_clock::now();
            harness.forward->run(full.batch, full_logits, tapped ? &tap : nullptr);
            device.synchronize();
            std::printf("prefill %d tokens as one chunk: %.1f ms\n", n, seconds_since(t0) * 1e3);
            const auto dump = [&](const std::string& path, std::size_t offset, std::size_t bytes) {
                if (path.empty()) { return; }
                std::vector<std::byte> host(bytes);
                taps.copy_to_host(host.data(), bytes, offset);
                std::ofstream file(path, std::ios::binary);
                file.write(reinterpret_cast<const char*>(host.data()), static_cast<std::streamsize>(bytes));
            };
            dump(job.residuals_path, 0, blocks * residual_bytes);
            dump(job.routes_path, blocks * residual_bytes, blocks * route_bytes);
            dump(job.blocks_path, io_offset, blocks * 4 * io_bytes);
            prefill = to_float(logits, vocab);
        }
        const auto best = top(prefill.data(), vocab, 5);
        std::printf("prefill top-5 at position %d:", n - 1);
        for (int id : best) { std::printf(" %d (%.3f)", id, prefill[id]); }
        std::printf("\n");
        if (!job.logits_path.empty()) {
            std::ofstream file(job.logits_path, std::ios::binary);
            file.write(reinterpret_cast<const char*>(prefill.data()), static_cast<std::streamsize>(vocab * sizeof(float)));
        }

        // 2. All but the last N tokens, then one decode call per token (timed).
        Harness harness(parameters, device, context, std::max<std::int32_t>(n - job.decode, 1), staging_slots);
        const std::int32_t head = n - job.decode;
        auto prompt             = make_call(c, volume, tokens, 0, head, false);
        Tensor step_logits(logits.p, DType::FP32, {static_cast<std::int32_t>(vocab), 1});
        harness.forward->run(prompt.batch, step_logits);
        device.synchronize();
        std::vector<double> step_ms;
        for (std::int32_t p = head; p < n; ++p) {
            auto step = make_call(c, volume, tokens, p, 1, false);
            t0        = std::chrono::steady_clock::now();
            harness.forward->run(step.batch, step_logits);
            device.synchronize();
            step_ms.push_back(seconds_since(t0) * 1e3);
        }
        const auto decode = to_float(logits, vocab);
        double max_diff   = 0;
        for (std::size_t i = 0; i < vocab; ++i) { max_diff = std::max(max_diff, std::abs(double(decode[i]) - prefill[i])); }
        const auto decode_best = top(decode.data(), vocab, 5);
        std::printf("decode %d steps: median %.1f ms per token (%.1f tok/s), first %.1f ms\n", job.decode,
                    median(step_ms), 1e3 / median(step_ms), step_ms.front());
        std::printf("decode top-5:");
        for (int id : decode_best) { std::printf(" %d (%.3f)", id, decode[id]); }
        std::printf("\nmax |decode - prefill| logit = %.4f\n", max_diff);
        print_memory("done");
        const bool ok = decode_best.front() == best.front();
        std::printf(ok ? "forward checks passed\n" : "FAIL: decode disagrees with prefill\n");
        return ok ? 0 : 1;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL: %s\n", error.what());
        return 1;
    }
}
