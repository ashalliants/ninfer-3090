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
//
// Rows mode (several sequences decoded together, Forward::run_rows):
//
//   ninfer_qwen4_exp_forward_real_test --rows STEPS [--time ROUNDS] [--frames-mib N] [--staging SLOTS]
//       PROMPT PROMPT [PROMPT ...]
//
// PROMPT is TOKENS or text:FILE (UTF-8 text, tokenized as is, no chat template); 1 to 8 prompts.
//  1. Each prompt alone through the single-sequence path (a Forward of its own): prefilled in calls
//     of 1024, then STEPS greedy decode calls. Its greedy ids and every decode call's logits are the
//     reference.
//  2. One Forward over one state slot per prompt: each prompt prefilled into its slot (select_row),
//     the state saved; then for B = 1 .. P the first B slots decode STEPS rounds together from the
//     saved state (run_rows, eager). Every row's greedy ids must equal its reference; the logits are
//     compared bit for bit too.
//  3. --time ROUNDS: every expert the B = P run routed in its first ROUNDS rounds is made resident
//     in device frames (most used first, as many as fit in the free VRAM or --frames-mib), then the
//     first ROUNDS rounds of each configuration are timed on the GPU (CUDA events around the replay
//     of a captured CUDA graph, the way the Program decodes): the single-sequence call on slot 0 (S)
//     against B rows, in S R_B R_B S order for every B, twice. Reports the median ms per round, the
//     distinct experts per layer and misses per round, and the cost per extra row.
//     --configs LIST (0 = the single-sequence call, B = B rows) instead runs the listed
//     configurations once each inside a cudaProfilerStart/Stop range, for an nsys kernel breakdown.
//
// KLD mode (the decode-path quality gate against a llama.cpp base; docs/maintainer/flash-next-plan.md):
//
//   ninfer_qwen4_exp_forward_real_test --kld-base FILE [--chunks N] [--window T] [--tsv OUT.tsv]
//       [--staging SLOTS] [--hc-int8]
//
// FILE is a `llama-perplexity --kl-divergence-base` file; its n_ctx-token chunks hold the tokens.
// Each chunk (the first N, default all) is a fresh sequence: positions [0, n_ctx/2) are prefilled as
// one call, then every target in [n_ctx/2 + 1, n_ctx) is predicted by a decode forward, T targets
// per call (default 3, at most 8): one column through the plain decode call, wider windows through
// the verification call the Program runs for an MTP round (T - 1 drafts) with every column committed
// by the GDN fold, qsa_commit_tails and ple_conv_commit; a partial last window is a narrower round.
// Each row's FP32 logits over the public token domain are scored against the base row (score_row);
// the rows go to OUT.tsv (default kld-rows.tsv) and the summary to stdout. Numerics options run
// under the gate as flags: --hc-int8 makes the mixers' int8 copies once (ex::HcInt8Copies) and every
// chunk's forward reads them (ForwardOptions::hc_int8) in its decode and verification calls; the
// prompt half (more than 16 columns) reads the BF16 weights.
//
// --hc-int8 prefill mode (ForwardOptions::hc_int8 leaves calls of more than 16 columns bit-exact):
//
//   ninfer_qwen4_exp_forward_real_test --hc-int8-prefill TOKENS [--staging SLOTS]
//
// From a fresh state, twice (off, then with the mixers' int8 copies): all but the last token are
// prefilled as one call (at least 17 tokens, so every mixer call is wider than the int8 route),
// then the last is decoded by one call. The prompt call's logits must be bitwise equal; the decode
// call's must differ (it reads the copies, a check that the option is in effect).
//
// Shared-expert-stream mode (ForwardOptions::shared_expert_stream is bit-exact):
//
//   ninfer_qwen4_exp_forward_real_test --shared-expert-stream TOKENS [--decode N] [--staging SLOTS]
//
// All but the last N tokens (default 4; more than 8 must remain, so the prompt call is wider than the
// GGML decode route and its Linears use the side stream's carve) are prefilled as one call, then each
// of the N is decoded by one call, from a fresh state, four times: the shared expert after the routed
// experts or beside them on its side stream, with the decode calls eager or each captured in a CUDA
// graph and replayed. The
// prompt call's last logits and every decode call's must be bitwise equal between the serial and the
// side-stream run of the same decode mode.
//
// GDN head-map mode (ForwardOptions::gdn_head_map is bit-exact):
//
//   ninfer_qwen4_exp_forward_real_test --gdn-head-map TOKENS [--window W] [--decode N] [--staging SLOTS]
//
// From a fresh state, twice (q/k tiled by copies, then read in place through the tiled head map):
// all but the last W + N tokens are prefilled as one call (at least 16 tokens, the GDN's chunked
// route), the next W (default 3, 2..8) are one verification call (the recorded recurrence, every
// column's logits) committed in full by the GDN fold, then each of the last N (default 2) is decoded
// by one call (the recurrent kernel). Every logits column must be bitwise equal between the runs.
//
// MoE precast mode (ForwardOptions::moe_precast_input is bit-exact):
//
//   ninfer_qwen4_exp_forward_real_test --moe-precast-input TOKENS [--window W] [--decode N] [--staging SLOTS]
//
// The same calls as the head-map mode (a prompt call wider than 8 columns, whose experts take the
// narrow and the wide route; a verification window; decode calls), each routed-expert call casting
// its input to A8 per block, then once per call. Every logits column must be bitwise equal.
//
// MoE register gate/up mode (ForwardOptions::moe_gate_up_registers is bit-exact):
//
//   ninfer_qwen4_exp_forward_real_test --moe-gate-up-registers TOKENS [--window W] [--decode N] [--staging SLOTS]
//
// The same calls, first with the default gate/up kernel (per-block casts, rows staged in shared
// memory), then with the precast input and the register gate/up kernel. Every logits column must be
// bitwise equal.
//
// MoE one-launch route and dispatch mode (ForwardOptions::moe_fused_route is bit-exact):
//
//   ninfer_qwen4_exp_forward_real_test --moe-fused-route TOKENS [--window W] [--decode N] [--staging SLOTS]
//
// The same calls, first with moe_route then moe_dispatch, then with moe_route_dispatch (one launch
// for the verification and decode calls; the prompt call, wider than 8 columns, takes the two Ops
// either way). Every logits column must be bitwise equal.

#include "artifact/reader.h"
#include "core/arena.h"
#include "core/decode_graph.h"
#include "core/device.h"
#include "core/gdn_replay_records.h"
#include "core/layout.h"
#include "core/linear_attention_state.h"
#include "core/paged_kv_cache.h"
#include "models/qwen4_exp/execution/forward.h"
#include "models/qwen4_exp/execution/parameters.h"
#include "models/qwen4_exp/frontend/ngram_hash.h"
#include "models/qwen4_exp/model.h"
#include "models/qwen4_exp/program/ngram_volume.h"
#include "models/qwen4_exp/program/program.h"
#include "models/qwen3_5/frontend/tokenizer.h"
#include "ninfer/ops/gdn_replay.h"
#include "ninfer/ops/ple.h"
#include "ninfer/ops/qsa.h"

#include "evaluation.h"

#include <cuda_profiler_api.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iterator>
#include <memory>
#include <numeric>
#include <optional>
#include <span>
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
namespace ppl = ninfer::perplexity;

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
            std::int32_t staging_slots, ex::ForwardOptions options = {})
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
        state.ple_slot = Tensor(p, DType::I32, {1}); // in the zeroed slack

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
            ex::Forward::workspace_bytes(c, columns, static_cast<std::uint32_t>(context), device.execution_view(),
                                         options.moe_precast_input) +
            (options.shared_expert_stream ? ex::Forward::shared_expert_workspace_bytes(c, columns) : 0));
        views   = state;
        forward = std::make_unique<ex::Forward>(parameters, device, *work, std::move(state), std::move(kv),
                                                std::move(experts), std::nullopt, options);
    }

    const q4::TextConfig& config;
    ex::ForwardState views; // the state's tensors (the Forward holds the same views)
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

// ------------------------------------------------------------------------------- rows mode

constexpr std::int32_t kPrefillChunk = 1024;

std::int32_t argmax(const float* v, std::size_t n) {
    return static_cast<std::int32_t>(std::max_element(v, v + n) - v);
}

// A prompt argument: TOKENS as for the other modes, or text:FILE (UTF-8, tokenized as is).
std::vector<std::int32_t> parse_prompt(const std::string& argument, const q4::FrontendResources& resources) {
    if (argument.rfind("text:", 0) != 0) { return parse_tokens(argument); }
    std::ifstream file(argument.substr(5), std::ios::binary);
    if (!file || resources.tokenizer == nullptr) { throw std::invalid_argument("cannot tokenize " + argument); }
    const std::string text((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    const std::vector<int> ids = resources.tokenizer->encode(text);
    return {ids.begin(), ids.end()};
}

// The n-gram rows of the last position of `history` (column-major [row bytes, heads]).
std::vector<std::byte> last_rows(const q4::TextConfig& c, const q4::NgramVolume& volume,
                                 const std::vector<std::int32_t>& history) {
    const q4::NgramHash hash(c.ple.ngram, 0);
    std::vector<std::int32_t> padded(c.ple.ngram.ngram_size - 1, c.eos_token_id);
    padded.insert(padded.end(), history.begin(), history.end());
    std::vector<std::uint32_t> row_ids(hash.heads());
    hash.row_ids(padded, 1, row_ids.data());
    std::vector<std::byte> rows(row_ids.size() * c.ple.table.row_bytes);
    volume.read_rows(row_ids, rows);
    return rows;
}

// `slots` sequences in one Forward: slotted recurrent state, one KV page pool with a block table per
// slot (slot s owns pages [s P, s P + P)), and the expert sources (no resident experts until
// make_resident). The state and the KV can be saved once and restored before every run.
struct RowsHarness {
    RowsHarness(const ex::Parameters& parameters, DeviceContext& device, std::int32_t slots, std::int32_t context,
                std::int32_t staging_slots, std::uint64_t frame_budget_bytes)
        : config(parameters.model.config()), parameters(parameters), device(device), slots(slots) {
        const auto& c   = config;
        const auto dims = [](auto v) { return static_cast<std::int32_t>(v); };
        const std::int32_t C = dims(c.gdn.conv_channels()), K = dims(c.gdn.conv_kernel - 1);
        const std::int32_t Dk = dims(c.gdn.key_head_dim), Dv = dims(c.gdn.value_head_dim), Hv = dims(c.gdn.value_heads);
        const std::int32_t di = dims(c.qsa.index_head_dim), r = dims(c.qsa.compress_ratio);
        const std::size_t n         = static_cast<std::size_t>(slots);
        const std::size_t conv      = std::size_t(C) * K * 2 * n, recurrent = std::size_t(Dk) * Dv * Hv * 4 * n;
        const std::size_t ple_bytes = std::size_t(c.residual_width()) * c.ple.conv_span() * 2 * n;
        const std::size_t tail      = std::size_t(di) * (r - 1) * 2 * n;
        state_backing = DeviceBuffer(c.gdn_layers * (conv + recurrent) + ple_bytes + c.attention_layers * tail + 4096);
        state_backing.fill(0);
        auto* p = static_cast<std::byte*>(state_backing.p);
        ex::ForwardRowsState state;
        for (std::uint32_t l = 0; l < c.gdn_layers; ++l) {
            state.gdn_conv.emplace_back(p, DType::BF16, std::initializer_list<std::int32_t>{C, K, slots});
            p += conv;
            state.gdn_recurrent.emplace_back(p, DType::FP32, std::initializer_list<std::int32_t>{Dk, Dv, Hv, slots});
            p += recurrent;
        }
        state.ple_conv = Tensor(p, DType::BF16, {dims(c.residual_width()), dims(c.ple.conv_span()), slots});
        p += ple_bytes;
        for (std::uint32_t l = 0; l < c.attention_layers; ++l) {
            state.qsa_tails.emplace_back(p, DType::BF16, std::initializer_list<std::int32_t>{di, r - 1, slots});
            p += tail;
        }
        std::vector<std::int32_t> ids(n);
        std::iota(ids.begin(), ids.end(), 0);
        state.slot_ids = Tensor(p, DType::I32, {slots});
        state_backing.copy_from_host(ids.data(), n * 4, static_cast<std::size_t>(p - static_cast<std::byte*>(state_backing.p)));

        const std::int32_t pages = (context + kPagedKVPageSize - 1) / kPagedKVPageSize, pool = pages * slots;
        const std::int32_t D = dims(c.attention.head_dim), Hkv = dims(c.attention.kv_heads);
        const std::size_t codes  = std::size_t(D) * kPagedKVPageSize * Hkv * pool;
        const std::size_t scales = std::size_t(D / 64) * kPagedKVPageSize * Hkv * pool * 2;
        const std::size_t pooled = std::size_t(di / r) * kPagedKVPageSize * pool * 2;
        const std::size_t table_bytes = (std::size_t(pool) * 4 + 255) / 256 * 256;
        kv_backing = DeviceBuffer(table_bytes + c.attention_layers * (2 * codes + 2 * scales + pooled) + 4096);
        kv_backing.fill(0);
        std::vector<std::int32_t> tables(static_cast<std::size_t>(pool));
        std::iota(tables.begin(), tables.end(), 0); // slot s, page i: s * pages + i
        kv_backing.copy_from_host(tables.data(), tables.size() * 4);
        auto* kvp = static_cast<std::byte*>(kv_backing.p);
        ex::ForwardRowsKV kv;
        kv.block_tables     = Tensor(kvp, DType::I32, {pages, slots});
        kv.max_visible_keys = static_cast<std::uint32_t>(context);
        kvp += table_bytes;
        for (std::uint32_t l = 0; l < c.attention_layers; ++l) {
            PagedKVLayerView view;
            view.k_pages = Tensor(kvp, DType::I8, {D, kPagedKVPageSize, Hkv, pool});
            kvp += codes;
            view.v_pages = Tensor(kvp, DType::I8, {D, kPagedKVPageSize, Hkv, pool});
            kvp += codes;
            view.k_scale_pages = Tensor(kvp, DType::FP16, {D / 64, kPagedKVPageSize, Hkv, pool});
            kvp += scales;
            view.v_scale_pages = Tensor(kvp, DType::FP16, {D / 64, kPagedKVPageSize, Hkv, pool});
            kvp += scales;
            view.head_dim     = D;
            view.num_kv_heads = Hkv;
            view.storage      = KvCacheStorage::Int8Group64;
            kv.layers.push_back(view);
            kv.pooled.emplace_back(kvp, DType::BF16, std::initializer_list<std::int32_t>{di / r, kPagedKVPageSize, 1, pool});
            kvp += pooled;
        }

        // Expert sources: frame tables (all -1 until make_resident), the frame pool, staging, the
        // route log (every call's routed ids per layer).
        const std::size_t E = c.moe.experts, L = c.num_hidden_layers;
        tables_host.assign(L * E, -1);
        frames_backing = DeviceBuffer(L * E * 4);
        frames_backing.fill(0xFF);
        for (const auto& bank : parameters.model.expert_banks()) { frame_stride = std::max(frame_stride, bank.record_stride); }
        staging = DeviceBuffer(static_cast<std::size_t>(staging_slots) * frame_stride);
        route_stride = c.moe.top_k * static_cast<std::size_t>(kPrefillChunk);
        route_log    = DeviceBuffer(L * route_stride * 4);
        frame_capacity = static_cast<std::int32_t>(frame_budget_bytes / frame_stride);
        if (frame_capacity > 0) { frame_pool = DeviceBuffer(static_cast<std::size_t>(frame_capacity) * frame_stride); }
        ex::ForwardExperts experts;
        for (std::size_t l = 0; l < L; ++l) {
            experts.frames.push_back(static_cast<const std::int32_t*>(frames_backing.p) + l * E);
        }
        experts.frame_base    = static_cast<const std::uint8_t*>(frame_pool.p);
        experts.frame_stride  = frame_stride;
        experts.staging_base  = static_cast<std::uint8_t*>(staging.p);
        experts.staging_slots = staging_slots;
        experts.route_log     = static_cast<std::int32_t*>(route_log.p);
        experts.route_stride  = route_stride;
        work = std::make_unique<WorkspaceArena>(
            ex::Forward::workspace_bytes(c, kPrefillChunk, static_cast<std::uint32_t>(context), device.execution_view(),
                                         false));
        forward = std::make_unique<ex::Forward>(parameters, device, *work, std::move(state), std::move(kv),
                                                std::move(experts));
    }

    void save() {
        state_saved = DeviceBuffer(state_backing.bytes);
        kv_saved    = DeviceBuffer(kv_backing.bytes);
        CUDA_CHECK(cudaMemcpy(state_saved.p, state_backing.p, state_backing.bytes, cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(kv_saved.p, kv_backing.p, kv_backing.bytes, cudaMemcpyDeviceToDevice));
    }
    void restore() {
        device.synchronize();
        CUDA_CHECK(cudaMemcpy(state_backing.p, state_saved.p, state_backing.bytes, cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(kv_backing.p, kv_saved.p, kv_backing.bytes, cudaMemcpyDeviceToDevice));
    }

    // Makes the given (layer, expert) pairs resident, in order, as far as the frames reach.
    std::int32_t make_resident(const std::vector<std::pair<std::uint32_t, std::uint32_t>>& experts) {
        const std::size_t E = config.moe.experts;
        std::int32_t used   = 0;
        for (const auto& [l, e] : experts) {
            if (used == frame_capacity) { break; }
            const ex::MoeParameters& moe = parameters.layers[l].moe;
            CUDA_CHECK(cudaMemcpy(static_cast<std::byte*>(frame_pool.p) + std::size_t(used) * frame_stride,
                                  moe.bank->host_records + std::size_t(e) * moe.bank->record_stride,
                                  moe.bank->record_stride, cudaMemcpyHostToDevice));
            tables_host[l * E + e] = used++;
        }
        frames_backing.copy_from_host(tables_host.data(), tables_host.size() * 4);
        return used;
    }

    // The routed ids of the last call of B columns: [layer][column][k].
    std::vector<std::int32_t> routes(std::int32_t columns) const {
        const std::size_t L = config.num_hidden_layers, K = config.moe.top_k;
        std::vector<std::int32_t> all(L * route_stride), out(L * K * static_cast<std::size_t>(columns));
        route_log.copy_to_host(all.data(), all.size() * 4);
        for (std::size_t l = 0; l < L; ++l) {
            std::copy_n(all.begin() + static_cast<std::ptrdiff_t>(l * route_stride), K * columns,
                        out.begin() + static_cast<std::ptrdiff_t>(l * K * columns));
        }
        return out;
    }

    const q4::TextConfig& config;
    const ex::Parameters& parameters;
    DeviceContext& device;
    std::int32_t slots;
    std::uint64_t frame_stride = 0;
    std::size_t route_stride   = 0;
    std::int32_t frame_capacity = 0;
    std::vector<std::int32_t> tables_host;
    DeviceBuffer state_backing, kv_backing, frames_backing, frame_pool, staging, route_log, state_saved, kv_saved;
    std::unique_ptr<WorkspaceArena> work;
    std::unique_ptr<ex::Forward> forward;
};

// Device inputs of one rows call at fixed addresses (a captured graph reads them every replay).
struct RowsInputs {
    RowsInputs(const q4::TextConfig& c, std::int32_t rows) : rows(rows) {
        const q4::NgramHash hash(c.ple.ngram, 0);
        row_bytes = c.ple.table.row_bytes * hash.heads();
        buffer    = DeviceBuffer(4096 + row_bytes * rows);
        auto* p   = static_cast<std::byte*>(buffer.p);
        batch.slot_ids         = Tensor(p, DType::I32, {rows});
        batch.ids              = Tensor(p + 256, DType::I32, {rows});
        batch.positions        = Tensor(p + 512, DType::I32, {rows});
        batch.rope_positions   = Tensor(p + 768, DType::I32, {rows, 3});
        batch.row_rope         = Tensor(p + 1024, DType::I32, {3, rows});
        batch.block_start_rope = Tensor(p + 1280, DType::I32, {3, rows});
        batch.ngram_rows       = Tensor(p + 4096, DType::U8,
                                        {static_cast<std::int32_t>(c.ple.table.row_bytes), static_cast<std::int32_t>(hash.heads()), rows});
        for (std::int32_t b = 0; b < rows; ++b) { batch.slots.push_back(b); }
    }

    // Column b: slot b's token `ids[b]` at position `positions[b]`, with that position's n-gram rows.
    void stage(const q4::TextConfig& c, const std::vector<std::int32_t>& ids, const std::vector<std::int32_t>& positions,
               const std::vector<std::vector<std::byte>>& ngram) {
        // Words at the tensors' byte offsets / 4; a text token's three RoPE axes equal its position.
        const auto R = static_cast<std::int32_t>(c.qsa.compress_ratio);
        std::vector<std::int32_t> words(384, 0);
        for (std::int32_t b = 0; b < rows; ++b) {
            const std::int32_t p = positions[b];
            words[b]       = b;     // slot_ids
            words[64 + b]  = ids[b];
            words[128 + b] = p;
            for (int a = 0; a < 3; ++a) {
                words[192 + a * rows + b] = p;         // rope_positions [B, 3] axis-major
                words[256 + 3 * b + a]    = p;         // row_rope [3, B]
                words[320 + 3 * b + a]    = p / R * R; // block_start_rope [3, B]
            }
        }
        buffer.copy_from_host(words.data(), words.size() * 4);
        for (std::int32_t b = 0; b < rows; ++b) {
            buffer.copy_from_host(ngram[b].data(), row_bytes, 4096 + row_bytes * b);
        }
    }

    std::int32_t rows;
    std::size_t row_bytes = 0;
    DeviceBuffer buffer;
    ex::ForwardRowsBatch batch;
};

// Device inputs of one single-sequence decode call at fixed addresses.
struct SingleInputs {
    explicit SingleInputs(const q4::TextConfig& c) {
        const q4::NgramHash hash(c.ple.ngram, 0);
        row_bytes = c.ple.table.row_bytes * hash.heads();
        buffer    = DeviceBuffer(4096 + row_bytes);
        auto* p   = static_cast<std::byte*>(buffer.p);
        batch.ids              = Tensor(p, DType::I32, {1});
        batch.positions        = Tensor(p + 256, DType::I32, {1});
        batch.rope_positions   = Tensor(p + 512, DType::I32, {1, 3});
        batch.block_start_rope = Tensor(p + 768, DType::I32, {3});
        batch.logit_columns    = Tensor(p + 1024, DType::I32, {1});
        batch.ngram_rows       = Tensor(p + 4096, DType::U8,
                                        {static_cast<std::int32_t>(c.ple.table.row_bytes), static_cast<std::int32_t>(hash.heads()), 1});
    }

    void stage(const q4::TextConfig& c, std::int32_t id, std::int32_t position, const std::vector<std::byte>& ngram) {
        const std::int32_t block = position / static_cast<std::int32_t>(c.qsa.compress_ratio) *
                                   static_cast<std::int32_t>(c.qsa.compress_ratio);
        std::vector<std::int32_t> words(257, 0);
        words[0]   = id;
        words[64]  = position;
        words[128] = words[129] = words[130] = position;
        words[192] = words[193] = words[194] = block;
        words[256] = 0;
        buffer.copy_from_host(words.data(), words.size() * 4);
        buffer.copy_from_host(ngram.data(), row_bytes, 4096);
    }

    std::size_t row_bytes = 0;
    DeviceBuffer buffer;
    ex::ForwardBatch batch;
};

// Rows mode: see the file comment.
int rows_mode(int argc, char** argv, const char* artifact_env) {
    std::int32_t steps = 0, time_rounds = 0, staging_slots = 64;
    std::int64_t frames_mib = -1;
    std::vector<std::string> prompt_args;
    std::vector<std::int32_t> configs;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg.rfind("--", 0) != 0) {
            prompt_args.push_back(arg);
            continue;
        }
        if (i + 1 >= argc) { throw std::invalid_argument(arg + " needs a value"); }
        const std::string value = argv[++i];
        if (arg == "--rows") {
            steps = std::stoi(value);
        } else if (arg == "--time") {
            time_rounds = std::stoi(value);
        } else if (arg == "--frames-mib") {
            frames_mib = std::stoll(value);
        } else if (arg == "--staging") {
            staging_slots = std::stoi(value);
        } else if (arg == "--configs") {
            configs = parse_tokens(value);
        } else {
            throw std::invalid_argument("unknown option " + arg);
        }
    }
    const auto P = static_cast<std::int32_t>(prompt_args.size());
    if (steps < 1 || P < 1 || P > ex::kMaxForwardRows || time_rounds < 0 || time_rounds > steps) {
        throw std::invalid_argument("--rows STEPS needs 1 to 8 prompts, and --time ROUNDS at most STEPS");
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
    std::printf("loaded in %.1f s: %.2f GiB device, %.2f GiB pinned host\n", seconds_since(t0),
                gib(model->storage_stats().device_capacity_bytes), gib(model->storage_stats().pinned_bytes));
    const ex::Parameters parameters(*model);
    const auto& c         = model->config();
    const char* ngram_env = std::getenv("NINFER_TEST_QWEN4EXP_NGRAM");
    q4::NgramVolume volume(ngram_env != nullptr ? std::filesystem::path(ngram_env) : q4::default_ngram_volume(artifact_env),
                           c.ple.table);
    volume.set_keepalive(std::chrono::milliseconds(100), std::chrono::milliseconds(60000));
    const std::size_t vocab = c.vocab_size;

    std::vector<std::vector<std::int32_t>> prompts;
    std::int32_t longest = 0;
    for (const auto& arg : prompt_args) {
        prompts.push_back(parse_prompt(arg, model->resources()));
        if (prompts.back().size() < 2) { throw std::invalid_argument("a prompt needs at least two tokens"); }
        longest = std::max<std::int32_t>(longest, static_cast<std::int32_t>(prompts.back().size()));
    }
    const std::int32_t context = (longest + steps + 1 + 63) / 64 * 64;
    std::printf("rows mode: %d prompts (", P);
    for (const auto& p : prompts) { std::printf(" %zu", p.size()); }
    std::printf(" tokens), %d decode steps each, context %d, staging slots %d\n", steps, context, staging_slots);

    // 1. Each prompt alone through the single-sequence path: greedy ids [steps + 1] (the prefill's
    // argmax, then one per decode call) and each decode call's logits.
    std::vector<std::vector<std::int32_t>> reference(static_cast<std::size_t>(P));
    std::vector<std::vector<std::vector<float>>> reference_logits(static_cast<std::size_t>(P));
    DeviceBuffer logits(vocab * ex::kMaxForwardRows * sizeof(float));
    std::vector<float> host(vocab * ex::kMaxForwardRows);
    double ngram_ms = 0.0;
    std::int64_t ngram_reads = 0;
    for (std::int32_t i = 0; i < P; ++i) {
        auto tokens = prompts[static_cast<std::size_t>(i)];
        const auto n = static_cast<std::int32_t>(tokens.size());
        Harness harness(parameters, device, context, std::min(kPrefillChunk, n), staging_slots);
        Tensor one(logits.p, DType::FP32, {static_cast<std::int32_t>(vocab), 1});
        for (std::int32_t first = 0; first < n; first += kPrefillChunk) {
            auto call = make_call(c, volume, tokens, first, std::min(kPrefillChunk, n - first), false);
            harness.forward->run(call.batch, one);
            device.synchronize();
        }
        logits.copy_to_host(host.data(), vocab * 4);
        auto& ids = reference[static_cast<std::size_t>(i)];
        ids.push_back(argmax(host.data(), vocab));
        t0 = std::chrono::steady_clock::now();
        for (std::int32_t k = 0; k < steps; ++k) {
            tokens.push_back(ids.back());
            const auto ngram_start = std::chrono::steady_clock::now();
            auto step = make_call(c, volume, tokens, n + k, 1, false);
            ngram_ms += seconds_since(ngram_start) * 1e3;
            ++ngram_reads;
            harness.forward->run(step.batch, one);
            device.synchronize();
            logits.copy_to_host(host.data(), vocab * 4);
            reference_logits[static_cast<std::size_t>(i)].emplace_back(host.begin(), host.begin() + vocab);
            ids.push_back(argmax(host.data(), vocab));
        }
        std::printf("prompt %d alone: %d tokens prefilled, %d decode calls in %.1f s (eager, every expert staged)\n", i, n,
                    steps, seconds_since(t0));
    }
    std::printf("n-gram rows of one decode column (hash, read, stage; host): mean %.3f ms over %lld reads\n",
                ngram_ms / static_cast<double>(ngram_reads), static_cast<long long>(ngram_reads));

    // 2. One Forward over P slots.
    std::size_t free_bytes = 0, total_bytes = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
    const std::uint64_t reserve = 3ULL << 30; // the rows harness's state, KV, workspace and graphs
    std::uint64_t frame_budget  = time_rounds > 0 && free_bytes > reserve ? free_bytes - reserve : 0;
    if (frames_mib >= 0) { frame_budget = std::min<std::uint64_t>(frame_budget, std::uint64_t(frames_mib) << 20); }
    RowsHarness rows(parameters, device, P, context, staging_slots, frame_budget);
    for (std::int32_t i = 0; i < P; ++i) {
        const auto& tokens = prompts[static_cast<std::size_t>(i)];
        const auto n       = static_cast<std::int32_t>(tokens.size());
        rows.forward->select_row(i);
        Tensor one(logits.p, DType::FP32, {static_cast<std::int32_t>(vocab), 1});
        for (std::int32_t first = 0; first < n; first += kPrefillChunk) {
            auto call = make_call(c, volume, tokens, first, std::min(kPrefillChunk, n - first), false);
            rows.forward->run(call.batch, one);
            device.synchronize();
        }
        logits.copy_to_host(host.data(), vocab * 4);
        if (argmax(host.data(), vocab) != reference[static_cast<std::size_t>(i)].front()) {
            std::printf("FAIL: slot %d's prefill disagrees with the prompt alone\n", i);
            return 1;
        }
    }
    rows.save();

    // The n-gram rows of every decode column: column k of prompt i is position n_i + k.
    std::vector<std::vector<std::vector<std::byte>>> ngram(static_cast<std::size_t>(P));
    for (std::int32_t i = 0; i < P; ++i) {
        auto tokens = prompts[static_cast<std::size_t>(i)];
        for (std::int32_t k = 0; k < steps; ++k) {
            tokens.push_back(reference[static_cast<std::size_t>(i)][static_cast<std::size_t>(k)]);
            ngram[static_cast<std::size_t>(i)].push_back(last_rows(c, volume, tokens));
        }
    }
    const auto stage_round = [&](RowsInputs& inputs, std::int32_t k) {
        std::vector<std::int32_t> ids, positions;
        std::vector<std::vector<std::byte>> rows_ngram;
        for (std::int32_t b = 0; b < inputs.rows; ++b) {
            ids.push_back(reference[static_cast<std::size_t>(b)][static_cast<std::size_t>(k)]);
            positions.push_back(static_cast<std::int32_t>(prompts[static_cast<std::size_t>(b)].size()) + k);
            rows_ngram.push_back(ngram[static_cast<std::size_t>(b)][static_cast<std::size_t>(k)]);
        }
        inputs.stage(c, ids, positions, rows_ngram);
    };

    // B rows from the saved state, `steps` rounds: every row's ids and logits against its reference.
    std::vector<std::vector<std::int32_t>> discovered; // the routes of the widest run, per round
    bool all_equal = true;
    for (std::int32_t B = 1; B <= P; ++B) {
        rows.restore();
        RowsInputs inputs(c, B);
        Tensor out(logits.p, DType::FP32, {static_cast<std::int32_t>(vocab), B});
        std::int32_t ids_equal = 0, logits_equal = 0;
        double max_diff = 0.0;
        for (std::int32_t k = 0; k < steps; ++k) {
            stage_round(inputs, k);
            rows.forward->run_rows(inputs.batch, out);
            device.synchronize();
            logits.copy_to_host(host.data(), vocab * B * 4);
            if (B == P) { discovered.push_back(rows.routes(B)); }
            for (std::int32_t b = 0; b < B; ++b) {
                const float* row     = host.data() + vocab * b;
                const auto& expected = reference_logits[static_cast<std::size_t>(b)][static_cast<std::size_t>(k)];
                ids_equal += argmax(row, vocab) == reference[static_cast<std::size_t>(b)][static_cast<std::size_t>(k) + 1];
                logits_equal += std::equal(row, row + vocab, expected.begin());
                for (std::size_t v = 0; v < vocab; ++v) { max_diff = std::max(max_diff, std::abs(double(row[v]) - expected[v])); }
            }
        }
        const bool equal = ids_equal == B * steps;
        all_equal        = all_equal && equal;
        std::printf("B = %d: greedy ids equal to each prompt alone in %d of %d row-steps, logits bitwise equal in %d, "
                    "max |logit diff| %.6f%s\n",
                    B, ids_equal, B * steps, logits_equal, max_diff, equal ? "" : "  <-- FAIL");
    }
    if (!all_equal) {
        std::printf("FAIL: rows decode disagrees with the prompts decoded alone\n");
        return 1;
    }
    if (time_rounds == 0) {
        std::printf("rows checks passed\n");
        return 0;
    }

    // 3. Timing. Every expert the widest run routed, most used first, in device frames.
    const std::size_t L = c.num_hidden_layers, K = c.moe.top_k, E = c.moe.experts;
    std::vector<std::uint32_t> uses(L * E, 0);
    for (std::int32_t k = 0; k < time_rounds; ++k) {
        const auto& r = discovered[static_cast<std::size_t>(k)];
        for (std::size_t l = 0; l < L; ++l) {
            for (std::size_t j = 0; j < K * static_cast<std::size_t>(P); ++j) { ++uses[l * E + r[l * K * P + j]]; }
        }
    }
    std::vector<std::pair<std::uint32_t, std::uint32_t>> wanted;
    for (std::size_t i = 0; i < uses.size(); ++i) {
        if (uses[i] > 0) { wanted.emplace_back(static_cast<std::uint32_t>(i / E), static_cast<std::uint32_t>(i % E)); }
    }
    std::stable_sort(wanted.begin(), wanted.end(), [&](const auto& a, const auto& b) {
        return uses[a.first * E + a.second] > uses[b.first * E + b.second];
    });
    const std::int32_t resident = rows.make_resident(wanted);
    std::printf("timing: %zu distinct experts routed in %d rounds of B = %d; %d resident (%d frames of %.2f MiB)\n",
                wanted.size(), time_rounds, P, resident, rows.frame_capacity, rows.frame_stride / 1048576.0);

    struct Sample {
        std::vector<double> ms;
        double distinct = 0.0, misses = 0.0;
        std::int32_t rounds = 0;
    };
    // One timed run: restore, then `time_rounds` rounds of B rows (B = 0: the single-sequence call on
    // slot 0), round 0 eager, round 1 captured, every later round a replay of the graph.
    const auto timed = [&](std::int32_t B, Sample& sample) {
        rows.restore();
        const std::int32_t columns = std::max(B, 1);
        RowsInputs inputs(c, columns);
        SingleInputs single(c);
        Tensor out(logits.p, DType::FP32, {static_cast<std::int32_t>(vocab), columns});
        DecodeGraphDefinition graph;
        DecodeGraphExecutable exec;
        cudaEvent_t start = nullptr, stop = nullptr;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
        const auto body = [&] {
            if (B == 0) {
                rows.forward->select_row(0);
                rows.forward->run(single.batch, out);
            } else {
                rows.forward->run_rows(inputs.batch, out);
            }
        };
        for (std::int32_t k = 0; k < time_rounds; ++k) {
            if (B == 0) {
                single.stage(c, reference[0][static_cast<std::size_t>(k)], static_cast<std::int32_t>(prompts[0].size()) + k,
                             ngram[0][static_cast<std::size_t>(k)]);
            } else {
                stage_round(inputs, k);
            }
            if (k == 0) {
                body();
            } else {
                if (!exec.ready()) {
                    graph.capture(device.stream, body);
                    exec.instantiate(graph);
                    exec.upload(device.stream);
                }
                CUDA_CHECK(cudaEventRecord(start, device.stream));
                exec.launch(device.stream);
                CUDA_CHECK(cudaEventRecord(stop, device.stream));
            }
            device.synchronize();
            if (k > 0) {
                float ms = 0.0F;
                CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
                sample.ms.push_back(ms);
            }
            logits.copy_to_host(host.data(), vocab * columns * 4);
            for (std::int32_t b = 0; b < columns; ++b) {
                if (argmax(host.data() + vocab * b, vocab) !=
                    reference[static_cast<std::size_t>(b)][static_cast<std::size_t>(k) + 1]) {
                    throw std::runtime_error("timed decode diverged from the reference");
                }
            }
            const auto r = rows.routes(columns);
            for (std::size_t l = 0; l < L; ++l) {
                std::vector<std::int32_t> ids(r.begin() + static_cast<std::ptrdiff_t>(l * K * columns),
                                              r.begin() + static_cast<std::ptrdiff_t>((l + 1) * K * columns));
                std::sort(ids.begin(), ids.end());
                ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
                sample.distinct += static_cast<double>(ids.size()) / static_cast<double>(L);
                for (const std::int32_t e : ids) { sample.misses += rows.tables_host[l * E + static_cast<std::size_t>(e)] < 0; }
            }
            ++sample.rounds;
        }
        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
    };
    std::vector<Sample> samples(static_cast<std::size_t>(P + 1));
    if (configs.empty()) {
        // ABBA against the single-sequence call: S R_B R_B S for B = 1 .. P, twice.
        for (int repeat = 0; repeat < 2; ++repeat) {
            for (std::int32_t B = 1; B <= P; ++B) {
                for (const std::int32_t run : {0, B, B, 0}) { timed(run, samples[static_cast<std::size_t>(run)]); }
            }
        }
    } else {
        // The listed configurations once each, inside a profiler capture range (nsys
        // --capture-range=cudaProfilerApi attributes kernel time to exactly these rounds).
        CUDA_CHECK(cudaProfilerStart());
        for (const std::int32_t run : configs) {
            if (run < 0 || run > P) { throw std::invalid_argument("--configs entries are 0 (single) to P"); }
            timed(run, samples[static_cast<std::size_t>(run)]);
        }
        CUDA_CHECK(cudaProfilerStop());
    }
    const auto report = [&](const char* name, const Sample& s) {
        const auto [lo, hi] = std::minmax_element(s.ms.begin(), s.ms.end());
        std::printf("%-26s median %.3f ms per round [%.3f-%.3f] over %zu rounds; %.2f distinct experts per layer, "
                    "%.2f misses per round\n",
                    name, median(s.ms), *lo, *hi, s.ms.size(), s.distinct / s.rounds, s.misses / s.rounds);
    };
    report("single-sequence call", samples[0]);
    for (std::int32_t B = 1; B <= P; ++B) {
        const std::string name = "rows call, B = " + std::to_string(B);
        report(name.c_str(), samples[static_cast<std::size_t>(B)]);
    }
    if (samples[0].ms.empty()) {
        std::printf("rows checks passed\n");
        return 0;
    }
    const double base   = median(samples[0].ms), base_d = samples[0].distinct / samples[0].rounds;
    const double base_m = samples[0].misses / samples[0].rounds;
    for (std::int32_t B = 2; B <= P; ++B) {
        const Sample& s = samples[static_cast<std::size_t>(B)];
        if (s.ms.empty()) { continue; }
        const double delta = (median(s.ms) - base) / (B - 1);
        // The spike's terms: per layer 3.67 us per distinct expert and 4.3 us per column (Op bench),
        // and 43.6 us per miss (its fit of the server's C = 1 decode).
        const double expert = 48.0 * (3.67 * (s.distinct / s.rounds - base_d) + 4.3 * (B - 1)) / 1000.0 / (B - 1);
        const double misses = 0.0436 * (s.misses / s.rounds - base_m) / (B - 1);
        std::printf("B = %d: +%.3f ms per extra row over the single-sequence call; of which routed-expert work "
                    "(bench model) %.3f ms, extra misses (43.6 us each) %.3f ms, non-expert %.3f ms\n",
                    B, delta, expert, misses, delta - expert - misses);
    }
    std::printf("rows checks passed\n");
    return 0;
}

// -------------------------------------------------------------------------------- KLD mode

// The widest verification call the Program runs: the anchor and at most 7 MTP drafts
// (program.cpp plan_layout).
constexpr std::int32_t kMaxVerifyColumns = 8;

// The bytes from layer 0's tensor to layer 1's (uniform per-layer strides), or `single` with one layer.
std::size_t layer_stride(const std::vector<Tensor>& layers, std::size_t single) {
    if (layers.size() < 2) { return single; }
    return static_cast<std::size_t>(static_cast<const std::byte*>(layers[1].data) -
                                    static_cast<const std::byte*>(layers[0].data));
}

// What a Harness's one sequence needs for W-column verification calls (W in [2, max_width]) and
// their commit, as the Program's SpecLayout holds it: the GDN records planned at the widest width
// (a narrower width's planes fit in them), the PLE convolution inputs, the raw index keys, copies of
// the QSA tails, the convolution's discarded window, the slot word (0) and commit words 0..max_width.
struct Verifier {
    static std::int32_t dim(std::uint64_t v) { return static_cast<std::int32_t>(v); }
    static std::size_t align256(std::size_t v) { return (v + 255) / 256 * 256; }

    Verifier(const Harness& h, std::int32_t max_width, bool gdn_head_map = false)
        : h(h), c(h.config), gdn_head_map(gdn_head_map) {
        di = dim(c.qsa.index_head_dim);
        r  = dim(c.qsa.compress_ratio);
        LayoutBuilder builder;
        records_bytes = align256(plan_gdn_replay_records(builder, records_spec(max_width)).payload_bytes());
        tail_stride   = layer_stride(h.views.qsa_tails, h.views.qsa_tails[0].bytes());
        const auto W  = static_cast<std::size_t>(max_width);
        const std::size_t ple_bytes   = align256(std::size_t(c.residual_width()) * W * 2);
        const std::size_t keys_bytes  = align256(std::size_t(di) * W * c.attention_layers * 2);
        const std::size_t tails_bytes = align256(tail_stride * c.attention_layers);
        const std::size_t conv_bytes  = align256(std::size_t(c.gdn.conv_channels()) * (c.gdn.conv_kernel - 1) * 2);
        backing = DeviceBuffer(records_bytes + ple_bytes + keys_bytes + tails_bytes + conv_bytes + 256 +
                               align256(4 * (W + 1)));
        backing.fill(0);
        auto* p      = static_cast<std::byte*>(backing.p);
        ple_inputs   = p + records_bytes;
        qsa_keys     = ple_inputs + ple_bytes;
        tail_copies  = qsa_keys + keys_bytes;
        conv         = tail_copies + tails_bytes;
        slot         = conv + conv_bytes; // holds 0
        commit_words = slot + 256;
        std::vector<std::int32_t> words(W + 1);
        std::iota(words.begin(), words.end(), 0);
        backing.copy_from_host(words.data(), words.size() * 4, static_cast<std::size_t>(commit_words - p));
        folds.resize(W + 1);
    }

    // The Program's records (one key head per value head, or the key heads under the tiled map).
    GdnReplayRecordSpec records_spec(std::int32_t width) const { return q4::gdn_records_spec(c, width, gdn_head_map); }

    ex::ForwardVerify view(std::int32_t W) const {
        LayoutBuilder builder;
        ex::ForwardVerify v;
        v.gdn        = GdnReplayRecords(DeviceSpan{backing.p, records_bytes},
                                        plan_gdn_replay_records(builder, records_spec(W)));
        v.ple_inputs = Tensor(ple_inputs, DType::BF16, {dim(c.residual_width()), W});
        v.qsa_keys   = Tensor(qsa_keys, DType::BF16, {di, W, dim(c.attention_layers)});
        for (std::uint32_t l = 0; l < c.attention_layers; ++l) {
            v.qsa_tails.emplace_back(tail_copies + tail_stride * l, DType::BF16,
                                     std::initializer_list<std::int32_t>{di, r - 1});
        }
        v.conv_scratch = Tensor(conv, DType::BF16, {dim(c.gdn.conv_channels()), dim(c.gdn.conv_kernel - 1)});
        v.initial_slot = Tensor(slot, DType::I32, {1});
        return v;
    }

    // Before a verification call: it pools from copies of the live tails.
    void copy_tails(cudaStream_t s) const {
        CUDA_CHECK(cudaMemcpyAsync(tail_copies, h.views.qsa_tails[0].data, tail_stride * c.attention_layers,
                                   cudaMemcpyDeviceToDevice, s));
    }

    // After a W-column verification call over `positions`: commits all W columns, as
    // commit_verified commits an accepted round (the GDN fold, qsa_commit_tails, ple_conv_commit).
    void commit(const ex::ForwardVerify& view, std::int32_t W, const Tensor& positions, cudaStream_t s) {
        auto& fold = folds.at(static_cast<std::size_t>(W));
        if (!fold) {
            // The one sequence's convolution windows and recurrences, one slot, a uniform layer stride.
            const auto stride = static_cast<std::int64_t>(layer_stride(h.views.gdn_conv, 0));
            LinearAttentionStateAllLayersView state;
            state.conv_layer0      = Tensor(h.views.gdn_conv[0].data, DType::BF16,
                                            {dim(c.gdn.conv_channels()), dim(c.gdn.conv_kernel - 1), 1});
            state.recurrent_layer0 = Tensor(h.views.gdn_recurrent[0].data, DType::FP32,
                                            {dim(c.gdn.key_head_dim), dim(c.gdn.value_head_dim),
                                             dim(c.gdn.value_heads), 1});
            state.conv_layer_stride_bytes      = stride;
            state.recurrent_layer_stride_bytes = stride;
            state.spec = LinearAttentionStatePoolSpec{.layers         = c.gdn_layers,
                                                      .conv_channels  = dim(c.gdn.conv_channels()),
                                                      .conv_width     = dim(c.gdn.conv_kernel - 1),
                                                      .value_heads    = dim(c.gdn.value_heads),
                                                      .value_head_dim = dim(c.gdn.value_head_dim),
                                                      .key_head_dim   = dim(c.gdn.key_head_dim),
                                                      .slot_count     = 1};
            fold.emplace(view.gdn, state);
        }
        const ops::GdnReplayFoldRow row{0, 0, W};
        fold->execute(std::span<const ops::GdnReplayFoldRow>(&row, 1), s);
        const Tensor count(reinterpret_cast<std::int32_t*>(commit_words) + W, DType::I32, {1});
        Tensor tail0 = h.views.qsa_tails[0];
        ops::qsa_commit_tails(view.qsa_keys, positions, count, tail0, static_cast<std::int64_t>(tail_stride), s);
        Tensor ple_state = h.views.ple_conv;
        ops::ple_conv_commit(view.ple_inputs.view({dim(c.residual_width()), W, 1}), count, ple_state,
                             h.views.ple_slot, s);
    }

    const Harness& h;
    const q4::TextConfig& c;
    bool gdn_head_map = false;
    std::int32_t di = 0, r = 0;
    std::size_t records_bytes = 0, tail_stride = 0;
    DeviceBuffer backing;
    std::byte *ple_inputs = nullptr, *qsa_keys = nullptr, *tail_copies = nullptr, *conv = nullptr, *slot = nullptr,
              *commit_words = nullptr;
    std::vector<std::optional<ops::GdnReplayFoldPlan>> folds;
};

// KLD mode: see the file comment.
int kld_mode(int argc, char** argv, const char* artifact_env) {
    std::optional<std::filesystem::path> base_path;
    std::optional<int> chunks;
    std::int32_t window = ppl::kDefaultDecodeWindow, staging_slots = 64;
    std::string tsv_path = "kld-rows.tsv";
    bool hc_int8         = false;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--hc-int8") {
            hc_int8 = true;
            continue;
        }
        if (i + 1 >= argc) { throw std::invalid_argument(arg + " needs a value"); }
        const std::string value = argv[++i];
        if (arg == "--kld-base") {
            base_path = value;
        } else if (arg == "--chunks") {
            chunks = ppl::parse_kld_chunks(value);
        } else if (arg == "--window") {
            window = std::stoi(value);
            if (window < 1 || window > kMaxVerifyColumns) {
                throw std::invalid_argument("--window must be 1 to " + std::to_string(kMaxVerifyColumns) +
                                            ": the Flash-Next verify forward is at most the anchor and 7 MTP drafts");
            }
        } else if (arg == "--tsv") {
            tsv_path = value;
        } else if (arg == "--staging") {
            staging_slots = std::stoi(value);
        } else {
            throw std::invalid_argument("unknown option " + arg + " in --kld-base mode");
        }
    }
    if (!base_path) { throw std::invalid_argument("--kld-base FILE is required"); }
    const ppl::KldBase base  = ppl::read_kld_base(*base_path);
    const int chunk_count    = ppl::kld_chunk_count(base, chunks);
    const auto n_ctx         = static_cast<std::int32_t>(base.n_ctx);
    const std::int32_t first = static_cast<std::int32_t>(base.n_ctx / 2U) + 1; // the first target
    std::printf("KLD base %s: n_ctx %d, n_vocab %d, %d of %d chunks; window T = %d, staging slots %d%s\n",
                base.path.string().c_str(), n_ctx, base.n_vocab, chunk_count, base.n_chunk, window, staging_slots,
                hc_int8 ? "; --hc-int8" : "");

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
    std::printf("loaded in %.1f s: %.2f GiB device, %.2f GiB pinned host\n", seconds_since(t0),
                gib(model->storage_stats().device_capacity_bytes), gib(model->storage_stats().pinned_bytes));
    const ex::Parameters parameters(*model);
    const auto& c         = model->config();
    const char* ngram_env = std::getenv("NINFER_TEST_QWEN4EXP_NGRAM");
    q4::NgramVolume volume(ngram_env != nullptr ? std::filesystem::path(ngram_env)
                                                : q4::default_ngram_volume(artifact_env),
                           c.ple.table);
    volume.set_keepalive(std::chrono::milliseconds(100), std::chrono::milliseconds(60000));
    // --hc-int8: the mixers' int8 copies, made once from the BF16 weights for every chunk's forward.
    std::unique_ptr<ex::HcInt8Copies> hc_int8_copies;
    ex::ForwardOptions options;
    if (hc_int8) {
        auto mixers = ex::HcInt8Copies::mixers(parameters.layers, parameters.final_mixer,
                                               parameters.mtp ? &*parameters.mtp : nullptr);
        const std::uint64_t bytes = ex::HcInt8Copies::bytes(mixers);
        hc_int8_copies = std::make_unique<ex::HcInt8Copies>(std::move(mixers), bytes, device.stream);
        options.hc_int8 = hc_int8_copies.get();
        std::printf("--hc-int8: %.1f MiB of int8 copies\n", static_cast<double>(bytes) / (1 << 20));
    }
    const std::size_t vocab = c.vocab_size;
    // The comparison domain is the public token domain (llama.cpp may pad n_vocab past it).
    const auto valid_vocab = static_cast<std::int32_t>(model->resources().public_token_count);
    if (base.n_vocab < valid_vocab) {
        throw std::runtime_error(base.path.string() + " has " + std::to_string(base.n_vocab) +
                                 " vocabulary rows, fewer than the model's " + std::to_string(valid_vocab) +
                                 " public tokens");
    }

    std::ofstream tsv(tsv_path, std::ios::binary | std::ios::trunc);
    if (!tsv) { throw std::runtime_error("cannot create " + tsv_path); }
    tsv << "chunk\tposition\ttarget\tnll\tnll_base\tkld\tsame_top1\tdlogp_top1\n" << std::setprecision(9);
    std::vector<ppl::ScoredRow> rows;
    DeviceBuffer logits(vocab * static_cast<std::size_t>(window) * sizeof(float));
    std::vector<float> host(vocab * static_cast<std::size_t>(window));
    const std::int32_t context = std::max<std::int32_t>(256, (n_ctx + 63) / 64 * 64);
    const auto scoring_started = std::chrono::steady_clock::now();
    for (int chunk = 0; chunk < chunk_count; ++chunk) {
        const ppl::KldChunk input = ppl::kld_chunk(base, chunk);
        const auto& tokens        = input.tokens;
        // A fresh sequence: positions [0, first - 1) prefilled as one call, then every target
        // [first, n_ctx) predicted by a decode forward: targets [b, e) are the logits of positions
        // [b - 1, e - 1), one plain decode call when e - b = 1, else one verification call of
        // e - b columns (an MTP round of e - b - 1 drafts) committed in full.
        Harness harness(parameters, device, context, std::max(first - 1, window), staging_slots, options);
        std::optional<Verifier> verifier;
        if (window > 1) { verifier.emplace(harness, window); }
        {
            auto prompt = make_call(c, volume, tokens, 0, first - 1, false);
            Tensor last(logits.p, DType::FP32, {static_cast<std::int32_t>(vocab), 1});
            harness.forward->run(prompt.batch, last);
            device.synchronize();
        }
        for (std::int32_t b = first; b < n_ctx; b += window) {
            const std::int32_t W = std::min(window, n_ctx - b);
            auto call            = make_call(c, volume, tokens, b - 1, W, W > 1);
            Tensor out(logits.p, DType::FP32, {static_cast<std::int32_t>(vocab), W});
            if (W == 1) {
                harness.forward->run(call.batch, out);
                device.synchronize();
            } else {
                const ex::ForwardVerify view = verifier->view(W);
                call.batch.verify            = &view;
                verifier->copy_tails(device.stream);
                harness.forward->run(call.batch, out);
                verifier->commit(view, W, call.batch.positions, device.stream);
                device.synchronize();
            }
            logits.copy_to_host(host.data(), vocab * static_cast<std::size_t>(W) * sizeof(float));
            for (std::int32_t j = 0; j < W; ++j) {
                const std::int32_t position = b + j;
                const auto base_logprobs    = ppl::kld_base_logprobs(base, chunk, ppl::kld_row(base, position));
                const std::span<const float> row(host.data() + vocab * static_cast<std::size_t>(j), vocab);
                const ppl::ScoredRow scored =
                    ppl::score_row(row, base_logprobs, tokens[static_cast<std::size_t>(position)], valid_vocab);
                rows.push_back(scored);
                tsv << chunk << '\t' << position << '\t' << tokens[static_cast<std::size_t>(position)] << '\t'
                    << scored.nll << '\t' << scored.nll_base << '\t' << scored.kld << '\t' << (scored.same_top1 ? 1 : 0)
                    << '\t' << scored.dlogp_top1 << '\n';
            }
        }
        if (!tsv) { throw std::runtime_error("cannot write " + tsv_path); }
        const ppl::KldSummary partial = ppl::summarize(rows);
        std::printf("chunk %d/%d: PPL %.4f, KLD mean %.5f, same top-1 %.2f%%, %.1f s elapsed\n", chunk + 1, chunk_count,
                    partial.ppl, partial.kld_mean, 100.0 * partial.same_top1, seconds_since(scoring_started));
    }
    tsv.flush();
    if (!tsv) { throw std::runtime_error("cannot write " + tsv_path); }
    const ppl::KldSummary summary = ppl::summarize(rows);
    double base_nll = 0.0;
    for (const auto& row : rows) { base_nll += row.nll_base; }
    std::printf("KL-divergence gate (decode path, T = %d, %d chunks, %zu rows)\n"
                "PPL:        %.6f +/- %.6f  (base %.6f)\n"
                "KLD mean:   %.6f\nKLD median: %.6f\nKLD p99:    %.6f\nKLD max:    %.6f\n"
                "same top-1: %.4f%%\nrows: %s\n",
                window, chunk_count, summary.rows, summary.ppl, summary.ppl_err,
                std::exp(base_nll / static_cast<double>(rows.size())), summary.kld_mean, summary.kld_median,
                summary.kld_p99, summary.kld_max, 100.0 * summary.same_top1, tsv_path.c_str());
    print_memory("done");
    return 0;
}

} // namespace

// The --hc-int8 prefill mode: see the file comment.
int hc_int8_prefill_mode(int argc, char** argv, const char* artifact_env) {
    std::vector<std::int32_t> tokens;
    std::int32_t staging_slots = 64;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--hc-int8-prefill") { continue; }
        if (arg.rfind("--", 0) != 0) {
            if (!tokens.empty()) { throw std::invalid_argument("--hc-int8-prefill takes one token list"); }
            tokens = parse_tokens(arg);
            continue;
        }
        if (i + 1 >= argc) { throw std::invalid_argument(arg + " needs a value"); }
        const std::string value = argv[++i];
        if (arg == "--staging") {
            staging_slots = std::stoi(value);
        } else {
            throw std::invalid_argument("unknown option " + arg + " in --hc-int8-prefill mode");
        }
    }
    const auto n            = static_cast<std::int32_t>(tokens.size());
    const std::int32_t head = n - 1;
    if (head <= ops::kHcInt8MaxColumns) {
        throw std::invalid_argument("--hc-int8-prefill needs at least " + std::to_string(ops::kHcInt8MaxColumns + 2) +
                                    " tokens: a prompt call wider than the int8 route, then one decode token");
    }
    DeviceContext device(0);
    artifact::Reader reader(artifact_env);
    if (!q4::is_qwen4_exp(reader)) {
        std::printf("SKIP: %s is not a Qwen4Exp artifact\n", artifact_env);
        return 77;
    }
    const auto model = q4::load_model(reader, device);
    device.synchronize();
    const ex::Parameters parameters(*model);
    const auto& c         = model->config();
    const char* ngram_env = std::getenv("NINFER_TEST_QWEN4EXP_NGRAM");
    q4::NgramVolume volume(ngram_env != nullptr ? std::filesystem::path(ngram_env)
                                                : q4::default_ngram_volume(artifact_env),
                           c.ple.table);
    auto mixers = ex::HcInt8Copies::mixers(parameters.layers, parameters.final_mixer,
                                           parameters.mtp ? &*parameters.mtp : nullptr);
    const std::uint64_t bytes = ex::HcInt8Copies::bytes(mixers);
    const ex::HcInt8Copies copies(std::move(mixers), bytes, device.stream);
    const std::size_t vocab    = c.vocab_size;
    const std::int32_t context = std::max<std::int32_t>(256, (n + 63) / 64 * 64);
    const Call prompt          = make_call(c, volume, tokens, 0, head, false);
    const Call step            = make_call(c, volume, tokens, head, 1, false);
    DeviceBuffer logits(vocab * sizeof(float));
    std::printf("prompt %d tokens as one call, then one decode call; staging slots %d\n", head, staging_slots);
    const auto run = [&](bool on) {
        const ex::ForwardOptions options{.hc_int8 = on ? &copies : nullptr};
        Harness harness(parameters, device, context, head, staging_slots, options);
        std::vector<std::vector<float>> rows;
        Tensor one(logits.p, DType::FP32, {static_cast<std::int32_t>(vocab), 1});
        for (const Call* call : {&prompt, &step}) {
            harness.forward->run(call->batch, one);
            device.synchronize();
            rows.push_back(to_float(logits, vocab));
        }
        return rows;
    };
    const auto off           = run(false);
    const auto on            = run(true);
    const bool prompt_equal  = std::memcmp(off[0].data(), on[0].data(), vocab * sizeof(float)) == 0;
    const bool decode_differ = std::memcmp(off[1].data(), on[1].data(), vocab * sizeof(float)) != 0;
    std::printf("prompt logits %s; decode logits %s with --hc-int8\n", prompt_equal ? "bitwise equal" : "DIFFER",
                decode_differ ? "differ" : "are bitwise EQUAL");
    if (prompt_equal && decode_differ) {
        std::printf("--hc-int8-prefill checks passed\n");
        return 0;
    }
    if (!prompt_equal) { std::printf("FAIL: --hc-int8 changed a call wider than 16 columns\n"); }
    if (!decode_differ) { std::printf("FAIL: the decode call did not read the int8 copies\n"); }
    return 1;
}

// The identity check of ForwardOptions::shared_expert_stream (see the header): four runs from a
// fresh state over the same device inputs, the shared expert serial or on its side stream, the
// decode calls eager or each captured in a CUDA graph and replayed once (the Program's decode). Each
// side-stream run's logits must be bitwise those of the serial run with the same decode mode.
int shared_expert_mode(int argc, char** argv, const char* artifact_env) {
    std::vector<std::int32_t> tokens;
    std::int32_t decode = 4, staging_slots = 64;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--shared-expert-stream") { continue; }
        if (arg.rfind("--", 0) != 0) {
            if (!tokens.empty()) { throw std::invalid_argument("--shared-expert-stream takes one token list"); }
            tokens = parse_tokens(arg);
            continue;
        }
        if (i + 1 >= argc) { throw std::invalid_argument(arg + " needs a value"); }
        const std::string value = argv[++i];
        if (arg == "--decode") {
            decode = std::stoi(value);
        } else if (arg == "--staging") {
            staging_slots = std::stoi(value);
        } else {
            throw std::invalid_argument("unknown option " + arg + " in --shared-expert-stream mode");
        }
    }
    const auto n = static_cast<std::int32_t>(tokens.size());
    if (decode < 1 || n - decode <= 8) {
        throw std::invalid_argument("--decode must be at least 1 and leave more than 8 prompt tokens (the prompt "
                                    "call must be wider than 8 columns to use the side stream's carve)");
    }

    DeviceContext device(0);
    artifact::Reader reader(artifact_env);
    if (!q4::is_qwen4_exp(reader)) {
        std::printf("SKIP: %s is not a Qwen4Exp artifact\n", artifact_env);
        return 77;
    }
    const auto model = q4::load_model(reader, device);
    device.synchronize();
    const ex::Parameters parameters(*model);
    const auto& c         = model->config();
    const char* ngram_env = std::getenv("NINFER_TEST_QWEN4EXP_NGRAM");
    q4::NgramVolume volume(ngram_env != nullptr ? std::filesystem::path(ngram_env)
                                                : q4::default_ngram_volume(artifact_env),
                           c.ple.table);
    const std::size_t vocab    = c.vocab_size;
    const std::int32_t context = std::max<std::int32_t>(256, (n + 63) / 64 * 64);
    const std::int32_t head    = n - decode;
    // Every run reads the same device inputs: the prompt as one call (wider than 8 columns, the side
    // stream's Linears take their Q8_1 scratch from the carve), then one call per decode token.
    const Call prompt = make_call(c, volume, tokens, 0, head, false);
    std::vector<Call> steps;
    for (std::int32_t p = head; p < n; ++p) { steps.push_back(make_call(c, volume, tokens, p, 1, false)); }
    DeviceBuffer logits(vocab * sizeof(float));
    Tensor out(logits.p, DType::FP32, {static_cast<std::int32_t>(vocab), 1});
    std::printf("prompt %d tokens as one call, then %d decode calls; staging slots %d\n", head, decode, staging_slots);

    // The logits of the prompt call, then of every decode call.
    const auto run = [&](bool side, bool graphs) {
        Harness harness(parameters, device, context, head, staging_slots, ex::ForwardOptions{.shared_expert_stream = side});
        std::vector<std::vector<float>> rows;
        harness.forward->run(prompt.batch, out);
        device.synchronize();
        rows.push_back(to_float(logits, vocab));
        for (const Call& step : steps) {
            if (graphs) {
                DecodeGraphDefinition graph;
                DecodeGraphExecutable exec;
                graph.capture(device.stream, [&] { harness.forward->run(step.batch, out); });
                exec.instantiate(graph);
                exec.launch(device.stream);
            } else {
                harness.forward->run(step.batch, out);
            }
            device.synchronize();
            rows.push_back(to_float(logits, vocab));
        }
        return rows;
    };
    bool ok = true;
    for (const bool graphs : {false, true}) {
        const auto serial = run(false, graphs);
        const auto side   = run(true, graphs);
        std::size_t equal = 0;
        for (std::size_t r = 0; r < serial.size(); ++r) {
            equal += std::memcmp(serial[r].data(), side[r].data(), vocab * sizeof(float)) == 0 ? 1 : 0;
        }
        std::printf("decode %s: %zu of %zu calls' logits bitwise equal with --shared-expert-stream\n",
                    graphs ? "in CUDA graphs" : "eager", equal, serial.size());
        ok &= equal == serial.size();
    }
    std::printf(ok ? "shared expert stream checks passed\n" : "FAIL: the shared expert stream changed the logits\n");
    return ok ? 0 : 1;
}

// The identity checks of ForwardOptions::gdn_head_map, ForwardOptions::moe_precast_input,
// ForwardOptions::moe_gate_up_registers and ForwardOptions::moe_fused_route (see the header; `flag`
// names the one checked, moe_gate_up_registers with the precast input it needs): the same calls from
// a fresh state without and with the option; every logits column must be bitwise equal.
int identity_mode(int argc, char** argv, const char* artifact_env, const std::string& flag) {
    std::vector<std::int32_t> tokens;
    std::int32_t window = 3, decode = 2, staging_slots = 64;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == flag) { continue; }
        if (arg.rfind("--", 0) != 0) {
            if (!tokens.empty()) { throw std::invalid_argument(flag + " takes one token list"); }
            tokens = parse_tokens(arg);
            continue;
        }
        if (i + 1 >= argc) { throw std::invalid_argument(arg + " needs a value"); }
        const std::string value = argv[++i];
        if (arg == "--window") {
            window = std::stoi(value);
        } else if (arg == "--decode") {
            decode = std::stoi(value);
        } else if (arg == "--staging") {
            staging_slots = std::stoi(value);
        } else {
            throw std::invalid_argument("unknown option " + arg + " in " + flag + " mode");
        }
    }
    if (window < 2 || window > kMaxVerifyColumns) {
        throw std::invalid_argument("--window must be 2 to " + std::to_string(kMaxVerifyColumns));
    }
    if (decode < 1) { throw std::invalid_argument("--decode must be at least 1"); }
    const auto n            = static_cast<std::int32_t>(tokens.size());
    const std::int32_t head = n - window - decode;
    if (head < 16) {
        throw std::invalid_argument(flag + " needs at least 16 prompt tokens before the window and the decode "
                                    "tokens (the GDN's chunked route, routed-expert calls wider than 8 columns)");
    }

    DeviceContext device(0);
    artifact::Reader reader(artifact_env);
    if (!q4::is_qwen4_exp(reader)) {
        std::printf("SKIP: %s is not a Qwen4Exp artifact\n", artifact_env);
        return 77;
    }
    const auto model = q4::load_model(reader, device);
    device.synchronize();
    const ex::Parameters parameters(*model);
    const auto& c         = model->config();
    const char* ngram_env = std::getenv("NINFER_TEST_QWEN4EXP_NGRAM");
    q4::NgramVolume volume(ngram_env != nullptr ? std::filesystem::path(ngram_env)
                                                : q4::default_ngram_volume(artifact_env),
                           c.ple.table);
    const std::size_t vocab    = c.vocab_size;
    const std::int32_t context = std::max<std::int32_t>(256, (n + 63) / 64 * 64);
    // Every run reads the same device inputs: the prompt, the verification window (every column's
    // logits), then one call per decode token.
    const Call prompt = make_call(c, volume, tokens, 0, head, false);
    Call verify       = make_call(c, volume, tokens, head, window, true);
    std::vector<Call> steps;
    for (std::int32_t p = head + window; p < n; ++p) { steps.push_back(make_call(c, volume, tokens, p, 1, false)); }
    DeviceBuffer logits(vocab * static_cast<std::size_t>(window) * sizeof(float));
    std::printf("prompt %d tokens as one call, a %d-column verification committed by the fold, %d decode calls; "
                "staging slots %d\n",
                head, window, decode, staging_slots);

    // Every logits column of the prompt, the verification and the decode calls, in order.
    const bool head_map_mode = flag == "--gdn-head-map";
    const auto run = [&](bool on) {
        const bool registers = on && flag == "--moe-gate-up-registers";
        const bool precast   = registers || (on && flag == "--moe-precast-input");
        const ex::ForwardOptions options{.gdn_head_map          = on && head_map_mode,
                                         .moe_precast_input     = precast,
                                         .moe_gate_up_registers = registers,
                                         .moe_fused_route       = on && flag == "--moe-fused-route"};
        Harness harness(parameters, device, context, head, staging_slots, options);
        Verifier verifier(harness, window, options.gdn_head_map);
        std::vector<std::vector<float>> rows;
        Tensor one(logits.p, DType::FP32, {static_cast<std::int32_t>(vocab), 1});
        harness.forward->run(prompt.batch, one);
        device.synchronize();
        rows.push_back(to_float(logits, vocab));
        const ex::ForwardVerify view = verifier.view(window);
        verify.batch.verify          = &view;
        Tensor all(logits.p, DType::FP32, {static_cast<std::int32_t>(vocab), window});
        verifier.copy_tails(device.stream);
        harness.forward->run(verify.batch, all);
        verifier.commit(view, window, verify.batch.positions, device.stream);
        device.synchronize();
        verify.batch.verify = nullptr;
        const std::vector<float> columns = to_float(logits, vocab * static_cast<std::size_t>(window));
        for (std::int32_t j = 0; j < window; ++j) {
            rows.emplace_back(columns.begin() + static_cast<std::ptrdiff_t>(vocab) * j,
                              columns.begin() + static_cast<std::ptrdiff_t>(vocab) * (j + 1));
        }
        for (const Call& step : steps) {
            harness.forward->run(step.batch, one);
            device.synchronize();
            rows.push_back(to_float(logits, vocab));
        }
        return rows;
    };
    const auto off = run(false);
    const auto on  = run(true);
    std::size_t equal = 0;
    for (std::size_t r = 0; r < off.size(); ++r) {
        equal += std::memcmp(off[r].data(), on[r].data(), vocab * sizeof(float)) == 0 ? 1 : 0;
    }
    std::printf("%zu of %zu logits columns bitwise equal with %s (prompt, %d verified, %d decoded)\n", equal,
                off.size(), flag.c_str(), window, decode);
    const bool ok = equal == off.size();
    if (ok) {
        std::printf("%s checks passed\n", flag.c_str());
    } else {
        std::printf("FAIL: %s changed the logits\n", flag.c_str());
    }
    return ok ? 0 : 1;
}

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
        if (std::string(argv[1]) == "--rows") { return rows_mode(argc, argv, artifact_env); }
        if (std::any_of(argv + 1, argv + argc, [](const char* a) { return std::string(a) == "--kld-base"; })) {
            return kld_mode(argc, argv, artifact_env);
        }
        if (std::any_of(argv + 1, argv + argc, [](const char* a) { return std::string(a) == "--hc-int8-prefill"; })) {
            return hc_int8_prefill_mode(argc, argv, artifact_env);
        }
        if (std::any_of(argv + 1, argv + argc, [](const char* a) { return std::string(a) == "--shared-expert-stream"; })) {
            return shared_expert_mode(argc, argv, artifact_env);
        }
        for (const char* flag :
             {"--gdn-head-map", "--moe-precast-input", "--moe-gate-up-registers", "--moe-fused-route"}) {
            if (std::any_of(argv + 1, argv + argc, [&](const char* a) { return std::string(a) == flag; })) {
                return identity_mode(argc, argv, artifact_env, flag);
            }
        }
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
