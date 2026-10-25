// Qwen3.8-Flash-Next MTP speculation on the real artifact with its drafter (converted with
// --mtp-gguf). Skips (77) unless NINFER_TEST_QWEN4EXP_ARTIFACT names such an artifact; the n-gram
// volume is NINFER_TEST_QWEN4EXP_NGRAM or <artifact>.ngram.
//
// 1. Verification is exact. From the state after a prompt, a verification call over W consecutive
//    tokens (records instead of state updates) gives every column the logits and final residual of
//    one-column decode calls at those positions, bit for bit, and leaves the state unchanged.
//    Committing its first k columns (the GDN fold, qsa_commit_tails, ple_conv_commit) leaves exactly
//    the state k decode calls leave: every GDN convolution window and recurrent state, the PLE
//    history and the QSA tails. W in {2, 3, 5, 8, 5, 7} from the one prompt end (the wider commits
//    cross a 4-token QSA block, which is what exposed the multi-block tail; W = 5 runs twice to
//    show the result repeats); k in {1, W/2, W}. This is what makes greedy MTP
//    output equal plain greedy output.
// 2. The drafter's step against FP64 oracles evaluated on its own tapped inputs (each op chain
//    checked locally): the input fusion (unit-offset RMSNorms of the token embedding and of the
//    residual, the two BF16 projections, the broadcast sum), the MoE (router top 10 over the BF16
//    rows, 512 Q2_0 experts through the canonical A8 cast, the BF16 shared expert and its gate, the
//    combine) and the draft head (the IQ4_XS LM head rows on the Q8_1-cast final hidden, and its
//    argmax).
//
// 3. ForwardOptions::shared_expert_stream (the shared expert on a side stream) leaves the prompt's
//    logits and the drafter step's MoE output, logits, residual and draft bitwise unchanged.
//
//   ninfer_qwen4_exp_mtp_real_test [TOKENS] [--prompt N]
//
// TOKENS is @FILE or a comma-separated id list (default: the code fixture text); the first N
// (default 300) tokens are the prompt, the next 8 the verified ones.

#include "artifact/reader.h"
#include "core/arena.h"
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
#include "ninfer/ops/gdn_replay.h"
#include "ninfer/ops/ple.h"
#include "ninfer/ops/qsa.h"
#include "ops/ggml_blocks_decode.h"
#include "ops/offloaded_moe_fixtures.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <memory>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

#ifndef NINFER_SOURCE_DIR
#    define NINFER_SOURCE_DIR "."
#endif

using namespace ninfer;
namespace q4 = ninfer::models::qwen4_exp;
namespace ex = ninfer::models::qwen4_exp::execution;

namespace {

int g_failures = 0;

void expect(bool ok, const std::string& what) {
    std::printf("%s: %s\n", ok ? "ok" : "FAIL", what.c_str());
    if (!ok) { ++g_failures; }
}

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

std::int32_t dim(std::uint64_t v) { return static_cast<std::int32_t>(v); }
std::size_t align256(std::size_t v) { return (v + 255) / 256 * 256; }

template <class T>
std::vector<T> download(const void* device, std::size_t count) {
    std::vector<T> out(count);
    if (cudaMemcpy(out.data(), device, count * sizeof(T), cudaMemcpyDeviceToHost) != cudaSuccess) {
        throw std::runtime_error("download failed");
    }
    return out;
}

double bf16(std::uint16_t h) { return std::bit_cast<float>(static_cast<std::uint32_t>(h) << 16); }

std::uint16_t to_bf16(double v) {
    const auto u = std::bit_cast<std::uint32_t>(static_cast<float>(v));
    return static_cast<std::uint16_t>((u + 0x7FFFU + ((u >> 16) & 1U)) >> 16);
}
double round_bf16(double v) { return bf16(to_bf16(v)); }

// Normwise relative error of `got` against `want`, and the worst element relative to the largest.
struct Error {
    double normwise = 0, worst = 0;
};
Error error_of(const std::vector<double>& got, const std::vector<double>& want) {
    double e2 = 0, r2 = 0, emax = 0, rmax = 0;
    for (std::size_t i = 0; i < want.size(); ++i) {
        const double e = got[i] - want[i];
        e2 += e * e;
        r2 += want[i] * want[i];
        emax = std::max(emax, std::fabs(e));
        rmax = std::max(rmax, std::fabs(want[i]));
    }
    return {r2 > 0 ? std::sqrt(e2 / r2) : std::sqrt(e2), rmax > 0 ? emax / rmax : emax};
}

// State, KV (the text model's 12 attention layers and the drafter's), the verification records and
// the drafter's buffers of one sequence, laid out as the Program lays them out (uniform per-layer
// GDN stride, contiguous QSA tails), and the forward over them.
struct Harness {
    Harness(const ex::Parameters& parameters, DeviceContext& device, std::int32_t context, std::int32_t columns,
            ex::ForwardOptions options = {})
        : c(parameters.model.config()) {
        conv_bytes      = std::size_t(c.gdn.conv_channels()) * (c.gdn.conv_kernel - 1) * 2;
        recurrent_bytes = std::size_t(c.gdn.key_head_dim) * c.gdn.value_head_dim * c.gdn.value_heads * 4;
        layer_stride    = align256(conv_bytes) + align256(recurrent_bytes);
        const std::size_t ple_bytes = std::size_t(c.residual_width()) * c.ple.conv_span() * 2;
        di = dim(c.qsa.index_head_dim);
        r  = dim(c.qsa.compress_ratio);
        tail_bytes = std::size_t(di) * (r - 1) * 2;
        state_bytes = c.gdn_layers * layer_stride + align256(ple_bytes) + c.attention_layers * align256(tail_bytes) + 256 +
                      align256(tail_bytes);
        state = DeviceBuffer(state_bytes);
        state.fill(0);
        auto* p = static_cast<std::byte*>(state.p);
        ex::ForwardState fs;
        for (std::uint32_t l = 0; l < c.gdn_layers; ++l) {
            fs.gdn_conv.emplace_back(p, DType::BF16, std::initializer_list<std::int32_t>{dim(c.gdn.conv_channels()),
                                                                                       dim(c.gdn.conv_kernel - 1)});
            fs.gdn_recurrent.emplace_back(p + align256(conv_bytes), DType::FP32,
                                          std::initializer_list<std::int32_t>{dim(c.gdn.key_head_dim),
                                                                              dim(c.gdn.value_head_dim),
                                                                              dim(c.gdn.value_heads)});
            p += layer_stride;
        }
        fs.ple_conv = Tensor(p, DType::BF16, {dim(c.residual_width()), dim(c.ple.conv_span()), 1});
        ple_state   = fs.ple_conv;
        p += align256(ple_bytes);
        tails = p;
        for (std::uint32_t l = 0; l < c.attention_layers; ++l) {
            fs.qsa_tails.emplace_back(p, DType::BF16, std::initializer_list<std::int32_t>{di, r - 1});
            p += align256(tail_bytes);
        }
        fs.ple_slot = Tensor(p, DType::I32, {1});
        ple_slot    = fs.ple_slot;
        p += 256;
        mtp_tail = Tensor(p, DType::BF16, {di, r - 1});

        const std::int32_t pages = (context + kPagedKVPageSize - 1) / kPagedKVPageSize;
        const std::int32_t D = dim(c.attention.head_dim), Hkv = dim(c.attention.kv_heads);
        const std::size_t codes  = std::size_t(D) * kPagedKVPageSize * Hkv * pages;
        const std::size_t scales = std::size_t(D / 64) * kPagedKVPageSize * Hkv * pages * 2;
        const std::size_t pooled = std::size_t(di / r) * kPagedKVPageSize * pages * 2;
        const std::size_t layer_bytes = align256(2 * codes + 2 * scales + pooled);
        kv_bytes = (c.attention_layers + 1) * layer_bytes + 4096;
        kv       = DeviceBuffer(kv_bytes);
        kv.fill(0);
        std::vector<std::int32_t> table(static_cast<std::size_t>(pages));
        std::iota(table.begin(), table.end(), 0);
        auto* kvp = static_cast<std::byte*>(kv.p);
        ex::ForwardKV fkv;
        fkv.block_table = Tensor(kvp, DType::I32, {pages});
        kv.copy_from_host(table.data(), table.size() * sizeof(std::int32_t));
        kvp += 4096;
        fkv.max_visible_keys = static_cast<std::uint32_t>(context);
        std::optional<ex::ForwardMtp> drafter;
        for (std::uint32_t l = 0; l <= c.attention_layers; ++l) {
            PagedKVLayerView view;
            view.k_pages       = Tensor(kvp, DType::I8, {D, kPagedKVPageSize, Hkv, pages});
            view.v_pages       = Tensor(kvp + codes, DType::I8, {D, kPagedKVPageSize, Hkv, pages});
            view.k_scale_pages = Tensor(kvp + 2 * codes, DType::FP16, {D / 64, kPagedKVPageSize, Hkv, pages});
            view.v_scale_pages = Tensor(kvp + 2 * codes + scales, DType::FP16, {D / 64, kPagedKVPageSize, Hkv, pages});
            view.block_table   = fkv.block_table;
            view.head_dim      = D;
            view.num_kv_heads  = Hkv;
            view.storage       = KvCacheStorage::Int8Group64;
            const Tensor pool(kvp + 2 * codes + 2 * scales, DType::BF16, {di / r, kPagedKVPageSize, 1, pages});
            kvp += layer_bytes;
            if (l < c.attention_layers) {
                fkv.layers.push_back(view);
                fkv.pooled.push_back(pool);
            } else {
                drafter.emplace(ex::ForwardMtp{.kv = view, .pooled = pool, .tail = mtp_tail, .ones = Tensor{}});
            }
        }
        // The verification records (widest W = 8), PLE inputs, raw index keys, tail copies, the
        // convolution's discarded window, the slot word, the drafter's ones and residual buffers.
        constexpr std::int32_t kWidth = 8;
        LayoutBuilder builder;
        const auto layout = plan_gdn_replay_records(builder, records_spec(kWidth));
        records_bytes     = align256(layout.payload_bytes());
        const std::size_t w = c.residual_width();
        spec = DeviceBuffer(records_bytes + align256(w * kWidth * 2) + align256(std::size_t(di) * kWidth * c.attention_layers * 2) +
                            c.attention_layers * align256(tail_bytes) + align256(conv_bytes) + 256 +
                            align256(std::size_t(c.hc.streams) * ex::kMtpChunkColumns * 4) +
                            align256(w * std::size_t(columns) * 2) + 4096);
        spec.fill(0);
        auto* s = static_cast<std::byte*>(spec.p);
        std::size_t at = records_bytes;
        ple_inputs = s + at;
        at += align256(w * kWidth * 2);
        qsa_keys = s + at;
        at += align256(std::size_t(di) * kWidth * c.attention_layers * 2);
        tail_copies = s + at;
        at += c.attention_layers * align256(tail_bytes);
        conv_scratch = s + at;
        at += align256(conv_bytes);
        slot = s + at;
        at += 256;
        std::vector<float> ones(static_cast<std::size_t>(c.hc.streams) * ex::kMtpChunkColumns, 1.0F);
        spec.copy_from_host(ones.data(), ones.size() * 4, at);
        drafter->ones = Tensor(s + at, DType::FP32, {dim(c.hc.streams), ex::kMtpChunkColumns});
        at += align256(ones.size() * 4);
        residuals = Tensor(s + at, DType::BF16, {dim(w), columns});

        frames = DeviceBuffer(sizeof(std::int32_t) * c.moe.experts * c.num_hidden_layers);
        frames.fill(0xFF); // every text expert from the pinned bank (placement does not change bits)
        ex::ForwardExperts experts;
        for (std::uint32_t l = 0; l < c.num_hidden_layers; ++l) {
            experts.frames.push_back(static_cast<const std::int32_t*>(frames.p) + l * c.moe.experts);
        }
        std::uint64_t stride = 0;
        for (const auto& bank : parameters.model.expert_banks()) { stride = std::max(stride, bank.record_stride); }
        staging                = DeviceBuffer(64 * stride);
        experts.staging_base   = static_cast<std::uint8_t*>(staging.p);
        experts.staging_slots  = 64;
        const auto keys        = static_cast<std::uint32_t>(context);
        work = std::make_unique<WorkspaceArena>(
            std::max(ex::Forward::workspace_bytes(c, columns, keys, device.execution_view()),
                     ex::Forward::mtp_workspace_bytes(c, std::min(columns, ex::kMtpChunkColumns), keys,
                                                      device.execution_view())) +
            w * ex::kMtpChunkColumns * 6 +
            (options.shared_expert_stream ? ex::Forward::shared_expert_workspace_bytes(c, columns) : 0));
        forward = std::make_unique<ex::Forward>(parameters, device, *work, std::move(fs), std::move(fkv),
                                                std::move(experts), std::move(drafter), options);
    }

    GdnReplayRecordSpec records_spec(std::int32_t width) const {
        return {.layers          = dim(c.gdn_layers),
                .record_capacity = 1,
                .width           = width,
                .conv_channels   = dim(c.gdn.conv_channels()),
                .qk_heads        = dim(c.gdn.value_heads),
                .value_heads     = dim(c.gdn.value_heads),
                .key_dim         = dim(c.gdn.key_head_dim),
                .value_dim       = dim(c.gdn.value_head_dim)};
    }

    GdnReplayRecords records(std::int32_t width) const {
        LayoutBuilder builder;
        return GdnReplayRecords(DeviceSpan{spec.p, records_bytes}, plan_gdn_replay_records(builder, records_spec(width)));
    }

    ex::ForwardVerify verify_view(std::int32_t W) const {
        ex::ForwardVerify v;
        v.gdn        = records(W);
        v.ple_inputs = Tensor(ple_inputs, DType::BF16, {dim(c.residual_width()), W});
        v.qsa_keys   = Tensor(qsa_keys, DType::BF16, {di, W, dim(c.attention_layers)});
        for (std::uint32_t l = 0; l < c.attention_layers; ++l) {
            v.qsa_tails.emplace_back(static_cast<std::byte*>(tail_copies) + align256(tail_bytes) * l, DType::BF16,
                                     std::initializer_list<std::int32_t>{di, r - 1});
        }
        v.conv_scratch = Tensor(conv_scratch, DType::BF16, {dim(c.gdn.conv_channels()), dim(c.gdn.conv_kernel - 1)});
        v.initial_slot = Tensor(slot, DType::I32, {1});
        return v;
    }

    LinearAttentionStateAllLayersView state_view() const {
        LinearAttentionStateAllLayersView view;
        view.conv_layer0      = Tensor(state.p, DType::BF16, {dim(c.gdn.conv_channels()), dim(c.gdn.conv_kernel - 1), 1});
        view.recurrent_layer0 = Tensor(static_cast<std::byte*>(state.p) + align256(conv_bytes), DType::FP32,
                                       {dim(c.gdn.key_head_dim), dim(c.gdn.value_head_dim), dim(c.gdn.value_heads), 1});
        view.conv_layer_stride_bytes      = static_cast<std::int64_t>(layer_stride);
        view.recurrent_layer_stride_bytes = static_cast<std::int64_t>(layer_stride);
        view.spec = LinearAttentionStatePoolSpec{.layers         = c.gdn_layers,
                                                 .conv_channels  = dim(c.gdn.conv_channels()),
                                                 .conv_width     = dim(c.gdn.conv_kernel - 1),
                                                 .value_heads    = dim(c.gdn.value_heads),
                                                 .value_head_dim = dim(c.gdn.value_head_dim),
                                                 .key_head_dim   = dim(c.gdn.key_head_dim),
                                                 .slot_count     = 1};
        return view;
    }

    const q4::TextConfig& c;
    std::size_t conv_bytes = 0, recurrent_bytes = 0, layer_stride = 0, tail_bytes = 0, state_bytes = 0, kv_bytes = 0;
    std::size_t records_bytes = 0;
    std::int32_t di = 0, r = 0;
    DeviceBuffer state, kv, spec, frames, staging;
    Tensor ple_state, ple_slot, mtp_tail, residuals;
    void *tails = nullptr, *ple_inputs = nullptr, *qsa_keys = nullptr, *tail_copies = nullptr, *conv_scratch = nullptr,
         *slot = nullptr;
    std::unique_ptr<WorkspaceArena> work;
    std::unique_ptr<ex::Forward> forward;
};

// Where two state images first differ, and how many bytes differ per region.
void describe_difference(const Harness& h, const std::vector<std::uint8_t>& a, const std::vector<std::uint8_t>& b) {
    std::size_t gdn_conv = 0, gdn_recurrent = 0, other = 0, first = a.size();
    const std::size_t gdn_end = h.c.gdn_layers * h.layer_stride;
    for (std::size_t i = 0; i < a.size(); ++i) {
        if (a[i] == b[i]) { continue; }
        first = std::min(first, i);
        if (i < gdn_end) {
            ((i % h.layer_stride) < align256(h.conv_bytes) ? gdn_conv : gdn_recurrent) += 1;
        } else {
            ++other;
        }
    }
    if (first == a.size()) { return; }
    std::printf("  state differs: first byte %zu (GDN layer %zu), GDN conv %zu B, GDN recurrent %zu B, "
                "PLE/tails %zu B (PLE ends at %zu)\n",
                first, first / h.layer_stride, gdn_conv, gdn_recurrent, other,
                gdn_end + align256(std::size_t(h.c.residual_width()) * h.c.ple.conv_span() * 2));
}

// Device inputs of one call over positions [first, first + width) of `tokens`, every column's
// logits when `every`.
struct Call {
    DeviceBuffer buffer;
    ex::ForwardBatch batch;
};

Call make_call(const q4::TextConfig& c, const q4::NgramVolume& volume, const std::vector<std::int32_t>& tokens,
               std::int32_t first, std::int32_t width, bool every) {
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
    std::vector<std::int32_t> rope(3 * static_cast<std::size_t>(width));
    for (int a = 0; a < 3; ++a) {
        for (std::int32_t t = 0; t < width; ++t) { rope[a * width + t] = first + t; }
    }
    const std::int32_t block = first / dim(c.qsa.compress_ratio) * dim(c.qsa.compress_ratio);
    const std::vector<std::int32_t> block_rope{block, block, block};
    std::vector<std::int32_t> columns;
    if (every) {
        columns.resize(width);
        std::iota(columns.begin(), columns.end(), 0);
    } else {
        columns.push_back(width - 1);
    }
    Call call;
    const std::size_t i32 = sizeof(std::int32_t);
    call.buffer = DeviceBuffer(align256(ids.size() * i32) * 2 + align256(rope.size() * i32) + align256(3 * i32) +
                               align256(columns.size() * i32) + align256(ngram.size()));
    auto* base         = static_cast<std::byte*>(call.buffer.p);
    std::size_t offset = 0;
    auto put           = [&](const void* data, std::size_t n) {
        call.buffer.copy_from_host(data, n, offset);
        void* p = base + offset;
        offset += align256(n);
        return p;
    };
    call.batch.ids              = Tensor(put(ids.data(), ids.size() * i32), DType::I32, {width});
    call.batch.positions        = Tensor(put(positions.data(), positions.size() * i32), DType::I32, {width});
    call.batch.rope_positions   = Tensor(put(rope.data(), rope.size() * i32), DType::I32, {width, 3});
    call.batch.block_start_rope = Tensor(put(block_rope.data(), 3 * i32), DType::I32, {3});
    call.batch.logit_columns    = Tensor(put(columns.data(), columns.size() * i32), DType::I32, {dim(columns.size())});
    call.batch.ngram_rows       = Tensor(put(ngram.data(), ngram.size()), DType::U8,
                                         {dim(row_bytes), dim(heads), width});
    return call;
}

// A drafter call's device inputs: cells [first, first + n) of `tokens` (cell c's next token is
// tokens[c + 1]).
struct Cells {
    DeviceBuffer buffer;
    Tensor ids, positions, rope, block;
};

Cells make_cells(const q4::TextConfig& c, const std::vector<std::int32_t>& tokens, std::int32_t first, std::int32_t n) {
    std::vector<std::int32_t> words(static_cast<std::size_t>(n) * 5 + 3);
    for (std::int32_t i = 0; i < n; ++i) {
        words[i]     = tokens[static_cast<std::size_t>(first + i + 1)];
        words[n + i] = first + i;
        for (int a = 0; a < 3; ++a) { words[2 * n + a * n + i] = first + i; }
    }
    const std::int32_t block = first / dim(c.qsa.compress_ratio) * dim(c.qsa.compress_ratio);
    for (int a = 0; a < 3; ++a) { words[5 * n + a] = block; }
    Cells out;
    out.buffer = DeviceBuffer(words.size() * 4);
    out.buffer.copy_from_host(words.data(), words.size() * 4);
    auto* p       = static_cast<std::int32_t*>(out.buffer.p);
    out.ids       = Tensor(p, DType::I32, {n});
    out.positions = Tensor(p + n, DType::I32, {n});
    out.rope      = Tensor(p + 2 * n, DType::I32, {n, 3});
    out.block     = Tensor(p + 5 * n, DType::I32, {3});
    return out;
}

// ------------------------------------------------------------------------------- 1. verification

void test_verification(Harness& h, DeviceContext& device, const q4::NgramVolume& volume,
                       const std::vector<std::int32_t>& tokens, std::int32_t prompt) {
    const auto& c        = h.c;
    const std::size_t V  = c.vocab_size, Wr = c.residual_width();
    const cudaStream_t s = device.stream;
    DeviceBuffer logits(V * 8 * sizeof(float));
    // The prompt as one call.
    {
        auto call = make_call(c, volume, tokens, 0, prompt, false);
        Tensor out(logits.p, DType::FP32, {dim(V), 1});
        h.forward->run(call.batch, out);
        device.synchronize();
    }
    DeviceBuffer state0(h.state_bytes), kv0(h.kv_bytes);
    CUDA_CHECK(cudaMemcpy(state0.p, h.state.p, h.state_bytes, cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(kv0.p, h.kv.p, h.kv_bytes, cudaMemcpyDeviceToDevice));
    const auto restore = [&](bool with_kv) {
        CUDA_CHECK(cudaMemcpy(h.state.p, state0.p, h.state_bytes, cudaMemcpyDeviceToDevice));
        if (with_kv) { CUDA_CHECK(cudaMemcpy(h.kv.p, kv0.p, h.kv_bytes, cudaMemcpyDeviceToDevice)); }
    };
    constexpr std::int32_t kMax = 8;
    // One-column decode calls: each position's logits, residual and the state after it.
    std::vector<std::vector<float>> seq_logits;
    std::vector<std::vector<std::uint16_t>> seq_residual;
    std::vector<std::vector<std::uint8_t>> seq_state;
    for (std::int32_t j = 0; j < kMax; ++j) {
        auto call               = make_call(c, volume, tokens, prompt + j, 1, false);
        call.batch.residual_out = h.residuals.slice(1, 0, 1);
        Tensor out(logits.p, DType::FP32, {dim(V), 1});
        h.forward->run(call.batch, out);
        device.synchronize();
        seq_logits.push_back(download<float>(logits.p, V));
        seq_residual.push_back(download<std::uint16_t>(h.residuals.data, Wr));
        seq_state.push_back(download<std::uint8_t>(h.state.p, h.state_bytes));
    }
    const auto state_bytes0 = download<std::uint8_t>(state0.p, h.state_bytes);
    for (const std::int32_t W : {2, 3, 5, 8, 5, 7}) {
        restore(true);
        const ex::ForwardVerify view = h.verify_view(W);
        auto call                    = make_call(c, volume, tokens, prompt, W, true);
        call.batch.verify            = &view;
        call.batch.residual_out      = h.residuals.slice(1, 0, W);
        CUDA_CHECK(cudaMemcpyAsync(h.tail_copies, h.tails, c.attention_layers * align256(h.tail_bytes),
                                   cudaMemcpyDeviceToDevice, s));
        Tensor out(logits.p, DType::FP32, {dim(V), W});
        h.forward->run(call.batch, out);
        device.synchronize();
        const auto got_logits   = download<float>(logits.p, V * W);
        const auto got_residual = download<std::uint16_t>(h.residuals.data, Wr * W);
        bool logits_equal = true, residual_equal = true;
        for (std::int32_t j = 0; j < W; ++j) {
            logits_equal = logits_equal && std::memcmp(got_logits.data() + V * j, seq_logits[j].data(), V * 4) == 0;
            residual_equal =
                residual_equal && std::memcmp(got_residual.data() + Wr * j, seq_residual[j].data(), Wr * 2) == 0;
        }
        expect(logits_equal, "W=" + std::to_string(W) + ": every column's logits equal the one-column decode's");
        expect(residual_equal, "W=" + std::to_string(W) + ": every column's final residual equals the decode's");
        expect(download<std::uint8_t>(h.state.p, h.state_bytes) == state_bytes0,
               "W=" + std::to_string(W) + ": the verification leaves the recurrent state, PLE history and tails unchanged");
        // Commits of k columns from the unchanged state and the records.
        const ops::GdnReplayFoldPlan fold(view.gdn, h.state_view());
        for (const std::int32_t k : {1, std::max(1, W / 2), W}) {
            restore(false);
            const std::vector<std::int32_t> commit_words(1, k);
            DeviceBuffer commit(4);
            commit.copy_from_host(commit_words.data(), 4);
            const ops::GdnReplayFoldRow row{0, 0, k};
            fold.execute(std::span<const ops::GdnReplayFoldRow>(&row, 1), s);
            Tensor tail0(h.tails, DType::BF16, {h.di, h.r - 1});
            ops::qsa_commit_tails(Tensor(h.qsa_keys, DType::BF16, {h.di, W, dim(c.attention_layers)}), call.batch.positions,
                                  Tensor(commit.p, DType::I32, {1}), tail0,
                                  static_cast<std::int64_t>(align256(h.tail_bytes)), s);
            Tensor ple_state = h.ple_state;
            ops::ple_conv_commit(Tensor(h.ple_inputs, DType::BF16, {dim(Wr), W, 1}), Tensor(commit.p, DType::I32, {1}),
                                 ple_state, h.ple_slot, s);
            device.synchronize();
            const auto got_state = download<std::uint8_t>(h.state.p, h.state_bytes);
            describe_difference(h, got_state, seq_state[static_cast<std::size_t>(k - 1)]);
            expect(got_state == seq_state[static_cast<std::size_t>(k - 1)],
                   "W=" + std::to_string(W) + " k=" + std::to_string(k) +
                       ": the commit leaves the state of k decode calls (GDN windows and states, PLE history, tails)");
        }
    }
    restore(true);
}

// ----------------------------------------------------------------------- 2. the drafter's oracles

// fl32 RMSNorm with a unit-offset BF16 gain, rounded to BF16 (rmsnorm's output boundary).
std::vector<double> rmsnorm_unit(const std::vector<double>& x, const std::vector<std::uint16_t>& gamma, double eps) {
    double squares = 0;
    for (const double v : x) { squares += v * v; }
    const double inv = 1.0 / std::sqrt(squares / static_cast<double>(x.size()) + eps);
    std::vector<double> out(x.size());
    for (std::size_t i = 0; i < x.size(); ++i) { out[i] = round_bf16(x[i] * inv * (1.0 + bf16(gamma[i]))); }
    return out;
}

// out[n] = sum_k w[n][k] x[k] over BF16 rows [N][K].
std::vector<double> rows_dot(const std::vector<std::uint16_t>& w, std::size_t N, const std::vector<double>& x) {
    const std::size_t K = x.size();
    std::vector<double> out(N);
    for (std::size_t n = 0; n < N; ++n) {
        double sum = 0;
        for (std::size_t k = 0; k < K; ++k) { sum += bf16(w[n * K + k]) * x[k]; }
        out[n] = sum;
    }
    return out;
}

std::vector<double> to_double(const std::vector<std::uint16_t>& v) {
    std::vector<double> out(v.size());
    for (std::size_t i = 0; i < v.size(); ++i) { out[i] = bf16(v[i]); }
    return out;
}

std::vector<std::uint16_t> tensor_words(const Tensor& t) { return download<std::uint16_t>(t.data, t.numel()); }
std::vector<std::uint16_t> weight_words(const Weight& w) {
    return download<std::uint16_t>(w.qdata != nullptr ? w.qdata : w.payload,
                                   static_cast<std::size_t>(w.n) * static_cast<std::size_t>(w.k));
}

void test_drafter(Harness& h, DeviceContext& device, const ex::Parameters& parameters, const q4::NgramVolume& volume,
                  const std::vector<std::int32_t>& tokens, std::int32_t prompt) {
    const auto& c        = h.c;
    const auto& p        = *parameters.mtp;
    const std::size_t V  = c.vocab_size, H = c.hidden_size, Wr = c.residual_width(), S = c.hc.streams;
    const double eps     = c.rms_norm_eps;
    // The prompt with its residuals exported, then the drafter's cells of the prompt (cell c pairs
    // residual c with token c + 1), then one full step at the last prompt position with the token
    // after it.
    DeviceBuffer residuals(Wr * prompt * 2);
    {
        auto call               = make_call(c, volume, tokens, 0, prompt, false);
        call.batch.residual_out = Tensor(residuals.p, DType::BF16, {dim(Wr), prompt});
        DeviceBuffer logits(V * sizeof(float));
        Tensor out(logits.p, DType::FP32, {dim(V), 1});
        h.forward->run(call.batch, out);
    }
    for (std::int32_t first = 0; first < prompt - 1; first += ex::kMtpChunkColumns) {
        const std::int32_t n = std::min(ex::kMtpChunkColumns, prompt - 1 - first);
        const Cells cells    = make_cells(c, tokens, first, n);
        ex::MtpCall call;
        call.residuals        = Tensor(static_cast<std::byte*>(residuals.p) + Wr * 2 * first, DType::BF16, {dim(Wr), n});
        call.ids              = cells.ids;
        call.positions        = cells.positions;
        call.rope_positions   = cells.rope;
        call.block_start_rope = cells.block;
        call.tail             = h.mtp_tail;
        h.forward->run_mtp(call);
    }
    const std::int32_t cell = prompt - 1;
    const Cells step        = make_cells(c, tokens, cell, 1);
    DeviceBuffer taps(Wr * 2 * 2 + H * 2 * 6 + 64 + V * 4 + 4096);
    auto* t = static_cast<std::byte*>(taps.p);
    ex::MtpCall::Tap tap;
    std::size_t at = 0;
    const auto carve = [&](DType dtype, std::initializer_list<std::int32_t> shape) {
        Tensor out(t + at, dtype, shape);
        at += align256(out.bytes());
        return out;
    };
    tap.fused     = carve(DType::BF16, {dim(Wr), 1});
    tap.mixer_in  = carve(DType::BF16, {dim(H), 1});
    tap.attention = carve(DType::BF16, {dim(H), 1});
    tap.moe_in    = carve(DType::BF16, {dim(H), 1});
    tap.routes    = carve(DType::I32, {dim(c.moe.top_k), 1});
    tap.moe_out   = carve(DType::BF16, {dim(H), 1});
    tap.final_x   = carve(DType::BF16, {dim(H), 1});
    tap.logits    = carve(DType::FP32, {dim(V), 1});
    const Tensor out_residual = carve(DType::BF16, {dim(Wr), 1});
    const Tensor draft        = carve(DType::I32, {1});
    ex::MtpCall call;
    call.residuals        = Tensor(static_cast<std::byte*>(residuals.p) + Wr * 2 * cell, DType::BF16, {dim(Wr), 1});
    call.ids              = step.ids;
    call.positions        = step.positions;
    call.rope_positions   = step.rope;
    call.block_start_rope = step.block;
    call.tail             = h.mtp_tail;
    call.kv_only          = false;
    call.residual_out     = out_residual;
    call.drafts           = draft;
    call.tap              = &tap;
    h.forward->run_mtp(call);
    device.synchronize();

    // (a) The input fusion from the step's residual and token.
    {
        const auto R     = to_double(download<std::uint16_t>(call.residuals.data, Wr));
        const auto fused = to_double(tensor_words(tap.fused));
        const auto id    = tokens[static_cast<std::size_t>(cell + 1)];
        const Weight& emb = parameters.token_embedding;
        const std::size_t row_bytes = static_cast<std::size_t>(emb.k) / 256 * 136;
        const auto row   = download<std::byte>(static_cast<const std::byte*>(emb.qdata) + row_bytes * id, row_bytes);
        std::vector<float> decoded(H);
        test::ggml::decode_blocks(QType::GGML_IQ4_XS, row, decoded);
        std::vector<double> x0(H);
        for (std::size_t i = 0; i < H; ++i) { x0[i] = round_bf16(decoded[i]); }
        const auto e  = rmsnorm_unit(x0, tensor_words(p.embedding_norm), eps);
        auto ep       = rows_dot(weight_words(p.embedding_projection.weight), H, e);
        for (auto& v : ep) { v = round_bf16(v); }
        const auto hn = rmsnorm_unit(R, tensor_words(p.hidden_norm), eps);
        const auto fc = weight_words(p.hidden_projection.weight);
        std::vector<double> want(Wr);
        for (std::size_t sidx = 0; sidx < S; ++sidx) {
            const std::vector<double> stream(hn.begin() + sidx * H, hn.begin() + (sidx + 1) * H);
            const auto projected = rows_dot(fc, H, stream);
            for (std::size_t d = 0; d < H; ++d) { want[sidx * H + d] = round_bf16(round_bf16(projected[d]) + ep[d]); }
        }
        const Error err = error_of(fused, want);
        std::printf("fusion against FP64: normwise %.3e, worst %.3e\n", err.normwise, err.worst);
        expect(err.normwise <= 0x1p-7 && err.worst <= 0x1p-4, "the drafter's input fusion is within BF16 rounding of FP64");
    }
    // (b) The MoE from its tapped input.
    {
        const auto xm_words = tensor_words(tap.moe_in);
        const auto xm       = to_double(xm_words);
        const std::size_t E = c.moe.experts, K = c.moe.top_k, I = c.moe.shared_intermediate;
        auto logits         = rows_dot(tensor_words(p.router), E, xm);
        const double score  = rows_dot(tensor_words(p.shared_score), 1, xm)[0];
        std::vector<int> order(E);
        std::iota(order.begin(), order.end(), 0);
        std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return logits[a] > logits[b]; });
        const auto routes = download<std::int32_t>(tap.routes.data, K);
        std::vector<int> got(routes.begin(), routes.end()), want(order.begin(), order.begin() + K);
        std::vector<int> got_sorted = got, want_sorted = want;
        std::sort(got_sorted.begin(), got_sorted.end());
        std::sort(want_sorted.begin(), want_sorted.end());
        const double margin = logits[order[K - 1]] - logits[order[K]];
        std::printf("router: top-%zu margin %.3e\n", K, margin);
        expect(got_sorted == want_sorted || margin < 1e-4, "the drafter's routed experts are the FP64 router's top 10");
        // Weights over the GPU's selection (so a near-tie swap does not fail the rest).
        double sum = 0;
        std::vector<double> w(K);
        for (std::size_t i = 0; i < K; ++i) { sum += w[i] = std::exp(logits[got[i]] - logits[got[0]]); }
        const auto& bank = *p.bank;
        std::vector<double> want_y(H, 0.0);
        for (std::size_t i = 0; i < K; ++i) {
            const auto record = download<std::uint8_t>(bank.device_records + bank.record_stride * got[i],
                                                       ops::offloaded_moe::record_geometry(bank.format).bytes);
            const auto decoded = test::moe::decode_record(bank.format, record.data());
            const auto y       = test::moe::expert_oracle(decoded, xm_words.data());
            for (std::size_t d = 0; d < H; ++d) { want_y[d] += (w[i] / sum) * round_bf16(y[d]); }
        }
        auto g = rows_dot(tensor_words(p.shared_gate), I, xm);
        auto u = rows_dot(tensor_words(p.shared_up), I, xm);
        std::vector<double> product(I);
        for (std::size_t j = 0; j < I; ++j) {
            const double gj = round_bf16(g[j]), uj = round_bf16(u[j]);
            product[j]      = round_bf16(gj / (1.0 + std::exp(-gj)) * uj);
        }
        const auto shared = rows_dot(tensor_words(p.shared_down), H, product);
        const double gate = 1.0 / (1.0 + std::exp(-score));
        for (std::size_t d = 0; d < H; ++d) { want_y[d] = round_bf16(want_y[d] + gate * round_bf16(shared[d])); }
        const Error err = error_of(to_double(tensor_words(tap.moe_out)), want_y);
        std::printf("MoE against FP64: normwise %.3e, worst %.3e\n", err.normwise, err.worst);
        expect(err.normwise <= 0x1p-7 && err.worst <= 0x1p-4,
               "the drafter's MoE (Q2_0 experts, shared expert, combine) is within its A8 and BF16 boundaries of FP64");
    }
    // (c) The draft head from the final mixer's output.
    {
        const auto x_words = tensor_words(tap.final_x);
        std::vector<float> xf(H);
        for (std::size_t i = 0; i < H; ++i) { xf[i] = static_cast<float>(bf16(x_words[i])); }
        std::vector<double> xq(H);
        for (std::size_t g = 0; g < H / 32; ++g) {
            std::int8_t q[32];
            const double d = test::moe::a8_cast(xf.data() + 32 * g, 32, q);
            for (int j = 0; j < 32; ++j) { xq[32 * g + j] = d * q[j]; }
        }
        const Weight& head          = parameters.output_head;
        const std::size_t row_bytes = H / 256 * 136;
        const auto rows             = download<std::byte>(head.qdata, row_bytes * V);
        std::vector<double> want(V);
        std::vector<float> decoded(H);
        for (std::size_t v = 0; v < V; ++v) {
            test::ggml::decode_blocks(QType::GGML_IQ4_XS, std::span<const std::byte>(rows.data() + row_bytes * v, row_bytes),
                                decoded);
            double sum = 0;
            for (std::size_t k = 0; k < H; ++k) { sum += decoded[k] * xq[k]; }
            want[v] = sum;
        }
        const auto got_f = download<float>(tap.logits.data, V);
        std::vector<double> got(got_f.begin(), got_f.end());
        const Error err = error_of(got, want);
        const auto domain = static_cast<std::size_t>(parameters.model.resources().public_token_count);
        const auto best   = static_cast<std::int32_t>(std::max_element(want.begin(), want.begin() + domain) - want.begin());
        std::vector<double> sorted(want.begin(), want.begin() + domain);
        std::partial_sort(sorted.begin(), sorted.begin() + 2, sorted.end(), std::greater<>());
        const auto drafted = download<std::int32_t>(draft.data, 1)[0];
        std::printf("head against FP64: normwise %.3e, worst %.3e; FP64 argmax %d (margin %.3e), draft %d\n",
                    err.normwise, err.worst, best, sorted[0] - sorted[1], drafted);
        expect(err.normwise <= 0x1p-12, "the draft head's FP32 logits are within FP32 summation error of FP64");
        expect(drafted == best || sorted[0] - sorted[1] < 0x1p-6 * std::fabs(sorted[0]),
               "the draft is the FP64 head's argmax (or a BF16-rounding tie)");
    }
}

} // namespace

// ------------------------------------------------------------------ 3. the shared expert stream
// The prompt and one full drafter step after it (as test_drafter runs them), for comparing two
// Forwards: the prompt's last logits, and the step's MoE output, head logits, output residual and
// draft.
struct DrafterRun {
    std::vector<float> prompt_logits, logits;
    std::vector<std::uint16_t> moe_out, residual;
    std::int32_t draft = -1;
};

DrafterRun run_drafter(Harness& h, DeviceContext& device, const q4::NgramVolume& volume,
                       const std::vector<std::int32_t>& tokens, std::int32_t prompt) {
    const auto& c       = h.c;
    const std::size_t V = c.vocab_size, H = c.hidden_size, Wr = c.residual_width();
    DrafterRun out;
    DeviceBuffer residuals(Wr * prompt * 2);
    {
        auto call               = make_call(c, volume, tokens, 0, prompt, false);
        call.batch.residual_out = Tensor(residuals.p, DType::BF16, {dim(Wr), prompt});
        DeviceBuffer logits(V * sizeof(float));
        Tensor logits_out(logits.p, DType::FP32, {dim(V), 1});
        h.forward->run(call.batch, logits_out);
        device.synchronize();
        out.prompt_logits = download<float>(logits.p, V);
    }
    for (std::int32_t first = 0; first < prompt - 1; first += ex::kMtpChunkColumns) {
        const std::int32_t n = std::min(ex::kMtpChunkColumns, prompt - 1 - first);
        const Cells cells    = make_cells(c, tokens, first, n);
        ex::MtpCall call;
        call.residuals        = Tensor(static_cast<std::byte*>(residuals.p) + Wr * 2 * first, DType::BF16, {dim(Wr), n});
        call.ids              = cells.ids;
        call.positions        = cells.positions;
        call.rope_positions   = cells.rope;
        call.block_start_rope = cells.block;
        call.tail             = h.mtp_tail;
        h.forward->run_mtp(call);
    }
    const std::int32_t cell = prompt - 1;
    const Cells step        = make_cells(c, tokens, cell, 1);
    DeviceBuffer taps(H * 2 + V * 4 + Wr * 2 + 4 + 4096);
    std::size_t at   = 0;
    const auto carve = [&](DType dtype, std::initializer_list<std::int32_t> shape) {
        Tensor t(static_cast<std::byte*>(taps.p) + at, dtype, shape);
        at += align256(t.bytes());
        return t;
    };
    ex::MtpCall::Tap tap;
    tap.moe_out                   = carve(DType::BF16, {dim(H), 1});
    tap.logits                    = carve(DType::FP32, {dim(V), 1});
    const Tensor residual         = carve(DType::BF16, {dim(Wr), 1});
    const Tensor draft            = carve(DType::I32, {1});
    ex::MtpCall call;
    call.residuals        = Tensor(static_cast<std::byte*>(residuals.p) + Wr * 2 * cell, DType::BF16, {dim(Wr), 1});
    call.ids              = step.ids;
    call.positions        = step.positions;
    call.rope_positions   = step.rope;
    call.block_start_rope = step.block;
    call.tail             = h.mtp_tail;
    call.kv_only          = false;
    call.residual_out     = residual;
    call.drafts           = draft;
    call.tap              = &tap;
    h.forward->run_mtp(call);
    device.synchronize();
    out.moe_out  = download<std::uint16_t>(tap.moe_out.data, H);
    out.logits   = download<float>(tap.logits.data, V);
    out.residual = download<std::uint16_t>(residual.data, Wr);
    out.draft    = download<std::int32_t>(draft.data, 1).front();
    return out;
}

template <class T>
bool bitwise_equal(const std::vector<T>& a, const std::vector<T>& b) {
    return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(T)) == 0;
}

// ForwardOptions::shared_expert_stream runs the shared expert of the text model's MoE blocks and of
// the drafter's on a side stream: every output equals the serial run's, bit for bit.
void test_shared_expert_stream(const ex::Parameters& parameters, DeviceContext& device, const q4::NgramVolume& volume,
                               const std::vector<std::int32_t>& tokens, std::int32_t prompt, std::int32_t context) {
    DrafterRun serial, side;
    {
        Harness h(parameters, device, context, prompt);
        serial = run_drafter(h, device, volume, tokens, prompt);
    }
    {
        Harness h(parameters, device, context, prompt, ex::ForwardOptions{.shared_expert_stream = true});
        side = run_drafter(h, device, volume, tokens, prompt);
    }
    expect(bitwise_equal(serial.prompt_logits, side.prompt_logits),
           "shared expert stream: the prompt's logits are bitwise the serial run's");
    expect(bitwise_equal(serial.moe_out, side.moe_out) && bitwise_equal(serial.logits, side.logits) &&
               bitwise_equal(serial.residual, side.residual) && serial.draft == side.draft,
           "shared expert stream: the drafter step's MoE output, logits, residual and draft are bitwise the serial run's");
}

int main(int argc, char** argv) {
    const char* artifact_env = std::getenv("NINFER_TEST_QWEN4EXP_ARTIFACT");
    if (artifact_env == nullptr) {
        std::printf("SKIP: set NINFER_TEST_QWEN4EXP_ARTIFACT to a Qwen3.8-Flash-Next artifact with its MTP drafter\n");
        return 77;
    }
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    try {
        std::string tokens_arg = "@" + std::string(NINFER_SOURCE_DIR) + "/tests/fixtures/qwen4_exp/ppl_tokens-code.ids";
        std::int32_t prompt    = 300;
        for (int i = 1; i < argc; ++i) {
            const std::string a = argv[i];
            if (a == "--prompt" && i + 1 < argc) {
                prompt = std::stoi(argv[++i]);
            } else {
                tokens_arg = a;
            }
        }
        const auto tokens = parse_tokens(tokens_arg);
        if (prompt < 2 || static_cast<std::int32_t>(tokens.size()) < prompt + 9) {
            throw std::invalid_argument("the token list needs the prompt plus 9 tokens");
        }
        DeviceContext device(0);
        artifact::Reader reader(artifact_env);
        if (!q4::is_qwen4_exp(reader) || !q4::has_mtp(reader)) {
            std::printf("SKIP: %s is not a Qwen4Exp artifact with its MTP drafter\n", artifact_env);
            return 77;
        }
        const auto model = q4::load_model(reader, device, /*mtp=*/true);
        device.synchronize();
        const ex::Parameters parameters(*model);
        const auto& c = model->config();
        const char* ngram_env = std::getenv("NINFER_TEST_QWEN4EXP_NGRAM");
        q4::NgramVolume volume(ngram_env != nullptr ? std::filesystem::path(ngram_env) : q4::default_ngram_volume(artifact_env),
                               c.ple.table);
        const std::int32_t context = (prompt + 64 + 63) / 64 * 64;
        {
            Harness h(parameters, device, context, prompt);
            test_verification(h, device, volume, tokens, prompt);
        }
        {
            Harness h(parameters, device, context, prompt);
            test_drafter(h, device, parameters, volume, tokens, prompt);
        }
        test_shared_expert_stream(parameters, device, volume, tokens, prompt, context);
        std::printf(g_failures == 0 ? "qwen4_exp mtp checks passed\n" : "FAIL: %d qwen4_exp mtp check(s)\n", g_failures);
        return g_failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL: %s\n", error.what());
        return 1;
    }
}
