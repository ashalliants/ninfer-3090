// Adapted from Infernix a3edb450 src/models/qwen4_exp/execution/forward.cpp (Apache-2.0).
// Modified for NInfer-3090: see forward.h. Per block the composition is Infernix's: PLE injection
// (its layer), attention-side mixer, GDN or QSA, injection, MLP-side mixer, MoE, injection. The
// GGUF's layouts change three steps: the attention query and gate rows interleave per head, the
// GDN value heads are in tiled order (value head h reads key head h % key_heads), and the indexer
// projections are BF16 rows projected in FP32.

#include "models/qwen4_exp/execution/forward.h"

#include "ninfer/ops/argmax.h"
#include "ninfer/ops/cast.h"
#include "ninfer/ops/causal_conv1d_silu.h"
#include "ninfer/ops/embedding.h"
#include "ninfer/ops/gated_delta_net.h"
#include "ninfer/ops/gated_rmsnorm.h"
#include "ninfer/ops/gdn_gating.h"
#include "ninfer/ops/hyper_connection.h"
#include "ninfer/ops/kv_cache_append.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/linear_pair.h"
#include "ninfer/ops/offloaded_sparse_moe.h"
#include "ninfer/ops/ple.h"
#include "ninfer/ops/projection_fp32.h"
#include "ninfer/ops/rmsnorm.h"
#include "ninfer/ops/rope.h"
#include "ninfer/ops/rows.h"
#include "ninfer/ops/scatter.h"
#include "ninfer/ops/sigmoid_mul.h"
#include "ninfer/ops/silu_mul.h"

#include <algorithm>
#include <cmath>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>

namespace ninfer::models::qwen4_exp::execution {
namespace {

std::int32_t dim(std::uint64_t v) { return static_cast<std::int32_t>(v); }

void project(const Tensor& x, const LinearParameters& p, Tensor& out, WorkspaceArena& work, cudaStream_t s) {
    ops::linear(x, p.weight, out, p.policy, work, s);
}

// Two projections of one input: one linear_pair launch for decode-width calls over GGML weights
// (the same outputs as two linears, bit for bit) when both policies admit A8 (the only GGML profile;
// linear() rejects A16Only for them, so the pair must not bypass that), else two linears.
void project_pair(const Tensor& x, const LinearParameters& a, const LinearParameters& b, Tensor& out_a, Tensor& out_b,
                  WorkspaceArena& work, cudaStream_t s) {
    if (x.ne[1] <= 8 && a.weight.layout == QuantLayout::GgmlBlocks && b.weight.layout == QuantLayout::GgmlBlocks &&
        ops::allows_a8(a.policy) && ops::allows_a8(b.policy)) {
        ops::linear_pair(x, a.weight, b.weight, out_a, out_b, s);
        return;
    }
    project(x, a, out_a, work, s);
    project(x, b, out_b, work, s);
}

void copy_tap(std::vector<Tensor>* targets, std::uint32_t layer, const Tensor& source, cudaStream_t s) {
    if (targets == nullptr) { return; }
    if (cudaMemcpyAsync(targets->at(layer).data, source.data, source.bytes(), cudaMemcpyDeviceToDevice, s) !=
        cudaSuccess) {
        throw std::runtime_error("Qwen4Exp forward: block tap copy failed");
    }
}

// q/k [D, Hk, T] -> [D, Hv, T] with out head h = in head h % Hk: the GGUF's tiled pairing expressed
// as the recurrence's own one-to-one pairing (the normalization is per head row, so it commutes).
void tile_heads(const Tensor& in, Tensor& out, cudaStream_t s) {
    const std::size_t row  = static_cast<std::size_t>(in.ne[0]) * sizeof(std::uint16_t);
    const std::size_t from = row * in.ne[1], to = row * out.ne[1];
    for (std::int32_t r = 0; r < out.ne[1] / in.ne[1]; ++r) {
        CUDA_CHECK(cudaMemcpy2DAsync(static_cast<std::byte*>(out.data) + r * from, to, in.data, from, from, in.ne[2],
                                     cudaMemcpyDeviceToDevice, s));
    }
}

} // namespace

ops::QsaIndexerGeometry qsa_geometry(const TextConfig& c) {
    return {.index_heads      = dim(c.qsa.index_heads),
            .index_head_dim   = dim(c.qsa.index_head_dim),
            .rotary_dim       = dim(c.rope.rotary_dim),
            .block_tokens     = dim(c.qsa.compress_ratio),
            .budget_tokens    = dim(c.qsa.budget),
            .theta            = c.rope.theta,
            .eps              = c.rms_norm_eps,
            .unit_offset_norm = true};
}

Forward::Forward(const Parameters& parameters, DeviceContext& device, WorkspaceArena& work, ForwardState state,
                 ForwardKV kv, ForwardExperts experts, std::optional<ForwardMtp> mtp)
    : parameters_(parameters), config_(parameters.model.config()), device_(device), work_(&work),
      state_(std::move(state)), kv_(std::move(kv)), experts_(std::move(experts)), mtp_(std::move(mtp)) {
    if (state_.gdn_conv.size() != config_.gdn_layers || state_.gdn_recurrent.size() != config_.gdn_layers ||
        state_.ple_slot.dtype != DType::I32 || state_.ple_slot.numel() != 1 ||
        state_.qsa_tails.size() != config_.attention_layers || kv_.layers.size() != config_.attention_layers ||
        kv_.pooled.size() != config_.attention_layers || experts_.frames.size() != config_.num_hidden_layers) {
        throw std::invalid_argument("Qwen4Exp forward: state, KV or expert residency is incomplete");
    }
    if (mtp_.has_value() != parameters_.mtp.has_value()) {
        throw std::invalid_argument("Qwen4Exp forward: the MTP drafter's parameters and state go together");
    }
    if (mtp_) {
        // Every drafter expert is resident: expert e's record is frame e of its device bank.
        const auto E = static_cast<std::int32_t>(config_.moe.experts);
        std::vector<std::int32_t> identity(static_cast<std::size_t>(E));
        for (std::int32_t e = 0; e < E; ++e) { identity[static_cast<std::size_t>(e)] = e; }
        mtp_frames_backing_ = DeviceBuffer(identity.size() * sizeof(std::int32_t));
        mtp_frames_backing_.copy_from_host(identity.data(), mtp_frames_backing_.bytes);
        mtp_frames_ = Tensor(mtp_frames_backing_.p, DType::I32, {E});
    }
}

Forward::Forward(const Parameters& parameters, DeviceContext& device, WorkspaceArena& work, ForwardRowsState state,
                 ForwardRowsKV kv, ForwardExperts experts, std::optional<ForwardRowsMtp> mtp)
    : parameters_(parameters), config_(parameters.model.config()), device_(device), work_(&work),
      experts_(std::move(experts)) {
    if (mtp.has_value() != parameters_.mtp.has_value()) {
        throw std::invalid_argument("Qwen4Exp forward: the MTP drafter's parameters and state go together");
    }
    const auto& c        = config_;
    const auto slots     = static_cast<std::int32_t>(state.slot_ids.numel());
    const std::int32_t C = dim(c.gdn.conv_channels()), K = dim(c.gdn.conv_kernel - 1);
    const std::int32_t Dk = dim(c.gdn.key_head_dim), Dv = dim(c.gdn.value_head_dim), Hv = dim(c.gdn.value_heads);
    const std::int32_t Di = dim(c.qsa.index_head_dim), R = dim(c.qsa.compress_ratio);
    const auto shaped = [](const Tensor& t, DType dtype, std::initializer_list<std::int32_t> shape) {
        if (t.data == nullptr || t.dtype != dtype || !t.is_contiguous()) { return false; }
        int axis = 0;
        for (const std::int32_t n : shape) {
            if (t.ne[axis++] != n) { return false; }
        }
        return true;
    };
    bool valid = slots >= 1 && shaped(state.slot_ids, DType::I32, {slots}) &&
                 state.gdn_conv.size() == c.gdn_layers && state.gdn_recurrent.size() == c.gdn_layers &&
                 state.qsa_tails.size() == c.attention_layers && kv.layers.size() == c.attention_layers &&
                 kv.pooled.size() == c.attention_layers && kv.block_tables.dtype == DType::I32 &&
                 kv.block_tables.is_contiguous() && kv.block_tables.ne[1] == slots && kv.block_tables.ne[2] == 1 &&
                 experts_.frames.size() == c.num_hidden_layers &&
                 shaped(state.ple_conv, DType::BF16, {dim(c.residual_width()), dim(c.ple.conv_span()), slots});
    for (std::uint32_t l = 0; valid && l < c.gdn_layers; ++l) {
        valid = shaped(state.gdn_conv[l], DType::BF16, {C, K, slots}) &&
                shaped(state.gdn_recurrent[l], DType::FP32, {Dk, Dv, Hv, slots});
    }
    for (std::uint32_t l = 0; valid && l < c.attention_layers; ++l) {
        valid = shaped(state.qsa_tails[l], DType::BF16, {Di, R - 1, slots});
    }
    if (!valid) { throw std::invalid_argument("Qwen4Exp forward: the slotted state, KV or expert residency is incomplete"); }
    Rows rows{std::move(state), std::move(kv), {}, {}};
    const std::int32_t pages = rows.kv.block_tables.ne[0];
    for (std::int32_t s = 0; s < slots; ++s) {
        ForwardState view;
        for (std::uint32_t l = 0; l < c.gdn_layers; ++l) {
            view.gdn_conv.push_back(rows.state.gdn_conv[l].slice(2, s, 1).view({C, K}));
            view.gdn_recurrent.push_back(rows.state.gdn_recurrent[l].slice(3, s, 1).view({Dk, Dv, Hv}));
        }
        view.ple_conv = rows.state.ple_conv;
        view.ple_slot = rows.state.slot_ids.slice(0, s, 1);
        for (std::uint32_t l = 0; l < c.attention_layers; ++l) {
            view.qsa_tails.push_back(rows.state.qsa_tails[l].slice(2, s, 1).view({Di, R - 1}));
        }
        ForwardKV table;
        table.block_table      = rows.kv.block_tables.slice(1, s, 1).view({pages});
        table.pooled           = rows.kv.pooled;
        table.max_visible_keys = rows.kv.max_visible_keys;
        for (PagedKVLayerView layer : rows.kv.layers) {
            layer.block_table = table.block_table;
            table.layers.push_back(layer);
        }
        if (mtp) {
            if (mtp->tails.data == nullptr || !shaped(mtp->tails, DType::BF16, {Di, R - 1, slots}) ||
                mtp->pooled.data == nullptr || mtp->ones.data == nullptr) {
                throw std::invalid_argument("Qwen4Exp forward: the slotted drafter state is incomplete");
            }
            ForwardMtp drafter;
            drafter.kv             = mtp->kv;
            drafter.kv.block_table = table.block_table;
            drafter.pooled         = mtp->pooled;
            drafter.tail           = mtp->tails.slice(2, s, 1).view({Di, R - 1});
            drafter.ones           = mtp->ones;
            rows.slot_mtp.push_back(std::move(drafter));
        }
        rows.slot_state.push_back(std::move(view));
        rows.slot_kv.push_back(std::move(table));
    }
    if (mtp) {
        // Every drafter expert is resident: expert e's record is frame e of its device bank.
        const auto E = static_cast<std::int32_t>(config_.moe.experts);
        std::vector<std::int32_t> identity(static_cast<std::size_t>(E));
        for (std::int32_t e = 0; e < E; ++e) { identity[static_cast<std::size_t>(e)] = e; }
        mtp_frames_backing_ = DeviceBuffer(identity.size() * sizeof(std::int32_t));
        mtp_frames_backing_.copy_from_host(identity.data(), mtp_frames_backing_.bytes);
        mtp_frames_ = Tensor(mtp_frames_backing_.p, DType::I32, {E});
    }
    rows_ = std::move(rows);
    select_row(0);
}

void Forward::select_row(std::int32_t slot) {
    if (!rows_ || slot < 0 || slot >= static_cast<std::int32_t>(rows_->slot_state.size())) {
        throw std::invalid_argument("Qwen4Exp forward: no such state slot");
    }
    state_ = rows_->slot_state[static_cast<std::size_t>(slot)];
    kv_    = rows_->slot_kv[static_cast<std::size_t>(slot)];
    if (!rows_->slot_mtp.empty()) { mtp_ = rows_->slot_mtp[static_cast<std::size_t>(slot)]; }
}

std::size_t Forward::workspace_bytes(const TextConfig& c, std::int32_t columns, std::uint32_t max_visible_keys,
                                     DeviceExecutionView execution) {
    const std::size_t t = static_cast<std::size_t>(columns), bf = 2, f = 4;
    const std::size_t h = c.hidden_size, w = c.residual_width(), k = c.moe.top_k;
    std::size_t bytes = w * t * bf;                                               // residual
    bytes += h * t * bf * 2 + c.hc.streams * t * f;                               // mixer x, inject
    bytes += (c.hc.rank + c.hc.streams) * t * (bf + f) * 2 + w * t * f * 2;       // mixer scratch bound
    const std::size_t gdn = c.gdn.conv_channels() * t * bf * 2 + c.gdn.value_width() * t * bf * 4 +
                            2 * std::size_t(c.gdn.value_heads) * c.gdn.key_head_dim * t * bf +
                            std::size_t(c.gdn.value_heads) * t * (bf * 4 + f * 2) +
                            ops::gated_delta_net_workspace_capacity_bytes(dim(c.gdn.value_heads),
                                                                          dim(c.gdn.value_heads), 1, columns);
    const auto geometry = qsa_geometry(c);
    const std::size_t index_rows = std::size_t(c.qsa.index_heads + 1) * c.qsa.index_head_dim;
    const std::size_t attention =
        2 * std::size_t(c.attention.query_width()) * t * bf * 3 + 2 * std::size_t(c.attention.key_width()) * t * bf * 2 +
        index_rows * t * (bf + f) + std::size_t(c.qsa.budget / c.qsa.compress_ratio + 1) * t * f +
        ops::qsa_select_blocks_workspace_capacity_bytes(geometry, {max_visible_keys}, 1, columns, execution) +
        ops::qsa_attention_workspace_capacity_bytes({dim(c.attention.head_dim), dim(c.attention.heads),
                                                     dim(c.attention.kv_heads)},
                                                    geometry, KvCacheStorage::Int8Group64, 1, columns, execution);
    const std::size_t ple = std::size_t(c.ple.embed_dim) * t * bf + (w + h) * t * bf * 2 + w * t * bf * 2;
    const std::int32_t entries = dim(k * t);
    const std::size_t moe = (c.moe.experts + 1) * t * f + k * t * (f * 2) + t * f +
                            ops::moe_dispatch_bytes(dim(c.moe.experts), entries) + h * k * t * bf +
                            ops::moe_experts_workspace_bytes(std::min(dim(c.moe.experts), entries), entries, columns) +
                            std::size_t(c.moe.shared_intermediate) * t * bf * 3 + h * t * bf * 2;
    bytes += std::max({gdn, attention, ple, moe});
    // Linear's activation casts and the composed mixer route, plus alignment slack per allocation.
    bytes += w * t * (bf + f) * 2 + (256ULL << 20);
    return bytes;
}

void Forward::run(const ForwardBatch& batch, Tensor& logits, const ForwardTap* tap) {
    if (logits.dtype != DType::FP32 || logits.ne[0] != dim(config_.vocab_size) ||
        logits.ne[1] != batch.logit_columns.ne[0]) {
        throw std::invalid_argument("Qwen4Exp forward: batch geometry is invalid");
    }
    run_front(batch, 0, tap);
    run_back(batch, 0, logits, tap);
}

std::uint32_t Forward::rows_layer() const noexcept {
    for (std::uint32_t l = 0; l < config_.num_hidden_layers; ++l) {
        if (parameters_.layers[l].ple) { return l; }
    }
    return config_.num_hidden_layers;
}

void Forward::run_front(const ForwardBatch& batch, std::uint32_t end, const ForwardTap* tap) {
    const cudaStream_t s = device_.stream;
    const std::int32_t T = batch.ids.ne[0], H = dim(config_.hidden_size), W = dim(config_.residual_width());
    if (T <= 0 || batch.positions.ne[0] != T || batch.rope_positions.ne[0] != T || batch.rope_positions.ne[1] != 3 ||
        batch.block_start_rope.ne[0] != 3 || end > rows_layer()) {
        throw std::invalid_argument("Qwen4Exp forward: batch geometry is invalid");
    }
    work_->reset();
    residual_ = work_->alloc(DType::BF16, {W, T});
    {
        auto scope = work_->scope();
        Tensor x0  = work_->alloc(DType::BF16, {H, T});
        ops::embedding(batch.ids, parameters_.token_embedding, x0, s);
        if (batch.vision != nullptr && batch.vision->columns.data != nullptr) {
            ops::scatter(batch.vision->embeddings, batch.vision->columns, x0, s);
        }
        ops::hyper_connection_expand(x0, dim(config_.hc.streams), residual_, s);
    }
    layers(0, end, batch, tap);
}

void Forward::layers(std::uint32_t begin, std::uint32_t end, const ForwardBatch& batch, const ForwardTap* tap) {
    for (std::uint32_t l = begin; l < end; ++l) {
        try {
            layer(l, batch, residual_, tap);
        } catch (const std::exception& error) {
            throw std::runtime_error("qwen4_exp/layers/" + std::to_string(l) + ": " + error.what());
        }
    }
}

void Forward::run_back(const ForwardBatch& batch, std::uint32_t begin, Tensor& logits, const ForwardTap* tap) {
    const cudaStream_t s = device_.stream;
    const std::int32_t T = batch.ids.ne[0], H = dim(config_.hidden_size);
    if (residual_.data == nullptr || residual_.ne[1] != T || logits.dtype != DType::FP32 ||
        logits.ne[0] != dim(config_.vocab_size) || logits.ne[1] != batch.logit_columns.ne[0]) {
        throw std::invalid_argument("Qwen4Exp forward: the back part needs its front's batch and FP32 logits");
    }
    layers(begin, config_.num_hidden_layers, batch, tap);
    Tensor& residual = residual_;
    if (batch.residual_out.data != nullptr) {
        if (batch.residual_out.bytes() != residual.bytes()) {
            throw std::invalid_argument("Qwen4Exp forward: residual_out must be BF16 [S*H, T]");
        }
        CUDA_CHECK(cudaMemcpyAsync(batch.residual_out.data, residual.data, residual.bytes(), cudaMemcpyDeviceToDevice, s));
    }
    if (batch.mtp_chunk != nullptr) { mtp_cells(*batch.mtp_chunk, residual); }
    auto scope       = work_->scope();
    Tensor final_x   = mix(parameters_.final_mixer, residual, nullptr);
    Tensor last    = work_->alloc(DType::BF16, {H, batch.logit_columns.ne[0]});
    ops::gather_columns(final_x, batch.logit_columns, last, s);
    ops::projection_fp32(last, parameters_.output_head, logits, s);
}

void Forward::layer(std::uint32_t layer, const ForwardBatch& batch, Tensor& residual, const ForwardTap* tap) {
    const cudaStream_t s = device_.stream;
    const std::int32_t T = batch.ids.ne[0];
    const auto& block    = parameters_.layers[layer];
    auto scope           = work_->scope();
    if (block.ple) { ple(*block.ple, residual, batch); }
    Tensor inject = work_->alloc(DType::FP32, {dim(config_.hc.streams), T});
    Tensor xa     = mix(block.attn_hc, residual, &inject);
    const std::uint32_t compact = config_.compact_layer_indices[layer];
    Tensor y = config_.layer_types[layer] == MixerKind::Gdn
                   ? gdn(std::get<GdnParameters>(block.mixer), xa, compact, batch)
                   : attention(std::get<AttentionParameters>(block.mixer), xa, compact, batch);
    ops::hyper_connection_inject(y, inject, residual, s);
    Tensor xm = mix(block.mlp_hc, residual, &inject);
    Tensor ym = moe(block.moe, xm, layer, tap != nullptr && tap->routes != nullptr ? &tap->routes->at(layer) : nullptr);
    ops::hyper_connection_inject(ym, inject, residual, s);
    if (tap != nullptr) {
        copy_tap(tap->mixer_inputs, layer, xa, s);
        copy_tap(tap->mixer_outputs, layer, y, s);
        copy_tap(tap->moe_inputs, layer, xm, s);
        copy_tap(tap->moe_outputs, layer, ym, s);
        copy_tap(tap->residuals, layer, residual, s);
    }
}

Tensor Forward::mix(const HyperConnectionParameters& p, const Tensor& residual, Tensor* inject) {
    const std::int32_t T = residual.ne[1], H = dim(config_.hidden_size);
    Tensor x = work_->alloc(DType::BF16, {H, T});
    ops::hyper_connection_mix(residual, p.norm, p.down.weight, p.up.weight, p.down.policy, dim(config_.hc.streams),
                              dim(config_.hc.rank), config_.rms_norm_eps, x, p.combine ? inject : nullptr, *work_,
                              device_.stream);
    return x;
}

void Forward::ple(const PleParameters& p, Tensor& residual, const ForwardBatch& batch) {
    // A verification call records the convolution inputs for its commit and leaves the history.
    // Otherwise the one sequence's slot is updated in place. The slot index is state (no per-call
    // memset: a memset node splits the decode graph's kernel chain).
    const bool verify = batch.verify != nullptr;
    ple_inject(p, residual, batch.ngram_rows, verify, verify ? &batch.verify->ple_inputs : nullptr, state_.ple_conv,
               state_.ple_slot, verify ? Tensor{} : state_.ple_slot);
}

void Forward::ple_inject(const PleParameters& p, Tensor& residual, const Tensor& ngram_rows, bool columnwise,
                         const Tensor* normalized_out, Tensor& states, const Tensor& source, const Tensor& destination) {
    const cudaStream_t s = device_.stream;
    const std::int32_t T = residual.ne[1], W = residual.ne[0], H = dim(config_.hidden_size);
    auto scope = work_->scope();
    Tensor e   = work_->alloc(DType::BF16, {dim(config_.ple.embed_dim), T});
    ops::ple_embed(ngram_rows, config_.ple.table.format, e, s);
    Tensor kv = work_->alloc(DType::BF16, {W + H, T});
    project_verify(e, p.projection, kv, columnwise);
    Tensor key   = work_->alloc(DType::BF16, {W, T});
    Tensor value = work_->alloc(DType::BF16, {H, T});
    {
        Tensor* parts[] = {&key, &value};
        ops::split_rows(kv, parts, s);
    }
    Tensor gated      = work_->alloc(DType::BF16, {W, T});
    Tensor normalized = normalized_out != nullptr ? *normalized_out : work_->alloc(DType::BF16, {W, T});
    ops::ple_gate(key, value, residual, p.key_norm, p.query_norm, p.conv_norm, dim(config_.hc.streams),
                  config_.rms_norm_eps, gated, normalized, s);
    ops::ple_conv_inject(gated, normalized, p.convolution, dim(config_.ple.conv_dilation), states, source, destination,
                         residual, s);
}

void Forward::project_verify(const Tensor& x, const LinearParameters& p, Tensor& out, bool columnwise) {
    const cudaStream_t s = device_.stream;
    if (!columnwise || x.ne[1] == 1) {
        project(x, p, out, *work_, s);
        return;
    }
    for (std::int32_t t = 0; t < x.ne[1]; ++t) {
        Tensor column = out.slice(1, t, 1);
        project(x.slice(1, t, 1), p, column, *work_, s);
    }
}

Tensor Forward::gdn(const GdnParameters& p, const Tensor& x, std::uint32_t index, const ForwardBatch& batch) {
    const cudaStream_t s = device_.stream;
    const auto& g        = config_.gdn;
    const std::int32_t T = x.ne[1], H = dim(config_.hidden_size);
    const std::int32_t C = dim(g.conv_channels()), KW = dim(g.key_width()), VW = dim(g.value_width());
    const std::int32_t NV = dim(g.value_heads), NK = dim(g.key_heads), DK = dim(g.key_head_dim),
                       DV = dim(g.value_head_dim);
    // A verification call projects its convolution inputs straight into the replay records.
    GdnReplayRecordLayer records;
    if (batch.verify != nullptr) { records = batch.verify->gdn.layer(static_cast<std::int32_t>(index), 1); }
    Tensor qkv = batch.verify != nullptr ? records.conv.view({C, T}) : work_->alloc(DType::BF16, {C, T});
    project(x, p.qkv, qkv, *work_, s);
    Tensor z = work_->alloc(DType::BF16, {VW, T});
    project(x, p.z, z, *work_, s);
    Tensor ab = work_->alloc(DType::BF16, {2 * NV, T});
    project_verify(x, p.control, ab, batch.verify != nullptr);
    Tensor a = work_->alloc(DType::BF16, {NV, T});
    Tensor b = work_->alloc(DType::BF16, {NV, T});
    {
        Tensor* parts[] = {&a, &b};
        ops::split_rows(ab, parts, s);
    }
    Tensor gate = work_->alloc(DType::FP32, {NV, T});
    Tensor beta = work_->alloc(DType::FP32, {NV, T});
    ops::gdn_gating_decay(a, b, p.decay, p.dt_bias, gate, beta, s);

    Tensor q = work_->alloc(DType::BF16, {KW, T});
    Tensor k = work_->alloc(DType::BF16, {KW, T});
    Tensor v = work_->alloc(DType::BF16, {VW, T});
    Tensor& conv_state = state_.gdn_conv.at(index);
    // Verification reads the live convolution window and discards the advanced one.
    Tensor conv_out = batch.verify != nullptr ? batch.verify->conv_scratch : conv_state;
    ops::causal_conv1d_silu_split(qkv, p.convolution, conv_state, conv_out, q, k, v, s);
    Tensor qt = work_->alloc(DType::BF16, {DK, NV, T});
    Tensor kt = work_->alloc(DType::BF16, {DK, NV, T});
    tile_heads(q.view({DK, NK, T}), qt, s);
    tile_heads(k.view({DK, NK, T}), kt, s);
    Tensor o = work_->alloc(DType::BF16, {DV, NV, T});
    const float scale = static_cast<float>(1.0 / std::sqrt(static_cast<double>(DK)));
    if (batch.verify != nullptr) {
        // The recurrence from the live state, which it leaves unchanged; the tiled q/k (one key
        // head per value head), v and {g, beta} go to the records the commit folds.
        Tensor& recurrent = state_.gdn_recurrent.at(index);
        const Tensor states = recurrent.view({DK, DV, NV, 1});
        Tensor out          = o.view({DV, NV, T, 1});
        ops::gated_delta_net_replay_record(qt.view({DK, NV, T, 1}), kt.view({DK, NV, T, 1}), v.view({DV, NV, T, 1}),
                                           gate.view({NV, T, 1}), beta.view({NV, T, 1}), scale, states, Tensor{},
                                           batch.verify->initial_slot, records.key, records.value, records.gate,
                                           out, s);
    } else {
        ops::gated_delta_net(qt, kt, v.view({DV, NV, T}), gate, beta, scale, /*normalize_qk=*/true, *work_,
                             state_.gdn_recurrent.at(index), o, device_.execution_view());
    }
    Tensor normed = work_->alloc(DType::BF16, {DV, NV * T});
    ops::gated_rmsnorm(o.view({DV, NV * T}), p.norm, z.view({DV, NV * T}),
                       g.output_gate == GateActivation::Sigmoid ? ops::RmsGate::Sigmoid : ops::RmsGate::Silu,
                       config_.rms_norm_eps, normed, device_.execution_view());
    Tensor y = work_->alloc(DType::BF16, {H, T});
    project(normed.view({VW, T}), p.output, y, *work_, s);
    return y;
}

Tensor Forward::attention(const AttentionParameters& p, const Tensor& x, std::uint32_t index,
                          const ForwardBatch& batch) {
    const cudaStream_t s = device_.stream;
    const auto& a        = config_.attention;
    const std::int32_t T = x.ne[1], H = dim(config_.hidden_size);
    const std::int32_t D = dim(a.head_dim), NH = dim(a.heads), KVH = dim(a.kv_heads), QW = dim(a.query_width());
    const std::int32_t IH = dim(config_.qsa.index_heads), ID = dim(config_.qsa.index_head_dim);
    // Query and gate rows interleave per head: [D query rows, D gate rows] for each of NH heads.
    Tensor qg = work_->alloc(DType::BF16, {2 * QW, T});
    project(x, p.query_gate, qg, *work_, s);
    Tensor q    = work_->alloc(DType::BF16, {D, NH, T});
    Tensor gate = work_->alloc(DType::BF16, {D, NH, T});
    {
        Tensor qf = q.view({D, NH * T}), gf = gate.view({D, NH * T});
        Tensor* parts[] = {&qf, &gf};
        ops::split_rows(qg.view({2 * D, NH * T}), parts, s);
    }
    Tensor k = work_->alloc(DType::BF16, {D, KVH, T});
    Tensor v = work_->alloc(DType::BF16, {D, KVH, T});
    {
        Tensor kf = k.view({D * KVH, T}), vf = v.view({D * KVH, T});
        project_pair(x, p.key, p.value, kf, vf, *work_, s);
    }
    // The indexer's BF16 rows in FP32, then rounded once to BF16 (Linear's BF16 output boundary).
    Tensor indexed = work_->alloc(DType::FP32, {IH * ID + ID, T});
    {
        const Tensor* rows[] = {&p.index_query, &p.index_key};
        ops::projection_fp32(x, rows, indexed, s);
    }
    Tensor index_rows = work_->alloc(DType::BF16, {IH * ID + ID, T});
    ops::cast_fp32_to_bf16(indexed, index_rows, s);
    Tensor iq = work_->alloc(DType::BF16, {ID, IH, T});
    // A verification call records the raw index keys for the tail commit.
    Tensor ik = batch.verify != nullptr
                    ? batch.verify->qsa_keys.slice(2, static_cast<std::int32_t>(index), 1).view({ID, T})
                    : work_->alloc(DType::BF16, {ID, T});
    {
        Tensor iqf      = iq.view({IH * ID, T});
        Tensor* parts[] = {&iqf, &ik};
        ops::split_rows(index_rows, parts, s);
    }
    Tensor qn = work_->alloc(DType::BF16, {D, NH, T});
    Tensor kn = work_->alloc(DType::BF16, {D, KVH, T});
    {
        Tensor qn_rows = qn.view({D, NH * T}), kn_rows = kn.view({D, KVH * T});
        ops::rmsnorm(q.view({D, NH * T}), p.query_norm, config_.rms_norm_eps, true, qn_rows, s);
        ops::rmsnorm(k.view({D, KVH * T}), p.key_norm, config_.rms_norm_eps, true, kn_rows, s);
    }
    // Interleaved MRoPE over the three-axis RoPE positions (an image's tokens take the frontend's
    // grid positions, later tokens their index plus the prompt's rope delta). A text token's equal
    // axes give the 1-D rotation bit for bit (the kernel computes position[axis] * frequency).
    ops::rope(batch.rope_positions, dim(config_.rope.rotary_dim), config_.rope.theta, qn, kn, device_.execution_view());

    const PagedKVLayerView& cache = kv_.layers.at(index);
    ops::kv_cache_append(kn, v, batch.positions, cache, s);
    const auto geometry = qsa_geometry(config_);
    // Verification pools from a copy of the live tail (and advances the copy).
    Tensor tail = batch.verify != nullptr ? batch.verify->qsa_tails.at(index) : state_.qsa_tails.at(index);
    ops::qsa_pool_keys(ik, batch.positions, batch.rope_positions, batch.block_start_rope, p.index_key_norm, geometry,
                       kv_.block_table, tail, kv_.pooled.at(index), s);
    ops::qsa_index_query(batch.rope_positions, p.index_query_norm, geometry, iq, s);
    const std::int32_t top = dim(config_.qsa.budget / config_.qsa.compress_ratio);
    Tensor selected = work_->alloc(DType::I32, {top, T});
    Tensor counts   = work_->alloc(DType::I32, {T});
    ops::qsa_select_blocks(iq, batch.positions, kv_.block_table, kv_.pooled.at(index), geometry,
                           {kv_.max_visible_keys}, *work_, selected, counts, device_.execution_view());
    Tensor out = work_->alloc(DType::BF16, {D, NH, T});
    ops::qsa_attention(qn, batch.positions, selected, counts, {D, NH, KVH}, geometry,
                       static_cast<float>(1.0 / std::sqrt(static_cast<double>(D))), cache, *work_, out,
                       device_.execution_view());
    // Output gate: out * sigmoid(gate), then o_proj.
    ops::sigmoid_mul(gate, out, s);
    Tensor y = work_->alloc(DType::BF16, {H, T});
    project(out.view({QW, T}), p.output, y, *work_, s);
    return y;
}

Tensor Forward::moe(const MoeParameters& p, const Tensor& x, std::uint32_t layer, Tensor* route_tap) {
    const cudaStream_t s = device_.stream;
    const auto& m        = config_.moe;
    const std::int32_t T = x.ne[1], H = dim(config_.hidden_size), E = dim(m.experts), K = dim(m.top_k);
    Tensor logits = work_->alloc(DType::FP32, {E + 1, T});
    {
        const Tensor* rows[] = {&p.router, &p.shared_score};
        ops::projection_fp32(x, rows, logits, s);
    }
    ops::MoeRouting routing{work_->alloc(DType::I32, {K, T}), work_->alloc(DType::FP32, {K, T}),
                            work_->alloc(DType::FP32, {T})};
    ops::moe_route(logits, K, routing, s);
    if (route_tap != nullptr &&
        cudaMemcpyAsync(route_tap->data, routing.ids.data, routing.ids.bytes(), cudaMemcpyDeviceToDevice, s) !=
            cudaSuccess) {
        throw std::runtime_error("Qwen4Exp forward: route tap copy failed");
    }
    std::int32_t* route_log = nullptr; // moe_dispatch writes the raw ids there (same layout as routing.ids)
    if (experts_.route_log != nullptr) {
        if (static_cast<std::size_t>(K) * T > experts_.route_stride) {
            throw std::logic_error("Qwen4Exp forward: the call exceeds the route log");
        }
        route_log = experts_.route_log + static_cast<std::size_t>(layer) * experts_.route_stride;
    }
    const DeviceSpan dispatch_bytes = work_->alloc_bytes(ops::moe_dispatch_bytes(E, K * T));
    ops::MoeDispatch dispatch       = ops::carve_moe_dispatch(dispatch_bytes.data, E, K * T);
    ops::moe_dispatch(routing, E, dispatch, route_log, s);
    ops::MoeExpertSource source{.format        = p.bank->format,
                                .frame_base    = experts_.frame_base,
                                .frames        = experts_.frames.at(layer),
                                .host_records  = p.bank->device_records,
                                .record_stride = p.bank->record_stride,
                                .frame_stride  = experts_.frame_stride,
                                .error         = experts_.error,
                                .staging_base  = experts_.staging_base,
                                .staging_slots = experts_.staging_slots,
                                .zero_copy_misses = experts_.zero_copy_misses};
    ExpertStream* stream = experts_.stream != nullptr && experts_.stream->active() ? experts_.stream : nullptr;
    if (stream != nullptr) {
        source.frames = stream->table(layer);
        // Every non-resident expert streamed: no staging pass (one pass over every job).
        if (stream->covers_all()) { source.staging_slots = 0; }
        stream->before_experts(layer, s);
    }
    Tensor outputs              = work_->alloc(DType::BF16, {H, K * T});
    const std::int32_t max_jobs = std::min(E, K * T);
    const DeviceSpan expert_ws  = work_->alloc_bytes(ops::moe_experts_workspace_bytes(max_jobs, K * T, T));
    ops::moe_experts(x, dispatch, source, K, max_jobs, expert_ws.data, outputs, s);
    if (stream != nullptr) { stream->after_experts(layer, s); }

    const std::int32_t I = dim(m.shared_intermediate);
    Tensor g = work_->alloc(DType::BF16, {I, T});
    Tensor u = work_->alloc(DType::BF16, {I, T});
    project_pair(x, p.shared_gate, p.shared_up, g, u, *work_, s);
    Tensor product = work_->alloc(DType::BF16, {I, T});
    ops::silu_mul(g, u, product, s);
    Tensor shared = work_->alloc(DType::BF16, {H, T});
    project(product, p.shared_down, shared, *work_, s);
    Tensor y = work_->alloc(DType::BF16, {H, T});
    ops::moe_combine(outputs, routing, shared, y, s);
    return y;
}

// ------------------------------------------------------------------------- several sequences
// Column b of a rows call is one decode step of slot slots[b]. The per-column Ops (embedding, the
// mixers, GGML Linears, norms, RoPE, the index query, MoE and the head) take the B columns as one
// call; the two BF16 Linears run one column at a time; the convolution windows, recurrent states and
// PLE histories are updated in each row's slot by the slotted Ops; the QSA Ops that read or write
// a row's KV, pooled keys and index-key tail run once per row through its own block table.

void Forward::run_rows(const ForwardRowsBatch& batch, Tensor& logits) {
    run_rows_front(batch);
    run_rows_back(batch, logits);
}

void Forward::run_rows_front(const ForwardRowsBatch& batch) {
    if (!rows_) { throw std::logic_error("Qwen4Exp forward: run_rows needs the slotted state"); }
    const auto B     = static_cast<std::int32_t>(batch.slots.size());
    const auto slots = static_cast<std::int32_t>(rows_->slot_state.size());
    bool distinct    = B >= 1 && B <= kMaxForwardRows;
    for (std::int32_t b = 0; distinct && b < B; ++b) {
        const std::int32_t slot = batch.slots[static_cast<std::size_t>(b)];
        distinct = slot >= 0 && slot < slots &&
                   std::find(batch.slots.begin(), batch.slots.begin() + b, slot) == batch.slots.begin() + b;
    }
    if (!distinct || batch.slot_ids.numel() != B || batch.ids.ne[0] != B || batch.positions.ne[0] != B ||
        batch.rope_positions.ne[0] != B || batch.rope_positions.ne[1] != 3 || batch.row_rope.ne[0] != 3 ||
        batch.row_rope.ne[1] != B || batch.block_start_rope.ne[0] != 3 || batch.block_start_rope.ne[1] != B ||
        batch.ngram_rows.ne[2] != B) {
        throw std::invalid_argument("Qwen4Exp forward: rows batch geometry is invalid");
    }
    const cudaStream_t s = device_.stream;
    const std::int32_t H = dim(config_.hidden_size), W = dim(config_.residual_width());
    work_->reset();
    residual_ = work_->alloc(DType::BF16, {W, B});
    {
        auto scope = work_->scope();
        Tensor x0  = work_->alloc(DType::BF16, {H, B});
        ops::embedding(batch.ids, parameters_.token_embedding, x0, s);
        ops::hyper_connection_expand(x0, dim(config_.hc.streams), residual_, s);
    }
    const std::uint32_t split = rows_layer();
    for (std::uint32_t l = 0; l < split; ++l) {
        try {
            layer_rows(l, batch);
        } catch (const std::exception& error) {
            throw std::runtime_error("qwen4_exp/rows/layers/" + std::to_string(l) + ": " + error.what());
        }
    }
}

void Forward::run_rows_back(const ForwardRowsBatch& batch, Tensor& logits) {
    const auto B = static_cast<std::int32_t>(batch.slots.size());
    if (residual_.data == nullptr || residual_.ne[1] != B || logits.dtype != DType::FP32 ||
        logits.ne[0] != dim(config_.vocab_size) || logits.ne[1] != B) {
        throw std::invalid_argument("Qwen4Exp forward: the rows back part needs its front's batch and FP32 logits");
    }
    const cudaStream_t s = device_.stream;
    for (std::uint32_t l = rows_layer(); l < config_.num_hidden_layers; ++l) {
        try {
            layer_rows(l, batch);
        } catch (const std::exception& error) {
            throw std::runtime_error("qwen4_exp/rows/layers/" + std::to_string(l) + ": " + error.what());
        }
    }
    if (batch.residual_out.data != nullptr) {
        if (batch.residual_out.bytes() != residual_.bytes()) {
            throw std::invalid_argument("Qwen4Exp forward: residual_out must be BF16 [S*H, B]");
        }
        CUDA_CHECK(cudaMemcpyAsync(batch.residual_out.data, residual_.data, residual_.bytes(),
                                   cudaMemcpyDeviceToDevice, s));
    }
    {
        auto scope     = work_->scope();
        Tensor final_x = mix(parameters_.final_mixer, residual_, nullptr);
        ops::projection_fp32(final_x, parameters_.output_head, logits, s);
    }
    residual_ = Tensor{};
}

void Forward::layer_rows(std::uint32_t layer, const ForwardRowsBatch& batch) {
    const cudaStream_t s = device_.stream;
    const std::int32_t B = residual_.ne[1];
    const auto& block    = parameters_.layers[layer];
    auto scope           = work_->scope();
    if (block.ple) {
        ple_inject(*block.ple, residual_, batch.ngram_rows, /*columnwise=*/true, nullptr, rows_->state.ple_conv,
                   batch.slot_ids, batch.slot_ids);
    }
    Tensor inject = work_->alloc(DType::FP32, {dim(config_.hc.streams), B});
    Tensor xa     = mix(block.attn_hc, residual_, &inject);
    const std::uint32_t compact = config_.compact_layer_indices[layer];
    Tensor y = config_.layer_types[layer] == MixerKind::Gdn
                   ? gdn_rows(std::get<GdnParameters>(block.mixer), xa, compact, batch)
                   : attention_rows(std::get<AttentionParameters>(block.mixer), xa, compact, batch);
    ops::hyper_connection_inject(y, inject, residual_, s);
    Tensor xm = mix(block.mlp_hc, residual_, &inject);
    Tensor ym = moe(block.moe, xm, layer, nullptr);
    ops::hyper_connection_inject(ym, inject, residual_, s);
}

Tensor Forward::gdn_rows(const GdnParameters& p, const Tensor& x, std::uint32_t index, const ForwardRowsBatch& batch) {
    const cudaStream_t s = device_.stream;
    const auto& g        = config_.gdn;
    const std::int32_t B = x.ne[1], H = dim(config_.hidden_size);
    const std::int32_t C = dim(g.conv_channels()), KW = dim(g.key_width()), VW = dim(g.value_width());
    const std::int32_t NV = dim(g.value_heads), NK = dim(g.key_heads), DK = dim(g.key_head_dim),
                       DV = dim(g.value_head_dim);
    Tensor qkv = work_->alloc(DType::BF16, {C, B});
    project(x, p.qkv, qkv, *work_, s);
    Tensor z = work_->alloc(DType::BF16, {VW, B});
    project(x, p.z, z, *work_, s);
    Tensor ab = work_->alloc(DType::BF16, {2 * NV, B});
    project_verify(x, p.control, ab, /*columnwise=*/true);
    Tensor a = work_->alloc(DType::BF16, {NV, B});
    Tensor b = work_->alloc(DType::BF16, {NV, B});
    {
        Tensor* parts[] = {&a, &b};
        ops::split_rows(ab, parts, s);
    }
    Tensor gate = work_->alloc(DType::FP32, {NV, B});
    Tensor beta = work_->alloc(DType::FP32, {NV, B});
    ops::gdn_gating_decay(a, b, p.decay, p.dt_bias, gate, beta, s);

    // Each row's convolution window, read and replaced in its slot (the snapshot form at one column
    // per row: the same arithmetic as the single-sequence split form).
    Tensor conv     = work_->alloc(DType::BF16, {C, B});
    Tensor conv_out = conv.view({C, 1, B});
    ops::causal_conv1d_silu_snapshot(qkv.view({C, 1, B}), p.convolution, rows_->state.gdn_conv.at(index), Tensor{},
                                     batch.slot_ids, batch.slot_ids, conv_out, s);
    Tensor q = work_->alloc(DType::BF16, {KW, B});
    Tensor k = work_->alloc(DType::BF16, {KW, B});
    Tensor v = work_->alloc(DType::BF16, {VW, B});
    {
        Tensor* parts[] = {&q, &k, &v};
        ops::split_rows(conv, parts, s);
    }
    Tensor qt = work_->alloc(DType::BF16, {DK, NV, B});
    Tensor kt = work_->alloc(DType::BF16, {DK, NV, B});
    tile_heads(q.view({DK, NK, B}), qt, s);
    tile_heads(k.view({DK, NK, B}), kt, s);
    // Each row's recurrence from its slot, in place (the one-token kernel of a T = 1 call).
    Tensor o           = work_->alloc(DType::BF16, {DV, NV, B});
    Tensor o_rows      = o.view({DV, NV, 1, B});
    const float scale = static_cast<float>(1.0 / std::sqrt(static_cast<double>(DK)));
    ops::gated_delta_net_batch_update(qt.view({DK, NV, 1, B}), kt.view({DK, NV, 1, B}), v.view({DV, NV, 1, B}),
                                      gate.view({NV, 1, B}), beta.view({NV, 1, B}), scale, /*normalize_qk=*/true,
                                      rows_->state.gdn_recurrent.at(index), batch.slot_ids, batch.slot_ids, o_rows, s);
    Tensor normed = work_->alloc(DType::BF16, {DV, NV * B});
    ops::gated_rmsnorm(o.view({DV, NV * B}), p.norm, z.view({DV, NV * B}),
                       g.output_gate == GateActivation::Sigmoid ? ops::RmsGate::Sigmoid : ops::RmsGate::Silu,
                       config_.rms_norm_eps, normed, device_.execution_view());
    Tensor y = work_->alloc(DType::BF16, {H, B});
    project(normed.view({VW, B}), p.output, y, *work_, s);
    return y;
}

Tensor Forward::attention_rows(const AttentionParameters& p, const Tensor& x, std::uint32_t index,
                               const ForwardRowsBatch& batch) {
    const cudaStream_t s = device_.stream;
    const auto& a        = config_.attention;
    const std::int32_t B = x.ne[1], H = dim(config_.hidden_size);
    const std::int32_t D = dim(a.head_dim), NH = dim(a.heads), KVH = dim(a.kv_heads), QW = dim(a.query_width());
    const std::int32_t IH = dim(config_.qsa.index_heads), ID = dim(config_.qsa.index_head_dim);
    Tensor qg = work_->alloc(DType::BF16, {2 * QW, B});
    project(x, p.query_gate, qg, *work_, s);
    Tensor q    = work_->alloc(DType::BF16, {D, NH, B});
    Tensor gate = work_->alloc(DType::BF16, {D, NH, B});
    {
        Tensor qf = q.view({D, NH * B}), gf = gate.view({D, NH * B});
        Tensor* parts[] = {&qf, &gf};
        ops::split_rows(qg.view({2 * D, NH * B}), parts, s);
    }
    Tensor k = work_->alloc(DType::BF16, {D, KVH, B});
    Tensor v = work_->alloc(DType::BF16, {D, KVH, B});
    {
        Tensor kf = k.view({D * KVH, B}), vf = v.view({D * KVH, B});
        project_pair(x, p.key, p.value, kf, vf, *work_, s);
    }
    Tensor indexed = work_->alloc(DType::FP32, {IH * ID + ID, B});
    {
        const Tensor* rows[] = {&p.index_query, &p.index_key};
        ops::projection_fp32(x, rows, indexed, s);
    }
    Tensor index_rows = work_->alloc(DType::BF16, {IH * ID + ID, B});
    ops::cast_fp32_to_bf16(indexed, index_rows, s);
    Tensor iq = work_->alloc(DType::BF16, {ID, IH, B});
    Tensor ik = work_->alloc(DType::BF16, {ID, B});
    {
        Tensor iqf      = iq.view({IH * ID, B});
        Tensor* parts[] = {&iqf, &ik};
        ops::split_rows(index_rows, parts, s);
    }
    Tensor qn = work_->alloc(DType::BF16, {D, NH, B});
    Tensor kn = work_->alloc(DType::BF16, {D, KVH, B});
    {
        Tensor qn_rows = qn.view({D, NH * B}), kn_rows = kn.view({D, KVH * B});
        ops::rmsnorm(q.view({D, NH * B}), p.query_norm, config_.rms_norm_eps, true, qn_rows, s);
        ops::rmsnorm(k.view({D, KVH * B}), p.key_norm, config_.rms_norm_eps, true, kn_rows, s);
    }
    ops::rope(batch.rope_positions, dim(config_.rope.rotary_dim), config_.rope.theta, qn, kn, device_.execution_view());
    const auto geometry = qsa_geometry(config_);
    ops::qsa_index_query(batch.rope_positions, p.index_query_norm, geometry, iq, s);
    const std::int32_t top = dim(config_.qsa.budget / config_.qsa.compress_ratio);
    Tensor selected = work_->alloc(DType::I32, {top, B});
    Tensor counts   = work_->alloc(DType::I32, {B});
    Tensor out      = work_->alloc(DType::BF16, {D, NH, B});
    const float scale = static_cast<float>(1.0 / std::sqrt(static_cast<double>(D)));
    for (std::int32_t b = 0; b < B; ++b) {
        const auto slot               = static_cast<std::size_t>(batch.slots[static_cast<std::size_t>(b)]);
        const ForwardKV& kv           = rows_->slot_kv[slot];
        const PagedKVLayerView& cache = kv.layers.at(index);
        const Tensor position         = batch.positions.slice(0, b, 1);
        const Tensor rope             = batch.row_rope.slice(1, b, 1).view({1, 3});
        ops::kv_cache_append(kn.slice(2, b, 1), v.slice(2, b, 1), position, cache, s);
        Tensor tail   = rows_->slot_state[slot].qsa_tails.at(index);
        Tensor pooled = kv.pooled.at(index);
        ops::qsa_pool_keys(ik.slice(1, b, 1), position, rope, batch.block_start_rope.slice(1, b, 1).view({3}),
                           p.index_key_norm, geometry, kv.block_table, tail, pooled, s);
        Tensor row_selected = selected.slice(1, b, 1);
        Tensor row_count    = counts.slice(0, b, 1);
        ops::qsa_select_blocks(iq.slice(2, b, 1), position, kv.block_table, pooled, geometry, {kv.max_visible_keys},
                               *work_, row_selected, row_count, device_.execution_view());
        Tensor row_out = out.slice(2, b, 1);
        ops::qsa_attention(qn.slice(2, b, 1), position, row_selected, row_count, {D, NH, KVH}, geometry, scale, cache,
                           *work_, row_out, device_.execution_view());
    }
    ops::sigmoid_mul(gate, out, s);
    Tensor y = work_->alloc(DType::BF16, {H, B});
    project(out.view({QW, B}), p.output, y, *work_, s);
    return y;
}

// ------------------------------------------------------------------------------- MTP drafter
// Adapted from Infernix a3edb450 src/models/qwen4_exp/execution/forward.cpp:666-785 (run_mtp,
// mtp_block). Modified for NInfer-3090: one sequence; BF16 projections in FP32 rows
// (projection_fp32) where K <= 3072, BF16 Linears for the fusion and attention-output
// projections; the routed experts are Q2_0 records served by offloaded_sparse_moe's narrow route
// from the device bank (every expert resident) instead of resident_moe; the draft head is the
// text model's IQ4_XS head over the public tokens (no proposal head, no draft log-probability).

namespace {

// FP32 rows of a BF16 projection, then one BF16 rounding (the drafter's projections).
Tensor project_rows(const Tensor& x, std::initializer_list<const Tensor*> rows, std::int32_t n, WorkspaceArena& work,
                    cudaStream_t s) {
    Tensor wide = work.alloc(DType::FP32, {n, x.ne[1]});
    ops::projection_fp32(x, std::span<const Tensor* const>(rows.begin(), rows.size()), wide, s);
    Tensor out = work.alloc(DType::BF16, {n, x.ne[1]});
    ops::cast_fp32_to_bf16(wide, out, s);
    return out;
}

} // namespace

Tensor Forward::mtp_attention(const MtpParameters& p, const Tensor& x, const MtpCall& call) {
    const cudaStream_t s = device_.stream;
    const auto& a        = config_.attention;
    const std::int32_t T = x.ne[1], H = dim(config_.hidden_size);
    const std::int32_t D = dim(a.head_dim), NH = dim(a.heads), KVH = dim(a.kv_heads), QW = dim(a.query_width());
    const std::int32_t IH = dim(config_.qsa.index_heads), ID = dim(config_.qsa.index_head_dim);
    ForwardMtp& m         = *mtp_;
    Tensor kv = project_rows(x, {&p.key, &p.value}, 2 * D * KVH, *work_, s);
    Tensor k  = work_->alloc(DType::BF16, {D, KVH, T});
    Tensor v  = work_->alloc(DType::BF16, {D, KVH, T});
    {
        Tensor kf = k.view({D * KVH, T}), vf = v.view({D * KVH, T});
        Tensor* parts[] = {&kf, &vf};
        ops::split_rows(kv, parts, s);
    }
    Tensor index_rows = project_rows(x, {&p.index_query, &p.index_key}, IH * ID + ID, *work_, s);
    Tensor iq         = work_->alloc(DType::BF16, {ID, IH, T});
    Tensor ik         = call.key_records.data != nullptr ? call.key_records.view({ID, T}) : work_->alloc(DType::BF16, {ID, T});
    {
        Tensor iqf      = iq.view({IH * ID, T});
        Tensor* parts[] = {&iqf, &ik};
        ops::split_rows(index_rows, parts, s);
    }
    Tensor kn = work_->alloc(DType::BF16, {D, KVH, T});
    {
        Tensor kn_rows = kn.view({D, KVH * T});
        ops::rmsnorm(k.view({D, KVH * T}), p.key_norm, config_.rms_norm_eps, true, kn_rows, s);
    }
    const auto geometry = qsa_geometry(config_);
    Tensor tail         = call.tail;
    if (call.kv_only) {
        ops::rope(call.rope_positions, dim(config_.rope.rotary_dim), config_.rope.theta, kn, device_.execution_view());
        ops::kv_cache_append(kn, v, call.positions, m.kv, s);
        ops::qsa_pool_keys(ik, call.positions, call.rope_positions, call.block_start_rope, p.index_key_norm, geometry,
                           kv_.block_table, tail, m.pooled, s);
        return Tensor{};
    }
    Tensor qg   = project_rows(x, {&p.query_gate}, 2 * QW, *work_, s);
    Tensor q    = work_->alloc(DType::BF16, {D, NH, T});
    Tensor gate = work_->alloc(DType::BF16, {D, NH, T});
    {
        Tensor qf = q.view({D, NH * T}), gf = gate.view({D, NH * T});
        Tensor* parts[] = {&qf, &gf};
        ops::split_rows(qg.view({2 * D, NH * T}), parts, s);
    }
    Tensor qn = work_->alloc(DType::BF16, {D, NH, T});
    {
        Tensor qn_rows = qn.view({D, NH * T});
        ops::rmsnorm(q.view({D, NH * T}), p.query_norm, config_.rms_norm_eps, true, qn_rows, s);
    }
    ops::rope(call.rope_positions, dim(config_.rope.rotary_dim), config_.rope.theta, qn, kn, device_.execution_view());
    ops::kv_cache_append(kn, v, call.positions, m.kv, s);
    ops::qsa_pool_keys(ik, call.positions, call.rope_positions, call.block_start_rope, p.index_key_norm, geometry,
                       kv_.block_table, tail, m.pooled, s);
    ops::qsa_index_query(call.rope_positions, p.index_query_norm, geometry, iq, s);
    const std::int32_t top = dim(config_.qsa.budget / config_.qsa.compress_ratio);
    Tensor selected = work_->alloc(DType::I32, {top, T});
    Tensor counts   = work_->alloc(DType::I32, {T});
    ops::qsa_select_blocks(iq, call.positions, kv_.block_table, m.pooled, geometry, {kv_.max_visible_keys}, *work_,
                           selected, counts, device_.execution_view());
    Tensor out = work_->alloc(DType::BF16, {D, NH, T});
    ops::qsa_attention(qn, call.positions, selected, counts, {D, NH, KVH}, geometry,
                       static_cast<float>(1.0 / std::sqrt(static_cast<double>(D))), m.kv, *work_, out,
                       device_.execution_view());
    ops::sigmoid_mul(gate, out, s);
    Tensor y = work_->alloc(DType::BF16, {H, T});
    project(out.view({QW, T}), p.output, y, *work_, s);
    return y;
}

void Forward::run_mtp(const MtpCall& call) {
    work_->reset();
    mtp_block(call);
}

void Forward::mtp_cells(const MtpChunk& chunk, const Tensor& residual) {
    if (!mtp_) { throw std::logic_error("Qwen4Exp forward: MTP cells without the drafter"); }
    const cudaStream_t s = device_.stream;
    const std::int32_t T = residual.ne[1], W = dim(config_.residual_width());
    const std::int32_t cells = (chunk.prepend ? 1 : 0) + chunk.columns;
    const std::size_t column = static_cast<std::size_t>(W) * 2;
    if (chunk.columns < 0 || chunk.columns > T || chunk.saved.bytes() != column ||
        (cells > 0 && (chunk.ids.ne[0] != cells || chunk.positions.ne[0] != cells || chunk.rope_positions == nullptr ||
                       chunk.block_start_rope == nullptr))) {
        throw std::invalid_argument("Qwen4Exp forward: MTP chunk does not match the call");
    }
    // Cell v is `saved` (v = 0 when prepending) or call column v - prepend.
    const std::int32_t first_column = chunk.prepend ? -1 : 0;
    for (std::int32_t begin = 0; begin < cells; begin += kMtpChunkColumns) {
        const std::int32_t n = std::min(kMtpChunkColumns, cells - begin);
        auto scope  = work_->scope();
        Tensor rows = work_->alloc(DType::BF16, {W, n});
        auto* out   = static_cast<std::byte*>(rows.data);
        std::int32_t from = first_column + begin;
        if (from < 0) {
            CUDA_CHECK(cudaMemcpyAsync(out, chunk.saved.data, column, cudaMemcpyDeviceToDevice, s));
            out += column;
            from = 0;
        }
        const std::int32_t rest = first_column + begin + n - from;
        if (rest > 0) {
            CUDA_CHECK(cudaMemcpyAsync(out, static_cast<const std::byte*>(residual.data) + column * from, column * rest,
                                       cudaMemcpyDeviceToDevice, s));
        }
        const std::int32_t k = begin / kMtpChunkColumns;
        MtpCall call{.residuals        = rows,
                     .ids              = chunk.ids.slice(0, begin, n),
                     .positions        = chunk.positions.slice(0, begin, n),
                     .rope_positions   = Tensor(chunk.rope_positions + 3 * static_cast<std::size_t>(begin), DType::I32, {n, 3}),
                     .block_start_rope = Tensor(chunk.block_start_rope + 3 * static_cast<std::size_t>(k), DType::I32, {3}),
                     .tail             = mtp_->tail,
                     .kv_only          = true,
                     .vision           = chunk.vision != nullptr ? &chunk.vision[k] : nullptr};
        mtp_block(call);
    }
    CUDA_CHECK(cudaMemcpyAsync(chunk.saved.data, static_cast<const std::byte*>(residual.data) + column * (T - 1), column,
                               cudaMemcpyDeviceToDevice, s));
}

void Forward::mtp_block(const MtpCall& call) {
    if (!mtp_ || !parameters_.mtp) { throw std::logic_error("Qwen4Exp forward: the MTP drafter is not loaded"); }
    const cudaStream_t s = device_.stream;
    const auto& p        = *parameters_.mtp;
    const std::int32_t T = call.ids.ne[0], H = dim(config_.hidden_size), W = dim(config_.residual_width());
    const std::int32_t S = dim(config_.hc.streams);
    if (T <= 0 || call.residuals.ne[0] != W || call.residuals.ne[1] != T || call.positions.ne[0] != T ||
        call.rope_positions.ne[0] != T || call.rope_positions.ne[1] != 3 || call.block_start_rope.ne[0] != 3 ||
        call.tail.data == nullptr || (!call.kv_only && (T != 1 || call.residual_out.data == nullptr ||
                                                        call.drafts.data == nullptr)) ||
        T > mtp_->ones.ne[1]) {
        throw std::invalid_argument("Qwen4Exp forward: MTP call geometry is invalid");
    }
    auto scope = work_->scope();
    try {
        // Input fusion: R = fc_hidden(each stream of RMSNorm_{S*H}(rows)) + fc_embedding(RMSNorm_H(embed(id))).
        Tensor x0 = work_->alloc(DType::BF16, {H, T});
        ops::embedding(call.ids, parameters_.token_embedding, x0, s);
        if (call.vision != nullptr && call.vision->columns.data != nullptr) {
            ops::scatter(call.vision->embeddings, call.vision->columns, x0, s);
        }
        Tensor e = work_->alloc(DType::BF16, {H, T});
        ops::rmsnorm(x0, p.embedding_norm, config_.rms_norm_eps, true, e, s);
        Tensor ep = work_->alloc(DType::BF16, {H, T});
        project(e, p.embedding_projection, ep, *work_, s);
        Tensor hn = work_->alloc(DType::BF16, {W, T});
        ops::rmsnorm(call.residuals, p.hidden_norm, config_.rms_norm_eps, true, hn, s);
        Tensor residual = work_->alloc(DType::BF16, {W, T});
        {
            Tensor per_stream = residual.view({H, S * T});
            project(hn.view({H, S * T}), p.hidden_projection, per_stream, *work_, s);
        }
        ops::hyper_connection_inject(ep, mtp_->ones.slice(1, 0, T), residual, s);
        const auto tap = [&](const Tensor& target, const Tensor& source) {
            if (call.tap == nullptr || target.data == nullptr) { return; }
            if (target.bytes() != source.bytes()) { throw std::invalid_argument("Qwen4Exp forward: MTP tap shape"); }
            CUDA_CHECK(cudaMemcpyAsync(target.data, source.data, source.bytes(), cudaMemcpyDeviceToDevice, s));
        };
        if (call.tap != nullptr) { tap(call.tap->fused, residual); }

        Tensor inject = work_->alloc(DType::FP32, {S, T});
        Tensor xa     = mix(p.attn_hc, residual, &inject);
        Tensor y      = mtp_attention(p, xa, call);
        if (call.kv_only) { return; }
        if (call.tap != nullptr) {
            tap(call.tap->mixer_in, xa);
            tap(call.tap->attention, y);
        }
        ops::hyper_connection_inject(y, inject, residual, s);

        // MoE over the device-resident Q2_0 experts, with the shared expert.
        Tensor xm            = mix(p.mlp_hc, residual, &inject);
        const auto& mc       = config_.moe;
        const std::int32_t E = dim(mc.experts), K = dim(mc.top_k), SI = dim(mc.shared_intermediate);
        Tensor logits        = work_->alloc(DType::FP32, {E + 1, T});
        {
            const Tensor* rows[] = {&p.router, &p.shared_score};
            ops::projection_fp32(xm, rows, logits, s);
        }
        ops::MoeRouting routing{work_->alloc(DType::I32, {K, T}), work_->alloc(DType::FP32, {K, T}),
                                work_->alloc(DType::FP32, {T})};
        ops::moe_route(logits, K, routing, s);
        const DeviceSpan dispatch_bytes = work_->alloc_bytes(ops::moe_dispatch_bytes(E, K * T));
        ops::MoeDispatch dispatch       = ops::carve_moe_dispatch(dispatch_bytes.data, E, K * T);
        ops::moe_dispatch(routing, E, dispatch, nullptr, s);
        const ops::MoeExpertSource source{.format        = p.bank->format,
                                          .frame_base    = p.bank->device_records,
                                          .frames        = static_cast<const std::int32_t*>(mtp_frames_.data),
                                          .host_records  = p.bank->device_records,
                                          .record_stride = p.bank->record_stride,
                                          .frame_stride  = p.bank->record_stride};
        Tensor outputs              = work_->alloc(DType::BF16, {H, K * T});
        const std::int32_t max_jobs = std::min(E, K * T);
        const DeviceSpan expert_ws  = work_->alloc_bytes(ops::moe_experts_workspace_bytes(max_jobs, K * T, T));
        ops::moe_experts(xm, dispatch, source, K, max_jobs, expert_ws.data, outputs, s);
        Tensor gu = project_rows(xm, {&p.shared_gate, &p.shared_up}, 2 * SI, *work_, s);
        Tensor g  = work_->alloc(DType::BF16, {SI, T});
        Tensor u  = work_->alloc(DType::BF16, {SI, T});
        {
            Tensor* parts[] = {&g, &u};
            ops::split_rows(gu, parts, s);
        }
        Tensor product = work_->alloc(DType::BF16, {SI, T});
        ops::silu_mul(g, u, product, s);
        Tensor shared = project_rows(product, {&p.shared_down}, H, *work_, s);
        Tensor ym     = work_->alloc(DType::BF16, {H, T});
        ops::moe_combine(outputs, routing, shared, ym, s);
        if (call.tap != nullptr) {
            tap(call.tap->moe_in, xm);
            tap(call.tap->routes, routing.ids);
            tap(call.tap->moe_out, ym);
        }
        ops::hyper_connection_inject(ym, inject, residual, s);
        CUDA_CHECK(cudaMemcpyAsync(call.residual_out.data, residual.data, residual.bytes(), cudaMemcpyDeviceToDevice, s));

        // The draft: argmax of the text head over the public tokens, from BF16 logits.
        Tensor final_x = mix(p.final_mixer, residual, nullptr);
        Tensor head    = work_->alloc(DType::FP32, {dim(config_.vocab_size), T});
        ops::projection_fp32(final_x, parameters_.output_head, head, s);
        if (call.tap != nullptr) {
            tap(call.tap->final_x, final_x);
            tap(call.tap->logits, head);
        }
        Tensor narrow = work_->alloc(DType::BF16, {dim(config_.vocab_size), T});
        ops::cast_fp32_to_bf16(head, narrow, s);
        Tensor drafts = call.drafts;
        ops::argmax(narrow, drafts, dim(parameters_.model.resources().public_token_count), s);
    } catch (const std::exception& error) {
        throw std::runtime_error(std::string("qwen4_exp/mtp: ") + error.what());
    }
}

std::size_t Forward::mtp_workspace_bytes(const TextConfig& c, std::int32_t columns, std::uint32_t max_visible_keys,
                                         DeviceExecutionView execution) {
    const std::size_t t = static_cast<std::size_t>(columns), bf = 2, f = 4;
    const std::size_t h = c.hidden_size, w = c.residual_width(), v = c.vocab_size;
    // A block at `columns` columns (as the text forward sizes one), the fusion's tensors, the
    // FP32 projection rows and the head's logits of one column.
    return workspace_bytes(c, columns, max_visible_keys, execution) + (h * 4 + w * 2) * t * bf +
           (2 * std::size_t(c.attention.query_width()) + 2 * std::size_t(c.attention.key_width()) + 1024) * t * f +
           v * (f + bf) + (8ULL << 20);
}

} // namespace ninfer::models::qwen4_exp::execution
