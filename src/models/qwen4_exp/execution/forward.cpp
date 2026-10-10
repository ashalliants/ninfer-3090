// Adapted from Infernix a3edb450 src/models/qwen4_exp/execution/forward.cpp (Apache-2.0).
// Modified for NInfer-3090: see forward.h. Per block the composition is Infernix's: PLE injection
// (its layer), attention-side mixer, GDN or QSA, injection, MLP-side mixer, MoE, injection. The
// GGUF's layouts change three steps: the attention query and gate rows interleave per head, the
// GDN value heads are in tiled order (value head h reads key head h % key_heads), and the indexer
// projections are BF16 rows projected in FP32.

#include "models/qwen4_exp/execution/forward.h"

#include "ninfer/ops/cast.h"
#include "ninfer/ops/causal_conv1d_silu.h"
#include "ninfer/ops/embedding.h"
#include "ninfer/ops/gated_delta_net.h"
#include "ninfer/ops/gated_rmsnorm.h"
#include "ninfer/ops/gdn_gating.h"
#include "ninfer/ops/hyper_connection.h"
#include "ninfer/ops/kv_cache_append.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/offloaded_sparse_moe.h"
#include "ninfer/ops/ple.h"
#include "ninfer/ops/projection_fp32.h"
#include "ninfer/ops/rmsnorm.h"
#include "ninfer/ops/rope.h"
#include "ninfer/ops/rows.h"
#include "ninfer/ops/sigmoid_mul.h"
#include "ninfer/ops/silu_mul.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>

namespace ninfer::models::qwen4_exp::execution {
namespace {

std::int32_t dim(std::uint64_t v) { return static_cast<std::int32_t>(v); }

void project(const Tensor& x, const LinearParameters& p, Tensor& out, WorkspaceArena& work, cudaStream_t s) {
    ops::linear(x, p.weight, out, p.policy, work, s);
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
                 ForwardKV kv, ForwardExperts experts)
    : parameters_(parameters), config_(parameters.model.config()), device_(device), work_(work),
      state_(std::move(state)), kv_(std::move(kv)), experts_(std::move(experts)) {
    if (state_.gdn_conv.size() != config_.gdn_layers || state_.gdn_recurrent.size() != config_.gdn_layers ||
        state_.qsa_tails.size() != config_.attention_layers || kv_.layers.size() != config_.attention_layers ||
        kv_.pooled.size() != config_.attention_layers || experts_.frames.size() != config_.num_hidden_layers) {
        throw std::invalid_argument("Qwen4Exp forward: state, KV or expert residency is incomplete");
    }
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
    const cudaStream_t s = device_.stream;
    const std::int32_t T = batch.ids.ne[0], H = dim(config_.hidden_size), W = dim(config_.residual_width());
    if (T <= 0 || batch.positions.ne[0] != T || batch.rope_positions.ne[0] != T || batch.rope_positions.ne[1] != 3 ||
        batch.block_start_rope.ne[0] != 3 || logits.dtype != DType::FP32 ||
        logits.ne[0] != dim(config_.vocab_size) || logits.ne[1] != batch.logit_columns.ne[0]) {
        throw std::invalid_argument("Qwen4Exp forward: batch geometry is invalid");
    }
    work_.reset();
    Tensor residual = work_.alloc(DType::BF16, {W, T});
    {
        auto scope = work_.scope();
        Tensor x0  = work_.alloc(DType::BF16, {H, T});
        ops::embedding(batch.ids, parameters_.token_embedding, x0, s);
        ops::hyper_connection_expand(x0, dim(config_.hc.streams), residual, s);
    }
    for (std::uint32_t l = 0; l < config_.num_hidden_layers; ++l) {
        try {
            layer(l, batch, residual, tap);
        } catch (const std::exception& error) {
            throw std::runtime_error("qwen4_exp/layers/" + std::to_string(l) + ": " + error.what());
        }
    }
    auto scope     = work_.scope();
    Tensor final_x = mix(parameters_.final_mixer, residual, nullptr);
    Tensor last    = work_.alloc(DType::BF16, {H, batch.logit_columns.ne[0]});
    ops::gather_columns(final_x, batch.logit_columns, last, s);
    ops::projection_fp32(last, parameters_.output_head, logits, s);
}

void Forward::layer(std::uint32_t layer, const ForwardBatch& batch, Tensor& residual, const ForwardTap* tap) {
    const cudaStream_t s = device_.stream;
    const std::int32_t T = batch.ids.ne[0];
    const auto& block    = parameters_.layers[layer];
    auto scope           = work_.scope();
    if (block.ple) { ple(*block.ple, residual, batch); }
    Tensor inject = work_.alloc(DType::FP32, {dim(config_.hc.streams), T});
    Tensor xa     = mix(block.attn_hc, residual, &inject);
    const std::uint32_t compact = config_.compact_layer_indices[layer];
    Tensor y = config_.layer_types[layer] == MixerKind::Gdn
                   ? gdn(std::get<GdnParameters>(block.mixer), xa, compact)
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
    Tensor x = work_.alloc(DType::BF16, {H, T});
    ops::hyper_connection_mix(residual, p.norm, p.down.weight, p.up.weight, p.down.policy, dim(config_.hc.streams),
                              dim(config_.hc.rank), config_.rms_norm_eps, x, p.combine ? inject : nullptr, work_,
                              device_.stream);
    return x;
}

void Forward::ple(const PleParameters& p, Tensor& residual, const ForwardBatch& batch) {
    const cudaStream_t s = device_.stream;
    const std::int32_t T = residual.ne[1], W = residual.ne[0], H = dim(config_.hidden_size);
    auto scope = work_.scope();
    Tensor e   = work_.alloc(DType::BF16, {dim(config_.ple.embed_dim), T});
    ops::ple_embed(batch.ngram_rows, config_.ple.table.format, e, s);
    Tensor kv = work_.alloc(DType::BF16, {W + H, T});
    project(e, p.projection, kv, work_, s);
    Tensor key   = work_.alloc(DType::BF16, {W, T});
    Tensor value = work_.alloc(DType::BF16, {H, T});
    {
        Tensor* parts[] = {&key, &value};
        ops::split_rows(kv, parts, s);
    }
    Tensor gated      = work_.alloc(DType::BF16, {W, T});
    Tensor normalized = work_.alloc(DType::BF16, {W, T});
    ops::ple_gate(key, value, residual, p.key_norm, p.query_norm, p.conv_norm, dim(config_.hc.streams),
                  config_.rms_norm_eps, gated, normalized, s);
    // One sequence in state slot 0, updated in place.
    Tensor slot = work_.alloc(DType::I32, {1});
    CUDA_CHECK(cudaMemsetAsync(slot.data, 0, sizeof(std::int32_t), s));
    ops::ple_conv_inject(gated, normalized, p.convolution, dim(config_.ple.conv_dilation), state_.ple_conv, slot,
                         slot, residual, s);
}

Tensor Forward::gdn(const GdnParameters& p, const Tensor& x, std::uint32_t index) {
    const cudaStream_t s = device_.stream;
    const auto& g        = config_.gdn;
    const std::int32_t T = x.ne[1], H = dim(config_.hidden_size);
    const std::int32_t C = dim(g.conv_channels()), KW = dim(g.key_width()), VW = dim(g.value_width());
    const std::int32_t NV = dim(g.value_heads), NK = dim(g.key_heads), DK = dim(g.key_head_dim),
                       DV = dim(g.value_head_dim);
    Tensor qkv = work_.alloc(DType::BF16, {C, T});
    project(x, p.qkv, qkv, work_, s);
    Tensor z = work_.alloc(DType::BF16, {VW, T});
    project(x, p.z, z, work_, s);
    Tensor ab = work_.alloc(DType::BF16, {2 * NV, T});
    project(x, p.control, ab, work_, s);
    Tensor a = work_.alloc(DType::BF16, {NV, T});
    Tensor b = work_.alloc(DType::BF16, {NV, T});
    {
        Tensor* parts[] = {&a, &b};
        ops::split_rows(ab, parts, s);
    }
    Tensor gate = work_.alloc(DType::FP32, {NV, T});
    Tensor beta = work_.alloc(DType::FP32, {NV, T});
    ops::gdn_gating_decay(a, b, p.decay, p.dt_bias, gate, beta, s);

    Tensor q = work_.alloc(DType::BF16, {KW, T});
    Tensor k = work_.alloc(DType::BF16, {KW, T});
    Tensor v = work_.alloc(DType::BF16, {VW, T});
    Tensor& conv_state = state_.gdn_conv.at(index);
    ops::causal_conv1d_silu_split(qkv, p.convolution, conv_state, conv_state, q, k, v, s);
    Tensor qt = work_.alloc(DType::BF16, {DK, NV, T});
    Tensor kt = work_.alloc(DType::BF16, {DK, NV, T});
    tile_heads(q.view({DK, NK, T}), qt, s);
    tile_heads(k.view({DK, NK, T}), kt, s);
    Tensor o = work_.alloc(DType::BF16, {DV, NV, T});
    const float scale = static_cast<float>(1.0 / std::sqrt(static_cast<double>(DK)));
    ops::gated_delta_net(qt, kt, v.view({DV, NV, T}), gate, beta, scale, /*normalize_qk=*/true, work_,
                         state_.gdn_recurrent.at(index), o, device_.execution_view());
    Tensor normed = work_.alloc(DType::BF16, {DV, NV * T});
    ops::gated_rmsnorm(o.view({DV, NV * T}), p.norm, z.view({DV, NV * T}),
                       g.output_gate == GateActivation::Sigmoid ? ops::RmsGate::Sigmoid : ops::RmsGate::Silu,
                       config_.rms_norm_eps, normed, device_.execution_view());
    Tensor y = work_.alloc(DType::BF16, {H, T});
    project(normed.view({VW, T}), p.output, y, work_, s);
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
    Tensor qg = work_.alloc(DType::BF16, {2 * QW, T});
    project(x, p.query_gate, qg, work_, s);
    Tensor q    = work_.alloc(DType::BF16, {D, NH, T});
    Tensor gate = work_.alloc(DType::BF16, {D, NH, T});
    {
        Tensor qf = q.view({D, NH * T}), gf = gate.view({D, NH * T});
        Tensor* parts[] = {&qf, &gf};
        ops::split_rows(qg.view({2 * D, NH * T}), parts, s);
    }
    Tensor k = work_.alloc(DType::BF16, {D, KVH, T});
    Tensor v = work_.alloc(DType::BF16, {D, KVH, T});
    {
        Tensor kf = k.view({D * KVH, T}), vf = v.view({D * KVH, T});
        project(x, p.key, kf, work_, s);
        project(x, p.value, vf, work_, s);
    }
    // The indexer's BF16 rows in FP32, then rounded once to BF16 (Linear's BF16 output boundary).
    Tensor indexed = work_.alloc(DType::FP32, {IH * ID + ID, T});
    {
        const Tensor* rows[] = {&p.index_query, &p.index_key};
        ops::projection_fp32(x, rows, indexed, s);
    }
    Tensor index_rows = work_.alloc(DType::BF16, {IH * ID + ID, T});
    ops::cast_fp32_to_bf16(indexed, index_rows, s);
    Tensor iq = work_.alloc(DType::BF16, {ID, IH, T});
    Tensor ik = work_.alloc(DType::BF16, {ID, T});
    {
        Tensor iqf      = iq.view({IH * ID, T});
        Tensor* parts[] = {&iqf, &ik};
        ops::split_rows(index_rows, parts, s);
    }
    Tensor qn = work_.alloc(DType::BF16, {D, NH, T});
    Tensor kn = work_.alloc(DType::BF16, {D, KVH, T});
    {
        Tensor qn_rows = qn.view({D, NH * T}), kn_rows = kn.view({D, KVH * T});
        ops::rmsnorm(q.view({D, NH * T}), p.query_norm, config_.rms_norm_eps, true, qn_rows, s);
        ops::rmsnorm(k.view({D, KVH * T}), p.key_norm, config_.rms_norm_eps, true, kn_rows, s);
    }
    // Text tokens: the three RoPE axes are equal, so interleaved MRoPE is the 1-D rotation.
    ops::rope(batch.positions, dim(config_.rope.rotary_dim), config_.rope.theta, qn, kn, device_.execution_view());

    const PagedKVLayerView& cache = kv_.layers.at(index);
    ops::kv_cache_append(kn, v, batch.positions, cache, s);
    const auto geometry = qsa_geometry(config_);
    ops::qsa_pool_keys(ik, batch.positions, batch.rope_positions, batch.block_start_rope, p.index_key_norm, geometry,
                       kv_.block_table, state_.qsa_tails.at(index), kv_.pooled.at(index), s);
    ops::qsa_index_query(batch.rope_positions, p.index_query_norm, geometry, iq, s);
    const std::int32_t top = dim(config_.qsa.budget / config_.qsa.compress_ratio);
    Tensor selected = work_.alloc(DType::I32, {top, T});
    Tensor counts   = work_.alloc(DType::I32, {T});
    ops::qsa_select_blocks(iq, batch.positions, kv_.block_table, kv_.pooled.at(index), geometry,
                           {kv_.max_visible_keys}, work_, selected, counts, device_.execution_view());
    Tensor out = work_.alloc(DType::BF16, {D, NH, T});
    ops::qsa_attention(qn, batch.positions, selected, counts, {D, NH, KVH}, geometry,
                       static_cast<float>(1.0 / std::sqrt(static_cast<double>(D))), cache, work_, out,
                       device_.execution_view());
    // Output gate: out * sigmoid(gate), then o_proj.
    ops::sigmoid_mul(gate, out, s);
    Tensor y = work_.alloc(DType::BF16, {H, T});
    project(out.view({QW, T}), p.output, y, work_, s);
    return y;
}

Tensor Forward::moe(const MoeParameters& p, const Tensor& x, std::uint32_t layer, Tensor* route_tap) {
    const cudaStream_t s = device_.stream;
    const auto& m        = config_.moe;
    const std::int32_t T = x.ne[1], H = dim(config_.hidden_size), E = dim(m.experts), K = dim(m.top_k);
    Tensor logits = work_.alloc(DType::FP32, {E + 1, T});
    {
        const Tensor* rows[] = {&p.router, &p.shared_score};
        ops::projection_fp32(x, rows, logits, s);
    }
    ops::MoeRouting routing{work_.alloc(DType::I32, {K, T}), work_.alloc(DType::FP32, {K, T}),
                            work_.alloc(DType::FP32, {T})};
    ops::moe_route(logits, K, routing, s);
    if (route_tap != nullptr &&
        cudaMemcpyAsync(route_tap->data, routing.ids.data, routing.ids.bytes(), cudaMemcpyDeviceToDevice, s) !=
            cudaSuccess) {
        throw std::runtime_error("Qwen4Exp forward: route tap copy failed");
    }
    const DeviceSpan dispatch_bytes = work_.alloc_bytes(ops::moe_dispatch_bytes(E, K * T));
    ops::MoeDispatch dispatch       = ops::carve_moe_dispatch(dispatch_bytes.data, E, K * T);
    ops::moe_dispatch(routing, E, dispatch, nullptr, s);
    ops::MoeExpertSource source{.format        = p.bank->format,
                                .frame_base    = experts_.frame_base,
                                .frames        = experts_.frames.at(layer),
                                .host_records  = p.bank->device_records,
                                .record_stride = p.bank->record_stride,
                                .error         = experts_.error,
                                .staging_base  = experts_.staging_base,
                                .staging_slots = experts_.staging_slots};
    Tensor outputs              = work_.alloc(DType::BF16, {H, K * T});
    const std::int32_t max_jobs = std::min(E, K * T);
    const DeviceSpan expert_ws  = work_.alloc_bytes(ops::moe_experts_workspace_bytes(max_jobs, K * T, T));
    ops::moe_experts(x, dispatch, source, K, max_jobs, expert_ws.data, outputs, s);

    const std::int32_t I = dim(m.shared_intermediate);
    Tensor g = work_.alloc(DType::BF16, {I, T});
    Tensor u = work_.alloc(DType::BF16, {I, T});
    project(x, p.shared_gate, g, work_, s);
    project(x, p.shared_up, u, work_, s);
    Tensor product = work_.alloc(DType::BF16, {I, T});
    ops::silu_mul(g, u, product, s);
    Tensor shared = work_.alloc(DType::BF16, {H, T});
    project(product, p.shared_down, shared, work_, s);
    Tensor y = work_.alloc(DType::BF16, {H, T});
    ops::moe_combine(outputs, routing, shared, y, s);
    return y;
}

} // namespace ninfer::models::qwen4_exp::execution
