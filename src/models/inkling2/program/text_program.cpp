// Written for NInfer-3090. See text_program.h; the order of operations is the spec's ("Forward, in order") and
// tools/inkling2/reference.py Model.forward, which the golden comparison ties to llama.cpp.

#include "models/inkling2/program/text_program.h"

#include "ninfer/ops/cast.h"
#include "ninfer/ops/depth_attention.h"
#include "ninfer/ops/fp32_rows.h"
#include "ninfer/ops/moe_situ.h"
#include "ninfer/ops/projection_fp32.h"
#include "ninfer/ops/relative_bias_attention.h"
#include "ninfer/ops/short_conv.h"

#include <array>
#include <cstddef>
#include <stdexcept>
#include <string>

namespace ninfer::models::inkling2 {
namespace {

constexpr std::size_t kAlign = 256;

std::size_t padded(std::size_t bytes) { return (bytes + kAlign - 1) / kAlign * kAlign; }

std::int32_t i32(std::uint32_t v) { return static_cast<std::int32_t>(v); }

void sync(cudaStream_t stream) {
    if (cudaStreamSynchronize(stream) != cudaSuccess) { throw std::runtime_error("inkling2: stream synchronization failed"); }
}

// Every allocation one token costs in a forward call (persistent across visits and per visit), with slack.
std::size_t workspace_bytes(const TextConfig& c, std::int32_t chunk) {
    const std::size_t h = c.hidden_size, q = std::size_t(c.heads) * c.head_dim, kv = std::size_t(c.kv_heads) * c.head_dim;
    const std::size_t per_visit_kept = 6 * h * 4; // ffn_inp, the output and a core visit's four loop messages
    const std::size_t per_visit_temp = 12 * h * 4 + 3 * h * 2 + (2 * q + std::size_t(c.heads) * (c.d_rel + 1)) * 4 +
                                       4 * (kv + 64) * 4 + 2 * q * 4 + q * 2 + 4 * h * 2 + 4 * h * 4 + 64 * 4 * 2 +
                                       std::size_t(c.experts_per_token) * (c.moe_intermediate + h) * 4 + 4 * c.experts_per_token * 4 +
                                       ops::moe_route_sigmoid_topk_workspace_bytes(i32(c.experts), 1) +
                                       ops::depth_attention_mix_workspace_bytes(i32(c.hidden_size), 1, ops::kDepthAttentionMaxSources);
    const std::size_t output = std::size_t(c.vocab_size) * 4 + 4 * h * 4;
    const std::size_t per_token = c.num_layer_visits * per_visit_kept + per_visit_temp + output;
    // Each allocation pads to 256 bytes; a visit makes about 60.
    return per_token * static_cast<std::size_t>(chunk) + (c.num_layer_visits * 2 + 64) * kAlign * 4 + (64U << 20);
}

// The limits, validated before anything is sized from them.
ProgramLimits checked(ProgramLimits limits) {
    if (limits.max_chunk < 1 || limits.max_context < limits.max_chunk) { throw std::invalid_argument("inkling2: invalid program limits"); }
    return limits;
}

} // namespace

TextProgram::TextProgram(const Model& model, ProgramLimits limits, cudaStream_t stream)
    : model_(model), config_(model.config()), limits_(limits), stream_(stream),
      workspace_(workspace_bytes(model.config(), checked(limits).max_chunk)) {
    const TextConfig& c = config_;
    const std::size_t kv = std::size_t(c.kv_heads) * c.head_dim, taps = c.conv_kernel_size - 1;
    const std::size_t local_cap = c.sliding_window - 1 + static_cast<std::size_t>(limits.max_chunk);
    std::size_t total = 0;
    std::vector<std::size_t> capacity(c.num_layer_visits);
    for (std::uint32_t v = 0; v < c.num_layer_visits; ++v) {
        capacity[v] = c.local[c.physical(v)] ? local_cap : static_cast<std::size_t>(limits.max_context);
        total += 2 * padded(kv * capacity[v] * 4) + 2 * padded(kv * taps * 4) + 2 * padded(std::size_t(c.hidden_size) * taps * 4);
    }
    state_memory_ = DeviceBuffer(total);
    auto* base = static_cast<std::byte*>(state_memory_.p);
    std::size_t offset = 0;
    const auto carve = [&](std::int32_t rows, std::int32_t cols) {
        Tensor t(base + offset, DType::FP32, {rows, cols});
        offset += padded(t.bytes());
        return t;
    };
    for (std::uint32_t v = 0; v < c.num_layer_visits; ++v) {
        VisitState s;
        s.k_cache   = carve(i32(kv), i32(capacity[v]));
        s.v_cache   = carve(i32(kv), i32(capacity[v]));
        s.k_hist    = carve(i32(kv), i32(taps));
        s.v_hist    = carve(i32(kv), i32(taps));
        s.attn_hist = carve(i32(c.hidden_size), i32(taps));
        s.mlp_hist  = carve(i32(c.hidden_size), i32(taps));
        state_.push_back(s);
    }
    reset();
}

TextProgram::~TextProgram() {
    if (decode_graph_ != nullptr) { cudaGraphExecDestroy(decode_graph_); }
}

void TextProgram::reset() {
    for (auto& s : state_) {
        for (Tensor* t : {&s.k_hist, &s.v_hist, &s.attn_hist, &s.mlp_hist}) {
            if (cudaMemsetAsync(t->data, 0, t->bytes(), stream_) != cudaSuccess) { throw std::runtime_error("inkling2: cannot clear state"); }
        }
    }
    position_ = 0;
}

Tensor TextProgram::project(const Tensor& x, std::span<const WeightId> weights, std::span<const std::int32_t> rows) {
    const std::int32_t k = x.ne[0], cols = x.ne[1];
    std::array<Tensor, 4> views;
    std::array<const Tensor*, 4> pointers{};
    std::int32_t total = 0;
    for (std::size_t i = 0; i < weights.size(); ++i) {
        views[i]    = model_.tensor(weights[i], {k, rows[i]});
        pointers[i] = &views[i];
        total += rows[i];
    }
    Tensor out = workspace_.alloc(DType::FP32, {total, cols});
    ops::projection_fp32(x, std::span<const Tensor* const>(pointers.data(), weights.size()), out, stream_);
    return out;
}

void TextProgram::control(const ControllerWeights& w, const Tensor& c, const Tensor& m, Tensor& out) {
    const std::int32_t h = c.ne[0], cols = c.ne[1], rank = i32(config_.loop_gate_rank);
    const auto scope = workspace_.scope();
    Tensor both = workspace_.alloc(DType::BF16, {2 * h, cols});
    Tensor first = both.slice(0, 0, h), second = both.slice(0, h, h);
    ops::rmsnorm_fp32(c, nullptr, config_.rms_norm_eps, first, stream_);
    ops::rmsnorm_fp32(m, nullptr, config_.rms_norm_eps, second, stream_);
    const WeightId down[] = {w.down};
    const std::int32_t down_rows[] = {rank};
    Tensor hidden = project(both, down, down_rows);
    Tensor hidden_bf16 = workspace_.alloc(DType::BF16, {rank, cols});
    ops::cast_fp32_to_bf16(hidden, hidden_bf16, stream_);
    const WeightId gates[] = {w.gain, w.forget};
    const std::int32_t gate_rows[] = {h, h};
    Tensor gf = project(hidden_bf16, gates, gate_rows);
    ops::forget_gain_combine(c, m, gf, out, stream_);
}

Tensor TextProgram::visit(std::uint32_t v, const Tensor& cur, std::vector<Tensor>& blocks, const Tensor* messages, std::int64_t start,
                          const Trace* trace) {
    const TextConfig& c   = config_;
    const std::uint32_t p = c.physical(v);
    const LayerWeights& w = model_.weights().layers.at(p);
    VisitState& st        = state_.at(v);
    const bool local      = c.local[p];
    const std::int32_t h = i32(c.hidden_size), cols = cur.ne[1], heads = i32(c.heads), d = i32(c.head_dim),
                       kv = i32(c.kv_heads * c.head_dim), q = heads * d, rel = heads * i32(c.d_rel), k = i32(c.conv_kernel_size);
    const float eps = c.rms_norm_eps;

    // The visit's outputs outlive it (blocks and the loop bookkeeping keep them); everything else is scoped.
    Tensor ffn_inp = workspace_.alloc(DType::FP32, {h, cols});
    Tensor next    = workspace_.alloc(DType::FP32, {h, cols});
    const auto scope = workspace_.scope();

    // 3b-3d: attention inputs by depth attention, loop messages, input norms.
    Tensor xq = cur, xk = cur, xv = cur;
    if (p > 0) {
        std::vector<const Tensor*> sources;
        for (const Tensor& b : blocks) { sources.push_back(&b); }
        sources.push_back(&cur);
        if (local) {
            xq = xk = xv = workspace_.alloc(DType::FP32, {h, cols});
            Tensor* outs[] = {&xq};
            ops::depth_attention_mix(sources, model_.tensor(w.res_query, {h, 1}), eps, workspace_, outs, stream_);
        } else {
            xq = workspace_.alloc(DType::FP32, {h, cols});
            xk = workspace_.alloc(DType::FP32, {h, cols});
            xv = workspace_.alloc(DType::FP32, {h, cols});
            Tensor* outs[] = {&xq, &xk, &xv};
            ops::depth_attention_mix(sources, model_.tensor(w.res_query, {h, 3}), eps, workspace_, outs, stream_);
        }
    }
    if (messages != nullptr) {
        Tensor cq = workspace_.alloc(DType::FP32, {h, cols}), ck = workspace_.alloc(DType::FP32, {h, cols}),
               cv = workspace_.alloc(DType::FP32, {h, cols});
        control(w.loop[0], xq, messages[0], cq);
        control(w.loop[1], xk, messages[1], ck);
        control(w.loop[2], xv, messages[2], cv);
        xq = cq;
        xk = ck;
        xv = cv;
    }
    Tensor nq = workspace_.alloc(DType::BF16, {h, cols}), nk = workspace_.alloc(DType::BF16, {h, cols}),
           nv = workspace_.alloc(DType::BF16, {h, cols});
    if (local) {
        const Tensor gamma = model_.tensor(w.norm, {h});
        ops::rmsnorm_fp32(xq, &gamma, eps, nq, stream_);
        ops::rmsnorm_fp32(xk, &gamma, eps, nk, stream_);
        ops::rmsnorm_fp32(xv, &gamma, eps, nv, stream_);
    } else {
        const Tensor gq = model_.tensor(w.q_norm, {h}), gk = model_.tensor(w.k_norm, {h}), gv = model_.tensor(w.v_norm, {h});
        ops::rmsnorm_fp32(xq, &gq, eps, nq, stream_);
        ops::rmsnorm_fp32(xk, &gk, eps, nk, stream_);
        ops::rmsnorm_fp32(xv, &gv, eps, nv, stream_);
    }

    // 3e: projections and short convolutions.
    const WeightId q_weights[] = {w.q, w.rel, w.gate, w.sink};
    const std::int32_t q_rows[] = {q, rel, q, heads};
    Tensor pq = project(nq, q_weights, q_rows);
    Tensor q_t = pq.slice(0, 0, q), r_t = pq.slice(0, q, rel), gate = pq.slice(0, q + rel, q), sink = pq.slice(0, 2 * q + rel, heads);
    const std::int32_t dyn_rows = i32(c.kv_heads) * k;
    Tensor kconv = workspace_.alloc(DType::FP32, {kv, cols}), vconv = workspace_.alloc(DType::FP32, {kv, cols});
    if (local) {
        const WeightId kw[] = {w.k}, vw[] = {w.v};
        const std::int32_t kr[] = {kv};
        Tensor pk = project(nk, kw, kr), pv = project(nv, vw, kr);
        ops::short_conv_fp32(pk, model_.tensor(w.k_conv, {k, kv}), nullptr, d, st.k_hist, kconv, stream_);
        ops::short_conv_fp32(pv, model_.tensor(w.v_conv, {k, kv}), nullptr, d, st.v_hist, vconv, stream_);
    } else {
        const WeightId kw[] = {w.k, w.k_conv_proj}, vw[] = {w.v, w.v_conv_proj};
        const std::int32_t kr[] = {kv, dyn_rows};
        Tensor pk = project(nk, kw, kr), pv = project(nv, vw, kr);
        const Tensor kd = pk.slice(0, kv, dyn_rows), vd = pv.slice(0, kv, dyn_rows);
        ops::short_conv_fp32(pk.slice(0, 0, kv), model_.tensor(w.k_conv, {k, kv}), &kd, d, st.k_hist, kconv, stream_);
        ops::short_conv_fp32(pv.slice(0, 0, kv), model_.tensor(w.v_conv, {k, kv}), &vd, d, st.v_hist, vconv, stream_);
    }

    // 3f: clips, value norm, log scaling of the global layers' q and r.
    const ops::PositionLogScale tau{start, c.log_scaling_n_floor, c.log_scaling_alpha, start_offset_};
    const Tensor q_clip = model_.tensor(w.q_clip, {d, heads}), rel_clip = model_.tensor(w.rel_clip, {i32(c.d_rel), heads});
    ops::head_l2_clip_fp32(q_t, d, &q_clip, c.qk_clip, local ? nullptr : &tau, stream_);
    ops::head_l2_clip_fp32(r_t, i32(c.d_rel), &rel_clip, c.rel_clip, local ? nullptr : &tau, stream_);
    ops::head_l2_clip_fp32(kconv, d, nullptr, c.qk_clip, nullptr, stream_);
    ops::head_unit_rms_fp32(vconv, d, eps, stream_);

    // 3g: attention, output gate, output projection and its convolution.
    ops::RelativeBiasAttentionShape shape{heads, i32(c.kv_heads), d, i32(c.d_rel), local ? i32(c.sliding_window) : 0};
    Tensor o = workspace_.alloc(DType::FP32, {q, cols});
    const Tensor rel_bias = model_.tensor(w.rel_bias, {i32(c.d_rel), i32(c.rel_extent_of(p))});
    ops::relative_bias_sink_attention(shape, q_t, r_t, sink, kconv, vconv, rel_bias, st.k_cache, st.v_cache, start, start_offset_, o, stream_);
    Tensor gated = workspace_.alloc(DType::BF16, {q, cols});
    ops::sigmoid_gate_fp32(o, gate, gated, stream_);
    const WeightId ow[] = {w.output};
    const std::int32_t orows[] = {h};
    Tensor a_in = project(gated, ow, orows);
    Tensor a = workspace_.alloc(DType::FP32, {h, cols});
    ops::short_conv_fp32(a_in, model_.tensor(w.post_conv, {k, h}), nullptr, 1, st.attn_hist, a, stream_);

    // 3h: residual and block boundary.
    if (p % c.attn_res_block_size == 0) {
        blocks.push_back(cur);
        if (cudaMemcpyAsync(ffn_inp.data, a.data, a.bytes(), cudaMemcpyDeviceToDevice, stream_) != cudaSuccess) {
            throw std::runtime_error("inkling2: copy failed");
        }
    } else {
        ops::add_fp32(cur, a, ffn_inp, stream_);
    }

    // 3i-3j: MoE input by depth attention, loop message, norm, routed experts, output convolution.
    std::vector<const Tensor*> sources;
    for (const Tensor& b : blocks) { sources.push_back(&b); }
    sources.push_back(&ffn_inp);
    Tensor z = workspace_.alloc(DType::FP32, {h, cols});
    {
        Tensor* outs[] = {&z};
        ops::depth_attention_mix(sources, model_.tensor(w.ffn_res_query, {h, 1}), eps, workspace_, outs, stream_);
    }
    if (messages != nullptr) {
        Tensor zc = workspace_.alloc(DType::FP32, {h, cols});
        control(w.loop[3], z, messages[3], zc);
        z = zc;
    }
    const Tensor ffn_norm = model_.tensor(w.ffn_norm, {h});
    ops::rmsnorm_fp32(z, &ffn_norm, eps, z, stream_);
    const std::int32_t top_k = i32(c.experts_per_token);
    Tensor ids = workspace_.alloc(DType::I32, {top_k, cols}), weights = workspace_.alloc(DType::FP32, {top_k, cols});
    ops::moe_route_sigmoid_topk(z, model_.tensor(w.router, {h, i32(c.experts)}), model_.tensor(w.router_bias, {i32(c.experts)}), top_k,
                                c.route_scale, workspace_, ids, weights, stream_);
    Tensor y = workspace_.alloc(DType::FP32, {h, cols});
    ops::moe_situ_experts(model_.expert_bank(p), z, ids, weights, c.gate_softcap, c.up_softcap, workspace_, y, stream_);
    Tensor mo = workspace_.alloc(DType::FP32, {h, cols});
    ops::short_conv_fp32(y, model_.tensor(w.mlp_post_conv, {k, h}), nullptr, 1, st.mlp_hist, mo, stream_);
    ops::add_fp32(ffn_inp, mo, next, stream_);

    if (trace != nullptr && *trace) {
        sync(stream_);
        (*trace)("attn_out-" + std::to_string(v), a);
        (*trace)("ffn_inp-" + std::to_string(v), ffn_inp);
        (*trace)("l_out-" + std::to_string(v), next);
        (*trace)("route_ids-" + std::to_string(v), ids);
    }
    return next;
}

Tensor TextProgram::forward(std::span<const std::int32_t> tokens, const Trace* trace) {
    const TextConfig& c = config_;
    const std::int32_t cols = static_cast<std::int32_t>(tokens.size()), h = i32(c.hidden_size);
    if (cols < 1 || cols > limits_.max_chunk) { throw std::invalid_argument("inkling2: chunk size outside 1..max_chunk"); }
    if (position_ + cols > limits_.max_context) { throw std::invalid_argument("inkling2: the sequence would pass max_context"); }
    for (const auto token : tokens) {
        if (token < 0 || static_cast<std::uint32_t>(token) >= c.unpadded_vocab_size) { throw std::invalid_argument("inkling2: token id out of range"); }
    }
    const bool tracing = trace != nullptr && *trace;
    if (cols == 1 && !tracing && limits_.decode_graph && stream_ != nullptr) { return forward_graph(tokens[0]); }
    workspace_.reset();
    Tensor ids = workspace_.alloc(DType::I32, {cols});
    if (cudaMemcpyAsync(ids.data, tokens.data(), tokens.size_bytes(), cudaMemcpyHostToDevice, stream_) != cudaSuccess) {
        throw std::runtime_error("inkling2: token upload failed");
    }
    Tensor logits = run(ids, position_, trace);
    position_ += cols;
    return logits;
}

Tensor TextProgram::forward_graph(std::int32_t token) {
    struct Step {
        std::int64_t position;
        std::int32_t token, pad;
    };
    const Step host{position_, token, 0};
    if (step_.bytes == 0) { step_ = DeviceBuffer(sizeof(Step)); }
    // From pageable memory: the copy is staged before the call returns, so `host` may go out of scope.
    if (cudaMemcpyAsync(step_.p, &host, sizeof host, cudaMemcpyHostToDevice, stream_) != cudaSuccess) {
        throw std::runtime_error("inkling2: step upload failed");
    }
    if (decode_graph_ == nullptr) {
        workspace_.reset();
        const Tensor ids(static_cast<char*>(step_.p) + offsetof(Step, token), DType::I32, {1});
        if (cudaStreamBeginCapture(stream_, cudaStreamCaptureModeThreadLocal) != cudaSuccess) {
            throw std::runtime_error("inkling2: cannot capture the decode graph");
        }
        start_offset_ = static_cast<const std::int64_t*>(step_.p);
        cudaGraph_t graph = nullptr;
        try {
            graph_logits_ = run(ids, 0, nullptr);
        } catch (...) {
            start_offset_ = nullptr;
            if (cudaStreamEndCapture(stream_, &graph) == cudaSuccess && graph != nullptr) { cudaGraphDestroy(graph); }
            throw;
        }
        start_offset_ = nullptr;
        if (cudaStreamEndCapture(stream_, &graph) != cudaSuccess) { throw std::runtime_error("inkling2: decode graph capture failed"); }
        const cudaError_t made = cudaGraphInstantiate(&decode_graph_, graph, 0);
        cudaGraphDestroy(graph);
        if (made != cudaSuccess) {
            decode_graph_ = nullptr;
            throw std::runtime_error(std::string("inkling2: cannot instantiate the decode graph: ") + cudaGetErrorString(made));
        }
    }
    if (cudaGraphLaunch(decode_graph_, stream_) != cudaSuccess) { throw std::runtime_error("inkling2: decode graph launch failed"); }
    position_ += 1;
    return graph_logits_;
}

Tensor TextProgram::run(const Tensor& ids, std::int64_t start, const Trace* trace) {
    const TextConfig& c = config_;
    const std::int32_t cols = ids.ne[0], h = i32(c.hidden_size);

    // 1: embedding and its norm.
    Tensor cur = workspace_.alloc(DType::FP32, {h, cols});
    ops::embedding_bf16_fp32(ids, model_.tensor(model_.weights().embedding, {h, i32(c.vocab_size)}), cur, stream_);
    const Tensor embed_norm = model_.tensor(model_.weights().embedding_norm, {h});
    ops::rmsnorm_fp32(cur, &embed_norm, c.rms_norm_eps, cur, stream_);

    // 3: the visits, with the loop bookkeeping of 3a.
    std::vector<Tensor> blocks, prelude_blocks, bank;
    Tensor prelude_cur, bank_last;
    const Tensor loop_q = model_.tensor(model_.weights().loop_res_query, {h, i32(2 * c.core_layers())});
    const Tensor loop_kv = model_.tensor(model_.weights().loop_kv_res_query, {h, i32(2 * c.core_layers())});
    for (std::uint32_t v = 0; v < c.num_layer_visits; ++v) {
        const std::uint32_t p = c.physical(v), pass = c.pass(v);
        const bool core = c.is_core(p);
        if (core && p == c.loop_core_start) {
            if (pass == 0) {
                prelude_blocks = blocks;
                prelude_cur    = cur;
            } else {
                bank.assign(blocks.begin() + static_cast<std::ptrdiff_t>(prelude_blocks.size()) + 1, blocks.end());
                bank_last = cur;
                blocks    = prelude_blocks;
                cur       = prelude_cur;
            }
        }
        std::array<Tensor, 4> messages; // attn, k, v, mlp
        const bool has_messages = core && pass > 0;
        if (has_messages) {
            const std::int32_t j = i32(p - c.loop_core_start);
            for (Tensor& m : messages) { m = workspace_.alloc(DType::FP32, {h, cols}); }
            std::vector<const Tensor*> sources;
            for (const Tensor& b : bank) { sources.push_back(&b); }
            sources.push_back(&bank_last);
            Tensor* attn_mlp[] = {&messages[0], &messages[3]};
            Tensor* k_v[]      = {&messages[1], &messages[2]};
            ops::depth_attention_mix(sources, loop_q.slice(1, 2 * j, 2), c.rms_norm_eps, workspace_, attn_mlp, stream_);
            ops::depth_attention_mix(sources, loop_kv.slice(1, 2 * j, 2), c.rms_norm_eps, workspace_, k_v, stream_);
        }
        cur = visit(v, cur, blocks, has_messages ? messages.data() : nullptr, start, trace);
    }

    // 4: output.
    std::vector<const Tensor*> sources;
    for (const Tensor& b : blocks) { sources.push_back(&b); }
    sources.push_back(&cur);
    Tensor x = workspace_.alloc(DType::FP32, {h, cols});
    {
        Tensor* outs[] = {&x};
        const Tensor output_query = model_.tensor(model_.weights().output_res_query, {h, 1});
        ops::depth_attention_mix(sources, output_query, c.rms_norm_eps, workspace_, outs, stream_);
    }
    const Tensor out_norm = model_.tensor(model_.weights().output_norm, {h});
    ops::rmsnorm_fp32(x, &out_norm, c.rms_norm_eps, x, stream_);
    if (trace != nullptr && *trace) {
        sync(stream_);
        (*trace)("result_norm", x);
    }
    Tensor xb = workspace_.alloc(DType::BF16, {h, cols});
    ops::cast_fp32_to_bf16(x, xb, stream_);
    const WeightId head[] = {model_.weights().head};
    const std::int32_t head_rows[] = {i32(c.vocab_size)};
    Tensor logits = project(xb, head, head_rows);
    ops::tanh_softcap_fp32(logits, c.final_logit_softcap, logits, stream_);
    return logits;
}

} // namespace ninfer::models::inkling2
