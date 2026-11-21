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

#include "core/copy_batch.h"

#include <array>
#include <cstddef>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>

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
                                       ops::depth_attention_mix_workspace_bytes(i32(c.hidden_size), 1, ops::kDepthAttentionMaxSources) +
                                       ops::loop_controller_workspace_bytes(i32(c.hidden_size), i32(c.loop_gate_rank), 1,
                                                                            ops::kLoopControllerMaxJobs);
    const std::size_t output = std::size_t(c.vocab_size) * 4 + 4 * h * 4;
    const std::size_t per_token = c.num_layer_visits * per_visit_kept + per_visit_temp + output;
    // Each allocation pads to 256 bytes; a visit makes about 60.
    return per_token * static_cast<std::size_t>(chunk) + (c.num_layer_visits * 2 + 64) * kAlign * 4 + (64U << 20);
}

// The limits, validated before anything is sized from them.
ProgramLimits checked(ProgramLimits limits) {
    if (limits.max_chunk < 1 || limits.max_context < limits.max_chunk) {
        throw std::invalid_argument("inkling2: invalid program limits (max_chunk positive, at most max_context)");
    }
    if (limits.max_rollback < 0 || limits.max_rollback > limits.max_chunk) {
        throw std::invalid_argument("inkling2: max_rollback must be in 0..max_chunk");
    }
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
        if (limits.max_rollback > 0) {
            const std::size_t saved = taps + static_cast<std::size_t>(limits.max_rollback);
            total += 2 * padded(kv * saved * 4) + 2 * padded(std::size_t(c.hidden_size) * saved * 4);
        }
    }
    for (const auto tap : limits.tap_visits) {
        if (tap > c.num_layer_visits) { throw std::invalid_argument("inkling2: a tap visit past the final state"); }
        total += padded(std::size_t(c.hidden_size) * static_cast<std::size_t>(limits.max_chunk) * 4);
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
        if (limits.max_rollback > 0) {
            const std::int32_t saved = i32(taps) + limits.max_rollback;
            s.k_saved    = carve(i32(kv), saved);
            s.v_saved    = carve(i32(kv), saved);
            s.attn_saved = carve(i32(c.hidden_size), saved);
            s.mlp_saved  = carve(i32(c.hidden_size), saved);
        }
        state_.push_back(s);
    }
    for (std::size_t i = 0; i < limits.tap_visits.size(); ++i) { taps_.push_back(carve(i32(c.hidden_size), limits.max_chunk)); }
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
    position_  = 0;
    last_cols_ = 0;
}

namespace {
// Calls f(slot, count) for the ring slots of positions [first, first + n) of a ring of `capacity` (at most two runs).
template <class F> void ring_runs(std::int64_t first, std::int64_t n, std::int64_t capacity, F&& f) {
    while (n > 0) {
        const std::int64_t slot = first % capacity, run = std::min(n, capacity - slot);
        f(slot, run);
        first += run, n -= run;
    }
}
} // namespace

std::size_t TextProgram::snapshot_bytes() const {
    std::size_t total = 0;
    for (std::uint32_t v = 0; v < config_.num_layer_visits; ++v) {
        const VisitState& s = state_[v];
        total += s.k_hist.bytes() + s.v_hist.bytes() + s.attn_hist.bytes() + s.mlp_hist.bytes();
        if (config_.local[config_.physical(v)]) {
            total += 2 * static_cast<std::size_t>(s.k_cache.ne[0]) * 4 * (config_.sliding_window - 1);
        }
    }
    return total;
}

void TextProgram::save(Snapshot& snapshot) const {
    if (snapshot.memory.bytes < snapshot_bytes()) { throw std::invalid_argument("inkling2: the snapshot is too small"); }
    auto* out = static_cast<std::byte*>(snapshot.memory.p);
    CopyBatch copies(stream_);
    const std::int64_t first = std::max<std::int64_t>(0, position_ - (config_.sliding_window - 1));
    for (std::uint32_t v = 0; v < config_.num_layer_visits; ++v) {
        const VisitState& s = state_[v];
        for (const Tensor* t : {&s.k_hist, &s.v_hist, &s.attn_hist, &s.mlp_hist}) {
            copies.add(out, t->data, t->bytes());
            out += t->bytes();
        }
        if (!config_.local[config_.physical(v)]) { continue; }
        const std::size_t column = static_cast<std::size_t>(s.k_cache.ne[0]) * 4;
        for (const Tensor* t : {&s.k_cache, &s.v_cache}) {
            auto* base = out;
            ring_runs(first, position_ - first, t->ne[1], [&](std::int64_t slot, std::int64_t run) {
                copies.add(base, static_cast<const std::byte*>(t->data) + static_cast<std::size_t>(slot) * column,
                           static_cast<std::size_t>(run) * column);
                base += static_cast<std::size_t>(run) * column;
            });
            out += column * (config_.sliding_window - 1);
        }
    }
    copies.flush();
    snapshot.position = position_;
}

void TextProgram::restore(const Snapshot& snapshot) {
    if (snapshot.position < 0 || snapshot.position > limits_.max_context) { throw std::invalid_argument("inkling2: an empty snapshot"); }
    const auto* in = static_cast<const std::byte*>(snapshot.memory.p);
    CopyBatch copies(stream_);
    const std::int64_t first = std::max<std::int64_t>(0, snapshot.position - (config_.sliding_window - 1));
    for (std::uint32_t v = 0; v < config_.num_layer_visits; ++v) {
        VisitState& s = state_[v];
        for (Tensor* t : {&s.k_hist, &s.v_hist, &s.attn_hist, &s.mlp_hist}) {
            copies.add(t->data, in, t->bytes());
            in += t->bytes();
        }
        if (!config_.local[config_.physical(v)]) { continue; }
        const std::size_t column = static_cast<std::size_t>(s.k_cache.ne[0]) * 4;
        for (Tensor* t : {&s.k_cache, &s.v_cache}) {
            const auto* base = in;
            ring_runs(first, snapshot.position - first, t->ne[1], [&](std::int64_t slot, std::int64_t run) {
                copies.add(static_cast<std::byte*>(t->data) + static_cast<std::size_t>(slot) * column, base,
                           static_cast<std::size_t>(run) * column);
                base += static_cast<std::size_t>(run) * column;
            });
            in += column * (config_.sliding_window - 1);
        }
    }
    copies.flush();
    position_  = snapshot.position;
    last_cols_ = 0;
}

Tensor TextProgram::tap(std::size_t i) const {
    if (i >= taps_.size() || last_cols_ < 1) { throw std::out_of_range("inkling2: no such tap"); }
    return taps_[i].slice(1, 0, last_cols_);
}

void TextProgram::copy_tap(std::uint32_t visit, const Tensor& value) {
    for (std::size_t i = 0; i < taps_.size(); ++i) {
        if (limits_.tap_visits[i] != visit) { continue; }
        const std::size_t column = std::size_t(config_.hidden_size) * 4;
        if (cudaMemcpy2DAsync(taps_[i].data, column, value.data, static_cast<std::size_t>(value.nb[1]), column, value.ne[1],
                              cudaMemcpyDeviceToDevice, stream_) != cudaSuccess) {
            throw std::runtime_error("inkling2: tap copy failed");
        }
    }
}

void TextProgram::save_input(Tensor& saved, const Tensor& x) {
    if (!saving_) { return; }
    const std::size_t column = static_cast<std::size_t>(x.ne[0]) * 4, taps = config_.conv_kernel_size - 1;
    if (cudaMemcpy2DAsync(static_cast<std::byte*>(saved.data) + taps * column, column, x.data, static_cast<std::size_t>(x.nb[1]),
                          column, x.ne[1], cudaMemcpyDeviceToDevice, stream_) != cudaSuccess) {
        throw std::runtime_error("inkling2: cannot save a convolution input");
    }
}

void TextProgram::keep(std::int32_t n) {
    if (n < 1 || n > last_cols_) { throw std::invalid_argument("inkling2: keep() outside 1..the last forward's tokens"); }
    if (n == last_cols_) { return; }
    if (last_cols_ > limits_.max_rollback) { throw std::invalid_argument("inkling2: the last forward cannot be cut back"); }
    // History after the first n inputs: columns n .. n + K - 2 of [old history | inputs].
    CopyBatch copies(stream_);
    for (auto& s : state_) {
        for (auto [hist, saved] : {std::pair{&s.k_hist, &s.k_saved}, std::pair{&s.v_hist, &s.v_saved},
                                   std::pair{&s.attn_hist, &s.attn_saved}, std::pair{&s.mlp_hist, &s.mlp_saved}}) {
            const std::size_t column = static_cast<std::size_t>(hist->ne[0]) * 4;
            copies.add(hist->data, static_cast<const std::byte*>(saved->data) + static_cast<std::size_t>(n) * column, hist->bytes());
        }
    }
    copies.flush();
    position_ -= last_cols_ - n;
    last_cols_ = n;
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

void TextProgram::control(std::span<const ControllerWeights* const> w, std::span<const Tensor* const> c,
                          std::span<const Tensor* const> m, std::span<Tensor* const> out) {
    const std::int32_t h = i32(config_.hidden_size), rank = i32(config_.loop_gate_rank);
    std::array<Tensor, 3 * ops::kLoopControllerMaxJobs> views;
    std::array<ops::LoopControllerJob, ops::kLoopControllerMaxJobs> jobs;
    for (std::size_t j = 0; j < w.size(); ++j) {
        views[3 * j]     = model_.tensor(w[j]->down, {2 * h, rank});
        views[3 * j + 1] = model_.tensor(w[j]->gain, {rank, h});
        views[3 * j + 2] = model_.tensor(w[j]->forget, {rank, h});
        jobs[j] = {c[j], m[j], &views[3 * j], &views[3 * j + 1], &views[3 * j + 2], out[j]};
    }
    ops::loop_controller_fp32(std::span<const ops::LoopControllerJob>(jobs.data(), w.size()), config_.rms_norm_eps, workspace_, stream_);
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
        const ControllerWeights* cw[] = {&w.loop[0], &w.loop[1], &w.loop[2]};
        const Tensor* cin[] = {&xq, &xk, &xv};
        const Tensor* min[] = {&messages[0], &messages[1], &messages[2]};
        Tensor* cout[] = {&cq, &ck, &cv};
        control(cw, cin, min, cout);
        xq = cq;
        xk = ck;
        xv = cv;
    }
    // A local visit without loop messages has one input for q, k and v (one MIX query, one norm): normalize it once
    // and project k and v in one call.
    const bool one_input = local && messages == nullptr;
    Tensor nq = workspace_.alloc(DType::BF16, {h, cols}), nk = nq, nv = nq;
    if (!one_input) {
        nk = workspace_.alloc(DType::BF16, {h, cols});
        nv = workspace_.alloc(DType::BF16, {h, cols});
    }
    {
        const Tensor gq = model_.tensor(local ? w.norm : w.q_norm, {h});
        const Tensor gk = local ? gq : model_.tensor(w.k_norm, {h}), gv = local ? gq : model_.tensor(w.v_norm, {h});
        const ops::RmsNormJob norms[] = {{&xq, &gq, &nq}, {&xk, &gk, &nk}, {&xv, &gv, &nv}};
        ops::rmsnorm_fp32_batch(std::span<const ops::RmsNormJob>(norms, one_input ? 1 : 3), eps, stream_);
    }

    // 3e: projections and short convolutions.
    const WeightId q_weights[] = {w.q, w.rel, w.gate, w.sink};
    const std::int32_t q_rows[] = {q, rel, q, heads};
    Tensor pq = project(nq, q_weights, q_rows);
    Tensor q_t = pq.slice(0, 0, q), r_t = pq.slice(0, q, rel), gate = pq.slice(0, q + rel, q), sink = pq.slice(0, 2 * q + rel, heads);
    const std::int32_t dyn_rows = i32(c.kv_heads) * k;
    Tensor kconv = workspace_.alloc(DType::FP32, {kv, cols}), vconv = workspace_.alloc(DType::FP32, {kv, cols});
    if (local) {
        Tensor pk, pv;
        if (one_input) {
            const WeightId kvw[] = {w.k, w.v};
            const std::int32_t kvr[] = {kv, kv};
            const Tensor both = project(nk, kvw, kvr);
            pk = both.slice(0, 0, kv);
            pv = both.slice(0, kv, kv);
        } else {
            const WeightId kw[] = {w.k}, vw[] = {w.v};
            const std::int32_t kr[] = {kv};
            pk = project(nk, kw, kr);
            pv = project(nv, vw, kr);
        }
        const Tensor kt = model_.tensor(w.k_conv, {k, kv}), vt = model_.tensor(w.v_conv, {k, kv});
        ops::ShortConvJob jobs[2];
        jobs[0].x = &pk, jobs[0].taps = &kt, jobs[0].history = &st.k_hist, jobs[0].out = &kconv;
        jobs[1].x = &pv, jobs[1].taps = &vt, jobs[1].history = &st.v_hist, jobs[1].out = &vconv;
        save_input(st.k_saved, pk);
        save_input(st.v_saved, pv);
        ops::short_conv_fp32_batch(jobs, d, stream_);
    } else {
        const WeightId kw[] = {w.k, w.k_conv_proj}, vw[] = {w.v, w.v_conv_proj};
        const std::int32_t kr[] = {kv, dyn_rows};
        Tensor pk = project(nk, kw, kr), pv = project(nv, vw, kr);
        const Tensor kd = pk.slice(0, kv, dyn_rows), vd = pv.slice(0, kv, dyn_rows);
        const Tensor kx = pk.slice(0, 0, kv), vx = pv.slice(0, 0, kv);
        const Tensor kt = model_.tensor(w.k_conv, {k, kv}), vt = model_.tensor(w.v_conv, {k, kv});
        ops::ShortConvJob jobs[2];
        jobs[0].x = &kx, jobs[0].taps = &kt, jobs[0].dynamic = &kd, jobs[0].history = &st.k_hist, jobs[0].out = &kconv;
        jobs[1].x = &vx, jobs[1].taps = &vt, jobs[1].dynamic = &vd, jobs[1].history = &st.v_hist, jobs[1].out = &vconv;
        save_input(st.k_saved, kx);
        save_input(st.v_saved, vx);
        ops::short_conv_fp32_batch(jobs, d, stream_);
    }

    // 3f: clips, value norm, log scaling of the global layers' q and r.
    const ops::PositionLogScale tau{start, c.log_scaling_n_floor, c.log_scaling_alpha, start_offset_};
    const Tensor q_clip = model_.tensor(w.q_clip, {d, heads}), rel_clip = model_.tensor(w.rel_clip, {i32(c.d_rel), heads});
    {
        ops::HeadNormJob jobs[4];
        jobs[0].x = &q_t, jobs[0].head_dim = d, jobs[0].scale = &q_clip, jobs[0].threshold = c.qk_clip;
        jobs[0].log_scale = local ? nullptr : &tau;
        jobs[1].x = &r_t, jobs[1].head_dim = i32(c.d_rel), jobs[1].scale = &rel_clip, jobs[1].threshold = c.rel_clip;
        jobs[1].log_scale = local ? nullptr : &tau;
        jobs[2].x = &kconv, jobs[2].head_dim = d, jobs[2].threshold = c.qk_clip;
        jobs[3].x = &vconv, jobs[3].head_dim = d, jobs[3].unit_rms = true, jobs[3].eps = eps;
        ops::head_norms_fp32(jobs, stream_);
    }

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
    const Tensor post_taps = model_.tensor(w.post_conv, {k, h});

    // 3h: the convolution with the residual and block boundary: a block's first layer starts from a alone.
    {
        const bool boundary = p % c.attn_res_block_size == 0;
        if (boundary) { blocks.push_back(cur); }
        ops::ShortConvJob job;
        job.x = &a_in, job.taps = &post_taps, job.history = &st.attn_hist;
        save_input(st.attn_saved, a_in);
        if (boundary) {
            job.out = &ffn_inp;
        } else {
            job.out = &a, job.residual = &cur, job.sum = &ffn_inp;
        }
        ops::short_conv_fp32_batch(std::span<const ops::ShortConvJob>(&job, 1), 1, stream_);
        if (boundary) { a = ffn_inp; } // for the trace
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
        const ControllerWeights* cw[] = {&w.loop[3]};
        const Tensor* cin[] = {&z};
        const Tensor* min[] = {&messages[3]};
        Tensor* cout[] = {&zc};
        control(cw, cin, min, cout);
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
    {
        const Tensor mlp_taps = model_.tensor(w.mlp_post_conv, {k, h});
        ops::ShortConvJob job;
        job.x = &y, job.taps = &mlp_taps, job.history = &st.mlp_hist, job.residual = &ffn_inp, job.sum = &next;
        save_input(st.mlp_saved, y);
        ops::short_conv_fp32_batch(std::span<const ops::ShortConvJob>(&job, 1), 1, stream_);
    }

    if (trace != nullptr && *trace) {
        sync(stream_);
        (*trace)("attn_out-" + std::to_string(v), a);
        (*trace)("ffn_inp-" + std::to_string(v), ffn_inp);
        (*trace)("l_out-" + std::to_string(v), next);
        (*trace)("route_ids-" + std::to_string(v), ids);
    }
    return next;
}

Tensor TextProgram::forward(std::span<const std::int32_t> tokens, std::span<const InputEmbeddings> embeddings, const Trace* trace) {
    const TextConfig& c = config_;
    const std::int32_t cols = static_cast<std::int32_t>(tokens.size()), h = i32(c.hidden_size);
    if (cols < 1 || cols > limits_.max_chunk) { throw std::invalid_argument("inkling2: chunk size outside 1..max_chunk"); }
    if (position_ + cols > limits_.max_context) { throw std::invalid_argument("inkling2: the sequence would pass max_context"); }
    for (const auto token : tokens) {
        if (token < 0 || static_cast<std::uint32_t>(token) >= c.unpadded_vocab_size) { throw std::invalid_argument("inkling2: token id out of range"); }
    }
    std::int32_t next = 0;
    for (const auto& e : embeddings) {
        const bool shaped = e.values.data != nullptr && e.values.dtype == DType::FP32 && e.values.ne[0] == h && e.values.ne[1] >= 1 &&
                            e.values.ne[2] == 1 && e.values.ne[3] == 1 && e.values.is_contiguous();
        if (!shaped || e.column < next || e.column + e.values.ne[1] > cols) {
            throw std::invalid_argument("inkling2: input embeddings must be FP32 [hidden, n] in ascending columns of the chunk");
        }
        next = e.column + e.values.ne[1];
    }
    const bool tracing = trace != nullptr && *trace;
    if (cols == 1 && embeddings.empty() && !tracing && limits_.decode_graph && stream_ != nullptr) { return forward_graph(tokens[0]); }
    workspace_.reset();
    Tensor ids = workspace_.alloc(DType::I32, {cols});
    if (cudaMemcpyAsync(ids.data, tokens.data(), tokens.size_bytes(), cudaMemcpyHostToDevice, stream_) != cudaSuccess) {
        throw std::runtime_error("inkling2: token upload failed");
    }
    saving_ = cols >= 2 && cols <= limits_.max_rollback;
    if (saving_) {
        // Each history before the forward, ahead of the inputs save_input() appends.
        CopyBatch copies(stream_);
        for (auto& s : state_) {
            for (auto [hist, saved] : {std::pair{&s.k_hist, &s.k_saved}, std::pair{&s.v_hist, &s.v_saved},
                                       std::pair{&s.attn_hist, &s.attn_saved}, std::pair{&s.mlp_hist, &s.mlp_saved}}) {
                copies.add(saved->data, hist->data, hist->bytes());
            }
        }
        copies.flush();
    }
    last_cols_ = 0;
    Tensor logits;
    inputs_ = embeddings;
    try {
        logits = run(ids, position_, trace);
    } catch (...) {
        saving_ = false;
        inputs_ = {};
        throw;
    }
    saving_ = false;
    inputs_ = {};
    position_ += cols;
    last_cols_ = cols;
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
        // Ends the capture and returns its graph; a graph handle is destroyed whatever the capture's result.
        const auto end_capture = [&]() -> cudaGraph_t {
            start_offset_ = nullptr;
            cudaGraph_t graph = nullptr;
            const cudaError_t ended = cudaStreamEndCapture(stream_, &graph);
            if (ended != cudaSuccess) {
                if (graph != nullptr) { cudaGraphDestroy(graph); }
                return nullptr;
            }
            return graph;
        };
        try {
            graph_logits_ = run(ids, 0, nullptr);
        } catch (...) {
            if (cudaGraph_t graph = end_capture()) { cudaGraphDestroy(graph); }
            throw;
        }
        cudaGraph_t graph = end_capture();
        if (graph == nullptr) { throw std::runtime_error("inkling2: decode graph capture failed"); }
        const cudaError_t made = cudaGraphInstantiate(&decode_graph_, graph, 0);
        cudaGraphDestroy(graph);
        if (made != cudaSuccess) {
            decode_graph_ = nullptr;
            throw std::runtime_error(std::string("inkling2: cannot instantiate the decode graph: ") + cudaGetErrorString(made));
        }
    }
    if (cudaGraphLaunch(decode_graph_, stream_) != cudaSuccess) { throw std::runtime_error("inkling2: decode graph launch failed"); }
    position_ += 1;
    last_cols_ = 1;
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
    for (const auto& e : inputs_) {
        if (cudaMemcpyAsync(static_cast<float*>(cur.data) + static_cast<std::int64_t>(e.column) * h, e.values.data, e.values.bytes(),
                            cudaMemcpyDeviceToDevice, stream_) != cudaSuccess) {
            throw std::runtime_error("inkling2: input embedding copy failed");
        }
    }

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
        copy_tap(v, cur);
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
    copy_tap(c.num_layer_visits, x);
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
