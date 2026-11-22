// Written for NInfer-3090. See dspark_program.h; the order of operations is llama.cpp's graph_inkling2 (context
// injection, block backbone, Markov head) and build_inkling2_head.

#include "models/inkling2/program/dspark_program.h"

#include "core/copy_batch.h"
#include "ninfer/ops/cast.h"
#include "ninfer/ops/dspark_fp32.h"
#include "ninfer/ops/fp32_rows.h"
#include "ninfer/ops/projection_fp32.h"

#include <cuda_bf16.h>

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>

namespace ninfer::models::inkling2 {
namespace {

constexpr std::size_t kAlign = 256;

std::size_t padded(std::size_t bytes) { return (bytes + kAlign - 1) / kAlign * kAlign; }

std::int32_t i32(std::uint64_t v) { return static_cast<std::int32_t>(v); }

void cuda(cudaError_t error, const char* what) {
    if (error != cudaSuccess) { throw std::runtime_error(std::string("inkling2 dspark: ") + what + ": " + cudaGetErrorString(error)); }
}

ops::DenseMatrix dense(const Model::Matrix& m) { return {m.data, m.format, m.rows, m.cols}; }

const DSparkConfig& require_dspark(const Model& model) {
    if (model.dspark_config() == nullptr) { throw std::invalid_argument("inkling2: the artifact has no DSpark drafter"); }
    return *model.dspark_config();
}

} // namespace

DSparkProgram::DSparkProgram(const Model& model, std::int32_t max_context, std::int32_t max_chunk, cudaStream_t stream,
                             const ops::Q8Weight* head_q8, bool kv_fp16, std::int32_t global_window, bool head_copy)
    : model_(model), config_(require_dspark(model)), text_(model.config()), max_context_(max_context), max_chunk_(max_chunk),
      stream_(stream),
      workspace_((std::size_t(model.config().vocab_size) * require_dspark(model).block_size * 4) +
                 std::size_t(std::max(max_chunk, std::int32_t(require_dspark(model).block_size))) * 64 * 1024 +
                 ops::dspark_block_attention_workspace_bytes({i32(require_dspark(model).heads), 0, 0, 0}, i32(require_dspark(model).block_size)) +
                 (64U << 20)) {
    const DSparkConfig& c = config_;
    if (max_chunk < 1 || max_chunk > max_context || max_chunk > i32(c.sliding_window)) {
        throw std::invalid_argument("inkling2 dspark: max_chunk must be in 1..min(max_context, the draft window)");
    }
    if (c.target_visits.size() > 8 || c.head_dim != 128) { throw std::invalid_argument("inkling2 dspark: unsupported geometry"); }
    if (global_window < 0 || (global_window > 0 && global_window < i32(c.sliding_window))) {
        throw std::invalid_argument("inkling2 dspark: global_window must be 0 or at least the sliding window");
    }
    const bool windowed = global_window > 0 && global_window < max_context;
    const std::size_t kv = std::size_t(c.kv_heads) * c.head_dim, h = c.hidden_size, r = c.markov_rank, v = text_.vocab_size;
    std::size_t total = padded(h * c.block_size * 4) + padded((2 * r + 2 * h) * 4) + padded(v * 4) +
                        padded(std::size_t(c.block_size) * 4) + padded(std::size_t(c.block_size) * 4) + padded(std::size_t(c.block_size) * 8);
    std::vector<std::size_t> capacity(c.layers);
    for (std::uint32_t i = 0; i < c.layers; ++i) {
        window_.push_back(c.local[i] ? i32(c.sliding_window) : windowed ? global_window : 0);
        capacity[i] = window_[i] > 0 ? static_cast<std::size_t>(window_[i]) : static_cast<std::size_t>(max_context);
        total += 2 * padded(kv * capacity[i] * (kv_fp16 ? 2 : 4));
    }
    state_memory_ = DeviceBuffer(total);
    auto* base    = static_cast<std::byte*>(state_memory_.p);
    std::size_t offset = 0;
    const auto carve = [&](DType dtype, std::initializer_list<std::int32_t> ne) {
        Tensor t(base + offset, dtype, ne);
        offset += padded(t.bytes());
        return t;
    };
    for (std::uint32_t i = 0; i < c.layers; ++i) {
        const DType ring = kv_fp16 ? DType::FP16 : DType::FP32;
        layers_.push_back({carve(ring, {i32(kv), i32(capacity[i])}), carve(ring, {i32(kv), i32(capacity[i])})});
    }
    masks_    = carve(DType::FP32, {i32(h), i32(c.block_size)});
    joint_in_ = carve(DType::FP32, {i32(2 * r + 2 * h)});
    hb_       = carve(DType::FP32, {i32(v)});
    ids_      = carve(DType::I32, {i32(c.block_size)});
    probs_    = carve(DType::FP32, {i32(c.block_size)});
    draws_    = carve(DType::I32, {2 * i32(c.block_size)}); // FP64 [block]
    cuda(cudaMemsetAsync(state_memory_.p, 0, total, stream_), "clear");

    const DSparkWeights& w = model_.dspark_weights();
    const Tensor mask      = model_.tensor(w.mask_embedding, {i32(h)});
    for (std::uint32_t i = 1; i < c.block_size; ++i) {
        cuda(cudaMemcpyAsync(static_cast<float*>(masks_.data) + i * h, mask.data, h * 4, cudaMemcpyDeviceToDevice, stream_), "masks");
    }
    // The Markov head's W_head W_o and W_head b_o, once: the transposed W_o as FP32 [H, R] (column r = W_o[:, r]).
    const Tensor wo = model_.tensor(w.hidden_out, {i32(r), i32(h)});
    std::vector<__nv_bfloat16> words(h * r);
    cuda(cudaMemcpyAsync(words.data(), wo.data, words.size() * 2, cudaMemcpyDeviceToHost, stream_), "W_o download");
    cuda(cudaStreamSynchronize(stream_), "W_o download");
    std::vector<float> transposed(h * r);
    for (std::size_t row = 0; row < h; ++row) {
        for (std::size_t col = 0; col < r; ++col) { transposed[col * h + row] = __bfloat162float(words[row * r + col]); }
    }
    DeviceBuffer wo_t(transposed.size() * 4);
    cuda(cudaMemcpyAsync(wo_t.p, transposed.data(), transposed.size() * 4, cudaMemcpyHostToDevice, stream_), "W_o upload");
    const Tensor head = model_.tensor(model_.weights().head, {i32(h), i32(v)});
    // The head the target reads: BF16, or its Q8_G32_FP16 re-encoding (values [V, H], scales [V, H / 32]).
    const ops::DenseMatrix head_matrix = head_q8 != nullptr ? ops::DenseMatrix{head_q8->values, QType::Q8_G32_FP16, i32(v), i32(h), head_q8->scales}
                                                            : ops::DenseMatrix{head.data, QType::BF16, i32(v), i32(h)};
    const Tensor wo_tensor(wo_t.p, DType::FP32, {i32(h), i32(r)});
    // The drafter's compact forms (lossy: they change only drafts, which the target's verify licenses): P as Q8 with
    // a scale per 32 of R, and the head as Q8 for the base logits: the target's re-encoding when it reads one (the
    // drafts then come from the logits that verify them), else the drafter's own copy.
    {
        DeviceBuffer pt_memory(v * r * 4);
        Tensor pt(pt_memory.p, DType::FP32, {i32(v), i32(r)});
        ops::dense_projection_fp32(wo_tensor, head_matrix, nullptr, pt, stream_);
        markov_ = ops::MarkovHead{ops::MarkovForm::Q8, nullptr, nullptr, i32(v), i32(r)};
        const std::size_t codes = padded(ops::markov_head_code_bytes(markov_.form, i32(v), i32(r)));
        markov_memory_ = DeviceBuffer(codes + ops::markov_head_scale_bytes(markov_.form, i32(v), i32(r)));
        markov_.codes  = markov_memory_.p;
        markov_.scales = reinterpret_cast<const float*>(static_cast<std::byte*>(markov_memory_.p) + codes);
        ops::dspark_markov_quantize(pt, markov_, stream_);
        cuda(cudaStreamSynchronize(stream_), "P"); // before pt_memory is freed
    }
    if (head_q8 != nullptr) {
        head_q8_ = *head_q8;
    } else if (!head_copy) {
        head_bf16_ = head_matrix;
    } else {
        const std::size_t values = padded(std::size_t(v) * h);
        head_memory_ = DeviceBuffer(values + std::size_t(v) * (h / 32) * 2);
        auto* codes  = static_cast<std::int8_t*>(head_memory_.p);
        head_q8_     = ops::Q8Weight{codes, static_cast<std::byte*>(head_memory_.p) + values, i32(h), i32(v)};
        ops::quantize_q8_g32(head, codes, static_cast<std::byte*>(head_memory_.p) + values, stream_);
    }
    const Tensor bo = model_.tensor(w.hidden_out_bias, {i32(h), 1});
    ops::dense_projection_fp32(bo, head_matrix, nullptr, hb_, stream_);
    // The Q8_0 matrices the block and the context read, split once. The layers' matrices are in pinned host memory
    // (the split reads them there); one that is not split is copied to the device as it is.
    std::vector<Model::Matrix> q8, copied;
    const auto want = [&](WeightId id, bool host) {
        const Model::Matrix m = model_.matrix(id);
        if (m.format == QType::GGML_Q8_0 && m.cols % 128 == 0) {
            q8.push_back(m);
        } else if (host) {
            copied.push_back(m);
        }
    };
    for (const DSparkLayerWeights& l : w.layers) {
        for (const WeightId id : {l.q, l.k, l.v, l.output, l.gate, l.up, l.down, l.context_k, l.context_v}) { want(id, true); }
    }
    want(w.joint, false);
    const auto stored_bytes = [](const Model::Matrix& m) {
        const std::size_t n = std::size_t(m.rows) * m.cols;
        switch (m.format) {
        case QType::GGML_Q8_0: return n / 32 * 34;
        case QType::BF16: return n * 2;
        case QType::FP32: return n * 4;
        default: throw std::invalid_argument("inkling2 dspark: a layer matrix of an unsupported format");
        }
    };
    std::size_t copy_bytes = 0;
    for (const Model::Matrix& m : copied) { copy_bytes += padded(stored_bytes(m)); }
    copy_memory_ = DeviceBuffer(std::max<std::size_t>(copy_bytes, 1));
    for (std::size_t i = 0, at = 0; i < copied.size(); ++i) {
        auto* to = static_cast<std::byte*>(copy_memory_.p) + at;
        at += padded(stored_bytes(copied[i]));
        cuda(cudaMemcpyAsync(to, copied[i].data, stored_bytes(copied[i]), cudaMemcpyHostToDevice, stream_), "layer matrix");
        split_.emplace(copied[i].data, ops::DenseMatrix{to, copied[i].format, copied[i].rows, copied[i].cols});
    }
    std::size_t split_bytes = 0;
    for (const Model::Matrix& m : q8) { split_bytes += padded(std::size_t(m.rows) * m.cols) + padded(std::size_t(m.rows) * m.cols / 16); }
    split_memory_ = DeviceBuffer(std::max<std::size_t>(split_bytes, 1));
    std::size_t at = 0;
    for (const Model::Matrix& m : q8) {
        auto* values = static_cast<std::byte*>(split_memory_.p) + at;
        at += padded(std::size_t(m.rows) * m.cols);
        auto* scales = static_cast<std::byte*>(split_memory_.p) + at;
        at += padded(std::size_t(m.rows) * m.cols / 16);
        ops::split_q8_0(m.data, m.rows, m.cols, values, scales, stream_);
        split_.emplace(m.data, ops::DenseMatrix{values, QType::Q8_G32_FP16, m.rows, m.cols, scales});
    }
    cuda(cudaStreamSynchronize(stream_), "precompute");
}

ops::DenseMatrix DSparkProgram::matrix(const Model::Matrix& m) const {
    const auto it = split_.find(m.data);
    return it != split_.end() ? it->second : dense(m);
}

void DSparkProgram::extend(std::span<const Tensor> taps, std::int32_t n) {
    const DSparkConfig& c = config_;
    if (taps.size() != c.target_visits.size()) { throw std::invalid_argument("inkling2 dspark: one tap per target visit"); }
    if (n < 1 || n > max_chunk_ || position_ + n > max_context_) { throw std::invalid_argument("inkling2 dspark: bad extend size"); }
    const std::int32_t h = i32(c.hidden_size), kv = i32(std::size_t(c.kv_heads) * c.head_dim), r = i32(c.markov_rank);
    std::vector<Tensor> cols;
    std::vector<const Tensor*> pointers;
    for (const Tensor& t : taps) {
        if (t.ne[0] != h || t.ne[1] < n) { throw std::invalid_argument("inkling2 dspark: a tap is not [hidden, >= n]"); }
        cols.push_back(t.slice(1, 0, n));
    }
    for (const Tensor& t : cols) { pointers.push_back(&t); }
    const DSparkWeights& w = model_.dspark_weights();
    const Tensor fuse      = model_.tensor(w.fuse, {i32(c.target_visits.size()), i32(c.layers + 1)});
    workspace_.reset();
    for (std::uint32_t i = 0; i < c.layers; ++i) {
        const DSparkLayerWeights& l = w.layers[i];
        Tensor ctx                  = workspace_.alloc(DType::FP32, {h, n});
        const Tensor norm           = model_.tensor(l.context_norm, {h});
        ops::dspark_fuse_taps_fp32(pointers, fuse, static_cast<int>(i), norm, c.rms_norm_eps, ctx, stream_);
        Tensor k = workspace_.alloc(DType::FP32, {kv, n}), v = workspace_.alloc(DType::FP32, {kv, n});
        ops::dense_projection_fp32(ctx, matrix(model_.matrix(l.context_k)), nullptr, k, stream_);
        ops::dense_projection_fp32(ctx, matrix(model_.matrix(l.context_v)), nullptr, v, stream_);
        ops::head_rmsnorm_rope_fp32(k, i32(c.head_dim), model_.tensor(l.context_k_norm, {i32(c.head_dim)}), c.rms_norm_eps, true,
                                    position_, c.rope_theta, stream_);
        // Columns into the ring slots (position mod capacity). FP16 rings round through dspark_ring_write; FP32 rings
        // take the columns as they are, in at most two contiguous async copies (no kernel).
        Layer& layer = layers_[i];
        if (layer.k_ring.dtype != DType::FP32) {
            ops::dspark_ring_write(k, layer.k_ring, position_, stream_);
            ops::dspark_ring_write(v, layer.v_ring, position_, stream_);
            continue;
        }
        const std::int64_t capacity = layer.k_ring.ne[1];
        const std::size_t column    = static_cast<std::size_t>(kv) * 4;
        for (std::int32_t done = 0; done < n;) {
            const std::int64_t slot = (position_ + done) % capacity;
            const std::int32_t run  = static_cast<std::int32_t>(std::min<std::int64_t>(n - done, capacity - slot));
            for (auto [ring, src] : {std::pair{&layer.k_ring, &k}, std::pair{&layer.v_ring, &v}}) {
                cuda(cudaMemcpyAsync(static_cast<std::byte*>(ring->data) + slot * column, static_cast<const std::byte*>(src->data) + done * column,
                                     run * column, cudaMemcpyDeviceToDevice, stream_),
                     "ring write");
            }
            done += run;
        }
    }
    // The Markov head's context: the last committed position's fused taps (row `layers`, its own norm).
    std::vector<Tensor> last;
    std::vector<const Tensor*> last_pointers;
    for (const Tensor& t : cols) { last.push_back(t.slice(1, n - 1, 1)); }
    for (const Tensor& t : last) { last_pointers.push_back(&t); }
    Tensor f_rnn(static_cast<float*>(joint_in_.data) + 2 * r + h, DType::FP32, {h, 1});
    ops::dspark_fuse_taps_fp32(last_pointers, fuse, static_cast<int>(c.layers), model_.tensor(w.rnn_context_norm, {h}), c.rms_norm_eps,
                               f_rnn, stream_);
    position_ += n;
}

std::vector<std::int32_t> DSparkProgram::draft(std::int32_t anchor, std::int32_t n_drafts, const ops::DraftPick* pick,
                                               std::span<const double> canonical, std::vector<float>* probs) {
    const DSparkConfig& c = config_;
    if (position_ < 1) { throw std::invalid_argument("inkling2 dspark: drafting needs a committed context"); }
    if (anchor < 0 || static_cast<std::uint32_t>(anchor) >= text_.unpadded_vocab_size) { throw std::invalid_argument("inkling2 dspark: bad anchor"); }
    const std::int32_t h = i32(c.hidden_size), kv = i32(std::size_t(c.kv_heads) * c.head_dim), q = i32(std::size_t(c.heads) * c.head_dim);
    const std::int32_t r = i32(c.markov_rank), vocab = i32(text_.vocab_size), f = i32(c.intermediate_size);
    if (n_drafts < 1 || n_drafts > i32(c.block_size) - 1) { throw std::invalid_argument("inkling2 dspark: 1 to block_size - 1 drafts"); }
    const DSparkWeights& w = model_.dspark_weights();
    const std::int32_t b = i32(c.block_size);
    const bool sampled = pick != nullptr && pick->temperature > 0.0F;
    if (sampled && canonical.size() < static_cast<std::size_t>(n_drafts)) {
        throw std::invalid_argument("inkling2 dspark: a sampled draft takes one uniform per draft");
    }
    // The picks that report the drafter's probabilities (or sample) run dspark_draft_pick; plain greedy drafts keep
    // argmax_fp32 (the same tokens, without the probability's pass).
    const bool picked = sampled || probs != nullptr;
    workspace_.reset();
    cuda(cudaMemcpyAsync(ids_.data, &anchor, 4, cudaMemcpyHostToDevice, stream_), "anchor upload");
    if (sampled) {
        cuda(cudaMemcpyAsync(draws_.data, canonical.data(), std::size_t(n_drafts) * 8, cudaMemcpyHostToDevice, stream_), "uniforms upload");
    }

    // The block: the anchor's normed embedding, then the masks.
    Tensor x = workspace_.alloc(DType::FP32, {h, b});
    cuda(cudaMemcpyAsync(x.data, masks_.data, std::size_t(h) * b * 4, cudaMemcpyDeviceToDevice, stream_), "block");
    {
        Tensor first = x.slice(1, 0, 1);
        ops::embedding_bf16_fp32(ids_.slice(0, 0, 1), model_.tensor(model_.weights().embedding, {h, vocab}), first, stream_);
        const Tensor embed_norm = model_.tensor(model_.weights().embedding_norm, {h});
        ops::rmsnorm_fp32(first, &embed_norm, text_.rms_norm_eps, first, stream_);
    }
    // A block projection: K <= 3072 (all but the MLP's down) by BF16 activations into projection_q8_fp32 over the split
    // Q8_0 values (the same weights; the activations rounded to BF16), the rest by dense_projection_fp32.
    const auto project = [&](const Tensor& in, WeightId id, Tensor& out) {
        const ops::DenseMatrix m = matrix(model_.matrix(id));
        if (m.format == QType::Q8_G32_FP16 && m.cols <= 3072) {
            Tensor xb16 = workspace_.alloc(DType::BF16, {in.ne[0], in.ne[1]});
            ops::cast_fp32_to_bf16(in, xb16, stream_);
            const ops::Q8Weight q{static_cast<const std::int8_t*>(m.data), m.scales, m.cols, m.rows};
            ops::projection_q8_fp32(xb16, std::span<const ops::Q8Weight>(&q, 1), out, stream_);
        } else {
            ops::dense_projection_fp32(in, m, nullptr, out, stream_);
        }
    };
    for (std::uint32_t i = 0; i < c.layers; ++i) {
        const DSparkLayerWeights& l = w.layers[i];
        const Tensor attn_norm      = model_.tensor(l.attn_norm, {h});
        Tensor n                    = workspace_.alloc(DType::FP32, {h, b});
        ops::rmsnorm_fp32(x, &attn_norm, c.rms_norm_eps, n, stream_);
        Tensor qt = workspace_.alloc(DType::FP32, {q, b}), kt = workspace_.alloc(DType::FP32, {kv, b}), vt = workspace_.alloc(DType::FP32, {kv, b});
        project(n, l.q, qt);
        project(n, l.k, kt);
        project(n, l.v, vt);
        ops::head_rmsnorm_rope_fp32(qt, i32(c.head_dim), model_.tensor(l.q_norm, {i32(c.head_dim)}), c.rms_norm_eps, true, position_,
                                    c.rope_theta, stream_);
        ops::head_rmsnorm_rope_fp32(kt, i32(c.head_dim), model_.tensor(l.k_norm, {i32(c.head_dim)}), c.rms_norm_eps, true, position_,
                                    c.rope_theta, stream_);
        Tensor attn = workspace_.alloc(DType::FP32, {q, b});
        const ops::DSparkAttentionShape shape{i32(c.heads), i32(c.kv_heads), i32(c.head_dim), window_[i]};
        Tensor parts = workspace_.alloc(DType::FP32, {i32(ops::dspark_block_attention_workspace_bytes(shape, b) / 4)});
        ops::dspark_block_attention_fp32(qt, layers_[i].k_ring, layers_[i].v_ring, kt, vt, position_, shape, attn, parts, stream_);
        Tensor o = workspace_.alloc(DType::FP32, {h, b});
        project(attn, l.output, o);
        ops::add_fp32(x, o, x, stream_);
        const Tensor ffn_norm = model_.tensor(l.ffn_norm, {h});
        ops::rmsnorm_fp32(x, &ffn_norm, c.rms_norm_eps, n, stream_);
        Tensor g = workspace_.alloc(DType::FP32, {f, b}), u = workspace_.alloc(DType::FP32, {f, b});
        project(n, l.gate, g);
        project(n, l.up, u);
        ops::silu_mul_fp32(g, u, g, stream_);
        project(g, l.down, o);
        ops::add_fp32(x, o, x, stream_);
    }
    const Tensor out_norm = model_.tensor(w.output_norm, {h});
    ops::rmsnorm_fp32(x, &out_norm, c.rms_norm_eps, x, stream_);
    // The base logits of the drafted columns 1 .. n through the text head, as the target computes its own (BF16
    // activations, FP32 sums); column i of `base` is block position i + 1.
    Tensor xb = workspace_.alloc(DType::BF16, {h, n_drafts});
    ops::cast_fp32_to_bf16(x.slice(1, 1, n_drafts), xb, stream_);
    Tensor base = workspace_.alloc(DType::FP32, {vocab, n_drafts});
    if (head_q8_.values != nullptr) {
        ops::projection_q8_fp32(xb, std::span<const ops::Q8Weight>(&head_q8_, 1), base, stream_);
    } else {
        ops::dense_projection_fp32(x.slice(1, 1, n_drafts), head_bf16_, nullptr, base, stream_);
    }

    // The Markov chain: state from 0, prev from the anchor; each step's argmax feeds the next.
    cuda(cudaMemsetAsync(joint_in_.data, 0, static_cast<std::size_t>(r) * 4, stream_), "state");
    Tensor state  = joint_in_.slice(0, 0, r), prev_row = joint_in_.slice(0, r, r).reshape({r, 1});
    Tensor joint  = workspace_.alloc(DType::FP32, {3 * r}), latent = workspace_.alloc(DType::FP32, {r});
    const Tensor w1 = model_.tensor(w.markov_embedding, {r, vocab});
    const Tensor joint_bias = model_.tensor(w.joint_bias, {3 * r});
    const Tensor joint_in   = joint_in_.reshape({i32(joint_in_.numel()), 1});
    Tensor joint_out        = joint.reshape({3 * r, 1});
    // A BF16 joint runs by BF16 inputs into projection_fp32 (its 768 rows in enough blocks); a Q8_0 one (the converter
    // admits either) by dense_projection_fp32 over its own format, as before.
    const ops::DenseMatrix joint_matrix = matrix(model_.matrix(w.joint));
    const bool joint_bf16               = joint_matrix.format == QType::BF16;
    Tensor joint_weight;
    if (joint_bf16) { joint_weight = model_.tensor(w.joint, {i32(joint_in_.numel()), 3 * r}); }
    const Tensor* joint_weights[] = {&joint_weight};
    for (std::int32_t i = 1; i <= n_drafts; ++i) {
        ops::embedding_bf16_fp32(ids_.slice(0, i - 1, 1), w1, prev_row, stream_);
        cuda(cudaMemcpyAsync(static_cast<float*>(joint_in_.data) + 2 * r, static_cast<const float*>(x.data) + std::size_t(i) * h,
                             std::size_t(h) * 4, cudaMemcpyDeviceToDevice, stream_),
             "x_i");
        if (joint_bf16) {
            Tensor jb = workspace_.alloc(DType::BF16, {i32(joint_in_.numel()), 1});
            ops::cast_fp32_to_bf16(joint_in, jb, stream_);
            ops::projection_fp32(jb, joint_weights, joint_out, stream_);
            ops::add_fp32(joint, joint_bias, joint, stream_);
        } else {
            ops::dense_projection_fp32(joint_in, joint_matrix, &joint_bias, joint_out, stream_);
        }
        ops::dspark_markov_update_fp32(joint, state, latent, stream_);
        Tensor logits = base.slice(1, i - 1, 1).reshape({vocab});
        ops::dspark_markov_logits(logits, hb_, markov_, latent, c.final_logit_softcap, logits, stream_);
        Tensor next = ids_.slice(0, i, 1);
        if (picked) {
            Tensor p = probs_.slice(0, i - 1, 1);
            ops::dspark_draft_pick(logits.reshape({vocab, 1}), i32(text_.unpadded_vocab_size), pick != nullptr ? *pick : ops::DraftPick{},
                                   static_cast<const double*>(draws_.data) + (i - 1), next, p, stream_);
        } else {
            ops::argmax_fp32(logits.reshape({vocab, 1}), i32(text_.unpadded_vocab_size), next, stream_);
        }
    }
    std::vector<std::int32_t> drafts(static_cast<std::size_t>(n_drafts));
    cuda(cudaMemcpyAsync(drafts.data(), static_cast<const std::int32_t*>(ids_.data) + 1, drafts.size() * 4, cudaMemcpyDeviceToHost, stream_),
         "drafts");
    if (probs != nullptr) {
        probs->resize(drafts.size());
        cuda(cudaMemcpyAsync(probs->data(), probs_.data, drafts.size() * 4, cudaMemcpyDeviceToHost, stream_), "draft probabilities");
    }
    cuda(cudaStreamSynchronize(stream_), "draft");
    return drafts;
}

std::size_t DSparkProgram::snapshot_bytes() const {
    std::size_t total = joint_in_.bytes();
    for (std::size_t i = 0; i < layers_.size(); ++i) {
        if (window_[i] > 0) { total += layers_[i].k_ring.bytes() + layers_[i].v_ring.bytes(); }
    }
    return total;
}

void DSparkProgram::save(Snapshot& snapshot) const {
    if (snapshot.memory.bytes < snapshot_bytes()) { throw std::invalid_argument("dspark: the snapshot is too small"); }
    auto* out = static_cast<std::byte*>(snapshot.memory.p);
    CopyBatch copies(stream_);
    copies.add(out, joint_in_.data, joint_in_.bytes());
    out += joint_in_.bytes();
    for (std::size_t i = 0; i < layers_.size(); ++i) {
        if (window_[i] == 0) { continue; }
        for (const Tensor* t : {&layers_[i].k_ring, &layers_[i].v_ring}) {
            copies.add(out, t->data, t->bytes());
            out += t->bytes();
        }
    }
    copies.flush();
    snapshot.position = position_;
}

void DSparkProgram::restore(const Snapshot& snapshot) {
    if (snapshot.position < 0) { throw std::invalid_argument("dspark: an empty snapshot"); }
    const auto* in = static_cast<const std::byte*>(snapshot.memory.p);
    CopyBatch copies(stream_);
    copies.add(joint_in_.data, in, joint_in_.bytes());
    in += joint_in_.bytes();
    for (std::size_t i = 0; i < layers_.size(); ++i) {
        if (window_[i] == 0) { continue; }
        for (const Tensor* t : {&layers_[i].k_ring, &layers_[i].v_ring}) {
            copies.add(t->data, in, t->bytes());
            in += t->bytes();
        }
    }
    copies.flush();
    position_ = snapshot.position;
}

} // namespace ninfer::models::inkling2
