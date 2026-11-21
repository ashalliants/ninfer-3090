// Written for NInfer-3090. See dspark_program.h; the order of operations is llama.cpp's graph_inkling2 (context
// injection, block backbone, Markov head) and build_inkling2_head.

#include "models/inkling2/program/dspark_program.h"

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

DSparkProgram::DSparkProgram(const Model& model, std::int32_t max_context, std::int32_t max_chunk, cudaStream_t stream)
    : model_(model), config_(require_dspark(model)), text_(model.config()), max_context_(max_context), max_chunk_(max_chunk),
      stream_(stream),
      workspace_((std::size_t(model.config().vocab_size) * require_dspark(model).block_size * 4) +
                 std::size_t(std::max(max_chunk, std::int32_t(require_dspark(model).block_size))) * 64 * 1024 + (64U << 20)) {
    const DSparkConfig& c = config_;
    if (max_chunk < 1 || max_chunk > max_context || max_chunk > i32(c.sliding_window)) {
        throw std::invalid_argument("inkling2 dspark: max_chunk must be in 1..min(max_context, the draft window)");
    }
    if (c.target_visits.size() > 8 || c.head_dim != 128) { throw std::invalid_argument("inkling2 dspark: unsupported geometry"); }
    const std::size_t kv = std::size_t(c.kv_heads) * c.head_dim, h = c.hidden_size, r = c.markov_rank, v = text_.vocab_size;
    std::size_t total = padded(h * c.block_size * 4) + padded((2 * r + 2 * h) * 4) + padded(v * r * 4) + padded(v * 4) +
                        padded(std::size_t(c.block_size) * 4);
    std::vector<std::size_t> capacity(c.layers);
    for (std::uint32_t i = 0; i < c.layers; ++i) {
        capacity[i] = c.local[i] ? c.sliding_window : static_cast<std::size_t>(max_context);
        total += 2 * padded(kv * capacity[i] * 4);
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
        layers_.push_back({carve(DType::FP32, {i32(kv), i32(capacity[i])}), carve(DType::FP32, {i32(kv), i32(capacity[i])})});
    }
    masks_    = carve(DType::FP32, {i32(h), i32(c.block_size)});
    joint_in_ = carve(DType::FP32, {i32(2 * r + 2 * h)});
    pt_       = carve(DType::FP32, {i32(v), i32(r)});
    hb_       = carve(DType::FP32, {i32(v)});
    ids_      = carve(DType::I32, {i32(c.block_size)});
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
    const ops::DenseMatrix head_matrix{head.data, QType::BF16, i32(v), i32(h)};
    const Tensor wo_tensor(wo_t.p, DType::FP32, {i32(h), i32(r)});
    ops::dense_projection_fp32(wo_tensor, head_matrix, nullptr, pt_, stream_);
    const Tensor bo = model_.tensor(w.hidden_out_bias, {i32(h), 1});
    ops::dense_projection_fp32(bo, head_matrix, nullptr, hb_, stream_);
    cuda(cudaStreamSynchronize(stream_), "precompute");
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
        ops::dense_projection_fp32(ctx, dense(model_.matrix(l.context_k)), nullptr, k, stream_);
        ops::dense_projection_fp32(ctx, dense(model_.matrix(l.context_v)), nullptr, v, stream_);
        ops::head_rmsnorm_rope_fp32(k, i32(c.head_dim), model_.tensor(l.context_k_norm, {i32(c.head_dim)}), c.rms_norm_eps, true,
                                    position_, c.rope_theta, stream_);
        // Columns into the ring slots (position mod capacity): at most two contiguous runs.
        const Layer& layer           = layers_[i];
        const std::int64_t capacity  = layer.k_ring.ne[1];
        const std::size_t column     = static_cast<std::size_t>(kv) * 4;
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

std::vector<std::int32_t> DSparkProgram::draft(std::int32_t anchor) {
    const DSparkConfig& c = config_;
    if (position_ < 1) { throw std::invalid_argument("inkling2 dspark: drafting needs a committed context"); }
    if (anchor < 0 || static_cast<std::uint32_t>(anchor) >= text_.unpadded_vocab_size) { throw std::invalid_argument("inkling2 dspark: bad anchor"); }
    const std::int32_t h = i32(c.hidden_size), kv = i32(std::size_t(c.kv_heads) * c.head_dim), q = i32(std::size_t(c.heads) * c.head_dim);
    const std::int32_t b = i32(c.block_size), r = i32(c.markov_rank), vocab = i32(text_.vocab_size), f = i32(c.intermediate_size);
    const DSparkWeights& w = model_.dspark_weights();
    workspace_.reset();
    cuda(cudaMemcpyAsync(ids_.data, &anchor, 4, cudaMemcpyHostToDevice, stream_), "anchor upload");

    // The block: the anchor's normed embedding, then the masks.
    Tensor x = workspace_.alloc(DType::FP32, {h, b});
    cuda(cudaMemcpyAsync(x.data, masks_.data, masks_.bytes(), cudaMemcpyDeviceToDevice, stream_), "block");
    {
        Tensor first = x.slice(1, 0, 1);
        ops::embedding_bf16_fp32(ids_.slice(0, 0, 1), model_.tensor(model_.weights().embedding, {h, vocab}), first, stream_);
        const Tensor embed_norm = model_.tensor(model_.weights().embedding_norm, {h});
        ops::rmsnorm_fp32(first, &embed_norm, text_.rms_norm_eps, first, stream_);
    }
    for (std::uint32_t i = 0; i < c.layers; ++i) {
        const DSparkLayerWeights& l = w.layers[i];
        const Tensor attn_norm      = model_.tensor(l.attn_norm, {h});
        Tensor n                    = workspace_.alloc(DType::FP32, {h, b});
        ops::rmsnorm_fp32(x, &attn_norm, c.rms_norm_eps, n, stream_);
        Tensor qt = workspace_.alloc(DType::FP32, {q, b}), kt = workspace_.alloc(DType::FP32, {kv, b}), vt = workspace_.alloc(DType::FP32, {kv, b});
        ops::dense_projection_fp32(n, dense(model_.matrix(l.q)), nullptr, qt, stream_);
        ops::dense_projection_fp32(n, dense(model_.matrix(l.k)), nullptr, kt, stream_);
        ops::dense_projection_fp32(n, dense(model_.matrix(l.v)), nullptr, vt, stream_);
        ops::head_rmsnorm_rope_fp32(qt, i32(c.head_dim), model_.tensor(l.q_norm, {i32(c.head_dim)}), c.rms_norm_eps, true, position_,
                                    c.rope_theta, stream_);
        ops::head_rmsnorm_rope_fp32(kt, i32(c.head_dim), model_.tensor(l.k_norm, {i32(c.head_dim)}), c.rms_norm_eps, true, position_,
                                    c.rope_theta, stream_);
        Tensor attn = workspace_.alloc(DType::FP32, {q, b});
        const ops::DSparkAttentionShape shape{i32(c.heads), i32(c.kv_heads), i32(c.head_dim), c.local[i] ? i32(c.sliding_window) : 0};
        ops::dspark_block_attention_fp32(qt, layers_[i].k_ring, layers_[i].v_ring, kt, vt, position_, shape, attn, stream_);
        Tensor o = workspace_.alloc(DType::FP32, {h, b});
        ops::dense_projection_fp32(attn, dense(model_.matrix(l.output)), nullptr, o, stream_);
        ops::add_fp32(x, o, x, stream_);
        const Tensor ffn_norm = model_.tensor(l.ffn_norm, {h});
        ops::rmsnorm_fp32(x, &ffn_norm, c.rms_norm_eps, n, stream_);
        Tensor g = workspace_.alloc(DType::FP32, {f, b}), u = workspace_.alloc(DType::FP32, {f, b});
        ops::dense_projection_fp32(n, dense(model_.matrix(l.gate)), nullptr, g, stream_);
        ops::dense_projection_fp32(n, dense(model_.matrix(l.up)), nullptr, u, stream_);
        ops::silu_mul_fp32(g, u, g, stream_);
        ops::dense_projection_fp32(g, dense(model_.matrix(l.down)), nullptr, o, stream_);
        ops::add_fp32(x, o, x, stream_);
    }
    const Tensor out_norm = model_.tensor(w.output_norm, {h});
    ops::rmsnorm_fp32(x, &out_norm, c.rms_norm_eps, x, stream_);
    // The base logits through the text head, as the target computes its own (BF16 activations, FP32 sums).
    Tensor xb = workspace_.alloc(DType::BF16, {h, b});
    ops::cast_fp32_to_bf16(x, xb, stream_);
    Tensor base = workspace_.alloc(DType::FP32, {vocab, b});
    {
        const Tensor head      = model_.tensor(model_.weights().head, {h, vocab});
        const Tensor* heads[] = {&head};
        ops::projection_fp32(xb, heads, base, stream_);
    }

    // The Markov chain: state from 0, prev from the anchor; each step's argmax feeds the next.
    cuda(cudaMemsetAsync(joint_in_.data, 0, static_cast<std::size_t>(r) * 4, stream_), "state");
    Tensor state  = joint_in_.slice(0, 0, r), prev_row = joint_in_.slice(0, r, r).reshape({r, 1});
    Tensor joint  = workspace_.alloc(DType::FP32, {3 * r}), latent = workspace_.alloc(DType::FP32, {r});
    const Tensor w1 = model_.tensor(w.markov_embedding, {r, vocab});
    const Tensor joint_bias = model_.tensor(w.joint_bias, {3 * r});
    const Tensor joint_in   = joint_in_.reshape({i32(joint_in_.numel()), 1});
    Tensor joint_out        = joint.reshape({3 * r, 1});
    for (std::int32_t i = 1; i < b; ++i) {
        ops::embedding_bf16_fp32(ids_.slice(0, i - 1, 1), w1, prev_row, stream_);
        cuda(cudaMemcpyAsync(static_cast<float*>(joint_in_.data) + 2 * r, static_cast<const float*>(x.data) + std::size_t(i) * h,
                             std::size_t(h) * 4, cudaMemcpyDeviceToDevice, stream_),
             "x_i");
        ops::dense_projection_fp32(joint_in, dense(model_.matrix(w.joint)), &joint_bias, joint_out, stream_);
        ops::dspark_markov_update_fp32(joint, state, latent, stream_);
        Tensor logits = base.slice(1, i, 1).reshape({vocab});
        ops::dspark_markov_logits_fp32(logits, hb_, pt_, latent, c.final_logit_softcap, logits, stream_);
        Tensor next = ids_.slice(0, i, 1);
        ops::argmax_fp32(logits.reshape({vocab, 1}), i32(text_.unpadded_vocab_size), next, stream_);
    }
    std::vector<std::int32_t> drafts(b - 1);
    cuda(cudaMemcpyAsync(drafts.data(), static_cast<const std::int32_t*>(ids_.data) + 1, drafts.size() * 4, cudaMemcpyDeviceToHost, stream_),
         "drafts");
    cuda(cudaStreamSynchronize(stream_), "draft");
    return drafts;
}

} // namespace ninfer::models::inkling2
