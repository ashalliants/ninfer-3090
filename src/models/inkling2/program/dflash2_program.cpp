// Written for NInfer-3090. See dflash2_program.h; the order of operations is the release's modelling code
// (DFlash2DraftModel: fused context, five convolved sliding layers, the target head's top 16, the candidate selector).

#include "models/inkling2/program/dflash2_program.h"

#include "core/copy_batch.h"
#include "core/host_gather.h"
#include "ninfer/ops/cast.h"
#include "ninfer/ops/dspark_fp32.h"
#include "ninfer/ops/fp32_rows.h"
#include "ninfer/ops/projection_fp32.h"

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
    if (error != cudaSuccess) { throw std::runtime_error(std::string("inkling2 dflash2: ") + what + ": " + cudaGetErrorString(error)); }
}

} // namespace

std::size_t DFlash2Program::workspace_bytes(const Model& model, const DFlash2Config& c, std::int32_t max_chunk) {
    const std::size_t h = c.hidden_size, kv = std::size_t(c.kv_heads) * c.head_dim, taps = c.target_visits.size();
    const std::size_t chunk = std::max<std::size_t>(max_chunk, c.block_size);
    // extend: the taps side by side, their A8 form, the fused context and its A8 form, one layer's K and V.
    const std::size_t extend = padded(chunk * taps * h * 4) + padded(ops::a8_columns_bytes(i32(taps * h), i32(chunk))) + padded(chunk * h * 4) +
                               padded(ops::a8_columns_bytes(i32(h), i32(chunk))) + 2 * padded(chunk * kv * 4);
    // draft: the block's activations (a few MB), the base logits and the candidates.
    const std::size_t draft = std::size_t(model.config().vocab_size) * c.block_size * 4 + (16U << 20) +
                              ops::dspark_block_attention_workspace_bytes({i32(c.heads), 0, 0, 0}, i32(c.block_size));
    return std::max(extend, draft) + (8U << 20);
}

namespace {
DeviceArena drafter_workspace(const Model& model, const DFlash2Config& config, std::int32_t max_chunk, DeviceSpan borrowed) {
    const std::size_t bytes = DFlash2Program::workspace_bytes(model, config, max_chunk);
    if (borrowed.data == nullptr) { return DeviceArena(bytes); }
    if (borrowed.bytes < bytes) { throw std::invalid_argument("inkling2 dflash2: the borrowed workspace is smaller than the drafter needs"); }
    return DeviceArena(borrowed);
}
} // namespace

DFlash2Program::DFlash2Program(const Model& model, const DFlash2Release& release, std::int32_t max_context, std::int32_t max_chunk,
                               cudaStream_t stream, const ops::Q8Weight* head_q8, bool kv_fp16, bool head_copy, DeviceSpan workspace)
    : model_(model), config_(release.config()), text_(model.config()), max_context_(max_context), max_chunk_(max_chunk), stream_(stream),
      workspace_(drafter_workspace(model, release.config(), max_chunk, workspace)) {
    const DFlash2Config& c = config_;
    if (max_chunk < 1 || max_chunk > max_context || max_chunk > i32(c.sliding_window)) {
        throw std::invalid_argument("inkling2 dflash2: max_chunk must be in 1..min(max_context, the sliding window)");
    }
    const std::int32_t h = i32(c.hidden_size), kv = i32(std::size_t(c.kv_heads) * c.head_dim), v = i32(text_.vocab_size);
    const std::int32_t b = i32(c.block_size), window = i32(c.sliding_window);

    // The weights. BF16 matrices go through a staging buffer into their 8-bit forms; vectors widen to FP32.
    std::size_t staging_bytes = 0;
    for (const char* name : {"fc.weight", "layers.0.mlp.gate_proj.weight", "layers.0.self_attn.q_proj.weight"}) {
        staging_bytes = std::max(staging_bytes, release.words(name).size() * 2);
    }
    DeviceBuffer staging(staging_bytes);
    const auto q8 = [&](const std::string& name) {
        const auto& shape = release.shape(name);
        const std::int32_t rows = i32(shape[0]), cols = i32(shape[1]);
        const auto words        = release.words(name);
        cuda(cudaMemcpyAsync(staging.p, words.data(), words.size() * 2, cudaMemcpyHostToDevice, stream_), "matrix upload");
        const std::size_t values = padded(std::size_t(rows) * cols);
        weight_memory_.emplace_back(values + std::size_t(rows) * (cols / 32) * 2);
        auto* base = static_cast<std::byte*>(weight_memory_.back().p);
        ops::quantize_q8_g32(Tensor(staging.p, DType::BF16, {cols, rows}), reinterpret_cast<std::int8_t*>(base), base + values, stream_);
        cuda(cudaStreamSynchronize(stream_), "matrix encode"); // the staging buffer is reused
        return ops::DenseMatrix{base, QType::Q8_G32_FP16, rows, cols, base + values};
    };
    const auto fp32 = [&](const std::string& name, std::initializer_list<std::int32_t> ne) {
        const auto words = release.words(name);
        std::vector<float> values(words.size());
        for (std::size_t i = 0; i < words.size(); ++i) {
            const std::uint32_t bits = std::uint32_t(words[i]) << 16;
            std::memcpy(&values[i], &bits, 4);
        }
        weight_memory_.emplace_back(values.size() * 4);
        cuda(cudaMemcpy(weight_memory_.back().p, values.data(), values.size() * 4, cudaMemcpyHostToDevice), "vector upload");
        return Tensor(weight_memory_.back().p, DType::FP32, ne);
    };
    const auto bf16 = [&](const std::string& name, std::initializer_list<std::int32_t> ne) {
        const auto words = release.words(name);
        weight_memory_.emplace_back(words.size() * 2);
        cuda(cudaMemcpy(weight_memory_.back().p, words.data(), words.size() * 2, cudaMemcpyHostToDevice), "codebook upload");
        return Tensor(weight_memory_.back().p, DType::BF16, ne);
    };
    fc_          = q8("fc.weight");
    hidden_norm_ = fp32("hidden_norm.weight", {h});
    output_norm_ = fp32("norm.weight", {h});
    selector_    = q8("candidate_selector.hidden_projection.weight");
    predecessor_ = bf16("candidate_selector.predecessor_codebook", {ops::kDFlash2Rank, v});
    successor_   = bf16("candidate_selector.successor_codebook", {ops::kDFlash2Rank, v});
    const std::int32_t d = i32(c.head_dim);
    for (std::uint32_t i = 0; i < c.layers; ++i) {
        const std::string p = "layers." + std::to_string(i) + ".";
        Layer l;
        l.q              = q8(p + "self_attn.q_proj.weight");
        l.k              = q8(p + "self_attn.k_proj.weight");
        l.v              = q8(p + "self_attn.v_proj.weight");
        l.output         = q8(p + "self_attn.o_proj.weight");
        l.gate           = q8(p + "mlp.gate_proj.weight");
        l.up             = q8(p + "mlp.up_proj.weight");
        l.down           = q8(p + "mlp.down_proj.weight");
        l.attention_conv = q8(p + "attention_conv.kernel_projection.weight");
        l.mlp_conv       = q8(p + "mlp_conv.kernel_projection.weight");
        l.input_norm     = fp32(p + "input_layernorm.weight", {h});
        l.post_norm      = fp32(p + "post_attention_layernorm.weight", {h});
        l.q_norm         = fp32(p + "self_attn.q_norm.weight", {d});
        l.k_norm         = fp32(p + "self_attn.k_norm.weight", {d});
        l.attention_base = fp32(p + "attention_conv.base_kernel", {h, 4});
        l.mlp_base       = fp32(p + "mlp_conv.base_kernel", {h, 4});
        layers_.push_back(std::move(l));
    }

    // State: each layer's rings (its last `window` positions), the masks, the round's tokens and probabilities.
    const std::size_t ring = padded(std::size_t(kv) * window * (kv_fp16 ? 2 : 4));
    const std::size_t total = c.layers * 2 * ring + padded(std::size_t(h) * b * 4) + 2 * padded(std::size_t(b) * 4) + padded(std::size_t(b) * 8);
    state_memory_ = DeviceBuffer(total);
    auto* base    = static_cast<std::byte*>(state_memory_.p);
    std::size_t offset = 0;
    const auto carve = [&](DType dtype, std::initializer_list<std::int32_t> ne) {
        Tensor t(base + offset, dtype, ne);
        offset += padded(t.bytes());
        return t;
    };
    for (Layer& l : layers_) {
        l.k_ring = carve(kv_fp16 ? DType::FP16 : DType::FP32, {kv, window});
        l.v_ring = carve(kv_fp16 ? DType::FP16 : DType::FP32, {kv, window});
    }
    masks_  = carve(DType::FP32, {h, b});
    tokens_ = carve(DType::I32, {b});
    probs_  = carve(DType::FP32, {b});
    draws_  = carve(DType::I32, {2 * b}); // FP64 [block]
    cuda(cudaMemsetAsync(state_memory_.p, 0, total, stream_), "clear");
    // The masks: the target's embedding of the mask token, normed by its embedding norm (as the anchor's).
    {
        const Tensor table      = model_.tensor(model_.weights().embedding, {h, v});
        const Tensor embed_norm = model_.tensor(model_.weights().embedding_norm, {h});
        const std::int32_t mask = i32(c.mask_token_id);
        cuda(cudaMemcpyAsync(tokens_.data, &mask, 4, cudaMemcpyHostToDevice, stream_), "mask id");
        Tensor first = masks_.slice(1, 1, 1);
        ops::embedding_bf16_fp32(tokens_.slice(0, 0, 1), table, first, stream_);
        ops::rmsnorm_fp32(first, &embed_norm, text_.rms_norm_eps, first, stream_);
        for (std::int32_t i = 2; i < b; ++i) {
            cuda(cudaMemcpyAsync(static_cast<float*>(masks_.data) + std::size_t(i) * h, first.data, std::size_t(h) * 4, cudaMemcpyDeviceToDevice,
                                 stream_),
                 "masks");
        }
    }
    // The head the base logits read (as DSparkProgram's).
    const Tensor head = model_.tensor(model_.weights().head, {h, v});
    if (head_q8 != nullptr) {
        head_q8_ = *head_q8;
    } else if (!head_copy) {
        head_bf16_ = ops::DenseMatrix{head.data, QType::BF16, v, h};
    } else {
        const std::size_t codes = padded(std::size_t(v) * h / 2);
        head_memory_ = DeviceBuffer(codes + std::size_t(v) * (h / 32) * 2);
        head_q4_     = ops::Q4Matrix{head_memory_.p, static_cast<std::byte*>(head_memory_.p) + codes, v, h};
        ops::quantize_q4_rows(head, head_memory_.p, static_cast<std::byte*>(head_memory_.p) + codes, stream_);
    }
    cuda(cudaStreamSynchronize(stream_), "precompute");
}

DFlash2Program::~DFlash2Program() {
    // A copy may still run after wait_copy() (a stream wait) cleared copy_pending_: it reads this Program's rings.
    if (copy_done_ != nullptr) {
        (void)cudaEventSynchronize(copy_done_);
        cudaEventDestroy(copy_done_);
    }
}

void DFlash2Program::extend(std::span<const Tensor> taps, std::int32_t n) {
    const DFlash2Config& c = config_;
    wait_copy(); // its ring writes after a save_async copy that may still read them
    if (taps.size() != c.target_visits.size()) { throw std::invalid_argument("inkling2 dflash2: one tap per target visit"); }
    if (n < 1 || n > max_chunk_) { throw std::invalid_argument("inkling2 dflash2: bad extend size"); }
    const std::int32_t h = i32(c.hidden_size), kv = i32(std::size_t(c.kv_heads) * c.head_dim), nt = i32(taps.size());
    workspace_.reset();
    // The taps side by side ([tap 0; tap 1; ...] per column), as fc reads them.
    Tensor joined = workspace_.alloc(DType::FP32, {nt * h, n});
    for (std::int32_t i = 0; i < nt; ++i) {
        const Tensor& t = taps[static_cast<std::size_t>(i)];
        if (t.dtype != DType::FP32 || t.ne[0] != h || t.ne[1] < n || t.nb[0] != 4) {
            throw std::invalid_argument("inkling2 dflash2: a tap is not FP32 [hidden, >= n]");
        }
        cuda(cudaMemcpy2DAsync(static_cast<float*>(joined.data) + std::size_t(i) * h, std::size_t(nt) * h * 4, t.data, t.ne[1] == 1 ? h * 4 : t.nb[1],
                               std::size_t(h) * 4, static_cast<std::size_t>(n), cudaMemcpyDeviceToDevice, stream_),
             "taps");
    }
    const auto a8 = [&](const Tensor& x) {
        const std::size_t bytes = ops::a8_columns_bytes(i32(x.ne[0]), i32(x.ne[1]));
        Tensor storage          = workspace_.alloc(DType::U8, {i32(bytes)});
        return ops::a8_quantize_fp32(x, storage.data, bytes, stream_);
    };
    Tensor fused = workspace_.alloc(DType::FP32, {h, n});
    ops::dense_projection_a8_fp32(a8(joined), fc_, nullptr, fused, stream_);
    ops::rmsnorm_fp32(fused, &hidden_norm_, c.rms_norm_eps, fused, stream_);
    const ops::A8Columns fa = a8(fused);
    Tensor k = workspace_.alloc(DType::FP32, {kv, n}), v = workspace_.alloc(DType::FP32, {kv, n});
    for (Layer& l : layers_) {
        ops::dense_projection_a8_fp32(fa, l.k, nullptr, k, stream_);
        ops::dense_projection_a8_fp32(fa, l.v, nullptr, v, stream_);
        ops::head_rmsnorm_rope_fp32(k, i32(c.head_dim), l.k_norm, c.rms_norm_eps, true, position_, c.rope_theta, stream_);
        // A chunk longer than the ring keeps only its last `window` columns (the earlier ones would be overwritten).
        const std::int32_t keep = std::min<std::int32_t>(n, i32(l.k_ring.ne[1]));
        const Tensor ks = k.slice(1, n - keep, keep), vs = v.slice(1, n - keep, keep);
        ops::dspark_ring_write(ks, l.k_ring, position_ + (n - keep), stream_);
        ops::dspark_ring_write(vs, l.v_ring, position_ + (n - keep), stream_);
    }
    position_ += n;
}

std::vector<std::int32_t> DFlash2Program::draft(std::int32_t anchor, std::int32_t n_drafts, const ops::DraftPick* pick,
                                                std::span<const double> canonical, std::vector<float>* probs) {
    const DFlash2Config& c = config_;
    if (position_ < 1) { throw std::invalid_argument("inkling2 dflash2: drafting needs a committed context"); }
    if (anchor < 0 || static_cast<std::uint32_t>(anchor) >= text_.unpadded_vocab_size) { throw std::invalid_argument("inkling2 dflash2: bad anchor"); }
    if (n_drafts < 1 || n_drafts > max_drafts()) { throw std::invalid_argument("inkling2 dflash2: 1 to block_size - 1 drafts"); }
    const std::int32_t h = i32(c.hidden_size), kv = i32(std::size_t(c.kv_heads) * c.head_dim), q = i32(std::size_t(c.heads) * c.head_dim);
    const std::int32_t vocab = i32(text_.vocab_size), f = i32(c.intermediate_size), b = i32(c.block_size), groups = h / i32(c.conv_group);
    const bool sampled = pick != nullptr && pick->temperature > 0.0F;
    if (sampled && canonical.size() < static_cast<std::size_t>(n_drafts)) {
        throw std::invalid_argument("inkling2 dflash2: a sampled draft takes one uniform per draft");
    }
    workspace_.reset();
    cuda(cudaMemcpyAsync(tokens_.data, &anchor, 4, cudaMemcpyHostToDevice, stream_), "anchor upload");
    if (sampled) {
        cuda(cudaMemcpyAsync(draws_.data, canonical.data(), std::size_t(n_drafts) * 8, cudaMemcpyHostToDevice, stream_), "uniforms upload");
    }
    // The block: the anchor's normed embedding, then the masks.
    Tensor x = workspace_.alloc(DType::FP32, {h, b});
    cuda(cudaMemcpyAsync(x.data, masks_.data, std::size_t(h) * b * 4, cudaMemcpyDeviceToDevice, stream_), "block");
    {
        Tensor first            = x.slice(1, 0, 1);
        const Tensor embed_norm = model_.tensor(model_.weights().embedding_norm, {h});
        ops::embedding_bf16_fp32(tokens_.slice(0, 0, 1), model_.tensor(model_.weights().embedding, {h, vocab}), first, stream_);
        ops::rmsnorm_fp32(first, &embed_norm, text_.rms_norm_eps, first, stream_);
    }
    const auto a8 = [&](const Tensor& in) {
        const std::size_t bytes = ops::a8_columns_bytes(i32(in.ne[0]), i32(in.ne[1]));
        Tensor storage          = workspace_.alloc(DType::U8, {i32(bytes)});
        return ops::a8_quantize_fp32(in, storage.data, bytes, stream_);
    };
    const auto project = [&](const ops::A8Columns& in, const ops::DenseMatrix& m, Tensor& out) { ops::dense_projection_a8_fp32(in, m, nullptr, out, stream_); };
    Tensor n     = workspace_.alloc(DType::FP32, {h, b}), prepared = workspace_.alloc(DType::FP32, {h, b});
    Tensor delta = workspace_.alloc(DType::FP32, {4 * groups, b}), z = workspace_.alloc(DType::FP32, {h, b});
    Tensor qt = workspace_.alloc(DType::FP32, {q, b}), kt = workspace_.alloc(DType::FP32, {kv, b}), vt = workspace_.alloc(DType::FP32, {kv, b});
    Tensor attn = workspace_.alloc(DType::FP32, {q, b});
    Tensor g = workspace_.alloc(DType::FP32, {f, b}), u = workspace_.alloc(DType::FP32, {f, b});
    const ops::DSparkAttentionShape shape{i32(c.heads), i32(c.kv_heads), i32(c.head_dim), i32(c.sliding_window)};
    Tensor parts = workspace_.alloc(DType::FP32, {i32(ops::dspark_block_attention_workspace_bytes(shape, b) / 4)});
    for (Layer& l : layers_) {
        // Attention, its input and output each through the layer's dynamic convolution (side 0 in, side 1 out).
        ops::rmsnorm_fp32(x, &l.input_norm, c.rms_norm_eps, n, stream_);
        project(a8(n), l.attention_conv, delta);
        ops::dflash2_conv_fp32(n, l.attention_base, delta, i32(c.conv_group), 0, prepared, false, stream_);
        const ops::A8Columns pa = a8(prepared);
        project(pa, l.q, qt);
        project(pa, l.k, kt);
        project(pa, l.v, vt);
        ops::head_rmsnorm_rope_fp32(qt, i32(c.head_dim), l.q_norm, c.rms_norm_eps, true, position_, c.rope_theta, stream_);
        ops::head_rmsnorm_rope_fp32(kt, i32(c.head_dim), l.k_norm, c.rms_norm_eps, true, position_, c.rope_theta, stream_);
        ops::dspark_block_attention_fp32(qt, l.k_ring, l.v_ring, kt, vt, position_, shape, attn, parts, stream_);
        project(a8(attn), l.output, z);
        ops::dflash2_conv_fp32(z, l.attention_base, delta, i32(c.conv_group), 1, x, true, stream_);
        // The MLP, the same way.
        ops::rmsnorm_fp32(x, &l.post_norm, c.rms_norm_eps, n, stream_);
        project(a8(n), l.mlp_conv, delta);
        ops::dflash2_conv_fp32(n, l.mlp_base, delta, i32(c.conv_group), 0, prepared, false, stream_);
        const ops::A8Columns pm = a8(prepared);
        project(pm, l.gate, g);
        project(pm, l.up, u);
        ops::silu_mul_fp32(g, u, g, stream_);
        project(a8(g), l.down, z);
        ops::dflash2_conv_fp32(z, l.mlp_base, delta, i32(c.conv_group), 1, x, true, stream_);
    }
    ops::rmsnorm_fp32(x, &output_norm_, c.rms_norm_eps, x, stream_);
    // The drafted columns 1 .. n through the target's head (its logits before the softcap), and the selector's g.
    const Tensor drafted    = x.slice(1, 1, n_drafts);
    const ops::A8Columns xa = a8(drafted);
    Tensor logits           = workspace_.alloc(DType::FP32, {vocab, n_drafts});
    if (head_q4_.codes != nullptr) {
        ops::dense_projection_a8q4_fp32(xa, head_q4_, logits, stream_);
    } else if (head_q8_.values != nullptr) {
        Tensor xb = workspace_.alloc(DType::BF16, {h, n_drafts});
        ops::cast_fp32_to_bf16(drafted, xb, stream_);
        ops::projection_q8_fp32(xb, std::span<const ops::Q8Weight>(&head_q8_, 1), logits, stream_);
    } else {
        ops::dense_projection_fp32(drafted, head_bf16_, nullptr, logits, stream_);
    }
    Tensor gsel = workspace_.alloc(DType::FP32, {ops::kDFlash2Rank, n_drafts});
    project(xa, selector_, gsel);
    const std::int32_t valid = i32(text_.unpadded_vocab_size);
    const std::int32_t count = (valid + ops::kCandidateTile - 1) / ops::kCandidateTile * ops::kDFlash2TopK;
    Tensor cand   = workspace_.alloc(DType::I32, {count, n_drafts});
    Tensor ids    = workspace_.alloc(DType::I32, {ops::kDFlash2TopK, n_drafts});
    Tensor scores = workspace_.alloc(DType::FP32, {ops::kDFlash2TopK, n_drafts});
    ops::tile_top_candidates(logits, nullptr, valid, ops::kDFlash2TopK, cand, stream_);
    ops::dflash2_top_candidates(cand, logits, ids, scores, stream_);
    ops::dflash2_select(ids, scores, gsel, predecessor_, successor_, c.final_logit_softcap, sampled ? *pick : ops::DraftPick{},
                        sampled ? static_cast<const double*>(draws_.data) : nullptr, static_cast<std::int32_t*>(tokens_.data),
                        probs != nullptr ? static_cast<float*>(probs_.data) : nullptr, stream_);
    std::vector<std::int32_t> drafts(static_cast<std::size_t>(n_drafts));
    cuda(cudaMemcpyAsync(drafts.data(), static_cast<const std::int32_t*>(tokens_.data) + 1, drafts.size() * 4, cudaMemcpyDeviceToHost, stream_),
         "drafts");
    if (probs != nullptr) {
        probs->resize(drafts.size());
        cuda(cudaMemcpyAsync(probs->data(), probs_.data, drafts.size() * 4, cudaMemcpyDeviceToHost, stream_), "draft probabilities");
    }
    cuda(cudaStreamSynchronize(stream_), "draft");
    return drafts;
}

// The byte runs of a ring that hold positions [max(0, position - capacity), position): a snapshot copies those.
template <class F> void DFlash2Program::ring_rows(const Tensor& ring, F&& f) const {
    const std::int64_t capacity = ring.ne[1];
    const std::int64_t first    = std::max<std::int64_t>(0, position_ - capacity);
    const auto column           = static_cast<std::size_t>(ring.nb[1]);
    for (std::int64_t p = first; p < position_;) {
        const std::int64_t slot = p % capacity, run = std::min(position_ - p, capacity - slot);
        f(static_cast<std::size_t>(slot) * column, static_cast<std::size_t>(run) * column);
        p += run;
    }
}

std::size_t DFlash2Program::snapshot_bytes() const {
    std::size_t total = 0;
    for (const Layer& l : layers_) { total += l.k_ring.bytes() + l.v_ring.bytes(); }
    return total;
}

void DFlash2Program::wait_copy() {
    if (!copy_pending_) { return; }
    if (cudaEventQuery(copy_done_) != cudaSuccess) { cuda(cudaStreamWaitEvent(stream_, copy_done_, 0), "snapshot copy wait"); }
    copy_pending_ = false;
}

void DFlash2Program::save_async(Snapshot& snapshot, cudaStream_t copy) {
    if (snapshot.memory.size() < snapshot_bytes()) { throw std::invalid_argument("dflash2: the snapshot is too small"); }
    if (copy_done_ != nullptr) { cuda(cudaEventSynchronize(copy_done_), "snapshot copy"); } // the descriptors are its until then
    copy_pending_ = false;
    if (copy_done_ == nullptr) { cuda(cudaEventCreateWithFlags(&copy_done_, cudaEventDisableTiming), "event"); }
    // Only extend() writes the rings, and it waits for this copy; the copy follows this stream's work so far.
    cudaEvent_t ready = nullptr;
    cuda(cudaEventCreateWithFlags(&ready, cudaEventDisableTiming), "event");
    cuda(cudaEventRecord(ready, stream_), "record");
    cuda(cudaStreamWaitEvent(copy, ready, 0), "wait");
    cudaEventDestroy(ready);
    require_identity_mapped(snapshot.memory.data()); // the copy kernel writes it by its host address
    auto* out = static_cast<std::byte*>(snapshot.memory.data());
    std::vector<GatherCopy> copies;
    for (const Layer& l : layers_) {
        for (const Tensor* t : {&l.k_ring, &l.v_ring}) {
            ring_rows(*t, [&](std::size_t at, std::size_t bytes) {
                const std::size_t row = bytes % 2048 == 0 ? 2048 : bytes;
                copies.push_back({out + at, static_cast<const std::byte*>(t->data) + at, row, row, row, bytes / row});
            });
            out += t->bytes();
        }
    }
    split_gather_copies(copies, 256U << 10);
    if (copy_descriptors_.size() < copies.size() * sizeof(GatherCopy)) { copy_descriptors_ = PinnedHostBuffer(copies.size() * sizeof(GatherCopy)); }
    auto* descriptors = static_cast<GatherCopy*>(copy_descriptors_.data());
    std::copy(copies.begin(), copies.end(), descriptors);
    if (!copies.empty()) { gather_copies_paced(std::span<const GatherCopy>(descriptors, copies.size()), GatherPace{}, copy); }
    cuda(cudaEventRecord(copy_done_, copy), "record");
    copy_pending_     = true;
    snapshot.position = position_;
}

void DFlash2Program::save(Snapshot& snapshot) {
    if (snapshot.memory.size() < snapshot_bytes()) { throw std::invalid_argument("dflash2: the snapshot is too small"); }
    wait_copy();
    auto* out = static_cast<std::byte*>(snapshot.memory.data());
    CopyBatch copies(stream_);
    for (const Layer& l : layers_) {
        for (const Tensor* t : {&l.k_ring, &l.v_ring}) {
            ring_rows(*t, [&](std::size_t at, std::size_t bytes) { copies.add(out + at, static_cast<const std::byte*>(t->data) + at, bytes); });
            out += t->bytes();
        }
    }
    copies.flush();
    snapshot.position = position_;
}

void DFlash2Program::restore(const Snapshot& snapshot) {
    if (snapshot.position < 0) { throw std::invalid_argument("dflash2: an empty snapshot"); }
    wait_copy();
    position_      = snapshot.position; // ring_rows: the snapshot's positions
    const auto* in = static_cast<const std::byte*>(snapshot.memory.data());
    CopyBatch copies(stream_);
    for (Layer& l : layers_) {
        for (const Tensor* t : {&l.k_ring, &l.v_ring}) {
            ring_rows(*t, [&](std::size_t at, std::size_t bytes) { copies.add(static_cast<std::byte*>(t->data) + at, in + at, bytes); });
            in += t->bytes();
        }
    }
    copies.flush();
}

} // namespace ninfer::models::inkling2
