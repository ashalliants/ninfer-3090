// Written for NInfer-3090. See forward.h.

#include "models/dsv4/execution/forward.h"

#include "core/weight_view.h"
#include "ninfer/ops/cast.h"
#include "ninfer/ops/embedding.h"
#include "ninfer/ops/hc_sinkhorn.h"
#include "ninfer/ops/projection_fp32.h"
#include "ninfer/ops/rmsnorm.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>

namespace ninfer::models::dsv4 {
namespace {

void cuda_ok(cudaError_t error, const char* what) {
    if (error != cudaSuccess) { throw std::runtime_error(std::string("dsv4 forward: ") + what + ": " + cudaGetErrorString(error)); }
}

constexpr std::size_t kSlack = 1 << 20;

} // namespace

struct Forward::Impl {
    const Model& model;
    ExpertSource& experts;
    ForwardOptions options;
    cudaStream_t stream;
    const TextConfig& config;
    FfnParams ffn_params_;

    std::vector<AttentionBinding> attention;
    std::vector<FfnWeights> ffn;
    std::vector<DeviceBuffer> ffn_norms; // BF16 copies of each FFN norm (the artifact stores FP32)
    std::vector<DeviceBuffer> owned;     // BF16 copy of the output norm
    Tensor output_norm;
    Weight embedding_table, head;

    AttentionState state;
    DeviceBuffer residual_buffer, mixes_buffer, identity_buffer, logits_buffer, tokens_buffer;
    Tensor identity_pre;
    DeviceArena workspace;
    Tensor residual_view;
    Tensor residual_last;
    Tensor logits;
    std::int32_t position = 0;
    std::int32_t last_columns = 0; // the last call's token count: what residual() exposes
    // NINFER_DSV4_PROFILE=1: each stage is timed on the host between stream synchronizations (slows the run
    // down; profile_report() returns the numbers).
    bool profile = [] {
        const char* v = std::getenv("NINFER_DSV4_PROFILE");
        return v != nullptr && v[0] != '\0' && std::string(v) != "0";
    }();
    bool recording = false; // only decode steps from position 16 on (the warm ones) are recorded
    double stage_seconds[6] = {};
    std::int64_t stage_calls = 0;
    std::chrono::steady_clock::time_point mark;
    void begin() {
        if (!recording) { return; }
        cudaStreamSynchronize(stream);
        mark = std::chrono::steady_clock::now();
    }
    void end(int stage) {
        if (!recording) { return; }
        cudaStreamSynchronize(stream);
        stage_seconds[stage] += std::chrono::duration<double>(std::chrono::steady_clock::now() - mark).count();
    }

    static std::size_t workspace_bytes(const Model& model, const ForwardOptions& options) {
        const TextConfig& c = model.config();
        const std::int32_t T = static_cast<std::int32_t>(options.limits.max_chunk);
        std::size_t attention_bytes = attention_workspace_bytes(c, options.limits);
        std::size_t ffn_bytes       = 0;
        const FfnParams p           = ffn_params(c, 0);
        DeviceBuffer norm_owner;
        ffn_bytes = ffn_workspace_bytes(p, bind_ffn_weights(model, 0, norm_owner), T);
        const std::size_t H = c.hidden_size;
        // collapsed + normed activations of both sublayers, the attention output, the final norm and head.
        const std::size_t activations = 4 * H * T * 2 + 2 * H * 2 + std::size_t(c.vocab_size) * 4;
        return std::max(attention_bytes, ffn_bytes) + activations + 8 * kSlack;
    }

    Tensor norm_bf16(WeightId id, std::int32_t n) {
        Tensor stored = weight_tensor(model.weight(id).view, {n});
        if (stored.dtype == DType::BF16) { return stored; }
        owned.emplace_back(static_cast<std::size_t>(n) * 2);
        Tensor rounded(owned.back().p, DType::BF16, {n});
        ops::cast_fp32_to_bf16(stored, rounded, stream);
        return rounded;
    }

    Impl(const Model& m, ExpertSource& e, ForwardOptions o, cudaStream_t s)
        : model(m), experts(e), options(o), stream(s), config(m.config()), ffn_params_(ffn_params(m.config(), 0)),
          state(m.config(), o.limits), workspace(workspace_bytes(m, o)) {
        const std::int32_t H = static_cast<std::int32_t>(config.hidden_size), S = static_cast<std::int32_t>(config.hc_streams),
                           T = static_cast<std::int32_t>(options.limits.max_chunk);
        if (S != ops::kHcSinkhornStreams) { throw std::invalid_argument("dsv4 forward: four hyper-connection streams are required"); }
        owned.reserve(4);
        ffn_norms.reserve(config.num_hidden_layers);
        for (std::uint32_t l = 0; l < config.num_hidden_layers; ++l) {
            const BlockWeights& b = model.weights().layers[l];
            attention.push_back(bind_attention_views(model, b.attention, stream));
            ffn_norms.emplace_back();
            ffn.push_back(bind_ffn_weights(model, l, ffn_norms.back()));
        }
        output_norm     = norm_bf16(model.weights().output_norm, H);
        embedding_table = native_weight(model.weight(model.weights().token_embedding).view);
        head            = native_weight(model.weight(model.weights().output_head).view);

        residual_buffer = DeviceBuffer(std::size_t(S) * H * T * 2);
        residual_view   = Tensor(residual_buffer.p, DType::BF16, {S * H, T});
        // Three sets of mixes (the attention's, and two FFN ones alternating between blocks) and the identity pre.
        const std::size_t per_set = (std::size_t(S) * 2 + std::size_t(S) * S) * T * 4;
        mixes_buffer              = DeviceBuffer(3 * per_set + 3 * 256);
        identity_buffer           = DeviceBuffer(std::size_t(S) * T * 4);
        std::vector<float> identity(std::size_t(S) * T, 0.0F);
        for (std::int32_t t = 0; t < T; ++t) { identity[std::size_t(t) * S] = 1.0F; }
        identity_buffer.copy_from_host(identity.data(), identity.size() * 4);
        identity_pre  = Tensor(identity_buffer.p, DType::FP32, {S, T});
        logits_buffer = DeviceBuffer(std::size_t(config.vocab_size) * 4);
        logits        = Tensor(logits_buffer.p, DType::FP32, {static_cast<std::int32_t>(config.vocab_size), 1});
        tokens_buffer = DeviceBuffer(std::size_t(T) * 4);
    }

    // Mixes of T columns carved from one of the three sets.
    HcMixes mixes(int which, std::int32_t T) {
        DeviceArena arena(DeviceSpan{static_cast<std::byte*>(mixes_buffer.p) + std::size_t(which) * (mixes_buffer.bytes / 3), mixes_buffer.bytes / 3});
        return alloc_hc_mixes(arena, ffn_params_, T);
    }

    Tensor columns(const Tensor& whole, std::int32_t T) const {
        Tensor t = whole;
        t.ne[1]  = T;
        return t;
    }
};

Forward::Forward(const Model& model, ExpertSource& experts, ForwardOptions options, cudaStream_t stream)
    : impl_(std::make_unique<Impl>(model, experts, options, stream)) {}

Forward::~Forward() = default;

std::string Forward::profile_report() const {
    const Impl& s = *impl_;
    std::string out;
    if (!s.profile || s.stage_calls == 0) { return out; }
    static const char* names[6] = {"embedding", "attention mixes+collapse+norm", "attention layer", "attention expand", "ffn sublayer", "head"};
    for (int i = 0; i < 6; ++i) {
        char line[160];
        std::snprintf(line, sizeof line, "dsv4 profile: %-30s %9.3f ms per decoded token\n", names[i],
                      1000.0 * s.stage_seconds[i] / double(s.stage_calls));
        out += line;
    }
    return out;
}

std::size_t Forward::device_bytes() const noexcept {
    const Impl& i = *impl_;
    return i.state.device_bytes() + i.residual_buffer.bytes + i.mixes_buffer.bytes + i.logits_buffer.bytes + i.workspace.capacity();
}

const Tensor& Forward::residual() const {
    // The buffer holds max_chunk columns; the view is the last call's.
    Impl& s = *impl_;
    s.residual_last = s.residual_view;
    s.residual_last.ne[1] = std::max(s.last_columns, 1);
    return s.residual_last;
}

const Tensor& Forward::run(std::span<const std::int32_t> tokens, std::int32_t start) {
    Impl& s              = *impl_;
    const TextConfig& c  = s.config;
    const std::int32_t T = static_cast<std::int32_t>(tokens.size());
    const std::int32_t H = static_cast<std::int32_t>(c.hidden_size), S = static_cast<std::int32_t>(c.hc_streams);
    if (T < 1 || T > static_cast<std::int32_t>(s.options.limits.max_chunk)) { throw std::invalid_argument("dsv4 forward: chunk length outside [1, max_chunk]"); }
    if (start != 0 && start != s.position) { throw std::invalid_argument("dsv4 forward: start is not the next position"); }
    if (std::int64_t(start) + T > s.options.limits.max_context) { throw std::invalid_argument("dsv4 forward: context exceeds max_context"); }
    cudaStream_t stream = s.stream;

    // Embedding into one stream, then copied into the others.
    s.recording = s.profile && T == 1 && start >= 16;
    s.begin();
    cuda_ok(cudaMemcpyAsync(s.tokens_buffer.p, tokens.data(), std::size_t(T) * 4, cudaMemcpyHostToDevice, stream), "token upload");
    Tensor ids(s.tokens_buffer.p, DType::I32, {T});
    Tensor residual = s.columns(s.residual_view, T);
    Tensor first(s.residual_buffer.p, DType::BF16, {H, T});
    // Stream 0 of every column is the embedding; the layout is [S * H, T], stream-major within a column.
    {
        auto scope = s.workspace.scope();
        Tensor emb = s.workspace.alloc(DType::BF16, {H, T});
        ops::embedding(ids, s.embedding_table, emb, stream);
        for (std::int32_t k = 0; k < S; ++k) {
            cuda_ok(cudaMemcpy2DAsync(static_cast<std::byte*>(residual.data) + std::size_t(k) * H * 2, std::size_t(S) * H * 2, emb.data,
                                      std::size_t(H) * 2, std::size_t(H) * 2, T, cudaMemcpyDeviceToDevice, stream),
                    "embedding copy");
        }
    }

    s.end(0);
    s.begin();
    HcMixes attn_mixes = s.mixes(0, T);
    HcMixes ffn_mixes[2] = {s.mixes(1, T), s.mixes(2, T)};
    const Tensor* pre    = &s.identity_pre;
    Tensor pre_view      = s.columns(s.identity_pre, T);
    pre                  = &pre_view;

    for (std::uint32_t l = 0; l < c.num_hidden_layers; ++l) {
        const BlockWeights& b = s.model.weights().layers[l];
        // Attention sublayer.
        {
            auto scope = s.workspace.scope();
            const Tensor attn_fn    = weight_tensor(s.model.weight(b.attn_hc.fn).view, {S * H, (2 + S) * S});
            const Tensor attn_scale = weight_tensor(s.model.weight(b.attn_hc.scale).view, {3});
            const Tensor attn_base  = weight_tensor(s.model.weight(b.attn_hc.base).view, {(2 + S) * S});
            ops::hc_sinkhorn_mixes(residual, attn_fn, attn_scale, attn_base, c.rms_norm_eps, c.hc_epsilon,
                                   static_cast<std::int32_t>(c.hc_iterations), attn_mixes.pre, attn_mixes.post, attn_mixes.comb, stream);
            Tensor collapsed = s.workspace.alloc(DType::BF16, {H, T});
            Tensor x         = s.workspace.alloc(DType::BF16, {H, T});
            Tensor out       = s.workspace.alloc(DType::BF16, {H, T});
            ops::hc_collapse(residual, *pre, collapsed, stream);
            ops::rmsnorm(collapsed, s.attention[l].views.attn_norm, c.rms_norm_eps, false, x, stream);
            s.end(1);
            s.begin();
            attention_layer(c, l, s.attention[l].views, s.state, x, start, out, s.workspace, stream);
            s.end(2);
            s.begin();
            ops::hc_expand_mix(out, attn_mixes.post, attn_mixes.comb, residual, stream);
        }
        // FFN sublayer; its mixes alternate between two buffers so the next block can collapse with them.
        s.end(3);
        s.begin();
        HcMixes& mixes = ffn_mixes[l & 1];
        FfnInput input{residual, &attn_mixes.pre, mixes};
        ffn_forward(ffn_params(c, l), s.ffn[l], l, s.experts, input, s.workspace, stream);
        s.end(4);
        pre = &mixes.pre;
        s.begin();
    }

    // Final collapse with the last FFN's pre, then the last position only through norm and head.
    {
        auto scope = s.workspace.scope();
        Tensor collapsed = s.workspace.alloc(DType::BF16, {H, T});
        ops::hc_collapse(residual, *pre, collapsed, stream);
        Tensor last(static_cast<std::byte*>(collapsed.data) + std::size_t(T - 1) * H * 2, DType::BF16, {H, 1});
        Tensor normed = s.workspace.alloc(DType::BF16, {H, 1});
        ops::rmsnorm(last, s.output_norm, c.rms_norm_eps, false, normed, stream);
        ops::projection_fp32(normed, s.head, s.logits, stream);
    }
    s.end(5);
    if (s.recording) { ++s.stage_calls; }
    s.position     = start + T;
    s.last_columns = T;
    return s.logits;
}

std::int32_t Forward::argmax() {
    // The logits are FP32 and the device argmax reads BF16, so the largest of the vocabulary's values is
    // found on the host (one 0.5 MB download per generated token).
    Impl& s = *impl_;
    std::vector<float> host(s.config.vocab_size);
    cuda_ok(cudaMemcpyAsync(host.data(), s.logits.data, host.size() * 4, cudaMemcpyDeviceToHost, s.stream), "logits download");
    cuda_ok(cudaStreamSynchronize(s.stream), "logits sync");
    return static_cast<std::int32_t>(std::max_element(host.begin(), host.end()) - host.begin());
}

} // namespace ninfer::models::dsv4
