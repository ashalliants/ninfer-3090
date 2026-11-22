// Written for NInfer-3090. See forward.h.

#include "models/dsv4/execution/forward.h"

#include "core/weight_view.h"
#include "models/dsv4/execution/engram.h"
#include "models/dsv4/execution/engram_sidecar.h"
#include "ninfer/ops/cast.h"
#include "ninfer/ops/embedding.h"
#include "ninfer/ops/hc_sinkhorn.h"
#include "ninfer/ops/projection_fp32.h"
#include "ninfer/ops/rmsnorm.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <optional>
#include <stdexcept>
#include <string>

namespace ninfer::models::dsv4 {
namespace {

void cuda_ok(cudaError_t error, const char* what) {
    if (error != cudaSuccess) { throw std::runtime_error(std::string("dsv4 forward: ") + what + ": " + cudaGetErrorString(error)); }
}

constexpr std::size_t kSlack = 1 << 20;

// The Engram folder's checked contents, or nothing without a folder (the layers then run uninjected).
std::optional<EngramSidecar> load_sidecar(const Model& model, const ForwardOptions& options) {
    if (options.engram_dir.empty()) { return std::nullopt; }
    return load_engram_sidecar(options.engram_dir, model.config());
}

std::unique_ptr<qwen4_exp::NgramVolume> open_volume(const std::optional<EngramSidecar>& sidecar) {
    if (!sidecar) { return nullptr; }
    return std::make_unique<qwen4_exp::NgramVolume>(sidecar->volume_path, sidecar->table);
}

// The layer's Engram weights and how its injection runs.
struct EngramLayer {
    Weight kv;
    Tensor gate;
    ops::LinearPolicy policy = ops::LinearPolicy::AllowA8;
};

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

    // The Engram (declared before the workspace, whose size includes the injection's scratch).
    std::optional<EngramSidecar> sidecar;
    std::unique_ptr<qwen4_exp::NgramVolume> engram_volume;
    std::unique_ptr<Engram> engram;
    EngramHasher::History engram_history;
    std::vector<std::optional<EngramLayer>> engram_layers; // per model layer; engaged on Engram layers
    std::vector<std::uint32_t> engram_index;               // per model layer: the Engram's own index

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
    // Decode graphs (NINFER_DSV4_GRAPHS=0 turns them off). A one-column step runs each layer as a few CUDA graphs
    // around the steps that cannot be captured: the attention layer (its kernels take the position as an
    // argument) and the host's read of the routed ids. Graphs are captured on the first use, after one eager
    // one-column step has initialised the Ops, and live as long as the Forward.
    struct LayerGraphs {
        cudaGraphExec_t front = nullptr; // attention mixes, collapse, norm (only where the previous layer's tail did not run it)
        cudaGraphExec_t route = nullptr; // attention expand, FFN mixes, collapse, norm, router, dispatch
        cudaGraphExec_t shared = nullptr; // the shared expert
        cudaGraphExec_t attn_front = nullptr, attn_back = nullptr; // the attention layer's two stretches (AttentionRunner)
        cudaGraphExec_t tail  = nullptr; // routed experts, weighted sum, expand, then the next layer's front (or the head)
    };
    bool graphs = [] {
        const char* v = std::getenv("NINFER_DSV4_GRAPHS");
        return v == nullptr || std::string(v) != "0";
    }();
    cudaStream_t capture_stream = nullptr;
    std::vector<LayerGraphs> layer_graphs;
    bool warmed = false; // one eager one-column step has run
    DeviceBuffer tail_collapsed, tail_normed;
    bool recording = false; // only one-column steps (decode) are recorded
    double stage_seconds[8] = {};
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

    ~Impl() {
        for (LayerGraphs& g : layer_graphs) {
            for (cudaGraphExec_t e : {g.front, g.route, g.shared, g.tail, g.attn_front, g.attn_back}) {
                if (e != nullptr) { cudaGraphExecDestroy(e); }
            }
        }
        if (capture_stream != nullptr) { cudaStreamDestroy(capture_stream); }
    }

    // Runs `body(stream)`: directly, or (`use`) as the graph `exec`, captured from `body(capture_stream)` the
    // first time. A capture that fails throws (a decode must not silently change how it runs).
    template <class Body>
    void segment(cudaGraphExec_t& exec, bool use, Body&& body) {
        if (!use || !graphs) {
            body(stream);
            return;
        }
        if (exec == nullptr) {
            cudaGraph_t graph = nullptr;
            cudaError_t status = cudaStreamBeginCapture(capture_stream, cudaStreamCaptureModeRelaxed);
            if (status == cudaSuccess) {
                try {
                    body(capture_stream);
                } catch (...) {
                    cudaGraph_t partial = nullptr;
                    cudaStreamEndCapture(capture_stream, &partial);
                    if (partial != nullptr) { cudaGraphDestroy(partial); }
                    throw;
                }
                status = cudaStreamEndCapture(capture_stream, &graph);
            }
            if (status == cudaSuccess) { status = cudaGraphInstantiate(&exec, graph, 0); }
            if (graph != nullptr) { cudaGraphDestroy(graph); }
            if (status != cudaSuccess) {
                cudaGetLastError();
                exec = nullptr;
                throw std::runtime_error(std::string("dsv4 forward: CUDA graph capture failed: ") + cudaGetErrorString(status) +
                                         " (NINFER_DSV4_GRAPHS=0 decodes without graphs)");
            }
        }
        cuda_ok(cudaGraphLaunch(exec, stream), "graph launch");
    }

    static std::size_t workspace_bytes(const Model& model, const ForwardOptions& options, std::size_t engram_bytes) {
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
        return std::max({attention_bytes, ffn_bytes, engram_bytes}) + activations + 8 * kSlack;
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
          sidecar(load_sidecar(m, o)), engram_volume(open_volume(sidecar)),
          engram(sidecar ? std::make_unique<Engram>(m.config(), sidecar->params, *engram_volume, static_cast<std::uint32_t>(o.limits.max_chunk))
                         : nullptr),
          state(m.config(), o.limits),
          workspace(workspace_bytes(
              m, o, engram ? engram->apply_workspace_bytes(static_cast<std::uint32_t>(o.limits.max_chunk), ops::LinearPolicy::AllowA8) : 0)) {
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
        engram_layers.resize(config.num_hidden_layers);
        engram_index.assign(config.num_hidden_layers, 0);
        if (engram) {
            engram_history     = engram->hasher().start();
            std::uint32_t next = 0;
            for (std::uint32_t l = 0; l < config.num_hidden_layers; ++l) {
                const BlockWeights& b = model.weights().layers[l];
                if (!config.layers[l].engram) { continue; }
                if (!b.engram) { throw std::invalid_argument("dsv4 forward: an Engram layer has no Engram weights"); }
                engram_index[l]  = next++;
                engram_layers[l] = EngramLayer{native_weight(model.weight(b.engram->kv).view), weight_tensor(model.weight(b.engram->gate).view, {S, H}),
                                               model.weight(b.engram->kv).policy};
            }
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
        tail_collapsed = DeviceBuffer(std::size_t(H) * T * 2);
        tail_normed    = DeviceBuffer(std::size_t(H) * 2);
        layer_graphs.resize(config.num_hidden_layers);
        cuda_ok(cudaStreamCreateWithFlags(&capture_stream, cudaStreamNonBlocking), "capture stream");
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

bool Forward::engram_applied() const noexcept { return impl_->engram != nullptr; }

void Forward::reset_profile() noexcept {
    for (double& v : impl_->stage_seconds) { v = 0; }
    impl_->stage_calls = 0;
}

std::string Forward::profile_report() const {
    const Impl& s = *impl_;
    std::string out;
    if (!s.profile || s.stage_calls == 0) { return out; }
    static const char* names[6] = {"embedding", "attention mixes+collapse+norm", "attention layer", "attn expand + ffn route",
                                   "shared launch + ids wait + expert read (host)", "experts + next front (GPU)"};
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
    // The decode graphs are instantiated on the first warm decode step; 27 MiB were measured for 43 layers on the
    // real artifact (test_forward prints it), so each layer is allowed 1 MiB.
    const std::size_t graph_allowance = i.graphs ? std::size_t(i.layer_graphs.size()) << 20 : 0;
    return i.state.device_bytes() + i.residual_buffer.bytes + i.mixes_buffer.bytes + i.logits_buffer.bytes + i.workspace.capacity() +
           i.tail_collapsed.bytes + i.tail_normed.bytes + graph_allowance;
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

    // Engram: the hash history restarts with a sequence; the rows of every Engram layer for these tokens are read
    // from the volume on the host before any device work is queued (the ids depend on the tokens alone). A call
    // that fails leaves the history where it was, so a retry at the same start hashes alike.
    struct HistoryGuard {
        EngramHasher::History& history;
        EngramHasher::History saved;
        bool done = false;
        ~HistoryGuard() {
            if (!done) { history = saved; }
        }
    } history_guard{s.engram_history, s.engram_history};
    if (s.engram) {
        if (start == 0) { s.engram_history = s.engram->hasher().start(); }
        const Engram::Run run{&s.engram_history, tokens};
        s.engram->stage(std::span<const Engram::Run>(&run, 1));
    }
    // Embedding into one stream, then copied into the others.
    s.recording = s.profile && T == 1 && start >= 1;
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
    HcMixes attn_mixes   = s.mixes(0, T);
    HcMixes ffn_mixes[2] = {s.mixes(1, T), s.mixes(2, T)};
    Tensor pre           = s.columns(s.identity_pre, T);
    const bool use_graphs = s.graphs && T == 1 && s.warmed;
    const std::uint32_t L = c.num_hidden_layers;

    // The attention sublayer's front: this layer's mixes, the collapse of the streams and the norm.
    const auto attn_front = [&](std::uint32_t l, const Tensor& pre_in, Tensor& collapsed, Tensor& x, cudaStream_t st) {
        const BlockWeights& b   = s.model.weights().layers[l];
        const Tensor attn_fn    = weight_tensor(s.model.weight(b.attn_hc.fn).view, {S * H, (2 + S) * S});
        const Tensor attn_scale = weight_tensor(s.model.weight(b.attn_hc.scale).view, {3});
        const Tensor attn_base  = weight_tensor(s.model.weight(b.attn_hc.base).view, {(2 + S) * S});
        ops::hc_sinkhorn_mixes(residual, attn_fn, attn_scale, attn_base, c.rms_norm_eps, c.hc_epsilon,
                               static_cast<std::int32_t>(c.hc_iterations), attn_mixes.pre, attn_mixes.post, attn_mixes.comb, st);
        ops::hc_collapse(residual, pre_in, collapsed, st);
        ops::rmsnorm(collapsed, s.attention[l].views.attn_norm, c.rms_norm_eps, false, x, st);
    };

    for (std::uint32_t l = 0; l < L; ++l) {
        Impl::LayerGraphs& g = s.layer_graphs[l];
        // Buffers that live across the layer's segments; the same arena state gives the same addresses every
        // step, which is what the captured graphs rely on.
        auto layer_scope = s.workspace.scope();
        Tensor collapsed = s.workspace.alloc(DType::BF16, {H, T});
        Tensor x         = s.workspace.alloc(DType::BF16, {H, T});
        Tensor out       = s.workspace.alloc(DType::BF16, {H, T});
        const bool front_here = l == 0 || s.engram_layers[l].has_value(); // else the previous layer's tail ran it
        if (s.engram_layers[l]) {
            // The residual streams receive the layer's gated n-gram value before its block.
            EngramLayer& e = *s.engram_layers[l];
            auto scope     = s.workspace.scope();
            (void)s.engram->apply(s.engram_index[l], e.kv, e.gate, residual, e.policy, s.workspace, stream);
        }
        s.begin();
        if (front_here) {
            const Tensor pre_in = pre;
            s.segment(g.front, use_graphs, [&](cudaStream_t st) { attn_front(l, pre_in, collapsed, x, st); });
        }
        s.end(1);
        s.begin();
        {
            auto scope = s.workspace.scope();
            struct Runner final : AttentionRunner {
                Impl& impl;
                Impl::LayerGraphs& graphs;
                Runner(Impl& i, Impl::LayerGraphs& g) : impl(i), graphs(g) {}
                void run(int slot, const std::function<void(cudaStream_t)>& body) override {
                    impl.segment(slot == 0 ? graphs.attn_front : graphs.attn_back, true, body);
                }
            } runner(s, g);
            attention_layer(c, l, s.attention[l].views, s.state, x, start, out, s.workspace, stream, use_graphs ? &runner : nullptr);
        }
        s.end(2);
        s.begin();
        // FFN sublayer; its mixes alternate between two buffers so the next block can collapse with them.
        HcMixes& mixes  = ffn_mixes[l & 1];
        FfnInput input{residual, &attn_mixes.pre, mixes};
        const FfnParams fp = ffn_params(c, l);
        const std::int32_t entries = fp.top_k * T;
        FfnCall call = ffn_carve(fp, T, s.workspace);
        s.segment(g.route, use_graphs, [&](cudaStream_t st) {
            ops::hc_expand_mix(out, attn_mixes.post, attn_mixes.comb, residual, st);
            ffn_route(fp, s.ffn[l], call, input, st);
        });
        // The ids start their way to the host now, and the shared expert (which does not depend on them) is
        // enqueued before the source waits for them, so a source that reads records from a file overlaps those
        // reads with the shared expert's work.
        const auto* ids = static_cast<const std::int32_t*>(call.ids.data);
        s.end(3);
        s.begin();
        s.experts.begin(l, ids, entries, stream);
        s.segment(g.shared, use_graphs, [&](cudaStream_t st) { ffn_shared(fp, s.ffn[l], call, s.workspace, st); });
        const ops::MoeExpertSource source = s.experts.prepare(l, ids, entries, stream);
        s.end(4);
        s.begin();
        const bool last = l + 1 == L;
        const bool next_front = !last && !s.engram_layers[l + 1];
        const Tensor mixes_pre = mixes.pre;
        s.segment(g.tail, use_graphs, [&](cudaStream_t st) {
            ffn_experts(fp, call, source, input, st);
            if (next_front) {
                attn_front(l + 1, mixes_pre, collapsed, x, st);
            } else if (last) {
                // Final collapse with the last FFN's pre, then the last position only through norm and head.
                Tensor all(s.tail_collapsed.p, DType::BF16, {H, T});
                ops::hc_collapse(residual, mixes_pre, all, st);
                Tensor last_col(static_cast<std::byte*>(all.data) + std::size_t(T - 1) * H * 2, DType::BF16, {H, 1});
                Tensor normed(s.tail_normed.p, DType::BF16, {H, 1});
                ops::rmsnorm(last_col, s.output_norm, c.rms_norm_eps, false, normed, st);
                ops::projection_fp32(normed, s.head, s.logits, st);
            }
        });
        pre = mixes.pre;
        s.end(5);
    }
    if (T == 1) { s.warmed = true; }
    if (s.recording) { ++s.stage_calls; }
    s.position     = start + T;
    s.last_columns = T;
    history_guard.done = true;
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
