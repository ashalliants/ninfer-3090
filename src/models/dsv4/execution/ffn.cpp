// Written for NInfer-3090. See ffn.h.

#include "models/dsv4/execution/ffn.h"

#include "core/weight_view.h"
#include "models/dsv4/model.h"
#include "ninfer/ops/hc_sinkhorn.h"
#include "ninfer/ops/moe_mix.h"
#include "ninfer/ops/moe_route_sqrt_softplus.h"
#include "ninfer/ops/projection_fp32.h"
#include "ninfer/ops/rmsnorm.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <vector>
#include <stdexcept>

namespace ninfer::models::dsv4 {
namespace {

constexpr std::size_t kAlign = 256;

std::size_t padded(std::size_t bytes) { return (bytes + kAlign - 1) / kAlign * kAlign; }
// An allocation of `elements` values of `size` bytes, as DeviceArena::alloc pads it.
std::size_t tensor_bytes(std::size_t elements, std::size_t size) { return padded(elements * size); }

std::size_t linear_scratch(const FfnWeights& w, std::int32_t columns) {
    const auto need = [&](const Weight& weight) {
        return ops::linear_workspace_capacity_bytes(weight.qtype, weight.n, weight.k, w.shared_policy, columns, columns);
    };
    return std::max({need(w.shared_gate), need(w.shared_up), need(w.shared_down)}) + 2 * kAlign;
}

std::int32_t max_jobs(const FfnParams& p, std::int32_t columns) { return std::min(p.experts, p.top_k * columns); }

ops::offloaded_moe::MoeShape shape_of(const FfnParams& p) { return {p.hidden, p.intermediate}; }

// Whether two tensors' byte ranges intersect (a view into the middle of a buffer counts).
bool overlaps(const Tensor& a, const Tensor& b) {
    const auto begin_a = reinterpret_cast<std::uintptr_t>(a.data), begin_b = reinterpret_cast<std::uintptr_t>(b.data);
    return begin_a < begin_b + b.bytes() && begin_b < begin_a + a.bytes();
}

void require_supported(const FfnParams& p) {
    if (p.streams != ops::kHcSinkhornStreams) { throw std::invalid_argument("dsv4 ffn: the hyper-connection ops serve four streams"); }
    if (p.hidden <= 0 || p.experts <= 0 || p.top_k <= 0 || p.intermediate <= 0) { throw std::invalid_argument("dsv4 ffn: empty geometry"); }
}

} // namespace

FfnParams ffn_params(const TextConfig& config, std::uint32_t layer) {
    FfnParams p;
    p.hidden        = static_cast<std::int32_t>(config.hidden_size);
    p.streams       = static_cast<std::int32_t>(config.hc_streams);
    p.experts       = static_cast<std::int32_t>(config.experts);
    p.top_k         = static_cast<std::int32_t>(config.experts_per_token);
    p.intermediate  = static_cast<std::int32_t>(config.moe_intermediate);
    p.norm_eps      = config.rms_norm_eps;
    p.hc_eps        = config.hc_epsilon;
    p.hc_iterations = static_cast<std::int32_t>(config.hc_iterations);
    p.route_scale   = config.route_scale;
    p.norm_topk     = config.norm_topk;
    p.clamp_shared  = config.layers.at(layer).clamp_shared;
    // moe_experts_narrow clamps the routed experts' SwiGLU at the constant 10 (Dsv4Dims); another limit has no route yet.
    if (config.layers.at(layer).clamp_experts != 10.0F) {
        throw std::invalid_argument("dsv4 ffn: the routed experts' SwiGLU clamp must be 10 (the narrow Op's constant)");
    }
    return p;
}

FfnWeights bind_ffn_weights(const Model& model, std::uint32_t layer, DeviceBuffer& norm_owner) {
    const TextConfig& c   = model.config();
    const BlockWeights& b = model.weights().layers.at(layer);
    const std::int32_t H = static_cast<std::int32_t>(c.hidden_size), S = static_cast<std::int32_t>(c.hc_streams),
                       E = static_cast<std::int32_t>(c.experts), mix = (2 + S) * S;
    FfnWeights w;
    w.hc_fn        = weight_tensor(model.weight(b.ffn_hc.fn).view, {S * H, mix});
    w.hc_scale     = weight_tensor(model.weight(b.ffn_hc.scale).view, {3});
    w.hc_base      = weight_tensor(model.weight(b.ffn_hc.base).view, {mix});
    {
        const Tensor stored = weight_tensor(model.weight(b.moe.norm).view, {H});
        if (stored.dtype == DType::BF16) {
            w.norm = stored;
        } else {
            std::vector<float> values(static_cast<std::size_t>(H));
            if (cudaMemcpy(values.data(), stored.data, values.size() * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) {
                throw std::runtime_error("dsv4 ffn: cannot read the stored norm");
            }
            std::vector<std::uint16_t> bits(values.size());
            for (std::size_t i = 0; i < values.size(); ++i) {
                std::uint32_t u;
                std::memcpy(&u, &values[i], 4);
                u += 0x7fffU + ((u >> 16) & 1U); // round to nearest even (norm weights are finite)
                bits[i] = static_cast<std::uint16_t>(u >> 16);
            }
            norm_owner = DeviceBuffer(bits.size() * 2);
            norm_owner.copy_from_host(bits.data(), bits.size() * 2);
            w.norm = Tensor(norm_owner.p, DType::BF16, {H});
        }
    }
    w.router       = weight_tensor(model.weight(b.moe.router).view, {H, E});
    w.router_bias  = weight_tensor(model.weight(b.moe.router_bias).view, {E});
    w.shared_gate  = native_weight(model.weight(b.moe.shared.gate).view);
    w.shared_up    = native_weight(model.weight(b.moe.shared.up).view);
    w.shared_down  = native_weight(model.weight(b.moe.shared.down).view);
    w.shared_policy = model.weight(b.moe.shared.gate).policy;
    if (model.weight(b.moe.shared.up).policy != w.shared_policy || model.weight(b.moe.shared.down).policy != w.shared_policy) {
        throw std::invalid_argument("dsv4 ffn: the shared expert's three projections must share one activation policy");
    }
    return w;
}

HcMixes alloc_hc_mixes(DeviceArena& arena, const FfnParams& p, std::int32_t columns) {
    HcMixes m;
    m.pre  = arena.alloc(DType::FP32, {p.streams, columns});
    m.post = arena.alloc(DType::FP32, {p.streams, columns});
    m.comb = arena.alloc(DType::FP32, {p.streams * p.streams, columns});
    return m;
}

std::size_t ffn_workspace_bytes(const FfnParams& p, const FfnWeights& w, std::int32_t T) {
    require_supported(p);
    const std::size_t H = static_cast<std::size_t>(p.hidden), I = static_cast<std::size_t>(p.intermediate), E = static_cast<std::size_t>(p.experts),
                      K = static_cast<std::size_t>(p.top_k), cols = static_cast<std::size_t>(T);
    const std::int32_t entries = p.top_k * T, jobs = max_jobs(p, T);
    std::size_t total = 0;
    total += 2 * tensor_bytes(H * cols, 2);                     // collapsed, normed
    total += tensor_bytes(E * cols, 4);                         // router logits
    total += tensor_bytes(K * cols, 4) * 2;                     // ids, weights
    total += tensor_bytes(cols, 4);                             // shared gate (unused by V4, the Op's struct wants one)
    total += padded(ops::moe_dispatch_bytes(p.experts, entries)) + kAlign;
    total += padded(ops::moe_experts_narrow_workspace_bytes(shape_of(p), jobs, entries, T)) + kAlign;
    total += tensor_bytes(H * K * cols, 2);                     // routed outputs
    total += 3 * tensor_bytes(I * cols, 2);                     // gate, up, activation
    total += 2 * tensor_bytes(H * cols, 2);                     // shared output, y
    total += linear_scratch(w, T);
    return total + 4 * kAlign;
}

void ffn_forward(const FfnParams& p, const FfnWeights& w, std::uint32_t layer, ExpertSource& experts, FfnInput in, DeviceArena& arena,
                 cudaStream_t stream) {
    require_supported(p);
    Tensor& residual = in.residual;
    const std::int32_t T = residual.ne[1];
    if (residual.ne[0] != p.streams * p.hidden) { throw std::invalid_argument("dsv4 ffn: residual is not [streams * hidden, T]"); }
    if (in.collapse_pre != nullptr && (overlaps(*in.collapse_pre, in.mixes.pre) || overlaps(*in.collapse_pre, in.mixes.post) ||
                                       overlaps(*in.collapse_pre, in.mixes.comb))) {
        throw std::invalid_argument("dsv4 ffn: collapse_pre must not overlap any of this call's own mixes buffers (double-buffer the mixes)");
    }
    const auto scope     = arena.scope();
    const std::int32_t K = p.top_k, E = p.experts, entries = K * T, jobs = max_jobs(p, T);

    // Mixes of this sublayer, then the collapsed, normalised input.
    ops::hc_sinkhorn_mixes(residual, w.hc_fn, w.hc_scale, w.hc_base, p.norm_eps, p.hc_eps, p.hc_iterations, in.mixes.pre, in.mixes.post,
                           in.mixes.comb, stream);
    Tensor collapsed = arena.alloc(DType::BF16, {p.hidden, T});
    Tensor x         = arena.alloc(DType::BF16, {p.hidden, T});
    ops::hc_collapse(residual, in.collapse_pre != nullptr ? *in.collapse_pre : in.mixes.pre, collapsed, stream);
    ops::rmsnorm(collapsed, w.norm, p.norm_eps, false, x, stream);

    // Router: FP32 logits from the BF16 weights, then the selection.
    Tensor logits  = arena.alloc(DType::FP32, {E, T});
    Tensor ids     = arena.alloc(DType::I32, {K, T});
    Tensor weights = arena.alloc(DType::FP32, {K, T});
    const Tensor* router[] = {&w.router};
    ops::projection_fp32(x, router, logits, stream);
    ops::moe_route_sqrt_softplus(logits, w.router_bias, K, p.route_scale, p.norm_topk, ids, weights, stream);

    // Routed experts: dispatch, then the records from wherever they live. The ids start their way to the
    // host now, and the shared expert (which does not depend on them) is enqueued before the source waits for
    // them, so a source that reads records from a file overlaps those reads with the shared expert's work.
    ops::MoeRouting routing;
    routing.ids         = ids;
    routing.weights     = weights;
    routing.shared_gate = arena.alloc(DType::FP32, {T}); // unused: V4's shared expert is not gated
    const DeviceSpan dispatch_bytes = arena.alloc_bytes(ops::moe_dispatch_bytes(E, entries));
    ops::MoeDispatch dispatch       = ops::carve_moe_dispatch(dispatch_bytes.data, E, entries);
    ops::moe_dispatch(routing, E, dispatch, nullptr, stream);
    experts.begin(layer, static_cast<const std::int32_t*>(ids.data), entries, stream);

    // Shared expert: gate and up, the clamped SwiGLU, down.
    Tensor gate = arena.alloc(DType::BF16, {p.intermediate, T}), up = arena.alloc(DType::BF16, {p.intermediate, T}),
           act = arena.alloc(DType::BF16, {p.intermediate, T}), shared = arena.alloc(DType::BF16, {p.hidden, T}),
           y = arena.alloc(DType::BF16, {p.hidden, T});
    const auto project = [&](const Tensor& from, const Weight& weight, Tensor& to) {
        const auto call = arena.scope(); // the Op's transient storage, stream-ordered before the next call reuses it
        ops::linear(from, weight, to, w.shared_policy, arena, stream);
    };
    project(x, w.shared_gate, gate);
    project(x, w.shared_up, up);
    ops::clamped_swiglu(gate, up, p.clamp_shared, act, stream);
    project(act, w.shared_down, shared);

    const ops::MoeExpertSource source = experts.prepare(layer, static_cast<const std::int32_t*>(ids.data), entries, stream);
    const DeviceSpan expert_ws        = arena.alloc_bytes(ops::moe_experts_narrow_workspace_bytes(shape_of(p), jobs, entries, T));
    Tensor routed                     = arena.alloc(DType::BF16, {p.hidden, entries});
    ops::moe_experts_narrow(shape_of(p), x, dispatch, source, K, jobs, expert_ws.data, routed, stream);
    ops::weighted_slot_sum(routed, weights, shared, y, stream);

    // Back into the streams.
    ops::hc_expand_mix(y, in.mixes.post, in.mixes.comb, residual, stream);
}

} // namespace ninfer::models::dsv4
