#pragma once

// Written for NInfer-3090.
//
// The FFN sublayer of a DeepSeek-V4.1-Flash block, as one chain of Ops (the chain validated on the real
// layer-0 weights by tests/models/dsv4/test_ffn_layer.cpp against tools/dsv4/hyper_connection.py + moe.py):
//
//   hc_sinkhorn_mixes(R)                         this sublayer's pre / post / comb
//   hc_collapse(R, pre) -> rmsnorm               the sublayer input (pre: see FfnInput::collapse_pre)
//   projection_fp32(router) -> moe_route_sqrt_softplus     the k experts and their weights per column
//   moe_dispatch -> ExpertSource -> moe_experts_narrow     the routed experts, unweighted per entry
//   shared expert (Q5_K gate/up, clamped_swiglu, Q5_K down)
//   weighted_slot_sum                            y = sum_i w_i * routed_i + shared
//   hc_expand_mix(y, post, comb, R)              R updated in place: the next residual streams
//
// Wiring across sublayers (hyper_connection.block_streams): a sublayer's `post` and `comb` expand ITS OWN
// output into the streams, but the `pre` that collapses the streams into a sublayer's input is the one the
// PREVIOUS sublayer's mixes produced: attention collapses with the previous block's FFN `pre`, and the FFN
// collapses with the attention's `pre`. So the FFN takes the attention's pre (collapse_pre) and returns its
// own mixes for the next block's attention to collapse with. With collapse_pre null the FFN collapses with
// its own pre (the layer test's wiring, which checks every stage against the oracle).
//
// Any column count T >= 1 (decode: 1). No allocation or host synchronization of its own: scratch comes
// from the caller's arena, so with a graph-capturable ExpertSource (resident) a decode step can be captured.

#include "core/arena.h"
#include "core/tensor.h"
#include "core/weight.h"
#include "models/dsv4/config.h"
#include "models/dsv4/execution/expert_source.h"
#include "ninfer/ops/linear.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::models::dsv4 {

class Model;

struct FfnParams {
    std::int32_t hidden = 0, streams = 0, experts = 0, top_k = 0, intermediate = 0;
    float norm_eps = 0, hc_eps = 0, route_scale = 0, clamp_shared = 0;
    std::int32_t hc_iterations = 0;
    bool norm_topk             = false;
};

// One layer's FFN weights as bound views (device memory; the experts live behind the ExpertSource).
struct FfnWeights {
    Tensor hc_fn;       // FP32 [S*H, 4S + ... = (2 + S) * S]
    Tensor hc_scale;    // FP32 [3]
    Tensor hc_base;     // FP32 [(2 + S) * S]
    Tensor norm;        // BF16 [H]
    Tensor router;      // BF16 [H, E]
    Tensor router_bias; // FP32 [E]
    Weight shared_gate, shared_up, shared_down; // Q5_K [I, H], [I, H], [H, I]
    ops::LinearPolicy shared_policy = ops::LinearPolicy::AllowA8;
};

// The pre, post and comb mixes of one sublayer for T columns: FP32 [S, T], [S, T], [S*S, T].
struct HcMixes {
    Tensor pre, post, comb;
};

struct FfnInput {
    Tensor& residual;                 // BF16 [S*H, T], updated in place into the next streams
    const Tensor* collapse_pre = nullptr; // FP32 [S, T]: the pre that collapses the streams (null: this sublayer's own)
    HcMixes& mixes;                   // written: this sublayer's mixes (the next sublayer's collapse_pre is mixes.pre)
};

[[nodiscard]] FfnParams ffn_params(const TextConfig& config, std::uint32_t layer);
// Binds layer `layer` of a loaded model (device views). The artifact stores the norm in FP32 but rmsnorm
// reads BF16, so the norm is rounded once into `norm_owner` (which must outlive the weights).
[[nodiscard]] FfnWeights bind_ffn_weights(const Model& model, std::uint32_t layer, DeviceBuffer& norm_owner);
// The mixes' tensors for T columns, carved from `arena` (they outlive the FFN call: the caller chains them).
[[nodiscard]] HcMixes alloc_hc_mixes(DeviceArena& arena, const FfnParams& params, std::int32_t columns);

// Arena bytes ffn_forward needs for `columns` columns (bounds every allocation including alignment).
[[nodiscard]] std::size_t ffn_workspace_bytes(const FfnParams& params, const FfnWeights& weights, std::int32_t columns);

// The tensors one FFN call passes between its phases, all carved from the caller's arena: a caller that keeps
// the arena scope open (or carves the same way each call) gets the same addresses, which is what lets it
// capture each phase in its own CUDA graph once and replay it.
struct FfnCall {
    Tensor collapsed, x, logits, ids, weights, gate, up, act, shared, routed, y;
    ops::MoeRouting routing;
    ops::MoeDispatch dispatch;
    DeviceSpan expert_ws;
    std::int32_t columns = 0;
};

// Carves the call's tensors from `arena` (the caller owns the scope); no device work.
[[nodiscard]] FfnCall ffn_carve(const FfnParams& params, std::int32_t columns, DeviceArena& arena);
// Phase 1: mixes, collapse, norm, router, dispatch (no host work, capturable).
void ffn_route(const FfnParams& params, const FfnWeights& weights, FfnCall& call, FfnInput input, cudaStream_t stream);
// Phase 2: the shared expert (capturable). Its Op scratch is carved from `arena` and released inside.
void ffn_shared(const FfnParams& params, const FfnWeights& weights, FfnCall& call, DeviceArena& arena, cudaStream_t stream);
// Phase 3: the routed experts from `source`, the weighted sum and the expansion back into the streams (capturable).
// It is ffn_routed, then ffn_combine; a caller that runs the shared expert beside the routed experts (on another stream,
// ffn_shared's output being needed only by the combine) calls the two itself and joins that stream between them.
void ffn_experts(const FfnParams& params, FfnCall& call, const ops::MoeExpertSource& source, FfnInput input, cudaStream_t stream);
// Phase 3a: the routed experts' outputs (call.routed).
void ffn_routed(const FfnParams& params, FfnCall& call, const ops::MoeExpertSource& source, cudaStream_t stream);
// Phase 3b: the weighted sum with the shared expert's output (call.shared) and the expansion back into the streams.
void ffn_combine(FfnCall& call, FfnInput input, cudaStream_t stream);

// Runs the sublayer on `stream`. Scratch is carved from `workspace` inside a scope and released on return.
void ffn_forward(const FfnParams& params, const FfnWeights& weights, std::uint32_t layer, ExpertSource& experts,
                 FfnInput input, DeviceArena& workspace, cudaStream_t stream);

} // namespace ninfer::models::dsv4
