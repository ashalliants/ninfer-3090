#pragma once

// Adapted from Infernix a3edb450 include/infernix/ops/hyper_connection.h (Apache-2.0).
// Modified for NInfer-3090: the norm weight is the stored FP32 multiplier instead of a BF16
// unit-offset weight, Infernix's Q8 weight form is dropped (an int8 form with Basalt's group rule,
// made at load from the BF16 weights, takes its place: HcInt8Weights), and the contract follows
// op-development §3.

#include "core/arena.h"
#include "core/tensor.h"
#include "core/weight.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/offloaded_sparse_moe.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <span>

namespace ninfer::ops {

/// Values per binary16 scale of an int8 hyper-connection matrix.
inline constexpr std::int32_t kHcInt8Group = 32;
/// The most columns hyper_connection_mix's int8 form serves (wider calls use the BF16 weights).
inline constexpr std::int32_t kHcInt8MaxColumns = 16;

/// The int8 copy of a hyper-connection matrix [N, K] (hc_int8_quantize): codes int8 [N, K], row r
/// at codes + r * K; scales binary16 bits [N, K / kHcInt8Group], row r at scales + r * K / 32.
/// Element (r, k) represents codes[r, k] * scales[r, k / 32]. Non-owning; device memory.
struct HcInt8Weights {
    const std::int8_t* codes    = nullptr;
    const std::uint16_t* scales = nullptr;
    std::int32_t n = 0, k = 0;
};

/// The bytes of an [N, K] copy's codes and of its scales.
[[nodiscard]] constexpr std::size_t hc_int8_code_bytes(std::int32_t n, std::int32_t k) noexcept {
    return static_cast<std::size_t>(n) * static_cast<std::size_t>(k);
}

[[nodiscard]] constexpr std::size_t hc_int8_scale_bytes(std::int32_t n, std::int32_t k) noexcept {
    return static_cast<std::size_t>(n) * static_cast<std::size_t>(k / kHcInt8Group) *
           sizeof(std::uint16_t);
}

/**
 * Op: hc_int8_quantize (the int8 copy of a BF16 hyper-connection matrix)
 *
 * Math / indexing:
 *   For row r and group g (values k = 32g .. 32g + 31 of the row) of the BF16 values w:
 *     a       = max_k |w[r, k]|
 *     s[r, g] = f16(f32(a / 127)), then the next binary16 up when 127.5 * s < a
 *     q[r, k] = clamp(rint(f32(w[r, k] / s[r, g])), -127, 127), or 0 when s[r, g] = 0
 *   f32 and f16 round to nearest even and rint rounds half to even; s = 0 exactly when the group
 *   is all zeros, and every |q * s - w| <= s / 2. The rule is Basalt's (8d73a39 hc8_quant_kernel)
 *   except where its scale is a low binary16 subnormal: there Basalt's s can leave the largest
 *   value clamped by more than half a step (or, rounded to zero, unencoded), and s is raised.
 *
 * Logical shapes:
 *   w BF16 [N, K] (row r at stride max(padded_shape[1], K) elements); codes int8 [N, K]; scales
 *   binary16 [N, K / 32]; status one 32-bit word. N >= 1, K a positive multiple of 32.
 *
 * Supported domain:
 *   w is BF16, Contiguous, 16-byte aligned with a row stride that is a multiple of 8. codes and
 *   scales are 16-byte aligned and hold at least hc_int8_code_bytes and hc_int8_scale_bytes.
 *
 * Numeric:
 *   Exact: the codes and scales are the bits of the formula (ops/hyper_connection/hc_int8_codec.h
 *   is its host form). A group with an infinite or NaN value, or with f32(a / 127) >= 65520 (its
 *   binary16 rounding would be infinite), is refused: its codes and scale are written as zero.
 *
 * Effects:
 *   Writes codes and scales; writes a nonzero value to *status when a group is refused (it never
 *   clears it). w is read only; no output may overlap an input or another output.
 *
 * Workspace: none. Execution: stream-ordered; graph capturable.
 */
void hc_int8_quantize(const Weight& w, DeviceSpan codes, DeviceSpan scales, DeviceSpan status,
                      cudaStream_t stream);

/**
 * Op: hyper_connection_mix (the mixer that reads a block's input from hyper-connection residual
 * streams)
 *
 * Math / indexing:
 *   The residual R holds S streams of H features per column, stream-major: R[s*H + d, t] is
 *   feature d of stream s. For every column t independently:
 *     Rn[s*H + d] = R[s*H + d] * (mean_d' R[s*H + d']^2 + eps)^(-1/2) * w[s*H + d]
 *     z           = W_down Rn                       (rank rows, then S injection rows if present)
 *     m[k]        = SiLU(z[k] / S)                  for k < rank
 *     inject[s]   = 2 * sigmoid(z[rank + s] / S)    for s < S, only when inject is requested
 *     u           = W_up m
 *     x[d]        = (1/S) * sum_s sigmoid(u[s*H + d]) * Rn[s*H + d]
 *   w is the norm's full multiplier (a checkpoint's unit-offset 1 + gamma is stored as such).
 *
 * Logical shapes:
 *   residual BF16 [S*H, T]; norm_weight FP32 [S*H]; down [rank (+ S), S*H]; up [S*H, rank];
 *   x BF16 [H, T]; inject FP32 [S, T]. down has rank + S rows exactly when inject is requested.
 *   T >= 1.
 *
 * Supported domain:
 *   BF16 form: down and up are contiguous BF16 weights registered with Linear for their [N, K];
 *   every T. The geometry S = 4, H = 2560, rank a multiple of 8 up to 512 is the one served
 *   without Linear at T <= 16. Int8 form: down and up are HcInt8Weights (hc_int8_quantize) with
 *   16-byte aligned codes and scales, at S = 4, H = 2560, rank a multiple of 32 up to 512 and
 *   1 <= T <= kHcInt8MaxColumns only (a caller runs wider calls on the BF16 weights); anything
 *   else is refused. residual is 16-byte aligned; eps > 0.
 *
 * Numeric:
 *   Arithmetic is FP32 from the represented inputs; the int8 form's weights are the codes' values
 *   q * s (exact in FP32). Two profiles: at T <= 16 with the S = 4, H = 2560 geometry every
 *   intermediate (Rn, z, m, u) stays FP32 and only x is rounded to BF16, so a column's x and inject
 *   do not depend on T within 1..16; otherwise Rn, z, m and u are rounded to BF16 (the composed
 *   profile) and Linear's BF16 criterion applies to z and u. The int8 form (the first profile)
 *   takes W_up m from an FP16 high and low split of m with products and sums in FP32: each m[k]
 *   enters as hi + lo within max(2^-22 |m[k]|, 2^-25) of m[k], for |m[k]| < 65520 (beyond, hi
 *   overflows to infinity). The oracle is the formula in FP64 on the represented weights.
 *
 * Effects:
 *   Writes x and, when requested, inject. residual and the weights are read only. No output may
 *   overlap an input.
 *
 * Workspace:
 *   hyper_connection_mix_workspace_capacity_bytes() for every T in [1, max_tokens]. It must not
 *   overlap any operand.
 *
 * Execution:
 *   Stream-ordered; graph capturable.
 */
void hyper_connection_mix(const Tensor& residual, const Tensor& norm_weight, const Weight& down,
                          const Weight& up, LinearPolicy policy, std::int32_t streams,
                          std::int32_t rank, float eps, Tensor& x, Tensor* inject,
                          WorkspaceArena& workspace, cudaStream_t stream);
void hyper_connection_mix(const Tensor& residual, const Tensor& norm_weight,
                          const HcInt8Weights& down, const HcInt8Weights& up, std::int32_t streams,
                          std::int32_t rank, float eps, Tensor& x, Tensor* inject,
                          WorkspaceArena& workspace, cudaStream_t stream);

/// The workspace hyper_connection_mix needs for every T in [1, max_tokens] (the int8 form:
/// max_tokens <= kHcInt8MaxColumns).
[[nodiscard]] std::size_t hyper_connection_mix_workspace_capacity_bytes(
    const Weight& down, const Weight& up, LinearPolicy policy, std::int32_t streams,
    std::int32_t rank, std::int32_t max_tokens);
[[nodiscard]] std::size_t hyper_connection_mix_workspace_capacity_bytes(const HcInt8Weights& down,
                                                                        const HcInt8Weights& up,
                                                                        std::int32_t streams,
                                                                        std::int32_t rank,
                                                                        std::int32_t max_tokens);

/// hc_fused_read's outputs (device memory). x BF16 [H, T]; inject FP32 [S, T]; logits FP32
/// [E + 1, T]; routing as moe_route writes it (ids I32 [k, T], weights FP32 [k, T], shared_gate
/// FP32 [T]); dispatch carved for E experts and k * T entries (carve_moe_dispatch); route_log
/// optional I32 [k * T]; a8_codes int8 [T][H] (hc_fused_read_a8_code_bytes) and a8_scales FP32
/// [T][H / 32] (hc_fused_read_a8_scale_bytes), the planes MoeA8Input passes to moe_experts.
struct HcFusedReadOutputs {
    Tensor x;
    Tensor inject;
    Tensor logits;
    MoeRouting routing;
    MoeDispatch dispatch;
    std::int32_t* route_log = nullptr;
    DeviceSpan a8_codes;
    DeviceSpan a8_scales;
};

[[nodiscard]] constexpr std::size_t hc_fused_read_a8_code_bytes(std::int32_t hidden,
                                                                std::int32_t columns) noexcept {
    return static_cast<std::size_t>(hidden) * static_cast<std::size_t>(columns);
}

[[nodiscard]] constexpr std::size_t hc_fused_read_a8_scale_bytes(std::int32_t hidden,
                                                                 std::int32_t columns) noexcept {
    return static_cast<std::size_t>(hidden / 32) * static_cast<std::size_t>(columns) *
           sizeof(float);
}

/**
 * Op: hc_fused_read (a routed-expert block's input read from the hyper-connection residual: the
 *     previous block's inject, the int8 mixer, the router, the routing and dispatch, and the A8
 *     cast of the mixer's output)
 *
 * Math / indexing:
 *   The composition, for every column t of T:
 *     1. residual = hyper_connection_inject(y, prev_inject, residual)  (in place)
 *     2. x, inject = hyper_connection_mix(residual, norm_weight, down, up), the int8 form
 *     3. logits = projection_fp32(x, router)       (the E router rows, then the shared-gate row)
 *     4. routing = moe_route(logits, top_k)
 *     5. dispatch, route_log = moe_dispatch(routing, E), the entries within each expert in
 *        ascending entry index t * k + slot (offloaded_moe::plan_dispatch, as moe_route_dispatch
 *        writes them)
 *     6. a8 = the canonical A8 cast of x (offloaded_sparse_moe.h, A8): per 32 values of a column,
 *        codes and the FP32 scale d, as moe_experts' precast input holds them.
 *
 * Logical shapes:
 *   residual BF16 [S*H, T]; y BF16 [H, T]; prev_inject FP32 [S, T]; norm_weight FP32 [S*H]; down
 *   [rank + S, S*H] and up [S*H, rank] (HcInt8Weights); router BF16 [H, N_i] per segment,
 *   sum N_i = E + 1; outputs as HcFusedReadOutputs. 1 <= T <= kHcInt8MaxColumns.
 *
 * Supported domain:
 *   The int8 mixer's domain (hyper_connection_mix, int8 form: S = 4, H = 2560, rank a multiple of
 *   32 up to 512) with the injection rows; one to four router segments, each contiguous and
 *   16-byte aligned; 1 <= E <= 512 and 1 <= top_k <= min(16, E). residual, y, x and the A8
 *   planes are 16-byte aligned; every tensor is contiguous. eps > 0.
 *
 * Numeric:
 *   Each step's own contract: steps 1, 3, 4, 5 and 6 are exact given their inputs (step 1 as one
 *   fused multiply-add rounded once, which hyper_connection_inject's kernel is by contraction),
 *   and step 2 is the int8 mixer's arithmetic, so every output is bitwise that of the composition of those Ops
 *   (the fixed dispatch order included). The oracle is that composition; x also meets the int8
 *   mixer's FP64 criterion.
 *
 * Effects:
 *   Updates residual in place (step 1); writes every output. y, prev_inject, norm_weight, the
 *   weights and the router are read only. No output may overlap an input or another output (in
 *   particular inject must not overlap prev_inject).
 *
 * Workspace:
 *   hc_fused_read_workspace_capacity_bytes() for every T in [1, max_tokens]. It must not overlap
 *   any operand. The workspace stays in device use until the call's last kernel (the routing)
 *   finishes on `stream`, after the call returns and after `mixed` is recorded: work ordered only
 *   after `mixed` (a side stream forked there) must not be handed that memory, so a caller keeps it
 *   allocated until such work has joined `stream` again.
 *
 * Execution:
 *   Stream-ordered; graph capturable; no host synchronization. When `mixed` is set it is recorded
 *   on `stream` once residual, x, inject and the A8 planes are written and before the routing
 *   (steps 3 to 5) is enqueued, so a side stream can wait for x alone (see Workspace).
 */
void hc_fused_read(Tensor& residual, const Tensor& y, const Tensor& prev_inject,
                   const Tensor& norm_weight, const HcInt8Weights& down, const HcInt8Weights& up,
                   std::int32_t streams, std::int32_t rank, float eps,
                   std::span<const Tensor* const> router, std::int32_t top_k,
                   HcFusedReadOutputs& out, WorkspaceArena& workspace, cudaStream_t stream,
                   cudaEvent_t mixed = nullptr);

/// The workspace hc_fused_read needs beyond the int8 mixer's (its routing kernel's counters).
inline constexpr std::size_t kHcFusedReadExtraWorkspaceBytes = 256;

/// The workspace hc_fused_read needs for every T in [1, max_tokens] (max_tokens <=
/// kHcInt8MaxColumns): hyper_connection_mix_workspace_capacity_bytes of the int8 form plus
/// kHcFusedReadExtraWorkspaceBytes. The second form takes the geometry alone (down has rank + S
/// rows), for a caller that plans before the copies exist; both give the same bytes.
[[nodiscard]] std::size_t hc_fused_read_workspace_capacity_bytes(const HcInt8Weights& down,
                                                                 const HcInt8Weights& up,
                                                                 std::int32_t streams,
                                                                 std::int32_t rank,
                                                                 std::int32_t max_tokens);
[[nodiscard]] std::size_t hc_fused_read_workspace_capacity_bytes(std::int32_t streams,
                                                                 std::int32_t rank,
                                                                 std::int32_t max_tokens);

/**
 * Op: hyper_connection_inject
 *
 * Math / indexing:
 *   residual[s*H + d, t] = bf16(residual[s*H + d, t] + inject[s, t] * y[d, t])
 *
 * Logical shapes:
 *   y BF16 [H, T]; inject FP32 [S, T]; residual BF16 [S*H, T]; T >= 1.
 *
 * Numeric:
 *   One FP32 multiply-add per element, rounded once to BF16.
 *
 * Effects:
 *   Updates residual in place; y and inject are read only and must not overlap it.
 *
 * Workspace: none. Execution: stream-ordered; graph capturable.
 */
void hyper_connection_inject(const Tensor& y, const Tensor& inject, Tensor& residual,
                             cudaStream_t stream);

/**
 * Op: hyper_connection_expand
 *
 * Math / indexing:
 *   residual[s*H + d, t] = x[d, t] for every stream s (the first block's residual from the
 *   embedding). Exact copy.
 *
 * Logical shapes:
 *   x BF16 [H, T]; residual BF16 [S*H, T]; T >= 1.
 *
 * Effects:
 *   Writes residual; x must not overlap it.
 *
 * Workspace: none. Execution: stream-ordered; graph capturable.
 */
void hyper_connection_expand(const Tensor& x, std::int32_t streams, Tensor& residual,
                             cudaStream_t stream);

} // namespace ninfer::ops
