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

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

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
