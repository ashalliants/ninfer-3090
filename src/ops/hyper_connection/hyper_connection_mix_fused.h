#pragma once

// Adapted from Infernix a3edb450 src/ops/hyper_connection/hyper_connection_mix_fused.h
// (Apache-2.0).
// Modified for NInfer-3090: BF16 contiguous weights (Infernix's Q8 form is dropped) and, with
// --hc-int8, int8 copies (HcInt8Weights); the norm weight is the stored FP32 multiplier. Private to
// the Op.

#include "core/tensor.h"
#include "core/weight.h"
#include "ninfer/ops/hyper_connection.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::detail {

// Columns the fused route serves in one call.
inline constexpr std::int32_t kHcMixFusedMaxColumns = 16;

// Whether the fused route serves this problem: both weights BF16 contiguous, the registered
// geometry (4 streams of 2560, rank a multiple of 8 up to 512) and 1..16 columns.
[[nodiscard]] bool hc_mix_fused_supported(const Weight& down, const Weight& up,
                                          std::int32_t hidden, std::int32_t streams,
                                          std::int32_t rank, std::int32_t columns) noexcept;

// FP32 split-K partials [S][down rows][T] and per-stream inverse RMS [S][T].
[[nodiscard]] std::size_t hc_mix_fused_workspace_bytes(std::int32_t down_rows,
                                                       std::int32_t streams,
                                                       std::int32_t columns) noexcept;

// K1a then K1b. `partials` holds hc_mix_fused_workspace_bytes; inject (FP32 [S, T]) may be null.
void hc_mix_fused(const Tensor& residual, const Tensor& norm_weight, const Weight& down,
                  const Weight& up, std::int32_t streams, std::int32_t rank, float eps, Tensor& x,
                  Tensor* inject, void* partials, cudaStream_t stream);

// The int8 form (--hc-int8; hyper_connection_mix_int8.cu): the same geometry with int8 copies
// (rank a multiple of 32 up to 512) and 1..16 columns.
[[nodiscard]] bool hc_mix_int8_fused_supported(const HcInt8Weights& down, const HcInt8Weights& up,
                                               std::int32_t hidden, std::int32_t streams,
                                               std::int32_t rank, std::int32_t columns) noexcept;

// Where the per-stream inverse RMS [S][T] starts in `partials`: after the FP32 split-K partials
// [S][down rows][T], 256-byte aligned.
[[nodiscard]] inline std::size_t hc_mix_fused_inv_offset(std::int32_t down_rows,
                                                         std::int32_t streams,
                                                         std::int32_t columns) noexcept {
    return (static_cast<std::size_t>(streams) * down_rows * columns * sizeof(float) + 255) / 256 *
           256;
}

// K1a over the int8 down copy (hyper_connection_mix_fused.cu): the same kernel template as the
// BF16 route with each 8-weight chunk decoded to q * s, so the partials and inverse RMS are those
// of the BF16 route over the represented weights. `partials` as for hc_mix_fused.
void hc_mix_down_partials_int8(const Tensor& residual, const Tensor& norm_weight,
                               const HcInt8Weights& down, std::int32_t streams, float eps,
                               void* partials, cudaStream_t stream);

// K1a then the int8 K1b (FP16 hi/lo mma.sync over the int8 up copy, then the collapse).
void hc_mix_int8_fused(const Tensor& residual, const Tensor& norm_weight, const HcInt8Weights& down,
                       const HcInt8Weights& up, std::int32_t streams, std::int32_t rank, float eps,
                       Tensor& x, Tensor* inject, void* partials, cudaStream_t stream);

} // namespace ninfer::ops::detail
