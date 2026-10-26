#pragma once

// Adapted from Infernix a3edb450 src/ops/offloaded_sparse_moe/cuda/wide_expert.h (Apache-2.0).
// Modified for NInfer-3090: GGML expert records on sm_86. The work planning (64-column tiles of
// every wide job, per pass) is Infernix's; the GEMMs are new: exact int8 mma.sync on decoded GGML
// codes with the canonical A8 activations, summed in the narrow route's order, so a wide expert's
// bits equal the narrow route's and the CPU engine's (Infernix's W4A4 wide route only met an FP64
// bound; its W4A16 route is exact, and this one follows that contract). cp.async replaces TMA and
// the mbarrier ring.
//
// The wide route of offloaded_sparse_moe: every routed expert with more than kMaxColumns columns in
// a call (prefill chunks) is computed by an integer tensor-core grouped GEMM instead of the dp4a
// narrow route. The route of an expert is a function of its column count alone, never of placement,
// and both routes return the same bits, so the choice is invisible in the outputs.
//
// Exactness. canonical_ggml.h fixes the arithmetic of one row value: per 32-value sub-block s the
// exact integer P_s, the lane partial l = s % 32 accumulating fma(c_s * d_s, float(P_s), .) in
// ascending s, then the butterfly over the 32 partials. The butterfly is a perfect binary tree whose
// leaves in depth-first order are the partials l = bitreverse5(n), n = 0..31, so the GEMM visits K in
// that order (sub-blocks l, l + 32, l + 64 of each partial in turn) and folds each finished leaf into
// a five-entry stack of subtree sums, one per tree level: the same additions on the same operands.
// P_s comes from the MMA's exact int32 (one m16n8k32 when both halves of the sub-block share their
// multiplier, else one m16n8k16 per half), read as a binary32 through the accumulator's bias
// 0x4B400000, which needs |sum| < 2^22: a k32 sum is at most 32 * 127^2 = 516,128 (|w| <= 127, the
// IQ4 codebooks; |q| <= 127). float(P_s) = m * A (one multiplier) or m1 * B + m0 * A is exact below
// 2^24: |P_s| <= 32 * 32 * 127^2 = 16,516,096 for IQ4_XS (|m| <= 32, signed), less for every other
// format (canonical_ggml.h step 2), so m * A, m0 * A and the fma are exact.

#include "ninfer/ops/offloaded_sparse_moe.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::offloaded_moe::wide {

inline constexpr int kTileColumns = 64; // columns (routed entries of one expert) per work tile
// A call of at least this many columns sends every job to the wide route (the narrow kernels do not
// run): nearly every expert then has more than kMaxColumns columns, and the bits are the same.
inline constexpr int kAllWideColumns = 512;
inline bool all_wide(std::int32_t columns) { return columns >= kAllWideColumns; }

// The wide route's part of moe_experts' workspace, after the narrow route's.
struct Workspace {
    std::int8_t* xq     = nullptr; // [columns][32 leaves][128 bytes]: the A8 cast of x, leaf-major
    std::int32_t* skip  = nullptr; // [max_jobs]: the narrow kernels skip a job when nonzero
    std::int32_t* tiles = nullptr; // [max_tiles][2]: (job, first column) of each wide tile
    std::int32_t* pass_tiles = nullptr; // [passes + 1]: first tile of each pass
};

[[nodiscard]] std::int32_t max_tiles(std::int32_t max_jobs, std::int32_t entries);
[[nodiscard]] std::size_t workspace_bytes(std::int32_t max_jobs, std::int32_t entries, std::int32_t columns);
[[nodiscard]] Workspace carve(void* base, std::int32_t max_jobs, std::int32_t entries, std::int32_t columns);

// The record geometry the wide kernels read (device-copyable).
struct Geometry {
    std::uint32_t up_offset;
    std::uint32_t down_offset;
    std::uint32_t gate_up_row_bytes;
    std::uint32_t down_row_bytes;
};

// Before the passes: the A8 cast of every column of x, the call's wide tiles in passes of pass_jobs
// jobs, and the skip flags of the narrow kernels (a CPU-served job or a wide job).
void prepare(const std::uint16_t* x, std::int32_t columns, const MoeDispatch& dispatch, std::int32_t max_jobs,
             std::int32_t pass_jobs, const std::int32_t* cpu_flags, const Workspace& workspace, cudaStream_t stream);

// The wide jobs of pass `pass` (jobs [pass * pass_jobs, ...)): gate/up with SwiGLU and the A8 cast of
// h into the entries' h rows (codes and scales in place of the narrow route's FP32 h), then down into
// `outputs`. job_records: the records the stage kernel resolved for the pass, or null (each record
// read where it lives). The formats are the record's parts (ggml_record_parts).
void run_pass(QType gate_up_format, QType down_format, const MoeDispatch& dispatch, const MoeExpertSource& source,
              const Geometry& g, std::int32_t top_k, std::int32_t max_jobs, std::int32_t pass_jobs, std::int32_t pass,
              const std::uint8_t* const* job_records, const Workspace& workspace, float* h_rows,
              std::uint16_t* outputs, cudaStream_t stream);

} // namespace ninfer::ops::offloaded_moe::wide
