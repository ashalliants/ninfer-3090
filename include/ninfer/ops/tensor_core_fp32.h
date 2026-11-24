#pragma once

#include "core/arena.h"
#include "core/tensor.h"
#include "ninfer/ops/moe_situ.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <span>

namespace ninfer::ops {

/**
 * Op: projection_tc_fp32 (not bit-exact: projection_fp32 on the FP16 tensor cores)
 *
 * Math / indexing:
 *   out[n, t] = (sum_k W16[n, k] X16[k, t]) 2^-e_n 2^-e_t for the rows n of `weights` concatenated in order, with
 *   W16 = FP16(w 2^e_n) per weight row (its absmax in [2^14, 2^15)) and X16 = FP16(x 2^e_t) per column of the
 *   weight's input; the sum is cuBLAS's FP32-accumulating tensor-op GEMM. A BF16 value converts exactly unless its
 *   scaled magnitude is below FP16's normal range (2^-14; bf16_fp16_inexact counts those), so against
 *   projection_fp32 the difference is the summation order and those values.
 *
 * Logical shapes:
 *   inputs: one BF16 [K, T] for every weight, or one per weight; weight i BF16 [K, N_i]; out FP32 [sum N_i, T]
 *   contiguous. K a multiple of 8.
 *
 * Workspace: projection_tc_workspace_bytes(K, max N_i, T) from `workspace`, released on return.
 * Execution: stream-ordered; not graph capturable on the first call of a device (cuBLAS's handle is made then).
 */
[[nodiscard]] std::size_t projection_tc_workspace_bytes(std::int32_t k, std::int32_t max_rows, std::int32_t cols);
void projection_tc_fp32(std::span<const Tensor* const> inputs, std::span<const Tensor* const> weights, Tensor& out,
                        DeviceArena& workspace, cudaStream_t stream);

/**
 * Op: moe_situ_experts_tc (not bit-exact: moe_situ_experts on the FP16 tensor cores)
 *
 * moe_situ_experts' math with the entries grouped by expert (the ids are read on the host: the call syncs) and the
 * expert's inputs in FP16 under a power of two per entry (absmax in [2^14, 2^15)).
 *   NVFP4 banks: every decoded weight value is exact in FP16 unscaled (at most 6 significant bits, magnitudes in
 *   [2^-10, 2688]). One kernel per matrix runs every expert: a CTA takes 128 rows of one expert by up to 64 of its
 *   entries, decodes the blocks into shared memory and multiplies on the tensor cores (WMMA, FP32 accumulate). The
 *   gate/up kernel applies situ to its rows' pairs and stores the activation in FP16 under one power of two for the
 *   whole call (situ is at most gate_cap up_cap in magnitude); the down kernel scatters each entry's column.
 *   BF16 banks: each expert's rows in FP16 under a power of two per row, cuBLAS FP32-accumulating GEMMs, the
 *   activation under a power of two per entry.
 *   BF16 banks with blas_min_entries > 0 and row_exponents set: the NVFP4 banks' kernels and hot-expert split, the
 *   kernels decoding each row (BF16 or MXFP8) into FP16 under its exponent, the same FP16 values the cuBLAS route
 *   multiplies, and the activation under the kernels' one power of two; the hot experts take the cuBLAS route above.
 *   NVFP4 banks with blas_min_entries > 0: an expert with at least that many entries in the call is decoded whole
 *   into FP16 (exact, as above) and multiplied by cuBLAS (FP32 accumulate) in batches of up to
 *   kExpertsBlasBatch entries, its inputs and activation scaled as the kernels scale them; the other experts take
 *   the kernels. Only the order of the FP32 sums differs between the two, so which route an expert takes is a
 *   question of speed (cuBLAS wins once an expert has enough entries to amortise its decode).
 * The combine over the k slots is moe_situ_experts' (in order, one FMA each from 0). gate_cap and up_cap positive.
 * Workspace: moe_situ_experts_tc_workspace_bytes(bank, k * T) from `workspace`, released on return. The FP16 inputs
 *   take one row per column (NVFP4) or per entry of the expert with the most (BF16 banks, whose GEMM results take as
 *   many); given `columns` T (a column's k ids distinct, so no expert has more than T entries) and `combined` false
 *   (weights null: out holds the entries) the bound is tighter. A call with blas_min_entries > 0 needs the query's
 *   with the same value (an NVFP4 expert's decoded rows and one batch's operands, about 46 MiB at H 2304, I 1152).
 * Execution: stream-ordered; syncs the stream; not graph capturable.
 */
inline constexpr int kExpertsBlasBatch = 1024;

/// The BF16 banks' row exponents (SituExpertBank::row_exponents): one per row, per expert its 2I gate/up rows then its
/// H down rows, E (2I + H) in all; each is the e that moe_situ_experts_tc's BF16 route scales the row by (absmax 2^e in
/// [2^14, 2^15), 0 for an all-zero row). Reads the bank in either form (the same values, so the same exponents);
/// asynchronous on `stream`.
[[nodiscard]] std::int64_t situ_bank_row_exponent_count(const SituExpertBank& bank);
void situ_bank_row_exponents(const SituExpertBank& bank, std::int16_t* out, cudaStream_t stream);
[[nodiscard]] std::size_t moe_situ_experts_tc_workspace_bytes(const SituExpertBank& bank, std::int64_t entries, std::int64_t columns = 0,
                                                              bool combined = true, int blas_min_entries = 0);
void moe_situ_experts_tc(const SituExpertBank& bank, const Tensor& z, const Tensor& ids, const Tensor* weights, float gate_cap,
                         float up_cap, DeviceArena& workspace, Tensor& out, cudaStream_t stream, int blas_min_entries = 0);

/// The values of NVFP4 rows (records of an expert bank: every gate, up and down row of every expert) that the row
/// scaling does not carry exactly into FP16 (a diagnostic; syncs).
[[nodiscard]] std::uint64_t nvfp4_bank_fp16_inexact(const SituExpertBank& bank, cudaStream_t stream);

/// The values of BF16 rows [K, N] that the row scaling above does not carry exactly into FP16 (a diagnostic; syncs).
[[nodiscard]] std::uint64_t bf16_fp16_inexact(const Tensor& w, cudaStream_t stream);

} // namespace ninfer::ops
