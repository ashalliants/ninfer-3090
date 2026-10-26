#pragma once

#include "core/weight.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops {

/**
 * Gathers one embedding row per token:
 *
 *   ideal[d,t] = dequantize(table)[ids[t],d].
 *
 * `ids` is contiguous I32 [T], `out` is contiguous BF16 [D,T], and every id is in
 * [0,vocab). `table` has logical shape [vocab,D] and is contiguous BF16, Q4_G64_FP16 or
 * Q6_G64_FP16 RowSplit, Q8_G32_FP16 RowSplit, FP8_E4M3FN_ROW_BF16 RowScale, or GGML_IQ4_XS or
 * GGML_IQ3_S ggml_blocks_v1 (any vocab, D a multiple of 256; each value is ggml's
 * dequantize_row_iq4_xs, dl = d * (ls - 32) then dl * kvalue, or dequantize_row_iq3_s, db = d *
 * (1 + 2 scale) then db * grid byte with the sign bit applied as a negation, in FP32, rounded once
 * to BF16: an exact output). Dense BF16 values
 * are copied bit-exactly. For quantized tables, the oracle independently decodes each code and
 * multiplies it by the exact stored scale in FP64; the BF16 output is promoted and compared
 * directly with that ideal. Final output storage rounding belongs to the quantized embedding
 * criterion, not the oracle. The registered domains are Q4 `[248320,5120]` or `[248320,2048]`,
 * Q6 `[248320,5120]` or `[248320,2048]`, Q8 `[248320,2048]` or `[248320,5120]`, and FP8
 * `[248320,5120]`. Q4/Q6/Q8 scales are FP16; FP8 has one BF16 multiplier per
 * row and requires 4-byte-aligned output storage. `out` must not overlap `ids` or any table plane.
 * There is no workspace or persistent state side effect.
 */
void embedding(const Tensor& ids, const Weight& table, Tensor& out, cudaStream_t stream);

/**
 * Op: embedding gather from a table held in two planes
 *
 * Math / indexing:
 *   ideal[d,t] = dequantize(row(ids[t]))[d], where row(i) is row i of `hot` when i < hot_rows and
 *   row i of `cold` otherwise.
 *
 * Logical shapes:
 *   `ids` contiguous I32 [T], every id in [0,vocab); `out` contiguous BF16 [D,T]; `cold` the whole
 *   table [vocab,D]; `hot` [hot_rows,D], 1 <= hot_rows <= vocab. Both planes hold the same rows
 *   [0,hot_rows): the Op reads those rows from `hot` and never reads them from `cold`.
 *
 * Supported domain:
 *   GGML_IQ4_XS ggml_blocks_v1 planes (D a multiple of 256), each plane in any memory the device can
 *   read (device memory, or mapped page-locked host memory through its device pointer).
 *
 * Numeric:
 *   Each value is decoded as `embedding` decodes a GGML_IQ4_XS table (an exact output): the result is
 *   bitwise `embedding(ids, table, out)` for a table whose rows are the planes' rows as above.
 *
 * Effects:
 *   Writes `out`. `out` must not overlap `ids` or either plane.
 *
 * Workspace: none.
 */
void embedding_split(const Tensor& ids, const Weight& hot, const Weight& cold, std::int32_t hot_rows, Tensor& out,
                     cudaStream_t stream);

} // namespace ninfer::ops
