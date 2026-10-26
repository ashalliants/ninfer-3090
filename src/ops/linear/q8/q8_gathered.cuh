#pragma once

// Row gathering for the exact stage of two_stage_head (--lm-head-two-stage): the dense head's own
// sliced-K and MMA kernels, run over a device-built list of row ids instead of every row. A row's
// result depends only on its weights and the activation, never on which rows share its tile, so
// each gathered row gets the dense head's bits. CTAs past the list's device count retire at once.

#include <cuda_bf16.h>

#include <cstdint>

namespace ninfer::ops::detail {

// `ids` null maps list position i to row i (every row, for the whole-head fallback). Positions
// past `*count` inside the last CTA repeat the last listed row and their stores are dropped.
template <int RowsPerCta>
struct Q8GatheredRows {
    static constexpr int kOutputRowsPerCta = RowsPerCta;
    const std::int32_t* ids;
    const std::int32_t* count;

    __device__ __forceinline__ bool skip_cta(int row0) const { return row0 >= *count; }

    __device__ __forceinline__ int weight_row(int row0, int local_row) const {
        int at        = row0 + local_row;
        const int end = *count;
        at            = at < end ? at : end - 1;
        return ids != nullptr ? ids[at] : at;
    }

    __device__ __forceinline__ int weight_row(int row0, int local_row, int) const {
        return weight_row(row0, local_row);
    }
};

// Stores list position `row` at its head row of the BF16 [rows, T] logits, rounded as
// LinearBf16Output rounds.
struct Q8ScatteredBf16Output {
    __nv_bfloat16* data;
    std::int32_t rows;
    const std::int32_t* ids;
    const std::int32_t* count;

    __device__ __forceinline__ void store(int row, int token, float value) const {
        if (row >= *count) { return; }
        const int head_row = ids != nullptr ? ids[row] : row;
        data[static_cast<std::int64_t>(token) * rows + head_row] = __float2bfloat16_rn(value);
    }
};

} // namespace ninfer::ops::detail
