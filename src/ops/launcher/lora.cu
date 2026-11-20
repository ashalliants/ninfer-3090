// Implements: include/ninfer/ops/lora.h
// Single variant: a split-K shrink into FP32 partials, then a per-output expand (kernel/lora.cuh).
#include "ops/launcher/lora.h"

#include "core/device.h" // CUDA_CHECK
#include "ops/kernel/lora.cuh"

#include <algorithm>
#include <cuda_bf16.h>

namespace ninfer::ops::detail {

static_assert(kLoraKernelAdapters == kLoraMaxAdapters && kLoraKernelRows == kLoraMaxRows);
static_assert(kLoraKernelOutputs == kLoraMaxOutputs);

std::int32_t lora_splits(std::int32_t k) noexcept { return (k + kLoraSplitK - 1) / kLoraSplitK; }

void lora_apply_launch(const Tensor& x, const LoraDown& down, std::span<const LoraUp> up,
                       const LoraColumns& columns, float* partial, cudaStream_t stream) {
    const std::int32_t k      = x.ne[0];
    const std::int32_t total  = columns.rows * columns.width;
    const std::int32_t splits = lora_splits(k);

    LoraShrinkParams shrink{};
    shrink.x    = static_cast<const __nv_bfloat16*>(x.data);
    shrink.x_ld = x.nb[1] / static_cast<std::int64_t>(sizeof(__nv_bfloat16));
    for (int s = 0; s < kLoraKernelAdapters; ++s) {
        shrink.a[s] = static_cast<const __nv_bfloat16*>(down.a[static_cast<std::size_t>(s)]);
    }
    for (int row = 0; row < kLoraKernelRows; ++row) {
        shrink.slot[row] = static_cast<std::int8_t>(
            row < columns.rows ? columns.slot[static_cast<std::size_t>(row)] : -1);
    }
    shrink.rank    = down.rank;
    shrink.k       = k;
    shrink.columns = total;
    shrink.width   = columns.width;
    shrink.partial = partial;
    // The narrowest tile that holds the call (decode and verify widths), else the widest, with two
    // rank rows per warp past eight columns where the rank allows (kernel/lora.cuh).
    const auto launch_shrink = [&]<int Cols, int RowsPerWarp>() {
        const dim3 grid(static_cast<unsigned>((total + Cols - 1) / Cols),
                        static_cast<unsigned>(down.rank / (kLoraShrinkRows * RowsPerWarp)),
                        static_cast<unsigned>(splits));
        lora_shrink_kernel<Cols, RowsPerWarp><<<grid, kLoraThreads, 0, stream>>>(shrink);
    };
    if (total <= 1) {
        launch_shrink.template operator()<1, 1>();
    } else if (total <= 2) {
        launch_shrink.template operator()<2, 1>();
    } else if (total <= 4) {
        launch_shrink.template operator()<4, 1>();
    } else if (total <= kLoraShrinkColumns || down.rank % (2 * kLoraShrinkRows) != 0) {
        launch_shrink.template operator()<kLoraShrinkColumns, 1>();
    } else {
        launch_shrink.template operator()<kLoraShrinkColumns, 2>();
    }
    CUDA_CHECK(cudaGetLastError());

    LoraExpandParams expand{};
    std::int32_t widest = 0, deepest = 0;
    for (std::size_t o = 0; o < up.size(); ++o) {
        LoraExpandOutput& out = expand.outputs[o];
        out.out               = static_cast<__nv_bfloat16*>(up[o].output.data);
        out.ld                = up[o].output.nb[1] / static_cast<std::int64_t>(sizeof(__nv_bfloat16));
        for (int s = 0; s < kLoraKernelAdapters; ++s) {
            out.b[s] = static_cast<const __nv_bfloat16*>(up[o].b[static_cast<std::size_t>(s)]);
        }
        out.n           = up[o].output.ne[0];
        out.rank_offset = up[o].rank_offset;
        out.rank_count  = up[o].rank_count;
        widest          = std::max(widest, out.n);
        deepest         = std::max(deepest, out.rank_count);
    }
    for (int row = 0; row < kLoraKernelRows; ++row) { expand.slot[row] = shrink.slot[row]; }
    expand.rank    = down.rank;
    expand.columns = total;
    expand.width   = columns.width;
    expand.splits  = splits;
    expand.partial = partial;
    const auto launch_expand = [&]<int Cols>() {
        const dim3 grid(static_cast<unsigned>((widest + kLoraThreads - 1) / kLoraThreads),
                        static_cast<unsigned>((total + Cols - 1) / Cols),
                        static_cast<unsigned>(up.size()));
        const std::size_t shared = static_cast<std::size_t>(deepest) * Cols * sizeof(float);
        lora_expand_kernel<Cols><<<grid, kLoraThreads, shared, stream>>>(expand);
    };
    if (total <= 1) {
        launch_expand.template operator()<1>();
    } else if (total <= 2) {
        launch_expand.template operator()<2>();
    } else if (total <= 4) {
        launch_expand.template operator()<4>();
    } else if (total <= kLoraExpandColumns) {
        launch_expand.template operator()<kLoraExpandColumns>();
    } else {
        constexpr int rows_per_block = kLoraThreads * kLoraExpandRows;
        const dim3 grid(static_cast<unsigned>((widest + rows_per_block - 1) / rows_per_block),
                        static_cast<unsigned>((total + kLoraExpandColumns - 1) / kLoraExpandColumns),
                        static_cast<unsigned>(up.size()));
        const std::size_t shared =
            static_cast<std::size_t>(deepest) * kLoraExpandColumns * sizeof(float);
        lora_expand_rows_kernel<kLoraExpandColumns, kLoraExpandRows>
            <<<grid, kLoraThreads, shared, stream>>>(expand);
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
