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
    const dim3 shrink_grid(static_cast<unsigned>((total + kLoraShrinkColumns - 1) / kLoraShrinkColumns),
                           static_cast<unsigned>(down.rank / kLoraShrinkRows),
                           static_cast<unsigned>(splits));
    lora_shrink_kernel<<<shrink_grid, kLoraThreads, 0, stream>>>(shrink);
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
    const dim3 expand_grid(static_cast<unsigned>((widest + kLoraThreads - 1) / kLoraThreads),
                           static_cast<unsigned>((total + kLoraExpandColumns - 1) / kLoraExpandColumns),
                           static_cast<unsigned>(up.size()));
    const std::size_t shared = static_cast<std::size_t>(deepest) * kLoraExpandColumns * sizeof(float);
    lora_expand_kernel<<<expand_grid, kLoraThreads, shared, stream>>>(expand);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
