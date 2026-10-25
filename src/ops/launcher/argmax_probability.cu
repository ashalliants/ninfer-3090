// Implements: include/ninfer/ops/argmax_probability.h
// Match: wrapper-validated contiguous FP32 logits, I32 ids and FP32 probs.
// Algorithm assumptions: per-tile partials in caller workspace, then one combining CTA per column.
#include "ops/launcher/argmax_probability.h"

#include "core/device.h" // CUDA_CHECK
#include "core/layout.h"
#include "ops/common/math.h"
#include "ops/common/token_slices.h"
#include "ops/kernel/argmax_probability.cuh"

#include <cstdint>

namespace ninfer::ops::detail {
namespace {

struct Partials {
    double* sums          = nullptr;
    float* keys           = nullptr;
    std::int32_t* indices = nullptr;
    float* maxima         = nullptr;
};

std::int32_t tiles_of(std::int32_t valid_rows) { return div_up(valid_rows, kArgmaxProbabilityTileRows); }

// One recipe for the sizing query (WorkspaceLayoutBuilder) and the call (WorkspaceArena).
template <class Allocator>
Partials allocate_partials(Allocator& allocator, std::int32_t valid_rows, std::int32_t columns) {
    const std::size_t n = static_cast<std::size_t>(tiles_of(valid_rows)) * static_cast<std::size_t>(columns);
    Partials out;
    out.sums    = static_cast<double*>(allocator.alloc_bytes(n * sizeof(double)).data);
    out.keys    = static_cast<float*>(allocator.alloc_bytes(n * sizeof(float)).data);
    out.indices = static_cast<std::int32_t*>(allocator.alloc_bytes(n * sizeof(std::int32_t)).data);
    out.maxima  = static_cast<float*>(allocator.alloc_bytes(n * sizeof(float)).data);
    return out;
}

} // namespace

std::size_t argmax_probability_workspace_bytes(std::int32_t valid_rows, std::int32_t columns) {
    WorkspaceLayoutBuilder layout;
    (void)allocate_partials(layout, valid_rows, columns);
    return layout.peak_bytes(1);
}

void argmax_probability_launch(const Tensor& logits, Tensor& ids, Tensor& probs, std::int32_t valid_rows,
                               WorkspaceArena& workspace, cudaStream_t stream) {
    const std::int32_t physical_rows = logits.ne[0];
    const std::int32_t columns       = logits.ne[1];
    const std::int32_t tiles         = tiles_of(valid_rows);
    const Partials partials          = allocate_partials(workspace, valid_rows, columns);
    const auto* source               = static_cast<const float*>(logits.data);
    for_each_token_slice(columns, 1, [&](int offset, int count) {
        const std::size_t at = static_cast<std::size_t>(offset) * tiles;
        const dim3 grid(static_cast<unsigned int>(tiles), static_cast<unsigned int>(count));
        argmax_probability_tiles_kernel<<<grid, kArgmaxProbabilityBlock, 0, stream>>>(
            source + static_cast<std::int64_t>(offset) * physical_rows, partials.keys + at, partials.indices + at,
            partials.maxima + at, partials.sums + at, valid_rows, physical_rows);
        CUDA_CHECK(cudaGetLastError());
    });
    argmax_probability_combine_kernel<<<static_cast<unsigned int>(columns), kArgmaxProbabilityBlock, 0, stream>>>(
        source, partials.keys, partials.indices, partials.maxima, partials.sums, static_cast<std::int32_t*>(ids.data),
        static_cast<float*>(probs.data), tiles, physical_rows);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
