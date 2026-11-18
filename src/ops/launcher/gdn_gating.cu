// ninfer::ops - gdn_gating launcher: grid/block/stream configuration + kernel launch.
#include "ops/launcher/gdn_gating.h"

#include "ops/common/math.h"
#include "ops/kernel/gdn_gating.cuh"
#include "core/device.h" // CUDA_CHECK

#include <algorithm>
#include <cstdint>

namespace ninfer::ops::detail {

void gdn_gating_launch(const Tensor& a, const Tensor& b, const Tensor& rate, const Tensor& dt_bias,
                       bool decay, Tensor& g, Tensor& beta, cudaStream_t stream) {
    gdn_gating_launch(static_cast<const __nv_bfloat16*>(a.data), static_cast<const __nv_bfloat16*>(b.data), 48, rate,
                      dt_bias, decay, g, beta, stream);
}

void gdn_gating_launch(const __nv_bfloat16* av, const __nv_bfloat16* bv, int ab_stride, const Tensor& rate,
                       const Tensor& dt_bias, bool decay, Tensor& g, Tensor& beta, cudaStream_t stream) {
    const std::int64_t n = g.numel();
    constexpr int kBlock = 256;
    const int grid =
        static_cast<int>(std::max<std::int64_t>(1, div_up(n, static_cast<std::int64_t>(kBlock))));

    const auto* rv    = static_cast<const float*>(rate.data);
    const auto* bias  = static_cast<const float*>(dt_bias.data);
    auto* gv          = static_cast<float*>(g.data);
    auto* betav       = static_cast<float*>(beta.data);
    if (decay) {
        gdn_gating_kernel<true><<<grid, kBlock, 0, stream>>>(av, bv, rv, bias, gv, betav, n, ab_stride);
    } else {
        gdn_gating_kernel<false><<<grid, kBlock, 0, stream>>>(av, bv, rv, bias, gv, betav, n, ab_stride);
    }
    CUDA_CHECK(cudaGetLastError());
}

void gdn_gate_job_launch(const GdnGateJob& job, cudaStream_t stream) {
    if (job.n == 0) { return; }
    constexpr int kBlock = 256;
    const int grid =
        static_cast<int>(std::max<std::int64_t>(1, div_up(job.n, static_cast<std::int64_t>(kBlock))));
    gdn_gating_kernel<true><<<grid, kBlock, 0, stream>>>(job.a, job.b, job.decay, job.dt_bias, job.g, job.beta,
                                                         job.n, job.ab_stride);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
