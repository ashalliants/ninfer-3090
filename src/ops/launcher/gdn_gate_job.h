#pragma once

// ninfer::ops - the decay form of the GDN gates (gdn_gating_decay) as a job another kernel can run
// beside its own work: element i = t * 48 + h reads a and b at t * ab_stride + h.

#include <cuda_bf16.h>

#include <cstdint>

namespace ninfer::ops {

struct GdnGateJob {
    const __nv_bfloat16* a = nullptr;
    const __nv_bfloat16* b = nullptr;
    const float* decay     = nullptr;
    const float* dt_bias   = nullptr;
    float* g               = nullptr;
    float* beta            = nullptr;
    std::int64_t n         = 0; // 0: no job
    int ab_stride          = 48;
};

} // namespace ninfer::ops
