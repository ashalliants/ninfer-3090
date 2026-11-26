// NVFP4 SwiGLU above 16 tokens on the A16 route: the NVFP4 A16 MMA kernel with the row-major SwiGLU
// epilogue, as fp8_linear_swiglu_matrix_launch does for FP8. Gate and up rows share a tile, and the
// kernel scales both accumulators by the weight's multiplier in FP32 before SiLU, so the
// intermediate never rounds to BF16. This is the only wide route on targets without NVFP4 tensor
// cores (sm_70, sm_80-89), where A16 is the only policy NVFP4 weights admit.
#include "core/weight.h"
#include "ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_plan.h"

#include "core/device.h"
#include "ops/linear/nvfp4/nvfp4_instances.cuh"
#include "ops/linear/nvfp4/nvfp4_schedule.cuh"
#include "ops/linear/nvfp4/nvfp4_template_launch.cuh"
#include "ops/linear_swiglu/row_major_mma_epilogue.cuh"

#include <cuda_bf16.h>

namespace ninfer::ops::detail {

void nvfp4_linear_swiglu_matrix_launch(const Tensor& x, const Weight& weight, Tensor& out,
                                       cudaStream_t stream) {
    const auto p = nvfp4_a16_operands(x, weight);
    const LinearBf16Output output{static_cast<__nv_bfloat16*>(out.data), weight.n / 2};
    const auto mma = [&]<class Schedule>() {
        using S = Nvfp4ScheduleInstance<Schedule, 5120>;
        launch_nvfp4_a16_mma<S>(p, output, SwiGluRowMajorMmaEpilogue{}, stream,
                                SwiGluRowMajorMmaRows<S>{});
    };
    if (x.ne[1] <= 64)
        return mma.template operator()<Nvfp4A16MmaSchedule<32, 64, 128, 16, 16, 1, 3>>();
    if (x.ne[1] <= 96)
        return mma.template operator()<Nvfp4A16MmaSchedule<64, 96, 128, 64, 16, 1, 2>>();
    mma.template operator()<Nvfp4A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>>();
}

} // namespace ninfer::ops::detail
