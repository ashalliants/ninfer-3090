#include "ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_plan.h"
#include "ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_epilogue.cuh"
#include "ops/linear/nvfp4/nvfp4_template_launch.cuh"

namespace ninfer::ops::detail {
void nvfp4_linear_swiglu_decode_launch(const Tensor& x, const Weight& w, Tensor& y,
                                       cudaStream_t stream) {
    using S = Nvfp4ScheduleInstance<
        Nvfp4A16GemvSchedule<8, 2, 16, 4, Nvfp4ScaleAccess::Direct, Nvfp4CodeCache::Default, 2>,
        5120>;
    launch_nvfp4_a16_gemv<S>(nvfp4_a16_operands(x, w),
                             LinearBf16Output{static_cast<__nv_bfloat16*>(y.data), w.n / 2},
                             Nvfp4SwiGluEpilogue{}, stream, Nvfp4SwiGluRows<1>{});
}

bool nvfp4_linear_swiglu_stream_launch(const Tensor& x, const Weight& w, Tensor& y,
                                       cudaStream_t stream) {
    return try_launch_nvfp4_stream<LinearBf16Output, Nvfp4SwiGluEpilogue, Nvfp4SwiGluRows<1>>(
        nvfp4_a16_operands(x, w), LinearBf16Output{static_cast<__nv_bfloat16*>(y.data), w.n / 2},
        Nvfp4SwiGluEpilogue{}, stream);
}
} // namespace ninfer::ops::detail
