#include "core/weight.h"
#include "ops/gdn_input_proj/nvfp4/nvfp4_gdn_input_plan.h"

#include "core/device.h"
#include "ops/gdn_input_proj/nvfp4/nvfp4_gdn_input_output.cuh"
#include "ops/linear/nvfp4/nvfp4_schedule.cuh"
#include "ops/linear/nvfp4/nvfp4_template_launch.cuh"

namespace ninfer::ops::detail {

void nvfp4_gdn_input_decode_launch(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                                   cudaStream_t stream) {
    using Geometry = Nvfp4N16384K5120;
    using Schedule =
        Nvfp4A16GemvSchedule<8, 2, 16, 4, Nvfp4ScaleAccess::StagedRaw, Nvfp4CodeCache::Default, 2>;

    launch_nvfp4_a16_gemv<Nvfp4ScheduleInstance<Schedule, Geometry::kInputRows>>(
        nvfp4_a16_operands(x, weight),
        Nvfp4GdnInputOutput{static_cast<__nv_bfloat16*>(qkv.data),
                            static_cast<__nv_bfloat16*>(z.data)},
        LinearIdentityEpilogue{}, stream);
}

bool nvfp4_stream_admits(std::int32_t tokens, cudaStream_t stream) {
    if constexpr (!kStreamDp4aCompiled) {
        return false;
    } else {
        constexpr std::int32_t kHidden = Nvfp4N16384K5120::kInputRows;
        return tokens >= 1 && tokens <= kStreamGemvMaxTokens && stream_dp4a_enabled(stream) &&
               (tokens <= kStreamDp4aMaxCols ||
                stream_tile_gemm_supported<Nvfp4StreamFormat>(kHidden, kHidden, kHidden));
    }
}

bool nvfp4_gdn_input_stream_launch(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                                   cudaStream_t stream) {
    return try_launch_nvfp4_stream<Nvfp4GdnInputOutput, LinearIdentityEpilogue, Nvfp4IdentityRows>(
        nvfp4_a16_operands(x, weight),
        Nvfp4GdnInputOutput{static_cast<__nv_bfloat16*>(qkv.data),
                            static_cast<__nv_bfloat16*>(z.data)},
        LinearIdentityEpilogue{}, stream);
}

} // namespace ninfer::ops::detail
