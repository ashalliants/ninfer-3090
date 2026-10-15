#include "ops/linear/q4/q4_shapes.h"
#include "ops/linear/q4/q4_ksplit_launch.cuh"

namespace ninfer::ops::detail {
// The DFlash2 adapter of a ternary target is Q4. This shape is its hidden-width output or
// feature projection; the ladder is n5120_k6144.cu's, not separately measured at this K.
Q4Launch select_q4_n5120_k4096(std::int32_t tokens) {
    if (tokens <= 8) return launch_q4_ksplit<5120, 4096, 8>;
    if (tokens <= 24) return launch_q4_ksplit<5120, 4096, 24>;
    if (tokens <= 32) return launch_q4_a16_mma_r32_t32_k64_wr16_wt8_s4_a4_b2;
    if (tokens <= 64) return launch_q4_a16_mma_r32_t32_k64_wr16_wt16_s3_a3_b2;
    if (tokens <= 80) return launch_q4_a16_mma_r64_t80;
    if (tokens <= 96) return launch_q4_a16_mma_r64_t96;
    if (tokens <= 128) return launch_q4_a16_mma_r64_t64_k64_wr32_wt16_s2_a2_b2;
    if (tokens <= 160) return launch_q4_a16_mma_r64_t80;
    if (tokens <= 192) return launch_q4_a16_mma_r64_t96;
    if (tokens <= 224) return launch_q4_a16_mma_r64_t112;
    if (tokens <= 256) return launch_q4_a16_mma_r64_t128;
    if (tokens <= 320) return launch_q4_a16_mma_r64_t80;
    return launch_q4_a16_mma_r64_t128;
}

} // namespace ninfer::ops::detail
