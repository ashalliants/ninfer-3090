// Pins the sm_70 route decisions (docs/maintainer/volta-port.md): which shapes and token bands leave
// the emulated mma for GEMV / SIMT schedules, and that the INT8-G64 FA2 prompt plan never asks for
// the eight-warp CTA that does not fit Volta's shared memory. Pure host: no device work.
#include "ops/linear/q4/q4_dispatch.h"
#include "ops/linear/q4/q4_launch.h"
#include "ops/linear/t2/t2_dispatch.h"
#include "ops/softmax_attention/dense/causal_cache/prompt_i8_fa2_plan.h"

#include <cstdint>
#include <iostream>

namespace {

using namespace ninfer::ops::detail;

int failures = 0;

void expect(bool ok, const char* what) {
    if (!ok) {
        std::cerr << "FAIL: " << what << '\n';
        ++failures;
    }
}

} // namespace

int main() {
#if defined(NINFER_SM70)
    // Q4 decode: the K=5120 / K=6144 attention and GDN projections take the GEMV instances.
    for (const std::int32_t n : {4096, 6144, 7168}) {
        expect(select_q4_a16_launch(n, 5120, 1) == launch_q4_a16_gemv_r1_w8_direct,
               "q4 n=4096/6144/7168 k=5120 T=1 -> GEMV");
        expect(select_q4_a16_launch(n, 5120, 2) != launch_q4_a16_gemv_r1_w8_direct,
               "q4 k=5120 T=2 keeps the sm_86 table");
    }
    expect(select_q4_a16_launch(5120, 6144, 1) == launch_q4_a16_gemv_r1_w8_k6144,
           "q4 n=5120 k=6144 T=1 -> GEMV");
    // The K = 4096 / 17408 / 25600 shapes have no valid GEMV instance and stay on the table.
    for (const std::int32_t k : {4096, 17408, 25600}) {
        expect(select_q4_a16_launch(5120, k, 1) != launch_q4_a16_gemv_r1_w8_direct &&
                   select_q4_a16_launch(5120, k, 1) != launch_q4_a16_gemv_r1_w8_k6144,
               "q4 ksplit shapes without a GEMV instance are not rerouted");
    }

    // Ternary: GEMV at T=1, SIMT for T<=8, then the sm_86 tensor-core bands.
    expect(select_t2_a16_launch(6144, 5120, 1) == launch_t2_gemv_r8_w1_k5120, "t2 T=1 -> GEMV");
    for (const std::int32_t t : {2, 3, 4}) {
        expect(select_t2_a16_launch(6144, 5120, t) == launch_t2_simt_r8_c4, "t2 T=2..4 -> SIMT c4");
    }
    for (const std::int32_t t : {5, 8}) {
        expect(select_t2_a16_launch(6144, 5120, t) == launch_t2_simt_r8_c8, "t2 T=5..8 -> SIMT c8");
    }
    expect(select_t2_a16_launch(6144, 5120, 9) == launch_t2_small_t_v2, "t2 T=9 -> small-T mma");
    expect(select_t2_a16_launch(6144, 5120, 64) == launch_t2_mma_r64_c64, "t2 T=64 -> mma");
#endif

    // FA2 INT8 prompt plan. Eight warps need 100,352 B (INT8 values) or 84,992 B (packed values).
    expect(causal_prompt_fa2_eight_warps_fit(true), "packed-value eight-warp CTA fits every target");
#if defined(NINFER_SM70)
    expect(!causal_prompt_fa2_eight_warps_fit(false), "INT8-value eight-warp CTA exceeds 96 KiB");
    // A launch that would split or take eight warps on sm_86 stays one four-warp CTA grid.
    const CausalPromptFa2Plan plan = causal_prompt_fa2_plan(24, 2048, 131072, 80, false);
    expect(plan.warps == 4 && plan.splits == 1, "no eight-warp plan without the shape");
    const CausalPromptFa2Plan narrow = causal_prompt_fa2_plan(24, 32, 4096, 80, false);
    expect(narrow.warps == 4 && narrow.splits == 1, "narrow launch is four warps too");
#else
    expect(causal_prompt_fa2_eight_warps_fit(false), "INT8-value eight-warp CTA fits 99 KiB");
#endif
    const CausalPromptFa2Plan narrow8 = causal_prompt_fa2_plan(24, 32, 4096, 80, true);
    expect(narrow8.warps == 8, "with the shape, a narrow launch takes eight warps");

    if (failures != 0) return 1;
    std::cout << "volta routes: OK\n";
    return 0;
}
