#pragma once

// NINFER_NO_COOPERATIVE_KERNELS=1: never pick a schedule that is launched cooperatively
// (cudaLaunchAttributeCooperative + cooperative_groups grid sync); take its non-cooperative
// fallback instead. Off by default, read once per process.
//
// It exists for profiling. Under Nsight Systems on sm_70 (Tesla V100) every cooperative grid-sync
// kernel faults with cudaErrorInvalidPc, so a model whose routes include one (Qwen3.6-35B-A3B's
// BF16 GDN gating projection and its fused norm-gating route) cannot be profiled without it. Set
// it when running under nsys on sm_70. It is harmless elsewhere: it only moves those calls to the
// slower non-cooperative schedule, which may round differently (the split-K reduction order
// changes), so leave it unset for measurements and bit-identity checks.
//
// The cooperative launch sites it covers (grep cudaLaunchAttributeCooperative /
// cudaLaunchCooperativeKernel / this_grid().sync()):
//   - ops/gdn_gating_proj/bf16: MmaCooperativeSplit32/16/8/4/2 of the gating projection, and the
//     35B norm-gating MmaCooperativeSplit32 route (cols <= 16), which falls back to Composed.

#include <cstdlib>

namespace ninfer::ops {

// The setting a value of NINFER_NO_COOPERATIVE_KERNELS asks for: "1" disables cooperative
// kernels; unset, empty, "0" or anything else leaves them on (the default, as with the other
// NINFER_* opt-ins).
constexpr bool cooperative_kernels_disabled_by(const char* value) noexcept {
    return value != nullptr && value[0] == '1' && value[1] == '\0';
}

// Whether cooperative schedules may be picked in this process (NINFER_NO_COOPERATIVE_KERNELS,
// read on first use and fixed for the life of the process).
inline bool cooperative_kernels_allowed() noexcept {
    static const bool allowed =
        !cooperative_kernels_disabled_by(std::getenv("NINFER_NO_COOPERATIVE_KERNELS"));
    return allowed;
}

} // namespace ninfer::ops
