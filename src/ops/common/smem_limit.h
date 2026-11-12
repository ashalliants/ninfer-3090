#pragma once

namespace ninfer::ops {

// The largest dynamic shared memory one block can opt into with
// cudaFuncAttributeMaxDynamicSharedMemorySize. GA10x reports 99 KiB (100 KiB per SM, 1 KiB reserved
// by the driver); Volta's opt-in limit and its per-SM capacity are both 96 KiB. Every schedule that
// sizes its tiles to the limit static_asserts against this, so a schedule that cannot launch on the
// target fails to compile instead of failing at runtime.
#if defined(NINFER_SM70)
inline constexpr int kMaxDynamicSharedBytes = 96 * 1024;
inline constexpr int kSharedBytesPerSm      = 96 * 1024;
#else
inline constexpr int kMaxDynamicSharedBytes = 99 * 1024;
inline constexpr int kSharedBytesPerSm      = 100 * 1024;
#endif

// sm_120 (RTX 50) has the GA10x limit. The FP8 A8 and NVFP4 A4 mma routes and every TMA route only
// ever execute there, but their schedules still compile as dead code on the other targets, so they
// assert against this and not against the build target's limit.
inline constexpr int kSm120MaxDynamicSharedBytes = 99 * 1024;

} // namespace ninfer::ops
