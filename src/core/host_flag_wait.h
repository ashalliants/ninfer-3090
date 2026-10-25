#pragma once

// Adapted from Basalt 8d73a39 src/kernels/cuda/verify_kernels.cu (wait_flag_ge) (MIT, third_party/basalt/LICENSE).
// Modified for NInfer-3090: a wrap-aware comparison, a device sequence counter (one captured
// launch waits for each replay's own value), and a timeout that reports to an error word. Core
// infrastructure beside upload_pinned (core/pinned_upload.h): it orders stream work after a host
// write and has no tensor effect, so it is not an Op.

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer {

// A wait gives up when the flag has not reached its value for this long (GPU time): 1 s, under the
// 2 s of Windows' display watchdog (TDR), the same as the CPU-miss wait of offloaded_sparse_moe.
inline constexpr std::uint64_t kWaitFlagTimeoutNs = 1000000000ULL;
// What a timed-out wait writes to its error word.
inline constexpr std::uint32_t kWaitFlagErrorTimeout = 0xFF03;

/**
 * wait_flag_ge / wait_flag_sequence: a stream-ordered wait for a host flag
 *
 * Math / indexing:
 *   reached(f, v) = (int32)(f - v) >= 0, the wrap-aware f >= v of 32-bit counters (f is at or up
 *   to 2^31 - 1 past v, modulo 2^32).
 *   wait_flag_ge:       want = value.
 *   wait_flag_sequence: want = *sequence + 1 (mod 2^32); *sequence' = want.
 *   The call completes once reached(*mapped, want) holds. When it has not held for
 *   kWaitFlagTimeoutNs, the call writes kWaitFlagErrorTimeout to *error and completes; with
 *   error == nullptr it traps instead (the context is lost).
 *
 * Logical shapes:
 *   mapped, error: one U32 word each, in host memory mapped to the device (their device
 *   addresses); sequence: one U32 word in device memory.
 *
 * Supported domain:
 *   Any value; the flag's writer raises it monotonically (each raise at most 2^31 - 1 past the
 *   value a waiter wants).
 *
 * Numeric:
 *   Integer only; exact.
 *
 * Effects:
 *   Reads *mapped with system-scope acquire semantics: stream work after the call sees every
 *   write the host made before the store that raised the flag (the host's release store). Writes
 *   *error only on a timeout; wait_flag_sequence writes *sequence. Nothing else.
 *
 * Workspace:
 *   None.
 *
 * Execution:
 *   Stream-ordered and graph capturable: a captured wait_flag_sequence waits for a new value at
 *   each replay, so the flag's writer raises it once per replay (to the count of replays) and a
 *   value left from an earlier replay never releases a later one. The device spins (sleeping
 *   between reads) until then, so work enqueued after the call starts only after the flag is
 *   raised; the host must raise it without waiting for the stream. Under serialized launches
 *   (CUDA_LAUNCH_BLOCKING=1, compute-sanitizer) a launch returns only once the wait is done, so a
 *   host that raises the flag after the launch always meets the timeout.
 */
void wait_flag_ge(const volatile std::uint32_t* mapped, std::uint32_t value, std::uint32_t* error,
                  cudaStream_t stream);
void wait_flag_sequence(const volatile std::uint32_t* mapped, std::uint32_t* sequence,
                        std::uint32_t* error, cudaStream_t stream);

} // namespace ninfer
