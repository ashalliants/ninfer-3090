#pragma once

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops {

/// One device-to-device copy of copy_runs: `bytes` from src + offset * src_step to dst. dst, src, bytes and src_step
/// multiples of 16; the copies of one call must not overlap each other's destinations or a source another writes.
struct CopyRun {
    void* dst            = nullptr;
    const void* src      = nullptr;
    std::uint64_t bytes  = 0;
    std::uint64_t src_step = 0;
};

/**
 * Op: copy_runs (exact: a byte copy)
 *
 *   for each run r of runs[0, count): dst_r[0, bytes_r) = (src_r + offset * src_step_r)[0, bytes_r)
 *
 * runs: `count` CopyRun records in device memory (they stay valid until the launch completes). One launch, in place
 * of a driver call a copy: a call of a few hundred small copies otherwise holds the host for hundreds of
 * microseconds (cudaMemcpyBatchAsync), with the GPU idle when it waits on that work. Asynchronous on `stream`.
 */
void copy_runs(const CopyRun* runs, std::int32_t count, std::int64_t offset, cudaStream_t stream);

/// Whether a run meets copy_runs's alignment (dst, src, bytes and src_step multiples of 16).
[[nodiscard]] bool copy_run_aligned(const CopyRun& run) noexcept;

} // namespace ninfer::ops
