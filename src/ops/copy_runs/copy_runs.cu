#include "ninfer/ops/copy_runs.h"

#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

constexpr int kThreads = 256, kBlocksPerRun = 4;

// grid (count, kBlocksPerRun): block (r, b) copies run r's 16-byte words b * 256 + t, then every kBlocksPerRun * 256th.
__global__ void __launch_bounds__(kThreads) copy_runs_kernel(const CopyRun* __restrict__ runs, std::int64_t offset) {
    const CopyRun run = runs[blockIdx.x];
    auto* dst         = static_cast<uint4*>(run.dst);
    const auto* src   = reinterpret_cast<const uint4*>(static_cast<const unsigned char*>(run.src) + offset * static_cast<std::int64_t>(run.src_step));
    const std::uint64_t words = run.bytes / 16;
    for (std::uint64_t i = blockIdx.y * kThreads + threadIdx.x; i < words; i += kBlocksPerRun * kThreads) { dst[i] = src[i]; }
}

} // namespace

bool copy_run_aligned(const CopyRun& run) noexcept {
    return reinterpret_cast<std::uintptr_t>(run.dst) % 16 == 0 && reinterpret_cast<std::uintptr_t>(run.src) % 16 == 0 && run.bytes % 16 == 0 &&
           run.src_step % 16 == 0;
}

void copy_runs(const CopyRun* runs, std::int32_t count, std::int64_t offset, cudaStream_t stream) {
    if (count < 0 || count > 65535 || (count > 0 && runs == nullptr)) { throw std::invalid_argument("copy_runs: 0 to 65535 runs, in device memory"); }
    if (count == 0) { return; }
    copy_runs_kernel<<<dim3(count, kBlocksPerRun), kThreads, 0, stream>>>(runs, offset);
    const cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) { throw std::runtime_error(std::string("copy_runs: ") + cudaGetErrorString(error)); }
}

} // namespace ninfer::ops
