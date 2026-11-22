#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace ninfer {

// One 2D copy for gather_copies: `height` rows of `width` bytes. `source` is page-locked Host memory
// (cudaHostAlloc, mapped under unified addressing) or Device memory.
struct GatherCopy {
    void* destination               = nullptr;
    const void* source              = nullptr;
    std::uint64_t destination_pitch = 0;
    std::uint64_t source_pitch      = 0;
    std::uint64_t width             = 0;
    std::uint64_t height            = 0;
};

// Runs a list of 2D copies as one kernel that reads Host pages over PCIe, instead of copy-engine
// work. A GeForce card under WDDM exposes one copy engine (asyncEngineCount = 1), so a bulk H2D
// there holds every later H2D from any stream behind it -- including the few bytes of inputs a
// forward pass uploads. A gather leaves the engine free and reaches the PCIe rate with a few SMs.
// `descriptors` lives in page-locked Host memory (cudaHostAlloc) and must stay valid and unchanged
// until the kernel has completed on `stream`.
void gather_copies(std::span<const GatherCopy> descriptors, cudaStream_t stream);

// A background gather that a forward on another stream runs beside: `blocks` blocks of `threads` threads, each thread
// sleeping `pace_ns` after each 16-byte word it copies (about blocks * threads * 16 / pace_ns bytes per ns). On a V100
// whose link takes device-to-host writes at about 3.3 GB/s, an unpaced gather to the host stalls the forward's kernels
// beside it for about the copy's own time (its writes back up into the memory system), and a copy-engine copy holds the
// forward's own small device-to-host copies behind it; one paced block at about 2 GB/s costs the forward a few ms per
// 240 MB.
// Throws std::logic_error unless the page-locked Host memory at `host` (cudaHostAlloc, mapped) has its own address on
// the device: a GatherCopy carries Host addresses as the kernel reads them.
void require_identity_mapped(const void* host);

struct GatherPace {
    unsigned blocks   = 1;
    unsigned threads  = 128;
    unsigned pace_ns  = 1000;
};
void gather_copies_paced(std::span<const GatherCopy> descriptors, GatherPace pace, cudaStream_t stream);

// Splits copies by rows into pieces of at most `piece_bytes` (at least one row), so the kernel's
// blocks, which take whole pieces, share a long copy.
void split_gather_copies(std::vector<GatherCopy>& copies, std::size_t piece_bytes);

} // namespace ninfer
