#include "core/host_gather.h"

#include "core/device.h"

#include <algorithm>
#include <stdexcept>

namespace ninfer {
namespace {

// 32 blocks of 512 threads, one copy piece per block at a time, reach the PCIe 4.0 x16 H2D rate on
// an RTX 3090 (~23 GB/s) and leave most SMs to the forward pass running beside the copy.
constexpr unsigned kGatherBlocks  = 32;
constexpr unsigned kGatherThreads = 512;

template <class Word>
__device__ void copy_rows(const GatherCopy& copy, std::size_t row_words, unsigned pace_ns) {
    auto* destination        = static_cast<Word*>(copy.destination);
    const auto* source       = static_cast<const Word*>(copy.source);
    const std::size_t dpitch = copy.destination_pitch / sizeof(Word);
    const std::size_t spitch = copy.source_pitch / sizeof(Word);
    const std::size_t total  = row_words * copy.height;
    for (std::size_t index = threadIdx.x; index < total; index += blockDim.x) {
        const std::size_t row = index / row_words, column = index - row * row_words;
        destination[row * dpitch + column] = source[row * spitch + column];
        if (pace_ns != 0) { __nanosleep(pace_ns); }
    }
}

// Each block copies whole pieces; split_gather_copies bounds a piece so the blocks share the work.
__global__ void gather_kernel(const GatherCopy* __restrict__ descriptors, std::uint32_t count, unsigned pace_ns) {
    for (std::uint32_t d = blockIdx.x; d < count; d += gridDim.x) {
        const GatherCopy copy = descriptors[d];
        const auto bits = reinterpret_cast<std::uintptr_t>(copy.destination) |
                          reinterpret_cast<std::uintptr_t>(copy.source) | copy.destination_pitch |
                          copy.source_pitch | copy.width;
        if (bits % sizeof(int4) == 0) {
            copy_rows<int4>(copy, copy.width / sizeof(int4), pace_ns);
        } else {
            copy_rows<unsigned char>(copy, copy.width, pace_ns);
        }
    }
}

} // namespace

void split_gather_copies(std::vector<GatherCopy>& copies, std::size_t piece_bytes) {
    std::vector<GatherCopy> out;
    out.reserve(copies.size());
    for (const GatherCopy& copy : copies) {
        const std::uint64_t rows =
            std::max<std::uint64_t>(1, piece_bytes / std::max<std::uint64_t>(1, copy.width));
        for (std::uint64_t row = 0; row < copy.height; row += rows) {
            GatherCopy piece  = copy;
            piece.destination = static_cast<unsigned char*>(copy.destination) +
                                row * copy.destination_pitch;
            piece.source = static_cast<const unsigned char*>(copy.source) + row * copy.source_pitch;
            piece.height = std::min(rows, copy.height - row);
            out.push_back(piece);
        }
    }
    copies = std::move(out);
}

namespace {

void launch_gather(std::span<const GatherCopy> descriptors, unsigned max_blocks, unsigned threads, unsigned pace_ns,
                   cudaStream_t stream) {
    if (descriptors.empty()) { return; }
    if (descriptors.size() > UINT32_MAX) { throw std::overflow_error("too many gather copies"); }
    void* mapped = nullptr;
    CUDA_CHECK(cudaHostGetDevicePointer(&mapped, const_cast<GatherCopy*>(descriptors.data()), 0));
    // Descriptors carry Host addresses as the kernel reads them, which holds under unified
    // addressing; the descriptor buffer is allocated the same way as every Host source.
    if (mapped != static_cast<const void*>(descriptors.data())) {
        throw std::logic_error("pinned Host memory is not identity-mapped on this device");
    }
    const auto blocks = static_cast<unsigned>(std::min<std::size_t>(max_blocks, descriptors.size()));
    gather_kernel<<<blocks, threads, 0, stream>>>(static_cast<const GatherCopy*>(mapped),
                                                  static_cast<std::uint32_t>(descriptors.size()), pace_ns);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

void require_identity_mapped(const void* host) {
    void* mapped = nullptr;
    CUDA_CHECK(cudaHostGetDevicePointer(&mapped, const_cast<void*>(host), 0));
    if (mapped != host) { throw std::logic_error("pinned Host memory is not identity-mapped on this device"); }
}

void gather_copies(std::span<const GatherCopy> descriptors, cudaStream_t stream) {
    launch_gather(descriptors, kGatherBlocks, kGatherThreads, 0, stream);
}

void gather_copies_paced(std::span<const GatherCopy> descriptors, GatherPace pace, cudaStream_t stream) {
    if (pace.blocks == 0 || pace.threads == 0 || pace.threads > 1024 || pace.threads % 32 != 0) {
        throw std::invalid_argument("gather_copies_paced: blocks > 0 and threads a multiple of 32 up to 1024");
    }
    launch_gather(descriptors, pace.blocks, pace.threads, pace.pace_ns, stream);
}

} // namespace ninfer
