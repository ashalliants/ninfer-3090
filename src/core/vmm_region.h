#pragma once

// Adapted from NInfer-all 0dcb1da9 src/core/vmm.h (Apache-2.0).
// Modified for NInfer-3090: it sits on this tree's cuda_vmm.h helpers instead of replacing them, and
// a mapping WDDM cannot make resident (CUDA_ERROR_NOT_READY from cuMemSetAccess) is a refusal, like
// an out-of-memory cuMemCreate.

#include "core/arena.h"

#include <cuda.h>

#include <cstddef>
#include <vector>

namespace ninfer {

// Device memory at an address fixed for the region's lifetime whose physical backing can be released
// and recreated: suspending a model gives its memory back to the device without invalidating
// anything that captured an address inside it (tensors, block tables, CUDA Graphs). Contents do not
// survive a release; an owner that needs them copies them out first (DeviceSnapshot).
class VmmRegion {
public:
    struct Options {
        int device = 0;
        // Further devices granted direct access, for memory a peer reads or writes across the bus.
        std::vector<int> peers;
        // Physical allocation unit, a multiple of the device granularity. Larger pieces map and
        // release faster; the last piece is shortened to the reservation. Zero picks 256 MiB.
        std::size_t piece_bytes = 0;
    };

    VmmRegion() noexcept = default;
    VmmRegion(std::size_t bytes, Options options);
    ~VmmRegion();

    VmmRegion(const VmmRegion&)            = delete;
    VmmRegion& operator=(const VmmRegion&) = delete;
    VmmRegion(VmmRegion&& other) noexcept;
    VmmRegion& operator=(VmmRegion&& other) noexcept;

    // Whether `device` supports the driver's virtual memory management. A suspendable region refuses
    // to start without it rather than fall back to an allocation that cannot be released.
    [[nodiscard]] static bool supported(int device);

    // The usable extent, `bytes` from the constructor; the reservation is rounded up past it.
    [[nodiscard]] DeviceSpan span() const noexcept;
    [[nodiscard]] std::size_t reserved_bytes() const noexcept { return reserved_; }
    [[nodiscard]] int device() const noexcept { return options_.device; }
    [[nodiscard]] bool backed() const noexcept { return backed_; }
    [[nodiscard]] explicit operator bool() const noexcept { return base_ != 0; }

    // Unmaps and frees every physical piece. The caller has drained all work that addresses the
    // region. Idempotent. If the driver refuses part of it, it throws after releasing everything it
    // could; the region is then not backed, restore_backing() maps what is missing, and a repeated
    // release only retries the pieces still held.
    void release_backing();

    // Maps fresh physical pieces at the same addresses; their contents are undefined. When the
    // device cannot supply them, every piece this call created is released again before it throws,
    // so the region stays unbacked and the caller may retry once memory is free. Idempotent.
    void restore_backing();

private:
    void reset() noexcept;
    CUresult release_pieces() noexcept;

    CUdeviceptr base_     = 0;
    std::size_t bytes_    = 0;
    std::size_t reserved_ = 0;
    std::size_t piece_    = 0;
    Options options_{};
    std::vector<int> access_; // owner first, then peers
    // One entry per piece. A piece can be created but unmapped (its release failed part way), so
    // mapping and handle ownership are tracked separately and a retry repeats only what is left.
    struct Piece {
        CUmemGenericAllocationHandle handle = 0; // 0: no physical allocation held
        bool mapped                         = false;
    };
    std::vector<Piece> pieces_;
    bool backed_ = false; // every piece is mapped
};

} // namespace ninfer
