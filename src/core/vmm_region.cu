// Adapted from NInfer-all 0dcb1da9 src/core/vmm.cu (Apache-2.0).
// Modified for NInfer-3090: it uses this tree's cuda_vmm.h helpers, and a mapping WDDM cannot make
// resident (CUDA_ERROR_NOT_READY from cuMemSetAccess) is reported as a refusal, like an
// out-of-memory cuMemCreate.

#include "core/vmm_region.h"

#include "core/cuda_vmm.h"

#include <algorithm>
#include <stdexcept>
#include <string>
#include <utility>

namespace ninfer {
namespace {

constexpr std::size_t kDefaultPieceBytes = 256ULL * 1024ULL * 1024ULL;

// Maps `handle` at `va` and grants every device in `devices` read/write access. Throws when the
// memory cannot be made resident; nothing stays mapped then.
void map_piece(CUdeviceptr va, std::size_t bytes, CUmemGenericAllocationHandle handle,
               const std::vector<int>& devices) {
    vmm::check(cuMemMap(va, bytes, 0, handle, 0), "cuMemMap");
    std::vector<CUmemAccessDesc> access(devices.size());
    for (std::size_t index = 0; index < devices.size(); ++index) {
        access[index].location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        access[index].location.id   = devices[index];
        access[index].flags         = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    }
    const CUresult result = cuMemSetAccess(va, bytes, access.data(), access.size());
    if (result != CUDA_SUCCESS) {
        (void)cuMemUnmap(va, bytes);
        if (vmm::grow_refused(result)) {
            throw std::runtime_error(
                "the device could not make the mapped memory resident (out of memory or the "
                "WDDM budget)");
        }
        vmm::check(result, "cuMemSetAccess");
    }
}

} // namespace

bool VmmRegion::supported(int device) { return vmm::supported(device); }

VmmRegion::VmmRegion(std::size_t bytes, Options options) : options_(std::move(options)) {
    if (bytes == 0) { throw std::invalid_argument("a suspendable region must not be empty"); }
    if (!vmm::supported(options_.device)) {
        throw std::invalid_argument(
            "model suspend requires CUDA virtual memory management on device " +
            std::to_string(options_.device));
    }
    const std::size_t granule = vmm::granularity(options_.device);
    piece_ = options_.piece_bytes == 0 ? kDefaultPieceBytes : options_.piece_bytes;
    piece_ = (piece_ + granule - 1) / granule * granule;
    bytes_ = bytes;
    // The reservation is rounded to whole granules only; the last piece is shortened to it, so a
    // small region does not hold a full piece of physical memory.
    reserved_ = (bytes + granule - 1) / granule * granule;
    access_.push_back(options_.device);
    for (const int peer : options_.peers) {
        if (std::find(access_.begin(), access_.end(), peer) == access_.end()) {
            access_.push_back(peer);
        }
    }
    vmm::check(cuMemAddressReserve(&base_, reserved_, granule, 0, 0), "cuMemAddressReserve");
    try {
        restore_backing();
    } catch (...) {
        (void)cuMemAddressFree(base_, reserved_);
        base_ = 0;
        throw;
    }
}

VmmRegion::~VmmRegion() { reset(); }

VmmRegion::VmmRegion(VmmRegion&& other) noexcept
    : base_(std::exchange(other.base_, 0)), bytes_(std::exchange(other.bytes_, 0)),
      reserved_(std::exchange(other.reserved_, 0)), piece_(std::exchange(other.piece_, 0)),
      options_(std::move(other.options_)), access_(std::move(other.access_)),
      handles_(std::move(other.handles_)), backed_(std::exchange(other.backed_, false)) {}

VmmRegion& VmmRegion::operator=(VmmRegion&& other) noexcept {
    if (this != &other) {
        reset();
        base_     = std::exchange(other.base_, 0);
        bytes_    = std::exchange(other.bytes_, 0);
        reserved_ = std::exchange(other.reserved_, 0);
        piece_    = std::exchange(other.piece_, 0);
        options_  = std::move(other.options_);
        access_   = std::move(other.access_);
        handles_  = std::move(other.handles_);
        backed_   = std::exchange(other.backed_, false);
    }
    return *this;
}

void VmmRegion::reset() noexcept {
    if (base_ == 0) { return; }
    release_pieces();
    (void)cuMemAddressFree(base_, reserved_);
    base_ = 0;
}

DeviceSpan VmmRegion::span() const noexcept {
    return DeviceSpan{reinterpret_cast<void*>(base_), bytes_};
}

void VmmRegion::release_pieces() noexcept {
    std::size_t offset = 0;
    for (const CUmemGenericAllocationHandle handle : handles_) {
        const std::size_t size = std::min(piece_, reserved_ - offset);
        (void)cuMemUnmap(base_ + offset, size);
        (void)cuMemRelease(handle);
        offset += size;
    }
    handles_.clear();
    backed_ = false;
}

void VmmRegion::release_backing() {
    if (!backed_) { return; }
    release_pieces();
}

void VmmRegion::restore_backing() {
    if (backed_) { return; }
    handles_.clear();
    try {
        std::size_t offset = 0;
        while (offset < reserved_) {
            const std::size_t size = std::min(piece_, reserved_ - offset);
            CUmemGenericAllocationHandle handle{};
            const CUmemAllocationProp prop = vmm::device_prop(options_.device);
            const CUresult created         = cuMemCreate(&handle, size, &prop, 0);
            if (created != CUDA_SUCCESS) {
                if (vmm::grow_refused(created)) {
                    throw std::runtime_error(
                        "the device has no free memory to restore a suspended region");
                }
                vmm::check(created, "cuMemCreate");
            }
            try {
                map_piece(base_ + offset, size, handle, access_);
            } catch (...) {
                (void)cuMemRelease(handle);
                throw;
            }
            handles_.push_back(handle);
            offset += size;
        }
    } catch (...) {
        release_pieces();
        throw;
    }
    backed_ = true;
}

} // namespace ninfer
