// Written for NInfer-3090. See expert_source.h.

#include "models/dsv4/execution/expert_source.h"

#include <algorithm>
#include <chrono>
#include <span>
#include <stdexcept>
#include <string>

namespace ninfer::models::dsv4 {
namespace {

void cuda_require(cudaError_t status, const char* what) {
    if (status != cudaSuccess) { throw std::runtime_error(std::string("dsv4 expert source: ") + what + ": " + cudaGetErrorString(status)); }
}

std::int32_t* device_minus_ones(std::uint32_t count) {
    std::int32_t* p = nullptr;
    cuda_require(cudaMalloc(&p, static_cast<std::size_t>(count) * sizeof(std::int32_t)), "cudaMalloc");
    cuda_require(cudaMemset(p, 0xFF, static_cast<std::size_t>(count) * sizeof(std::int32_t)), "cudaMemset");
    return p;
}

using Clock = std::chrono::steady_clock;
double seconds_since(Clock::time_point t) { return std::chrono::duration<double>(Clock::now() - t).count(); }

} // namespace

ResidentExpertSource::ResidentExpertSource(std::span<const ExpertBank> banks, std::uint32_t experts)
    : banks_(banks.begin(), banks.end()), experts_(experts) {
    for (const auto& bank : banks_) {
        if (bank.device_records == nullptr) { throw std::invalid_argument("ResidentExpertSource: a bank has no device_records"); }
    }
    no_frames_ = device_minus_ones(experts_);
}

ResidentExpertSource::~ResidentExpertSource() { cudaFree(no_frames_); }

ops::MoeExpertSource ResidentExpertSource::prepare(std::uint32_t layer, const std::int32_t*, std::int32_t, cudaStream_t) {
    const ExpertBank& bank = banks_.at(layer);
    ops::MoeExpertSource source;
    source.format        = bank.format;
    source.frame_base    = bank.device_records; // never read: no expert has a frame
    source.frames        = no_frames_;
    source.host_records  = bank.device_records;
    source.record_stride = bank.record_stride;
    return source; // staging_slots == 0: records are read in place
}

FileExpertSource::FileExpertSource(const artifact::Reader& reader, std::span<const ExpertBank> banks, std::uint32_t experts,
                                   FileExpertOptions options)
    : reader_(reader), banks_(banks.begin(), banks.end()), experts_(experts), options_(options) {
    for (const auto& bank : banks_) {
        if (bank.record_stride == 0 || bank.experts != experts_) { throw std::invalid_argument("FileExpertSource: a bank is not a file bank"); }
        slot_bytes_ = std::max<std::uint64_t>(slot_bytes_, bank.record_stride);
    }
    if (slot_bytes_ == 0 || options_.staging_slots <= 0 || options_.max_entries <= 0) { throw std::invalid_argument("FileExpertSource: bad options"); }
    slots_ = static_cast<std::uint32_t>(options_.cache_bytes / slot_bytes_);
    const std::uint32_t need = std::min<std::uint32_t>(experts_, static_cast<std::uint32_t>(options_.max_entries));
    if (slots_ < need) {
        throw std::invalid_argument("FileExpertSource: cache_bytes holds " + std::to_string(slots_) + " records of " + std::to_string(slot_bytes_) +
                                    " bytes; one call can route " + std::to_string(need) + " distinct experts");
    }
    cuda_require(cudaHostAlloc(reinterpret_cast<void**>(&cache_host_), static_cast<std::size_t>(slots_) * slot_bytes_,
                               cudaHostAllocMapped | cudaHostAllocPortable),
                 "cudaHostAlloc (record cache)");
    void* device = nullptr;
    cuda_require(cudaHostGetDevicePointer(&device, cache_host_, 0), "cudaHostGetDevicePointer");
    cache_device_ = static_cast<std::uint8_t*>(device);
    slot_.assign(slots_, Slot{});
    slot_of_.assign(static_cast<std::size_t>(banks_.size()) * experts_, -1);
    staging_   = DeviceBuffer(static_cast<std::size_t>(options_.staging_slots) * slot_bytes_);
    no_frames_ = device_minus_ones(experts_);
    cuda_require(cudaMalloc(reinterpret_cast<void**>(&table_device_), experts_ * sizeof(void*)), "cudaMalloc");
    cuda_require(cudaHostAlloc(reinterpret_cast<void**>(&table_host_), experts_ * sizeof(void*), cudaHostAllocPortable), "cudaHostAlloc");
    cuda_require(cudaHostAlloc(reinterpret_cast<void**>(&ids_host_), static_cast<std::size_t>(options_.max_entries) * sizeof(std::int32_t),
                               cudaHostAllocPortable),
                 "cudaHostAlloc");
}

FileExpertSource::~FileExpertSource() {
    cudaFree(no_frames_);
    cudaFree(table_device_);
    cudaFreeHost(table_host_);
    cudaFreeHost(ids_host_);
    cudaFreeHost(cache_host_);
}

void FileExpertSource::clear_cache() {
    slot_.assign(slots_, Slot{});
    slot_of_.assign(slot_of_.size(), -1);
}

ops::MoeExpertSource FileExpertSource::prepare(std::uint32_t layer, const std::int32_t* ids_device, std::int32_t count,
                                               cudaStream_t stream) {
    if (count <= 0 || count > options_.max_entries) { throw std::invalid_argument("FileExpertSource: entry count outside max_entries"); }
    const ExpertBank& bank = banks_.at(layer);
    ++call_;
    ++stats_.calls;

    // The routed ids: one host synchronization (it also waits for everything enqueued before them).
    auto t0 = Clock::now();
    cuda_require(cudaMemcpyAsync(ids_host_, ids_device, static_cast<std::size_t>(count) * sizeof(std::int32_t), cudaMemcpyDeviceToHost, stream), "ids copy");
    cuda_require(cudaStreamSynchronize(stream), "ids sync");
    stats_.sync_seconds += seconds_since(t0);

    std::vector<std::uint8_t> routed(experts_, 0);
    for (std::int32_t i = 0; i < count; ++i) {
        const std::int32_t e = ids_host_[i];
        if (e >= 0 && static_cast<std::uint32_t>(e) < experts_) { routed[static_cast<std::size_t>(e)] = 1; }
    }

    // Hits first (so a miss never evicts a record this call needs), then the misses.
    const std::size_t base = static_cast<std::size_t>(layer) * experts_;
    std::vector<std::uint32_t> missing;
    for (std::uint32_t e = 0; e < experts_; ++e) {
        if (routed[e] == 0) { continue; }
        const std::int32_t s = slot_of_[base + e];
        if (s >= 0) {
            slot_[static_cast<std::size_t>(s)].last_used = call_;
            ++stats_.hits;
        } else {
            missing.push_back(e);
        }
    }
    t0 = Clock::now();
    for (const std::uint32_t e : missing) {
        std::int32_t victim = -1;
        for (std::uint32_t s = 0; s < slots_; ++s) {
            const Slot& slot = slot_[s];
            if (slot.key < 0) {
                victim = static_cast<std::int32_t>(s);
                break;
            }
            if (slot.last_used != call_ && (victim < 0 || slot.last_used < slot_[static_cast<std::size_t>(victim)].last_used)) {
                victim = static_cast<std::int32_t>(s);
            }
        }
        if (victim < 0) { throw std::runtime_error("FileExpertSource: no free cache slot (more routed experts than slots)"); }
        Slot& slot = slot_[static_cast<std::size_t>(victim)];
        if (slot.key >= 0) { slot_of_[static_cast<std::size_t>(slot.key)] = -1; }
        slot.key          = -1; // not valid until the read lands
        auto* destination = reinterpret_cast<std::byte*>(cache_host_ + static_cast<std::uint64_t>(victim) * slot_bytes_);
        reader_.read_into(bank.artifact_offset + static_cast<std::uint64_t>(e) * bank.record_stride,
                          std::span<std::byte>(destination, static_cast<std::size_t>(bank.record_stride)));
        slot.key           = static_cast<std::int64_t>(base + e);
        slot.last_used     = call_;
        slot_of_[base + e] = victim;
        ++stats_.misses;
        stats_.bytes_read += bank.record_stride;
    }
    stats_.read_seconds += seconds_since(t0);

    // The address table of this call's records (the Op reads an entry only for a routed expert).
    for (std::uint32_t e = 0; e < experts_; ++e) {
        table_host_[e] = routed[e] != 0 ? cache_device_ + static_cast<std::uint64_t>(slot_of_[base + e]) * slot_bytes_ : nullptr;
    }
    cuda_require(cudaMemcpyAsync(table_device_, table_host_, experts_ * sizeof(void*), cudaMemcpyHostToDevice, stream), "table upload");

    ops::MoeExpertSource source;
    source.format        = bank.format;
    source.frame_base    = static_cast<const std::uint8_t*>(staging_.p); // never read: no expert has a frame
    source.frames        = no_frames_;
    source.record_stride = bank.record_stride;
    source.staging_base  = static_cast<std::uint8_t*>(staging_.p);
    source.staging_slots = options_.staging_slots;
    ops::set_host_slots(source, table_device_);
    return source;
}

} // namespace ninfer::models::dsv4
