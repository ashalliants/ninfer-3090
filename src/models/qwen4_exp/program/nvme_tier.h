#pragma once

// The NVMe tier of the Qwen3.8-Flash-Next context cache (docs/maintainer/flash-next-context-cache.md,
// "Storage tiers"): every checkpoint slab is an offset in a scratch file on local disk, and an
// optional pinned RAM cache of frames holds recent slabs in front of it.
//
// A capture copies into RAM frames and a writer thread writes them to the file at once (write-through:
// a frame is evictable as soon as its write lands, so RAM never holds the only copy for long); with
// no frames, or a capture larger than half of them, it goes through a pinned bounce buffer straight to
// the file. A restore gathers the slabs RAM holds from their frames and reads the rest from the file
// through the bounce buffer, two halves in turn, so one half's reads overlap the other's copy to the
// device. A slab whose write failed (a full disk) keeps its frame, so its bytes stay correct; a
// capture or import whose own write fails throws, and the caller drops it (a cache miss).

#include "core/scratch_file.h"
#include "models/qwen4_exp/program/context_cache.h"

#include <cuda_runtime.h>

#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <filesystem>
#include <list>
#include <memory>
#include <mutex>
#include <span>
#include <thread>
#include <vector>

namespace ninfer::models::qwen4_exp {

// The transfers of the tier's backing file (a ScratchFile; tests substitute failures).
class TierFile {
public:
    using Request = ScratchFile::Request;
    virtual ~TierFile()                                  = default;
    virtual void read(std::span<const Request> requests)  = 0;
    virtual void write(std::span<const Request> requests) = 0;
};

// A ScratchFile of `bytes` at `path` (created; removed when closed or the process ends).
[[nodiscard]] std::unique_ptr<TierFile> open_scratch_tier_file(const std::filesystem::path& path, std::uint64_t bytes);

class NvmeTier final : public CheckpointTier {
public:
    struct Options {
        std::uint32_t slabs     = 0; // the file holds slabs * slab_bytes
        std::size_t slab_bytes  = 0; // a multiple of ScratchFile::kAlignment
        std::uint32_t frames    = 0; // pinned RAM frames (one slab each); 0: none
        std::uint32_t bounce_slabs = 0; // slabs per half of the pinned bounce buffer (at least 1)
    };

    struct Stats {
        std::uint64_t slabs_written    = 0; // to the file
        std::uint64_t slabs_read       = 0; // from the file (restores and exports)
        std::uint64_t slabs_from_ram   = 0; // restored from a frame
        std::uint64_t write_failures   = 0; // frame writes that failed (the frame stays resident)
        std::uint32_t frames           = 0;
        std::uint32_t cached           = 0; // frames holding a slab
        std::uint32_t dirty            = 0; // waiting for or in their write
        std::uint32_t stuck            = 0; // whose write failed
    };

    // Throws when the pinned memory cannot be allocated.
    NvmeTier(Options options, std::unique_ptr<TierFile> file);
    ~NvmeTier() override;
    NvmeTier(const NvmeTier&)            = delete;
    NvmeTier& operator=(const NvmeTier&) = delete;

    [[nodiscard]] std::uint32_t slabs() const noexcept override { return options_.slabs; }
    [[nodiscard]] std::size_t slab_bytes() const noexcept override { return options_.slab_bytes; }
    void copy(std::span<const SlabCopy> copies, bool to_tier, cudaStream_t stream) override;
    void read(std::span<const std::uint32_t> slabs, std::span<std::byte> bytes) override;
    void write(std::span<const std::uint32_t> slabs, std::span<const std::byte> bytes) override;
    void discard(std::uint32_t slab) noexcept override;

    [[nodiscard]] Stats stats() const;
    // Waits until no frame is waiting for or in its write.
    void flush();

private:
    enum class State : std::uint8_t { Free, Filling, Dirty, Writing, Clean, Stuck };
    struct Frame {
        std::uint32_t slab = 0;
        State state        = State::Free;
        std::uint64_t generation = 0;
        bool free_after_write    = false;
        std::list<std::uint32_t>::iterator lru; // valid while Clean
    };
    struct Pinned {
        void* p           = nullptr;
        std::size_t bytes = 0;
    };

    std::byte* frame(std::uint32_t f) const noexcept;
    std::byte* half(std::uint32_t h) const noexcept;
    // Under mutex_: a frame for `slab` (its own, a free one, or the least recently used clean one;
    // waits for the writer when every frame is dirty), or -1 when none can be had (every frame
    // stuck), or when `wait` is false and none is free or clean.
    std::int64_t acquire(std::unique_lock<std::mutex>& lock, std::uint32_t slab, bool wait);
    void make_clean(std::uint32_t f);
    void release_frame(std::uint32_t f);
    // Under mutex_: waits until none of `slabs` is being written and drops their frames.
    void forget(std::unique_lock<std::mutex>& lock, std::span<const std::uint32_t> slabs);
    void gather(std::vector<GatherCopy>& copies, Pinned& descriptors, cudaStream_t stream);
    // The file requests that move sorted `slabs` from/to a bounce half (consecutive slabs are
    // consecutive in the half and in the file).
    std::vector<TierFile::Request> half_requests(std::span<const std::uint32_t> slabs, std::uint32_t h) const;
    void direct_capture(std::span<const SlabCopy> copies, std::span<const std::uint32_t> slabs, cudaStream_t stream);
    void writer_loop() noexcept;

    Options options_;
    std::unique_ptr<TierFile> file_;
    Pinned frames_memory_, bounce_;
    Pinned descriptors_[3]; // frames, bounce half 0, bounce half 1
    cudaEvent_t half_done_[2] = {nullptr, nullptr};

    mutable std::mutex mutex_;
    std::condition_variable wake_, progress_;
    std::vector<std::int64_t> frame_of_; // per slab: its frame, -1 when only on disk
    std::vector<Frame> frames_;
    std::vector<std::uint32_t> free_frames_;
    std::list<std::uint32_t> clean_;                                      // least recently used first
    std::deque<std::pair<std::uint32_t, std::uint64_t>> queue_;             // frames to write
    std::vector<std::uint32_t> writing_;                                  // slabs being written
    Stats stats_;
    bool stop_ = false;
    std::thread writer_;
};

} // namespace ninfer::models::qwen4_exp
