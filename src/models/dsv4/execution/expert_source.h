#pragma once

// Written for NInfer-3090.
//
// Where a DeepSeek-V4.1-Flash layer's routed expert records come from, behind one interface so the FFN
// sublayer (ffn.h) is the same code either way:
//
//   ResidentExpertSource  every bank is one block the GPU can read (pinned host memory from the load, or
//                         device memory): the Op reads the records in place, nothing moves.
//   FileExpertSource      banks left in the artifact (plan_load with in_file): after the router the host
//                         reads the routed ids back, preads the records it does not have into a bounded
//                         pinned LRU cache, and hands the Op a table of those records' addresses; the Op's
//                         staging pass copies them into a small device ring. The outputs equal the
//                         resident ones bit for bit (the Op's results do not depend on where a record lives).
//
// Tiers (FileExpertSource): a pinned RAM pool (LRU) above the file and, when device_cache_bytes is set, a
// pool of device frames above that (frequency first, expert_tiers.h). Records in a frame are read by the Op
// in place; a record the policy wants in a frame is copied there from its RAM slot on a copy stream while the
// call's own Op reads it from RAM, and is usable from the next call. The file reads of a call run on a small
// thread pool, in file order. qwen4_exp's expert cache (program/expert_cache, program/ram_slots.h) was not
// reused: it is one global policy with per-layer slot pools, write-back and a prefill stream, all driven by the
// Program's step loop; this source is the same idea at the size one sequence needs, and the Program's tiers can
// replace it without touching the FFN, which sees only begin() and prepare().

#include "artifact/reader.h"
#include "core/arena.h"
#include "models/dsv4/execution/expert_tiers.h"
#include "models/dsv4/model.h"
#include "ninfer/ops/offloaded_sparse_moe.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace ninfer::models::dsv4 {

class ExpertSource {
public:
    virtual ~ExpertSource() = default;

    // The Op's source for layer `layer`'s call whose routed expert ids (I32, `count` of them, entries
    // outside [0, experts) ignored) are at `ids_device`, which the router has already written on
    // `stream`. A source that has to look at the ids synchronizes `stream`; everything it enqueues
    // is ordered before the caller's moe_experts_narrow.
    [[nodiscard]] virtual ops::MoeExpertSource prepare(std::uint32_t layer, const std::int32_t* ids_device,
                                                       std::int32_t count, cudaStream_t stream) = 0;
    // Optional: starts the copy of the ids to the host right after the router, so work enqueued between
    // begin() and prepare() (the shared expert) runs while prepare() reads records from the file. The same
    // arguments as the prepare() that follows it.
    virtual void begin(std::uint32_t /*layer*/, const std::int32_t* /*ids_device*/, std::int32_t /*count*/, cudaStream_t /*stream*/) {}
    // Whether prepare() only enqueues work (no host synchronization), so a decode step can be captured.
    [[nodiscard]] virtual bool graph_capturable() const noexcept = 0;
};

// Banks the GPU reads in place: ExpertBank::device_records must be set (host_records is not used).
class ResidentExpertSource final : public ExpertSource {
public:
    ResidentExpertSource(std::span<const ExpertBank> banks, std::uint32_t experts);
    ~ResidentExpertSource() override;
    ResidentExpertSource(const ResidentExpertSource&)            = delete;
    ResidentExpertSource& operator=(const ResidentExpertSource&) = delete;

    [[nodiscard]] ops::MoeExpertSource prepare(std::uint32_t layer, const std::int32_t* ids_device, std::int32_t count,
                                               cudaStream_t stream) override;
    [[nodiscard]] bool graph_capturable() const noexcept override { return true; }

private:
    std::vector<ExpertBank> banks_;
    std::uint32_t experts_   = 0;
    std::int32_t* no_frames_ = nullptr; // device [E], all -1: every expert is read from the bank
};

struct FileExpertOptions {
    // Pinned RAM for cached records (rounded down to whole slots of the largest layer's record stride).
    std::uint64_t cache_bytes = 4ULL << 30;
    // Device ring the Op stages missing records through; a call with more distinct experts runs in passes.
    std::int32_t staging_slots = 64;
    // The most routed entries (k * T) of one call.
    std::int32_t max_entries = 6 * 2048;
    // Device memory for the VRAM tier: whole frames of the largest layer's record stride; 0 has no VRAM tier.
    std::uint64_t device_cache_bytes = 0;
    // Threads that pread the missing records of a call (the calling thread is one of them): a drive that
    // serves several requests at once (an SSD) wants 8 or more, a hard disk 1.
    // Clamped to 1 on Windows, whose artifact file reads do not run concurrently.
    std::uint32_t read_threads = 1;
    // Misses of a one-column call the Op reads zero-copy over PCIe instead of staging them (the Op's
    // zero_copy_misses); 0 stages every miss.
    std::int32_t zero_copy_misses = 0;
    // At most this many records are promoted to the VRAM tier per call (a prefill chunk routes most of the
    // experts, which would otherwise be copied in whole).
    std::uint32_t max_promotions = 16;
    // The use counts of the VRAM policy are halved every this many calls.
    std::uint32_t age_period = 1U << 14;
};

class FileExpertSource final : public ExpertSource {
public:
    struct Stats {
        std::uint64_t calls = 0, hits = 0, misses = 0, bytes_read = 0; // hits = vram_hits + ram_hits
        std::uint64_t vram_hits = 0, ram_hits = 0, promotions = 0;
        double read_seconds = 0; // host time in preads
        double sync_seconds = 0; // host time waiting for the ids (includes the GPU work queued before them)
    };

    // `banks`: per layer, ExpertBank::artifact_offset / record_stride / format (LoadPlan::file_banks()).
    // `reader` must outlive the source. Throws when the cache holds fewer records than one call needs
    // (min(experts, max_entries) of the largest layer): size cache_bytes for that.
    FileExpertSource(const artifact::Reader& reader, std::span<const ExpertBank> banks, std::uint32_t experts,
                     FileExpertOptions options = {});
    ~FileExpertSource() override;
    FileExpertSource(const FileExpertSource&)            = delete;
    FileExpertSource& operator=(const FileExpertSource&) = delete;

    [[nodiscard]] ops::MoeExpertSource prepare(std::uint32_t layer, const std::int32_t* ids_device, std::int32_t count,
                                               cudaStream_t stream) override;
    void begin(std::uint32_t layer, const std::int32_t* ids_device, std::int32_t count, cudaStream_t stream) override;
    [[nodiscard]] bool graph_capturable() const noexcept override { return false; }

    [[nodiscard]] const Stats& stats() const noexcept { return stats_; }
    void reset_stats() noexcept { stats_ = {}; }
    // Forgets every cached record (a cold cache again).
    void clear_cache();
    [[nodiscard]] std::uint32_t cache_slots() const noexcept { return slots_; }
    // Device bytes this source holds: the frame pool, the record staging ring, the frame table and the record address table.
    [[nodiscard]] std::uint64_t device_bytes() const noexcept {
        return frame_pool_.bytes + staging_.bytes + static_cast<std::uint64_t>(experts_) * (sizeof(std::int32_t) + sizeof(void*));
    }
    [[nodiscard]] std::uint32_t device_frames() const noexcept { return frames_; }

private:
    class ReadPool;
    void release() noexcept; // frees the pinned and device allocations (also on a failed construction)
    void issue_ids_copy(const std::int32_t* ids_device, std::int32_t count, cudaStream_t stream);
    void poll_copies();

    const artifact::Reader& reader_;
    std::vector<ExpertBank> banks_;
    std::uint32_t experts_ = 0;
    FileExpertOptions options_;
    std::uint64_t slot_bytes_   = 0;
    std::uint32_t slots_        = 0;
    std::uint8_t* cache_host_   = nullptr; // pinned, mapped
    std::uint8_t* cache_device_ = nullptr;
    std::uint32_t frames_ = 0;
    ExpertTierPolicy policy_;
    DeviceBuffer frame_pool_;
    DeviceBuffer staging_;
    std::int32_t* frames_device_ = nullptr; // device [E]: the call's frame per expert, -1 for none
    std::int32_t* frames_host_   = nullptr; // pinned [E]
    cudaStream_t copy_stream_    = nullptr; // promotions RAM -> VRAM
    cudaEvent_t ids_event_       = nullptr;
    struct Batch {
        std::uint64_t id;
        cudaEvent_t event;
    };
    std::vector<Batch> batches_;   // in flight, oldest first
    std::vector<cudaEvent_t> free_events_;
    std::uint64_t batch_id_ = 0, waited_batch_ = 0;
    bool ids_pending_       = false;
    std::unique_ptr<ReadPool> pool_;
    const std::uint8_t** table_device_ = nullptr; // device [E]: the call's record addresses
    const std::uint8_t** table_host_   = nullptr; // pinned [E]
    std::int32_t* ids_host_            = nullptr; // pinned [max_entries]
    Stats stats_;
};

} // namespace ninfer::models::dsv4
