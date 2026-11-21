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
// qwen4_exp's expert cache and RAM slots (program/expert_cache, program/ram_slots.h) were not reused: they
// are one global LFRU policy over layer x expert keys with per-layer slot pools, VRAM frames with promotion
// and write-back, and a prefill stream, all driven by the Program's step loop. This source needs a single
// LRU over a flat pinned pool and nothing else, and it can be replaced by the Program's tiers later
// without touching the FFN, which sees only prepare().

#include "artifact/reader.h"
#include "core/arena.h"
#include "models/dsv4/model.h"
#include "ninfer/ops/offloaded_sparse_moe.h"

#include <cuda_runtime.h>

#include <cstdint>
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
};

class FileExpertSource final : public ExpertSource {
public:
    struct Stats {
        std::uint64_t calls = 0, hits = 0, misses = 0, bytes_read = 0;
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
    [[nodiscard]] bool graph_capturable() const noexcept override { return false; }

    [[nodiscard]] const Stats& stats() const noexcept { return stats_; }
    void reset_stats() noexcept { stats_ = {}; }
    // Forgets every cached record (a cold cache again).
    void clear_cache();
    [[nodiscard]] std::uint32_t cache_slots() const noexcept { return slots_; }
    // Device bytes this source holds: the record staging ring, the no-frame table and the record address table.
    [[nodiscard]] std::uint64_t device_bytes() const noexcept {
        return staging_.bytes + static_cast<std::uint64_t>(experts_) * (sizeof(std::int32_t) + sizeof(void*));
    }

private:
    struct Slot {
        std::int64_t key        = -1; // layer * experts + expert, or -1 when free
        std::uint64_t last_used = 0;  // call number that last used it
    };

    void release() noexcept; // frees the pinned and device allocations (also on a failed construction)

    const artifact::Reader& reader_;
    std::vector<ExpertBank> banks_;
    std::uint32_t experts_ = 0;
    FileExpertOptions options_;
    std::uint64_t slot_bytes_   = 0;
    std::uint32_t slots_        = 0;
    std::uint8_t* cache_host_   = nullptr; // pinned, mapped
    std::uint8_t* cache_device_ = nullptr;
    std::vector<Slot> slot_;
    std::vector<std::int32_t> slot_of_; // per key: its slot or -1
    std::uint64_t call_ = 0;
    DeviceBuffer staging_;
    std::int32_t* no_frames_           = nullptr; // device [E], all -1
    const std::uint8_t** table_device_ = nullptr; // device [E]: the call's record addresses
    const std::uint8_t** table_host_   = nullptr; // pinned [E]
    std::int32_t* ids_host_            = nullptr; // pinned [max_entries]
    Stats stats_;
};

} // namespace ninfer::models::dsv4
