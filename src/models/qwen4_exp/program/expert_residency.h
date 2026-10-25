#pragma once

// Adapted from Infernix a3edb450 src/models/qwen4_exp/program/expert_residency.h (Apache-2.0).
// Modified for NInfer-3090: frames hold ggml_expert_record_v1 records of each layer's bank (the
// stride is the largest bank's); no SSD tier or landing frames (their Op and Program paths are not
// ported); frame lending for prefill, whose give-back restores the displaced experts; a grow
// refills its new frames by score; uploads use core/pinned_upload.h.
//
// The VRAM expert cache of the Qwen4Exp Program: a pool of device frames, each holding one routed
// expert's record, the per-layer frame tables the MoE kernels read (frame or -1: stage the record
// from the pinned bank), the route log of every round, and the LFRU policy
// (expert_cache::CacheController) that decides promotions and evictions at round boundaries.
//
// Ordering. Rounds run on the compute stream and the Program synchronizes at the end of each. At
// the boundary the policy's evictions are applied to the device tables on the compute stream
// before the next round, and the promotions' copies run on a copy stream that first waits for that
// table update, so no kernel can read a frame while it is overwritten. A promoted expert becomes
// visible in its table only after its copy has completed. Expert outputs are placement-invariant
// (ninfer/ops/offloaded_sparse_moe.h), so the round a promotion lands in does not change any result.
//
// Size. The frames live in a VMM arena whose base never moves: resize() backs more frames chunk by
// chunk, or releases the top chunks after evicting the lowest-score experts and moving the survivors
// below the new top, so kernels and captured graphs keep their addresses. Without VMM the frames are
// one allocation of the first size.

#include "core/arena.h"
#include "core/device.h"
#include "core/vmm_arena.h"
#include "core/vram_budget.h"
#include "models/qwen4_exp/config.h"
#include "models/qwen4_exp/program/expert_cache/expert_cache.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace ninfer::models::qwen4_exp {

class ExpertResidency {
public:
    struct Stats {
        std::uint64_t routed     = 0; // routed (layer, expert) uses, summed over rounds' groups
        std::uint64_t hits       = 0; // of those, resident in a frame at the round's start
        std::uint64_t promotions = 0; // expert copies issued
        std::uint64_t seeded     = 0; // experts loaded by the warm start
    };

    using SavedState = expert_cache::SavedState;
    [[nodiscard]] SavedState saved_state() const;
    // Every key's LFRU use count (U32 [layers][experts]; zeros without a policy): the prior the
    // expert stream's CPU share ranks a layer's misses by before the layer routes.
    void use_counts(std::vector<std::uint32_t>& out) const;
    // Before the first round: seeds the counts (each capped at `count_cap`, so a stale expert
    // yields to new uses) and loads at most `max_keys` of the ranked keys, best first, into free
    // frames, waiting for the copies. Returns the number loaded.
    std::uint32_t warm_start(const SavedState& state, std::uint32_t count_cap, std::uint32_t max_keys,
                             cudaStream_t compute);

    // banks[l]: layer l's pinned host records (`strides[l]` bytes apart); a frame holds the largest.
    // max_columns: the most columns one round routes. The cache starts with no frames: resize()
    // backs them.
    ExpertResidency(const TextConfig& config, std::vector<const std::uint8_t*> banks,
                    std::vector<std::uint64_t> strides, std::int32_t max_columns, int device);
    ~ExpertResidency();
    ExpertResidency(const ExpertResidency&)            = delete;
    ExpertResidency& operator=(const ExpertResidency&) = delete;

    // Device bytes besides the frames: the frame tables and the route log.
    [[nodiscard]] static std::uint64_t table_bytes(const TextConfig& config, std::int32_t max_columns) noexcept;
    // Frames the cache uses at most: every routed expert but one.
    [[nodiscard]] static std::uint32_t max_frames(const TextConfig& config) noexcept;
    // Bytes the frames are mapped in (whole chunks); a grow or shrink moves in these steps.
    static constexpr std::size_t kChunkBytes = 64ULL << 20;

    struct Resize {
        std::uint32_t frames  = 0;     // frames backed afterwards
        std::uint64_t spilled = 0;     // bytes of a chunk the driver placed in system memory (given back)
        bool refused          = false; // the device had no memory for the next chunk
        std::uint32_t refilled = 0;     // a grow: experts loaded into the new frames
        VramSnapshot at_spill;         // the reading after a spilled chunk (diagnostics)
    };
    // Backs `frames` frames (clamped to frame_limit()). Call only between rounds, with nothing in
    // flight on the compute stream: promotion copies are waited for, then a shrink evicts the
    // lowest-score experts, moves the survivors above the new top below it on `compute` and
    // releases the top chunks; a grow maps chunks one at a time, stops at the first the driver
    // refuses or places in system memory (`vram` tells), and fills the new frames from the queue.
    // Without VMM only the first call allocates.
    Resize resize(std::uint32_t frames, cudaStream_t compute, VramBudgetSource& vram);

    // Whether the pool can resize after its first size (VMM).
    [[nodiscard]] bool elastic() const noexcept { return arena_ != nullptr; }
    [[nodiscard]] std::uint32_t frame_limit() const noexcept { return limit_; }
    // Bytes the frames occupy now (whole chunks with VMM).
    [[nodiscard]] std::uint64_t pool_bytes() const noexcept;

    [[nodiscard]] const std::uint8_t* frame_base() const noexcept;
    [[nodiscard]] std::uint64_t frame_stride() const noexcept { return stride_; }
    [[nodiscard]] const std::int32_t* table(std::uint32_t layer) const noexcept;
    // The host copy of every layer's table (I32 [layers][experts]), equal to the device tables
    // once before_round has uploaded them.
    [[nodiscard]] const std::int32_t* host_table() const noexcept {
        return static_cast<const std::int32_t*>(table_host_.data());
    }
    [[nodiscard]] std::int32_t* route_log() noexcept { return static_cast<std::int32_t*>(route_device_.p); }
    [[nodiscard]] std::size_t route_stride() const noexcept { return route_stride_; }
    [[nodiscard]] std::uint32_t frames() const noexcept { return frames_; }

    // Before a round: publishes the loads that completed since the last round and uploads the table
    // when it changed.
    void before_round(cudaStream_t compute);
    // After the round's kernels are enqueued: downloads its route log with the round.
    void enqueue_route_download(cudaStream_t compute, std::int32_t columns);
    // After the round completed (the compute stream was synchronized): runs the policy over the
    // round's routes and issues its evictions and promotions. `per_layer_budget` caps the
    // promotions one layer's routes may start.
    void after_round(cudaStream_t compute, std::int32_t columns, std::size_t per_layer_budget);
    // Waits for the promotion copies in flight and publishes them.
    void drain(cudaStream_t compute);

    // Frame lending (Infernix design section 19.3.2): a contiguous run of frames serves prefill (the
    // expert stream's ring and the wide calls' workspace) and comes back. Lent frames hold no expert
    // and the pool does not resize while a lease is out.
    struct FrameLease {
        std::uint32_t first = 0, count = 0;
        DeviceSpan memory; // the run's bytes
        [[nodiscard]] bool valid() const noexcept { return count != 0; }
    };
    // Frames that may be lent now: backed and above the quarter serving always keeps.
    [[nodiscard]] std::uint32_t lendable() const noexcept;
    // Between rounds (nothing in flight on `compute`): waits for the promotion copies, evicts the
    // experts of the cheapest run of `count` frames and uploads the table on `compute`.
    FrameLease lend(std::uint32_t count, cudaStream_t compute);
    // Returns a lease once `compute` has finished every access to it, then refills the free frames
    // on the copy stream behind `compute`: first with the experts the lease displaced (the set
    // before the prompt), then with the highest-score experts that are not resident. Returns the
    // number of refill loads.
    std::uint32_t give_back(FrameLease& lease, cudaStream_t compute);
    [[nodiscard]] std::uint32_t lent_frames() const noexcept;

    [[nodiscard]] const Stats& stats() const noexcept { return stats_; }

private:
    // After the controller took a run back (commands_ holds the loads that gave back queued): reloads
    // displaced_[displaced_from..] and issues every queued load. The frames are already returned when
    // this runs, so a throw from it leaves them returned.
    std::uint32_t settle_return(cudaStream_t compute, std::size_t displaced_from);
    void upload_table(cudaStream_t compute);
    // Publishes the promotions whose copies have completed.
    void publish_landed();
    // Issues commands_' promotion copies on the copy stream, after the table update on `compute`.
    void issue_loads(cudaStream_t compute);
    // Loads non-resident experts into the free frames: those of `first`, then the others with
    // uses, each group highest LFRU score first. Returns the number loaded.
    std::uint32_t refill(cudaStream_t compute, std::span<const std::uint32_t> first);

    const TextConfig& c_;
    std::vector<const std::uint8_t*> banks_;
    std::vector<std::uint64_t> strides_; // per layer: the bank's record stride
    std::uint64_t stride_ = 0;           // one frame: the largest record stride
    std::uint32_t frames_ = 0;
    std::uint32_t limit_  = 0;
    std::uint32_t experts_ = 0, layers_ = 0, top_k_ = 0;
    std::size_t route_stride_ = 0;

    std::unique_ptr<VmmArena> arena_;
    DeviceBuffer frame_memory_; // without VMM
    DeviceBuffer table_device_;
    PinnedHostBuffer table_host_{1};
    DeviceBuffer route_device_;
    PinnedHostBuffer route_host_{1};
    bool table_dirty_ = true;

    std::unique_ptr<expert_cache::CacheController> controller_;
    std::vector<expert_cache::CacheController::Command> commands_;
    std::vector<std::uint32_t> group_;
    std::vector<std::uint32_t> displaced_; // the experts the current lease evicted
    std::vector<std::uint8_t> seen_;
    std::uint64_t round_ = 0;

    cudaStream_t copy_stream_ = nullptr;
    cudaEvent_t table_ready_  = nullptr;
    struct Batch {
        cudaEvent_t done = nullptr;
        struct Load {
            std::uint32_t key, frame;
            std::uint64_t serial;
        };
        std::vector<Load> loads;
    };
    std::vector<Batch> in_flight_;
    std::vector<cudaEvent_t> spare_events_;
    // Promotions per completion event (~3.5 ms of link at Gen4 x16): a large promotion publishes in
    // steps.
    static constexpr std::size_t kLoadsPerEvent = 64;
    Stats stats_;
};

} // namespace ninfer::models::qwen4_exp
