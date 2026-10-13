#pragma once

// Adapted from Infernix a3edb450 src/models/qwen4_exp/memory_plan.h (Apache-2.0).
// Modified for NInfer-3090: full mode only (every expert pinned in RAM); the SSD tier is not ported, so a ledger that cannot hold every expert refuses to start.

// The startup memory plans of Qwen3.8-Flash-Next (docs/maintainer/qwen3_8-flash-next-design.md
// §19.3.7). Host RAM: what the experts may take once a reserve stays free for the system and every
// other allocation the engine will make is planned; one reserve covers every pinned allocation.
// Device memory: what the expert frames may take once the fixed allocations, a reserve for
// NInfer's own later allocations and a headroom for the display and other programs are left.

#include "core/host_memory.h"
#include "core/vram_budget.h"
#include "ninfer/types.h"

#include <cstdint>
#include <optional>
#include <string>

namespace ninfer::models::qwen4_exp {

inline constexpr std::uint64_t kLedgerGiB = 1ULL << 30;
// Physical Host memory left free for the OS and other programs when no reserve is given.
inline constexpr std::uint64_t kDefaultRamHeadroomBytes = 2ULL << 30;

struct HostMemoryDemand {
    // Physical memory left free for the OS and other programs (EngineOptions::ram_headroom_bytes).
    std::uint64_t reserve = kDefaultRamHeadroomBytes;
    // Pinned expert banks (all layers).
    std::uint64_t expert_banks = 0;
    // The model's other pinned weights.
    std::uint64_t other_pinned = 0;
    // Pinned and mapped allocations made after the model loads (the Program's buffers).
    std::uint64_t later_pinned = 0;
    // The context cache's pinned Host tier (fit_context_cache sizes it).
    std::uint64_t context_cache = 0;
    // Large pageable allocations made after planning (the n-gram row cache).
    std::uint64_t pageable = 0;
    // Transient load staging (bounce and upload slots), freed when the load ends.
    std::uint64_t load_staging = 0;
    // The process's pageable growth after planning.
    std::uint64_t process_growth = kLedgerGiB;
    // Page-frame and lock bookkeeping, as a fraction of every pinned byte.
    double lock_overhead = 0.0025;
    // Beyond the banks, so the system is not run at the reserve's edge.
    std::uint64_t full_margin = kLedgerGiB;
};

struct HostMemoryLedger {
    HostMemorySnapshot snapshot;
    HostMemoryDemand demand;
    // Everything but the experts: other and later pins with their lock overhead, pageable caches,
    // staging and growth.
    std::uint64_t planned = 0;
    // What the experts may take (pinned bytes including their lock overhead), never more than
    // they need.
    std::uint64_t expert_ram = 0;
    // The experts' need: the banks with their lock overhead plus the margin.
    std::uint64_t full_need = 0;
    // The commit limit, not physical memory, bounded expert_ram (a small page file).
    bool commit_limited = false;
    // Every pinned byte (banks, other and later pins), compared with the Linux memory-lock limit.
    std::uint64_t pinned_total = 0;
    // The Linux RLIMIT_MEMLOCK is below pinned_total, so pinning would fail whatever is free.
    bool lock_limited = false;
    // Every expert fits in RAM. Without the SSD tier (not ported) startup refuses otherwise.
    bool fits = false;

    // One log line: the inputs and the outcome.
    [[nodiscard]] std::string describe() const;
    // One plain line for the console: free RAM and whether the experts fit.
    [[nodiscard]] std::string summary() const;
};

[[nodiscard]] HostMemoryLedger plan_host_memory(const HostMemorySnapshot& snapshot, const HostMemoryDemand& demand);

// The context cache's Host tier beside everything else the ledger plans: `requested` bytes when they
// fit with the experts (their full need and margin) above the reserve, else the most that fits in
// whole `granule`s; zero below `minimum` (too small to hold one long checkpoint) or when the experts
// do not fit at all. `demand.context_cache` is ignored (it is what this decides).
struct ContextCacheFit {
    std::uint64_t bytes = 0;
    bool shrunk         = false; // less than requested
};
[[nodiscard]] ContextCacheFit fit_context_cache(const HostMemorySnapshot& snapshot, HostMemoryDemand demand,
                                                std::uint64_t requested, std::uint64_t minimum,
                                                std::uint64_t granule);

// ---------------------------------------------------------------- device memory

inline constexpr std::uint64_t kLedgerMiB = 1ULL << 20;
// The margin kept below the OS budget (Windows WDDM). The display headroom applies to device free
// memory only, so a budget below free memory is not stacked on it.
inline constexpr std::uint64_t kVramBudgetMarginBytes = 64 * kLedgerMiB;
// Device memory one CUDA graph executable may take once instantiated (graphs are captured lazily,
// after sizing).
inline constexpr std::uint64_t kGraphExecutableBytes = 4 * kLedgerMiB;

struct VramDemand {
    // --vram-headroom-mib; empty selects it from the display state.
    std::optional<std::uint64_t> headroom;
    // The frame pool shrinks at runtime when the display or another program needs memory. Without
    // that nothing absorbs their growth, so the automatic headroom and the graph reserve are larger.
    bool elastic = false;
    // CUDA graph executables the Program may instantiate after sizing, and what each may take.
    std::uint32_t graphs      = 0;
    std::uint64_t graph_bytes = kGraphExecutableBytes;
    // Device allocations still to be made besides the frames (taken from free memory and budget).
    std::uint64_t fixed = 0;
    // One expert frame, and the most frames the residency uses.
    std::uint64_t frame_bytes = 0;
    std::uint32_t max_frames  = 0;
};

struct VramSizing {
    VramSnapshot snapshot;
    VramDemand demand;
    std::uint64_t headroom = 0; // left free for the display and other programs
    std::uint64_t reserve  = 0; // for NInfer's own allocations after sizing (graph executables)
    // What the frames may take; negative when the fixed allocations alone do not fit.
    std::int64_t frame_bytes = 0;
    bool budget_limited      = false; // the OS budget, not free memory, bounded frame_bytes
    std::uint32_t frames     = 0;

    // One log line: the inputs and the outcome.
    [[nodiscard]] std::string describe() const;
};

// CUDA graph executables a Program instantiates at most: one decode graph per batch size, and with
// speculation a verification graph per batch size and width, and the MTP drafter's catch-up (per
// batch size and width) and draft graphs (per batch size and step). The count is per decode forward:
// a caller that splits one decode into several graphs (the front/back split) multiplies it by the
// number of parts, as the qwen4_exp Program's sizing does (2 * graph_bound(1, 1, 0)).
[[nodiscard]] std::uint32_t graph_bound(std::uint32_t lanes, std::uint32_t max_width, std::uint32_t mtp_draft_tokens);
// The headroom left for the display and other programs: the requested one, else 256 MiB without a
// display and, with one (or when unknown), 512 MiB for an elastic pool or 1 GiB for a fixed one.
[[nodiscard]] std::uint64_t display_headroom(DisplayState display, bool elastic, std::optional<std::uint64_t> requested);
// The reserve for graph executables instantiated after sizing: max(256 MiB, 128 MiB + one
// allowance per graph), the graph count capped at 32 for an elastic pool, which absorbs the rest.
[[nodiscard]] std::uint64_t internal_reserve(std::uint32_t graphs, std::uint64_t graph_bytes, bool elastic);
// frame_bytes = min(F - fixed - headroom, B - fixed - 64 MiB) - reserve, with F the device free
// memory and B the OS budget less this process's usage (no budget: F's bound alone); frames =
// clamp(frame_bytes / frame size, 0, max_frames).
[[nodiscard]] VramSizing size_expert_frames(const VramSnapshot& snapshot, const VramDemand& demand);

// The elastic KV pool's growth (design §19.3.11): the expert frames to release before the pool maps
// `added` bytes, so the growth never makes the device hold more than it did. `spare` is the memory
// the sizing function would give the frames anyway (free memory above the headroom and reserve);
// the rest comes out of the frames, one frame chunk over, because the frame pool releases whole
// chunks of `chunk_bytes` and keeps a partly used one: releasing f frames frees at least
// f * frame_bytes - chunk_bytes. Clamped to `frames`.
[[nodiscard]] std::uint32_t kv_growth_frames(std::uint64_t added, std::uint64_t spare, std::uint32_t frames,
                                             std::uint64_t frame_bytes, std::uint64_t chunk_bytes) noexcept;

// The runtime law of an elastic frame pool (design §19.3.7): from the latest snapshot, the pool's
// size and the time, the frame count it should have. Pure and single-threaded; the Program guards
// it. The reserve is spent only by NInfer's own allocations after sizing (graph executables): with
// an OS budget, the growth of this process's usage other than the pool's.
class VramControl {
public:
    // `demand` as sized at startup (elastic); `usage` and `pool_bytes` at that moment.
    VramControl(const VramDemand& demand, std::uint64_t chunk_bytes, std::uint64_t usage, std::uint64_t pool_bytes,
                double grow_delay_seconds = 30.0);

    struct Decision {
        std::uint32_t frames = 0; // the frame count to resize to (the current one: no change)
        bool shrink          = false;
        bool grow            = false;
        bool pressure        = false; // free memory below half the headroom, or the budget exceeded
    };
    // Shrinks at once when free memory falls below half the headroom or this process is over its
    // OS budget, to one chunk below the sizing function's target; grows toward the target once it
    // has stayed two chunks above the pool for the grow delay, by at most four chunks a boundary
    // while rounds run and all at once when `idle`.
    [[nodiscard]] Decision decide(const VramSnapshot& snapshot, std::uint64_t pool_bytes, std::uint32_t frames,
                                  double now_seconds, bool idle);
    // The frame count the sizing function gives now (no hysteresis).
    [[nodiscard]] std::uint32_t target(const VramSnapshot& snapshot, std::uint64_t pool_bytes) const;
    [[nodiscard]] std::uint64_t headroom(DisplayState display) const;
    [[nodiscard]] std::uint64_t reserve_left(const VramSnapshot& snapshot, std::uint64_t pool_bytes) const;
    // A fixed allocation grew (positive) or shrank by `bytes` since sizing (the elastic KV pool, design
    // §19.3.11): it is not the reserve's use.
    void account_fixed(std::int64_t bytes) noexcept {
        usage_at_sizing_ = static_cast<std::uint64_t>(static_cast<std::int64_t>(usage_at_sizing_) + bytes);
    }

private:
    [[nodiscard]] std::int64_t frame_bytes(const VramSnapshot& snapshot, std::uint64_t pool_bytes) const;

    VramDemand demand_;
    std::uint64_t chunk_;
    std::uint64_t usage_at_sizing_;
    std::uint64_t pool_at_sizing_;
    double grow_delay_;
    double grow_since_ = -1.0; // when the target first stood two chunks above the pool; < 0: not now
};

} // namespace ninfer::models::qwen4_exp
