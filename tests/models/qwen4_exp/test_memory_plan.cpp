// Adapted from Infernix a3edb450 tests/models/qwen4_exp/test_memory_plan.cpp (Apache-2.0).
// Modified for NInfer-3090: the RAM ledger is full mode only (fits or refuses; no SSD tier or
// --expert-ram-mib), with this fork's 33,812 MiB of IQ2 expert banks; the VRAM tests are unchanged.

// The Qwen3.8-Flash-Next startup memory plans (design §19.3.7). RT1, the RAM ledger: the experts
// take what the reserve and every planned allocation leave, full mode needs the banks plus the
// margin, the commit limit bounds the experts like physical memory, and the reserve is applied
// once. RT7, the VRAM sizing: the display headroom by display state, the OS budget without
// stacking the headroom on it, the graph reserve, fixed allocations that do not fit, the frame
// clamp, and the spill test with noisy free-memory readings. RT9, the runtime law of the elastic
// pool: shrink on low free memory or an exceeded budget, hysteresis, the grow delay and per-boundary
// cap, display hot-plug, and the reserve spent by the engine's own later allocations. The elastic
// KV pool's growth: the frames it takes free at least what it maps beyond the spare memory.

#include "models/qwen4_exp/memory_plan.h"

#include <algorithm>
#include <cstdio>
#include <deque>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::models::qwen4_exp;

namespace {

constexpr std::uint64_t kMiB = 1ULL << 20;
constexpr std::uint64_t kGiB = 1ULL << 30;

void require(bool value, const std::string& message) {
    if (!value) { throw std::runtime_error(message); }
}

// This fork's model: the 33,812 MiB of IQ2 expert banks (every other weight is on the device).
HostMemoryDemand flash_next() {
    HostMemoryDemand d;
    d.reserve      = 2 * kGiB;
    d.expert_banks = 33'812 * kMiB;
    d.later_pinned = 256 * kMiB;
    d.pageable     = 90 * kMiB;
    d.load_staging = 264 * kMiB;
    return d;
}

HostMemorySnapshot snapshot(std::uint64_t available, std::uint64_t commit) {
    HostMemorySnapshot s;
    s.total_physical     = 64 * kGiB;
    s.available_physical = available;
    s.available_commit   = commit;
    return s;
}

std::uint64_t need_for_full(const HostMemoryDemand& d) {
    const HostMemoryLedger probe = plan_host_memory(snapshot(0, 0), d);
    return d.reserve + probe.planned + probe.full_need;
}

void test_fit_boundary() {
    const HostMemoryDemand d = flash_next();
    const std::uint64_t need = need_for_full(d);
    const HostMemoryLedger at = plan_host_memory(snapshot(need, 200 * kGiB), d);
    require(at.fits && at.expert_ram == at.full_need, "exactly enough RAM holds every expert");
    const HostMemoryLedger below = plan_host_memory(snapshot(need - 1, 200 * kGiB), d);
    require(!below.fits && below.expert_ram == at.full_need - 1 && !below.commit_limited,
            "one byte less does not fit, and the refusal is not blamed on the page file");
    require(at.full_need > d.expert_banks + d.full_margin, "the banks' lock overhead is counted");
    require(at.full_need - d.full_margin - d.expert_banks == (d.expert_banks * 25 + 9'999) / 10'000,
            "lock overhead is 0.25 % of the banks, rounded up");
    // This box with the model's measured neighbourhood: ~44 GiB available before loading holds
    // every expert at the 2 GiB default, ~36 GiB does not.
    require(plan_host_memory(snapshot(44 * kGiB, 120 * kGiB), d).fits, "44 GiB available holds every expert");
    require(!plan_host_memory(snapshot(36 * kGiB, 120 * kGiB), d).fits, "36 GiB available is refused");
}

void test_reserve_applied_once() {
    HostMemoryDemand d = flash_next();
    const std::uint64_t available = 30 * kGiB;
    const HostMemoryLedger a      = plan_host_memory(snapshot(available, 200 * kGiB), d);
    d.reserve += 3 * kGiB;
    const HostMemoryLedger b = plan_host_memory(snapshot(available, 200 * kGiB), d);
    require(a.expert_ram - b.expert_ram == 3 * kGiB, "raising the reserve by 3 GiB takes 3 GiB from the experts");
    require(a.expert_ram == available - flash_next().reserve - a.planned, "the experts take exactly what is left");
}

void test_planned_terms() {
    HostMemoryDemand d = flash_next();
    const HostMemoryLedger base = plan_host_memory(snapshot(30 * kGiB, 200 * kGiB), d);
    d.later_pinned += kGiB;
    const HostMemoryLedger more_pins = plan_host_memory(snapshot(30 * kGiB, 200 * kGiB), d);
    require(base.expert_ram - more_pins.expert_ram == kGiB + (kGiB * 25 + 9'999) / 10'000,
            "a later pin costs its bytes plus its lock overhead");
    d = flash_next();
    d.pageable += kGiB;
    const HostMemoryLedger more_cache = plan_host_memory(snapshot(30 * kGiB, 200 * kGiB), d);
    require(base.expert_ram - more_cache.expert_ram == kGiB, "a pageable cache costs its bytes, no lock overhead");
}

void test_commit_limit() {
    const HostMemoryDemand d = flash_next();
    const std::uint64_t need = need_for_full(d);
    // Plenty of physical memory, a small page file.
    const HostMemoryLedger small_commit = plan_host_memory(snapshot(need + 10 * kGiB, need - 5 * kGiB), d);
    require(!small_commit.fits && small_commit.commit_limited && small_commit.expert_ram + 5 * kGiB == small_commit.full_need,
            "the commit limit bounds the experts and is reported");
    require(small_commit.summary().find("page file") != std::string::npos, "the refusal names the page file");
    const HostMemoryLedger enough = plan_host_memory(snapshot(need + 10 * kGiB, need + kGiB), d);
    require(enough.fits && !enough.commit_limited, "enough commit holds every expert");
}

void test_lock_limit() {
    const HostMemoryDemand d = flash_next();
    const std::uint64_t need = need_for_full(d);
    HostMemorySnapshot s     = snapshot(need + 10 * kGiB, need + 10 * kGiB);
    s.lock_limit             = 8 * kGiB;
    const HostMemoryLedger low = plan_host_memory(s, d);
    require(!low.fits && low.lock_limited, "a memory-lock limit below the pinned total refuses");
    require(low.summary().find("memory-lock limit") != std::string::npos, "the refusal names the lock limit");
    s.lock_limit = 64 * kGiB;
    require(plan_host_memory(s, d).fits, "a lock limit above the pinned total fits");
    s.lock_limit.reset();
    require(plan_host_memory(s, d).fits, "no lock limit fits");
    // The pinned load staging counts toward the limit while it overlaps the banks, up to the later pins.
    HostMemoryDemand staged = d;
    staged.later_pinned     = 0;
    staged.load_staging     = 2 * kGiB;
    const std::uint64_t peak = staged.expert_banks + staged.other_pinned + staged.load_staging;
    s.lock_limit             = peak;
    require(plan_host_memory(s, staged).pinned_total == peak && !plan_host_memory(s, staged).lock_limited,
            "a lock limit equal to the staging peak fits");
    s.lock_limit = peak - 1;
    require(plan_host_memory(s, staged).lock_limited, "a lock limit one byte under the staging peak refuses");
}

void test_starved_and_describe() {
    const HostMemoryDemand d     = flash_next();
    const HostMemoryLedger empty = plan_host_memory(snapshot(kGiB, kGiB), d);
    require(empty.expert_ram == 0 && !empty.fits, "no room leaves no expert RAM");
    const std::string line = plan_host_memory(snapshot(44 * kGiB, 120 * kGiB), d).describe();
    require(line.find("RAM ledger: available 44.00 GiB") == 0 && line.find("reserve 2.00") != std::string::npos &&
                line.find("every expert fits") != std::string::npos,
            "the ledger line names the inputs and the outcome: " + line);
    const std::string refusal = empty.summary();
    require(refusal.find("short") != std::string::npos && refusal.find("Close other programs") != std::string::npos,
            "a refusal says how much is missing and what to do: " + refusal);
    // The shortfall is everything the start needs (reserve, planned, experts) less what is free.
    char expected[64];
    std::snprintf(expected, sizeof(expected), "%.1f GiB short",
                  static_cast<double>(need_for_full(d) - kGiB) / static_cast<double>(kGiB));
    require(refusal.find(expected) != std::string::npos, std::string("the refusal names ") + expected + ": " + refusal);
}

// ---------------------------------------------------------------- RT7

constexpr std::uint64_t kRecord = 2'764'800; // Infernix's NVFP4 record: the sizing law is format-blind

VramSnapshot card(std::uint64_t free, DisplayState display) {
    VramSnapshot s;
    s.device_free  = free;
    s.device_total = 32'607 * kMiB;
    s.display      = display;
    s.outputs      = display == DisplayState::Attached ? 1 : display == DisplayState::Headless ? 0 : -1;
    return s;
}

VramDemand demand(std::uint64_t fixed, std::uint32_t graphs) {
    VramDemand d;
    d.graphs      = graphs;
    d.fixed       = fixed;
    d.frame_bytes = kRecord;
    d.max_frames  = 48 * 512 - 1;
    return d;
}

void test_graph_bound() {
    require(graph_bound(1, 1, 0) == 1, "plain C = 1: one decode graph");
    require(graph_bound(1, 8, 3) == 1 + 8 + 8 + 3, "C = 1, W = 8 with MTP 3");
    require(graph_bound(2, 16, 0) == 2 + 32, "n-gram only: decode and verification graphs");
    require(graph_bound(8, 16, 7) == 320, "C = 8 with MTP 7 and n-gram 15");
}

void test_display_headroom() {
    require(display_headroom(DisplayState::Headless, false, std::nullopt) == 256 * kMiB, "headless: 256 MiB");
    require(display_headroom(DisplayState::Attached, false, std::nullopt) == 1024 * kMiB,
            "a display with a fixed pool: 1 GiB");
    require(display_headroom(DisplayState::Attached, true, std::nullopt) == 512 * kMiB,
            "a display with an elastic pool: 512 MiB");
    require(display_headroom(DisplayState::Unknown, false, std::nullopt) == 1024 * kMiB,
            "an unknown display counts as attached");
    require(display_headroom(DisplayState::Attached, false, 0) == 0 &&
                display_headroom(DisplayState::Headless, false, 3 * kGiB) == 3 * kGiB,
            "a requested headroom wins in every state");
}

void test_internal_reserve() {
    require(internal_reserve(1, kGraphExecutableBytes, false) == 256 * kMiB, "few graphs: the 256 MiB floor");
    require(internal_reserve(40, kGraphExecutableBytes, false) == 128 * kMiB + 160 * kMiB,
            "a fixed pool reserves for every graph");
    require(internal_reserve(320, kGraphExecutableBytes, true) == 256 * kMiB,
            "an elastic pool counts at most 32 graphs");
}

void test_sizing() {
    // Headless, no OS budget: frames take free - fixed - headroom - reserve.
    const std::uint64_t free  = 28 * kGiB;
    const std::uint64_t fixed = 3 * kGiB;
    const VramSizing headless = size_expert_frames(card(free, DisplayState::Headless), demand(fixed, 20));
    const std::int64_t expect = static_cast<std::int64_t>(free - fixed - 256 * kMiB - 256 * kMiB);
    require(headless.frame_bytes == expect && headless.frames == static_cast<std::uint32_t>(expect / kRecord) &&
                !headless.budget_limited,
            "headless sizing");
    const VramSizing attached = size_expert_frames(card(free, DisplayState::Attached), demand(fixed, 20));
    require(attached.frame_bytes == expect - static_cast<std::int64_t>(768 * kMiB),
            "a display takes 768 MiB more with a fixed pool");

    // An OS budget below free memory bounds the frames with its 64 MiB margin only: the display
    // headroom is not stacked on top of it.
    VramSnapshot budgeted = card(free, DisplayState::Attached);
    budgeted.has_budget   = true;
    budgeted.local_budget = 26 * kGiB;
    budgeted.local_usage  = 0;
    const VramSizing low  = size_expert_frames(budgeted, demand(fixed, 20));
    require(low.budget_limited &&
                low.frame_bytes == static_cast<std::int64_t>(26 * kGiB - fixed - 64 * kMiB - 256 * kMiB),
            "the budget's bound: B - fixed - 64 MiB - reserve");
    budgeted.local_budget = 30 * kGiB; // above F - headroom: the free-memory bound applies
    const VramSizing high = size_expert_frames(budgeted, demand(fixed, 20));
    require(!high.budget_limited && high.frame_bytes == attached.frame_bytes, "a roomy budget changes nothing");
    budgeted.local_budget = 27 * kGiB;
    budgeted.local_usage  = 2 * kGiB; // usage counts against the budget
    require(size_expert_frames(budgeted, demand(fixed, 20)).frame_bytes ==
                static_cast<std::int64_t>(25 * kGiB - fixed - 64 * kMiB - 256 * kMiB),
            "the budget's bound subtracts this process's usage");

    // The fixed allocations do not fit: negative bytes, no frames, and the line says so.
    const VramSizing over = size_expert_frames(card(2 * kGiB, DisplayState::Headless), demand(fixed, 20));
    require(over.frame_bytes < 0 && over.frames == 0, "an overflow leaves no frames");
    require(over.describe().find("the fixed allocations do not fit") != std::string::npos,
            "an overflow is named: " + over.describe());

    // Every expert fits: the frames stop at max_frames.
    VramDemand all = demand(0, 1);
    all.max_frames = 100;
    require(size_expert_frames(card(free, DisplayState::Headless), all).frames == 100, "frames clamp to max_frames");

    VramDemand set   = demand(fixed, 20);
    set.headroom     = 0;
    const auto line  = size_expert_frames(card(free, DisplayState::Attached), set).describe();
    require(line.find("headroom 0 (set)") != std::string::npos && line.find("display attached") != std::string::npos,
            "the sizing line names the headroom's source and the display: " + line);
}

class ScriptedSource final : public VramBudgetSource {
public:
    explicit ScriptedSource(std::deque<std::uint64_t> readings) : readings_(std::move(readings)) {}
    VramSnapshot query() override {
        VramSnapshot s;
        s.device_free = readings_.front();
        if (readings_.size() > 1) { readings_.pop_front(); }
        return s;
    }

private:
    std::deque<std::uint64_t> readings_;
};

void test_spill() {
    const std::uint64_t n = 1 * kGiB;
    require(spill_shortfall(10 * kGiB, 9 * kGiB, n) == 0, "a resident step");
    require(spill_shortfall(10 * kGiB, 9 * kGiB + 30 * kMiB, n) == 0, "a shortfall within the tolerance");
    require(spill_shortfall(10 * kGiB, 10 * kGiB - 512 * kMiB, n) == 512 * kMiB, "half the step spilled");
    require(spill_shortfall(10 * kGiB, 10 * kGiB + kMiB, n) == n, "free memory rose: all of it missing");
    require(spill_shortfall(10 * kGiB, 8 * kGiB, n) == 0, "another program allocated meanwhile: no spill");

    // A transient high reading is re-read before a spill is reported.
    ScriptedSource noisy({10 * kGiB, 10 * kGiB, 9 * kGiB});
    SpillGuard guard(noisy);
    guard.begin();
    require(guard.end(n) == 0, "a second reading clears a transient shortfall");
    ScriptedSource spilled({10 * kGiB, 10 * kGiB - 100 * kMiB, 10 * kGiB - 100 * kMiB});
    SpillGuard confirm(spilled);
    confirm.begin();
    require(confirm.end(n) == n - 100 * kMiB, "a confirmed shortfall is reported");

    // The test seam replaces the sources opened afterwards.
    ninfer::testing::set_vram_budget_source([] { return card(5 * kGiB, DisplayState::Headless); });
    const VramSnapshot faked = open_vram_budget_source(0)->query();
    ninfer::testing::set_vram_budget_source({});
    require(faked.device_free == 5 * kGiB && faked.display == DisplayState::Headless, "the seam's source answers");
}

// ---------------------------------------------------------------- RT9

constexpr std::uint64_t kChunk = 64 * kMiB;

// A card where this process has `pool` bytes of frames and `other` bytes of everything else, the OS
// budget is 31,419 MiB and free memory is budget - usage (as RM0a measured on the RTX 5090).
VramSnapshot wddm(std::uint64_t pool, std::uint64_t other, DisplayState display = DisplayState::Headless,
                  std::uint64_t budget = 31'419 * kMiB) {
    VramSnapshot s;
    s.device_total = 32'579 * kMiB;
    s.has_budget   = true;
    s.local_budget = budget;
    s.local_usage  = pool + other;
    s.device_free  = budget > s.local_usage ? budget - s.local_usage : 0;
    s.display      = display;
    s.outputs      = display == DisplayState::Attached ? 1 : 0;
    return s;
}

std::uint32_t frames_in(std::uint64_t bytes) { return static_cast<std::uint32_t>(bytes / kRecord); }

void test_control_law() {
    VramDemand d;
    d.graphs      = 20;
    d.frame_bytes = kRecord;
    d.max_frames  = 48 * 512 - 1;
    const std::uint64_t other = 6 * kGiB; // dense weights, KV, workspace, staging, context
    // Sizing left headroom (256) + reserve (256) free, on the chunk grid.
    const std::uint64_t pool = (31'419 * kMiB - other - 512 * kMiB) / kChunk * kChunk;
    VramControl control(d, kChunk, pool + other, pool);
    const std::uint32_t frames = frames_in(pool);
    double t = 0.0;

    auto steady = control.decide(wddm(pool, other), pool, frames, t, true);
    require(!steady.shrink && !steady.grow && steady.frames == frames, "the sized pool is stable when idle");

    // Another program takes 2 GiB: this process's budget falls by it, free memory to 0.
    const std::uint64_t squeezed_budget = 31'419 * kMiB - 2 * kGiB;
    auto shrink = control.decide(wddm(pool, other, DisplayState::Headless, squeezed_budget), pool, frames, t += 1, false);
    require(shrink.shrink && shrink.pressure && shrink.frames < frames, "low free memory shrinks at once");
    const std::uint64_t new_pool = (static_cast<std::uint64_t>(shrink.frames) * kRecord + kChunk - 1) / kChunk * kChunk;
    // After the shrink the same pressure no longer holds and the pool is stable.
    const VramSnapshot after = wddm(new_pool, other, DisplayState::Headless, squeezed_budget);
    auto settled             = control.decide(after, new_pool, shrink.frames, t += 1, false);
    require(!settled.shrink && !settled.pressure && settled.frames == shrink.frames,
            "one chunk below the target: the next reading does not shrink again");
    require(after.device_free >= control.headroom(DisplayState::Headless), "the shrink restores the headroom");

    // The other program exits: room for the startup pool again. No growth before the delay.
    VramSnapshot freed = wddm(new_pool, other);
    for (double step = 0; step < 29.5; step += 1.0) {
        const auto wait = control.decide(freed, new_pool, shrink.frames, t + step, false);
        require(!wait.grow && wait.frames == shrink.frames, "no growth before 30 s");
    }
    const auto grow = control.decide(freed, new_pool, shrink.frames, t + 30.0, false);
    require(grow.grow && grow.frames == frames_in(new_pool + 4 * kChunk),
            "during rounds the pool grows by at most four chunks per boundary");
    const auto idle_grow = control.decide(freed, new_pool, shrink.frames, t + 30.5, true);
    require(idle_grow.grow && idle_grow.frames >= frames - frames_in(kChunk) && idle_grow.frames <= frames,
            "an idle engine grows to the target at once");

    // Hysteresis: one chunk of room never grows the pool.
    VramControl tight(d, kChunk, pool + other, pool);
    const VramSnapshot one_chunk = wddm(pool - kChunk, other);
    for (double step = 0; step < 120; step += 10) {
        require(!tight.decide(one_chunk, pool - kChunk, frames_in(pool - kChunk), step, true).grow,
                "a target one chunk above the pool does not grow it");
    }

    // A display appears and the desktop takes 400 MiB: the headroom rises from 256 to 512 MiB
    // (elastic) and the remaining ~170 MiB free is below half of it.
    VramControl plugged(d, kChunk, pool + other, pool);
    const VramSnapshot display = wddm(pool, other, DisplayState::Attached, 31'419 * kMiB - 400 * kMiB);
    require(plugged.headroom(DisplayState::Attached) == 512 * kMiB, "elastic headroom with a display");
    require(plugged.decide(display, pool, frames, 0.0, false).shrink, "a display appearing shrinks the pool");

    // the engine's own later allocations (graph executables, 64 MiB) spend the reserve, not the frames:
    // the target falls by what remains of the reserve's use, and growth never eats into it.
    VramControl reserve(d, kChunk, pool + other, pool);
    const VramSnapshot graphs = wddm(pool, other + 64 * kMiB);
    require(reserve.reserve_left(graphs, pool) == 256 * kMiB - 64 * kMiB, "graph memory spends the reserve");
    require(reserve.target(graphs, pool) <= frames, "an idle start never grows into the internal reserve");
    require(!reserve.decide(graphs, pool, frames, 100.0, true).shrink, "spending the reserve is not pressure");

    // The OS budget falls below this process's usage: shrink even with free memory reported.
    VramSnapshot trimmed = wddm(pool, other);
    trimmed.local_budget = pool + other - kGiB;
    trimmed.device_free  = kGiB;
    require(control.decide(trimmed, pool, frames, 200.0, false).shrink, "an exceeded budget shrinks the pool");
}

// The elastic KV pool's growth takes its memory from the frames before it maps (the order whose
// absence overcommitted WDDM and crashed 128K requests): releasing kv_growth_frames frames through
// the frame pool's chunk rule frees at least what the KV adds beyond the spare memory, from any
// fill level of the top chunk.
void test_kv_growth() {
    const std::uint64_t chunk = 64 * kMiB;
    require(kv_growth_frames(100 * kMiB, 100 * kMiB, 1000, kMiB, chunk) == 0, "spare memory covers the growth");
    require(kv_growth_frames(100 * kMiB, 200 * kMiB, 1000, kMiB, chunk) == 0, "more spare than growth");
    require(kv_growth_frames(5 * kGiB, 0, 1000, kMiB, chunk) == 1000, "the growth takes at most every frame");
    require(kv_growth_frames(kMiB, 0, 1000, 0, chunk) == 0, "no frames, nothing to take");
    // The 128K case on the RTX 3090: 1,344 MiB of KV, IQ2 frames of 1.44 MiB, no spare memory.
    const std::uint64_t iq2 = 1'474'560;
    const std::uint32_t take = kv_growth_frames(1344 * kMiB, 0, 12'258, iq2, chunk);
    require(take > 1344 * kMiB / iq2 && take <= 1344 * kMiB / iq2 + chunk / iq2 + 2, "128K growth takes ~1 chunk over");
    for (const std::uint64_t stride : {iq2, std::uint64_t{1'048'576}, std::uint64_t{3'000'000},
                                         std::uint64_t{67'108'864}, std::uint64_t{100'000'000}}) {
        for (const std::uint64_t added : {2 * kMiB, 130 * kMiB, 1344 * kMiB, 3500 * kMiB}) {
            for (const std::uint64_t spare : {std::uint64_t{0}, kMiB, 100 * kMiB, 2000 * kMiB}) {
                for (const std::uint32_t frames : {3000U, 12'258U}) {
                    // The frame pool maps whole chunks: from the frames' bytes up to one chunk above.
                    for (const std::uint64_t slack : {std::uint64_t{0}, chunk / 2, chunk - 1}) {
                        const std::uint64_t used = static_cast<std::uint64_t>(frames) * stride;
                        const std::uint64_t mapped = (used + slack + chunk - 1) / chunk * chunk;
                        const std::uint32_t give = kv_growth_frames(added, spare, frames, stride, chunk);
                        const std::uint64_t keep = static_cast<std::uint64_t>(frames - give) * stride;
                        std::uint64_t left = mapped;
                        while (left >= keep + chunk) { left -= chunk; } // ExpertResidency::resize's release
                        const std::uint64_t freed = mapped - left;
                        const std::uint64_t needed = added > spare ? added - spare : 0;
                        require(freed >= needed || give == frames,
                                "the frames release what the KV adds (stride " + std::to_string(stride) + ", added " +
                                    std::to_string(added >> 20) + " MiB, spare " + std::to_string(spare >> 20) + ")");
                    }
                }
            }
        }
    }
}

// --hot-embedding-rows: the token embedding (248,320 IQ4_XS rows of 2,560 values, 1,360 bytes each)
// leaves the dense device weights for the pinned block, whole, and its first N rows get a device
// copy. With N = 8192 (Basalt's) the device holds (V - N) rows fewer and the pinned demand grows by
// the whole table; with N = 0 nothing changes.
void test_hot_embedding_ledgers() {
    constexpr std::uint64_t kVocab = 248'320, kHidden = 2'560, kRow = 10 * 136;
    const HotEmbeddingPlan off = plan_hot_embedding(kVocab, kHidden, 0);
    require(off.row_bytes == 0 && off.table == 0 && off.hot == 0, "N = 0 plans nothing");
    const HotEmbeddingPlan on = plan_hot_embedding(kVocab, kHidden, 8192);
    require(on.row_bytes == kRow, "a row is hidden / 256 IQ4_XS blocks of 136 bytes");
    require(on.table == kVocab * kRow && on.table == 337'715'200ULL, "the whole table moves to the pinned block");
    require(on.hot == 8192 * kRow && on.hot == 11'141'120ULL, "the hot copy is the first 8192 rows");
    // Device: the dense weights lose the table, the Program's fixed allocations gain the hot rows.
    require(on.table - on.hot == (kVocab - 8192) * kRow && on.table - on.hot == 326'574'080ULL,
            "the device plan shrinks by (V - N) rows");
    require(plan_hot_embedding(kVocab, kHidden, static_cast<std::uint32_t>(kVocab)).hot == on.table,
            "N = V copies every row");
    bool refused = false;
    try {
        (void)plan_hot_embedding(kVocab, kHidden, static_cast<std::uint32_t>(kVocab) + 1);
    } catch (const std::invalid_argument& error) {
        refused = std::string(error.what()).find("--hot-embedding-rows") != std::string::npos;
    }
    require(refused, "N above the vocabulary is refused naming --hot-embedding-rows");
    // The two planes are IQ4_XS rows (embedding_split): an IQ3_S embedding (the GSQ-RCO IQ3_XXS
    // pack's) is refused with the flag, and accepted without it.
    refused = false;
    try {
        require_hot_embedding_format(ninfer::QType::GGML_IQ3_S, 8192);
    } catch (const std::invalid_argument& error) {
        refused = std::string(error.what()).find("--hot-embedding-rows") != std::string::npos &&
                  std::string(error.what()).find("ggml_iq3_s") != std::string::npos;
    }
    require(refused, "an IQ3_S embedding with --hot-embedding-rows is refused naming the flag and the format");
    require_hot_embedding_format(ninfer::QType::GGML_IQ3_S, 0);
    require_hot_embedding_format(ninfer::QType::GGML_IQ4_XS, 8192);

    // Host: the pinned block holds the banks, and with the option the table too; the ledger plans
    // the table as other_pinned and the rest as the banks.
    const HostMemoryDemand base = flash_next();
    HostMemoryDemand plain      = base;
    plan_pinned_weights(plain, base.expert_banks, off);
    require(plain.expert_banks == base.expert_banks && plain.other_pinned == 0, "N = 0 leaves the pinned terms");
    HostMemoryDemand hot = base;
    plan_pinned_weights(hot, base.expert_banks + on.table, on);
    require(hot.expert_banks == base.expert_banks && hot.other_pinned == on.table,
            "the table is other_pinned and the banks are unchanged");
    const HostMemorySnapshot s = snapshot(60 * kGiB, 200 * kGiB);
    const HostMemoryLedger a = plan_host_memory(s, plain), b = plan_host_memory(s, hot);
    require(b.pinned_total - a.pinned_total == kVocab * kRow, "the pinned total grows by the whole table");
    // The lock overhead is rounded up over every other pin at once: within a byte of 0.25 % of the table.
    const std::uint64_t overhead = (on.table * 25) / 10'000;
    require(b.planned - a.planned >= on.table + overhead && b.planned - a.planned <= on.table + overhead + 1,
            "the ledger plans the table with its lock overhead");
    require(b.full_need == a.full_need, "the experts' need is unchanged");
    char pinned[64];
    std::snprintf(pinned, sizeof(pinned), "(%.1f GiB pinned)",
                  static_cast<double>(hot.expert_banks + on.table) / static_cast<double>(kGiB));
    require(b.fits && b.summary().find(pinned) != std::string::npos, "the summary counts the pinned table");
}

// Partial RAM residency (--partial-ram-residency): with every layer's record the same size the slot
// pool is (E_total - vram_frames + spares) * stride, spares being 2 x the most experts one round
// promotes per layer, in every layer; the VRAM-only experts are the frames less the load margin,
// the warm start's ranking first and then every layer in turn; a layer never gets more slots than
// experts; and per-layer strides weigh each layer's slots.
void partial_plan_bytes() {
    constexpr std::uint32_t kLayers = 48, kExperts = 512;
    constexpr std::uint64_t kStride = 1'510'400;
    PartialRamDemand d;
    d.experts     = kExperts;
    d.strides.assign(kLayers, kStride);
    d.frame_bytes = kStride;
    d.frames      = 9'600;
    const std::uint32_t margin = static_cast<std::uint32_t>((kPartialFrameMarginBytes + kStride - 1) / kStride);
    const PartialRamPlan p = plan_partial_ram(d);
    const std::uint32_t vram = d.frames - margin;
    require(p.vram_keys.size() == vram, "the VRAM-only experts are the frames less the load margin");
    require(kPartialSparesPerLayer == 2 * 8, "spares are 2 x the 8 promotions a round makes per layer");
    const std::uint64_t spares = std::uint64_t(kLayers) * kPartialSparesPerLayer;
    require(p.bytes == (std::uint64_t(kLayers) * kExperts - vram + spares) * kStride,
            "host = (E_total - vram_frames + spares) * stride");
    require(p.cap_warning.empty() && p.slots.size() == kLayers, "no cap: the plan fits");
    std::set<std::uint32_t> unique(p.vram_keys.begin(), p.vram_keys.end());
    require(unique.size() == p.vram_keys.size() && *unique.rbegin() < kLayers * kExperts, "VRAM keys are distinct");
    // Without a ranking every layer takes its turn: the first 48 keys are expert 0 of each layer.
    require(p.vram_keys[0] == 0 && p.vram_keys[1] == kExperts && p.vram_keys[kLayers] == 1, "round robin over layers");
    // A ranking comes first; repeats and keys out of range are skipped.
    PartialRamDemand ranked = d;
    ranked.ranked = {7, 7, 1'000'000, 3 * kExperts + 5};
    const PartialRamPlan r = plan_partial_ram(ranked);
    require(r.vram_keys[0] == 7 && r.vram_keys[1] == 3 * kExperts + 5 && r.vram_keys[2] == 0, "the ranking leads");
    require(r.bytes == p.bytes, "the same counts of slots in total");
    // A cache smaller than the margin keeps nothing only in VRAM: every expert has a slot, no spares
    // beyond the layer's experts.
    PartialRamDemand tiny = d;
    tiny.frames = margin;
    const PartialRamPlan t = plan_partial_ram(tiny);
    require(t.vram_keys.empty() && t.bytes == std::uint64_t(kLayers) * kExperts * kStride,
            "a cache within the margin: every expert in RAM, slots capped at the experts");
    // Per-layer strides: each layer's slots at its own stride.
    PartialRamDemand mixed = d;
    for (std::uint32_t l = 0; l < kLayers; ++l) { mixed.strides[l] = l % 2 == 0 ? 1'382'400 : 2'329'600; }
    mixed.frame_bytes = 2'329'600;
    mixed.frames      = 4'800 + static_cast<std::uint32_t>((kPartialFrameMarginBytes + mixed.frame_bytes - 1) /
                                                          mixed.frame_bytes);
    const PartialRamPlan m = plan_partial_ram(mixed);
    std::uint64_t expect = 0;
    for (std::uint32_t l = 0; l < kLayers; ++l) {
        expect += std::uint64_t(kExperts - 100 + kPartialSparesPerLayer) * mixed.strides[l];
    }
    require(m.vram_keys.size() == 4'800 && m.bytes == expect, "100 VRAM-only experts per layer, each at its stride");
    require(partial_place_bounce_bytes(kStride) == 2 * kPartialPlaceRecords * kStride,
            "the placement's bounce is two halves of 16 records, a named later pin");
    require(wddm_pin_cap(64 * kGiB) == 24 * kGiB && wddm_pin_cap(12 * kGiB) == 0, "the WDDM cap is RAM/2 - 8 GiB");
}

// Over the WDDM page-lock cap (cudaHostRegister's, as Basalt reports it) the plan warns with what
// would fit: the fewest frames whose slot pool fits, as the largest --vram-headroom-mib that gives
// them, or the --expert-cache-mib that does when that option bounded the cache, or no fit at all;
// within the cap, or with no cap, it does not warn. The cap never changes the plan itself.
void partial_plan_wddm_cap_warning() {
    constexpr std::uint32_t kLayers = 48, kExperts = 512;
    constexpr std::uint64_t kStride = 2'329'600;
    PartialRamDemand d;
    d.experts      = kExperts;
    d.strides.assign(kLayers, kStride);
    d.frame_bytes  = kStride;
    d.frames       = 6'000;
    d.other_pinned = 300 * kMiB;
    d.headroom     = 4 * kGiB;
    const PartialRamPlan open = plan_partial_ram(d);
    require(open.cap_warning.empty(), "no cap: no warning");
    d.pin_cap = open.bytes + d.other_pinned;
    require(plan_partial_ram(d).cap_warning.empty(), "exactly at the cap fits");
    d.pin_cap = open.bytes + d.other_pinned - 100 * kStride;
    const PartialRamPlan over = plan_partial_ram(d);
    require(!over.cap_warning.empty() && over.frames_needed == d.frames + 100, "100 more frames would fit the cap");
    const std::uint64_t headroom = d.headroom - 100 * kStride;
    require(over.cap_warning.find("--vram-headroom-mib " + std::to_string(headroom / kMiB)) != std::string::npos &&
                over.cap_warning.find("--partial-ram-residency") != std::string::npos,
            "the warning names the largest headroom that fits: " + over.cap_warning);
    // Fits at the needed frame count, and not one frame fewer.
    PartialRamDemand at = d;
    at.frames = over.frames_needed;
    require(plan_partial_ram(at).cap_warning.empty(), "the named frame count fits");
    at.frames = over.frames_needed - 1;
    require(!plan_partial_ram(at).cap_warning.empty(), "one frame fewer does not");
    // --expert-cache-mib bounded the cache: the advice is the cache size.
    PartialRamDemand limited = d;
    limited.cache_limited = true;
    const std::uint64_t cache_mib = (std::uint64_t(over.frames_needed) * kStride + kMiB - 1) / kMiB;
    require(plan_partial_ram(limited).cap_warning.find("--expert-cache-mib " + std::to_string(cache_mib)) != std::string::npos,
            "a bounded cache is told the size that fits");
    // More frames than the headroom can give: no headroom fits.
    PartialRamDemand short_headroom = d;
    short_headroom.headroom = 50 * kStride;
    require(plan_partial_ram(short_headroom).cap_warning.find("--vram-headroom-mib") == std::string::npos,
            "a headroom that cannot give the frames is not offered");
    // A cap below the pins that never fit.
    PartialRamDemand hopeless = d;
    hopeless.pin_cap = d.other_pinned / 2;
    const PartialRamPlan never = plan_partial_ram(hopeless);
    require(!never.cap_warning.empty() && never.frames_needed == 0, "no frame count fits a cap below the other pins");
    require(over.bytes == open.bytes && over.vram_keys == open.vram_keys, "the cap only warns: the plan is unchanged");
    require(over.cap_warning.find("cudaHostRegister") != std::string::npos &&
                over.cap_warning.find("cudaHostAlloc") != std::string::npos,
            "the warning says the cap is cudaHostRegister's and cudaHostAlloc may still succeed");
}

} // namespace

int main() {
    try {
        test_fit_boundary();
        test_reserve_applied_once();
        test_planned_terms();
        test_commit_limit();
        test_lock_limit();
        test_starved_and_describe();
        test_graph_bound();
        test_display_headroom();
        test_internal_reserve();
        test_sizing();
        test_spill();
        test_control_law();
        test_kv_growth();
        test_hot_embedding_ledgers();
        partial_plan_bytes();
        partial_plan_wddm_cap_warning();
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL: %s\n", error.what());
        return 1;
    }
    std::printf("qwen4_exp memory plan checks passed\n");
    return 0;
}
