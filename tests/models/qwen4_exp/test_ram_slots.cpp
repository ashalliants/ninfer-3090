// Partial RAM residency's host bookkeeping (program/ram_slots.h, --partial-ram-residency): a layer's
// RAM slots hand out and take back slots and refuse when full; the swap planner names the write-backs
// an eviction needs and refuses an admission without a RAM copy; and over random evict/admit
// sequences, with write-backs and promotions landing late, every expert keeps at least one valid copy
// and no frame is reused while a write-back still reads it.

#include "models/qwen4_exp/program/ram_slots.h"

#include <algorithm>
#include <cstdio>
#include <map>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

using namespace ninfer::models::qwen4_exp::partial;

namespace {

int g_failures = 0;

void check(bool ok, const char* what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++g_failures;
    }
}

template <class F>
bool throws(F&& f) {
    try {
        f();
    } catch (const std::exception&) {
        return true;
    }
    return false;
}

void slots_acquire_release() {
    RamSlots slots(3, 1024);
    check(slots.slots() == 3 && slots.free_slots() == 3 && slots.bytes() == 3 * 1024, "three slots of 1 KiB");
    const auto a = slots.acquire(7), b = slots.acquire(9);
    check(a && b && *a == 0 && *b == 1, "slots are handed out lowest first");
    check(slots.acquire(7) == a && slots.free_slots() == 1, "a key that holds a slot gets its own back");
    check(slots.address(7) == 0 && slots.address(9) == 1024, "addresses are slot * stride");
    check(slots.holds(7) && !slots.holds(8), "holds names the keys with a slot");
    slots.release(7);
    check(!slots.holds(7) && slots.free_slots() == 2, "a released slot is free again");
    const auto c = slots.acquire(11);
    check(c && *c == 0, "the lowest free slot is reused first");
    check(throws([&] { slots.release(7); }), "releasing a key without a slot is refused");
    check(throws([&] { (void)slots.address(7); }), "the address of a key without a slot is refused");
    check(throws([] { RamSlots bad(2, 0); }), "a zero stride is refused");
}

void slots_exhaustion() {
    RamSlots slots(2, 64);
    check(slots.acquire(1) && slots.acquire(2), "two keys take both slots");
    check(!slots.acquire(3) && slots.free_slots() == 0, "a third key finds none");
    check(slots.acquire(2).has_value(), "a holder still gets its own slot when the pool is full");
    slots.release(1);
    check(slots.acquire(3).has_value() && slots.address(3) == 0, "a freed slot serves the next key");
    RamSlots none(0, 64);
    check(!none.acquire(0), "an empty pool hands out nothing");
}

// Two layers of 4 experts; layer 0 has 2 slots (2 experts live only in VRAM), layer 1 has 3.
Residency small_residency() {
    const std::vector<std::uint32_t> slots{2, 3};
    const std::vector<std::uint64_t> strides{256, 512};
    Residency r(4, slots, strides);
    r.place_vram(0, 0); // layer 0 experts 0 and 1 in frames 0 and 1
    r.place_vram(1, 1);
    r.place_ram(2);
    r.place_ram(3);
    r.place_vram(4, 2); // layer 1 expert 0 in frame 2
    for (ExpertKey k = 5; k < 8; ++k) { r.place_ram(k); }
    return r;
}

void residency_layout() {
    Residency r = small_residency();
    check(r.pool_bytes() == 2 * 256 + 3 * 512 && r.layer_base(1) == 2 * 256, "layers follow one another in the pool");
    check(r.ram_offset(3) == 256 && r.ram_offset(5) == 512, "a key's offset is its layer's base plus its slot");
    check(r.frame(0) == 0 && !r.in_ram(0) && r.in_ram(2) && r.frame(2) == -1, "placements are recorded");
    check(throws([&] { (void)r.place_ram(0); }), "a layer with no free slot refuses a RAM placement");
    r.check();
    const auto changed = r.take_changed();
    check(changed.size() == 5 && r.take_changed().empty(), "placements in RAM are reported once");
}

void swap_plan_names_writebacks() {
    Residency r = small_residency();
    r.publish(2, 3); // a promotion of key 2 into frame 3 landed; its RAM copy stays
    const std::vector<ExpertKey> evict{0, 2}, admit{3};
    const SwapPlan plan = plan_swap(r, evict, admit);
    check(plan.to_ram == std::vector<ExpertKey>{0}, "only the VRAM-only expert needs a write-back");
    check(plan.to_vram == std::vector<ExpertKey>{3}, "the admitted expert is copied from RAM");
    const std::vector<ExpertKey> resident{1};
    check(plan_swap(r, {}, resident).to_vram.empty(), "an expert already in VRAM is not copied again");
    r.evict(2);
    check(r.frame(2) == -1 && r.in_ram(2), "an expert with a RAM copy leaves VRAM at once");
    const std::vector<ExpertKey> back{0};
    const std::vector<std::uint32_t> one{1};
    const std::vector<std::uint64_t> stride{64};
    const Residency empty(2, one, stride);
    check(throws([&] { (void)plan_swap(empty, {}, back); }), "an admitted expert with no RAM copy is refused");
}

void writeback_and_reclaim() {
    Residency r = small_residency();
    r.publish(2, 3); // key 2 in VRAM with its copy
    // Layer 0's two slots hold keys 2 and 3: evicting key 0 must reclaim key 2's copy.
    check(r.slots(0).free_slots() == 0, "layer 0 is full");
    check(r.reserve_eviction(0), "a VRAM-only expert may leave when a slot can be reclaimed");
    check(!r.in_ram(2) && r.frame(2) == 3, "the reclaimed expert lives only in VRAM");
    check(!r.reserve_eviction(1), "no second slot: key 2 may not give its slot twice, key 3 is not in VRAM");
    const auto offset = r.evict(0);
    check(offset.has_value() && r.writing_back(0) && r.frame(0) == 0, "the write-back starts; the frame stays valid");
    const std::vector<ExpertKey> back{0};
    check(plan_swap(r, {}, back).to_vram == back, "an expert being written back may be admitted again");
    r.check();
    r.complete_writeback(0);
    check(r.in_ram(0) && r.frame(0) == -1 && r.ram_offset(0) == *offset, "the landed write-back is the copy");
    r.clear_reservations();
    r.check();
    check(r.reclaimed() == 1, "one slot was reclaimed");
    // A key approved this round keeps its copy: it cannot be reclaimed from.
    Residency s = small_residency();
    s.publish(2, 3);
    check(s.reserve_eviction(2), "a key with a copy may leave");
    check(!s.reserve_eviction(0), "the leaving key's slot is not reclaimed");
    s.clear_reservations();
    check(s.reserve_eviction(0), "after the round it may be");
}

// Random rounds over 3 layers of 16 experts and 20 frames: each round evicts some VRAM experts the
// gate approves and admits some RAM experts into free frames; write-backs and promotions land a
// random number of rounds later. After every step every key has a valid copy, a frame being written
// back is never handed out, and the swap plan's write-backs are exactly the VRAM-only victims.
void swap_plan_invariant() {
    constexpr std::uint32_t kLayers = 3, kExperts = 16, kFrames = 20;
    const std::vector<std::uint32_t> slots{12, 11, 14};
    const std::vector<std::uint64_t> strides{256, 512, 768};
    std::mt19937 rng(20261026);
    for (int trial = 0; trial < 20; ++trial) {
        Residency r(kExperts, slots, strides);
        std::vector<std::uint32_t> free_frames;
        std::map<ExpertKey, std::uint32_t> in_vram; // the policy's view: resident keys and their frames
        // Initial placement: enough VRAM-only experts per layer that every other one has a slot.
        std::uint32_t next_frame = 0;
        for (std::uint32_t l = 0; l < kLayers; ++l) {
            const std::uint32_t vram = kExperts - slots[l] + 1; // one spare slot per layer
            for (std::uint32_t e = 0; e < kExperts; ++e) {
                const ExpertKey key = l * kExperts + e;
                if (e < vram) {
                    r.place_vram(key, next_frame);
                    in_vram[key] = next_frame++;
                } else {
                    r.place_ram(key);
                }
            }
        }
        for (std::uint32_t f = next_frame; f < kFrames; ++f) { free_frames.push_back(f); }
        r.check();
        struct Pending {
            ExpertKey key;
            std::uint32_t frame;
            int rounds;
        };
        std::vector<Pending> writebacks, loads;
        std::set<std::uint32_t> held; // frames whose write-back is in flight
        bool ok = true;
        for (int round = 0; round < 300 && ok; ++round) {
            // Land some of the work in flight (write-backs first, as the copy stream orders them).
            for (auto it = writebacks.begin(); it != writebacks.end();) {
                if (--it->rounds <= 0) {
                    r.complete_writeback(it->key);
                    held.erase(it->frame);
                    free_frames.push_back(it->frame);
                    it = writebacks.erase(it);
                } else {
                    ++it;
                }
            }
            for (auto it = loads.begin(); it != loads.end();) {
                // A load of a key being written back reads its slot after the write-back on the
                // same copy stream, so it never lands first.
                if (--it->rounds <= 0 && !r.writing_back(it->key)) {
                    r.publish(it->key, it->frame);
                    it = loads.erase(it);
                } else {
                    ++it;
                }
            }
            r.check();
            // Victims: a few residents the gate approves.
            std::vector<ExpertKey> evict;
            for (const auto& [key, frame] : in_vram) {
                if (rng() % 5 != 0) { continue; }
                const bool loading =
                    std::any_of(loads.begin(), loads.end(), [&](const Pending& p) { return p.key == key; });
                if (loading) { continue; } // keep the simulation's landing simple
                if (r.reserve_eviction(key)) { evict.push_back(key); }
            }
            // Admissions: RAM experts into free frames.
            std::vector<ExpertKey> admit;
            for (ExpertKey key = 0; key < kLayers * kExperts; ++key) {
                if (in_vram.contains(key) || rng() % 7 != 0) { continue; }
                if (std::find(evict.begin(), evict.end(), key) != evict.end()) { continue; }
                admit.push_back(key);
            }
            const SwapPlan plan = plan_swap(r, evict, admit);
            for (const ExpertKey key : evict) {
                const bool vram_only = r.frame(key) >= 0 && !r.in_ram(key);
                const bool named = std::find(plan.to_ram.begin(), plan.to_ram.end(), key) != plan.to_ram.end();
                if (vram_only != named) { ok = false; }
            }
            for (const ExpertKey key : evict) {
                const std::uint32_t frame = in_vram.at(key);
                in_vram.erase(key);
                if (r.evict(key)) {
                    held.insert(frame);
                    writebacks.push_back({key, frame, 1 + static_cast<int>(rng() % 3)});
                } else {
                    free_frames.push_back(frame);
                }
            }
            r.clear_reservations();
            for (const ExpertKey key : plan.to_vram) {
                if (free_frames.empty()) { break; }
                const std::uint32_t frame = free_frames.back();
                free_frames.pop_back();
                if (held.contains(frame)) { ok = false; } // a frame under write-back was handed out
                in_vram[key] = frame;
                loads.push_back({key, frame, 1 + static_cast<int>(rng() % 3)});
            }
            try {
                r.check();
            } catch (const std::exception&) {
                ok = false;
            }
            for (std::uint32_t l = 0; l < kLayers; ++l) {
                if (r.slots(l).slots() != slots[l]) { ok = false; }
            }
        }
        check(ok, "every key keeps a valid copy, write-backs are the VRAM-only victims, held frames stay held");
    }
}

} // namespace

int main() {
    slots_acquire_release();
    slots_exhaustion();
    residency_layout();
    swap_plan_names_writebacks();
    writeback_and_reclaim();
    swap_plan_invariant();
    if (g_failures) {
        std::fprintf(stderr, "%d check(s) failed\n", g_failures);
        return 1;
    }
    std::printf("partial RAM residency checks passed\n");
    return 0;
}
