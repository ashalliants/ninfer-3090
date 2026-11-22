// Written for NInfer-3090.
//
// The residency policy of the DeepSeek-V4.1-Flash expert tiers (src/models/dsv4/execution/expert_tiers.h):
// RAM slots evict the lowest priority record (by default: fewest uses, with dynamic aging) that the call did not use and
// no copy holds; VRAM frames take a record when one is free, else only one used more often than the least used frame.

#include "models/dsv4/execution/expert_tiers.h"

#include <cstdio>
#include <initializer_list>

using ninfer::models::dsv4::ExpertTierPolicy;
using ninfer::models::dsv4::RamEviction;
using ninfer::models::dsv4::TierPolicyOptions;

namespace {

int failures = 0;

void check(bool ok, const char* what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++failures;
    }
}

// One call routing `keys`: touches them all, then gives the ones without a RAM slot a slot.
void route(ExpertTierPolicy& p, std::initializer_list<std::uint32_t> keys) {
    p.begin_call();
    for (const auto k : keys) { p.touch(k); }
    for (const auto k : keys) {
        if (p.ram_of(k) < 0) { (void)p.acquire_ram(k); }
    }
}

TierPolicyOptions with(RamEviction e, bool exclusive = false) {
    TierPolicyOptions o;
    o.ram_eviction = e;
    o.exclusive    = exclusive;
    return o;
}

void ram_lru() {
    ExpertTierPolicy p(16, 3, 0, with(RamEviction::Lru));
    route(p, {0, 1});
    route(p, {2});
    check(p.ram_of(0) >= 0 && p.ram_of(1) >= 0 && p.ram_of(2) >= 0, "three keys fit three slots");
    route(p, {1}); // 0 is now the least recently used
    route(p, {3});
    check(p.ram_of(0) < 0, "the least recently used key lost its slot");
    check(p.ram_of(1) >= 0 && p.ram_of(2) >= 0 && p.ram_of(3) >= 0, "the others kept theirs");
    // A call never evicts what it routed: three keys in a three slot pool all stay.
    route(p, {4, 5, 6});
    check(p.ram_of(4) >= 0 && p.ram_of(5) >= 0 && p.ram_of(6) >= 0, "one call's keys all hold slots");
    // A fourth key in the same call finds no victim.
    p.begin_call();
    for (const std::uint32_t k : {4U, 5U, 6U, 7U}) { p.touch(k); }
    check(p.acquire_ram(7) == ExpertTierPolicy::kNone, "no slot is free when the call uses them all");
}

// Hot records that a stream of once-used records would flush from an LRU pool (six scan records between two uses of
// the hot ones, in a pool of four) survive it under a frequency-aware policy.
void ram_scan_resistance() {
    for (const RamEviction e : {RamEviction::Lru, RamEviction::Lfu, RamEviction::Segmented, RamEviction::Lfuda}) {
        ExpertTierPolicy p(256, 4, 0, with(e));
        for (int i = 0; i < 8; ++i) { route(p, {0, 1}); } // two hot records
        std::uint32_t scan = 10;
        int resident       = 0;
        for (int round = 0; round < 10; ++round) {
            for (int i = 0; i < 6; ++i) { route(p, {scan++}); }
            resident += (p.ram_of(0) >= 0 ? 1 : 0) + (p.ram_of(1) >= 0 ? 1 : 0); // are they still there when used again?
            route(p, {0, 1});
        }
        if (e == RamEviction::Lru) {
            check(resident == 0, "LRU loses the hot records to the scan");
        } else {
            check(resident == 20, "a frequency-aware policy keeps the hot records through the scan");
        }
    }
}

// The hot set moves: records that were hot long ago do not stay forever (Lfuda's aging; Lfu's halving).
void ram_follows_the_workload() {
    TierPolicyOptions o = with(RamEviction::Lfuda);
    o.age_period        = 64;
    ExpertTierPolicy p(64, 3, 0, o);
    for (int i = 0; i < 20; ++i) { route(p, {0, 1, 2}); }
    for (int round = 0; round < 40; ++round) { route(p, {10, 11, 12}); }
    check(p.ram_of(10) >= 0 && p.ram_of(11) >= 0 && p.ram_of(12) >= 0, "the new hot set replaced the old one");
    check(p.ram_of(0) < 0 && p.ram_of(1) < 0 && p.ram_of(2) < 0, "the old hot set left");
}

void ram_segmented_protects_repeats() {
    ExpertTierPolicy p(32, 3, 0, with(RamEviction::Segmented));
    route(p, {0});
    route(p, {0}); // used twice: protected
    route(p, {1});
    route(p, {2});
    route(p, {3}); // evicts a once-used key (1, the older), not the protected 0
    check(p.ram_of(0) >= 0 && p.ram_of(1) < 0 && p.ram_of(2) >= 0 && p.ram_of(3) >= 0, "a once-used key leaves before a repeated one");
}

// With exclusive tiers the slot of a record that also has a frame is the first to be reused.
void ram_exclusive_prefers_framed_records() {
    ExpertTierPolicy p(32, 3, 2, with(RamEviction::Lfuda, true));
    for (int i = 0; i < 5; ++i) { route(p, {0}); }
    route(p, {1, 2});
    (void)p.admit_frame(0); // key 0 is the most used and is now served from a frame
    route(p, {3});
    check(p.ram_of(0) < 0 && p.ram_of(1) >= 0 && p.ram_of(2) >= 0 && p.ram_of(3) >= 0, "the framed record gave up its RAM slot");
    check(p.frame_of(0) >= 0, "and keeps its frame");
    ExpertTierPolicy q(32, 3, 2, with(RamEviction::Lfuda, false));
    for (int i = 0; i < 5; ++i) { route(q, {0}); }
    route(q, {1, 2});
    (void)q.admit_frame(0);
    route(q, {3});
    check(q.ram_of(0) >= 0, "an inclusive policy keeps it");
}

void frames_resize_after_construction() {
    ExpertTierPolicy p(32, 4, 0);
    route(p, {0});
    check(!p.admit_frame(0), "no frames yet");
    p.resize_frames(2);
    check(p.frames() == 2 && p.admit_frame(0).has_value(), "frames added after construction are used");
    p.resize_frames(0);
    check(p.frame_of(0) < 0 && !p.admit_frame(0), "and removed again");
}

void frame_margin_slows_promotion() {
    TierPolicyOptions o = with(RamEviction::Lfuda);
    o.frame_margin      = 2;
    ExpertTierPolicy p(32, 8, 1, o);
    route(p, {0});
    (void)p.admit_frame(0);
    route(p, {1});
    route(p, {1});
    route(p, {1});
    check(!p.admit_frame(1), "a margin of 2 needs more than two extra uses");
    route(p, {1});
    check(p.admit_frame(1).has_value(), "three extra uses take the frame");
}

void ram_holds() {
    ExpertTierPolicy p(16, 2, 0);
    route(p, {0, 1});
    const std::int32_t slot0 = p.ram_of(0);
    const std::int32_t slot1 = p.ram_of(1);
    p.hold_ram(slot0, 7);
    p.begin_call();
    p.touch(2);
    p.touch(3);
    check(p.acquire_ram(2) == slot1 && p.ram_of(0) == slot0, "a held slot is not a victim");
    check(p.acquire_ram(3) == ExpertTierPolicy::kNone, "with one slot held and the other just filled, nothing is free");
    p.set_completed(7);
    check(p.acquire_ram(3) == slot0, "the slot is a victim once its copy completed");
    check(p.ram_of(0) < 0, "the evicted key has no slot");
}

void frames_fill_then_frequency() {
    ExpertTierPolicy p(32, 8, 2);
    route(p, {0});
    const auto a = p.admit_frame(0);
    check(a && a->evicted < 0, "a free frame is taken");
    route(p, {1});
    const auto b = p.admit_frame(1);
    check(a && b && b->frame != a->frame, "the second free frame");
    // Key 0 is used a second time, key 1 once; a newcomer used once finds no frame.
    route(p, {0});
    route(p, {2});
    check(!p.admit_frame(2), "a key used once does not displace keys used as often");
    // Used three times it displaces the least used frame (key 1, used once).
    route(p, {2});
    route(p, {2});
    const auto c = p.admit_frame(2);
    check(c && c->evicted == 1, "a more frequent key displaces the least used frame");
    check(c && p.frame_of(1) < 0 && p.frame_of(2) == c->frame && p.frame_of(0) >= 0, "frame ownership follows");
}

void frames_are_protected_within_a_call() {
    ExpertTierPolicy p(32, 8, 2);
    route(p, {0});
    (void)p.admit_frame(0);
    route(p, {1});
    (void)p.admit_frame(1);
    route(p, {1}); // key 0 is now the least used frame
    for (int i = 0; i < 3; ++i) { route(p, {2}); }
    p.begin_call();
    p.touch(2);
    p.touch(0); // the call uses key 0: it is not a victim even though it is the least used
    const auto c = p.admit_frame(2);
    check(c && c->evicted == 1, "a frame the call uses is not taken");
    p.begin_call();
    p.touch(5);
    p.touch(2);
    p.touch(0);
    check(!p.admit_frame(5), "no frame is taken when the call uses them all");
}

void aging_follows_the_workload() {
    ExpertTierPolicy p(32, 8, 1, 4);
    for (int i = 0; i < 6; ++i) { route(p, {0}); }
    (void)p.admit_frame(0);
    check(p.frequency(0) >= 3, "use counts accumulate");
    // Unused for a long time, key 0's count decays and a new key takes its place.
    for (int i = 0; i < 40; ++i) { p.begin_call(); }
    check(p.frequency(0) <= 1, "counts are halved every age_period calls");
    route(p, {7});
    check(p.admit_frame(7).has_value(), "the new key displaces the faded one");
}

void failed_fills_roll_back() {
    ExpertTierPolicy p(8, 2, 1);
    route(p, {0});
    check(p.ram_of(0) >= 0, "key 0 got a slot");
    p.drop_ram(0);
    check(p.ram_of(0) < 0, "a dropped RAM slot is no longer resident");
    route(p, {1, 2});
    check(p.ram_of(1) >= 0 && p.ram_of(2) >= 0, "the dropped slot is reusable");
    check(p.admit_frame(1).has_value(), "key 1 gets the frame");
    p.drop_frame(1);
    check(p.frame_of(1) < 0, "a dropped frame is no longer resident");
    route(p, {2});
    check(p.admit_frame(2).has_value(), "the dropped frame is reusable");
}

void no_frames_never_admits() {
    ExpertTierPolicy p(8, 2, 0);
    route(p, {0});
    check(!p.admit_frame(0), "a policy without frames admits nothing");
    p.clear();
    check(p.ram_of(0) < 0, "clear forgets the slots");
}

} // namespace

int main() {
    ram_lru();
    ram_scan_resistance();
    ram_follows_the_workload();
    ram_segmented_protects_repeats();
    ram_exclusive_prefers_framed_records();
    frames_resize_after_construction();
    frame_margin_slows_promotion();
    ram_holds();
    frames_fill_then_frequency();
    frames_are_protected_within_a_call();
    aging_follows_the_workload();
    failed_fills_roll_back();
    no_frames_never_admits();
    if (failures == 0) { std::printf("PASS\n"); }
    return failures == 0 ? 0 : 1;
}
