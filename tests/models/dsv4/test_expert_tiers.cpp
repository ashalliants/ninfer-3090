// Written for NInfer-3090.
//
// The residency policy of the DeepSeek-V4.1-Flash expert tiers (src/models/dsv4/execution/expert_tiers.h):
// RAM slots evict the least recently used record that the call did not use and no copy holds; VRAM frames
// take a record when one is free, else only one used more often than the least used frame.

#include "models/dsv4/execution/expert_tiers.h"

#include <cstdio>
#include <initializer_list>

using ninfer::models::dsv4::ExpertTierPolicy;

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

void ram_lru() {
    ExpertTierPolicy p(16, 3, 0);
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
    ram_holds();
    frames_fill_then_frequency();
    frames_are_protected_within_a_call();
    aging_follows_the_workload();
    no_frames_never_admits();
    if (failures == 0) { std::printf("PASS\n"); }
    return failures == 0 ? 0 : 1;
}
