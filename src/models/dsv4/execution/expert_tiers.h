#pragma once

// Written for NInfer-3090.
//
// The residency policy of DeepSeek-V4.1-Flash's routed expert records over two cache tiers above the
// artifact file, with no CUDA in it so it is unit-tested on the host (tests/models/dsv4/test_expert_tiers.cpp)
// and replayed against recorded routing traces (tests/models/dsv4/test_expert_trace_replay.cpp):
//
//   RAM slots   a pinned pool of `ram_slots` records. Which record leaves is chosen by RamEviction; the default
//               (Lfuda) is frequency first with dynamic aging, so a record used often survives a long scan of
//               records used once (a prefill chunk routes most experts of a layer, a decode token six). A slot can
//               be held (a copy to a device frame is reading it) and is then not a victim. With `exclusive`, the
//               slot of a record that also has a device frame is the first victim: the frame already serves it, so
//               RAM and VRAM together hold more distinct records.
//   VRAM frames a device pool of `frames` records the Op reads in place. Frequency first: a record is
//               admitted into a full pool only when it has been used more often than the least used
//               frame (plus `frame_margin`), so one pass over many experts does not flush the hot set. Ties and
//               the choice among equally used frames go to the least recently used.
//
// Keys are layer * experts + expert. A key's use count is bumped by touch() once per call it is routed in
// and halved for every key every `age_period` calls, so the hot set follows the workload (LFRU: frequency
// with aging, recency as the tie-break). The count outlives residency: a record that was evicted and comes
// back is not a stranger (a ghost list, as ARC keeps, without the extra structure). Nothing a call has
// touched is evicted by that call.

#include <cstdint>
#include <optional>
#include <stdexcept>
#include <vector>

namespace ninfer::models::dsv4 {

struct TierCounters {
    std::uint64_t vram_hits = 0;      // routed expert read from a device frame
    std::uint64_t ram_hits  = 0;      // routed expert read from a pinned RAM slot
    std::uint64_t misses    = 0;      // routed expert read from the file
    std::uint64_t promotions   = 0;   // records copied RAM -> VRAM
    std::uint64_t frame_evictions = 0;
};

// Which RAM slot loses its record when a miss needs one.
enum class RamEviction : std::uint8_t {
    Lru,       // least recently used
    Lfu,       // lowest use count, least recently used among equals
    Segmented, // records used at least twice are protected from records used once (2Q / SLRU); LRU inside each
    Lfuda,     // LFU with dynamic aging: priority = use count + the priority of the last victim (no scan pollution, no stale hot set)
};

struct TierPolicyOptions {
    RamEviction ram_eviction = RamEviction::Lfuda;
    bool exclusive           = true;        // a record in a device frame is the first to leave RAM
    std::uint32_t age_period = 1U << 14;    // use counts are halved every this many calls
    std::uint32_t frame_margin = 0;         // a newcomer needs more than the victim's count plus this
};

class ExpertTierPolicy {
public:
    static constexpr std::int32_t kNone = -1;

    ExpertTierPolicy(std::uint32_t keys, std::uint32_t ram_slots, std::uint32_t frames, TierPolicyOptions options = {})
        : options_(options), age_period_(options.age_period == 0 ? 1 : options.age_period), key_(keys), ram_(ram_slots), frame_(frames) {}
    ExpertTierPolicy(std::uint32_t keys, std::uint32_t ram_slots, std::uint32_t frames, std::uint32_t age_period)
        : ExpertTierPolicy(keys, ram_slots, frames, with_age(age_period)) {}

    void begin_call() {
        ++call_;
        if (call_ % age_period_ == 0) {
            for (auto& k : key_) { k.freq >>= 1; }
        }
    }

    [[nodiscard]] const TierPolicyOptions& options() const noexcept { return options_; }
    [[nodiscard]] std::uint32_t keys() const noexcept { return static_cast<std::uint32_t>(key_.size()); }
    [[nodiscard]] std::uint32_t ram_slots() const noexcept { return static_cast<std::uint32_t>(ram_.size()); }
    [[nodiscard]] std::uint32_t frames() const noexcept { return static_cast<std::uint32_t>(frame_.size()); }
    [[nodiscard]] std::uint64_t call() const noexcept { return call_; }
    [[nodiscard]] std::int32_t frame_of(std::uint32_t key) const { return key_.at(key).frame; }
    [[nodiscard]] std::int32_t ram_of(std::uint32_t key) const { return key_.at(key).ram; }
    [[nodiscard]] std::uint32_t frequency(std::uint32_t key) const { return key_.at(key).freq; }

    // Gives the device tier `frames` frames (a pool sized after the policy was built, once the free device
    // memory is known). Forgets which keys had a frame; the RAM tier is untouched.
    void resize_frames(std::uint32_t frames) {
        for (auto& k : key_) { k.frame = kNone; }
        frame_.assign(frames, Frame{});
    }

    // The key is routed in this call: counts the use and protects whatever it already occupies.
    void touch(std::uint32_t key) {
        Key& k = key_.at(key);
        if (k.touched == call_) { return; }
        k.touched = call_;
        ++k.freq;
        if (k.ram >= 0) {
            Ram& r      = ram_[static_cast<std::size_t>(k.ram)];
            r.last_used = call_;
            r.priority  = floor_ + k.freq;
        }
        if (k.frame >= 0) { frame_[static_cast<std::size_t>(k.frame)].last_used = call_; }
    }

    // A RAM slot for `key` (which has none), evicting the best victim that this call has not used and no
    // copy holds; kNone when there is none. The caller fills the slot before the next call.
    [[nodiscard]] std::int32_t acquire_ram(std::uint32_t key) {
        if (key_.at(key).ram >= 0) { throw std::logic_error("expert tiers: key already has a RAM slot"); }
        std::int32_t victim = kNone;
        for (std::size_t s = 0; s < ram_.size(); ++s) {
            const Ram& r = ram_[s];
            if (r.key < 0) {
                victim = static_cast<std::int32_t>(s);
                break;
            }
            if (r.last_used == call_ || r.hold > completed_) { continue; }
            if (victim < 0 || ram_before(r, ram_[static_cast<std::size_t>(victim)])) { victim = static_cast<std::int32_t>(s); }
        }
        if (victim < 0) { return kNone; }
        Ram& r = ram_[static_cast<std::size_t>(victim)];
        if (r.key >= 0) {
            if (r.priority > floor_) { floor_ = r.priority; } // the aging of Lfuda: later records start above what just left
            key_[static_cast<std::size_t>(r.key)].ram = kNone;
        }
        r.key       = static_cast<std::int64_t>(key);
        r.last_used = call_;
        r.hold      = 0;
        r.priority  = floor_ + key_[key].freq;
        key_[key].ram = victim;
        return victim;
    }

    // Copies that read RAM slots carry a batch number; a slot held by batch b is not a victim until
    // set_completed(n) with n >= b.
    void hold_ram(std::int32_t slot, std::uint64_t batch) { ram_.at(static_cast<std::size_t>(slot)).hold = batch; }
    void set_completed(std::uint64_t batch) noexcept { completed_ = batch > completed_ ? batch : completed_; }
    [[nodiscard]] std::uint64_t completed() const noexcept { return completed_; }

    struct Admission {
        std::int32_t frame;
        std::int64_t evicted; // key that lost the frame, or -1
    };
    // Gives `key` (already used by this call, normally resident in RAM) a device frame when the policy
    // wants it there: a free frame, else the least used frame this call did not use, if `key` has been used
    // more often. The caller copies the record into the frame before any later call reads it.
    [[nodiscard]] std::optional<Admission> admit_frame(std::uint32_t key) {
        Key& k = key_.at(key);
        if (k.frame >= 0 || frame_.empty()) { return std::nullopt; }
        std::int32_t victim = kNone;
        for (std::size_t f = 0; f < frame_.size(); ++f) {
            const Frame& fr = frame_[f];
            if (fr.key < 0) {
                victim = static_cast<std::int32_t>(f);
                break;
            }
            if (fr.last_used == call_) { continue; }
            if (victim < 0) {
                victim = static_cast<std::int32_t>(f);
                continue;
            }
            const Frame& best = frame_[static_cast<std::size_t>(victim)];
            const std::uint32_t fa = key_[static_cast<std::size_t>(fr.key)].freq, fb = key_[static_cast<std::size_t>(best.key)].freq;
            if (fa < fb || (fa == fb && fr.last_used < best.last_used)) { victim = static_cast<std::int32_t>(f); }
        }
        if (victim < 0) { return std::nullopt; }
        Frame& fr            = frame_[static_cast<std::size_t>(victim)];
        std::int64_t evicted = -1;
        if (fr.key >= 0) {
            if (key_[static_cast<std::size_t>(fr.key)].freq + options_.frame_margin >= k.freq) { return std::nullopt; }
            evicted = fr.key;
            key_[static_cast<std::size_t>(fr.key)].frame = kNone;
        }
        fr.key       = static_cast<std::int64_t>(key);
        fr.last_used = call_;
        k.frame      = victim;
        return Admission{victim, evicted};
    }

    // Undo of acquire_ram / admit_frame for a record that could not be filled: the key no longer owns the slot or frame.
    void drop_ram(std::uint32_t key) {
        Key& k = key_.at(key);
        if (k.ram < 0) { return; }
        ram_[static_cast<std::size_t>(k.ram)] = Ram{};
        k.ram                                 = kNone;
    }
    void drop_frame(std::uint32_t key) {
        Key& k = key_.at(key);
        if (k.frame < 0) { return; }
        frame_[static_cast<std::size_t>(k.frame)] = Frame{};
        k.frame                                   = kNone;
    }

    // Forgets everything (a cold start); the counters of the caller are its own.
    void clear() {
        for (auto& k : key_) { k = Key{}; }
        for (auto& r : ram_) { r = Ram{}; }
        for (auto& f : frame_) { f = Frame{}; }
        floor_ = 0;
    }

private:
    struct Key {
        std::int32_t ram = -1, frame = -1;
        std::uint32_t freq = 0;
        std::uint64_t touched = 0;
    };
    struct Ram {
        std::int64_t key = -1;
        std::uint64_t last_used = 0, hold = 0, priority = 0;
    };
    struct Frame {
        std::int64_t key = -1;
        std::uint64_t last_used = 0;
    };

    static TierPolicyOptions with_age(std::uint32_t age_period) {
        TierPolicyOptions o;
        o.age_period = age_period;
        return o;
    }

    // Whether the occupied slot `a` should be evicted before `b`.
    [[nodiscard]] bool ram_before(const Ram& a, const Ram& b) const {
        if (options_.exclusive) {
            const bool fa = key_[static_cast<std::size_t>(a.key)].frame >= 0, fb = key_[static_cast<std::size_t>(b.key)].frame >= 0;
            if (fa != fb) { return fa; }
        }
        const std::uint32_t qa = key_[static_cast<std::size_t>(a.key)].freq, qb = key_[static_cast<std::size_t>(b.key)].freq;
        switch (options_.ram_eviction) {
        case RamEviction::Lru: break;
        case RamEviction::Lfu:
            if (qa != qb) { return qa < qb; }
            break;
        case RamEviction::Segmented:
            if ((qa >= 2) != (qb >= 2)) { return qa < 2; }
            break;
        case RamEviction::Lfuda:
            if (a.priority != b.priority) { return a.priority < b.priority; }
            break;
        }
        return a.last_used < b.last_used;
    }

    TierPolicyOptions options_;
    std::uint32_t age_period_;
    std::uint64_t call_ = 0, completed_ = 0, floor_ = 0;
    std::vector<Key> key_;
    std::vector<Ram> ram_;
    std::vector<Frame> frame_;
};

} // namespace ninfer::models::dsv4
