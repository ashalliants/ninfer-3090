#pragma once

// Adapted from Basalt 8d73a39 include/basalt/core/expert_source.hpp (MIT, third_party/basalt/LICENSE).
// Basalt's resident RAM mode keeps a compact copy of the experts no VRAM slot holds; NInfer-3090
// keeps per-layer slot pools (its layers' records differ in size), keeps the RAM copy of a promoted
// expert, and writes an expert held only in VRAM back into a slot before its frame is reused.
//
// Partial RAM residency (FlashNextOptions::partial_ram_residency, --partial-ram-residency): host RAM
// holds the routed experts the VRAM expert cache does not, plus spare slots, instead of every bank
// whole. This is the host bookkeeping of where each expert's record lives; it allocates nothing.
//
// Every expert always has at least one valid copy: in a published VRAM frame, in a RAM slot, or in
// both.
// - At load each expert is read from the artifact into either a frame (the hottest ones, which then
//   live only in VRAM) or a RAM slot.
// - A promotion copies a record RAM -> VRAM and keeps the RAM copy. The slot pool is sized at startup,
//   so the copy costs no RAM, and evicting that expert later needs no copy at all.
// - Evicting an expert held only in VRAM first writes its record back into a RAM slot: a free one, or
//   one reclaimed from another expert of the layer that is in VRAM with a copy (which then lives only
//   in VRAM). Until the write-back lands the expert still reads from its frame, and the frame is not
//   reused before then.
//
// Keys are layer * experts + expert, as everywhere in the expert cache.

#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <span>
#include <unordered_map>
#include <vector>

namespace ninfer::models::qwen4_exp::partial {

using ExpertKey = std::uint32_t;

// One layer's RAM slots: `slots` slots of `stride` bytes (the layer's record stride), each holding
// at most one expert. Slots are handed out lowest first, so experts placed in ascending order fill
// consecutive slots.
class RamSlots {
public:
    RamSlots(std::size_t slots, std::size_t stride);

    // The key's slot: the one it holds, else a free one (nothing when every slot is taken).
    [[nodiscard]] std::optional<std::uint32_t> acquire(ExpertKey key);
    // Frees the key's slot; the key must hold one.
    void release(ExpertKey key);
    // The byte offset of the key's slot in the pool; the key must hold one.
    [[nodiscard]] std::uint64_t address(ExpertKey key) const;
    [[nodiscard]] bool holds(ExpertKey key) const { return slot_.contains(key); }

    [[nodiscard]] std::size_t slots() const noexcept { return slots_; }
    [[nodiscard]] std::size_t free_slots() const noexcept { return free_.size(); }
    [[nodiscard]] std::size_t stride() const noexcept { return stride_; }
    [[nodiscard]] std::uint64_t bytes() const noexcept { return static_cast<std::uint64_t>(slots_) * stride_; }

private:
    std::size_t slots_  = 0;
    std::size_t stride_ = 0;
    std::vector<std::uint32_t> free_; // the lowest free slot at the back
    std::unordered_map<ExpertKey, std::uint32_t> slot_;
};

// Where every routed expert of the model lives (one RamSlots pool per layer, laid out one after the
// other in one pinned block), and the transitions that keep at least one valid copy of each.
class Residency {
public:
    // Layer l has `slots[l]` slots of `strides[l]` bytes for its `experts` experts.
    Residency(std::uint32_t experts, std::span<const std::uint32_t> slots, std::span<const std::uint64_t> strides);

    [[nodiscard]] std::uint32_t layers() const noexcept { return static_cast<std::uint32_t>(slots_.size()); }
    [[nodiscard]] std::uint32_t experts() const noexcept { return experts_; }
    [[nodiscard]] std::uint32_t keys() const noexcept { return layers() * experts_; }
    [[nodiscard]] const RamSlots& slots(std::uint32_t layer) const { return slots_.at(layer); }
    // The first byte of layer l's slots in the pool, and the pool's size.
    [[nodiscard]] std::uint64_t layer_base(std::uint32_t layer) const { return bases_.at(layer); }
    [[nodiscard]] std::uint64_t pool_bytes() const noexcept { return pool_bytes_; }

    // The published VRAM frame holding the key, or -1.
    [[nodiscard]] std::int32_t frame(ExpertKey key) const { return frame_.at(key); }
    // Whether the key has a valid RAM copy.
    [[nodiscard]] bool in_ram(ExpertKey key) const { return state_.at(key) == kInRam; }
    // Whether a write-back into the key's slot is in flight (its frame is still valid).
    [[nodiscard]] bool writing_back(ExpertKey key) const { return state_.at(key) == kWritingBack; }
    // The pool offset of the key's slot (a valid copy, or one being written back).
    [[nodiscard]] std::uint64_t ram_offset(ExpertKey key) const;

    // Load: the key lives only in VRAM `frame`, or only in a RAM slot (returns its pool offset;
    // refused when its layer has no free slot).
    void place_vram(ExpertKey key, std::uint32_t frame);
    std::uint64_t place_ram(ExpertKey key);

    // A copy of the key into `frame` landed and is published (the RAM copy, if any, stays).
    void publish(ExpertKey key, std::uint32_t frame);

    // The eviction gate (expert_cache::LfruPolicy's `evictable`): whether the key may leave VRAM now.
    // It may when it is not in VRAM or has a RAM copy, or when its layer has a slot for its write-back:
    // a free one not yet reserved, else one reclaimed now from the layer's VRAM resident with a copy
    // that `score` ranks highest (lowest key without one), which then lives only in VRAM. Approved
    // keys are never reclaimed from until clear_reservations, so a key this round evicts keeps its copy.
    // A layer found with no slot to give is remembered until its slots or residents change, so
    // further VRAM-only keys of that layer are refused in O(1), without searching it again.
    using Score = std::function<double(ExpertKey)>;
    [[nodiscard]] bool reserve_eviction(ExpertKey key, const Score* score = nullptr);
    // The gate's classes (expert_cache::LfruPolicy::set_gate_classes): 2 * layer + 1 for a key held
    // only in VRAM (refused exactly when its layer has no slot to give, which lasts the step), else
    // 2 * layer (always approved). gate_classes() of them.
    [[nodiscard]] std::uint32_t gate_class(ExpertKey key) const {
        const std::uint32_t layer = key / experts_;
        return 2 * layer + (frame_.at(key) >= 0 && state_[key] == kNone && !leaving_[key] ? 1U : 0U);
    }
    [[nodiscard]] std::uint32_t gate_classes() const noexcept { return 2 * layers(); }
    void clear_reservations();

    // The key leaves VRAM. With a RAM copy it leaves at once (nothing returned). Without one a
    // write-back into a slot (reserved by reserve_eviction, else free) starts and its pool offset is
    // returned: the key keeps reading from its frame until complete_writeback.
    std::optional<std::uint64_t> evict(ExpertKey key);
    // The key's write-back landed: its RAM copy is valid and it no longer lives in VRAM.
    void complete_writeback(ExpertKey key);

    // Keys whose RAM copy or slot changed since the last call (placed, written back, reclaimed).
    [[nodiscard]] std::vector<ExpertKey> take_changed();
    [[nodiscard]] std::uint64_t reclaimed() const noexcept { return reclaimed_; }
    // Searches of a layer for a copy to reclaim (each O(experts)); tests bound them.
    [[nodiscard]] std::uint64_t reclaim_searches() const noexcept { return searches_; }

    // Throws std::logic_error when some key has no valid copy, or a layer holds more slots than it has.
    void check() const;

private:
    enum : std::uint8_t { kNone, kInRam, kWritingBack };
    [[nodiscard]] std::uint32_t layer_of(ExpertKey key) const;
    void mark(ExpertKey key);

    std::uint32_t experts_ = 0;
    std::vector<RamSlots> slots_;
    std::vector<std::uint64_t> bases_;
    std::uint64_t pool_bytes_ = 0;
    std::vector<std::int32_t> frame_;
    std::vector<std::uint8_t> state_;
    std::vector<std::uint32_t> reserved_; // per layer: free slots promised to this round's write-backs
    std::vector<std::uint8_t> leaving_;   // per key: approved by reserve_eviction this round
    std::vector<ExpertKey> leaving_keys_;
    std::vector<std::uint8_t> changed_;
    std::vector<ExpertKey> changed_keys_;
    std::uint64_t reclaimed_ = 0;
    std::uint64_t searches_  = 0;
    // Per layer: a generation that changes with its slots or VRAM residents, and the generation at
    // which it was last found with no slot to give.
    std::vector<std::uint64_t> generation_, blocked_at_;
    void touch(std::uint32_t layer) { ++generation_[layer]; }
};

// One round's moves: the evicted experts that need a write-back first (held only in VRAM), and the
// admitted experts copied RAM -> VRAM (each has a RAM copy, or one being written back that lands
// first on the copy stream).
struct SwapPlan {
    std::vector<ExpertKey> to_ram, to_vram;
};
// Refuses (std::logic_error) an admitted key with no RAM copy; keys already in VRAM are not moved.
[[nodiscard]] SwapPlan plan_swap(const Residency& residency, std::span<const ExpertKey> evict,
                                 std::span<const ExpertKey> admit);

} // namespace ninfer::models::qwen4_exp::partial
