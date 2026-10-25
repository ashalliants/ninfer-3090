#pragma once

// The draft head's context rows (--draft-context-rows). The reduced proposal head scores only its
// frequency-sorted subset of the vocabulary, so a name or identifier a request writes, if the
// subset lacks it, can never be drafted. The Program keeps those ids here, one fixed-capacity set
// shared by every lane, and the drafters score them from the full output head beside the subset
// (ops::linear_rows). Drafts only: verification decides every token. The idea is Basalt's (spare
// context rows of its MTP draft head, 8d73a39 src/core/mtp.cpp); this implementation is NInfer's.

#include "core/weight.h"
#include "ninfer/types.h"

#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace ninfer::models::qwen3_5::detail {

// A refcounted union of the ids each lane's text holds that the subset lacks, in `capacity` slots.
// An id takes the lowest free slot when it is first noted and keeps it while any lane holds it;
// a slot frees when its last lane is released. Ids noted while every slot is taken are dropped
// (the next note of the same id, once a slot is free, takes one).
class ContextRowSet {
public:
    // Lanes are bits of one 32-bit mask.
    static constexpr int kMaximumLanes = 32;

    // `subset_ids` are the proposal head's rows' ids, each in [0,vocab); `capacity` is in
    // [1,kMaximumDraftContextRows] (ninfer/types.h).
    ContextRowSet(std::span<const std::int32_t> subset_ids, std::int32_t vocab, int capacity);

    // `lane` holds every id of `ids` that is in [0,vocab) and outside the subset.
    void note(int lane, std::span<const std::int32_t> ids);
    // `lane` holds nothing; rows no other lane holds are freed.
    void release_lane(int lane) noexcept;

    // One id per slot, -1 for an empty slot: what the drafters' row ids hold.
    [[nodiscard]] std::span<const std::int32_t> slots() const noexcept { return slots_; }
    [[nodiscard]] int size() const noexcept { return size_; }
    [[nodiscard]] int capacity() const noexcept { return static_cast<int>(slots_.size()); }
    // Changes whenever slots() does.
    [[nodiscard]] std::uint64_t version() const noexcept { return version_; }

private:
    std::vector<std::uint8_t> in_subset_;
    std::vector<std::int32_t> slot_of_;
    std::vector<std::int32_t> slots_;
    std::vector<std::uint32_t> holders_;
    int size_            = 0;
    int first_free_      = 0;
    std::uint64_t version_ = 0;
};

// Why --draft-context-rows cannot score rows of an output head in `format` and `layout` whose
// rows are `columns` wide beside a proposal head (`proposal_indexed`: it has a token subset), or an
// empty string when it can.
[[nodiscard]] std::string draft_context_rows_refusal(QType format, QuantLayout layout,
                                                     std::int32_t columns, bool proposal_indexed);

} // namespace ninfer::models::qwen3_5::detail
