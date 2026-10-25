#include "models/qwen3_5/program/context_rows.h"

#include <stdexcept>

namespace ninfer::models::qwen3_5::detail {

ContextRowSet::ContextRowSet(std::span<const std::int32_t> subset_ids, std::int32_t vocab,
                             int capacity) {
    if (vocab <= 0) { throw std::invalid_argument("context rows need a positive vocabulary"); }
    if (capacity < 1 || static_cast<std::uint32_t>(capacity) > kMaximumDraftContextRows) {
        throw std::invalid_argument("--draft-context-rows must be in [1," +
                                    std::to_string(kMaximumDraftContextRows) + "]");
    }
    in_subset_.assign(static_cast<std::size_t>(vocab), 0);
    for (const std::int32_t id : subset_ids) {
        if (id < 0 || id >= vocab) {
            throw std::invalid_argument("a proposal head id is outside the vocabulary");
        }
        in_subset_[static_cast<std::size_t>(id)] = 1;
    }
    slot_of_.assign(static_cast<std::size_t>(vocab), -1);
    slots_.assign(static_cast<std::size_t>(capacity), -1);
    holders_.assign(static_cast<std::size_t>(capacity), 0U);
}

void ContextRowSet::note(int lane, std::span<const std::int32_t> ids) {
    if (lane < 0 || lane >= kMaximumLanes) {
        throw std::invalid_argument("context row lane is outside the lane mask");
    }
    const std::uint32_t bit = 1U << lane;
    const auto vocab        = static_cast<std::int32_t>(slot_of_.size());
    for (const std::int32_t id : ids) {
        if (id < 0 || id >= vocab || in_subset_[static_cast<std::size_t>(id)] != 0) { continue; }
        const std::int32_t held = slot_of_[static_cast<std::size_t>(id)];
        if (held >= 0) {
            holders_[static_cast<std::size_t>(held)] |= bit;
            continue;
        }
        if (size_ == capacity()) { continue; }
        const int slot = first_free_;
        slots_[static_cast<std::size_t>(slot)]   = id;
        holders_[static_cast<std::size_t>(slot)] = bit;
        slot_of_[static_cast<std::size_t>(id)]   = slot;
        ++size_;
        ++version_;
        while (first_free_ < capacity() && slots_[static_cast<std::size_t>(first_free_)] >= 0) {
            ++first_free_;
        }
    }
}

void ContextRowSet::release_lane(int lane) noexcept {
    if (lane < 0 || lane >= kMaximumLanes) { return; }
    const std::uint32_t bit = 1U << lane;
    for (int slot = 0; slot < capacity(); ++slot) {
        std::uint32_t& holders = holders_[static_cast<std::size_t>(slot)];
        if ((holders & bit) == 0) { continue; }
        holders &= ~bit;
        if (holders != 0) { continue; }
        std::int32_t& id                       = slots_[static_cast<std::size_t>(slot)];
        slot_of_[static_cast<std::size_t>(id)] = -1;
        id                                     = -1;
        --size_;
        ++version_;
        if (slot < first_free_) { first_free_ = slot; }
    }
}

std::string draft_context_rows_refusal(QType format, QuantLayout layout, std::int32_t columns,
                                       bool proposal_indexed) {
    if (!proposal_indexed) {
        return "--draft-context-rows adds rows the reduced proposal head lacks, and this "
               "artifact's proposal head has no token subset: remove --draft-context-rows";
    }
    if ((format != QType::Q8_G32_FP16 && format != QType::Q6_G64_FP16 &&
         format != QType::Q4_G64_FP16) ||
        layout != QuantLayout::RowSplit) {
        return "--draft-context-rows reads Q8, Q6 or Q4 row-split output heads (the default "
               "head, --lm-head-q6 or --lm-head-q4) and this artifact's output head is stored "
               "otherwise: remove --draft-context-rows";
    }
    if (columns <= 0 || columns % 128 != 0) {
        return "--draft-context-rows reads output heads whose hidden size is a multiple of 128: "
               "remove --draft-context-rows";
    }
    return {};
}

} // namespace ninfer::models::qwen3_5::detail
