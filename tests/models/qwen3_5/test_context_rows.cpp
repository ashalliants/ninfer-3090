// The draft head's context rows (--draft-context-rows, CPU only): the set of a request's own token
// ids that the reduced proposal head lacks, and the head formats the option runs with.
#include "models/qwen3_5/program/context_rows.h"

#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using ninfer::models::qwen3_5::detail::ContextRowSet;
using ninfer::models::qwen3_5::detail::draft_context_rows_refusal;

namespace {

int failures = 0;

void expect(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

std::vector<std::int32_t> slots(const ContextRowSet& set) {
    return {set.slots().begin(), set.slots().end()};
}

// Vocabulary 64, the subset the even ids, room for four rows.
ContextRowSet even_subset(int capacity = 4) {
    std::vector<std::int32_t> subset;
    for (std::int32_t id = 0; id < 64; id += 2) { subset.push_back(id); }
    return ContextRowSet(subset, 64, capacity);
}

void subset_ignored() {
    ContextRowSet set = even_subset();
    const std::vector<std::int32_t> ids{2, 4, 6, 8};
    set.note(0, ids);
    expect(set.size() == 0, "ids the proposal head holds take no context row");
    expect(slots(set) == std::vector<std::int32_t>({-1, -1, -1, -1}), "every slot stays empty");
    const std::vector<std::int32_t> outside{-5, 64, 1000, -1};
    set.note(0, outside);
    expect(set.size() == 0, "ids outside the vocabulary take no context row");
    const std::vector<std::int32_t> mixed{2, 3, 4};
    set.note(0, mixed);
    expect(slots(set) == std::vector<std::int32_t>({3, -1, -1, -1}),
           "only the id the subset lacks is kept");
}

void dedup() {
    ContextRowSet set = even_subset();
    const std::vector<std::int32_t> ids{7, 7, 9, 7, 9};
    set.note(0, ids);
    expect(set.size() == 2 && slots(set) == std::vector<std::int32_t>({7, 9, -1, -1}),
           "a repeated id takes one row");
    const std::uint64_t version = set.version();
    const std::vector<std::int32_t> again{9, 7};
    set.note(0, again);
    expect(set.version() == version, "noting held ids again leaves the slots unchanged");
    set.note(1, again);
    expect(set.size() == 2 && set.version() == version,
           "another lane noting held ids shares their rows");
}

void capacity_cap() {
    ContextRowSet set = even_subset(3);
    const std::vector<std::int32_t> ids{1, 3, 5, 7, 9};
    set.note(0, ids);
    expect(set.size() == 3 && slots(set) == std::vector<std::int32_t>({1, 3, 5}),
           "ids past the capacity are dropped");
    const std::vector<std::int32_t> late{11};
    set.note(1, late);
    expect(slots(set) == std::vector<std::int32_t>({1, 3, 5}), "a full set takes no new id");
    set.release_lane(0);
    expect(set.size() == 0, "the only lane's release empties the set");
    set.note(1, late);
    expect(slots(set) == std::vector<std::int32_t>({11, -1, -1}),
           "a released row is free for a new id");

    bool refused = false;
    try {
        ContextRowSet invalid(std::vector<std::int32_t>{}, 64, 0);
    } catch (const std::invalid_argument&) { refused = true; }
    expect(refused, "a set without capacity is refused");
    refused = false;
    try {
        ContextRowSet invalid(std::vector<std::int32_t>{70}, 64, 4);
    } catch (const std::invalid_argument&) { refused = true; }
    expect(refused, "a subset id outside the vocabulary is refused");
}

void stable_order() {
    ContextRowSet set = even_subset(6);
    const std::vector<std::int32_t> first{21, 13, 5};
    set.note(0, first);
    expect(slots(set) == std::vector<std::int32_t>({21, 13, 5, -1, -1, -1}),
           "rows fill in the order ids are first noted");
    const std::vector<std::int32_t> second{5, 1, 21, 3};
    set.note(1, second);
    expect(slots(set) == std::vector<std::int32_t>({21, 13, 5, 1, 3, -1}),
           "held ids keep their rows and new ids append");
    const std::vector<std::int32_t> third{13};
    set.note(1, third);
    set.release_lane(0);
    expect(slots(set) == std::vector<std::int32_t>({21, 13, 5, 1, 3, -1}),
           "rows another lane holds stay where they are");
}

void lane_refcounts() {
    ContextRowSet set = even_subset(6);
    const std::vector<std::int32_t> lane0{1, 3, 5};
    const std::vector<std::int32_t> lane1{5, 7};
    set.note(0, lane0);
    set.note(1, lane1);
    expect(slots(set) == std::vector<std::int32_t>({1, 3, 5, 7, -1, -1}), "the union of lanes");
    const std::uint64_t before = set.version();
    set.release_lane(0);
    expect(slots(set) == std::vector<std::int32_t>({-1, -1, 5, 7, -1, -1}),
           "releasing one lane keeps the rows another lane holds");
    expect(set.size() == 2 && set.version() != before, "a release that frees rows is a change");
    const std::vector<std::int32_t> lane2{9, 11, 13};
    set.note(2, lane2);
    expect(slots(set) == std::vector<std::int32_t>({9, 11, 5, 7, 13, -1}),
           "new ids take the lowest free rows");
    const std::uint64_t held = set.version();
    set.release_lane(3);
    expect(set.version() == held, "releasing a lane that holds nothing changes nothing");
    set.release_lane(1);
    set.release_lane(2);
    expect(set.size() == 0 && slots(set) == std::vector<std::int32_t>(6, -1),
           "the last holder's release frees every row");
    bool refused = false;
    try {
        set.note(ContextRowSet::kMaximumLanes, lane0);
    } catch (const std::invalid_argument&) { refused = true; }
    expect(refused, "a lane outside the mask is refused");
}

void refusals() {
    using ninfer::QType;
    using ninfer::QuantLayout;
    expect(draft_context_rows_refusal(QType::Q8_G32_FP16, QuantLayout::RowSplit, 5120, true).empty(),
           "the 27B's Q8 head runs");
    expect(draft_context_rows_refusal(QType::Q6_G64_FP16, QuantLayout::RowSplit, 2048, true).empty(),
           "a Q6 head (--lm-head-q6) runs");
    expect(draft_context_rows_refusal(QType::Q4_G64_FP16, QuantLayout::RowSplit, 2048, true).empty(),
           "a Q4 head (--lm-head-q4) runs");
    const auto names_flag = [](const std::string& why) {
        return why.find("--draft-context-rows") != std::string::npos;
    };
    expect(names_flag(draft_context_rows_refusal(QType::FP8_E4M3FN_ROW_BF16, QuantLayout::RowScale,
                                                 5120, true)),
           "an FP8 head is refused naming the flag");
    expect(names_flag(
               draft_context_rows_refusal(QType::T2_G128_FP16, QuantLayout::RowSplit, 5120, true)),
           "a ternary head is refused naming the flag");
    expect(names_flag(draft_context_rows_refusal(QType::Q8_G32_FP16, QuantLayout::RowSplitPanel,
                                                 5120, true)),
           "a panel-permuted head is refused naming the flag");
    expect(names_flag(
               draft_context_rows_refusal(QType::Q8_G32_FP16, QuantLayout::RowSplit, 5000, true)),
           "a hidden size off the 128-column layout is refused naming the flag");
    expect(names_flag(
               draft_context_rows_refusal(QType::Q8_G32_FP16, QuantLayout::RowSplit, 5120, false)),
           "a proposal head without a token subset is refused naming the flag");
}

} // namespace

int main() {
    subset_ignored();
    dedup();
    capacity_cap();
    stable_order();
    lane_refcounts();
    refusals();
    if (failures != 0) {
        std::cerr << failures << " context row check(s) failed\n";
        return 1;
    }
    std::cout << "ok\n";
    return 0;
}
