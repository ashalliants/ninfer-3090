// Spike tool (spike/qwen4exp-lanes): replays captured Qwen3.8-Flash-Next route traces through the
// real expert-cache policy (expert_cache::CacheController, the LFRU and frame pool the Program runs)
// with C lanes decoding in the same rounds. CPU only. See docs/maintainer/flash-next-lanes-spike.md.
//
// Traces come from NINFER_ROUTE_TRACE (the spike's instrumentation in ExpertResidency::after_round):
// per round {i32 columns, top_k, layers, budget, hits, routed, then layers * top_k * columns ids}.
// Each trace is one request: its prefill calls, then one-column decode rounds.
//
// Model of a C-lane server (stated in the document): the list is served once, from --offset on; each
// lane takes the next unserved conversation when its own ends (no conversation repeats, so no lane
// finds its own text's experts left over); a lane that starts a conversation runs its prefill calls as rounds of their own
// (the other lanes wait; prefill promotes nothing, as the wide calls do), then the free frames are
// refilled by LFRU score as the lease's give-back does; every decode round routes the union of the
// decoding lanes' next tokens per layer and promotes per the Program's decode_budget (one per layer
// per token while filling, then every `interval` tokens). With --width W each decoding lane routes its
// next W tokens in a round and advances --advance tokens (an MTP verification's proxy: accepted drafts
// are the real next tokens). A hit is an expert READY in a frame at the round's start, as the
// Program's statistics count it.
//
// Build (from a VS 2022 x64 environment):
//   cl /std:c++20 /O2 /EHsc /I src tools\flash_next\lanes_spike\lanes_replay.cpp
//      src\models\qwen4_exp\program\expert_cache\expert_cache.cpp /Fe:lanes_replay.exe
// Usage:
//   lanes_replay --frames F --lanes C [--width W --advance A] [--interval 4] [--warmup R]
//                [--rounds N] [--offset K] [--validate] trace_0.bin=PROMPT_TOKENS trace_1.bin=...
// --validate replays the traces in order at one lane with their recorded budgets and prints the
// real (recorded) and replayed decode hit rates per request.

#include "models/qwen4_exp/program/expert_cache/expert_cache.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using ninfer::models::qwen4_exp::expert_cache::CacheController;
using ninfer::models::qwen4_exp::expert_cache::ResidencyState;

constexpr std::uint32_t kLayers = 48, kExperts = 512, kTopK = 10;
constexpr std::uint32_t kKeys = kLayers * kExperts;

struct Round {
    std::int32_t columns = 0, budget = 0, hits = 0, routed = 0;
    std::vector<std::int32_t> ids; // [layers][top_k * columns]
};

struct Conversation {
    std::string name;
    std::uint32_t prompt = 0;
    std::vector<Round> prefill;
    std::vector<Round> decode; // one column each: token t's routes
};

Conversation load(const std::string& spec) {
    const auto eq = spec.rfind('=');
    if (eq == std::string::npos) { throw std::invalid_argument("trace argument must be path=prompt_tokens"); }
    Conversation c;
    c.name   = spec.substr(0, eq);
    c.prompt = static_cast<std::uint32_t>(std::stoul(spec.substr(eq + 1)));
    std::ifstream in(c.name, std::ios::binary);
    if (!in) { throw std::runtime_error("cannot open " + c.name); }
    std::uint32_t columns_seen = 0;
    for (;;) {
        std::int32_t head[6];
        if (!in.read(reinterpret_cast<char*>(head), sizeof(head))) { break; }
        if (head[1] != static_cast<std::int32_t>(kTopK) || head[2] != static_cast<std::int32_t>(kLayers)) {
            throw std::runtime_error("unexpected trace shape in " + c.name);
        }
        Round r;
        r.columns = head[0];
        r.budget  = head[3];
        r.hits    = head[4];
        r.routed  = head[5];
        r.ids.resize(static_cast<std::size_t>(kLayers) * kTopK * r.columns);
        in.read(reinterpret_cast<char*>(r.ids.data()), static_cast<std::streamsize>(r.ids.size() * 4));
        if (columns_seen < c.prompt) {
            columns_seen += static_cast<std::uint32_t>(r.columns);
            c.prefill.push_back(std::move(r));
        } else {
            if (r.columns != 1) { throw std::runtime_error("a decode round with more than one column in " + c.name); }
            c.decode.push_back(std::move(r));
        }
    }
    if (columns_seen != c.prompt) { throw std::runtime_error("prefill columns differ from the prompt in " + c.name); }
    return c;
}

struct Totals {
    std::uint64_t rounds = 0, routed = 0, hits = 0, misses = 0, layer_calls = 0, layer_calls_missing = 0;
    std::uint64_t layer_calls_over2 = 0, distinct = 0, promotions = 0, tokens = 0, columns = 0;
    std::uint64_t real_routed = 0, real_hits = 0; // validate: the recorded statistics of the same rounds
};

class Replay {
public:
    Replay(std::uint32_t frames, std::uint64_t interval)
        : ctl_(kKeys, frames, 0, 1, frames), frames_(frames), interval_(interval) {
        seen_.assign(kExperts, 0);
    }

    // One round: per layer the union of `columns` (each a span of top_k ids). Returns misses.
    void round(const std::vector<const std::int32_t*>& columns, const std::vector<std::size_t>& strides,
               std::size_t budget, Totals* t) {
        commands_.clear();
        std::uint64_t round_misses = 0;
        for (std::uint32_t layer = 0; layer < kLayers; ++layer) {
            group_.clear();
            for (std::size_t c = 0; c < columns.size(); ++c) {
                const std::int32_t* ids = columns[c] + static_cast<std::size_t>(layer) * strides[c];
                for (std::uint32_t k = 0; k < kTopK; ++k) {
                    const auto e = static_cast<std::uint32_t>(ids[k]);
                    if (e >= kExperts || seen_[e]) { continue; }
                    seen_[e] = 1;
                    group_.push_back(layer * kExperts + e);
                }
            }
            std::uint64_t misses = 0;
            for (const auto key : group_) {
                seen_[key - layer * kExperts] = 0;
                if (ctl_.entry(key).state != ResidencyState::kReady) { ++misses; }
            }
            if (t != nullptr) {
                t->routed += group_.size();
                t->hits += group_.size() - misses;
                t->misses += misses;
                t->distinct += group_.size();
                ++t->layer_calls;
                if (misses > 0) { ++t->layer_calls_missing; }
                if (misses > 2) { ++t->layer_calls_over2; }
            }
            round_misses += misses;
            ctl_.on_route(group_, round_, commands_, budget);
        }
        ++round_;
        ctl_.on_quiescent(commands_);
        std::uint64_t loads = 0;
        for (const auto& c : commands_) {
            if (c.kind == CacheController::Command::Kind::kCopy) { ++loads; }
        }
        promotions_ += loads;
        if (t != nullptr) {
            ++t->rounds;
            t->promotions += loads;
        }
        (void)round_misses;
    }

    // The Program's decode_budget for a round advancing `tokens`.
    std::size_t decode_budget(std::uint32_t tokens) {
        if (promotions_ < frames_) { return tokens; }
        budget_tokens_ += tokens;
        const std::uint64_t due = budget_tokens_ / interval_;
        budget_tokens_ %= interval_;
        return static_cast<std::size_t>(due);
    }

    // The give-back's refill: free frames take the best non-resident keys with uses.
    void refill() {
        const std::uint32_t room = ctl_.free_frames();
        if (room == 0) { return; }
        const auto& policy = ctl_.policy();
        std::vector<std::pair<double, std::uint32_t>> ranked;
        for (std::uint32_t key = 0; key < kKeys; ++key) {
            if (!policy.resident(key) && policy.count(key) > 0) { ranked.emplace_back(policy.score(key), key); }
        }
        const std::size_t take = std::min<std::size_t>(room, ranked.size());
        std::partial_sort(ranked.begin(), ranked.begin() + static_cast<std::ptrdiff_t>(take), ranked.end(),
                          [](const auto& a, const auto& b) { return a.first > b.first || (a.first == b.first && a.second < b.second); });
        std::vector<std::uint32_t> keys;
        for (std::size_t i = 0; i < take; ++i) { keys.push_back(ranked[i].second); }
        commands_.clear();
        const std::uint32_t loaded = ctl_.seed(keys, {}, commands_);
        promotions_ += loaded;
    }

private:
    CacheController ctl_;
    std::uint32_t frames_;
    std::uint64_t interval_;
    std::uint64_t round_ = 0, promotions_ = 0, budget_tokens_ = 0;
    std::vector<CacheController::Command> commands_;
    std::vector<std::uint32_t> group_;
    std::vector<std::uint8_t> seen_;
};

struct Lane {
    std::size_t conversation = 0; // index into the list
    std::size_t next_prefill = 0; // prefill rounds done
    std::size_t token        = 0; // next decode token
    bool decoding            = false;
};

} // namespace

int main(int argc, char** argv) {
    std::uint32_t frames = 0, lanes = 1, width = 1, advance = 1;
    std::uint64_t interval = 4, warmup = 300, max_rounds = 0, offset = 0;
    bool validate = false;
    std::vector<std::string> specs;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        const auto value    = [&]() -> std::string {
            if (++i >= argc) { throw std::invalid_argument(a + " needs a value"); }
            return argv[i];
        };
        if (a == "--frames") { frames = static_cast<std::uint32_t>(std::stoul(value())); }
        else if (a == "--lanes") { lanes = static_cast<std::uint32_t>(std::stoul(value())); }
        else if (a == "--width") { width = static_cast<std::uint32_t>(std::stoul(value())); }
        else if (a == "--advance") { advance = static_cast<std::uint32_t>(std::stoul(value())); }
        else if (a == "--interval") { interval = std::stoull(value()); }
        else if (a == "--warmup") { warmup = std::stoull(value()); }
        else if (a == "--rounds") { max_rounds = std::stoull(value()); }
        else if (a == "--validate") { validate = true; }
        else if (a == "--offset") { offset = std::stoull(value()); }
        else { specs.push_back(a); }
    }
    if (frames == 0 || lanes == 0 || specs.empty() || advance == 0 || advance > width) {
        std::fprintf(stderr, "usage: lanes_replay --frames F --lanes C [--width W --advance A] [--interval 4] "
                             "[--warmup R] [--rounds N] [--validate] trace=prompt_tokens ...\n");
        return 2;
    }
    std::vector<Conversation> list;
    for (const auto& s : specs) { list.push_back(load(s)); }

    Replay replay(frames, interval);
    Totals measured;

    if (validate) {
        // One lane, the conversations in capture order, the recorded budgets: the replay of the real run.
        Totals per;
        std::printf("validate: frames %u\n%-40s %8s %8s %10s %10s\n", frames, "conversation", "decode", "real hit",
                    "replay hit", "replay prefill");
        for (const auto& c : list) {
            Totals pre;
            for (const auto& r : c.prefill) {
                replay.round({r.ids.data()}, {static_cast<std::size_t>(kTopK) * r.columns},
                             static_cast<std::size_t>(r.budget), &pre);
            }
            replay.refill();
            Totals dec;
            for (const auto& r : c.decode) {
                dec.real_routed += static_cast<std::uint64_t>(r.routed);
                dec.real_hits += static_cast<std::uint64_t>(r.hits);
                replay.round({r.ids.data()}, {kTopK}, static_cast<std::size_t>(r.budget), &dec);
            }
            std::printf("%-40s %8zu %7.2f%% %9.2f%% %9.2f%%\n", c.name.c_str(), c.decode.size(),
                        100.0 * dec.real_hits / std::max<std::uint64_t>(1, dec.real_routed),
                        100.0 * dec.hits / std::max<std::uint64_t>(1, dec.routed),
                        100.0 * pre.hits / std::max<std::uint64_t>(1, pre.routed));
        }
        return 0;
    }

    // C lanes. The list is served once, from conversation `offset` on (wrapping), each lane taking the
    // next unserved conversation when its own finishes; a lane with none left goes idle and the run
    // ends when every lane is idle. Only rounds in which all C lanes decode are measured (after
    // `warmup` decode rounds), so the tail where fewer lanes remain does not dilute C.
    std::vector<Lane> state(lanes);
    std::vector<bool> idle(lanes, false);
    std::size_t served = 0;
    const auto take = [&](Lane& l, std::size_t i) {
        if (served >= list.size()) {
            idle[i] = true;
            return;
        }
        l              = Lane{};
        l.conversation = (offset + served++) % list.size();
    };
    for (std::size_t i = 0; i < state.size(); ++i) { take(state[i], i); }
    std::uint64_t decode_rounds = 0;
    const std::uint64_t limit   = max_rounds != 0 ? max_rounds + warmup : ~0ULL;
    while (decode_rounds < limit) {
        if (std::all_of(idle.begin(), idle.end(), [](bool b) { return b; })) { break; }
        // Prefill first: one lane's call per round, as its own round (promotes nothing).
        bool prefilled = false;
        for (std::size_t i = 0; i < state.size(); ++i) {
            auto& l = state[i];
            if (idle[i] || l.decoding) { continue; }
            const auto& c = list[l.conversation];
            if (l.next_prefill < c.prefill.size()) {
                const auto& r = c.prefill[l.next_prefill++];
                replay.round({r.ids.data()}, {static_cast<std::size_t>(kTopK) * r.columns}, 0, nullptr);
                prefilled = true;
                if (l.next_prefill == c.prefill.size()) {
                    replay.refill();
                    l.decoding = true;
                    l.token    = 0;
                }
                break;
            }
        }
        if (prefilled) { continue; }
        // A decode round over every decoding lane.
        std::vector<const std::int32_t*> columns;
        std::vector<std::size_t> strides;
        std::uint32_t tokens = 0;
        std::uint32_t active = 0;
        for (std::size_t i = 0; i < state.size(); ++i) {
            if (idle[i]) { continue; }
            ++active;
            auto& l       = state[i];
            const auto& c = list[l.conversation];
            for (std::uint32_t w = 0; w < width && l.token + w < c.decode.size(); ++w) {
                columns.push_back(c.decode[l.token + w].ids.data());
                strides.push_back(kTopK);
            }
            const std::uint32_t step = std::min<std::uint32_t>(advance, static_cast<std::uint32_t>(c.decode.size() - l.token));
            tokens += step;
        }
        Totals* t = decode_rounds >= warmup && active == lanes ? &measured : nullptr;
        if (t != nullptr) {
            t->tokens += tokens;
            t->columns += columns.size();
        }
        replay.round(columns, strides, replay.decode_budget(tokens), t);
        ++decode_rounds;
        for (std::size_t i = 0; i < state.size(); ++i) {
            if (idle[i]) { continue; }
            auto& l       = state[i];
            const auto& c = list[l.conversation];
            l.token += advance;
            if (l.token >= c.decode.size()) { take(l, i); }
        }
    }
    const double rounds = static_cast<double>(measured.rounds);
    std::printf("frames %u lanes %u width %u advance %u interval %llu: rounds %llu, hit %.2f%%, misses/round %.2f, "
                "distinct experts/layer %.2f, layer calls with a miss %.1f%%, with >2 misses %.1f%%, promotions/round "
                "%.2f, tokens/round %.2f, columns/round %.2f\n",
                frames, lanes, width, advance, static_cast<unsigned long long>(interval),
                static_cast<unsigned long long>(measured.rounds),
                100.0 * measured.hits / std::max<std::uint64_t>(1, measured.routed), measured.misses / rounds,
                static_cast<double>(measured.distinct) / std::max<std::uint64_t>(1, measured.layer_calls),
                100.0 * measured.layer_calls_missing / std::max<std::uint64_t>(1, measured.layer_calls),
                100.0 * measured.layer_calls_over2 / std::max<std::uint64_t>(1, measured.layer_calls),
                measured.promotions / rounds, measured.tokens / rounds, measured.columns / rounds);
    return 0;
}
