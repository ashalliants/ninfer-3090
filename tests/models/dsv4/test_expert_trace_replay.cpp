// Written for NInfer-3090.
//
// Replays the routed-expert access sequence of DeepSeek-V4.1-Flash through the expert tier policy
// (src/models/dsv4/execution/expert_tiers.h, the same code FileExpertSource runs) and counts where each use is
// served from, for different eviction policies and tier sizes. Host only: no GPU, no artifact.
//
//   no argument                    a synthetic trace (a skewed hot set that drifts, plus prompt-sized scans): checks that
//                                  the frequency-aware default serves more uses from the tiers than plain LRU and
//                                  that replaying is deterministic.
//   NINFER_DSV4_TRACE=FILE         a trace FileExpertSource wrote (NINFER_DSV4_EXPERT_TRACE=FILE while the model ran,
//                                  e.g. through ninfer_dsv4_expert_trace_record): prints a table of hit rates.
//   NINFER_DSV4_TRACE_RAM_MIB / _VRAM_MIB / _RECORD_MB   tier sizes of the table's first line (defaults 4096, 12288, 8.5).
//
// Trace file (little endian): "NDXT", u32 version 1, u32 layers, u32 experts; then per call u16 layer, u16 n, u16 expert[n]
// (the distinct experts the call routed to).

#include "models/dsv4/execution/expert_tiers.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <map>
#include <random>
#include <string>
#include <vector>

using namespace ninfer::models::dsv4;

namespace {

int failures = 0;

void check(bool ok, const char* what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++failures;
    }
}

struct Call {
    std::uint16_t layer;
    std::vector<std::uint16_t> experts;
};

struct Trace {
    std::uint32_t layers = 0, experts = 0;
    std::vector<Call> calls;
};

bool load_trace(const char* path, Trace& trace) {
    std::FILE* f = std::fopen(path, "rb");
    if (f == nullptr) { return false; }
    char magic[4];
    std::uint32_t head[3];
    bool ok = std::fread(magic, 1, 4, f) == 4 && std::memcmp(magic, "NDXT", 4) == 0 && std::fread(head, 4, 3, f) == 3 && head[0] == 1;
    if (ok) {
        trace.layers  = head[1];
        trace.experts = head[2];
        for (;;) {
            std::uint16_t h[2];
            const std::size_t got = std::fread(h, 1, sizeof(h), f); // bytes: a one-byte tail is a truncated header too
            if (got == 0 && std::feof(f) != 0) { break; }           // a clean end between calls
            Call c;
            if (got != sizeof(h) || h[0] >= trace.layers) { ok = false; break; }
            c.layer = h[0];
            c.experts.resize(h[1]);
            if (std::fread(c.experts.data(), 2, h[1], f) != h[1] || std::any_of(c.experts.begin(), c.experts.end(), [&](std::uint16_t e) { return e >= trace.experts; })) {
                ok = false; // a truncated or corrupt call: do not report hit rates for a prefix
                break;
            }
            trace.calls.push_back(std::move(c));
        }
        ok = ok && std::ferror(f) == 0;
    }
    std::fclose(f);
    return ok && !trace.calls.empty();
}

struct Result {
    std::uint64_t uses = 0, vram = 0, ram = 0, miss = 0;
    std::uint64_t decode_uses = 0, decode_vram = 0, decode_ram = 0, decode_miss = 0;
    std::uint64_t promotions = 0;
};

// What FileExpertSource::prepare does with the policy, minus the copies (they finish before the next call).
Result replay(const Trace& trace, std::uint32_t ram_slots, std::uint32_t frames, const TierPolicyOptions& options, std::uint32_t max_promotions = 16) {
    ExpertTierPolicy policy(trace.layers * trace.experts, ram_slots, frames, options);
    Result r;
    std::vector<std::uint32_t> keys;
    for (const Call& call : trace.calls) {
        policy.begin_call();
        const bool decode = call.experts.size() <= 8;
        keys.clear();
        for (const auto e : call.experts) { keys.push_back(call.layer * trace.experts + e); }
        for (const auto k : keys) { policy.touch(k); }
        for (const auto k : keys) {
            ++r.uses;
            decode ? ++r.decode_uses : 0;
            if (policy.frame_of(k) >= 0) {
                ++r.vram;
                decode ? ++r.decode_vram : 0;
            } else if (policy.ram_of(k) >= 0) {
                ++r.ram;
                decode ? ++r.decode_ram : 0;
            } else {
                ++r.miss;
                decode ? ++r.decode_miss : 0;
                (void)policy.acquire_ram(k);
            }
        }
        std::uint32_t promoted = 0;
        for (const auto k : keys) {
            if (promoted >= max_promotions) { break; }
            if (policy.frame_of(k) >= 0 || policy.ram_of(k) < 0) { continue; }
            if (policy.admit_frame(k)) {
                ++promoted;
                ++r.promotions;
            }
        }
    }
    return r;
}

// Belady's optimum for one cache of `capacity` records over the same use sequence: the best any policy could
// do with RAM and VRAM pooled (a bound, not an achievable result).
std::uint64_t belady_hits(const Trace& trace, std::uint32_t capacity) {
    if (capacity == 0) { return 0; }
    std::vector<std::uint32_t> seq;
    for (const Call& c : trace.calls) {
        for (const auto e : c.experts) { seq.push_back(c.layer * trace.experts + e); }
    }
    const std::size_t n = seq.size();
    std::vector<std::uint32_t> next(n);
    std::map<std::uint32_t, std::uint32_t> last;
    for (std::size_t i = n; i-- > 0;) {
        const auto it = last.find(seq[i]);
        next[i]       = it == last.end() ? UINT32_MAX : it->second;
        last[seq[i]]  = static_cast<std::uint32_t>(i);
    }
    std::map<std::uint32_t, std::uint32_t> cache; // key -> next use
    std::map<std::uint32_t, std::uint32_t> by_next; // next use -> key (unique: one key per position)
    std::uint64_t hits = 0;
    for (std::size_t i = 0; i < n; ++i) {
        const auto it = cache.find(seq[i]);
        if (it != cache.end()) {
            ++hits;
            by_next.erase(it->second);
        } else if (cache.size() >= capacity) {
            const auto far = std::prev(by_next.end());
            if (far->first <= next[i]) { continue; } // the newcomer is needed last: do not cache it
            cache.erase(far->second);
            by_next.erase(far);
        }
        cache[seq[i]] = next[i];
        by_next[next[i]] = seq[i];
    }
    return hits;
}

Trace synthetic_trace() {
    Trace t;
    t.layers  = 6;
    t.experts = 64;
    std::mt19937 rng(7);
    // Each layer has a popularity ranking over its experts (Zipf 1.1); the ranking rotates every phase.
    std::vector<double> weight(t.experts);
    for (std::uint32_t e = 0; e < t.experts; ++e) { weight[e] = 1.0 / std::pow(double(e + 1), 1.1); }
    for (int phase = 0; phase < 4; ++phase) {
        for (int token = 0; token < 600; ++token) {
            for (std::uint32_t l = 0; l < t.layers; ++l) {
                std::discrete_distribution<int> pick(weight.begin(), weight.end());
                Call c;
                c.layer = static_cast<std::uint16_t>(l);
                while (c.experts.size() < 6) {
                    const auto e = static_cast<std::uint16_t>((pick(rng) + phase * 17 + l * 5) % t.experts);
                    if (std::find(c.experts.begin(), c.experts.end(), e) == c.experts.end()) { c.experts.push_back(e); }
                }
                t.calls.push_back(std::move(c));
            }
            if (token % 150 == 149) { // a prompt: one call per layer touching almost every expert
                for (std::uint32_t l = 0; l < t.layers; ++l) {
                    Call c;
                    c.layer = static_cast<std::uint16_t>(l);
                    for (std::uint32_t e = 0; e < t.experts; ++e) {
                        if (rng() % 10 != 0) { c.experts.push_back(static_cast<std::uint16_t>(e)); }
                    }
                    t.calls.push_back(std::move(c));
                }
            }
        }
    }
    return t;
}

const char* name_of(RamEviction e) {
    switch (e) {
    case RamEviction::Lru: return "lru";
    case RamEviction::Lfu: return "lfu";
    case RamEviction::Segmented: return "segmented";
    case RamEviction::Lfuda: return "lfuda";
    }
    return "?";
}

double pct(std::uint64_t a, std::uint64_t b) { return b == 0 ? 0.0 : 100.0 * double(a) / double(b); }

void print_row(const char* label, const Result& r) {
    std::printf("  %-34s vram %5.1f%%  ram %5.1f%%  miss %5.1f%%   decode: vram %5.1f%%  ram %5.1f%%  miss %5.1f%%   promotions %llu\n", label, pct(r.vram, r.uses),
                pct(r.ram, r.uses), pct(r.miss, r.uses), pct(r.decode_vram, r.decode_uses), pct(r.decode_ram, r.decode_uses),
                pct(r.decode_miss, r.decode_uses), static_cast<unsigned long long>(r.promotions));
}

void sweep(const Trace& trace, std::uint32_t ram_slots, std::uint32_t frames) {
    std::printf("tiers: %u RAM slots + %u VRAM frames, over %zu calls, %u layers x %u experts\n", ram_slots, frames, trace.calls.size(), trace.layers,
                trace.experts);
    for (const RamEviction e : {RamEviction::Lru, RamEviction::Lfu, RamEviction::Segmented, RamEviction::Lfuda}) {
        for (const bool exclusive : {false, true}) {
            TierPolicyOptions o;
            o.ram_eviction = e;
            o.exclusive    = exclusive;
            const std::string label = std::string(name_of(e)) + (exclusive ? " exclusive" : " inclusive");
            print_row(label.c_str(), replay(trace, ram_slots, frames, o));
        }
    }
    for (const std::uint32_t age : {1U << 10, 1U << 12, 1U << 16}) {
        TierPolicyOptions o;
        o.age_period            = age;
        const std::string label = "default, counts halved every " + std::to_string(age) + " calls";
        print_row(label.c_str(), replay(trace, ram_slots, frames, o));
    }
    for (const std::uint32_t margin : {1U, 2U, 4U}) {
        TierPolicyOptions o;
        o.frame_margin          = margin;
        const std::string label = "default, frame margin " + std::to_string(margin);
        print_row(label.c_str(), replay(trace, ram_slots, frames, o));
    }
    for (const std::uint32_t cap : {4U, 64U, 100000U}) {
        const std::string label = "default, at most " + std::to_string(cap) + " promotions per call";
        print_row(label.c_str(), replay(trace, ram_slots, frames, TierPolicyOptions{}, cap));
    }
    const Result pooled = replay(trace, ram_slots, frames, TierPolicyOptions{});
    std::printf("  upper bound (Belady, RAM+VRAM pooled): %.1f%% of uses served from a tier; the default reaches %.1f%%\n",
                pct(belady_hits(trace, ram_slots + frames), pooled.uses), pct(pooled.vram + pooled.ram, pooled.uses));
}

} // namespace

int main() {
    if (const char* path = std::getenv("NINFER_DSV4_TRACE"); path != nullptr && path[0] != '\0') {
        Trace trace;
        if (!load_trace(path, trace)) {
            std::fprintf(stderr, "cannot read the trace %s\n", path);
            return 1;
        }
        const auto env = [](const char* name, double fallback) {
            const char* v = std::getenv(name);
            return v != nullptr && v[0] != '\0' ? std::atof(v) : fallback;
        };
        const double record = env("NINFER_DSV4_TRACE_RECORD_MB", 8.5) * 1e6;
        const auto slots    = [&](double mib) { return static_cast<std::uint32_t>(mib * 1048576.0 / record); };
        sweep(trace, slots(env("NINFER_DSV4_TRACE_RAM_MIB", 4096)), slots(env("NINFER_DSV4_TRACE_VRAM_MIB", 12288)));
        for (const double vram : {4096.0, 8192.0, 16384.0}) { sweep(trace, slots(env("NINFER_DSV4_TRACE_RAM_MIB", 4096)), slots(vram)); }
        for (const double ram : {2048.0, 8192.0}) { sweep(trace, slots(ram), slots(env("NINFER_DSV4_TRACE_VRAM_MIB", 12288))); }
        return 0;
    }

    const Trace trace = synthetic_trace();
    TierPolicyOptions lru;
    lru.ram_eviction = RamEviction::Lru;
    lru.exclusive    = false;
    const TierPolicyOptions fresh;
    const std::uint32_t ram = 48, frames = 48;
    const Result a = replay(trace, ram, frames, lru), b = replay(trace, ram, frames, fresh), c = replay(trace, ram, frames, fresh);
    std::printf("synthetic trace, %zu calls: lru %.1f%% from tiers, default %.1f%%\n", trace.calls.size(), pct(a.vram + a.ram, a.uses),
                pct(b.vram + b.ram, b.uses));
    check(b.vram == c.vram && b.ram == c.ram && b.miss == c.miss, "a replay is deterministic");
    check(b.uses == a.uses && a.vram + a.ram + a.miss == a.uses, "every use is counted once");
    check(b.vram + b.ram > a.vram + a.ram, "the frequency-aware default serves more uses from the tiers than LRU");
    check(b.vram + b.ram <= belady_hits(trace, ram + frames), "no policy beats the offline optimum");
    check(belady_hits(trace, 0) == 0, "no capacity, no hits");
    if (failures == 0) { std::printf("PASS\n"); }
    return failures == 0 ? 0 : 1;
}
