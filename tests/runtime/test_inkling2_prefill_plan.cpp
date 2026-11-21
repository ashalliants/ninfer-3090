// Written for NInfer-3090.
//
// Inkling-2's prefill partition: a prompt resumed from any snapshot a previous prompt can leave runs, from the resume
// point on, exactly the forward calls (start and width) of a cold prefill, at every chunk the Engine picks. The
// tensor-core routes dispatch on a call's width, so this is what makes a resumed prefill bit-equal to a cold one.

#include "runtime/engine/inkling2_prefill_plan.h"

#include <cstdint>
#include <cstdio>
#include <utility>
#include <vector>

namespace {

using ninfer::runtime::inkling2::prefill_call_width;
using ninfer::runtime::inkling2::prefill_snapshot_at;

using Calls = std::vector<std::pair<std::int32_t, std::int32_t>>; // (start, width)

Calls plan(std::int32_t from, std::int32_t n, std::int32_t chunk) {
    Calls calls;
    for (std::int32_t s = from, len = 0; s < n; s += len) {
        len = prefill_call_width(s, n, chunk);
        calls.emplace_back(s, len);
    }
    return calls;
}

} // namespace

int main() {
    int failures = 0;
    const auto check = [&](bool ok, const char* what, std::int32_t chunk, std::int32_t previous, std::int32_t n) {
        if (!ok) {
            std::fprintf(stderr, "FAILED: %s (chunk %d, previous prompt %d, prompt %d)\n", what, chunk, previous, n);
            ++failures;
        }
    };
    for (const std::int32_t chunk : {256, 512, 1024, 2048}) {
        for (std::int32_t previous = 1; previous <= 3 * chunk + 3; previous += 7) {
            const std::int32_t resume = prefill_snapshot_at(previous, chunk);
            check(resume % chunk == 0 && resume < previous, "a snapshot sits on a chunk boundary inside its prompt", chunk, previous, 0);
            // The previous prompt's own cold prefill cuts at its snapshot.
            bool cut = resume == 0;
            for (const auto& [s, len] : plan(0, previous, chunk)) { cut = cut || s + len == resume; }
            check(cut, "the previous prompt's forwards end at its snapshot", chunk, previous, 0);
            for (std::int32_t n = resume + 1; n <= resume + 2 * chunk + 1; n += 5) {
                const Calls cold = plan(0, n, chunk), resumed = plan(resume, n, chunk);
                Calls tail;
                for (const auto& call : cold) {
                    if (call.first >= resume) { tail.push_back(call); }
                }
                check(resumed == tail, "a resumed prefill runs a cold one's calls from the resume point", chunk, previous, n);
            }
        }
    }
    // The case the 256-position snapshot got wrong at chunk 512: a 512-token prompt leaves no snapshot (it would
    // resume [256,768) where a cold 1024-token prefill runs [0,512) [512,768) [768,1024)), and a 513-token one leaves 512.
    check(prefill_snapshot_at(512, 512) == 0 && prefill_snapshot_at(513, 512) == 512, "chunk 512 snapshots at 0 and 512", 512, 512, 1024);
    check(plan(0, 1024, 512) == Calls{{0, 512}, {512, 512}}, "a cold 1024-token prefill at chunk 512 runs two whole chunks", 512, 0, 1024);
    if (failures != 0) {
        std::fprintf(stderr, "FAIL: %d checks\n", failures);
        return 1;
    }
    std::printf("PASS\n");
    return 0;
}
