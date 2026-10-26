#include "serve/slots_report.h"

#include <iostream>
#include <limits>
#include <string>

namespace {

using ninfer::RuntimeLaneStats;
using ninfer::RuntimeStats;
using ninfer::RuntimeWaitingStats;

int check(bool condition, const std::string& message) {
    if (condition) { return 0; }
    std::cerr << message << '\n';
    return 1;
}

int expect_json(const std::string& actual, const std::string& expected, const char* what) {
    return check(actual == expected,
                 std::string(what) + "\n  expected: " + expected + "\n  actual:   " + actual);
}

// Three lanes, one per state, and two waiting requests, one more than were listed.
RuntimeStats fixed_stats() {
    RuntimeStats stats;
    stats.lane_count = 3;
    stats.lanes[0] = RuntimeLaneStats{.state                    = RuntimeLaneStats::State::Prefill,
                                      .request_id               = 7,
                                      .prompt_tokens            = 4096,
                                      .computed_prompt_tokens   = 1024,
                                      .reused_prompt_tokens     = 2048,
                                      .generated_tokens         = 0,
                                      .context_tokens           = 3072,
                                      .decode_tokens_per_second = 0.0,
                                      .drafted_tokens           = 0,
                                      .accepted_tokens          = 0,
                                      .relaxed_kept             = 0};
    stats.lanes[1] = RuntimeLaneStats{.state                    = RuntimeLaneStats::State::Decoding,
                                      .request_id               = 5,
                                      .prompt_tokens            = 100,
                                      .computed_prompt_tokens   = 100,
                                      .reused_prompt_tokens     = 0,
                                      .generated_tokens         = 41,
                                      .context_tokens           = 141,
                                      .decode_tokens_per_second = 62.5,
                                      .drafted_tokens           = 48,
                                      .accepted_tokens          = 30,
                                      .relaxed_kept             = 2};
    // Lane 2 is idle and lane 3 is past lane_count: it must not be listed.
    stats.lanes[3].state      = RuntimeLaneStats::State::Decoding;
    stats.lanes[3].request_id = 99;
    stats.waiting             = {RuntimeWaitingStats{.request_id = 8, .prompt_tokens = 512,
                                                     .waited_seconds = 1.25}};
    stats.waiting_requests    = 2;
    return stats;
}

} // namespace

int main() {
    int failures = 0;

    // render_slots_shape: the exact body for a fixed snapshot.
    failures += expect_json(
        ninfer::serve::render_slots_json(fixed_stats()),
        R"({"slots":[)"
        R"({"id":0,"state":"prefill","request_id":7,"prompt_tokens":4096,"prompt_done":3072,)"
        R"("reused":2048,"generated":0,"n_ctx_used":3072,"tokens_per_second":0.0,"drafted":0,)"
        R"("accepted":0,"relaxed_kept":0},)"
        R"({"id":1,"state":"decoding","request_id":5,"prompt_tokens":100,"prompt_done":100,)"
        R"("reused":0,"generated":41,"n_ctx_used":141,"tokens_per_second":62.5,"drafted":48,)"
        R"("accepted":30,"relaxed_kept":2},)"
        R"({"id":2,"state":"idle","request_id":null,"prompt_tokens":0,"prompt_done":0,)"
        R"("reused":0,"generated":0,"n_ctx_used":0,"tokens_per_second":0.0,"drafted":0,)"
        R"("accepted":0,"relaxed_kept":0}],)"
        R"("queue":[{"id":8,"prompt_tokens":512,"waited_s":1.25}],"queue_total":2})",
        "render_slots_shape: the /slots body for a fixed snapshot");

    // An Engine that has not published yet (or a scoring Engine) has no lanes.
    failures += expect_json(ninfer::serve::render_slots_json(RuntimeStats{}),
                            R"({"slots":[],"queue":[],"queue_total":0})",
                            "an empty snapshot renders empty lists");

    // JSON has no NaN or infinity: a rate that is not finite is reported as 0.
    RuntimeStats odd = fixed_stats();
    odd.lane_count   = 2;
    odd.lanes[1].decode_tokens_per_second = std::numeric_limits<double>::infinity();
    odd.waiting[0].waited_seconds         = std::numeric_limits<double>::quiet_NaN();
    const std::string body                = ninfer::serve::render_slots_json(odd);
    failures += check(body.find(R"("tokens_per_second":0.0,"drafted":48)") != std::string::npos &&
                          body.find(R"("waited_s":0.0})") != std::string::npos,
                      "a non-finite rate or wait was not rendered as 0: " + body);

    if (failures == 0) { std::cout << "slots report: ok\n"; }
    return failures == 0 ? 0 : 1;
}
