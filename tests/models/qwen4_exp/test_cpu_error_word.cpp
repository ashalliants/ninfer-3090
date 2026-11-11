// The CPU miss service's error word as the Program arms and reads it (CPU only, header only): a code
// a call left when it threw between its launch and its check must not fail the next healthy
// verification or prefill call, and a real code still reaches the call that set it.
#include "models/qwen4_exp/program/cpu_error_word.h"

#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>

using namespace ninfer::models::qwen4_exp;

namespace {

void require(bool condition, const std::string& message) {
    if (!condition) { throw std::runtime_error(message); }
}

constexpr std::uint32_t kTimedOut = (7U << 16U) | 0x0002U; // layer 7's CPU leg timed out

struct Device {
    std::uint32_t word = 0; // the mapped word a timed-out CPU leg writes
    int synchronized   = 0;
    auto sync() {
        return [this] { ++synchronized; };
    }
};

void a_code_reaches_its_call() {
    Device d;
    CpuErrorWord w(&d.word);
    w.arm(true, d.sync());
    d.word = kTimedOut; // the call's layer 7 timed out
    require(w.take_prefill() == kTimedOut, "the prefill call sees its own code");
    require(d.word == 0 && !w.armed() && !w.prefill_armed(), "the word is clear after the check");
    require(d.synchronized == 0, "a checked call costs no synchronization");

    w.arm(false, d.sync());
    d.word = kTimedOut;
    require(w.take() == kTimedOut, "a verification sees its own code");
    require(d.synchronized == 0, "still no synchronization");
}

void a_thrown_call_does_not_fail_the_next_prefill() {
    Device d;
    CpuErrorWord w(&d.word);
    w.arm(true, d.sync());
    d.word = kTimedOut; // ... and the call threw before check_cpu_prefill
    w.arm(true, d.sync());
    require(d.synchronized == 1, "the unchecked call's kernels finish before the word is cleared");
    require(d.word == 0, "the stale code is dropped");
    require(w.take_prefill() == 0, "the healthy prefill call passes");
}

void a_thrown_prefill_does_not_fail_the_next_verification() {
    Device d;
    CpuErrorWord w(&d.word);
    w.arm(true, d.sync());
    d.word = kTimedOut; // the prefill call threw
    w.arm(false, d.sync());
    require(d.synchronized == 1, "synchronized once before the verification");
    require(w.take() == 0, "the healthy verification passes");
    require(w.take_prefill() == 0, "and no prefill check is pending");
}

void a_thrown_verification_does_not_fail_the_next_prefill() {
    Device d;
    CpuErrorWord w(&d.word);
    w.arm(false, d.sync());
    d.word = kTimedOut; // the verification threw before check_cpu_verify
    w.arm(true, d.sync());
    require(d.synchronized == 1 && w.take_prefill() == 0, "the healthy prefill call passes");
}

void a_call_without_a_share_is_not_checked() {
    Device d;
    CpuErrorWord w(&d.word);
    w.arm(true, d.sync());
    d.word = kTimedOut; // the share call threw
    // The next prefill call is too wide for the channel: it hands no share and must not read the
    // stale code.
    w.skip();
    require(w.take_prefill() == 0, "a call without a share passes");
    require(w.armed(), "the stale code is still unchecked");
    w.arm(true, d.sync());
    require(d.synchronized == 1 && d.word == 0, "the next share call drops it after synchronizing");
    require(w.take_prefill() == 0, "and passes");
}

void unbound_is_inert() {
    CpuErrorWord w;
    int synchronized = 0;
    w.arm(true, [&] { ++synchronized; });
    w.arm(true, [&] { ++synchronized; });
    require(!w.bound() && !w.armed() && synchronized == 0, "no service: nothing armed");
    require(w.take() == 0 && w.take_prefill() == 0, "no service: no code");
}

} // namespace

int main() {
    try {
        a_code_reaches_its_call();
        a_thrown_call_does_not_fail_the_next_prefill();
        a_thrown_prefill_does_not_fail_the_next_verification();
        a_thrown_verification_does_not_fail_the_next_prefill();
        a_call_without_a_share_is_not_checked();
        unbound_is_inert();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL: %s\n", e.what());
        return 1;
    }
    std::puts("ninfer_qwen4_exp_cpu_error_word_test: ok");
    return 0;
}
