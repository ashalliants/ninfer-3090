#pragma once

// The CPU miss service's error word as the Program uses it (--cpu-verify-misses,
// --prefill-cpu-share). A layer whose CPU leg timed out writes layer << 16 | code to one mapped
// word, shared by verification rounds and prefill calls, and the Program reads it after the call
// synchronized. A call that throws between its launch and that read leaves its code behind; without
// care it would fail the next healthy call ("did not answer"). So every call that may hand the
// service jobs arms the word first: the code an unchecked earlier call left is dropped, after the
// device has finished that call's kernels (they could still write it).
//
// Host only, no CUDA: the Program passes the word's host address and a function that synchronizes
// the device.

#include <atomic>
#include <cstdint>

namespace ninfer::models::qwen4_exp {

class CpuErrorWord {
public:
    CpuErrorWord() = default;
    explicit CpuErrorWord(std::uint32_t* word) noexcept : word_(word) {}

    [[nodiscard]] bool bound() const noexcept { return word_ != nullptr; }

    // Before launching a call that may hand the service jobs. `synchronize` runs first when an
    // earlier armed call was never checked (it threw), so none of its kernels can write the word
    // after it is cleared. `prefill`: the call is a prefill or forced-token call handing its share,
    // which take_prefill() then checks.
    template <class Synchronize>
    void arm(bool prefill, Synchronize&& synchronize) {
        if (word_ == nullptr) { return; }
        if (armed_) { synchronize(); }
        ref().store(0, std::memory_order_relaxed);
        armed_   = true;
        prefill_ = prefill;
    }

    // Before launching a prefill or forced-token call that hands the service no share:
    // take_prefill() after it returns 0. A code an unchecked earlier call left stays in the word
    // until the next arm() drops it.
    void skip() noexcept { prefill_ = false; }

    // After the armed call synchronized: its code (0: every CPU leg answered), and the word clear.
    std::uint32_t take() noexcept {
        if (word_ == nullptr) { return 0; }
        armed_   = false;
        prefill_ = false;
        return ref().exchange(0, std::memory_order_relaxed);
    }

    // After a prefill or forced-token call synchronized: take() when that call handed its share,
    // else 0 (a verification checks its own round).
    std::uint32_t take_prefill() noexcept { return prefill_ ? take() : 0; }

    // Whether an armed call has not been checked yet.
    [[nodiscard]] bool armed() const noexcept { return armed_; }
    [[nodiscard]] bool prefill_armed() const noexcept { return prefill_; }

private:
    std::atomic_ref<std::uint32_t> ref() const noexcept { return std::atomic_ref<std::uint32_t>(*word_); }

    std::uint32_t* word_ = nullptr;
    bool armed_          = false;
    bool prefill_        = false;
};

} // namespace ninfer::models::qwen4_exp
