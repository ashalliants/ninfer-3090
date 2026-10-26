// The contract of wait_flag_ge / wait_flag_sequence (core/host_flag_wait.h): the stream-ordered
// wait for a flag in mapped host memory that the Flash-Next round graph's n-gram rows arrive
// behind (--ngram-flag-wait).
//
//   raised_later:     a host thread raises the flag after a delay; the wait holds back the stream
//                     work behind it until then (a D2H marker stays unwritten), then completes with
//                     the error word untouched.
//   already_reached:  a flag at or past the value (wrap-aware) releases at once.
//   never_raised:     the wait gives up after kWaitFlagTimeoutNs, writes kWaitFlagErrorTimeout and
//                     the stream continues.
//   stale_sequence:   wait_flag_sequence wants the counter's next value: a flag left at the
//                     previous round's value holds the next round back until it is raised again;
//                     the counter advances once per call, across the 2^32 wrap too.
//   replayed_graph:   one captured wait_flag_sequence waits for a new value at each replay.
//
// The expected values follow from the contract alone: the flag's value, the counter, the timeout.

#include "core/arena.h"
#include "core/host_flag_wait.h"

#include <cuda_runtime.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>

using namespace ninfer;

namespace {

void cuda_check(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

void cuda_synchronize(cudaStream_t stream) {
    cuda_check(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
}

bool cuda_unavailable() {
    int n               = 0;
    const cudaError_t e = cudaGetDeviceCount(&n);
    if (e == cudaSuccess) { return n == 0; }
    if (e == cudaErrorNoDevice || e == cudaErrorInsufficientDriver) { return true; }
    throw std::runtime_error(std::string("cudaGetDeviceCount: ") + cudaGetErrorString(e));
}

using Clock = std::chrono::steady_clock;

int failures = 0;

void check(bool ok, const std::string& what) {
    if (!ok) {
        std::cout << "FAIL: " << what << "\n";
        ++failures;
    }
}

double seconds_since(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}

// The mapped words: the flag, the error word and the marker the stream writes after the wait (each
// on its own 64-byte line), with their device addresses; the device marker source and the counter.
struct Words {
    PinnedHostBuffer host{256};
    std::uint32_t* flag   = nullptr;
    std::uint32_t* error  = nullptr;
    std::uint32_t* marker = nullptr;
    std::uint32_t* flag_dev  = nullptr;
    std::uint32_t* error_dev = nullptr;
    DeviceBuffer source{sizeof(std::uint32_t)};
    DeviceBuffer sequence{sizeof(std::uint32_t)};

    Words() {
        auto* words = static_cast<std::uint32_t*>(host.data());
        flag        = words;
        error       = words + 16;
        marker      = words + 32;
        void* mapped = nullptr;
        cuda_check(cudaHostGetDevicePointer(&mapped, host.data(), 0), "cudaHostGetDevicePointer");
        flag_dev  = static_cast<std::uint32_t*>(mapped);
        error_dev = flag_dev + 16;
        const std::uint32_t mark = 0xC0FFEE;
        source.copy_from_host(&mark, sizeof(mark));
        set_sequence(0);
        reset(0);
    }
    void reset(std::uint32_t value) {
        store(flag, value);
        store(error, 0);
        store(marker, 0);
    }
    // The marker back to "not yet written" between rounds (the flag keeps its value).
    void clear_marker() { store(marker, 0); }
    void set_sequence(std::uint32_t value) { sequence.copy_from_host(&value, sizeof(value)); }
    std::uint32_t read_sequence() const {
        std::uint32_t value = 0;
        sequence.copy_to_host(&value, sizeof(value));
        return value;
    }
    static void store(std::uint32_t* word, std::uint32_t value) {
        std::atomic_ref<std::uint32_t>(*word).store(value, std::memory_order_release);
    }
    static std::uint32_t load(std::uint32_t* word) {
        return std::atomic_ref<std::uint32_t>(*word).load(std::memory_order_acquire);
    }
    // The stream work behind a wait: the marker copied to mapped host memory.
    void enqueue_marker(cudaStream_t stream) {
        cuda_check(cudaMemcpyAsync(marker, source.p, sizeof(std::uint32_t),
                                   cudaMemcpyDeviceToHost, stream),
                   "marker copy");
    }
    [[nodiscard]] bool marked() const { return load(marker) == 0xC0FFEE; }
    // The wait released the work behind it and wrote no error.
    [[nodiscard]] bool clean() const { return marked() && load(error) == 0; }
};

// Raises the flag to `value` after `delay`, on its own thread (the host side of the protocol).
std::thread raise_later(std::uint32_t* flag, std::uint32_t value, std::chrono::milliseconds delay) {
    return std::thread([flag, value, delay] {
        std::this_thread::sleep_for(delay);
        Words::store(flag, value);
    });
}

// While the flag is low the work behind the wait must not have run.
void check_held(Words& w, cudaStream_t stream, const std::string& name) {
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    check(cudaStreamQuery(stream) == cudaErrorNotReady,
          name + ": the stream finished before the flag was raised");
    check(!w.marked(), name + ": work behind the wait ran before the flag was raised");
}

void raised_later(cudaStream_t stream) {
    Words w;
    const auto start = Clock::now();
    wait_flag_ge(w.flag_dev, 1, w.error_dev, stream);
    w.enqueue_marker(stream);
    std::thread host = raise_later(w.flag, 1, std::chrono::milliseconds(200));
    check_held(w, stream, "raised_later");
    host.join();
    cuda_synchronize(stream);
    const double took = seconds_since(start);
    check(w.marked(), "raised_later: the work behind the wait did not run");
    check(Words::load(w.error) == 0, "raised_later: the error word was written");
    check(took >= 0.19 && took < 0.9,
          "raised_later: the wait took " + std::to_string(took) + " s, not the delay");
    std::cout << "raised_later: released after " << took << " s\n";
}

void already_reached(cudaStream_t stream) {
    Words w;
    struct Case {
        std::uint32_t flag, value;
    };
    for (const Case c : {Case{1, 1}, Case{7, 3}, Case{2, 0xFFFFFFF0U}, Case{0x7FFFFFFFU, 0}}) {
        w.reset(c.flag);
        const auto start = Clock::now();
        wait_flag_ge(w.flag_dev, c.value, w.error_dev, stream);
        w.enqueue_marker(stream);
        cuda_synchronize(stream);
        const std::string name =
            "already_reached flag " + std::to_string(c.flag) + " value " + std::to_string(c.value);
        check(w.clean(), name + ": did not complete cleanly");
        check(seconds_since(start) < 0.5, name + ": waited");
    }
    // Below the value, wrap-aware: 0xFFFFFFFF is one short of 0.
    for (const Case c : {Case{0xFFFFFFFFU, 0}, Case{4, 5}}) {
        w.reset(c.flag);
        wait_flag_ge(w.flag_dev, c.value, w.error_dev, stream);
        w.enqueue_marker(stream);
        const std::string name =
            "below flag " + std::to_string(c.flag) + " value " + std::to_string(c.value);
        std::thread host = raise_later(w.flag, c.value, std::chrono::milliseconds(100));
        check_held(w, stream, name);
        host.join();
        cuda_synchronize(stream);
        check(w.clean(), name + ": did not complete cleanly once raised");
    }
    std::cout << "already_reached: reached flags release at once, lower ones (wrap-aware) wait\n";
}

void never_raised(cudaStream_t stream) {
    Words w;
    const auto start = Clock::now();
    wait_flag_ge(w.flag_dev, 5, w.error_dev, stream);
    w.enqueue_marker(stream);
    cuda_synchronize(stream);
    const double took   = seconds_since(start);
    const double target = static_cast<double>(kWaitFlagTimeoutNs) * 1e-9;
    check(Words::load(w.error) == kWaitFlagErrorTimeout,
          "never_raised: the error word does not hold the timeout");
    check(w.marked(), "never_raised: the stream did not continue after the timeout");
    check(took >= 0.9 * target && took < target + 1.0,
          "never_raised: gave up after " + std::to_string(took) + " s, not the timeout");
    std::cout << "never_raised: kWaitFlagErrorTimeout after " << took << " s\n";
}

void stale_sequence(cudaStream_t stream) {
    Words w;
    auto* sequence = static_cast<std::uint32_t*>(w.sequence.p);
    // Round 1 wants 1.
    wait_flag_sequence(w.flag_dev, sequence, w.error_dev, stream);
    w.enqueue_marker(stream);
    std::thread first = raise_later(w.flag, 1, std::chrono::milliseconds(100));
    check_held(w, stream, "stale_sequence round 1");
    first.join();
    cuda_synchronize(stream);
    check(w.clean(), "stale_sequence round 1: did not complete cleanly");
    check(w.read_sequence() == 1, "stale_sequence: the counter is not 1 after one call");
    // Round 2 with the flag still at round 1's value: it must wait for 2.
    w.clear_marker();
    wait_flag_sequence(w.flag_dev, sequence, w.error_dev, stream);
    w.enqueue_marker(stream);
    std::thread second = raise_later(w.flag, 2, std::chrono::milliseconds(100));
    check_held(w, stream, "stale_sequence round 2 (flag left at 1)");
    second.join();
    cuda_synchronize(stream);
    check(w.clean(), "stale_sequence round 2: did not complete cleanly");
    check(w.read_sequence() == 2, "stale_sequence: the counter is not 2 after two calls");
    // Across the wrap: the counter at 0xFFFFFFFF wants 0, and the flag at 0xFFFFFFFF holds it back.
    w.set_sequence(0xFFFFFFFFU);
    w.reset(0xFFFFFFFFU);
    wait_flag_sequence(w.flag_dev, sequence, w.error_dev, stream);
    w.enqueue_marker(stream);
    std::thread wrap = raise_later(w.flag, 0, std::chrono::milliseconds(100));
    check_held(w, stream, "stale_sequence across the wrap");
    wrap.join();
    cuda_synchronize(stream);
    check(w.clean(), "stale_sequence across the wrap: did not complete cleanly");
    check(w.read_sequence() == 0, "stale_sequence: the counter did not wrap to 0");
    std::cout << "stale_sequence: each call waits for the counter's next value\n";
}

void replayed_graph(cudaStream_t stream) {
    Words w;
    auto* sequence = static_cast<std::uint32_t*>(w.sequence.p);
    cudaGraph_t graph = nullptr;
    cuda_check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal), "begin capture");
    wait_flag_sequence(w.flag_dev, sequence, w.error_dev, stream);
    w.enqueue_marker(stream);
    cuda_check(cudaStreamEndCapture(stream, &graph), "end capture");
    cudaGraphExec_t exec = nullptr;
    cuda_check(cudaGraphInstantiate(&exec, graph, 0), "instantiate");
    for (std::uint32_t replay = 1; replay <= 3; ++replay) {
        w.clear_marker();
        cuda_check(cudaGraphLaunch(exec, stream), "launch");
        std::thread host = raise_later(w.flag, replay, std::chrono::milliseconds(100));
        check_held(w, stream, "replayed_graph replay " + std::to_string(replay));
        host.join();
        cuda_synchronize(stream);
        check(w.clean(),
              "replayed_graph replay " + std::to_string(replay) + ": did not complete cleanly");
    }
    check(w.read_sequence() == 3, "replayed_graph: the counter is not 3 after three replays");
    cuda_check(cudaGraphExecDestroy(exec), "exec destroy");
    cuda_check(cudaGraphDestroy(graph), "graph destroy");
    std::cout << "replayed_graph: three replays of one captured wait, each held until its value\n";
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    cudaStream_t stream = nullptr;
    cuda_check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "cudaStreamCreate");
    try {
        raised_later(stream);
        already_reached(stream);
        never_raised(stream);
        stale_sequence(stream);
        replayed_graph(stream);
    } catch (const std::exception& error) {
        std::cout << "FAIL: " << error.what() << "\n";
        ++failures;
    }
    cuda_check(cudaStreamDestroy(stream), "cudaStreamDestroy");
    std::cout << (failures == 0 ? "OK" : "FAIL") << " host flag wait contract\n";
    return failures == 0 ? 0 : 1;
}
