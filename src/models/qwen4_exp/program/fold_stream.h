#pragma once

// --gdn-async-commit (FlashNextOptions::gdn_async_commit): the stream an MTP verification commit's
// GDN replay fold runs on. Basalt 8d73a39 commits the GDN state on a stream of its own and drains
// what is left at the next window's start (src/kernels/cuda/verify_kernels.cu:358-450); this is the
// plain form of that idea (no code taken from it): the whole fold on the side stream, and an event
// the device stream waits for before the state is used again.

#include <cuda_runtime.h>

#include <functional>

namespace ninfer::models::qwen4_exp {

// One fold in flight at most. run() forks the stream from the device stream (the fold follows the
// verification that wrote its records), enqueues the fold there and records `done`, even when the
// enqueue throws; join() makes the device stream wait for `done`. The Program joins before anything
// that reads or writes a lane's recurrent state or the verification records again. Neither call may
// run inside a stream capture: the Program forks and joins between its graphs.
class GdnFoldStream {
public:
    GdnFoldStream();
    // Waits for the fold in flight, then destroys the stream and its events.
    ~GdnFoldStream();
    GdnFoldStream(const GdnFoldStream&)            = delete;
    GdnFoldStream& operator=(const GdnFoldStream&) = delete;

    // Joins the fold before (at most one in flight), forks the side stream from `main`, runs
    // `enqueue(side)` and records its end. The fold is pending until join().
    void run(cudaStream_t main, const std::function<void(cudaStream_t)>& enqueue);
    // The device stream `main` waits for the pending fold (nothing when none is pending).
    void join(cudaStream_t main);
    // join() for noexcept paths: when the wait cannot be enqueued, the host waits for the side
    // stream instead (its error, if any, is the device's and surfaces at the next device call).
    void settle(cudaStream_t main) noexcept;
    [[nodiscard]] bool pending() const noexcept { return pending_; }

private:
    void release() noexcept;

    cudaStream_t stream_ = nullptr;
    cudaEvent_t fork_    = nullptr;
    cudaEvent_t done_    = nullptr;
    bool pending_        = false;
};

} // namespace ninfer::models::qwen4_exp
