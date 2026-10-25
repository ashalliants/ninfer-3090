#include "models/qwen4_exp/program/fold_stream.h"

#include "core/device.h"

namespace ninfer::models::qwen4_exp {

GdnFoldStream::GdnFoldStream() {
    try {
        // The default priority, as the device stream's: the fold fills what the drafter leaves idle.
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
        CUDA_CHECK(cudaEventCreateWithFlags(&fork_, cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(&done_, cudaEventDisableTiming));
    } catch (...) {
        release();
        throw;
    }
}

GdnFoldStream::~GdnFoldStream() {
    if (stream_ != nullptr) { (void)cudaStreamSynchronize(stream_); }
    release();
}

void GdnFoldStream::release() noexcept {
    if (done_ != nullptr) { (void)cudaEventDestroy(done_); }
    if (fork_ != nullptr) { (void)cudaEventDestroy(fork_); }
    if (stream_ != nullptr) { (void)cudaStreamDestroy(stream_); }
    done_ = fork_ = nullptr;
    stream_       = nullptr;
}

void GdnFoldStream::run(cudaStream_t main, const std::function<void(cudaStream_t)>& enqueue) {
    join(main);
    CUDA_CHECK(cudaEventRecord(fork_, main));
    CUDA_CHECK(cudaStreamWaitEvent(stream_, fork_, 0));
    // From here whatever reached the side stream is waited for: `done` follows it even when the
    // enqueue throws halfway.
    pending_ = true;
    try {
        enqueue(stream_);
    } catch (...) {
        if (cudaEventRecord(done_, stream_) != cudaSuccess) { (void)cudaStreamSynchronize(stream_); }
        throw;
    }
    if (const cudaError_t recorded = cudaEventRecord(done_, stream_); recorded != cudaSuccess) {
        (void)cudaStreamSynchronize(stream_);
        CUDA_CHECK(recorded);
    }
}

void GdnFoldStream::join(cudaStream_t main) {
    if (!pending_) { return; }
    CUDA_CHECK(cudaStreamWaitEvent(main, done_, 0));
    pending_ = false;
}

void GdnFoldStream::settle(cudaStream_t main) noexcept {
    if (!pending_) { return; }
    pending_ = false;
    if (cudaStreamWaitEvent(main, done_, 0) != cudaSuccess) { (void)cudaStreamSynchronize(stream_); }
}

} // namespace ninfer::models::qwen4_exp
