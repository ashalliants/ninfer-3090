// Adapted from Infernix a3edb450 src/models/qwen4_exp/execution/expert_stream.cpp (Apache-2.0).
// Modified for NInfer-3090: see expert_stream.h (frame tables instead of slot tables, 2-D run
// copies, no gated chunks or SSD holds).

#include "models/qwen4_exp/execution/expert_stream.h"

#include "core/device.h"
#include "core/pinned_upload.h"
#include "ops/offloaded_sparse_moe/cpu_share.h"

#include <algorithm>
#include <atomic>
#include <span>
#include <stdexcept>

namespace ninfer::models::qwen4_exp::execution {
namespace {
std::atomic<std::uint64_t> g_cold_experts{0}; // ExpertStream::process_cold_experts
} // namespace

std::uint64_t ExpertStream::process_cold_experts() noexcept { return g_cold_experts.load(std::memory_order_relaxed); }

ExpertStream::ExpertStream(std::uint32_t layers, std::uint32_t experts, std::vector<Bank> banks,
                           std::uint8_t* frame_base, std::uint64_t frame_stride)
    : layers_(layers), experts_(experts), banks_(std::move(banks)), frame_base_(frame_base),
      frame_stride_(frame_stride) {
    if (layers_ == 0 || experts_ == 0 || banks_.size() != layers_ || frame_base_ == nullptr || frame_stride_ == 0) {
        throw std::invalid_argument("expert stream: empty geometry");
    }
    for (const auto& bank : banks_) {
        if (bank.records == nullptr || bank.stride == 0 || bank.stride > frame_stride_) {
            throw std::invalid_argument("expert stream: a bank's records do not fit a frame");
        }
    }
    tables_host_.assign(static_cast<std::size_t>(layers_) * experts_, -1);
    CUDA_CHECK(cudaHostAlloc(&tables_pinned_, table_bytes(), cudaHostAllocDefault));
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
    CUDA_CHECK(cudaEventCreateWithFlags(&start_, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&uploaded_, cudaEventDisableTiming));
    landed_.resize(layers_);
    consumed_.resize(layers_);
    for (std::uint32_t l = 0; l < layers_; ++l) {
        CUDA_CHECK(cudaEventCreateWithFlags(&landed_[l], cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(&consumed_[l], cudaEventDisableTiming));
    }
    streams_.assign(layers_, 0);
}

ExpertStream::~ExpertStream() {
    if (stream_ != nullptr) { (void)cudaStreamSynchronize(stream_); }
    if (uploading_) { (void)cudaEventSynchronize(uploaded_); }
    for (auto event : landed_) { (void)cudaEventDestroy(event); }
    for (auto event : consumed_) { (void)cudaEventDestroy(event); }
    if (uploaded_ != nullptr) { (void)cudaEventDestroy(uploaded_); }
    if (start_ != nullptr) { (void)cudaEventDestroy(start_); }
    if (stream_ != nullptr) { (void)cudaStreamDestroy(stream_); }
    if (tables_pinned_ != nullptr) { (void)cudaFreeHost(tables_pinned_); }
}

std::size_t ExpertStream::table_bytes() const noexcept {
    return static_cast<std::size_t>(layers_) * experts_ * sizeof(std::int32_t);
}

void ExpertStream::begin(const std::int32_t* residency, std::int32_t* tables, std::uint32_t ring_first,
                         std::uint32_t ring_frames, cudaStream_t compute, Share share) {
    if (active_) { throw std::logic_error("expert stream: a chunk is already streaming"); }
    if (ring_frames < 2 || tables == nullptr) { throw std::invalid_argument("expert stream: the ring holds no frames"); }
    if (share.num != 0 && share.counts == nullptr) { throw std::invalid_argument("expert stream: a share without counts"); }
    half_       = ring_frames / 2;
    ring_first_ = ring_first;
    tables_     = tables;
    // The previous chunk's table upload may still read the pinned copy rewritten below.
    if (uploading_) { CUDA_CHECK(cudaEventSynchronize(uploaded_)); }
    left_cold_ = false;
    cold_.assign(experts_, 0);
    for (std::uint32_t l = 0; l < layers_; ++l) {
        std::int32_t* row          = tables_host_.data() + static_cast<std::size_t>(l) * experts_;
        const std::int32_t* frames = residency + static_cast<std::size_t>(l) * experts_;
        std::uint32_t taken        = 0;
        if (share.num != 0) {
            // The layer's predicted-cold misses stay -1 (the MoE stages them or the CPU takes them).
            std::fill(cold_.begin(), cold_.end(), 0);
            const auto cold = ops::offloaded_moe::select_stream_cold(
                std::span(frames, experts_), std::span(share.counts + static_cast<std::size_t>(l) * experts_, experts_),
                static_cast<int>(half_), share.num, share.den);
            for (const std::int32_t e : cold) { cold_[static_cast<std::size_t>(e)] = 1; }
            stats_.cold += cold.size();
            g_cold_experts.fetch_add(cold.size(), std::memory_order_relaxed);
            left_cold_ = left_cold_ || !cold.empty();
        }
        for (std::uint32_t e = 0; e < experts_; ++e) {
            if (frames[e] >= 0) {
                row[e] = frames[e];
            } else if (cold_[e] != 0) {
                row[e] = -1;
            } else if (taken < half_) {
                row[e] = static_cast<std::int32_t>(ring_first_ + (l % 2) * half_ + taken++);
            } else {
                row[e] = -1;
                ++stats_.left;
            }
        }
        streams_[l] = taken > 0 ? 1 : 0;
    }
    std::copy(tables_host_.begin(), tables_host_.end(), static_cast<std::int32_t*>(tables_pinned_));
    // SM-read upload: it must not queue behind copies on a copy engine.
    upload_pinned(tables_, tables_pinned_, table_bytes(), compute);
    CUDA_CHECK(cudaEventRecord(uploaded_, compute));
    uploading_ = true;
    // The ring's previous readers (the last chunk's experts, the lender's last use) come first.
    CUDA_CHECK(cudaEventRecord(start_, compute));
    CUDA_CHECK(cudaStreamWaitEvent(stream_, start_, 0));
    active_ = true;
    enqueue(0);
    if (layers_ > 1) { enqueue(1); }
}

const std::int32_t* ExpertStream::table(std::uint32_t layer) const noexcept {
    return tables_ + static_cast<std::size_t>(layer) * experts_;
}

void ExpertStream::enqueue(std::uint32_t layer) {
    if (layer >= 2) { CUDA_CHECK(cudaStreamWaitEvent(stream_, consumed_[layer - 2], 0)); }
    const std::int32_t* row = tables_host_.data() + static_cast<std::size_t>(layer) * experts_;
    const std::uint32_t lo  = ring_first_ + (layer % 2) * half_, hi = lo + half_;
    const Bank& bank        = banks_[layer];
    // Runs of consecutive streamed experts land in consecutive ring frames: one 2-D copy per run
    // (records `stride` apart in the bank, frame_stride apart in the pool).
    for (std::uint32_t e = 0; e < experts_;) {
        const auto frame = static_cast<std::uint32_t>(row[e]);
        if (row[e] < 0 || frame < lo || frame >= hi) {
            ++e;
            continue;
        }
        std::uint32_t end = e + 1;
        while (end < experts_ && row[end] == row[end - 1] + 1 && static_cast<std::uint32_t>(row[end]) < hi) { ++end; }
        CUDA_CHECK(cudaMemcpy2DAsync(frame_base_ + static_cast<std::size_t>(frame) * frame_stride_, frame_stride_,
                                     bank.records + static_cast<std::size_t>(e) * bank.stride, bank.stride, bank.stride,
                                     end - e, cudaMemcpyHostToDevice, stream_));
        stats_.streamed += end - e;
        ++stats_.copies;
        e = end;
    }
    CUDA_CHECK(cudaEventRecord(landed_[layer], stream_));
}

void ExpertStream::drain() { CUDA_CHECK(cudaStreamSynchronize(stream_)); }

void ExpertStream::before_experts(std::uint32_t layer, cudaStream_t compute) {
    if (active_ && streams_[layer] != 0) { CUDA_CHECK(cudaStreamWaitEvent(compute, landed_[layer], 0)); }
}

void ExpertStream::after_experts(std::uint32_t layer, cudaStream_t compute) {
    if (!active_) { return; }
    CUDA_CHECK(cudaEventRecord(consumed_[layer], compute));
    if (layer + 2 < layers_) { enqueue(layer + 2); }
}

} // namespace ninfer::models::qwen4_exp::execution
