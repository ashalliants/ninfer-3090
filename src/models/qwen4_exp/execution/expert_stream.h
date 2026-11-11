#pragma once

// Adapted from Infernix a3edb450 src/models/qwen4_exp/execution/expert_stream.h (Apache-2.0).
// Modified for NInfer-3090: the ring is a run of frames lent by the expert cache, so a streamed
// expert is published as resident in a per-layer frame table (the cache's table with the streamed
// experts' ring frames) and the MoE Op reads it like any resident record (no `prefetched` source,
// no Op change); records copy with one 2-D copy per run of consecutive experts (bank records are
// narrower than frames for some layers); no SSD tier, no gated chunks, no holds.
//
// Prefill expert streaming: the records of a wide chunk's non-resident experts reach the device by
// copy-engine DMA, two layers ahead, into a ring of lent frames, instead of through the MoE's
// staging kernel, which copies with the SMs inside the layer's critical path. A wide chunk routes
// nearly every expert of every layer, so a layer's copies do not wait for its routing: they take
// the layer's non-resident experts in ascending id, up to one half of the ring. Layer l uses half
// l % 2; its copies wait until layer l - 2's experts finished reading that half, and layer l's MoE
// waits for them (one event per layer each way). Experts beyond the half's capacity keep the MoE's
// staging. Results are unchanged: the MoE's outputs do not depend on where a record lives.
//
// With a CPU share (--prefill-cpu-share, NInfer-3090) each layer also leaves out a predicted-cold
// set (ops::offloaded_moe::select_stream_cold over the expert cache's LFRU use counts, the only prior
// that exists before the layer routes): they stay -1 in the table, so the call's MoE stages them or
// hands its CPU share of them to the CPU miss service.

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <vector>

namespace ninfer::models::qwen4_exp::execution {

// ExpertStream::begin's CPU share: select_stream_cold of each layer's residency row and `counts` row
// (U32 [layers][experts], the expert cache's LFRU use counts), the half's capacity and num / den.
struct ExpertStreamShare {
    int num = 0, den = 1;
    const std::uint32_t* counts = nullptr;
};

class ExpertStream {
public:
    // Layer l's pinned host bank (device-copyable by DMA) and its record stride.
    struct Bank {
        const std::uint8_t* records = nullptr;
        std::uint64_t stride        = 0;
    };

    // frame_base / frame_stride: the expert cache's frame pool, which holds the ring.
    ExpertStream(std::uint32_t layers, std::uint32_t experts, std::vector<Bank> banks, std::uint8_t* frame_base,
                 std::uint64_t frame_stride);
    ~ExpertStream();
    ExpertStream(const ExpertStream&)            = delete;
    ExpertStream& operator=(const ExpertStream&) = delete;

    // Device bytes of the per-layer frame tables (I32 [layers][experts]).
    [[nodiscard]] std::size_t table_bytes() const noexcept;

    // Starts a chunk: the ring is frames [ring_first, ring_first + ring_frames) of the pool, `tables`
    // device memory of table_bytes(). From the cache's host tables `residency` (I32 [layers][experts],
    // frame or -1, as the chunk's kernels see them) it builds the streamed tables, uploads them on
    // `compute`, and enqueues the copies of layers 0 and 1 on the copy stream behind `compute`.
    //
    // `share`, when its num is nonzero, leaves each layer's predicted-cold experts out of the stream
    // (ExpertStreamShare).
    using Share = ExpertStreamShare;
    void begin(const std::int32_t* residency, std::int32_t* tables, std::uint32_t ring_first, std::uint32_t ring_frames,
               cudaStream_t compute, Share share = {});
    [[nodiscard]] bool active() const noexcept { return active_; }
    // Whether every non-resident expert of every layer streams this chunk: a half holds a layer's
    // every expert and no cold set was left out.
    [[nodiscard]] bool covers_all() const noexcept { return half_ >= experts_ && !left_cold_; }
    // Layer l's frame table for the chunk (device).
    [[nodiscard]] const std::int32_t* table(std::uint32_t layer) const noexcept;
    // Before layer `layer`'s experts: `compute` waits for the layer's copies.
    void before_experts(std::uint32_t layer, cudaStream_t compute);
    // After the layer's experts are enqueued on `compute`: the half is free once they finish; the
    // copies of layer + 2 follow.
    void after_experts(std::uint32_t layer, cudaStream_t compute);
    // After the chunk's last layer (every copy was waited for by `compute`).
    void end() noexcept { active_ = false; }
    // Blocks until the copy stream is idle. After an aborted chunk, copies of layers the compute
    // stream never consumed can still be writing the ring, which the lender is about to reclaim.
    void drain();

    struct Stats {
        std::uint64_t streamed = 0; // records copied
        std::uint64_t copies   = 0; // copy calls (runs of consecutive experts)
        std::uint64_t left     = 0; // non-resident experts beyond a half's capacity (left to staging)
        std::uint64_t cold     = 0; // predicted-cold experts left out for the CPU share (Share)
    };
    // The cold experts every stream of this process has left out (tests that cannot reach it).
    [[nodiscard]] static std::uint64_t process_cold_experts() noexcept;
    [[nodiscard]] const Stats& stats() const noexcept { return stats_; }

private:
    void enqueue(std::uint32_t layer);

    std::uint32_t layers_ = 0, experts_ = 0;
    std::vector<Bank> banks_;
    std::uint8_t* frame_base_   = nullptr;
    std::uint64_t frame_stride_ = 0;
    cudaStream_t stream_        = nullptr;
    cudaEvent_t start_          = nullptr;
    cudaEvent_t uploaded_       = nullptr;
    std::vector<cudaEvent_t> landed_, consumed_;
    std::vector<std::int32_t> tables_host_; // pageable staging of the streamed tables
    void* tables_pinned_     = nullptr;     // pinned copy uploaded by DMA
    std::int32_t* tables_    = nullptr;     // device
    std::uint32_t ring_first_ = 0, half_ = 0;
    std::vector<std::uint8_t> streams_; // per layer: whether it has copies
    bool active_    = false;
    bool uploading_ = false;
    bool left_cold_ = false; // this chunk left a cold set out of some layer
    std::vector<std::uint8_t> cold_; // scratch: one layer's cold mask
    Stats stats_;
};

} // namespace ninfer::models::qwen4_exp::execution
