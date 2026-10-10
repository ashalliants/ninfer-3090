#pragma once

// Adapted from Infernix a3edb450 src/models/qwen4_exp/program/ngram_volume.h (Apache-2.0).
// Modified for NInfer-3090: version 2 volumes of GGML block rows (the header names the row format,
// tools/artifact/ngram_volume.py); an optional keep-alive read, after Strata's PLE reader
// (src/ngram/ple_reader.cpp, MIT), which Infernix lacks; NInfer's namespace.
//
// The PLE n-gram table volume written by the converter: a 4 KiB header, then rows of `row_bytes`,
// `rows_per_block` per 4 KiB block, never straddling a block. Rows are read with unbuffered I/O so
// the 29 GB table never fills the page cache. A direct-mapped host cache keeps recently read rows;
// one call's missing rows are read once per distinct block, up to 64 blocks in flight through a
// fixed ring, so their latency overlaps and the memory stays bounded. A call with many distinct
// blocks (a prefill chunk) splits them over kReadThreads threads with a kParallelInFlight-block
// ring each. The volume must not be mapped (core/read_only_file.h).

#include "core/read_only_file.h"
#include "models/qwen4_exp/config.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <mutex>
#include <span>
#include <thread>
#include <vector>

namespace ninfer::models::qwen4_exp {

class NgramVolume {
public:
    // Validates the header against the artifact's table geometry, row format and volume id.
    NgramVolume(const std::filesystem::path& path, const NgramTableConfig& table);
    ~NgramVolume();
    NgramVolume(const NgramVolume&)            = delete;
    NgramVolume& operator=(const NgramVolume&) = delete;

    [[nodiscard]] const NgramTableConfig& table() const noexcept { return table_; }

    // Copies each row's bytes to out[i * row_bytes, (i + 1) * row_bytes). Not reentrant: one caller.
    void read_rows(std::span<const std::uint32_t> rows, std::span<std::byte> out) const;

    // Whether every row is in the host cache, so read_rows would read nothing from the volume.
    [[nodiscard]] bool cached(std::span<const std::uint32_t> rows) const noexcept;

    // Keeps the drive awake while the volume is in use: as long as rows were asked for within the
    // last `window`, one 4 KiB block (a different one each time, so the device really reads) is read
    // whenever `period` has passed without a volume read. An idle SSD may enter a power state whose
    // exit stalls the next read for tens of milliseconds. A zero period turns it off (the default).
    // A failed keep-alive read turns the keep-alive off, never the volume.
    void set_keepalive(std::chrono::milliseconds period, std::chrono::milliseconds window);

    // Row traffic since the volume was opened: rows requested, rows served by the host cache, 4 KiB
    // blocks read for rows, the wall time of the calls' volume reads, and keep-alive reads.
    struct Counters {
        std::uint64_t rows = 0, hits = 0, reads = 0, read_ns = 0, keepalive_reads = 0;
    };
    [[nodiscard]] Counters counters() const noexcept;

    // Bytes of the host row cache a volume of this geometry allocates.
    [[nodiscard]] static std::uint64_t cache_bytes(const NgramTableConfig& table) noexcept {
        return (std::uint64_t{1} << kCacheBits) * table.row_bytes;
    }

private:
    static constexpr unsigned kCacheBits             = 20; // 2^20 rows, ~94 MB with 90-byte rows
    static constexpr std::size_t kInFlight           = 64; // blocks read at once (one thread)
    static constexpr std::size_t kReadThreads        = 8;  // threads of a large call
    static constexpr std::size_t kParallelInFlight   = 16; // blocks in flight per thread of a large call
    static constexpr std::size_t kParallelBlocks     = 512; // distinct blocks that make a call large

    // The direct-mapped cache slot of a row.
    static std::size_t slot_of(std::uint32_t row) noexcept {
        return static_cast<std::size_t>((row * 2654435761U) >> (32U - kCacheBits));
    }
    void keepalive_loop();
    static std::int64_t now_ns() noexcept;

    ReadOnlyFile file_;
    NgramTableConfig table_;
    mutable std::vector<std::byte> ring_storage_; // kInFlight blocks, 4 KiB-aligned at ring_
    mutable std::byte* ring_ = nullptr;
    mutable std::vector<std::byte> parallel_storage_; // kReadThreads rings of kParallelInFlight blocks
    mutable std::byte* parallel_ring_ = nullptr;
    mutable std::vector<std::uint32_t> tags_; // row id per cache slot, UINT32_MAX when empty
    mutable std::vector<std::byte> cache_;    // row bytes per slot
    // One call's misses as (block << 32 | output index), sorted so each block's rows are adjacent;
    // the distinct blocks' offsets and where each block's misses begin.
    mutable std::vector<std::uint64_t> misses_;
    mutable std::vector<std::uint64_t> offsets_;
    mutable std::vector<std::size_t> first_miss_;
    mutable Counters counters_;

    // Keep-alive state: the thread reads these clocks (steady, ns) without the call's lock.
    mutable std::atomic<std::int64_t> last_request_ns_{0};
    mutable std::atomic<std::int64_t> last_read_ns_{0};
    std::atomic<std::uint64_t> keepalive_reads_{0};
    std::mutex keepalive_mutex_;
    std::condition_variable keepalive_wake_;
    std::chrono::milliseconds keepalive_period_{0}, keepalive_window_{0};
    bool keepalive_stop_ = false;
    std::thread keepalive_thread_;
};

// The volume beside the artifact: `<artifact>.ngram`.
[[nodiscard]] std::filesystem::path default_ngram_volume(const std::filesystem::path& artifact);

} // namespace ninfer::models::qwen4_exp
