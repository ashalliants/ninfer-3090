#pragma once

// Adapted from Infernix a3edb450 src/core/read_only_file.h (Apache-2.0).
// Modified for NInfer-3090: unbuffered reads only. Infernix's whole-file mapping is not ported: its
// one consumer here, the n-gram volume, must not be mapped, because on Windows every unbuffered
// read of a file with a mapped data section pays the cache manager's coherency work (Infernix
// measured random 4 KiB reads of the volume fall from ~220K/s to ~56K/s).

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <span>

namespace ninfer {

// A file read with unbuffered I/O (Windows FILE_FLAG_NO_BUFFERING, Linux O_DIRECT): offsets,
// lengths and buffers follow the device's sector alignment, which 4 KiB satisfies.
class ReadOnlyFile {
public:
    explicit ReadOnlyFile(const std::filesystem::path& path);
    ~ReadOnlyFile();

    ReadOnlyFile(ReadOnlyFile&&) noexcept;
    ReadOnlyFile& operator=(ReadOnlyFile&&) noexcept;
    ReadOnlyFile(const ReadOnlyFile&)            = delete;
    ReadOnlyFile& operator=(const ReadOnlyFile&) = delete;

    // Size of the file on disk now; external truncation or extension after open is observed.
    [[nodiscard]] std::uint64_t current_bytes() const noexcept;
    // Reads until `destination` is full or the file ends; returns the bytes read.
    std::size_t read_direct(std::uint64_t offset, std::span<std::byte> destination) const;

    // Unbuffered reads of `block_bytes` at each offset, at most ring.size() / block_bytes in flight,
    // each into a free block of `ring` (block-aligned), so their device latency overlaps (Windows:
    // overlapped reads polled without events; Linux: kernel AIO; other POSIX systems read
    // serially). consume(i, bytes) runs on the calling thread as read i completes, in completion
    // order; its ring block is reused once consume returns. Returns when every read has completed;
    // a failed or short read stops issuing, waits for the reads in flight and throws.
    void read_direct_blocks(std::span<const std::uint64_t> offsets, std::size_t block_bytes, std::span<std::byte> ring,
                            const std::function<void(std::size_t, std::span<const std::byte>)>& consume) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace ninfer
