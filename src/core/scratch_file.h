#pragma once

// A private scratch file read and written with unbuffered I/O (Windows FILE_FLAG_NO_BUFFERING, Linux
// O_DIRECT): a process-lifetime backing store on local disk. It is created new (it must not exist),
// sparse where the file system allows, so its disk use grows with what is written, and removed when
// it is closed or the process ends (Windows: FILE_FLAG_DELETE_ON_CLOSE; POSIX: unlinked once open),
// so a killed process leaves nothing behind. Offsets, lengths and buffer addresses must be multiples
// of kAlignment.

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>

namespace ninfer {

class ScratchFile {
public:
    static constexpr std::size_t kAlignment = 4096;

    struct Request {
        std::uint64_t offset = 0;
        std::byte* data      = nullptr;
        std::size_t bytes    = 0;
    };

    // Creates `path` with a logical size of `bytes` (a multiple of kAlignment). Throws
    // std::system_error when it cannot be created.
    ScratchFile(const std::filesystem::path& path, std::uint64_t bytes);
    ~ScratchFile();
    ScratchFile(const ScratchFile&)            = delete;
    ScratchFile& operator=(const ScratchFile&) = delete;

    [[nodiscard]] std::uint64_t bytes() const noexcept;
    [[nodiscard]] const std::filesystem::path& path() const noexcept;
    // Whether disk space is taken only as data is written (a sparse file on Windows; a size set by
    // ftruncate on POSIX).
    [[nodiscard]] bool sparse() const noexcept;

    // Runs every request, at most `depth` in flight (Windows: overlapped I/O; POSIX: one at a time),
    // and returns when all have completed. A failed or short transfer stops issuing, waits for the
    // ones in flight and throws std::system_error (a full disk on write: ERROR_DISK_FULL / ENOSPC).
    // Safe to call from several threads at once on disjoint byte ranges.
    void read(std::span<const Request> requests, std::size_t depth) const;
    void write(std::span<const Request> requests, std::size_t depth);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace ninfer
