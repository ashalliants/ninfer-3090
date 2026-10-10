// Adapted from Infernix a3edb450 src/core/read_only_file_win32.cpp (Apache-2.0).
// Modified for NInfer-3090: the unbuffered handle only (no buffered handle or mapping), NInfer's
// namespace.

#include "core/read_only_file.h"

#ifndef NOMINMAX
#    define NOMINMAX
#endif
#include <windows.h>

#include <algorithm>
#include <exception>
#include <limits>
#include <stdexcept>
#include <system_error>
#include <vector>

namespace ninfer {

struct ReadOnlyFile::Impl {
    HANDLE direct_file = INVALID_HANDLE_VALUE;
    std::uint64_t size = 0;

    explicit Impl(const std::filesystem::path& path) {
        // A read-only consumer imposes the loosest sharing, as POSIX open does.
        direct_file = ::CreateFileW(path.c_str(), GENERIC_READ,
                                    FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr,
                                    OPEN_EXISTING,
                                    FILE_ATTRIBUTE_NORMAL | FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED,
                                    nullptr);
        if (direct_file == INVALID_HANDLE_VALUE) {
            throw std::system_error(static_cast<int>(::GetLastError()), std::system_category(),
                                    "CreateFileW " + path.string());
        }
        LARGE_INTEGER file_size{};
        if (!::GetFileSizeEx(direct_file, &file_size) || file_size.QuadPart < 0) {
            const auto error = ::GetLastError();
            ::CloseHandle(direct_file);
            throw std::system_error(static_cast<int>(error), std::system_category(), "GetFileSizeEx " + path.string());
        }
        size = static_cast<std::uint64_t>(file_size.QuadPart);
    }

    ~Impl() {
        if (direct_file != INVALID_HANDLE_VALUE) { ::CloseHandle(direct_file); }
    }
};

ReadOnlyFile::ReadOnlyFile(const std::filesystem::path& path) : impl_(std::make_unique<Impl>(path)) {}

ReadOnlyFile::~ReadOnlyFile()                                  = default;
ReadOnlyFile::ReadOnlyFile(ReadOnlyFile&&) noexcept            = default;
ReadOnlyFile& ReadOnlyFile::operator=(ReadOnlyFile&&) noexcept = default;

std::uint64_t ReadOnlyFile::current_bytes() const noexcept {
    LARGE_INTEGER size{};
    if (!::GetFileSizeEx(impl_->direct_file, &size) || size.QuadPart < 0) { return impl_->size; }
    return static_cast<std::uint64_t>(size.QuadPart);
}

void ReadOnlyFile::read_direct_blocks(std::span<const std::uint64_t> offsets, std::size_t block_bytes,
                                      std::span<std::byte> ring,
                                      const std::function<void(std::size_t, std::span<const std::byte>)>& consume) const {
    if (block_bytes == 0 || block_bytes > std::numeric_limits<DWORD>::max() || ring.size() < block_bytes) {
        throw std::invalid_argument("direct block reads need a ring of at least one block");
    }
    const std::size_t slots     = ring.size() / block_bytes;
    constexpr std::size_t kIdle = std::numeric_limits<std::size_t>::max();
    // One OVERLAPPED per ring slot and no event objects: completion is polled, which is a memory
    // read (HasOverlappedIoCompleted), not a system call.
    std::vector<OVERLAPPED> operations(slots);
    std::vector<std::size_t> reading(slots, kIdle);
    std::vector<std::size_t> idle;
    idle.reserve(slots);
    for (std::size_t s = slots; s-- > 0;) { idle.push_back(s); }
    std::size_t next = 0, in_flight = 0;
    DWORD failure   = ERROR_SUCCESS;
    bool short_read = false;
    std::exception_ptr consume_error;
    const auto failed = [&] { return failure != ERROR_SUCCESS || short_read || consume_error; };
    while (in_flight > 0 || (next < offsets.size() && !failed())) {
        while (next < offsets.size() && !idle.empty() && !failed()) {
            const std::size_t slot = idle.back();
            OVERLAPPED& op         = operations[slot];
            op                     = OVERLAPPED{};
            op.Offset              = static_cast<DWORD>(offsets[next] & 0xffffffffULL);
            op.OffsetHigh          = static_cast<DWORD>(offsets[next] >> 32U);
            if (!::ReadFile(impl_->direct_file, ring.data() + slot * block_bytes, static_cast<DWORD>(block_bytes),
                            nullptr, &op)) {
                const DWORD error = ::GetLastError();
                if (error != ERROR_IO_PENDING) {
                    failure = error;
                    break;
                }
            }
            idle.pop_back();
            reading[slot] = next++;
            ++in_flight;
        }
        bool progressed = false;
        for (std::size_t slot = 0; slot < slots && in_flight > 0; ++slot) {
            if (reading[slot] == kIdle || !HasOverlappedIoCompleted(&operations[slot])) { continue; }
            DWORD bytes = 0;
            if (!::GetOverlappedResult(impl_->direct_file, &operations[slot], &bytes, FALSE)) {
                if (failure == ERROR_SUCCESS) { failure = ::GetLastError(); }
            } else if (bytes != block_bytes) {
                short_read = true;
            } else if (!failed()) {
                try {
                    consume(reading[slot], std::span<const std::byte>(ring.data() + slot * block_bytes, block_bytes));
                } catch (...) { consume_error = std::current_exception(); }
            }
            reading[slot] = kIdle;
            idle.push_back(slot);
            --in_flight;
            progressed = true;
        }
        if (!progressed && in_flight > 0) { YieldProcessor(); }
    }
    if (consume_error) { std::rethrow_exception(consume_error); }
    if (failure != ERROR_SUCCESS) {
        throw std::system_error(static_cast<int>(failure), std::system_category(), "direct block read");
    }
    if (short_read) { throw std::runtime_error("direct block read: a read did not complete in full"); }
}

std::size_t ReadOnlyFile::read_direct(std::uint64_t offset, std::span<std::byte> destination) const {
    constexpr auto max_file_offset = static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max());
    if (offset > max_file_offset || destination.size() > max_file_offset - offset) {
        throw std::overflow_error("direct file read exceeds platform I/O limits");
    }
    // Each call waits on its own event: without one, GetOverlappedResult waits on the file handle,
    // which any concurrent read of the same handle (read_direct_blocks on another thread) signals,
    // so it could return while this call's OVERLAPPED and buffer are still in use.
    const HANDLE event = ::CreateEventW(nullptr, TRUE, FALSE, nullptr);
    if (event == nullptr) {
        throw std::system_error(static_cast<int>(::GetLastError()), std::system_category(), "CreateEventW");
    }
    struct CloseEvent {
        HANDLE h;
        ~CloseEvent() { ::CloseHandle(h); }
    } close_event{event};
    std::size_t total = 0;
    while (total < destination.size()) {
        constexpr std::size_t max_read = 1ULL << 30;
        const auto amount              = static_cast<DWORD>(std::min(max_read, destination.size() - total));
        const std::uint64_t absolute   = offset + total;
        OVERLAPPED operation{};
        operation.Offset     = static_cast<DWORD>(absolute & 0xffffffffULL);
        operation.OffsetHigh = static_cast<DWORD>(absolute >> 32U);
        operation.hEvent     = event;
        DWORD bytes          = 0;
        if (!::ReadFile(impl_->direct_file, destination.data() + total, amount, nullptr, &operation)) {
            const auto error = ::GetLastError();
            if (error == ERROR_HANDLE_EOF) { break; }
            if (error != ERROR_IO_PENDING) {
                throw std::system_error(static_cast<int>(error), std::system_category(), "direct file read");
            }
        }
        // Completed at once or pending: the byte count comes from the operation either way.
        if (!::GetOverlappedResult(impl_->direct_file, &operation, &bytes, TRUE)) {
            const auto error = ::GetLastError();
            if (error == ERROR_HANDLE_EOF) { break; }
            throw std::system_error(static_cast<int>(error), std::system_category(), "direct file read");
        }
        total += bytes;
        if (bytes != amount) { break; }
    }
    return total;
}

} // namespace ninfer
