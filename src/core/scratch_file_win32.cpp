#include "core/scratch_file.h"

#ifndef NOMINMAX
#    define NOMINMAX
#endif
#include <windows.h>
#include <winioctl.h>

#include <algorithm>
#include <deque>
#include <limits>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

namespace ninfer {
namespace {

[[noreturn]] void fail(DWORD error, const std::string& what) {
    throw std::system_error(static_cast<int>(error), std::system_category(), what);
}

void check_request(const ScratchFile::Request& r, std::uint64_t size) {
    constexpr auto a = ScratchFile::kAlignment;
    if (r.offset % a != 0 || r.bytes % a != 0 || reinterpret_cast<std::uintptr_t>(r.data) % a != 0 || r.bytes == 0 ||
        r.bytes > std::numeric_limits<DWORD>::max() - a || r.offset > size || r.bytes > size - r.offset) {
        throw std::invalid_argument("scratch file transfer is not aligned or runs past the file");
    }
}

} // namespace

struct ScratchFile::Impl {
    HANDLE file = INVALID_HANDLE_VALUE;
    std::filesystem::path path;
    std::uint64_t size = 0;
    bool sparse        = false;

    ~Impl() {
        if (file != INVALID_HANDLE_VALUE) { ::CloseHandle(file); }
    }

    void run(std::span<const Request> requests, std::size_t depth, bool writing) const {
        for (const Request& r : requests) { check_request(r, size); }
        depth = std::max<std::size_t>(1, std::min(depth, requests.size()));
        struct Slot {
            OVERLAPPED op{};
            HANDLE event = nullptr;
            std::size_t request = 0;
        };
        std::vector<Slot> slots(depth);
        struct Events {
            std::vector<Slot>& s;
            ~Events() {
                for (auto& slot : s) {
                    if (slot.event != nullptr) { ::CloseHandle(slot.event); }
                }
            }
        } events{slots};
        for (auto& slot : slots) {
            slot.event = ::CreateEventW(nullptr, TRUE, FALSE, nullptr);
            if (slot.event == nullptr) { fail(::GetLastError(), "CreateEventW"); }
        }
        std::deque<std::size_t> in_flight; // slot indices, issue order
        std::vector<std::size_t> idle;
        for (std::size_t s = depth; s-- > 0;) { idle.push_back(s); }
        std::size_t next  = 0;
        DWORD failure     = ERROR_SUCCESS;
        bool short_count  = false;
        const auto failed = [&] { return failure != ERROR_SUCCESS || short_count; };
        while (!in_flight.empty() || (next < requests.size() && !failed())) {
            while (next < requests.size() && !idle.empty() && !failed()) {
                const std::size_t s = idle.back();
                Slot& slot          = slots[s];
                const Request& r    = requests[next];
                ::ResetEvent(slot.event);
                slot.op            = OVERLAPPED{};
                slot.op.Offset     = static_cast<DWORD>(r.offset & 0xffffffffULL);
                slot.op.OffsetHigh = static_cast<DWORD>(r.offset >> 32U);
                slot.op.hEvent     = slot.event;
                slot.request       = next;
                const BOOL ok = writing ? ::WriteFile(file, r.data, static_cast<DWORD>(r.bytes), nullptr, &slot.op)
                                        : ::ReadFile(file, r.data, static_cast<DWORD>(r.bytes), nullptr, &slot.op);
                if (!ok) {
                    const DWORD error = ::GetLastError();
                    if (error != ERROR_IO_PENDING) {
                        failure = error;
                        break;
                    }
                }
                idle.pop_back();
                in_flight.push_back(s);
                ++next;
            }
            if (in_flight.empty()) { break; }
            const std::size_t s = in_flight.front();
            in_flight.pop_front();
            Slot& slot  = slots[s];
            DWORD count = 0;
            if (!::GetOverlappedResult(file, &slot.op, &count, TRUE)) {
                if (failure == ERROR_SUCCESS) { failure = ::GetLastError(); }
            } else if (count != requests[slot.request].bytes) {
                short_count = true;
            }
            idle.push_back(s);
        }
        if (failure != ERROR_SUCCESS) {
            fail(failure, std::string(writing ? "scratch file write " : "scratch file read ") + path.string());
        }
        if (short_count) {
            fail(writing ? ERROR_DISK_FULL : ERROR_HANDLE_EOF,
                 std::string(writing ? "scratch file short write " : "scratch file short read ") + path.string());
        }
    }
};

ScratchFile::ScratchFile(const std::filesystem::path& path, std::uint64_t bytes) : impl_(std::make_unique<Impl>()) {
    if (bytes == 0 || bytes % kAlignment != 0) { throw std::invalid_argument("scratch file size must be a nonzero multiple of 4 KiB"); }
    impl_->path = path;
    impl_->size = bytes;
    // No sharing: nobody else opens it. Delete-on-close also covers a killed process.
    impl_->file = ::CreateFileW(path.c_str(), GENERIC_READ | GENERIC_WRITE, 0, nullptr, CREATE_NEW,
                                FILE_ATTRIBUTE_NORMAL | FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED |
                                    FILE_FLAG_DELETE_ON_CLOSE,
                                nullptr);
    if (impl_->file == INVALID_HANDLE_VALUE) { fail(::GetLastError(), "CreateFileW " + path.string()); }
    // Sparse: clusters are allocated as slabs are written, and a write far past the valid data does
    // not first zero everything before it.
    DWORD returned = 0;
    impl_->sparse  = ::DeviceIoControl(impl_->file, FSCTL_SET_SPARSE, nullptr, 0, nullptr, 0, &returned, nullptr) != 0;
    FILE_END_OF_FILE_INFO end{};
    end.EndOfFile.QuadPart = static_cast<LONGLONG>(bytes);
    if (!::SetFileInformationByHandle(impl_->file, FileEndOfFileInfo, &end, sizeof(end))) {
        fail(::GetLastError(), "SetFileInformationByHandle " + path.string());
    }
}

ScratchFile::~ScratchFile() = default;

std::uint64_t ScratchFile::bytes() const noexcept { return impl_->size; }
const std::filesystem::path& ScratchFile::path() const noexcept { return impl_->path; }
bool ScratchFile::sparse() const noexcept { return impl_->sparse; }

void ScratchFile::read(std::span<const Request> requests, std::size_t depth) const { impl_->run(requests, depth, false); }

void ScratchFile::write(std::span<const Request> requests, std::size_t depth) { impl_->run(requests, depth, true); }

} // namespace ninfer
