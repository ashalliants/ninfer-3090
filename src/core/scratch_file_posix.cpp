#include "core/scratch_file.h"

#include <cerrno>
#include <stdexcept>
#include <string>
#include <system_error>

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

namespace ninfer {
namespace {

void check_request(const ScratchFile::Request& r, std::uint64_t size) {
    constexpr auto a = ScratchFile::kAlignment;
    if (r.offset % a != 0 || r.bytes % a != 0 || reinterpret_cast<std::uintptr_t>(r.data) % a != 0 || r.bytes == 0 ||
        r.offset > size || r.bytes > size - r.offset) {
        throw std::invalid_argument("scratch file transfer is not aligned or runs past the file");
    }
}

} // namespace

struct ScratchFile::Impl {
    int fd = -1;
    std::filesystem::path path;
    std::uint64_t size = 0;

    ~Impl() {
        if (fd >= 0) { ::close(fd); }
    }

    void run(std::span<const Request> requests, bool writing) const {
        for (const Request& r : requests) { check_request(r, size); }
        for (const Request& r : requests) {
            std::size_t done = 0;
            while (done < r.bytes) {
                const auto offset = static_cast<off_t>(r.offset + done);
                const ssize_t n   = writing ? ::pwrite(fd, r.data + done, r.bytes - done, offset)
                                            : ::pread(fd, r.data + done, r.bytes - done, offset);
                if (n < 0 && errno == EINTR) { continue; }
                if (n <= 0) {
                    const int error = n < 0 ? errno : (writing ? ENOSPC : EIO);
                    throw std::system_error(error, std::generic_category(),
                                            std::string(writing ? "scratch file write " : "scratch file read ") +
                                                path.string());
                }
                done += static_cast<std::size_t>(n);
            }
        }
    }
};

ScratchFile::ScratchFile(const std::filesystem::path& path, std::uint64_t bytes) : impl_(std::make_unique<Impl>()) {
    if (bytes == 0 || bytes % kAlignment != 0) { throw std::invalid_argument("scratch file size must be a nonzero multiple of 4 KiB"); }
    impl_->path = path;
    impl_->size = bytes;
    impl_->fd   = ::open(path.c_str(), O_RDWR | O_CREAT | O_EXCL | O_CLOEXEC | O_DIRECT, 0600);
    if (impl_->fd < 0) { throw std::system_error(errno, std::generic_category(), "open " + path.string()); }
    // Anonymous from here on: nothing is left behind however the process ends.
    (void)::unlink(path.c_str());
    if (::ftruncate(impl_->fd, static_cast<off_t>(bytes)) != 0) {
        throw std::system_error(errno, std::generic_category(), "ftruncate " + path.string());
    }
}

ScratchFile::~ScratchFile() = default;

std::uint64_t ScratchFile::bytes() const noexcept { return impl_->size; }
const std::filesystem::path& ScratchFile::path() const noexcept { return impl_->path; }
bool ScratchFile::sparse() const noexcept { return true; }

void ScratchFile::read(std::span<const Request> requests, std::size_t) const { impl_->run(requests, false); }

void ScratchFile::write(std::span<const Request> requests, std::size_t) { impl_->run(requests, true); }

} // namespace ninfer
