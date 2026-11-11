// Adapted from Infernix a3edb450 src/core/read_only_file_posix.cpp (Apache-2.0).
// Modified for NInfer-3090: no mapping, NInfer's namespace.

#include "core/read_only_file.h"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cstdint>
#include <exception>
#include <limits>
#include <stdexcept>
#include <system_error>
#include <vector>

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#if defined(__linux__)
#include <linux/aio_abi.h>
#include <sys/syscall.h>
#endif

namespace ninfer {

#if defined(__linux__)
namespace {

// One Linux AIO context per thread, created on first use and kept for the thread's life instead of
// a fresh io_setup/io_destroy per read_direct_blocks call: a 2026-11-11 session measured io_destroy
// alone costing ~0.5 s a call on a rented box (TLB-shootdown IPI latency, most likely, under that
// box's virtualization), which -- paid on every decode-time n-gram read batch -- was the real cause
// of a measured decode collapse, not the reads themselves (io_submit/io_getevents stayed fast
// throughout). A raw syscall wrapper's destructor runs at thread exit, where the same cost is paid
// once per thread instead of once per call.
struct ThreadAioContext {
    aio_context_t handle = 0;
    unsigned capacity    = 0;

    ~ThreadAioContext() { reset(); }

    void reset() {
        if (capacity > 0) {
            ::syscall(SYS_io_destroy, handle);
            handle   = 0;
            capacity = 0;
        }
    }
};
thread_local ThreadAioContext g_aio_context;

// A context with at least `slots` capacity, reusing this thread's existing one when it already has
// enough room; 0 if io_setup fails, so the caller can fall back to serial reads as before.
aio_context_t acquire_aio_context(unsigned slots) {
    if (g_aio_context.capacity >= slots) { return g_aio_context.handle; }
    g_aio_context.reset();
    aio_context_t context = 0;
    if (::syscall(SYS_io_setup, slots, &context) != 0) { return 0; }
    g_aio_context.handle   = context;
    g_aio_context.capacity = slots;
    return context;
}

} // namespace
#endif

struct ReadOnlyFile::Impl {
    int fd             = -1;
    std::uint64_t size = 0;
    mutable std::atomic<std::uint64_t> serial_fallback_calls{0};

    explicit Impl(const std::filesystem::path& path) {
        fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC | O_DIRECT);
        if (fd < 0) { throw std::system_error(errno, std::generic_category(), "open " + path.string()); }
        struct stat status{};
        if (::fstat(fd, &status) != 0 || status.st_size < 0) {
            const int error = errno;
            ::close(fd);
            fd = -1;
            throw std::system_error(error, std::generic_category(), "fstat " + path.string());
        }
        size = static_cast<std::uint64_t>(status.st_size);
    }

    ~Impl() {
        if (fd >= 0) { ::close(fd); }
    }
};

ReadOnlyFile::ReadOnlyFile(const std::filesystem::path& path) : impl_(std::make_unique<Impl>(path)) {}

ReadOnlyFile::~ReadOnlyFile()                                  = default;
ReadOnlyFile::ReadOnlyFile(ReadOnlyFile&&) noexcept            = default;
ReadOnlyFile& ReadOnlyFile::operator=(ReadOnlyFile&&) noexcept = default;

std::uint64_t ReadOnlyFile::current_bytes() const noexcept {
    struct stat status{};
    if (::fstat(impl_->fd, &status) != 0 || status.st_size < 0) { return impl_->size; }
    return static_cast<std::uint64_t>(status.st_size);
}

void ReadOnlyFile::read_direct_blocks(std::span<const std::uint64_t> offsets, std::size_t block_bytes,
                                      std::span<std::byte> ring,
                                      const std::function<void(std::size_t, std::span<const std::byte>)>& consume) const {
    if (block_bytes == 0 || ring.size() < block_bytes) {
        throw std::invalid_argument("direct block reads need a ring of at least one block");
    }
    if (offsets.empty()) { return; }
#if defined(__linux__)
    // Kernel AIO on the O_DIRECT descriptor keeps one read per ring block in flight, on this
    // thread's persistent context (acquire_aio_context): see ThreadAioContext above.
    const std::size_t slots      = std::min(ring.size() / block_bytes, offsets.size());
    const aio_context_t context  = acquire_aio_context(static_cast<unsigned>(slots));
    if (context != 0) {
        std::vector<iocb> blocks(slots);
        std::vector<io_event> events(slots);
        std::vector<std::size_t> idle;
        idle.reserve(slots);
        for (std::size_t s = slots; s-- > 0;) { idle.push_back(s); }
        std::size_t next = 0, in_flight = 0;
        int failure     = 0;
        bool short_read = false;
        std::exception_ptr consume_error;
        const auto failed = [&] { return failure != 0 || short_read || consume_error; };
        while (in_flight > 0 || (next < offsets.size() && !failed())) {
            while (next < offsets.size() && !idle.empty() && !failed()) {
                const std::size_t slot = idle.back();
                constexpr auto max_offset = static_cast<std::uint64_t>(std::numeric_limits<off_t>::max());
                if (offsets[next] > max_offset || block_bytes > max_offset - offsets[next]) {
                    failure = EOVERFLOW;
                    break;
                }
                iocb& block          = blocks[slot];
                block                = iocb{};
                block.aio_fildes     = static_cast<std::uint32_t>(impl_->fd);
                block.aio_lio_opcode = IOCB_CMD_PREAD;
                block.aio_buf        = reinterpret_cast<std::uint64_t>(ring.data() + slot * block_bytes);
                block.aio_nbytes     = block_bytes;
                block.aio_offset     = static_cast<std::int64_t>(offsets[next]);
                block.aio_data       = (static_cast<std::uint64_t>(next) << 16U) | slot;
                iocb* submit         = &block;
                const long submitted = ::syscall(SYS_io_submit, context, 1L, &submit);
                if (submitted != 1) {
                    if (submitted < 0 && errno == EAGAIN && in_flight > 0) { break; }
                    if (submitted < 0 && errno == EINTR) { continue; }
                    failure = submitted < 0 ? errno : EIO;
                    break;
                }
                idle.pop_back();
                ++next;
                ++in_flight;
            }
            if (in_flight == 0) { continue; }
            const long done = ::syscall(SYS_io_getevents, context, 1L, static_cast<long>(slots), events.data(), nullptr);
            if (done < 0) {
                if (errno == EINTR) { continue; }
                // The kernel owns the buffers of reads in flight: wait them out before the ring is reused.
                const int error = errno;
                while (in_flight > 0) {
                    const long one = ::syscall(SYS_io_getevents, context, 1L, 1L, events.data(), nullptr);
                    if (one > 0) {
                        in_flight -= static_cast<std::size_t>(one);
                    } else if (one < 0 && errno != EINTR) {
                        break;
                    }
                }
                // This context's state after a failed io_getevents is not something to trust with
                // the next call: drop it now so acquire_aio_context sets up a fresh one next time.
                g_aio_context.reset();
                throw std::system_error(error, std::generic_category(), "direct block read: io_getevents");
            }
            for (long e = 0; e < done; ++e) {
                const io_event& event   = events[static_cast<std::size_t>(e)];
                const std::size_t slot  = static_cast<std::size_t>(event.data & 0xFFFFU);
                const std::size_t index = static_cast<std::size_t>(event.data >> 16U);
                if (event.res < 0) {
                    if (failure == 0) { failure = static_cast<int>(-event.res); }
                } else if (static_cast<std::uint64_t>(event.res) != block_bytes) {
                    short_read = true;
                } else if (!failed()) {
                    try {
                        consume(index, std::span<const std::byte>(ring.data() + slot * block_bytes, block_bytes));
                    } catch (...) { consume_error = std::current_exception(); }
                }
                idle.push_back(slot);
                --in_flight;
            }
        }
        // The context outlives this call (acquire_aio_context, reused by this thread's next one).
        if (consume_error) { std::rethrow_exception(consume_error); }
        if (failure != 0) { throw std::system_error(failure, std::generic_category(), "direct block read"); }
        if (short_read) { throw std::runtime_error("direct block read: a read did not complete in full"); }
        return;
    }
#endif
    // Without kernel AIO -- io_setup failed above (or this is not Linux): serial reads into the
    // first ring block, one at a time, forfeiting the ring's overlap entirely.
    impl_->serial_fallback_calls.fetch_add(1, std::memory_order_relaxed);
    const std::span<std::byte> block = ring.first(block_bytes);
    for (std::size_t i = 0; i < offsets.size(); ++i) {
        if (read_direct(offsets[i], block) != block_bytes) {
            throw std::runtime_error("direct block read: a read did not complete in full");
        }
        consume(i, block);
    }
}

std::size_t ReadOnlyFile::read_direct(std::uint64_t offset, std::span<std::byte> destination) const {
    // The whole range [offset, offset + size) must be representable as off_t, like the Win32 reader.
    constexpr auto max_offset = static_cast<std::uint64_t>(std::numeric_limits<off_t>::max());
    if (offset > max_offset || destination.size() > max_offset - offset ||
        destination.size() > static_cast<std::size_t>(std::numeric_limits<ssize_t>::max())) {
        throw std::overflow_error("direct file read exceeds platform I/O limits");
    }
    std::size_t total = 0;
    while (total < destination.size()) {
        const ssize_t bytes = ::pread(impl_->fd, destination.data() + total, destination.size() - total,
                                      static_cast<off_t>(offset + total));
        if (bytes < 0) {
            if (errno == EINTR) { continue; }
            throw std::system_error(errno, std::generic_category(), "direct file read");
        }
        if (bytes == 0) { break; }
        total += static_cast<std::size_t>(bytes);
    }
    return total;
}

std::uint64_t ReadOnlyFile::serial_fallback_calls() const noexcept {
    return impl_->serial_fallback_calls.load(std::memory_order_relaxed);
}

} // namespace ninfer
