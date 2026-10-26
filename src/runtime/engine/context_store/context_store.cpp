#include "runtime/engine/context_store/context_store.h"
#include "runtime/engine/context_store/image_id.h"

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstring>
#include <exception>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <unordered_set>
#include <utility>

#include <zstd.h>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/file.h>
#include <unistd.h>

#include <cerrno>
#endif

namespace ninfer::runtime {
namespace {

namespace fs = std::filesystem;

constexpr char kManifestMagic[8]     = {'N', 'C', 'T', 'X', 'S', 'T', 'O', '1'};
// Version 2 adds the shared flag after the creation time. Version 1 manifests (written before it)
// are still read, as not shared, so an upgrade does not discard a store's conversations.
constexpr std::uint32_t kManifestVersion = 2;
// A manifest names at most this many chunks and checkpoints; larger counts mean a corrupt file, not
// a big session (a 1M-token session at 32 MiB chunks is a few thousand chunks).
constexpr std::uint32_t kMaximumChunksPerImage = 1U << 20U;
constexpr std::uint32_t kMaximumCheckpoints    = 1024;
constexpr std::uint32_t kMaximumSegments       = 1U << 16U;
// Threads that read and verify the chunks of one image (load).
constexpr std::size_t kLoadThreads = 8;
// Chunks of one image downloaded at a time (fetch_image_chunks): enough to hide the round trip
// of each request, few enough to keep the memory in flight (up to this many chunks) modest.
constexpr std::size_t kFetchThreads = 8;

// Bytes outside the declared regions at least this long are chunked by content (see put());
// shorter ones travel inside the manifest.
constexpr std::size_t kContentChunkMinimumSegment = 64U << 10U;
// Content-defined chunks: never shorter or longer than these (except a segment's last), about
// 64 KiB past the minimum on average.
constexpr std::size_t kContentChunkMinimum = 16U << 10U;
constexpr std::size_t kContentChunkMaximum = 256U << 10U;
constexpr std::uint64_t kContentChunkMask  = 0xffffULL << 48U; // 16 bits: 1 in 65536 positions

// A gear hash (as in FastCDC): each byte shifts the hash and adds a fixed random word, so bit k
// depends on the last k + 1 bytes and the top bits on a 64-byte window. The table is fixed
// (splitmix64 from a constant seed): changing it changes every cut, so chunks written before
// would no longer be reused.
constexpr std::array<std::uint64_t, 256> make_gear_table() {
    std::array<std::uint64_t, 256> table{};
    std::uint64_t state = 0x6e696e6665722d63ULL; // "ninfer-c"
    for (auto& entry : table) {
        state += 0x9e3779b97f4a7c15ULL;
        std::uint64_t z = state;
        z               = (z ^ (z >> 30U)) * 0xbf58476d1ce4e5b9ULL;
        z               = (z ^ (z >> 27U)) * 0x94d049bb133111ebULL;
        entry           = z ^ (z >> 31U);
    }
    return table;
}
constexpr std::array<std::uint64_t, 256> kGearTable = make_gear_table();

// Length of the first content-defined chunk of `bytes`: the same content gives the same cuts
// wherever it sits, so a stretch an edit did not touch is cut, and hashed, as it was before.
std::size_t content_defined_cut(std::span<const std::uint8_t> bytes) {
    if (bytes.size() <= kContentChunkMinimum) { return bytes.size(); }
    const std::size_t end = std::min(bytes.size(), kContentChunkMaximum);
    std::uint64_t gear    = 0;
    for (std::size_t at = kContentChunkMinimum - 64; at < end; ++at) {
        gear = (gear << 1U) + kGearTable[bytes[at]];
        if (at >= kContentChunkMinimum && (gear & kContentChunkMask) == 0) { return at + 1; }
    }
    return end;
}

// xxHash64 (Yann Collet, BSD-2-Clause), little-endian hosts. Two independently seeded passes give
// the 128-bit chunk name. This is a content address for trusted local files, not a security
// boundary: it needs to make an accidental collision between two different chunks negligible.
constexpr std::uint64_t kPrime1 = 11400714785074694791ULL;
constexpr std::uint64_t kPrime2 = 14029467366897019727ULL;
constexpr std::uint64_t kPrime3 = 1609587929392839161ULL;
constexpr std::uint64_t kPrime4 = 9650029242287828579ULL;
constexpr std::uint64_t kPrime5 = 2870177450012600261ULL;

constexpr std::uint64_t rotate_left(std::uint64_t value, int bits) noexcept {
    return (value << bits) | (value >> (64 - bits));
}

std::uint64_t read64(const std::uint8_t* p) noexcept {
    std::uint64_t value;
    std::memcpy(&value, p, sizeof(value));
    return value;
}

std::uint32_t read32(const std::uint8_t* p) noexcept {
    std::uint32_t value;
    std::memcpy(&value, p, sizeof(value));
    return value;
}

constexpr std::uint64_t xxh_round(std::uint64_t acc, std::uint64_t input) noexcept {
    acc += input * kPrime2;
    acc = rotate_left(acc, 31);
    return acc * kPrime1;
}

constexpr std::uint64_t xxh_merge(std::uint64_t acc, std::uint64_t value) noexcept {
    acc ^= xxh_round(0, value);
    return acc * kPrime1 + kPrime4;
}

std::uint64_t xxh64(const std::uint8_t* p, std::size_t length, std::uint64_t seed) noexcept {
    const std::uint8_t* const end = p + length;
    std::uint64_t h;
    if (length >= 32) {
        const std::uint8_t* const limit = end - 32;
        std::uint64_t v1                = seed + kPrime1 + kPrime2;
        std::uint64_t v2                = seed + kPrime2;
        std::uint64_t v3                = seed;
        std::uint64_t v4                = seed - kPrime1;
        do {
            v1 = xxh_round(v1, read64(p));
            v2 = xxh_round(v2, read64(p + 8));
            v3 = xxh_round(v3, read64(p + 16));
            v4 = xxh_round(v4, read64(p + 24));
            p += 32;
        } while (p <= limit);
        h = rotate_left(v1, 1) + rotate_left(v2, 7) + rotate_left(v3, 12) + rotate_left(v4, 18);
        h = xxh_merge(h, v1);
        h = xxh_merge(h, v2);
        h = xxh_merge(h, v3);
        h = xxh_merge(h, v4);
    } else {
        h = seed + kPrime5;
    }
    h += static_cast<std::uint64_t>(length);
    while (p + 8 <= end) {
        h ^= xxh_round(0, read64(p));
        h = rotate_left(h, 27) * kPrime1 + kPrime4;
        p += 8;
    }
    if (p + 4 <= end) {
        h ^= static_cast<std::uint64_t>(read32(p)) * kPrime1;
        h = rotate_left(h, 23) * kPrime2 + kPrime3;
        p += 4;
    }
    while (p < end) {
        h ^= static_cast<std::uint64_t>(*p) * kPrime5;
        h = rotate_left(h, 11) * kPrime1;
        ++p;
    }
    h ^= h >> 33;
    h *= kPrime2;
    h ^= h >> 29;
    h *= kPrime3;
    h ^= h >> 32;
    return h;
}

std::string hex_key(const std::array<std::uint64_t, 2>& hash) {
    char text[33];
    std::snprintf(text, sizeof(text), "%016llx%016llx", static_cast<unsigned long long>(hash[0]),
                  static_cast<unsigned long long>(hash[1]));
    return text;
}

bool valid_id(const std::string& id) {
    if (id.empty() || id.size() > 64) { return false; }
    return std::all_of(id.begin(), id.end(), [](char c) {
        return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
    });
}

std::int64_t system_now_ms() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::system_clock::now().time_since_epoch())
        .count();
}

class Writer {
public:
    template <class T>
    void pod(T value) {
        static_assert(std::is_trivially_copyable_v<T>);
        bytes(&value, sizeof(T));
    }
    void bytes(const void* data, std::size_t count) {
        const auto* begin = static_cast<const std::uint8_t*>(data);
        out.insert(out.end(), begin, begin + count);
    }
    void string(const std::string& text) {
        pod<std::uint32_t>(static_cast<std::uint32_t>(text.size()));
        bytes(text.data(), text.size());
    }
    std::vector<std::uint8_t> out;
};

class Reader {
public:
    explicit Reader(std::span<const std::uint8_t> data) : data_(data) {}

    void bytes(void* out, std::size_t count) {
        if (count > data_.size() - cursor_) { throw std::runtime_error("manifest is truncated"); }
        std::memcpy(out, data_.data() + cursor_, count);
        cursor_ += count;
    }
    template <class T>
    T pod() {
        T value{};
        bytes(&value, sizeof(T));
        return value;
    }
    std::string string(std::size_t maximum) {
        const auto length = pod<std::uint32_t>();
        if (length > maximum) { throw std::runtime_error("manifest string is out of range"); }
        std::string text(length, '\0');
        bytes(text.data(), length);
        return text;
    }
    [[nodiscard]] std::size_t remaining() const noexcept { return data_.size() - cursor_; }

private:
    std::span<const std::uint8_t> data_;
    std::size_t cursor_ = 0;
};

std::vector<std::uint8_t> read_whole_file(const fs::path& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) { throw std::runtime_error("cannot open " + path.string()); }
    const std::streamsize size = file.tellg();
    if (size < 0) { throw std::runtime_error("cannot size " + path.string()); }
    std::vector<std::uint8_t> data(static_cast<std::size_t>(size));
    file.seekg(0);
    if (size != 0 && !file.read(reinterpret_cast<char*>(data.data()), size)) {
        throw std::runtime_error("cannot read " + path.string());
    }
    return data;
}

void remove_quietly(const fs::path& path) noexcept {
    std::error_code ignored;
    fs::remove(path, ignored);
}

// Compressed chunks move through zstd as streams of this size: a chunk is never held twice, once
// compressed and once raw, in the store's own buffers.
constexpr std::size_t kStreamBytes = std::size_t{256} << 10U;

// A compressed chunk is kept only when it is at least 3 % smaller than the raw one.
constexpr std::uint64_t compressed_limit(std::uint64_t raw) noexcept { return raw / 100 * 97; }

using CompressContext   = std::unique_ptr<ZSTD_CCtx, decltype(&ZSTD_freeCCtx)>;
using DecompressContext = std::unique_ptr<ZSTD_DCtx, decltype(&ZSTD_freeDCtx)>;

// Streams `bytes` through zstd into `staging`. The frame records the content size, so the raw
// length can be read from the first bytes of the file. Stops and returns nullopt as soon as the
// output passes the keep limit (the chunk does not compress enough to be worth it).
std::optional<std::uint64_t> compress_to_file(std::span<const std::uint8_t> bytes,
                                              const fs::path& staging, int level) {
    CompressContext context(ZSTD_createCCtx(), &ZSTD_freeCCtx);
    if (!context) { throw std::runtime_error("cannot create a zstd compression context"); }
    ZSTD_CCtx_setParameter(context.get(), ZSTD_c_compressionLevel, level);
    ZSTD_CCtx_setPledgedSrcSize(context.get(), bytes.size());
    std::ofstream file(staging, std::ios::binary | std::ios::trunc);
    std::vector<char> out(kStreamBytes);
    ZSTD_inBuffer input{bytes.data(), bytes.size(), 0};
    const std::uint64_t limit = compressed_limit(bytes.size());
    std::uint64_t total       = 0;
    for (;;) {
        ZSTD_outBuffer sink{out.data(), out.size(), 0};
        const std::size_t remaining = ZSTD_compressStream2(context.get(), &sink, &input, ZSTD_e_end);
        if (ZSTD_isError(remaining)) {
            throw std::runtime_error(std::string("zstd: ") + ZSTD_getErrorName(remaining));
        }
        file.write(out.data(), static_cast<std::streamsize>(sink.pos));
        total += sink.pos;
        if (total > limit) { return std::nullopt; }
        if (remaining == 0) { break; }
    }
    file.flush();
    if (!file.good()) { throw std::runtime_error("cannot write " + staging.string()); }
    return total;
}

// Decodes one zstd frame that must be exactly `out.size()` bytes. `next` fills the given buffer
// with the next compressed bytes and returns how many it wrote (0 at the end of the input). False
// for a damaged frame, a frame of another length, or input that ends early or runs on.
template <class Next>
bool decompress_stream(Next&& next, std::span<std::uint8_t> out) {
    DecompressContext context(ZSTD_createDCtx(), &ZSTD_freeDCtx);
    if (!context) { return false; }
    std::vector<char> buffer(kStreamBytes);
    ZSTD_outBuffer sink{out.data(), out.size(), 0};
    for (;;) {
        const std::size_t got = next(buffer);
        if (got == 0) { return false; } // ended before the frame did
        ZSTD_inBuffer input{buffer.data(), got, 0};
        while (input.pos < input.size) {
            const std::size_t more = ZSTD_decompressStream(context.get(), &sink, &input);
            if (ZSTD_isError(more)) { return false; }
            if (more == 0) {
                return sink.pos == out.size() && input.pos == input.size && next(buffer) == 0;
            }
            if (sink.pos == sink.size) { return false; } // the frame holds more than `out`
        }
    }
}

bool decompress_file(const fs::path& path, std::span<std::uint8_t> out) {
    std::ifstream file(path, std::ios::binary);
    if (!file) { return false; }
    return decompress_stream(
        [&](std::vector<char>& buffer) {
            file.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
            return static_cast<std::size_t>(file.gcount());
        },
        out);
}

bool decompress_memory(std::span<const std::uint8_t> packed, std::span<std::uint8_t> out) {
    std::size_t position = 0;
    return decompress_stream(
        [&](std::vector<char>& buffer) {
            const std::size_t count = std::min(buffer.size(), packed.size() - position);
            std::memcpy(buffer.data(), packed.data() + position, count);
            position += count;
            return count;
        },
        out);
}

// The raw length a compressed chunk file declares in its frame header; nullopt when the file is
// missing or does not start with a frame that records its size.
std::optional<std::uint64_t> compressed_content_size(const fs::path& path) {
    std::ifstream file(path, std::ios::binary);
    char head[32];
    if (!file) { return std::nullopt; }
    file.read(head, sizeof(head));
    const auto size = ZSTD_getFrameContentSize(head, static_cast<std::size_t>(file.gcount()));
    if (size == ZSTD_CONTENTSIZE_UNKNOWN || size == ZSTD_CONTENTSIZE_ERROR) { return std::nullopt; }
    return size;
}

} // namespace

std::array<std::uint64_t, 2> ContextStore::hash(std::span<const std::uint8_t> bytes) {
    return {xxh64(bytes.data(), bytes.size(), 0), xxh64(bytes.data(), bytes.size(), 0x9e3779b97f4a7c15ULL)};
}

std::string ContextStore::image_id(const std::array<std::uint64_t, 2>& digests,
                                   std::uint32_t frontier, bool shared) {
    return context_store_image_id(digests, frontier, shared);
}

// An exclusive lock on one file of the directory, held for the store's life. The directory's logic
// (refcounts, scan, garbage collection) is only sound with one process in it, so a second owner
// must fail at start-up rather than corrupt it. The OS releases the lock when the process dies.
struct ContextStore::OwnerLock {
    explicit OwnerLock(const fs::path& path) {
#ifdef _WIN32
        handle_ = CreateFileW(path.c_str(), GENERIC_READ | GENERIC_WRITE, 0, nullptr, OPEN_ALWAYS,
                              FILE_ATTRIBUTE_NORMAL, nullptr);
        if (handle_ == INVALID_HANDLE_VALUE) {
            const DWORD code = GetLastError();
            throw std::runtime_error(code == ERROR_SHARING_VIOLATION
                                         ? "in use by another process"
                                         : "cannot open " + path.string() + " (error " +
                                               std::to_string(code) + ")");
        }
#else
        fd_ = ::open(path.c_str(), O_RDWR | O_CREAT | O_CLOEXEC, 0644);
        if (fd_ < 0) { throw std::runtime_error("cannot open " + path.string()); }
        if (::flock(fd_, LOCK_EX | LOCK_NB) != 0) {
            const int code = errno;
            ::close(fd_);
            throw std::runtime_error(code == EWOULDBLOCK ? "in use by another process"
                                                         : "cannot lock " + path.string());
        }
#endif
    }
    ~OwnerLock() {
#ifdef _WIN32
        CloseHandle(handle_);
#else
        ::close(fd_);
#endif
    }
    OwnerLock(const OwnerLock&)            = delete;
    OwnerLock& operator=(const OwnerLock&) = delete;

private:
#ifdef _WIN32
    HANDLE handle_ = INVALID_HANDLE_VALUE;
#else
    int fd_ = -1;
#endif
};

ContextStore::ContextStore(Options options) : options_(std::move(options)) {
    if (options_.directory.empty()) {
        throw std::invalid_argument("context store directory is empty");
    }
    if (options_.chunk_bytes == 0 ||
        options_.chunk_bytes > std::numeric_limits<std::uint32_t>::max()) {
        throw std::invalid_argument("context store chunk size is out of range");
    }
    if (!options_.clock) { options_.clock = system_now_ms; }
    std::error_code error;
    for (const char* sub : {"chunks", "manifests", "tmp"}) {
        fs::create_directories(options_.directory / sub, error);
        if (error) {
            throw std::runtime_error("cannot create context store directory " +
                                     (options_.directory / sub).string() + ": " + error.message());
        }
    }
    if (!options_.peers.empty() && options_.peer_poll.count() <= 0) {
        throw std::invalid_argument("context store peer_poll must be positive");
    }
    for (const fs::path& peer : options_.peers) {
        std::error_code same_error;
        if (peer.empty() || fs::equivalent(peer, options_.directory, same_error)) {
            throw std::invalid_argument("a context store peer must be another store's directory");
        }
    }
    try {
        owner_ = std::make_unique<OwnerLock>(options_.directory / "owner.lock");
    } catch (const std::exception& failure) {
        throw std::runtime_error("context store directory " + options_.directory.string() + ": " +
                                 failure.what());
    }
    {
        std::scoped_lock io(io_mutex_);
        std::scoped_lock lock(mutex_);
        scan();
        maintain_locked(nullptr);
    }
    if (options_.remote) {
        remote_thread_ = std::thread([this] { remote_loop(); });
        enqueue_remote({RemoteTask::Kind::Refresh, std::string(), {}, {}, {}, {}});
    }
    if (!options_.peers.empty()) { peer_thread_ = std::thread([this] { peer_loop(); }); }
}

ContextStore::~ContextStore() {
    {
        std::scoped_lock lock(peer_mutex_);
        peer_stopping_ = true;
    }
    peer_cv_.notify_all();
    if (peer_thread_.joinable()) { peer_thread_.join(); }
    {
        std::scoped_lock lock(remote_mutex_);
        remote_stopping_ = true;
    }
    remote_cv_.notify_all();
    // A transfer in progress can take minutes on a bad link; the budget was spent before this point.
    if (options_.remote) { options_.remote->interrupt(); }
    if (remote_thread_.joinable()) { remote_thread_.join(); }
}

std::int64_t ContextStore::now_ms() const { return options_.clock(); }

fs::path ContextStore::manifest_path(const std::string& id) const {
    return options_.directory / "manifests" / (id + ".manifest");
}

fs::path ContextStore::published_path(const std::string& id) const {
    return options_.directory / "manifests" / (id + ".published");
}

void ContextStore::write_published_locked(const std::string& id,
                                          std::span<const std::uint8_t> manifest) const {
    // The manifest's own checksum, its last 8 bytes, names exactly the manifest published.
    if (manifest.size() < sizeof(std::uint64_t)) { return; }
    try {
        write_atomically(published_path(id), manifest.subspan(manifest.size() - sizeof(std::uint64_t)));
    } catch (const std::exception&) {
        // Without the marker a restart only treats the image as unpublished: removed whole on
        // eviction instead of kept for the bucket.
    }
}

fs::path ContextStore::used_path(const std::string& id) const {
    return options_.directory / "manifests" / (id + ".used");
}

fs::path ContextStore::chunk_path(const std::array<std::uint64_t, 2>& hash) const {
    const std::string key = hex_key(hash);
    return options_.directory / "chunks" / key.substr(0, 2) / (key + ".chunk");
}

fs::path ContextStore::compressed_chunk_path(const std::array<std::uint64_t, 2>& hash) const {
    const std::string key = hex_key(hash);
    return options_.directory / "chunks" / key.substr(0, 2) / (key + ".chunk.zst");
}

std::optional<ContextStore::StoredChunk> ContextStore::locate_chunk(const ChunkRef& chunk) const {
    std::error_code error;
    const auto raw = fs::file_size(chunk_path(chunk.hash), error);
    if (!error && raw == chunk.length) { return StoredChunk{false, raw}; }
    const fs::path packed = compressed_chunk_path(chunk.hash);
    const auto size       = fs::file_size(packed, error);
    if (error) { return std::nullopt; }
    const auto content = compressed_content_size(packed);
    if (!content || *content != chunk.length) { return std::nullopt; }
    return StoredChunk{true, size};
}

bool ContextStore::read_chunk(const ChunkRef& chunk, std::span<std::uint8_t> out) const {
    {
        std::ifstream file(chunk_path(chunk.hash), std::ios::binary);
        if (file && file.read(reinterpret_cast<char*>(out.data()),
                              static_cast<std::streamsize>(out.size()))) {
            return true;
        }
    }
    return decompress_file(compressed_chunk_path(chunk.hash), out);
}

std::pair<fs::path, ContextStore::StoredChunk>
ContextStore::write_chunk(const ChunkRef& chunk, std::span<const std::uint8_t> bytes) const {
    if (options_.compress_chunks) {
        const fs::path final_path = compressed_chunk_path(chunk.hash);
        const fs::path staging    = staging_path();
        std::error_code error;
        fs::create_directories(final_path.parent_path(), error);
        std::optional<std::uint64_t> packed;
        try {
            packed = compress_to_file(bytes, staging, options_.compression_level);
        } catch (...) {
            remove_quietly(staging);
            throw;
        }
        if (packed) {
            fs::rename(staging, final_path, error);
            if (error) {
                remove_quietly(staging);
                throw std::runtime_error("cannot publish " + final_path.string() + ": " +
                                         error.message());
            }
            return {final_path, StoredChunk{true, *packed}};
        }
        remove_quietly(staging); // not worth compressing: kept raw
    }
    write_atomically(chunk_path(chunk.hash), bytes);
    return {chunk_path(chunk.hash), StoredChunk{false, bytes.size()}};
}

void ContextStore::install_chunk_file(const ChunkRef& chunk, bool compressed,
                                      std::span<const std::uint8_t> file_bytes) const {
    write_atomically(compressed ? compressed_chunk_path(chunk.hash) : chunk_path(chunk.hash),
                     file_bytes);
}

std::vector<std::uint8_t> ContextStore::verified_chunk_file(const ChunkRef& chunk,
                                                            bool& compressed) const {
    const auto located = locate_chunk(chunk);
    if (!located) { throw std::runtime_error("a chunk of the image is missing on disk"); }
    std::vector<std::uint8_t> raw(chunk.length);
    if (!read_chunk(chunk, raw) || hash(raw) != chunk.hash) {
        throw std::runtime_error("a chunk of the image is damaged on disk");
    }
    compressed = located->compressed;
    return compressed ? read_whole_file(compressed_chunk_path(chunk.hash)) : raw;
}

fs::path ContextStore::staging_path() const {
    static std::atomic<std::uint64_t> counter{0};
    std::ostringstream name;
    name << std::this_thread::get_id() << '.' << counter.fetch_add(1) << ".tmp";
    return options_.directory / "tmp" / name.str();
}

void ContextStore::remove_chunk_files(const std::array<std::uint64_t, 2>& hash) const noexcept {
    remove_quietly(chunk_path(hash));
    remove_quietly(compressed_chunk_path(hash));
}

void ContextStore::write_atomically(const fs::path& path, std::span<const std::uint8_t> bytes) const {
    const fs::path staging = staging_path();
    std::error_code error;
    fs::create_directories(path.parent_path(), error);
    {
        std::ofstream file(staging, std::ios::binary | std::ios::trunc);
        file.write(reinterpret_cast<const char*>(bytes.data()),
                   static_cast<std::streamsize>(bytes.size()));
        file.flush();
        if (!file.good()) {
            file.close();
            remove_quietly(staging);
            throw std::runtime_error("cannot write " + staging.string());
        }
    }
    fs::rename(staging, path, error);
    if (error) {
        remove_quietly(staging);
        throw std::runtime_error("cannot publish " + path.string() + ": " + error.message());
    }
}

void ContextStore::add_references(const Entry& entry) {
    for (const ChunkRef& chunk : entry.chunks) {
        ChunkUse& use = chunks_[hex_key(chunk.hash)];
        // A chunk kept alive only by a queued upload is already counted and described.
        if (use.references++ == 0 && use.pins == 0) {
            use.length = chunk.length;
            // An image a remote holds is registered before its chunks are fetched.
            const auto located = locate_chunk(chunk);
            use.present        = located.has_value();
            use.stored         = located ? located->bytes : 0;
            if (use.present) { chunk_total_bytes_ += use.stored; }
        }
    }
}

void ContextStore::release_references(const Entry& entry) {
    for (const ChunkRef& chunk : entry.chunks) {
        const auto found = chunks_.find(hex_key(chunk.hash));
        if (found == chunks_.end()) { continue; }
        if (--found->second.references == 0 && found->second.pins == 0) {
            if (found->second.present) { chunk_total_bytes_ -= found->second.stored; }
            remove_chunk_files(chunk.hash);
            chunks_.erase(found);
        }
    }
}

void ContextStore::unpin_chunks_locked(const std::vector<ChunkRef>& chunks) {
    for (const ChunkRef& chunk : chunks) {
        const auto found = chunks_.find(hex_key(chunk.hash));
        if (found == chunks_.end() || found->second.pins == 0) { continue; }
        if (--found->second.pins == 0 && found->second.references == 0) {
            if (found->second.present) { chunk_total_bytes_ -= found->second.stored; }
            remove_chunk_files(chunk.hash);
            chunks_.erase(found);
        }
    }
}

void ContextStore::remove_entry_locked(const std::string& id, bool corrupt) {
    const auto found = entries_.find(id);
    if (found != entries_.end()) {
        release_references(found->second);
        manifest_total_bytes_ -= found->second.manifest_bytes;
        entries_.erase(found);
        if (corrupt) { ++stats_.corrupt_removed; }
    }
    remove_quietly(manifest_path(id));
    remove_quietly(used_path(id));
    remove_quietly(published_path(id));
}

void ContextStore::quarantine_peers_locked(Entry& entry, const std::vector<ChunkRef>* damaged) {
    if (damaged == nullptr) {
        if (!entry.peer_memo_key.empty()) { peer_skipped_[entry.peer_memo_key] = entry.peer_stamp; }
        return;
    }
    // A damaged chunk this store wrote, fetched or shared with another image says nothing about a
    // peer: that peer is asked again, and its intact copy can still complete the image. The damaged
    // files are discarded, so whatever replaces them is no longer the peer's.
    for (auto& [memo_key, fill] : entry.peer_fills) {
        for (const ChunkRef& chunk : *damaged) {
            if (fill.chunk_keys.erase(hex_key(chunk.hash)) != 0) {
                peer_skipped_[memo_key] = fill.stamp;
            }
        }
    }
}

void ContextStore::write_used(const std::string& id, std::int64_t last_used_ms) const {
    const std::string text = std::to_string(last_used_ms);
    try {
        write_atomically(used_path(id),
                         std::span<const std::uint8_t>(
                             reinterpret_cast<const std::uint8_t*>(text.data()), text.size()));
    } catch (const std::exception&) {
        // A stale use time only changes which image is evicted first.
    }
}

std::uint64_t ContextStore::used_bytes_locked() const {
    return chunk_total_bytes_ + manifest_total_bytes_;
}

bool ContextStore::read_manifest(const fs::path& path, Entry& entry,
                                 std::vector<Segment>* segments) const {
    try {
        return parse_manifest(read_whole_file(path), entry, segments);
    } catch (const std::exception&) {
        return false;
    }
}

bool ContextStore::parse_manifest(std::span<const std::uint8_t> data, Entry& entry,
                                  std::vector<Segment>* segments) const {
    try {
        if (data.size() < sizeof(kManifestMagic) + sizeof(std::uint64_t)) { return false; }
        const std::span<const std::uint8_t> body(data.data(), data.size() - sizeof(std::uint64_t));
        std::uint64_t stored = 0;
        std::memcpy(&stored, data.data() + body.size(), sizeof(stored));
        if (xxh64(body.data(), body.size(), 0x4e43u) != stored) { return false; }
        Reader reader(body);
        char magic[sizeof(kManifestMagic)];
        reader.bytes(magic, sizeof(magic));
        if (std::memcmp(magic, kManifestMagic, sizeof(magic)) != 0) { return false; }
        const auto version = reader.pod<std::uint32_t>();
        if (version != 1 && version != kManifestVersion) { return false; }
        Info& info       = entry.info;
        info.id          = reader.string(64);
        info.binding     = reader.string(1U << 16U);
        info.tokens      = reader.pod<std::uint32_t>();
        info.image_bytes = reader.pod<std::uint64_t>();
        info.created_ms  = reader.pod<std::int64_t>();
        info.shared      = version >= 2 && reader.pod<std::uint8_t>() != 0;
        const auto checkpoint_count = reader.pod<std::uint32_t>();
        if (checkpoint_count > kMaximumCheckpoints) { return false; }
        info.checkpoints.resize(checkpoint_count);
        for (CheckpointKey& key : info.checkpoints) {
            key.frontier     = reader.pod<std::uint32_t>();
            key.digests[0]   = reader.pod<std::uint64_t>();
            key.digests[1]   = reader.pod<std::uint64_t>();
            key.identity_tag = reader.pod<std::uint32_t>();
        }
        const auto segment_count = reader.pod<std::uint32_t>();
        if (segment_count > kMaximumSegments) { return false; }
        std::uint64_t covered = 0;
        std::uint64_t total_chunks = 0;
        for (std::uint32_t index = 0; index < segment_count; ++index) {
            Segment segment;
            segment.chunked = reader.pod<std::uint8_t>() != 0;
            segment.length  = reader.pod<std::uint64_t>();
            if (segment.chunked) {
                const auto chunk_count = reader.pod<std::uint32_t>();
                total_chunks += chunk_count;
                if (total_chunks > kMaximumChunksPerImage) { return false; }
                std::uint64_t chunk_sum = 0;
                segment.chunks.resize(chunk_count);
                for (ChunkRef& chunk : segment.chunks) {
                    chunk.hash[0] = reader.pod<std::uint64_t>();
                    chunk.hash[1] = reader.pod<std::uint64_t>();
                    chunk.length  = reader.pod<std::uint32_t>();
                    chunk_sum += chunk.length;
                    entry.chunks.push_back(chunk);
                }
                if (chunk_sum != segment.length) { return false; }
            } else {
                if (segment.length > reader.remaining()) { return false; }
                if (segments != nullptr) {
                    segment.inline_bytes.resize(static_cast<std::size_t>(segment.length));
                    reader.bytes(segment.inline_bytes.data(), segment.inline_bytes.size());
                } else {
                    std::vector<std::uint8_t> skipped(static_cast<std::size_t>(segment.length));
                    reader.bytes(skipped.data(), skipped.size());
                }
            }
            covered += segment.length;
            if (segments != nullptr) { segments->push_back(std::move(segment)); }
        }
        if (covered != info.image_bytes || reader.remaining() != 0 || !valid_id(info.id)) {
            return false;
        }
        entry.manifest_bytes = data.size();
        return true;
    } catch (const std::exception&) {
        return false;
    }
}

void ContextStore::scan() {
    std::error_code error;
    for (const auto& item : fs::directory_iterator(options_.directory / "tmp", error)) {
        remove_quietly(item.path());
    }
    for (const auto& item : fs::directory_iterator(options_.directory / "manifests", error)) {
        const fs::path& path = item.path();
        if (path.extension() != ".manifest") { continue; }
        Entry entry;
        const std::string id = path.stem().string();
        if (!valid_id(id) || !read_manifest(path, entry, nullptr) || entry.info.id != id) {
            // A manifest of this store's own naming that does not describe its image is a damaged
            // session; a file under another name was never one.
            if (valid_id(id)) {
                ++stats_.corrupt_removed;
                ++stats_.corrupt_manifest;
            }
            remove_quietly(path);
            remove_quietly(used_path(id));
            remove_quietly(published_path(id));
            continue;
        }
        entry.info.last_used_ms = entry.info.created_ms;
        if (options_.remote) {
            // Published before the restart, and this manifest is the one published (not a newer
            // write of the id whose upload never ran).
            try {
                const std::vector<std::uint8_t> marker   = read_whole_file(published_path(id));
                const std::vector<std::uint8_t> manifest = read_whole_file(path);
                entry.published = marker.size() == sizeof(std::uint64_t) &&
                                  manifest.size() >= marker.size() &&
                                  std::equal(marker.begin(), marker.end(),
                                             manifest.end() - static_cast<std::ptrdiff_t>(marker.size()));
            } catch (const std::exception&) {}
        }
        try {
            const std::vector<std::uint8_t> used = read_whole_file(used_path(id));
            const std::string text(used.begin(), used.end());
            entry.info.last_used_ms = std::max<std::int64_t>(entry.info.created_ms, std::stoll(text));
        } catch (const std::exception&) {}
        manifest_total_bytes_ += entry.manifest_bytes;
        add_references(entry);
        entry.generation = ++next_generation_;
        entries_.emplace(id, std::move(entry));
    }
    // Chunks no manifest names are left over from an interrupted write or an eviction.
    for (const auto& shard : fs::directory_iterator(options_.directory / "chunks", error)) {
        if (!shard.is_directory()) { continue; }
        for (const auto& item : fs::directory_iterator(shard.path(), error)) {
            const std::string name = item.path().filename().string();
            constexpr std::string_view kRaw = ".chunk";
            constexpr std::string_view kPacked = ".chunk.zst";
            const auto ends_with = [&](std::string_view suffix) {
                return name.size() > suffix.size() &&
                       name.compare(name.size() - suffix.size(), suffix.size(), suffix) == 0;
            };
            std::string key;
            if (ends_with(kPacked)) {
                key = name.substr(0, name.size() - kPacked.size());
            } else if (ends_with(kRaw)) {
                key = name.substr(0, name.size() - kRaw.size());
            }
            if (key.empty() || chunks_.find(key) == chunks_.end()) { remove_quietly(item.path()); }
        }
    }
    // An image whose chunk is missing or the wrong size can never be reassembled.
    std::vector<std::string> broken;
    for (const auto& [id, entry] : entries_) {
        for (const ChunkRef& chunk : entry.chunks) {
            if (!locate_chunk(chunk)) {
                if (options_.remote) {
                    // The remote still has it: a missing chunk is fetched when the image is
                    // loaded, and a short one is discarded so it is fetched again.
                    remove_chunk_files(chunk.hash);
                    continue;
                }
                broken.push_back(id);
                break;
            }
        }
    }
    stats_.corrupt_startup += broken.size();
    for (const std::string& id : broken) { remove_entry_locked(id, true); }
}

bool ContextStore::entry_local_locked(const Entry& entry) const {
    for (const ChunkRef& chunk : entry.chunks) {
        const auto found = chunks_.find(hex_key(chunk.hash));
        if (found == chunks_.end() || !found->second.present) { return false; }
    }
    return true;
}

bool ContextStore::entry_has_local_bytes_locked(const Entry& entry) const {
    // No chunk references: the whole image is inline in the manifest, which is always local.
    if (entry.chunks.empty()) { return true; }
    for (const ChunkRef& chunk : entry.chunks) {
        const auto found = chunks_.find(hex_key(chunk.hash));
        if (found != chunks_.end() && found->second.present) { return true; }
    }
    return false;
}

bool ContextStore::evict_oldest_locked(const std::string* protect) {
    // Shared prefixes go last: each serves every conversation that starts with it, and its chunks
    // are most of theirs. Otherwise the least recently used first.
    const auto evict_before = [](const Entry& a, const Entry& b) {
        if (a.info.shared != b.info.shared) { return !a.info.shared; }
        return a.info.last_used_ms < b.info.last_used_ms;
    };
    if (!options_.remote) {
        const Entry* oldest = nullptr;
        for (const auto& [id, entry] : entries_) {
            if (protect != nullptr && id == *protect) { continue; }
            if (oldest == nullptr || evict_before(entry, *oldest)) { oldest = &entry; }
        }
        if (oldest == nullptr) { return false; }
        const std::string id = oldest->info.id; // the entry, and its id, go with the removal
        remove_entry_locked(id, false);
        ++stats_.evicted_for_space;
        return true;
    }
    // With a remote the least recently used image with files of its own gives them up and stays
    // offered. Files a queued upload pins stay until it is done; an image evicted before is taken
    // again only for files it can free now, so every eviction makes progress.
    std::vector<Entry*> order;
    for (auto& [id, entry] : entries_) {
        if (protect == nullptr || id != *protect) { order.push_back(&entry); }
    }
    std::sort(order.begin(), order.end(),
              [&](const Entry* a, const Entry* b) { return evict_before(*a, *b); });
    const auto holders = whole_image_holders_locked();
    for (Entry* entry : order) {
        if (!entry->published) {
            // The bucket may not hold it (its upload is pending or failed): removed whole, as
            // without a remote, never kept as an entry whose chunks exist nowhere. A pending
            // upload still runs from its pinned files and a later listing registers the image.
            if (!entry_has_local_bytes_locked(*entry)) { continue; }
            const std::string id = entry->info.id; // the entry, and its id, go with the removal
            remove_entry_locked(id, false);
            ++stats_.evicted_for_space;
            return true;
        }
        const EvictionYield yield = evict_to_remote_locked(*entry, holders, false);
        if (yield.freed == 0 && (entry->evicted || yield.pinned == 0)) { continue; }
        (void)evict_to_remote_locked(*entry, holders, true);
        ++stats_.evicted_for_space;
        return true;
    }
    return false;
}

bool ContextStore::held_whole_locked(const Entry& entry) const {
    return !entry.evicted && entry_local_locked(entry);
}

std::unordered_map<std::string, std::uint32_t> ContextStore::whole_image_holders_locked() const {
    std::unordered_map<std::string, std::uint32_t> holders;
    for (const auto& [id, entry] : entries_) {
        if (!held_whole_locked(entry)) { continue; }
        std::unordered_set<std::string> seen;
        for (const ChunkRef& chunk : entry.chunks) {
            std::string key = hex_key(chunk.hash);
            if (seen.insert(key).second) { ++holders[std::move(key)]; }
        }
    }
    return holders;
}

ContextStore::EvictionYield ContextStore::evict_to_remote_locked(
    Entry& entry, const std::unordered_map<std::string, std::uint32_t>& holders, bool apply) {
    EvictionYield yield;
    const std::uint32_t self = held_whole_locked(entry) ? 1U : 0U;
    std::unordered_set<std::string> seen;
    for (const ChunkRef& chunk : entry.chunks) {
        const std::string key = hex_key(chunk.hash);
        if (!seen.insert(key).second) { continue; }
        // Chunk files another image the directory still holds whole also uses stay.
        const auto held            = holders.find(key);
        const std::uint32_t others = (held == holders.end() ? 0U : held->second) - self;
        const auto found = chunks_.find(key);
        if (found == chunks_.end() || !found->second.present || others != 0) { continue; }
        // A queued upload still reads a pinned file: a later eviction takes it.
        if (found->second.pins != 0) {
            yield.pinned += found->second.stored;
            continue;
        }
        yield.freed += found->second.stored;
        if (!apply) { continue; }
        remove_chunk_files(chunk.hash);
        chunk_total_bytes_ -= found->second.stored;
        found->second.present = false;
        found->second.stored  = 0;
    }
    if (apply) { entry.evicted = true; }
    return yield;
}

void ContextStore::maintain_locked(const std::string* protect) {
    if (options_.ttl.count() > 0) {
        const std::int64_t horizon = now_ms() - options_.ttl.count() * 1000;
        std::vector<std::string> expired;
        for (const auto& [id, entry] : entries_) {
            // An image only the bucket holds ages there, by its lifecycle rule, not by this
            // directory's TTL; the next refresh drops it from the index when the bucket does.
            if (options_.remote && (entry.evicted || !entry_has_local_bytes_locked(entry))) {
                continue;
            }
            if ((protect == nullptr || id != *protect) && entry.info.last_used_ms < horizon) {
                expired.push_back(id);
            }
        }
        for (const std::string& id : expired) {
            // With a remote only the directory's copy goes, the bucket's still offered, once the
            // bucket is known to hold it; otherwise the image is removed whole.
            if (options_.remote && entries_.at(id).published) {
                (void)evict_to_remote_locked(entries_.at(id), whole_image_holders_locked(), true);
            } else {
                remove_entry_locked(id, false);
            }
            ++stats_.expired;
        }
    }
    if (options_.max_bytes != 0) {
        while (used_bytes_locked() > options_.max_bytes &&
               entries_.size() > (protect != nullptr ? 1U : 0U)) {
            if (!evict_oldest_locked(protect)) { break; }
        }
    }
}

void ContextStore::maintain() {
    std::scoped_lock io(io_mutex_);
    std::scoped_lock lock(mutex_);
    maintain_locked(nullptr);
}

ContextStore::PutResult ContextStore::put(const Description& description,
                                          std::span<const std::uint8_t> image,
                                          std::span<const Region> regions) {
    if (!valid_id(description.id)) { throw std::invalid_argument("context store id is invalid"); }
    if (description.checkpoints.size() > kMaximumCheckpoints) {
        throw std::invalid_argument("too many checkpoint keys");
    }
    std::uint64_t cursor = 0;
    for (const Region& region : regions) {
        if (region.length == 0 || region.offset < cursor ||
            region.offset > image.size() || region.length > image.size() - region.offset) {
            throw std::invalid_argument("context store regions must be ordered and inside the image");
        }
        cursor = region.offset + region.length;
    }

    std::scoped_lock io(io_mutex_);
    PutResult result;
    result.image_bytes = image.size();
    std::vector<Segment> segments;
    std::vector<ChunkRef> all_chunks;
    std::vector<fs::path> written;
    // Chunks the index knew but the directory lacked, now written, with their size on disk.
    std::vector<std::pair<std::string, std::uint64_t>> refilled;
    std::uint64_t stored_written = 0;
    std::unordered_set<std::string> seen_in_put;
    try {
        // Stores `bytes` as one chunk of `segment`, writing it only when the directory lacks it.
        const auto add_chunk = [&](Segment& segment, std::span<const std::uint8_t> bytes) {
            ChunkRef chunk{.hash = hash(bytes), .length = static_cast<std::uint32_t>(bytes.size())};
            const std::string key = hex_key(chunk.hash);
            bool present;
            bool known;
            {
                std::scoped_lock lock(mutex_);
                const auto found = chunks_.find(key);
                known            = found != chunks_.end();
                // An image registered from the bucket names chunks the directory does not hold.
                present = known && found->second.present;
            }
            if (!present && seen_in_put.find(key) == seen_in_put.end()) {
                const auto [path, stored] = write_chunk(chunk, bytes);
                written.push_back(path);
                if (known) { refilled.emplace_back(key, stored.bytes); }
                result.chunk_bytes_written += bytes.size();
                stored_written += stored.bytes;
            } else {
                result.chunk_bytes_reused += bytes.size();
            }
            seen_in_put.insert(key);
            segment.chunks.push_back(chunk);
            all_chunks.push_back(chunk);
        };
        // The bytes between regions (the image's header: its token ledger and identity, several
        // MB for a deep context) change a little at many places each turn, as every per-token table
        // in it grows. Beyond a small size they are cut where their content says (not at fixed
        // offsets), so an unchanged stretch keeps its chunks however much the tables before it
        // grew, and the manifest itself stays small.
        const auto add_between = [&](std::uint64_t begin, std::uint64_t end) {
            if (end <= begin) { return; }
            Segment segment;
            segment.length = end - begin;
            const std::span<const std::uint8_t> bytes =
                image.subspan(static_cast<std::size_t>(begin), static_cast<std::size_t>(end - begin));
            if (bytes.size() < kContentChunkMinimumSegment) {
                segment.inline_bytes.assign(bytes.begin(), bytes.end());
            } else {
                segment.chunked = true;
                for (std::size_t done = 0; done < bytes.size();) {
                    const std::size_t size = content_defined_cut(bytes.subspan(done));
                    add_chunk(segment, bytes.subspan(done, size));
                    done += size;
                }
            }
            segments.push_back(std::move(segment));
        };
        std::uint64_t position = 0;
        for (const Region& region : regions) {
            add_between(position, region.offset);
            Segment segment;
            segment.chunked = true;
            segment.length  = region.length;
            for (std::uint64_t done = 0; done < region.length; done += options_.chunk_bytes) {
                const std::uint64_t size = std::min(options_.chunk_bytes, region.length - done);
                add_chunk(segment, image.subspan(static_cast<std::size_t>(region.offset + done),
                                                 static_cast<std::size_t>(size)));
            }
            segments.push_back(std::move(segment));
            position = region.offset + region.length;
        }
        add_between(position, image.size());

        Writer manifest;
        const std::int64_t now = now_ms();
        manifest.bytes(kManifestMagic, sizeof(kManifestMagic));
        manifest.pod(kManifestVersion);
        manifest.string(description.id);
        manifest.string(description.binding);
        manifest.pod<std::uint32_t>(description.tokens);
        manifest.pod<std::uint64_t>(image.size());
        manifest.pod<std::int64_t>(now);
        manifest.pod<std::uint8_t>(description.shared ? 1 : 0);
        manifest.pod<std::uint32_t>(static_cast<std::uint32_t>(description.checkpoints.size()));
        for (const CheckpointKey& key : description.checkpoints) {
            manifest.pod(key.frontier);
            manifest.pod(key.digests[0]);
            manifest.pod(key.digests[1]);
            manifest.pod(key.identity_tag);
        }
        manifest.pod<std::uint32_t>(static_cast<std::uint32_t>(segments.size()));
        for (const Segment& segment : segments) {
            manifest.pod<std::uint8_t>(segment.chunked ? 1 : 0);
            manifest.pod<std::uint64_t>(segment.length);
            if (segment.chunked) {
                manifest.pod<std::uint32_t>(static_cast<std::uint32_t>(segment.chunks.size()));
                for (const ChunkRef& chunk : segment.chunks) {
                    manifest.pod(chunk.hash[0]);
                    manifest.pod(chunk.hash[1]);
                    manifest.pod(chunk.length);
                }
            } else {
                manifest.bytes(segment.inline_bytes.data(), segment.inline_bytes.size());
            }
        }
        const std::uint64_t checksum =
            xxh64(manifest.out.data(), manifest.out.size(), 0x4e43u);
        manifest.pod(checksum);
        write_atomically(manifest_path(description.id), manifest.out);

        Entry entry;
        entry.info = Info{.id           = description.id,
                          .binding      = description.binding,
                          .tokens       = description.tokens,
                          .image_bytes  = image.size(),
                          .created_ms   = now,
                          .last_used_ms = now,
                          .checkpoints  = description.checkpoints,
                          .shared       = description.shared};
        entry.manifest_bytes = manifest.out.size();
        entry.chunks         = std::move(all_chunks);

        std::scoped_lock lock(mutex_);
        const auto previous = entries_.find(description.id);
        // Reference the new image before releasing the one it replaces, so shared chunks survive.
        Entry replaced;
        const bool had_previous = previous != entries_.end();
        if (had_previous) {
            replaced = std::move(previous->second);
            entries_.erase(previous);
            manifest_total_bytes_ -= replaced.manifest_bytes;
        }
        manifest_total_bytes_ += entry.manifest_bytes;
        for (const auto& [key, stored] : refilled) {
            const auto found = chunks_.find(key);
            if (found != chunks_.end() && !found->second.present) {
                found->second.present = true;
                found->second.stored  = stored;
                chunk_total_bytes_ += stored;
            }
        }
        add_references(entry);
        // The upload works from this snapshot: the entry may be replaced or evicted before it runs.
        std::vector<ChunkRef> upload_chunks;
        if (options_.remote) {
            upload_chunks = entry.chunks;
            for (const ChunkRef& chunk : upload_chunks) { ++chunks_[hex_key(chunk.hash)].pins; }
        }
        entry.generation = ++next_generation_;
        entries_.emplace(description.id, std::move(entry));
        {
            // A fresh local write supersedes whatever made an older image of this id unusable.
            std::scoped_lock remote_lock(remote_mutex_);
            remote_failed_.erase(description.id);
        }
        if (had_previous) { release_references(replaced); }
        write_used(description.id, now);
        std::vector<std::string> superseded;
        std::vector<std::string> prefixes_used;
        if (!description.prefix_digests.empty()) {
            for (auto& [other_id, other] : entries_) {
                if (other_id == description.id || other.info.binding != description.binding ||
                    description.shared || other.info.tokens > description.tokens ||
                    other.info.checkpoints.empty()) {
                    continue;
                }
                const CheckpointKey& endpoint = other.info.checkpoints.front();
                if (!description.checkpoints.empty() &&
                    endpoint.identity_tag != description.checkpoints.front().identity_tag) {
                    continue;
                }
                if (endpoint.frontier >= description.prefix_digests.size() ||
                    description.prefix_digests[endpoint.frontier] != endpoint.digests) {
                    continue;
                }
                if (other.info.shared) {
                    // Every conversation reproduces the shared prefix it started from; that prefix
                    // still serves the other conversations, so it is never superseded by one. Its
                    // conversations reuse it from memory without loading it, so this write is
                    // its use: otherwise it would age as unused and be the first image evicted
                    // (or expire, here and in the bucket) while every conversation relies on it.
                    other.info.last_used_ms = now;
                    write_used(other_id, now);
                    prefixes_used.push_back(other_id);
                    continue;
                }
                superseded.push_back(other_id);
            }
            if (options_.remote) {
                // Restarts the bucket copy's age (at most once a day per image).
                for (const std::string& prefix : prefixes_used) {
                    enqueue_remote({RemoteTask::Kind::Touch, prefix, {}, {}, {}, {}});
                }
            }
            for (const std::string& other_id : superseded) {
                remove_entry_locked(other_id, false);
                ++stats_.superseded;
            }
            if (options_.remote && !superseded.empty()) {
                // Until this image's upload deletes their manifests from the bucket, a refresh
                // must not register them again.
                std::scoped_lock remote_lock(remote_mutex_);
                for (const std::string& other_id : superseded) {
                    remote_failed_[other_id] = {std::numeric_limits<std::int64_t>::max(), now};
                }
            }
        }
        ++stats_.puts;
        stats_.bytes_written += result.chunk_bytes_written;
        stats_.bytes_stored += stored_written;
        stats_.bytes_reused += result.chunk_bytes_reused;
        const std::uint64_t evictions_before = stats_.evicted_for_space + stats_.expired;
        maintain_locked(&description.id);
        result.evicted_images = stats_.evicted_for_space + stats_.expired - evictions_before;
        if (options_.remote) {
            // The new entry is protected from the maintenance above, so it is still there.
            enqueue_upload(description.id, entries_.at(description.id).generation,
                           std::vector<std::uint8_t>(manifest.out.begin(), manifest.out.end()),
                           std::move(upload_chunks), std::move(superseded));
        }
    } catch (...) {
        for (const fs::path& path : written) { remove_quietly(path); }
        std::scoped_lock lock(mutex_);
        ++stats_.put_failures;
        throw;
    }
    return result;
}

std::optional<std::vector<std::uint8_t>> ContextStore::load(
    const std::string& id, bool touch, LoadReport* report,
    std::chrono::steady_clock::time_point deadline) {
    if (!valid_id(id)) { return std::nullopt; }
    // Downloads only, timed apart from reading and reassembling the local files.
    const auto fetch = [&](std::uint64_t& generation) {
        const auto started  = std::chrono::steady_clock::now();
        std::uint64_t bytes = 0;
        const FetchStatus status = fetch_image_chunks(id, generation, &bytes, deadline);
        if (report != nullptr) {
            report->expired = report->expired || status == FetchStatus::Expired;
            report->fetched_bytes += bytes;
            report->fetch_ns += static_cast<std::uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now() - started)
                    .count());
        }
        return status;
    };
    if (options_.remote) {
        bool needs_fetch = false;
        {
            std::scoped_lock lock(mutex_);
            const auto found = entries_.find(id);
            needs_fetch      = found != entries_.end() && !entry_local_locked(found->second);
        }
        if (needs_fetch) {
            // Network time: no lock is held, so puts and lookups carry on meanwhile.
            std::uint64_t generation = 0;
            const FetchStatus status = fetch(generation);
            if (status != FetchStatus::Complete) {
                {
                    std::scoped_lock lock(mutex_);
                    ++stats_.loads;
                    ++stats_.load_misses;
                }
                if (status == FetchStatus::Corrupt) { settle_failed_fetch(id, generation); }
                return std::nullopt;
            }
        }
    }
    // One pass reads the local files. With a remote, a local chunk found damaged is discarded and
    // fetched again, then the pass is repeated once.
    for (int pass = 0; pass < 2; ++pass) {
        std::vector<ChunkRef> damaged;
        {
            std::scoped_lock io(io_mutex_);
            Entry entry;
            std::vector<Segment> segments;
            {
                std::scoped_lock lock(mutex_);
                if (pass == 0) { ++stats_.loads; }
                if (entries_.find(id) == entries_.end()) {
                    if (pass == 0) { ++stats_.load_misses; }
                    return std::nullopt;
                }
            }
            // `bad_chunks` nullptr: the manifest or its layout is at fault.
            const auto fail = [&](const std::vector<ChunkRef>* bad_chunks =
                                      nullptr) -> std::optional<std::vector<std::uint8_t>> {
                std::scoped_lock lock(mutex_);
                const auto found = entries_.find(id);
                if (found != entries_.end()) {
                    quarantine_peers_locked(found->second, bad_chunks);
                    ++(bad_chunks == nullptr ? stats_.corrupt_manifest : stats_.corrupt_chunk);
                }
                remove_entry_locked(id, true);
                ++stats_.load_misses;
                return std::nullopt;
            };
            if (!read_manifest(manifest_path(id), entry, &segments)) { return fail(); }
            const auto image_bytes = static_cast<std::size_t>(entry.info.image_bytes);
            std::vector<std::uint8_t> image(image_bytes);
            std::uint8_t* const buffer = image.data();
            struct Piece {
                const ChunkRef* chunk;
                std::size_t position;
            };
            std::vector<Piece> pieces;
            std::size_t position = 0;
            for (const Segment& segment : segments) {
                if (!segment.chunked) {
                    if (segment.inline_bytes.size() > image_bytes - position) { return fail(); }
                    std::memcpy(buffer + position, segment.inline_bytes.data(),
                                segment.inline_bytes.size());
                    position += segment.inline_bytes.size();
                    continue;
                }
                for (const ChunkRef& chunk : segment.chunks) {
                    if (chunk.length > image_bytes - position) { return fail(); }
                    pieces.push_back({&chunk, position});
                    position += chunk.length;
                }
            }
            if (position != image_bytes) { return fail(); }
            // Reading and verifying a chunk is independent of every other chunk; one thread is
            // bound by hashing and page faults at well under the drive's sequential rate, so the
            // chunks are read by several.
            std::atomic<std::size_t> next{0};
            std::mutex damage_mutex;
            std::exception_ptr reader_failure;
            const auto read_pieces = [&] {
                try {
                    for (std::size_t index = next++; index < pieces.size(); index = next++) {
                        const Piece& piece = pieces[index];
                        // A compressed chunk is decoded as a stream straight into the image.
                        const std::span<std::uint8_t> target(buffer + piece.position,
                                                             piece.chunk->length);
                        if (!read_chunk(*piece.chunk, target) ||
                            hash(target) != piece.chunk->hash) {
                            std::scoped_lock damage(damage_mutex);
                            damaged.push_back(*piece.chunk);
                        }
                    }
                } catch (...) {
                    // An exception leaving a thread body would terminate the process: keep the
                    // first, stop the other readers, and rethrow from load() once all are joined.
                    std::scoped_lock damage(damage_mutex);
                    if (!reader_failure) { reader_failure = std::current_exception(); }
                    next = pieces.size();
                }
            };
            const std::size_t threads =
                std::min<std::size_t>({pieces.size(), kLoadThreads,
                                       std::max(1U, std::thread::hardware_concurrency())});
            std::vector<std::thread> readers;
            try {
                for (std::size_t t = 1; t < threads; ++t) { readers.emplace_back(read_pieces); }
            } catch (...) {} // fewer threads: the calling thread still reads every piece
            read_pieces();
            for (auto& reader : readers) { reader.join(); }
            if (reader_failure) { std::rethrow_exception(reader_failure); }
            // Every damaged chunk is discarded and fetched again together (with a remote), so one
            // bad chunk does not leave the next to fail the retry.
            const bool intact = damaged.empty();
            if (!intact && !(options_.remote && pass == 0)) { return fail(&damaged); }
            if (intact) {
                if (touch) {
                    std::scoped_lock lock(mutex_);
                    const auto found = entries_.find(id);
                    if (found != entries_.end()) {
                        found->second.info.last_used_ms = now_ms();
                        write_used(id, found->second.info.last_used_ms);
                    }
                }
                // Only a use restarts the age of the bucket's copy: a probe or a restore that is
                // abandoned (touch false) must not keep a session alive there.
                if (touch) { enqueue_remote({RemoteTask::Kind::Touch, id, {}, {}, {}, {}}); }
                return image;
            }
            // Discard the damaged local copies; the remote holds the good ones. A peer that linked
            // one of them is not linked from again, unchanged, while the fetch runs.
            std::scoped_lock lock(mutex_);
            const auto held = entries_.find(id);
            if (held != entries_.end()) { quarantine_peers_locked(held->second, &damaged); }
            for (const ChunkRef& chunk : damaged) {
                const auto found = chunks_.find(hex_key(chunk.hash));
                if (found != chunks_.end() && found->second.present) {
                    found->second.present = false;
                    chunk_total_bytes_ -= found->second.stored;
                    found->second.stored = 0;
                }
                remove_chunk_files(chunk.hash);
            }
        }
        std::uint64_t generation = 0;
        const FetchStatus status = fetch(generation);
        if (status != FetchStatus::Complete) {
            {
                std::scoped_lock lock(mutex_);
                ++stats_.load_misses;
            }
            // Gone or damaged in the bucket too: nothing can repair it, so do not offer it again.
            if (status == FetchStatus::Corrupt) { settle_failed_fetch(id, generation); }
            return std::nullopt;
        }
    }
    return std::nullopt;
}

std::vector<ContextStore::Info> ContextStore::list() const {
    std::scoped_lock lock(mutex_);
    std::vector<Info> out;
    out.reserve(entries_.size());
    for (const auto& [id, entry] : entries_) {
        out.push_back(entry.info);
        out.back().fetch_bytes = fetch_bytes_locked(entry);
        out.back().local       = out.back().fetch_bytes == 0 && entry_local_locked(entry);
    }
    std::sort(out.begin(), out.end(), [](const Info& left, const Info& right) {
        return left.last_used_ms > right.last_used_ms;
    });
    return out;
}

std::optional<ContextStore::Info> ContextStore::find(const std::string& id) const {
    std::scoped_lock lock(mutex_);
    const auto found = entries_.find(id);
    if (found == entries_.end()) { return std::nullopt; }
    Info info        = found->second.info;
    info.fetch_bytes = fetch_bytes_locked(found->second);
    info.local       = info.fetch_bytes == 0 && entry_local_locked(found->second);
    return info;
}

void ContextStore::for_each_checkpoint(const CheckpointMatch& match,
                                       const CheckpointVisit& visit) const {
    std::scoped_lock lock(mutex_);
    for (const auto& [id, entry] : entries_) {
        std::optional<Info> info; // as list() reports it, once a key matches
        for (const CheckpointKey& key : entry.info.checkpoints) {
            if (!match(entry.info, key)) { continue; }
            if (!info) {
                info.emplace(entry.info);
                info->fetch_bytes = fetch_bytes_locked(entry);
                info->local       = info->fetch_bytes == 0 && entry_local_locked(entry);
            }
            visit(*info, key);
        }
    }
}

std::uint64_t ContextStore::fetch_bytes_locked(const Entry& entry) const {
    std::uint64_t bytes = 0;
    std::unordered_set<std::string> seen;
    for (const ChunkRef& chunk : entry.chunks) {
        std::string key  = hex_key(chunk.hash);
        const auto found = chunks_.find(key);
        if ((found == chunks_.end() || !found->second.present) && seen.insert(std::move(key)).second) {
            bytes += chunk.length;
        }
    }
    return bytes;
}

bool ContextStore::erase(const std::string& id) {
    if (!valid_id(id)) { return false; }
    std::scoped_lock io(io_mutex_);
    std::scoped_lock lock(mutex_);
    if (entries_.find(id) == entries_.end()) { return false; }
    remove_entry_locked(id, false);
    return true;
}

void ContextStore::touch(const std::string& id) {
    std::scoped_lock io(io_mutex_);
    std::scoped_lock lock(mutex_);
    const auto found = entries_.find(id);
    if (found == entries_.end()) { return; }
    found->second.info.last_used_ms = now_ms();
    write_used(id, found->second.info.last_used_ms);
    enqueue_remote({RemoteTask::Kind::Touch, id, {}, {}, {}, {}});
}

ContextStore::Stats ContextStore::stats() const {
    std::scoped_lock lock(mutex_);
    Stats out        = stats_;
    out.used_bytes   = used_bytes_locked();
    out.images       = entries_.size();
    if (options_.remote) {
        for (const auto& [id, entry] : entries_) {
            if (!entry_local_locked(entry)) { ++out.remote_images; }
        }
    }
    return out;
}

std::string ContextStore::remote_manifest_key(const std::string& id) const {
    return options_.remote_prefix + "manifests/" + id + ".manifest";
}

std::string ContextStore::remote_chunk_key(const std::array<std::uint64_t, 2>& hash,
                                           bool compressed) const {
    const std::string key = hex_key(hash);
    return options_.remote_prefix + "chunks/" + key.substr(0, 2) + "/" + key +
           (compressed ? ".chunk.zst" : ".chunk");
}

std::optional<std::pair<std::string, bool>>
ContextStore::find_remote_chunk(const std::array<std::uint64_t, 2>& hash) {
    for (const bool compressed : {options_.compress_chunks, !options_.compress_chunks}) {
        std::string key = remote_chunk_key(hash, compressed);
        if (options_.remote->exists(key)) { return std::make_pair(std::move(key), compressed); }
    }
    return std::nullopt;
}

void ContextStore::enqueue_remote(RemoteTask task) {
    if (!options_.remote) { return; }
    {
        std::scoped_lock lock(remote_mutex_);
        if (remote_stopping_) { return; }
        std::size_t background = 0;
        for (const RemoteTask& queued : remote_queue_) {
            if (queued.kind == task.kind && queued.id == task.id) { return; }
            if (queued.kind != RemoteTask::Kind::Upload) { ++background; }
        }
        // Touches, prefetches and refreshes are best-effort and asked for again by the next use:
        // while a stalled bucket holds the worker they must not pile up without bound.
        constexpr std::size_t kMaximumQueuedBackgroundTasks = 64;
        if (task.kind != RemoteTask::Kind::Upload && background >= kMaximumQueuedBackgroundTasks) {
            return;
        }
        remote_queue_.push_back(std::move(task));
    }
    remote_cv_.notify_one();
}

// Called with io_mutex_ and mutex_ held, the chunks already pinned. One queued upload per image: a
// newer write of the image replaces it. Each queued upload keeps its chunk files on disk, so the
// queue is bounded and the oldest is given up beyond it (its image is uploaded again by its next write).
void ContextStore::enqueue_upload(const std::string& id, std::uint64_t generation,
                                  std::vector<std::uint8_t> manifest, std::vector<ChunkRef> chunks,
                                  std::vector<std::string> superseded) {
    constexpr std::size_t kMaximumQueuedUploads = 16;
    std::vector<ChunkRef> unused;
    bool queued = false;
    {
        std::scoped_lock lock(remote_mutex_);
        if (remote_stopping_) {
            unused = std::move(chunks);
            release_retirements_locked(superseded);
        } else {
            auto ready = std::chrono::steady_clock::now() + options_.remote_upload_delay;
            // A queued upload of an image this write superseded is not needed: this one has
            // everything it has. Its wait and the manifests it was to delete pass to this one.
            std::vector<std::string> retire;
            for (auto it = remote_queue_.begin(); it != remote_queue_.end();) {
                const bool replaced = it->kind == RemoteTask::Kind::Upload &&
                                      std::find(superseded.begin(), superseded.end(), it->id) !=
                                          superseded.end();
                if (!replaced) {
                    ++it;
                    continue;
                }
                ready = std::min(ready, it->ready);
                retire.insert(retire.end(), it->retire.begin(), it->retire.end());
                unused.insert(unused.end(), it->chunks.begin(), it->chunks.end());
                ++stats_.remote_uploads_skipped; // stats_ is guarded by mutex_, which put() holds
                it = remote_queue_.erase(it);
            }
            retire.insert(retire.end(), superseded.begin(), superseded.end());
            std::size_t uploads = 0;
            for (RemoteTask& task : remote_queue_) {
                if (task.kind != RemoteTask::Kind::Upload) { continue; }
                ++uploads;
                if (task.id == id) {
                    unused.insert(unused.end(), task.chunks.begin(), task.chunks.end());
                    task.manifest   = std::move(manifest);
                    task.chunks     = std::move(chunks);
                    task.retire.insert(task.retire.end(), retire.begin(), retire.end());
                    task.ready      = std::min(task.ready, ready);
                    task.generation = generation;
                    queued          = true;
                    break;
                }
            }
            if (!queued) {
                if (uploads >= kMaximumQueuedUploads) {
                    for (auto it = remote_queue_.begin(); it != remote_queue_.end(); ++it) {
                        if (it->kind == RemoteTask::Kind::Upload) {
                            unused.insert(unused.end(), it->chunks.begin(), it->chunks.end());
                            release_retirements_locked(it->retire);
                            remote_queue_.erase(it);
                            break;
                        }
                    }
                }
                remote_queue_.push_back(RemoteTask{RemoteTask::Kind::Upload, id, std::move(manifest),
                                                   std::move(chunks), std::move(retire), ready,
                                                   generation});
                queued = true;
            }
        }
    }
    unpin_chunks_locked(unused);
    if (queued) { remote_cv_.notify_one(); }
}

void ContextStore::release_retirements_locked(const std::vector<std::string>& ids) {
    for (const std::string& id : ids) {
        const auto found = remote_failed_.find(id);
        // Only the block put() placed (no bucket time can pass it), not a failure of its own.
        if (found != remote_failed_.end() &&
            found->second.remote_modified_ms == std::numeric_limits<std::int64_t>::max()) {
            remote_failed_.erase(found);
        }
    }
}

void ContextStore::remote_loop() {
    auto next_refresh = std::chrono::steady_clock::now() + options_.remote_refresh;
    std::unique_lock lock(remote_mutex_);
    // The first task that may run now: an upload waits out its delay unless a flush is waiting.
    const auto runnable = [&] {
        const auto now = std::chrono::steady_clock::now();
        return std::find_if(remote_queue_.begin(), remote_queue_.end(), [&](const RemoteTask& task) {
            return task.kind != RemoteTask::Kind::Upload || remote_flushes_ != 0 || task.ready <= now;
        });
    };
    for (;;) {
        // Woken by a new task, a flush, or the earliest delayed upload coming due.
        while (!remote_stopping_ && runnable() == remote_queue_.end() &&
               std::chrono::steady_clock::now() < next_refresh) {
            auto wake = next_refresh;
            for (const RemoteTask& queued : remote_queue_) { wake = std::min(wake, queued.ready); }
            remote_cv_.wait_until(lock, wake);
        }
        if (remote_stopping_) { break; }
        RemoteTask task{RemoteTask::Kind::Refresh, std::string(), {}, {}, {}, {}};
        if (const auto next = runnable(); next != remote_queue_.end()) {
            task = std::move(*next);
            remote_queue_.erase(next);
        } else if (std::chrono::steady_clock::now() >= next_refresh) {
            next_refresh = std::chrono::steady_clock::now() + options_.remote_refresh;
        } else {
            continue;
        }
        remote_busy_ = true;
        lock.unlock();
        try {
            switch (task.kind) {
            case RemoteTask::Kind::Upload: upload_image(task); break;
            case RemoteTask::Kind::Refresh: (void)refresh_remote(); break;
            case RemoteTask::Kind::Touch: touch_remote_image(task.id); break;
            case RemoteTask::Kind::Prefetch: {
                std::uint64_t generation = 0;
                const FetchStatus status = fetch_image_chunks(task.id, generation);
                if (status == FetchStatus::Complete) {
                    std::scoped_lock io(io_mutex_);
                    std::scoped_lock state(mutex_);
                    maintain_locked(&task.id);
                } else if (status == FetchStatus::Corrupt) {
                    // Damaged or gone in the bucket: do not offer it again.
                    settle_failed_fetch(task.id, generation);
                }
                break;
            }
            }
        } catch (...) {}
        prune_remote_state();
        lock.lock();
        remote_busy_ = false;
        if (remote_queue_.empty()) { remote_idle_cv_.notify_all(); }
    }
    // Uploads that never ran release the chunk files they were holding.
    std::vector<ChunkRef> pinned;
    for (RemoteTask& task : remote_queue_) {
        if (task.kind == RemoteTask::Kind::Upload) {
            pinned.insert(pinned.end(), task.chunks.begin(), task.chunks.end());
            release_retirements_locked(task.retire);
        }
    }
    remote_queue_.clear();
    lock.unlock();
    std::scoped_lock io(io_mutex_);
    std::scoped_lock state(mutex_);
    unpin_chunks_locked(pinned);
}

bool ContextStore::flush_remote(std::chrono::steady_clock::time_point deadline, bool final_flush) {
    if (!options_.remote) { return true; }
    std::unique_lock lock(remote_mutex_);
    // At the end of the run, queued refreshes, prefetches and touches are best-effort: drop them
    // rather than let them use the budget the uploads need.
    if (final_flush) {
        std::erase_if(remote_queue_,
                      [](const RemoteTask& task) { return task.kind != RemoteTask::Kind::Upload; });
    }
    // Delayed uploads run now.
    ++remote_flushes_;
    remote_cv_.notify_all();
    const bool drained = remote_idle_cv_.wait_until(
        lock, deadline, [&] { return remote_queue_.empty() && !remote_busy_; });
    --remote_flushes_;
    return drained;
}

bool ContextStore::prefetch(const std::string& id) {
    if (!options_.remote || !valid_id(id)) { return false; }
    {
        std::scoped_lock lock(mutex_);
        const auto found = entries_.find(id);
        if (found == entries_.end() || entry_local_locked(found->second)) { return false; }
    }
    enqueue_remote({RemoteTask::Kind::Prefetch, id, {}, {}, {}, {}});
    return true;
}

void ContextStore::upload_image(RemoteTask& task) {
    std::uint64_t objects = 0;
    std::uint64_t bytes   = 0;
    bool succeeded        = false;
    try {
        const std::int64_t now   = now_ms();
        const std::int64_t trust = options_.remote_touch_interval.count() * 1000;
        std::unordered_set<std::string> seen;
        for (const ChunkRef& chunk : task.chunks) {
            // Confirmations are tracked by content hash: the bucket may hold either form.
            const std::string key = hex_key(chunk.hash);
            if (!seen.insert(key).second) { continue; }
            {
                // A bucket's lifecycle rule can expire an object, so a confirmation is trusted for a
                // while, not for good.
                std::scoped_lock lock(remote_mutex_);
                const auto known = uploaded_.find(key);
                if (known != uploaded_.end() && now - known->second < trust) { continue; }
            }
            if (!find_remote_chunk(chunk.hash)) {
                // The file is pinned, so it is there; if it is not intact the image is not published.
                // A compressed chunk goes up as it is on disk.
                bool compressed = false;
                const std::vector<std::uint8_t> data = verified_chunk_file(chunk, compressed);
                options_.remote->put(remote_chunk_key(chunk.hash, compressed), data);
                ++objects;
                bytes += data.size();
            }
            std::scoped_lock lock(remote_mutex_);
            uploaded_[key] = now;
        }
        // The manifest goes last: an image another engine sees always has its chunks.
        options_.remote->put(remote_manifest_key(task.id), task.manifest);
        ++objects;
        bytes += task.manifest.size();
        succeeded = true;
    } catch (const std::exception&) {}
    if (succeeded) {
        {
            // Only now may the directory give this image's chunks up to the bucket.
            std::scoped_lock io(io_mutex_);
            std::scoped_lock lock(mutex_);
            const auto found = entries_.find(task.id);
            if (found != entries_.end() && found->second.generation == task.generation) {
                found->second.published    = true;
                found->second.published_at = std::chrono::steady_clock::now();
                write_published_locked(task.id, task.manifest);
            }
        }
        // The images this one superseded leave the bucket, or every engine keeps registering
        // them and a new one downloads every turn of every conversation at start-up. Their
        // chunks are mostly this image's too; the rest expire under the bucket's lifecycle rule.
        // Deletions that failed before are retried here.
        std::vector<std::string> retire = std::move(task.retire);
        {
            std::scoped_lock lock(remote_mutex_);
            retire.insert(retire.end(), retire_backlog_.begin(), retire_backlog_.end());
            retire_backlog_.clear();
        }
        for (std::size_t index = 0; index < retire.size(); ++index) {
            const std::string& retired = retire[index];
            {
                std::scoped_lock lock(mutex_);
                // Written here again since: that write is the bucket's copy now.
                if (entries_.find(retired) != entries_.end()) { continue; }
            }
            try {
                options_.remote->remove(remote_manifest_key(retired));
            } catch (const std::exception&) {
                // The bucket is failing: keep the rest (bounded) for the next upload.
                constexpr std::size_t kMaximumRetireBacklog = 4096;
                std::scoped_lock lock(remote_mutex_);
                for (std::size_t rest = index;
                     rest < retire.size() && retire_backlog_.size() < kMaximumRetireBacklog; ++rest) {
                    retire_backlog_.push_back(retire[rest]);
                }
                break;
            }
            std::scoped_lock lock(mutex_);
            ++stats_.remote_retired;
            std::scoped_lock remote_lock(remote_mutex_);
            remote_failed_.erase(retired);
        }
    } else {
        // Not published: the images it superseded are still the bucket's latest copies.
        std::scoped_lock lock(remote_mutex_);
        release_retirements_locked(task.retire);
    }
    std::scoped_lock io(io_mutex_);
    std::scoped_lock lock(mutex_);
    unpin_chunks_locked(task.chunks);
    task.chunks.clear();
    stats_.remote_uploads += objects;
    stats_.remote_upload_bytes += bytes;
    if (!succeeded) { ++stats_.remote_upload_failures; }
}

void ContextStore::prune_remote_state() {
    const std::int64_t now = now_ms();
    if (now - last_prune_ms_ < 60'000) { return; }
    last_prune_ms_            = now;
    const std::int64_t horizon = options_.remote_touch_interval.count() * 1000;
    std::scoped_lock lock(remote_mutex_);
    for (auto it = uploaded_.begin(); it != uploaded_.end();) {
        it = now - it->second >= horizon ? uploaded_.erase(it) : std::next(it);
    }
    for (auto it = remote_touched_.begin(); it != remote_touched_.end();) {
        it = now - it->second >= horizon ? remote_touched_.erase(it) : std::next(it);
    }
    for (auto it = remote_failed_.begin(); it != remote_failed_.end();) {
        it = now - it->second.failed_ms >= horizon ? remote_failed_.erase(it) : std::next(it);
    }
}

void ContextStore::touch_remote_image(const std::string& id) {
    std::vector<ChunkRef> chunks;
    std::uint64_t generation = 0;
    // An image imported whole from a peer is the peer's to upload: its age in the bucket is
    // restarted, but what the bucket lacks is not put there from here.
    bool peer_image = false;
    {
        std::scoped_lock lock(mutex_);
        const auto found = entries_.find(id);
        if (found == entries_.end()) { return; }
        chunks     = found->second.chunks;
        generation = found->second.generation;
        peer_image = !found->second.peer_memo_key.empty();
    }
    const std::int64_t now = now_ms();
    {
        std::scoped_lock lock(remote_mutex_);
        const auto last = remote_touched_.find(id);
        if (last != remote_touched_.end() &&
            now - last->second < options_.remote_touch_interval.count() * 1000) {
            return;
        }
    }
    // Touched only counts once every object is confirmed or put back; otherwise the next request
    // tries again instead of trusting a repair that did not happen.
    bool complete = true;
    try {
        for (const ChunkRef& chunk : chunks) {
            const std::string key = hex_key(chunk.hash); // confirmations are tracked by content hash
            // The bucket may hold either form of the chunk; the form this store writes is tried first.
            if (options_.remote->touch(remote_chunk_key(chunk.hash, options_.compress_chunks)) ||
                options_.remote->touch(remote_chunk_key(chunk.hash, !options_.compress_chunks))) {
                std::scoped_lock lock(remote_mutex_);
                uploaded_[key] = now;
                continue;
            }
            {
                // The bucket lost it (expired under a lifecycle rule): what was confirmed is stale.
                std::scoped_lock lock(remote_mutex_);
                uploaded_.erase(key);
            }
            if (peer_image) {
                complete = false;
                continue;
            }
            // Put it back from the directory while the image is still in use, if the directory
            // has an intact copy.
            std::vector<std::uint8_t> data;
            bool compressed = false;
            try {
                data = verified_chunk_file(chunk, compressed);
            } catch (const std::exception&) {
                complete = false;
                continue;
            }
            options_.remote->put(remote_chunk_key(chunk.hash, compressed), data);
            std::scoped_lock lock(mutex_);
            ++stats_.remote_uploads;
            stats_.remote_upload_bytes += data.size();
            std::scoped_lock remote_lock(remote_mutex_);
            uploaded_[key] = now;
        }
        if (!options_.remote->touch(remote_manifest_key(id))) {
            // A manifest goes up only after every chunk it names is confirmed or put back
            // (manifest-last): with a chunk still missing, publishing it would offer other engines
            // an image they cannot read. The next use retries.
            if (!complete || peer_image) { return; }
            // Republish only the manifest of the image the chunks above were taken from: if the id
            // has been rewritten since, the new image has its own upload and this one is stale.
            std::vector<std::uint8_t> manifest;
            bool replaced = false;
            bool read     = false;
            {
                std::scoped_lock io(io_mutex_);
                std::scoped_lock lock(mutex_);
                const auto current = entries_.find(id);
                replaced = current == entries_.end() || current->second.generation != generation;
                if (!replaced) {
                    try {
                        manifest = read_whole_file(manifest_path(id));
                        read     = true;
                    } catch (const std::exception&) {}
                }
            }
            if (replaced) { return; }
            if (read) {
                options_.remote->put(remote_manifest_key(id), manifest);
            } else {
                complete = false;
            }
        }
        if (complete) {
            {
                // Every chunk and the manifest are confirmed in the bucket; recorded on disk too, so
                // a restart still knows (the manifest touched or put back is the one on disk).
                std::scoped_lock io(io_mutex_);
                std::scoped_lock lock(mutex_);
                const auto found = entries_.find(id);
                if (found != entries_.end() && found->second.generation == generation) {
                    found->second.published    = true;
                    found->second.published_at = std::chrono::steady_clock::now();
                    try {
                        write_published_locked(id, read_whole_file(manifest_path(id)));
                    } catch (const std::exception&) {}
                }
            }
            std::scoped_lock lock(remote_mutex_);
            remote_touched_[id] = now;
        }
    } catch (const std::exception&) {}
}

ContextStore::FetchStatus ContextStore::fetch_image_chunks(
    const std::string& id, std::uint64_t& generation, std::uint64_t* fetched,
    std::chrono::steady_clock::time_point deadline) {
    std::vector<ChunkRef> chunks;
    generation = 0;
    std::atomic<std::uint64_t> fetched_bytes{0};
    // However this returns, report what was downloaded.
    struct ReportFetched {
        std::atomic<std::uint64_t>& bytes;
        std::uint64_t* out;
        ~ReportFetched() {
            if (out != nullptr) { *out += bytes.load(); }
        }
    } report_fetched{fetched_bytes, fetched};
    {
        std::scoped_lock lock(mutex_);
        const auto found = entries_.find(id);
        if (found == entries_.end()) { return FetchStatus::Corrupt; }
        // Each missing chunk once.
        std::unordered_set<std::string> seen;
        for (const ChunkRef& chunk : found->second.chunks) {
            std::string key   = hex_key(chunk.hash);
            const auto held   = chunks_.find(key);
            if ((held == chunks_.end() || !held->second.present) && seen.insert(std::move(key)).second) {
                chunks.push_back(chunk);
            }
        }
        generation = found->second.generation;
    }
    // A request may be waiting on this: the chunks are fetched several at a time, so a session of
    // a few hundred objects costs a few round trips, not a few hundred. The first failure stops the
    // others; a damaged chunk outranks a transient error.
    std::atomic<std::size_t> next{0};
    std::atomic<bool> transient{false};
    std::atomic<bool> corrupt{false};
    // A request waits on this until its deadline only: past it, no chunk is started (one in flight
    // ends within the transfer's own timeouts), so the reader is free for the next request.
    std::atomic<bool> expired{false};
    const auto fetch_chunk_range = [&] {
        for (std::size_t index = next++; index < chunks.size() && !transient && !corrupt && !expired;
             index = next++) {
            if (std::chrono::steady_clock::now() >= deadline) {
                expired = true;
                return;
            }
            const ChunkRef& chunk = chunks[index];
            std::optional<std::vector<std::uint8_t>> data;
            bool compressed = false;
            try {
                // Whichever form an engine uploaded; the form this store writes is tried first.
                for (const bool form : {options_.compress_chunks, !options_.compress_chunks}) {
                    data = options_.remote->get(remote_chunk_key(chunk.hash, form));
                    if (data) {
                        compressed = form;
                        break;
                    }
                }
            } catch (const std::exception&) {
                std::scoped_lock lock(mutex_);
                ++stats_.remote_download_failures;
                ++stats_.remote_download_failures_transfer;
                transient = true;
                return;
            }
            bool intact = false;
            if (data) {
                if (compressed) {
                    std::vector<std::uint8_t> raw(chunk.length);
                    intact = decompress_memory(*data, raw) && hash(raw) == chunk.hash;
                } else {
                    intact = data->size() == chunk.length && hash(*data) == chunk.hash;
                }
            }
            if (!intact) {
                std::scoped_lock lock(mutex_);
                ++stats_.remote_download_failures;
                ++(data ? stats_.remote_download_failures_damaged : stats_.remote_download_failures_missing);
                corrupt = true;
                return;
            }
            if (!install_chunk(chunk, compressed, *data)) {
                corrupt = true;
                return;
            }
            std::scoped_lock lock(mutex_);
            ++stats_.remote_downloads;
            stats_.remote_download_bytes += data->size(); // what crossed the network
            // Raw bytes, the unit Info::fetch_bytes prices a fetch in, so the rate learned from
            // this report applies to it whether or not the chunk travelled compressed.
            fetched_bytes += chunk.length;
        }
    };
    const auto fetch_chunks = [&] {
        // Runs on its own thread: nothing may escape it (that would terminate the process). A
        // failure here that is not the bucket's (a full disk installing the chunk, an allocation)
        // makes the fetch fail like a transfer error, and a later load retries.
        try {
            fetch_chunk_range();
        } catch (...) {
            transient = true;
            std::scoped_lock lock(mutex_);
            ++stats_.remote_download_failures;
            ++stats_.remote_download_failures_install;
        }
    };
    const std::size_t threads = std::min(chunks.size(), kFetchThreads);
    std::vector<std::thread> fetchers;
    try {
        for (std::size_t t = 1; t < threads; ++t) { fetchers.emplace_back(fetch_chunks); }
    } catch (...) {} // fewer threads: the calling thread still fetches every chunk
    fetch_chunks();
    for (auto& fetcher : fetchers) { fetcher.join(); }
    if (corrupt) { return FetchStatus::Corrupt; }
    if (transient) { return FetchStatus::Transient; }
    if (expired) { return FetchStatus::Expired; }
    std::scoped_lock lock(mutex_);
    const auto found = entries_.find(id);
    // Whole again: it is evicted like any image the directory holds.
    if (found != entries_.end() && found->second.generation == generation) {
        found->second.evicted = false;
    }
    return FetchStatus::Complete;
}

bool ContextStore::install_chunk(const ChunkRef& chunk, bool compressed,
                                 std::span<const std::uint8_t> file_bytes) {
    std::scoped_lock io(io_mutex_);
    std::scoped_lock lock(mutex_);
    const auto found = chunks_.find(hex_key(chunk.hash));
    if (found == chunks_.end() || found->second.references == 0) { return false; } // image removed
    if (found->second.present) { return true; }
    install_chunk_file(chunk, compressed, file_bytes);
    found->second.present = true;
    found->second.stored  = file_bytes.size();
    chunk_total_bytes_ += found->second.stored;
    return true;
}

void ContextStore::settle_failed_fetch(const std::string& id, std::uint64_t generation) {
    std::scoped_lock io(io_mutex_);
    std::scoped_lock lock(mutex_);
    const auto found = entries_.find(id);
    // Replaced while the fetch ran: the failure belongs to an image that is already gone.
    if (found == entries_.end() || found->second.generation != generation) { return; }
    // Completed meanwhile from a peer's copy: the bucket's failure no longer matters here.
    if (entry_local_locked(found->second)) { return; }
    // An image written or loaded here has no bucket time: it is not retried until the entry expires.
    const std::int64_t remote_modified = found->second.remote_modified_ms != 0
                                             ? found->second.remote_modified_ms
                                             : std::numeric_limits<std::int64_t>::max();
    ++stats_.corrupt_remote;
    remove_entry_locked(id, true);
    std::scoped_lock remote_lock(remote_mutex_);
    remote_failed_[id] = {remote_modified, now_ms()};
}

std::size_t ContextStore::refresh_remote(std::chrono::steady_clock::time_point deadline) {
    if (!options_.remote) { return 0; }
    // The wait for another refresh counts against the deadline too.
    std::unique_lock refreshing(refresh_mutex_, std::defer_lock);
    if (!refreshing.try_lock_until(deadline)) { return 0; }
    return refresh_remote_locked(deadline);
}

std::shared_ptr<const ContextStore::Listing>
ContextStore::listing_since(std::chrono::steady_clock::time_point asked,
                            std::chrono::steady_clock::time_point deadline) {
    std::unique_lock lock(listing_mutex_);
    for (;;) {
        // A listing asked for at or after `asked` shows everything this caller needs to see; one
        // asked for before it may miss what was uploaded since, however recently it finished.
        if (latest_listing_ && latest_listing_->listed_at >= asked) { return latest_listing_; }
        if (!listing_started_) { break; }
        // One listing runs at a time. One in progress that started after `asked` serves this
        // caller; one that started before it may miss what was uploaded just before this caller
        // asked, so the caller waits for it to end and then shares the next one: the first to
        // wake takes it, and the others find a listing started after they asked and wait for it.
        // However many callers arrive during a listing, one more listing follows it.
        if (listing_cv_.wait_until(lock, deadline) == std::cv_status::timeout &&
            !(latest_listing_ && latest_listing_->listed_at >= asked)) {
            return nullptr;
        }
    }
    if (std::chrono::steady_clock::now() >= deadline) { return nullptr; }
    const auto started = std::chrono::steady_clock::now();
    listing_started_   = started;
    lock.unlock();
    // However this ends, the waiters are told: they take the listing, or list for themselves.
    struct Finish {
        ContextStore& store;
        std::chrono::steady_clock::time_point started;
        ~Finish() {
            {
                std::scoped_lock lock(store.listing_mutex_);
                if (store.listing_started_ == started) { store.listing_started_.reset(); }
            }
            store.listing_cv_.notify_all();
        }
    } finish{*this, started};
    auto listing       = std::make_shared<Listing>();
    listing->listed_at = started;
    listing->objects   = options_.remote->list(options_.remote_prefix + "manifests/");
    std::scoped_lock published(listing_mutex_);
    latest_listing_ = listing; // one listing at a time: this is the newest
    return listing;
}

std::string ContextStore::listed_id(const std::string& key) const {
    const std::string manifests        = options_.remote_prefix + "manifests/";
    constexpr std::string_view kSuffix = ".manifest";
    if (key.size() <= manifests.size() + kSuffix.size() || key.compare(0, manifests.size(), manifests) != 0 ||
        key.compare(key.size() - kSuffix.size(), kSuffix.size(), kSuffix) != 0) {
        return {};
    }
    std::string id = key.substr(manifests.size(), key.size() - manifests.size() - kSuffix.size());
    return valid_id(id) ? id : std::string();
}

bool ContextStore::register_remote_image(const ObjectInfo& object, const std::string& id,
                                         std::chrono::steady_clock::time_point listed_at) {
    {
        std::scoped_lock lock(mutex_);
        if (entries_.find(id) != entries_.end()) { return false; }
    }
    {
        std::scoped_lock lock(remote_mutex_);
        const auto failed = remote_failed_.find(id);
        if (failed != remote_failed_.end()) {
            // Given up on: skipped until the bucket lists the manifest as written since, which
            // means another engine has replaced the image that failed.
            if (object.modified_ms <= failed->second.remote_modified_ms) { return false; }
            remote_failed_.erase(failed);
        }
    }
    const std::optional<std::vector<std::uint8_t>> data = options_.remote->get(object.key);
    if (!data) {
        // Listed but gone by the time it was read (deleted, or expired, since the listing).
        std::scoped_lock lock(mutex_);
        ++stats_.remote_download_failures;
        ++stats_.remote_download_failures_missing;
        return false;
    }
    {
        std::scoped_lock lock(mutex_);
        ++stats_.remote_manifest_downloads;
        stats_.remote_manifest_download_bytes += data->size();
    }
    Entry entry;
    if (!parse_manifest(*data, entry, nullptr) || entry.info.id != id) {
        {
            std::scoped_lock lock(mutex_);
            ++stats_.remote_download_failures;
            ++stats_.remote_download_failures_damaged;
        }
        std::scoped_lock lock(remote_mutex_);
        remote_failed_[id] = {object.modified_ms, now_ms()};
        return false;
    }
    entry.info.last_used_ms = std::max(entry.info.created_ms, object.modified_ms);
    entry.published         = true; // listed, so the bucket holds its manifest
    // A listing a refresh asked for before this one was taken may not show the image: it does not
    // prove the image gone (a lookup registers images while a refresh runs).
    entry.published_at      = listed_at;
    std::scoped_lock io(io_mutex_);
    std::scoped_lock lock(mutex_);
    // A refresh and a lookup may register the same image at once: the first one wins.
    if (entries_.find(id) != entries_.end()) { return false; }
    write_atomically(manifest_path(id), *data);
    write_published_locked(id, *data);
    manifest_total_bytes_ += entry.manifest_bytes;
    add_references(entry);
    // The listing shows the manifest, not its chunks (a lifecycle rule or a person may have
    // removed one): their bucket presence stays unknown, so a write that shares them checks
    // before it skips the upload.
    entry.generation         = ++next_generation_;
    entry.remote_modified_ms = object.modified_ms;
    entries_.emplace(id, std::move(entry));
    return true;
}

ContextStore::LookupResult
ContextStore::lookup_remote(std::chrono::steady_clock::time_point asked,
                            std::chrono::steady_clock::time_point deadline,
                            const std::function<bool(const std::string& id)>& wanted) {
    LookupResult result;
    if (!options_.remote) { return result; }
    try {
        const std::shared_ptr<const Listing> listing = listing_since(asked, deadline);
        if (!listing) { return result; }
        for (const ObjectInfo& object : listing->objects) {
            const std::string id = listed_id(object.key);
            if (id.empty() || !wanted(id)) { continue; }
            if (std::chrono::steady_clock::now() < deadline &&
                register_remote_image(object, id, listing->listed_at)) {
                ++result.registered;
            }
            std::scoped_lock lock(mutex_);
            if (entries_.find(id) != entries_.end()) { ++result.held; }
        }
    } catch (const std::exception&) {
        std::scoped_lock lock(mutex_);
        ++stats_.remote_download_failures;
        ++stats_.remote_download_failures_listing;
    }
    return result;
}

std::size_t ContextStore::refresh_remote_locked(std::chrono::steady_clock::time_point deadline) {
    std::size_t added = 0;
    bool truncated    = false;
    try {
        // Taken now: a refresh is what lookups fall back on for every image they leave, so it
        // never reuses an older listing (a lookup may reuse this one).
        const std::shared_ptr<const Listing> listing =
            listing_since(std::chrono::steady_clock::now(), deadline);
        if (!listing) { return 0; }
        std::unordered_set<std::string> listed;
        for (const ObjectInfo& object : listing->objects) {
            if (std::chrono::steady_clock::now() >= deadline) {
                truncated = true;
                break;
            }
            // The bucket's lifecycle rule is what expires an image there; this directory's TTL
            // governs only the files it holds.
            const std::string id = listed_id(object.key);
            if (id.empty()) { continue; }
            listed.insert(id);
            if (register_remote_image(object, id, listing->listed_at)) { ++added; }
        }
        // An image the bucket no longer lists (expired, or removed by someone) is not offered
        // from the index any more, unless the directory still holds some of it.
        std::scoped_lock io(io_mutex_);
        std::scoped_lock lock(mutex_);
        std::vector<std::string> gone;
        // A listing cut short by the deadline is incomplete, so it proves nothing is gone.
        if (!truncated) {
            for (auto& [id, entry] : entries_) {
                if (listed.count(id) != 0) { continue; }
                if (!entry_has_local_bytes_locked(entry)) {
                    // Registered (or published) from a listing taken after this one: not gone.
                    if (entry.published_at >= listing->listed_at) { continue; }
                    gone.push_back(id);
                } else if (entry.published && entry.published_at < listing->listed_at) {
                    // Kept for its local bytes, but no longer in the bucket: eviction must remove
                    // it whole, not keep an entry whose chunks would exist nowhere. (A manifest
                    // published after the listing began may be missing from it, so that one
                    // proves nothing.)
                    entry.published = false;
                    remove_quietly(published_path(id));
                }
            }
        }
        for (const std::string& id : gone) { remove_entry_locked(id, false); }
        stats_.remote_removed += gone.size();
        ++stats_.remote_refreshes;
    } catch (const std::exception&) {
        std::scoped_lock lock(mutex_);
        ++stats_.remote_download_failures;
        ++stats_.remote_download_failures_listing;
    }
    return added;
}

void ContextStore::peer_loop() {
    std::unique_lock lock(peer_mutex_);
    while (!peer_stopping_) {
        peer_cv_.wait_for(lock, options_.peer_poll, [&] { return peer_stopping_; });
        if (peer_stopping_) { break; }
        lock.unlock();
        try {
            (void)sync_peers();
        } catch (...) {} // a peer that cannot be read this time is read again next time
        lock.lock();
    }
}

std::size_t ContextStore::sync_peers() {
    std::size_t imported = 0;
    std::unordered_set<std::string> listed;
    for (const fs::path& peer : options_.peers) {
        std::error_code error;
        fs::directory_iterator item(peer / "manifests", error);
        if (error) { continue; } // the peer has not started yet, or has no store
        for (; item != fs::directory_iterator(); item.increment(error)) {
            if (error) { break; }
            const fs::path& path = item->path();
            if (path.extension() != ".manifest") { continue; }
            const std::string id = path.stem().string();
            if (!valid_id(id)) { continue; }
            const std::string memo_key = peer.string() + "|" + id;
            listed.insert(memo_key);
            std::error_code time_error;
            const std::int64_t stamp = static_cast<std::int64_t>(
                fs::last_write_time(path, time_error).time_since_epoch().count());
            if (time_error) {
                // The peer removed it since the listing: count it, so the skip metrics show the loss.
                std::scoped_lock lock(mutex_);
                ++stats_.peer_skipped;
                continue;
            }
            {
                std::scoped_lock lock(mutex_);
                // An image the bucket listing registered by name only is still worth the peer's
                // copy: linking it spares load() the download.
                const auto held = entries_.find(id);
                if (held != entries_.end() && entry_local_locked(held->second)) { continue; }
                const auto skipped = peer_skipped_.find(memo_key);
                if (skipped != peer_skipped_.end() && skipped->second == stamp) { continue; }
            }
            if (import_peer_image(peer, id, stamp)) { ++imported; }
        }
    }
    // A skipped image the peers no longer list is forgotten, so the memo cannot outgrow them.
    std::scoped_lock lock(mutex_);
    for (auto it = peer_skipped_.begin(); it != peer_skipped_.end();) {
        it = listed.count(it->first) != 0 ? std::next(it) : peer_skipped_.erase(it);
    }
    return imported;
}

bool ContextStore::import_peer_image(const fs::path& peer, const std::string& id,
                                     std::int64_t stamp) {
    const std::string memo_key = peer.string() + "|" + id;
    const auto skip            = [&] {
        std::scoped_lock lock(mutex_);
        peer_skipped_[memo_key] = stamp;
        ++stats_.peer_skipped;
        return false;
    };
    std::vector<std::uint8_t> data;
    try {
        data = read_whole_file(peer / "manifests" / (id + ".manifest"));
    } catch (const std::exception&) {
        // The peer removed it since the listing: count it, so the skip metrics show the loss.
        std::scoped_lock lock(mutex_);
        ++stats_.peer_skipped;
        return false;
    }
    Entry entry;
    if (!parse_manifest(data, entry, nullptr) || entry.info.id != id) { return skip(); }
    // The peer's last use of the image, as its `.used` sidecar records it (creation time when it
    // has none): the TTL and the LRU order both go by use, as for the store's own images.
    entry.info.last_used_ms = entry.info.created_ms;
    try {
        const std::vector<std::uint8_t> used = read_whole_file(peer / "manifests" / (id + ".used"));
        entry.info.last_used_ms = std::max<std::int64_t>(
            entry.info.created_ms, std::stoll(std::string(used.begin(), used.end())));
    } catch (const std::exception&) {}
    if (options_.ttl.count() > 0 &&
        entry.info.last_used_ms < now_ms() - options_.ttl.count() * 1000) {
        return skip();
    }

    std::scoped_lock io(io_mutex_);
    std::scoped_lock lock(mutex_);
    // An entry the index already holds (registered from the bucket, its chunks not fetched yet)
    // keeps its own manifest and generation; only the chunk files it lacks are linked, found by
    // their content hash, so the peer's manifest need not be the same write of the image.
    const auto held_entry = entries_.find(id);
    Entry* const held     = held_entry != entries_.end() ? &held_entry->second : nullptr;
    if (held != nullptr && entry_local_locked(*held)) { return false; }
    const std::vector<ChunkRef>& wanted = held != nullptr ? held->chunks : entry.chunks;
    const auto skip_locked = [&] {
        peer_skipped_[memo_key] = stamp;
        ++stats_.peer_skipped;
        return false;
    };

    // The chunks this store does not already hold, each once, in whichever form the peer holds it.
    // The fit check counts the bytes the links will occupy (a compressed chunk takes its file size).
    struct Source {
        const ChunkRef* chunk;
        fs::path path;
        bool compressed;
        std::uint64_t size;
    };
    std::vector<Source> needed;
    std::unordered_set<std::string> seen;
    std::uint64_t new_bytes = held != nullptr ? 0 : entry.manifest_bytes; // a held manifest is counted
    for (const ChunkRef& chunk : wanted) {
        const std::string key = hex_key(chunk.hash);
        if (!seen.insert(key).second) { continue; }
        const auto known = chunks_.find(key);
        if (known != chunks_.end() && known->second.present) { continue; }
        const fs::path raw_source    = peer / "chunks" / key.substr(0, 2) / (key + ".chunk");
        const fs::path packed_source = peer / "chunks" / key.substr(0, 2) / (key + ".chunk.zst");
        std::error_code error;
        std::uint64_t size = fs::file_size(raw_source, error);
        if (!error && size == chunk.length) {
            needed.push_back({&chunk, raw_source, false, size});
        } else {
            error.clear();
            size = fs::file_size(packed_source, error);
            const auto content = error ? std::nullopt : compressed_content_size(packed_source);
            if (error || !content || *content != chunk.length) {
                ++stats_.peer_link_failures;
                return skip_locked(); // the peer evicted it meanwhile, or it is damaged
            }
            needed.push_back({&chunk, packed_source, true, size});
        }
        new_bytes += size;
    }
    // Nothing is imported that would be evicted at once, or the store would link, evict and link
    // the same image again at every poll.
    if (options_.max_bytes != 0 && used_bytes_locked() + new_bytes > options_.max_bytes) {
        return skip_locked();
    }

    // Hard-link each chunk. A link is independent of the peer's file from then on: the peer
    // deleting its copy does not touch ours.
    std::vector<fs::path> created;
    std::vector<std::pair<const ChunkRef*, StoredChunk>> linked;
    std::uint64_t linked_bytes = 0;
    const auto roll_back       = [&] {
        for (const fs::path& path : created) { remove_quietly(path); }
    };
    for (const Source& source : needed) {
        std::error_code error;
        remove_chunk_files(source.chunk->hash); // a stale file of ours under the name would block the link
        const fs::path target = source.compressed ? compressed_chunk_path(source.chunk->hash)
                                                  : chunk_path(source.chunk->hash);
        fs::create_directories(target.parent_path(), error);
        fs::create_hard_link(source.path, target, error);
        if (error) {
            ++stats_.peer_link_failures; // another volume, or the peer removed it just now
            roll_back();
            return skip_locked();
        }
        created.push_back(target);
        linked.emplace_back(source.chunk, StoredChunk{source.compressed, source.size});
        linked_bytes += source.size;
    }
    if (held == nullptr) {
        try {
            write_atomically(manifest_path(id), data);
        } catch (const std::exception&) {
            roll_back();
            return skip_locked();
        }
    }

    // Chunks the index knew by name only (an image registered from the bucket) now have files.
    for (const auto& [chunk, stored] : linked) {
        const auto known = chunks_.find(hex_key(chunk->hash));
        if (known != chunks_.end() && !known->second.present) {
            known->second.present = true;
            known->second.stored  = stored.bytes;
            chunk_total_bytes_ += stored.bytes;
        }
    }
    peer_skipped_.erase(memo_key);
    ++stats_.peer_images;
    stats_.peer_bytes_linked += linked_bytes;
    // Only the chunks linked here are the peer's: one this store already held is not.
    const auto record_fill = [&](Entry& target) {
        Entry::PeerFill& fill = target.peer_fills[memo_key];
        fill.stamp            = stamp;
        for (const auto& [chunk, stored] : linked) { fill.chunk_keys.insert(hex_key(chunk->hash)); }
    };
    if (held != nullptr) {
        // Same references, manifest and generation: a fetch of this entry in flight finds the
        // chunks present and skips them. The manifest stays the bucket's, so the image is still
        // this store's to repair in the bucket. The use time is the later of the bucket's and the
        // peer's.
        record_fill(*held);
        if (entry.info.last_used_ms > held->info.last_used_ms) {
            held->info.last_used_ms = entry.info.last_used_ms;
            write_used(id, held->info.last_used_ms);
        }
        return true;
    }
    entry.peer_memo_key     = memo_key;
    entry.peer_stamp        = stamp;
    record_fill(entry);
    manifest_total_bytes_ += entry.manifest_bytes;
    add_references(entry);
    entry.generation = ++next_generation_;
    entries_.emplace(id, std::move(entry));
    write_used(id, entries_.find(id)->second.info.last_used_ms); // our own sidecar, so a restart keeps it
    return true;
}

} // namespace ninfer::runtime
