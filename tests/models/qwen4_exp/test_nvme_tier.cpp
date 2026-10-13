// The Qwen3.8-Flash-Next context cache's NVMe tier (program/nvme_tier.h) on a real scratch file and
// the GPU (no model artifact): device bytes captured into slabs come back exactly, whether the slabs
// are in RAM frames, written through and evicted to the file, or captured straight to the file
// through the bounce halves; host reads and writes (checkpoint images) agree with device copies; a
// discarded slab rewritten holds its new bytes; with a failing disk a frame whose write failed keeps
// serving its bytes from RAM, a direct capture or a host write throws, and a failed read throws;
// a tier with no frames works from the file alone.
//
// --bench DIR [MiB]: capture and restore MiB (default 512) of 912 KiB slabs through the file in DIR,
// five times each, with no RAM frames (the restore speed of a checkpoint that is only on disk) and
// with frames (from RAM), and prints the medians.

#include "models/qwen4_exp/program/nvme_tier.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

using namespace ninfer;
using namespace ninfer::models::qwen4_exp;

namespace {

void require(bool value, const std::string& message) {
    if (!value) { throw std::runtime_error(message); }
}

void cuda(cudaError_t status, const char* what) {
    if (status != cudaSuccess) { throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status)); }
}

// A scratch file whose reads or writes can be made to fail (a full or failing disk).
class FlakyFile final : public TierFile {
public:
    explicit FlakyFile(std::unique_ptr<TierFile> inner) : inner_(std::move(inner)) {}
    void read(std::span<const Request> requests) override {
        if (fail_reads) { throw std::system_error(EIO, std::generic_category(), "test: read failed"); }
        inner_->read(requests);
    }
    void write(std::span<const Request> requests) override {
        if (fail_writes) { throw std::system_error(ENOSPC, std::generic_category(), "test: the disk is full"); }
        inner_->write(requests);
    }
    std::atomic<bool> fail_reads{false}, fail_writes{false};

private:
    std::unique_ptr<TierFile> inner_;
};

std::filesystem::path scratch(const char* name) {
    return std::filesystem::current_path() / (std::string("ninfer_nvme_tier_test_") + name + ".tier");
}

// Every slab of a test holds two parts: a long row at its start (copied one row per slab, runs of
// consecutive slabs as one copy) and short rows further in (copied per slab).
constexpr std::size_t kSlab      = 64 * 1024;
constexpr std::size_t kLong      = 32 * 1024;
constexpr std::size_t kShortAt   = 32 * 1024;
constexpr std::size_t kShort     = 256;
constexpr std::size_t kShortStep = 300;
constexpr std::size_t kShortRows = 100;

struct Device {
    std::byte* data   = nullptr;
    std::size_t bytes = 0;
    explicit Device(std::size_t n) : bytes(n) { cuda(cudaMalloc(&data, n), "cudaMalloc"); }
    ~Device() { cudaFree(data); }
    Device(const Device&)            = delete;
    Device& operator=(const Device&) = delete;
};

// The copies between slabs (ascending) and a device region laid out as [long rows][short rows].
std::vector<SlabCopy> copies_for(const std::vector<std::uint32_t>& slabs, std::byte* device) {
    std::vector<SlabCopy> out;
    std::byte* shorts = device + slabs.size() * kLong;
    for (std::size_t i = 0; i < slabs.size();) {
        std::size_t j = i + 1;
        while (j < slabs.size() && slabs[j] == slabs[j - 1] + 1) { ++j; }
        out.push_back(SlabCopy{.slab = slabs[i], .offset = 0, .slab_pitch = kSlab, .device = device + i * kLong,
                               .device_pitch = kLong, .width = kLong, .rows = j - i});
        i = j;
    }
    for (std::size_t i = 0; i < slabs.size(); ++i) {
        out.push_back(SlabCopy{.slab = slabs[i], .offset = kShortAt, .slab_pitch = kShortStep,
                               .device = shorts + i * kShortRows * kShort, .device_pitch = kShort, .width = kShort,
                               .rows = kShortRows});
    }
    return out;
}

std::size_t region_bytes(std::size_t slabs) { return slabs * (kLong + kShortRows * kShort); }

std::vector<std::byte> pattern(std::size_t bytes, std::uint32_t seed) {
    std::vector<std::byte> out(bytes);
    std::uint32_t x = seed * 2654435761U + 1U;
    for (auto& b : out) {
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        b = static_cast<std::byte>(x & 0xff);
    }
    return out;
}

struct Harness {
    NvmeTier& tier;
    cudaStream_t stream;

    void capture(const std::vector<std::uint32_t>& slabs, const std::vector<std::byte>& bytes) {
        Device d(bytes.size());
        cuda(cudaMemcpy(d.data, bytes.data(), bytes.size(), cudaMemcpyHostToDevice), "upload");
        // The tier works on a non-blocking stream, which the legacy-stream upload does not order.
        cuda(cudaDeviceSynchronize(), "upload");
        const auto copies = copies_for(slabs, d.data);
        tier.copy(copies, true, stream);
        cuda(cudaStreamSynchronize(stream), "capture");
    }

    std::vector<std::byte> restore(const std::vector<std::uint32_t>& slabs) {
        Device d(region_bytes(slabs.size()));
        cuda(cudaMemset(d.data, 0xEE, d.bytes), "memset");
        cuda(cudaDeviceSynchronize(), "memset");
        const auto copies = copies_for(slabs, d.data);
        tier.copy(copies, false, stream);
        cuda(cudaStreamSynchronize(stream), "restore");
        std::vector<std::byte> out(d.bytes);
        cuda(cudaMemcpy(out.data(), d.data, d.bytes, cudaMemcpyDeviceToHost), "download");
        return out;
    }
};

// Equal, or says where the first difference is.
bool same(const std::vector<std::byte>& a, const std::vector<std::byte>& b) {
    if (a == b) { return true; }
    std::size_t i = 0;
    while (i < a.size() && i < b.size() && a[i] == b[i]) { ++i; }
    std::size_t differ = 0;
    for (std::size_t k = 0; k < std::min(a.size(), b.size()); ++k) { differ += a[k] != b[k] ? 1 : 0; }
    std::fprintf(stderr, "first difference at byte %zu of %zu (%zu differ; got %d, expected %d)\n", i, a.size(), differ,
                 i < a.size() ? static_cast<int>(a[i]) : -1, i < b.size() ? static_cast<int>(b[i]) : -1);
    return false;
}

std::vector<std::uint32_t> range(std::uint32_t first, std::uint32_t count) {
    std::vector<std::uint32_t> out(count);
    std::iota(out.begin(), out.end(), first);
    return out;
}

void test_frames_eviction_and_direct(cudaStream_t stream) {
    NvmeTier::Options options{.slabs = 256, .slab_bytes = kSlab, .frames = 16, .bounce_slabs = 4};
    NvmeTier tier(options, open_scratch_tier_file(scratch("a"), 256ULL * kSlab));
    Harness h{tier, stream};
    // Through RAM frames (6 slabs, at most half the frames), written through to the file.
    const auto first = std::vector<std::uint32_t>{3, 4, 5, 9, 10, 40};
    const auto a     = pattern(region_bytes(first.size()), 1);
    h.capture(first, a);
    tier.flush();
    auto stats = tier.stats();
    require(stats.slabs_written == 6 && stats.dirty == 0 && stats.cached == 6, "a framed capture is written through");
    require(same(h.restore(first), a), "a framed capture restores from RAM");
    require(tier.stats().slabs_from_ram == 6 && tier.stats().slabs_read == 0, "it came from RAM");
    // Two more captures of 8 evict the least recently used frames to the file.
    const auto b = pattern(region_bytes(8), 2);
    const auto c = pattern(region_bytes(8), 3);
    h.capture(range(100, 8), b);
    tier.flush();
    h.capture(range(120, 8), c);
    tier.flush();
    const auto before = tier.stats();
    require(same(h.restore(first), a), "evicted slabs restore from the file");
    const auto after = tier.stats();
    require(after.slabs_read - before.slabs_read > 0, "some of them came from the file");
    require(same(h.restore(range(120, 8)), c) && same(h.restore(range(100, 8)), b), "the later captures restore");
    // A capture larger than half the frames goes straight to the file through the bounce halves.
    const auto big = range(200, 12);
    const auto d   = pattern(region_bytes(big.size()), 4);
    const auto written = tier.stats().slabs_written;
    h.capture(big, d);
    require(tier.stats().slabs_written == written + 12, "a large capture is written directly");
    require(same(h.restore(big), d), "a direct capture restores from the file");
    // A slab freed and captured again holds its new bytes (RAM and file).
    for (const std::uint32_t s : first) { tier.discard(s); }
    const auto e = pattern(region_bytes(first.size()), 5);
    h.capture(first, e);
    tier.flush();
    require(same(h.restore(first), e), "a rewritten slab restores its new bytes");
    // Host whole-slab I/O (images) agrees with the device copies.
    std::vector<std::byte> slabs(first.size() * kSlab);
    tier.read(first, slabs);
    const auto host = pattern(slabs.size(), 6);
    tier.write(range(150, 6), host);
    std::vector<std::byte> back(host.size());
    tier.read(range(150, 6), back);
    require(back == host, "a host write reads back");
    tier.write(range(160, 6), slabs);
    require(same(h.restore(range(160, 6)), e), "slabs copied through the host restore as the originals");
    std::printf("ok: frames, eviction, direct captures and host I/O\n");
}

void test_failing_disk(cudaStream_t stream) {
    auto flaky_owner = std::make_unique<FlakyFile>(open_scratch_tier_file(scratch("b"), 128ULL * kSlab));
    FlakyFile& flaky = *flaky_owner;
    NvmeTier tier({.slabs = 128, .slab_bytes = kSlab, .frames = 16, .bounce_slabs = 4}, std::move(flaky_owner));
    Harness h{tier, stream};
    flaky.fail_writes = true;
    const auto slabs = range(10, 4);
    const auto a     = pattern(region_bytes(slabs.size()), 7);
    h.capture(slabs, a);
    tier.flush();
    const auto stats = tier.stats();
    require(stats.write_failures >= 4 && stats.stuck == 4, "failed writes leave their frames stuck in RAM");
    require(same(h.restore(slabs), a), "a stuck frame still restores its bytes");
    bool threw = false;
    try {
        h.capture(range(40, 12), pattern(region_bytes(12), 8));
    } catch (const std::system_error&) { threw = true; }
    require(threw, "a direct capture on a full disk throws");
    threw = false;
    try {
        tier.write(range(60, 2), pattern(2 * kSlab, 9));
    } catch (const std::system_error&) { threw = true; }
    require(threw, "a host write on a full disk throws");
    flaky.fail_writes = false;
    const auto b = pattern(region_bytes(12), 10);
    h.capture(range(40, 12), b); // direct
    flaky.fail_reads = true;
    threw = false;
    try {
        (void)h.restore(range(40, 12));
    } catch (const std::system_error&) { threw = true; }
    require(threw, "a failed read throws");
    flaky.fail_reads = false;
    require(same(h.restore(range(40, 12)), b), "the disk recovered: the bytes are there");
    for (const std::uint32_t s : slabs) { tier.discard(s); }
    require(tier.stats().stuck == 0, "freeing a stuck slab frees its frame");
    std::printf("ok: a failing disk degrades to errors the caller turns into misses\n");
}

void test_no_frames(cudaStream_t stream) {
    NvmeTier tier({.slabs = 64, .slab_bytes = kSlab, .frames = 0, .bounce_slabs = 3}, open_scratch_tier_file(scratch("c"), 64ULL * kSlab));
    Harness h{tier, stream};
    const auto slabs = std::vector<std::uint32_t>{0, 1, 2, 7, 8, 30, 31, 32, 33, 63};
    const auto a     = pattern(region_bytes(slabs.size()), 11);
    h.capture(slabs, a);
    require(same(h.restore(slabs), a), "a tier without frames restores from the file");
    require(tier.stats().slabs_from_ram == 0 && tier.stats().slabs_read == slabs.size(), "all of it from the file");
    std::printf("ok: no RAM frames\n");
}

double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

// The tier's own speed: capture and restore `mib` of 912 KiB slabs (the real page record) five times.
void bench(const std::filesystem::path& dir, std::size_t mib, cudaStream_t stream) {
    constexpr std::size_t slab = 933'888;
    const auto count           = static_cast<std::uint32_t>((mib << 20) / slab);
    Device device(static_cast<std::size_t>(count) * slab);
    cuda(cudaMemset(device.data, 0x5A, device.bytes), "memset");
    std::vector<SlabCopy> copies;
    copies.push_back(SlabCopy{.slab = 0, .offset = 0, .slab_pitch = slab, .device = device.data, .device_pitch = slab,
                              .width = slab, .rows = count});
    for (const std::uint32_t frames : {0U, 2 * count + 2}) {
        NvmeTier tier({.slabs = count, .slab_bytes = slab, .frames = frames, .bounce_slabs = 70},
                      open_scratch_tier_file(dir / "ninfer_nvme_tier_bench.tier", static_cast<std::uint64_t>(count) * slab));
        std::vector<double> capture, restore;
        for (int run = 0; run < 5; ++run) {
            for (std::uint32_t s = 0; s < count; ++s) { tier.discard(s); }
            auto start = std::chrono::steady_clock::now();
            tier.copy(copies, true, stream);
            cuda(cudaStreamSynchronize(stream), "capture");
            tier.flush();
            capture.push_back(std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count());
            start = std::chrono::steady_clock::now();
            tier.copy(copies, false, stream);
            cuda(cudaStreamSynchronize(stream), "restore");
            restore.push_back(std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count());
        }
        const double bytes = static_cast<double>(count) * slab;
        std::printf("bench: %s, %.0f MiB: capture %.1f ms median (%.2f GB/s, written through), restore %.1f ms median "
                    "(%.2f GB/s); restore runs:",
                    frames == 0 ? "file only" : "RAM frames", bytes / (1 << 20), median(capture) * 1e3,
                    bytes / median(capture) / 1e9, median(restore) * 1e3, bytes / median(restore) / 1e9);
        for (const double r : restore) { std::printf(" %.1f", r * 1e3); }
        std::printf(" ms\n");
    }
}

} // namespace

int main(int argc, char** argv) {
    try {
        int devices = 0;
        if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
            std::printf("skipped: no CUDA device\n");
            return 77;
        }
        cudaStream_t stream = nullptr;
        cuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "stream");
        if (argc >= 3 && std::string(argv[1]) == "--bench") {
            bench(argv[2], argc >= 4 ? std::stoul(argv[3]) : 512, stream);
            return 0;
        }
        test_frames_eviction_and_direct(stream);
        test_failing_disk(stream);
        test_no_frames(stream);
        cudaStreamDestroy(stream);
        std::printf("qwen4_exp NVMe tier tests passed\n");
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL: %s\n", error.what());
        return 1;
    }
}
