// End-to-end read and write throughput of the durable context store (ContextStore put/load), the
// path a stored conversation takes when it is written behind or read back for a request. CPU and
// disk only; no GPU.
//
//   ninfer_context_store_bench put  --dir D --gib N     write one N-GiB image (incompressible bytes)
//   ninfer_context_store_bench load --dir D [--repeat R] read every image in D back, R times
//
// For a cold read, copy the directory with unbuffered I/O (Windows: `xcopy /J /E /I D D2`) so none
// of it is in the file cache, then `load --dir D2`. The first load of a directory is the cold
// figure; later repeats read from the file cache.
#include "runtime/engine/context_store/context_store.h"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <random>
#include <string>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

int usage() {
    std::fprintf(stderr, "usage: ninfer_context_store_bench put --dir D --gib N\n"
                         "       ninfer_context_store_bench load --dir D [--repeat R]\n");
    return 2;
}

} // namespace

int main(int argc, char** argv) try {
    if (argc < 2) { return usage(); }
    const std::string mode = argv[1];
    std::string dir;
    double gib       = 1.0;
    int repeat       = 1;
    for (int i = 2; i + 1 < argc; i += 2) {
        if (std::strcmp(argv[i], "--dir") == 0) {
            dir = argv[i + 1];
        } else if (std::strcmp(argv[i], "--gib") == 0) {
            gib = std::atof(argv[i + 1]);
        } else if (std::strcmp(argv[i], "--repeat") == 0) {
            repeat = std::atoi(argv[i + 1]);
        } else {
            return usage();
        }
    }
    if (dir.empty() || gib <= 0.0 || repeat <= 0) { return usage(); }
    ninfer::runtime::ContextStore::Options options;
    options.directory = dir;
    ninfer::runtime::ContextStore store(std::move(options));
    if (mode == "put") {
        const auto bytes = static_cast<std::size_t>(gib * double(1ULL << 30U));
        std::vector<std::uint8_t> image(bytes);
        std::mt19937_64 random(20261013);
        for (std::size_t i = 0; i + 8 <= image.size(); i += 8) {
            const std::uint64_t value = random();
            std::memcpy(image.data() + i, &value, 8);
        }
        ninfer::runtime::ContextStore::Description description;
        description.id      = "00000000000000000000000000000001";
        description.binding = "context-store-bench";
        description.tokens  = 1;
        description.checkpoints.push_back({.frontier = 1, .digests = {1, 2}});
        // Everything but a small header is chunked, as a checkpoint image's payload is.
        const ninfer::runtime::ContextStore::Region region{.offset = 4096,
                                                           .length = image.size() - 4096};
        const auto started = Clock::now();
        const auto result  = store.put(description, image, {&region, 1});
        const double seconds = std::chrono::duration<double>(Clock::now() - started).count();
        std::printf("put bytes=%zu written=%llu seconds=%.3f GBps=%.3f\n", image.size(),
                    static_cast<unsigned long long>(result.chunk_bytes_written), seconds,
                    double(image.size()) / seconds / 1e9);
        return 0;
    }
    if (mode != "load") { return usage(); }
    for (int pass = 0; pass < repeat; ++pass) {
        for (const auto& info : store.list()) {
            const auto started = Clock::now();
            const auto image   = store.load(info.id, false);
            const double seconds = std::chrono::duration<double>(Clock::now() - started).count();
            if (!image) {
                std::fprintf(stderr, "load %s failed\n", info.id.c_str());
                return 1;
            }
            std::printf("load pass=%d bytes=%zu seconds=%.3f GBps=%.3f\n", pass, image->size(),
                        seconds, double(image->size()) / seconds / 1e9);
        }
    }
    return 0;
} catch (const std::exception& error) {
    std::fprintf(stderr, "error: %s\n", error.what());
    return 1;
}
