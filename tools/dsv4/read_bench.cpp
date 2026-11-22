// Written for NInfer-3090.
//
// Micro-benchmark of how DeepSeek-V4.1-Flash's routed expert records can be read from a drive: random reads of
// `record` bytes (an expert record is about 9.3 MB) at 4 KiB aligned offsets of a big file, in batches whose
// offsets are sorted (what FileExpertSource does for the missing records of one call). Nothing is written.
//
//   g++ -std=c++20 -O2 -pthread tools/dsv4/read_bench.cpp -o read_bench
//   ./read_bench FILE MODE THREADS [records=48] [record_mb=9.3] [seed=1] [chunk_kb=0] [batch=16]
//
// MODE: buffered  one pread per record (the page cache is in the path)
//       direct    O_DIRECT, aligned buffers (fails on a file system without it)
//       willneed  posix_fadvise(WILLNEED) for the whole batch, then buffered preads
//       random    posix_fadvise(RANDOM) on the file (no kernel readahead), then buffered preads
//       mmap      mmap the file and copy each record out (page faults do the reads)
// THREADS: reader threads sharing a batch. chunk_kb > 0 splits each record into chunks of that size, which the
// threads read independently (several requests in flight for one record). Prints MB/s and ms per record.

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace {

struct Job {
    std::uint64_t offset, size;
    std::uint8_t* destination;
};

} // namespace

int main(int argc, char** argv) {
    if (argc < 4) {
        std::fprintf(stderr, "usage: %s FILE MODE THREADS [records] [record_mb] [seed] [chunk_kb] [batch]\n", argv[0]);
        return 2;
    }
    const char* path       = argv[1];
    const std::string mode = argv[2];
    const int threads      = std::max(1, std::atoi(argv[3]));
    const int records      = argc > 4 ? std::atoi(argv[4]) : 48;
    const double record_mb = argc > 5 ? std::atof(argv[5]) : 9.3;
    const unsigned seed    = argc > 6 ? static_cast<unsigned>(std::atoi(argv[6])) : 1;
    const std::uint64_t chunk = argc > 7 ? static_cast<std::uint64_t>(std::atoll(argv[7])) * 1024 : 0;
    const int batch        = argc > 8 ? std::atoi(argv[8]) : 16;
    if (records <= 0 || batch <= 0 || record_mb <= 0 || (chunk != 0 && chunk < 4096)) {
        std::fprintf(stderr, "records, batch and record_mb must be positive and chunk_kb 0 or at least 4\n");
        return 2;
    }
    const bool direct      = mode == "direct";
    const int fd           = ::open(path, O_RDONLY | (direct ? O_DIRECT : 0));
    if (fd < 0) {
        std::perror("open");
        return 1;
    }
    struct stat st {};
    ::fstat(fd, &st);
    const std::uint64_t size   = static_cast<std::uint64_t>(st.st_size);
    const std::uint64_t record = (static_cast<std::uint64_t>(record_mb * 1e6) + 4095) / 4096 * 4096;
    if (size < record + 4096) {
        std::fprintf(stderr, "the file (%llu bytes) must hold one record and a page more (%llu bytes)\n", static_cast<unsigned long long>(size), static_cast<unsigned long long>(record));
        return 2;
    }
    if (mode == "random") { ::posix_fadvise(fd, 0, 0, POSIX_FADV_RANDOM); }
    const std::uint8_t* map = nullptr;
    if (mode == "mmap") {
        void* m = ::mmap(nullptr, size, PROT_READ, MAP_SHARED, fd, 0);
        if (m == MAP_FAILED) {
            std::perror("mmap");
            return 1;
        }
        map = static_cast<const std::uint8_t*>(m);
        ::madvise(m, size, MADV_RANDOM);
    }
    std::mt19937_64 rng(seed);
    std::vector<std::uint64_t> offsets(static_cast<std::size_t>(records));
    for (auto& o : offsets) { o = (rng() % ((size - record) / 4096)) * 4096; }
    std::vector<std::uint8_t*> buffers;
    for (int i = 0; i < batch; ++i) {
        void* p = nullptr;
        if (::posix_memalign(&p, 4096, record) != 0) { return 1; }
        std::memset(p, 0, record);
        buffers.push_back(static_cast<std::uint8_t*>(p));
    }
    std::atomic<std::uint64_t> total{0};
    std::atomic<bool> failed{false};
    const auto start = std::chrono::steady_clock::now();
    for (int first = 0; first < records; first += batch) {
        const int n = std::min(batch, records - first);
        std::vector<std::uint64_t> sorted(offsets.begin() + first, offsets.begin() + first + n);
        std::sort(sorted.begin(), sorted.end());
        std::vector<Job> jobs;
        for (int i = 0; i < n; ++i) {
            const std::uint64_t piece = chunk == 0 ? record : chunk;
            for (std::uint64_t at = 0; at < record; at += piece) { jobs.push_back({sorted[i] + at, std::min(piece, record - at), buffers[i] + at}); }
            if (mode == "willneed") { ::posix_fadvise(fd, static_cast<off_t>(sorted[i]), static_cast<off_t>(record), POSIX_FADV_WILLNEED); }
        }
        std::atomic<std::size_t> next{0};
        const auto work = [&] {
            for (;;) {
                const std::size_t j = next.fetch_add(1);
                if (j >= jobs.size()) { return; }
                const Job& job = jobs[j];
                if (map != nullptr) {
                    std::memcpy(job.destination, map + job.offset, job.size);
                    total += job.size;
                    continue;
                }
                std::uint64_t done = 0;
                while (done < job.size) {
                    const ssize_t r = ::pread(fd, job.destination + done, job.size - done, static_cast<off_t>(job.offset + done));
                    if (r <= 0) {
                        std::perror("pread");
                        failed = true;
                        return;
                    }
                    done += static_cast<std::uint64_t>(r);
                }
                total += done;
            }
        };
        std::vector<std::thread> pool;
        for (int t = 1; t < threads; ++t) { pool.emplace_back(work); }
        work();
        for (auto& t : pool) { t.join(); }
        if (failed) { return 1; }
    }
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    std::printf("%-8s threads %2d chunk %5llu KiB: %7.1f MB/s, %6.1f ms per %.1f MB record (%d records, %.1f s)\n", mode.c_str(), threads,
                static_cast<unsigned long long>(chunk / 1024), static_cast<double>(total) / 1e6 / seconds, 1000.0 * seconds / records,
                static_cast<double>(record) / 1e6, records, seconds);
    return 0;
}
