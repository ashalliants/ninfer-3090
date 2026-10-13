// Unbuffered sequential write then read of one file, 32 MiB requests (FILE_FLAG_NO_BUFFERING).
// usage: nvme_seq.exe <path> <GiB>
#include <windows.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>

int main(int argc, char** argv) {
    if (argc < 3) { std::fprintf(stderr, "usage: nvme_seq <path> <GiB>\n"); return 2; }
    const char* path = argv[1];
    const unsigned long long total = std::strtoull(argv[2], nullptr, 10) << 30;
    const DWORD block = 32u << 20;
    auto* buffer = static_cast<unsigned char*>(VirtualAlloc(nullptr, block, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE));
    for (DWORD i = 0; i < block; ++i) buffer[i] = static_cast<unsigned char>(i * 2654435761u >> 24);
    using C = std::chrono::steady_clock;
    HANDLE h = CreateFileA(path, GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS,
                           FILE_FLAG_NO_BUFFERING | FILE_FLAG_WRITE_THROUGH, nullptr);
    if (h == INVALID_HANDLE_VALUE) { std::fprintf(stderr, "open for write failed %lu\n", GetLastError()); return 1; }
    auto t0 = C::now();
    for (unsigned long long done = 0; done < total; done += block) {
        buffer[0] = static_cast<unsigned char>(done >> 25);
        DWORD n = 0;
        if (!WriteFile(h, buffer, block, &n, nullptr) || n != block) { std::fprintf(stderr, "write failed\n"); return 1; }
    }
    FlushFileBuffers(h);
    CloseHandle(h);
    double ws = std::chrono::duration<double>(C::now() - t0).count();
    h = CreateFileA(path, GENERIC_READ, 0, nullptr, OPEN_EXISTING, FILE_FLAG_NO_BUFFERING | FILE_FLAG_SEQUENTIAL_SCAN, nullptr);
    if (h == INVALID_HANDLE_VALUE) { std::fprintf(stderr, "open for read failed\n"); return 1; }
    t0 = C::now();
    unsigned long long got = 0;
    for (;;) {
        DWORD n = 0;
        if (!ReadFile(h, buffer, block, &n, nullptr)) { std::fprintf(stderr, "read failed\n"); return 1; }
        if (n == 0) break;
        got += n;
    }
    double rs = std::chrono::duration<double>(C::now() - t0).count();
    CloseHandle(h);
    DeleteFileA(path);
    std::printf("bytes=%llu write_s=%.3f write_GBps=%.3f read_s=%.3f read_GBps=%.3f\n", total, ws,
                total / ws / 1e9, rs, got / rs / 1e9);
    return 0;
}
