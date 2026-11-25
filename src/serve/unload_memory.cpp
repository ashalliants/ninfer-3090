#include "serve/unload_memory.h"

#include "serve/generation_service.h"

#include <algorithm>
#include <cstdio>
#include <exception>

#if defined(__GLIBC__)
#include <malloc.h>
#endif
#if defined(__linux__)
#include <unistd.h>
#endif

namespace ninfer::serve {

UnloadStoreSave save_context_before_unload(GenerationService& service) noexcept {
    UnloadStoreSave out;
    try {
        if (!service.engine_options().context_store.enabled()) { return out; }
        if (service.has_failed()) {
            out.skipped = "the model failed";
            return out;
        }
        const auto residency = service.residency();
        if (residency.enabled && residency.state == ninfer::ModelResidency::Suspended) {
            out.skipped = "the model is asleep";
            return out;
        }
        const std::chrono::milliseconds budget = std::min<std::chrono::milliseconds>(
            std::chrono::seconds(service.options().context_store_flush_seconds),
            kUnloadStoreSaveBudget);
        out.attempted      = true;
        const auto started = std::chrono::steady_clock::now();
        try {
            const auto report = service.flush_context_store(budget);
            out.written       = report.written;
            out.unsaved       = report.unsaved;
        } catch (const std::exception& error) {
            out.error = error.what();
        } catch (...) { out.error = "unknown error"; }
        out.seconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    } catch (const std::exception& error) {
        out.error = error.what();
    } catch (...) { out.error = "unknown error"; }
    return out;
}

std::uint64_t resident_bytes() noexcept {
#if defined(__linux__)
    std::FILE* file = std::fopen("/proc/self/statm", "r");
    if (file == nullptr) { return 0; }
    unsigned long long size = 0;
    unsigned long long resident = 0;
    const int read = std::fscanf(file, "%llu %llu", &size, &resident);
    std::fclose(file);
    const long page = sysconf(_SC_PAGESIZE);
    if (read != 2 || page <= 0) { return 0; }
    return static_cast<std::uint64_t>(resident) * static_cast<std::uint64_t>(page);
#else
    return 0;
#endif
}

HeapRelease release_freed_heap() noexcept {
    HeapRelease out;
    out.resident_before = resident_bytes();
#if defined(__GLIBC__)
    // Walks every arena and gives back the free pages inside them, not only the top of each heap:
    // an Engine's loader and worker threads free gigabytes into per-thread arenas that a plain
    // free() never shrinks.
    (void)malloc_trim(0);
    out.trimmed = true;
#endif
    out.resident_after = resident_bytes();
    return out;
}

namespace {

std::string gib(std::uint64_t bytes) {
    char text[32];
    std::snprintf(text, sizeof(text), "%.2f", static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0));
    return text;
}

} // namespace

std::string describe_unload(const UnloadStoreSave& save, const HeapRelease& heap) {
    std::string out;
    char text[160];
    if (!save.error.empty()) {
        std::snprintf(text, sizeof(text), "context store: save failed after %.1f s (", save.seconds);
        out += text + save.error + "), unloading anyway";
    } else if (save.attempted) {
        std::snprintf(text, sizeof(text), "context store: %llu saved in %.1f s",
                      static_cast<unsigned long long>(save.written), save.seconds);
        out += text;
        if (save.unsaved != 0) {
            std::snprintf(text, sizeof(text), ", %llu not saved (out of time)",
                          static_cast<unsigned long long>(save.unsaved));
            out += text;
        }
    } else if (!save.skipped.empty()) {
        out += "context store: not saved (" + save.skipped + ")";
    }
    if (heap.resident_before != 0 && heap.resident_after != 0) {
        if (!out.empty()) { out += " | "; }
        out += "RAM " + gib(heap.resident_before) + " -> " + gib(heap.resident_after) + " GiB";
        if (heap.resident_after < heap.resident_before) {
            out += " (" + gib(heap.resident_before - heap.resident_after) +
                   " GiB returned to the system)";
        }
    }
    return out;
}

} // namespace ninfer::serve
