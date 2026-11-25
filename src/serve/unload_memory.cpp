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

namespace {

class ServiceTarget final : public UnloadStoreTarget {
public:
    explicit ServiceTarget(GenerationService& service) : service_(service) {}
    bool store_enabled() const override { return service_.engine_options().context_store.enabled(); }
    bool failed() const override { return service_.has_failed(); }
    bool asleep() const override {
        const auto residency = service_.residency();
        return residency.enabled && residency.state == ninfer::ModelResidency::Suspended;
    }
    void hold_asleep() override { (void)service_.suspend(false); }
    std::uint32_t flush_seconds() const override { return service_.options().context_store_flush_seconds; }
    ninfer::ContextStoreFlushReport flush(std::chrono::milliseconds budget) override {
        return service_.flush_context_store(budget);
    }

private:
    GenerationService& service_;
};

} // namespace

UnloadStoreSave save_context_before_unload(GenerationService& service) noexcept {
    ServiceTarget target(service);
    return save_context_before_unload(target);
}

UnloadStoreSave save_context_before_unload(UnloadStoreTarget& target) noexcept {
    UnloadStoreSave out;
    try {
        if (!target.store_enabled()) { return out; }
        if (target.failed()) {
            out.skipped = "the model failed";
            return out;
        }
        if (target.asleep()) {
            out.skipped = "the model is asleep";
            try {
                target.hold_asleep();
            } catch (const std::exception& error) {
                out.error = std::string("could not keep it asleep: ") + error.what();
            } catch (...) { out.error = "could not keep it asleep"; }
            return out;
        }
        const std::chrono::milliseconds budget = std::min<std::chrono::milliseconds>(
            std::chrono::seconds(target.flush_seconds()), kUnloadStoreSaveBudget);
        out.attempted      = true;
        const auto started = std::chrono::steady_clock::now();
        try {
            const auto report = target.flush(budget);
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
#if __GLIBC__ > 2 || (__GLIBC__ == 2 && __GLIBC_MINOR__ >= 33)
    const struct mallinfo2 info = mallinfo2();
    out.heap_in_use             = static_cast<std::uint64_t>(info.uordblks) + static_cast<std::uint64_t>(info.hblkhd);
#endif
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
    if (!save.error.empty() && !save.attempted) {
        out += "context store: not saved (" + save.skipped + "; " + save.error + ")";
    } else if (!save.error.empty()) {
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
        if (heap.heap_in_use != 0) { out += "; " + gib(heap.heap_in_use) + " GiB of heap still in use"; }
    }
    return out;
}

} // namespace ninfer::serve
