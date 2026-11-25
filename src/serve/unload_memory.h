#pragma once

// What the router does around a model's unload so that switching models leaves the machine as it
// found it: the model's retained conversations go to its context store first (when it has one),
// and once its Engine is gone the heap it freed goes back to the system.

#include <chrono>
#include <cstdint>
#include <string>

namespace ninfer::serve {

class GenerationService;

// The most a switch waits for the context store, whatever --context-store-flush-seconds says (that
// bound is for a server shutting down, where nobody is waiting for the next model).
inline constexpr std::chrono::seconds kUnloadStoreSaveBudget{20};

// The context store write before an unload.
struct UnloadStoreSave {
    bool attempted = false;      // the model has a context store and was asked to write
    std::string skipped;         // why it was not attempted (empty: no store, or attempted)
    std::uint64_t written = 0;   // conversations written during the save
    std::uint64_t unsaved = 0;   // retained conversations still not stored when the budget ran out
    double seconds        = 0.0;
    std::string error;           // the save failed; the unload went ahead
};

// Writes what the model's context store does not yet hold, within the smaller of
// --context-store-flush-seconds and kUnloadStoreSaveBudget, through the store's own write path
// (Engine::flush_context_store: the same eligibility and size limits as any store write). Never
// throws: a failure is reported in `error`. Skipped when the store is off, the Engine failed, or
// the model is asleep (waking it would claim the device memory the next model needs).
[[nodiscard]] UnloadStoreSave save_context_before_unload(GenerationService& service) noexcept;

// Returns memory the process freed but the allocator kept (glibc's per-thread arenas hold freed
// blocks until trimmed) to the system, and measures the resident set around it.
struct HeapRelease {
    std::uint64_t resident_before = 0; // bytes; 0 when not measurable on this platform
    std::uint64_t resident_after  = 0;
    bool trimmed                  = false; // the allocator was asked to give memory back
};
[[nodiscard]] HeapRelease release_freed_heap() noexcept;

// This process's resident set in bytes (/proc/self/statm), or 0 when not available.
[[nodiscard]] std::uint64_t resident_bytes() noexcept;

// One line for the router log, e.g. "context store: 3 saved in 0.4 s | RAM 7.31 -> 1.05 GiB
// (6.26 GiB returned to the system)".
[[nodiscard]] std::string describe_unload(const UnloadStoreSave& save, const HeapRelease& heap);

} // namespace ninfer::serve
