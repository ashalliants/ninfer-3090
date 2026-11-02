#pragma once

#include <algorithm>
#include <cstddef>
#include <span>
#include <vector>

namespace ninfer::runtime {

// A byte range of the Host backing. Pure placement arithmetic: no allocator, no checkpoints.
struct HostRun {
    std::size_t offset = 0;
    std::size_t bytes  = 0;

    friend bool operator==(const HostRun&, const HostRun&) = default;
};

// The free runs of the Host backing, and the bytes the cache may still occupy: the general budget
// less the occupied bytes it counts, which a snapshot reserve can make smaller than the runs' total.
struct HostFreeRuns {
    std::vector<HostRun> runs;
    std::size_t general_capacity = 0;
    std::size_t occupied         = 0; // all occupied bytes; those above the budget sit in the reserve

    [[nodiscard]] std::size_t budget_free() const {
        return general_capacity - std::min(occupied, general_capacity);
    }
};

// Sorts `runs` and merges those that touch, as the allocator does when an extent is released.
inline void coalesce_host_runs(std::vector<HostRun>& runs) {
    runs.erase(std::remove_if(runs.begin(), runs.end(), [](const HostRun& r) { return !r.bytes; }),
               runs.end());
    std::sort(runs.begin(), runs.end(),
              [](const HostRun& a, const HostRun& b) { return a.offset < b.offset; });
    std::size_t kept = 0;
    for (std::size_t i = 0; i < runs.size(); ++i) {
        if (kept && runs[kept - 1].offset + runs[kept - 1].bytes >= runs[i].offset) {
            const auto end = std::max(runs[kept - 1].offset + runs[kept - 1].bytes,
                                      runs[i].offset + runs[i].bytes);
            runs[kept - 1].bytes = end - runs[kept - 1].offset;
        } else {
            runs[kept++] = runs[i];
        }
    }
    runs.resize(kept);
}

// Bytes that allocations of whole `unit`s could take from `runs` (coalesced): a run shorter than
// one unit holds none. Unit 0 counts every byte.
[[nodiscard]] inline std::size_t host_placeable_bytes(std::span<const HostRun> runs,
                                                      std::size_t unit) {
    std::size_t bytes = 0;
    for (const auto& run : runs) { bytes += unit ? run.bytes / unit * unit : run.bytes; }
    return bytes;
}

[[nodiscard]] inline std::size_t longest_host_run(std::span<const HostRun> runs) {
    std::size_t longest = 0;
    for (const auto& run : runs) { longest = std::max(longest, run.bytes); }
    return longest;
}

// The free runs once `freed` ranges (disjoint from `free` and from each other) are released.
[[nodiscard]] inline HostFreeRuns host_runs_after_release(const HostFreeRuns& free,
                                                          std::span<const HostRun> freed) {
    HostFreeRuns out = free;
    out.runs.insert(out.runs.end(), freed.begin(), freed.end());
    coalesce_host_runs(out.runs);
    std::size_t bytes = 0;
    for (const auto& run : freed) { bytes += run.bytes; }
    out.occupied -= std::min(out.occupied, bytes);
    return out;
}

// Whether a demotion's destination can be allocated from `free`: `bytes` of whole `unit` pages
// (placed in as many runs as needed), or, for one state image (`contiguous`), one run of `bytes`.
// Either way the cache's budget must also have the bytes free.
[[nodiscard]] inline bool host_places(const HostFreeRuns& free, std::size_t bytes, std::size_t unit,
                                      bool contiguous) {
    if (bytes > free.budget_free()) { return false; }
    return contiguous ? longest_host_run(free.runs) >= bytes
                      : host_placeable_bytes(free.runs, unit) >= bytes;
}

} // namespace ninfer::runtime
