#pragma once

#include <chrono>
#include <exception>
#include <optional>
#include <utility>

namespace ninfer::runtime {

// One scan of the public shared prefixes that the store does not hold yet, kept apart from the
// Engine so its fault isolation can be tested without a device. The Engine passes the callbacks
// that touch the Program and the store.
//
// The rules, each with a test in tests/runtime/test_shared_prefix_scan.cpp:
//   - A candidate the store already holds is recorded by `already_stored` and skipped.
//   - A candidate whose export throws is skipped, not fatal: the next one is tried.
//   - At most one image reaches the sink per scan, and the scan ends there whatever the sink
//     answers. A sink that returns false or throws counts as a failed scan.
//   - A failed scan (every candidate failed to export, or the sink refused) asks for a back-off
//     of kSharedPrefixBackoff instead of the usual rescan interval.
inline constexpr std::chrono::seconds kSharedPrefixRescan{1};
inline constexpr std::chrono::seconds kSharedPrefixBackoff{4};

// The delay before the next scan.
//   candidates         the handles still to export, in order
//   already_stored(h)  true when the store has h's image; records that itself
//   export_image(h)    the image for h; throws when h cannot be exported
//   sink(image)        true when the image was accepted; may throw
//   accepted(h, bytes) called once, after the sink accepts h's image of that many bytes
// An image type needs a `bytes` member with size().
template <class Candidates, class AlreadyStored, class ExportImage, class Sink, class Accepted>
[[nodiscard]] std::chrono::seconds
scan_shared_prefixes(const Candidates& candidates, AlreadyStored&& already_stored,
                     ExportImage&& export_image, Sink&& sink, Accepted&& accepted) noexcept {
    try {
        bool failed = false;
        for (const auto handle : candidates) {
            if (already_stored(handle)) { continue; }
            // One prefix that cannot be exported (the Program refuses it, or a transient failure)
            // must not keep the others from being tried. One export per scan.
            std::optional<decltype(export_image(handle))> image;
            try {
                image.emplace(export_image(handle));
            } catch (...) {
                failed = true;
                continue;
            }
            // The sink ends the scan whatever it answers: a rejection (full queue, shutdown) or a
            // throw backs off like a failed scan instead of exporting another prefix.
            const auto bytes = image->bytes.size();
            bool taken       = false;
            try { taken = sink(std::move(*image)); } catch (...) {}
            if (!taken) { return kSharedPrefixBackoff; }
            accepted(handle, bytes);
            return kSharedPrefixRescan;
        }
        // Every candidate failed: back off instead of retrying them each second.
        return failed ? kSharedPrefixBackoff : kSharedPrefixRescan;
    } catch (...) { return kSharedPrefixBackoff; }
}

}  // namespace ninfer::runtime
