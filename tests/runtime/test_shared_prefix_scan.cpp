#include "runtime/engine/shared_prefix_scan.h"

#include <chrono>
#include <cstddef>
#include <iostream>
#include <set>
#include <stdexcept>
#include <utility>
#include <vector>

// Fault injection for the shared-prefix export scan (EngineCore::persist_shared_prefix). The
// callbacks fail or throw on demand, and the logs show which candidates were exported and sunk.

namespace {

using ninfer::runtime::kSharedPrefixBackoff;
using ninfer::runtime::kSharedPrefixRescan;
using ninfer::runtime::scan_shared_prefixes;

struct Image {
    std::vector<std::byte> bytes;
};

enum class SinkMode { Accept, Reject, Throw };

struct Harness {
    std::vector<int> candidates;
    std::set<int> already_stored;
    std::set<int> export_fails;
    SinkMode sink_mode = SinkMode::Accept;

    std::vector<int> exported;    // export attempts, including the failed ones
    std::vector<int> sunk;        // candidates whose image reached the sink
    std::vector<int> stored_now;  // already-stored hits the callback recorded
    std::vector<std::pair<int, std::size_t>> accepted;

    std::chrono::seconds run() {
        int sinking = -1;
        return scan_shared_prefixes(
            candidates,
            [&](int handle) {
                const bool held = already_stored.contains(handle);
                if (held) { stored_now.push_back(handle); }
                return held;
            },
            [&](int handle) {
                exported.push_back(handle);
                if (export_fails.contains(handle)) { throw std::runtime_error("export refused"); }
                sinking = handle;
                return Image{std::vector<std::byte>(static_cast<std::size_t>(handle) * 10)};
            },
            [&](Image&& image) {
                sunk.push_back(sinking);
                if (image.bytes.size() != static_cast<std::size_t>(sinking) * 10) {
                    throw std::logic_error("the sink got a different image");
                }
                if (sink_mode == SinkMode::Throw) { throw std::runtime_error("sink threw"); }
                return sink_mode == SinkMode::Accept;
            },
            [&](int handle, std::size_t bytes) { accepted.emplace_back(handle, bytes); });
    }
};

void require(bool condition, const char* message) {
    if (!condition) { throw std::runtime_error(message); }
}

using Ints  = std::vector<int>;
using Pairs = std::vector<std::pair<int, std::size_t>>;

void test_first_fails_second_exports_and_scan_ends() {
    Harness h{.candidates = {1, 2, 3}, .export_fails = {1}};
    require(h.run() == kSharedPrefixRescan, "a scan that exported one prefix did not rescan at 1 s");
    require(h.exported == Ints{1, 2}, "the failed first export did not fall through to the second only");
    require(h.sunk == Ints{2}, "the second prefix was not the one sunk");
    require(h.accepted == Pairs{{2, 20}}, "the accepted export was not recorded with its size");
}

void test_every_export_fails_backs_off_without_sinking() {
    Harness h{.candidates = {1, 2, 3}, .export_fails = {1, 2, 3}};
    require(h.run() == kSharedPrefixBackoff, "a scan where every export failed did not back off 4 s");
    require(h.exported == Ints{1, 2, 3}, "not every candidate was tried");
    require(h.sunk.empty() && h.accepted.empty(), "an image reached the sink although every export failed");
}

void test_sink_rejection_ends_the_scan_with_backoff() {
    Harness h{.candidates = {1, 2}, .sink_mode = SinkMode::Reject};
    require(h.run() == kSharedPrefixBackoff, "a rejecting sink did not back off 4 s");
    require(h.exported == Ints{1} && h.sunk == Ints{1}, "the scan went past a rejecting sink");
    require(h.accepted.empty(), "a rejected image was recorded as stored");
}

void test_sink_throw_ends_the_scan_with_backoff() {
    Harness h{.candidates = {1, 2}, .sink_mode = SinkMode::Throw};
    require(h.run() == kSharedPrefixBackoff, "a throwing sink did not back off 4 s");
    require(h.exported == Ints{1} && h.sunk == Ints{1}, "the scan went past a throwing sink");
    require(h.accepted.empty(), "an image whose sink threw was recorded as stored");
}

void test_failed_export_then_sink_rejection_ends_after_one_sink_call() {
    Harness h{.candidates = {1, 2, 3}, .export_fails = {1}, .sink_mode = SinkMode::Reject};
    require(h.run() == kSharedPrefixBackoff, "export failure plus a rejecting sink did not back off");
    require(h.exported == Ints{1, 2} && h.sunk == Ints{2}, "the scan did not end after the one sink call");
}

void test_stored_candidates_are_recorded_and_skipped() {
    Harness h{.candidates = {1, 2, 3}, .already_stored = {1, 2}};
    require(h.run() == kSharedPrefixRescan, "a scan past stored candidates did not rescan at 1 s");
    require(h.stored_now == Ints{1, 2}, "stored candidates were not recorded");
    require(h.exported == Ints{3}, "a stored candidate was exported again");
}

void test_nothing_to_do_rescans_at_the_normal_pace() {
    Harness none{};
    require(none.run() == kSharedPrefixRescan, "an empty scan backed off");
    Harness stored{.candidates = {1}, .already_stored = {1}};
    require(stored.run() == kSharedPrefixRescan, "a scan of only stored prefixes backed off");
    require(stored.exported.empty() && stored.sunk.empty(), "a stored prefix reached the exporter or the sink");
}

void test_stored_hit_does_not_hide_an_earlier_failure() {
    // A failure followed only by already-stored candidates is still a failed scan.
    Harness h{.candidates = {1, 2}, .already_stored = {2}, .export_fails = {1}};
    require(h.run() == kSharedPrefixBackoff, "a failed export followed by a stored candidate did not back off");
}

void test_throwing_store_lookup_backs_off() {
    const Ints candidates{1};
    const auto delay = scan_shared_prefixes(
        candidates, [](int) -> bool { throw std::runtime_error("store lookup failed"); },
        [](int) { return Image{}; }, [](Image&&) { return true; }, [](int, std::size_t) {});
    require(delay == kSharedPrefixBackoff, "a failing store lookup did not back off 4 s");
}

}  // namespace

int main() {
    try {
        test_first_fails_second_exports_and_scan_ends();
        test_every_export_fails_backs_off_without_sinking();
        test_sink_rejection_ends_the_scan_with_backoff();
        test_sink_throw_ends_the_scan_with_backoff();
        test_failed_export_then_sink_rejection_ends_after_one_sink_call();
        test_stored_candidates_are_recorded_and_skipped();
        test_nothing_to_do_rescans_at_the_normal_pace();
        test_stored_hit_does_not_hide_an_earlier_failure();
        test_throwing_store_lookup_backs_off();
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
    std::cout << "ok\n";
    return 0;
}
