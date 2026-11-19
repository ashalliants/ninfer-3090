// Expert extents: the layout packs aligned, padded records; the reader returns each wanted record's
// exact bytes whatever the mix of sizes, request order and ring depth, and refuses what would read
// garbage (a bad index, an extent past the file, an unaligned or too small ring).

#include "core/expert_extents.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <span>
#include <stdexcept>
#include <vector>

namespace {

using ninfer::ExpertExtent;
using ninfer::ExpertExtentReader;
using ninfer::kExtentAlignment;

int g_failures = 0;
void check(bool ok, const char* what) {
    if (!ok) {
        std::cerr << "FAILED: " << what << '\n';
        ++g_failures;
    }
}

template <class E, class F>
bool throws(F&& f) {
    try {
        f();
    } catch (const E&) { return true; } catch (...) { return false; }
    return false;
}

// A byte that depends on the record and its position, so a read from the wrong place shows.
std::byte pattern(std::size_t record, std::uint64_t i) {
    return static_cast<std::byte>((record * 131 + i * 7 + (i >> 8)) & 0xFF);
}

// An aligned buffer of `bytes` carved from a vector.
struct AlignedRing {
    std::vector<std::byte> storage;
    std::span<std::byte> bytes;
    explicit AlignedRing(std::size_t n) : storage(n + kExtentAlignment) {
        const auto address = reinterpret_cast<std::uintptr_t>(storage.data());
        const auto skew    = (kExtentAlignment - address % kExtentAlignment) % kExtentAlignment;
        bytes              = std::span<std::byte>(storage.data() + skew, n);
    }
};

void test_pack() {
    const std::vector<std::uint64_t> sizes{1, 4096, 4097, 10000};
    const auto packed = ninfer::pack_expert_extents(sizes, 8192);
    check(packed.extents.size() == 4, "pack: one extent per record");
    check(packed.extents[0].offset == 8192 && packed.extents[1].offset == 8192 + 4096 &&
              packed.extents[2].offset == 8192 + 8192 && packed.extents[3].offset == 8192 + 16384,
          "pack: each extent starts at the previous one's padded end");
    check(packed.extents[2].bytes == 4097, "pack: bytes stay unpadded");
    check(packed.end == 8192 + 4096 + 4096 + 8192 + 12288, "pack: end is the last padded end");
    check(ninfer::pack_expert_extents({}, 0).end == 0, "pack: empty input");

    const std::vector<std::uint64_t> zero{4096, 0};
    check(throws<std::invalid_argument>([&] { (void)ninfer::pack_expert_extents(zero, 0); }), "pack: zero bytes refused");
    check(throws<std::invalid_argument>([&] { (void)ninfer::pack_expert_extents(sizes, 100); }),
          "pack: misaligned first offset refused");
    const std::vector<std::uint64_t> huge{std::numeric_limits<std::uint64_t>::max() / 2};
    check(throws<std::overflow_error>([&] { (void)ninfer::pack_expert_extents(huge, 0); }), "pack: oversize record refused");
    const std::vector<std::uint64_t> two{std::uint64_t{1} << 62, std::uint64_t{1} << 62};
    check(throws<std::overflow_error>([&] { (void)ninfer::pack_expert_extents(two, 0); }),
          "pack: a layout past the largest file offset refused");
}

void test_read(const std::filesystem::path& path) {
    // Three quantisations' worth of record sizes, interleaved like per-layer types.
    std::vector<std::uint64_t> sizes;
    for (std::size_t i = 0; i < 40; ++i) { sizes.push_back(i % 3 == 0 ? 9000 : (i % 3 == 1 ? 4096 : 20001)); }
    const auto packed = ninfer::pack_expert_extents(sizes, 4096); // a header block first
    {
        std::vector<char> file(packed.end, 0);
        for (std::size_t r = 0; r < sizes.size(); ++r) {
            for (std::uint64_t i = 0; i < sizes[r]; ++i) {
                file[packed.extents[r].offset + i] = static_cast<char>(pattern(r, i));
            }
        }
        std::ofstream(path, std::ios::binary).write(file.data(), static_cast<std::streamsize>(file.size()));
    }
    const ninfer::ReadOnlyFile file(path);
    const ExpertExtentReader reader(file, packed.extents);
    check(reader.size() == sizes.size(), "reader: size");

    // A scattered, repeating-free request touching every size.
    std::vector<std::uint32_t> wanted{39, 0, 1, 2, 17, 5, 38, 20, 3, 4, 33};
    for (const std::size_t depth : {std::size_t{1}, std::size_t{2}, std::size_t{8}}) {
        AlignedRing ring(reader.ring_bytes(wanted, depth));
        std::vector<int> seen(wanted.size(), 0);
        bool exact = true;
        reader.read(wanted, ring.bytes, [&](std::size_t position, std::span<const std::byte> record) {
            ++seen[position];
            const std::uint32_t extent = wanted[position];
            if (record.size() != sizes[extent]) { exact = false; return; }
            for (std::uint64_t i = 0; i < record.size(); ++i) {
                if (record[i] != pattern(extent, i)) { exact = false; return; }
            }
        });
        check(exact, "reader: every record's bytes are exact");
        check(std::all_of(seen.begin(), seen.end(), [](int n) { return n == 1; }), "reader: each wanted record consumed once");
    }

    // The same extent twice in one request is read twice and delivered at both positions.
    {
        std::vector<std::uint32_t> twice{4, 4, 7};
        AlignedRing ring(reader.ring_bytes(twice, 2));
        std::vector<int> seen(twice.size(), 0);
        reader.read(twice, ring.bytes, [&](std::size_t position, std::span<const std::byte>) { ++seen[position]; });
        check(seen[0] == 1 && seen[1] == 1 && seen[2] == 1, "reader: a repeated extent is delivered at each position");
    }

    // Nothing wanted is a no-op even with no ring.
    reader.read({}, {}, [&](std::size_t, std::span<const std::byte>) { check(false, "empty request consumed"); });

    // Refusals.
    AlignedRing ring(reader.ring_bytes(wanted, 1));
    const auto noop = [](std::size_t, std::span<const std::byte>) {};
    const std::vector<std::uint32_t> bad{1, 999};
    check(throws<std::out_of_range>([&] { reader.read(bad, ring.bytes, noop); }), "reader: a bad index is refused");
    check(throws<std::invalid_argument>([&] { reader.read(wanted, ring.bytes.subspan(1), noop); }),
          "reader: an unaligned ring is refused");
    check(throws<std::invalid_argument>([&] { reader.read(wanted, ring.bytes.first(4096), noop); }),
          "reader: a ring below one block of the largest extent is refused");
    check(throws<std::out_of_range>([&] {
              (void)reader.ring_bytes(bad, 1);
          }), "reader: ring_bytes refuses a bad index");

    // A consumer's exception propagates.
    check(throws<std::runtime_error>([&] {
              reader.read(wanted, ring.bytes,
                          [](std::size_t, std::span<const std::byte>) { throw std::runtime_error("stop"); });
          }),
          "reader: a consumer's exception propagates");

    // Construction refuses extents the file cannot satisfy.
    std::vector<ExpertExtent> past = packed.extents;
    past.push_back({packed.end, 1});
    check(throws<std::out_of_range>([&] { const ExpertExtentReader r(file, past); }), "reader: an extent past the file is refused");
    // Near-maximum sizes must not wrap to a zero-byte padded size (which would divide by zero in read).
    for (const std::uint64_t huge : {std::numeric_limits<std::uint64_t>::max(),
                                     std::numeric_limits<std::uint64_t>::max() - kExtentAlignment + 2,
                                     std::uint64_t{1} << 63}) {
        std::vector<ExpertExtent> wrap{{0, huge}};
        check(throws<std::out_of_range>([&] { const ExpertExtentReader r(file, wrap); }),
              "reader: a near-maximum extent size is refused, not wrapped");
    }
    std::vector<ExpertExtent> misaligned{{100, 10}};
    check(throws<std::invalid_argument>([&] { const ExpertExtentReader r(file, misaligned); }),
          "reader: a misaligned extent is refused");
    std::vector<ExpertExtent> empty{{4096, 0}};
    check(throws<std::invalid_argument>([&] { const ExpertExtentReader r(file, empty); }), "reader: an empty extent is refused");
}

} // namespace

int main() {
    // In the working directory (the build tree), not a tmpfs that may refuse unbuffered opens.
    const auto path = std::filesystem::current_path() / "ninfer_expert_extents_test.bin";
    test_pack();
    try {
        test_read(path);
    } catch (const std::exception& e) {
        std::cerr << "FAILED: unexpected exception: " << e.what() << '\n';
        ++g_failures;
    }
    std::filesystem::remove(path);
    if (g_failures == 0) { std::cout << "expert_extents: passed\n"; }
    return g_failures == 0 ? 0 : 1;
}
