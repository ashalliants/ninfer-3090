// Adapted from Infernix a3edb450 tests/models/qwen4_exp/test_ngram_volume.cpp (Apache-2.0).
// Modified for NInfer-3090: version 2 volumes of 90-byte IQ4_NL-sized rows (45 per block, the
// Flash-Next geometry); refusals of a version 1 header and another row format; the keep-alive.
//
// NgramVolume::read_rows on a synthetic volume. The oracle is the closed-form row contents f(r, k),
// never the volume: random rows, duplicates and rows sharing a block read each distinct block once
// (counters), more than 64 distinct blocks wrap the ring, a second call hits the host cache, two
// rows of one cache slot both return their own bytes, a row past the table throws, and a volume of
// the wrong size, version or row format is refused at open. A call of thousands of distinct blocks
// (a prefill chunk) takes the multi-threaded route: the same oracle, counters and cache.
//
// ngram_window_rows (program/ngram_rows.h) against ids derived by hand: the EOS-padded window of
// each column laid out here and handed to NgramHash::row_ids directly. And the prefetch
// --ngram-prefetch does: the anchor column's rows read ahead are cached when verification reads
// the whole window.

#include "models/qwen4_exp/frontend/ngram_hash.h"
#include "models/qwen4_exp/program/ngram_rows.h"
#include "models/qwen4_exp/program/ngram_volume.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace q4 = ninfer::models::qwen4_exp;

namespace {

constexpr std::uint32_t kRowBytes     = 90; // 160 IQ4_NL values
constexpr std::uint32_t kBlockBytes   = 4096;
constexpr std::uint32_t kRowsPerBlock = kBlockBytes / kRowBytes;
constexpr std::uint64_t kRows         = 1ULL << 21; // twice the cache's slots: slot collisions exist

std::uint8_t f(std::uint64_t row, std::size_t k) { return static_cast<std::uint8_t>((row * 29U + k * 13U + 7U) & 0xFFU); }

int failures = 0;

void check(bool ok, const std::string& what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what.c_str());
        ++failures;
    }
}

template <class T>
void put(std::vector<std::byte>& header, std::size_t at, T value) {
    std::memcpy(header.data() + at, &value, sizeof(T));
}

std::vector<std::byte> header_of(const q4::NgramTableConfig& table, std::uint32_t version, const char* format) {
    std::vector<std::byte> header(4096);
    std::memcpy(header.data(), "NINFERNG", 8);
    put<std::uint32_t>(header, 8, version);
    put<std::uint32_t>(header, 12, table.header_bytes);
    put<std::uint64_t>(header, 16, table.rows);
    put<std::uint32_t>(header, 24, table.row_bytes);
    put<std::uint32_t>(header, 28, table.rows_per_block);
    put<std::uint32_t>(header, 32, table.block_bytes);
    put<std::uint64_t>(header, 36, table.blocks);
    std::memcpy(header.data() + 44, table.volume_id.data(), 16);
    std::memcpy(header.data() + 60, format, std::strlen(format));
    return header;
}

q4::NgramTableConfig make_volume(const std::filesystem::path& path) {
    q4::NgramTableConfig table;
    table.format         = ninfer::QType::GGML_IQ4_NL;
    table.rows           = kRows;
    table.row_bytes      = kRowBytes;
    table.rows_per_block = kRowsPerBlock;
    table.block_bytes    = kBlockBytes;
    table.header_bytes   = 4096;
    table.blocks         = (kRows + kRowsPerBlock - 1) / kRowsPerBlock;
    table.file_bytes     = table.header_bytes + table.blocks * kBlockBytes;
    for (std::size_t i = 0; i < table.volume_id.size(); ++i) { table.volume_id[i] = static_cast<std::uint8_t>(i * 17); }
    const auto header = header_of(table, 2, "ggml_iq4_nl");
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    out.write(reinterpret_cast<const char*>(header.data()), static_cast<std::streamsize>(header.size()));
    std::vector<std::uint8_t> block(kBlockBytes);
    for (std::uint64_t b = 0; b < table.blocks; ++b) {
        std::fill(block.begin(), block.end(), std::uint8_t{0});
        for (std::uint32_t r = 0; r < kRowsPerBlock && b * kRowsPerBlock + r < kRows; ++r) {
            for (std::uint32_t k = 0; k < kRowBytes; ++k) { block[r * kRowBytes + k] = f(b * kRowsPerBlock + r, k); }
        }
        out.write(reinterpret_cast<const char*>(block.data()), static_cast<std::streamsize>(block.size()));
    }
    return table;
}

bool rows_match(std::span<const std::uint32_t> rows, const std::vector<std::byte>& out) {
    for (std::size_t i = 0; i < rows.size(); ++i) {
        for (std::uint32_t k = 0; k < kRowBytes; ++k) {
            if (static_cast<std::uint8_t>(out[i * kRowBytes + k]) != f(rows[i], k)) { return false; }
        }
    }
    return true;
}

std::vector<std::byte> read(const q4::NgramVolume& volume, const std::vector<std::uint32_t>& rows) {
    std::vector<std::byte> out(rows.size() * kRowBytes);
    volume.read_rows(rows, out);
    return out;
}

std::uint64_t distinct_blocks(const std::vector<std::uint32_t>& rows) {
    std::set<std::uint32_t> blocks;
    for (const auto r : rows) { blocks.insert(r / kRowsPerBlock); }
    return blocks.size();
}

// A copy of the volume with a different header, refused at open.
bool refused(const std::filesystem::path& path, const q4::NgramTableConfig& table, std::uint32_t version,
             const char* format) {
    const auto copy = std::filesystem::path(path.string() + ".bad");
    std::filesystem::copy_file(path, copy, std::filesystem::copy_options::overwrite_existing);
    {
        const auto header = header_of(table, version, format);
        std::fstream out(copy, std::ios::binary | std::ios::in | std::ios::out);
        out.write(reinterpret_cast<const char*>(header.data()), static_cast<std::streamsize>(header.size()));
    }
    bool threw = false;
    try {
        const q4::NgramVolume bad(copy, table);
    } catch (const std::runtime_error&) { threw = true; }
    std::error_code ignored;
    std::filesystem::remove(copy, ignored);
    return threw;
}

// A small hash: EOS 7, 3-grams, two heads per n-gram (four heads), every row id below kRows.
q4::NgramHash make_hash() {
    q4::NgramConfig config;
    config.vocab_size                         = 1000;
    config.eos_token_id                       = 7;
    config.ngram_size                         = 3;
    config.heads_per_ngram                    = 2;
    config.ngram_vocab_size_base              = 100000;
    config.make_ngram_vocab_size_divisible_by = 1;
    return q4::NgramHash(config, 0);
}

// The ids of the last `count` tokens of `padded`, which holds the two tokens before them first.
std::vector<std::uint32_t> ids_of(const q4::NgramHash& hash, const std::vector<std::int32_t>& padded, std::size_t count) {
    std::vector<std::uint32_t> ids(count * hash.heads());
    hash.row_ids(padded, count, ids.data());
    return ids;
}

void check_window_rows() {
    const q4::NgramHash hash = make_hash();
    const std::vector<std::int32_t> history{10, 20, 30, 40, 50, 7, 60, 70};
    const std::vector<std::int32_t> none;
    using Span = std::span<const std::int32_t>;
    const auto rows = [&](Span h, Span d, std::int64_t begin, int width) { return q4::ngram_window_rows(hash, h, d, begin, width); };

    // The sequence start: the missing context is EOS (the window of position 0 is {EOS, EOS, 10}).
    check(rows(history, none, 0, 2) == ids_of(hash, {7, 7, 10, 20}, 2), "window rows at position 0 are EOS-padded");
    check(rows(history, none, 1, 1) == ids_of(hash, {7, 10, 20}, 1), "window rows at position 1 pad one EOS");
    check(rows(history, none, 1, 3) == ids_of(hash, {7, 10, 20, 30, 40}, 3), "a window from position 1 pads one EOS");
    // Mid-sequence.
    check(rows(history, none, 2, 3) == ids_of(hash, {10, 20, 30, 40, 50}, 3), "window rows mid-sequence");
    check(rows(history, none, 6, 2) == ids_of(hash, {50, 7, 60, 70}, 2), "window rows across an EOS in the history");
    check(rows(history, none, 7, 1) == ids_of(hash, {7, 60, 70}, 1), "window rows of the last token");
    // The drafts continue the history: any split of one sequence gives the same rows.
    const std::vector<std::int32_t> head3{10, 20, 30}, drafts{40, 50};
    check(rows(head3, drafts, 2, 3) == ids_of(hash, {10, 20, 30, 40, 50}, 3), "drafts continue the history");
    check(rows(head3, drafts, 2, 3) == rows(std::vector<std::int32_t>{10, 20, 30, 40, 50}, none, 2, 3),
          "a history split from its drafts gives the whole sequence's rows");
    check(rows(head3, drafts, 4, 1) == ids_of(hash, {30, 40, 50}, 1), "a column wholly in the drafts");
    // The anchor column's rows are the first column's: the prefix of the whole window.
    const auto whole  = rows(head3, drafts, 2, 3);
    const auto anchor = rows(head3, none, 2, 1);
    check(anchor.size() == hash.heads() && std::equal(anchor.begin(), anchor.end(), whole.begin()),
          "the anchor column's rows are the window's first heads");
    check(whole.size() == 3U * hash.heads(), "a window has width * heads rows");
    bool threw = false;
    // The output-span form (the Program's per-token path) writes the same ids, whatever its kept
    // scratch held before, and refuses an output of the wrong size.
    std::vector<std::int32_t> window{99, 98, 97, 96, 95, 94, 93};
    std::vector<std::uint32_t> out(3U * hash.heads(), 0xFFFFFFFFU);
    q4::ngram_window_rows(hash, head3, drafts, 2, 3, window, out);
    check(out == whole, "the output-span form equals the vector form");
    std::vector<std::uint32_t> one(hash.heads());
    q4::ngram_window_rows(hash, history, none, 0, 1, window, one);
    check(one == rows(history, none, 0, 1), "a reused scratch window starts afresh (EOS padding)");
    threw = false;
    try {
        q4::ngram_window_rows(hash, history, none, 0, 2, window, one);
    } catch (const std::invalid_argument&) { threw = true; }
    check(threw, "an output of the wrong size throws");
    // Out of the sequence.
    threw = false;
    try { (void)rows(head3, drafts, 3, 3); } catch (const std::invalid_argument&) { threw = true; }
    check(threw, "columns past the sequence throw");
    threw = false;
    try { (void)rows(head3, drafts, -1, 1); } catch (const std::invalid_argument&) { threw = true; }
    check(threw, "a negative begin throws");
}

} // namespace

int main() {
    check_window_rows();
    const auto path = std::filesystem::temp_directory_path() / "ninfer_ngram_volume_test.ngram";
    try {
        const q4::NgramTableConfig table = make_volume(path);
        {
            q4::NgramVolume volume(path, table);
            std::mt19937 rng(20261004);

            // Random rows (cold): one read per distinct block.
            std::vector<std::uint32_t> rows(500);
            for (auto& r : rows) { r = static_cast<std::uint32_t>(rng() % kRows); }
            auto before = volume.counters();
            check(rows_match(rows, read(volume, rows)), "random rows equal the oracle");
            check(volume.counters().reads - before.reads == distinct_blocks(rows), "one read per distinct block");

            // Duplicates and rows sharing a block, in one call: one read per distinct block.
            const std::uint32_t base = 1'000'000 / kRowsPerBlock * kRowsPerBlock; // a block's first row
            std::vector<std::uint32_t> shared{base, base + 1, base + 5, base, base + 1, base + kRowsPerBlock * 3};
            before = volume.counters();
            check(rows_match(shared, read(volume, shared)), "duplicates and shared blocks equal the oracle");
            check(volume.counters().reads - before.reads == 2, "two distinct blocks, two reads");

            // More than 64 distinct blocks: the ring wraps.
            std::vector<std::uint32_t> many;
            for (std::uint32_t b = 0; b < 200; ++b) { many.push_back((b * 37 + 11) * kRowsPerBlock + b % kRowsPerBlock); }
            before = volume.counters();
            check(rows_match(many, read(volume, many)), "200 distinct blocks equal the oracle");
            check(volume.counters().reads - before.reads == 200, "200 reads through a 64-block ring");

            // The same rows again: host-cache hits, no reads.
            before = volume.counters();
            check(rows_match(many, read(volume, many)), "cached rows equal the oracle");
            check(volume.counters().reads == before.reads && volume.counters().hits - before.hits == many.size(),
                  "a second call hits the cache");
            check(volume.cached(many), "cached() reports rows already read");
            check(!volume.cached(std::vector<std::uint32_t>{static_cast<std::uint32_t>(kRows)}),
                  "cached() rejects a row past the table");

            // Two rows of one cache slot (the slot is a hash of the row id; brute-force a collision).
            const auto slot_of = [](std::uint32_t row) { return (row * 2654435761U) >> 12U; };
            std::uint32_t b = 0;
            for (std::uint32_t r = 1; r < kRows && b == 0; ++r) {
                if (slot_of(r) == slot_of(0)) { b = r; }
            }
            check(b != 0, "the test found two rows sharing a cache slot");
            const std::vector<std::uint32_t> pair{0, b, 0, b};
            check(rows_match(pair, read(volume, pair)), "rows sharing a cache slot return their own bytes");
            check(rows_match(pair, read(volume, pair)), "and again after evicting each other");

            // A prefill-sized call: the multi-threaded route, then every row from the cache.
            std::vector<std::uint32_t> large(20000);
            for (auto& r : large) { r = static_cast<std::uint32_t>(rng() % kRows); }
            for (std::size_t i = 0; i < 2000; ++i) { large[rng() % large.size()] = large[rng() % large.size()]; }
            const std::uint64_t large_blocks = [&] {
                std::set<std::uint32_t> missing;
                for (const auto r : large) {
                    if (!volume.cached(std::vector<std::uint32_t>{r})) { missing.insert(r / kRowsPerBlock); }
                }
                return static_cast<std::uint64_t>(missing.size());
            }();
            check(large_blocks >= 4000, "the large call has thousands of uncached blocks");
            before = volume.counters();
            check(rows_match(large, read(volume, large)), "a prefill-sized call equals the oracle");
            check(volume.counters().reads - before.reads == large_blocks, "one read per distinct uncached block");
            check(rows_match(large, read(volume, large)), "the large call's rows again equal the oracle");

            bool threw = false;
            try {
                (void)read(volume, {static_cast<std::uint32_t>(kRows)});
            } catch (const std::out_of_range&) { threw = true; }
            check(threw, "a row past the table throws out_of_range");

            // Keep-alive: while rows were asked for within the window, one block read per period of
            // volume idleness; none once the window has passed; reads still equal the oracle.
            volume.set_keepalive(std::chrono::milliseconds(20), std::chrono::milliseconds(400));
            (void)read(volume, rows);
            const auto start = volume.counters();
            std::this_thread::sleep_for(std::chrono::milliseconds(250));
            const auto during = volume.counters().keepalive_reads - start.keepalive_reads;
            check(during >= 4 && during <= 14, "keep-alive reads every 20 ms of idleness (" + std::to_string(during) + ")");
            check(volume.counters().reads == start.reads, "keep-alive reads are not counted as row reads");
            std::this_thread::sleep_for(std::chrono::milliseconds(400)); // past the window
            const auto settled = volume.counters().keepalive_reads;
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
            check(volume.counters().keepalive_reads == settled, "no keep-alive reads after the window");
            check(rows_match(rows, read(volume, rows)), "rows equal the oracle with the keep-alive on");
            volume.set_keepalive(std::chrono::milliseconds(0), std::chrono::milliseconds(0));

            // The default period, 40 ms (Basalt's, against an SSD that sleeps after about 100 ms
            // idle). A loaded host delays the keep-alive thread by tens of milliseconds, so the
            // upper bound is the best of five idle spells: at least one read within three periods;
            // and in every spell none before half the period has passed.
            volume.set_keepalive(std::chrono::milliseconds(40), std::chrono::milliseconds(2000));
            using Clock = std::chrono::steady_clock;
            auto best = Clock::duration::max();
            bool early = false;
            for (int attempt = 0; attempt < 5; ++attempt) {
                (void)read(volume, rows);
                const auto idle_from  = Clock::now();
                const auto idle_start = volume.counters().keepalive_reads;
                std::this_thread::sleep_for(std::chrono::milliseconds(15));
                early = early || volume.counters().keepalive_reads != idle_start;
                while (Clock::now() - idle_from < std::chrono::milliseconds(500)) {
                    if (volume.counters().keepalive_reads > idle_start) {
                        best = std::min(best, Clock::now() - idle_from);
                        break;
                    }
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
                }
            }
            check(!early, "no 40 ms keep-alive read before half the period");
            check(best <= std::chrono::milliseconds(120),
                  "a 40 ms keep-alive reads within three periods of idleness (best " +
                      std::to_string(std::chrono::duration_cast<std::chrono::milliseconds>(best).count()) + " ms)");
            volume.set_keepalive(std::chrono::milliseconds(0), std::chrono::milliseconds(0));
        }

        {
            // The prefetch: the anchor column's rows read while the draft graph runs are cached
            // when verification reads the whole window (anchor and drafts).
            q4::NgramVolume volume(path, table);
            const q4::NgramHash hash = make_hash();
            const std::vector<std::int32_t> history{10, 20, 30, 40, 50}; // the anchor is the last token
            const std::int64_t anchor_at = 4;
            const auto anchor            = q4::ngram_window_rows(hash, history, {}, anchor_at, 1);
            check(anchor.size() == hash.heads(), "the anchor column has one row per head");
            check(!volume.cached(anchor), "the anchor's rows are not cached before the prefetch");
            auto before = volume.counters();
            check(rows_match(anchor, read(volume, anchor)), "the prefetched anchor rows equal the oracle");
            check(volume.cached(anchor), "the anchor's rows are cached after the prefetch");
            check(volume.counters().reads > before.reads, "the prefetch reads the volume");

            // Verification: the anchor and three drafts, as upload_rows reads them.
            const std::vector<std::int32_t> drafts{61, 62, 63};
            const auto window = q4::ngram_window_rows(hash, history, drafts, anchor_at, 4);
            check(std::equal(anchor.begin(), anchor.end(), window.begin()), "verification reads the prefetched rows first");
            before = volume.counters();
            check(rows_match(window, read(volume, window)), "the verification window equals the oracle");
            const auto after = volume.counters();
            check(after.hits - before.hits >= anchor.size(), "the anchor's rows are served from the cache");
            check(after.rows - before.rows == window.size(), "verification asks for every row of the window");
        }

        q4::NgramTableConfig wrong = table;
        wrong.file_bytes += 4096;
        bool threw = false;
        try {
            const q4::NgramVolume bad(path, wrong);
        } catch (const std::runtime_error&) { threw = true; }
        check(threw, "a volume of the wrong size is refused at open");
        check(refused(path, table, 1, "ggml_iq4_nl"), "a version 1 (Infernix FP8) volume is refused");
        check(refused(path, table, 2, "ggml_q8_0"), "a volume of another row format is refused");
        q4::NgramTableConfig other = table;
        other.volume_id[3] ^= 1;
        threw = false;
        try {
            const q4::NgramVolume bad(path, other);
        } catch (const std::runtime_error&) { threw = true; }
        check(threw, "a volume of another id is refused");
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL: %s\n", error.what());
        ++failures;
    }
    std::error_code ignored;
    std::filesystem::remove(path, ignored);
    std::printf(failures == 0 ? "ngram volume checks passed\n" : "FAIL: %d checks failed\n", failures);
    return failures == 0 ? 0 : 1;
}
