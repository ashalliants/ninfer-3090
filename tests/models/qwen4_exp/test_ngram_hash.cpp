// Adapted from Infernix a3edb450 tests/models/qwen4_exp/test_ngram_hash.cpp (Apache-2.0).
// Modified for NInfer-3090: also checks the Qwen3.8-Flash-Next GGUF's literal tables (the converter
// proves them equal to the derivation; this pins the C++ hash to the same numbers).
//
// PLE n-gram row ids against the fixture written by Infernix's tools/flash_next/ngram.py
// (tests/fixtures/qwen4_exp/ngram_rows.txt, Infernix a3edb450): multipliers, head primes, padded
// table size and every row, exactly.

#include "models/qwen4_exp/frontend/ngram_hash.h"

#include <cinttypes>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#ifndef NINFER_SOURCE_DIR
#    define NINFER_SOURCE_DIR "."
#endif

using namespace ninfer::models::qwen4_exp;

namespace {

// GGUF qwen4exp.ple.* of Qwen3.8-Flash-Next GSQ-RCO IQ2_XS (vocabulary 248320, seed 1234, base
// 20000000, padding 128): the literal multipliers, head sizes and offsets the file stores.
int check_flash_next() {
    const NgramConfig c{.vocab_size                         = 248320,
                        .eos_token_id                       = 248044,
                        .ngram_size                         = 3,
                        .heads_per_ngram                    = 8,
                        .ngram_vocab_size_base              = 20000000,
                        .make_ngram_vocab_size_divisible_by = 128,
                        .seed                               = 1234};
    const NgramHash hash(c, 0);
    const std::vector<std::uint64_t> multipliers{23703573157769ULL, 20109073645365ULL, 8052911324071ULL};
    const std::vector<std::uint64_t> sizes{20000003, 20000023, 20000033, 20000047, 20000059, 20000063,
                                           20000069, 20000077, 20000081, 20000093, 20000107, 20000147,
                                           20000153, 20000159, 20000161, 20000171};
    const std::vector<std::uint64_t> offsets{0,         20000003,  40000026,  60000059,  80000106,  100000165,
                                             120000228, 140000297, 160000374, 180000455, 200000548, 220000655,
                                             240000802, 260000955, 280001114, 300001275};
    int failures = 0;
    failures += std::vector<std::uint64_t>(hash.multipliers().begin(), hash.multipliers().end()) != multipliers;
    failures += std::vector<std::uint64_t>(hash.sizes().begin(), hash.sizes().end()) != sizes;
    failures += std::vector<std::uint64_t>(hash.offsets().begin(), hash.offsets().end()) != offsets;
    failures += hash.table_rows() != 320001536ULL;
    if (failures != 0) { std::fprintf(stderr, "FAIL: the hash differs from the Flash-Next GGUF's tables\n"); }
    return failures;
}

} // namespace

int main() {
    std::ifstream in(std::string(NINFER_SOURCE_DIR) + "/tests/fixtures/qwen4_exp/ngram_rows.txt");
    if (!in) {
        std::fprintf(stderr, "FAIL: fixture unreadable\n");
        return 1;
    }
    std::string line, tag;
    std::getline(in, line);
    NgramConfig c;
    std::sscanf(line.c_str(), "# vocab %u eos %d n %u heads %u base %" SCNu64 " div %" SCNu64 " seed %" SCNu64,
                &c.vocab_size, &c.eos_token_id, &c.ngram_size, &c.heads_per_ngram, &c.ngram_vocab_size_base,
                &c.make_ngram_vocab_size_divisible_by, &c.seed);
    std::getline(in, line);
    std::istringstream hs(line);
    hs >> tag;
    std::vector<std::int32_t> history;
    for (std::int32_t t; hs >> t;) { history.push_back(t); }
    long failures = 0, rows_checked = 0;
    std::vector<NgramHash> layers{NgramHash(c, 0), NgramHash(c, 1)};
    std::vector<std::vector<std::uint32_t>> rows(2);
    const std::size_t count = history.size() - (c.ngram_size - 1);
    for (int l = 0; l < 2; ++l) {
        rows[l].resize(count * layers[l].heads());
        layers[l].row_ids(history, count, rows[l].data());
    }
    while (std::getline(in, line)) {
        std::istringstream s(line);
        s >> tag;
        int layer = 0;
        s >> layer;
        const NgramHash& h = layers[static_cast<std::size_t>(layer)];
        if (tag == "L") {
            std::string kind;
            s >> kind;
            std::vector<std::uint64_t> v;
            std::string w;
            std::uint64_t padded = 0;
            while (s >> w) {
                if (w == "P") {
                    s >> padded;
                    break;
                }
                v.push_back(std::stoull(w));
            }
            const auto got = kind == "M" ? h.multipliers() : h.sizes();
            failures += !(std::vector<std::uint64_t>(got.begin(), got.end()) == v);
            if (kind == "S") { failures += h.table_rows() != padded; }
        } else if (tag == "R") {
            std::size_t p = 0;
            s >> p;
            for (std::uint32_t j = 0; j < h.heads(); ++j) {
                std::uint32_t want = 0;
                s >> want;
                failures += rows[static_cast<std::size_t>(layer)][p * h.heads() + j] != want;
            }
            ++rows_checked;
        }
    }
    std::printf("n-gram rows: %ld positions checked, %ld mismatches\n", rows_checked, failures);
    failures += check_flash_next();
    if (failures != 0 || rows_checked < 700) {
        std::fprintf(stderr, "FAIL: n-gram row ids differ from the reference\n");
        return 1;
    }
    std::printf("all qwen4_exp n-gram checks passed\n");
    return 0;
}
