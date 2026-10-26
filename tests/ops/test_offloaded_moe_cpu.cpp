// The CPU routes of offloaded_sparse_moe (host only):
//   - the canonical A8 cast against an independent FP64 formulation, bit for bit, on random,
//     tie, subnormal, zero, overflow and NaN groups;
//   - the exact sub-block decode of every expert block format (gate/up IQ2_S, IQ2_XXS, IQ2_XS, IQ1_M,
//     IQ3_XXS, IQ3_S, IQ4_XS; down Q2_0, IQ4_NL): c * m * w equals ggml's reference value
//     (tests/ops/ggml_blocks_decode.h) bit for bit;
//   - expert_forward_cpu (scalar) against the FP64 expert oracle over the decoded record and the
//     A8-cast activations, for each record format at T in {1, 2, 7, 8};
//   - AVX2 = scalar, and the worker team at 1, 2, 3, 8 and 16 workers = scalar, bit for bit;
//   - extreme_magnitude_records: the largest |P_s| of the canonical arithmetic (IQ4_XS m = -32 and
//     |w| = 127, IQ3_XXS grid 62, IQ4_NL |w| = 127, all against codes +-127) is exact on both ISAs,
//     and AVX2's byte-pair sums do not saturate;
//   - both entry points reject a job with a null record or a null active x/y column; the team's
//     optional progress word advances during a round;
//   - coresident_blocks_values: the register gate/up kernel's co-resident grid leaves an eighth of
//     the SMs to the side streams (offloaded_moe::coresident_blocks).

#include "ninfer/ops/offloaded_sparse_moe.h"
#include "offloaded_moe_fixtures.h"
#include "ops/common/canonical_ggml.h"
#include "ops/offloaded_sparse_moe/coresident.h"
#include "ops/offloaded_sparse_moe/cpu/expert_cpu.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <cstdint>
#include <iterator>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>

namespace {

using namespace ninfer;
using namespace ninfer::test::moe;
namespace canon  = ninfer::ops::canon;
namespace detail = ninfer::ops::offloaded_moe::detail;

void test_a8_cast() {
    std::mt19937 rng(7);
    std::normal_distribution<float> n(0.0F, 1.0F);
    std::uniform_int_distribution<int> pick(0, 9);
    int groups = 0;
    for (int trial = 0; trial < 20000; ++trial) {
        float v[32];
        const int kind = trial < 16 ? trial % 8 : pick(rng);
        for (float& x : v) { x = n(rng); }
        switch (kind) {
        case 0: for (float& x : v) { x = 0.0F; } break;                           // a = 0
        case 1: for (float& x : v) { x *= 1e-38F; } break;                        // tiny a: 127 / a would overflow without the 2^100 prescale
        case 2: for (int j = 0; j < 32; ++j) { v[j] = static_cast<float>(j - 16) * 0.5F; } break; // ties
        case 3: v[5] = std::numeric_limits<float>::quiet_NaN(); break;            // NaN ignored, code 0
        case 4: for (float& x : v) { x *= 3e4F; } break;                          // large
        case 5: v[0] = -v[1]; v[2] = 127.0F; break;                               // exact 127
        default: break;
        }
        // BF16 inputs (the x route) and FP32 inputs (the h route).
        std::uint16_t b[32];
        float bv[32];
        for (int j = 0; j < 32; ++j) {
            b[j]  = bf16_bits(v[j]);
            bv[j] = bf16_value(b[j]);
            if (std::isnan(v[j])) { b[j] = 0x7FC0; bv[j] = std::numeric_limits<float>::quiet_NaN(); }
        }
        std::int8_t q1[32], q2[32], r1[32], r2[32];
        const float d1 = canon::a8_quantize_bf16(b, q1);
        const float e1 = a8_cast(bv, 32, r1);
        const float d2 = canon::a8_quantize_f32(v, q2);
        const float e2 = a8_cast(v, 32, r2);
        check(std::bit_cast<std::uint32_t>(d1) == std::bit_cast<std::uint32_t>(e1) && std::memcmp(q1, r1, 32) == 0,
              "BF16 A8 cast differs from the independent formulation");
        check(std::bit_cast<std::uint32_t>(d2) == std::bit_cast<std::uint32_t>(e2) && std::memcmp(q2, r2, 32) == 0,
              "FP32 A8 cast differs from the independent formulation");
        ++groups;
    }
    std::printf("A8 cast: %d groups equal to the FP64 formulation\n", groups);
}

void test_decode() {
    std::mt19937 rng(11);
    for (QType format : {QType::GGML_IQ2_S, QType::GGML_IQ2_XXS, QType::GGML_IQ1_M, QType::GGML_Q2_0,
                         QType::GGML_IQ2_XS, QType::GGML_IQ3_XXS, QType::GGML_IQ3_S, QType::GGML_IQ4_XS,
                         QType::GGML_IQ4_NL}) {
        const auto block   = ggml_block(format);
        const int blocks   = 4096;
        std::vector<std::uint8_t> bytes(static_cast<std::size_t>(blocks) * block.bytes);
        fill_blocks(format, bytes.data(), blocks, 1.0F, rng);
        // A second batch with extreme scales: the smallest and largest normal fp16.
        std::vector<std::uint8_t> extreme(bytes.begin(), bytes.begin() + 64 * block.bytes);
        fill_blocks(format, extreme.data(), 32, 1.3e-4F, rng);
        fill_blocks(format, extreme.data() + 32 * block.bytes, 32, 3.0e4F, rng);
        for (const auto* set : {&bytes, &extreme}) {
            const std::size_t n = set->size() / block.bytes;
            const auto ref = ninfer::test::ggml::decode_blocks(
                format, std::span(reinterpret_cast<const std::byte*>(set->data()), set->size()));
            const int subs = static_cast<int>(n * block.values / 32);
            for (int s = 0; s < subs; ++s) {
                const auto sub = detail::decode_sub(format, set->data(), s);
                for (int j = 0; j < 32; ++j) {
                    const float value = sub.c * static_cast<float>(sub.m[j / 16]) * static_cast<float>(sub.w[j]);
                    const float want  = ref[static_cast<std::size_t>(s) * 32 + j];
                    check(std::bit_cast<std::uint32_t>(value) == std::bit_cast<std::uint32_t>(want) ||
                              (value == 0.0F && want == 0.0F),
                          "expert sub-block decode differs from ggml's reference value");
                }
            }
        }
        std::printf("decode %d: c * m * w equals the reference on %d blocks\n", static_cast<int>(format), blocks + 64);
    }
}

std::vector<std::vector<std::uint16_t>> run_forward(ops::offloaded_moe::CpuIsa isa, QType format,
                                                    const std::uint8_t* record, const std::vector<std::uint16_t>& x,
                                                    int ncols) {
    std::vector<std::vector<std::uint16_t>> y(ncols, std::vector<std::uint16_t>(om::kHidden));
    const std::uint16_t* xp[om::kMaxColumns];
    std::uint16_t* yp[om::kMaxColumns];
    for (int c = 0; c < ncols; ++c) {
        xp[c] = x.data() + static_cast<std::size_t>(c) * om::kHidden;
        yp[c] = y[c].data();
    }
    ops::offloaded_moe::expert_forward_cpu(isa, format, record, ncols, xp, yp);
    return y;
}

void test_oracle_and_isa() {
    using ops::offloaded_moe::CpuIsa;
    const bool avx2 = ops::offloaded_moe::cpu_isa_supported(CpuIsa::kAvx2);
    std::printf("AVX2 route %s\n", avx2 ? "available" : "NOT available on this CPU: AVX2 = scalar not checked");
    std::mt19937 rng(23);
    for (QType format : kRecordFormats) {
        const auto record  = random_record(format, rng);
        const auto decoded = decode_record(format, record.data());
        for (int t : {1, 2, 7, 8}) {
            const auto x      = random_x(t, rng);
            const auto scalar = run_forward(CpuIsa::kScalar, format, record.data(), x, t);
            double worst_norm = 0.0, worst_point = 0.0;
            for (int c = 0; c < t; ++c) {
                const auto want = expert_oracle(decoded, x.data() + static_cast<std::size_t>(c) * om::kHidden);
                const auto crit = measure(scalar[c], want);
                worst_norm      = std::max(worst_norm, crit.normwise);
                worst_point     = std::max(worst_point, crit.worst);
                check(accept(crit), std::string("scalar expert outside the A8 criterion: ") + format_name(format));
            }
            std::printf("%-14s T=%d: scalar vs FP64 normwise %.3e, worst %.3e\n", format_name(format), t, worst_norm,
                        worst_point);
            if (avx2) {
                const auto vec = run_forward(CpuIsa::kAvx2, format, record.data(), x, t);
                check(vec == scalar, std::string("AVX2 differs from scalar: ") + format_name(format));
            }
        }
        // Every column count with AVX2, against scalar (each count is its own AVX2 instance).
        if (avx2) {
            const auto x = random_x(om::kMaxColumns, rng);
            for (int t = 1; t <= om::kMaxColumns; ++t) {
                check(run_forward(CpuIsa::kAvx2, format, record.data(), x, t) ==
                          run_forward(CpuIsa::kScalar, format, record.data(), x, t),
                      "AVX2 differs from scalar at some column count");
            }
            std::printf("%-14s AVX2 = scalar at T = 1..8\n", format_name(format));
        }
    }
}

void test_team() {
    using ops::offloaded_moe::CpuIsa;
    std::mt19937 rng(31);
    // One job per format plus one sharing the first job's record, column counts 1..8 in turn.
    std::vector<std::vector<std::uint8_t>> records;
    std::vector<QType> formats;
    for (int j = 0; j < kRecordFormatCount; ++j) {
        formats.push_back(kRecordFormats[j]);
        records.push_back(random_record(formats.back(), rng));
    }
    formats.push_back(formats[0]);
    records.push_back(records[0]);
    const auto x = random_x(om::kMaxColumns, rng);
    std::vector<std::vector<std::vector<std::uint16_t>>> want;
    for (std::size_t j = 0; j < records.size(); ++j) {
        want.push_back(run_forward(CpuIsa::kScalar, formats[j], records[j].data(), x, static_cast<int>(j % 8) + 1));
    }
    for (CpuIsa isa : {CpuIsa::kScalar, CpuIsa::kAvx2}) {
        if (!ops::offloaded_moe::cpu_isa_supported(isa)) { continue; }
        for (int workers : {1, 2, 3, 8, 16}) {
            ops::offloaded_moe::CpuExpertTeam team(
                {.workers = workers, .isa = isa, .max_jobs = static_cast<int>(records.size())});
            std::vector<std::vector<std::vector<std::uint16_t>>> got(records.size());
            std::vector<ops::offloaded_moe::CpuExpertJob> jobs(records.size());
            for (std::size_t j = 0; j < records.size(); ++j) {
                auto& job  = jobs[j];
                job.record = records[j].data();
                job.format = formats[j];
                job.ncols  = static_cast<int>(j % 8) + 1;
                got[j].assign(job.ncols, std::vector<std::uint16_t>(om::kHidden, 0xFFFF));
                for (int c = 0; c < job.ncols; ++c) {
                    job.x[c] = x.data() + static_cast<std::size_t>(c) * om::kHidden;
                    job.y[c] = got[j][c].data();
                }
            }
            for (int round = 0; round < 2; ++round) { team.run(jobs); } // a second round reuses the team
            for (std::size_t j = 0; j < records.size(); ++j) {
                check(got[j] == want[j], "the worker team differs from expert_forward_cpu");
            }
        }
        std::printf("team (%s): 1, 2, 3, 8, 16 workers = scalar on %zu jobs\n", ops::offloaded_moe::cpu_isa_name(isa),
                    records.size());
    }
}

// One row of `blocks` blocks of `format` with fp16 d = 1 whose every sub-block is at the largest |P_s|
// the format can reach against codes +-127, with a few variants per row:
//   - IQ4_XS: ls = 0 (m = -32) or ls = 63 (m = 31), codes 0 (w = -127) or 15 (w = 113);
//   - IQ3_XXS: ls = 15 (m = 31), the grid entry of the largest magnitudes (it holds 62s), random signs;
//   - IQ4_NL: codes 0 or 15.
std::vector<std::uint8_t> extreme_row(QType format, int blocks, std::mt19937& rng) {
    const auto block = ggml_block(format);
    std::vector<std::uint8_t> row(static_cast<std::size_t>(blocks) * block.bytes, 0);
    std::uniform_int_distribution<int> bit(0, 1);
    std::uniform_int_distribution<int> byte(0, 255);
    // IQ3_XXS: the grid entry of the largest magnitude sum (no entry holds four 62s; this one holds
    // 62s, the largest value).
    int grid62 = 0, best = 0;
    for (int i = 0; i < 256; ++i) {
        const std::uint32_t g = ninfer::ggml_tables::iq3xxs_grid[i];
        const int sum = static_cast<int>((g & 0xFF) + ((g >> 8) & 0xFF) + ((g >> 16) & 0xFF) + (g >> 24));
        if (sum > best) {
            best   = sum;
            grid62 = i;
        }
    }
    const std::uint32_t g62 = ninfer::ggml_tables::iq3xxs_grid[grid62];
    check((g62 & 0xFF) == 62 || ((g62 >> 8) & 0xFF) == 62 || ((g62 >> 16) & 0xFF) == 62 || (g62 >> 24) == 62,
          "the largest iq3xxs_grid entry holds a 62");
    for (int b = 0; b < blocks; ++b) {
        std::uint8_t* p = row.data() + static_cast<std::size_t>(b) * block.bytes;
        put_u16(p, 0x3C00); // d = 1
        switch (format) {
        case QType::GGML_IQ4_XS: {
            // ls = 0 everywhere in the first block (every m = -32), mixed 0 / 63 after it.
            const bool mixed = b > 0;
            std::uint16_t high = 0;
            for (int ib = 0; ib < 8; ++ib) {
                const int ls = mixed && bit(rng) != 0 ? 63 : 0;
                p[4 + ib / 2] |= static_cast<std::uint8_t>((ls & 15) << (4 * (ib % 2)));
                high |= static_cast<std::uint16_t>((ls >> 4) << (2 * ib));
            }
            put_u16(p + 2, high);
            for (int i = 0; i < 128; ++i) {
                p[8 + i] = mixed ? static_cast<std::uint8_t>((bit(rng) ? 0x0F : 0x00) | (bit(rng) ? 0xF0 : 0x00)) : 0;
            }
            break;
        }
        case QType::GGML_IQ3_XXS:
            for (int i = 0; i < 64; ++i) { p[2 + i] = static_cast<std::uint8_t>(grid62); }
            for (int ib = 0; ib < 8; ++ib) {
                const std::uint32_t signs = b == 0 ? 0U : static_cast<std::uint32_t>(byte(rng) | (byte(rng) << 8) |
                                                                                    (byte(rng) << 16) | ((byte(rng) & 15) << 24));
                const std::uint32_t word  = (15U << 28) | signs;
                for (int k = 0; k < 4; ++k) { p[66 + 4 * ib + k] = static_cast<std::uint8_t>(word >> (8 * k)); }
            }
            break;
        case QType::GGML_IQ4_NL:
            for (int i = 0; i < 16; ++i) {
                p[2 + i] = b == 0 ? 0 : static_cast<std::uint8_t>((bit(rng) ? 0x0F : 0x00) | (bit(rng) ? 0xF0 : 0x00));
            }
            break;
        default: check(false, "extreme_row: no extreme pattern for this format");
        }
    }
    return row;
}

void test_extreme_magnitude_records() {
    using ops::offloaded_moe::CpuIsa;
    const bool avx2 = ops::offloaded_moe::cpu_isa_supported(CpuIsa::kAvx2);
    std::mt19937 rng(41);
    struct Case {
        QType format;
        int k;
        float c; // the canonical scale c of a sub-block with d = 1
    };
    for (const Case& item : {Case{QType::GGML_IQ4_XS, om::kHidden, 1.0F}, Case{QType::GGML_IQ3_XXS, om::kHidden, 0.25F},
                             Case{QType::GGML_IQ4_NL, om::kIntermediate, 1.0F}}) {
        const auto block   = ggml_block(item.format);
        const int blocks   = item.k / static_cast<int>(block.values);
        const auto row     = extreme_row(item.format, blocks, rng);
        const auto ref     = ninfer::test::ggml::decode_blocks(
            item.format, std::span(reinterpret_cast<const std::byte*>(row.data()), row.size()));
        const int subs     = item.k / 32;
        // Column 0: every code +127; column 1: random +-127. Every activation scale is 1.
        std::vector<std::int8_t> q(2 * static_cast<std::size_t>(item.k));
        std::uniform_int_distribution<int> bit(0, 1);
        for (int j = 0; j < item.k; ++j) {
            q[j]          = 127;
            q[item.k + j] = static_cast<std::int8_t>(bit(rng) != 0 ? 127 : -127);
        }
        const std::vector<float> d(static_cast<std::size_t>(subs), 1.0F);
        const detail::A8Act acts[2] = {{q.data(), d.data()}, {q.data() + item.k, d.data()}};
        std::int64_t largest = 0;
        for (int c = 0; c < 2; ++c) {
            // The canonical row value from exact int64 sums over the reference decoder's values:
            // W / c is an integer m w for these blocks.
            float partial[canon::kLanes] = {};
            for (int s = 0; s < subs; ++s) {
                std::int64_t p = 0;
                for (int j = 0; j < 32; ++j) {
                    const float w = ref[static_cast<std::size_t>(32 * s + j)] / item.c;
                    check(w == std::nearbyint(w), "the extreme block's W / c is an integer");
                    p += static_cast<std::int64_t>(w) * q[static_cast<std::size_t>(c) * item.k + 32 * s + j];
                }
                largest = std::max(largest, p < 0 ? -p : p);
                check(p > -(1LL << 24) && p < (1LL << 24), "|P_s| < 2^24");
                partial[s % canon::kLanes] = canon::accumulate(partial[s % canon::kLanes], item.c, 1.0F,
                                                               static_cast<std::int32_t>(p));
            }
            const float want = canon::reduce_lanes(partial);
            for (CpuIsa isa : {CpuIsa::kScalar, CpuIsa::kAvx2}) {
                if (!ops::offloaded_moe::cpu_isa_supported(isa)) { continue; }
                float got = 0.0F;
                detail::row_values(isa, item.format, row.data(), row.size(), item.k, 0, 1, &acts[c], 1, &got);
                check(std::bit_cast<std::uint32_t>(got) == std::bit_cast<std::uint32_t>(want),
                      std::string("extreme row value differs from the exact sum on ") + ops::offloaded_moe::cpu_isa_name(isa));
            }
        }
        std::printf("extreme %-8s largest |P_s| %lld (< 2^24 = 16777216): exact on scalar%s\n",
                    item.format == QType::GGML_IQ4_XS ? "iq4_xs" : item.format == QType::GGML_IQ3_XXS ? "iq3_xxs" : "iq4_nl",
                    static_cast<long long>(largest), avx2 ? " and AVX2" : "");
        if (item.format == QType::GGML_IQ4_XS) {
            check(largest == 32LL * 32 * 127 * 127, "the IQ4_XS case reaches |P_s| = 32 * 32 * 127^2");
        }
    }
    // A whole IQ4_XS+IQ4_NL expert of extreme blocks against saturating activations: AVX2 = scalar,
    // every output finite.
    const QType format = QType::GGML_REC_IQ4_XS_IQ4_NL;
    const auto g       = om::record_geometry(format);
    std::vector<std::uint8_t> record(g.bytes, 0);
    for (int r = 0; r < om::kIntermediate; ++r) {
        for (const std::uint64_t base : {std::uint64_t{0}, g.up_offset}) {
            auto part = extreme_row(QType::GGML_IQ4_XS, om::kHidden / 256, rng);
            put_u16(part.data(), fp16_bits(1.0e-4F)); // keep h finite: d = 1e-4 in the first block
            for (std::size_t b = 1; b < part.size() / 136; ++b) { put_u16(part.data() + 136 * b, fp16_bits(1.0e-4F)); }
            std::copy(part.begin(), part.end(), record.begin() + static_cast<std::ptrdiff_t>(base + r * g.gate_up_row_bytes));
        }
    }
    for (int r = 0; r < om::kHidden; ++r) {
        auto part = extreme_row(QType::GGML_IQ4_NL, om::kIntermediate / 32, rng);
        for (std::size_t b = 0; b < part.size() / 18; ++b) { put_u16(part.data() + 18 * b, fp16_bits(1.0e-3F)); }
        std::copy(part.begin(), part.end(), record.begin() + static_cast<std::ptrdiff_t>(g.down_offset + r * g.down_row_bytes));
    }
    std::vector<std::uint16_t> x(static_cast<std::size_t>(om::kMaxColumns) * om::kHidden);
    for (std::size_t i = 0; i < x.size(); ++i) { x[i] = bf16_bits((i % 7 == 0 ? -1.0F : 1.0F) * (i % 3 == 0 ? 0.5F : 1.0F)); }
    const auto scalar = run_forward(CpuIsa::kScalar, format, record.data(), x, om::kMaxColumns);
    for (const auto& column : scalar) {
        for (const std::uint16_t v : column) { check(std::isfinite(bf16_value(v)), "an extreme expert output is not finite"); }
    }
    if (avx2) {
        check(run_forward(CpuIsa::kAvx2, format, record.data(), x, om::kMaxColumns) == scalar,
              "AVX2 differs from scalar on the extreme IQ4_XS+IQ4_NL record");
    }
    std::printf("extreme iq4_xs+iq4_nl record: finite outputs%s\n", avx2 ? ", AVX2 = scalar" : "");
}

template <class F>
bool rejects(F&& f) {
    try {
        f();
    } catch (const std::invalid_argument&) {
        return true;
    }
    return false;
}

void test_rejects_missing_columns() {
    using ops::offloaded_moe::CpuIsa;
    std::mt19937 rng(37);
    const QType format = kRecordFormats[0];
    const auto record  = random_record(format, rng);
    const auto x       = random_x(2, rng);
    std::vector<std::uint16_t> y0(om::kHidden), y1(om::kHidden);
    const std::uint16_t* xs[om::kMaxColumns] = {x.data(), x.data() + om::kHidden};
    std::uint16_t* ys[om::kMaxColumns]       = {y0.data(), y1.data()};
    const std::uint16_t* xs_gap[om::kMaxColumns] = {x.data(), nullptr};
    std::uint16_t* ys_gap[om::kMaxColumns]       = {y0.data(), nullptr};
    auto forward = [&](const std::uint8_t* r, const std::uint16_t* const* xp, std::uint16_t* const* yp) {
        return rejects([&] { ops::offloaded_moe::expert_forward_cpu(CpuIsa::kScalar, format, r, 2, xp, yp); });
    };
    check(forward(nullptr, xs, ys), "expert_forward_cpu accepted a null record");
    check(forward(record.data(), xs_gap, ys), "expert_forward_cpu accepted a null active x");
    check(forward(record.data(), xs, ys_gap), "expert_forward_cpu accepted a null active y");
    ops::offloaded_moe::CpuExpertTeam team({.workers = 2, .isa = CpuIsa::kScalar, .max_jobs = 1});
    ops::offloaded_moe::CpuExpertJob job;
    job.record = record.data();
    job.format = format;
    job.ncols  = 2;
    job.x[0]   = x.data();
    job.y[0]   = y0.data();
    job.y[1]   = y1.data(); // x[1] stays null
    check(rejects([&] { team.run({&job, 1}); }), "CpuExpertTeam accepted a null active x");
    job.x[1] = x.data() + om::kHidden;
    job.y[1] = nullptr;
    check(rejects([&] { team.run({&job, 1}); }), "CpuExpertTeam accepted a null active y");
    job.y[1] = y1.data();
    team.run({&job, 1}); // the same team still runs a valid job after the rejections
    // The optional progress word (CpuMissService's heartbeat) advances during every round.
    std::uint32_t progress = 0;
    ops::offloaded_moe::CpuExpertTeam ticking({.workers = 3, .isa = CpuIsa::kScalar, .max_jobs = 1, .progress = &progress});
    ticking.run({&job, 1});
    const std::uint32_t after_one = progress;
    ticking.run({&job, 1});
    check(after_one > 0 && progress > after_one, "the team's progress word did not advance during a round");
    std::printf("rejects a null record and null active columns; the progress word advances\n");
}

// The grid of GateUpVariant::Registers: (SMs - SMs / 8) * blocks per SM, an eighth of the SMs (rounded
// down) left to the side streams.
void test_coresident_blocks_values() {
    check(om::coresident_blocks(82, 2) == 144, "82 SMs (RTX 3090) at 2 blocks: 144");
    check(om::coresident_blocks(1, 2) == 2, "1 SM at 2 blocks: 2");
    check(om::coresident_blocks(8, 3) == 21, "8 SMs at 3 blocks: 21");
    check(om::coresident_blocks(7, 1) == 7, "7 SMs keep every SM");
    std::printf("coresident_blocks_values: 82 SMs x 2 -> 144, 1 x 2 -> 2\n");
}

} // namespace

int main() {
    test_a8_cast();
    test_decode();
    test_oracle_and_isa();
    test_team();
    test_extreme_magnitude_records();
    test_rejects_missing_columns();
    test_coresident_blocks_values();
    std::printf("offloaded_moe_cpu: passed\n");
    return 0;
}
