// The CPU routes of offloaded_sparse_moe (host only):
//   - the canonical A8 cast against an independent FP64 formulation, bit for bit, on random,
//     tie, subnormal, zero, overflow and NaN groups;
//   - the exact sub-block decode of IQ2_S, IQ2_XXS, IQ1_M and Q2_0: c * m * w equals ggml's
//     reference value (tests/ops/ggml_blocks_decode.h) bit for bit;
//   - expert_forward_cpu (scalar) against the FP64 expert oracle over the decoded record and the
//     A8-cast activations, for each record format at T in {1, 2, 7, 8};
//   - AVX2 = scalar, and the worker team at 1, 2, 3, 8 and 16 workers = scalar, bit for bit.

#include "ninfer/ops/offloaded_sparse_moe.h"
#include "offloaded_moe_fixtures.h"
#include "ops/common/canonical_ggml.h"
#include "ops/offloaded_sparse_moe/cpu/expert_cpu.h"

#include <bit>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <random>
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
        case 1: for (float& x : v) { x *= 1e-38F; } break;                        // subnormal: 127 / a overflows
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
    for (QType format : {QType::GGML_IQ2_S, QType::GGML_IQ2_XXS, QType::GGML_IQ1_M, QType::GGML_Q2_0}) {
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
            std::printf("%-13s T=%d: scalar vs FP64 normwise %.3e, worst %.3e\n", format_name(format), t, worst_norm,
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
            std::printf("%-13s AVX2 = scalar at T = 1..8\n", format_name(format));
        }
    }
}

void test_team() {
    using ops::offloaded_moe::CpuIsa;
    std::mt19937 rng(31);
    // Eight jobs: every format, column counts 1..8, two jobs sharing a record.
    std::vector<std::vector<std::uint8_t>> records;
    std::vector<QType> formats;
    for (int j = 0; j < 7; ++j) {
        formats.push_back(kRecordFormats[j % 3]);
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
            ops::offloaded_moe::CpuExpertTeam team({.workers = workers, .isa = isa, .max_jobs = 8});
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
        std::printf("team (%s): 1, 2, 3, 8, 16 workers = scalar on 8 jobs\n", ops::offloaded_moe::cpu_isa_name(isa));
    }
}

} // namespace

int main() {
    test_a8_cast();
    test_decode();
    test_oracle_and_isa();
    test_team();
    std::printf("offloaded_moe_cpu: passed\n");
    return 0;
}
