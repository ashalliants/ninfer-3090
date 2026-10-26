// The int8 tensor-core decode route of the GGML linears (LinearPolicy::AllowA8GgmlDecodeMma, T <= 8),
// on the host: src/ops/linear/ggml/ggml_mma_emulation.h reproduces the kernel's arithmetic one IEEE
// operation at a time, and the GPU suite (ninfer_linear_ggml_test) checks the device against it bit
// for bit. Every format the GGML linear registry routes is covered: Q8_0, Q2_0, IQ4_NL, IQ4_XS,
// IQ3_S, Q6_K, Q5_K and Q4_K. (Basalt's dense_tc also takes Q3_K; no registered problem uses it and
// NInfer has no Q3_K block format, so there is nothing to route.)
//
// 1. decomposition: the exact sub-block decode (c, m_h, w, and c_min m_min for Q4_K/Q5_K) gives
//    ggml's value bit for bit, c m w (- c_min m_min), against the exact host decoder
//    (ops/ggml_blocks_decode.h, itself qualified against ggml b11316) on the golden fixtures and on
//    random blocks.
// 2. emulation_within_fp64: every registered (format, N, K) problem, Linear and LinearAdd, at T = 8
//    columns against the naive FP64 product of the exactly decoded weight and the Q8_1-cast
//    activation (linear.h's ideal): the BF16 output within the GGML suite's criterion
//    (tests/ops/linear/test_ggml.cpp) and the FP32 row value within the accumulation bound. A row's
//    arithmetic depends only on the format and K, so each problem checks its first 512 rows.
// 3. emulation_column_invariant: the kernel's tile distribution (emulate_tile: eight warps, the
//    in-register butterfly steps, then three across warps) equals the canonical row order
//    (emulate_row) bit for bit for T = 1..8, with the columns in any position: a column's bits do not
//    depend on T or on where it sits.
// 4. int32_range_safe: on extreme blocks (the largest |w| and |m| each format can encode, against
//    codes of +-127 matched to the weights' signs) every 16- and 32-value MMA sum stays below 2^22
//    and every scaled product m_0 S_0 + m_1 S_1 (and m_min A) below 2^24, so the int32 sums are
//    exact and float(P) is exact. The same bounds hold analytically for every format.

#include "core/weight.h"
#include "ops/ggml_blocks_decode.h"
#include "ops/ggml_linear_emulation.h"
#include "ops/linear/ggml/ggml_mma_emulation.h"
#include "ops/linear/ggml/ggml_shapes.h"
#include "ops/op_check.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

using namespace ninfer;
namespace mma = ninfer::ops::ggml_mma;
namespace em  = ninfer::test::ggml_linear;

// The GGML suite's criterion (tests/ops/linear/test_ggml.cpp).
constexpr test::ReductionCriterion kGgmlQ81Criterion{1.0 / 256.0, 0.0, test::kBf16GrossRelativeFloor};

constexpr QType kFormats[] = {QType::GGML_Q8_0,  QType::GGML_Q2_0, QType::GGML_IQ4_NL, QType::GGML_IQ4_XS,
                              QType::GGML_IQ3_S, QType::GGML_Q6_K, QType::GGML_Q5_K,   QType::GGML_Q4_K};

int g_failures = 0;

void check(bool ok, const std::string& what) {
    if (!ok) {
        std::cerr << "FAIL: " << what << '\n';
        ++g_failures;
    }
}

const char* name_of(QType format) {
    switch (format) {
    case QType::GGML_Q8_0: return "ggml_q8_0";
    case QType::GGML_Q2_0: return "ggml_q2_0";
    case QType::GGML_IQ4_NL: return "ggml_iq4_nl";
    case QType::GGML_IQ4_XS: return "ggml_iq4_xs";
    case QType::GGML_IQ3_S: return "ggml_iq3_s";
    case QType::GGML_Q6_K: return "ggml_q6_k";
    case QType::GGML_Q5_K: return "ggml_q5_k";
    case QType::GGML_Q4_K: return "ggml_q4_k";
    default: return "?";
    }
}

bool same_bits(float a, float b) {
    return std::isnan(a) ? std::isnan(b) : std::bit_cast<std::uint32_t>(a) == std::bit_cast<std::uint32_t>(b);
}

std::vector<std::uint8_t> read_file(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file.good()) { throw std::runtime_error("cannot open fixture " + path.string()); }
    return {std::istreambuf_iterator<char>(file), {}};
}

std::uint16_t fp16_bits(bool negative, int exponent, int mantissa) {
    return static_cast<std::uint16_t>((negative ? 0x8000 : 0) | ((exponent + 15) << 10) | (mantissa & 0x3FF));
}

// Random blocks (every code byte pattern is valid in these formats) with finite binary16 scales of
// either sign, as the GPU suite makes them.
std::vector<std::uint8_t> random_blocks(QType format, std::size_t blocks, std::uint32_t seed) {
    const GgmlBlock block = ggml_block(format);
    std::vector<std::uint8_t> bytes(blocks * block.bytes);
    std::mt19937 rng(seed);
    for (auto& b : bytes) b = static_cast<std::uint8_t>(rng());
    const std::size_t d_offset = format == QType::GGML_Q6_K ? 208 : 0;
    const int base             = format == QType::GGML_Q8_0 || format == QType::GGML_IQ4_NL ? -12 : -14;
    for (std::size_t i = 0; i < blocks; ++i) {
        for (int field = 0; field < (mma::affine(format) ? 2 : 1); ++field) {
            const auto h = fp16_bits(rng() & 1, base + static_cast<int>(rng() % 4), static_cast<int>(rng() & 0x3FF));
            std::memcpy(&bytes[i * block.bytes + d_offset + 2 * field], &h, 2);
        }
    }
    return bytes;
}

// BF16 [K,T]; each 32-value group gets its own magnitude, and the first group of every column is
// zero (the cast's amax = 0 rule).
std::vector<std::uint16_t> make_activation(std::int32_t k, std::int32_t t, std::uint32_t seed) {
    std::vector<std::uint16_t> x(static_cast<std::size_t>(k) * t);
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> unit(-0.5F, 0.5F);
    for (std::int32_t col = 0; col < t; ++col) {
        for (std::int32_t g = 0; g < k / 32; ++g) {
            const float scale = std::ldexp(1.0F, static_cast<int>(rng() % 9) - 6);
            for (int i = 0; i < 32; ++i) {
                x[static_cast<std::size_t>(col) * k + 32 * g + i] = em::bf16_bits(g == 0 ? 0.0F : unit(rng) * scale);
            }
        }
    }
    return x;
}

template <class F>
void parallel_rows(std::int32_t rows, F&& body) {
    const unsigned workers = std::max(1U, std::min(16U, std::thread::hardware_concurrency()));
    std::vector<std::thread> pool;
    for (unsigned id = 0; id < workers; ++id) {
        pool.emplace_back([&, id] {
            for (std::int32_t r = static_cast<std::int32_t>(id); r < rows; r += static_cast<std::int32_t>(workers)) body(r);
        });
    }
    for (auto& worker : pool) worker.join();
}

// 1. The exact sub-block decode against ggml's values.
void decomposition(QType format) {
    const std::filesystem::path root = std::filesystem::path(NINFER_SOURCE_DIR) / "tests" / "fixtures" / "ggml";
    const auto golden     = read_file(root / (std::string(name_of(format)) + ".blocks"));
    const auto random     = random_blocks(format, 4096, 0xD3C0 + static_cast<int>(format));
    const GgmlBlock block = ggml_block(format);
    const int per_block   = static_cast<int>(block.values) / 32;
    for (const auto* source : {&golden, &random}) {
        const auto exact = test::ggml::decode_blocks(format, std::as_bytes(std::span(*source)));
        std::size_t bad  = 0;
        for (std::size_t s = 0; s < exact.size() / 32; ++s) {
            const auto* b       = source->data() + (s / per_block) * block.bytes;
            const mma::Sub sub = mma::decode_sub(format, b, static_cast<int>(s % per_block));
            for (int j = 0; j < 32; ++j) {
                float value = (sub.c * static_cast<float>(sub.m[j / 16])) * static_cast<float>(sub.w[j]);
                if (mma::affine(format)) value -= sub.c_min * static_cast<float>(sub.m_min);
                bad += !same_bits(value, exact[32 * s + j]);
            }
        }
        const std::string what = std::string(name_of(format)) + (source == &golden ? " golden" : " random") +
                                 " blocks: c m w is ggml's value";
        check(bad == 0, what + " (" + std::to_string(bad) + " differ)");
        if (bad == 0) { std::cout << what << " for all " << exact.size() << " values\n"; }
    }
}

// The magnitude that bounds a row value's FP32 error: sum over k of (|c m w_k| + |c_min m_min|) *
// |q_k| * d, in FP64. The route adds K / 32 (2 K / 32 for the affine formats) fma terms into the
// lane partials, each rounded once with its product c * d, then 31 additions: well within
// K * 2^-24 of this magnitude.
double term_magnitude(QType format, const std::uint8_t* row, const em::Q81& x, std::int32_t k, std::int32_t col) {
    const int subs = k / 32;
    double sum     = 0.0;
    for (int s = 0; s < subs; ++s) {
        const mma::Sub sub   = mma::decode_sub(format, row, s);
        const std::int8_t* q = x.q.data() + static_cast<std::size_t>(col) * k + 32 * s;
        const double d       = x.d[static_cast<std::size_t>(col) * subs + s];
        for (int j = 0; j < 32; ++j) {
            const double w = std::fabs(static_cast<double>(sub.c) * sub.m[j / 16] * sub.w[j]) +
                             std::fabs(static_cast<double>(sub.c_min) * sub.m_min);
            sum += w * std::fabs(static_cast<double>(q[j])) * d;
        }
    }
    return sum;
}

// 2. One registered problem: its first rows against the FP64 ideal, T = 8 (`residual`: LinearAdd).
void problem(QType format, std::int32_t n, std::int32_t k, bool residual) {
    constexpr std::int32_t kColumns = ops::detail::kGgmlDecodeMaxTokens;
    const std::int32_t rows         = std::min<std::int32_t>(n, 512);
    const std::uint32_t seed        = static_cast<std::uint32_t>(n * 131 + k * 7 + static_cast<int>(format));
    const std::size_t row_bytes     = mma::row_bytes(format, k);
    const auto blocks = random_blocks(format, static_cast<std::size_t>(rows) * (k / ggml_block(format).values), seed);
    const auto x_bits = make_activation(k, kColumns, seed ^ 0x5bd1e995U);
    const em::Q81 x   = em::cast_q81(x_bits, k, kColumns);
    std::vector<std::uint16_t> old(static_cast<std::size_t>(rows) * kColumns);
    {
        std::mt19937 rng(seed + 1);
        std::uniform_real_distribution<float> unit(-4.0F, 4.0F);
        for (auto& r : old) r = em::bf16_bits(unit(rng));
    }
    std::vector<double> ideal(old.size()), got(old.size());
    std::vector<double> ratio(static_cast<std::size_t>(rows), 0.0);
    parallel_rows(rows, [&](std::int32_t r) {
        const auto* row  = blocks.data() + static_cast<std::size_t>(r) * row_bytes;
        const auto exact = test::ggml::decode_blocks(format, std::as_bytes(std::span(row, row_bytes)));
        for (std::int32_t col = 0; col < kColumns; ++col) {
            double sum = 0.0;
            for (std::int32_t i = 0; i < k; ++i) {
                const std::size_t xi = static_cast<std::size_t>(col) * k + i;
                sum += static_cast<double>(exact[i]) * static_cast<double>(x.q[xi]) * static_cast<double>(x.d[xi / 32]);
            }
            float acc = 0.0F;
            ops::ggml_mma_emulate(format, reinterpret_cast<const std::byte*>(row), x.q.data() + static_cast<std::size_t>(col) * k,
                                  x.d.data() + static_cast<std::size_t>(col) * (k / 32), k, &acc);
            const double bound = term_magnitude(format, row, x, k, col) * k * 0x1p-24 + 1e-37;
            ratio[r]           = std::max(ratio[r], std::fabs(static_cast<double>(acc) - sum) / bound);
            const std::size_t o = static_cast<std::size_t>(col) * rows + r;
            ideal[o]            = sum + (residual ? static_cast<double>(em::bf16_value(old[o])) : 0.0);
            got[o]              = em::bf16_value(em::store(acc, residual ? &old[o] : nullptr));
        }
    });
    const std::string label = std::string(residual ? "linear_add " : "linear ") + name_of(format) + " [" +
                              std::to_string(n) + "," + std::to_string(k) + "]";
    const double worst = *std::max_element(ratio.begin(), ratio.end());
    check(worst <= 1.0, label + ": FP32 row value within sum |terms| K 2^-24 (worst " + std::to_string(worst) + ")");
    const auto count = static_cast<std::int64_t>(ideal.size());
    const auto stats = test::compute_reduction_stats(got.data(), ideal.data(), count);
    check(test::reduction_passes(stats, count, kGgmlQ81Criterion),
          label + " within the GGML criterion (relative L2 " + std::to_string(stats.relative_l2) + ")");
    std::cout << label << " T=8: relative L2 " << stats.relative_l2 << ", worst |error| " << stats.maximum_absolute_error
              << " (largest |ideal| " << stats.maximum_absolute_reference << "), FP32 error at most " << worst
              << " of the accumulation bound\n";
}

void emulation_within_fp64() {
    std::size_t count = 0;
    for (const auto& shape : ops::detail::kGgmlLinearShapes) {
        check(mma::supported(shape.format), std::string("registered format ") + name_of(shape.format) + " has an emulation");
        problem(shape.format, shape.n, shape.k, false);
        ++count;
        if (shape.linear_add) {
            problem(shape.format, shape.n, shape.k, true);
            ++count;
        }
    }
    check(count == std::size(ops::detail::kGgmlLinearShapes) + 8, "every registered Linear and LinearAdd problem was checked");
}

// 3. The tile distribution equals the row order, for every T and column position.
void emulation_column_invariant() {
    for (const QType format : kFormats) {
        for (const std::int32_t k : {640, 2560, 6144}) {
            if (!ops::detail::ggml_k_registered(format, k)) { continue; }
            const std::uint32_t seed = static_cast<std::uint32_t>(k * 17 + static_cast<int>(format));
            const auto blocks = random_blocks(format, static_cast<std::size_t>(mma::kRows) * (k / ggml_block(format).values), seed);
            const auto x_bits = make_activation(k, mma::kColumns, seed ^ 0x9e3779b9U);
            const em::Q81 x   = em::cast_q81(x_bits, k, mma::kColumns);
            const std::size_t row_bytes = mma::row_bytes(format, k);
            // The canonical row value of every (row, column).
            std::vector<float> want(static_cast<std::size_t>(mma::kRows) * mma::kColumns);
            for (int col = 0; col < mma::kColumns; ++col) {
                for (int r = 0; r < mma::kRows; ++r) {
                    ops::ggml_mma_emulate(format, reinterpret_cast<const std::byte*>(blocks.data() + r * row_bytes),
                                          x.q.data() + static_cast<std::size_t>(col) * k,
                                          x.d.data() + static_cast<std::size_t>(col) * (k / 32), k,
                                          &want[static_cast<std::size_t>(col) * mma::kRows + r]);
                }
            }
            std::size_t bad = 0;
            for (int t = 1; t <= mma::kColumns; ++t) {
                // T columns, the source columns rotated by T so each lands in different positions.
                std::vector<std::int8_t> q(static_cast<std::size_t>(t) * k);
                std::vector<float> d(static_cast<std::size_t>(t) * (k / 32));
                std::vector<int> source(static_cast<std::size_t>(t));
                for (int c = 0; c < t; ++c) {
                    source[c] = (c + t) % mma::kColumns;
                    std::copy_n(x.q.data() + static_cast<std::size_t>(source[c]) * k, k, q.data() + static_cast<std::size_t>(c) * k);
                    std::copy_n(x.d.data() + static_cast<std::size_t>(source[c]) * (k / 32), k / 32,
                                d.data() + static_cast<std::size_t>(c) * (k / 32));
                }
                std::vector<float> got(static_cast<std::size_t>(t) * mma::kRows);
                mma::emulate_tile(format, reinterpret_cast<const std::byte*>(blocks.data()), q.data(), d.data(), k, t, got.data());
                for (int c = 0; c < t; ++c) {
                    for (int r = 0; r < mma::kRows; ++r) {
                        bad += !same_bits(got[static_cast<std::size_t>(c) * mma::kRows + r],
                                          want[static_cast<std::size_t>(source[c]) * mma::kRows + r]);
                    }
                }
            }
            const std::string what = std::string(name_of(format)) + " K=" + std::to_string(k) +
                                     ": the tile order equals the row order bit for bit at T=1..8";
            check(bad == 0, what + " (" + std::to_string(bad) + " differ)");
            if (bad == 0) { std::cout << what << '\n'; }
        }
    }
}

// 4. Extreme blocks. Each format's encodable maxima of |w| and |m| (ggml_mma_emulation.h, step 1):
// the analytic worst case of a 32-value sum, and of the scaled product.
struct Extreme {
    int max_w;
    int max_m;
};

Extreme analytic(QType format) {
    switch (format) {
    case QType::GGML_Q8_0: return {128, 1};
    case QType::GGML_Q2_0: return {2, 1};
    case QType::GGML_IQ4_NL: return {127, 1};
    case QType::GGML_IQ4_XS: return {127, 32};
    case QType::GGML_IQ3_S: return {15, 31};
    case QType::GGML_Q6_K: return {32, 128};
    case QType::GGML_Q5_K: return {31, 63};
    case QType::GGML_Q4_K: return {15, 63};
    default: break;
    }
    throw std::invalid_argument("no analytic bound");
}

// Blocks that reach each format's largest |w| and |m|: every byte one pattern, then d = 1. For IQ3_S
// the codes are set to the grid entry with the largest magnitude, then the largest sum.
std::vector<std::vector<std::uint8_t>> extreme_blocks(QType format) {
    const GgmlBlock block = ggml_block(format);
    std::vector<std::vector<std::uint8_t>> out;
    for (const std::uint8_t pattern : {0x00, 0xFF, 0x80, 0x7F, 0x55, 0xAA, 0x88, 0x11}) {
        std::vector<std::uint8_t> b(block.bytes, pattern);
        if (format == QType::GGML_IQ3_S) {
            int best = 0, best_score = -1;
            for (int i = 0; i < 512; ++i) {
                int largest = 0, sum = 0;
                for (int j = 0; j < 4; ++j) {
                    const int v = static_cast<int>((ggml_tables::iq3s_grid[i] >> (8 * j)) & 0xFF);
                    largest     = std::max(largest, v);
                    sum += v;
                }
                if (256 * largest + sum > best_score) { best = i, best_score = 256 * largest + sum; }
            }
            for (int j = 0; j < 64; ++j) b[2 + j] = static_cast<std::uint8_t>(best & 0xFF);
            for (int j = 0; j < 8; ++j) b[66 + j] = best >> 8 ? 0xFF : 0x00;
        }
        const std::uint16_t one = fp16_bits(false, 0, 0);
        std::memcpy(b.data() + (format == QType::GGML_Q6_K ? 208 : 0), &one, 2);
        if (mma::affine(format)) std::memcpy(b.data() + 2, &one, 2);
        out.push_back(std::move(b));
    }
    return out;
}

void int32_range_safe() {
    for (const QType format : kFormats) {
        const Extreme e = analytic(format);
        // The analytic worst cases (|q| <= 127).
        const long long sum_bound     = 32LL * e.max_w * 127;
        const long long product_bound = sum_bound * e.max_m;
        const long long offset_bound  = mma::affine(format) ? 63LL * 32 * 127 : 0;
        check(sum_bound < (1LL << 22), std::string(name_of(format)) + ": the analytic 32-value sum bound is below 2^22");
        check(product_bound < (1LL << 24) && offset_bound < (1LL << 24),
              std::string(name_of(format)) + ": the analytic scaled-product bound is below 2^24");
        long long max_sum = 0, max_half = 0, max_product = 0, max_offset = 0;
        int seen_w = 0, seen_m = 0;
        const int per_block = static_cast<int>(ggml_block(format).values) / 32;
        for (const auto& b : extreme_blocks(format)) {
            for (int ib = 0; ib < per_block; ++ib) {
                const mma::Sub sub = mma::decode_sub(format, b.data(), ib);
                for (const int sign : {1, -1}) {
                    std::int8_t q[32];
                    for (int j = 0; j < 32; ++j) q[j] = static_cast<std::int8_t>(sub.w[j] == 0 ? 0 : (sub.w[j] > 0 ? 127 : -127) * sign);
                    const auto h = mma::half_sums(sub, q);
                    max_half     = std::max({max_half, std::llabs(h.s0), std::llabs(h.s1)});
                    max_sum      = std::max(max_sum, std::llabs(static_cast<long long>(h.s0) + h.s1));
                    // P with the halves' multiplier signs matched to their sums: the largest |P|.
                    const long long p = std::llabs(static_cast<long long>(sub.m[0]) * h.s0) +
                                        std::llabs(static_cast<long long>(sub.m[1]) * h.s1);
                    max_product = std::max(max_product, p);
                    std::int8_t all[32];
                    std::fill_n(all, 32, static_cast<std::int8_t>(127 * sign));
                    max_offset = std::max(max_offset, std::llabs(static_cast<long long>(sub.m_min) * mma::code_sum(all)));
                    const auto exact = mma::product(sub, q);
                    check(static_cast<long long>(static_cast<float>(exact)) == exact,
                          std::string(name_of(format)) + ": float(P) is exact on an extreme block");
                }
                for (int j = 0; j < 32; ++j) seen_w = std::max(seen_w, std::abs(sub.w[j]));
                seen_m = std::max({seen_m, std::abs(sub.m[0]), std::abs(sub.m[1])});
            }
        }
        const std::string label = name_of(format);
        check(seen_w == e.max_w && seen_m == e.max_m,
              label + ": the extreme blocks reach the format's largest |w| and |m| (|w| " + std::to_string(seen_w) +
                  ", |m| " + std::to_string(seen_m) + ")");
        check(max_sum < (1LL << 22) && max_half < (1LL << 22), label + ": the MMA sums of extreme blocks are below 2^22");
        check(max_product < (1LL << 24) && max_offset < (1LL << 24),
              label + ": the scaled products of extreme blocks are below 2^24");
        std::cout << label << ": extreme blocks: |S| at most " << max_sum << " (< 2^22), |P| at most " << max_product
                  << " (< 2^24), |m_min A| at most " << max_offset << '\n';
    }
}

} // namespace

int main() {
    try {
        for (const QType format : kFormats) decomposition(format);
        emulation_within_fp64();
        emulation_column_invariant();
        int32_range_safe();
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
    if (g_failures != 0) {
        std::cerr << g_failures << " failures\n";
        return 1;
    }
    std::cout << "GGML int8 tensor-core decode emulation: passed\n";
    return 0;
}
