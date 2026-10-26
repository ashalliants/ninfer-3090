// The GGML linears' arithmetic for the affine K-quants Q4_K and Q5_K, on the host
// (ops/ggml_linear_emulation.h reproduces both routes one IEEE operation at a time; the GPU suite
// ninfer_linear_ggml_test checks the device against this emulation bit for bit).
//
// 1. The sub-block decomposition the device decoder forms (codes q, scale = d * sc, offset =
//    dmin * m) reproduces ggml b11316's dequantize_row_q4_K/q5_K: binary32(binary32(scale * q) -
//    offset) equals the exact decoder (ops/ggml_blocks_decode.h, itself qualified against ggml's
//    goldens) bit for bit, on the golden fixtures (every scale and min field, the special binary16
//    d and dmin) and on random blocks.
// 2. Every registered (Q4_K or Q5_K, N, K) problem, at T = 1..8 and on both routes' summation
//    orders (a column's arithmetic depends on neither T nor the column's position, so the wide
//    order over the same columns stands for every T > 8), against the naive FP64 product of the
//    exactly decoded weight and the Q8_1-cast activation (linear.h's ideal): the BF16 output
//    within the suite's criterion, and the FP32 accumulator within the documented accumulation
//    bound. A row's arithmetic depends only on the format and K, so each problem checks its first
//    1024 rows. LinearAdd problems add a BF16 residual.
// 3. Rows built to cancel (d sc q close to dmin m for every value) stay within the FP32 bound.

#include "core/weight.h"
#include "ops/ggml_blocks_decode.h"
#include "ops/ggml_linear_emulation.h"
#include "ops/linear/ggml/ggml_shapes.h"
#include "ops/op_check.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
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
namespace em = ninfer::test::ggml_linear;

// The GGML suite's criterion (tests/ops/linear/test_ggml.cpp): A16-class arithmetic against an
// oracle that already holds the Q8_1 cast.
constexpr test::ReductionCriterion kGgmlQ81Criterion{1.0 / 256.0, 0.0, test::kBf16GrossRelativeFloor};

int g_failures = 0;

void check(bool ok, const std::string& what) {
    if (!ok) {
        std::cerr << "FAIL: " << what << '\n';
        ++g_failures;
    }
}

const char* name_of(QType format) { return format == QType::GGML_Q4_K ? "ggml_q4_k" : "ggml_q5_k"; }

std::vector<std::uint8_t> read_file(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file.good()) { throw std::runtime_error("cannot open fixture " + path.string()); }
    return {std::istreambuf_iterator<char>(file), {}};
}

bool same_bits(float a, float b) {
    return std::isnan(a) ? std::isnan(b) : std::bit_cast<std::uint32_t>(a) == std::bit_cast<std::uint32_t>(b);
}

std::uint16_t fp16_bits(bool negative, int exponent, int mantissa) {
    return static_cast<std::uint16_t>((negative ? 0x8000 : 0) | ((exponent + 15) << 10) | (mantissa & 0x3FF));
}

// Random blocks with finite binary16 d and dmin of either sign (every other byte pattern is valid).
std::vector<std::uint8_t> random_blocks(QType format, std::size_t blocks, std::uint32_t seed) {
    const GgmlBlock block = ggml_block(format);
    std::vector<std::uint8_t> bytes(blocks * block.bytes);
    std::mt19937 rng(seed);
    for (auto& b : bytes) b = static_cast<std::uint8_t>(rng());
    for (std::size_t i = 0; i < blocks; ++i) {
        for (const int at : {0, 2}) {
            const auto h = fp16_bits(rng() & 1, -14 + static_cast<int>(rng() % 4), static_cast<int>(rng() & 0x3FF));
            std::memcpy(&bytes[i * block.bytes + at], &h, 2);
        }
    }
    return bytes;
}

// 1. The device decomposition against the exact decoder.
void decomposition(QType format) {
    const std::filesystem::path root = std::filesystem::path(NINFER_SOURCE_DIR) / "tests" / "fixtures" / "ggml";
    const auto golden = read_file(root / (std::string(name_of(format)) + ".blocks"));
    const auto random = random_blocks(format, 4096, 0x51ED + static_cast<int>(format));
    const GgmlBlock block = ggml_block(format);
    for (const auto* source : {&golden, &random}) {
        const auto exact = test::ggml::decode_blocks(format, std::as_bytes(std::span(*source)));
        std::size_t bad = 0;
        for (std::size_t s = 0; s < exact.size() / 32; ++s) {
            const auto* row = source->data() + (s / 8) * block.bytes;
            const auto sub  = em::affine_sub(format, row, static_cast<int>(s % 8));
            for (int l = 0; l < 32; ++l) {
                const float product = sub.scale * static_cast<float>(sub.q[l]);
                bad += !same_bits(product - sub.offset, exact[32 * s + l]);
            }
        }
        const std::string what = std::string(name_of(format)) + (source == &golden ? " golden" : " random") +
                                 " blocks: scale q - offset is ggml's value";
        check(bad == 0, what + " (" + std::to_string(bad) + " differ)");
        if (bad == 0) { std::cout << what << " for all " << exact.size() << " values\n"; }
    }
}

// BF16 [K,T] whose 32-value groups each get their own magnitude; the first group of every column
// is zero (the cast's amax = 0 rule).
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

// Rows of `blocks` against the oracle, both route orders, T = 1..8 (`residual`: LinearAdd).
void problem(const std::string& label, QType format, std::int32_t rows, std::int32_t k,
             const std::vector<std::uint8_t>& blocks, bool residual, std::uint32_t seed, bool report_bf16) {
    constexpr std::int32_t kColumns = ops::detail::kGgmlDecodeMaxTokens;
    const std::size_t row_bytes     = static_cast<std::size_t>(k) / 256 * ggml_block(format).bytes;
    const auto x_bits               = make_activation(k, kColumns, seed);
    const em::Q81 x                 = em::cast_q81(x_bits, k, kColumns);
    const auto xm                   = em::activation_sums(x);
    std::vector<std::uint16_t> old(static_cast<std::size_t>(rows) * kColumns);
    {
        std::mt19937 rng(seed + 1);
        std::uniform_real_distribution<float> unit(-4.0F, 4.0F);
        for (auto& r : old) r = em::bf16_bits(unit(rng));
    }
    std::vector<double> ideal(old.size());
    std::vector<double> got[2] = {std::vector<double>(old.size()), std::vector<double>(old.size())};
    std::vector<double> ratio(static_cast<std::size_t>(rows), 0.0);
    parallel_rows(rows, [&](std::int32_t r) {
        const auto* row  = blocks.data() + static_cast<std::size_t>(r) * row_bytes;
        const auto exact = test::ggml::decode_blocks(format, std::as_bytes(std::span(row, row_bytes)));
        const auto subs  = em::affine_row(format, row, k);
        for (std::int32_t t = 0; t < kColumns; ++t) {
            double sum = 0.0;
            for (std::int32_t i = 0; i < k; ++i) {
                sum += static_cast<double>(exact[i]) * static_cast<double>(x.q[static_cast<std::size_t>(t) * k + i]) *
                       static_cast<double>(x.d[(static_cast<std::size_t>(t) * k + i) / 32]);
            }
            const std::size_t o    = static_cast<std::size_t>(t) * rows + r;
            const double bound     = em::term_magnitude(subs, x, k, t) * k * 0x1p-24 + 1e-37;
            const std::uint16_t* rr = residual ? &old[o] : nullptr;
            ideal[o]               = sum + (residual ? static_cast<double>(em::bf16_value(old[o])) : 0.0);
            for (const auto route : {em::Route::Decode, em::Route::Wide}) {
                const float acc = em::row_dot(route, subs, x, xm, k, t);
                ratio[r]        = std::max(ratio[r], std::fabs(static_cast<double>(acc) - sum) / bound);
                got[route == em::Route::Wide][o] = em::bf16_value(em::store(acc, rr));
            }
        }
    });
    const double worst = *std::max_element(ratio.begin(), ratio.end());
    check(worst <= 1.0, label + ": FP32 accumulator within sum |terms| K 2^-24 (worst " + std::to_string(worst) + ")");
    if (report_bf16) {
        for (int route = 0; route < 2; ++route) {
            const auto count = static_cast<std::int64_t>(ideal.size());
            const auto stats = test::compute_reduction_stats(got[route].data(), ideal.data(), count);
            const std::string what = label + (route == 0 ? " decode" : " wide") + " T=1..8";
            check(test::reduction_passes(stats, count, kGgmlQ81Criterion),
                  what + " within the GGML criterion (relative L2 " + std::to_string(stats.relative_l2) + ")");
            std::cout << what << ": relative L2 " << stats.relative_l2 << ", worst |error| "
                      << stats.maximum_absolute_error << " (largest |ideal| " << stats.maximum_absolute_reference
                      << ")\n";
        }
    }
    std::cout << label << ": FP32 error at most " << worst << " of the accumulation bound\n";
}

// 2. Every registered affine problem.
void registered_problems() {
    std::size_t count = 0;
    for (const auto& shape : ops::detail::kGgmlLinearShapes) {
        if (!em::is_affine(shape.format)) { continue; }
        const std::int32_t rows  = std::min<std::int32_t>(shape.n, 1024);
        const std::uint32_t seed = static_cast<std::uint32_t>(shape.n * 131 + shape.k * 7 + static_cast<int>(shape.format));
        const auto blocks        = random_blocks(shape.format, static_cast<std::size_t>(rows) * (shape.k / 256), seed);
        for (const bool add : {false, true}) {
            if (add && !shape.linear_add) { continue; }
            const std::string label = std::string(add ? "linear_add " : "linear ") + name_of(shape.format) + " [" +
                                      std::to_string(shape.n) + "," + std::to_string(shape.k) + "]";
            problem(label, shape.format, rows, shape.k, blocks, add, seed ^ 0x5bd1e995U, true);
            ++count;
        }
    }
    check(count >= 14, "every registered Q4_K and Q5_K problem was checked");
}

// 3. Cancelling rows: sc = 1, m = 8, d = dmin and every code 8 (or near it), so each weight is
// d (q - 8) with q in {7, 8, 9}: the two partial products cancel to a small remainder.
void cancelling(QType format) {
    constexpr std::int32_t rows = 64, k = 2560;
    auto blocks = random_blocks(format, static_cast<std::size_t>(rows) * (k / 256), 0xCA7 + static_cast<int>(format));
    const GgmlBlock block = ggml_block(format);
    std::mt19937 rng(7);
    for (std::size_t i = 0; i < blocks.size() / block.bytes; ++i) {
        std::uint8_t* b = blocks.data() + i * block.bytes;
        std::memcpy(b + 2, b, 2); // dmin = d
        for (int j = 0; j < 4; ++j) {
            b[4 + j]  = 1; // sc_j = 1 for j < 4, no high bits for j >= 4
            b[8 + j]  = 8; // m_j = 8
            b[12 + j] = static_cast<std::uint8_t>(1 | (8 << 4)); // sc = 1, m = 8 for j >= 4
        }
        const bool q5 = format == QType::GGML_Q5_K;
        if (q5) std::memset(b + 16, 0, 32); // no fifth bits
        for (int l = 0; l < 128; ++l) {
            const int lo = 7 + static_cast<int>(rng() % 3), hi = 7 + static_cast<int>(rng() % 3);
            b[(q5 ? 48 : 16) + l] = static_cast<std::uint8_t>(lo | (hi << 4));
        }
    }
    problem(std::string("cancelling ") + name_of(format) + " [64,2560]", format, rows, k, blocks, false, 0xC0DE, false);
}

} // namespace

int main() {
    try {
        for (const QType format : {QType::GGML_Q4_K, QType::GGML_Q5_K}) decomposition(format);
        registered_problems();
        for (const QType format : {QType::GGML_Q4_K, QType::GGML_Q5_K}) cancelling(format);
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
    if (g_failures != 0) {
        std::cerr << g_failures << " failures\n";
        return 1;
    }
    std::cout << "GGML affine arithmetic: passed\n";
    return 0;
}
