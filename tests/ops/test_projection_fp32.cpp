// Adapted from Infernix a3edb450 tests/ops/linear/test_projection_fp32.cpp (Apache-2.0).
// Modified for NInfer-3090: the BF16 cases are kept (the q8_g32_fp16/q4_g64_fp16 and BF16 tall
// and tensor-core head cases are not ported, nor are those routes), and GGML IQ4_XS head cases are
// new.
//
// projection_fp32 against the FP64 product of the represented weights and activations, within the
// documented FP32 accumulation bound sum_k |w x| * K * 2^-24, plus its column invariance: a
// column's bits do not depend on the batch width or the column's position. For the IQ4_XS head
// the weight is decoded by the test-owned exact decoder (ops/ggml_blocks_decode.h) and x passes
// through an independent host copy of the Q8_1 cast, the contract's activation cast.

#include "ninfer/ops/projection_fp32.h"

#include "core/arena.h"
#include "core/weight.h"
#include "ops/ggml_blocks_decode.h"
#include "ops/op_tester.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <iostream>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {
using ninfer::DType;
using ninfer::QType;
using ninfer::Tensor;
using ninfer::test::bf16_to_f32;
using ninfer::test::f32_to_bf16;

int g_failures = 0;

void check(bool ok, const std::string& what) {
    if (!ok) {
        std::cerr << "FAIL: " << what << "\n";
        ++g_failures;
    }
}

template <class T> T* device(const std::vector<T>& v) {
    T* p = nullptr;
    if (cudaMalloc(&p, std::max<std::size_t>(1, v.size()) * sizeof(T)) != cudaSuccess ||
        cudaMemcpy(p, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice) != cudaSuccess) {
        throw std::runtime_error("device copy failed");
    }
    return p;
}

// x: BF16 bits [t][k] and their values.
void random_activations(std::int32_t k, std::int32_t t, std::uint32_t seed,
                        std::vector<std::uint16_t>& bits, std::vector<float>& values) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> n(0.0F, 1.0F);
    bits.resize(static_cast<std::size_t>(k) * t);
    values.resize(bits.size());
    for (std::size_t i = 0; i < bits.size(); ++i) {
        bits[i]   = f32_to_bf16(n(rng));
        values[i] = bf16_to_f32(bits[i]);
    }
}

// w(r, i): the represented weight value of row r, column i.
using WeightAt = std::function<double(std::int32_t, std::int32_t)>;

// Checks out [t][n] against the FP64 product on `rows` (all rows when empty).
void verify(const std::string& tag, const std::vector<float>& out, const WeightAt& w,
            const std::vector<float>& x, std::int32_t n, std::int32_t k, std::int32_t t,
            const std::vector<std::int32_t>& rows) {
    std::vector<std::int32_t> all;
    if (rows.empty()) {
        for (std::int32_t r = 0; r < n; ++r) { all.push_back(r); }
    }
    const auto& list = rows.empty() ? all : rows;
    const unsigned workers = std::max(1U, std::min(16U, std::thread::hardware_concurrency()));
    std::vector<long> bad(workers, 0);
    std::vector<double> worst(workers, 0.0);
    std::vector<std::thread> pool;
    for (unsigned id = 0; id < workers; ++id) {
        pool.emplace_back([&, id] {
            for (std::size_t j = id; j < list.size(); j += workers) {
                const std::int32_t r = list[j];
                for (std::int32_t c = 0; c < t; ++c) {
                    double ref = 0, mag = 0;
                    for (std::int32_t i = 0; i < k; ++i) {
                        const double p = w(r, i) * static_cast<double>(x[static_cast<std::size_t>(c) * k + i]);
                        ref += p;
                        mag += std::fabs(p);
                    }
                    const double got   = out[static_cast<std::size_t>(c) * n + r];
                    const double bound = mag * k * std::ldexp(1.0, -24) + 1e-30;
                    worst[id] = std::max(worst[id], std::fabs(got - ref) / bound);
                    if (!(std::fabs(got - ref) <= bound)) { ++bad[id]; }
                }
            }
        });
    }
    for (auto& thread : pool) { thread.join(); }
    long outside = 0;
    double max_ratio = 0;
    for (unsigned id = 0; id < workers; ++id) {
        outside += bad[id];
        max_ratio = std::max(max_ratio, worst[id]);
    }
    std::cout << tag << ": worst error " << max_ratio << " of the bound, " << outside << " outside\n";
    check(outside == 0, tag + " within the FP32 accumulation bound");
}

// Segments of `rows` rows each, concatenated in order (one to four weights).
void bf16_case(const std::vector<std::int32_t>& rows, std::int32_t k, std::uint32_t seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> d(0.0F, 0.05F);
    std::vector<float> wv;
    std::vector<std::uint16_t*> devices;
    std::vector<Tensor> tensors;
    std::int32_t n = 0;
    std::string label;
    for (const std::int32_t segment : rows) {
        std::vector<std::uint16_t> part(static_cast<std::size_t>(segment) * k);
        for (auto& v : part) { v = f32_to_bf16(d(rng)); }
        for (auto v : part) { wv.push_back(bf16_to_f32(v)); }
        devices.push_back(device(part));
        tensors.push_back(Tensor(devices.back(), DType::BF16, {k, segment}));
        n += segment;
        label += (label.empty() ? "" : "+") + std::to_string(segment);
    }
    std::vector<const Tensor*> weights;
    for (const auto& t : tensors) { weights.push_back(&t); }
    const WeightAt w = [&](std::int32_t r, std::int32_t i) {
        return static_cast<double>(wv[static_cast<std::size_t>(r) * k + i]);
    };
    std::vector<std::uint16_t> wide_bits;
    std::vector<float> wide_values;
    random_activations(k, 17, seed + 7, wide_bits, wide_values);
    std::vector<float> single;
    // Decode widths (every T up to 8) take the one-row-per-warp mapping, wider calls the four-row
    // one; a column's bits must not depend on either.
    for (const std::int32_t t : {1, 2, 3, 4, 5, 6, 7, 8, 9, 17}) {
        auto* dx = device(std::vector<std::uint16_t>(
            wide_bits.begin(), wide_bits.begin() + static_cast<std::ptrdiff_t>(k) * t));
        float* dout = nullptr;
        cudaMalloc(&dout, sizeof(float) * n * t);
        Tensor x(dx, DType::BF16, {k, t}), out(dout, DType::FP32, {n, t});
        ninfer::ops::projection_fp32(x, weights, out, nullptr);
        if (cudaDeviceSynchronize() != cudaSuccess) { throw std::runtime_error("projection bf16 failed"); }
        std::vector<float> host(static_cast<std::size_t>(n) * t);
        cudaMemcpy(host.data(), dout, host.size() * sizeof(float), cudaMemcpyDeviceToHost);
        const std::vector<float> values(wide_values.begin(),
                                        wide_values.begin() + static_cast<std::ptrdiff_t>(k) * t);
        verify("bf16 N=" + label + " K=" + std::to_string(k) + " T=" + std::to_string(t), host, w,
               values, n, k, t, {});
        if (t == 1) {
            single = host;
        } else {
            check(std::memcmp(single.data(), host.data(), sizeof(float) * n) == 0,
                  "bf16 column invariance N=" + std::to_string(n) + " T=" + std::to_string(t));
        }
        cudaFree(dx);
        cudaFree(dout);
    }
    for (auto* p : devices) { cudaFree(p); }
}

// Prefill widths take the wide mapping (64 columns and more): every column's bits must equal the
// narrow mapping's for the same column (calls of at most 63 columns), and the first and last
// columns meet the FP64 bound.
void bf16_wide_case(const std::vector<std::int32_t>& rows, std::int32_t k, std::int32_t t,
                    std::uint32_t seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> d(0.0F, 0.05F);
    std::vector<float> wv;
    std::vector<std::uint16_t*> devices;
    std::vector<Tensor> tensors;
    std::int32_t n = 0;
    for (const std::int32_t segment : rows) {
        std::vector<std::uint16_t> part(static_cast<std::size_t>(segment) * k);
        for (auto& v : part) { v = f32_to_bf16(d(rng)); }
        for (auto v : part) { wv.push_back(bf16_to_f32(v)); }
        devices.push_back(device(part));
        tensors.push_back(Tensor(devices.back(), DType::BF16, {k, segment}));
        n += segment;
    }
    std::vector<const Tensor*> weights;
    for (const auto& w : tensors) { weights.push_back(&w); }
    std::vector<std::uint16_t> bits;
    std::vector<float> values;
    random_activations(k, t, seed + 3, bits, values);
    auto* dx    = device(bits);
    float* dout = nullptr;
    cudaMalloc(&dout, sizeof(float) * n * t);
    const auto run = [&](std::int32_t first, std::int32_t width) {
        Tensor x(dx + static_cast<std::size_t>(first) * k, DType::BF16, {k, width});
        Tensor out(dout + static_cast<std::size_t>(first) * n, DType::FP32, {n, width});
        ninfer::ops::projection_fp32(x, weights, out, nullptr);
        if (cudaDeviceSynchronize() != cudaSuccess) { throw std::runtime_error("projection bf16 wide failed"); }
        std::vector<float> host(static_cast<std::size_t>(n) * width);
        cudaMemcpy(host.data(), out.data, host.size() * sizeof(float), cudaMemcpyDeviceToHost);
        return host;
    };
    const std::vector<float> wide = run(0, t);
    long differing = 0;
    for (std::int32_t first = 0; first < t; first += 63) {
        const std::int32_t width        = std::min(63, t - first);
        const std::vector<float> narrow = run(first, width);
        for (std::int32_t c = 0; c < width; ++c) {
            if (std::memcmp(narrow.data() + static_cast<std::size_t>(c) * n,
                            wide.data() + static_cast<std::size_t>(first + c) * n, sizeof(float) * n) != 0) {
                ++differing;
            }
        }
    }
    const std::string label =
        "bf16 wide N=" + std::to_string(n) + " K=" + std::to_string(k) + " T=" + std::to_string(t);
    std::cout << label << ": " << differing << " columns differ from the narrow mapping\n";
    check(differing == 0, label + " bitwise equal to the narrow mapping");
    const WeightAt w = [&](std::int32_t r, std::int32_t i) {
        return static_cast<double>(wv[static_cast<std::size_t>(r) * k + i]);
    };
    verify(label + " (first columns)", wide, w, values, n, k, 3, {});
    const std::vector<float> tail(wide.end() - static_cast<std::ptrdiff_t>(n) * 3, wide.end());
    const std::vector<float> tail_x(values.end() - static_cast<std::ptrdiff_t>(k) * 3, values.end());
    verify(label + " (last columns)", tail, w, tail_x, n, k, 3, {});
    cudaFree(dx);
    cudaFree(dout);
    for (auto* p : devices) { cudaFree(p); }
}

// --- the IQ4_XS head -------------------------------------------------------------------------------

// The contract's Q8_1 cast of BF16 values [t][k], host side: per 32 values d = amax / 127 and
// q = rne(x * (127 / amax)); returns the represented values q * d.
std::vector<float> cast_q81(const std::vector<std::uint16_t>& x) {
    std::vector<float> out(x.size());
    for (std::size_t g = 0; g < x.size() / 32; ++g) {
        float amax = 0.0F;
        for (int i = 0; i < 32; ++i) amax = std::max(amax, std::fabs(bf16_to_f32(x[32 * g + i])));
        const float inverse = amax == 0.0F ? 0.0F : 127.0F / amax;
        const float d       = amax / 127.0F;
        for (int i = 0; i < 32; ++i) {
            const float q = std::nearbyint(bf16_to_f32(x[32 * g + i]) * inverse);
            out[32 * g + i] = static_cast<float>(static_cast<double>(q) * static_cast<double>(d));
        }
    }
    return out;
}

// Random IQ4_XS blocks (every code byte is valid) with finite FP16 super-scales of either sign.
std::vector<std::uint8_t> iq4_xs_blocks(std::int32_t n, std::int32_t k, std::uint32_t seed) {
    const ninfer::GgmlBlock block = ninfer::ggml_block(QType::GGML_IQ4_XS);
    const std::size_t blocks      = static_cast<std::size_t>(n) * (k / block.values);
    std::vector<std::uint8_t> bytes(blocks * block.bytes);
    std::mt19937 rng(seed);
    for (auto& b : bytes) b = static_cast<std::uint8_t>(rng());
    for (std::size_t i = 0; i < blocks; ++i) {
        const auto h = static_cast<std::uint16_t>(((rng() & 1U) << 15) |
                                                  ((static_cast<unsigned>(-14 + static_cast<int>(rng() % 4)) + 15U) << 10) |
                                                  (rng() & 0x3FFU));
        std::memcpy(&bytes[i * block.bytes], &h, 2);
    }
    return bytes;
}

ninfer::Weight iq4_xs_weight(std::int32_t n, std::int32_t k, void* data) {
    const ninfer::GgmlBlock block = ninfer::ggml_block(QType::GGML_IQ4_XS);
    ninfer::Weight w;
    w.payload       = data;
    w.payload_bytes = static_cast<std::uint64_t>(n) * (k / block.values) * block.bytes;
    w.qtype         = QType::GGML_IQ4_XS;
    w.layout        = ninfer::QuantLayout::GgmlBlocks;
    w.ndim          = 2;
    w.n = w.shape[0] = w.padded_shape[0] = n;
    w.k = w.shape[1] = w.padded_shape[1] = k;
    w.qdata                              = data;
    w.group      = static_cast<std::int32_t>(block.values);
    w.group_size = block.values;
    return w;
}

// N rows of K = 2560; `sampled` checks every 977th row and the last against FP64 (the vocabulary
// head's 248,320 rows), otherwise every row.
void iq4_xs_case(std::int32_t n, std::uint32_t seed, bool sampled) {
    constexpr std::int32_t k = 2560;
    const auto bytes         = iq4_xs_blocks(n, k, seed);
    auto* dw                 = device(bytes);
    const ninfer::Weight weight = iq4_xs_weight(n, k, dw);
    std::vector<std::int32_t> rows;
    if (sampled) {
        for (std::int32_t r = 0; r < n; r += 977) { rows.push_back(r); }
        rows.push_back(n - 1);
    } else {
        for (std::int32_t r = 0; r < n; ++r) { rows.push_back(r); }
    }
    // The decoded rows the oracle reads (only those checked).
    const std::size_t row_bytes = static_cast<std::size_t>(k) / 256 * 136;
    std::vector<float> decoded(static_cast<std::size_t>(n) * (sampled ? 0 : k));
    std::vector<std::vector<float>> sampled_rows;
    std::vector<std::int32_t> index_of(sampled ? n : 0, -1);
    for (const std::int32_t r : rows) {
        const auto values = ninfer::test::ggml::decode_blocks(
            QType::GGML_IQ4_XS,
            std::as_bytes(std::span<const std::uint8_t>(bytes.data() + r * row_bytes, row_bytes)));
        if (sampled) {
            index_of[r] = static_cast<std::int32_t>(sampled_rows.size());
            sampled_rows.push_back(values);
        } else {
            std::copy(values.begin(), values.end(), decoded.begin() + static_cast<std::ptrdiff_t>(r) * k);
        }
    }
    const WeightAt w = [&](std::int32_t r, std::int32_t i) {
        return sampled ? static_cast<double>(sampled_rows[index_of[r]][i])
                       : static_cast<double>(decoded[static_cast<std::size_t>(r) * k + i]);
    };
    std::vector<std::uint16_t> wide_bits;
    std::vector<float> unused;
    random_activations(k, 17, seed + 1, wide_bits, unused);
    // One all-zero 32-value group per column exercises the cast's amax = 0 rule.
    for (std::int32_t c = 0; c < 17; ++c) {
        for (int i = 0; i < 32; ++i) wide_bits[static_cast<std::size_t>(c) * k + 64 + i] = 0;
    }
    std::vector<float> single;
    for (const std::int32_t t : {1, 2, 7, 8, 9, 16, 17}) {
        const std::vector<std::uint16_t> bits(wide_bits.begin(),
                                              wide_bits.begin() + static_cast<std::ptrdiff_t>(k) * t);
        auto* dx    = device(bits);
        float* dout = nullptr;
        cudaMalloc(&dout, sizeof(float) * static_cast<std::size_t>(n) * t);
        Tensor x(dx, DType::BF16, {k, t}), out(dout, DType::FP32, {n, t});
        ninfer::ops::projection_fp32(x, weight, out, nullptr);
        if (cudaDeviceSynchronize() != cudaSuccess) { throw std::runtime_error("projection iq4_xs failed"); }
        std::vector<float> host(static_cast<std::size_t>(n) * t);
        cudaMemcpy(host.data(), dout, host.size() * sizeof(float), cudaMemcpyDeviceToHost);
        verify("iq4_xs N=" + std::to_string(n) + " T=" + std::to_string(t), host, w, cast_q81(bits), n,
               k, t, rows);
        if (t == 1) {
            single = host;
        } else {
            // Column 0 and the last column each equal a single-column call of the same x.
            bool same = std::memcmp(single.data(), host.data(), sizeof(float) * n) == 0;
            Tensor last_x(dx + static_cast<std::size_t>(t - 1) * k, DType::BF16, {k, 1});
            float* dlast = nullptr;
            cudaMalloc(&dlast, sizeof(float) * n);
            Tensor last_out(dlast, DType::FP32, {n, 1});
            ninfer::ops::projection_fp32(last_x, weight, last_out, nullptr);
            cudaDeviceSynchronize();
            std::vector<float> last(n);
            cudaMemcpy(last.data(), dlast, sizeof(float) * n, cudaMemcpyDeviceToHost);
            same = same && std::memcmp(last.data(), host.data() + static_cast<std::size_t>(t - 1) * n,
                                       sizeof(float) * n) == 0;
            cudaFree(dlast);
            check(same, "iq4_xs column invariance (first and last column) T=" + std::to_string(t));
        }
        cudaFree(dx);
        cudaFree(dout);
    }
    // An odd N and another K are refused.
    for (const auto& [bad_n, bad_k] : {std::pair{n - 1, k}, std::pair{n, 6144}}) {
        bool threw = false;
        try {
            auto* dx = device(std::vector<std::uint16_t>(static_cast<std::size_t>(bad_k), 0));
            float* dout = nullptr;
            cudaMalloc(&dout, sizeof(float) * bad_n);
            Tensor x(dx, DType::BF16, {bad_k, 1}), out(dout, DType::FP32, {bad_n, 1});
            const ninfer::Weight bad = iq4_xs_weight(bad_n, bad_k, dw);
            try {
                ninfer::ops::projection_fp32(x, bad, out, nullptr);
            } catch (const std::invalid_argument&) { threw = true; }
            cudaFree(dx);
            cudaFree(dout);
        } catch (const std::runtime_error&) {}
        check(threw, "iq4_xs N=" + std::to_string(bad_n) + " K=" + std::to_string(bad_k) + " is refused");
    }
    cudaFree(dw);
}

} // namespace

int main() {
    if (ninfer::test::cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    try {
        bf16_case({512, 1}, 2560, 11); // router rows and the shared-expert gate, concatenated
        bf16_case({1000, 37}, 2560, 13);
        bf16_case({129}, 2560, 14);           // one segment
        bf16_case({7, 300, 1, 64}, 1024, 15); // four segments, one of a single row
        bf16_wide_case({512, 1}, 2560, 64, 31);         // the narrowest wide call
        bf16_wide_case({512, 1}, 2560, 4096, 37);       // a 4096-token prefill chunk's router
        bf16_wide_case({7, 300, 1, 64}, 3072, 301, 41); // K at its limit, partial CTA range and pass
        bf16_wide_case({129}, 1024, 135, 43);
        iq4_xs_case(1000, 51, false);
        iq4_xs_case(248320, 53, true); // the Flash-Next LM head, sampled rows
    } catch (const std::exception& e) {
        std::cerr << "FAIL: " << e.what() << "\n";
        return 1;
    }
    if (g_failures != 0) {
        std::cerr << g_failures << " check(s) failed\n";
        return 1;
    }
    std::cout << "OK projection_fp32\n";
    return 0;
}
