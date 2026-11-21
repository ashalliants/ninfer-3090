// Written for NInfer-3090. quantize_q8_g32 against a host copy of ggml's quantize_row_q8_0_ref (the codes and FP16
// scales exactly), and projection_q8_fp32 against the FP64 product of the decoded weights and x within the
// documented FP32 bound sum_k |w x| * K * 2^-24, plus its column invariance (a column's bits do not depend on T or
// its position) and the segment concatenation.

#include "ninfer/ops/projection_fp32.h"

#include "core/tensor.h"
#include "ops/op_tester.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using ninfer::DType;
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
template <class T> std::vector<T> host(const T* p, std::size_t n) {
    std::vector<T> v(n);
    if (cudaMemcpy(v.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost) != cudaSuccess) { throw std::runtime_error("host copy failed"); }
    return v;
}

struct Encoded {
    std::vector<std::int8_t> q;
    std::vector<std::uint16_t> d; // FP16 bits
    std::vector<double> w;        // decoded
};

// ggml's quantize_row_q8_0_ref, row by row.
Encoded encode(const std::vector<std::uint16_t>& w, int k, int rows) {
    Encoded e{std::vector<std::int8_t>(w.size()), std::vector<std::uint16_t>(w.size() / 32), std::vector<double>(w.size())};
    for (std::size_t b = 0; b < w.size() / 32; ++b) {
        float amax = 0.0F;
        for (int i = 0; i < 32; ++i) { amax = std::max(amax, std::fabs(bf16_to_f32(w[b * 32 + i]))); }
        const float d  = amax / 127.0F;
        const float id = d != 0.0F ? 1.0F / d : 0.0F;
        const __half h = __float2half_rn(d);
        std::memcpy(&e.d[b], &h, 2);
        for (int i = 0; i < 32; ++i) {
            e.q[b * 32 + i] = static_cast<std::int8_t>(std::round(bf16_to_f32(w[b * 32 + i]) * id));
            e.w[b * 32 + i] = static_cast<double>(__half2float(h)) * e.q[b * 32 + i];
        }
    }
    (void)k, (void)rows;
    return e;
}

void run(int k, std::vector<int> rows, int columns, std::uint32_t seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> normal(0.0F, 1.0F);
    std::vector<std::vector<std::uint16_t>> weights;
    std::vector<Encoded> encoded;
    std::vector<std::int8_t*> dq;
    std::vector<__half*> dd;
    std::vector<ninfer::ops::Q8Weight> q8;
    int total = 0;
    for (std::size_t s = 0; s < rows.size(); ++s) {
        std::vector<std::uint16_t> w(static_cast<std::size_t>(k) * rows[s]);
        for (std::size_t i = 0; i < w.size(); ++i) {
            // Mixed magnitudes, an all-zero block and an outlier per row.
            float v = normal(rng) * (i % 97 == 0 ? 20.0F : 0.05F);
            if (i / 32 == 3) { v = 0.0F; }
            w[i] = f32_to_bf16(v);
        }
        auto* dw = device(w);
        Tensor tw(dw, DType::BF16, {k, rows[s]});
        std::int8_t* values = nullptr;
        __half* scales      = nullptr;
        cudaMalloc(&values, w.size());
        cudaMalloc(&scales, w.size() / 32 * 2);
        ninfer::ops::quantize_q8_g32(tw, values, scales, nullptr);
        const auto e  = encode(w, k, rows[s]);
        const auto gq = host(values, w.size());
        std::vector<std::uint16_t> gd(w.size() / 32);
        cudaMemcpy(gd.data(), scales, gd.size() * 2, cudaMemcpyDeviceToHost);
        check(gq == e.q && gd == e.d, "quantize_q8_g32 matches ggml's quantize_row_q8_0_ref (K " + std::to_string(k) + ")");
        cudaFree(dw);
        weights.push_back(w), encoded.push_back(e), dq.push_back(values), dd.push_back(scales);
        q8.push_back({values, scales, k, rows[s]});
        total += rows[s];
    }
    std::vector<std::uint16_t> x(static_cast<std::size_t>(k) * columns);
    for (auto& v : x) { v = f32_to_bf16(normal(rng)); }
    auto* dx = device(x);
    float* dout = nullptr;
    cudaMalloc(&dout, static_cast<std::size_t>(total) * columns * 4);
    Tensor tx(dx, DType::BF16, {k, columns}), tout(dout, DType::FP32, {total, columns});
    ninfer::ops::projection_q8_fp32(tx, q8, tout, nullptr);
    const auto out = host(dout, static_cast<std::size_t>(total) * columns);
    double worst = 0.0;
    int base = 0;
    for (std::size_t s = 0; s < rows.size(); ++s) {
        for (int n = 0; n < rows[s]; ++n) {
            for (int t = 0; t < columns; ++t) {
                double exact = 0.0, bound = 0.0;
                for (int i = 0; i < k; ++i) {
                    const double p = encoded[s].w[static_cast<std::size_t>(n) * k + i] * bf16_to_f32(x[static_cast<std::size_t>(t) * k + i]);
                    exact += p, bound += std::fabs(p);
                }
                bound = bound * k * std::ldexp(1.0, -24) + 1e-30;
                const double err = std::fabs(out[static_cast<std::size_t>(t) * total + base + n] - exact);
                worst = std::max(worst, err / bound);
            }
        }
        base += rows[s];
    }
    check(worst <= 1.0, "projection_q8_fp32 within the FP32 bound (K " + std::to_string(k) + ", T " + std::to_string(columns) + ")");
    // Column invariance: each column alone gives the same bits.
    bool same = true;
    for (int t = 0; t < columns; ++t) {
        Tensor one(dx + static_cast<std::size_t>(t) * k, DType::BF16, {k, 1}), single(dout, DType::FP32, {total, 1});
        ninfer::ops::projection_q8_fp32(one, q8, single, nullptr);
        const auto got = host(dout, static_cast<std::size_t>(total));
        same = same && std::memcmp(got.data(), out.data() + static_cast<std::size_t>(t) * total, got.size() * 4) == 0;
    }
    check(same, "projection_q8_fp32 columns do not depend on T or position");
    std::cout << "projection_q8_fp32 K " << k << " segments " << rows.size() << " T " << columns << ": worst error " << worst
              << " of the bound\n";
    cudaFree(dx), cudaFree(dout);
    for (auto* p : dq) { cudaFree(p); }
    for (auto* p : dd) { cudaFree(p); }
}

// T past one launch's 65535 * 8 columns: the columns either side of the split match a one-column call bit for bit
// and the FP64 dot product within the FP32 bound.
void run_wide() {
    const int k = 32, rows = 5, columns = 65535 * 8 + 3;
    std::mt19937 rng(6);
    std::normal_distribution<float> normal(0.0F, 1.0F);
    std::vector<std::uint16_t> w(static_cast<std::size_t>(k) * rows), x(static_cast<std::size_t>(k) * columns);
    for (auto& v : w) { v = f32_to_bf16(normal(rng) * 0.05F); }
    for (auto& v : x) { v = f32_to_bf16(normal(rng)); }
    auto* dw = device(w);
    std::int8_t* values = nullptr;
    __half* scales      = nullptr;
    cudaMalloc(&values, w.size());
    cudaMalloc(&scales, w.size() / 32 * 2);
    ninfer::ops::quantize_q8_g32(Tensor(dw, DType::BF16, {k, rows}), values, scales, nullptr);
    const auto e = encode(w, k, rows);
    const ninfer::ops::Q8Weight q8{values, scales, k, rows};
    auto* dx    = device(x);
    float* dout = nullptr;
    cudaMalloc(&dout, static_cast<std::size_t>(rows) * columns * 4);
    Tensor tx(dx, DType::BF16, {k, columns}), tout(dout, DType::FP32, {rows, columns});
    ninfer::ops::projection_q8_fp32(tx, std::span(&q8, 1), tout, nullptr);
    const auto out = host(dout, static_cast<std::size_t>(rows) * columns);
    bool same = true;
    double worst = 0.0;
    float* single = nullptr;
    cudaMalloc(&single, static_cast<std::size_t>(rows) * 4);
    for (const int t : {0, 65535 * 8 - 1, 65535 * 8, columns - 1}) {
        Tensor one(dx + static_cast<std::size_t>(t) * k, DType::BF16, {k, 1}), to(single, DType::FP32, {rows, 1});
        ninfer::ops::projection_q8_fp32(one, std::span(&q8, 1), to, nullptr);
        const auto got = host(single, static_cast<std::size_t>(rows));
        same = same && std::memcmp(got.data(), out.data() + static_cast<std::size_t>(t) * rows, got.size() * 4) == 0;
        for (int n = 0; n < rows; ++n) {
            double exact = 0.0, bound = 0.0;
            for (int i = 0; i < k; ++i) {
                const double p = e.w[static_cast<std::size_t>(n) * k + i] * bf16_to_f32(x[static_cast<std::size_t>(t) * k + i]);
                exact += p, bound += std::fabs(p);
            }
            bound = bound * k * std::ldexp(1.0, -24) + 1e-30;
            worst = std::max(worst, std::fabs(out[static_cast<std::size_t>(t) * rows + n] - exact) / bound);
        }
    }
    check(same && worst <= 1.0, "projection_q8_fp32 past one launch's columns (T " + std::to_string(columns) + ")");
    std::cout << "projection_q8_fp32 T " << columns << " (two launches): the split's columns are a single call's bits, worst error "
              << worst << " of the bound\n";
    cudaFree(dw), cudaFree(values), cudaFree(scales), cudaFree(dx), cudaFree(dout), cudaFree(single);
}

// NINFER_Q8_BENCH: T = 1 timings of Inkling-2's decode shapes (BF16 projection_fp32 against projection_q8_fp32).
void bench() {
    const int shapes[][2] = {{2304, 6562}, {2304, 2048}, {2304, 1024}, {3072, 2304}, {2304, 201024}};
    cudaEvent_t a, b;
    cudaEventCreate(&a), cudaEventCreate(&b);
    for (const auto& [k, n] : shapes) {
        std::vector<std::uint16_t> w(static_cast<std::size_t>(k) * n, f32_to_bf16(0.01F)), x(k, f32_to_bf16(1.0F));
        auto* dw = device(w);
        auto* dx = device(x);
        std::int8_t* values = nullptr;
        __half* scales      = nullptr;
        float* out          = nullptr;
        cudaMalloc(&values, w.size()), cudaMalloc(&scales, w.size() / 16), cudaMalloc(&out, static_cast<std::size_t>(n) * 4);
        Tensor tw(dw, DType::BF16, {k, n}), tx(dx, DType::BF16, {k, 1}), to(out, DType::FP32, {n, 1});
        ninfer::ops::quantize_q8_g32(tw, values, scales, nullptr);
        const ninfer::ops::Q8Weight q8{values, scales, k, n};
        const Tensor* ws[] = {&tw};
        const auto time = [&](auto&& f) {
            for (int i = 0; i < 5; ++i) { f(); }
            cudaEventRecord(a);
            for (int i = 0; i < 50; ++i) { f(); }
            cudaEventRecord(b);
            cudaEventSynchronize(b);
            float ms = 0;
            cudaEventElapsedTime(&ms, a, b);
            return ms * 1e3 / 50;
        };
        const double bf = time([&] { ninfer::ops::projection_fp32(tx, ws, to, nullptr); });
        const double q  = time([&] { ninfer::ops::projection_q8_fp32(tx, std::span(&q8, 1), to, nullptr); });
        std::printf("K %d N %d: BF16 %.1f us (%.0f GB/s), Q8 %.1f us (%.0f GB/s)\n", k, n, bf, 2.0 * k * n / bf / 1e3, q,
                    (k * n * 1.0625) / q / 1e3);
        cudaFree(dw), cudaFree(dx), cudaFree(values), cudaFree(scales), cudaFree(out);
    }
}

} // namespace

int main() {
    if (std::getenv("NINFER_Q8_BENCH") != nullptr) {
        bench();
        return 0;
    }
    try {
        run(2304, {37}, 1, 1);
        run(2304, {64, 33, 64, 5}, 3, 2);
        run(3072, {70}, 8, 3);
        run(2304, {29, 40}, 19, 4); // past one CTA's 8 columns
        run(32, {5}, 2, 5);
        run_wide();
        // Refusals.
        bool refused = false;
        try {
            Tensor x(nullptr, DType::BF16, {48, 1}), out(nullptr, DType::FP32, {1, 1});
            ninfer::ops::Q8Weight w{};
            ninfer::ops::projection_q8_fp32(x, std::span(&w, 1), out, nullptr);
        } catch (const std::invalid_argument&) { refused = true; }
        check(refused, "projection_q8_fp32 refuses a bad call");
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << "\n";
        return 1;
    }
    std::cout << (g_failures ? "FAILED\n" : "PASS\n");
    return g_failures ? 1 : 0;
}
