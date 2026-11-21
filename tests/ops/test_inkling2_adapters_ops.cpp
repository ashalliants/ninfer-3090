// Written for NInfer-3090. low_rank_update_fp32 and direction_ablate_fp32 (include/ninfer/ops/inkling2_adapters.h)
// against FP64 references on random operands at Inkling-2-Mini's shapes (K 3072, N and C 2304), ranks 1, 2 and 16,
// columns at a padded stride; the ablation with a residual, at a block boundary (in place), and with w = 0.

#include "core/arena.h"
#include "core/tensor.h"
#include "ninfer/ops/inkling2_adapters.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using ninfer::DeviceBuffer;
using ninfer::DType;
using ninfer::Tensor;

int failures = 0;

void check(bool ok, const std::string& what) {
    std::printf("%s: %s\n", ok ? "ok  " : "FAIL", what.c_str());
    if (!ok) { ++failures; }
}

std::uint16_t to_bf16(float f) {
    std::uint32_t u;
    std::memcpy(&u, &f, 4);
    return static_cast<std::uint16_t>((u + 0x7FFFU + ((u >> 16) & 1U)) >> 16);
}
double from_bf16(std::uint16_t h) {
    const std::uint32_t u = static_cast<std::uint32_t>(h) << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

// A [rows, cols] tensor at column stride `stride` (>= rows) over `buffer`.
Tensor strided(const DeviceBuffer& buffer, DType dtype, std::int32_t rows, std::int32_t cols, std::int32_t stride) {
    Tensor t(buffer.p, dtype, {rows, cols});
    t.nb[1] = static_cast<std::int64_t>(stride) * static_cast<std::int64_t>(ninfer::dtype_size(dtype));
    t.nb[2] = t.nb[1] * cols, t.nb[3] = t.nb[2];
    return t;
}

std::string rel(double x) {
    char text[32];
    std::snprintf(text, sizeof text, "%.2e", x);
    return text;
}

std::vector<float> host(const DeviceBuffer& b) {
    std::vector<float> out(b.bytes / 4);
    b.copy_to_host(out.data(), b.bytes);
    return out;
}

} // namespace

int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
        std::printf("SKIP: no usable CUDA device\n");
        return 77;
    }
    try {
        std::mt19937 rng(11);
        std::normal_distribution<float> normal(0.0F, 1.0F);
        constexpr std::int32_t K = 3072, N = 2304, T = 9, stride_x = K + 64, stride_y = N + 32;
        for (const std::int32_t rank : {1, 2, 16}) {
            const float scale = rank == 2 ? 1.0F : 0.5F;
            std::vector<float> a(static_cast<std::size_t>(rank) * K), b(static_cast<std::size_t>(N) * rank), y0(static_cast<std::size_t>(stride_y) * T);
            std::vector<std::uint16_t> x(static_cast<std::size_t>(stride_x) * T);
            for (auto& v : a) { v = 0.02F * normal(rng); }
            for (auto& v : b) { v = 0.05F * normal(rng); }
            for (auto& v : y0) { v = normal(rng); }
            for (auto& v : x) { v = to_bf16(normal(rng)); }
            DeviceBuffer ad(a.size() * 4), bd(b.size() * 4), xd(x.size() * 2), yd(y0.size() * 4);
            ad.copy_from_host(a.data(), ad.bytes), bd.copy_from_host(b.data(), bd.bytes), xd.copy_from_host(x.data(), xd.bytes);
            yd.copy_from_host(y0.data(), yd.bytes);
            const Tensor xt = strided(xd, DType::BF16, K, T, stride_x);
            Tensor yt       = strided(yd, DType::FP32, N, T, stride_y);
            ninfer::ops::low_rank_update_fp32(xt, static_cast<const float*>(ad.p), static_cast<const float*>(bd.p), rank, scale, yt, nullptr);
            if (cudaDeviceSynchronize() != cudaSuccess) { throw std::runtime_error("low_rank_update_fp32 failed"); }
            const auto got = host(yd);
            double err = 0, mag = 0;
            bool padding = true;
            for (std::int32_t t = 0; t < T; ++t) {
                std::vector<double> u(static_cast<std::size_t>(rank), 0.0);
                for (std::int32_t j = 0; j < rank; ++j) {
                    for (std::int32_t k = 0; k < K; ++k) { u[j] += double(a[std::size_t(j) * K + k]) * from_bf16(x[std::size_t(t) * stride_x + k]); }
                    u[j] *= scale;
                }
                for (std::int32_t n = 0; n < N; ++n) {
                    double d = 0;
                    for (std::int32_t j = 0; j < rank; ++j) { d += double(b[std::size_t(n) * rank + j]) * u[j]; }
                    const std::size_t i = std::size_t(t) * stride_y + n;
                    err = std::max(err, std::abs(got[i] - (y0[i] + d)));
                    mag = std::max(mag, std::abs(y0[i] + d));
                }
                for (std::int32_t n = N; n < stride_y; ++n) { padding = padding && got[std::size_t(t) * stride_y + n] == y0[std::size_t(t) * stride_y + n]; }
            }
            check(err / mag < 2e-6 && padding, "low_rank_update_fp32 rank " + std::to_string(rank) + ": rel error " + rel(err / mag));
        }
        constexpr std::int32_t C = 2304;
        std::vector<float> r(C);
        double norm = 0;
        for (auto& v : r) { v = normal(rng), norm += double(v) * v; }
        for (auto& v : r) { v = static_cast<float>(v / std::sqrt(norm)); }
        DeviceBuffer rd(C * 4);
        rd.copy_from_host(r.data(), rd.bytes);
        for (const int mode : {0, 1, 2}) { // with a residual, in place (a block boundary), w = 0
            const float w = mode == 2 ? 0.0F : 0.87F;
            std::vector<float> y0(static_cast<std::size_t>(stride_y) * T), res(y0.size());
            for (auto& v : y0) { v = 2.0F * normal(rng); }
            for (auto& v : res) { v = 5.0F * normal(rng); }
            DeviceBuffer yd(y0.size() * 4), resd(res.size() * 4), sd(y0.size() * 4);
            yd.copy_from_host(y0.data(), yd.bytes), resd.copy_from_host(res.data(), resd.bytes);
            Tensor yt = strided(yd, DType::FP32, C, T, stride_y), rt = strided(resd, DType::FP32, C, T, stride_y);
            Tensor st = strided(mode == 1 ? yd : sd, DType::FP32, C, T, stride_y);
            ninfer::ops::direction_ablate_fp32(yt, static_cast<const float*>(rd.p), w, mode == 1 ? nullptr : &rt, st, nullptr);
            if (cudaDeviceSynchronize() != cudaSuccess) { throw std::runtime_error("direction_ablate_fp32 failed"); }
            const auto got = host(mode == 1 ? yd : sd), y_after = host(yd);
            double err = 0, mag = 0, residue = 0;
            bool exact = true;
            for (std::int32_t t = 0; t < T; ++t) {
                double d = 0;
                for (std::int32_t c = 0; c < C; ++c) { d += double(r[c]) * y0[std::size_t(t) * stride_y + c]; }
                double dot_after = 0;
                for (std::int32_t c = 0; c < C; ++c) {
                    const std::size_t i = std::size_t(t) * stride_y + c;
                    const double ablated = y0[i] - double(w) * d * r[c];
                    const double want    = ablated + (mode == 1 ? 0.0 : res[i]);
                    err = std::max(err, std::abs(got[i] - want)), mag = std::max(mag, std::abs(want));
                    dot_after += double(r[c]) * y_after[i];
                    if (mode == 2) { exact = exact && got[i] == res[i] + y0[i]; }
                }
                residue = std::max(residue, std::abs(dot_after - (1.0 - w) * d));
            }
            const char* name = mode == 0 ? "with the residual" : mode == 1 ? "in place" : "w = 0 (the plain residual add)";
            check(err / mag < 2e-6 && residue < 1e-4 && exact, std::string("direction_ablate_fp32 ") + name + ": rel error " + rel(err / mag));
        }
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL: %s\n", error.what());
        return 1;
    }
    std::printf(failures ? "FAILED\n" : "PASS\n");
    return failures ? 1 : 0;
}
