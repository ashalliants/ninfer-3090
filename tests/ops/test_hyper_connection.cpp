// Adapted from Infernix a3edb450 tests/ops/test_hyper_connection.cpp (Apache-2.0).
// Modified for NInfer-3090: BF16 weights only, the norm weight is an FP32 stored multiplier, the
// composed route is also checked past Linear's 512-column tier, and inject/expand get exact
// oracles.
//
// hyper_connection_mix against an FP64 oracle. The oracle evaluates the closed formula in FP64
// from the represented BF16 residual, the FP32 norm multiplier and the BF16 weights:
//
//   Rn = R * (mean R^2 + eps)^-1/2 * w per stream; z = W_down Rn; m = SiLU(z[:rank] / S);
//   inject = 2 sigmoid(z[rank:] / S); u = W_up m; x = (1/S) sum_s sigmoid(u[sH + d]) Rn[sH + d].
//
// Cases: Flash-Next's geometry (S = 4, H = 2560, rank 320; the block mixers carry 4 injection
// rows, the final mixer none) at T = 1, 4, 5, 8, 9, 16 (the fused route) and 17, 64, 513 (the
// composed route, across Linear's BF16 tiers); column invariance of the fused route (each column
// of a 16-column call bit-identical to that column alone, and a 5-column call equal to the
// 16-column call's first five); one graph-captured call bit-identical to the eager call; the
// workspace capacity covers every T. Criteria: the fused route rounds only x to BF16 (FP32
// intermediates), so |x - ref| <= 2^-8 |ref| + 2^-12 rms(ref), and injects to 1e-5 relative +
// 1e-6; the composed route rounds Rn, z, m and u to BF16, so its bound is 2^-5 |ref| + 2^-6
// rms(ref) and injects to 2^-7 relative + 2^-9.

#include "ninfer/ops/hyper_connection.h"

#include "ops/linear/linear_test_common.h"
#include "ops/op_tester.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

namespace t = ninfer::test;
using ninfer::DType;
using ninfer::Tensor;

constexpr int kS = 4, kH = 2560, kRank = 320, kW = kS * kH;
constexpr float kEps = 1e-6F;

int failures = 0;

void check(bool ok, const std::string& what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what.c_str());
        ++failures;
    }
}

std::uint32_t next(std::uint32_t& state) { return state = state * 1664525U + 1013904223U; }
float uniform(std::uint32_t& state) {
    return static_cast<float>(static_cast<std::int32_t>(next(state) >> 8) - (1 << 23)) /
           static_cast<float>(1 << 23);
}

struct Device {
    void* p = nullptr;
    explicit Device(std::size_t bytes) { t::cuda_check(cudaMalloc(&p, bytes), "cudaMalloc"); }
    ~Device() { cudaFree(p); }
    Device(const Device&)            = delete;
    Device& operator=(const Device&) = delete;
};

struct Reference {
    std::vector<double> x;      // [T][H]
    std::vector<double> inject; // [T][S]
};

Reference oracle(const std::vector<std::uint16_t>& r, const std::vector<float>& w,
                 const std::vector<float>& down, int down_rows, const std::vector<float>& up,
                 int T) {
    Reference out;
    out.x.assign(static_cast<std::size_t>(T) * kH, 0.0);
    out.inject.assign(static_cast<std::size_t>(T) * kS, 0.0);
    // Columns are independent; split them over host threads.
    const unsigned workers = std::max(1U, std::min<unsigned>(std::thread::hardware_concurrency(),
                                                             static_cast<unsigned>(T)));
    std::vector<std::thread> pool;
    for (unsigned worker = 0; worker < workers; ++worker) {
        pool.emplace_back([&, worker] {
            std::vector<double> rn(kW), z(down_rows), m(kRank);
            for (int c = static_cast<int>(worker); c < T; c += static_cast<int>(workers)) {
                for (int s = 0; s < kS; ++s) {
                    double ss = 0;
                    for (int d = 0; d < kH; ++d) {
                        const double v = t::bf16_to_f32(r[static_cast<std::size_t>(c) * kW + s * kH + d]);
                        ss += v * v;
                    }
                    const double inv = 1.0 / std::sqrt(ss / kH + kEps);
                    for (int d = 0; d < kH; ++d) {
                        const std::size_t i = static_cast<std::size_t>(s) * kH + d;
                        rn[i] = t::bf16_to_f32(r[static_cast<std::size_t>(c) * kW + i]) * inv *
                                static_cast<double>(w[i]);
                    }
                }
                for (int k = 0; k < down_rows; ++k) {
                    double acc = 0;
                    for (int i = 0; i < kW; ++i) {
                        acc += static_cast<double>(down[static_cast<std::size_t>(k) * kW + i]) * rn[i];
                    }
                    z[k] = acc / kS;
                }
                for (int k = 0; k < kRank; ++k) { m[k] = z[k] / (1.0 + std::exp(-z[k])); }
                for (int k = kRank; k < down_rows; ++k) {
                    out.inject[static_cast<std::size_t>(c) * kS + (k - kRank)] = 2.0 / (1.0 + std::exp(-z[k]));
                }
                for (int d = 0; d < kH; ++d) {
                    double sum = 0;
                    for (int s = 0; s < kS; ++s) {
                        const std::size_t row = static_cast<std::size_t>(s) * kH + d;
                        double u              = 0;
                        for (int k = 0; k < kRank; ++k) {
                            u += static_cast<double>(up[row * kRank + k]) * m[k];
                        }
                        sum += rn[row] / (1.0 + std::exp(-u));
                    }
                    out.x[static_cast<std::size_t>(c) * kH + d] = sum / kS;
                }
            }
        });
    }
    for (auto& thread : pool) { thread.join(); }
    return out;
}

// BF16 weights of the model's scale: per 32-element group a gain g in [2, 6), values uniform in
// (-1, 1) * g / sqrt(K), rounded to BF16 (the oracle reads the rounded values), so z / S and u stay
// of order one as in the model.
void randomize_bf16(t::quantized_weight::PackedWeight& w, std::uint32_t seed) {
    const int n = w.weight.n, k = w.weight.k;
    std::uint32_t state = seed;
    for (int r = 0; r < n; ++r) {
        double gain = 0;
        for (int c = 0; c < k; ++c) {
            if (c % 32 == 0) {
                gain = 2.0 + 4.0 * static_cast<double>(next(state) >> 8) / static_cast<double>(1 << 24);
            }
            const auto bits = t::f32_to_bf16(
                static_cast<float>(uniform(state) * gain / std::sqrt(static_cast<double>(k))));
            const std::size_t at = (static_cast<std::size_t>(r) * k + c) * 2;
            w.payload[at]        = static_cast<std::uint8_t>(bits & 0xFFU);
            w.payload[at + 1]    = static_cast<std::uint8_t>(bits >> 8);
        }
    }
}

std::vector<float> decode_bf16(const t::quantized_weight::PackedWeight& w) {
    std::vector<float> out(static_cast<std::size_t>(w.weight.n) * w.weight.k);
    for (std::size_t i = 0; i < out.size(); ++i) {
        out[i] = t::bf16_to_f32(
            static_cast<std::uint16_t>(w.payload[2 * i] | (w.payload[2 * i + 1] << 8)));
    }
    return out;
}

struct Problem {
    int down_rows = 0;
    t::quantized_weight::PackedWeight down, up;
    std::vector<float> down_values, up_values; // decoded independently
    std::unique_ptr<Device> down_payload, up_payload;
    ninfer::Weight down_w, up_w;
    std::vector<float> norm; // the stored multiplier, 1 + gamma
    std::unique_ptr<Device> norm_device;
};

Problem make_problem(bool inject) {
    Problem p;
    p.down_rows = kRank + (inject ? kS : 0);
    p.down      = t::linear::make_bf16_weight(p.down_rows, kW, inject ? 511U : 513U);
    p.up        = t::linear::make_bf16_weight(kW, kRank, 519U);
    randomize_bf16(p.down, inject ? 611U : 613U);
    randomize_bf16(p.up, 617U);
    p.down_values  = decode_bf16(p.down);
    p.up_values    = decode_bf16(p.up);
    p.down_payload = std::make_unique<Device>(p.down.payload.size());
    p.up_payload   = std::make_unique<Device>(p.up.payload.size());
    t::cuda_check(cudaMemcpy(p.down_payload->p, p.down.payload.data(), p.down.payload.size(),
                             cudaMemcpyHostToDevice),
                  "down");
    t::cuda_check(cudaMemcpy(p.up_payload->p, p.up.payload.data(), p.up.payload.size(),
                             cudaMemcpyHostToDevice),
                  "up");
    p.down_w            = p.down.device_weight(p.down_payload->p);
    p.up_w              = p.up.device_weight(p.up_payload->p);
    std::uint32_t state = 77U;
    p.norm.resize(kW);
    // As stored by the GGUF: FP32 1 + gamma with gamma a BF16 value.
    for (auto& v : p.norm) { v = 1.0F + t::bf16_to_f32(t::f32_to_bf16(0.25F * uniform(state))); }
    p.norm_device = std::make_unique<Device>(kW * 4);
    t::cuda_check(cudaMemcpy(p.norm_device->p, p.norm.data(), kW * 4, cudaMemcpyHostToDevice),
                  "norm");
    return p;
}

std::vector<std::uint16_t> residual(int T, std::uint32_t seed) {
    std::vector<std::uint16_t> r(static_cast<std::size_t>(T) * kW);
    std::uint32_t state = seed;
    for (std::size_t i = 0; i < r.size(); ++i) {
        // Streams at different scales, so the per-stream norms matter.
        const int s = static_cast<int>((i % kW) / kH);
        r[i]        = t::f32_to_bf16(uniform(state) * static_cast<float>(1 << s));
    }
    return r;
}

struct Result {
    std::vector<std::uint16_t> x;
    std::vector<float> inject;
};

Result run(Problem& p, const std::vector<std::uint16_t>& r, int T, bool inject,
           ninfer::WorkspaceArena& ws, cudaStream_t stream, bool graph = false) {
    Device rd(r.size() * 2), xd(static_cast<std::size_t>(kH) * T * 2),
        id(static_cast<std::size_t>(kS) * T * 4);
    t::cuda_check(cudaMemcpy(rd.p, r.data(), r.size() * 2, cudaMemcpyHostToDevice), "residual");
    const Tensor R(rd.p, DType::BF16, {kW, T});
    const Tensor w(p.norm_device->p, DType::FP32, {kW});
    Tensor x(xd.p, DType::BF16, {kH, T});
    Tensor inj(id.p, DType::FP32, {kS, T});
    const auto call = [&] {
        ninfer::ops::hyper_connection_mix(R, w, p.down_w, p.up_w, ninfer::ops::LinearPolicy::A16Only,
                                          kS, kRank, kEps, x, inject ? &inj : nullptr, ws, stream);
    };
    if (graph) {
        cudaGraph_t g        = nullptr;
        cudaGraphExec_t exec = nullptr;
        t::cuda_check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal), "capture");
        call();
        t::cuda_check(cudaStreamEndCapture(stream, &g), "end capture");
        t::cuda_check(cudaGraphInstantiate(&exec, g, 0), "instantiate");
        t::cuda_check(cudaGraphLaunch(exec, stream), "launch");
        t::cuda_check(cudaStreamSynchronize(stream), "sync");
        cudaGraphExecDestroy(exec);
        cudaGraphDestroy(g);
    } else {
        call();
        t::cuda_check(cudaStreamSynchronize(stream), "sync");
    }
    Result out;
    out.x.resize(static_cast<std::size_t>(kH) * T);
    t::cuda_check(cudaMemcpy(out.x.data(), xd.p, out.x.size() * 2, cudaMemcpyDeviceToHost), "x");
    if (inject) {
        out.inject.resize(static_cast<std::size_t>(kS) * T);
        t::cuda_check(cudaMemcpy(out.inject.data(), id.p, out.inject.size() * 4,
                                 cudaMemcpyDeviceToHost),
                      "inject");
    }
    return out;
}

void compare(const Result& got, const Reference& ref, bool fused, bool inject,
             const std::string& label) {
    double ss = 0;
    for (const double v : ref.x) { ss += v * v; }
    const double rms = std::sqrt(ss / static_cast<double>(ref.x.size()));
    const double rel = fused ? std::ldexp(1.0, -8) : std::ldexp(1.0, -5);
    // The composed route rounds Rn, z, m and u to BF16; where the S streams cancel, x is small
    // against their sizes, hence its absolute floor.
    const double abs = rms * (fused ? std::ldexp(1.0, -12) : std::ldexp(1.0, -6));
    double worst     = 0;
    std::size_t bad  = 0;
    for (std::size_t i = 0; i < ref.x.size(); ++i) {
        const double v = static_cast<double>(t::bf16_to_f32(got.x[i]));
        const double e = std::fabs(v - ref.x[i]);
        if (!std::isfinite(v)) { ++bad; continue; }
        worst = std::max(worst, e / (rel * std::fabs(ref.x[i]) + abs));
        bad += e > rel * std::fabs(ref.x[i]) + abs ? 1U : 0U;
    }
    std::printf("%s: x worst %.3f of the bound\n", label.c_str(), worst);
    check(bad == 0, label + ": x within its bound (" + std::to_string(bad) + " outside, worst " +
                        std::to_string(worst) + " of the bound)");
    if (inject) {
        // inject lies in (0, 2): a relative bound plus an absolute floor where it saturates.
        const double irel = fused ? 1e-5 : std::ldexp(1.0, -7);
        const double iabs = fused ? 1e-6 : std::ldexp(1.0, -9);
        double iworst     = 0;
        for (std::size_t i = 0; i < ref.inject.size(); ++i) {
            const double e = std::fabs(got.inject[i] - ref.inject[i]);
            iworst = std::max(iworst, std::isfinite(got.inject[i]) ? e / (irel * std::fabs(ref.inject[i]) + iabs)
                                                                   : 1e30);
        }
        check(iworst <= 1.0, label + ": inject within its bound (worst " + std::to_string(iworst) +
                                 " of the bound)");
    }
}

// hyper_connection_inject and hyper_connection_expand: exact against the formula with one FP32
// multiply-add (inject) or a copy (expand).
void inject_and_expand(cudaStream_t stream) {
    for (const int T : {1, 5, 64}) {
        std::uint32_t state = 900U + static_cast<std::uint32_t>(T);
        std::vector<std::uint16_t> y(static_cast<std::size_t>(kH) * T), r(static_cast<std::size_t>(kW) * T);
        std::vector<float> inj(static_cast<std::size_t>(kS) * T);
        for (auto& v : y) { v = t::f32_to_bf16(uniform(state)); }
        for (auto& v : r) { v = t::f32_to_bf16(4.0F * uniform(state)); }
        for (auto& v : inj) { v = 1.0F + uniform(state); }
        Device yd(y.size() * 2), rd(r.size() * 2), id(inj.size() * 4);
        t::cuda_check(cudaMemcpy(yd.p, y.data(), y.size() * 2, cudaMemcpyHostToDevice), "y");
        t::cuda_check(cudaMemcpy(rd.p, r.data(), r.size() * 2, cudaMemcpyHostToDevice), "r");
        t::cuda_check(cudaMemcpy(id.p, inj.data(), inj.size() * 4, cudaMemcpyHostToDevice), "inj");
        Tensor Y(yd.p, DType::BF16, {kH, T}), R(rd.p, DType::BF16, {kW, T}), I(id.p, DType::FP32, {kS, T});
        ninfer::ops::hyper_connection_inject(Y, I, R, stream);
        t::cuda_check(cudaStreamSynchronize(stream), "sync");
        std::vector<std::uint16_t> got(r.size()), want(r.size());
        t::cuda_check(cudaMemcpy(got.data(), rd.p, got.size() * 2, cudaMemcpyDeviceToHost), "got");
        for (int c = 0; c < T; ++c) {
            for (int s = 0; s < kS; ++s) {
                for (int d = 0; d < kH; ++d) {
                    const std::size_t i = static_cast<std::size_t>(c) * kW + s * kH + d;
                    // One FP32 fma-free multiply-add: the kernel may contract it, so accept either.
                    const float a = t::bf16_to_f32(r[i]);
                    const float b = inj[static_cast<std::size_t>(c) * kS + s];
                    const float v = t::bf16_to_f32(y[static_cast<std::size_t>(c) * kH + d]);
                    const std::uint16_t fused   = t::f32_to_bf16(std::fma(b, v, a));
                    const std::uint16_t unfused = t::f32_to_bf16(a + b * v);
                    want[i] = got[i] == unfused ? unfused : fused;
                }
            }
        }
        check(got == want, "inject T=" + std::to_string(T) + " equals the formula");
        ninfer::ops::hyper_connection_expand(Y, kS, R, stream);
        t::cuda_check(cudaStreamSynchronize(stream), "sync");
        t::cuda_check(cudaMemcpy(got.data(), rd.p, got.size() * 2, cudaMemcpyDeviceToHost), "got");
        bool same = true;
        for (int c = 0; c < T; ++c) {
            for (int s = 0; s < kS; ++s) {
                for (int d = 0; d < kH; ++d) {
                    same = same && got[static_cast<std::size_t>(c) * kW + s * kH + d] ==
                                       y[static_cast<std::size_t>(c) * kH + d];
                }
            }
        }
        check(same, "expand T=" + std::to_string(T) + " copies x to every stream");
    }
}

} // namespace

int main() {
    try {
        if (!t::linear::cuda_available()) {
            std::printf("SKIP: no CUDA device\n");
            return 77;
        }
        cudaStream_t stream = nullptr;
        t::cuda_check(cudaStreamCreate(&stream), "stream");
        for (const bool inject : {true, false}) {
            Problem p = make_problem(inject);
            const std::size_t capacity = ninfer::ops::hyper_connection_mix_workspace_capacity_bytes(
                p.down_w, p.up_w, ninfer::ops::LinearPolicy::A16Only, kS, kRank, 513);
            ninfer::WorkspaceArena ws(capacity);
            const std::string tag = inject ? "inject" : "no inject";
            for (const int T : {1, 4, 5, 8, 9, 16, 17, 64, 513}) {
                if (T == 513 && !inject) { continue; } // one composed case past Linear's 512 tier
                const auto r     = residual(T, 1000U + static_cast<std::uint32_t>(T));
                const Result got = run(p, r, T, inject, ws, stream);
                compare(got, oracle(r, p.norm, p.down_values, p.down_rows, p.up_values, T), T <= 16,
                        inject, tag + " T=" + std::to_string(T));
                check(ws.peak_used() <= capacity,
                      tag + ": the workspace capacity covers T=" + std::to_string(T));
            }
            // Column invariance of the fused route.
            const auto r16   = residual(16, 4242U);
            const Result all = run(p, r16, 16, inject, ws, stream);
            bool same        = true;
            for (int c = 0; c < 16 && same; ++c) {
                const std::vector<std::uint16_t> one(r16.begin() + static_cast<std::ptrdiff_t>(c) * kW,
                                                     r16.begin() + static_cast<std::ptrdiff_t>(c + 1) * kW);
                const Result alone = run(p, one, 1, inject, ws, stream);
                same = std::equal(alone.x.begin(), alone.x.end(),
                                  all.x.begin() + static_cast<std::ptrdiff_t>(c) * kH) &&
                       (!inject || std::equal(alone.inject.begin(), alone.inject.end(),
                                              all.inject.begin() + static_cast<std::ptrdiff_t>(c) * kS));
            }
            check(same, tag + ": each column of a 16-column call equals that column alone");
            const std::vector<std::uint16_t> r5(r16.begin(), r16.begin() + 5 * kW);
            const Result five = run(p, r5, 5, inject, ws, stream);
            check(std::equal(five.x.begin(), five.x.end(), all.x.begin()),
                  tag + ": a 5-column call equals the first five");
            const Result captured = run(p, r5, 5, inject, ws, stream, true);
            check(captured.x == five.x && captured.inject == five.inject,
                  tag + ": a graph-captured call equals the eager call");
            // The composed route under capture as well.
            const auto r17          = residual(17, 77U);
            const Result eager17    = run(p, r17, 17, inject, ws, stream);
            const Result captured17 = run(p, r17, 17, inject, ws, stream, true);
            check(captured17.x == eager17.x && captured17.inject == eager17.inject,
                  tag + ": a graph-captured composed call equals the eager call");
        }
        // Shapes that disagree are refused.
        {
            Problem p = make_problem(true);
            ninfer::WorkspaceArena ws(1 << 20);
            Device rd(static_cast<std::size_t>(kW) * 2), xd(kH * 2);
            const Tensor R(rd.p, DType::BF16, {kW, 1});
            const Tensor w(p.norm_device->p, DType::FP32, {kW});
            Tensor x(xd.p, DType::BF16, {kH, 1});
            bool threw = false;
            try {
                // The down projection carries injection rows the call does not ask for.
                ninfer::ops::hyper_connection_mix(R, w, p.down_w, p.up_w,
                                                  ninfer::ops::LinearPolicy::A16Only, kS, kRank,
                                                  kEps, x, nullptr, ws, stream);
            } catch (const std::invalid_argument&) { threw = true; }
            check(threw, "a down projection with unused injection rows is refused");
            threw = false;
            try {
                const Tensor bf16_norm(p.norm_device->p, DType::BF16, {kW});
                ninfer::ops::hyper_connection_mix(R, bf16_norm, p.down_w, p.up_w,
                                                  ninfer::ops::LinearPolicy::A16Only, kS, kRank,
                                                  kEps, x, nullptr, ws, stream);
            } catch (const std::invalid_argument&) { threw = true; }
            check(threw, "a BF16 norm weight is refused");
        }
        inject_and_expand(stream);
        cudaStreamDestroy(stream);
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL: %s\n", error.what());
        ++failures;
    }
    std::printf(failures == 0 ? "hyper_connection checks passed\n" : "FAIL: %d checks failed\n",
                failures);
    return failures == 0 ? 0 : 1;
}
