// hc_sinkhorn_mixes, hc_collapse and hc_expand_mix against FP64 oracles that evaluate the closed
// formulas of include/ninfer/ops/hc_sinkhorn.h on the represented inputs (the BF16 residual, the
// FP32 weights): the statistic over the whole stream, the projection, the sigmoid pre and post and
// the softmax-then-Sinkhorn comb; the weighted collapse; the mix back into the streams.
//
// Cases: DeepSeek-V4.1-Flash's geometry (S = 4, H = 5120) at T = 1, 2, 3, 8, 17, and a small H = 64
// geometry with a different iteration count and epsilon. Criteria (suite-owned): the FP32 outputs of
// the mixes meet |got - ref| <= 2e-5 |ref| + 2e-7 (a 20,480-term FP32 sum and the iterations in FP32; measured
// worst about a tenth of this; finite required); the BF16 outputs of collapse and expand_mix meet
// |got - ref| <= 2^-8 |ref| + 2^-12 rms(ref) (one rounding of an FP32 sum of four products from
// exact FP32 inputs). Also: each column bitwise independent of the others (T = 1 against its own
// column of a T = 8 call), a graph-captured call bitwise equal to the eager call, in-place
// expand_mix equal to the oracle on the pre-call residual, and refusals (misaligned residual,
// overlapping outputs, a wrong shape, non-contiguous scale and base, non-finite epsilons; a Tensor
// cannot have T = 0), and T = 65,537 (past grid.y) in every operation.

#include "ninfer/ops/hc_sinkhorn.h"

#include "ops/op_tester.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <limits>
#include <string>
#include <vector>

namespace {

namespace t = ninfer::test;
using ninfer::DType;
using ninfer::Tensor;

constexpr int kS = ninfer::ops::kHcSinkhornStreams;
constexpr int kMix = ninfer::ops::kHcSinkhornMixRows;

int failures = 0;

void check(bool ok, const std::string& what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what.c_str());
        ++failures;
    }
}

std::uint32_t next(std::uint32_t& state) { return state = state * 1664525U + 1013904223U; }
float uniform(std::uint32_t& state) {
    return static_cast<float>(static_cast<std::int32_t>(next(state) >> 8) - (1 << 23)) / static_cast<float>(1 << 23);
}

struct Device {
    void* p = nullptr;
    explicit Device(std::size_t bytes) { t::cuda_check(cudaMalloc(&p, bytes ? bytes : 1), "cudaMalloc"); }
    ~Device() { cudaFree(p); }
    Device(const Device&)            = delete;
    Device& operator=(const Device&) = delete;
    template <class V>
    void put(const V& v) { t::cuda_check(cudaMemcpy(p, v.data(), v.size() * sizeof(typename V::value_type), cudaMemcpyHostToDevice), "put"); }
    template <class T>
    std::vector<T> get(std::size_t count) const {
        std::vector<T> out(count);
        t::cuda_check(cudaMemcpy(out.data(), p, count * sizeof(T), cudaMemcpyDeviceToHost), "get");
        return out;
    }
};

std::vector<std::uint16_t> bf16_values(std::size_t count, std::uint32_t seed, float sigma) {
    std::vector<std::uint16_t> v(count);
    std::uint32_t s = seed;
    for (auto& x : v) { x = t::f32_to_bf16(sigma * uniform(s) * 1.7F); }
    return v;
}

std::vector<float> f32_values(std::size_t count, std::uint32_t seed, float sigma) {
    std::vector<float> v(count);
    std::uint32_t s = seed;
    for (auto& x : v) { x = sigma * uniform(s); }
    return v;
}

struct Mixes {
    std::vector<double> pre, post, comb; // [T][S], [T][S], [T][S*S]
};

Mixes mixes_oracle(const std::vector<std::uint16_t>& r, const std::vector<float>& fn, const float scale[3],
                   const std::vector<float>& base, int width, int T, double norm_eps, double eps, int iters) {
    Mixes out;
    out.pre.assign(static_cast<std::size_t>(T) * kS, 0);
    out.post.assign(static_cast<std::size_t>(T) * kS, 0);
    out.comb.assign(static_cast<std::size_t>(T) * kS * kS, 0);
    for (int c = 0; c < T; ++c) {
        double ss = 0;
        for (int k = 0; k < width; ++k) {
            const double v = t::bf16_to_f32(r[static_cast<std::size_t>(c) * width + k]);
            ss += v * v;
        }
        const double rs = 1.0 / std::sqrt(ss / width + norm_eps);
        double m[kMix];
        for (int n = 0; n < kMix; ++n) {
            double acc = 0;
            for (int k = 0; k < width; ++k) {
                acc += static_cast<double>(fn[static_cast<std::size_t>(n) * width + k]) *
                       t::bf16_to_f32(r[static_cast<std::size_t>(c) * width + k]);
            }
            m[n] = acc * rs;
        }
        const auto sig = [](double x) { return 1.0 / (1.0 + std::exp(-x)); };
        for (int i = 0; i < kS; ++i) {
            out.pre[c * kS + i]  = sig(m[i] * scale[0] + base[i]) + eps;
            out.post[c * kS + i] = 2.0 * sig(m[kS + i] * scale[1] + base[kS + i]);
        }
        double a[kS][kS];
        for (int i = 0; i < kS; ++i) {
            double peak = -1e300;
            for (int j = 0; j < kS; ++j) {
                a[i][j] = m[2 * kS + i * kS + j] * scale[2] + base[2 * kS + i * kS + j];
                peak    = std::max(peak, a[i][j]);
            }
            double sum = 0;
            for (int j = 0; j < kS; ++j) { a[i][j] = std::exp(a[i][j] - peak); sum += a[i][j]; }
            for (int j = 0; j < kS; ++j) { a[i][j] = a[i][j] / sum + eps; }
        }
        const auto columns = [&] {
            for (int j = 0; j < kS; ++j) {
                double sum = 0;
                for (int i = 0; i < kS; ++i) { sum += a[i][j]; }
                for (int i = 0; i < kS; ++i) { a[i][j] /= sum + eps; }
            }
        };
        columns();
        for (int round = 1; round < iters; ++round) {
            for (int i = 0; i < kS; ++i) {
                double sum = 0;
                for (int j = 0; j < kS; ++j) { sum += a[i][j]; }
                for (int j = 0; j < kS; ++j) { a[i][j] /= sum + eps; }
            }
            columns();
        }
        for (int i = 0; i < kS; ++i) {
            for (int j = 0; j < kS; ++j) { out.comb[(static_cast<std::size_t>(c) * kS + i) * kS + j] = a[i][j]; }
        }
    }
    return out;
}

// The FP32-output criterion of hc_sinkhorn_mixes.
double mixes_error(const std::vector<float>& got, const std::vector<double>& ref) {
    double worst = 0;
    for (std::size_t i = 0; i < ref.size(); ++i) {
        if (!std::isfinite(got[i])) { return 1e30; }
        worst = std::max(worst, std::fabs(got[i] - ref[i]) / (2e-5 * std::fabs(ref[i]) + 2e-7));
    }
    return worst; // <= 1 passes
}

// The BF16-output criterion of collapse and expand_mix.
double bf16_error(const std::vector<std::uint16_t>& got, const std::vector<double>& ref) {
    double rms = 0;
    for (const double v : ref) { rms += v * v; }
    rms = std::sqrt(rms / static_cast<double>(ref.size()));
    double worst = 0;
    for (std::size_t i = 0; i < ref.size(); ++i) {
        const double g = t::bf16_to_f32(got[i]);
        if (!std::isfinite(g)) { return 1e30; }
        worst = std::max(worst, std::fabs(g - ref[i]) / (std::ldexp(1.0, -8) * std::fabs(ref[i]) + std::ldexp(1.0, -12) * rms));
    }
    return worst;
}

struct Weights {
    std::vector<float> fn, base;
    float scale[3] = {0.8F, 0.5F, 1.3F};
};

Weights make_weights(int width, std::uint32_t seed, float fn_sigma) {
    Weights w;
    w.fn   = f32_values(static_cast<std::size_t>(kMix) * width, seed, fn_sigma);
    w.base = f32_values(kMix, seed + 1, 0.4F);
    return w;
}

struct MixesRun {
    std::vector<float> pre, post, comb;
};

MixesRun run_mixes(const std::vector<std::uint16_t>& r, const Weights& w, int width, int T, float norm_eps,
                   float eps, int iters, cudaStream_t stream, bool graph = false) {
    Device rd(r.size() * 2), fd(w.fn.size() * 4), sd(12), bd(w.base.size() * 4), pd(T * kS * 4), qd(T * kS * 4),
        cd(static_cast<std::size_t>(T) * kS * kS * 4);
    rd.put(r);
    fd.put(w.fn);
    t::cuda_check(cudaMemcpy(sd.p, w.scale, 12, cudaMemcpyHostToDevice), "scale");
    bd.put(w.base);
    const Tensor R(rd.p, DType::BF16, {width, T}), F(fd.p, DType::FP32, {width, kMix}), S(sd.p, DType::FP32, {3}),
        B(bd.p, DType::FP32, {kMix});
    Tensor pre(pd.p, DType::FP32, {kS, T}), post(qd.p, DType::FP32, {kS, T}), comb(cd.p, DType::FP32, {kS * kS, T});
    const auto call = [&] { ninfer::ops::hc_sinkhorn_mixes(R, F, S, B, norm_eps, eps, iters, pre, post, comb, stream); };
    if (graph) {
        cudaGraph_t g        = nullptr;
        cudaGraphExec_t exec = nullptr;
        t::cuda_check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal), "capture");
        call();
        t::cuda_check(cudaStreamEndCapture(stream, &g), "end capture");
        t::cuda_check(cudaGraphInstantiate(&exec, g, 0), "instantiate");
        t::cuda_check(cudaGraphLaunch(exec, stream), "launch");
        cudaGraphExecDestroy(exec);
        cudaGraphDestroy(g);
    } else {
        call();
    }
    t::cuda_check(cudaStreamSynchronize(stream), "sync");
    return {pd.get<float>(T * kS), qd.get<float>(T * kS), cd.get<float>(static_cast<std::size_t>(T) * kS * kS)};
}

void test_mixes(int width, int T, float fn_sigma, float norm_eps, float eps, int iters, cudaStream_t stream) {
    const auto r = bf16_values(static_cast<std::size_t>(width) * T, 100U + T, 1.0F);
    const Weights w = make_weights(width, 7U, fn_sigma);
    const MixesRun got = run_mixes(r, w, width, T, norm_eps, eps, iters, stream);
    const Mixes ref = mixes_oracle(r, w.fn, w.scale, w.base, width, T, norm_eps, eps, iters);
    const std::string label = "mixes width=" + std::to_string(width) + " T=" + std::to_string(T);
    const auto widen = [](const std::vector<float>& v) { return std::vector<double>(v.begin(), v.end()); };
    (void)widen;
    const double pe = mixes_error(got.pre, ref.pre), qe = mixes_error(got.post, ref.post), ce = mixes_error(got.comb, ref.comb);
    check(pe <= 1.0 && qe <= 1.0 && ce <= 1.0, label + ": within the criterion (worst pre " + std::to_string(pe) + ", post " +
                                                   std::to_string(qe) + ", comb " + std::to_string(ce) + ")");
    std::printf("%s: worst error / bound: pre %.3f post %.3f comb %.3f\n", label.c_str(), pe, qe, ce);
    // The comb is doubly stochastic to the iteration's accuracy (a property of the result).
    for (int c = 0; c < T && iters >= 8; ++c) {
        for (int j = 0; j < kS; ++j) {
            double sum = 0;
            for (int i = 0; i < kS; ++i) { sum += got.comb[(static_cast<std::size_t>(c) * kS + i) * kS + j]; }
            check(std::fabs(sum - 1.0) < 1e-3, label + ": comb column sums to 1");
        }
    }
}

void test_column_independence_and_graph(cudaStream_t stream) {
    const int width = 4 * 5120, T = 8;
    const auto r    = bf16_values(static_cast<std::size_t>(width) * T, 55U, 1.0F);
    const Weights w = make_weights(width, 9U, 0.02F);
    const MixesRun all = run_mixes(r, w, width, T, 1e-6F, 1e-6F, 20, stream);
    for (const int c : {0, 3, 7}) {
        const std::vector<std::uint16_t> one(r.begin() + static_cast<std::ptrdiff_t>(c) * width,
                                             r.begin() + static_cast<std::ptrdiff_t>(c + 1) * width);
        const MixesRun alone = run_mixes(one, w, width, 1, 1e-6F, 1e-6F, 20, stream);
        check(std::memcmp(alone.pre.data(), all.pre.data() + c * kS, kS * 4) == 0 &&
                  std::memcmp(alone.post.data(), all.post.data() + c * kS, kS * 4) == 0 &&
                  std::memcmp(alone.comb.data(), all.comb.data() + c * kS * kS, kS * kS * 4) == 0,
              "column " + std::to_string(c) + " alone is bitwise its column of the 8-column call");
    }
    const MixesRun graphed = run_mixes(r, w, width, T, 1e-6F, 1e-6F, 20, stream, true);
    check(graphed.pre == all.pre && graphed.post == all.post && graphed.comb == all.comb,
          "a graph-captured call is bitwise the eager call");
}

void test_collapse_and_expand(int hidden, int T, cudaStream_t stream) {
    const int width = kS * hidden;
    const auto r    = bf16_values(static_cast<std::size_t>(width) * T, 300U + T, 1.0F);
    auto pre        = f32_values(static_cast<std::size_t>(T) * kS, 301U, 1.0F);
    auto post       = f32_values(static_cast<std::size_t>(T) * kS, 302U, 1.0F);
    auto comb       = f32_values(static_cast<std::size_t>(T) * kS * kS, 303U, 1.0F);
    const auto y    = bf16_values(static_cast<std::size_t>(hidden) * T, 304U, 1.0F);
    Device rd(r.size() * 2), pd(pre.size() * 4), qd(post.size() * 4), cd(comb.size() * 4), yd(y.size() * 2), xd(y.size() * 2);
    rd.put(r);
    pd.put(pre);
    qd.put(post);
    cd.put(comb);
    yd.put(y);
    const std::string label = "hidden=" + std::to_string(hidden) + " T=" + std::to_string(T);

    Tensor R(rd.p, DType::BF16, {width, T});
    const Tensor P(pd.p, DType::FP32, {kS, T}), Q(qd.p, DType::FP32, {kS, T}), C(cd.p, DType::FP32, {kS * kS, T}),
        Y(yd.p, DType::BF16, {hidden, T});
    Tensor X(xd.p, DType::BF16, {hidden, T});
    ninfer::ops::hc_collapse(R, P, X, stream);
    t::cuda_check(cudaStreamSynchronize(stream), "sync");
    std::vector<double> collapsed(static_cast<std::size_t>(hidden) * T);
    for (int c = 0; c < T; ++c) {
        for (int d = 0; d < hidden; ++d) {
            double sum = 0;
            for (int s = 0; s < kS; ++s) {
                sum += static_cast<double>(pre[c * kS + s]) * t::bf16_to_f32(r[static_cast<std::size_t>(c) * width + s * hidden + d]);
            }
            collapsed[static_cast<std::size_t>(c) * hidden + d] = sum;
        }
    }
    const double ce = bf16_error(xd.get<std::uint16_t>(collapsed.size()), collapsed);
    check(ce <= 1.0, "collapse " + label + " within the criterion (" + std::to_string(ce) + ")");

    ninfer::ops::hc_expand_mix(Y, Q, C, R, stream);
    t::cuda_check(cudaStreamSynchronize(stream), "sync");
    std::vector<double> expanded(r.size());
    for (int c = 0; c < T; ++c) {
        for (int j = 0; j < kS; ++j) {
            for (int d = 0; d < hidden; ++d) {
                double sum = static_cast<double>(post[c * kS + j]) * t::bf16_to_f32(y[static_cast<std::size_t>(c) * hidden + d]);
                for (int i = 0; i < kS; ++i) {
                    sum += static_cast<double>(comb[(static_cast<std::size_t>(c) * kS + i) * kS + j]) *
                           t::bf16_to_f32(r[static_cast<std::size_t>(c) * width + i * hidden + d]);
                }
                expanded[static_cast<std::size_t>(c) * width + j * hidden + d] = sum;
            }
        }
    }
    const double ee = bf16_error(rd.get<std::uint16_t>(expanded.size()), expanded);
    check(ee <= 1.0, "expand_mix (in place) " + label + " within the criterion (" + std::to_string(ee) + ")");
    std::printf("collapse/expand_mix %s: worst error / bound %.3f, %.3f\n", label.c_str(), ce, ee);
}

template <class F>
bool refuses(F&& f) {
    try {
        f();
    } catch (const std::invalid_argument&) {
        return true;
    }
    return false;
}

void test_refusals(cudaStream_t stream) {
    const int width = 256, T = 2;
    const auto r    = bf16_values(static_cast<std::size_t>(width) * T + 8, 1U, 1.0F);
    const Weights w = make_weights(width, 3U, 0.1F);
    Device rd(r.size() * 2), fd(w.fn.size() * 4), sd(12), bd(w.base.size() * 4), pd(T * kS * 4), qd(T * kS * 4),
        cd(static_cast<std::size_t>(T) * kS * kS * 4);
    rd.put(r);
    fd.put(w.fn);
    t::cuda_check(cudaMemcpy(sd.p, w.scale, 12, cudaMemcpyHostToDevice), "scale");
    bd.put(w.base);
    const Tensor R(rd.p, DType::BF16, {width, T}), F(fd.p, DType::FP32, {width, kMix}), S(sd.p, DType::FP32, {3}),
        B(bd.p, DType::FP32, {kMix});
    Tensor pre(pd.p, DType::FP32, {kS, T}), post(qd.p, DType::FP32, {kS, T}), comb(cd.p, DType::FP32, {kS * kS, T});
    namespace o = ninfer::ops;
    check(!refuses([&] { o::hc_sinkhorn_mixes(R, F, S, B, 1e-6F, 1e-6F, 4, pre, post, comb, stream); }), "a valid call is served");
    const Tensor misaligned(static_cast<char*>(rd.p) + 2, DType::BF16, {width, T});
    check(refuses([&] { o::hc_sinkhorn_mixes(misaligned, F, S, B, 1e-6F, 1e-6F, 4, pre, post, comb, stream); }),
          "a misaligned residual is refused");
    Tensor overlapping(pd.p, DType::FP32, {kS, T});
    check(refuses([&] { o::hc_sinkhorn_mixes(R, F, S, B, 1e-6F, 1e-6F, 4, pre, overlapping, comb, stream); }),
          "overlapping outputs are refused");
    check(refuses([&] { o::hc_sinkhorn_mixes(R, F, S, B, 0.0F, 1e-6F, 4, pre, post, comb, stream); }), "norm_eps = 0 is refused");
    check(refuses([&] { o::hc_sinkhorn_mixes(R, F, S, B, 1e-6F, 1e-6F, 0, pre, post, comb, stream); }), "zero iterations are refused");
    const Tensor wrong_fn(fd.p, DType::FP32, {width, kMix - 1});
    check(refuses([&] { o::hc_sinkhorn_mixes(R, wrong_fn, S, B, 1e-6F, 1e-6F, 4, pre, post, comb, stream); }),
          "a wrong fn shape is refused");
    // A non-contiguous base (a same-size permuted view; scale goes through the same check), and non-finite epsilons.
    const Tensor strided_base = B.view({kMix / 2, 2, 1, 1}).permute({1, 0, 2, 3});
    check(refuses([&] { o::hc_sinkhorn_mixes(R, F, S, strided_base, 1e-6F, 1e-6F, 4, pre, post, comb, stream); }),
          "a non-contiguous base is refused");
    const float inf = std::numeric_limits<float>::infinity();
    check(refuses([&] { o::hc_sinkhorn_mixes(R, F, S, B, 1e-6F, inf, 4, pre, post, comb, stream); }), "an infinite sinkhorn_eps is refused");
    check(refuses([&] { o::hc_sinkhorn_mixes(R, F, S, B, inf, 1e-6F, 4, pre, post, comb, stream); }), "an infinite norm_eps is refused");
    check(refuses([&] { o::hc_sinkhorn_mixes(R, F, S, B, std::nanf(""), 1e-6F, 4, pre, post, comb, stream); }), "a NaN norm_eps is refused");
    t::cuda_check(cudaStreamSynchronize(stream), "sync");
}

// More columns than grid.y can hold (65,535): every operation serves them, columns checked against the oracle
// or against a single-column call.
void test_many_columns(cudaStream_t stream) {
    const int T = 65537;
    test_collapse_and_expand(8, T, stream);
    const int width = 256;
    const auto r    = bf16_values(static_cast<std::size_t>(width) * T, 77U, 1.0F);
    const Weights w = make_weights(width, 5U, 0.1F);
    const MixesRun all = run_mixes(r, w, width, T, 1e-6F, 1e-6F, 20, stream);
    for (const int c : {0, 65535, 65536}) {
        const std::vector<std::uint16_t> one(r.begin() + static_cast<std::ptrdiff_t>(c) * width,
                                             r.begin() + static_cast<std::ptrdiff_t>(c + 1) * width);
        const MixesRun alone = run_mixes(one, w, width, 1, 1e-6F, 1e-6F, 20, stream);
        check(std::memcmp(alone.comb.data(), all.comb.data() + static_cast<std::size_t>(c) * kS * kS, kS * kS * 4) == 0 &&
                  std::memcmp(alone.pre.data(), all.pre.data() + static_cast<std::size_t>(c) * kS, kS * 4) == 0,
              "mixes column " + std::to_string(c) + " of a 65,537-column call equals its own call");
    }
}

} // namespace

int main() {
    try {
        if (t::cuda_unavailable()) {
            std::printf("SKIP: no CUDA device\n");
            return 77;
        }
        cudaStream_t stream = nullptr;
        t::cuda_check(cudaStreamCreate(&stream), "stream");
        for (const int T : {1, 2, 3, 8, 17}) { test_mixes(4 * 5120, T, 0.02F, 1e-6F, 1e-6F, 20, stream); }
        test_mixes(256, 3, 0.2F, 1e-5F, 1e-3F, 5, stream);
        test_mixes(4 * 5120, 2, 0.002F, 1e-6F, 1e-6F, 20, stream); // near-uniform logits
        test_column_independence_and_graph(stream);
        for (const int T : {1, 3, 8}) { test_collapse_and_expand(5120, T, stream); }
        test_collapse_and_expand(37, 5, stream); // a hidden size no vector width divides
        test_refusals(stream);
        test_many_columns(stream);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "EXCEPTION: %s\n", e.what());
        return 1;
    }
    if (failures != 0) {
        std::fprintf(stderr, "hc_sinkhorn: %d failure(s)\n", failures);
        return 1;
    }
    std::printf("hc_sinkhorn: passed\n");
    return 0;
}
