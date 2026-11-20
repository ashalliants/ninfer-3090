// rope_pairs against an FP64 oracle of its formula (include/ninfer/ops/rope_pairs.h): adjacent
// element pairs of the last `rotary_dim` elements of each head rotated by position * inv_freq[i],
// forward or inverse, the leading elements untouched.
//
// Cases: DeepSeek-V4.1-Flash's query (D = 512, H = 64, R = 64) and kv (H = 1) geometries with a plain
// frequency table (theta 10,000) and a YaRN-blended table (theta 40,000, factor 16, original length
// 65,536, beta 32 and 1 as the model repo's `precompute_freqs_cis`) at positions from 0 to past a
// million; both directions; R = D (the whole head); R = 2; a small odd geometry. Criteria
// (suite-owned): rotated pairs |got - ref| <= half a BF16 ulp of ref + 1e-6 |z| where |z| is the
// pair's magnitude (one BF16 rounding of an FP32 evaluation with FP32 cos and sin of an FP64
// angle); unrotated elements bit-identical to the input. Also forward then inverse returning the
// input to within two roundings, a graph-captured call bitwise equal to the eager call, and refusals.

#include "ninfer/ops/rope_pairs.h"

#include "ops/op_tester.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <string>
#include <vector>

namespace {

namespace t = ninfer::test;
using ninfer::DType;
using ninfer::Tensor;

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
};

constexpr double kPi = 3.141592653589793238462643383279502884;

// The model repo's `precompute_freqs_cis` frequencies (inference/model.py) in double: plain, or YaRN when original_len > 0.
std::vector<float> frequencies(int rotary, double base, double factor, int original_len, double beta_fast, double beta_slow) {
    std::vector<double> f(rotary / 2);
    for (int i = 0; i < rotary / 2; ++i) { f[i] = 1.0 / std::pow(base, 2.0 * i / rotary); }
    if (original_len > 0) {
        const auto corrected = [&](double rotations) {
            return rotary * std::log(original_len / (rotations * 2 * kPi)) / (2 * std::log(base));
        };
        const double low  = std::max(std::floor(corrected(beta_fast)), 0.0);
        const double high = std::min(std::ceil(corrected(beta_slow)), static_cast<double>(rotary - 1));
        for (int i = 0; i < rotary / 2; ++i) {
            const double ramp   = std::clamp((i - low) / std::max(high - low, 1e-3), 0.0, 1.0);
            const double smooth = 1.0 - ramp;
            f[i] = f[i] / factor * (1 - smooth) + f[i] * smooth;
        }
    }
    return std::vector<float>(f.begin(), f.end());
}

struct Problem {
    int D, H, R;
    bool inverse;
    std::vector<std::uint16_t> x;
    std::vector<std::int32_t> positions;
    std::vector<float> freq;
};

Problem make_problem(int D, int H, int R, bool inverse, const std::vector<std::int32_t>& positions, std::vector<float> freq,
                     std::uint32_t seed) {
    Problem p{D, H, R, inverse, {}, positions, std::move(freq)};
    std::uint32_t s = seed;
    p.x.resize(static_cast<std::size_t>(D) * H * positions.size());
    for (auto& v : p.x) { v = t::f32_to_bf16(uniform(s) * 3.0F); }
    return p;
}

std::vector<std::uint16_t> run(const Problem& p, cudaStream_t stream, bool graph = false) {
    const int T = static_cast<int>(p.positions.size());
    Device xd(p.x.size() * 2), pd(p.positions.size() * 4), fd(p.freq.size() * 4);
    xd.put(p.x);
    pd.put(p.positions);
    fd.put(p.freq);
    const Tensor positions(pd.p, DType::I32, {T}), freq(fd.p, DType::FP32, {p.R / 2});
    Tensor x(xd.p, DType::BF16, {p.D, p.H, T});
    const auto call = [&] { ninfer::ops::rope_pairs(positions, freq, p.R, p.inverse, x, stream); };
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
    std::vector<std::uint16_t> got(p.x.size());
    t::cuda_check(cudaMemcpy(got.data(), xd.p, got.size() * 2, cudaMemcpyDeviceToHost), "get");
    return got;
}

void verify(const Problem& p, const std::vector<std::uint16_t>& got, const std::string& label) {
    double worst = 0;
    const int T = static_cast<int>(p.positions.size());
    for (int c = 0; c < T; ++c) {
        for (int h = 0; h < p.H; ++h) {
            const std::size_t at = (static_cast<std::size_t>(c) * p.H + h) * p.D;
            for (int d = 0; d < p.D - p.R; ++d) {
                if (got[at + d] != p.x[at + d]) {
                    check(false, label + ": an unrotated element changed");
                    return;
                }
            }
            for (int i = 0; i < p.R / 2; ++i) {
                const std::size_t e = at + (p.D - p.R) + 2 * i;
                const double a = t::bf16_to_f32(p.x[e]), b = t::bf16_to_f32(p.x[e + 1]);
                const double angle = static_cast<double>(p.positions[c]) * static_cast<double>(p.freq[i]);
                const double cs = std::cos(angle), sn = p.inverse ? -std::sin(angle) : std::sin(angle);
                const double ref[2] = {a * cs - b * sn, a * sn + b * cs};
                const double magnitude = std::hypot(a, b);
                for (int k = 0; k < 2; ++k) {
                    const float g = t::bf16_to_f32(got[e + k]);
                    if (!std::isfinite(g)) {
                        check(false, label + ": a non-finite output");
                        return;
                    }
                    const double half_ulp = ref[k] == 0.0 ? 0.0 : std::ldexp(1.0, std::ilogb(std::fabs(ref[k])) - 8);
                    worst = std::max(worst, std::fabs(g - ref[k]) / (half_ulp + 1e-6 * magnitude + 1e-30));
                }
            }
        }
    }
    check(worst <= 1.0, label + ": within the criterion (worst " + std::to_string(worst) + " of the bound)");
    std::printf("%s: worst error / bound %.3f\n", label.c_str(), worst);
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

} // namespace

int main() {
    try {
        if (t::cuda_unavailable()) {
            std::printf("SKIP: no CUDA device\n");
            return 77;
        }
        cudaStream_t stream = nullptr;
        t::cuda_check(cudaStreamCreate(&stream), "stream");
        const std::vector<std::int32_t> spread = {0, 1, 2, 127, 128, 4095, 65535, 65536, 300000, 1048575};
        const std::vector<float> plain = frequencies(64, 10000.0, 1.0, 0, 32, 1);
        const std::vector<float> yarn  = frequencies(64, 40000.0, 16.0, 65536, 32, 1);
        for (const bool inverse : {false, true}) {
            const std::string dir = inverse ? " inverse" : " forward";
            verify(make_problem(512, 64, 64, inverse, spread, plain, 11U), run(make_problem(512, 64, 64, inverse, spread, plain, 11U), stream),
                   "query D=512 H=64 R=64 plain" + dir);
            verify(make_problem(512, 64, 64, inverse, spread, yarn, 12U), run(make_problem(512, 64, 64, inverse, spread, yarn, 12U), stream),
                   "query D=512 H=64 R=64 YaRN" + dir);
            verify(make_problem(512, 1, 64, inverse, spread, yarn, 13U), run(make_problem(512, 1, 64, inverse, spread, yarn, 13U), stream),
                   "kv D=512 H=1 R=64 YaRN" + dir);
        }
        {
            const Problem p = make_problem(64, 3, 64, false, {0, 5, 77}, frequencies(64, 10000.0, 1.0, 0, 32, 1), 21U);
            verify(p, run(p, stream), "whole head R = D = 64");
            const Problem q = make_problem(96, 5, 2, false, {3, 9}, {0.37F}, 22U);
            verify(q, run(q, stream), "R = 2, odd head count");
        }

        // Forward then inverse returns the input to within the two BF16 roundings.
        {
            Problem fwd = make_problem(512, 4, 64, false, spread, yarn, 31U);
            const std::vector<std::uint16_t> rotated = run(fwd, stream);
            Problem back = fwd;
            back.inverse = true;
            back.x       = rotated;
            const std::vector<std::uint16_t> restored = run(back, stream);
            double worst = 0;
            for (std::size_t i = 0; i < restored.size(); ++i) {
                const double want = t::bf16_to_f32(fwd.x[i]), have = t::bf16_to_f32(restored[i]);
                worst = std::max(worst, std::fabs(have - want) / (std::fabs(want) * std::ldexp(1.0, -7) + 0.02));
            }
            check(worst <= 1.0, "forward then inverse restores the input (worst " + std::to_string(worst) + ")");
        }

        {
            const Problem p = make_problem(512, 64, 64, false, spread, yarn, 41U);
            check(run(p, stream) == run(p, stream, true), "a graph-captured call is bitwise the eager call");
        }

        // Refusals.
        {
            const Problem p = make_problem(128, 4, 64, false, {1, 2}, plain, 51U);
            Device xd(p.x.size() * 2), pd(8), fd(p.freq.size() * 4);
            const Tensor positions(pd.p, DType::I32, {2}), freq(fd.p, DType::FP32, {32});
            Tensor x(xd.p, DType::BF16, {128, 4, 2});
            namespace o = ninfer::ops;
            check(!refuses([&] { o::rope_pairs(positions, freq, 64, false, x, stream); }), "a valid call is served");
            check(refuses([&] { o::rope_pairs(positions, freq, 63, false, x, stream); }), "an odd rotary_dim is refused");
            check(refuses([&] { o::rope_pairs(positions, freq, 130, false, x, stream); }), "rotary_dim past D is refused");
            check(refuses([&] { o::rope_pairs(positions, freq, 0, false, x, stream); }), "rotary_dim 0 is refused");
            const Tensor short_freq(fd.p, DType::FP32, {31});
            check(refuses([&] { o::rope_pairs(positions, short_freq, 64, false, x, stream); }), "a frequency table of the wrong length is refused");
            const Tensor wrong_positions(pd.p, DType::I32, {1});
            check(refuses([&] { o::rope_pairs(wrong_positions, freq, 64, false, x, stream); }), "positions for another column count are refused");
            {
                Device odd_x(63 * 2 * 2 * 2);
                Tensor odd(odd_x.p, DType::BF16, {63, 2, 2});
                check(refuses([&] { o::rope_pairs(positions, freq, 62, false, odd, stream); }), "an odd head dimension is refused");
            }
            const Tensor float_positions(pd.p, DType::FP32, {2});
            check(refuses([&] { o::rope_pairs(float_positions, freq, 64, false, x, stream); }), "non-I32 positions are refused");
            Tensor aliased(pd.p, DType::BF16, {2, 1, 2});
            const Tensor one_freq(fd.p, DType::FP32, {1});
            check(refuses([&] { o::rope_pairs(positions, one_freq, 2, false, aliased, stream); }), "x overlapping positions is refused");
            t::cuda_check(cudaStreamSynchronize(stream), "sync");
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "EXCEPTION: %s\n", e.what());
        return 1;
    }
    if (failures != 0) {
        std::fprintf(stderr, "rope_pairs: %d failure(s)\n", failures);
        return 1;
    }
    std::printf("rope_pairs: passed\n");
    return 0;
}
