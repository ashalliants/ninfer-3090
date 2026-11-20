// engram_gate_inject against an FP64 oracle of its formula (include/ninfer/ops/engram.h):
// DeepSeek-V4's Engram gate (a normalised dot product of each stream against its key, signed square
// root, sigmoid) and the write of the shared value into every stream.
//
// Cases: V4.1-Flash's geometry (S = 4 streams of H = 5120) at T = 1, 3 and 64 with FP32 and BF16
// gate weights (the GGUF stores FP32); other shapes (H = 8, S = 1, H = 16384, H = 648 off the block
// size); stream magnitudes far from one (the norms divide them out); a zero key and a zero stream
// (the gate clamps at |dot| = 1e-6, so g = 1e-3); a key opposite the stream (negative dot).
//
// Criterion (suite-owned), per output element: |got - ref| <= half a BF16 ulp of ref (one rounding;
// 2^-9 |ref|) + 2^-24 |x| (the FP32 add) + |value| * 0.25 * dg, where dg bounds the FP32 evaluation
// error of the gate's argument g = sign(dot) sqrt(max(|dot|, 1e-6)): with A = rx rk sum |x w key| /
// sqrt(H), |d dot| <= (H + 16) 2^-24 (A + 2 |dot|) covers the three FP32 sums, the two reciprocal
// roots and the product (every term at its worst case, not its random-walk size), and dg = |d dot|
// / (2 sqrt(max(|dot|, 1e-6))); 0.25 is the sigmoid's largest slope, and 4 ulps of the sigmoid
// itself are added. Also: a column's result is bitwise the same at every T and position; a
// graph-captured call is bitwise the eager call; refusals.

#include "ninfer/ops/engram.h"

#include "ops/op_tester.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
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
    return static_cast<float>(static_cast<std::int32_t>(next(state) >> 8) - (1 << 23)) /
           static_cast<float>(1 << 23);
}

struct Device {
    void* p = nullptr;

    explicit Device(std::size_t bytes) {
        t::cuda_check(cudaMalloc(&p, bytes ? bytes : 1), "cudaMalloc");
    }

    ~Device() { cudaFree(p); }

    Device(const Device&)            = delete;
    Device& operator=(const Device&) = delete;

    template <class V>
    void put(const V& v) {
        t::cuda_check(cudaMemcpy(p, v.data(), v.size() * sizeof(typename V::value_type),
                                 cudaMemcpyHostToDevice),
                      "put");
    }
};

struct Problem {
    int S, H, T;
    bool fp32_weight;
    float eps = 1e-6F;
    std::vector<std::uint16_t> x, kv; // [T][S * H], [T][(S + 1) * H]
    std::vector<float> w;             // S * H (exact in BF16 when the weight is BF16)
};

// scale_x scales the streams, scale_k the keys; mode: 0 random, 1 zero keys, 2 zero streams, 3 key
// = -stream.
Problem make_problem(int S, int H, int T, bool fp32_weight, float scale_x, float scale_k,
                     std::uint32_t seed, int mode = 0) {
    Problem p{S, H, T, fp32_weight};
    std::uint32_t s = seed;
    p.x.resize(static_cast<std::size_t>(S) * H * T);
    p.kv.resize(static_cast<std::size_t>(S + 1) * H * T);
    p.w.resize(static_cast<std::size_t>(S) * H);
    for (auto& v : p.w) {
        const float f = 1.0F + 0.4F * uniform(s);
        v             = fp32_weight ? f : t::bf16_to_f32(t::f32_to_bf16(f));
    }
    for (int c = 0; c < T; ++c) {
        for (int i = 0; i < S * H; ++i) {
            const float xv = mode == 2 ? 0.0F : scale_x * uniform(s) * 2.0F;
            p.x[static_cast<std::size_t>(c) * S * H + i] = t::f32_to_bf16(xv);
            float key                                    = mode == 1 ? 0.0F : scale_k * uniform(s);
            if (mode == 3) { key = -t::bf16_to_f32(t::f32_to_bf16(xv)); }
            p.kv[static_cast<std::size_t>(c) * (S + 1) * H + i] = t::f32_to_bf16(key);
        }
        for (int i = 0; i < H; ++i) {
            p.kv[static_cast<std::size_t>(c) * (S + 1) * H + static_cast<std::size_t>(S) * H + i] =
                t::f32_to_bf16(3.0F * uniform(s));
        }
    }
    return p;
}

std::vector<std::uint16_t> run(const Problem& p, cudaStream_t stream, bool graph = false) {
    Device xd(p.x.size() * 2), kd(p.kv.size() * 2), wd(p.w.size() * 4);
    xd.put(p.x);
    kd.put(p.kv);
    if (p.fp32_weight) {
        wd.put(p.w);
    } else {
        std::vector<std::uint16_t> bits;
        for (const float f : p.w) { bits.push_back(t::f32_to_bf16(f)); }
        wd.put(bits);
    }
    Tensor residual(xd.p, DType::BF16, {p.S * p.H, p.T});
    const Tensor kv(kd.p, DType::BF16, {(p.S + 1) * p.H, p.T});
    const Tensor w(wd.p, p.fp32_weight ? DType::FP32 : DType::BF16, {p.H, p.S});
    const auto call = [&] { ninfer::ops::engram_gate_inject(kv, w, p.S, p.eps, residual, stream); };
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
    constexpr double u = 5.960464477539063e-8; // 2^-24
    double worst = 0, worst_gate = 0;
    for (int c = 0; c < p.T; ++c) {
        const std::size_t xo = static_cast<std::size_t>(c) * p.S * p.H,
                          ko = static_cast<std::size_t>(c) * (p.S + 1) * p.H;
        for (int s = 0; s < p.S; ++s) {
            double xx = 0, kk = 0, xwk = 0, absxwk = 0;
            for (int d = 0; d < p.H; ++d) {
                const double xv = t::bf16_to_f32(p.x[xo + s * p.H + d]),
                             kv = t::bf16_to_f32(p.kv[ko + s * p.H + d]);
                xx += xv * xv;
                kk += kv * kv;
                xwk += xv * p.w[s * p.H + d] * kv;
                absxwk += std::fabs(xv * p.w[s * p.H + d] * kv);
            }
            const double rx    = 1.0 / std::sqrt(xx / p.H + p.eps),
                         rk    = 1.0 / std::sqrt(kk / p.H + p.eps);
            const double dot   = xwk * rx * rk / std::sqrt(static_cast<double>(p.H));
            const double a     = absxwk * rx * rk / std::sqrt(static_cast<double>(p.H));
            const double g     = std::copysign(std::sqrt(std::max(std::fabs(dot), 1e-6)), dot);
            const double gate  = 1.0 / (1.0 + std::exp(-g));
            const double d_dot = (p.H + 16) * u * (a + 2 * std::fabs(dot));
            const double dg    = d_dot / (2.0 * std::sqrt(std::max(std::fabs(dot), 1e-6)));
            for (int d = 0; d < p.H; ++d) {
                const double xv = t::bf16_to_f32(p.x[xo + s * p.H + d]);
                const double value =
                    t::bf16_to_f32(p.kv[ko + static_cast<std::size_t>(p.S) * p.H + d]);
                const double ref = xv + gate * value;
                const float have = t::bf16_to_f32(got[xo + s * p.H + d]);
                if (!std::isfinite(have)) {
                    check(false, label + ": a non-finite output");
                    return;
                }
                const double half_ulp =
                    ref == 0.0 ? 0.0 : std::ldexp(1.0, std::ilogb(std::fabs(ref)) - 8);
                const double bound =
                    half_ulp + u * std::fabs(xv) + std::fabs(value) * (0.25 * dg + 4 * u) + 1e-30;
                worst      = std::max(worst, std::fabs(have - ref) / bound);
                worst_gate = std::max(worst_gate, std::fabs(have - ref) / (half_ulp + 1e-30));
            }
        }
    }
    check(worst <= 1.0,
          label + ": within the criterion (worst " + std::to_string(worst) + " of the bound)");
    std::printf("%s: worst error / bound %.3f (error / half-ulp %.2f)\n", label.c_str(), worst,
                worst_gate);
}

void test_case(const std::string& label, const Problem& p, cudaStream_t stream) {
    verify(p, run(p, stream), label);
}

template <class F>
bool refuses(F&& f) {
    try {
        f();
    } catch (const std::invalid_argument&) { return true; }
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
        test_case("S=4 H=5120 T=1, FP32 gate (V4.1-Flash decode)",
                  make_problem(4, 5120, 1, true, 1.0F, 1.0F, 1U), stream);
        test_case("S=4 H=5120 T=3, FP32 gate", make_problem(4, 5120, 3, true, 1.0F, 1.0F, 2U),
                  stream);
        test_case("S=4 H=5120 T=64, BF16 gate", make_problem(4, 5120, 64, false, 1.5F, 0.7F, 3U),
                  stream);
        test_case("S=4 H=648 T=5 (off the block size)",
                  make_problem(4, 648, 5, true, 1.0F, 1.0F, 4U), stream);
        test_case("S=1 H=8 T=9", make_problem(1, 8, 9, true, 1.0F, 1.0F, 5U), stream);
        test_case("S=2 H=16384 T=2 (the widest)", make_problem(2, 16384, 2, true, 1.0F, 1.0F, 6U),
                  stream);
        test_case("S=4 H=5120 T=2, streams 1e3 and keys 1e-3 (the norms divide them out)",
                  make_problem(4, 5120, 2, true, 1000.0F, 0.001F, 7U), stream);
        test_case("S=4 H=5120 T=2, streams 1e-3 and keys 1e3",
                  make_problem(4, 5120, 2, true, 0.001F, 1000.0F, 8U), stream);
        test_case("S=4 H=5120 T=2, zero keys (dot = 0: the clamp)",
                  make_problem(4, 5120, 2, true, 1.0F, 1.0F, 9U, 1), stream);
        test_case("S=4 H=5120 T=2, zero streams",
                  make_problem(4, 5120, 2, true, 1.0F, 1.0F, 10U, 2), stream);
        test_case("S=4 H=5120 T=2, key opposite the stream (negative dot)",
                  make_problem(4, 5120, 2, true, 1.0F, 1.0F, 11U, 3), stream);
        {
            Problem p = make_problem(4, 5120, 2, true, 1.0F, 1.0F, 12U);
            p.eps     = 1e-20F; // the GGUF's epsilon
            test_case("S=4 H=5120 T=2, eps = 1e-20 (the GGUF's)", p, stream);
        }
        {
            // A column's result does not depend on T or its position.
            const Problem p = make_problem(4, 5120, 5, true, 1.0F, 1.0F, 21U);
            const auto all  = run(p, stream);
            for (const int c : {0, 3, 4}) {
                Problem one{p.S, p.H, 1, p.fp32_weight, p.eps, {}, {}, p.w};
                one.x.assign(p.x.begin() + static_cast<std::ptrdiff_t>(c) * p.S * p.H,
                             p.x.begin() + static_cast<std::ptrdiff_t>(c + 1) * p.S * p.H);
                one.kv.assign(p.kv.begin() + static_cast<std::ptrdiff_t>(c) * (p.S + 1) * p.H,
                              p.kv.begin() + static_cast<std::ptrdiff_t>(c + 1) * (p.S + 1) * p.H);
                const auto alone = run(one, stream);
                check(std::equal(alone.begin(), alone.end(),
                                 all.begin() + static_cast<std::ptrdiff_t>(c) * p.S * p.H),
                      "column " + std::to_string(c) + " alone is bitwise its result inside T = 5");
            }
            check(run(p, stream) == run(p, stream, true),
                  "a graph-captured call is bitwise the eager call");
        }
        {
            const Problem p = make_problem(4, 64, 2, true, 1.0F, 1.0F, 31U);
            Device xd(p.x.size() * 2), kd(p.kv.size() * 2), wd(p.w.size() * 4),
                od(p.x.size() * 2 + 256);
            Tensor residual(xd.p, DType::BF16, {256, 2});
            const Tensor kv(kd.p, DType::BF16, {320, 2}), w(wd.p, DType::FP32, {64, 4});
            namespace o = ninfer::ops;
            check(!refuses([&] { o::engram_gate_inject(kv, w, 4, 1e-6F, residual, stream); }),
                  "a valid call is served");
            check(refuses([&] { o::engram_gate_inject(kv, w, 0, 1e-6F, residual, stream); }),
                  "zero streams are refused");
            check(refuses([&] { o::engram_gate_inject(kv, w, 4, 0.0F, residual, stream); }),
                  "eps = 0 is refused");
            check(refuses([&] { o::engram_gate_inject(kv, w, 3, 1e-6F, residual, stream); }),
                  "streams that do not divide the rows are refused");
            const Tensor short_kv(kd.p, DType::BF16, {256, 2});
            check(refuses([&] { o::engram_gate_inject(short_kv, w, 4, 1e-6F, residual, stream); }),
                  "a kv without the value rows is refused");
            const Tensor other_t(kd.p, DType::BF16, {320, 1});
            check(refuses([&] { o::engram_gate_inject(other_t, w, 4, 1e-6F, residual, stream); }),
                  "a kv of another column count is refused");
            const Tensor short_w(wd.p, DType::FP32, {64, 3});
            check(refuses([&] { o::engram_gate_inject(kv, short_w, 4, 1e-6F, residual, stream); }),
                  "a gate weight of the wrong size is refused");
            const Tensor int_w(wd.p, DType::I32, {64, 4});
            check(refuses([&] { o::engram_gate_inject(kv, int_w, 4, 1e-6F, residual, stream); }),
                  "an integer gate weight is refused");
            const Tensor float_kv(kd.p, DType::FP32, {320, 2});
            check(refuses([&] { o::engram_gate_inject(float_kv, w, 4, 1e-6F, residual, stream); }),
                  "a non-BF16 kv is refused");
            Tensor odd(xd.p, DType::BF16, {4 * 12, 2}); // H = 12: not a multiple of 8
            const Tensor odd_kv(kd.p, DType::BF16, {5 * 12, 2});
            const Tensor odd_w(wd.p, DType::FP32, {12, 4});
            check(refuses([&] { o::engram_gate_inject(odd_kv, odd_w, 4, 1e-6F, odd, stream); }),
                  "an H that is not a multiple of 8 is refused");
            Tensor aliased(kd.p, DType::BF16, {256, 2});
            check(refuses([&] { o::engram_gate_inject(kv, w, 4, 1e-6F, aliased, stream); }),
                  "a residual overlapping kv is refused");
            Tensor misaligned(static_cast<char*>(xd.p) + 2, DType::BF16, {256, 1});
            const Tensor kv1(kd.p, DType::BF16, {320, 1});
            check(refuses([&] { o::engram_gate_inject(kv1, w, 4, 1e-6F, misaligned, stream); }),
                  "a misaligned residual is refused");
            t::cuda_check(cudaStreamSynchronize(stream), "sync");
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "EXCEPTION: %s\n", e.what());
        return 1;
    }
    if (failures != 0) {
        std::fprintf(stderr, "engram_gate_inject: %d failure(s)\n", failures);
        return 1;
    }
    std::printf("engram_gate_inject: passed\n");
    return 0;
}
