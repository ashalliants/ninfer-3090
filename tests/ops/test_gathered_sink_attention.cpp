// gathered_sink_attention against an FP64 oracle of its formula (include/ninfer/ops/
// gathered_sink_attention.h): softmax over each query's listed rows of one shared KV table, the
// rows serving as both keys and values, with a per-head sink logit added to the denominator only.
//
// Cases (8 or more columns with D = 256 or 512 run on tensor cores, whose BF16 softmax weights widen the criterion by 2^-8 of the weighted mean magnitude of the rows): DeepSeek-V4.1-Flash's geometry (D = 512, H = 64, K = 640 = 128 window + 512 compressed,
// N = 700 rows) at T = 1 (decode), 3 and 70; a small geometry with an odd head count and a list
// that is not a multiple of the 4-row group (D = 128, H = 5, K = 7); a one-row table (N = 1); D = 64
// and 256; lists with invalid (-1) and out-of-range entries and duplicated rows; a column with no
// valid row (exact zeros); logits far above the sink and below it (scores up to about 40, so the
// stable softmax is exercised) and a sink far above the scores. Criteria (suite-owned): every
// output element |got - ref| <= half a BF16 ulp of ref + 2e-5 rms(ref over the head's D elements)
// (one BF16 rounding of an FP32 evaluation of the formula, the allowance covering the FP32
// accumulation: about 170 FP32 ulps of the output's scale); finite
// required; the no-valid-row column exactly zero. Also a graph-captured call bitwise equal to the
// eager call, each column independent of the others (a T = 1 call equals its column of a T = 3
// call, bitwise), and refusals.

#include "ninfer/ops/gathered_sink_attention.h"

#include "ops/op_tester.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <limits>
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

struct Problem {
    int D, H, T, K, N;
    float scale;
    std::vector<std::uint16_t> q, kv;
    std::vector<std::int32_t> idx;
    std::vector<float> sink;
};

struct Shape {
    int D, H, T, K, N;
    float q_sigma = 1.0F;   // scales the query (and so the scores)
    float sink_mean = 0.0F; // the sink logits' centre
    float invalid_fraction = 0.2F;
};

Problem make_problem(const Shape& s, std::uint32_t seed) {
    Problem p{s.D, s.H, s.T, s.K, s.N, 1.0F / std::sqrt(static_cast<float>(s.D)), {}, {}, {}, {}};
    std::uint32_t r = seed;
    p.q.resize(static_cast<std::size_t>(s.D) * s.H * s.T);
    for (auto& v : p.q) { v = t::f32_to_bf16(s.q_sigma * uniform(r) * 1.7F); }
    p.kv.resize(static_cast<std::size_t>(s.D) * s.N);
    for (auto& v : p.kv) { v = t::f32_to_bf16(uniform(r) * 1.7F); }
    p.idx.resize(static_cast<std::size_t>(s.K) * s.T);
    for (auto& v : p.idx) {
        const float u = (uniform(r) + 1.0F) * 0.5F;
        if (u < s.invalid_fraction) {
            v = (next(r) & 1U) != 0 ? -1 : s.N + static_cast<std::int32_t>(next(r) % 50U); // invalid or out of range
        } else {
            v = static_cast<std::int32_t>(next(r) % static_cast<std::uint32_t>(s.N)); // duplicates occur
        }
    }
    p.sink.resize(s.H);
    for (auto& v : p.sink) { v = s.sink_mean + 2.0F * uniform(r); }
    return p;
}

std::vector<std::uint16_t> run(const Problem& p, cudaStream_t stream, bool graph = false) {
    const std::size_t out_count = p.q.size();
    Device qd(p.q.size() * 2), kd(p.kv.size() * 2), id(p.idx.size() * 4), sd(p.sink.size() * 4), od(out_count * 2);
    qd.put(p.q);
    kd.put(p.kv);
    id.put(p.idx);
    sd.put(p.sink);
    t::cuda_check(cudaMemset(od.p, 0xFF, out_count * 2), "poison out");
    const Tensor Q(qd.p, DType::BF16, {p.D, p.H, p.T}), KV(kd.p, DType::BF16, {p.D, p.N}), I(id.p, DType::I32, {p.K, p.T}),
        S(sd.p, DType::FP32, {p.H});
    Tensor out(od.p, DType::BF16, {p.D, p.H, p.T});
    const auto call = [&] { ninfer::ops::gathered_sink_attention(Q, KV, I, S, p.scale, out, stream); };
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
    std::vector<std::uint16_t> got(out_count);
    t::cuda_check(cudaMemcpy(got.data(), od.p, out_count * 2, cudaMemcpyDeviceToHost), "get");
    return got;
}

// The formula, naively, in FP64 from the represented BF16 inputs. `valid` reports whether the column had a row.
std::vector<double> oracle(const Problem& p, int column, int head, bool* any_valid, std::vector<double>* weighted_abs = nullptr) {
    std::vector<int> rows;
    for (int j = 0; j < p.K; ++j) {
        const std::int32_t r = p.idx[static_cast<std::size_t>(column) * p.K + j];
        if (r >= 0 && r < p.N) { rows.push_back(r); }
    }
    *any_valid = !rows.empty();
    std::vector<double> out(p.D, 0.0), absmean(p.D, 0.0);
    if (weighted_abs != nullptr) { weighted_abs->assign(p.D, 0.0); }
    if (rows.empty()) { return out; }
    const std::uint16_t* q = p.q.data() + (static_cast<std::size_t>(column) * p.H + head) * p.D;
    std::vector<double> s(rows.size());
    double m = p.sink[head];
    for (std::size_t j = 0; j < rows.size(); ++j) {
        double dot = 0;
        for (int d = 0; d < p.D; ++d) {
            dot += static_cast<double>(t::bf16_to_f32(q[d])) * t::bf16_to_f32(p.kv[static_cast<std::size_t>(rows[j]) * p.D + d]);
        }
        s[j] = dot * p.scale;
        m = std::max(m, s[j]);
    }
    double denom = std::exp(static_cast<double>(p.sink[head]) - m);
    for (std::size_t j = 0; j < rows.size(); ++j) {
        const double e = std::exp(s[j] - m);
        denom += e;
        for (int d = 0; d < p.D; ++d) {
            const double v = t::bf16_to_f32(p.kv[static_cast<std::size_t>(rows[j]) * p.D + d]);
            out[d] += e * v;
            absmean[d] += e * std::fabs(v);
        }
    }
    for (auto& v : out) { v /= denom; }
    // sum_j w_j |v_j| / sum_j w_j (the sink is not a row), the scale of the BF16 weight rounding.
    double wsum = denom - std::exp(static_cast<double>(p.sink[head]) - m);
    if (weighted_abs != nullptr) {
        for (int d = 0; d < p.D; ++d) { (*weighted_abs)[d] = absmean[d] / wsum; }
    }
    return out;
}

// The tensor-core route (contract: 8 or more columns, D = 256 or 512) multiplies BF16 softmax weights.
bool tensor_core_call(const Problem& p) { return p.T >= 8 && (p.D == 256 || p.D == 512); }

void verify(const Problem& p, const std::vector<std::uint16_t>& got, const std::string& label) {
    const bool tc = tensor_core_call(p);
    double worst = 0;
    int empty_columns = 0;
    for (int c = 0; c < p.T; ++c) {
        for (int h = 0; h < p.H; ++h) {
            bool any = false;
            std::vector<double> wabs;
            const std::vector<double> ref = oracle(p, c, h, &any, &wabs);
            double rms = 0;
            for (const double v : ref) { rms += v * v; }
            rms = std::sqrt(rms / p.D);
            if (!any && h == 0) { ++empty_columns; }
            for (int d = 0; d < p.D; ++d) {
                const float g = t::bf16_to_f32(got[(static_cast<std::size_t>(c) * p.H + h) * p.D + d]);
                if (!std::isfinite(g)) {
                    check(false, label + ": a non-finite output");
                    return;
                }
                if (!any) {
                    check(g == 0.0F, label + ": a column with no valid row is exactly zero");
                    continue;
                }
                // Half a BF16 ulp of the reference (the one rounding) plus an FP32 allowance.
                const double half_ulp = ref[d] == 0.0 ? 0.0 : std::ldexp(1.0, std::ilogb(std::fabs(ref[d])) - 8);
                const double rounding = tc ? std::ldexp(wabs[d], -8) : 0.0; // BF16 weights
                worst = std::max(worst, std::fabs(g - ref[d]) / (half_ulp + 2e-5 * rms + rounding + 1e-30));
            }
        }
    }
    check(worst <= 1.0, label + ": within the criterion (worst " + std::to_string(worst) + " of the bound)");
    std::printf("%s: worst error / bound %.3f%s\n", label.c_str(), worst, empty_columns ? " (with empty columns)" : "");
}

void test_case(const std::string& label, const Shape& s, std::uint32_t seed, cudaStream_t stream) {
    const Problem p = make_problem(s, seed);
    verify(p, run(p, stream), label);
}

// NINFER_BENCH=1: time the V4 geometry (not a pass/fail criterion).
void bench(int columns, cudaStream_t stream) {
    const Problem p = make_problem({512, 64, columns, 640, 700, 1.0F, 0.0F, 0.0F}, 7U);
    Device qd(p.q.size() * 2), kd(p.kv.size() * 2), id(p.idx.size() * 4), sd(p.sink.size() * 4), od(p.q.size() * 2);
    qd.put(p.q);
    kd.put(p.kv);
    id.put(p.idx);
    sd.put(p.sink);
    const Tensor Q(qd.p, DType::BF16, {p.D, p.H, p.T}), KV(kd.p, DType::BF16, {p.D, p.N}), I(id.p, DType::I32, {p.K, p.T}),
        S(sd.p, DType::FP32, {p.H});
    Tensor out(od.p, DType::BF16, {p.D, p.H, p.T});
    cudaEvent_t a = nullptr, b = nullptr;
    t::cuda_check(cudaEventCreate(&a), "event");
    t::cuda_check(cudaEventCreate(&b), "event");
    for (int i = 0; i < 3; ++i) { ninfer::ops::gathered_sink_attention(Q, KV, I, S, p.scale, out, stream); }
    constexpr int kIters = 20;
    t::cuda_check(cudaEventRecord(a, stream), "record");
    for (int i = 0; i < kIters; ++i) { ninfer::ops::gathered_sink_attention(Q, KV, I, S, p.scale, out, stream); }
    t::cuda_check(cudaEventRecord(b, stream), "record");
    t::cuda_check(cudaEventSynchronize(b), "sync");
    float ms = 0;
    t::cuda_check(cudaEventElapsedTime(&ms, a, b), "elapsed");
    std::printf("bench D=512 H=64 K=640 T=%d: %.1f us per call\n", columns, 1000.0 * ms / kIters);
    cudaEventDestroy(a);
    cudaEventDestroy(b);
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
        for (const int T : {1, 3, 8, 9, 70}) {
            test_case("D=512 H=64 K=640 N=700 T=" + std::to_string(T), {512, 64, T, 640, 700}, 11U + T, stream);
        }
        test_case("D=128 H=5 K=7 N=40 (odd heads, a partial row group)", {128, 5, 4, 7, 40}, 21U, stream);
        test_case("D=128 H=4 K=3 N=1 (one-row table)", {128, 4, 3, 3, 1, 1.0F, 0.0F, 0.3F}, 31U, stream);
        test_case("D=64 H=8 K=33 N=100", {64, 8, 5, 33, 100}, 41U, stream);
        test_case("D=256 H=8 K=33 N=100", {256, 8, 5, 33, 100}, 51U, stream);
        test_case("tensor cores, D=256 H=40 (a partial head block) K=70 N=100 T=12", {256, 40, 12, 70, 100}, 52U, stream);
        test_case("tensor cores, D=512 H=64 K=1 N=30 T=9 (a list of one row)", {512, 64, 9, 1, 30, 1.0F, 0.0F, 0.0F}, 53U, stream);
        test_case("tensor cores, large scores (q x 8)", {512, 64, 10, 130, 200, 8.0F}, 54U, stream);
        test_case("tensor cores, sink far above the scores", {512, 33, 9, 130, 200, 1.0F, 30.0F}, 55U, stream);
        test_case("tensor cores, mostly invalid lists", {512, 64, 12, 130, 200, 1.0F, 0.0F, 0.97F}, 56U, stream);
        test_case("large scores (q x 8)", {512, 8, 4, 130, 200, 8.0F}, 61U, stream);
        test_case("sink far above the scores", {512, 8, 3, 130, 200, 1.0F, 30.0F}, 71U, stream);
        test_case("sink far below the scores", {512, 8, 3, 130, 200, 6.0F, -60.0F}, 81U, stream);
        test_case("mostly invalid lists", {512, 8, 6, 130, 200, 1.0F, 0.0F, 0.97F}, 91U, stream);

        // A column with no valid row is exactly zero; its neighbours are unaffected.
        {
            Problem p = make_problem({128, 4, 3, 9, 50}, 101U);
            for (int j = 0; j < p.K; ++j) { p.idx[static_cast<std::size_t>(1) * p.K + j] = (j & 1) != 0 ? -1 : p.N + j; }
            verify(p, run(p, stream), "an empty middle column");
        }

        // Column independence and graph capture, bitwise.
        {
            const Problem p3 = make_problem({512, 64, 3, 640, 700}, 111U);
            const std::vector<std::uint16_t> all = run(p3, stream), graphed = run(p3, stream, true);
            check(all == graphed, "a graph-captured call is bitwise the eager call");
            Problem p1 = p3;
            p1.T = 1;
            const std::size_t per_column = static_cast<std::size_t>(p3.D) * p3.H;
            p1.q.assign(p3.q.begin() + per_column, p3.q.begin() + 2 * per_column);
            p1.idx.assign(p3.idx.begin() + p3.K, p3.idx.begin() + 2 * p3.K);
            const std::vector<std::uint16_t> one = run(p1, stream);
            check(std::memcmp(one.data(), all.data() + per_column, per_column * 2) == 0, "a column's output does not depend on the other columns");
        }

        // The tensor-core route: each column's output is its own, whatever the chunk it came in.
        {
            const Problem pw = make_problem({512, 64, 20, 640, 700}, 131U);
            const std::vector<std::uint16_t> whole = run(pw, stream);
            Problem pn = pw;
            pn.T = 9;
            const std::size_t per_column = static_cast<std::size_t>(pw.D) * pw.H;
            pn.q.assign(pw.q.begin() + 5 * per_column, pw.q.begin() + 14 * per_column);
            pn.idx.assign(pw.idx.begin() + 5 * pw.K, pw.idx.begin() + 14 * pw.K);
            const std::vector<std::uint16_t> part = run(pn, stream);
            check(std::memcmp(part.data(), whole.data() + 5 * per_column, 9 * per_column * 2) == 0,
                  "a tensor-core column's output does not depend on the other columns");
            const std::vector<std::uint16_t> graphed = run(pw, stream, true);
            check(whole == graphed, "a graph-captured tensor-core call is bitwise the eager call");
        }

        if (std::getenv("NINFER_BENCH") != nullptr) {
            for (const int columns : {1, 16, 512}) { bench(columns, stream); }
        }

        // Refusals.
        {
            const Problem p = make_problem({512, 8, 2, 16, 20}, 121U);
            Device qd(p.q.size() * 2), kd(p.kv.size() * 2), id(p.idx.size() * 4), sd(p.sink.size() * 4), od(p.q.size() * 2);
            const Tensor Q(qd.p, DType::BF16, {512, 8, 2}), KV(kd.p, DType::BF16, {512, 20}), I(id.p, DType::I32, {16, 2}),
                S(sd.p, DType::FP32, {8});
            Tensor out(od.p, DType::BF16, {512, 8, 2});
            namespace o = ninfer::ops;
            check(!refuses([&] { o::gathered_sink_attention(Q, KV, I, S, 0.044F, out, stream); }), "a valid call is served");
            check(refuses([&] { o::gathered_sink_attention(Q, KV, I, S, 0.0F, out, stream); }), "a zero scale is refused");
            check(refuses([&] { o::gathered_sink_attention(Q, KV, I, S, -1.0F, out, stream); }), "a negative scale is refused");
            check(refuses([&] { o::gathered_sink_attention(Q, KV, I, S, std::numeric_limits<float>::infinity(), out, stream); }),
                  "an infinite scale is refused");
            const Tensor odd_q(qd.p, DType::BF16, {96, 8, 2});
            Tensor odd_out(od.p, DType::BF16, {96, 8, 2});
            const Tensor odd_kv(kd.p, DType::BF16, {96, 20});
            check(refuses([&] { o::gathered_sink_attention(odd_q, odd_kv, I, S, 0.1F, odd_out, stream); }), "head dimension 96 is refused");
            const Tensor wrong_kv(kd.p, DType::BF16, {256, 20});
            check(refuses([&] { o::gathered_sink_attention(Q, wrong_kv, I, S, 0.1F, out, stream); }), "a KV width other than D is refused");
            const Tensor wrong_idx(id.p, DType::I32, {16, 3});
            check(refuses([&] { o::gathered_sink_attention(Q, KV, wrong_idx, S, 0.1F, out, stream); }), "an index list for another column count is refused");
            const Tensor wrong_sink(sd.p, DType::FP32, {7});
            check(refuses([&] { o::gathered_sink_attention(Q, KV, I, wrong_sink, 0.1F, out, stream); }), "a sink of the wrong length is refused");
            {
                // More heads than the grid serves: refused before any launch (the buffers are large enough for the shapes).
                Device big_q(512 * 1025 * 2 * 2), big_out(512 * 1025 * 2 * 2), big_sink(1025 * 4);
                const Tensor many_q(big_q.p, DType::BF16, {512, 1025, 2}), many_sink(big_sink.p, DType::FP32, {1025});
                Tensor many_out(big_out.p, DType::BF16, {512, 1025, 2});
                check(refuses([&] { o::gathered_sink_attention(many_q, KV, I, many_sink, 0.1F, many_out, stream); }),
                      "more than 1,024 heads are refused");
            }
            Tensor aliased(qd.p, DType::BF16, {512, 8, 2});
            check(refuses([&] { o::gathered_sink_attention(Q, KV, I, S, 0.1F, aliased, stream); }), "an output overlapping q is refused");
            const Tensor float_idx(id.p, DType::FP32, {16, 2});
            check(refuses([&] { o::gathered_sink_attention(Q, KV, float_idx, S, 0.1F, out, stream); }), "a non-I32 index list is refused");
            t::cuda_check(cudaStreamSynchronize(stream), "sync");
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "EXCEPTION: %s\n", e.what());
        return 1;
    }
    if (failures != 0) {
        std::fprintf(stderr, "gathered_sink_attention: %d failure(s)\n", failures);
        return 1;
    }
    std::printf("gathered_sink_attention: passed\n");
    return 0;
}
