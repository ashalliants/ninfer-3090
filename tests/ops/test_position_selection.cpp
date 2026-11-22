// weighted_relu_dot_scores and top_k_positions (include/ninfer/ops/position_selection.h): DeepSeek-V4's
// indexer scores against an FP64 oracle, and its top-k selection against an exact oracle.
//
// Scores: V4.1-Flash's geometry (D = 128 index dimension, H = 32 index heads) at T = 1 and 5 with a
// few thousand positions and ragged reachable lengths (zero, partial tiles, all, and past N, and
// negative), and the other head counts (8, 16, 64) and dimensions (8, 64, 256 where D*H <= 8192), plus chunks of 8 or more columns (the tensor-core route) at 16/32/64 heads.
// Criterion (suite-owned): |got - ref| <= 5e-7 * |scale| * sum_h |w_h| |dot_h| (a few FP32 ulps of
// the terms' magnitude; measured worst about a quarter of it); unreachable positions exactly -inf. Top-k: exact equality with a stable
// sort by (score descending, position ascending) of the reachable positions, written ascending plus
// the offset and padded with -1, on scores with many exact ties (values from a small set, -0 beside
// +0, -inf among reachable positions), at V4's k = 512 over a few thousand and 300,000 positions, at
// n < k, n = k, n = 0, k = 1, and k = 4,096. Also a scores-to-selection chain against the NumPy-style
// full oracle, graph capture bitwise equal to eager, and refusals.

#include "ninfer/ops/position_selection.h"

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
#include <numeric>
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
    template <class T>
    std::vector<T> get(std::size_t count) const {
        std::vector<T> out(count);
        t::cuda_check(cudaMemcpy(out.data(), p, count * sizeof(T), cudaMemcpyDeviceToHost), "get");
        return out;
    }
};

template <class F>
bool refuses(F&& f) {
    try {
        f();
    } catch (const std::invalid_argument&) {
        return true;
    }
    return false;
}

// ---------------------------------------------------------------------------------- scores

struct ScoreCase {
    int D, H, T, N;
    float scale;
    std::vector<std::uint16_t> q, k, w;
    std::vector<std::int32_t> valid;
};

ScoreCase make_scores(int D, int H, int T, int N, std::vector<std::int32_t> valid, std::uint32_t seed, float scale = 0.0625F) {
    ScoreCase c{D, H, T, N, scale, {}, {}, {}, std::move(valid)};
    std::uint32_t s = seed;
    c.q.resize(static_cast<std::size_t>(D) * H * T);
    c.k.resize(static_cast<std::size_t>(D) * N);
    c.w.resize(static_cast<std::size_t>(H) * T);
    for (auto& v : c.q) { v = t::f32_to_bf16(uniform(s) * 2.0F); }
    for (auto& v : c.k) { v = t::f32_to_bf16(uniform(s) * 2.0F); }
    for (auto& v : c.w) { v = t::f32_to_bf16(uniform(s)); }
    return c;
}

std::vector<float> run_scores(const ScoreCase& c, cudaStream_t stream, bool graph = false) {
    Device qd(c.q.size() * 2), kd(c.k.size() * 2), wd(c.w.size() * 2), vd(c.valid.size() * 4),
        od(static_cast<std::size_t>(c.N) * c.T * 4);
    qd.put(c.q);
    kd.put(c.k);
    wd.put(c.w);
    vd.put(c.valid);
    const Tensor Q(qd.p, DType::BF16, {c.D, c.H, c.T}), K(kd.p, DType::BF16, {c.D, c.N}), W(wd.p, DType::BF16, {c.H, c.T}),
        V(vd.p, DType::I32, {c.T});
    Tensor out(od.p, DType::FP32, {c.N, c.T});
    if (graph) {
        cudaGraph_t g        = nullptr;
        cudaGraphExec_t exec = nullptr;
        t::cuda_check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal), "capture");
        ninfer::ops::weighted_relu_dot_scores(Q, K, W, V, c.scale, out, stream);
        t::cuda_check(cudaStreamEndCapture(stream, &g), "end capture");
        t::cuda_check(cudaGraphInstantiate(&exec, g, 0), "instantiate");
        t::cuda_check(cudaGraphLaunch(exec, stream), "launch");
        cudaGraphExecDestroy(exec);
        cudaGraphDestroy(g);
    } else {
        ninfer::ops::weighted_relu_dot_scores(Q, K, W, V, c.scale, out, stream);
    }
    t::cuda_check(cudaStreamSynchronize(stream), "sync");
    return od.get<float>(static_cast<std::size_t>(c.N) * c.T);
}

void verify_scores(const ScoreCase& c, const std::vector<float>& got, const std::string& label) {
    double worst = 0;
    for (int s = 0; s < c.T; ++s) {
        const int lim = std::min(std::max(c.valid[s], 0), c.N);
        for (int n = 0; n < c.N; ++n) {
            const float have = got[static_cast<std::size_t>(s) * c.N + n];
            if (n >= lim) {
                check(have == -std::numeric_limits<float>::infinity(), label + ": an unreachable position is -inf");
                continue;
            }
            double ref = 0, magnitude = 0;
            for (int h = 0; h < c.H; ++h) {
                double dot = 0;
                for (int d = 0; d < c.D; ++d) {
                    dot += static_cast<double>(t::bf16_to_f32(c.q[(static_cast<std::size_t>(s) * c.H + h) * c.D + d])) *
                           t::bf16_to_f32(c.k[static_cast<std::size_t>(n) * c.D + d]);
                }
                const double w = t::bf16_to_f32(c.w[static_cast<std::size_t>(s) * c.H + h]);
                ref += w * std::max(dot, 0.0);
                magnitude += std::fabs(w * dot);
            }
            ref *= c.scale;
            if (!std::isfinite(have)) {
                check(false, label + ": a non-finite score");
                return;
            }
            worst = std::max(worst, std::fabs(have - ref) / (5e-7 * std::fabs(c.scale) * magnitude + 1e-30));
        }
    }
    check(worst <= 1.0, label + ": within the criterion (worst " + std::to_string(worst) + " of the bound)");
    std::printf("%s: worst error / bound %.3f\n", label.c_str(), worst);
}

// ---------------------------------------------------------------------------------- top-k

struct TopCase {
    int N, T, k, offset;
    std::vector<float> scores;
    std::vector<std::int32_t> valid;
};

// `levels` distinct score values (so ties are everywhere); 0 means continuous. Some -inf, -0 and +0.
TopCase make_top(int N, int T, int k, int offset, std::vector<std::int32_t> valid, std::uint32_t seed, int levels) {
    TopCase c{N, T, k, offset, {}, std::move(valid)};
    std::uint32_t s = seed;
    c.scores.resize(static_cast<std::size_t>(N) * T);
    for (auto& v : c.scores) {
        if (levels > 0) {
            v = static_cast<float>(static_cast<int>(next(s) % static_cast<std::uint32_t>(levels))) - levels / 2;
            if (v == 0.0F && (next(s) & 1U) != 0) { v = -0.0F; }
        } else {
            v = uniform(s) * 5.0F;
        }
        if (levels > 0 && next(s) % 23U == 0) { v = -std::numeric_limits<float>::infinity(); }
    }
    return c;
}

std::vector<std::int32_t> run_top(const TopCase& c, cudaStream_t stream, bool graph = false) {
    Device sd(c.scores.size() * 4), vd(c.valid.size() * 4), od(static_cast<std::size_t>(c.k) * c.T * 4);
    sd.put(c.scores);
    vd.put(c.valid);
    t::cuda_check(cudaMemset(od.p, 0x7F, static_cast<std::size_t>(c.k) * c.T * 4), "poison out");
    const Tensor S(sd.p, DType::FP32, {c.N, c.T}), V(vd.p, DType::I32, {c.T});
    Tensor out(od.p, DType::I32, {c.k, c.T});
    const auto call = [&] { ninfer::ops::top_k_positions(S, V, c.k, c.offset, out, stream); };
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
    return od.get<std::int32_t>(static_cast<std::size_t>(c.k) * c.T);
}

std::vector<std::int32_t> top_oracle(const TopCase& c) {
    std::vector<std::int32_t> out(static_cast<std::size_t>(c.k) * c.T, -1);
    for (int s = 0; s < c.T; ++s) {
        const int n = std::min(std::max(c.valid[s], 0), c.N);
        const int m = std::min(c.k, n);
        const float* row = c.scores.data() + static_cast<std::size_t>(s) * c.N;
        std::vector<int> order(n);
        std::iota(order.begin(), order.end(), 0);
        std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return row[a] > row[b]; }); // equal scores keep position order
        std::vector<int> chosen(order.begin(), order.begin() + m);
        std::sort(chosen.begin(), chosen.end());
        for (int i = 0; i < m; ++i) { out[static_cast<std::size_t>(s) * c.k + i] = chosen[i] + c.offset; }
    }
    return out;
}

void test_top(const std::string& label, const TopCase& c, cudaStream_t stream) {
    const std::vector<std::int32_t> got = run_top(c, stream), want = top_oracle(c);
    check(got == want, label + ": equals the exact selection");
    std::printf("%s: %s\n", label.c_str(), got == want ? "exact" : "MISMATCH");
}

// NINFER_BENCH=1: time V4 sizes (not a pass/fail criterion): a decode step over a 32k-token context and a
// 512-token prefill chunk over 16k tokens, both at ratio 2 (16,384 and 8,192 compressed positions).
void bench(int columns, int positions, cudaStream_t stream) {
    const ScoreCase c = make_scores(128, 32, columns, positions, std::vector<std::int32_t>(columns, positions), 5U);
    Device qd(c.q.size() * 2), kd(c.k.size() * 2), wd(c.w.size() * 2), vd(c.valid.size() * 4),
        sd(static_cast<std::size_t>(positions) * columns * 4), od(static_cast<std::size_t>(512) * columns * 4);
    qd.put(c.q);
    kd.put(c.k);
    wd.put(c.w);
    vd.put(c.valid);
    const Tensor Q(qd.p, DType::BF16, {128, 32, columns}), K(kd.p, DType::BF16, {128, positions}),
        W(wd.p, DType::BF16, {32, columns}), V(vd.p, DType::I32, {columns});
    Tensor S(sd.p, DType::FP32, {positions, columns}), out(od.p, DType::I32, {512, columns});
    cudaEvent_t e[3];
    for (auto& ev : e) { t::cuda_check(cudaEventCreate(&ev), "event"); }
    constexpr int kIters = 10;
    for (int i = 0; i < 2; ++i) {
        ninfer::ops::weighted_relu_dot_scores(Q, K, W, V, c.scale, S, stream);
        ninfer::ops::top_k_positions(S, V, 512, 0, out, stream);
    }
    t::cuda_check(cudaEventRecord(e[0], stream), "record");
    for (int i = 0; i < kIters; ++i) { ninfer::ops::weighted_relu_dot_scores(Q, K, W, V, c.scale, S, stream); }
    t::cuda_check(cudaEventRecord(e[1], stream), "record");
    for (int i = 0; i < kIters; ++i) { ninfer::ops::top_k_positions(S, V, 512, 0, out, stream); }
    t::cuda_check(cudaEventRecord(e[2], stream), "record");
    t::cuda_check(cudaEventSynchronize(e[2]), "sync");
    float scores_ms = 0, top_ms = 0;
    t::cuda_check(cudaEventElapsedTime(&scores_ms, e[0], e[1]), "elapsed");
    t::cuda_check(cudaEventElapsedTime(&top_ms, e[1], e[2]), "elapsed");
    std::printf("bench T=%d N=%d: scores %.1f us, top-512 %.1f us\n", columns, positions, 1000.0 * scores_ms / kIters,
                1000.0 * top_ms / kIters);
    for (auto& ev : e) { cudaEventDestroy(ev); }
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

        {
            const ScoreCase c = make_scores(128, 32, 1, 3000, {3000}, 11U);
            verify_scores(c, run_scores(c, stream), "scores D=128 H=32 T=1 N=3000");
        }
        {
            // Reachable lengths: none, one position, a partial tile, a whole tile, the table, past it, negative.
            const ScoreCase c = make_scores(128, 32, 7, 700, {0, 1, 333, 384, 700, 9999, -5}, 12U);
            verify_scores(c, run_scores(c, stream), "scores D=128 H=32 ragged valid lengths");
        }
        for (const int H : {8, 16, 64}) {
            const ScoreCase c = make_scores(H == 64 ? 128 : 64, H, 3, 300, {300, 150, 10}, 13U + H);
            verify_scores(c, run_scores(c, stream), "scores H=" + std::to_string(H));
        }
        for (const int D : {8, 256}) {
            const ScoreCase c = make_scores(D, 32, 3, 250, {250, 100, 3}, 20U + D);
            verify_scores(c, run_scores(c, stream), "scores D=" + std::to_string(D));
        }

        // Chunks of 8 or more columns (tensor cores on Ampere and newer): V4's geometry with a prefill
        // chunk's reachable lengths (rising by two per column), the other head counts and dimensions the
        // route serves, an odd position count (rows not 8-byte aligned) and a block that reaches nothing.
        {
            std::vector<std::int32_t> rising(37);
            for (int i = 0; i < 37; ++i) { rising[i] = 250 + 2 * i; }
            const ScoreCase c = make_scores(128, 32, 37, 700, rising, 81U);
            verify_scores(c, run_scores(c, stream), "scores D=128 H=32 T=37 N=700 rising lengths (tensor cores)");
        }
        {
            const ScoreCase c = make_scores(128, 32, 11, 251, {0, 1, 33, 64, 250, 251, 9999, -5, 100, 7, 251}, 82U);
            verify_scores(c, run_scores(c, stream), "scores D=128 H=32 T=11 N=251 ragged lengths (tensor cores)");
        }
        {
            const ScoreCase c = make_scores(128, 32, 16, 400, std::vector<std::int32_t>(16, 0), 83U);
            verify_scores(c, run_scores(c, stream), "scores T=16 nothing reachable (tensor cores)");
        }
        // The route seam (8 columns) and its neighbours, the scoring call captured in a graph.
        for (const int T : {7, 8, 9}) {
            std::vector<std::int32_t> lengths(T);
            for (int i = 0; i < T; ++i) { lengths[i] = 300 - 17 * i; }
            const ScoreCase c = make_scores(128, 32, T, 300, lengths, 70U + T);
            const std::vector<float> eager = run_scores(c, stream);
            verify_scores(c, eager, "scores D=128 H=32 N=300 T=" + std::to_string(T) + " (route seam)");
            check(run_scores(c, stream, true) == eager, "a graph-captured scores call is bitwise the eager call at T=" + std::to_string(T));
        }
        for (const int H : {16, 32, 64}) {
            for (const int D : {64, 128, 256}) {
                if (H / 16 * D > 256) { continue; }
                const ScoreCase c = make_scores(D, H, 9, 333, {333, 300, 200, 100, 64, 63, 33, 1, 250}, 90U + H + D);
                verify_scores(c, run_scores(c, stream), "scores H=" + std::to_string(H) + " D=" + std::to_string(D) + " T=9 (tensor cores)");
            }
        }
        {
            const ScoreCase c = make_scores(128, 32, 10, 3000, {3000, 2999, 2000, 1500, 1000, 700, 513, 64, 32, 5}, 99U);
            const std::vector<float> eager = run_scores(c, stream);
            verify_scores(c, eager, "scores D=128 H=32 T=10 N=3000 (tensor cores)");
            // A column's scores do not depend on the other columns in its chunk.
            ScoreCase one = c;
            one.T = 9;
            one.q.assign(c.q.begin() + 128 * 32, c.q.end());
            one.w.assign(c.w.begin() + 32, c.w.end());
            one.valid.assign(c.valid.begin() + 1, c.valid.end());
            const std::vector<float> tail = run_scores(one, stream);
            check(std::memcmp(tail.data(), eager.data() + 3000, 9 * 3000 * 4) == 0, "tensor-core scores of a column do not depend on the other columns");
        }

        test_top("top-k k=512 N=3000 T=5 continuous", make_top(3000, 5, 512, 128, {3000, 2999, 1500, 1000, 513}, 31U, 0), stream);
        test_top("top-k k=512 N=3000 T=5 ties (6 levels, -inf, +-0)", make_top(3000, 5, 512, 0, {3000, 2999, 1500, 1000, 513}, 32U, 6), stream);
        test_top("top-k k=512 N=3000 ties (2 levels)", make_top(3000, 3, 512, 7, {3000, 1200, 700}, 33U, 2), stream);
        test_top("top-k k=512 n < k, n = k, n = 0", make_top(1000, 4, 512, 0, {100, 512, 0, 1}, 34U, 5), stream);
        test_top("top-k k=1", make_top(500, 3, 1, 0, {500, 40, 1}, 35U, 4), stream);
        test_top("top-k k=4096 N=5000", make_top(5000, 2, 4096, 0, {5000, 4097}, 36U, 9), stream);
        test_top("top-k N=300000 k=512 T=2", make_top(300000, 2, 512, 1000, {300000, 123457}, 37U, 0), stream);
        test_top("top-k N=300000 k=512 ties", make_top(300000, 2, 512, 0, {300000, 123457}, 38U, 4), stream);
        test_top("top-k valid past N and negative", make_top(200, 3, 16, 0, {100000, -3, 200}, 39U, 5), stream);
        {
            TopCase all_equal = make_top(2000, 2, 512, 0, {2000, 1999}, 40U, 0);
            std::fill(all_equal.scores.begin(), all_equal.scores.end(), 1.5F);
            test_top("top-k all scores equal (the first k positions)", all_equal, stream);
            TopCase all_neg_inf = make_top(2000, 2, 512, 0, {2000, 700}, 41U, 0);
            std::fill(all_neg_inf.scores.begin(), all_neg_inf.scores.end(), -std::numeric_limits<float>::infinity());
            test_top("top-k all reachable scores -inf (still the first k positions)", all_neg_inf, stream);
        }

        // The two Ops chained, against the oracles composed.
        {
            const ScoreCase c = make_scores(128, 32, 4, 1500, {1500, 1499, 900, 513}, 51U);
            Device qd(c.q.size() * 2), kd(c.k.size() * 2), wd(c.w.size() * 2), vd(16), sd(static_cast<std::size_t>(c.N) * c.T * 4), od(512 * 4 * 4);
            qd.put(c.q);
            kd.put(c.k);
            wd.put(c.w);
            vd.put(c.valid);
            const Tensor Q(qd.p, DType::BF16, {128, 32, 4}), K(kd.p, DType::BF16, {128, 1500}), W(wd.p, DType::BF16, {32, 4}),
                V(vd.p, DType::I32, {4});
            Tensor S(sd.p, DType::FP32, {1500, 4}), out(od.p, DType::I32, {512, 4});
            ninfer::ops::weighted_relu_dot_scores(Q, K, W, V, c.scale, S, stream);
            ninfer::ops::top_k_positions(S, V, 512, 64, out, stream);
            t::cuda_check(cudaStreamSynchronize(stream), "sync");
            const std::vector<std::int32_t> got = od.get<std::int32_t>(512 * 4);
            const std::vector<float> scores     = sd.get<float>(static_cast<std::size_t>(1500) * 4);
            TopCase via{1500, 4, 512, 64, scores, c.valid};
            check(got == top_oracle(via), "the chain selects the exact top-k of the scores it computed");
        }

        if (std::getenv("NINFER_BENCH") != nullptr) {
            bench(1, 16384, stream);
            bench(512, 8192, stream);
        }

        // Graph capture is bitwise the eager call.
        {
            const TopCase c = make_top(3000, 5, 512, 0, {3000, 2999, 1500, 1000, 513}, 61U, 6);
            check(run_top(c, stream) == run_top(c, stream, true), "a graph-captured top-k is bitwise the eager call");
        }

        // Refusals.
        {
            const ScoreCase c = make_scores(128, 32, 2, 100, {100, 100}, 71U);
            Device qd(c.q.size() * 2), kd(c.k.size() * 2), wd(c.w.size() * 2), vd(8), sd(100 * 2 * 4), id(512 * 2 * 4);
            const Tensor Q(qd.p, DType::BF16, {128, 32, 2}), K(kd.p, DType::BF16, {128, 100}), W(wd.p, DType::BF16, {32, 2}),
                V(vd.p, DType::I32, {2});
            Tensor S(sd.p, DType::FP32, {100, 2}), out(id.p, DType::I32, {512, 2});
            namespace o = ninfer::ops;
            check(!refuses([&] { o::weighted_relu_dot_scores(Q, K, W, V, 0.1F, S, stream); }), "a valid scores call is served");
            check(refuses([&] { o::weighted_relu_dot_scores(Q, K, W, V, std::numeric_limits<float>::infinity(), S, stream); }),
                  "an infinite weight_scale is refused");
            const Tensor odd_heads(qd.p, DType::BF16, {128, 24, 2});
            check(refuses([&] { o::weighted_relu_dot_scores(odd_heads, K, W, V, 0.1F, S, stream); }), "24 heads are refused");
            const Tensor wide(qd.p, DType::BF16, {512, 32, 2});
            const Tensor wide_k(kd.p, DType::BF16, {512, 100});
            check(refuses([&] { o::weighted_relu_dot_scores(wide, wide_k, W, V, 0.1F, S, stream); }), "D * H past 8192 is refused");
            const Tensor wrong_k(kd.p, DType::BF16, {64, 100});
            check(refuses([&] { o::weighted_relu_dot_scores(Q, wrong_k, W, V, 0.1F, S, stream); }), "a key width other than D is refused");
            Tensor bad_scores(sd.p, DType::FP32, {99, 2});
            check(refuses([&] { o::weighted_relu_dot_scores(Q, K, W, V, 0.1F, bad_scores, stream); }), "a scores table of another size is refused");
            Tensor aliased(kd.p, DType::FP32, {100, 2});
            check(refuses([&] { o::weighted_relu_dot_scores(Q, K, W, V, 0.1F, aliased, stream); }), "scores overlapping an input is refused");
            const Tensor top_scores(sd.p, DType::FP32, {100, 2});
            check(!refuses([&] { o::top_k_positions(top_scores, V, 512, 0, out, stream); }), "a valid top-k call is served");
            check(refuses([&] { o::top_k_positions(top_scores, V, 0, 0, out, stream); }), "k = 0 is refused");
            check(refuses([&] { o::top_k_positions(top_scores, V, 4097, 0, out, stream); }), "k past 4,096 is refused");
            check(refuses([&] { o::top_k_positions(top_scores, V, 512, -1, out, stream); }), "a negative offset is refused");
            check(refuses([&] { o::top_k_positions(top_scores, V, 512, 2147483647 - 98, out, stream); }),
                  "an offset that overflows the I32 output with N = 100 is refused");
            check(!refuses([&] { o::top_k_positions(top_scores, V, 512, 2147483647 - 99, out, stream); }),
                  "the largest offset that fits (offset + N - 1 = INT32_MAX) is served");
            {
                // More positions than the grid serves: refused before any launch (the shapes are larger than the buffers).
                const int too_many = 65535 * 128 + 1;
                const Tensor big_k(kd.p, DType::BF16, {128, too_many});
                Tensor big_scores(sd.p, DType::FP32, {too_many, 2});
                check(refuses([&] { o::weighted_relu_dot_scores(Q, big_k, W, V, 0.1F, big_scores, stream); }),
                      "more than 65,535 tiles of positions are refused");
            }
            Tensor wrong_out(id.p, DType::I32, {256, 2});
            check(refuses([&] { o::top_k_positions(top_scores, V, 512, 0, wrong_out, stream); }), "an output of another width is refused");
            Tensor aliased_out(sd.p, DType::I32, {100, 2});
            check(refuses([&] { o::top_k_positions(top_scores, V, 100, 0, aliased_out, stream); }), "an output overlapping the scores is refused");
            t::cuda_check(cudaStreamSynchronize(stream), "sync");
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "EXCEPTION: %s\n", e.what());
        return 1;
    }
    if (failures != 0) {
        std::fprintf(stderr, "position_selection: %d failure(s)\n", failures);
        return 1;
    }
    std::printf("position_selection: passed\n");
    return 0;
}
