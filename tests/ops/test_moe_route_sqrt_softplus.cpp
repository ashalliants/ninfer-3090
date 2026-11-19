// moe_route_sqrt_softplus against an FP64 oracle of its formula (include/ninfer/ops/
// moe_route_sqrt_softplus.h): scores sqrt(softplus(z)), selection by score + bias with ties to the
// lower expert id, weights from the unbiased scores, optional normalisation, route scale.
//
// Cases: DeepSeek-V4.1-Flash's router (E = 384, k = 6, normalised, scale 1.5) at T = 1, 2, 33 and
// 4,097; a router without normalisation and with another scale (E = 256, k = 8); every expert
// selected (E = 5, k = 5); the widest (E = 512, k = 16); k = 1; logits past the softplus linear
// threshold (z > 20); exact ties (equal logit and bias on several experts, so the lower id must come
// first). Criteria (suite-owned): ids exactly equal to the oracle's order, except in a column whose
// oracle keys have two adjacent selected-or-next experts within 1e-5 of each other (FP32 evaluation
// of expf/log1pf/sqrtf may order those either way; such columns are counted and must be rare, and
// there the device's ids must still be a valid selection: every device id's key is within 1e-5 of
// the oracle's k-th key or above); weights |got - ref| <= 2e-6 |ref| + 1e-8 (about 17 FP32 ulps) against the oracle
// evaluated on the ids the device chose; finite required. Also a graph-captured call bitwise equal
// to the eager call and refusals.

#include "ninfer/ops/moe_route_sqrt_softplus.h"

#include "ops/op_tester.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
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
};

double sqrt_softplus(double z) { return std::sqrt(z > 20.0 ? z : std::log1p(std::exp(z))); }

struct Problem {
    int experts, top_k, columns;
    float scale;
    bool normalise;
    std::vector<float> logits, bias;
};

Problem make_problem(int experts, int top_k, int columns, float scale, bool normalise, std::uint32_t seed,
                     float logit_sigma = 2.0F) {
    Problem p{experts, top_k, columns, scale, normalise, {}, {}};
    std::uint32_t s = seed;
    p.logits.resize(static_cast<std::size_t>(experts) * columns);
    for (auto& v : p.logits) { v = logit_sigma * uniform(s) * 1.7F; }
    p.bias.resize(experts);
    for (auto& v : p.bias) { v = 0.3F * uniform(s); }
    return p;
}

struct Result {
    std::vector<std::int32_t> ids;
    std::vector<float> weights;
};

Result run(const Problem& p, cudaStream_t stream, bool graph = false) {
    const std::size_t n = static_cast<std::size_t>(p.top_k) * p.columns;
    Device ld(p.logits.size() * 4), bd(p.bias.size() * 4), id(n * 4), wd(n * 4);
    t::cuda_check(cudaMemcpy(ld.p, p.logits.data(), p.logits.size() * 4, cudaMemcpyHostToDevice), "logits");
    t::cuda_check(cudaMemcpy(bd.p, p.bias.data(), p.bias.size() * 4, cudaMemcpyHostToDevice), "bias");
    const Tensor L(ld.p, DType::FP32, {p.experts, p.columns}), B(bd.p, DType::FP32, {p.experts});
    Tensor ids(id.p, DType::I32, {p.top_k, p.columns}), weights(wd.p, DType::FP32, {p.top_k, p.columns});
    const auto call = [&] { ninfer::ops::moe_route_sqrt_softplus(L, B, p.top_k, p.scale, p.normalise, ids, weights, stream); };
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
    Result r;
    r.ids.resize(n);
    r.weights.resize(n);
    t::cuda_check(cudaMemcpy(r.ids.data(), id.p, n * 4, cudaMemcpyDeviceToHost), "ids");
    t::cuda_check(cudaMemcpy(r.weights.data(), wd.p, n * 4, cudaMemcpyDeviceToHost), "weights");
    return r;
}

// Returns the number of columns where an adjacent-key near tie let the order differ.
int verify(const Problem& p, const Result& got, const std::string& label) {
    int ambiguous = 0;
    double worst = 0;
    for (int c = 0; c < p.columns; ++c) {
        std::vector<double> score(p.experts), key(p.experts);
        for (int e = 0; e < p.experts; ++e) {
            score[e] = sqrt_softplus(p.logits[static_cast<std::size_t>(c) * p.experts + e]);
            key[e]   = score[e] + static_cast<double>(p.bias[e]);
        }
        std::vector<int> order(p.experts);
        std::iota(order.begin(), order.end(), 0);
        std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return key[a] > key[b]; });
        bool near_tie = false;
        const int upto = std::min(p.experts - 1, p.top_k); // the pairs among the selected and the next
        for (int i = 0; i < upto; ++i) { near_tie = near_tie || key[order[i]] - key[order[i + 1]] < 1e-5 && key[order[i]] != key[order[i + 1]]; }
        const std::int32_t* ids = got.ids.data() + static_cast<std::size_t>(c) * p.top_k;
        bool same = true;
        for (int i = 0; i < p.top_k; ++i) { same = same && ids[i] == order[i]; }
        if (near_tie) {
            ++ambiguous;
            for (int i = 0; i < p.top_k; ++i) {
                check(ids[i] >= 0 && ids[i] < p.experts && key[ids[i]] >= key[order[p.top_k - 1]] - 1e-5,
                      label + ": an ambiguous column still selects a valid expert");
            }
        } else {
            check(same, label + ": column " + std::to_string(c) + " selects the oracle's experts in order");
        }
        // Weights on the device's own ids.
        double sum = 0;
        for (int i = 0; i < p.top_k; ++i) { sum += score[ids[i]]; }
        for (int i = 0; i < p.top_k; ++i) {
            const double w = (p.normalise && p.top_k > 1 ? score[ids[i]] / (sum + 1e-20) : score[ids[i]]) * p.scale;
            const float g  = got.weights[static_cast<std::size_t>(c) * p.top_k + i];
            if (!std::isfinite(g)) {
                check(false, label + ": a non-finite weight");
                continue;
            }
            worst = std::max(worst, std::fabs(g - w) / (2e-6 * std::fabs(w) + 1e-8));
        }
    }
    check(worst <= 1.0, label + ": weights within the criterion (worst " + std::to_string(worst) + ")");
    std::printf("%s: worst weight error / bound %.3f, %d ambiguous of %d columns\n", label.c_str(), worst, ambiguous, p.columns);
    return ambiguous;
}

void test_case(const std::string& label, const Problem& p, cudaStream_t stream) {
    const Result got = run(p, stream);
    const int ambiguous = verify(p, got, label);
    check(ambiguous * 50 <= p.columns + 50, label + ": near-ties are rare");
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
        for (const int T : {1, 2, 33, 4097}) {
            test_case("E=384 k=6 T=" + std::to_string(T), make_problem(384, 6, T, 1.5F, true, 11U + T), stream);
        }
        test_case("E=256 k=8 unnormalised scale 2.5", make_problem(256, 8, 9, 2.5F, false, 21U), stream);
        test_case("E=5 k=5", make_problem(5, 5, 4, 1.0F, true, 31U), stream);
        test_case("E=512 k=16", make_problem(512, 16, 6, 1.0F, true, 41U), stream);
        test_case("E=384 k=1", make_problem(384, 1, 5, 1.0F, true, 51U), stream);
        test_case("E=384 k=6 logits past the softplus threshold", make_problem(384, 6, 5, 1.5F, true, 61U, 25.0F), stream);

        // Exact ties: equal logit and bias on experts {10, 11, 12, 300}: the lower ids come first.
        {
            Problem p = make_problem(384, 6, 3, 1.0F, true, 71U);
            for (int c = 0; c < p.columns; ++c) {
                for (const int e : {10, 11, 12, 300}) {
                    p.logits[static_cast<std::size_t>(c) * 384 + e] = 30.0F; // far above the rest: all four are selected
                    p.bias[e] = 0.25F;
                }
            }
            const Result got = run(p, stream);
            for (int c = 0; c < p.columns; ++c) {
                const std::int32_t* ids = got.ids.data() + static_cast<std::size_t>(c) * 6;
                check(ids[0] == 10 && ids[1] == 11 && ids[2] == 12 && ids[3] == 300, "tied experts come out lowest id first");
            }
            verify(p, got, "ties");
        }

        // Graph capture is bitwise the eager call.
        {
            const Problem p = make_problem(384, 6, 17, 1.5F, true, 81U);
            const Result a = run(p, stream), b = run(p, stream, true);
            check(a.ids == b.ids && std::memcmp(a.weights.data(), b.weights.data(), a.weights.size() * 4) == 0,
                  "a graph-captured call is bitwise the eager call");
        }

        // Refusals.
        {
            const Problem p = make_problem(384, 6, 2, 1.0F, true, 91U);
            Device ld(p.logits.size() * 4), bd(p.bias.size() * 4), id(2 * 6 * 4), wd(2 * 6 * 4);
            const Tensor L(ld.p, DType::FP32, {384, 2}), B(bd.p, DType::FP32, {384});
            Tensor ids(id.p, DType::I32, {6, 2}), weights(wd.p, DType::FP32, {6, 2});
            namespace o = ninfer::ops;
            check(!refuses([&] { o::moe_route_sqrt_softplus(L, B, 6, 1.0F, true, ids, weights, stream); }), "a valid call is served");
            check(refuses([&] { o::moe_route_sqrt_softplus(L, B, 17, 1.0F, true, ids, weights, stream); }), "k past 16 is refused");
            check(refuses([&] { o::moe_route_sqrt_softplus(L, B, 0, 1.0F, true, ids, weights, stream); }), "k = 0 is refused");
            check(refuses([&] { o::moe_route_sqrt_softplus(L, B, 6, std::numeric_limits<float>::infinity(), true, ids, weights, stream); }),
                  "an infinite route_scale is refused");
            const Tensor wide(ld.p, DType::FP32, {513, 1});
            const Tensor wide_bias(bd.p, DType::FP32, {513});
            check(refuses([&] { o::moe_route_sqrt_softplus(wide, wide_bias, 6, 1.0F, true, ids, weights, stream); }),
                  "more than 512 experts are refused");
            Tensor overlapping(ld.p, DType::FP32, {6, 2});
            check(refuses([&] { o::moe_route_sqrt_softplus(L, B, 6, 1.0F, true, ids, overlapping, stream); }),
                  "an output overlapping an input is refused");
            const Tensor wrong_bias(bd.p, DType::FP32, {383});
            check(refuses([&] { o::moe_route_sqrt_softplus(L, wrong_bias, 6, 1.0F, true, ids, weights, stream); }), "a wrong bias size is refused");
            t::cuda_check(cudaStreamSynchronize(stream), "sync");
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "EXCEPTION: %s\n", e.what());
        return 1;
    }
    if (failures != 0) {
        std::fprintf(stderr, "moe_route_sqrt_softplus: %d failure(s)\n", failures);
        return 1;
    }
    std::printf("moe_route_sqrt_softplus: passed\n");
    return 0;
}
