// softmax_pool against an FP64 oracle of its formula (include/ninfer/ops/softmax_pool.h): each
// channel pools every group of `ratio` consecutive tokens with a softmax over the group.
//
// Cases: DeepSeek-V4.1-Flash's compressor width (D = 512) at its ratio 2 (the GGUF's
// compress_ratios hold 0, 2 and 1; ratio 1 is a plain projection and does not pool) over a prompt
// with a trailing partial group, and at larger ratios for generality (4, 128, up to the largest, 256); a tall group count (70,000 groups, past grid.y's 65,535); a width that is not a multiple
// of the block (D = 100); scores far apart (a softmax that is nearly one-hot); equal scores (a plain
// mean). Criteria (suite-owned): |got - ref| <= half a BF16 ulp of ref + 1e-5 max|kv in the group|
// (one BF16 rounding of an FP32 evaluation; the allowance covers expf and the FP32 sums over up to
// 256 terms); finite required; tokens past the last full group have no effect (checked by poisoning
// them). Also a graph-captured call bitwise equal to the eager call and refusals.

#include "ninfer/ops/softmax_pool.h"

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
    int D, T, ratio, G;
    std::vector<std::uint16_t> kv, score;
};

// score_sigma scales the scores; a sigma of 0 makes every score equal. `poison_tail` fills the
// tokens past the last full group with huge values (they must not be read).
Problem make_problem(int D, int T, int ratio, float score_sigma, std::uint32_t seed, bool poison_tail = false) {
    Problem p{D, T, ratio, T / ratio, {}, {}};
    std::uint32_t s = seed;
    p.kv.resize(static_cast<std::size_t>(D) * T);
    p.score.resize(static_cast<std::size_t>(D) * T);
    for (auto& v : p.kv) { v = t::f32_to_bf16(uniform(s) * 3.0F); }
    for (auto& v : p.score) { v = t::f32_to_bf16(score_sigma * uniform(s) * 1.7F); }
    if (poison_tail) {
        for (std::size_t i = static_cast<std::size_t>(D) * p.G * ratio; i < p.kv.size(); ++i) {
            p.kv[i]    = t::f32_to_bf16(30000.0F);
            p.score[i] = t::f32_to_bf16(30000.0F);
        }
    }
    return p;
}

std::vector<std::uint16_t> run(const Problem& p, cudaStream_t stream, bool graph = false) {
    Device kd(p.kv.size() * 2), sd(p.score.size() * 2), od(static_cast<std::size_t>(p.D) * p.G * 2);
    kd.put(p.kv);
    sd.put(p.score);
    t::cuda_check(cudaMemset(od.p, 0xFF, static_cast<std::size_t>(p.D) * p.G * 2), "poison out");
    const Tensor KV(kd.p, DType::BF16, {p.D, p.T}), S(sd.p, DType::BF16, {p.D, p.T});
    Tensor out(od.p, DType::BF16, {p.D, p.G});
    const auto call = [&] { ninfer::ops::softmax_pool(KV, S, p.ratio, out, stream); };
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
    std::vector<std::uint16_t> got(static_cast<std::size_t>(p.D) * p.G);
    t::cuda_check(cudaMemcpy(got.data(), od.p, got.size() * 2, cudaMemcpyDeviceToHost), "get");
    return got;
}

void verify(const Problem& p, const std::vector<std::uint16_t>& got, const std::string& label) {
    double worst = 0;
    for (int g = 0; g < p.G; ++g) {
        for (int d = 0; d < p.D; ++d) {
            double m = -1e300, biggest = 0;
            for (int i = 0; i < p.ratio; ++i) { m = std::max<double>(m, t::bf16_to_f32(p.score[(static_cast<std::size_t>(g) * p.ratio + i) * p.D + d])); }
            double denom = 0, pooled = 0;
            for (int i = 0; i < p.ratio; ++i) {
                const std::size_t at = (static_cast<std::size_t>(g) * p.ratio + i) * p.D + d;
                const double e = std::exp(static_cast<double>(t::bf16_to_f32(p.score[at])) - m);
                denom += e;
                pooled += e * t::bf16_to_f32(p.kv[at]);
                biggest = std::max<double>(biggest, std::fabs(t::bf16_to_f32(p.kv[at])));
            }
            const double ref = pooled / denom;
            const float have = t::bf16_to_f32(got[static_cast<std::size_t>(g) * p.D + d]);
            if (!std::isfinite(have)) {
                check(false, label + ": a non-finite output");
                return;
            }
            const double half_ulp = ref == 0.0 ? 0.0 : std::ldexp(1.0, std::ilogb(std::fabs(ref)) - 8);
            worst = std::max(worst, std::fabs(have - ref) / (half_ulp + 1e-5 * biggest + 1e-30));
        }
    }
    check(worst <= 1.0, label + ": within the criterion (worst " + std::to_string(worst) + " of the bound)");
    std::printf("%s: worst error / bound %.3f\n", label.c_str(), worst);
}

void test_case(const std::string& label, const Problem& p, cudaStream_t stream) { verify(p, run(p, stream), label); }

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
        test_case("D=512 ratio 4, 1,003 tokens (partial group)", make_problem(512, 1003, 4, 1.0F, 11U, true), stream);
        test_case("D=512 ratio 128, 1,000 tokens", make_problem(512, 1000, 128, 1.0F, 12U, true), stream);
        test_case("D=512 ratio 2, 1,001 tokens (V4.1-Flash; partial group)", make_problem(512, 1001, 2, 1.0F, 13U, true), stream);
        test_case("D=512 ratio 256", make_problem(512, 600, 256, 1.0F, 14U, true), stream);
        test_case("D=100 ratio 4 (a width off the block size)", make_problem(100, 41, 4, 1.0F, 15U, true), stream);
        test_case("D=512 ratio 4, scores far apart", make_problem(512, 64, 4, 40.0F, 16U), stream);
        test_case("D=512 ratio 8, equal scores (a mean)", make_problem(512, 64, 8, 0.0F, 17U), stream);
        test_case("D=16 ratio 4, 70,000 groups (past grid.y)", make_problem(16, 280000, 4, 1.0F, 18U), stream);
        {
            const Problem p = make_problem(512, 1003, 4, 1.0F, 21U);
            check(run(p, stream) == run(p, stream, true), "a graph-captured call is bitwise the eager call");
        }
        {
            const Problem p = make_problem(512, 16, 4, 1.0F, 31U);
            Device kd(p.kv.size() * 2), sd(p.score.size() * 2), od(512 * 4 * 2);
            const Tensor KV(kd.p, DType::BF16, {512, 16}), S(sd.p, DType::BF16, {512, 16});
            Tensor out(od.p, DType::BF16, {512, 4});
            namespace o = ninfer::ops;
            check(!refuses([&] { o::softmax_pool(KV, S, 4, out, stream); }), "a valid call is served");
            check(refuses([&] { o::softmax_pool(KV, S, 1, out, stream); }), "ratio 1 is refused");
            check(refuses([&] { o::softmax_pool(KV, S, 257, out, stream); }), "a ratio past 256 is refused");
            Device big(512 * 5 * 2);
            Tensor out5(big.p, DType::BF16, {512, 5});
            check(refuses([&] { o::softmax_pool(KV, S, 4, out5, stream); }), "more groups than the tokens hold are refused");
            const Tensor wrong_score(sd.p, DType::BF16, {512, 15});
            check(refuses([&] { o::softmax_pool(KV, wrong_score, 4, out, stream); }), "a score of another length is refused");
            const Tensor float_kv(kd.p, DType::FP32, {512, 16});
            check(refuses([&] { o::softmax_pool(float_kv, S, 4, out, stream); }), "a non-BF16 input is refused");
            Tensor aliased(kd.p, DType::BF16, {512, 4});
            check(refuses([&] { o::softmax_pool(KV, S, 4, aliased, stream); }), "an output overlapping an input is refused");
            t::cuda_check(cudaStreamSynchronize(stream), "sync");
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "EXCEPTION: %s\n", e.what());
        return 1;
    }
    if (failures != 0) {
        std::fprintf(stderr, "softmax_pool: %d failure(s)\n", failures);
        return 1;
    }
    std::printf("softmax_pool: passed\n");
    return 0;
}
