// clamped_swiglu and weighted_slot_sum against FP64 oracles of their formulas (include/ninfer/ops/
// moe_mix.h): the shared expert's clamped SwiGLU and the weighted sum of a token's routed expert
// outputs plus the shared output.
//
// SwiGLU cases: DeepSeek-V4.1-Flash's shared expert width (2,304) at T = 1, 7 and 33 with the limit
// 10 and activations wide enough that both clamps bite; no limit (0); a 4-D shape with an odd count;
// gates far below zero (silu underflowing to zero) and above the limit. Slot-sum cases: V4's mix
// (H = 5120, k = 6) at T = 1, 2 and 33; k = 1 and k = 64; a row count off the block size; 70,000 columns.
// Criteria (suite-owned): SwiGLU |got - ref| <= half a BF16 ulp of ref + 4e-6 |ref| (one BF16
// rounding of an FP32 evaluation with expf); slot sum |got - ref| <= half a BF16 ulp of ref +
// 2^-22 (k + 1) (sum_s |w_s o_s| + |extra|) (the FP32 chain, with cancellation allowed for);
// finite required. Also graph capture bitwise equal to eager, and refusals.

#include "ninfer/ops/moe_mix.h"

#include "ops/op_tester.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
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
    std::vector<std::uint16_t> bf16(std::size_t count) const {
        std::vector<std::uint16_t> out(count);
        t::cuda_check(cudaMemcpy(out.data(), p, count * 2, cudaMemcpyDeviceToHost), "get");
        return out;
    }
};

std::vector<std::uint16_t> bf16_values(std::size_t count, std::uint32_t seed, float sigma) {
    std::vector<std::uint16_t> v(count);
    std::uint32_t s = seed;
    for (auto& x : v) { x = t::f32_to_bf16(sigma * uniform(s) * 1.7F); }
    return v;
}

double half_ulp(double ref) { return ref == 0.0 ? 0.0 : std::ldexp(1.0, std::ilogb(std::fabs(ref)) - 8); }

template <class F>
bool refuses(F&& f) {
    try {
        f();
    } catch (const std::invalid_argument&) {
        return true;
    }
    return false;
}

template <class Call>
std::vector<std::uint16_t> run_graphed(Call&& call, Device& out, std::size_t count, cudaStream_t stream, bool graph) {
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
    return out.bf16(count);
}

// ---------------------------------------------------------------------------------- SwiGLU

void swiglu_case(const std::string& label, std::initializer_list<std::int32_t> shape, float limit, float gate_sigma, float up_sigma,
                 std::uint32_t seed, cudaStream_t stream, bool graph_too = false) {
    std::size_t count = 1;
    for (const auto d : shape) { count *= static_cast<std::size_t>(d); }
    const auto gate = bf16_values(count, seed, gate_sigma), up = bf16_values(count, seed + 1, up_sigma);
    Device gd(count * 2), ud(count * 2), od(count * 2);
    gd.put(gate);
    ud.put(up);
    t::cuda_check(cudaMemset(od.p, 0xFF, count * 2), "poison out");
    const Tensor G(gd.p, DType::BF16, shape), U(ud.p, DType::BF16, shape);
    Tensor out(od.p, DType::BF16, shape);
    const auto got = run_graphed([&] { ninfer::ops::clamped_swiglu(G, U, limit, out, stream); }, od, count, stream, false);
    double worst = 0;
    for (std::size_t i = 0; i < count; ++i) {
        double g = t::bf16_to_f32(gate[i]), u = t::bf16_to_f32(up[i]);
        if (limit > 0.0F) {
            g = std::min(g, static_cast<double>(limit));
            u = std::min(std::max(u, -static_cast<double>(limit)), static_cast<double>(limit));
        }
        const double ref  = g / (1.0 + std::exp(-g)) * u;
        const float value = t::bf16_to_f32(got[i]);
        if (!std::isfinite(value)) {
            check(false, label + ": a non-finite output");
            return;
        }
        worst = std::max(worst, std::fabs(value - ref) / (half_ulp(ref) + 4e-6 * std::fabs(ref) + 1e-30));
    }
    check(worst <= 1.0, label + ": within the criterion (worst " + std::to_string(worst) + " of the bound)");
    std::printf("swiglu %s: worst error / bound %.3f\n", label.c_str(), worst);
    if (graph_too) {
        Device od2(count * 2);
        Tensor out2(od2.p, DType::BF16, shape);
        const auto graphed = run_graphed([&] { ninfer::ops::clamped_swiglu(G, U, limit, out2, stream); }, od2, count, stream, true);
        check(graphed == got, label + ": a graph-captured call is bitwise the eager call");
    }
}

// ---------------------------------------------------------------------------------- slot sum

void slot_case(const std::string& label, int H, int k, int T, std::uint32_t seed, cudaStream_t stream, bool graph_too = false) {
    const auto outputs = bf16_values(static_cast<std::size_t>(H) * k * T, seed, 3.0F), extra = bf16_values(static_cast<std::size_t>(H) * T, seed + 1, 3.0F);
    std::vector<float> weights(static_cast<std::size_t>(k) * T);
    std::uint32_t s = seed + 2;
    for (auto& w : weights) { w = uniform(s) * 1.5F; }
    Device od(outputs.size() * 2), wd(weights.size() * 4), ed(extra.size() * 2), yd(static_cast<std::size_t>(H) * T * 2);
    od.put(outputs);
    wd.put(weights);
    ed.put(extra);
    t::cuda_check(cudaMemset(yd.p, 0xFF, static_cast<std::size_t>(H) * T * 2), "poison y");
    const Tensor O(od.p, DType::BF16, {H, k * T}), W(wd.p, DType::FP32, {k, T}), E(ed.p, DType::BF16, {H, T});
    Tensor Y(yd.p, DType::BF16, {H, T});
    const auto got = run_graphed([&] { ninfer::ops::weighted_slot_sum(O, W, E, Y, stream); }, yd, static_cast<std::size_t>(H) * T, stream, false);
    double worst = 0;
    for (int c = 0; c < T; ++c) {
        for (int h = 0; h < H; ++h) {
            double ref = 0, magnitude = std::fabs(t::bf16_to_f32(extra[static_cast<std::size_t>(c) * H + h]));
            for (int slot = 0; slot < k; ++slot) {
                const double term = static_cast<double>(weights[static_cast<std::size_t>(c) * k + slot]) *
                                    t::bf16_to_f32(outputs[(static_cast<std::size_t>(c) * k + slot) * H + h]);
                ref += term;
                magnitude += std::fabs(term);
            }
            ref += t::bf16_to_f32(extra[static_cast<std::size_t>(c) * H + h]);
            const float value = t::bf16_to_f32(got[static_cast<std::size_t>(c) * H + h]);
            if (!std::isfinite(value)) {
                check(false, label + ": a non-finite output");
                return;
            }
            worst = std::max(worst, std::fabs(value - ref) / (half_ulp(ref) + std::ldexp(1.0, -22) * (k + 1) * magnitude + 1e-30));
        }
    }
    check(worst <= 1.0, label + ": within the criterion (worst " + std::to_string(worst) + " of the bound)");
    std::printf("slot sum %s: worst error / bound %.3f\n", label.c_str(), worst);
    if (graph_too) {
        Device yd2(static_cast<std::size_t>(H) * T * 2);
        Tensor Y2(yd2.p, DType::BF16, {H, T});
        const auto graphed = run_graphed([&] { ninfer::ops::weighted_slot_sum(O, W, E, Y2, stream); }, yd2, static_cast<std::size_t>(H) * T, stream, true);
        check(graphed == got, label + ": a graph-captured call is bitwise the eager call");
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
        for (const int T : {1, 7, 33}) {
            swiglu_case("D=2304 T=" + std::to_string(T) + " limit 10 (both clamps bite)", {2304, T}, 10.0F, 9.0F, 9.0F, 11U + T, stream, T == 7);
        }
        swiglu_case("D=2304 T=5 no limit", {2304, 5}, 0.0F, 4.0F, 4.0F, 21U, stream);
        swiglu_case("4-D odd count [7,5,3,3]", {7, 5, 3, 3}, 10.0F, 9.0F, 9.0F, 22U, stream);
        swiglu_case("gates far below and above the limit", {1024, 4}, 10.0F, 40.0F, 40.0F, 23U, stream);
        swiglu_case("limit 2.5", {512, 3}, 2.5F, 3.0F, 3.0F, 24U, stream);

        for (const int T : {1, 2, 33}) { slot_case("H=5120 k=6 T=" + std::to_string(T), 5120, 6, T, 31U + T, stream, T == 2); }
        slot_case("H=1000 k=1 T=9", 1000, 1, 9, 41U, stream);
        slot_case("H=300 k=64 T=3", 300, 64, 3, 42U, stream);
        slot_case("H=64 k=6 T=70000 (past grid.y)", 64, 6, 70000, 43U, stream);

        {
            Device gd(2304 * 2), ud(2304 * 2), od(2304 * 2), wd(6 * 4), outs(5120 * 6 * 2), ex(5120 * 2), y(5120 * 2);
            const Tensor G(gd.p, DType::BF16, {2304, 1}), U(ud.p, DType::BF16, {2304, 1});
            Tensor out(od.p, DType::BF16, {2304, 1});
            namespace o = ninfer::ops;
            check(!refuses([&] { o::clamped_swiglu(G, U, 10.0F, out, stream); }), "a valid swiglu call is served");
            check(refuses([&] { o::clamped_swiglu(G, U, -1.0F, out, stream); }), "a negative limit is refused");
            check(refuses([&] { o::clamped_swiglu(G, U, std::numeric_limits<float>::quiet_NaN(), out, stream); }), "a NaN limit is refused");
            check(refuses([&] { o::clamped_swiglu(G, U, std::numeric_limits<float>::infinity(), out, stream); }), "an infinite limit is refused");
            const Tensor wrong(ud.p, DType::BF16, {2304, 2});
            check(refuses([&] { o::clamped_swiglu(G, wrong, 10.0F, out, stream); }), "mismatched shapes are refused");
            Tensor aliased(gd.p, DType::BF16, {2304, 1});
            check(refuses([&] { o::clamped_swiglu(G, U, 10.0F, aliased, stream); }), "an output overlapping an input is refused");
            const Tensor float_gate(gd.p, DType::FP32, {2304, 1});
            check(refuses([&] { o::clamped_swiglu(float_gate, U, 10.0F, out, stream); }), "a non-BF16 input is refused");

            const Tensor O(outs.p, DType::BF16, {5120, 6}), W(wd.p, DType::FP32, {6, 1}), E(ex.p, DType::BF16, {5120, 1});
            Tensor Y(y.p, DType::BF16, {5120, 1});
            check(!refuses([&] { o::weighted_slot_sum(O, W, E, Y, stream); }), "a valid slot-sum call is served");
            const Tensor few(outs.p, DType::BF16, {5120, 5});
            check(refuses([&] { o::weighted_slot_sum(few, W, E, Y, stream); }), "outputs that are not k x T columns are refused");
            const Tensor float_extra(ex.p, DType::FP32, {5120, 1});
            check(refuses([&] { o::weighted_slot_sum(O, W, float_extra, Y, stream); }), "a non-BF16 extra is refused");
            Tensor Y_aliased(ex.p, DType::BF16, {5120, 1});
            check(refuses([&] { o::weighted_slot_sum(O, W, E, Y_aliased, stream); }), "y overlapping extra is refused");
            const Tensor many(wd.p, DType::FP32, {65, 1});
            const Tensor many_out(outs.p, DType::BF16, {5120, 65});
            check(refuses([&] { o::weighted_slot_sum(many_out, many, E, Y, stream); }), "more than 64 slots are refused");
            t::cuda_check(cudaStreamSynchronize(stream), "sync");
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "EXCEPTION: %s\n", e.what());
        return 1;
    }
    if (failures != 0) {
        std::fprintf(stderr, "moe_mix: %d failure(s)\n", failures);
        return 1;
    }
    std::printf("moe_mix: passed\n");
    return 0;
}
