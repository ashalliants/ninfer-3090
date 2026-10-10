// The GPU side of offloaded_sparse_moe without the expert route:
//
//   - moe_route: exact top-10 of 512 against an FP64 ranking (ties to the lower id), the FP32
//     logits drawn from FP64 values; a column whose FP64 gap between ranks 10 and 11 is inside the
//     FP32 rounding bound is counted, not required to agree. Weights and the shared gate against
//     FP64.
//   - moe_dispatch: counts, offsets, jobs and entries exactly, on the one-CTA and three-kernel routes.
//   - moe_combine against FP64 over its represented inputs.

#include "core/arena.h"
#include "core/tensor.h"
#include "ninfer/ops/offloaded_sparse_moe.h"
#include "offloaded_moe_fixtures.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <random>
#include <string>
#include <vector>

namespace {

using namespace ninfer;
using namespace ninfer::test::moe;
namespace ops = ninfer::ops;

void cuda_ok(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "CUDA %s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

template <class T>
std::vector<T> download(const void* device, std::size_t count) {
    std::vector<T> out(count);
    cuda_ok(cudaMemcpy(out.data(), device, count * sizeof(T), cudaMemcpyDeviceToHost), "download");
    return out;
}

// ------------------------------------------------------------------------------------- routing

struct RouteRef {
    std::vector<int> ids;
    std::vector<double> weights;
    double shared = 0.0;
    double gap    = 0.0; // FP64 gap between ranks k and k + 1
};

RouteRef route_oracle(const double* logits, int experts, int k) {
    std::vector<int> order(experts);
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return logits[a] > logits[b]; });
    RouteRef r;
    r.ids.assign(order.begin(), order.begin() + k);
    r.gap    = k < experts ? logits[order[k - 1]] - logits[order[k]] : 1e300;
    double s = 0.0;
    for (int i = 0; i < k; ++i) { s += std::exp(logits[r.ids[i]] - logits[r.ids[0]]); }
    for (int i = 0; i < k; ++i) { r.weights.push_back(std::exp(logits[r.ids[i]] - logits[r.ids[0]]) / s); }
    r.shared = 1.0 / (1.0 + std::exp(-logits[experts]));
    return r;
}

struct DeviceRouting {
    DeviceBuffer logits, ids, weights, shared_gate, dispatch;
    ops::MoeRouting routing;
    ops::MoeDispatch view;
    int experts = 0, k = 0, columns = 0;

    DeviceRouting(const std::vector<float>& host_logits, int e, int top_k, int t)
        : logits(host_logits.size() * 4), ids(static_cast<std::size_t>(top_k) * t * 4),
          weights(static_cast<std::size_t>(top_k) * t * 4), shared_gate(static_cast<std::size_t>(t) * 4),
          dispatch(ops::moe_dispatch_bytes(e, top_k * t)), experts(e), k(top_k), columns(t) {
        logits.copy_from_host(host_logits.data(), host_logits.size() * 4);
        routing.ids         = Tensor(ids.p, DType::I32, {top_k, t});
        routing.weights     = Tensor(weights.p, DType::FP32, {top_k, t});
        routing.shared_gate = Tensor(shared_gate.p, DType::FP32, {t});
        view                = ops::carve_moe_dispatch(dispatch.p, e, top_k * t);
        const Tensor l(logits.p, DType::FP32, {e + 1, t});
        ops::moe_route(l, top_k, routing, nullptr);
        ops::moe_dispatch(routing, e, view, nullptr, nullptr);
        cuda_ok(cudaDeviceSynchronize(), "route/dispatch");
    }
};

// FP32 logits [E + 1, T] drawn from FP64 values; columns `ties` hold exact ties at the top.
std::vector<double> random_logits(int experts, int columns, std::mt19937& rng, std::vector<float>& as_f32) {
    std::normal_distribution<double> n(0.0, 2.0);
    std::vector<double> l(static_cast<std::size_t>(experts + 1) * columns);
    for (auto& v : l) { v = n(rng); }
    as_f32.resize(l.size());
    for (std::size_t i = 0; i < l.size(); ++i) { as_f32[i] = static_cast<float>(l[i]); }
    return l;
}

void test_route_and_dispatch() {
    std::mt19937 rng(101);
    constexpr int kExperts = 512, kTopK = 10;
    for (int columns : {1, 9, 257}) {
        std::vector<float> f32;
        auto l64 = random_logits(kExperts, columns, rng, f32);
        // Exact ties: column 0 all equal (ids 0..9), column 1 (when present) a 3-way tie for first
        // and a tie across the selection boundary.
        for (int e = 0; e < kExperts; ++e) { f32[e] = 0.25F; }
        if (columns > 1) {
            float* c = f32.data() + (kExperts + 1);
            for (int e : {400, 7, 129}) { c[e] = 90.0F; }
            for (int e = 0; e < 12; ++e) { c[300 + e] = 50.0F; }
        }
        DeviceRouting r(f32, kExperts, kTopK, columns);
        const auto ids     = download<std::int32_t>(r.ids.p, static_cast<std::size_t>(kTopK) * columns);
        const auto weights = download<float>(r.weights.p, static_cast<std::size_t>(kTopK) * columns);
        const auto shared  = download<float>(r.shared_gate.p, columns);
        int boundary = 0;
        for (int t = 0; t < columns; ++t) {
            std::vector<double> col(kExperts + 1);
            for (int e = 0; e <= kExperts; ++e) { col[e] = f32[static_cast<std::size_t>(t) * (kExperts + 1) + e]; }
            const auto exact = route_oracle(col.data(), kExperts, kTopK); // on the FP32 values
            for (int i = 0; i < kTopK; ++i) {
                check(ids[static_cast<std::size_t>(t) * kTopK + i] == exact.ids[i], "route ids differ from the exact ranking");
                const double w = weights[static_cast<std::size_t>(t) * kTopK + i];
                check(std::fabs(w - exact.weights[i]) <= 2e-6 * exact.weights[i] + 1e-12, "route weight outside 2e-6");
            }
            check(std::fabs(shared[t] - exact.shared) <= 1e-6, "shared gate outside 1e-6");
            // Against the FP64 values the FP32 logits came from: the same set wherever the FP64 gap at
            // the boundary exceeds the FP32 rounding of the two logits (2^-24 relative each).
            if (t >= 2) {
                const auto ref   = route_oracle(l64.data() + static_cast<std::size_t>(t) * (kExperts + 1), kExperts, kTopK);
                const double top = std::fabs(l64[static_cast<std::size_t>(t) * (kExperts + 1) + ref.ids[kTopK - 1]]);
                if (ref.gap > 2.0 * top * 0x1p-24 + 1e-30) {
                    std::vector<int> a(ids.begin() + static_cast<std::ptrdiff_t>(t) * kTopK,
                                       ids.begin() + static_cast<std::ptrdiff_t>(t + 1) * kTopK);
                    auto b = ref.ids;
                    std::sort(a.begin(), a.end());
                    std::sort(b.begin(), b.end());
                    check(a == b, "route set differs from the FP64 top-10 outside the tie bound");
                } else {
                    ++boundary;
                }
            }
        }
        check(ids[0] == 0 && ids[9] == 9, "an all-equal column selects ids 0..9");
        if (columns > 1) {
            const std::int32_t* c = ids.data() + kTopK;
            check(c[0] == 7 && c[1] == 129 && c[2] == 400 && c[3] == 300 && c[9] == 306,
                  "exact ties go to the lower id, in selection order");
        }
        // Dispatch, exactly.
        const int entries = kTopK * columns;
        const auto counts  = download<std::int32_t>(r.view.counts, kExperts);
        const auto offsets = download<std::int32_t>(r.view.offsets, kExperts + 1);
        const auto jobs    = download<std::int32_t>(r.view.jobs, kExperts);
        const auto job_count = download<std::int32_t>(r.view.job_count, 1)[0];
        const auto ent     = download<std::int32_t>(r.view.entries, entries);
        std::vector<int> want_counts(kExperts, 0);
        for (int i = 0; i < entries; ++i) { ++want_counts[ids[i]]; }
        int at = 0, used = 0;
        for (int e = 0; e < kExperts; ++e) {
            check(counts[e] == want_counts[e] && offsets[e] == at, "dispatch counts/offsets");
            std::vector<int> got(ent.begin() + at, ent.begin() + at + want_counts[e]);
            std::sort(got.begin(), got.end());
            std::vector<int> want;
            for (int i = 0; i < entries; ++i) {
                if (ids[i] == e) { want.push_back(i); }
            }
            check(got == want, "dispatch entries of an expert");
            if (want_counts[e] > 0) { check(jobs[used++] == e, "dispatch jobs ascending"); }
            at += want_counts[e];
        }
        check(offsets[kExperts] == entries && job_count == used, "dispatch totals");
        std::printf("route+dispatch T=%d: exact; %d column(s) inside the FP64 tie bound\n", columns, boundary);
    }
}

// moe_combine against FP64 over its represented inputs (BF16 outputs and shared, FP32 routing).
void test_combine() {
    std::mt19937 rng(151);
    constexpr int kExperts = 512, kTopK = 10;
    for (int columns : {1, 7, 64}) {
        std::vector<float> f32;
        (void)random_logits(kExperts, columns, rng, f32);
        DeviceRouting r(f32, kExperts, kTopK, columns);
        const auto weights = download<float>(r.weights.p, static_cast<std::size_t>(kTopK) * columns);
        const auto gate    = download<float>(r.shared_gate.p, columns);
        const auto outputs = random_x(kTopK * columns, rng);
        const auto shared  = random_x(columns, rng);
        DeviceBuffer d_out(outputs.size() * 2), d_shared(shared.size() * 2), d_y(shared.size() * 2);
        d_out.copy_from_host(outputs.data(), outputs.size() * 2);
        d_shared.copy_from_host(shared.data(), shared.size() * 2);
        const Tensor ot(d_out.p, DType::BF16, {om::kHidden, kTopK * columns});
        const Tensor st(d_shared.p, DType::BF16, {om::kHidden, columns});
        Tensor yt(d_y.p, DType::BF16, {om::kHidden, columns});
        ops::moe_combine(ot, r.routing, st, yt, nullptr);
        cuda_ok(cudaDeviceSynchronize(), "combine");
        const auto got = download<std::uint16_t>(d_y.p, shared.size());
        std::vector<double> want(shared.size());
        for (int t = 0; t < columns; ++t) {
            for (int d = 0; d < om::kHidden; ++d) {
                double s = static_cast<double>(gate[t]) * bf16_value(shared[static_cast<std::size_t>(t) * om::kHidden + d]);
                for (int k = 0; k < kTopK; ++k) {
                    s += static_cast<double>(weights[static_cast<std::size_t>(t) * kTopK + k]) *
                         bf16_value(outputs[(static_cast<std::size_t>(t) * kTopK + k) * om::kHidden + d]);
                }
                want[static_cast<std::size_t>(t) * om::kHidden + d] = s;
            }
        }
        const auto crit = measure(got, want);
        check(accept(crit), "moe_combine outside the BF16 criterion against FP64");
        std::printf("combine T=%d vs FP64: normwise %.3e, worst %.3e\n", columns, crit.normwise, crit.worst);
    }
}

} // namespace

int main() {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
        std::printf("no CUDA device: skipped\n");
        return 77;
    }
    test_route_and_dispatch();
    test_combine();
    std::printf("offloaded_moe_layer: passed\n");
    return 0;
}
