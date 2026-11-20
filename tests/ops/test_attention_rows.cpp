// attention_rows (include/ninfer/ops/attention_rows.h) against exact host formulas, which are the
// reachability rule and tools/dsv4/attention.py's `window_indices` and `select_candidate_blocks`
// written naively. All four ops are exact (integers and compares), so every comparison is bitwise.
//
// Cases: compressed_lengths for ratios 1, 2 and 7 at several starts; attention_row_lists at the
// start of a sequence (window rows before position 0 are -1), mid-sequence, with and without a
// compressed part; block_max_scores with ties, -inf scores, a partial last block and the newest
// block forced to +inf; mask_scores_outside_blocks fed by top_k_positions over block_max_scores,
// compared with a host reimplementation of `select_candidate_blocks` (score descending, lower block
// on a tie, kept only when finite or newest). Also a graph-captured call equal to the eager call
// and refusals.

#include "ninfer/ops/attention_rows.h"
#include "ninfer/ops/position_selection.h"

#include "ops/op_tester.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <limits>
#include <numeric>
#include <stdexcept>
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

    template <class E>
    std::vector<E> get(std::size_t n) const {
        std::vector<E> v(n);
        t::cuda_check(cudaMemcpy(v.data(), p, n * sizeof(E), cudaMemcpyDeviceToHost), "get");
        return v;
    }
};

template <class F>
bool refuses(F&& f) {
    try {
        f();
    } catch (const std::invalid_argument&) { return true; }
    return false;
}

constexpr float kInf = std::numeric_limits<float>::infinity();

void test_lengths(cudaStream_t stream) {
    for (const int ratio : {1, 2, 7}) {
        for (const int start : {0, 1, 5, 1000}) {
            const int T = 37;
            Device d(T * 4);
            Tensor out(d.p, DType::I32, {T});
            ninfer::ops::compressed_lengths(start, ratio, out, stream);
            t::cuda_check(cudaStreamSynchronize(stream), "sync");
            const auto got = d.get<std::int32_t>(T);
            bool ok        = true;
            for (int s = 0; s < T; ++s) { ok = ok && got[s] == (start + s + 1) / ratio; }
            check(ok, "compressed_lengths ratio " + std::to_string(ratio) + " start " +
                          std::to_string(start));
        }
    }
}

void test_row_lists(cudaStream_t stream) {
    const int window = 8;
    for (const int start : {0, 3, 7, 8, 100}) {
        for (const int Kt : {0, 5}) {
            const int T = 11, raw_base = 40, comp_base = 7;
            std::vector<std::int32_t> topk(static_cast<std::size_t>(std::max(Kt, 1)) * T);
            for (std::size_t i = 0; i < topk.size(); ++i) {
                topk[i] = (i % 3 == 0) ? -1 : static_cast<std::int32_t>(i % 29);
            }
            Device tk(topk.size() * 4), rd(static_cast<std::size_t>(window + Kt) * T * 4);
            tk.put(topk);
            Tensor TK(tk.p, DType::I32, {std::max(Kt, 1), T}),
                rows(rd.p, DType::I32, {window + Kt, T});
            ninfer::ops::attention_row_lists(Kt > 0 ? &TK : nullptr, start, window, raw_base,
                                             comp_base, rows, stream);
            t::cuda_check(cudaStreamSynchronize(stream), "sync");
            const auto got = rd.get<std::int32_t>(static_cast<std::size_t>(window + Kt) * T);
            bool ok        = true;
            for (int s = 0; s < T; ++s) {
                for (int j = 0; j < window + Kt; ++j) {
                    std::int32_t want;
                    if (j < window) {
                        want = (start + s - (window - 1) + j >= 0) ? raw_base + s + j : -1;
                    } else {
                        const std::int32_t v =
                            topk[static_cast<std::size_t>(s) * Kt + (j - window)];
                        want = v >= 0 ? comp_base + v : -1;
                    }
                    ok = ok && got[static_cast<std::size_t>(s) * (window + Kt) + j] == want;
                }
            }
            check(ok, "attention_row_lists start " + std::to_string(start) + " Kt " +
                          std::to_string(Kt));
        }
    }
}

// select_candidate_blocks for one column on the host: the kept block ids ascending.
std::vector<std::int32_t> host_blocks(const std::vector<float>& col, int reach, int bs,
                                      int topk_blocks) {
    const int n = static_cast<int>(col.size()), nb = (n + bs - 1) / bs;
    std::vector<float> score(nb, -kInf);
    for (int b = 0; b < nb; ++b) {
        for (int p = b * bs; p < std::min(n, (b + 1) * bs); ++p) {
            score[b] = std::max(score[b], p < reach ? col[p] : -kInf);
        }
    }
    if (reach >= 1) { score[(reach - 1) / bs] = kInf; }
    std::vector<int> order(nb);
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return score[a] > score[b]; });
    std::vector<std::int32_t> keep;
    for (int i = 0; i < std::min(topk_blocks, nb); ++i) {
        if (score[order[i]] > -kInf) { keep.push_back(order[i]); }
    }
    std::sort(keep.begin(), keep.end());
    return keep;
}

void test_blocks(cudaStream_t stream) {
    struct Case {
        int N, T, bs, kb;
    };

    for (const Case c :
         {Case{83, 9, 8, 4}, Case{64, 5, 8, 100}, Case{300, 6, 2, 20}, Case{1, 3, 8, 2}}) {
        std::uint32_t seed = 12345U + c.N;
        std::vector<float> scores(static_cast<std::size_t>(c.N) * c.T);
        std::vector<std::int32_t> valid(c.T);
        for (int s = 0; s < c.T; ++s) {
            valid[s] = static_cast<std::int32_t>(next(seed) % (c.N + 1));
            if (s == 0) { valid[s] = 0; }
            if (s == 1) { valid[s] = c.N; }
            for (int p = 0; p < c.N; ++p) {
                // Few distinct values: ties are common (the relu makes zeros in the real scores).
                float v = static_cast<float>(static_cast<int>(next(seed) >> 20) % 5) - 1.0F;
                scores[static_cast<std::size_t>(s) * c.N + p] = p < valid[s] ? v : -kInf;
            }
        }
        const int nb = (c.N + c.bs - 1) / c.bs;
        Device sd(scores.size() * 4), vd(c.T * 4), bd(static_cast<std::size_t>(nb) * c.T * 4),
            kd(static_cast<std::size_t>(c.kb) * c.T * 4);
        sd.put(scores);
        vd.put(valid);
        Tensor S(sd.p, DType::FP32, {c.N, c.T}), V(vd.p, DType::I32, {c.T}),
            B(bd.p, DType::FP32, {nb, c.T}), K(kd.p, DType::I32, {c.kb, c.T});
        ninfer::ops::block_max_scores(S, V, c.bs, B, stream);
        // The reachable blocks of column s are those below ceil(valid / bs): its own valid tensor.
        std::vector<std::int32_t> block_valid(c.T);
        for (int s = 0; s < c.T; ++s) { block_valid[s] = (valid[s] + c.bs - 1) / c.bs; }
        Device bv(c.T * 4);
        bv.put(block_valid);
        Tensor BV(bv.p, DType::I32, {c.T});
        ninfer::ops::top_k_positions(B, BV, c.kb, 0, K, stream);
        ninfer::ops::mask_scores_outside_blocks(K, c.bs, S, stream);
        t::cuda_check(cudaStreamSynchronize(stream), "sync");
        const auto gotb = bd.get<float>(static_cast<std::size_t>(nb) * c.T);
        const auto gots = sd.get<float>(scores.size());
        const auto gotk = kd.get<std::int32_t>(static_cast<std::size_t>(c.kb) * c.T);
        bool ok_b = true, ok_k = true, ok_m = true;
        for (int s = 0; s < c.T; ++s) {
            const std::vector<float> col(scores.begin() + static_cast<std::size_t>(s) * c.N,
                                         scores.begin() + static_cast<std::size_t>(s + 1) * c.N);
            const int reach = valid[s];
            for (int b = 0; b < nb; ++b) {
                float want = -kInf;
                if (b * c.bs < reach) {
                    want = b == (reach - 1) / c.bs ? kInf : -kInf;
                    if (want != kInf) {
                        for (int p = b * c.bs; p < std::min(c.N, (b + 1) * c.bs); ++p) {
                            want = std::max(want, col[p]);
                        }
                    }
                }
                ok_b = ok_b && gotb[static_cast<std::size_t>(s) * nb + b] == want;
            }
            const auto keep = host_blocks(col, reach, c.bs, c.kb);
            std::vector<std::int32_t> got_keep;
            for (int i = 0; i < c.kb; ++i) {
                const std::int32_t v = gotk[static_cast<std::size_t>(s) * c.kb + i];
                if (v >= 0) { got_keep.push_back(v); }
            }
            ok_k = ok_k && got_keep == keep;
            for (int p = 0; p < c.N; ++p) {
                const bool in    = std::binary_search(keep.begin(), keep.end(), p / c.bs);
                const float want = in ? col[p] : -kInf;
                ok_m             = ok_m && gots[static_cast<std::size_t>(s) * c.N + p] == want;
            }
        }
        const std::string label = "N " + std::to_string(c.N) + " bs " + std::to_string(c.bs) +
                                  " kb " + std::to_string(c.kb);
        check(ok_b, "block_max_scores " + label);
        check(ok_k, "top_k_positions over block maxima equals select_candidate_blocks " + label);
        check(ok_m, "mask_scores_outside_blocks " + label);
    }
}

void test_graph(cudaStream_t stream) {
    const int T = 21;
    Device d(T * 4), e(T * 4);
    Tensor out(d.p, DType::I32, {T}), out2(e.p, DType::I32, {T});
    ninfer::ops::compressed_lengths(9, 2, out2, stream);
    cudaGraph_t g        = nullptr;
    cudaGraphExec_t exec = nullptr;
    t::cuda_check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal), "capture");
    ninfer::ops::compressed_lengths(9, 2, out, stream);
    t::cuda_check(cudaStreamEndCapture(stream, &g), "end capture");
    t::cuda_check(cudaGraphInstantiate(&exec, g, 0), "instantiate");
    t::cuda_check(cudaGraphLaunch(exec, stream), "launch");
    t::cuda_check(cudaStreamSynchronize(stream), "sync");
    cudaGraphExecDestroy(exec);
    cudaGraphDestroy(g);
    check(d.get<std::int32_t>(T) == e.get<std::int32_t>(T),
          "a graph-captured compressed_lengths equals the eager call");
}

void test_refusals(cudaStream_t stream) {
    Device a(1024), b(1024);
    Tensor i32(a.p, DType::I32, {4}), f32(b.p, DType::FP32, {4, 2}), small(b.p, DType::I32, {4, 2});
    check(refuses([&] { ninfer::ops::compressed_lengths(-1, 2, i32, stream); }),
          "negative start refused");
    check(refuses([&] { ninfer::ops::compressed_lengths(0, 0, i32, stream); }), "ratio 0 refused");
    check(refuses([&] { ninfer::ops::attention_row_lists(nullptr, 0, 8, 0, 0, small, stream); }),
          "rows of the wrong width refused");
    check(refuses([&] { ninfer::ops::block_max_scores(f32, i32, 8, f32, stream); }),
          "block_max_scores with an aliased out refused");
    check(refuses([&] { ninfer::ops::mask_scores_outside_blocks(small, 0, f32, stream); }),
          "block size 0 refused");
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
        test_lengths(stream);
        test_row_lists(stream);
        test_blocks(stream);
        test_graph(stream);
        test_refusals(stream);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "EXCEPTION: %s\n", e.what());
        return 1;
    }
    if (failures != 0) {
        std::fprintf(stderr, "attention_rows: %d failure(s)\n", failures);
        return 1;
    }
    std::printf("attention_rows: passed\n");
    return 0;
}
