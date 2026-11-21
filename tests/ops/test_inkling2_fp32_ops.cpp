// Written for NInfer-3090.
//
// The FP32 Ops of Inkling-2-Mini's text model (fp32_rows.h, depth_attention.h, short_conv.h,
// relative_bias_attention.h, moe_situ.h) against the FP64 oracles of tests/ops/inkling2_reference.h, which
// tests/ops/inkling2_reference_test.cpp ties to the NumPy reference and through it to llama.cpp. Shapes are the
// model's (H 2304, 24/8 heads of 128, d_rel 16, conv kernel 4, 128 experts of I 1152, top 8) except where a
// smaller case reaches a boundary faster (a KV ring that wraps, a relative extent shorter than the context).
//
// Criterion (suite-owned): every output within tol * (1 + |ref|) of the oracle, with tol 2e-6 (about 16 FP32 ulps:
// the reductions are FP32 chains of at most a few thousand bounded terms) and, for the experts, within 1e-5 of the
// output's largest magnitude (two chained K = 2304 and 1152 dot products of random data, which cancel); finite
// required. On sm_70 (2026-11-20) the largest errors were 2.0e-7 and 6.7e-7. Stateful Ops are run as two calls and must equal one call over the whole
// sequence; routing must select the oracle's experts except at a score tie within 1e-6.

#include "ninfer/ops/depth_attention.h"
#include "ninfer/ops/fp32_rows.h"
#include "ninfer/ops/moe_situ.h"
#include "ninfer/ops/projection_fp32.h"
#include "ninfer/ops/relative_bias_attention.h"
#include "ninfer/ops/tensor_core_fp32.h"
#include "ninfer/ops/short_conv.h"

#include "core/arena.h"
#include "ops/inkling2_reference.h"
#include "ops/op_tester.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <iterator>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

namespace t   = ninfer::test;
namespace ref = ninfer::test::inkling2_ref;
using ninfer::DType;
using ninfer::Tensor;
using Vec = std::vector<double>;

int failures = 0;

void check(bool ok, const std::string& what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what.c_str());
        ++failures;
    }
}

std::uint32_t seed_state = 12345;
float uniform(float lo = -1.0F, float hi = 1.0F) {
    seed_state = seed_state * 1664525U + 1013904223U;
    return lo + (hi - lo) * static_cast<float>(seed_state >> 8) / static_cast<float>(1 << 24);
}
std::vector<float> randoms(std::size_t n, float lo = -1.0F, float hi = 1.0F) {
    std::vector<float> v(n);
    for (auto& x : v) { x = uniform(lo, hi); }
    return v;
}
Vec widen(const std::vector<float>& v) { return Vec(v.begin(), v.end()); }

// Device copy of host floats, kept alive for the test.
struct Buf {
    ninfer::DeviceBuffer b;
    explicit Buf(const std::vector<float>& v) : b(t::to_device(v)) {}
    explicit Buf(std::size_t n) : b(n * 4) { b.fill(0); }
    Tensor tensor(std::initializer_list<std::int32_t> ne, DType dtype = DType::FP32) { return Tensor(b.p, dtype, ne); }
    std::vector<float> get(std::size_t n) const { return t::from_device<float>(b, n); }
};

void compare(const std::string& label, const std::vector<float>& got, const Vec& want, double tol) {
    check(got.size() == want.size(), label + ": size");
    double worst = 0;
    std::size_t at = 0;
    for (std::size_t i = 0; i < std::min(got.size(), want.size()); ++i) {
        if (!std::isfinite(got[i])) {
            check(false, label + ": non-finite output");
            return;
        }
        const double e = std::fabs(double(got[i]) - want[i]) / (1.0 + std::fabs(want[i]));
        if (e > worst) {
            worst = e;
            at    = i;
        }
    }
    std::printf("%-44s max scaled error %.2e (index %zu)\n", label.c_str(), worst, at);
    check(worst <= tol, label + ": outside tolerance");
}

// Host [T, C] row-major (oracle) <-> device [C, T] columns are the same memory order.

void test_rows() {
    const int h = 2304, cols = 3;
    auto x = randoms(h * cols, -3, 3), w = randoms(h, 0.5, 1.5), b = randoms(h * cols, -2, 2);
    Buf dx(x), dw(w), db(b), out(h * cols), wide(2 * h * cols);
    Tensor tx = dx.tensor({h, cols}), tw = dw.tensor({h}), to = out.tensor({h, cols});
    ninfer::ops::rmsnorm_fp32(tx, &tw, 1e-6F, to, nullptr);
    Vec want;
    for (int c = 0; c < cols; ++c) {
        const Vec u = ref::unit_rms(Vec(x.begin() + c * h, x.begin() + (c + 1) * h));
        for (int d = 0; d < h; ++d) { want.push_back(u[d] * w[d]); }
    }
    compare("rmsnorm_fp32 weighted", out.get(h * cols), want, 2e-6);
    // Unweighted into the upper half of a [2H, T] tensor (a strided output).
    Tensor upper = wide.tensor({2 * h, cols}).slice(0, h, h);
    ninfer::ops::rmsnorm_fp32(tx, nullptr, 1e-6F, upper, nullptr);
    const auto all = wide.get(2 * h * cols);
    std::vector<float> got;
    for (int c = 0; c < cols; ++c) { got.insert(got.end(), all.begin() + c * 2 * h + h, all.begin() + (c + 1) * 2 * h); }
    Vec unit;
    for (int c = 0; c < cols; ++c) {
        const Vec u = ref::unit_rms(Vec(x.begin() + c * h, x.begin() + (c + 1) * h));
        unit.insert(unit.end(), u.begin(), u.end());
    }
    compare("rmsnorm_fp32 unit, strided out", got, unit, 2e-6);
    // In place (out == x), as the program normalizes the embedding, the FFN input and the final state.
    Buf inplace(x);
    Tensor ti = inplace.tensor({h, cols});
    ninfer::ops::rmsnorm_fp32(ti, &tw, 1e-6F, ti, nullptr);
    compare("rmsnorm_fp32 in place", inplace.get(h * cols), want, 2e-6);

    Tensor tb = db.tensor({h, cols});
    ninfer::ops::sigmoid_gate_fp32(tx, tb, to, nullptr);
    Vec gated(x.size());
    for (std::size_t i = 0; i < x.size(); ++i) { gated[i] = x[i] / (1.0 + std::exp(-double(b[i]))); }
    compare("sigmoid_gate_fp32", out.get(h * cols), gated, 2e-6);
    ninfer::ops::tanh_softcap_fp32(tx, 2.0F, to, nullptr);
    Vec capped(x.size());
    for (std::size_t i = 0; i < x.size(); ++i) { capped[i] = 2.0 * std::tanh(x[i] / 2.0); }
    compare("tanh_softcap_fp32", out.get(h * cols), capped, 2e-6);
    ninfer::ops::add_fp32(tx, tb, to, nullptr);
    Vec sum(x.size());
    for (std::size_t i = 0; i < x.size(); ++i) { sum[i] = double(x[i]) + b[i]; }
    compare("add_fp32", out.get(h * cols), sum, 1e-7);
}

void test_mix_and_control() {
    const int h = 2304, cols = 3, sources = 5, queries = 3, rank = 64;
    std::vector<std::vector<float>> src;
    std::vector<Buf> dev;
    dev.reserve(sources);
    for (int j = 0; j < sources; ++j) {
        src.push_back(randoms(h * cols, -1.0F - j, 1.0F + j));
        dev.emplace_back(src.back());
    }
    auto q = randoms(h * queries, -0.3F, 0.3F);
    Buf dq(q);
    std::vector<Buf> outs;
    outs.reserve(queries);
    for (int i = 0; i < queries; ++i) { outs.emplace_back(static_cast<std::size_t>(h * cols)); }
    std::vector<Tensor> st, ot;
    for (auto& d : dev) { st.push_back(d.tensor({h, cols})); }
    for (auto& o : outs) { ot.push_back(o.tensor({h, cols})); }
    std::vector<const Tensor*> sp;
    std::vector<Tensor*> op;
    for (auto& s : st) { sp.push_back(&s); }
    for (auto& o : ot) { op.push_back(&o); }
    ninfer::DeviceArena mix_arena(ninfer::ops::depth_attention_mix_workspace_bytes(h, cols, sources) + 4096);
    ninfer::ops::depth_attention_mix(sp, dq.tensor({h, queries}), 1e-6F, mix_arena, op, nullptr);
    for (int i = 0; i < queries; ++i) {
        Vec want;
        for (int c = 0; c < cols; ++c) {
            std::vector<Vec> s;
            for (int j = 0; j < sources; ++j) { s.emplace_back(src[j].begin() + c * h, src[j].begin() + (c + 1) * h); }
            const Vec m = ref::mix(s, Vec(q.begin() + i * h, q.begin() + (i + 1) * h), h);
            want.insert(want.end(), m.begin(), m.end());
        }
        compare("depth_attention_mix query " + std::to_string(i), outs[i].get(h * cols), want, 2e-6);
    }

    // Three controllers in one call against the FP64 oracle over the same BF16 weights: job 1 writes over its c,
    // job 2 over its m.
    struct Job {
        std::vector<float> c, m, down, gain, forget;
    };
    std::vector<Job> host(3);
    std::vector<Buf> cs, ms, outs_c;
    std::vector<ninfer::DeviceBuffer> wd, wg, wf;
    cs.reserve(3), ms.reserve(3), outs_c.reserve(3);
    for (int j = 0; j < 3; ++j) {
        Job& job = host[j];
        job.c = randoms(h * cols, -1.0F - j, 1.0F + j), job.m = randoms(h * cols);
        job.down = randoms(rank * 2 * h, -0.02F, 0.02F), job.gain = randoms(h * rank, -0.1F, 0.1F), job.forget = randoms(h * rank, -0.1F, 0.1F);
        t::round_to_bf16(job.down), t::round_to_bf16(job.gain), t::round_to_bf16(job.forget);
        cs.emplace_back(job.c), ms.emplace_back(job.m), outs_c.emplace_back(static_cast<std::size_t>(h * cols));
        wd.push_back(t::to_device_bf16(job.down)), wg.push_back(t::to_device_bf16(job.gain)), wf.push_back(t::to_device_bf16(job.forget));
    }
    std::vector<Tensor> tc, tm, tout, tdown, tgain, tforget;
    for (int j = 0; j < 3; ++j) {
        tc.push_back(cs[j].tensor({h, cols})), tm.push_back(ms[j].tensor({h, cols})), tout.push_back(outs_c[j].tensor({h, cols}));
        tdown.emplace_back(wd[j].p, DType::BF16, std::initializer_list<std::int32_t>{2 * h, rank});
        tgain.emplace_back(wg[j].p, DType::BF16, std::initializer_list<std::int32_t>{rank, h});
        tforget.emplace_back(wf[j].p, DType::BF16, std::initializer_list<std::int32_t>{rank, h});
    }
    ninfer::ops::LoopControllerJob jobs[3];
    for (int j = 0; j < 3; ++j) {
        Tensor* target = j == 1 ? &tc[j] : j == 2 ? &tm[j] : &tout[j];
        jobs[j] = {&tc[j], &tm[j], &tdown[j], &tgain[j], &tforget[j], target};
    }
    ninfer::DeviceArena control_arena(ninfer::ops::loop_controller_workspace_bytes(h, rank, cols, 3));
    ninfer::ops::loop_controller_fp32(jobs, 1e-6F, control_arena, nullptr);
    for (int j = 0; j < 3; ++j) {
        Vec want;
        for (int c = 0; c < cols; ++c) {
            const Vec vc(host[j].c.begin() + c * h, host[j].c.begin() + (c + 1) * h), vm(host[j].m.begin() + c * h, host[j].m.begin() + (c + 1) * h);
            const Vec y = ref::control(vc, vm, widen(host[j].down), widen(host[j].gain), widen(host[j].forget));
            want.insert(want.end(), y.begin(), y.end());
        }
        Buf& got = j == 1 ? cs[j] : j == 2 ? ms[j] : outs_c[j];
        compare("loop_controller_fp32 job " + std::to_string(j) + (j == 1 ? " (out = c)" : j == 2 ? " (out = m)" : ""), got.get(h * cols),
                want, 2e-6);
    }
    bool threw = false;
    try {
        (void)ninfer::ops::loop_controller_workspace_bytes(2300, rank, cols, 1);
    } catch (const std::invalid_argument&) { threw = true; }
    check(threw, "loop_controller_workspace_bytes accepted H = 2300");
}

void test_short_conv(bool dynamic) {
    const int channels = dynamic ? 1024 : 2304, k = 4, head = 128, cols = 5, heads = channels / head;
    auto x = randoms(channels * cols), taps = randoms(channels * k, -0.5F, 0.5F), dyn = randoms(heads * k * cols, -0.5F, 0.5F);
    Vec history((k - 1) * channels, 0.0);
    const Vec dv = widen(dyn);
    const Vec want = ref::short_conv(widen(x), history, widen(taps), channels, k, dynamic ? &dv : nullptr, head);
    Buf dx(x), dt(taps), dd(dyn), dh(static_cast<std::size_t>(channels * (k - 1))), dout(static_cast<std::size_t>(channels * cols));
    Tensor th = dh.tensor({channels, k - 1});
    // Two calls (2 then 3 columns) carry the history.
    for (int part = 0; part < 2; ++part) {
        const int s = part == 0 ? 0 : 2, len = part == 0 ? 2 : 3;
        Tensor tx = dx.tensor({channels, cols}).slice(1, s, len), to = dout.tensor({channels, cols}).slice(1, s, len);
        Tensor tdyn = dd.tensor({heads * k, cols}).slice(1, s, len);
        ninfer::ops::short_conv_fp32(tx, dt.tensor({k, channels}), dynamic ? &tdyn : nullptr, head, th, to, nullptr);
    }
    const std::string label = dynamic ? "short_conv_fp32 dynamic (2 + 3 columns)" : "short_conv_fp32 static (2 + 3 columns)";
    compare(label, dout.get(channels * cols), want, 2e-6);
    std::vector<float> hist_got = dh.get(channels * (k - 1));
    compare(label + " history", hist_got, history, 0);
}

void test_clips() {
    const int heads = 24, d = 128, cols = 3;
    auto x = randoms(heads * d * cols, -4, 4), scale = randoms(heads * d, 0.5, 2);
    Buf dx(x), ds(scale);
    Tensor tx = dx.tensor({heads * d, cols});
    const Tensor ts = ds.tensor({d, heads});
    // Position 20000 as a host start of 19000 plus a device offset of 1000.
    const std::int64_t offset = 1000;
    ninfer::DeviceBuffer doff(sizeof offset);
    check(cudaMemcpy(doff.p, &offset, sizeof offset, cudaMemcpyHostToDevice) == cudaSuccess, "offset upload");
    const ninfer::ops::PositionLogScale tau{19000, 8192, 0.1F, static_cast<const std::int64_t*>(doff.p)};
    ninfer::ops::head_l2_clip_fp32(tx, d, &ts, 16.0F * std::sqrt(2.0F), &tau, nullptr);
    Vec want;
    for (int c = 0; c < cols; ++c)
        for (int hh = 0; hh < heads; ++hh) {
            Vec v(x.begin() + (c * heads + hh) * d, x.begin() + (c * heads + hh + 1) * d);
            Vec sc(scale.begin() + hh * d, scale.begin() + (hh + 1) * d);
            v = ref::l2_clip(v, 16.0 * std::sqrt(2.0), sc.data());
            for (double& e : v) { e *= ref::log_scale(20000 + c, 8192, 0.1); }
            want.insert(want.end(), v.begin(), v.end());
        }
    compare("head_l2_clip_fp32 (scale, tau)", dx.get(x.size()), want, 2e-6);
    Buf dy(x);
    Tensor ty = dy.tensor({heads * d, cols});
    ninfer::ops::head_unit_rms_fp32(ty, d, 1e-6F, nullptr);
    Vec unit;
    for (int i = 0; i < heads * cols; ++i) {
        const Vec u = ref::unit_rms(Vec(x.begin() + i * d, x.begin() + (i + 1) * d));
        unit.insert(unit.end(), u.begin(), u.end());
    }
    compare("head_unit_rms_fp32", dy.get(x.size()), unit, 2e-6);
}

bool same_bits(const std::vector<float>& a, const std::vector<float>& b) {
    return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
}

// The batched launches give each job the same bits as its own call.
void test_batches() {
    namespace ops = ninfer::ops;
    const int h = 2304, cols = 3;
    {
        // Three weighted norms, BF16 out (the q, k, v norms), the second in a strided slice.
        std::vector<Buf> xs, ws;
        for (int j = 0; j < 3; ++j) {
            xs.emplace_back(randoms(h * cols, -3, 3));
            ws.emplace_back(randoms(h, 0.5, 1.5));
        }
        Buf batched(3 * h * cols), single(3 * h * cols); // BF16 [H, T] x 3 fits in half of each
        std::vector<Tensor> tx, tw, tb, ts;
        for (int j = 0; j < 3; ++j) {
            tx.push_back(xs[j].tensor({h, cols}));
            tw.push_back(ws[j].tensor({h}));
            const auto out = [&](Buf& b) {
                Tensor wide(static_cast<char*>(b.b.p) + static_cast<std::size_t>(j) * h * cols * 2 * 2, DType::BF16, {2 * h, cols});
                return j == 1 ? wide.slice(0, h, h) : Tensor(wide.data, DType::BF16, {h, cols});
            };
            tb.push_back(out(batched));
            ts.push_back(out(single));
        }
        const ops::RmsNormJob jobs[] = {{&tx[0], &tw[0], &tb[0]}, {&tx[1], &tw[1], &tb[1]}, {&tx[2], nullptr, &tb[2]}};
        ops::rmsnorm_fp32_batch(jobs, 1e-6F, nullptr);
        ops::rmsnorm_fp32(tx[0], &tw[0], 1e-6F, ts[0], nullptr);
        ops::rmsnorm_fp32(tx[1], &tw[1], 1e-6F, ts[1], nullptr);
        ops::rmsnorm_fp32(tx[2], nullptr, 1e-6F, ts[2], nullptr);
        check(same_bits(batched.get(3 * h * cols), single.get(3 * h * cols)), "rmsnorm_fp32_batch: bits differ from single calls");
        std::printf("%-44s bit-exact\n", "rmsnorm_fp32_batch (3 jobs, BF16, strided)");
    }
    {
        // q clip with log scale, r clip with log scale (d 16), k clip, v unit rms: one launch.
        const int heads = 24, kv = 8, d = 128, dr = 16;
        auto q = randoms(heads * d * cols, -4, 4), r = randoms(heads * dr * cols, -4, 4), k = randoms(kv * d * cols, -40, 40),
             v = randoms(kv * d * cols, -3, 3);
        auto qs = randoms(heads * d, 0.5, 2), rs = randoms(heads * dr, 0.5, 2);
        Buf bq(q), br(r), bk(k), bv(v), sq(q), sr(r), sk(k), sv(v), dqs(qs), drs(rs);
        const Tensor tqs = dqs.tensor({d, heads}), trs = drs.tensor({dr, heads});
        const ops::PositionLogScale tau{20000, 8192, 0.1F, nullptr};
        Tensor a[] = {bq.tensor({heads * d, cols}), br.tensor({heads * dr, cols}), bk.tensor({kv * d, cols}), bv.tensor({kv * d, cols})};
        Tensor b[] = {sq.tensor({heads * d, cols}), sr.tensor({heads * dr, cols}), sk.tensor({kv * d, cols}), sv.tensor({kv * d, cols})};
        ops::HeadNormJob jobs[4];
        jobs[0].x = &a[0], jobs[0].head_dim = d, jobs[0].scale = &tqs, jobs[0].threshold = 22.6F, jobs[0].log_scale = &tau;
        jobs[1].x = &a[1], jobs[1].head_dim = dr, jobs[1].scale = &trs, jobs[1].threshold = 5.0F, jobs[1].log_scale = &tau;
        jobs[2].x = &a[2], jobs[2].head_dim = d, jobs[2].threshold = 22.6F;
        jobs[3].x = &a[3], jobs[3].head_dim = d, jobs[3].unit_rms = true, jobs[3].eps = 1e-6F;
        ops::head_norms_fp32(jobs, nullptr);
        ops::head_l2_clip_fp32(b[0], d, &tqs, 22.6F, &tau, nullptr);
        ops::head_l2_clip_fp32(b[1], dr, &trs, 5.0F, &tau, nullptr);
        ops::head_l2_clip_fp32(b[2], d, nullptr, 22.6F, nullptr, nullptr);
        ops::head_unit_rms_fp32(b[3], d, 1e-6F, nullptr);
        const bool ok = same_bits(bq.get(q.size()), sq.get(q.size())) && same_bits(br.get(r.size()), sr.get(r.size())) &&
                        same_bits(bk.get(k.size()), sk.get(k.size())) && same_bits(bv.get(v.size()), sv.get(v.size()));
        check(ok, "head_norms_fp32: bits differ from single calls");
        // And the batch against the oracle for the k clip (no scale): every head clipped to 22.6.
        Vec want;
        for (int i = 0; i < kv * cols; ++i) {
            const Vec u = ref::l2_clip(Vec(k.begin() + i * d, k.begin() + (i + 1) * d), 22.6, nullptr);
            want.insert(want.end(), u.begin(), u.end());
        }
        compare("head_norms_fp32 (4 jobs) k clip vs oracle", bk.get(k.size()), want, 2e-6);
    }
    {
        // Two dynamic convolutions (k and v), then a static one fused with its residual add, over 2 + 1 columns.
        const int channels = 1024, k = 4, head = 128, heads = channels / head;
        Buf x0(randoms(channels * cols)), x1(randoms(channels * cols)), t0(randoms(channels * k, -0.5F, 0.5F)),
            t1(randoms(channels * k, -0.5F, 0.5F)), dyn0(randoms(heads * k * cols, -0.5F, 0.5F)), dyn1(randoms(heads * k * cols, -0.5F, 0.5F)),
            res(randoms(channels * cols, -2, 2));
        Buf hb0(channels * (k - 1)), hb1(channels * (k - 1)), hs0(channels * (k - 1)), hs1(channels * (k - 1)), hb2(channels * (k - 1)),
            hs2(channels * (k - 1));
        Buf ob0(channels * cols), ob1(channels * cols), os0(channels * cols), os1(channels * cols), sumb(channels * cols),
            outs(channels * cols), sums(channels * cols);
        const Tensor tt0 = t0.tensor({k, channels}), tt1 = t1.tensor({k, channels});
        Tensor thb0 = hb0.tensor({channels, k - 1}), thb1 = hb1.tensor({channels, k - 1}), ths0 = hs0.tensor({channels, k - 1}),
               ths1 = hs1.tensor({channels, k - 1}), thb2 = hb2.tensor({channels, k - 1}), ths2 = hs2.tensor({channels, k - 1});
        for (int part = 0; part < 2; ++part) {
            const int s = part == 0 ? 0 : 2, len = part == 0 ? 2 : 1;
            const auto cut = [&](Buf& b, int rows) { return b.tensor({rows, cols}).slice(1, s, len); };
            const Tensor tx0 = cut(x0, channels), tx1 = cut(x1, channels), td0 = cut(dyn0, heads * k), td1 = cut(dyn1, heads * k),
                         tr = cut(res, channels);
            Tensor tob0 = cut(ob0, channels), tob1 = cut(ob1, channels), tos0 = cut(os0, channels), tos1 = cut(os1, channels),
                   tsb = cut(sumb, channels), tout = cut(outs, channels), tss = cut(sums, channels);
            ops::ShortConvJob jobs[2];
            jobs[0].x = &tx0, jobs[0].taps = &tt0, jobs[0].dynamic = &td0, jobs[0].history = &thb0, jobs[0].out = &tob0;
            jobs[1].x = &tx1, jobs[1].taps = &tt1, jobs[1].dynamic = &td1, jobs[1].history = &thb1, jobs[1].out = &tob1;
            ops::short_conv_fp32_batch(jobs, head, nullptr);
            ops::short_conv_fp32(tx0, tt0, &td0, head, ths0, tos0, nullptr);
            ops::short_conv_fp32(tx1, tt1, &td1, head, ths1, tos1, nullptr);
            ops::ShortConvJob fused;
            fused.x = &tx0, fused.taps = &tt1, fused.history = &thb2, fused.residual = &tr, fused.sum = &tsb;
            ops::short_conv_fp32_batch(std::span<const ops::ShortConvJob>(&fused, 1), 1, nullptr);
            ops::short_conv_fp32(tx0, tt1, nullptr, 1, ths2, tout, nullptr);
            ops::add_fp32(tr, tout, tss, nullptr);
        }
        const std::size_t n = static_cast<std::size_t>(channels) * cols, nh = static_cast<std::size_t>(channels) * (k - 1);
        check(same_bits(ob0.get(n), os0.get(n)) && same_bits(ob1.get(n), os1.get(n)), "short_conv_fp32_batch: outputs differ");
        check(same_bits(hb0.get(nh), hs0.get(nh)) && same_bits(hb1.get(nh), hs1.get(nh)) && same_bits(hb2.get(nh), hs2.get(nh)),
              "short_conv_fp32_batch: histories differ");
        check(same_bits(sumb.get(n), sums.get(n)), "short_conv_fp32_batch: fused sum differs from conv + add_fp32");
        std::printf("%-44s bit-exact\n", "head_norms / short_conv batches, fused sum");
    }
    {
        // More heads than a grid's y holds: two unit-RMS jobs of 40000 one-value heads over 2 columns (each head
        // becomes its value's sign), and a clip job of 128-value heads after them.
        const int n = 40000, d = 128, heads = 4;
        auto u = randoms(static_cast<std::size_t>(n) * cols, -2, 2), w = randoms(static_cast<std::size_t>(n) * cols, -2, 2),
             c = randoms(static_cast<std::size_t>(heads) * d * cols, -40, 40);
        Buf bu(u), bw(w), bc(c);
        Tensor tu = bu.tensor({n, cols}), tw = bw.tensor({n, cols}), tc = bc.tensor({heads * d, cols});
        ops::HeadNormJob jobs[3];
        jobs[0].x = &tu, jobs[0].head_dim = 1, jobs[0].unit_rms = true, jobs[0].eps = 1e-6F;
        jobs[1].x = &tw, jobs[1].head_dim = 1, jobs[1].unit_rms = true, jobs[1].eps = 1e-6F;
        jobs[2].x = &tc, jobs[2].head_dim = d, jobs[2].threshold = 22.6F;
        ops::head_norms_fp32(jobs, nullptr);
        Vec want_u, want_w, want_c;
        for (float e : u) { want_u.push_back(e / std::sqrt(double(e) * e + 1e-6)); }
        for (float e : w) { want_w.push_back(e / std::sqrt(double(e) * e + 1e-6)); }
        for (int i = 0; i < heads * cols; ++i) {
            const Vec v = ref::l2_clip(Vec(c.begin() + i * d, c.begin() + (i + 1) * d), 22.6, nullptr);
            want_c.insert(want_c.end(), v.begin(), v.end());
        }
        compare("head_norms_fp32 80004 heads: job 0", bu.get(u.size()), want_u, 2e-6);
        compare("head_norms_fp32 80004 heads: job 1", bw.get(w.size()), want_w, 2e-6);
        compare("head_norms_fp32 80004 heads: job 2 (past 65535)", bc.get(c.size()), want_c, 2e-6);
    }
    // Refusals: mismatched T, too many jobs, sum without residual.
    {
        Buf a(h * 2), b(h * 3), o(h * 3);
        Tensor ta = a.tensor({h, 2}), tb = b.tensor({h, 3}), to = o.tensor({h, 3}), to2 = o.tensor({h, 2});
        const ops::RmsNormJob mixed[] = {{&ta, nullptr, &to2}, {&tb, nullptr, &to}};
        bool threw = false;
        try { ops::rmsnorm_fp32_batch(mixed, 1e-6F, nullptr); } catch (const std::invalid_argument&) { threw = true; }
        check(threw, "rmsnorm_fp32_batch accepted jobs of different T");
        ops::HeadNormJob many[5];
        for (auto& job : many) { job.x = &tb, job.head_dim = 128, job.threshold = 1.0F; }
        threw = false;
        try { ops::head_norms_fp32(many, nullptr); } catch (const std::invalid_argument&) { threw = true; }
        check(threw, "head_norms_fp32 accepted 5 jobs");
    }
}

// Positions 0..n-1 through relative_bias_sink_attention in calls of `chunk`, against the oracle over the whole sequence.
// With device_start each call passes start 0 and the position as the device offset (how a decode graph runs).
// Only positions from check_from on are compared (the FP64 oracle is quadratic in n).
// With split the calls get a split workspace (calls of at most kSplitMaxColumns columns split their keys).
void test_attention(int window, int extent, int n, int chunk, const std::string& label, bool device_start = false,
                    int check_from = 0, bool split = false, bool tiled = false) {
    const int heads = 24, kvh = 8, d = 128, rel = 16, qw = heads * d, kvw = kvh * d;
    auto q = randoms(qw * n, -0.3F, 0.3F), r = randoms(heads * rel * n), sink = randoms(heads * n, -2, 2), k = randoms(kvw * n),
         v = randoms(kvw * n), table = randoms(extent * rel, -0.5, 0.5);
    const int capacity = window > 0 ? window - 1 + chunk : n;
    Buf dq(q), dr(r), ds(sink), dk(k), dv(v), dt(table), kc(static_cast<std::size_t>(kvw * capacity)),
        vc(static_cast<std::size_t>(kvw * capacity)), out(static_cast<std::size_t>(qw * n));
    Tensor tkc = kc.tensor({kvw, capacity}), tvc = vc.tensor({kvw, capacity});
    // The same calls with a sigmoid gate into BF16 (own caches): the bits of sigmoid_gate_fp32 of the ungated out.
    Buf dg(randoms(qw * n, -3, 3)), kc2(static_cast<std::size_t>(kvw * capacity)), vc2(static_cast<std::size_t>(kvw * capacity));
    Tensor tkc2 = kc2.tensor({kvw, capacity}), tvc2 = vc2.tensor({kvw, capacity});
    ninfer::DeviceBuffer gated(static_cast<std::size_t>(qw) * n * 2), want_gated(static_cast<std::size_t>(qw) * n * 2);
    ninfer::DeviceBuffer doff(sizeof(std::int64_t));
    const ninfer::ops::RelativeBiasAttentionShape split_shape{heads, kvh, d, rel, window};
    const std::size_t split_bytes = split ? ninfer::ops::relative_bias_attention_split_bytes(split_shape, chunk, capacity) : 0;
    Buf ws(std::max<std::size_t>(split_bytes / 4, 1));
    const Tensor split_ws = ws.tensor({static_cast<std::int32_t>(std::max<std::size_t>(split_bytes / 4, 1))});
    const Tensor* ws_arg = split_bytes > 0 ? &split_ws : nullptr;
    ninfer::DeviceBuffer counters(std::size_t(ninfer::ops::kSplitMaxColumns) * heads * 4);
    counters.fill(0);
    auto* counters_arg = split_bytes > 0 ? static_cast<std::int32_t*>(counters.p) : nullptr;
    for (int s = 0; s < n; s += chunk) {
        const int len = std::min(chunk, n - s);
        const std::int64_t position = s;
        if (device_start) {
            check(cudaMemcpy(doff.p, &position, sizeof position, cudaMemcpyHostToDevice) == cudaSuccess, "offset upload");
        }
        Tensor to = out.tensor({qw, n}).slice(1, s, len);
        const Tensor kn = dk.tensor({kvw, n}).slice(1, s, len), vn = dv.tensor({kvw, n}).slice(1, s, len);
        ninfer::ops::relative_bias_sink_attention({heads, kvh, d, rel, window}, dq.tensor({qw, n}).slice(1, s, len),
                                                  dr.tensor({heads * rel, n}).slice(1, s, len), ds.tensor({heads, n}).slice(1, s, len),
                                                  &kn, &vn,
                                                  dt.tensor({rel, extent}), tkc, tvc, device_start ? 0 : s,
                                                  device_start ? static_cast<const std::int64_t*>(doff.p) : nullptr, nullptr, to,
                                                  nullptr, ws_arg, counters_arg, tiled);
        Tensor tg = Tensor(gated.p, DType::BF16, {qw, n}).slice(1, s, len);
        const Tensor gate = dg.tensor({qw, n}).slice(1, s, len);
        ninfer::ops::relative_bias_sink_attention({heads, kvh, d, rel, window}, dq.tensor({qw, n}).slice(1, s, len),
                                                  dr.tensor({heads * rel, n}).slice(1, s, len), ds.tensor({heads, n}).slice(1, s, len),
                                                  &kn, &vn,
                                                  dt.tensor({rel, extent}), tkc2, tvc2, device_start ? 0 : s,
                                                  device_start ? static_cast<const std::int64_t*>(doff.p) : nullptr, &gate, tg,
                                                  nullptr, ws_arg, counters_arg, tiled);
    }
    if (split_bytes > 0) {
        std::vector<std::int32_t> left(std::size_t(ninfer::ops::kSplitMaxColumns) * heads);
        counters.copy_to_host(left.data(), left.size() * 4);
        check(std::all_of(left.begin(), left.end(), [](std::int32_t c) { return c == 0; }), label + ": the split counters are not left zero");
    }
    {
        // A gated BF16 out without storage is refused before the launch (the caches are not appended to).
        const int len  = std::min(chunk, n);
        const Tensor gate = dg.tensor({qw, n}).slice(1, 0, len);
        Tensor no_out(nullptr, DType::BF16, {qw, len});
        const Tensor kn = dk.tensor({kvw, n}).slice(1, 0, len), vn = dv.tensor({kvw, n}).slice(1, 0, len);
        bool refused = false;
        try {
            ninfer::ops::relative_bias_sink_attention({heads, kvh, d, rel, window}, dq.tensor({qw, n}).slice(1, 0, len),
                                                      dr.tensor({heads * rel, n}).slice(1, 0, len), ds.tensor({heads, n}).slice(1, 0, len),
                                                      &kn, &vn, dt.tensor({rel, extent}), tkc2, tvc2, 0, nullptr, &gate, no_out, nullptr);
        } catch (const std::invalid_argument&) { refused = true; }
        check(refused, "relative_bias_sink_attention takes a BF16 out without storage");
        // The pre-appended path (no k_new, v_new) over caches without storage is refused too, before the launch.
        Tensor no_k(nullptr, DType::FP32, {kvw, tkc2.ne[1]}), no_v(nullptr, DType::FP32, {kvw, tkc2.ne[1]});
        Tensor fp32_out = out.tensor({qw, n}).slice(1, 0, len);
        for (const bool k_missing : {true, false}) {
            bool no_cache = false;
            try {
                ninfer::ops::relative_bias_sink_attention({heads, kvh, d, rel, window}, dq.tensor({qw, n}).slice(1, 0, len),
                                                          dr.tensor({heads * rel, n}).slice(1, 0, len), ds.tensor({heads, n}).slice(1, 0, len),
                                                          nullptr, nullptr, dt.tensor({rel, extent}), k_missing ? no_k : tkc2,
                                                          k_missing ? tvc2 : no_v, 0, nullptr, nullptr, fp32_out, nullptr);
            } catch (const std::invalid_argument&) { no_cache = true; }
            check(no_cache, std::string("relative_bias_sink_attention reads a pre-appended ") + (k_missing ? "k_cache" : "v_cache") +
                                " without storage");
        }
    }
    {
        Tensor tw(want_gated.p, DType::BF16, {qw, n});
        ninfer::ops::sigmoid_gate_fp32(out.tensor({qw, n}), dg.tensor({qw, n}), tw, nullptr);
        std::vector<std::uint16_t> a(static_cast<std::size_t>(qw) * n), b(a.size());
        check(cudaMemcpy(a.data(), gated.p, a.size() * 2, cudaMemcpyDeviceToHost) == cudaSuccess &&
                  cudaMemcpy(b.data(), want_gated.p, b.size() * 2, cudaMemcpyDeviceToHost) == cudaSuccess,
              "gated download");
        check(a == b, label + ": the gated BF16 out differs from sigmoid_gate_fp32 of the ungated one");
    }
    Vec want;
    for (int p = check_from; p < n; ++p) {
        std::vector<bool> visible(n);
        Vec bias(static_cast<std::size_t>(heads) * n, 0.0);
        for (int j = 0; j < n; ++j) {
            visible[j] = j <= p && (window == 0 || p - j < window);
            for (int hh = 0; hh < heads; ++hh) {
                const Vec rr(r.begin() + (p * heads + hh) * rel, r.begin() + (p * heads + hh + 1) * rel);
                bias[hh * n + j] = ref::relative_bias(rr.data(), widen(table), rel, p - j);
            }
        }
        const Vec o = ref::attention(Vec(q.begin() + p * qw, q.begin() + (p + 1) * qw), widen(k), widen(v), bias,
                                     Vec(sink.begin() + p * heads, sink.begin() + (p + 1) * heads), visible, heads, kvh, d);
        want.insert(want.end(), o.begin(), o.end());
    }
    const auto got = out.get(qw * n);
    compare(label, std::vector<float>(got.begin() + static_cast<std::ptrdiff_t>(check_from) * qw, got.end()), want, 2e-6);
}

// head_norms_fp32 with a KvAppend: x the bits of the plain call, and the ring slots (start + t) mod capacity
// (wrapping) hold those columns.
void test_head_norm_append() {
    const int rows = 1024, cols = 3, capacity = 4, d = 128;
    const auto x = randoms(rows * cols, -2, 2);
    Buf a(x), b(x), cache(static_cast<std::size_t>(rows) * capacity);
    for (const bool unit : {false, true}) {
        Tensor ta = a.tensor({rows, cols}), tb = b.tensor({rows, cols}), tc = cache.tensor({rows, capacity});
        const ninfer::ops::KvAppend append{&tc, 2, nullptr};
        ninfer::ops::HeadNormJob ja, jb;
        ja.x = &ta, jb.x = &tb;
        ja.head_dim = jb.head_dim = d;
        if (unit) {
            ja.unit_rms = jb.unit_rms = true;
            ja.eps = jb.eps = 1e-6F;
        } else {
            ja.threshold = jb.threshold = 3.0F;
        }
        ja.append = &append;
        ninfer::ops::head_norms_fp32(std::span<const ninfer::ops::HeadNormJob>(&ja, 1), nullptr);
        ninfer::ops::head_norms_fp32(std::span<const ninfer::ops::HeadNormJob>(&jb, 1), nullptr);
        const auto ga = a.get(rows * cols), gb = b.get(rows * cols), gc = cache.get(rows * capacity);
        bool same = std::memcmp(ga.data(), gb.data(), ga.size() * 4) == 0;
        for (int c = 0; c < cols; ++c) { same &= std::memcmp(gc.data() + ((2 + c) % capacity) * rows, gb.data() + c * rows, rows * 4) == 0; }
        check(same, "head_norms_fp32 with an append differs from the plain call or its ring slots");
    }
    {
        // An append to a cache without storage is refused before the launch.
        Tensor ta = a.tensor({rows, cols}), no_cache(nullptr, DType::FP32, {rows, capacity});
        const ninfer::ops::KvAppend append{&no_cache, 0, nullptr};
        ninfer::ops::HeadNormJob job;
        job.x = &ta, job.head_dim = d, job.unit_rms = true, job.eps = 1e-6F, job.append = &append;
        bool refused = false;
        try {
            ninfer::ops::head_norms_fp32(std::span<const ninfer::ops::HeadNormJob>(&job, 1), nullptr);
        } catch (const std::invalid_argument&) { refused = true; }
        check(refused, "head_norms_fp32 appends to a cache without storage");
    }
    std::printf("head_norms_fp32 with a KvAppend: the plain call's bits, in the wrapped ring slots too\n");
}

// The FP16 KV rings (opt-in, lossy). head_norms_fp32 with a KvAppend into an FP16 ring: x the plain call's bits and
// the slots hold those columns rounded to nearest. relative_bias_sink_attention over FP16 rings of n positions: the FP64
// oracle over the decoded keys and values.
void test_kv_fp16() {
    const int heads = 24, kvh = 8, d = 128, rel = 16, qw = heads * d, kvw = kvh * d;
    {
        const DType form = DType::FP16;
        const char* name = "FP16";
        const int slot_rows = kvw;
        const std::size_t slot_bytes = static_cast<std::size_t>(slot_rows) * 2;
        // Host encoding and decoding of one column of kvw values.
        const auto encode = [&](const float* x, std::uint8_t* slot, float* decoded) {
            for (int i = 0; i < kvw; ++i) {
                const __half hv = __float2half_rn(x[i]);
                std::memcpy(slot + 2 * i, &hv, 2);
                decoded[i] = __half2float(hv);
            }
        };
        {
            const int cols = 3, capacity = 4;
            const auto x = randoms(kvw * cols, -2, 2);
            Buf a(x), b(x);
            ninfer::DeviceBuffer ring(slot_bytes * capacity);
            check(cudaMemset(ring.p, 0, slot_bytes * capacity) == cudaSuccess, "ring clear");
            Tensor ta = a.tensor({kvw, cols}), tb = b.tensor({kvw, cols}), tc(ring.p, form, {slot_rows, capacity});
            const ninfer::ops::KvAppend append{&tc, 2, nullptr};
            ninfer::ops::HeadNormJob ja, jb;
            ja.x = &ta, jb.x = &tb, ja.head_dim = jb.head_dim = d, ja.threshold = jb.threshold = 3.0F, ja.append = &append;
            ninfer::ops::head_norms_fp32(std::span<const ninfer::ops::HeadNormJob>(&ja, 1), nullptr);
            ninfer::ops::head_norms_fp32(std::span<const ninfer::ops::HeadNormJob>(&jb, 1), nullptr);
            const auto ga = a.get(kvw * cols), gb = b.get(kvw * cols);
            std::vector<std::uint8_t> got(slot_bytes * capacity), want(slot_bytes * capacity, 0);
            check(cudaMemcpy(got.data(), ring.p, got.size(), cudaMemcpyDeviceToHost) == cudaSuccess, "ring download");
            std::vector<float> decoded(kvw);
            for (int c = 0; c < cols; ++c) { encode(gb.data() + c * kvw, want.data() + ((2 + c) % capacity) * slot_bytes, decoded.data()); }
            check(std::memcmp(ga.data(), gb.data(), ga.size() * 4) == 0 && got == want,
                  std::string("head_norms_fp32 append into a ") + name + " ring: x or the slots differ from the host encoding");
        }
        {
            // 600 keys: the last 8 columns also go through the split kernel (three splits).
            const int n = 600, extent = 64, tail = 8;
            auto q = randoms(qw * n, -0.3F, 0.3F), r = randoms(heads * rel * n), sink = randoms(heads * n, -2, 2), k = randoms(kvw * n),
                 v = randoms(kvw * n), table = randoms(extent * rel, -0.5, 0.5);
            std::vector<std::uint8_t> kring(slot_bytes * n), vring(slot_bytes * n);
            std::vector<float> kd(k.size()), vd(v.size());
            for (int t = 0; t < n; ++t) {
                encode(k.data() + t * kvw, kring.data() + t * slot_bytes, kd.data() + t * kvw);
                encode(v.data() + t * kvw, vring.data() + t * slot_bytes, vd.data() + t * kvw);
            }
            ninfer::DeviceBuffer dk(kring.size()), dv(vring.size());
            check(cudaMemcpy(dk.p, kring.data(), kring.size(), cudaMemcpyHostToDevice) == cudaSuccess &&
                      cudaMemcpy(dv.p, vring.data(), vring.size(), cudaMemcpyHostToDevice) == cudaSuccess,
                  "ring upload");
            Buf dq(q), dr(r), ds(sink), dt(table), out(static_cast<std::size_t>(qw * n));
            Tensor tk(dk.p, form, {slot_rows, n}), tv(dv.p, form, {slot_rows, n}), to = out.tensor({qw, n});
            ninfer::ops::relative_bias_sink_attention({heads, kvh, d, rel, 0}, dq.tensor({qw, n}), dr.tensor({heads * rel, n}),
                                                      ds.tensor({heads, n}), nullptr, nullptr, dt.tensor({rel, extent}), tk, tv, 0,
                                                      nullptr, nullptr, to, nullptr);
            Vec want;
            const Vec table64 = widen(table), kd64 = widen(kd), vd64 = widen(vd);
            for (int p = 0; p < n; ++p) {
                std::vector<bool> visible(n);
                Vec bias(static_cast<std::size_t>(heads) * n, 0.0);
                for (int j = 0; j < n; ++j) {
                    visible[j] = j <= p;
                    for (int hh = 0; hh < heads; ++hh) {
                        const Vec rr(r.begin() + (p * heads + hh) * rel, r.begin() + (p * heads + hh + 1) * rel);
                        bias[hh * n + j] = ref::relative_bias(rr.data(), table64, rel, p - j);
                    }
                }
                const Vec o = ref::attention(Vec(q.begin() + p * qw, q.begin() + (p + 1) * qw), kd64, vd64, bias,
                                             Vec(sink.begin() + p * heads, sink.begin() + (p + 1) * heads), visible, heads, kvh, d);
                want.insert(want.end(), o.begin(), o.end());
            }
            compare(std::string("attention over ") + name + " rings, 600 keys", out.get(qw * n), want, 2e-6);
            {
                // Wide calls on the tensor cores (FP16 Q and softmax weights, FP32 accumulation): chunks of 200 of
                // every key, and a window of 100 (chunks of 150).
                for (const int window : {0, 100}) {
                    const int chunk = window == 0 ? 200 : 150;
                    Buf out3(static_cast<std::size_t>(qw * n));
                    for (int s0 = 0; s0 < n; s0 += chunk) {
                        const int len = std::min(chunk, n - s0);
                        Tensor to3 = out3.tensor({qw, n}).slice(1, s0, len);
                        ninfer::ops::relative_bias_sink_attention({heads, kvh, d, rel, window}, dq.tensor({qw, n}).slice(1, s0, len),
                                                                  dr.tensor({heads * rel, n}).slice(1, s0, len),
                                                                  ds.tensor({heads, n}).slice(1, s0, len), nullptr, nullptr,
                                                                  dt.tensor({rel, extent}), tk, tv, s0, nullptr, nullptr, to3, nullptr,
                                                                  nullptr, nullptr, true);
                    }
                    Vec want3 = want;
                    if (window > 0) {
                        want3.clear();
                        for (int p = 0; p < n; ++p) {
                            std::vector<bool> visible(n);
                            Vec bias(static_cast<std::size_t>(heads) * n, 0.0);
                            for (int j = 0; j < n; ++j) {
                                visible[j] = j <= p && p - j < window;
                                for (int hh = 0; hh < heads; ++hh) {
                                    const Vec rr(r.begin() + (p * heads + hh) * rel, r.begin() + (p * heads + hh + 1) * rel);
                                    bias[hh * n + j] = ref::relative_bias(rr.data(), table64, rel, p - j);
                                }
                            }
                            const Vec o = ref::attention(Vec(q.begin() + p * qw, q.begin() + (p + 1) * qw), kd64, vd64, bias,
                                                         Vec(sink.begin() + p * heads, sink.begin() + (p + 1) * heads), visible, heads, kvh, d);
                            want3.insert(want3.end(), o.begin(), o.end());
                        }
                    }
                    compare(std::string("tensor-core attention over FP16 rings, 600 keys, window ") + std::to_string(window), out3.get(qw * n),
                            want3, 3e-3);
                }
                // A v_cache 8 bytes off a 16-byte boundary (FP16-aligned, not uint4-aligned): the flash route's
                // vector value loads would fault or misread, so the call takes the tiled route, which loads each value.
                ninfer::DeviceBuffer dv_off(vring.size() + 16);
                char* const v_base = static_cast<char*>(dv_off.p) + 8;
                check(cudaMemcpy(v_base, vring.data(), vring.size(), cudaMemcpyHostToDevice) == cudaSuccess, "offset ring upload");
                Tensor tv_off(v_base, form, {slot_rows, n});
                Buf out4(static_cast<std::size_t>(qw * n));
                for (int s0 = 0; s0 < n; s0 += 200) {
                    const int len = std::min(200, n - s0);
                    Tensor to4 = out4.tensor({qw, n}).slice(1, s0, len);
                    ninfer::ops::relative_bias_sink_attention({heads, kvh, d, rel, 0}, dq.tensor({qw, n}).slice(1, s0, len),
                                                              dr.tensor({heads * rel, n}).slice(1, s0, len), ds.tensor({heads, n}).slice(1, s0, len),
                                                              nullptr, nullptr, dt.tensor({rel, extent}), tk, tv_off, s0, nullptr, nullptr, to4,
                                                              nullptr, nullptr, nullptr, true);
                }
                check(cudaDeviceSynchronize() == cudaSuccess, "wide attention over an unaligned v_cache ran");
                compare("wide attention over FP16 rings with v_cache 8 bytes off 16, 600 keys", out4.get(qw * n), want, 3e-3);
            }
            {
                const ninfer::ops::RelativeBiasAttentionShape shape{heads, kvh, d, rel, 0};
                const std::size_t split_bytes = ninfer::ops::relative_bias_attention_split_bytes(shape, tail, n);
                check(split_bytes > 0, "the FP16 split case splits");
                Buf ws(split_bytes / 4), out2(static_cast<std::size_t>(qw * tail));
                const Tensor split_ws = ws.tensor({static_cast<std::int32_t>(split_bytes / 4)});
                ninfer::DeviceBuffer counters(std::size_t(ninfer::ops::kSplitMaxColumns) * heads * 4);
                counters.fill(0);
                Tensor to2 = out2.tensor({qw, tail});
                ninfer::ops::relative_bias_sink_attention(shape, dq.tensor({qw, n}).slice(1, n - tail, tail),
                                                          dr.tensor({heads * rel, n}).slice(1, n - tail, tail),
                                                          ds.tensor({heads, n}).slice(1, n - tail, tail), nullptr, nullptr,
                                                          dt.tensor({rel, extent}), tk, tv, n - tail, nullptr, nullptr, to2, nullptr,
                                                          &split_ws, static_cast<std::int32_t*>(counters.p));
                compare(std::string("split attention over ") + name + " rings, last 8 of 600 keys", out2.get(qw * tail),
                        Vec(want.end() - static_cast<std::ptrdiff_t>(qw) * tail, want.end()), 2e-6);
            }
            // Appending k_new and v_new is FP32 only.
            const Tensor kn = dq.tensor({kvw, n}), vn = dq.tensor({kvw, n});
            bool refused = false;
            try {
                ninfer::ops::relative_bias_sink_attention({heads, kvh, d, rel, 0}, dq.tensor({qw, n}), dr.tensor({heads * rel, n}),
                                                          ds.tensor({heads, n}), &kn, &vn, dt.tensor({rel, extent}), tk, tv, 0, nullptr,
                                                          nullptr, to, nullptr);
            } catch (const std::invalid_argument&) { refused = true; }
            check(refused, std::string("relative_bias_sink_attention appends k_new into a ") + name + " ring");
        }
    }
    std::printf("KV rings FP16: appends match the host encoding, attention matches the oracle over the decoded rings\n");
}

// projection_fp32_inputs: each weight's rows by its own input, the bits of projection_fp32 on that input alone.
void test_projection_inputs() {
    const int k = 2304, ns[] = {64, 8, 32, 12};
    for (const int cols : {1, 3, 8}) {
        std::vector<ninfer::DeviceBuffer> xb, wb;
        std::vector<Tensor> xs, ws;
        for (int i = 0; i < 3; ++i) { xb.push_back(t::to_device_bf16(randoms(k * cols))); }
        for (int n : ns) { wb.push_back(t::to_device_bf16(randoms(k * n, -0.05F, 0.05F))); }
        for (int i = 0; i < 3; ++i) { xs.emplace_back(xb[i].p, DType::BF16, std::initializer_list<std::int32_t>{k, cols}); }
        for (int i = 0; i < 4; ++i) { ws.emplace_back(wb[i].p, DType::BF16, std::initializer_list<std::int32_t>{k, ns[i]}); }
        const Tensor* inputs[] = {&xs[0], &xs[1], &xs[1], &xs[2]};
        const Tensor* weights[] = {&ws[0], &ws[1], &ws[2], &ws[3]};
        const int total = 64 + 8 + 32 + 12;
        Buf joint(static_cast<std::size_t>(total) * cols);
        Tensor tj = joint.tensor({total, cols});
        ninfer::ops::projection_fp32_inputs(inputs, weights, tj, nullptr);
        const auto got = joint.get(static_cast<std::size_t>(total) * cols);
        bool same = true;
        int row = 0;
        for (int i = 0; i < 4; ++i) {
            Buf one(static_cast<std::size_t>(ns[i]) * cols);
            Tensor to = one.tensor({ns[i], cols});
            const Tensor* w1[] = {weights[i]};
            ninfer::ops::projection_fp32(*inputs[i], w1, to, nullptr);
            const auto want = one.get(static_cast<std::size_t>(ns[i]) * cols);
            for (int c = 0; c < cols; ++c) {
                same &= std::memcmp(got.data() + static_cast<std::size_t>(c) * total + row, want.data() + static_cast<std::size_t>(c) * ns[i],
                                    ns[i] * 4) == 0;
            }
            row += ns[i];
        }
        check(same, "projection_fp32_inputs differs from projection_fp32 per input");
    }
    std::printf("projection_fp32_inputs: each weight's rows the bits of its own projection_fp32 (T 1, 3, 8)\n");

    // The FP64 oracle, independent of the projection kernels: kProjectionMaxWeights segments (every N a multiple of
    // 4, so each CTA's four rows come from one input) over five distinct inputs, each output the FP64 dot product of
    // its BF16 weight row and BF16 input column within the FP32 bound sum_k |w x| * K * 2^-24.
    const int ks[]   = {2304, 3072};
    const int segs[] = {64, 8, 32, 12, 4, 16, 8, 20};
    static_assert(std::size(segs) == ninfer::ops::kProjectionMaxWeights);
    const int which[] = {0, 1, 1, 2, 3, 0, 4, 2}; // the input of each segment
    const auto bf16 = [](std::vector<float> v) {
        for (float& x : v) { x = t::bf16_to_f32(t::f32_to_bf16(x)); }
        return v;
    };
    for (const int kk : ks) {
        for (const int cols : {1, 5, 8}) {
            std::vector<std::vector<float>> xh, wh;
            std::vector<ninfer::DeviceBuffer> xb, wb;
            std::vector<Tensor> xs, ws;
            for (int i = 0; i < 5; ++i) {
                xh.push_back(bf16(randoms(static_cast<std::size_t>(kk) * cols)));
                xb.push_back(t::to_device_bf16(xh.back()));
                xs.emplace_back(xb.back().p, DType::BF16, std::initializer_list<std::int32_t>{kk, cols});
            }
            int total = 0;
            for (const int n : segs) {
                wh.push_back(bf16(randoms(static_cast<std::size_t>(kk) * n, -0.05F, 0.05F)));
                wb.push_back(t::to_device_bf16(wh.back()));
                ws.emplace_back(wb.back().p, DType::BF16, std::initializer_list<std::int32_t>{kk, n});
                total += n;
            }
            std::vector<const Tensor*> inputs, weights;
            for (std::size_t i = 0; i < std::size(segs); ++i) {
                inputs.push_back(&xs[static_cast<std::size_t>(which[i])]);
                weights.push_back(&ws[i]);
            }
            Buf out(static_cast<std::size_t>(total) * cols);
            Tensor to = out.tensor({total, cols});
            ninfer::ops::projection_fp32_inputs(inputs, weights, to, nullptr);
            const auto got = out.get(static_cast<std::size_t>(total) * cols);
            int worst_row = -1;
            int row = 0;
            for (std::size_t i = 0; i < std::size(segs); ++i) {
                const auto& x = xh[static_cast<std::size_t>(which[i])];
                for (int n = 0; n < segs[i]; ++n, ++row) {
                    for (int c = 0; c < cols; ++c) {
                        double dot = 0.0, magnitude = 0.0;
                        for (int k = 0; k < kk; ++k) {
                            const double product = double(wh[i][static_cast<std::size_t>(n) * kk + k]) * double(x[static_cast<std::size_t>(c) * kk + k]);
                            dot += product;
                            magnitude += std::abs(product);
                        }
                        const double bound = magnitude * kk * std::ldexp(1.0, -24);
                        if (std::abs(double(got[static_cast<std::size_t>(c) * total + row]) - dot) > bound) { worst_row = row; }
                    }
                }
            }
            check(worst_row < 0, "projection_fp32_inputs departs from the FP64 oracle at K " + std::to_string(kk) + ", T " +
                                     std::to_string(cols) + " (row " + std::to_string(worst_row) + ")");
        }
    }
    std::printf("projection_fp32_inputs: %zu segments over 5 inputs within the FP64 oracle's bound (K 2304, 3072; T 1, 5, 8)\n",
                std::size(segs));
}

void test_route() {
    const int h = 2304, e = 128, k = 8, cols = 4;
    auto z = randoms(h * cols), router = randoms(e * h, -0.05F, 0.05F), bias = randoms(e, -0.01F, 0.01F);
    Buf dz(z), dr(router), db(bias), ids(static_cast<std::size_t>(k * cols)), w(static_cast<std::size_t>(k * cols));
    Tensor tid = ids.tensor({k, cols}, DType::I32), tw = w.tensor({k, cols});
    ninfer::DeviceArena arena(ninfer::ops::moe_route_sigmoid_topk_workspace_bytes(e, cols) + 4096);
    ninfer::ops::moe_route_sigmoid_topk(dz.tensor({h, cols}), nullptr, dr.tensor({h, e}), db.tensor({e}), k, 8.0F, arena, tid, tw, nullptr);
    const auto got_ids = t::from_device<std::int32_t>(ids.b, k * cols);
    Vec want_w;
    bool same = true;
    for (int c = 0; c < cols; ++c) {
        const auto route = ref::route(Vec(z.begin() + c * h, z.begin() + (c + 1) * h), widen(router), widen(bias), k, 8.0);
        for (int i = 0; i < k; ++i) { same &= route.ids[i] == got_ids[c * k + i]; }
        want_w.insert(want_w.end(), route.weights.begin(), route.weights.end());
    }
    check(same, "moe_route_sigmoid_topk: selected experts differ from the oracle (random scores have no tie within 1e-6)");
    std::printf("%-44s experts %s\n", "moe_route_sigmoid_topk", same ? "equal" : "DIFFER");
    compare("moe_route_sigmoid_topk weights", w.get(k * cols), want_w, 2e-6);

    // The input norm in the routing launch: z the bits of rmsnorm_fp32, ids and weights those of routing it. D = 2240
    // is not a multiple of 256 (the norm's thread split) and E = 8 makes each block write several passes of z.
    for (const auto [rows, experts] : {std::pair{2304, 128}, std::pair{2240, 8}}) {
        auto x = randoms(rows * cols, -2, 2), gain = randoms(rows, 0.5F, 1.5F), rt = randoms(experts * rows, -0.05F, 0.05F),
             bs = randoms(experts, -0.01F, 0.01F);
        const int kk = std::min(k, experts);
        Buf dx(x), dgain(gain), drt(rt), dbs(bs), zn(static_cast<std::size_t>(rows) * cols), zr(static_cast<std::size_t>(rows) * cols),
            ids2(static_cast<std::size_t>(kk * cols)), w2(static_cast<std::size_t>(kk * cols)), ids3(static_cast<std::size_t>(kk * cols)),
            w3(static_cast<std::size_t>(kk * cols));
        ninfer::DeviceArena arena2(ninfer::ops::moe_route_sigmoid_topk_workspace_bytes(experts, cols) + 4096);
        Tensor tzn = zn.tensor({rows, cols}), tzr = zr.tensor({rows, cols}), tg = dgain.tensor({rows});
        Tensor ti2 = ids2.tensor({kk, cols}, DType::I32), tw2 = w2.tensor({kk, cols}), ti3 = ids3.tensor({kk, cols}, DType::I32),
               tw3 = w3.tensor({kk, cols});
        const ninfer::ops::RouteInputNorm norm{&tg, 1e-6F, &tzn};
        ninfer::ops::moe_route_sigmoid_topk(dx.tensor({rows, cols}), &norm, drt.tensor({rows, experts}), dbs.tensor({experts}), kk, 8.0F,
                                            arena2, ti2, tw2, nullptr);
        ninfer::ops::rmsnorm_fp32(dx.tensor({rows, cols}), &tg, 1e-6F, tzr, nullptr);
        ninfer::ops::moe_route_sigmoid_topk(tzr, nullptr, drt.tensor({rows, experts}), dbs.tensor({experts}), kk, 8.0F, arena2, ti3, tw3,
                                            nullptr);
        const auto a = zn.get(rows * cols), b = zr.get(rows * cols), wa = w2.get(kk * cols), wb = w3.get(kk * cols);
        const bool z_same = std::memcmp(a.data(), b.data(), a.size() * 4) == 0, w_same = std::memcmp(wa.data(), wb.data(), wa.size() * 4) == 0;
        const bool ids_same = t::from_device<std::int32_t>(ids2.b, kk * cols) == t::from_device<std::int32_t>(ids3.b, kk * cols);
        check(z_same && w_same && ids_same, "moe_route_sigmoid_topk with its input norm differs from rmsnorm_fp32 then routing");
        // Refused before any launch: a norm weight without storage, and a normalized z that partly overlaps z.
        const auto route_refuses = [&](const Tensor& in, const ninfer::ops::RouteInputNorm& bad) {
            try {
                ninfer::ops::moe_route_sigmoid_topk(in, &bad, drt.tensor({rows, experts}), dbs.tensor({experts}), kk, 8.0F, arena2, ti2,
                                                    tw2, nullptr);
            } catch (const std::invalid_argument&) { return true; }
            return false;
        };
        const Tensor no_gain(nullptr, DType::FP32, {rows});
        check(route_refuses(dx.tensor({rows, cols}), {&no_gain, 1e-6F, &tzn}), "moe_route_sigmoid_topk takes a null norm weight");
        Tensor shifted = tzn;
        shifted.data   = static_cast<float*>(zn.b.p) + 1; // one element into z: never read, the Op refuses first
        check(route_refuses(tzn, {&tg, 1e-6F, &shifted}), "moe_route_sigmoid_topk takes a normalized z overlapping z");
        if (experts >= cols) { // the router's first columns as the normalized z (never written: the Op refuses first)
            Tensor on_router = drt.tensor({rows, experts}).slice(1, 0, cols);
            check(route_refuses(dx.tensor({rows, cols}), {&tg, 1e-6F, &on_router}), "moe_route_sigmoid_topk takes a normalized z on the router");
        }
        {
            // z and the normalized z interleaved in one allocation, column by column: their bounding spans overlap but
            // no byte is shared, so the Op runs, and routes as with separate buffers.
            Buf both(static_cast<std::size_t>(2 * rows) * cols);
            check(cudaMemcpy2D(both.b.p, static_cast<std::size_t>(2 * rows) * 4, dx.b.p, static_cast<std::size_t>(rows) * 4,
                               static_cast<std::size_t>(rows) * 4, cols, cudaMemcpyDeviceToDevice) == cudaSuccess,
                  "interleave z");
            const Tensor whole = both.tensor({2 * rows, cols});
            Tensor zi = whole.slice(0, 0, rows), zo = whole.slice(0, rows, rows);
            const ninfer::ops::RouteInputNorm interleaved{&tg, 1e-6F, &zo};
            bool ran = true;
            try {
                ninfer::ops::moe_route_sigmoid_topk(zi, &interleaved, drt.tensor({rows, experts}), dbs.tensor({experts}), kk, 8.0F, arena2,
                                                    ti2, tw2, nullptr);
            } catch (const std::invalid_argument&) { ran = false; }
            const bool same = ran && std::memcmp(w2.get(kk * cols).data(), w3.get(kk * cols).data(), static_cast<std::size_t>(kk) * cols * 4) == 0 &&
                              t::from_device<std::int32_t>(ids2.b, kk * cols) == t::from_device<std::int32_t>(ids3.b, kk * cols);
            check(same, "moe_route_sigmoid_topk refuses or misroutes a normalized z interleaved with z");
        }
        std::printf("moe_route_sigmoid_topk D %d E %d with the input norm: the bits of rmsnorm_fp32, then routing\n", rows, experts);
    }

    // Exact ties (a zero router: every score 1/2) among 40 experts, bias steering two of them: the steered two
    // first, then the lowest ids.
    const int few = 40, picks = 5;
    std::vector<float> zero(static_cast<std::size_t>(few) * h, 0.0F), steer(few, 0.0F);
    steer[33] = 0.5F;
    steer[7]  = 0.25F;
    Buf dzero(zero), dsteer(steer), tie_ids(static_cast<std::size_t>(picks * cols)), tie_w(static_cast<std::size_t>(picks * cols));
    Tensor tti = tie_ids.tensor({picks, cols}, DType::I32), ttw = tie_w.tensor({picks, cols});
    ninfer::DeviceArena tie_arena(ninfer::ops::moe_route_sigmoid_topk_workspace_bytes(few, cols) + 4096);
    ninfer::ops::moe_route_sigmoid_topk(dz.tensor({h, cols}), nullptr, dzero.tensor({h, few}), dsteer.tensor({few}), picks, 8.0F,
                                        tie_arena, tti, ttw, nullptr);
    const auto tied = t::from_device<std::int32_t>(tie_ids.b, picks * cols);
    bool order = true;
    for (int c = 0; c < cols; ++c) {
        const std::int32_t want[] = {33, 7, 0, 1, 2};
        for (int i = 0; i < picks; ++i) { order &= tied[c * picks + i] == want[i]; }
    }
    check(order, "moe_route_sigmoid_topk: ties go to the lower id, after the bias-steered experts");
    const auto tw_got = tie_w.get(picks * cols);
    check(std::all_of(tw_got.begin(), tw_got.end(), [](float x) { return std::fabs(x - 1.6F) < 1e-6F; }), "tied weights are 8 / 5");
}

std::uint64_t align256(std::uint64_t v) { return (v + 255) / 256 * 256; }

// Random NVFP4 blocks with scales in a realistic range (no 0x7F).
std::vector<std::uint8_t> nvfp4_blocks(std::size_t blocks) {
    std::vector<std::uint8_t> b(blocks * 36);
    for (std::size_t i = 0; i < blocks; ++i) {
        for (int s = 0; s < 4; ++s) { b[i * 36 + s] = static_cast<std::uint8_t>(0x20 + (seed_state = seed_state * 1664525U + 1013904223U) % 0x30); }
        for (int j = 4; j < 36; ++j) { b[i * 36 + j] = static_cast<std::uint8_t>((seed_state = seed_state * 1664525U + 1013904223U) >> 24); }
    }
    return b;
}

// cols > 2 draws the ids at random (k * cols >= 64 entries take the grouped kernels) and also checks every column
// against its own T = 1 call, bit for bit.
void test_experts(bool nvfp4, int h, int inter, int experts, const std::string& label, int cols = 2) {
    const int k = 4, entries = k * cols;
    auto z = randoms(h * cols);
    std::vector<int> ids = {0, experts - 1, 1, 2 % experts, 1, 0, experts - 1, 2 % experts};
    if (cols != 2) {
        ids.resize(static_cast<std::size_t>(entries));
        for (int c = 0; c < cols; ++c) // k distinct experts per column, as the router gives
            for (int i = 0; i < k; ++i) { ids[c * k + i] = (static_cast<int>(uniform(0.0F, 1.0F) * experts) + i * (experts / k)) % experts; }
    }
    auto weights = randoms(entries, 0.1F, 2.0F);
    auto gu_scale = randoms(experts, 0.5F, 1.5F), down_scale = randoms(experts, 0.5F, 1.5F);
    std::vector<Vec> gate_up(experts), down(experts); // decoded [2I, H] and [H, I], row-major
    ninfer::ops::SituExpertBank bank;
    bank.experts = experts;
    bank.hidden = h;
    bank.intermediate = inter;
    ninfer::DeviceBuffer records, bgu, bd, dgs, dds;
    if (nvfp4) {
        const std::uint64_t gate_row = h / 64 * 36, down_row = inter / 64 * 36, gate_bytes = gate_row * inter;
        const std::uint64_t up_off = align256(gate_bytes), down_off = align256(up_off + gate_bytes),
                            stride = align256(down_off + down_row * h);
        std::vector<std::uint8_t> host(stride * experts, 0);
        for (int e = 0; e < experts; ++e) {
            const auto g = nvfp4_blocks(inter * h / 64), u = nvfp4_blocks(inter * h / 64), dn = nvfp4_blocks(h * inter / 64);
            std::copy(g.begin(), g.end(), host.begin() + e * stride);
            std::copy(u.begin(), u.end(), host.begin() + e * stride + up_off);
            std::copy(dn.begin(), dn.end(), host.begin() + e * stride + down_off);
            gate_up[e] = ref::decode_nvfp4(g);
            const Vec du = ref::decode_nvfp4(u);
            gate_up[e].insert(gate_up[e].end(), du.begin(), du.end());
            down[e] = ref::decode_nvfp4(dn);
        }
        records = t::to_device(host);
        dgs = t::to_device(gu_scale);
        dds = t::to_device(down_scale);
        bank.format = ninfer::QType::GGML_REC_NVFP4_NVFP4;
        bank.records = static_cast<const std::uint8_t*>(records.p);
        bank.record_stride = stride;
        bank.gate_up_scale = static_cast<const float*>(dgs.p);
        bank.down_scale = static_cast<const float*>(dds.p);
    } else {
        std::vector<float> gu(static_cast<std::size_t>(experts) * 2 * inter * h), dn(static_cast<std::size_t>(experts) * h * inter);
        for (auto& x : gu) { x = t::bf16_to_f32(t::f32_to_bf16(uniform(-0.05F, 0.05F))); }
        for (auto& x : dn) { x = t::bf16_to_f32(t::f32_to_bf16(uniform(-0.05F, 0.05F))); }
        for (int e = 0; e < experts; ++e) {
            gate_up[e] = Vec(gu.begin() + std::size_t(e) * 2 * inter * h, gu.begin() + std::size_t(e + 1) * 2 * inter * h);
            down[e]    = Vec(dn.begin() + std::size_t(e) * h * inter, dn.begin() + std::size_t(e + 1) * h * inter);
        }
        bgu = t::to_device_bf16(gu);
        bd = t::to_device_bf16(dn);
        bank.format = ninfer::QType::BF16;
        bank.gate_up = bgu.p;
        bank.down = bd.p;
    }
    Vec want(static_cast<std::size_t>(h) * cols, 0.0);
    for (int c = 0; c < cols; ++c)
        for (int i = 0; i < k; ++i) {
            const int e = ids[c * k + i];
            const double sg = nvfp4 ? gu_scale[e] : 1.0, sd = nvfp4 ? down_scale[e] : 1.0;
            Vec a(inter);
            for (int r = 0; r < inter; ++r) {
                double g = 0, u = 0;
                for (int d = 0; d < h; ++d) {
                    g += gate_up[e][std::size_t(r) * h + d] * z[c * h + d];
                    u += gate_up[e][std::size_t(inter + r) * h + d] * z[c * h + d];
                }
                a[r] = ref::situ(g * sg, u * sg, 4.0, 25.0);
            }
            for (int d = 0; d < h; ++d) {
                double y = 0;
                for (int r = 0; r < inter; ++r) { y += down[e][std::size_t(d) * inter + r] * a[r]; }
                want[c * h + d] += weights[c * k + i] * y * sd;
            }
        }
    Buf dz(z), dw(weights), dout(static_cast<std::size_t>(h * cols));
    ninfer::DeviceBuffer did = t::to_device_i32(ids);
    ninfer::DeviceArena arena(ninfer::ops::moe_situ_experts_workspace_bytes(bank, entries) + 4096);
    Tensor tout = dout.tensor({h, cols});
    const Tensor tw = dw.tensor({k, cols});
    ninfer::ops::moe_situ_experts(bank, dz.tensor({h, cols}), Tensor(did.p, DType::I32, {k, cols}), &tw, 4.0F, 25.0F, arena, tout, nullptr);
    {
        // Without weights the entries' outputs come back uncombined; the MLP convolution combining them in its launch
        // gives the bits of the experts' own combine followed by the convolution (with its residual sum).
        Buf entries_out(static_cast<std::size_t>(h) * k * cols), combined(static_cast<std::size_t>(h) * cols), taps(randoms(4 * h)),
            res(randoms(h * cols)), h1(randoms(3 * h)), h2(std::vector<float>(h1.get(3 * h))), s1(static_cast<std::size_t>(h) * cols),
            s2(static_cast<std::size_t>(h) * cols);
        Tensor te = entries_out.tensor({h, k * cols}), tcomb = combined.tensor({h, cols}), tt = taps.tensor({4, h}), tr = res.tensor({h, cols}),
               th1 = h1.tensor({h, 3}), th2 = h2.tensor({h, 3}), ts1 = s1.tensor({h, cols}), ts2 = s2.tensor({h, cols});
        ninfer::ops::moe_situ_experts(bank, dz.tensor({h, cols}), Tensor(did.p, DType::I32, {k, cols}), nullptr, 4.0F, 25.0F, arena, te,
                                      nullptr);
        ninfer::ops::ShortConvJob plain, fused;
        plain.x = &tout, plain.taps = &tt, plain.history = &th1, plain.residual = &tr, plain.sum = &ts1;
        const ninfer::ops::ConvCombine cb{&te, &tw, &tcomb};
        fused.combine = &cb, fused.taps = &tt, fused.history = &th2, fused.residual = &tr, fused.sum = &ts2;
        ninfer::ops::short_conv_fp32_batch(std::span<const ninfer::ops::ShortConvJob>(&plain, 1), 1, nullptr);
        ninfer::ops::short_conv_fp32_batch(std::span<const ninfer::ops::ShortConvJob>(&fused, 1), 1, nullptr);
        const auto eq = [](const Buf& x, const Buf& y, std::size_t n) {
            const auto a = x.get(n), b = y.get(n);
            return std::memcmp(a.data(), b.data(), n * 4) == 0;
        };
        const std::size_t n = static_cast<std::size_t>(h) * cols;
        check(eq(dout, combined, n) && eq(s1, s2, n) && eq(h1, h2, 3 * static_cast<std::size_t>(h)),
              label + ": the convolution's combine differs from the experts' combine then the convolution");
        std::printf("%-44s combined in the MLP convolution: bit-exact\n", label.c_str());
        // A combine's entries or weights without storage are refused before the launch.
        const Tensor no_entries(nullptr, DType::FP32, {h, k * cols}), no_weights(nullptr, DType::FP32, {k, cols});
        for (const auto& bad : {ninfer::ops::ConvCombine{&no_entries, &tw, &tcomb}, ninfer::ops::ConvCombine{&te, &no_weights, &tcomb}}) {
            ninfer::ops::ShortConvJob job = fused;
            job.combine  = &bad;
            bool refused = false;
            try {
                ninfer::ops::short_conv_fp32_batch(std::span<const ninfer::ops::ShortConvJob>(&job, 1), 1, nullptr);
            } catch (const std::invalid_argument&) { refused = true; }
            check(refused, label + ": a combine without storage was accepted");
        }
        // The unweighted route writes its entries straight into out: an out without storage, or of any shape but
        // exactly [H, k * T] ([H, T] with weights), is refused before the launch.
        const Tensor no_out(nullptr, DType::FP32, {h, k * cols});
        Tensor flat(entries_out.b.p, DType::FP32, {h * k * cols}), as_3d(entries_out.b.p, DType::FP32, {h, cols, k});
        Tensor weighted_wide(entries_out.b.p, DType::FP32, {h, k * cols});
        const auto experts_refuse = [&](const Tensor* weights, Tensor bad) {
            try {
                ninfer::ops::moe_situ_experts(bank, dz.tensor({h, cols}), Tensor(did.p, DType::I32, {k, cols}), weights, 4.0F, 25.0F, arena,
                                              bad, nullptr);
            } catch (const std::invalid_argument&) { return true; }
            return false;
        };
        check(experts_refuse(nullptr, no_out) && experts_refuse(nullptr, flat) && experts_refuse(nullptr, as_3d) &&
                  (k == 1 || experts_refuse(&tw, weighted_wide)),
              label + ": moe_situ_experts takes an out without storage or of the wrong rank-2 shape");
    }
    if (nvfp4) { // the vector loads need 16-byte aligned records: an offset base is refused, not read
        auto shifted = bank;
        shifted.records += 4;
        bool refused = false;
        try {
            ninfer::ops::moe_situ_experts(shifted, dz.tensor({h, cols}), Tensor(did.p, DType::I32, {k, cols}), &tw, 4.0F, 25.0F, arena, tout,
                                          nullptr);
        } catch (const std::invalid_argument&) { refused = true; }
        check(refused, label + ": unaligned NVFP4 records were accepted");
    }
    double scale = 0;
    for (double x : want) { scale = std::max(scale, std::fabs(x)); }
    Vec scaled(want.size());
    std::vector<float> got = dout.get(h * cols);
    // Compare relative to the output's scale: these sums of thousands of random products cancel.
    std::vector<float> got_scaled(got.size());
    for (std::size_t i = 0; i < want.size(); ++i) {
        scaled[i]     = want[i] / scale;
        got_scaled[i] = static_cast<float>(got[i] / scale);
    }
    compare(label, got_scaled, scaled, 1e-5);
    if (nvfp4 ? h % 128 == 0 && inter % 64 == 0 : h % 8 == 0 && inter % 8 == 0) {
        // The tensor-core route (FP16 inputs and activations, FP32 accumulation) against the same oracle.
        ninfer::DeviceArena tc_arena(ninfer::ops::moe_situ_experts_tc_workspace_bytes(bank, entries) + 4096);
        Buf tc_out(static_cast<std::size_t>(h) * cols);
        Tensor ttc = tc_out.tensor({h, cols});
        ninfer::ops::moe_situ_experts_tc(bank, dz.tensor({h, cols}), Tensor(did.p, DType::I32, {k, cols}), &tw, 4.0F, 25.0F, tc_arena, ttc,
                                         nullptr);
        const std::vector<float> tc_got = tc_out.get(h * cols);
        std::vector<float> tc_scaled(tc_got.size());
        for (std::size_t i = 0; i < tc_got.size(); ++i) { tc_scaled[i] = static_cast<float>(tc_got[i] / scale); }
        compare(label + " (tensor cores)", tc_scaled, scaled, 3e-3);
    }
    if (cols > 2) {
        bool same = true;
        for (int c = 0; c < cols; ++c) {
            Buf one(static_cast<std::size_t>(h));
            Tensor to = one.tensor({h, 1});
            const Tensor zc = dz.tensor({h, cols}).slice(1, c, 1);
            const Tensor wc(static_cast<float*>(dw.b.p) + c * k, DType::FP32, {k, 1});
            ninfer::ops::moe_situ_experts(bank, zc, Tensor(static_cast<std::int32_t*>(did.p) + c * k, DType::I32, {k, 1}), &wc, 4.0F, 25.0F,
                                          arena, to, nullptr);
            const auto col = one.get(h);
            same = same && std::memcmp(col.data(), got.data() + static_cast<std::size_t>(c) * h, h * sizeof(float)) == 0;
        }
        check(same, label + ": a column differs from its own T = 1 call");
        std::printf("%-44s every column bit-exact with T = 1\n", label.c_str());
    }
}

// NINFER_INKLING2_OP_BENCH=1: decode-shaped timing (T = 1, top 8 of 32 resident experts at the model's shape, each
// call a different 8 so nothing stays in L2), printed as microseconds and effective weight bandwidth. Not a criterion.
void bench_experts(bool nvfp4, int cols = 1) {
    const int h = 2304, inter = 1152, experts = cols > 1 ? 128 : 32, k = 8, iters = cols > 1 ? 5 : 40;
    ninfer::ops::SituExpertBank bank;
    bank.experts = experts;
    bank.hidden = h;
    bank.intermediate = inter;
    ninfer::DeviceBuffer records, bgu, bd, scales;
    std::uint64_t per_expert = 0;
    if (nvfp4) {
        const std::uint64_t gate_row = h / 64 * 36, down_row = inter / 64 * 36, gate_bytes = gate_row * inter;
        const std::uint64_t up_off = align256(gate_bytes), down_off = align256(up_off + gate_bytes),
                            stride = align256(down_off + down_row * h);
        std::vector<std::uint8_t> host(stride * experts, 0);
        for (int e = 0; e < experts; ++e) {
            const auto blocks = nvfp4_blocks(stride / 36);
            std::copy(blocks.begin(), blocks.end(), host.begin() + e * stride);
        }
        records = t::to_device(host);
        scales  = t::to_device(std::vector<float>(2 * experts, 1.0F));
        bank.format        = ninfer::QType::GGML_REC_NVFP4_NVFP4;
        bank.records       = static_cast<const std::uint8_t*>(records.p);
        bank.record_stride = stride;
        bank.gate_up_scale = static_cast<const float*>(scales.p);
        bank.down_scale    = static_cast<const float*>(scales.p) + experts;
        per_expert         = 2 * gate_bytes + down_row * h;
    } else {
        bgu = t::to_device_bf16(randoms(static_cast<std::size_t>(experts) * 2 * inter * h, -0.05F, 0.05F));
        bd  = t::to_device_bf16(randoms(static_cast<std::size_t>(experts) * h * inter, -0.05F, 0.05F));
        bank.format  = ninfer::QType::BF16;
        bank.gate_up = bgu.p;
        bank.down    = bd.p;
        per_expert   = 3ULL * inter * h * 2;
    }
    // T = 1: each call a different 8 of the 32 experts. T > 1: every column 8 distinct experts of 128 at random.
    std::vector<int> ids(static_cast<std::size_t>(k) * iters * cols);
    for (int i = 0; i < iters; ++i)
        for (int c = 0; c < cols; ++c)
            for (int j = 0; j < k; ++j) {
                ids[(i * cols + c) * k + j] = cols == 1 ? (i * k + j) % experts
                                                        : (static_cast<int>(uniform(0.0F, 1.0F) * 16) + j * 16) % experts;
            }
    Buf dz(randoms(static_cast<std::size_t>(h) * cols)), dw(std::vector<float>(static_cast<std::size_t>(k) * cols, 1.0F)),
        dout(static_cast<std::size_t>(h) * cols);
    ninfer::DeviceBuffer did = t::to_device_i32(ids);
    ninfer::DeviceArena arena(ninfer::ops::moe_situ_experts_workspace_bytes(bank, static_cast<std::int64_t>(k) * cols) + 4096);
    Tensor tout = dout.tensor({h, cols});
    const auto run = [&](int i) {
        const Tensor tw = dw.tensor({k, cols});
        ninfer::ops::moe_situ_experts(bank, dz.tensor({h, cols}),
                                      Tensor(static_cast<std::int32_t*>(did.p) + static_cast<std::size_t>(i) * k * cols, DType::I32, {k, cols}),
                                      &tw, 4.0F, 25.0F, arena, tout, nullptr);
    };
    run(0);
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    for (int i = 0; i < iters; ++i) { run(i); }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    const double us = 1e3 * ms / iters;
    if (cols > 1) {
        std::printf("bench moe_situ_experts %s T=%d top 8 of 128: %.1f us, %.1f us a column\n", nvfp4 ? "NVFP4" : "BF16 ", cols, us, us / cols);
    }
    if (cols == 1) std::printf("bench moe_situ_experts %s T=1 top 8: %.1f us, %.0f GB/s of expert weights\n", nvfp4 ? "NVFP4" : "BF16 ", us,
                k * per_expert / (us * 1e3));
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

// projection_fp32 at Inkling-2's decode shapes (T = 1): time per call and weight bandwidth.
void bench_projection() {
    struct Shape {
        int k, n;
        const char* what;
    };
    const Shape shapes[] = {{2304, 6552, "q|rel|gate|sink"}, {2304, 2048, "local k|v"}, {2304, 1056, "global k|dyn"},
                            {3072, 2304, "output"},          {4608, 64, "control down"}, {64, 4608, "control gates"},
                            {2304, 201024, "head"}};
    for (const Shape& shape : shapes) {
        const int iters = shape.n > 100000 ? 10 : 50, copies = shape.n > 100000 ? 1 : 8; // rotate copies past the L2
        ninfer::DeviceBuffer w = t::to_device_bf16(randoms(static_cast<std::size_t>(shape.k) * shape.n * copies, -0.05F, 0.05F));
        ninfer::DeviceBuffer x = t::to_device_bf16(randoms(static_cast<std::size_t>(shape.k)));
        Buf out(static_cast<std::size_t>(shape.n));
        Tensor tx(x.p, DType::BF16, {shape.k, 1}), to = out.tensor({shape.n, 1});
        const auto run = [&](int i) {
            const Tensor tw(static_cast<char*>(w.p) + static_cast<std::size_t>(i % copies) * shape.k * shape.n * 2, DType::BF16,
                            {shape.k, shape.n});
            const Tensor* ws[] = {&tw};
            ninfer::ops::projection_fp32(tx, ws, to, nullptr);
        };
        run(0);
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);
        for (int i = 0; i < iters; ++i) { run(i); }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms = 0;
        cudaEventElapsedTime(&ms, start, stop);
        const double us = 1e3 * ms / iters;
        std::printf("bench projection_fp32 K %5d N %6d T 1 %-16s %8.1f us, %4.0f GB/s\n", shape.k, shape.n, shape.what, us,
                    2.0 * shape.k * shape.n / (us * 1e3));
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
}

} // namespace

// More columns than a grid's y holds (70000 > 65535): every column is the same, so each must give the first
// column's bits, and the first must meet the oracle.
// Wide calls (the column-tiled route scores and controller apply, the segmented convolution) against one call per
// column: the same bits.
void test_wide_bits() {
    const auto same_bits = [](const std::vector<float>& a, const std::vector<float>& b) {
        return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * 4) == 0;
    };
    const int cols = 203;
    {
        const int h = 2304, e = 128, k = 8;
        auto x = randoms(h * cols, -2, 2), gain = randoms(h, 0.5F, 1.5F), router = randoms(e * h, -0.05F, 0.05F), bias = randoms(e, -0.01F, 0.01F);
        Buf dx(x), dg(gain), dr(router), db(bias);
        for (const bool with_norm : {false, true}) {
            Buf ids(static_cast<std::size_t>(k) * cols), w(static_cast<std::size_t>(k) * cols), z(static_cast<std::size_t>(h) * cols);
            Buf ids1(static_cast<std::size_t>(k) * cols), w1(static_cast<std::size_t>(k) * cols), z1(static_cast<std::size_t>(h) * cols);
            ninfer::DeviceArena arena(ninfer::ops::moe_route_sigmoid_topk_workspace_bytes(e, cols) + 4096);
            const Tensor tg = dg.tensor({h});
            Tensor tz = z.tensor({h, cols});
            ninfer::ops::RouteInputNorm norm{&tg, 1e-6F, &tz};
            Tensor tid = ids.tensor({k, cols}, DType::I32), tw = w.tensor({k, cols});
            ninfer::ops::moe_route_sigmoid_topk(dx.tensor({h, cols}), with_norm ? &norm : nullptr, dr.tensor({h, e}), db.tensor({e}), k, 8.0F, arena,
                                                tid, tw, nullptr);
            for (int c = 0; c < cols; ++c) {
                Tensor tz1 = z1.tensor({h, cols}).slice(1, c, 1);
                ninfer::ops::RouteInputNorm norm1{&tg, 1e-6F, &tz1};
                Tensor tid1 = ids1.tensor({k, cols}, DType::I32).slice(1, c, 1), tw1 = w1.tensor({k, cols}).slice(1, c, 1);
                ninfer::ops::moe_route_sigmoid_topk(dx.tensor({h, cols}).slice(1, c, 1), with_norm ? &norm1 : nullptr, dr.tensor({h, e}),
                                                    db.tensor({e}), k, 8.0F, arena, tid1, tw1, nullptr);
            }
            bool ok = same_bits(ids.get(static_cast<std::size_t>(k) * cols), ids1.get(static_cast<std::size_t>(k) * cols)) &&
                      same_bits(w.get(static_cast<std::size_t>(k) * cols), w1.get(static_cast<std::size_t>(k) * cols));
            if (with_norm) { ok = ok && same_bits(z.get(static_cast<std::size_t>(h) * cols), z1.get(static_cast<std::size_t>(h) * cols)); }
            check(ok, std::string("moe_route_sigmoid_topk") + (with_norm ? " with the norm" : "") + ": the wide call differs from per-column calls");
        }
    }
    {
        const int h = 2304, rank = 64;
        auto c = randoms(h * cols), m = randoms(h * cols), down = randoms(rank * 2 * h, -0.02F, 0.02F), gain = randoms(h * rank, -0.1F, 0.1F),
             forget = randoms(h * rank, -0.1F, 0.1F);
        t::round_to_bf16(down), t::round_to_bf16(gain), t::round_to_bf16(forget);
        Buf dc(c), dm(m), out(static_cast<std::size_t>(h) * cols), out1(static_cast<std::size_t>(h) * cols);
        auto wd = t::to_device_bf16(down), wg = t::to_device_bf16(gain), wf = t::to_device_bf16(forget);
        const Tensor tdown(wd.p, DType::BF16, {2 * h, rank}), tgain(wg.p, DType::BF16, {rank, h}), tforget(wf.p, DType::BF16, {rank, h});
        ninfer::DeviceArena arena(ninfer::ops::loop_controller_workspace_bytes(h, rank, cols, 1));
        {
            const Tensor tc = dc.tensor({h, cols}), tm = dm.tensor({h, cols});
            Tensor to = out.tensor({h, cols});
            const ninfer::ops::LoopControllerJob job{&tc, &tm, &tdown, &tgain, &tforget, &to};
            ninfer::ops::loop_controller_fp32(std::span(&job, 1), 1e-6F, arena, nullptr);
        }
        for (int i = 0; i < cols; ++i) {
            const Tensor tc = dc.tensor({h, cols}).slice(1, i, 1), tm = dm.tensor({h, cols}).slice(1, i, 1);
            Tensor to = out1.tensor({h, cols}).slice(1, i, 1);
            const ninfer::ops::LoopControllerJob job{&tc, &tm, &tdown, &tgain, &tforget, &to};
            ninfer::ops::loop_controller_fp32(std::span(&job, 1), 1e-6F, arena, nullptr);
        }
        check(same_bits(out.get(static_cast<std::size_t>(h) * cols), out1.get(static_cast<std::size_t>(h) * cols)),
              "loop_controller_fp32: the wide call differs from per-column calls");
    }
    {
        const int channels = 2304, k = 4, top_k = 8;
        auto x = randoms(channels * cols), taps = randoms(channels * k, -0.5F, 0.5F), hist0 = randoms(channels * (k - 1)),
             entries = randoms(channels * top_k * cols), weights = randoms(top_k * cols, 0, 0.3F), residual = randoms(channels * cols);
        Buf dx(x), dt(taps), de(entries), dw(weights), dres(residual);
        for (const bool combine : {false, true}) {
            Buf h_wide(hist0), h_col(hist0), sum(static_cast<std::size_t>(channels) * cols), sum1(static_cast<std::size_t>(channels) * cols),
                xc(static_cast<std::size_t>(channels) * cols), xc1(static_cast<std::size_t>(channels) * cols);
            const auto run = [&](int s, int len, Buf& hist, Buf& out_sum, Buf& out_x) {
                Tensor th = hist.tensor({channels, k - 1}), tsum = out_sum.tensor({channels, cols}).slice(1, s, len);
                const Tensor tx = dx.tensor({channels, cols}).slice(1, s, len), tres = dres.tensor({channels, cols}).slice(1, s, len);
                const Tensor te(static_cast<float*>(de.b.p) + static_cast<std::size_t>(s) * top_k * channels, DType::FP32, {channels, top_k * len});
                const Tensor tw(static_cast<float*>(dw.b.p) + static_cast<std::size_t>(s) * top_k, DType::FP32, {top_k, len});
                Tensor txc = out_x.tensor({channels, cols}).slice(1, s, len);
                const ninfer::ops::ConvCombine cb{&te, &tw, &txc};
                ninfer::ops::ShortConvJob job;
                if (combine) { job.combine = &cb; } else { job.x = &tx; }
                const Tensor taps_t = dt.tensor({k, channels});
                job.taps = &taps_t, job.history = &th, job.residual = &tres, job.sum = &tsum;
                ninfer::ops::short_conv_fp32_batch(std::span(&job, 1), 0, nullptr);
            };
            run(0, cols, h_wide, sum, xc);
            for (int i = 0; i < cols; ++i) { run(i, 1, h_col, sum1, xc1); }
            bool ok = same_bits(sum.get(static_cast<std::size_t>(channels) * cols), sum1.get(static_cast<std::size_t>(channels) * cols)) &&
                      same_bits(h_wide.get(static_cast<std::size_t>(channels) * (k - 1)), h_col.get(static_cast<std::size_t>(channels) * (k - 1)));
            if (combine) { ok = ok && same_bits(xc.get(static_cast<std::size_t>(channels) * cols), xc1.get(static_cast<std::size_t>(channels) * cols)); }
            check(ok, std::string("short_conv_fp32") + (combine ? " with the combine" : "") + ": the wide call (or its history) differs from per-column calls");
        }
    }
    std::printf("wide route, controller and convolution calls: the bits of one call per column\n");
}

void test_many_columns() {
    const int h = 64, cols = 70000, e = 16, k = 2;
    const auto col0 = randoms(h, -1, 1), col1 = randoms(h, -2, 2), q = randoms(h, -0.3F, 0.3F);
    std::vector<float> a, b;
    for (int c = 0; c < cols; ++c) {
        a.insert(a.end(), col0.begin(), col0.end());
        b.insert(b.end(), col1.begin(), col1.end());
    }
    Buf da(a), db(b), dq(q), out(static_cast<std::size_t>(h) * cols);
    const Tensor ta = da.tensor({h, cols}), tb = db.tensor({h, cols});
    Tensor to = out.tensor({h, cols});
    const Tensor* sources[] = {&ta, &tb};
    Tensor* outs[] = {&to};
    ninfer::DeviceArena arena(ninfer::ops::depth_attention_mix_workspace_bytes(h, cols, 2) + 4096);
    ninfer::ops::depth_attention_mix(sources, dq.tensor({h, 1}), 1e-6F, arena, outs, nullptr);
    const auto got = out.get(static_cast<std::size_t>(h) * cols);
    bool same = true;
    for (int c = 1; c < cols && same; ++c) { same = std::equal(got.begin(), got.begin() + h, got.begin() + static_cast<std::ptrdiff_t>(c) * h); }
    check(same, "MIX over 70000 columns: a column differs from the first");
    compare("depth_attention_mix T = 70000, column 0", std::vector<float>(got.begin(), got.begin() + h),
            ref::mix({widen(col0), widen(col1)}, widen(q), h), 2e-6);

    const auto router = randoms(e * h, -0.5F, 0.5F), bias = randoms(e, -0.01F, 0.01F);
    Buf dr(router), dbias(bias), ids(static_cast<std::size_t>(k) * cols), w(static_cast<std::size_t>(k) * cols);
    Tensor tid = ids.tensor({k, cols}, DType::I32), tw = w.tensor({k, cols});
    ninfer::DeviceArena route_arena(ninfer::ops::moe_route_sigmoid_topk_workspace_bytes(e, cols) + 4096);
    ninfer::ops::moe_route_sigmoid_topk(ta, nullptr, dr.tensor({h, e}), dbias.tensor({e}), k, 8.0F, route_arena, tid, tw, nullptr);
    const auto got_ids = t::from_device<std::int32_t>(ids.b, static_cast<std::size_t>(k) * cols);
    const auto got_w = w.get(static_cast<std::size_t>(k) * cols);
    bool same_route = true;
    for (int c = 1; c < cols && same_route; ++c) {
        for (int i = 0; i < k; ++i) { same_route = same_route && got_ids[c * k + i] == got_ids[i] && got_w[c * k + i] == got_w[i]; }
    }
    check(same_route, "route over 70000 columns: a column differs from the first");
    const ref::Route want = ref::route(widen(col0), widen(router), widen(bias), k, 8.0);
    bool ids_match = true;
    for (int i = 0; i < k; ++i) { ids_match = ids_match && got_ids[i] == want.ids[i]; }
    check(ids_match, "route T = 70000: column 0's experts differ from the oracle");
    std::printf("T = 70000 (past a grid's y): MIX and route give every column the first one's bits\n");
}

// The workspace queries refuse what the Ops refuse.
void test_workspace_queries() {
    const auto refuses = [](auto query) {
        try {
            (void)query();
        } catch (const std::invalid_argument&) { return true; }
        return false;
    };
    namespace o = ninfer::ops;
    check(refuses([] { return o::depth_attention_mix_workspace_bytes(2304, 0, 2); }), "MIX query takes T = 0");
    check(refuses([] { return o::depth_attention_mix_workspace_bytes(2304, -1, 2); }), "MIX query takes T < 0");
    check(refuses([] { return o::depth_attention_mix_workspace_bytes(0, 1, 2); }), "MIX query takes D = 0");
    check(refuses([] { return o::depth_attention_mix_workspace_bytes(2304, 1, 9); }), "MIX query takes 9 sources");
    check(refuses([] { return o::moe_route_sigmoid_topk_workspace_bytes(0, 1); }), "route query takes E = 0");
    check(refuses([] { return o::moe_route_sigmoid_topk_workspace_bytes(257, 1); }), "route query takes E > 256");
    check(refuses([] { return o::moe_route_sigmoid_topk_workspace_bytes(128, 0); }), "route query takes T = 0");
    check(!refuses([] { return o::depth_attention_mix_workspace_bytes(2304, 70000, 8); }), "MIX query refuses T > 65535");
    check(!refuses([] { return o::moe_route_sigmoid_topk_workspace_bytes(256, 70000); }), "route query refuses T > 65535");
    std::printf("workspace queries refuse outside the domain\n");
}

int main() {
    if (t::cuda_unavailable()) {
        std::printf("SKIP: no usable CUDA device\n");
        return 77;
    }
    try {
        test_workspace_queries();
        test_many_columns();
        test_wide_bits();
        test_rows();
        test_mix_and_control();
        test_short_conv(false);
        test_short_conv(true);
        test_clips();
        test_batches();
        test_attention(1024, 1024, 12, 5, "attention window 1024 > context (5+5+2)");
        test_attention(0, 4, 12, 5, "attention global, extent 4 < context");
        test_attention(4, 4, 13, 3, "attention local window 4, ring wraps");
        test_attention(4, 4, 13, 1, "attention local window 4, decode steps");
        test_attention(4, 4, 13, 1, "attention local window 4, decode, device start", true);
        test_attention(0, 4, 12, 1, "attention global, decode, device start", true);
        test_attention(0, 1024, 1030, 515, "attention global, two key tiles (last 6 of 1030)", false, 1024);
        // Every key of a tile scored: one query sees 40 keys (16-key steps, a partial last one), all compared.
        test_attention(0, 64, 40, 40, "attention global, 40 keys in one call");
        test_attention(0, 1024, 1300, 4, "attention global split, chunks of 4 (last 4)", false, 1296, true);
        test_attention(0, 1024, 1300, 1, "attention global split, decode, device start (last 4)", true, 1296, true);
        test_attention(1024, 1024, 1300, 4, "attention local window 1024 split, ring wraps (last 4)", false, 1296, true);
        test_attention(0, 1024, 1300, 260, "attention global tiled, chunks of 260 (last 40)", false, 1260, false, true);
        test_attention(0, 64, 1300, 100, "attention global tiled, extent 64, chunks of 100 (last 40)", false, 1260, false, true);
        test_attention(1024, 1024, 2100, 300, "attention local window 1024 tiled, ring wraps (last 40)", false, 2060, false, true);
        test_attention(4, 4, 40, 19, "attention local window 4 tiled, partial tiles", false, 0, false, true);
        test_attention(0, 64, 200, 1, "attention global, 200 keys with the split workspace: the single block (last 8)", true, 192, true);
        test_head_norm_append();
        test_kv_fp16();
        test_projection_inputs();
        test_route();
        test_experts(true, 128, 64, 3, "moe_situ_experts NVFP4 small");
        test_experts(true, 2304, 1152, 3, "moe_situ_experts NVFP4 [3, 2304, 1152]");
        test_experts(false, 2304, 1152, 3, "moe_situ_experts BF16 [3, 2304, 1152]");
        test_experts(true, 128, 64, 8, "moe_situ_experts NVFP4 small, T 24 grouped", 24);
        test_experts(true, 2304, 1152, 16, "moe_situ_experts NVFP4 [16, 2304, 1152] T 17 grouped", 17);
        test_experts(false, 2304, 1152, 8, "moe_situ_experts BF16 [8, 2304, 1152] T 17 grouped", 17);
        // H = 8192: the grouped kernels' staged activations exceed shared memory, so every entry runs alone.
        test_experts(true, 8192, 128, 4, "moe_situ_experts NVFP4 [4, 8192, 128] T 17 per entry", 17);
        test_experts(false, 8192, 128, 4, "moe_situ_experts BF16 [4, 8192, 128] T 17 per entry", 17);
        if (const char* bench = std::getenv("NINFER_INKLING2_OP_BENCH"); bench != nullptr && *bench == '1') {
            bench_experts(true);
            bench_experts(false);
            bench_experts(true, 256);
            bench_projection();
        }
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL: %s\n", error.what());
        return 1;
    }
    if (failures) {
        std::fprintf(stderr, "%d failure(s)\n", failures);
        return 1;
    }
    std::printf("PASS\n");
    return 0;
}
