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
#include "ninfer/ops/relative_bias_attention.h"
#include "ninfer/ops/short_conv.h"

#include "core/arena.h"
#include "ops/inkling2_reference.h"
#include "ops/op_tester.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
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

    // The controller's combine given gf; the oracle's control() computes gf itself from D, G, F, so it is formed
    // here in FP64 and handed to both.
    auto cc = randoms(h * cols), mm = randoms(h * cols), down = randoms(rank * 2 * h, -0.02F, 0.02F),
         gain = randoms(h * rank, -0.1F, 0.1F), forget = randoms(h * rank, -0.1F, 0.1F);
    std::vector<float> gf(2 * h * cols);
    Vec want;
    for (int c = 0; c < cols; ++c) {
        const Vec vc(cc.begin() + c * h, cc.begin() + (c + 1) * h), vm(mm.begin() + c * h, mm.begin() + (c + 1) * h);
        Vec in = ref::unit_rms(vc);
        const Vec um = ref::unit_rms(vm);
        in.insert(in.end(), um.begin(), um.end());
        Vec z(rank, 0.0);
        for (int r = 0; r < rank; ++r)
            for (int j = 0; j < 2 * h; ++j) { z[r] += down[r * 2 * h + j] * in[j]; }
        for (int d = 0; d < h; ++d) {
            double g = 0, f = 0;
            for (int r = 0; r < rank; ++r) {
                g += gain[d * rank + r] * z[r];
                f += forget[d * rank + r] * z[r];
            }
            gf[c * 2 * h + d]     = static_cast<float>(g);
            gf[c * 2 * h + h + d] = static_cast<float>(f);
        }
        const Vec y = ref::control(vc, vm, widen(down), widen(gain), widen(forget));
        want.insert(want.end(), y.begin(), y.end());
    }
    Buf dc(cc), dm(mm), dgf(gf), dout(static_cast<std::size_t>(h * cols));
    Tensor tout = dout.tensor({h, cols});
    ninfer::ops::forget_gain_combine(dc.tensor({h, cols}), dm.tensor({h, cols}), dgf.tensor({2 * h, cols}), tout, nullptr);
    compare("forget_gain_combine (vs control oracle)", dout.get(h * cols), want, 2e-6);
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

// Positions 0..n-1 through relative_bias_sink_attention in calls of `chunk`, against the oracle over the whole sequence.
// With device_start each call passes start 0 and the position as the device offset (how a decode graph runs).
void test_attention(int window, int extent, int n, int chunk, const std::string& label, bool device_start = false) {
    const int heads = 24, kvh = 8, d = 128, rel = 16, qw = heads * d, kvw = kvh * d;
    auto q = randoms(qw * n, -0.3F, 0.3F), r = randoms(heads * rel * n), sink = randoms(heads * n, -2, 2), k = randoms(kvw * n),
         v = randoms(kvw * n), table = randoms(extent * rel, -0.5, 0.5);
    const int capacity = window > 0 ? window - 1 + chunk : n;
    Buf dq(q), dr(r), ds(sink), dk(k), dv(v), dt(table), kc(static_cast<std::size_t>(kvw * capacity)),
        vc(static_cast<std::size_t>(kvw * capacity)), out(static_cast<std::size_t>(qw * n));
    Tensor tkc = kc.tensor({kvw, capacity}), tvc = vc.tensor({kvw, capacity});
    ninfer::DeviceBuffer doff(sizeof(std::int64_t));
    for (int s = 0; s < n; s += chunk) {
        const int len = std::min(chunk, n - s);
        const std::int64_t position = s;
        if (device_start) {
            check(cudaMemcpy(doff.p, &position, sizeof position, cudaMemcpyHostToDevice) == cudaSuccess, "offset upload");
        }
        Tensor to = out.tensor({qw, n}).slice(1, s, len);
        ninfer::ops::relative_bias_sink_attention({heads, kvh, d, rel, window}, dq.tensor({qw, n}).slice(1, s, len),
                                                  dr.tensor({heads * rel, n}).slice(1, s, len), ds.tensor({heads, n}).slice(1, s, len),
                                                  dk.tensor({kvw, n}).slice(1, s, len), dv.tensor({kvw, n}).slice(1, s, len),
                                                  dt.tensor({rel, extent}), tkc, tvc, device_start ? 0 : s,
                                                  device_start ? static_cast<const std::int64_t*>(doff.p) : nullptr, to, nullptr);
    }
    Vec want;
    for (int p = 0; p < n; ++p) {
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
    compare(label, out.get(qw * n), want, 2e-6);
}

void test_route() {
    const int h = 2304, e = 128, k = 8, cols = 4;
    auto z = randoms(h * cols), router = randoms(e * h, -0.05F, 0.05F), bias = randoms(e, -0.01F, 0.01F);
    Buf dz(z), dr(router), db(bias), ids(static_cast<std::size_t>(k * cols)), w(static_cast<std::size_t>(k * cols));
    Tensor tid = ids.tensor({k, cols}, DType::I32), tw = w.tensor({k, cols});
    ninfer::DeviceArena arena(ninfer::ops::moe_route_sigmoid_topk_workspace_bytes(e, cols) + 4096);
    ninfer::ops::moe_route_sigmoid_topk(dz.tensor({h, cols}), dr.tensor({h, e}), db.tensor({e}), k, 8.0F, arena, tid, tw, nullptr);
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

    // Exact ties (a zero router: every score 1/2) among 40 experts, bias steering two of them: the steered two
    // first, then the lowest ids.
    const int few = 40, picks = 5;
    std::vector<float> zero(static_cast<std::size_t>(few) * h, 0.0F), steer(few, 0.0F);
    steer[33] = 0.5F;
    steer[7]  = 0.25F;
    Buf dzero(zero), dsteer(steer), tie_ids(static_cast<std::size_t>(picks * cols)), tie_w(static_cast<std::size_t>(picks * cols));
    Tensor tti = tie_ids.tensor({picks, cols}, DType::I32), ttw = tie_w.tensor({picks, cols});
    ninfer::DeviceArena tie_arena(ninfer::ops::moe_route_sigmoid_topk_workspace_bytes(few, cols) + 4096);
    ninfer::ops::moe_route_sigmoid_topk(dz.tensor({h, cols}), dzero.tensor({h, few}), dsteer.tensor({few}), picks, 8.0F,
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

void test_experts(bool nvfp4, int h, int inter, int experts, const std::string& label) {
    const int k = 4, cols = 2, entries = k * cols;
    auto z = randoms(h * cols);
    std::vector<int> ids = {0, experts - 1, 1, 2 % experts, 1, 0, experts - 1, 2 % experts};
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
    ninfer::ops::moe_situ_experts(bank, dz.tensor({h, cols}), Tensor(did.p, DType::I32, {k, cols}), dw.tensor({k, cols}), 4.0F, 25.0F,
                                  arena, tout, nullptr);
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
}

// NINFER_INKLING2_OP_BENCH=1: decode-shaped timing (T = 1, top 8 of 32 resident experts at the model's shape, each
// call a different 8 so nothing stays in L2), printed as microseconds and effective weight bandwidth. Not a criterion.
void bench_experts(bool nvfp4) {
    const int h = 2304, inter = 1152, experts = 32, k = 8, iters = 40;
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
    std::vector<int> ids(static_cast<std::size_t>(k) * iters);
    for (int i = 0; i < iters; ++i)
        for (int j = 0; j < k; ++j) { ids[i * k + j] = (i * k + j) % experts; }
    Buf dz(randoms(h)), dw(std::vector<float>(k, 1.0F)), dout(static_cast<std::size_t>(h));
    ninfer::DeviceBuffer did = t::to_device_i32(ids);
    ninfer::DeviceArena arena(ninfer::ops::moe_situ_experts_workspace_bytes(bank, k) + 4096);
    Tensor tout = dout.tensor({h, 1});
    const auto run = [&](int i) {
        ninfer::ops::moe_situ_experts(bank, dz.tensor({h, 1}), Tensor(static_cast<std::int32_t*>(did.p) + i * k, DType::I32, {k, 1}),
                                      dw.tensor({k, 1}), 4.0F, 25.0F, arena, tout, nullptr);
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
    std::printf("bench moe_situ_experts %s T=1 top 8: %.1f us, %.0f GB/s of expert weights\n", nvfp4 ? "NVFP4" : "BF16 ", us,
                k * per_expert / (us * 1e3));
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

} // namespace

int main() {
    if (t::cuda_unavailable()) {
        std::printf("SKIP: no usable CUDA device\n");
        return 77;
    }
    try {
        test_rows();
        test_mix_and_control();
        test_short_conv(false);
        test_short_conv(true);
        test_clips();
        test_attention(1024, 1024, 12, 5, "attention window 1024 > context (5+5+2)");
        test_attention(0, 4, 12, 5, "attention global, extent 4 < context");
        test_attention(4, 4, 13, 3, "attention local window 4, ring wraps");
        test_attention(4, 4, 13, 1, "attention local window 4, decode steps");
        test_attention(4, 4, 13, 1, "attention local window 4, decode, device start", true);
        test_attention(0, 4, 12, 1, "attention global, decode, device start", true);
        test_route();
        test_experts(true, 128, 64, 3, "moe_situ_experts NVFP4 small");
        test_experts(true, 2304, 1152, 3, "moe_situ_experts NVFP4 [3, 2304, 1152]");
        test_experts(false, 2304, 1152, 3, "moe_situ_experts BF16 [3, 2304, 1152]");
        if (const char* bench = std::getenv("NINFER_INKLING2_OP_BENCH"); bench != nullptr && *bench == '1') {
            bench_experts(true);
            bench_experts(false);
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
