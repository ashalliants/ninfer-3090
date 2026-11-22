// Written for NInfer-3090.
//
// The FP32 Ops of Inkling-2-Mini's DSpark drafter (include/ninfer/ops/dspark_fp32.h) against FP64 oracles written
// here from the header's formulas, at the drafter's shapes (H 2304, 24/8 heads of 128, FFN 9216, rank 256, blocks
// of 8) plus edge cases (a wrapped context ring, a window shorter than the context, an anchor at 0, argmax ties).
//
// Criterion: every output within tol * (1 + |ref|) of the oracle, finite, with tol 2e-6 (chains of at most a few
// thousand terms); a projection within 1e-7 * (1 + sum |terms|) (its random terms cancel, so |ref| says little);
// argmax exact.

#include "ninfer/ops/dspark_fp32.h"

#include "ops/op_tester.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

namespace t = ninfer::test;
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

std::uint32_t seed_state = 777;
float uniform(float lo = -1.0F, float hi = 1.0F) {
    seed_state = seed_state * 1664525U + 1013904223U;
    return lo + (hi - lo) * static_cast<float>(seed_state >> 8) / static_cast<float>(1 << 24);
}
std::vector<float> randoms(std::size_t n, float lo = -1.0F, float hi = 1.0F) {
    std::vector<float> v(n);
    for (auto& x : v) { x = uniform(lo, hi); }
    return v;
}

struct Buf {
    ninfer::DeviceBuffer b;
    explicit Buf(const std::vector<float>& v) : b(t::to_device(v)) {}
    explicit Buf(std::size_t bytes) : b(bytes) { b.fill(0); }
    Tensor tensor(std::initializer_list<std::int32_t> ne, DType dtype = DType::FP32) { return Tensor(b.p, dtype, ne); }
    std::vector<float> get(std::size_t n) const { return t::from_device<float>(b, n); }
};

ninfer::DeviceBuffer bytes_to_device(const std::vector<std::uint8_t>& v) {
    ninfer::DeviceBuffer b(v.size());
    if (cudaMemcpy(b.p, v.data(), v.size(), cudaMemcpyHostToDevice) != cudaSuccess) { throw std::runtime_error("copy"); }
    return b;
}

// Error scaled by 1 + bound[i] (default |want[i]|).
void compare(const std::string& label, const std::vector<float>& got, const Vec& want, double tol, const Vec* bound = nullptr) {
    check(got.size() == want.size(), label + ": size");
    double worst = 0;
    std::size_t at = 0;
    for (std::size_t i = 0; i < std::min(got.size(), want.size()); ++i) {
        if (!std::isfinite(got[i])) {
            check(false, label + ": non-finite output");
            return;
        }
        const double e = std::fabs(double(got[i]) - want[i]) / (1.0 + (bound ? (*bound)[i] : std::fabs(want[i])));
        if (e > worst) { worst = e, at = i; }
    }
    std::printf("%-48s max scaled error %.2e (index %zu: %.9g vs %.9g)\n", label.c_str(), worst, at, double(got[at]), want[at]);
    check(worst <= tol, label + ": outside tolerance");
}

// A random matrix [n, k] as Q8_0 blocks or BF16 words, and its exact values.
struct Matrix {
    std::vector<std::uint8_t> bytes;
    std::vector<double> values;
};

Matrix make_matrix(bool q8, int n, int k) {
    Matrix m;
    m.values.resize(static_cast<std::size_t>(n) * k);
    if (q8) {
        for (int r = 0; r < n; ++r) {
            for (int b = 0; b < k / 32; ++b) {
                const __half d       = __float2half(uniform(0.001F, 0.02F));
                const std::uint16_t w = __half_as_ushort(d);
                m.bytes.push_back(static_cast<std::uint8_t>(w & 0xFF));
                m.bytes.push_back(static_cast<std::uint8_t>(w >> 8));
                for (int i = 0; i < 32; ++i) {
                    const auto q = static_cast<std::int8_t>(static_cast<int>(uniform(-128.0F, 127.99F)));
                    m.bytes.push_back(static_cast<std::uint8_t>(q));
                    m.values[static_cast<std::size_t>(r) * k + 32 * b + i] = double(__half2float(d)) * q;
                }
            }
        }
    } else {
        m.bytes.resize(static_cast<std::size_t>(n) * k * 2);
        for (std::size_t i = 0; i < m.values.size(); ++i) {
            float f = uniform(-0.05F, 0.05F);
            std::uint32_t u;
            std::memcpy(&u, &f, 4);
            const auto word = static_cast<std::uint16_t>(u >> 16);
            u               = static_cast<std::uint32_t>(word) << 16;
            std::memcpy(&f, &u, 4);
            m.values[i] = f;
            std::memcpy(m.bytes.data() + 2 * i, &word, 2);
        }
    }
    return m;
}

// split: the Q8_0 blocks through split_q8_0 into the Q8_G32_FP16 form first (same values, so the same oracle).
void test_projection(bool q8, int n, int k, int cols, bool bias, double tol, bool split = false) {
    const std::string label = std::string("dense_projection_fp32 ") + (split ? "split Q8_0" : q8 ? "Q8_0" : "BF16") + " [" + std::to_string(n) + ", " +
                              std::to_string(k) + "] T " + std::to_string(cols) + (bias ? " + bias" : "");
    const Matrix m = make_matrix(q8, n, k);
    // x as a strided view: columns k + 32 floats apart.
    const int stride = k + 32;
    const auto x = randoms(static_cast<std::size_t>(stride) * cols, -2, 2);
    const auto b = randoms(n, -1, 1);
    auto dw      = bytes_to_device(m.bytes);
    Buf dx(x), db(b), out(static_cast<std::size_t>(n) * cols * 4);
    Tensor tx = dx.tensor({stride, cols}).slice(0, 0, k);
    Tensor tb = db.tensor({n}), to = out.tensor({n, cols});
    ninfer::DeviceBuffer values(split ? static_cast<std::size_t>(n) * k : 1), scales(split ? static_cast<std::size_t>(n) * k / 32 * 2 : 1);
    if (split) {
        ninfer::ops::split_q8_0(dw.p, n, k, values.p, scales.p, nullptr);
        ninfer::ops::dense_projection_fp32(tx, {values.p, ninfer::QType::Q8_G32_FP16, n, k, scales.p}, bias ? &tb : nullptr, to, nullptr);
    } else {
        ninfer::ops::dense_projection_fp32(tx, {dw.p, q8 ? ninfer::QType::GGML_Q8_0 : ninfer::QType::BF16, n, k}, bias ? &tb : nullptr, to,
                                          nullptr);
    }
    Vec want(static_cast<std::size_t>(n) * cols), magnitude(want.size());
    for (int c = 0; c < cols; ++c) {
        for (int r = 0; r < n; ++r) {
            double s = bias ? b[r] : 0.0, a = bias ? std::fabs(b[r]) : 0.0;
            for (int i = 0; i < k; ++i) {
                const double term = m.values[static_cast<std::size_t>(r) * k + i] * x[static_cast<std::size_t>(c) * stride + i];
                s += term;
                a += std::fabs(term);
            }
            want[static_cast<std::size_t>(c) * n + r] = s;
            magnitude[static_cast<std::size_t>(c) * n + r] = a;
        }
    }
    // Scaled by the sum of the terms' magnitudes: random terms cancel, so |want| alone says little.
    compare(label, out.get(want.size()), want, tol, &magnitude);
}

void test_head_norm_rope() {
    for (const bool rope : {true, false}) {
        const int d = 128, heads = 24, cols = 8;
        const std::int64_t start = 70001;
        const auto x = randoms(static_cast<std::size_t>(d) * heads * cols, -3, 3), w = randoms(d, 0.5, 1.5);
        Buf dx(x), dw(w);
        Tensor tx = dx.tensor({d * heads, cols}), tw = dw.tensor({d});
        ninfer::ops::head_rmsnorm_rope_fp32(tx, d, tw, 1e-6F, rope, start, 1e6F, nullptr);
        Vec want(x.size());
        for (int c = 0; c < cols; ++c) {
            for (int h = 0; h < heads; ++h) {
                const std::size_t o = (static_cast<std::size_t>(c) * heads + h) * d;
                double ss = 0;
                for (int i = 0; i < d; ++i) { ss += double(x[o + i]) * x[o + i]; }
                const double r = 1.0 / std::sqrt(ss / d + double(1e-6F));
                std::vector<double> y(d);
                for (int i = 0; i < d; ++i) { y[i] = x[o + i] * r * w[i]; }
                for (int i = 0; i < d / 2; ++i) {
                    const double a = double(start + c) * std::pow(1e6, -2.0 * i / d);
                    want[o + i]         = rope ? y[i] * std::cos(a) - y[i + d / 2] * std::sin(a) : y[i];
                    want[o + i + d / 2] = rope ? y[i + d / 2] * std::cos(a) + y[i] * std::sin(a) : y[i + d / 2];
                }
            }
        }
        compare(rope ? "head_rmsnorm_rope_fp32 (position 70001)" : "head_rmsnorm_rope_fp32 norm only", dx.get(x.size()), want, 2e-6);
    }
}

void test_silu() {
    const auto g = randoms(9216 * 8, -8, 8), u = randoms(9216 * 8, -2, 2);
    Buf dg(g), du(u), out(g.size() * 4);
    Tensor tg = dg.tensor({9216, 8}), tu = du.tensor({9216, 8}), to = out.tensor({9216, 8});
    ninfer::ops::silu_mul_fp32(tg, tu, to, nullptr);
    Vec want(g.size());
    for (std::size_t i = 0; i < g.size(); ++i) { want[i] = g[i] / (1.0 + std::exp(-double(g[i]))) * u[i]; }
    compare("silu_mul_fp32", out.get(g.size()), want, 2e-6);
}

void test_fuse_taps() {
    const int h = 2304, cols = 5, n = 4, rows = 4;
    std::vector<std::vector<float>> taps;
    std::vector<Buf> bufs;
    for (int i = 0; i < n; ++i) {
        taps.push_back(randoms(static_cast<std::size_t>(h) * cols, -4, 4));
        bufs.emplace_back(taps.back());
    }
    const auto fuse = randoms(rows * n, -2, 2), norm = randoms(h, 0.5, 1.5);
    Buf df(fuse), dn(norm), out(static_cast<std::size_t>(h) * cols * 4);
    std::vector<Tensor> tensors;
    for (auto& b : bufs) { tensors.push_back(b.tensor({h, cols})); }
    std::vector<const Tensor*> ptrs;
    for (auto& tt : tensors) { ptrs.push_back(&tt); }
    Tensor tf = df.tensor({n, rows}), tn = dn.tensor({h}), to = out.tensor({h, cols});
    for (const int row : {0, 3}) {
        ninfer::ops::dspark_fuse_taps_fp32(ptrs, tf, row, tn, 1e-6F, to, nullptr);
        double m = -1e300, total = 0;
        for (int i = 0; i < n; ++i) { m = std::max(m, double(fuse[row * n + i])); }
        Vec s(n);
        for (int i = 0; i < n; ++i) { total += s[i] = std::exp(fuse[row * n + i] - m); }
        Vec want(static_cast<std::size_t>(h) * cols);
        for (int c = 0; c < cols; ++c) {
            double ss = 0;
            Vec y(h);
            for (int d = 0; d < h; ++d) {
                for (int i = 0; i < n; ++i) { y[d] += s[i] / total * taps[i][static_cast<std::size_t>(c) * h + d]; }
                ss += y[d] * y[d];
            }
            const double r = 1.0 / std::sqrt(ss / h + double(1e-6F));
            for (int d = 0; d < h; ++d) { want[static_cast<std::size_t>(c) * h + d] = y[d] * r * norm[d]; }
        }
        compare("dspark_fuse_taps_fp32 row " + std::to_string(row), out.get(want.size()), want, 2e-6);
    }
}

// fp16: the rings are FP16, written by dspark_ring_write from FP32 columns (checked against the host's round to
// nearest), and the oracle reads the values they hold.
void test_attention(std::int64_t anchor, int capacity, int window, const std::string& label, int w = 8, bool fp16 = false) {
    const int d = 128, heads = 24, kv_heads = 8, kv = kv_heads * d;
    const auto q = randoms(static_cast<std::size_t>(heads) * d * w, -1, 1);
    auto kr = randoms(static_cast<std::size_t>(kv) * capacity, -1, 1), vr = randoms(static_cast<std::size_t>(kv) * capacity, -2, 2);
    const auto kb = randoms(static_cast<std::size_t>(kv) * w, -1, 1), vb = randoms(static_cast<std::size_t>(kv) * w, -2, 2);
    Buf dq(q), dkr(kr), dvr(vr), dkb(kb), dvb(vb), out(q.size() * 4);
    Tensor tq = dq.tensor({heads * d, w}), tkr = dkr.tensor({kv, capacity}), tvr = dvr.tensor({kv, capacity});
    Buf hk(static_cast<std::size_t>(kv) * capacity * 2), hv(static_cast<std::size_t>(kv) * capacity * 2);
    if (fp16) {
        // Written from slot 5 on, so the write wraps; the host keeps the same layout.
        Tensor fk = hk.tensor({kv, capacity}, DType::FP16), fv = hv.tensor({kv, capacity}, DType::FP16);
        ninfer::ops::dspark_ring_write(tkr, fk, 5, nullptr);
        ninfer::ops::dspark_ring_write(tvr, fv, 5, nullptr);
        for (auto [ring, host] : {std::pair{&hk, &kr}, std::pair{&hv, &vr}}) {
            std::vector<__half> got(static_cast<std::size_t>(kv) * capacity);
            if (cudaMemcpy(got.data(), ring->b.p, got.size() * 2, cudaMemcpyDeviceToHost) != cudaSuccess) { throw std::runtime_error("copy"); }
            std::vector<float> placed(host->size());
            bool exact = true;
            for (int c = 0; c < capacity; ++c) {
                const std::size_t slot = static_cast<std::size_t>((c + 5) % capacity);
                for (int i = 0; i < kv; ++i) {
                    const __half want = __float2half_rn((*host)[static_cast<std::size_t>(c) * kv + i]);
                    const __half have = got[slot * kv + i];
                    exact = exact && std::memcmp(&want, &have, 2) == 0;
                    placed[slot * kv + i] = __half2float(have);
                }
            }
            check(exact, "dspark_ring_write " + label + ": the FP16 slots are not the host's round to nearest");
            *host = placed;
        }
        tkr = hk.tensor({kv, capacity}, DType::FP16), tvr = hv.tensor({kv, capacity}, DType::FP16);
    }
    Tensor tkb = dkb.tensor({kv, w}), tvb = dvb.tensor({kv, w}), to = out.tensor({heads * d, w});
    const ninfer::ops::DSparkAttentionShape shape{heads, kv_heads, d, window};
    Buf ws(ninfer::ops::dspark_block_attention_workspace_bytes(shape, w));
    Tensor tws = ws.tensor({static_cast<std::int32_t>(ninfer::ops::dspark_block_attention_workspace_bytes(shape, w) / 4)});
    {
        // A workspace of the right size without storage is refused before either kernel runs.
        const Tensor no_ws(nullptr, DType::FP32, {tws.ne[0]});
        bool refused = false;
        try {
            ninfer::ops::dspark_block_attention_fp32(tq, tkr, tvr, tkb, tvb, anchor, shape, to, no_ws, nullptr);
        } catch (const std::invalid_argument&) { refused = true; }
        check(refused, "dspark_block_attention_fp32 takes a workspace without storage (" + label + ")");
    }
    ninfer::ops::dspark_block_attention_fp32(tq, tkr, tvr, tkb, tvb, anchor, shape, to, tws, nullptr);
    const std::int64_t lo = window > 0 ? std::max<std::int64_t>(0, anchor - window) : 0;
    Vec want(q.size());
    for (int t = 0; t < w; ++t) {
        for (int h = 0; h < heads; ++h) {
            const int g = h / (heads / kv_heads);
            std::vector<const float*> keys, values;
            for (std::int64_t j = lo; j < anchor; ++j) {
                keys.push_back(kr.data() + (j % capacity) * kv + g * d);
                values.push_back(vr.data() + (j % capacity) * kv + g * d);
            }
            for (int j = 0; j < w; ++j) {
                keys.push_back(kb.data() + static_cast<std::size_t>(j) * kv + g * d);
                values.push_back(vb.data() + static_cast<std::size_t>(j) * kv + g * d);
            }
            Vec s(keys.size());
            double m = -1e300;
            for (std::size_t j = 0; j < keys.size(); ++j) {
                for (int i = 0; i < d; ++i) { s[j] += double(q[(static_cast<std::size_t>(t) * heads + h) * d + i]) * keys[j][i]; }
                s[j] /= std::sqrt(double(d));
                m = std::max(m, s[j]);
            }
            double total = 0;
            for (auto& v : s) { total += v = std::exp(v - m); }
            for (int i = 0; i < d; ++i) {
                double acc = 0;
                for (std::size_t j = 0; j < keys.size(); ++j) { acc += s[j] * values[j][i]; }
                want[(static_cast<std::size_t>(t) * heads + h) * d + i] = acc / total;
            }
        }
    }
    compare("dspark_block_attention_fp32 " + label, out.get(want.size()), want, 2e-6);
}

void test_markov() {
    const int r = 256, v = 201024;
    const auto joint = randoms(3 * r, -4, 4), state = randoms(r, -1, 1);
    Buf dj(joint), ds(state), dl(r * 4);
    Tensor tj = dj.tensor({3 * r}), ts = ds.tensor({r}), tl = dl.tensor({r});
    ninfer::ops::dspark_markov_update_fp32(tj, ts, tl, nullptr);
    Vec want_s(r), want_l(r);
    for (int i = 0; i < r; ++i) {
        const double gate = 1.0 / (1.0 + std::exp(-double(joint[i]))), c = std::tanh(double(joint[r + i]));
        want_s[i] = c + gate * (state[i] - c);
        want_l[i] = std::tanh(double(joint[2 * r + i]));
    }
    compare("dspark_markov_update_fp32 state", ds.get(r), want_s, 2e-6);
    compare("dspark_markov_update_fp32 latent", dl.get(r), want_l, 2e-6);

    const auto base = randoms(v, -40, 40), hb = randoms(v, -1, 1), pt = randoms(static_cast<std::size_t>(r) * v, -0.2F, 0.2F);
    const auto latent = dl.get(r);
    Buf dbase(base), dhb(hb), dpt(pt), dlat(latent), out(static_cast<std::size_t>(v) * 4);
    Tensor tb = dbase.tensor({v}), th = dhb.tensor({v}), tp = dpt.tensor({v, r}), tlat = dlat.tensor({r}), to = out.tensor({v});
    ninfer::ops::dspark_markov_logits_fp32(tb, th, tp, tlat, 30.0F, to, nullptr);
    Vec want(v), magnitude(v);
    for (int i = 0; i < v; ++i) {
        double acc = 0, a = std::fabs(base[i]) + std::fabs(hb[i]);
        for (int k = 0; k < r; ++k) {
            const double term = double(pt[static_cast<std::size_t>(k) * v + i]) * latent[k];
            acc += term;
            a += std::fabs(term);
        }
        want[i]      = 30.0 * std::tanh((double(base[i]) + hb[i] + acc) / 30.0);
        magnitude[i] = a; // the logit's argument cancels; tanh' <= 1
    }
    compare("dspark_markov_logits_fp32", out.get(v), want, 1e-6, &magnitude);
    // In place over base, as the drafter runs it.
    ninfer::ops::dspark_markov_logits_fp32(tb, th, tp, tlat, 30.0F, tb, nullptr);
    compare("dspark_markov_logits_fp32 in place", dbase.get(v), want, 1e-6, &magnitude);
}

// The drafter's compact P (lossy by design): the codes follow the stated rounding exactly, and the logits equal the
// FP64 sum over the decoded values.
void test_markov_compact() {
    const int r = 256, v = 4104; // V a multiple of 8, not of 256
    const auto base = randoms(v, -40, 40), hb = randoms(v, -1, 1), pt = randoms(static_cast<std::size_t>(r) * v, -0.2F, 0.2F);
    const auto latent = randoms(r, -1, 1);
    Buf dbase(base), dhb(hb), dpt(pt), dlat(latent), out(static_cast<std::size_t>(v) * 4);
    Tensor tb = dbase.tensor({v}), th = dhb.tensor({v}), tp = dpt.tensor({v, r}), tlat = dlat.tensor({r}), to = out.tensor({v});
    using ninfer::ops::MarkovForm;
    for (const MarkovForm form : {MarkovForm::BF16, MarkovForm::Q8, MarkovForm::Q4}) {
        const char* name = form == MarkovForm::BF16 ? "BF16" : form == MarkovForm::Q8 ? "Q8" : "Q4";
        const std::size_t code_bytes = ninfer::ops::markov_head_code_bytes(form, v, r), scale_bytes = ninfer::ops::markov_head_scale_bytes(form, v, r);
        ninfer::DeviceBuffer codes(code_bytes), scales(std::max<std::size_t>(scale_bytes, 4));
        const ninfer::ops::MarkovHead head{form, codes.p, static_cast<const float*>(scales.p), v, r};
        ninfer::ops::dspark_markov_quantize(tp, head, nullptr);
        std::vector<std::uint8_t> hc(code_bytes);
        std::vector<float> hs(std::max<std::size_t>(scale_bytes, 4) / 4);
        cudaMemcpy(hc.data(), codes.p, code_bytes, cudaMemcpyDeviceToHost);
        cudaMemcpy(hs.data(), scales.p, hs.size() * 4, cudaMemcpyDeviceToHost);
        // Decode, checking each code against the host rounding.
        std::vector<double> p(static_cast<std::size_t>(r) * v);
        bool codes_exact = true;
        for (int g = 0; g < r / 32; ++g) {
            for (int i = 0; i < v; ++i) {
                float amax = 0.0F;
                for (int k = 32 * g; k < 32 * g + 32; ++k) { amax = std::max(amax, std::fabs(pt[static_cast<std::size_t>(k) * v + i])); }
                const float levels = form == MarkovForm::Q8 ? 127.0F : 7.0F, d = amax / levels, inv = d > 0.0F ? 1.0F / d : 0.0F;
                if (form != MarkovForm::BF16) { codes_exact &= hs[static_cast<std::size_t>(g) * v + i] == d; }
                for (int k = 32 * g; k < 32 * g + 32; ++k) {
                    const std::size_t at = static_cast<std::size_t>(k) * v + i;
                    if (form == MarkovForm::BF16) {
                        std::uint16_t word;
                        std::memcpy(&word, hc.data() + 2 * at, 2);
                        std::uint32_t bits;
                        std::memcpy(&bits, &pt[at], 4);
                        const std::uint32_t rounded = (bits + 0x7FFFU + ((bits >> 16) & 1U)) >> 16;
                        codes_exact &= word == rounded;
                        const std::uint32_t back = std::uint32_t(word) << 16;
                        float f;
                        std::memcpy(&f, &back, 4);
                        p[at] = f;
                    } else {
                        const float lim = levels, want = std::min(std::max(std::rint(pt[at] * inv), -lim), lim);
                        const int code = form == MarkovForm::Q8 ? static_cast<std::int8_t>(hc[at])
                                                                : static_cast<int>((hc[at / 2] >> (4 * (at % 2))) & 15U) - 8;
                        codes_exact &= code == static_cast<int>(want);
                        p[at] = double(d) * code;
                    }
                }
            }
        }
        check(codes_exact, std::string("dspark_markov_quantize ") + name + ": codes or scales differ from the rounding rule");
        ninfer::ops::dspark_markov_logits(tb, th, head, tlat, 30.0F, to, nullptr);
        Vec want(v), magnitude(v);
        for (int i = 0; i < v; ++i) {
            double acc = 0, a = std::fabs(base[i]) + std::fabs(hb[i]);
            for (int k = 0; k < r; ++k) {
                const double term = p[static_cast<std::size_t>(k) * v + i] * latent[k];
                acc += term;
                a += std::fabs(term);
            }
            want[i]      = 30.0 * std::tanh((double(base[i]) + hb[i] + acc) / 30.0);
            magnitude[i] = a;
        }
        compare(std::string("dspark_markov_logits ") + name, out.get(v), want, 1e-6, &magnitude);
    }
    // Refusals: an incomplete or overlapping descriptor never reaches a kernel.
    ninfer::DeviceBuffer codes(ninfer::ops::markov_head_code_bytes(MarkovForm::Q8, v, r) + 8);
    ninfer::DeviceBuffer scales(ninfer::ops::markov_head_scale_bytes(MarkovForm::Q8, v, r));
    const auto* sp = static_cast<const float*>(scales.p);
    const auto refused = [&](const ninfer::ops::MarkovHead& head, Tensor& logits, const char* what) {
        bool quantize = false, logits_refused = false;
        try {
            ninfer::ops::dspark_markov_quantize(tp, head, nullptr);
        } catch (const std::invalid_argument&) { quantize = true; }
        try {
            ninfer::ops::dspark_markov_logits(tb, th, head, tlat, 30.0F, logits, nullptr);
        } catch (const std::invalid_argument&) { logits_refused = true; }
        check(quantize && logits_refused, std::string("dspark_markov_quantize and dspark_markov_logits refuse ") + what);
    };
    refused({MarkovForm::Q8, nullptr, sp, v, r}, to, "null codes");
    refused({MarkovForm::Q8, codes.p, nullptr, v, r}, to, "null Q8 scales");
    refused({MarkovForm::Q4, codes.p, nullptr, v, r}, to, "null Q4 scales");
    refused({static_cast<MarkovForm>(7), codes.p, sp, v, r}, to, "an unknown form");
    refused({MarkovForm::Q8, static_cast<std::byte*>(codes.p) + 4, sp, v, r}, to, "misaligned codes");
    refused({MarkovForm::Q8, codes.p, sp, v, 2048}, to, "R past 1024");
    bool overlap = false;
    try {
        ninfer::ops::dspark_markov_quantize(tp, {MarkovForm::Q8, codes.p, static_cast<const float*>(codes.p), v, r}, nullptr);
    } catch (const std::invalid_argument&) { overlap = true; }
    check(overlap, "dspark_markov_quantize refuses scales overlapping the codes");
    Tensor onto_hb = dhb.tensor({v});
    bool aliased = false;
    try {
        ninfer::ops::dspark_markov_logits(tb, th, {MarkovForm::Q8, codes.p, sp, v, r}, tlat, 30.0F, onto_hb, nullptr);
    } catch (const std::invalid_argument&) { aliased = true; }
    check(aliased, "dspark_markov_logits refuses logits aliasing hb");
}

void test_argmax() {
    const int rows = 201024, valid = 200058, cols = 4;
    auto x = randoms(static_cast<std::size_t>(rows) * cols, -10, 10);
    // Column 0: a tie between 17 and 150000 (the lower wins). Column 1: the maximum in padding (ignored). Column 2:
    // NaNs everywhere but one value. Column 3: all equal (row 0).
    x[17] = x[150000] = 50.0F;
    x[rows + valid + 5] = 99.0F;
    for (int i = 0; i < rows; ++i) { x[2 * static_cast<std::size_t>(rows) + i] = std::numeric_limits<float>::quiet_NaN(); }
    x[2 * static_cast<std::size_t>(rows) + 123456] = -5.0F;
    for (int i = 0; i < rows; ++i) { x[3 * static_cast<std::size_t>(rows) + i] = 1.0F; }
    Buf dx(x), out(cols * 4);
    Tensor tx = dx.tensor({rows, cols}), to = out.tensor({cols}, DType::I32);
    ninfer::ops::argmax_fp32(tx, valid, to, nullptr);
    const auto got = t::from_device<std::int32_t>(out.b, cols);
    std::vector<std::int32_t> want(cols);
    for (int c = 0; c < cols; ++c) {
        int best = -1;
        for (int i = 0; i < valid; ++i) {
            const float v = x[static_cast<std::size_t>(c) * rows + i];
            if (!std::isnan(v) && (best < 0 || v > x[static_cast<std::size_t>(c) * rows + best])) { best = i; }
        }
        want[c] = best;
    }
    std::printf("argmax_fp32 got %d %d %d %d, want %d %d %d %d\n", got[0], got[1], got[2], got[3], want[0], want[1], want[2], want[3]);
    check(got == want && want[0] == 17 && want[2] == 123456 && want[3] == 0, "argmax_fp32");
    // No finite maximum: every value NaN (column 0), or an infinite one (column 1); column 2 is ordinary.
    for (int i = 0; i < rows; ++i) { x[i] = std::numeric_limits<float>::quiet_NaN(); }
    x[rows + 9] = std::numeric_limits<float>::infinity();
    Buf dy(x);
    Tensor ty = dy.tensor({rows, cols});
    ninfer::ops::argmax_fp32(ty, valid, to, nullptr, -1);
    const auto flagged = t::from_device<std::int32_t>(out.b, cols);
    ninfer::ops::argmax_fp32(ty, valid, to, nullptr);
    const auto defaulted = t::from_device<std::int32_t>(out.b, cols);
    std::printf("argmax_fp32 non-finite: got %d %d %d, default %d %d\n", flagged[0], flagged[1], flagged[2], defaulted[0], defaulted[1]);
    check(flagged[0] == -1 && flagged[1] == -1 && flagged[2] == 123456 && defaulted[0] == 0 && defaulted[1] == 0,
          "argmax_fp32 reports a column with no finite maximum");
}

// FP64 values on the device.
struct DeviceBufferD {
    ninfer::DeviceBuffer b;
    explicit DeviceBufferD(const std::vector<double>& v) : b(v.size() * 8) {
        if (cudaMemcpy(b.p, v.data(), v.size() * 8, cudaMemcpyHostToDevice) != cudaSuccess) { throw std::runtime_error("upload"); }
    }
    [[nodiscard]] const double* at(std::size_t j) const { return static_cast<const double*>(b.p) + j; }
};

// The host sampler's draw (Inkling2Core's sample(), without penalties) over the whole vocabulary, in FP64: the
// candidates by logit descending and id ascending, top_k, min_p, top_p, then the inverse CDF at u * kept mass.
std::pair<int, double> draw_oracle(const std::vector<float>& l, int valid, const ninfer::ops::DraftPick& p, double u) {
    std::vector<int> ids(static_cast<std::size_t>(valid));
    for (int i = 0; i < valid; ++i) { ids[static_cast<std::size_t>(i)] = i; }
    const auto better = [&](int a, int b) { return l[a] > l[b] || (l[a] == l[b] && a < b); };
    const auto k = static_cast<std::size_t>(std::clamp(p.top_k, 1, valid));
    std::partial_sort(ids.begin(), ids.begin() + static_cast<std::ptrdiff_t>(k), ids.end(), better);
    ids.resize(k);
    std::vector<double> w(k);
    double sum = 0;
    for (std::size_t i = 0; i < k; ++i) { sum += w[i] = std::exp((double(l[ids[i]]) - l[ids[0]]) / p.temperature); }
    for (double& x : w) { x /= sum; }
    std::size_t keep = k;
    if (p.min_p > 0) {
        while (keep > 1 && w[keep - 1] < p.min_p * w[0]) { --keep; }
    }
    if (p.top_p < 1) {
        double mass = 0;
        for (std::size_t i = 0; i < keep; ++i) {
            if ((mass += w[i]) >= p.top_p) {
                keep = i + 1;
                break;
            }
        }
    }
    double total = 0;
    for (std::size_t i = 0; i < keep; ++i) { total += w[i]; }
    double x = u * total;
    for (std::size_t i = 0; i < keep; ++i) {
        if ((x -= w[i]) < 0) { return {ids[i], w[i] / total}; }
    }
    return {ids[keep - 1], w[keep - 1] / total};
}

void test_draft_pick() {
    const int rows = 201024, valid = 200058;
    // A softcapped column: noise in [-10, 10], 60 tokens in a band near the top and one far ahead of them, and a
    // larger value in padding (ignored).
    auto x = randoms(rows, -10, 10);
    std::vector<int> band;
    for (int i = 0; i < 60; ++i) {
        const int id = 3 + 3331 * i;
        x[id]        = 14.0F + 0.11F * static_cast<float>((i * 37) % 60);
        band.push_back(id);
    }
    x[777] = 22.0F;
    x[valid + 3] = 29.0F;
    Buf dx(x), id(4), prob(4);
    std::vector<double> us;
    for (int i = 0; i < 48; ++i) { us.push_back((i + 0.37) / 48.0); }
    DeviceBufferD du(us);
    Tensor tx = dx.tensor({rows}), ti = id.tensor({1}, DType::I32), tp = prob.tensor({1});
    // Greedy: argmax_fp32's token and its softmax probability.
    ninfer::ops::dspark_draft_pick(tx, valid, {}, nullptr, ti, tp, nullptr);
    {
        double s = 0;
        for (int i = 0; i < valid; ++i) { s += std::exp(double(x[i]) - 22.0); }
        const int g = t::from_device<std::int32_t>(id.b, 1)[0];
        const float pr = t::from_device<float>(prob.b, 1)[0];
        std::printf("dspark_draft_pick greedy: %d p %.6f (want 777 p %.6f)\n", g, pr, 1.0 / s);
        check(g == 777 && std::abs(pr - 1.0 / s) < 1e-6, "dspark_draft_pick greedy");
    }
    const ninfer::ops::DraftPick picks[] = {{1.0F, 20, 1.0F, 0.0F}, {0.7F, 40, 0.9F, 0.0F}, {2.0F, 50, 1.0F, 0.05F}, {6.0F, 30, 0.95F, 0.02F}, {1.0F, 1000, 1.0F, 0.0F}};
    for (const auto& p : picks) {
        int same = 0, prob_ok = 0, varied = 0, last = -1;
        for (std::size_t j = 0; j < us.size(); ++j) {
            ninfer::ops::dspark_draft_pick(tx, valid, p, du.at(j), ti, tp, nullptr);
            const int g = t::from_device<std::int32_t>(id.b, 1)[0];
            const float pr = t::from_device<float>(prob.b, 1)[0];
            const auto [want, wp] = draw_oracle(x, valid, p, us[j]);
            if ((g != want || std::abs(pr - wp) >= 1e-5) && same + 3 > static_cast<int>(j)) {
                std::printf("  u %.4f: got %d p %.6f, want %d p %.6f\n", us[j], g, pr, want, wp);
            }
            same += g == want;
            prob_ok += std::abs(pr - wp) < 1e-5;
            varied += g != last;
            last = g;
        }
        std::printf("dspark_draft_pick t %.1f top-k %d top-p %.2f min-p %.2f: %d/%zu draws and %d probabilities as the host's, %d changes\n",
                    p.temperature, p.top_k, p.top_p, p.min_p, same, us.size(), prob_ok, varied);
        check(same == static_cast<int>(us.size()) && prob_ok == same && varied > 1, "dspark_draft_pick sampled draws the host sampler's tokens");
    }
}

} // namespace

int main() {
    try {
        test_projection(true, 3072, 2304, 8, false, 1e-7);
        test_projection(true, 2304, 9216, 8, false, 1e-7);
        test_projection(true, 1000, 64, 3, true, 1e-7); // rows not a multiple of the block's 8
        test_projection(false, 768, 5120, 1, true, 1e-7);
        test_projection(false, 2304, 256, 20, true, 1e-7); // 3 column tiles
        test_projection(true, 3072, 2304, 8, false, 1e-7, true);
        test_projection(true, 2304, 9216, 8, false, 1e-7, true);
        test_projection(true, 1000, 128, 3, true, 1e-7, true);  // rows not a multiple of the block's 32
        test_projection(true, 768, 5120, 20, true, 1e-7, true); // 3 column tiles
        test_head_norm_rope();
        test_silu();
        test_fuse_taps();
        test_attention(0, 1024, 1024, "anchor 0 (block only)");
        test_attention(700, 1024, 1024, "local, short context");
        test_attention(5000, 1024, 1024, "local, wrapped ring");
        test_attention(3000, 4096, 0, "global");
        test_attention(24, 1024, 0, "global, 32 keys (one run, no combine)");
        test_attention(33, 1024, 0, "global, 41 keys (two runs, a partial tile)");
        test_attention(3000, 4096, 0, "global, W 16", 16);
        // Past 23,040 keys, the most the single-block form held in shared memory.
        test_attention(30000, 30720, 0, "global, 30,008 keys", 16);
        test_attention(5000, 1024, 1024, "FP16 rings, local, wrapped ring", 16, true);
        test_attention(30000, 30720, 0, "FP16 rings, global, 30,008 keys", 16, true);
        test_markov();
        test_markov_compact();
        test_argmax();
        test_draft_pick();
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL: %s\n", error.what());
        return 1;
    }
    std::printf(failures ? "FAILED (%d)\n" : "PASS\n", failures);
    return failures ? 1 : 0;
}
