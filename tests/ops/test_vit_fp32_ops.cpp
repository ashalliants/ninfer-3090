// Written for NInfer-3090.
//
// The FP32 Ops of Inkling-2-Mini's vision tower (include/ninfer/ops/vit_fp32.h) against FP64 oracles written here
// from the header's formulas, at the tower's shapes (D 1024, 16 heads of 64, FFN 4096, 20 x 20 patches, a 32 x 32
// position grid) plus edge cases (one token, partial tiles, a grid resized down and up, an unresized grid).
//
// Criterion: every output within tol * (1 + |ref|) of the oracle, finite, with tol 2e-6; a linear within
// 1e-6 * (1 + sum |terms|) (its random terms cancel, so |ref| says little), and each of its columns bit-exact
// whatever T; patchify and merge exact.

#include "ninfer/ops/vit_fp32.h"

#include "ops/op_tester.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <tuple>
#include <string>
#include <vector>

namespace {

namespace t = ninfer::test;
using ninfer::DType;
using ninfer::QType;
using ninfer::Tensor;
using Vec = std::vector<double>;

int failures = 0;

void check(bool ok, const std::string& what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what.c_str());
        ++failures;
    }
}

std::uint32_t seed_state = 4242;
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
    Tensor tensor(std::initializer_list<std::int32_t> ne) { return Tensor(b.p, DType::FP32, ne); }
    std::vector<float> get(std::size_t n) const { return t::from_device<float>(b, n); }
};

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
    std::printf("%-52s max scaled error %.2e (index %zu: %.9g vs %.9g)\n", label.c_str(), worst, at, double(got[at]), want[at]);
    check(worst <= tol, label + ": outside tolerance");
}

void exact(const std::string& label, const std::vector<float>& got, const std::vector<float>& want) {
    const bool same = got.size() == want.size() && std::memcmp(got.data(), want.data(), got.size() * 4) == 0;
    std::printf("%-52s %s\n", label.c_str(), same ? "exact" : "DIFFERS");
    check(same, label + ": not exact");
}

double gelu(double x) { return 0.5 * x * (1 + std::erf(x / std::sqrt(2.0))); }

// ------------------------------------------------------------------------------------------------------ tests

void test_linear(bool bf16, int n, int k, int cols, bool bias) {
    const std::string label = std::string("vit_linear_fp32 ") + (bf16 ? "BF16" : "FP32") + " [" + std::to_string(n) + ", " +
                              std::to_string(k) + "] T " + std::to_string(cols) + (bias ? " + bias" : "");
    std::vector<double> wv(static_cast<std::size_t>(n) * k);
    std::vector<std::uint8_t> wb(wv.size() * (bf16 ? 2 : 4));
    for (std::size_t i = 0; i < wv.size(); ++i) {
        float f = uniform(-0.05F, 0.05F);
        std::uint32_t u;
        std::memcpy(&u, &f, 4);
        if (bf16) {
            const auto word = static_cast<std::uint16_t>(u >> 16);
            u               = static_cast<std::uint32_t>(word) << 16;
            std::memcpy(wb.data() + 2 * i, &word, 2);
            std::memcpy(&f, &u, 4);
        } else {
            std::memcpy(wb.data() + 4 * i, &f, 4);
        }
        wv[i] = f;
    }
    ninfer::DeviceBuffer dw(wb.size());
    check(cudaMemcpy(dw.p, wb.data(), wb.size(), cudaMemcpyHostToDevice) == cudaSuccess, label + ": upload");
    const auto x = randoms(static_cast<std::size_t>(k) * cols, -2, 2);
    const auto b = randoms(n);
    Buf dx(x), db(b), out(static_cast<std::size_t>(n) * cols * 4);
    Tensor tx = dx.tensor({k, cols}), tb = db.tensor({n}), to = out.tensor({n, cols});
    ninfer::ops::vit_linear_fp32(tx, dw.p, bf16 ? QType::BF16 : QType::FP32, n, bias ? &tb : nullptr, to, nullptr);
    Vec want(static_cast<std::size_t>(n) * cols), magnitude(want.size());
    for (int c = 0; c < cols; ++c) {
        for (int r = 0; r < n; ++r) {
            double s = bias ? b[r] : 0.0, m = bias ? std::fabs(b[r]) : 0.0;
            for (int i = 0; i < k; ++i) {
                const double term = wv[static_cast<std::size_t>(r) * k + i] * x[static_cast<std::size_t>(c) * k + i];
                s += term, m += std::fabs(term);
            }
            want[static_cast<std::size_t>(c) * n + r] = s, magnitude[static_cast<std::size_t>(c) * n + r] = m;
        }
    }
    const auto got = out.get(want.size());
    compare(label, got, want, 1e-6, &magnitude);
    // Column c alone (T = 1) gives the same bits.
    for (int c : {0, cols - 1}) {
        Buf one(static_cast<std::size_t>(n) * 4);
        Tensor xc = Tensor(static_cast<float*>(dx.b.p) + static_cast<std::size_t>(c) * k, DType::FP32, {k, 1});
        Tensor oc = one.tensor({n, 1});
        ninfer::ops::vit_linear_fp32(xc, dw.p, bf16 ? QType::BF16 : QType::FP32, n, bias ? &tb : nullptr, oc, nullptr);
        exact(label + " column " + std::to_string(c) + " alone", one.get(n),
              std::vector<float>(got.begin() + static_cast<std::ptrdiff_t>(c) * n, got.begin() + static_cast<std::ptrdiff_t>(c + 1) * n));
    }
}

void test_layer_norm(int h, int cols, bool in_place) {
    const auto x = randoms(static_cast<std::size_t>(h) * cols, -3, 5), w = randoms(h, 0.5F, 1.5F), b = randoms(h);
    Buf dx(x), dw(w), db(b), out(x.size() * 4);
    Tensor tx = dx.tensor({h, cols}), tw = dw.tensor({h}), tb = db.tensor({h}), to = in_place ? tx : out.tensor({h, cols});
    ninfer::ops::vit_layer_norm_fp32(tx, tw, tb, 1e-5F, to, nullptr);
    Vec want(x.size());
    for (int c = 0; c < cols; ++c) {
        double mean = 0, var = 0;
        for (int i = 0; i < h; ++i) { mean += x[static_cast<std::size_t>(c) * h + i]; }
        mean /= h;
        for (int i = 0; i < h; ++i) { var += std::pow(x[static_cast<std::size_t>(c) * h + i] - mean, 2); }
        const double r = 1.0 / std::sqrt(var / h + double(1e-5F));
        for (int i = 0; i < h; ++i) {
            want[static_cast<std::size_t>(c) * h + i] = (x[static_cast<std::size_t>(c) * h + i] - mean) * r * w[i] + b[i];
        }
    }
    compare("vit_layer_norm_fp32 [" + std::to_string(h) + ", " + std::to_string(cols) + "]" + (in_place ? " in place" : ""),
            in_place ? dx.get(x.size()) : out.get(x.size()), want, 2e-6);
}

void test_gelu_and_residual() {
    const int h = 4096, cols = 9;
    const auto x = randoms(static_cast<std::size_t>(h) * cols, -6, 6);
    Buf dx(x);
    Tensor tx = dx.tensor({h, cols});
    ninfer::ops::vit_gelu_fp32(tx, nullptr);
    Vec want(x.size());
    for (std::size_t i = 0; i < x.size(); ++i) { want[i] = gelu(x[i]); }
    compare("vit_gelu_fp32 [4096, 9]", dx.get(x.size()), want, 2e-6);

    const auto hv = randoms(static_cast<std::size_t>(1024) * cols), xv = randoms(hv.size(), -3, 3), s = randoms(1024, -0.2F, 0.2F);
    Buf dh(hv), dxv(xv), ds(s);
    Tensor th = dh.tensor({1024, cols});
    ninfer::ops::vit_scaled_residual_fp32(th, dxv.tensor({1024, cols}), ds.tensor({1024}), nullptr);
    Vec r(hv.size());
    for (std::size_t i = 0; i < hv.size(); ++i) { r[i] = hv[i] + double(s[i % 1024]) * xv[i]; }
    compare("vit_scaled_residual_fp32 [1024, 9]", dh.get(hv.size()), r, 1e-6);
}

void test_patchify() {
    const int p = 20, gw = 4, gh = 3, w = gw * p, h = gh * p;
    const auto pixels = randoms(static_cast<std::size_t>(w) * h * 3);
    Buf dp(pixels), out(pixels.size() * 4);
    Tensor to = out.tensor({3 * p * p, gw * gh});
    ninfer::ops::vit_patchify_fp32(Tensor(dp.b.p, DType::FP32, {w, h, 3}), p, to, nullptr);
    std::vector<float> want(pixels.size());
    for (int py = 0; py < gh; ++py) {
        for (int px = 0; px < gw; ++px) {
            for (int c = 0; c < 3; ++c) {
                for (int ky = 0; ky < p; ++ky) {
                    for (int kx = 0; kx < p; ++kx) {
                        want[static_cast<std::size_t>(py * gw + px) * 3 * p * p + c * p * p + ky * p + kx] =
                            pixels[(static_cast<std::size_t>(c) * h + py * p + ky) * w + px * p + kx];
                    }
                }
            }
        }
    }
    exact("vit_patchify_fp32 80 x 60, P 20", out.get(pixels.size()), want);
}

void test_embed(int gw, int gh) {
    const int d = 1024, g = 32, n = gw * gh;
    const auto patches = randoms(static_cast<std::size_t>(d) * n), table = randoms(static_cast<std::size_t>(d) * g * g);
    const auto cls = randoms(d), cls_pos = randoms(d);
    Buf dp(patches), dt(table), dc(cls), dcp(cls_pos), out(static_cast<std::size_t>(d) * (n + 1) * 4);
    Tensor to = out.tensor({d, n + 1});
    ninfer::ops::vit_embed_fp32(dp.tensor({d, n}), dt.tensor({d, g * g}), g, dc.tensor({d}), dcp.tensor({d}), gw, gh, to, nullptr);
    const auto lerp = [](int i, int dst, int src) {
        const double u = (i + 0.5) * src / dst - 0.5;
        const int f    = static_cast<int>(std::floor(u));
        const int i0 = std::clamp(f, 0, src - 1), i1 = std::clamp(f + 1, 0, src - 1);
        return std::tuple{i0, i1, std::clamp(u - i0, 0.0, 1.0)};
    };
    Vec want(static_cast<std::size_t>(d) * (n + 1));
    for (int i = 0; i < d; ++i) { want[i] = double(cls[i]) + cls_pos[i]; }
    for (int py = 0; py < gh; ++py) {
        for (int px = 0; px < gw; ++px) {
            const auto [x0, x1, wx] = lerp(px, gw, g);
            const auto [y0, y1, wy] = lerp(py, gh, g);
            for (int i = 0; i < d; ++i) {
                const auto at   = [&](int x, int y) { return double(table[static_cast<std::size_t>(y * g + x) * d + i]); };
                const double pos = at(x0, y0) * (1 - wx) * (1 - wy) + at(x1, y0) * wx * (1 - wy) + at(x0, y1) * (1 - wx) * wy +
                                   at(x1, y1) * wx * wy;
                want[static_cast<std::size_t>(1 + py * gw + px) * d + i] = patches[static_cast<std::size_t>(py * gw + px) * d + i] + pos;
            }
        }
    }
    compare("vit_embed_fp32 grid 32 -> " + std::to_string(gw) + " x " + std::to_string(gh), out.get(want.size()), want, 1e-6);
}

void test_rope_2d() {
    const int d = 64, heads = 3, gw = 5, gh = 4, cols = 1 + gw * gh;
    const auto x = randoms(static_cast<std::size_t>(d) * heads * cols, -2, 2);
    Buf dx(x);
    Tensor tx = dx.tensor({d * heads, cols});
    ninfer::ops::vit_rope_2d_fp32(tx, d, gw, 10000.0F, nullptr);
    Vec want(x.size());
    for (int c = 0; c < cols; ++c) {
        const int px = c == 0 ? 0 : (c - 1) % gw + 1, py = c == 0 ? 0 : (c - 1) / gw + 1;
        for (int h = 0; h < heads; ++h) {
            for (int half = 0; half < 2; ++half) {
                for (int i = 0; i < d / 4; ++i) {
                    const std::size_t at = (static_cast<std::size_t>(c) * heads + h) * d + half * (d / 2) + 2 * i;
                    const double angle   = (half ? py : px) * std::pow(10000.0, -2.0 * i / (d / 2));
                    want[at]             = x[at] * std::cos(angle) - x[at + 1] * std::sin(angle);
                    want[at + 1]         = x[at] * std::sin(angle) + x[at + 1] * std::cos(angle);
                }
            }
        }
    }
    compare("vit_rope_2d_fp32 D 64, 3 heads, 5 x 4", dx.get(x.size()), want, 2e-6);
}

void test_attention(int heads, int cols) {
    const int d = 64, rows = d * heads;
    const auto q = randoms(static_cast<std::size_t>(rows) * cols, -2, 2), k = randoms(q.size(), -2, 2), v = randoms(q.size(), -2, 2);
    Buf dq(q), dk(k), dv(v), out(q.size() * 4);
    Tensor to = out.tensor({rows, cols});
    ninfer::ops::vit_attention_fp32(dq.tensor({rows, cols}), dk.tensor({rows, cols}), dv.tensor({rows, cols}), d, to, nullptr);
    Vec want(q.size());
    std::vector<double> s(cols);
    for (int h = 0; h < heads; ++h) {
        for (int t0 = 0; t0 < cols; ++t0) {
            double m = -1e300;
            for (int j = 0; j < cols; ++j) {
                double acc = 0;
                for (int i = 0; i < d; ++i) {
                    acc += double(q[static_cast<std::size_t>(t0) * rows + h * d + i]) * k[static_cast<std::size_t>(j) * rows + h * d + i];
                }
                s[j] = acc / 8.0, m = std::max(m, s[j]);
            }
            double l = 0;
            for (int j = 0; j < cols; ++j) { l += s[j] = std::exp(s[j] - m); }
            for (int i = 0; i < d; ++i) {
                double acc = 0;
                for (int j = 0; j < cols; ++j) { acc += s[j] * v[static_cast<std::size_t>(j) * rows + h * d + i]; }
                want[static_cast<std::size_t>(t0) * rows + h * d + i] = acc / l;
            }
        }
    }
    compare("vit_attention_fp32 " + std::to_string(heads) + " heads, T " + std::to_string(cols), out.get(want.size()), want, 2e-6);
}

void test_merge() {
    const int d = 48, gw = 6, gh = 4, n = gw * gh;
    const auto h = randoms(static_cast<std::size_t>(d) * (n + 1));
    Buf dh(h), out(static_cast<std::size_t>(d) * n * 4);
    Tensor to = out.tensor({4 * d, n / 4});
    ninfer::ops::vit_merge_2x2_fp32(dh.tensor({d, n + 1}), gw, gh, to, nullptr);
    std::vector<float> want(static_cast<std::size_t>(d) * n);
    for (int Y = 0; Y < gh / 2; ++Y) {
        for (int X = 0; X < gw / 2; ++X) {
            for (int dy = 0; dy < 2; ++dy) {
                for (int dx = 0; dx < 2; ++dx) {
                    for (int e = 0; e < d; ++e) {
                        want[static_cast<std::size_t>(Y * (gw / 2) + X) * 4 * d + e + d * dx + 2 * d * dy] =
                            h[static_cast<std::size_t>(1 + (2 * Y + dy) * gw + 2 * X + dx) * d + e];
                    }
                }
            }
        }
    }
    exact("vit_merge_2x2_fp32 D 48, 6 x 4", out.get(want.size()), want);
}

} // namespace

int main() {
    try {
        test_linear(true, 1024, 1024, 70, true);
        test_linear(true, 512, 4096, 65, true);
        test_linear(true, 2304, 4096, 3, false);
        test_linear(false, 1024, 1200, 37, false);
        test_linear(false, 70, 50, 1, true);
        test_layer_norm(1024, 33, false);
        test_layer_norm(1024, 5, true);
        test_gelu_and_residual();
        test_patchify();
        test_embed(26, 13);
        test_embed(40, 36);
        test_embed(32, 32);
        test_rope_2d();
        test_attention(2, 1);
        test_attention(2, 33);
        test_attention(3, 257);
        test_merge();
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL: %s\n", error.what());
        return 1;
    }
    std::printf(failures ? "FAILED (%d)\n" : "PASS\n", failures);
    return failures ? 1 : 0;
}
