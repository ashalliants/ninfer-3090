// The PLE Ops (include/ninfer/ops/ple.h) against independent oracles. New for NInfer-3090:
// Infernix a3edb450 ships the Ops (src/ops/ple/ple.cu) but no standalone test of them.
//
// ple_embed is exact: each value is the exact FP32 decode of the IQ4_NL row by the test-owned
// decoder (ops/ggml_blocks_decode.h, qualified bit-for-bit against ggml b11316) rounded once to
// BF16, compared bit for bit. ple_gate and ple_conv_inject are checked against the formula in FP64
// from their represented inputs; each rounds its outputs once to BF16 from FP32 arithmetic, so the
// criterion is |got - ref| <= 2^-8 |ref| + 2^-12 rms(ref) with non-finite values failing. Every
// state write is an exact copy, so states compare bit for bit, including untouched slots.
//
// Geometry is Flash-Next's PLE layer: 16 rows of 90 bytes (5 IQ4_NL blocks) per column, S = 4
// streams of H = 2560, a K = 4 tap convolution at dilation 3 (span 9). Columns cover decode (1),
// a verification width (5), widths past the span (12) and a prefill chunk (64); two sequences per
// call; split calls against one long call (bitwise); ple_conv_commit against the in-place update.

#include "ninfer/ops/ple.h"

#include "ops/ggml_blocks_decode.h"
#include "ops/op_tester.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

namespace t = ninfer::test;
using ninfer::DType;
using ninfer::QType;
using ninfer::Tensor;

constexpr int kS = 4, kH = 2560, kC = kS * kH;
constexpr int kHeads = 16, kRowBytes = 90, kEmbed = kHeads * 160;
constexpr int kTaps = 4, kDilation = 3, kSpan = (kTaps - 1) * kDilation;
constexpr int kSlots = 3;
constexpr float kEps = 1e-6F;

int failures = 0;

void check(bool ok, const std::string& what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what.c_str());
        ++failures;
    }
}

template <class T> struct Buffer {
    void* p = nullptr;
    std::size_t count;
    explicit Buffer(const std::vector<T>& host) : count(host.size()) {
        t::cuda_check(cudaMalloc(&p, std::max<std::size_t>(1, count) * sizeof(T)), "cudaMalloc");
        t::cuda_check(cudaMemcpy(p, host.data(), count * sizeof(T), cudaMemcpyHostToDevice), "upload");
    }
    ~Buffer() { cudaFree(p); }
    Buffer(const Buffer&)            = delete;
    Buffer& operator=(const Buffer&) = delete;
    std::vector<T> read() const {
        std::vector<T> host(count);
        t::cuda_check(cudaMemcpy(host.data(), p, count * sizeof(T), cudaMemcpyDeviceToHost), "download");
        return host;
    }
};

std::vector<std::uint16_t> bf16_values(std::size_t n, std::mt19937& rng, float scale) {
    std::uniform_real_distribution<float> u(-1.0F, 1.0F);
    std::vector<std::uint16_t> out(n);
    for (auto& v : out) { v = t::f32_to_bf16(u(rng) * scale); }
    return out;
}

// The stored norm multiplier: FP32 1 + gamma with gamma a BF16 value.
std::vector<float> norm_values(std::size_t n, std::mt19937& rng) {
    std::uniform_real_distribution<float> u(-0.25F, 0.25F);
    std::vector<float> out(n);
    for (auto& v : out) { v = 1.0F + t::bf16_to_f32(t::f32_to_bf16(u(rng))); }
    return out;
}

float half_value(std::uint16_t h) {
    const int sign = (h >> 15) & 1, exponent = (h >> 10) & 31, mantissa = h & 1023;
    double value   = exponent == 0 ? std::ldexp(static_cast<double>(mantissa), -24)
                                   : std::ldexp(static_cast<double>(mantissa + 1024), exponent - 25);
    return static_cast<float>(sign ? -value : value);
}

double rms(const std::vector<double>& v) {
    double ss = 0;
    for (const double x : v) { ss += x * x; }
    return std::sqrt(ss / static_cast<double>(v.size()));
}

// The suite's one rounded-output criterion.
void pointwise(const std::vector<std::uint16_t>& got, const std::vector<double>& ref,
               const std::string& label) {
    const double floor = std::ldexp(rms(ref), -12);
    double worst       = 0;
    std::size_t bad    = 0;
    for (std::size_t i = 0; i < ref.size(); ++i) {
        const double v     = t::bf16_to_f32(got[i]);
        const double bound = std::ldexp(std::fabs(ref[i]), -8) + floor;
        if (!std::isfinite(v) || std::fabs(v - ref[i]) > bound) { ++bad; }
        if (std::isfinite(v)) { worst = std::max(worst, std::fabs(v - ref[i]) / bound); }
    }
    std::printf("%s: worst %.3f of the bound\n", label.c_str(), worst);
    check(bad == 0, label + ": " + std::to_string(bad) + " outputs outside the bound");
}

// --- ple_embed -------------------------------------------------------------------------------------

void embed_case(int T, cudaStream_t stream) {
    std::mt19937 rng(17U + static_cast<unsigned>(T));
    std::vector<std::uint8_t> rows(static_cast<std::size_t>(kRowBytes) * kHeads * T);
    for (auto& b : rows) { b = static_cast<std::uint8_t>(rng()); }
    // Finite FP16 scales of either sign over normal and subnormal exponents.
    for (std::size_t block = 0; block < rows.size() / 18; ++block) {
        const auto exponent = static_cast<unsigned>(rng() % 20); // biased 0..19: subnormals included
        const auto h = static_cast<std::uint16_t>(((rng() & 1U) << 15) | (exponent << 10) | (rng() & 0x3FFU));
        std::memcpy(&rows[block * 18], &h, 2);
    }
    Buffer<std::uint8_t> d_rows(rows);
    Buffer<std::uint16_t> d_out(std::vector<std::uint16_t>(static_cast<std::size_t>(kEmbed) * T, 0xFFFF));
    const Tensor in(d_rows.p, DType::U8, {kRowBytes, kHeads, T});
    Tensor out(d_out.p, DType::BF16, {kEmbed, T});
    ninfer::ops::ple_embed(in, QType::GGML_IQ4_NL, out, stream);
    t::cuda_check(cudaStreamSynchronize(stream), "embed");
    const auto values = t::ggml::decode_blocks(
        QType::GGML_IQ4_NL, std::as_bytes(std::span<const std::uint8_t>(rows)));
    std::vector<std::uint16_t> want(values.size());
    for (std::size_t i = 0; i < values.size(); ++i) { want[i] = t::f32_to_bf16(values[i]); }
    check(d_out.read() == want, "ple_embed T=" + std::to_string(T) + " equals the exact decode");
}

void embed_refusals(cudaStream_t stream) {
    Buffer<std::uint8_t> d_rows(std::vector<std::uint8_t>(kRowBytes * kHeads, 0));
    Buffer<std::uint16_t> d_out(std::vector<std::uint16_t>(kEmbed, 0));
    Tensor out(d_out.p, DType::BF16, {kEmbed, 1});
    bool threw = false;
    try {
        ninfer::ops::ple_embed(Tensor(d_rows.p, DType::U8, {kRowBytes, kHeads, 1}),
                               QType::GGML_Q8_0, out, stream);
    } catch (const std::invalid_argument&) { threw = true; }
    check(threw, "ple_embed refuses a format other than IQ4_NL");
    threw = false;
    try {
        ninfer::ops::ple_embed(Tensor(d_rows.p, DType::U8, {89, kHeads, 1}), QType::GGML_IQ4_NL,
                               out, stream);
    } catch (const std::invalid_argument&) { threw = true; }
    check(threw, "ple_embed refuses rows that are not whole blocks");
}

// --- ple_gate ----------------------------------------------------------------------------------------

void gate_case(int T, cudaStream_t stream) {
    std::mt19937 rng(101U + static_cast<unsigned>(T));
    auto key   = bf16_values(static_cast<std::size_t>(kC) * T, rng, 2.0F);
    auto value = bf16_values(static_cast<std::size_t>(kH) * T, rng, 1.0F);
    std::vector<std::uint16_t> residual(static_cast<std::size_t>(kC) * T);
    {
        std::uniform_real_distribution<float> u(-1.0F, 1.0F);
        for (std::size_t i = 0; i < residual.size(); ++i) {
            residual[i] = t::f32_to_bf16(u(rng) * static_cast<float>(1 << ((i % kC) / kH)));
        }
    }
    // Make the keys follow the residual for half of the streams, so g takes both signs and sizes.
    for (int c = 0; c < T; ++c) {
        for (int s = 0; s < kS; s += 2) {
            for (int d = 0; d < kH; ++d) {
                const std::size_t i = static_cast<std::size_t>(c) * kC + s * kH + d;
                key[i] = t::f32_to_bf16((c % 2 ? -1.0F : 1.0F) * t::bf16_to_f32(residual[i]));
            }
        }
    }
    const auto kn = norm_values(kC, rng), qn = norm_values(kC, rng), cn = norm_values(kC, rng);
    Buffer<std::uint16_t> d_key(key), d_value(value), d_res(residual);
    Buffer<float> d_kn(kn), d_qn(qn), d_cn(cn);
    Buffer<std::uint16_t> d_gated(std::vector<std::uint16_t>(key.size(), 0)),
        d_norm(std::vector<std::uint16_t>(key.size(), 0));
    Tensor gated(d_gated.p, DType::BF16, {kC, T}), normalized(d_norm.p, DType::BF16, {kC, T});
    ninfer::ops::ple_gate(Tensor(d_key.p, DType::BF16, {kC, T}), Tensor(d_value.p, DType::BF16, {kH, T}),
                          Tensor(d_res.p, DType::BF16, {kC, T}), Tensor(d_kn.p, DType::FP32, {kC}),
                          Tensor(d_qn.p, DType::FP32, {kC}), Tensor(d_cn.p, DType::FP32, {kC}), kS,
                          kEps, gated, normalized, stream);
    t::cuda_check(cudaStreamSynchronize(stream), "gate");
    std::vector<double> ref_gated(key.size()), ref_norm(key.size());
    for (int c = 0; c < T; ++c) {
        for (int s = 0; s < kS; ++s) {
            const std::size_t base = static_cast<std::size_t>(c) * kC + s * kH;
            double kss = 0, qss = 0;
            for (int d = 0; d < kH; ++d) {
                const double k = t::bf16_to_f32(key[base + d]), q = t::bf16_to_f32(residual[base + d]);
                kss += k * k;
                qss += q * q;
            }
            const double kinv = 1.0 / std::sqrt(kss / kH + kEps), qinv = 1.0 / std::sqrt(qss / kH + kEps);
            double dot = 0;
            for (int d = 0; d < kH; ++d) {
                const std::size_t ch = static_cast<std::size_t>(s) * kH + d;
                dot += t::bf16_to_f32(key[base + d]) * kinv * kn[ch] *
                       (t::bf16_to_f32(residual[base + d]) * qinv * qn[ch]);
            }
            double g          = dot / std::sqrt(static_cast<double>(kH));
            g                 = std::copysign(std::sqrt(std::max(std::fabs(g), 1e-6)), g);
            const double gate = 1.0 / (1.0 + std::exp(-g));
            double gss        = 0;
            for (int d = 0; d < kH; ++d) {
                const double v = gate * t::bf16_to_f32(value[static_cast<std::size_t>(c) * kH + d]);
                ref_gated[base + d] = v;
                gss += v * v;
            }
            const double ginv = 1.0 / std::sqrt(gss / kH + kEps);
            for (int d = 0; d < kH; ++d) {
                ref_norm[base + d] = ref_gated[base + d] * ginv * cn[static_cast<std::size_t>(s) * kH + d];
            }
        }
    }
    pointwise(d_gated.read(), ref_gated, "ple_gate gated T=" + std::to_string(T));
    pointwise(d_norm.read(), ref_norm, "ple_gate normalized T=" + std::to_string(T));
}

// --- ple_conv_inject and ple_conv_commit -------------------------------------------------------------

struct ConvInputs {
    std::vector<std::uint16_t> gated, normalized, residual, states;
    std::vector<float> weight; // FP32 [K, C]: channel c's taps at c*K + j (FP16 values, as the
                               // converter widens the GGUF's F16 taps)
};

ConvInputs conv_inputs(int T, std::uint32_t seed) {
    std::mt19937 rng(seed);
    ConvInputs in;
    in.gated      = bf16_values(static_cast<std::size_t>(kC) * T, rng, 1.0F);
    in.normalized = bf16_values(static_cast<std::size_t>(kC) * T, rng, 2.0F);
    in.residual   = bf16_values(static_cast<std::size_t>(kC) * T, rng, 8.0F);
    in.states     = bf16_values(static_cast<std::size_t>(kC) * kSpan * kSlots, rng, 2.0F);
    std::uniform_real_distribution<float> u(-0.6F, 0.6F);
    in.weight.resize(static_cast<std::size_t>(kC) * kTaps);
    for (auto& w : in.weight) {
        // FP16 bits of a value of magnitude below 1 (round toward zero is fine: it is the stored value).
        const float v  = u(rng);
        const int e    = v == 0.0F ? -25 : static_cast<int>(std::floor(std::log2(std::fabs(v))));
        const int m    = static_cast<int>(std::ldexp(std::fabs(v), 10 - e)) - 1024;
        w = half_value(e < -14 ? static_cast<std::uint16_t>(v < 0 ? 0x8000 : 0)
                               : static_cast<std::uint16_t>((v < 0 ? 0x8000 : 0) | ((e + 15) << 10) |
                                                            (m & 0x3FF)));
    }
    return in;
}

struct ConvResult {
    std::vector<std::uint16_t> residual, states;
};

ConvResult run_conv(const ConvInputs& in, int T, const std::vector<std::int32_t>& source,
                    const std::vector<std::int32_t>& destination, cudaStream_t stream,
                    const std::vector<std::uint16_t>* states_override = nullptr,
                    const std::vector<std::uint16_t>* residual_override = nullptr,
                    int first_column = 0) {
    const int sequences = static_cast<int>(source.size());
    const auto slice = [&](const std::vector<std::uint16_t>& v) {
        return std::vector<std::uint16_t>(v.begin() + static_cast<std::ptrdiff_t>(first_column) * kC,
                                          v.begin() + static_cast<std::ptrdiff_t>(first_column + T) * kC);
    };
    Buffer<std::uint16_t> d_gated(slice(in.gated)), d_norm(slice(in.normalized)),
        d_res(residual_override ? slice(*residual_override) : slice(in.residual)),
        d_states(states_override ? *states_override : in.states);
    Buffer<float> d_weight(in.weight);
    Buffer<std::int32_t> d_src(source), d_dst(destination.empty() ? std::vector<std::int32_t>{0} : destination);
    Tensor residual(d_res.p, DType::BF16, {kC, T}), states(d_states.p, DType::BF16, {kC, kSpan, kSlots});
    const Tensor dst = destination.empty() ? Tensor{} : Tensor(d_dst.p, DType::I32, {sequences});
    ninfer::ops::ple_conv_inject(Tensor(d_gated.p, DType::BF16, {kC, T}),
                                 Tensor(d_norm.p, DType::BF16, {kC, T}),
                                 Tensor(d_weight.p, DType::FP32, {kTaps, kC}), kDilation, states,
                                 Tensor(d_src.p, DType::I32, {sequences}), dst, residual, stream);
    t::cuda_check(cudaStreamSynchronize(stream), "conv");
    return {d_res.read(), d_states.read()};
}

// FP64 output and exact new state of the formula.
void conv_oracle(const ConvInputs& in, int T, const std::vector<std::int32_t>& source,
                 const std::vector<std::int32_t>& destination, std::vector<double>& residual,
                 std::vector<std::uint16_t>& states) {
    const int sequences = static_cast<int>(source.size()), width = T / sequences;
    residual.assign(static_cast<std::size_t>(kC) * T, 0.0);
    states = in.states;
    for (int i = 0; i < sequences; ++i) {
        const auto src_value = [&](int c, int p) -> std::uint16_t {
            return p >= 0 ? in.normalized[static_cast<std::size_t>(i * width + p) * kC + c]
                          : in.states[(static_cast<std::size_t>(source[i]) * kSpan + (kSpan + p)) * kC + c];
        };
        for (int u = 0; u < width; ++u) {
            for (int c = 0; c < kC; ++c) {
                double a = 0;
                for (int j = 0; j < kTaps; ++j) {
                    a += static_cast<double>(in.weight[static_cast<std::size_t>(c) * kTaps + j]) *
                         t::bf16_to_f32(src_value(c, u - (kTaps - 1 - j) * kDilation));
                }
                const std::size_t at = static_cast<std::size_t>(i * width + u) * kC + c;
                residual[at] = t::bf16_to_f32(in.residual[at]) + t::bf16_to_f32(in.gated[at]) +
                               a / (1.0 + std::exp(-a));
            }
        }
        if (!destination.empty()) {
            for (int k = 0; k < kSpan; ++k) {
                for (int c = 0; c < kC; ++c) {
                    states[(static_cast<std::size_t>(destination[i]) * kSpan + k) * kC + c] =
                        src_value(c, width - kSpan + k);
                }
            }
        }
    }
}

void conv_case(int T, const std::vector<std::int32_t>& source, const std::vector<std::int32_t>& destination,
               cudaStream_t stream) {
    const ConvInputs in = conv_inputs(T, 300U + static_cast<std::uint32_t>(T * 7 + source.size()));
    const ConvResult got = run_conv(in, T, source, destination, stream);
    std::vector<double> ref;
    std::vector<std::uint16_t> states;
    conv_oracle(in, T, source, destination, ref, states);
    std::string label = "ple_conv_inject T=" + std::to_string(T) + " sequences=" + std::to_string(source.size()) +
                        (destination.empty() ? " (no state update)" : "");
    pointwise(got.residual, ref, label + " residual");
    check(got.states == states, label + ": states equal the formula exactly (untouched slots included)");
}

// One call over W columns equals the same columns split into two calls (in place, one slot).
void conv_split_case(cudaStream_t stream) {
    constexpr int W = 12;
    const ConvInputs in    = conv_inputs(W, 777U);
    const ConvResult whole = run_conv(in, W, {1}, {1}, stream);
    for (const int first : {1, 5, 9, 11}) {
        const ConvResult a = run_conv(in, first, {1}, {1}, stream);
        const ConvResult b = run_conv(in, W - first, {1}, {1}, stream, &a.states, nullptr, first);
        std::vector<std::uint16_t> residual = a.residual;
        residual.insert(residual.end(), b.residual.begin(), b.residual.end());
        check(residual == whole.residual && b.states == whole.states,
              "ple_conv_inject: " + std::to_string(first) + " + " + std::to_string(W - first) +
                  " columns equal one call of " + std::to_string(W) + " bit for bit");
    }
}

// ple_conv_commit of n columns equals ple_conv_inject's in-place update over those n columns.
void commit_case(cudaStream_t stream) {
    constexpr int W = 6, B = 3;
    const ConvInputs in = conv_inputs(W * B, 999U);
    const std::vector<std::int32_t> commit = {0, 4, 6}, slots = {2, 0, 1};
    Buffer<std::uint16_t> d_norm(in.normalized), d_states(in.states);
    Buffer<std::int32_t> d_commit(commit), d_slots(slots);
    Tensor states(d_states.p, DType::BF16, {kC, kSpan, kSlots});
    ninfer::ops::ple_conv_commit(Tensor(d_norm.p, DType::BF16, {kC, W, B}),
                                 Tensor(d_commit.p, DType::I32, {B}), states,
                                 Tensor(d_slots.p, DType::I32, {B}), stream);
    t::cuda_check(cudaStreamSynchronize(stream), "commit");
    std::vector<std::uint16_t> want = in.states;
    for (int b = 0; b < B; ++b) {
        if (commit[b] == 0) { continue; }
        ConvInputs row = in;
        row.gated.assign(in.gated.begin() + static_cast<std::ptrdiff_t>(b) * W * kC,
                         in.gated.begin() + static_cast<std::ptrdiff_t>(b * W + commit[b]) * kC);
        row.normalized.assign(in.normalized.begin() + static_cast<std::ptrdiff_t>(b) * W * kC,
                              in.normalized.begin() + static_cast<std::ptrdiff_t>(b * W + commit[b]) * kC);
        row.residual.assign(row.gated.size(), 0);
        const ConvResult updated = run_conv(row, commit[b], {slots[b]}, {slots[b]}, stream, &want);
        want = updated.states;
    }
    check(d_states.read() == want,
          "ple_conv_commit equals the in-place ple_conv_inject update (n = 0 leaves the slot)");
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
        for (const int T : {1, 5, 64}) { embed_case(T, stream); }
        embed_refusals(stream);
        for (const int T : {1, 5, 64}) { gate_case(T, stream); }
        conv_case(1, {0}, {0}, stream);       // decode, in place
        conv_case(5, {2}, {1}, stream);       // narrower than the span, another slot
        conv_case(12, {1}, {1}, stream);      // wider than the span
        conv_case(64, {0}, {2}, stream);      // a prefill chunk
        conv_case(12, {2, 0}, {2, 0}, stream); // two sequences of 6, each in place
        conv_case(10, {1, 2}, {}, stream);    // verification: no state update
        conv_split_case(stream);
        commit_case(stream);
        cudaStreamDestroy(stream);
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL: %s\n", error.what());
        ++failures;
    }
    std::printf(failures == 0 ? "ple checks passed\n" : "FAIL: %d checks failed\n", failures);
    return failures == 0 ? 0 : 1;
}
