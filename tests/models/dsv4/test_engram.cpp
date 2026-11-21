// Written for NInfer-3090.
//
// DeepSeek-V4.1-Flash's Engram injection (src/models/dsv4/execution/engram.h) against
// tools/dsv4/engram_ref.py.
//
//  1. The n-gram hash: row ids exactly equal to the oracle's
//  (tests/fixtures/dsv4/engram_hash_golden.json, 4-grams,
//     8 heads, layers 1 and 14 at the real bucket counts), whole, chunked and token by token (the
//     history carries across prefill chunks and decode steps).
//  2. The whole device sequence at V4.1-Flash's geometry (24 IQ4_NL rows of 256 values per token
//  and layer, the
//     Q5_K engram_kv [25600, 6144], 4 streams of 5120) on a small synthetic volume written here (a
//     few thousand buckets per head; the real volume does not exist yet): the rows staged from the
//     volume are the volume's bytes, the projection is within a bound derived below of its FP64
//     value, and the gated write is within one BF16 rounding (plus the gate's FP32 evaluation
//     error) of the FP64 formula applied to the device's own projection. Run on the whole sequence
//     (the wide linear route), in uneven chunks and one token at a time (the decode route), a
//     graph-captured call equal to the eager call.
//  3. With NINFER_DSV4_ENGRAM_FIXTURE (tools/dsv4/export_engram_fixture.py: layer 1's REAL
//  engram_kv and gate, the
//     real token map and multipliers, 40 tokens, real rows from the GGUF in a sparse volume with
//     the real 768,022,850-row geometry, and the NumPy oracle's results): the same checks against
//     the oracle's FP32 rows, projection and output.
//
// Criteria (suite-owned). Projection (Q5_K weight, BF16 activations a, Q8_1-quantised per 32 values
// by the Op: d = amax / 127, so each quantised activation is within amax / 254 of a, plus the
// binary16 rounding of d, 2^-11 of the value): for every output n, |got - sum_k w_nk a_k| <= sum_k
// |w_nk| (amax_blk(k) / 254 + 2^-10 |a_k|) + 2^-24 K sum_k |w_nk a_k| (the FP32 accumulation, worst
// case) + 2^-9 |ref| (the BF16 output). Gated write: as in ninfer_engram_gate_inject_test (half a
// BF16 ulp, the FP32 add, the gate's evaluation error times the value). Against the oracle's FP32
// output the sum of both applies (the projection bound feeds the value and, through the gate, the
// key): reported, and required within 1% relative L2 of the written change.

#include "artifact/schema.h"
#include "core/arena.h"
#include "core/device.h"
#include "models/dsv4/config.h"
#include "models/dsv4/execution/engram.h"
#include "models/dsv4/execution/engram_sidecar.h"
#include "models/qwen3_5/frontend/digest.h"
#include "ninfer/ops/linear.h"
#include "ops/ggml_blocks_decode.h"
#include "ops/op_tester.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <numeric>
#include <random>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef NINFER_SOURCE_DIR
#    define NINFER_SOURCE_DIR "."
#endif

namespace {

namespace t    = ninfer::test;
namespace dsv4 = ninfer::models::dsv4;
namespace q4   = ninfer::models::qwen4_exp;
using ninfer::DType;
using ninfer::QType;
using ninfer::Tensor;
using ninfer::Weight;
using ninfer::artifact::Json;

int failures = 0;

void check(bool ok, const std::string& what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what.c_str());
        ++failures;
    }
}

template <class F>
bool throws(F&& f) {
    try {
        f();
    } catch (const std::exception&) { return true; }
    return false;
}

std::vector<std::byte> read_file(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) { throw std::runtime_error("cannot open " + path.string()); }
    in.seekg(0, std::ios::end);
    std::vector<std::byte> out(static_cast<std::size_t>(in.tellg()));
    in.seekg(0);
    in.read(reinterpret_cast<char*>(out.data()), static_cast<std::streamsize>(out.size()));
    return out;
}

template <class T>
std::vector<T> read_array(const std::filesystem::path& path) {
    const auto bytes = read_file(path);
    std::vector<T> out(bytes.size() / sizeof(T));
    std::memcpy(out.data(), bytes.data(), out.size() * sizeof(T));
    return out;
}

Json read_json(const std::filesystem::path& path) {
    const auto bytes = read_file(path);
    return ninfer::artifact::parse_json(
        std::string_view(reinterpret_cast<const char*>(bytes.data()), bytes.size()), path.string());
}

template <class T>
std::vector<T> numbers(const Json& v) {
    std::vector<T> out;
    for (const auto& e : v) { out.push_back(e.get<T>()); }
    return out;
}

constexpr std::uint32_t kHidden = 5120, kStreams = 4, kRowWidth = 256, kColumns = 24;
constexpr std::uint32_t kK = kColumns * kRowWidth;     // 6144
constexpr std::uint32_t kN = (kStreams + 1) * kHidden; // 25600

// A TextConfig with what the Engram reads: hidden size, streams, epsilon, the layer plan (layers 1
// and 14) and the Engram geometry.
dsv4::TextConfig text_config(std::vector<std::uint64_t> sizes, std::int32_t pad_token, float eps) {
    dsv4::TextConfig c;
    c.hidden_size       = kHidden;
    c.hc_streams        = kStreams;
    c.rms_norm_eps      = eps;
    c.num_hidden_layers = 40;
    c.layers.resize(40);
    c.layers[1].engram        = true;
    c.layers[14].engram       = true;
    c.engram.ngram_size       = 4;
    c.engram.heads            = 8;
    c.engram.row_width        = kRowWidth;
    c.engram.pad_token        = pad_token;
    c.engram.table_rows       = std::accumulate(sizes.begin(), sizes.end(), std::uint64_t{0});
    c.engram.head_vocab_sizes = std::move(sizes);
    return c;
}

// ---------------------------------------------------------------------------------------- 1. the
// hash

void test_hash_golden() {
    const Json golden =
        read_json(std::string(NINFER_SOURCE_DIR) + "/tests/fixtures/dsv4/engram_hash_golden.json");
    const auto sizes  = numbers<std::uint64_t>(golden.at("head_vocab_sizes"));
    const auto tokens = numbers<std::int32_t>(golden.at("tokens"));
    const dsv4::TextConfig config =
        text_config(sizes, golden.at("pad_token").get<std::int32_t>(), 1e-6F);
    dsv4::EngramHashParams params{numbers<std::int32_t>(golden.at("token_map")),
                                  numbers<std::int64_t>(golden.at("multipliers"))};
    const dsv4::EngramHasher hasher(config.engram, 2, params);
    check(hasher.table_rows() == 768022850,
          "the golden's bucket counts total the real table's rows");
    std::vector<std::uint32_t> want;
    for (const auto& token : golden.at("row_ids")) {
        for (const auto& layer : token) {
            for (const auto& id : layer) { want.push_back(id.get<std::uint32_t>()); }
        }
    }
    const std::size_t per_token = 2 * 24;
    check(want.size() == tokens.size() * per_token, "the golden's row ids have the expected shape");

    // Whole sequence.
    std::vector<std::uint32_t> got(want.size());
    auto history = hasher.start();
    hasher.append(history, tokens, got);
    check(got == want, "hash ids of the whole sequence equal the oracle's exactly");

    // Uneven chunks, then one token at a time (the history carries across calls).
    std::mt19937 rng(7);
    for (int round = 0; round < 3; ++round) {
        std::vector<std::uint32_t> chunked(want.size());
        auto h         = hasher.start();
        std::size_t at = 0;
        while (at < tokens.size()) {
            const std::size_t n =
                round == 2 ? 1 : std::min<std::size_t>(tokens.size() - at, 1 + rng() % 17);
            hasher.append(h, std::span<const std::int32_t>(tokens).subspan(at, n),
                          std::span<std::uint32_t>(chunked).subspan(at * per_token, n * per_token));
            at += n;
        }
        check(chunked == want, round == 2 ? "token-by-token hash ids equal the oracle's"
                                          : "chunked hash ids equal the oracle's");
    }
    // A history that is copied and continued gives the same ids as the original (a speculative
    // branch).
    {
        auto h = hasher.start();
        std::vector<std::uint32_t> head(20 * per_token), a(10 * per_token), b(10 * per_token);
        hasher.append(h, std::span<const std::int32_t>(tokens).first(20), head);
        auto copy = h;
        hasher.append(h, std::span<const std::int32_t>(tokens).subspan(20, 10), a);
        hasher.append(copy, std::span<const std::int32_t>(tokens).subspan(20, 10), b);
        check(a == b && std::equal(a.begin(), a.end(), want.begin() + 20 * per_token),
              "a copied history continues identically");
    }
    // Refusals.
    {
        auto h = hasher.start();
        std::vector<std::uint32_t> out(per_token);
        const std::int32_t bad[] = {static_cast<std::int32_t>(params.token_map.size())};
        check(throws([&] { hasher.append(h, bad, out); }),
              "a token outside the vocabulary is refused");
        const std::int32_t neg[] = {-1};
        check(throws([&] { hasher.append(h, neg, out); }), "a negative token is refused");
        const std::int32_t ok[] = {5};
        std::vector<std::uint32_t> wrong(per_token + 1);
        check(throws([&] { hasher.append(h, ok, wrong); }),
              "an output of the wrong size is refused");
        auto bad_params = params;
        bad_params.multipliers.pop_back();
        check(throws([&] { dsv4::EngramHasher(config.engram, 2, bad_params); }),
              "a missing multiplier is refused");
        bad_params                = params;
        bad_params.multipliers[0] = 0;
        check(throws([&] { dsv4::EngramHasher(config.engram, 2, bad_params); }),
              "a zero multiplier is refused");
        bad_params                = params;
        bad_params.multipliers[0] = INT64_MAX;
        check(throws([&] { dsv4::EngramHasher(config.engram, 2, bad_params); }),
              "a multiplier that could overflow is refused");
        check(throws([&] { dsv4::EngramHasher(config.engram, 3, params); }),
              "head_vocab_sizes of another layer count are refused");
    }
    std::printf("hash: %zu tokens x 2 layers x 24 columns equal the oracle's (whole, chunked, "
                "token by token)\n",
                tokens.size());
}

// ---------------------------------------------------------------------------------------- volumes

// Writes an n-gram volume (tools/artifact/ngram_volume.py's version 2 layout) of `rows` (row_bytes
// each).
q4::NgramTableConfig write_volume(const std::filesystem::path& path,
                                  const std::vector<std::byte>& rows, std::uint64_t count,
                                  const std::array<std::uint8_t, 16>& id) {
    constexpr std::uint32_t row_bytes = 144, per_block = 4096 / row_bytes;
    q4::NgramTableConfig c;
    c.format         = QType::GGML_IQ4_NL;
    c.rows           = count;
    c.row_bytes      = row_bytes;
    c.rows_per_block = per_block;
    c.block_bytes    = 4096;
    c.header_bytes   = 4096;
    c.blocks         = (count + per_block - 1) / per_block;
    c.file_bytes     = 4096 + c.blocks * 4096;
    c.volume_id      = id;
    std::vector<std::byte> file(c.file_bytes, std::byte{0});
    const auto put = [&](std::size_t at, const void* v, std::size_t n) {
        std::memcpy(file.data() + at, v, n);
    };
    const std::uint32_t version = 2;
    put(0, "NINFERNG", 8);
    put(8, &version, 4);
    put(12, &c.header_bytes, 4);
    put(16, &c.rows, 8);
    put(24, &c.row_bytes, 4);
    put(28, &c.rows_per_block, 4);
    put(32, &c.block_bytes, 4);
    put(36, &c.blocks, 8);
    put(44, id.data(), 16);
    put(60, "ggml_iq4_nl", 11);
    for (std::uint64_t r = 0; r < count; ++r) {
        std::memcpy(file.data() + 4096 + (r / per_block) * 4096 + (r % per_block) * row_bytes,
                    rows.data() + r * row_bytes, row_bytes);
    }
    std::ofstream out(path, std::ios::binary);
    out.write(reinterpret_cast<const char*>(file.data()),
              static_cast<std::streamsize>(file.size()));
    return c;
}

std::uint32_t lcg(std::uint32_t& s) { return s = s * 1664525U + 1013904223U; }

// Random valid IQ4_NL rows: 8 blocks of (binary16 scale, 16 bytes of nibbles). Scales are normal
// binary16s of 2^-5..2^-2.
std::vector<std::byte> random_iq4_rows(std::uint64_t rows, std::uint32_t seed) {
    std::vector<std::byte> out(rows * 144);
    std::uint32_t s = seed;
    for (std::uint64_t r = 0; r < rows * 8; ++r) {
        std::byte* b = out.data() + r * 18;
        const std::uint16_t d =
            static_cast<std::uint16_t>(0x2800U + (lcg(s) >> 16) % 0x0C00U); // 2^-5 .. 2^-2
        std::memcpy(b, &d, 2);
        for (int i = 0; i < 16; ++i) { b[2 + i] = static_cast<std::byte>(lcg(s) >> 24); }
    }
    return out;
}

// Random valid Q5_K blocks (176 bytes): scales d and dmin binary16 of about 2^-9..2^-6, every other
// byte random.
std::vector<std::byte> random_q5k(std::uint32_t n, std::uint32_t k, std::uint32_t seed) {
    std::vector<std::byte> out(static_cast<std::size_t>(n) * (k / 256) * 176);
    std::uint32_t s = seed;
    for (std::size_t i = 0; i < out.size(); ++i) { out[i] = static_cast<std::byte>(lcg(s) >> 24); }
    for (std::size_t b = 0; b < out.size() / 176; ++b) {
        const std::uint16_t d = static_cast<std::uint16_t>(0x1800U + (lcg(s) >> 16) % 0x0C00U),
                            m = static_cast<std::uint16_t>(0x1800U + (lcg(s) >> 16) % 0x0C00U);
        std::memcpy(out.data() + b * 176, &d, 2);
        std::memcpy(out.data() + b * 176 + 2, &m, 2);
    }
    return out;
}

Weight device_q5k(const ninfer::DeviceBuffer& storage, std::uint32_t n, std::uint32_t k) {
    const auto block = ninfer::ggml_block(QType::GGML_Q5_K);
    Weight w;
    w.payload       = storage.p;
    w.payload_bytes = static_cast<std::uint64_t>(n) * (k / block.values) * block.bytes;
    w.qtype         = QType::GGML_Q5_K;
    w.layout        = ninfer::QuantLayout::GgmlBlocks;
    w.ndim          = 2;
    w.n = w.shape[0] = w.padded_shape[0] = static_cast<std::int32_t>(n);
    w.k = w.shape[1] = w.padded_shape[1] = static_cast<std::int32_t>(k);
    w.qdata                              = storage.p;
    w.group                              = static_cast<std::int32_t>(block.values);
    w.group_size                         = block.values;
    return w;
}

// ---------------------------------------------------------------------------------------- checks

float bf16(float f) { return t::bf16_to_f32(t::f32_to_bf16(f)); }

// The device projection `kv_bits` [T][25600] against the FP64 product of the sampled rows of the
// Q5_K weight and the BF16 activations `act` [T][6144] (criterion in the header). Returns the worst
// error / bound.
double check_projection(const std::vector<std::byte>& blocks, const std::vector<float>& act,
                        std::uint32_t T, const std::vector<std::uint16_t>& kv_bits,
                        const std::vector<std::uint32_t>& sample) {
    const std::size_t row_bytes = static_cast<std::size_t>(kK / 256) * 176;
    std::vector<float> w(kK);
    double worst = 0;
    for (const std::uint32_t n : sample) {
        t::ggml::decode_blocks(QType::GGML_Q5_K,
                               std::span<const std::byte>(blocks).subspan(
                                   static_cast<std::size_t>(n) * row_bytes, row_bytes),
                               w);
        for (std::uint32_t c = 0; c < T; ++c) {
            const float* a = &act[static_cast<std::size_t>(c) * kK];
            double ref = 0, abs_dot = 0, quant = 0;
            for (std::uint32_t k0 = 0; k0 < kK; k0 += 32) {
                float amax = 0;
                for (std::uint32_t k = k0; k < k0 + 32; ++k) {
                    amax = std::max(amax, std::fabs(a[k]));
                }
                for (std::uint32_t k = k0; k < k0 + 32; ++k) {
                    ref += static_cast<double>(w[k]) * a[k];
                    abs_dot += std::fabs(static_cast<double>(w[k]) * a[k]);
                    quant += std::fabs(static_cast<double>(w[k])) *
                             (amax / 254.0 + std::ldexp(1.0, -10) * std::fabs(a[k]));
                }
            }
            const double bound = quant + std::ldexp(1.0, -24) * kK * abs_dot +
                                 std::ldexp(1.0, -9) * std::fabs(ref) + 1e-30;
            const double have = t::bf16_to_f32(kv_bits[static_cast<std::size_t>(c) * kN + n]);
            worst             = std::max(worst, std::fabs(have - ref) / bound);
        }
    }
    return worst;
}

// engram_gate_inject's criterion (include/ninfer/ops/engram.h) for the residual written from `x`
// and the device's projection. Returns the worst error / bound.
double check_inject(const std::vector<std::uint16_t>& x, const std::vector<std::uint16_t>& kv,
                    const std::vector<float>& gate, const std::vector<std::uint16_t>& out,
                    std::uint32_t T, float eps) {
    constexpr double u = 5.960464477539063e-8;
    double worst       = 0;
    const double H     = kHidden;
    for (std::uint32_t c = 0; c < T; ++c) {
        const std::size_t xo = static_cast<std::size_t>(c) * kStreams * kHidden,
                          ko = static_cast<std::size_t>(c) * kN;
        for (std::uint32_t s = 0; s < kStreams; ++s) {
            double xx = 0, kk = 0, xwk = 0, axwk = 0;
            for (std::uint32_t d = 0; d < kHidden; ++d) {
                const double xv  = t::bf16_to_f32(x[xo + s * kHidden + d]),
                             kv_ = t::bf16_to_f32(kv[ko + s * kHidden + d]);
                xx += xv * xv;
                kk += kv_ * kv_;
                xwk += xv * gate[s * kHidden + d] * kv_;
                axwk += std::fabs(xv * gate[s * kHidden + d] * kv_);
            }
            const double rx = 1.0 / std::sqrt(xx / H + eps), rk = 1.0 / std::sqrt(kk / H + eps);
            const double dot = xwk * rx * rk / std::sqrt(H), a = axwk * rx * rk / std::sqrt(H);
            const double g   = std::copysign(std::sqrt(std::max(std::fabs(dot), 1e-6)), dot);
            const double sig = 1.0 / (1.0 + std::exp(-g));
            const double dg  = (H + 16) * u * (a + 2 * std::fabs(dot)) /
                              (2.0 * std::sqrt(std::max(std::fabs(dot), 1e-6)));
            for (std::uint32_t d = 0; d < kHidden; ++d) {
                const double xv = t::bf16_to_f32(x[xo + s * kHidden + d]);
                const double val =
                    t::bf16_to_f32(kv[ko + static_cast<std::size_t>(kStreams) * kHidden + d]);
                const double ref = xv + sig * val;
                const double hu =
                    ref == 0.0 ? 0.0 : std::ldexp(1.0, std::ilogb(std::fabs(ref)) - 8);
                const double bnd =
                    hu + u * std::fabs(xv) + std::fabs(val) * (0.25 * dg + 4 * u) + 1e-30;
                worst = std::max(worst,
                                 std::fabs(t::bf16_to_f32(out[xo + s * kHidden + d]) - ref) / bnd);
            }
        }
    }
    return worst;
}

struct Applied {
    std::vector<std::uint16_t> out, kv;
};

// Runs apply for the staged call on a copy of x [T][4 * 5120] (BF16 bits).
Applied apply_once(dsv4::Engram& engram, std::uint32_t layer, const Weight& weight,
                   const Tensor& gate, const std::vector<std::uint16_t>& x, std::uint32_t T,
                   ninfer::WorkspaceArena& arena, cudaStream_t stream, bool graph) {
    ninfer::DeviceBuffer d_x(x.size() * 2);
    d_x.copy_from_host(x.data(), x.size() * 2);
    Tensor residual(d_x.p, DType::BF16,
                    {static_cast<std::int32_t>(kStreams * kHidden), static_cast<std::int32_t>(T)});
    Tensor projected;
    const auto call = [&] {
        projected = engram.apply(layer, weight, gate, residual, ninfer::ops::LinearPolicy::AllowA8,
                                 arena, stream);
    };
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
    Applied a;
    a.out.resize(x.size());
    a.kv.resize(static_cast<std::size_t>(kN) * T);
    t::cuda_check(cudaMemcpy(a.out.data(), d_x.p, a.out.size() * 2, cudaMemcpyDeviceToHost),
                  "get out");
    t::cuda_check(cudaMemcpy(a.kv.data(), projected.data, a.kv.size() * 2, cudaMemcpyDeviceToHost),
                  "get kv");
    return a;
}

// The BF16 activations the device feeds the projection for the staged call and layer: the staged
// rows' exact decode.
std::vector<float> decoded_activations(const dsv4::Engram& engram, std::uint32_t layer,
                                       std::uint32_t T) {
    std::vector<float> act(static_cast<std::size_t>(T) * kK);
    for (std::uint32_t c = 0; c < T; ++c) {
        for (std::uint32_t col = 0; col < kColumns; ++col) {
            t::ggml::decode_blocks(
                QType::GGML_IQ4_NL, engram.staged_row(layer, c, col),
                std::span<float>(act).subspan(static_cast<std::size_t>(c) * kK + col * kRowWidth,
                                              kRowWidth));
        }
    }
    for (auto& v : act) { v = bf16(v); }
    return act;
}

std::vector<std::uint32_t> sample_rows(std::uint32_t count, std::uint32_t seed) {
    std::vector<std::uint32_t> rows{0, kN - 1, (kStreams * kHidden) - 1,
                                    kStreams *
                                        kHidden}; // the first and last key rows, the value's first
    std::uint32_t s = seed;
    while (rows.size() < count) { rows.push_back(lcg(s) % kN); }
    return rows;
}

// ---------------------------------------------------------------------------------------- 2. the
// device sequence

void run_sequence(const std::string& label, dsv4::Engram& engram, const dsv4::TextConfig& config,
                  std::uint32_t layer, const Weight& weight, const Tensor& gate,
                  const std::vector<float>& gate_host, const std::vector<std::byte>& q5k,
                  const std::vector<std::uint16_t>& x, const std::vector<std::int32_t>& tokens,
                  ninfer::WorkspaceArena& arena, cudaStream_t stream) {
    const std::uint32_t T = static_cast<std::uint32_t>(tokens.size());
    const auto sample     = sample_rows(48, 99);
    const auto history0   = engram.hasher().start();

    // Whole sequence, then uneven chunks, then token by token: the staged row ids and bytes are
    // identical in all three, and each device result meets the criteria.
    std::vector<std::uint32_t> ids_whole;
    std::vector<std::byte> rows_whole;
    double worst_kv = 0, worst_inject = 0;
    for (int mode = 0; mode < 3; ++mode) {
        auto history = history0;
        std::vector<std::uint32_t> ids;
        std::vector<std::byte> rows;
        std::size_t at = 0;
        std::vector<std::size_t> steps;
        if (mode == 0) { steps = {T}; }
        if (mode == 1) { steps = {13, 1, 1, 7, T - 22}; }
        if (mode == 2) { steps.assign(T, 1); }
        for (const std::size_t n : steps) {
            const dsv4::Engram::Run run{&history,
                                        std::span<const std::int32_t>(tokens).subspan(at, n)};
            engram.stage(std::span<const dsv4::Engram::Run>(&run, 1));
            const std::uint32_t n32 = static_cast<std::uint32_t>(n);
            // Rows of this call for the layer, as staged.
            for (std::uint32_t c = 0; c < n32; ++c) {
                for (std::uint32_t col = 0; col < kColumns; ++col) {
                    const auto row = engram.staged_row(layer, c, col);
                    rows.insert(rows.end(), row.begin(), row.end());
                }
            }
            const auto got_ids = engram.row_ids();
            ids.insert(ids.end(), got_ids.begin(), got_ids.end());
            const std::vector<std::uint16_t> xs(
                x.begin() + static_cast<std::ptrdiff_t>(at * kStreams * kHidden),
                x.begin() + static_cast<std::ptrdiff_t>((at + n) * kStreams * kHidden));
            // Verify the decode-route calls (token by token) on a thinner sample to keep the host
            // work modest.
            const Applied a =
                apply_once(engram, layer, weight, gate, xs, n32, arena, stream, false);
            const auto act = decoded_activations(engram, layer, n32);
            const auto rows_checked =
                mode == 2 ? std::vector<std::uint32_t>(sample.begin(), sample.begin() + 12)
                          : sample;
            worst_kv     = std::max(worst_kv, check_projection(q5k, act, n32, a.kv, rows_checked));
            worst_inject = std::max(
                worst_inject, check_inject(xs, a.kv, gate_host, a.out, n32, config.rms_norm_eps));
            at += n;
        }
        if (mode == 0) {
            ids_whole  = ids;
            rows_whole = rows;
        } else {
            check(ids == ids_whole, label + ": row ids in " +
                                        (mode == 1 ? "chunks" : "single steps") +
                                        " equal the whole sequence's");
            check(rows == rows_whole, label + ": staged rows in " +
                                          (mode == 1 ? "chunks" : "single steps") +
                                          " equal the whole sequence's");
        }
    }
    check(worst_kv <= 1.0,
          label + ": projection within its bound (worst " + std::to_string(worst_kv) + ")");
    check(worst_inject <= 1.0,
          label + ": gated write within its bound (worst " + std::to_string(worst_inject) + ")");
    std::printf("%s: %u tokens whole / chunked / one at a time: projection worst error / bound "
                "%.3f, gated write %.3f\n",
                label.c_str(), T, worst_kv, worst_inject);

    // A graph-captured T = 1 call is bitwise the eager call.
    {
        auto history = history0;
        const dsv4::Engram::Run run{&history, std::span<const std::int32_t>(tokens).first(1)};
        engram.stage(std::span<const dsv4::Engram::Run>(&run, 1));
        const std::vector<std::uint16_t> x1(x.begin(), x.begin() + kStreams * kHidden);
        const Applied eager = apply_once(engram, layer, weight, gate, x1, 1, arena, stream, false);
        const Applied graph = apply_once(engram, layer, weight, gate, x1, 1, arena, stream, true);
        check(eager.out == graph.out && eager.kv == graph.kv,
              label + ": a graph-captured T = 1 call is bitwise the eager call");

        // Report only: the device time of one captured T = 1 layer (rows copy, decode, projection,
        // gated write).
        ninfer::DeviceBuffer d_x(x1.size() * 2);
        d_x.copy_from_host(x1.data(), x1.size() * 2);
        Tensor residual(d_x.p, DType::BF16, {static_cast<std::int32_t>(kStreams * kHidden), 1});
        cudaGraph_t g        = nullptr;
        cudaGraphExec_t exec = nullptr;
        t::cuda_check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal), "capture");
        (void)engram.apply(layer, weight, gate, residual, ninfer::ops::LinearPolicy::AllowA8, arena,
                           stream);
        t::cuda_check(cudaStreamEndCapture(stream, &g), "end capture");
        t::cuda_check(cudaGraphInstantiate(&exec, g, 0), "instantiate");
        for (int i = 0; i < 20; ++i) { t::cuda_check(cudaGraphLaunch(exec, stream), "warm"); }
        cudaEvent_t e0, e1;
        t::cuda_check(cudaEventCreate(&e0), "event");
        t::cuda_check(cudaEventCreate(&e1), "event");
        t::cuda_check(cudaEventRecord(e0, stream), "event");
        for (int i = 0; i < 200; ++i) { t::cuda_check(cudaGraphLaunch(exec, stream), "launch"); }
        t::cuda_check(cudaEventRecord(e1, stream), "event");
        t::cuda_check(cudaEventSynchronize(e1), "event");
        float ms = 0;
        t::cuda_check(cudaEventElapsedTime(&ms, e0, e1), "elapsed");
        std::printf("%s: one captured T = 1 layer: %.1f us (rows copy + decode + projection + "
                    "gated write)\n",
                    label.c_str(), ms * 1000.0F / 200.0F);
        cudaEventDestroy(e0);
        cudaEventDestroy(e1);
        cudaGraphExecDestroy(exec);
        cudaGraphDestroy(g);
    }
}

void test_synthetic(cudaStream_t stream) {
    // 48 distinct primes above 1000 as bucket counts (24 per layer).
    std::vector<std::uint64_t> sizes;
    for (std::uint64_t p = 1001; sizes.size() < 48; ++p) {
        bool prime = true;
        for (std::uint64_t d = 2; d * d <= p && prime; ++d) { prime = p % d != 0; }
        if (prime) { sizes.push_back(p); }
    }
    const dsv4::TextConfig config = text_config(sizes, 2, 1e-6F);
    const std::uint64_t rows      = config.engram.table_rows;
    std::mt19937 rng(11);
    dsv4::EngramHashParams params;
    const std::int32_t vocab = 5000, compressed = 2000;
    for (int i = 0; i < vocab; ++i) {
        params.token_map.push_back(i < compressed ? i
                                                  : static_cast<std::int32_t>(rng() % compressed));
    }
    for (int i = 0; i < 8; ++i) {
        params.multipliers.push_back(static_cast<std::int64_t>((rng() % 1000000007ULL) * 2 + 1));
    }

    const auto table = random_iq4_rows(rows, 5);
    const auto path  = std::filesystem::temp_directory_path() / "ninfer_dsv4_engram_test.ngram";
    const std::array<std::uint8_t, 16> id{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    const auto geometry = write_volume(path, table, rows, id);
    {
        const q4::NgramVolume volume(path, geometry);
        const std::uint32_t max_tokens = 64;
        dsv4::Engram engram(config, params, volume, max_tokens);

        const auto q5k = random_q5k(kN, kK, 77);
        ninfer::DeviceBuffer d_w(q5k.size());
        d_w.copy_from_host(q5k.data(), q5k.size());
        const Weight weight = device_q5k(d_w, kN, kK);
        std::vector<float> gate_host(kStreams * kHidden);
        std::uint32_t s = 3;
        for (auto& v : gate_host) {
            v = 1.0F + 0.5F * (static_cast<float>(lcg(s) >> 8) / 8388608.0F - 1.0F);
        }
        ninfer::DeviceBuffer d_gate(gate_host.size() * 4);
        d_gate.copy_from_host(gate_host.data(), gate_host.size() * 4);
        const Tensor gate(
            d_gate.p, DType::FP32,
            {static_cast<std::int32_t>(kHidden), static_cast<std::int32_t>(kStreams)});

        const std::uint32_t T = 40;
        std::vector<std::int32_t> tokens(T);
        for (auto& tok : tokens) { tok = static_cast<std::int32_t>(rng() % vocab); }
        tokens[5] = tokens[6] = tokens[7] = 2; // the pad token itself
        std::vector<std::uint16_t> x(static_cast<std::size_t>(T) * kStreams * kHidden);
        for (auto& v : x) {
            v = t::f32_to_bf16(1.5F * (static_cast<float>(lcg(s) >> 8) / 4194304.0F - 1.0F));
        }

        ninfer::WorkspaceArena arena(
            engram.apply_workspace_bytes(max_tokens, ninfer::ops::LinearPolicy::AllowA8) +
            (std::size_t{1} << 20));
        for (std::uint32_t layer = 0; layer < 2; ++layer) {
            run_sequence("synthetic volume, Engram layer " + std::to_string(layer), engram, config,
                         layer, weight, gate, gate_host, q5k, x, tokens, arena, stream);
        }

        // The staged rows are the volume's bytes: an independent read of the row ids against the
        // table in memory.
        {
            auto history = engram.hasher().start();
            const dsv4::Engram::Run run{&history, tokens};
            engram.stage(std::span<const dsv4::Engram::Run>(&run, 1));
            bool same = true;
            for (std::uint32_t layer = 0; layer < 2; ++layer) {
                for (std::uint32_t c = 0; c < T; ++c) {
                    for (std::uint32_t col = 0; col < kColumns; ++col) {
                        const std::uint32_t row =
                            engram.row_ids()[(static_cast<std::size_t>(c) * 2 + layer) * kColumns +
                                             col];
                        const auto got = engram.staged_row(layer, c, col);
                        same           = same && row < rows &&
                               std::memcmp(got.data(),
                                           table.data() + static_cast<std::size_t>(row) * 144,
                                           144) == 0;
                        // Each (layer, column) draws from its own bucket range of the table.
                        std::uint64_t lo = 0;
                        for (std::uint32_t i = 0; i < layer * kColumns + col; ++i) {
                            lo += sizes[i];
                        }
                        same = same && row >= lo && row < lo + sizes[layer * kColumns + col];
                    }
                }
            }
            check(same, "staged rows are the volume's bytes at ids inside their (layer, column) "
                        "bucket ranges");
        }

        // Refusals.
        {
            auto history = engram.hasher().start();
            const dsv4::Engram::Run none{&history, {}};
            check(throws([&] { engram.stage(std::span<const dsv4::Engram::Run>(&none, 1)); }),
                  "staging no tokens is refused");
            std::vector<std::int32_t> many(max_tokens + 1, 3);
            const dsv4::Engram::Run too_many{&history, many};
            check(throws([&] { engram.stage(std::span<const dsv4::Engram::Run>(&too_many, 1)); }),
                  "staging more than max_tokens is refused");
            {
                // A failing stage (a bad token after valid ones) leaves the history and the staged
                // call untouched.
                auto before = engram.hasher().start();
                auto h2     = before;
                std::vector<std::int32_t> mixed{3, 4, 1 << 30};
                const dsv4::Engram::Run bad{&h2, mixed};
                check(throws([&] { engram.stage(std::span<const dsv4::Engram::Run>(&bad, 1)); }),
                      "a stage with a bad token is refused");
                check(h2.tail == before.tail && engram.staged_tokens() == 0,
                      "a failed stage leaves the history and the staged call untouched");
            }
            const dsv4::Engram::Run nohistory{nullptr, tokens};
            check(throws([&] { engram.stage(std::span<const dsv4::Engram::Run>(&nohistory, 1)); }),
                  "a run without a history is refused");
            const dsv4::Engram::Run three{&history, std::span<const std::int32_t>(tokens).first(3)};
            engram.stage(std::span<const dsv4::Engram::Run>(&three, 1));
            ninfer::DeviceBuffer d_x(static_cast<std::size_t>(2) * kStreams * kHidden * 2);
            Tensor residual(d_x.p, DType::BF16, {static_cast<std::int32_t>(kStreams * kHidden), 2});
            check(throws([&] {
                      (void)engram.apply(0, weight, gate, residual,
                                         ninfer::ops::LinearPolicy::AllowA8, arena, stream);
                  }),
                  "a residual of another column count than the staged call is refused");
            cudaStream_t other = nullptr;
            t::cuda_check(cudaStreamCreate(&other), "stream");
            Tensor ok3(d_x.p, DType::BF16, {static_cast<std::int32_t>(kStreams * kHidden), 3});
            check(throws([&] {
                      (void)engram.apply(0, weight, gate, ok3, ninfer::ops::LinearPolicy::AllowA8,
                                         arena, other);
                  }),
                  "a second stream is refused (one copy fence per instance)");
            {
                // A launch failure (a refused gate) leaves a fresh instance unbound: another
                // stream still works.
                dsv4::Engram fresh(config, params, volume, 8);
                auto h3 = fresh.hasher().start();
                const dsv4::Engram::Run one{&h3, std::span<const std::int32_t>(tokens).first(1)};
                fresh.stage(std::span<const dsv4::Engram::Run>(&one, 1));
                Tensor r1(d_x.p, DType::BF16, {static_cast<std::int32_t>(kStreams * kHidden), 1});
                // The last Op refuses a gate of the wrong size, after the copy and the projection
                // were enqueued.
                const Tensor bad_gate(gate.data, DType::FP32,
                                      {static_cast<std::int32_t>(kHidden), 3});
                check(throws([&] {
                          (void)fresh.apply(0, weight, bad_gate, r1,
                                            ninfer::ops::LinearPolicy::AllowA8, arena, stream);
                      }),
                      "an apply whose last Op refuses its input fails");
                check(!throws([&] {
                    (void)fresh.apply(0, weight, gate, r1, ninfer::ops::LinearPolicy::AllowA8,
                                      arena, other);
                }),
                      "after a failed first apply the instance is not bound to its stream");
                t::cuda_check(cudaStreamSynchronize(other), "sync");
            }
            // after `fresh` (whose destructor synchronizes the stream it is bound to)
            cudaStreamDestroy(other);
            Tensor ok(d_x.p, DType::BF16, {static_cast<std::int32_t>(kStreams * kHidden), 3});
            check(throws([&] {
                      (void)engram.apply(2, weight, gate, ok, ninfer::ops::LinearPolicy::AllowA8,
                                         arena, stream);
                  }),
                  "a layer index past the Engram layers is refused");
        }
        // Another stage's volume: rows differing from the hash's total are refused.
        {
            auto bad = text_config(sizes, 2, 1e-6F);
            bad.engram.head_vocab_sizes.back() += 2;
            check(throws([&] { dsv4::Engram(bad, params, volume, 8); }),
                  "a volume with another row count than the hash's buckets is refused");
        }
    }
    std::filesystem::remove(path);

    // parse_engram_table.
    {
        Json record{{"format", "ggml_iq4_nl"},
                    {"rows", 768022850},
                    {"row_bytes", 144},
                    {"rows_per_block", 28},
                    {"block_bytes", 4096},
                    {"header_bytes", 4096},
                    {"blocks", 27429388},
                    {"file_bytes", 4096 + std::uint64_t{27429388} * 4096},
                    {"volume_id", "0102030405060708090a0b0c0d0e0f10"}};
        const auto c = dsv4::parse_engram_table(record, 256);
        check(c.rows == 768022850 && c.row_bytes == 144 && c.volume_id[0] == 1 &&
                  c.volume_id[15] == 16,
              "the real table's geometry parses");
        Json unbound         = record;
        unbound["volume_id"] = nullptr;
        check(throws([&] { (void)dsv4::parse_engram_table(unbound, 256); }),
              "an unbound volume_id is refused");
        Json malformed         = record;
        malformed["volume_id"] = "zz02030405060708090a0b0c0d0e0f10";
        check(throws([&] { (void)dsv4::parse_engram_table(malformed, 256); }),
              "a volume_id with non-hex characters is refused");
        Json wrong         = record;
        wrong["row_bytes"] = 90;
        check(throws([&] { (void)dsv4::parse_engram_table(wrong, 256); }),
              "a row size that does not fit the row width is refused");
        Json other      = record;
        other["format"] = "ggml_q8_0";
        check(throws([&] { (void)dsv4::parse_engram_table(other, 256); }),
              "a volume that is not IQ4_NL is refused");
    }
}

// ---------------------------------------------------------------------------------------- the sidecar folder

std::string sha_hex(const void* data, std::size_t n) {
    namespace fe = ninfer::models::qwen3_5::frontend;
    return fe::sha256_hex(fe::sha256(std::span<const std::uint8_t>(static_cast<const std::uint8_t*>(data), n)));
}

// The folder tools/convert/dsv4_engram_volume.py writes, at a small table: engram.json, the volume and the token map.
void test_sidecar() {
    std::vector<std::uint64_t> sizes;
    for (std::uint64_t p = 1001; sizes.size() < 48; ++p) {
        bool prime = true;
        for (std::uint64_t d = 2; d * d <= p && prime; ++d) { prime = p % d != 0; }
        if (prime) { sizes.push_back(p); }
    }
    dsv4::TextConfig config = text_config(sizes, 2, 1e-6F);
    config.vocab_size       = 5000;
    const std::vector<std::int64_t> multipliers{3, 5, 7, 9, 11, 13, 15, 17};
    std::vector<std::int32_t> token_map(config.vocab_size);
    for (std::size_t i = 0; i < token_map.size(); ++i) { token_map[i] = static_cast<std::int32_t>(i % 2000); }

    const auto dir = std::filesystem::temp_directory_path() / "ninfer_dsv4_sidecar_test";
    std::filesystem::remove_all(dir);
    std::filesystem::create_directories(dir);
    const std::array<std::uint8_t, 16> id{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    const auto table = write_volume(dir / "engram.volume", random_iq4_rows(config.engram.table_rows, 3), config.engram.table_rows, id);
    {
        std::ofstream out(dir / "engram.token_map.i32", std::ios::binary);
        out.write(reinterpret_cast<const char*>(token_map.data()), static_cast<std::streamsize>(token_map.size() * 4));
    }
    const auto header = read_file(dir / "engram.volume");
    Json record{{"format", "ggml_iq4_nl"}, {"rows", table.rows}, {"row_bytes", table.row_bytes}, {"rows_per_block", table.rows_per_block},
                {"block_bytes", 4096}, {"header_bytes", 4096}, {"blocks", table.blocks}, {"file_bytes", table.file_bytes},
                {"volume_id", "0102030405060708090a0b0c0d0e0f10"}};
    const Json manifest{
        {"format", "ninfer-dsv4-engram"}, {"version", 1},
        {"volume", {{"file", "engram.volume"}, {"table", record}, {"header_sha256", sha_hex(header.data(), 4096)}}},
        {"hash",
         {{"layers", {1, 14}}, {"ngram_size", 4}, {"heads", 8}, {"row_width", 256}, {"pad_token_id", 2}, {"head_vocab_sizes", sizes},
          {"multipliers", multipliers}, {"token_map_count", token_map.size()},
          {"token_map", {{"file", "engram.token_map.i32"}, {"sha256", sha_hex(token_map.data(), token_map.size() * 4)}}}}}};
    const auto write_manifest = [&](const Json& m) { std::ofstream(dir / "engram.json") << m.dump(); };
    write_manifest(manifest);

    const dsv4::EngramSidecar loaded = dsv4::load_engram_sidecar(dir, config);
    check(loaded.params.token_map == token_map && loaded.params.multipliers == multipliers, "the sidecar's token map and multipliers load");
    check(loaded.table.rows == config.engram.table_rows && loaded.volume_path == dir / "engram.volume", "the sidecar names the volume");
    {
        const q4::NgramVolume volume(loaded.volume_path, loaded.table); // the header's id matches the manifest's
        const dsv4::EngramHasher hasher(config.engram, 2, loaded.params);
        check(hasher.table_rows() == volume.table().rows, "the sidecar's hash addresses the volume's rows");
    }

    const auto refused = [&](const char* label, const Json& m, const dsv4::TextConfig& c) {
        write_manifest(m);
        check(throws([&] { (void)dsv4::load_engram_sidecar(dir, c); }), label);
    };
    Json bad = manifest;
    bad["hash"]["pad_token_id"] = 3;
    refused("a pad token that differs from the model's is refused", bad, config);
    bad = manifest;
    bad["hash"]["head_vocab_sizes"][0] = 1;
    refused("bucket counts that differ from the model's are refused", bad, config);
    bad = manifest;
    bad["hash"]["layers"] = Json::array({1, 15});
    refused("other Engram layers are refused", bad, config);
    bad = manifest;
    bad["volume"]["header_sha256"] = std::string(64, '0');
    refused("a volume header that is not the manifest's is refused", bad, config);
    bad = manifest;
    bad["version"] = 2;
    refused("another manifest version is refused", bad, config);
    dsv4::TextConfig other = config;
    other.vocab_size       = 4999;
    refused("a token map of another length than the vocabulary is refused", manifest, other);
    write_manifest(manifest);
    {
        std::fstream map(dir / "engram.token_map.i32", std::ios::in | std::ios::out | std::ios::binary);
        map.seekp(40);
        map.put('\x7f');
    }
    check(throws([&] { (void)dsv4::load_engram_sidecar(dir, config); }), "a damaged token map is refused");
    std::filesystem::remove(dir / "engram.json");
    check(throws([&] { (void)dsv4::load_engram_sidecar(dir, config); }), "a folder without engram.json is refused");
    std::filesystem::remove_all(dir);
}

// ---------------------------------------------------------------------------------------- 3. real
// weights

void test_real(const std::filesystem::path& dir, cudaStream_t stream) {
    const Json meta           = read_json(dir / "meta.json");
    const std::uint32_t T     = static_cast<std::uint32_t>(meta.at("tokens").size());
    const std::uint32_t layer = meta.at("layer_index").get<std::uint32_t>();
    const float eps           = meta.at("eps").get<float>();
    const auto sizes          = numbers<std::uint64_t>(meta.at("head_vocab_sizes"));
    const dsv4::TextConfig config =
        text_config(sizes, meta.at("pad_token").get<std::int32_t>(), eps);
    dsv4::EngramHashParams params{read_array<std::int32_t>(dir / "token_map.i32"),
                                  numbers<std::int64_t>(meta.at("multipliers"))};
    const auto tokens = numbers<std::int32_t>(meta.at("tokens"));

    // The hash with the real map, multipliers and bucket counts: exactly the oracle's row ids.
    const dsv4::EngramHasher hasher(config.engram, 2, params);
    std::vector<std::uint32_t> want;
    for (const auto& token : meta.at("row_ids")) {
        for (const auto& l : token) {
            for (const auto& id : l) { want.push_back(id.get<std::uint32_t>()); }
        }
    }
    std::vector<std::uint32_t> got(want.size());
    auto history = hasher.start();
    hasher.append(history, tokens, got);
    check(got == want, "real tables: hash ids equal the oracle's exactly");

    const auto geometry = dsv4::parse_engram_table(meta.at("engram_table"), kRowWidth);
    const q4::NgramVolume volume(dir / "volume.ngram", geometry);
    dsv4::Engram engram(config, params, volume, 64);
    auto h = engram.hasher().start();
    const dsv4::Engram::Run run{&h, tokens};
    engram.stage(std::span<const dsv4::Engram::Run>(&run, 1));

    // The rows read from the (sparse, real-geometry) volume are the GGUF's bytes, and decode to the
    // oracle's rows.
    const auto raw  = read_file(dir / "rows.bin"); // [token][layer][column][144]
    bool bytes_same = true;
    for (std::uint32_t c = 0; c < T; ++c) {
        for (std::uint32_t l = 0; l < 2; ++l) {
            for (std::uint32_t col = 0; col < kColumns; ++col) {
                bytes_same =
                    bytes_same &&
                    std::memcmp(engram.staged_row(l, c, col).data(),
                                raw.data() +
                                    ((static_cast<std::size_t>(c) * 2 + l) * kColumns + col) * 144,
                                144) == 0;
            }
        }
    }
    check(bytes_same, "real rows staged from the volume are the GGUF's bytes");
    const auto act      = decoded_activations(engram, layer, T);
    const auto ref_rows = read_array<float>(dir / "ref_rows.f32");
    double row_err      = 0;
    for (std::size_t i = 0; i < act.size(); ++i) {
        row_err = std::max<double>(row_err, std::fabs(act[i] - bf16(ref_rows[i])));
    }
    check(row_err == 0.0, "the decoded rows are the oracle's rows rounded to BF16, exactly");

    // The device: real engram_kv and gate.
    const auto q5k = [&] {
        const auto bytes = read_file(dir / "kv.bin");
        return bytes;
    }();
    check(q5k.size() == static_cast<std::size_t>(kN) * (kK / 256) * 176,
          "kv.bin is the Q5_K [25600, 6144] tensor");
    ninfer::DeviceBuffer d_w(q5k.size());
    d_w.copy_from_host(q5k.data(), q5k.size());
    const Weight weight  = device_q5k(d_w, kN, kK);
    const auto gate_host = read_array<float>(dir / "gate.bin");
    ninfer::DeviceBuffer d_gate(gate_host.size() * 4);
    d_gate.copy_from_host(gate_host.data(), gate_host.size() * 4);
    const Tensor gate(d_gate.p, DType::FP32,
                      {static_cast<std::int32_t>(kHidden), static_cast<std::int32_t>(kStreams)});
    const auto x = read_array<std::uint16_t>(dir / "x.bin");
    ninfer::WorkspaceArena arena(
        engram.apply_workspace_bytes(64, ninfer::ops::LinearPolicy::AllowA8) +
        (std::size_t{1} << 20));
    const Applied a = apply_once(engram, layer, weight, gate, x, T, arena, stream, false);

    const double w_kv  = check_projection(q5k, act, T, a.kv, sample_rows(96, 5));
    const double w_inj = check_inject(x, a.kv, gate_host, a.out, T, eps);
    check(w_kv <= 1.0,
          "real engram_kv: projection within its bound (worst " + std::to_string(w_kv) + ")");
    check(w_inj <= 1.0,
          "real engram_gate: gated write within its bound (worst " + std::to_string(w_inj) + ")");

    // Against the oracle's FP32 chain (unquantised activation, FP32 everywhere, no BF16 on the
    // projection).
    const auto ref_kv  = read_array<float>(dir / "ref_kv.f32");
    const auto ref_out = read_array<float>(dir / "ref_out.f32");
    double kv_num = 0, kv_den = 0, out_num = 0, change_den = 0, out_max = 0, change_max = 0;
    for (std::size_t i = 0; i < ref_kv.size(); ++i) {
        const double d = t::bf16_to_f32(a.kv[i]) - ref_kv[i];
        kv_num += d * d;
        kv_den += static_cast<double>(ref_kv[i]) * ref_kv[i];
    }
    for (std::size_t i = 0; i < ref_out.size(); ++i) {
        const double d      = t::bf16_to_f32(a.out[i]) - ref_out[i],
                     change = ref_out[i] - t::bf16_to_f32(x[i]);
        out_num += d * d;
        change_den += change * change;
        out_max    = std::max(out_max, std::fabs(d));
        change_max = std::max(change_max, std::fabs(change));
    }
    const double kv_rel = std::sqrt(kv_num / kv_den), out_rel = std::sqrt(out_num / change_den);
    std::printf("real layer %u weights, %u tokens: staged rows == GGUF bytes, decode == oracle "
                "rows; projection worst error / bound %.3f, "
                "gated write %.3f; vs the oracle's FP32 chain: kv rel L2 %.3g, out error / written "
                "change rel L2 %.3g (max |d| %.3g of max change %.3g)\n",
                meta.at("layer").get<std::uint32_t>(), T, w_kv, w_inj, kv_rel, out_rel, out_max,
                change_max);
    check(kv_rel < 0.01, "real weights: the projection is within 1% (rel L2) of the oracle's");
    check(out_rel < 0.01, "real weights: the written change is within 1% (rel L2) of the oracle's");
}

} // namespace

int main() {
    try {
        test_hash_golden();
        test_sidecar();
        if (t::cuda_unavailable()) {
            std::printf("SKIP the device part: no CUDA device\n");
            return failures == 0 ? 0 : 1;
        }
        cudaStream_t stream = nullptr;
        t::cuda_check(cudaStreamCreate(&stream), "stream");
        test_synthetic(stream);
        if (const char* dir = std::getenv("NINFER_DSV4_ENGRAM_FIXTURE")) {
            test_real(dir, stream);
        } else {
            std::printf("real weights: skipped (set NINFER_DSV4_ENGRAM_FIXTURE; "
                        "tools/dsv4/export_engram_fixture.py)\n");
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "EXCEPTION: %s\n", e.what());
        return 1;
    }
    if (failures != 0) {
        std::fprintf(stderr, "dsv4 engram: %d failure(s)\n", failures);
        return 1;
    }
    std::printf("dsv4 engram: passed\n");
    return 0;
}
