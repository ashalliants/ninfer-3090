// The second half of one DeepSeek-V4.1-Flash block (layer 0) with its REAL weights, run through the Ops on
// the card and compared stage by stage with the NumPy oracle (tools/dsv4/hyper_connection.py, moe.py) that
// tools/dsv4/export_ffn_fixture.py ran on the same weights:
//
//   residual streams R [4 x 5120, T]
//   hc_sinkhorn_mixes(R)                                  = pre, post, comb
//   hc_collapse(R, pre) -> rmsnorm(ffn_norm)              = collapsed, normed
//   projection_fp32(router) -> moe_route_sqrt_softplus    = the 6 experts and their weights per token
//   moe_dispatch -> moe_experts_narrow (384 real experts of the layer, IQ1_M gate/up/down)
//   linear(shared gate, up; Q5_K) -> clamped_swiglu -> linear(shared down; Q5_K)
//   weighted_slot_sum(routed outputs, weights, shared)    = y
//   hc_expand_mix(y, post, comb, R)                       = the next residual streams
//
// It runs only when NINFER_DSV4_FFN_FIXTURE names the exporter's output directory (about 3 GB: the whole
// layer's expert bank; the weights are the real model's, which the repository does not hold):
//   python3 tools/dsv4/export_ffn_fixture.py FIRST_SHARD.gguf DIR --layer 0 --tokens 8
// Criterion (suite-owned): every stage's relative L2 error against the oracle at most its bound below;
// the device rounds activations to BF16 and runs the experts and projections on Q8_1 / A8 activations,
// which the FP32 oracle does not, so the bounds are 2.5 to 5 times the measured errors (the values are
// printed; the hyper-connection coefficients agree to about 1e-7, the routed-plus-shared MoE output to
// about 1 percent with IQ1_M experts, the next residual streams to 0.2 percent). The router's selection is compared by recall (a near tie at the boundary may swap on the
// device's rounded inputs) and its weights by relative error on the experts both selected.

#include "ninfer/ops/cast.h"
#include "ninfer/ops/hc_sinkhorn.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/moe_mix.h"
#include "ninfer/ops/moe_route_sqrt_softplus.h"
#include "ninfer/ops/offloaded_sparse_moe.h"
#include "ninfer/ops/projection_fp32.h"
#include "ninfer/ops/rmsnorm.h"

#include "core/arena.h"
#include "core/weight.h"
#include "ops/op_tester.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <fstream>
#include <iterator>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace {

namespace t = ninfer::test;
namespace om = ninfer::ops::offloaded_moe;
using ninfer::DType;
using ninfer::QType;
using ninfer::Tensor;
using ninfer::Weight;
using ninfer::ops::LinearPolicy;

constexpr int kS = 4, kH = 5120, kInter = 2304, kExperts = 384;

int failures = 0;

void check(bool ok, const std::string& what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what.c_str());
        ++failures;
    }
}

struct Device {
    void* p = nullptr;
    std::size_t size = 0;
    explicit Device(std::size_t bytes) : size(bytes) { t::cuda_check(cudaMalloc(&p, bytes ? bytes : 1), "cudaMalloc"); }
    ~Device() { cudaFree(p); }
    Device(const Device&)            = delete;
    Device& operator=(const Device&) = delete;
    std::uint8_t* bytes() const { return static_cast<std::uint8_t*>(p); }
};

std::vector<std::uint8_t> read_file(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) { throw std::runtime_error("cannot read " + path); }
    return std::vector<std::uint8_t>(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
}

template <class T>
std::vector<T> read_vector(const std::string& path) {
    const auto raw = read_file(path);
    std::vector<T> v(raw.size() / sizeof(T));
    std::memcpy(v.data(), raw.data(), v.size() * sizeof(T));
    return v;
}

void upload(Device& d, const void* host, std::size_t bytes, std::size_t offset = 0) {
    t::cuda_check(cudaMemcpy(d.bytes() + offset, host, bytes, cudaMemcpyHostToDevice), "upload");
}

std::unique_ptr<Device> upload_file(const std::string& path) {
    const auto raw = read_file(path);
    auto d         = std::make_unique<Device>(raw.size());
    upload(*d, raw.data(), raw.size());
    return d;
}

// The bank, in chunks (it is about 3 GB).
std::unique_ptr<Device> upload_bank(const std::string& path, std::size_t bytes) {
    auto d = std::make_unique<Device>(bytes);
    std::ifstream in(path, std::ios::binary);
    if (!in) { throw std::runtime_error("cannot read " + path); }
    constexpr std::size_t kChunk = 256U << 20;
    std::vector<char> buffer(kChunk);
    for (std::size_t at = 0; at < bytes; at += kChunk) {
        const std::size_t n = std::min(kChunk, bytes - at);
        in.read(buffer.data(), static_cast<std::streamsize>(n));
        if (static_cast<std::size_t>(in.gcount()) != n) { throw std::runtime_error("short read of " + path); }
        upload(*d, buffer.data(), n, at);
    }
    return d;
}

Weight ggml_weight(QType format, int n, int k, const void* device_blocks) {
    const ninfer::GgmlBlock block = ninfer::ggml_block(format);
    Weight w;
    w.payload       = device_blocks;
    w.payload_bytes = static_cast<std::uint64_t>(n) * (k / block.values) * block.bytes;
    w.qtype         = format;
    w.layout        = ninfer::QuantLayout::GgmlBlocks;
    w.ndim          = 2;
    w.n = w.shape[0] = w.padded_shape[0] = n;
    w.k = w.shape[1] = w.padded_shape[1] = k;
    w.qdata                              = device_blocks;
    w.group                              = static_cast<std::int32_t>(block.values);
    w.group_size                         = block.values;
    return w;
}

std::vector<float> download_bf16_as_f32(const void* device, std::size_t count) {
    std::vector<std::uint16_t> raw(count);
    t::cuda_check(cudaMemcpy(raw.data(), device, count * 2, cudaMemcpyDeviceToHost), "download");
    std::vector<float> out(count);
    for (std::size_t i = 0; i < count; ++i) { out[i] = t::bf16_to_f32(raw[i]); }
    return out;
}

std::vector<float> download_f32(const void* device, std::size_t count) {
    std::vector<float> out(count);
    t::cuda_check(cudaMemcpy(out.data(), device, count * 4, cudaMemcpyDeviceToHost), "download");
    return out;
}

void compare(const std::string& stage, const std::vector<float>& got, const std::vector<float>& ref, double bound) {
    if (got.size() != ref.size()) {
        check(false, stage + ": sizes differ");
        return;
    }
    double num = 0, den = 0, worst = 0;
    for (std::size_t i = 0; i < got.size(); ++i) {
        if (!std::isfinite(got[i])) {
            check(false, stage + ": a non-finite value");
            return;
        }
        const double d = static_cast<double>(got[i]) - ref[i];
        num += d * d;
        den += static_cast<double>(ref[i]) * ref[i];
        worst = std::max(worst, std::fabs(d));
    }
    const double rel = std::sqrt(num / den), rms = std::sqrt(den / got.size());
    std::printf("%-10s relative L2 %.5f (bound %.5f), worst |error| %.4g over rms %.4g\n", stage.c_str(), rel, bound, worst, rms);
    check(rel <= bound, stage + ": relative L2 error within its bound");
}

std::map<std::string, std::string> read_config(const std::string& path) {
    std::map<std::string, std::string> config;
    const auto raw = read_file(path);
    std::istringstream in(std::string(raw.begin(), raw.end()));
    std::string key, value;
    while (in >> key >> value) { config[key] = value; }
    return config;
}

QType record_type(const std::string& name) {
    static const std::map<std::string, QType> kTypes = {
        {"ggml_rec_iq1_m_iq1_m", QType::GGML_REC_IQ1_M_IQ1_M},       {"ggml_rec_iq2_xxs_iq1_m", QType::GGML_REC_IQ2_XXS_IQ1_M},
        {"ggml_rec_iq2_xxs_iq2_xxs", QType::GGML_REC_IQ2_XXS_IQ2_XXS}, {"ggml_rec_iq2_xs_iq1_m", QType::GGML_REC_IQ2_XS_IQ1_M},
        {"ggml_rec_iq2_xxs_iq2_xs", QType::GGML_REC_IQ2_XXS_IQ2_XS},  {"ggml_rec_iq1_m_iq2_xxs", QType::GGML_REC_IQ1_M_IQ2_XXS},
        {"ggml_rec_iq2_xs_iq2_xxs", QType::GGML_REC_IQ2_XS_IQ2_XXS},  {"ggml_rec_iq2_xs_iq2_xs", QType::GGML_REC_IQ2_XS_IQ2_XS}};
    return kTypes.at(name);
}

} // namespace

int main() {
    try {
        const char* dir_env = std::getenv("NINFER_DSV4_FFN_FIXTURE");
        if (dir_env == nullptr) {
            std::printf("SKIP: NINFER_DSV4_FFN_FIXTURE is not set (export a layer with tools/dsv4/export_ffn_fixture.py)\n");
            return 0;
        }
        if (t::cuda_unavailable()) {
            std::printf("SKIP: no CUDA device\n");
            return 77;
        }
        const std::string dir = std::string(dir_env) + "/";
        cudaStream_t stream   = nullptr;
        t::cuda_check(cudaStreamCreate(&stream), "stream");
        namespace o_ = ninfer::ops;

        const auto config = read_config(dir + "config.txt");
        const auto number = [&](const char* key) { return std::stod(config.at(key)); };
        const int T = static_cast<int>(number("tokens")), k = static_cast<int>(number("topk"));
        check(number("hc") == kS && number("dim") == kH && number("experts") == kExperts && number("inter") == kInter,
              "the exported layer has V4.1-Flash's geometry");
        const auto ref = [&](const char* name) { return read_vector<float>(dir + name + ".bin"); };
        std::printf("layer %d, %d tokens, %d of %d experts, records %s\n", static_cast<int>(number("layer")), T, k, kExperts,
                    config.at("record_format").c_str());

        // Weights.
        const auto fn_d = upload_file(dir + "hc_fn.bin"), scale_d = upload_file(dir + "hc_scale.bin"), base_d = upload_file(dir + "hc_base.bin"),
                   bias_d = upload_file(dir + "bias.bin"), router_d = upload_file(dir + "router.bin");
        const auto norm_f32 = read_vector<float>(dir + "ffn_norm.bin");
        std::vector<std::uint16_t> norm_bits(norm_f32.size());
        for (std::size_t i = 0; i < norm_f32.size(); ++i) { norm_bits[i] = t::f32_to_bf16(norm_f32[i]); }
        Device norm_d(norm_bits.size() * 2);
        upload(norm_d, norm_bits.data(), norm_bits.size() * 2);
        const auto sg = upload_file(dir + "shexp_gate.bin"), su = upload_file(dir + "shexp_up.bin"), sd = upload_file(dir + "shexp_down.bin");
        const Weight shared_gate = ggml_weight(QType::GGML_Q5_K, kInter, kH, sg->p), shared_up = ggml_weight(QType::GGML_Q5_K, kInter, kH, su->p),
                     shared_down = ggml_weight(QType::GGML_Q5_K, kH, kInter, sd->p);
        const QType format         = record_type(config.at("record_format"));
        const std::uint64_t stride = static_cast<std::uint64_t>(number("record_stride"));
        const auto bank            = upload_bank(dir + "bank.bin", static_cast<std::size_t>(stride) * kExperts);
        std::printf("bank: %.2f GB on the card\n", static_cast<double>(bank->size) / 1e9);

        // Streams.
        const auto x_bits = read_file(dir + "x.bin");
        check(x_bits.size() == static_cast<std::size_t>(T) * kS * kH * 2, "x.bin is [tokens, 4 x hidden] BF16");
        Device residual(x_bits.size()), pre(static_cast<std::size_t>(kS) * T * 4), post(static_cast<std::size_t>(kS) * T * 4),
            comb(static_cast<std::size_t>(kS) * kS * T * 4), collapsed(static_cast<std::size_t>(kH) * T * 2), normed(static_cast<std::size_t>(kH) * T * 2),
            logits(static_cast<std::size_t>(kExperts) * T * 4), ids(static_cast<std::size_t>(k) * T * 4), weights(static_cast<std::size_t>(k) * T * 4),
            gate(static_cast<std::size_t>(kInter) * T * 2), up(static_cast<std::size_t>(kInter) * T * 2),
            activation(static_cast<std::size_t>(kInter) * T * 2), shared_out(static_cast<std::size_t>(kH) * T * 2),
            routed(static_cast<std::size_t>(kH) * k * T * 2), y(static_cast<std::size_t>(kH) * T * 2), shared_gate_unused(static_cast<std::size_t>(T) * 4);
        upload(residual, x_bits.data(), x_bits.size());

        const Tensor R(residual.p, DType::BF16, {kS * kH, T}), FN(fn_d->p, DType::FP32, {kS * kH, 24}), SCALE(scale_d->p, DType::FP32, {3}),
            BASE(base_d->p, DType::FP32, {24}), NW(norm_d.p, DType::BF16, {kH});
        Tensor PRE(pre.p, DType::FP32, {kS, T}), POST(post.p, DType::FP32, {kS, T}), COMB(comb.p, DType::FP32, {kS * kS, T}),
            XC(collapsed.p, DType::BF16, {kH, T}), XN(normed.p, DType::BF16, {kH, T});
        Tensor RESIDUAL(residual.p, DType::BF16, {kS * kH, T});

        const float norm_eps = static_cast<float>(number("norm_eps")), hc_eps = static_cast<float>(number("hc_eps"));
        o_::hc_sinkhorn_mixes(R, FN, SCALE, BASE, norm_eps, hc_eps, static_cast<int>(number("hc_iters")), PRE, POST, COMB, stream);
        t::cuda_check(cudaStreamSynchronize(stream), "mixes");
        compare("pre", download_f32(pre.p, static_cast<std::size_t>(kS) * T), ref("pre"), 5e-6);
        compare("post", download_f32(post.p, static_cast<std::size_t>(kS) * T), ref("post"), 5e-6);
        compare("comb", download_f32(comb.p, static_cast<std::size_t>(kS) * kS * T), ref("comb"), 5e-6);
        o_::hc_collapse(R, PRE, XC, stream);
        o_::rmsnorm(XC, NW, norm_eps, false, XN, stream);
        t::cuda_check(cudaStreamSynchronize(stream), "norm");
        compare("collapsed", download_bf16_as_f32(collapsed.p, static_cast<std::size_t>(kH) * T), ref("collapsed"), 0.004);
        compare("normed", download_bf16_as_f32(normed.p, static_cast<std::size_t>(kH) * T), ref("normed"), 0.006);

        // The router, in FP32 from the BF16 weights, then the selection.
        const Tensor RW(router_d->p, DType::BF16, {kH, kExperts});
        const Tensor* router_weights[] = {&RW};
        Tensor LOGITS(logits.p, DType::FP32, {kExperts, T}), IDS(ids.p, DType::I32, {k, T}), WEIGHTS(weights.p, DType::FP32, {k, T});
        const Tensor BIAS(bias_d->p, DType::FP32, {kExperts});
        o_::projection_fp32(XN, router_weights, LOGITS, stream);
        o_::moe_route_sqrt_softplus(LOGITS, BIAS, k, static_cast<float>(number("route_scale")), number("norm_topk_prob") != 0, IDS, WEIGHTS, stream);
        t::cuda_check(cudaStreamSynchronize(stream), "route");
        {
            const auto want_ids = read_vector<std::int32_t>(dir + "route_ids.bin");
            const auto want_w   = ref("route_weights");
            std::vector<std::int32_t> got_ids(static_cast<std::size_t>(k) * T);
            t::cuda_check(cudaMemcpy(got_ids.data(), ids.p, got_ids.size() * 4, cudaMemcpyDeviceToHost), "ids");
            const auto got_w = download_f32(weights.p, static_cast<std::size_t>(k) * T);
            std::size_t hits = 0;
            double worst_w = 0;
            for (int s = 0; s < T; ++s) {
                for (int j = 0; j < k; ++j) {
                    for (int i = 0; i < k; ++i) {
                        if (got_ids[static_cast<std::size_t>(s) * k + j] == want_ids[static_cast<std::size_t>(s) * k + i]) {
                            ++hits;
                            const double a = got_w[static_cast<std::size_t>(s) * k + j], b = want_w[static_cast<std::size_t>(s) * k + i];
                            worst_w = std::max(worst_w, std::fabs(a - b) / std::fabs(b));
                        }
                    }
                }
            }
            const double recall = static_cast<double>(hits) / static_cast<double>(k * T);
            std::printf("%-10s recall %.4f over %d entries, worst weight error %.4g relative\n", "route", recall, k * T, worst_w);
            check(recall >= 0.95, "the routed experts recall at least 95 percent of the oracle's");
            check(worst_w <= 0.03, "the weights of the experts both selected agree to 3 percent");
        }

        // The routed experts: dispatch from the device's own routing, then the real records.
        o_::MoeRouting routing;
        routing.ids         = IDS;
        routing.weights     = WEIGHTS;
        routing.shared_gate = Tensor(shared_gate_unused.p, DType::FP32, {T});
        Device dispatch_buffer(o_::moe_dispatch_bytes(kExperts, k * T)), route_log(static_cast<std::size_t>(k) * T * 4);
        o_::MoeDispatch view = o_::carve_moe_dispatch(dispatch_buffer.p, kExperts, k * T);
        o_::moe_dispatch(routing, kExperts, view, static_cast<std::int32_t*>(route_log.p), stream);
        std::vector<std::int32_t> frame_table(kExperts);
        for (int e = 0; e < kExperts; ++e) { frame_table[e] = e; }
        Device frames(frame_table.size() * 4);
        upload(frames, frame_table.data(), frame_table.size() * 4);
        o_::MoeExpertSource source;
        source.format        = format;
        source.frame_base    = bank->bytes();
        source.host_records  = bank->bytes(); // required even when every expert is resident (the miss path reads it); device memory is readable
        source.frames        = static_cast<const std::int32_t*>(frames.p);
        source.record_stride = stride;
        const int max_jobs   = std::min(kExperts, k * T);
        Device workspace(o_::moe_experts_narrow_workspace_bytes(om::kDsv4Shape, max_jobs, k * T, T, false));
        Tensor ROUTED(routed.p, DType::BF16, {kH, k * T});
        o_::moe_experts_narrow(om::kDsv4Shape, XN, view, source, k, max_jobs, workspace.p, ROUTED, stream);

        // The shared expert: gate and up, the clamped SwiGLU, down.
        const LinearPolicy policy = LinearPolicy::AllowA8;
        const auto run_linear = [&](const Tensor& in, const Weight& weight, Tensor& result) {
            const std::size_t capacity = o_::linear_workspace_capacity_bytes(weight.qtype, weight.n, weight.k, policy, T, T);
            ninfer::DeviceArena arena(std::max<std::size_t>(capacity, 256));
            o_::linear(in, weight, result, policy, arena, stream);
        };
        Tensor GATE(gate.p, DType::BF16, {kInter, T}), UP(up.p, DType::BF16, {kInter, T}), ACT(activation.p, DType::BF16, {kInter, T}),
            SHARED(shared_out.p, DType::BF16, {kH, T}), Y(y.p, DType::BF16, {kH, T});
        run_linear(XN, shared_gate, GATE);
        run_linear(XN, shared_up, UP);
        o_::clamped_swiglu(GATE, UP, static_cast<float>(number("clamp_shared")), ACT, stream);
        run_linear(ACT, shared_down, SHARED);
        o_::weighted_slot_sum(ROUTED, WEIGHTS, SHARED, Y, stream);
        t::cuda_check(cudaStreamSynchronize(stream), "moe");
        compare("y", download_bf16_as_f32(y.p, static_cast<std::size_t>(kH) * T), ref("y"), 0.03);

        // Back into the streams.
        o_::hc_expand_mix(Y, POST, COMB, RESIDUAL, stream);
        t::cuda_check(cudaStreamSynchronize(stream), "expand");
        compare("out", download_bf16_as_f32(residual.p, static_cast<std::size_t>(kS) * kH * T), ref("out"), 0.005);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "EXCEPTION: %s\n", e.what());
        return 1;
    }
    if (failures != 0) {
        std::fprintf(stderr, "dsv4 ffn layer: %d failure(s)\n", failures);
        return 1;
    }
    std::printf("dsv4 ffn layer: passed\n");
    return 0;
}
