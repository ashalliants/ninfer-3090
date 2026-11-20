// One DeepSeek-V4.1-Flash attention layer with its REAL weights, run through the Ops on the card and
// compared stage by stage with the NumPy oracle (tools/dsv4/attention.py) that
// tools/dsv4/export_layer_fixture.py ran on the same weights. A window-only layer (layer 0):
//
//   x -> linear(q_a, Q8_0) -> rmsnorm -> linear(q_b, Q5_K) -> rope_pairs          = q
//   x -> linear(kv, Q8_0)  -> rmsnorm -> rope_pairs                               = kv
//   gathered_sink_attention(q, kv, window rows, sink)                             = o_pre
//   rope_pairs(inverse) -> grouped linear(out_a, Q5_K, 8 groups of 4096 -> 1024)  = projected
//   linear(out_b, Q5_K, 8192 -> 5120)                                             = out
//
// and, for a compressing layer that owns its compressed KV and its indexer (layer 2, ratio 2), also
//
//   x -> projection_fp32(compressor kv, gate; BF16) -> softmax_pool -> rmsnorm    = latent
//   latent -> rope_pairs (positions 0, 2, 4, ...)                                 = ckv
//   latent -> projection(index key) -> rmsnorm -> rope_pairs                      = index_k
//   q_a's output -> projection(index query) -> rope_pairs; x -> projection(index weights)
//   weighted_relu_dot_scores -> top_k_positions (offset = the raw kv rows)        = topk
//   gathered_sink_attention over [window rows | selected compressed rows]
//
// It runs only when NINFER_DSV4_LAYER_FIXTURE names the exporter's output directory (the weights are
// the real model's, which the repository does not hold; export a layer with
//   python3 tools/dsv4/export_layer_fixture.py FIRST_SHARD.gguf DIR --layer 0 --tokens 200
//   python3 tools/dsv4/export_layer_fixture.py FIRST_SHARD.gguf DIR --layer 2 --tokens 1100).
// Criterion (suite-owned): every stage's relative L2 error against the oracle at most its bound
// below; the device rounds activations to BF16 and casts them to Q8_1 for the integer products,
// which the FP32 oracle does not, so the bounds are about 2.5 times the measured errors (0.6 to 1
// percent at every stage; the values are printed). The indexer's selection is compared by recall
// (the device's scores differ from the oracle's by those roundings, so a position at the selection
// boundary may swap). The grouped output projection's gather and scatter between the per-group
// linears are plain copies here; the model's program will do it in one step.

#include "ninfer/ops/cast.h"
#include "ninfer/ops/gathered_sink_attention.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/position_selection.h"
#include "ninfer/ops/projection_fp32.h"
#include "ninfer/ops/rmsnorm.h"
#include "ninfer/ops/rope_pairs.h"
#include "ninfer/ops/softmax_pool.h"

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
#include <set>
#include <sstream>
#include <string>
#include <vector>

namespace {

namespace t = ninfer::test;
using ninfer::DType;
using ninfer::QType;
using ninfer::Tensor;
using ninfer::Weight;
using ninfer::ops::LinearPolicy;

constexpr int kHidden = 5120, kQLora = 1280, kHeads = 64, kHeadDim = 512, kRope = 64, kGroups = 8,
              kGroupIn = kHeads * kHeadDim / kGroups, kGroupOut = 1024, kCompDim = 512;
constexpr float kEps = 1e-20F; // the GGUF's attention.layer_norm_rms_epsilon, as FP32

int failures = 0;

void check(bool ok, const std::string& what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what.c_str());
        ++failures;
    }
}

struct Device {
    void* p = nullptr;
    explicit Device(std::size_t bytes) { t::cuda_check(cudaMalloc(&p, bytes ? bytes : 1), "cudaMalloc"); }
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

void upload(Device& d, const void* host, std::size_t bytes) {
    t::cuda_check(cudaMemcpy(d.p, host, bytes, cudaMemcpyHostToDevice), "upload");
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

// Relative L2 error ||got - ref|| / ||ref||, and the worst element error over the reference's rms.
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

} // namespace

int main() {
    try {
        const char* dir_env = std::getenv("NINFER_DSV4_LAYER_FIXTURE");
        if (dir_env == nullptr) {
            std::printf("SKIP: NINFER_DSV4_LAYER_FIXTURE is not set (export a layer with tools/dsv4/export_layer_fixture.py)\n");
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
        const int ratio = static_cast<int>(number("ratio")), window = static_cast<int>(number("window"));
        check(number("rope_dim") == kRope && (ratio == 0 || ratio == 2), "a supported layer (rope 64; ratio 0 or 2)");
        std::printf("layer %d, compress ratio %d, %d tokens\n", static_cast<int>(number("layer")), ratio,
                    static_cast<int>(number("tokens")));

        // Inputs and the oracle's stages.
        const auto x_bits = read_file(dir + "x.bin");
        const int T       = static_cast<int>(x_bits.size() / 2 / kHidden);
        check(static_cast<std::size_t>(T) * kHidden * 2 == x_bits.size() && T >= 1, "x.bin is [tokens, hidden] BF16");
        const int G = ratio != 0 ? T / ratio : 0; // compressed rows
        const auto ref = [&](const char* name) { return read_vector<float>(dir + name + ".bin"); };

        // The projections, as the GGUF stores them.
        std::map<std::string, std::unique_ptr<Device>> blocks;
        const auto load = [&](const char* field) -> const Device& {
            const auto raw = read_file(dir + field + ".bin");
            auto buffer    = std::make_unique<Device>(raw.size());
            upload(*buffer, raw.data(), raw.size());
            return *(blocks[field] = std::move(buffer));
        };
        const Weight wq_a = ggml_weight(QType::GGML_Q8_0, kQLora, kHidden, load("wq_a").p),
                     wq_b = ggml_weight(QType::GGML_Q5_K, kHeads * kHeadDim, kQLora, load("wq_b").p),
                     wkv  = ggml_weight(QType::GGML_Q8_0, kHeadDim, kHidden, load("wkv").p),
                     wo_b = ggml_weight(QType::GGML_Q5_K, kHidden, kGroups * kGroupOut, load("wo_b").p);
        const Device& wo_a_blocks = load("wo_a");

        const auto bf16_vector = [&](const char* file) {
            const auto v = read_vector<float>(dir + file);
            std::vector<std::uint16_t> bits(v.size());
            for (std::size_t i = 0; i < v.size(); ++i) { bits[i] = t::f32_to_bf16(v[i]); }
            return bits;
        };
        const auto norm_weight = [&](const char* file) {
            const auto bits = bf16_vector(file);
            auto d          = std::make_unique<Device>(bits.size() * 2);
            upload(*d, bits.data(), bits.size() * 2);
            return d;
        };
        const auto q_norm_w = norm_weight("q_norm.bin"), kv_norm_w = norm_weight("kv_norm.bin");
        const auto sink     = read_vector<float>(dir + "attn_sink.bin");
        const auto inv_freq = read_vector<float>(dir + "inv_freq.bin");
        check(inv_freq.size() == kRope / 2, "inv_freq has rope_dim / 2 entries");

        std::vector<std::int32_t> positions(T);
        for (int i = 0; i < T; ++i) { positions[i] = i; }
        const int list = std::min(T, window); // the window rows of each query (the oracle's window_indices)
        std::vector<std::int32_t> window_rows(static_cast<std::size_t>(list) * T);
        for (int s = 0; s < T; ++s) {
            for (int j = 0; j < list; ++j) {
                const int index = std::max(s - window + 1, 0) + j;
                window_rows[static_cast<std::size_t>(s) * list + j] = index > s ? -1 : index;
            }
        }
        const int topk = ratio != 0 ? std::min(static_cast<int>(number("index_topk")), G) : 0;
        const int K    = list + topk;       // each query's row list: window rows, then selected compressed rows
        const int N    = T + G;             // the KV table: the raw rows, then the compressed ones

        Device x(x_bits.size()), qa(static_cast<std::size_t>(kQLora) * T * 2), qn(static_cast<std::size_t>(kQLora) * T * 2),
            q(static_cast<std::size_t>(kHeads) * kHeadDim * T * 2), kv_raw(static_cast<std::size_t>(kHeadDim) * T * 2),
            kv_table(static_cast<std::size_t>(kHeadDim) * N * 2), o(static_cast<std::size_t>(kHeads) * kHeadDim * T * 2),
            group_in(static_cast<std::size_t>(kGroupIn) * T * 2), group_out(static_cast<std::size_t>(kGroupOut) * T * 2),
            projected(static_cast<std::size_t>(kGroups) * kGroupOut * T * 2), out(static_cast<std::size_t>(kHidden) * T * 2),
            sink_d(kHeads * 4), freq_d(inv_freq.size() * 4), pos_d(static_cast<std::size_t>(T) * 4),
            rows_d(static_cast<std::size_t>(K) * T * 4);
        upload(x, x_bits.data(), x_bits.size());
        upload(sink_d, sink.data(), sink.size() * 4);
        upload(freq_d, inv_freq.data(), inv_freq.size() * 4);
        upload(pos_d, positions.data(), positions.size() * 4);
        // The window rows go into the first `list` slots of each column of the row list.
        {
            Device window_d(window_rows.size() * 4);
            upload(window_d, window_rows.data(), window_rows.size() * 4);
            t::cuda_check(cudaMemcpy2D(rows_d.p, static_cast<std::size_t>(K) * 4, window_d.p, static_cast<std::size_t>(list) * 4,
                                       static_cast<std::size_t>(list) * 4, T, cudaMemcpyDeviceToDevice),
                          "window rows");
        }

        const LinearPolicy policy = LinearPolicy::AllowA8;
        const auto run_linear = [&](const Tensor& in, const Weight& weight, Tensor& result) {
            const std::size_t capacity = o_::linear_workspace_capacity_bytes(weight.qtype, weight.n, weight.k, policy, in.ne[1], in.ne[1]);
            ninfer::DeviceArena workspace(std::max<std::size_t>(capacity, 256));
            o_::linear(in, weight, result, policy, workspace, stream);
        };
        // A BF16 projection through the FP32 form, rounded to BF16 (the model does the same).
        const auto run_bf16_projection = [&](const char* field, int n, int k, const Tensor& in, Tensor& result) {
            const auto& weight = load(field);
            const Tensor W(weight.p, DType::BF16, {k, n});
            const Tensor* weights[] = {&W};
            Device wide(static_cast<std::size_t>(n) * in.ne[1] * 4);
            Tensor f32(wide.p, DType::FP32, {n, in.ne[1]});
            o_::projection_fp32(in, weights, f32, stream);
            o_::cast_fp32_to_bf16(f32, result, stream);
            t::cuda_check(cudaStreamSynchronize(stream), "projection");
        };

        const Tensor X(x.p, DType::BF16, {kHidden, T}), QW(q_norm_w->p, DType::BF16, {kQLora}),
            KW(kv_norm_w->p, DType::BF16, {kHeadDim}), POS(pos_d.p, DType::I32, {T}), FREQ(freq_d.p, DType::FP32, {kRope / 2}),
            IDX(rows_d.p, DType::I32, {K, T}), SINK(sink_d.p, DType::FP32, {kHeads});
        Tensor QA(qa.p, DType::BF16, {kQLora, T}), QN(qn.p, DType::BF16, {kQLora, T}), Q(q.p, DType::BF16, {kHeads * kHeadDim, T}),
            Q3(q.p, DType::BF16, {kHeadDim, kHeads, T}), KVR(kv_raw.p, DType::BF16, {kHeadDim, T}),
            KV(kv_table.p, DType::BF16, {kHeadDim, T}), KV3(kv_table.p, DType::BF16, {kHeadDim, 1, T}),
            KVALL(kv_table.p, DType::BF16, {kHeadDim, N}), O(o.p, DType::BF16, {kHeadDim, kHeads, T});

        // Query: q_a, norm, q_b, RoPE.
        run_linear(X, wq_a, QA);
        o_::rmsnorm(QA, QW, kEps, false, QN, stream);
        t::cuda_check(cudaStreamSynchronize(stream), "q norm");
        compare("qr", download_bf16_as_f32(qn.p, static_cast<std::size_t>(kQLora) * T), ref("qr"), 0.015);
        run_linear(QN, wq_b, Q);
        o_::rope_pairs(POS, FREQ, kRope, false, Q3, stream);
        t::cuda_check(cudaStreamSynchronize(stream), "q");
        compare("q", download_bf16_as_f32(q.p, static_cast<std::size_t>(kHeads) * kHeadDim * T), ref("q"), 0.02);

        // The shared key/value head (written straight into the first N columns of the KV table).
        run_linear(X, wkv, KVR);
        o_::rmsnorm(KVR, KW, kEps, false, KV, stream);
        o_::rope_pairs(POS, FREQ, kRope, false, KV3, stream);
        t::cuda_check(cudaStreamSynchronize(stream), "kv");
        compare("kv", download_bf16_as_f32(kv_table.p, static_cast<std::size_t>(kHeadDim) * T), ref("kv"), 0.015);

        if (ratio != 0) {
            // The compressor: two projections, the pooling softmax, the norm, RoPE at positions 0, r, 2r, ...
            const int cdim = kCompDim;
            Device kvp(static_cast<std::size_t>(cdim) * T * 2), gate(static_cast<std::size_t>(cdim) * T * 2),
                pooled(static_cast<std::size_t>(cdim) * G * 2), latent(static_cast<std::size_t>(cdim) * G * 2),
                comp_pos(static_cast<std::size_t>(G) * 4);
            Tensor KVP(kvp.p, DType::BF16, {cdim, T}), GATE(gate.p, DType::BF16, {cdim, T}), POOLED(pooled.p, DType::BF16, {cdim, G}),
                LATENT(latent.p, DType::BF16, {cdim, G});
            run_bf16_projection("comp_wkv", cdim, kHidden, X, KVP);
            run_bf16_projection("comp_wgate", cdim, kHidden, X, GATE);
            o_::softmax_pool(KVP, GATE, ratio, POOLED, stream);
            const auto comp_norm_w = norm_weight("comp_norm.bin");
            const Tensor CW(comp_norm_w->p, DType::BF16, {cdim});
            o_::rmsnorm(POOLED, CW, kEps, false, LATENT, stream);
            t::cuda_check(cudaStreamSynchronize(stream), "latent");
            compare("latent", download_bf16_as_f32(latent.p, static_cast<std::size_t>(cdim) * G), ref("latent"), 0.01);

            std::vector<std::int32_t> comp_positions(G);
            for (int g = 0; g < G; ++g) { comp_positions[g] = g * ratio; }
            upload(comp_pos, comp_positions.data(), comp_positions.size() * 4);
            const Tensor CPOS(comp_pos.p, DType::I32, {G});
            // The rotated compressed rows are the table's columns T .. T + G - 1.
            std::uint8_t* ckv_at = kv_table.bytes() + static_cast<std::size_t>(kHeadDim) * T * 2;
            t::cuda_check(cudaMemcpyAsync(ckv_at, latent.p, static_cast<std::size_t>(cdim) * G * 2, cudaMemcpyDeviceToDevice, stream), "ckv");
            Tensor CKV3(ckv_at, DType::BF16, {kHeadDim, 1, G});
            o_::rope_pairs(CPOS, FREQ, kRope, false, CKV3, stream);
            t::cuda_check(cudaStreamSynchronize(stream), "ckv rope");
            compare("ckv", download_bf16_as_f32(ckv_at, static_cast<std::size_t>(kHeadDim) * G), ref("ckv"), 0.01);

            // The indexer: its key (from the latent), query (from q_a's normed output) and head weights.
            const int ih = static_cast<int>(number("index_heads")), id = static_cast<int>(number("index_head_dim"));
            Device ik_raw(static_cast<std::size_t>(id) * G * 2), index_k(static_cast<std::size_t>(id) * G * 2),
                iq(static_cast<std::size_t>(id) * ih * T * 2), iw(static_cast<std::size_t>(ih) * T * 2),
                valid(static_cast<std::size_t>(T) * 4), scores(static_cast<std::size_t>(G) * T * 4),
                selected(static_cast<std::size_t>(topk) * T * 4);
            Tensor IKR(ik_raw.p, DType::BF16, {id, G}), IK(index_k.p, DType::BF16, {id, G}), IK3(index_k.p, DType::BF16, {id, 1, G}),
                IQ(iq.p, DType::BF16, {id * ih, T}), IQ3(iq.p, DType::BF16, {id, ih, T}), IW(iw.p, DType::BF16, {ih, T});
            run_bf16_projection("idx_wk", id, cdim, LATENT, IKR);
            const auto idx_norm_w = norm_weight("idx_k_norm.bin");
            const Tensor IKW(idx_norm_w->p, DType::BF16, {id});
            o_::rmsnorm(IKR, IKW, kEps, false, IK, stream);
            o_::rope_pairs(CPOS, FREQ, kRope, false, IK3, stream);
            run_bf16_projection("idx_wq_b", id * ih, kQLora, QN, IQ);
            o_::rope_pairs(POS, FREQ, kRope, false, IQ3, stream);
            run_bf16_projection("idx_weights_proj", ih, kHidden, X, IW);
            t::cuda_check(cudaStreamSynchronize(stream), "indexer inputs");
            compare("index_k", download_bf16_as_f32(index_k.p, static_cast<std::size_t>(id) * G), ref("index_k"), 0.01);

            std::vector<std::int32_t> valid_rows(T);
            for (int s = 0; s < T; ++s) { valid_rows[s] = (s + 1) / ratio; } // compressed rows complete at query s
            upload(valid, valid_rows.data(), valid_rows.size() * 4);
            const Tensor VALID(valid.p, DType::I32, {T});
            Tensor SCORES(scores.p, DType::FP32, {G, T}), SELECTED(selected.p, DType::I32, {topk, T});
            o_::weighted_relu_dot_scores(IQ3, IK, IW, VALID, static_cast<float>(std::pow(id, -0.5) * std::pow(ih, -0.5)), SCORES, stream);
            o_::top_k_positions(SCORES, VALID, topk, T, SELECTED, stream);
            t::cuda_check(cudaStreamSynchronize(stream), "selection");
            // The selected rows go after the window rows of each query's list.
            t::cuda_check(cudaMemcpy2D(rows_d.bytes() + static_cast<std::size_t>(list) * 4, static_cast<std::size_t>(K) * 4, selected.p,
                                       static_cast<std::size_t>(topk) * 4, static_cast<std::size_t>(topk) * 4, T, cudaMemcpyDeviceToDevice),
                          "selected rows");

            // Recall of the oracle's selection (a swap at the selection boundary is a rounding, not a fault).
            const auto want = read_vector<std::int32_t>(dir + "topk.bin");
            std::vector<std::int32_t> got(static_cast<std::size_t>(topk) * T);
            t::cuda_check(cudaMemcpy(got.data(), selected.p, got.size() * 4, cudaMemcpyDeviceToHost), "selection download");
            const int want_width = static_cast<int>(number("topk_width"));
            check(want_width == topk, "the oracle's selection has the same width");
            std::size_t hits = 0, total = 0;
            for (int s = 0; s < T; ++s) {
                // Only real selections count toward recall; the -1 padding must agree in number, and the
                // device's row must be strictly ascending (unique) over its real entries.
                std::set<std::int32_t> expected;
                std::size_t expected_pad = 0, got_pad = 0;
                for (int j = 0; j < want_width; ++j) {
                    const std::int32_t v = want[static_cast<std::size_t>(s) * want_width + j];
                    if (v >= 0) { expected.insert(v); } else { ++expected_pad; }
                }
                std::int32_t previous = -1;
                for (int j = 0; j < topk; ++j) {
                    const std::int32_t v = got[static_cast<std::size_t>(s) * topk + j];
                    if (v < 0) {
                        ++got_pad;
                        continue;
                    }
                    check(v > previous, "the selection is strictly ascending");
                    previous = v;
                    hits += expected.count(v) != 0 ? 1 : 0;
                }
                total += expected.size();
                check(got_pad == expected_pad, "the selection pads with -1 exactly where the oracle does");
            }
            const double recall = static_cast<double>(hits) / static_cast<double>(total);
            std::printf("%-10s recall of the oracle's selection %.5f over %zu entries\n", "topk", recall, total);
            check(recall >= 0.99, "the selection recalls at least 99 percent of the oracle's");
        }

        // Attention over each query's rows.
        o_::gathered_sink_attention(Q3, KVALL, IDX, SINK, 1.0F / std::sqrt(static_cast<float>(kHeadDim)), O, stream);
        t::cuda_check(cudaStreamSynchronize(stream), "attention");
        compare("o_pre", download_bf16_as_f32(o.p, static_cast<std::size_t>(kHeads) * kHeadDim * T), ref("o_pre"), 0.02);

        // Remove the rotation, then the grouped output projection: group g reads the 4096 features of
        // heads 8g..8g+7 and writes its 1024 outputs (rows [1024 g, 1024 (g + 1)) of out_a's weight).
        o_::rope_pairs(POS, FREQ, kRope, true, O, stream);
        t::cuda_check(cudaStreamSynchronize(stream), "o");
        compare("o", download_bf16_as_f32(o.p, static_cast<std::size_t>(kHeads) * kHeadDim * T), ref("o"), 0.02);
        const std::size_t row_bytes = static_cast<std::size_t>(kGroupIn / 256) * 176; // Q5_K: 176 bytes per 256 values
        for (int g = 0; g < kGroups; ++g) {
            t::cuda_check(cudaMemcpy2DAsync(group_in.p, kGroupIn * 2, o.bytes() + static_cast<std::size_t>(g) * kGroupIn * 2,
                                            static_cast<std::size_t>(kHeads) * kHeadDim * 2, kGroupIn * 2, T, cudaMemcpyDeviceToDevice, stream),
                          "gather group");
            const Weight group_weight = ggml_weight(QType::GGML_Q5_K, kGroupOut, kGroupIn,
                                                    wo_a_blocks.bytes() + static_cast<std::size_t>(g) * kGroupOut * row_bytes);
            const Tensor GI(group_in.p, DType::BF16, {kGroupIn, T});
            Tensor GO(group_out.p, DType::BF16, {kGroupOut, T});
            run_linear(GI, group_weight, GO);
            t::cuda_check(cudaMemcpy2DAsync(projected.bytes() + static_cast<std::size_t>(g) * kGroupOut * 2,
                                            static_cast<std::size_t>(kGroups) * kGroupOut * 2, group_out.p, kGroupOut * 2,
                                            kGroupOut * 2, T, cudaMemcpyDeviceToDevice, stream),
                          "scatter group");
        }
        t::cuda_check(cudaStreamSynchronize(stream), "projected");
        compare("projected", download_bf16_as_f32(projected.p, static_cast<std::size_t>(kGroups) * kGroupOut * T), ref("projected"), 0.02);

        const Tensor PR(projected.p, DType::BF16, {kGroups * kGroupOut, T});
        Tensor OUT(out.p, DType::BF16, {kHidden, T});
        run_linear(PR, wo_b, OUT);
        t::cuda_check(cudaStreamSynchronize(stream), "out");
        compare("out", download_bf16_as_f32(out.p, static_cast<std::size_t>(kHidden) * T), ref("out"), 0.02);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "EXCEPTION: %s\n", e.what());
        return 1;
    }
    if (failures != 0) {
        std::fprintf(stderr, "dsv4 attention layer: %d failure(s)\n", failures);
        return 1;
    }
    std::printf("dsv4 attention layer: passed\n");
    return 0;
}
