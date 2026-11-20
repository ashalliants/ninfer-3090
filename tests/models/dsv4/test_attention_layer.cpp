// Written for NInfer-3090.
//
// The attention sublayer library (src/models/dsv4/execution/attention.{h,cpp}) on a REAL
// DeepSeek-V4.1-Flash layer: the library runs it with persistent per-sequence state over any
// chunking of the sequence, and every chunking must agree with the NumPy oracle
// (tools/dsv4/attention.py) that tools/dsv4/export_layer_fixture.py ran on the same weights. It
// runs only when NINFER_DSV4_LAYER_FIXTURE names exporter output directories (a ':'-separated list;
// the weights are the real model's, which the repository does not hold; export a layer with
//   python3 tools/dsv4/export_layer_fixture.py FIRST_SHARD.gguf DIR --layer 0 --tokens 200
//   python3 tools/dsv4/export_layer_fixture.py FIRST_SHARD.gguf DIR --layer 2 --tokens 1100).
//
// Per fixture (a window-only layer 0, or a compressing layer that owns its compressed KV and
// indexer):
//   * the library's RoPE frequencies equal the exporter's;
//   * the whole sequence as one chunk matches the oracle's `out` (relative L2 within the ops test's
//     bound 0.02, which the device's BF16 and Q8_1 activation rounding explains: measured about
//     1%), the compressed rows and index keys within 0.01, the selection by recall;
//   * the same sequence in odd chunks (1, 7, 128, 129, a remainder) and then token by token for the
//     last 20 tokens, and in chunks of 3 (every chunk boundary falls inside a compressor group),
//     matches the oracle within the same bound and the single chunk's output within it, and the
//     compressed rows and index keys it leaves in the state are BIT-IDENTICAL to the single chunk's
//     (the compressor is exact across boundaries: projections do not depend on the batch width, the
//     pooling is per group).
// For a compressing layer also, with that layer's weights standing in for three layers (an index
// source that owns the compressed KV and is the candidate source, an index source reading its KV,
// and a layer reusing the selection), with a small candidate block count:
//   * the second layer's selection lies inside the first layer's candidate blocks and contains
//   every
//     member of the first layer's unmasked selection that lies in them;
//   * the third layer (same weights, no indexer: it reuses the second's selection) outputs bit for
//   bit
//     what the second does, whole sequence and chunked.

#include "artifact/reader.h"
#include "core/arena.h"
#include "core/device.h"
#include "core/weight.h"
#include "models/dsv4/config.h"
#include "models/dsv4/execution/attention.h"
#include "models/dsv4/model.h"
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

namespace t    = ninfer::test;
namespace dsv4 = ninfer::models::dsv4;
using ninfer::DType;
using ninfer::QType;
using ninfer::Tensor;
using ninfer::Weight;

constexpr int kHidden = 5120, kQLora = 1280;

int failures = 0;

void check(bool ok, const std::string& what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what.c_str());
        ++failures;
    }
}

struct Device {
    void* p = nullptr;

    explicit Device(std::size_t bytes) {
        t::cuda_check(cudaMalloc(&p, bytes ? bytes : 1), "cudaMalloc");
    }

    ~Device() { cudaFree(p); }

    Device(const Device&)            = delete;
    Device& operator=(const Device&) = delete;
};

std::vector<std::uint8_t> read_file(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) { throw std::runtime_error("cannot read " + path); }
    return std::vector<std::uint8_t>(std::istreambuf_iterator<char>(in),
                                     std::istreambuf_iterator<char>());
}

template <class T>
std::vector<T> read_vector(const std::string& path) {
    const auto raw = read_file(path);
    std::vector<T> v(raw.size() / sizeof(T));
    std::memcpy(v.data(), raw.data(), v.size() * sizeof(T));
    return v;
}

std::map<std::string, std::string> read_config(const std::string& path) {
    std::map<std::string, std::string> config;
    const auto raw = read_file(path);
    std::istringstream in(std::string(raw.begin(), raw.end()));
    std::string key, value;
    while (in >> key >> value) { config[key] = value; }
    return config;
}

std::vector<float> download_bf16(const void* device, std::size_t count) {
    std::vector<std::uint16_t> raw(count);
    t::cuda_check(cudaMemcpy(raw.data(), device, count * 2, cudaMemcpyDeviceToHost), "download");
    std::vector<float> out(count);
    for (std::size_t i = 0; i < count; ++i) { out[i] = t::bf16_to_f32(raw[i]); }
    return out;
}

std::vector<std::uint16_t> download_bits(const void* device, std::size_t count) {
    std::vector<std::uint16_t> raw(count);
    t::cuda_check(cudaMemcpy(raw.data(), device, count * 2, cudaMemcpyDeviceToHost), "download");
    return raw;
}

double relative_l2(const std::vector<float>& got, const std::vector<float>& ref) {
    if (got.size() != ref.size()) { return 1e9; }
    double num = 0, den = 0;
    for (std::size_t i = 0; i < got.size(); ++i) {
        if (!std::isfinite(got[i])) { return 1e9; }
        const double d = static_cast<double>(got[i]) - ref[i];
        num += d * d;
        den += static_cast<double>(ref[i]) * ref[i];
    }
    return std::sqrt(num / std::max(den, 1e-300));
}

void compare(const std::string& what, const std::vector<float>& got, const std::vector<float>& ref,
             double bound) {
    const double rel = relative_l2(got, ref);
    std::printf("%-58s relative L2 %.5f (bound %.5f)\n", what.c_str(), rel, bound);
    check(rel <= bound, what + ": relative L2 within its bound");
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

// One exported layer: its weights on the device as the library takes them, and the oracle's
// outputs.
struct Fixture {
    std::string dir;
    std::map<std::string, std::string> config;
    int ratio = 0, tokens = 0, index_topk = 0;
    bool kv_source = false,
         index_source =
             false; // a layer reading another layer's KV (kv_source 0) follows its source
    std::vector<std::unique_ptr<Device>> buffers;
    dsv4::AttentionLayerViews views;
    std::unique_ptr<Device> x;

    double number(const char* key) const { return std::stod(config.at(key)); }

    const void* upload(const std::vector<std::uint8_t>& raw) {
        buffers.push_back(std::make_unique<Device>(raw.size()));
        t::cuda_check(cudaMemcpy(buffers.back()->p, raw.data(), raw.size(), cudaMemcpyHostToDevice),
                      "upload");
        return buffers.back()->p;
    }

    const void* blocks(const char* field) { return upload(read_file(dir + field + ".bin")); }

    Tensor bf16_matrix(const char* field, int k, int n) {
        return Tensor(const_cast<void*>(blocks(field)), DType::BF16, {k, n});
    }

    Tensor bf16_vector(const char* file, int n) { // the exporter writes norms in FP32
        const auto v = read_vector<float>(dir + file);
        std::vector<std::uint16_t> bits(v.size());
        for (std::size_t i = 0; i < v.size(); ++i) { bits[i] = t::f32_to_bf16(v[i]); }
        std::vector<std::uint8_t> raw(bits.size() * 2);
        std::memcpy(raw.data(), bits.data(), raw.size());
        return Tensor(const_cast<void*>(upload(raw)), DType::BF16, {n});
    }
};

std::unique_ptr<Fixture> load_fixture(const std::string& dir) {
    auto f    = std::make_unique<Fixture>();
    f->dir    = dir + "/";
    f->config = read_config(f->dir + "config.txt");
    f->ratio  = static_cast<int>(f->number("ratio"));
    check(f->number("rope_dim") == 64 && (f->ratio == 0 || f->ratio == 1 || f->ratio == 2) &&
              f->number("window") == 128,
          "a supported layer (rope 64, window 128; ratio 0, 1 or 2)");
    f->kv_source = f->config.count("kv_source") != 0 ? f->number("kv_source") != 0 : f->ratio != 0;
    f->index_source =
        f->config.count("index_source") != 0 ? f->number("index_source") != 0 : f->ratio != 0;
    f->index_topk     = f->ratio != 0 ? static_cast<int>(f->number("index_topk")) : 0;
    const auto x_bits = read_file(f->dir + "x.bin");
    f->tokens         = static_cast<int>(x_bits.size() / 2 / kHidden);
    f->x              = std::make_unique<Device>(x_bits.size());
    t::cuda_check(cudaMemcpy(f->x->p, x_bits.data(), x_bits.size(), cudaMemcpyHostToDevice),
                  "upload x");
    auto& v    = f->views;
    v.q_a      = ggml_weight(QType::GGML_Q8_0, kQLora, kHidden, f->blocks("wq_a"));
    v.q_b      = ggml_weight(QType::GGML_Q5_K, 64 * 512, kQLora, f->blocks("wq_b"));
    v.kv       = ggml_weight(QType::GGML_Q8_0, 512, kHidden, f->blocks("wkv"));
    v.out_a    = ggml_weight(QType::GGML_Q5_K, 8 * 1024, 4096, f->blocks("wo_a"));
    v.out_b    = ggml_weight(QType::GGML_Q5_K, kHidden, 8 * 1024, f->blocks("wo_b"));
    v.q_a_norm = f->bf16_vector("q_norm.bin", kQLora);
    v.kv_norm  = f->bf16_vector("kv_norm.bin", 512);
    {
        const auto sink = read_file(f->dir + "attn_sink.bin");
        v.sinks         = Tensor(const_cast<void*>(f->upload(sink)), DType::FP32, {64});
    }
    if (f->kv_source) {
        dsv4::CompressorViews c;
        c.kv = f->bf16_matrix("comp_wkv", kHidden, 512);
        if (f->ratio > 1) { c.gate = f->bf16_matrix("comp_wgate", kHidden, 512); }
        c.norm         = f->bf16_vector("comp_norm.bin", 512);
        c.index_k      = f->bf16_matrix("idx_wk", 512, 128);
        c.index_k_norm = f->bf16_vector("idx_k_norm.bin", 128);
        v.compressor   = c;
    }
    if (f->index_source) {
        v.indexer = dsv4::IndexerViews{f->bf16_matrix("idx_wq_b", kQLora, 32 * 128),
                                       f->bf16_matrix("idx_weights_proj", kHidden, 32)};
    }
    return f;
}

// V4.1-Flash's attention geometry with one entry per layer given.
dsv4::TextConfig make_config(const std::vector<dsv4::LayerPlan>& plans,
                             const std::vector<std::uint32_t>& owners) {
    dsv4::TextConfig c;
    c.hidden_size             = kHidden;
    c.num_hidden_layers       = static_cast<std::uint32_t>(plans.size());
    c.max_position_embeddings = 1048576;
    c.rms_norm_eps            = 1e-20F; // the GGUF's attention.layer_norm_rms_epsilon, as FP32
    c.heads                   = 64;
    c.head_dim                = 512;
    c.rope_dim                = 64;
    c.q_lora_rank             = kQLora;
    c.o_groups                = 8;
    c.o_lora_rank             = 1024;
    c.window                  = 128;
    c.rope_theta              = 10000.0F;
    c.compress_rope_theta     = 160000.0F;
    c.yarn                    = {16.0F, 65536, 32.0F, 1.0F};
    c.indexer_heads           = 32;
    c.indexer_head_dim        = 128;
    c.indexer_top_k           = 512;
    c.layers                  = plans;
    c.kv_owner                = owners;
    c.index_owner             = owners;
    return c;
}

struct RunResult {
    std::vector<float> out;                // BF16 [hidden, T] as float
    std::vector<std::uint16_t> compressed; // [512, groups] bits
    std::vector<std::uint16_t> index_keys; // [128, groups] bits
    std::vector<std::int32_t>
        topk; // the last chunk's selection [indexer_top_k, chunk] (offset 0, -1 padded)
};

// Chunk sizes covering `tokens`: the odd sizes while they fit, one remainder chunk, then `decode`
// single tokens.
std::vector<int> odd_plan(int tokens, int decode) {
    std::vector<int> sizes;
    int used = 0;
    for (const int s : {1, 7, 128, 129}) {
        if (used + s >= tokens - decode) { break; }
        sizes.push_back(s);
        used += s;
    }
    if (tokens - decode - used > 0) { sizes.push_back(tokens - decode - used); }
    for (int i = 0; i < decode; ++i) { sizes.push_back(1); }
    return sizes;
}

std::vector<int> threes_plan(int tokens, int decode) {
    std::vector<int> sizes;
    int used = 0;
    while (used + 3 <= tokens - decode) {
        sizes.push_back(3);
        used += 3;
    }
    if (tokens - decode - used > 0) { sizes.push_back(tokens - decode - used); }
    for (int i = 0; i < decode; ++i) { sizes.push_back(1); }
    return sizes;
}

// Runs layers 0..views.size()-1 of `config` over x in the chunking `sizes`; returns layer
// `report`'s output and the first layer's compressed state.
RunResult run_layers(const dsv4::TextConfig& config,
                     const std::vector<const dsv4::AttentionLayerViews*>& views, const Device& x,
                     int tokens, const std::vector<int>& sizes, std::size_t report,
                     cudaStream_t stream, std::vector<std::vector<float>>* all_outputs = nullptr) {
    const int widest = *std::max_element(sizes.begin(), sizes.end());
    const dsv4::AttentionLimits limits{static_cast<std::uint32_t>(tokens),
                                       static_cast<std::uint32_t>(widest)};
    dsv4::AttentionState state(config, limits);
    ninfer::DeviceArena work(dsv4::attention_workspace_bytes(config, limits));
    std::vector<std::unique_ptr<Device>> outs;
    for (std::size_t l = 0; l < views.size(); ++l) {
        outs.push_back(std::make_unique<Device>(static_cast<std::size_t>(kHidden) * tokens * 2));
    }
    int start = 0;
    for (const int n : sizes) {
        const Tensor chunk(static_cast<std::uint8_t*>(x.p) +
                               static_cast<std::size_t>(start) * kHidden * 2,
                           DType::BF16, {kHidden, n});
        for (std::size_t l = 0; l < views.size(); ++l) {
            Tensor out(static_cast<std::uint8_t*>(outs[l]->p) +
                           static_cast<std::size_t>(start) * kHidden * 2,
                       DType::BF16, {kHidden, n});
            dsv4::attention_layer(config, static_cast<std::uint32_t>(l), *views[l], state, chunk,
                                  start, out, work, stream);
        }
        start += n;
    }
    t::cuda_check(cudaStreamSynchronize(stream), "run");
    check(start == tokens, "the plan covers the sequence");
    RunResult r;
    r.out = download_bf16(outs[report]->p, static_cast<std::size_t>(kHidden) * tokens);
    if (all_outputs != nullptr) {
        for (auto& o : outs) {
            all_outputs->push_back(download_bf16(o->p, static_cast<std::size_t>(kHidden) * tokens));
        }
    }
    if (config.layers[0].compress_ratio != 0) {
        const int groups = tokens / static_cast<int>(config.layers[0].compress_ratio);
        r.compressed =
            download_bits(state.compressed_rows(0).data, static_cast<std::size_t>(512) * groups);
        r.index_keys =
            download_bits(state.index_keys(0).data, static_cast<std::size_t>(128) * groups);
        const Tensor topk = state.shared_topk();
        r.topk.resize(static_cast<std::size_t>(topk.ne[0]) * sizes.back());
        t::cuda_check(
            cudaMemcpy(r.topk.data(), topk.data, r.topk.size() * 4, cudaMemcpyDeviceToHost),
            "topk");
    }
    return r;
}

// A decode step (T = 1) captured into a CUDA graph and launched gives the bits of the same step run
// eagerly.
void test_graph_step(Fixture& f, const dsv4::TextConfig& config, cudaStream_t stream) {
    const int T = f.tokens;
    const dsv4::AttentionLimits limits{static_cast<std::uint32_t>(T),
                                       static_cast<std::uint32_t>(T - 1)};
    const auto step = [&](bool capture) {
        dsv4::AttentionState state(config, limits);
        ninfer::DeviceArena work(dsv4::attention_workspace_bytes(config, limits));
        Device out(static_cast<std::size_t>(kHidden) * T * 2);
        const Tensor prefix(f.x->p, DType::BF16, {kHidden, T - 1});
        Tensor prefix_out(out.p, DType::BF16, {kHidden, T - 1});
        dsv4::attention_layer(config, 0, f.views, state, prefix, 0, prefix_out, work, stream);
        const Tensor token(static_cast<std::uint8_t*>(f.x->p) +
                               static_cast<std::size_t>(T - 1) * kHidden * 2,
                           DType::BF16, {kHidden, 1});
        Tensor token_out(static_cast<std::uint8_t*>(out.p) +
                             static_cast<std::size_t>(T - 1) * kHidden * 2,
                         DType::BF16, {kHidden, 1});
        t::cuda_check(cudaStreamSynchronize(stream), "prefix");
        if (capture) {
            cudaGraph_t graph    = nullptr;
            cudaGraphExec_t exec = nullptr;
            t::cuda_check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal),
                          "begin capture");
            dsv4::attention_layer(config, 0, f.views, state, token, T - 1, token_out, work, stream);
            t::cuda_check(cudaStreamEndCapture(stream, &graph), "end capture");
            t::cuda_check(cudaGraphInstantiate(&exec, graph, 0), "instantiate");
            t::cuda_check(cudaGraphLaunch(exec, stream), "launch");
            t::cuda_check(cudaStreamSynchronize(stream), "graph");
            cudaGraphExecDestroy(exec);
            cudaGraphDestroy(graph);
        } else {
            dsv4::attention_layer(config, 0, f.views, state, token, T - 1, token_out, work, stream);
            t::cuda_check(cudaStreamSynchronize(stream), "eager");
        }
        return download_bits(token_out.data, kHidden);
    };
    check(step(false) == step(true),
          "a graph-captured decode step equals the eager step bit for bit");
}

void test_fixture(const std::string& dir, cudaStream_t stream) {
    auto f = load_fixture(dir);
    std::printf("== %s: ratio %d, %d tokens\n", dir.c_str(), f->ratio, f->tokens);
    dsv4::LayerPlan plan;
    plan.compress_ratio = static_cast<std::uint32_t>(f->ratio);
    plan.kv_source = plan.index_source = f->ratio != 0;
    dsv4::TextConfig config            = make_config({plan}, {0});
    if (f->ratio == 1) {
        config.candidate_layer = 0, config.candidate_blocks = 2048, config.candidate_block_size = 8;
    } // covers every test position
    const int T = f->tokens;

    // The frequencies the exporter ran the oracle with.
    const auto want_freq = read_vector<float>(f->dir + "inv_freq.bin");
    const auto got_freq  = dsv4::rope_inv_freqs(config, 0);
    check(want_freq.size() == got_freq.size(), "rope_inv_freqs has rope_dim / 2 entries");
    double worst_freq = 0;
    for (std::size_t i = 0; i < std::min(want_freq.size(), got_freq.size()); ++i) {
        worst_freq = std::max<double>(worst_freq, std::fabs(got_freq[i] - want_freq[i]) /
                                                      std::max(std::fabs(want_freq[i]), 1e-30F));
    }
    std::printf("rope_inv_freqs: worst relative difference from the exporter's %.3g\n", worst_freq);
    check(worst_freq < 1e-6, "rope_inv_freqs equals the exporter's frequencies");

    const auto ref_out = read_vector<float>(f->dir + "out.bin");
    const std::vector<const dsv4::AttentionLayerViews*> views{&f->views};
    const RunResult whole = run_layers(config, views, *f->x, T, {T}, 0, stream);
    compare("single chunk vs oracle: out", whole.out, ref_out, 0.02);
    if (f->ratio != 0) {
        const int groups = T / f->ratio;
        std::vector<float> ckv(static_cast<std::size_t>(512) * groups),
            ik(static_cast<std::size_t>(128) * groups);
        for (std::size_t i = 0; i < ckv.size(); ++i) {
            ckv[i] = t::bf16_to_f32(whole.compressed[i]);
        }
        for (std::size_t i = 0; i < ik.size(); ++i) { ik[i] = t::bf16_to_f32(whole.index_keys[i]); }
        compare("single chunk vs oracle: compressed rows", ckv,
                read_vector<float>(f->dir + "ckv.bin"), 0.01);
        compare("single chunk vs oracle: index keys", ik,
                read_vector<float>(f->dir + "index_k.bin"), 0.01);
        // The selection by recall: the oracle's rows list positions offset by the raw row count T
        // (-1 invalid).
        const auto want  = read_vector<std::int32_t>(f->dir + "topk.bin");
        const int width  = static_cast<int>(f->number("topk_width")),
                  k      = static_cast<int>(whole.topk.size()) / T;
        std::size_t hits = 0, total = 0;
        for (int s = 0; s < T; ++s) {
            std::set<std::int32_t> expected;
            for (int j = 0; j < width; ++j) {
                if (want[static_cast<std::size_t>(s) * width + j] >= 0) {
                    expected.insert(want[static_cast<std::size_t>(s) * width + j] - T);
                }
            }
            for (int j = 0; j < k; ++j) {
                const std::int32_t v = whole.topk[static_cast<std::size_t>(s) * k + j];
                if (v < 0) { continue; }
                ++total;
                hits += expected.count(v) != 0 ? 1 : 0;
            }
        }
        const double recall =
            total != 0 ? static_cast<double>(hits) / static_cast<double>(total) : 0.0;
        std::printf("%-58s recall %.5f over %zu entries\n", "single chunk vs oracle: selection",
                    recall, total);
        check(total > 0 && recall >= 0.99,
              "the selection recalls at least 99 percent of the oracle's");
    }

    const std::vector<std::pair<std::string, std::vector<int>>> plans{
        {"odd chunks, then 20 decode steps", odd_plan(T, 20)},
        {"chunks of 3, then 20 decode steps", threes_plan(T, 20)}};
    for (const auto& [label, sizes] : plans) {
        const RunResult r = run_layers(config, views, *f->x, T, sizes, 0, stream);
        compare(label + " vs oracle: out", r.out, ref_out, 0.02);
        compare(label + " vs single chunk: out", r.out, whole.out, 0.02);
        if (f->ratio != 0) {
            check(r.compressed == whole.compressed,
                  label + ": the compressed rows are bit-identical to the single chunk's");
            check(r.index_keys == whole.index_keys,
                  label + ": the index keys are bit-identical to the single chunk's");
        }
    }
    test_graph_step(*f, config, stream);
}

// With NINFER_TEST_DSV4_ARTIFACT (a converted artifact): bind_attention_views over the real model's
// parameters of a window-only layer, a kv + index source, an index source reading its KV, and a
// ratio-1 layer reading its KV, and run each over 8 random tokens (finite output): the library
// takes the Program's own bound views.
void test_bound_views(const char* path, cudaStream_t stream) {
    ninfer::artifact::Reader reader(path);
    auto plan = dsv4::plan_load(reader, true);
    ninfer::DeviceContext device(0);
    auto model                = dsv4::materialize_model(std::move(plan), device);
    const dsv4::TextConfig& c = model->config();
    const int T               = 8;
    std::vector<std::uint16_t> bits(static_cast<std::size_t>(kHidden) * T);
    std::uint32_t seed = 7;
    for (auto& b : bits) {
        seed = seed * 1664525U + 1013904223U;
        b    = t::f32_to_bf16(static_cast<float>(static_cast<std::int32_t>(seed >> 8) - (1 << 23)) /
                              static_cast<float>(1 << 23));
    }
    Device x(bits.size() * 2), out(bits.size() * 2);
    t::cuda_check(cudaMemcpy(x.p, bits.data(), bits.size() * 2, cudaMemcpyHostToDevice),
                  "upload x");
    const dsv4::AttentionLimits limits{64, static_cast<std::uint32_t>(T)};
    dsv4::AttentionState state(c, limits);
    ninfer::DeviceArena work(dsv4::attention_workspace_bytes(c, limits));
    std::printf(
        "== the real artifact's bound views (state %.1f MB for %u layers at 64 positions)\n",
        double(state.device_bytes()) / 1e6, c.num_hidden_layers);
    for (std::uint32_t layer = 0; layer <= 21;
         ++layer) { // every layer up to 21 in order: kv owners and index sources precede their
                    // readers
        const auto binding =
            dsv4::bind_attention_views(*model, model->weights().layers[layer].attention, stream);
        const auto& views = binding.views;
        check(views.q_a.n == kQLora && views.q_a.k == kHidden && views.out_a.n == 8192 &&
                  views.out_b.n == kHidden,
              "bound GGML shapes");
        check(views.q_a_norm.dtype == DType::BF16 && views.sinks.dtype == DType::FP32 &&
                  views.sinks.ne[0] == 64,
              "bound norm and sink types");
        const Tensor in(x.p, DType::BF16, {kHidden, T});
        Tensor result(out.p, DType::BF16, {kHidden, T});
        dsv4::attention_layer(c, layer, views, state, in, 0, result, work, stream);
        t::cuda_check(cudaStreamSynchronize(stream), "layer");
        const auto values = download_bf16(out.p, bits.size());
        bool finite       = true;
        for (const float v : values) { finite = finite && std::isfinite(v); }
        check(finite, "layer " + std::to_string(layer) + " gives a finite output");
    }
    std::printf("bound views of layers 0-21 ran (ratio 0, 2 and 1; sources and readers)\n");
}

// Which positions the selection of column s lists (-1 padding dropped).
std::set<std::int32_t> column_set(const std::vector<std::int32_t>& v, int width, int s) {
    std::set<std::int32_t> out;
    for (int i = 0; i < width; ++i) {
        if (v[static_cast<std::size_t>(s) * width + i] >= 0) {
            out.insert(v[static_cast<std::size_t>(s) * width + i]);
        }
    }
    return out;
}

// Three layers with the compressing fixture's weights: A (kv + index source, candidate source), B
// (index source reading A's KV) and C (reads B's selection).
void test_shared_layers(const std::string& dir, cudaStream_t stream) {
    auto f = load_fixture(dir);
    if (f->ratio == 0) { return; }
    std::printf(
        "== %s: candidate blocks and a shared selection (three layers on one set of weights)\n",
        dir.c_str());
    const int T = std::min(f->tokens, 400);
    dsv4::LayerPlan a, b, c;
    a.compress_ratio = b.compress_ratio = c.compress_ratio = static_cast<std::uint32_t>(f->ratio);
    a.kv_source = a.index_source = true;
    b.index_source               = true;
    dsv4::TextConfig config      = make_config({a, b, c}, {0, 0, 0});
    config.indexer_top_k =
        32; // small selections and a small candidate set, so the mask bites at 400 tokens
    config.candidate_layer       = 0;
    config.candidate_blocks      = 24;
    config.candidate_block_size  = 4;
    dsv4::AttentionLayerViews va = f->views, vb = f->views, vc = f->views;
    vb.compressor.reset();
    vc.compressor.reset();
    vc.indexer.reset();

    // One chunk: look at the selections between layers by running the layers one at a time.
    {
        const dsv4::AttentionLimits limits{static_cast<std::uint32_t>(T),
                                           static_cast<std::uint32_t>(T)};
        dsv4::AttentionState state(config, limits);
        ninfer::DeviceArena work(dsv4::attention_workspace_bytes(config, limits));
        Device out_a(static_cast<std::size_t>(kHidden) * T * 2),
            out_b(static_cast<std::size_t>(kHidden) * T * 2),
            out_c(static_cast<std::size_t>(kHidden) * T * 2);
        const Tensor x(f->x->p, DType::BF16, {kHidden, T});
        Tensor oa(out_a.p, DType::BF16, {kHidden, T}), ob(out_b.p, DType::BF16, {kHidden, T}),
            oc(out_c.p, DType::BF16, {kHidden, T});
        const auto fetch = [&](const Tensor& tensor, int rows) {
            std::vector<std::int32_t> v(static_cast<std::size_t>(rows) * T);
            for (int s = 0; s < T; ++s) {
                t::cuda_check(cudaMemcpy(v.data() + static_cast<std::size_t>(s) * rows,
                                         static_cast<const std::uint8_t*>(tensor.data) +
                                             static_cast<std::size_t>(s) * tensor.ne[0] * 4,
                                         static_cast<std::size_t>(rows) * 4,
                                         cudaMemcpyDeviceToHost),
                              "fetch");
            }
            return v;
        };
        dsv4::attention_layer(config, 0, va, state, x, 0, oa, work, stream);
        t::cuda_check(cudaStreamSynchronize(stream), "layer A");
        const auto unmasked = fetch(state.shared_topk(), 32);
        const auto blocks   = fetch(state.candidate_blocks(), 24);
        dsv4::attention_layer(config, 1, vb, state, x, 0, ob, work, stream);
        t::cuda_check(cudaStreamSynchronize(stream), "layer B");
        const auto masked = fetch(state.shared_topk(), 32);
        dsv4::attention_layer(config, 2, vc, state, x, 0, oc, work, stream);
        t::cuda_check(cudaStreamSynchronize(stream), "layer C");

        bool inside = true, complete = true, masking_bit = false;
        int checked = 0;
        for (int s = 0; s < T; ++s) {
            const int reach    = (s + 1) / f->ratio; // positions reachable
            const auto kept    = column_set(blocks, 24, s);
            int kept_reachable = 0;
            for (int p = 0; p < reach; ++p) { kept_reachable += kept.count(p / 4) != 0 ? 1 : 0; }
            const auto first = column_set(unmasked, 32, s), second = column_set(masked, 32, s);
            if (kept_reachable < 32) {
                continue;
            } // too few survive: the model keeps masked positions to fill the selection
            ++checked;
            for (const auto p : second) { inside = inside && kept.count(p / 4) != 0; }
            for (const auto p : first) {
                if (kept.count(p / 4) != 0) { complete = complete && second.count(p) != 0; }
                masking_bit = masking_bit || kept.count(p / 4) == 0;
            }
        }
        std::printf("candidate blocks: %d queries checked; the mask changed a selection: %s\n",
                    checked, masking_bit ? "yes" : "no");
        check(checked > 100,
              "enough queries have at least 32 reachable positions inside the candidate blocks");
        check(masking_bit,
              "the candidate mask removes some of the unmasked selection (the test bites)");
        check(inside, "the second index layer's selection lies inside the candidate blocks");
        check(complete, "the second index layer keeps every unmasked pick that lies inside the "
                        "candidate blocks");
        check(download_bits(out_b.p, static_cast<std::size_t>(kHidden) * T) ==
                  download_bits(out_c.p, static_cast<std::size_t>(kHidden) * T),
              "a layer reusing the selection outputs bit for bit what the index layer with its "
              "weights does");
    }

    // Chunked, with a decode tail: the layer reusing the selection still equals the one that made
    // it.
    {
        std::vector<std::vector<float>> outs;
        const std::vector<const dsv4::AttentionLayerViews*> views{&va, &vb, &vc};
        const std::vector<int> plan = odd_plan(T, 20);
        (void)run_layers(config, views, *f->x, T, plan, 2, stream, &outs);
        check(outs[1] == outs[2], "chunked: the reusing layer equals the index layer bit for bit");
        std::vector<std::vector<float>> whole;
        (void)run_layers(config, views, *f->x, T, {T}, 2, stream, &whole);
        compare("chunked vs single chunk, layer B (candidate mask active)", outs[1], whole[1],
                0.05);
    }
}

// A ratio-1 kv + index source (the candidate layer, whose 2048 blocks of 8 cover every position
// here) followed by a layer reading its compressed KV and selection: each layer against its own
// oracle output, any chunking.
void test_pair(const std::string& source_dir, const std::string& reader_dir, cudaStream_t stream) {
    auto a = load_fixture(source_dir), b = load_fixture(reader_dir);
    std::printf("== %s then %s (ratio %d)\n", source_dir.c_str(), reader_dir.c_str(), a->ratio);
    check(a->tokens == b->tokens && a->ratio == b->ratio && a->kv_source && !b->kv_source,
          "a source and a layer reading its KV");
    dsv4::LayerPlan source, reader;
    source.compress_ratio = reader.compress_ratio = static_cast<std::uint32_t>(a->ratio);
    source.kv_source = source.index_source = true;
    dsv4::TextConfig config                = make_config({source, reader}, {0, 0});
    if (a->ratio == 1) {
        config.candidate_layer = 0, config.candidate_blocks = 2048, config.candidate_block_size = 8;
    }
    const int T = a->tokens;
    const std::vector<const dsv4::AttentionLayerViews*> views{&a->views, &b->views};
    const auto ref_a = read_vector<float>(a->dir + "out.bin"),
               ref_b = read_vector<float>(b->dir + "out.bin");
    std::vector<std::vector<float>> whole;
    const RunResult single = run_layers(config, views, *a->x, T, {T}, 0, stream, &whole);
    compare("source, single chunk vs oracle: out", whole[0], ref_a, 0.02);
    compare("reader, single chunk vs oracle: out", whole[1], ref_b, 0.02);
    const std::vector<std::pair<std::string, std::vector<int>>> plans{
        {"odd chunks, then 20 decode steps", odd_plan(T, 20)},
        {"chunks of 3, then 20 decode steps", threes_plan(T, 20)}};
    for (const auto& [label, sizes] : plans) {
        std::vector<std::vector<float>> outs;
        const RunResult r = run_layers(config, views, *a->x, T, sizes, 0, stream, &outs);
        compare("source, " + label + " vs oracle: out", outs[0], ref_a, 0.02);
        compare("reader, " + label + " vs oracle: out", outs[1], ref_b, 0.02);
        compare("reader, " + label + " vs single chunk: out", outs[1], whole[1], 0.02);
        check(r.compressed == single.compressed && r.index_keys == single.index_keys,
              label +
                  ": the compressed rows and index keys are bit-identical to the single chunk's");
    }
}

} // namespace

int main() {
    try {
        const char* env = std::getenv("NINFER_DSV4_LAYER_FIXTURE");
        if (env == nullptr) {
            std::printf("SKIP: NINFER_DSV4_LAYER_FIXTURE is not set (export layers with "
                        "tools/dsv4/export_layer_fixture.py)\n");
            return 0;
        }
        if (t::cuda_unavailable()) {
            std::printf("SKIP: no CUDA device\n");
            return 77;
        }
        cudaStream_t stream = nullptr;
        t::cuda_check(cudaStreamCreate(&stream), "stream");
        if (const char* artifact = std::getenv("NINFER_TEST_DSV4_ARTIFACT")) {
            test_bound_views(artifact, stream);
        }
        std::istringstream list(env);
        std::string dir;
        std::vector<std::string> dirs;
        while (std::getline(list, dir, ':')) {
            if (!dir.empty()) { dirs.push_back(dir); }
        }
        for (std::size_t i = 0; i < dirs.size(); ++i) {
            const auto config = read_config(dirs[i] + "/config.txt");
            if (config.count("kv_source") != 0 &&
                config.at("kv_source") == "0") { // a reader follows its source in the list
                check(i > 0, "a layer reading another's KV needs its source before it in "
                             "NINFER_DSV4_LAYER_FIXTURE");
                if (i > 0) { test_pair(dirs[i - 1], dirs[i], stream); }
                continue;
            }
            test_fixture(dirs[i], stream);
            if (config.count("ratio") != 0 && config.at("ratio") == "2") {
                test_shared_layers(dirs[i], stream);
            }
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "EXCEPTION: %s\n", e.what());
        return 1;
    }
    if (failures != 0) {
        std::fprintf(stderr, "dsv4 attention layer library: %d failure(s)\n", failures);
        return 1;
    }
    std::printf("dsv4 attention layer library: passed\n");
    return 0;
}
