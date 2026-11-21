// Written for NInfer-3090.
//
// Inkling-2-Mini's opt-in adapters (src/models/inkling2/program/adapters.h) on the real artifact and the two published
// adapter files:
//
//  1. Loading: the files load; unsupported forms (a GGUF graft, an ablation of another scope, a DoRA config, a LoRA
//     on another module) are refused with their reason.
//  2. The Ops against FP64 (low_rank_update_fp32 with the adapter's real A and B; direction_ablate_fp32 with its real
//     directions, with a residual and in place).
//  3. The LoRA's effective wo_ud: rows of projection_fp32 + low_rank_update_fp32 over the identity equal
//     base + scale B A (FP64) to FP32 rounding, and rounded to BF16 agree with BF16(base + scale B A): the rows the
//     publisher's merged checkpoint holds.
//  4. The program: an ablation at one visit changes that visit's attention write (inside a block and at a block's
//     first layer, where the write is ffn_inp itself) and MLP write as the FP64 reference of the base run's write says;
//     a program whose adapters were cleared gives the base logits bit for bit; with both adapters on, the decode graph
//     gives the eager forward's logits bit for bit.
//  5. Report (not judged): with NINFER_INKLING2_KLD_TEXT (a text file), the KLD of each adapter's next-token
//     distribution from the base model's, and their top-1 agreement, over its first NINFER_INKLING2_KLD_TOKENS (2048).
//
// Environment: NINFER_TEST_INKLING2_ARTIFACT, NINFER_INKLING2_LORA (the PEFT directory), NINFER_INKLING2_ABLATION
// (the safetensors file); skips (77) without them.

#include "artifact/reader.h"
#include "core/arena.h"
#include "core/device.h"
#include "core/tensor.h"
#include "models/inkling2/frontend/frontend.h"
#include "models/inkling2/model.h"
#include "models/inkling2/program/adapters.h"
#include "models/inkling2/program/text_program.h"
#include "ninfer/ops/inkling2_adapters.h"
#include "ninfer/ops/projection_fp32.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

namespace ink = ninfer::models::inkling2;
namespace fs  = std::filesystem;
using ninfer::DeviceBuffer;
using ninfer::DType;
using ninfer::Tensor;

int failures = 0;

void check(bool ok, const std::string& what) {
    std::printf("%s: %s\n", ok ? "ok  " : "FAIL", what.c_str());
    if (!ok) { ++failures; }
}

void cuda(cudaError_t e, const char* what) {
    if (e != cudaSuccess) { throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e)); }
}

std::uint16_t to_bf16(float f) {
    std::uint32_t u;
    std::memcpy(&u, &f, 4);
    return static_cast<std::uint16_t>((u + 0x7FFFU + ((u >> 16) & 1U)) >> 16);
}
float from_bf16(std::uint16_t h) {
    const std::uint32_t u = static_cast<std::uint32_t>(h) << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

std::vector<float> download(const Tensor& t, cudaStream_t stream) {
    std::vector<float> out(static_cast<std::size_t>(t.ne[0]) * t.ne[1]);
    cuda(cudaMemcpy2DAsync(out.data(), static_cast<std::size_t>(t.ne[0]) * 4, t.data, static_cast<std::size_t>(t.nb[1]),
                           static_cast<std::size_t>(t.ne[0]) * 4, static_cast<std::size_t>(t.ne[1]), cudaMemcpyDeviceToHost, stream),
         "download");
    cuda(cudaStreamSynchronize(stream), "sync");
    return out;
}

// The safetensors bytes of F32 tensors and a metadata object (for the refusal cases).
std::string safetensors(const std::vector<std::pair<std::string, std::vector<std::uint64_t>>>& tensors, const std::string& metadata) {
    std::string header = "{";
    std::uint64_t offset = 0;
    for (const auto& [name, shape] : tensors) {
        std::uint64_t count = 1;
        std::string dims;
        for (const auto d : shape) { count *= d, dims += (dims.empty() ? "" : ",") + std::to_string(d); }
        header += "\"" + name + "\":{\"dtype\":\"F32\",\"shape\":[" + dims + "],\"data_offsets\":[" + std::to_string(offset) + "," +
                  std::to_string(offset + 4 * count) + "]},";
        offset += 4 * count;
    }
    header += "\"__metadata__\":" + metadata + "}";
    std::string out(8, '\0');
    const std::uint64_t n = header.size();
    for (int i = 0; i < 8; ++i) { out[static_cast<std::size_t>(i)] = static_cast<char>((n >> (8 * i)) & 0xFF); }
    out += header;
    std::vector<float> zeros(offset / 4, 0.25F);
    out.append(reinterpret_cast<const char*>(zeros.data()), offset);
    return out;
}

void write(const fs::path& path, const std::string& bytes) {
    std::ofstream out(path, std::ios::binary);
    out << bytes;
    if (!out) { throw std::runtime_error("cannot write " + path.string()); }
}

template <class F> bool refused(F&& f, const std::string& needle) {
    try {
        f();
    } catch (const std::invalid_argument& error) {
        const bool found = std::string(error.what()).find(needle) != std::string::npos;
        if (!found) { std::printf("      (refused, but: %s)\n", error.what()); }
        return found;
    }
    return false;
}

double max_rel(const std::vector<float>& got, const std::vector<double>& want) {
    double err = 0, scale = 1e-30;
    for (std::size_t i = 0; i < want.size(); ++i) {
        err   = std::max(err, std::abs(double(got[i]) - want[i]));
        scale = std::max(scale, std::abs(want[i]));
    }
    return err / scale;
}

// Per-column softmax over the first `valid` rows of [rows, cols] logits, as log-probabilities.
void log_softmax(const float* col, std::int32_t valid, std::vector<double>& out) {
    out.resize(static_cast<std::size_t>(valid));
    double m = -1e300;
    for (std::int32_t v = 0; v < valid; ++v) { m = std::max(m, double(col[v])); }
    double s = 0;
    for (std::int32_t v = 0; v < valid; ++v) { s += std::exp(double(col[v]) - m); }
    const double z = m + std::log(s);
    for (std::int32_t v = 0; v < valid; ++v) { out[static_cast<std::size_t>(v)] = double(col[v]) - z; }
}

} // namespace

int main() {
    const char* artifact_path = std::getenv("NINFER_TEST_INKLING2_ARTIFACT");
    const char* lora_path     = std::getenv("NINFER_INKLING2_LORA");
    const char* ablation_path = std::getenv("NINFER_INKLING2_ABLATION");
    if (artifact_path == nullptr || lora_path == nullptr || ablation_path == nullptr) {
        std::printf("SKIP: set NINFER_TEST_INKLING2_ARTIFACT, NINFER_INKLING2_LORA and NINFER_INKLING2_ABLATION\n");
        return 77;
    }
    try {
        ninfer::artifact::Reader reader(artifact_path);
        const ink::TextConfig config = ink::parse_config(reader.directory());
        const std::int32_t h = static_cast<std::int32_t>(config.hidden_size), q = static_cast<std::int32_t>(config.heads * config.head_dim);

        // 1. Loading and refusals.
        const ink::LoraAdapter lora = ink::load_lora_adapter({"ou", lora_path}, config);
        const ink::VisitAblation ablation = ink::load_visit_ablation({"sd", ablation_path}, config);
        std::vector<std::uint32_t> adapted;
        for (std::uint32_t l = 0; l < lora.layers.size(); ++l) {
            if (lora.layers[l]) { adapted.push_back(l); }
        }
        std::printf("lora: rank %d alpha %g scale %g, layers %u..%u (%zu)\n", lora.rank, lora.alpha, lora.scale, adapted.front(), adapted.back(),
                    adapted.size());
        const auto nonzero = [](const std::vector<float>& w) { return std::count_if(w.begin(), w.end(), [](float x) { return x != 0; }); };
        std::printf("ablation: %u visits, %td attention and %td MLP weights nonzero\n", ablation.visits, nonzero(ablation.attn_w),
                    nonzero(ablation.mlp_w));
        check(lora.rank == 2 && lora.scale == 1.0F && adapted.size() == 20 && adapted.front() == 4 && adapted.back() == 23,
              "the OpenUncensored LoRA: r 2, alpha 2, wo_ud of layers 4..23");
        check(ablation.visits == config.num_layer_visits, "the ablation has one direction pair per visit");
        {
            const fs::path tmp = fs::temp_directory_path() / ("ninfer-inkling2-adapters-" + std::to_string(std::random_device{}()));
            fs::create_directories(tmp);
            write(tmp / "cv.gguf", std::string("GGUF\3\0\0\0", 8) + std::string(64, '\0'));
            check(refused([&] { (void)ink::load_visit_ablation({"cv", tmp / "cv.gguf"}, config); }, "GGUF"), "a GGUF graft is refused");
            const std::vector<std::pair<std::string, std::vector<std::uint64_t>>> layout = {
                {"attn_dirs", {config.num_layer_visits, config.hidden_size}}, {"attn_w", {config.num_layer_visits}},
                {"mlp_dirs", {config.num_layer_visits, config.hidden_size}}, {"mlp_w", {config.num_layer_visits}}};
            write(tmp / "global.safetensors", safetensors(layout, R"({"params":"{\"scope\": \"global\"}"})"));
            check(refused([&] { (void)ink::load_visit_ablation({"g", tmp / "global.safetensors"}, config); }, "per_visit"),
                  "an ablation of scope global is refused");
            write(tmp / "short.safetensors", safetensors({layout[0], layout[1], layout[3], {"mlp_dirs", {4, 4}}}, R"({"params":"{\"scope\": \"per_visit\"}"})"));
            check(refused([&] { (void)ink::load_visit_ablation({"s", tmp / "short.safetensors"}, config); }, "shape"),
                  "an ablation of another shape is refused");
            const fs::path dora = tmp / "dora", other = tmp / "other";
            fs::create_directories(dora), fs::create_directories(other);
            write(dora / "adapter_config.json", R"({"peft_type":"LORA","r":2,"lora_alpha":2,"use_dora":true})");
            write(dora / "adapter_model.safetensors", safetensors({}, "{}"));
            check(refused([&] { (void)ink::load_lora_adapter({"d", dora}, config); }, "use_dora"), "a DoRA adapter is refused");
            write(other / "adapter_config.json", R"({"peft_type":"LORA","r":2,"lora_alpha":2})");
            write(other / "adapter_model.safetensors",
                  safetensors({{"base_model.model.model.llm.layers.4.attn.wq_du.lora_A.weight", {2, config.hidden_size}}}, "{}"));
            check(refused([&] { (void)ink::load_lora_adapter({"o", other}, config); }, "attn.wo_ud"), "a LoRA on another module is refused");
            fs::remove_all(tmp);
        }

        // 2. The Ops against FP64.
        ninfer::DeviceContext device(0);
        cudaStream_t stream = nullptr;
        cuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "stream");
        const ink::DeviceLora dlora(lora);
        const ink::DeviceAblation dablation(ablation);
        std::mt19937 rng(7);
        std::normal_distribution<float> normal(0.0F, 1.0F);
        {
            constexpr std::int32_t T = 7;
            const std::uint32_t layer = 11;
            std::vector<std::uint16_t> xb(static_cast<std::size_t>(q) * T);
            for (auto& v : xb) { v = to_bf16(normal(rng)); }
            std::vector<float> y0(static_cast<std::size_t>(h) * T);
            for (auto& v : y0) { v = 4.0F * normal(rng); }
            DeviceBuffer xd(xb.size() * 2), yd(y0.size() * 4);
            xd.copy_from_host(xb.data(), xb.size() * 2);
            yd.copy_from_host(y0.data(), y0.size() * 4);
            Tensor x(xd.p, DType::BF16, {q, T}), y(yd.p, DType::FP32, {h, T});
            ninfer::ops::low_rank_update_fp32(x, dlora.a[layer], dlora.b[layer], dlora.rank, dlora.scale, y, stream);
            const auto got = download(y, stream);
            std::vector<double> want(got.size());
            double delta_scale = 0;
            for (std::int32_t t = 0; t < T; ++t) {
                std::vector<double> col(static_cast<std::size_t>(q));
                for (std::int32_t k = 0; k < q; ++k) { col[static_cast<std::size_t>(k)] = from_bf16(xb[static_cast<std::size_t>(t) * q + k]); }
                const auto d = ink::lora_reference_delta(*lora.layers[layer], lora.rank, lora.scale, col);
                for (std::int32_t n = 0; n < h; ++n) {
                    want[static_cast<std::size_t>(t) * h + n] = y0[static_cast<std::size_t>(t) * h + n] + d[static_cast<std::size_t>(n)];
                    delta_scale = std::max(delta_scale, std::abs(d[static_cast<std::size_t>(n)]));
                }
            }
            const double err = max_rel(got, want);
            std::printf("      low_rank_update_fp32: max |delta| %.3e, max rel error %.2e\n", delta_scale, err);
            check(err < 2e-6 && delta_scale > 0, "low_rank_update_fp32 (layer 11's A and B) against FP64");
        }
        for (const bool with_residual : {true, false}) {
            constexpr std::int32_t T = 5;
            const std::uint32_t v = 20;
            const float w = ablation.attn_w[v] != 0 ? ablation.attn_w[v] : 1.0F;
            std::vector<float> y0(static_cast<std::size_t>(h) * T), res(y0.size());
            for (auto& x : y0) { x = 3.0F * normal(rng); }
            for (auto& x : res) { x = 10.0F * normal(rng); }
            DeviceBuffer yd(y0.size() * 4), rd(res.size() * 4), sd(y0.size() * 4);
            yd.copy_from_host(y0.data(), y0.size() * 4);
            rd.copy_from_host(res.data(), res.size() * 4);
            Tensor y(yd.p, DType::FP32, {h, T}), r(rd.p, DType::FP32, {h, T}), s(with_residual ? sd.p : yd.p, DType::FP32, {h, T});
            ninfer::ops::direction_ablate_fp32(y, dablation.attn_dir[v], w, with_residual ? &r : nullptr, s, stream);
            const auto got = download(s, stream);
            std::vector<double> want(got.size());
            for (std::int32_t t = 0; t < T; ++t) {
                const std::vector<double> col(y0.begin() + std::ptrdiff_t(t) * h, y0.begin() + std::ptrdiff_t(t + 1) * h);
                const auto a = ink::ablation_reference(ablation.attn_dirs.data() + std::size_t(v) * h, w, col);
                for (std::int32_t c = 0; c < h; ++c) {
                    want[static_cast<std::size_t>(t) * h + c] = a[static_cast<std::size_t>(c)] + (with_residual ? res[static_cast<std::size_t>(t) * h + c] : 0.0);
                }
            }
            const double err = max_rel(got, want);
            std::printf("      direction_ablate_fp32 (%s): max rel error %.2e\n", with_residual ? "with the residual" : "in place", err);
            check(err < 2e-6, std::string("direction_ablate_fp32 against FP64, ") + (with_residual ? "with the residual add" : "in place"));
        }

        // 3. The LoRA's effective wo_ud rows.
        const auto model = ink::load_model(reader, device, {.vision = false, .dspark = false});
        {
            const std::uint32_t layer = 4;
            std::vector<std::uint16_t> eye(static_cast<std::size_t>(q) * q, 0);
            for (std::int32_t k = 0; k < q; ++k) { eye[static_cast<std::size_t>(k) * q + k] = to_bf16(1.0F); }
            DeviceBuffer xd(eye.size() * 2), yd(static_cast<std::size_t>(h) * q * 4);
            xd.copy_from_host(eye.data(), eye.size() * 2);
            Tensor x(xd.p, DType::BF16, {q, q}), y(yd.p, DType::FP32, {h, q});
            const Tensor wo = model->tensor(model->weights().layers[layer].output, {q, h});
            const Tensor* weights[] = {&wo};
            ninfer::ops::projection_fp32(x, weights, y, stream);
            const auto base = download(y, stream); // column k = wo_ud[:, k]
            ninfer::ops::low_rank_update_fp32(x, dlora.a[layer], dlora.b[layer], dlora.rank, dlora.scale, y, stream);
            const auto merged = download(y, stream);
            const auto& pair  = *lora.layers[layer];
            for (const std::int32_t n : {7, 1500}) {
                double err = 0, row_scale = 1e-30;
                std::size_t bf16_same = 0;
                for (std::int32_t k = 0; k < q; ++k) {
                    double want = base[static_cast<std::size_t>(k) * h + n];
                    for (std::int32_t j = 0; j < lora.rank; ++j) {
                        want += double(lora.scale) * pair.b[static_cast<std::size_t>(n) * lora.rank + j] * pair.a[static_cast<std::size_t>(j) * q + k];
                    }
                    const float got = merged[static_cast<std::size_t>(k) * h + n];
                    err = std::max(err, std::abs(got - want)), row_scale = std::max(row_scale, std::abs(want));
                    bf16_same += to_bf16(got) == to_bf16(static_cast<float>(want));
                }
                err /= row_scale; // relative to the row's largest value (an element can cancel to near zero)
                std::printf("      layer %u wo_ud row %d: max rel error %.2e from base + scale B A (FP64); BF16 equal in %zu/%d\n", layer, n,
                            err, bf16_same, q);
                check(err < 1e-6 && bf16_same * 1000 >= static_cast<std::size_t>(q) * 999,
                      "layer " + std::to_string(layer) + " wo_ud row " + std::to_string(n) + " is base + scale B A");
            }
        }

        // 4. The program.
        const ink::Frontend frontend(model->resources(), {});
        const auto prompt = frontend.tokenize_text("Inkling-2 adapters: the quick brown fox jumps over the lazy dog, then counts 1, 2, 3.");
        ink::ProgramLimits limits{512, 64};
        limits.decode_graph = false;
        ink::TextProgram program(*model, limits, stream);
        const auto run = [&](ink::AdapterSelection sel, std::map<std::string, std::vector<float>>* trace) {
            program.set_adapters(sel);
            program.reset();
            const ink::TextProgram::Trace capture = [&](const std::string& name, const Tensor& t) {
                if (trace != nullptr && t.dtype == DType::FP32) { (*trace)[name] = download(t, stream); }
            };
            return download(program.forward(prompt, trace != nullptr ? &capture : nullptr), stream);
        };
        std::map<std::string, std::vector<float>> base;
        const auto base_logits = run({}, &base);
        const std::int32_t T  = static_cast<std::int32_t>(prompt.size());
        const auto column = [&](const std::vector<float>& m, std::int32_t t) { return std::vector<double>(m.begin() + std::ptrdiff_t(t) * h, m.begin() + std::ptrdiff_t(t + 1) * h); };
        // One visit's write ablated alone (weight 1 there, 0 elsewhere): visits before it are the base run's, bit for bit.
        const auto single = [&](std::uint32_t v, bool attention) {
            ink::VisitAblation one = ablation;
            std::fill(one.attn_w.begin(), one.attn_w.end(), 0.0F);
            std::fill(one.mlp_w.begin(), one.mlp_w.end(), 0.0F);
            (attention ? one.attn_w : one.mlp_w)[v] = 1.0F;
            return std::make_unique<ink::DeviceAblation>(one);
        };
        const auto vs = [](std::uint32_t x) { return std::to_string(x); };
        std::uint32_t inside = 0, boundary = 0;
        for (std::uint32_t v = 1; v < config.num_layer_visits; ++v) {
            const bool first = config.physical(v) % config.attn_res_block_size == 0;
            if (first && boundary == 0 && v > 8) { boundary = v; }
            if (!first && inside == 0 && v > 8) { inside = v; }
        }
        for (const auto [v, attention] : {std::pair{inside, true}, std::pair{boundary, true}, std::pair{inside + 1, false}}) {
            const auto dev = single(v, attention);
            std::map<std::string, std::vector<float>> got;
            (void)run({nullptr, dev.get()}, &got);
            const float* r = (attention ? ablation.attn_dirs : ablation.mlp_dirs).data() + std::size_t(v) * h;
            std::vector<double> want;
            std::vector<float> have;
            for (std::int32_t t = 0; t < T; ++t) {
                // The write y (base), and what holds it after the residual add: ffn_inp (attention) or l_out (MLP).
                const auto sum = column(base[(attention ? "ffn_inp-" : "l_out-") + vs(v)], t);
                const auto y   = attention ? column(base["attn_out-" + vs(v)], t)
                                           : [&] { auto d = sum; const auto f = column(base["ffn_inp-" + vs(v)], t); for (std::size_t c = 0; c < d.size(); ++c) { d[c] -= f[c]; } return d; }();
                const auto ablated = ink::ablation_reference(r, 1.0, y);
                for (std::int32_t c = 0; c < h; ++c) { want.push_back(sum[static_cast<std::size_t>(c)] - y[static_cast<std::size_t>(c)] + ablated[static_cast<std::size_t>(c)]); }
                const auto g = got[(attention ? "ffn_inp-" : "l_out-") + vs(v)];
                have.insert(have.end(), g.begin() + std::ptrdiff_t(t) * h, g.begin() + std::ptrdiff_t(t + 1) * h);
            }
            const double err = max_rel(have, want);
            const bool before_same = v == 0 || got["l_out-" + vs(v - 1)] == base["l_out-" + vs(v - 1)];
            std::printf("      visit %u (layer %u%s) %s write: rel error %.2e from the FP64 ablation of the base write\n", v, config.physical(v),
                        config.physical(v) % config.attn_res_block_size == 0 ? ", a block's first" : "", attention ? "attention" : "MLP", err);
            check(err < 1e-5 && before_same, "the ablation at visit " + vs(v) + "'s " + (attention ? "attention" : "MLP") + " write");
        }
        {
            const auto again = run({}, nullptr);
            check(again == base_logits, "cleared adapters give the base logits bit for bit");
        }
        {
            ink::ProgramLimits graph_limits{512, 64};
            ink::TextProgram graphed(*model, graph_limits, stream);
            const ink::AdapterSelection both{&dlora, &dablation};
            graphed.set_adapters(both);
            program.set_adapters(both);
            graphed.reset(), program.reset();
            (void)graphed.forward(prompt), (void)program.forward(prompt);
            bool same = true;
            for (const std::int32_t token : {1000, 2000, 3000, 4000}) {
                const std::int32_t one[] = {token};
                same = same && download(graphed.forward(one), stream) == download(program.forward(one), stream);
            }
            check(same, "with both adapters the decode graph gives the eager forward's logits bit for bit");
        }

        // 5. KLD report.
        if (const char* text_path = std::getenv("NINFER_INKLING2_KLD_TEXT")) {
            std::ifstream in(text_path);
            std::stringstream text;
            text << in.rdbuf();
            auto tokens = frontend.tokenize_text(text.str());
            const std::size_t want = std::getenv("NINFER_INKLING2_KLD_TOKENS") ? std::strtoul(std::getenv("NINFER_INKLING2_KLD_TOKENS"), nullptr, 10) : 2048;
            tokens.resize(std::min(tokens.size(), want));
            constexpr std::int32_t kChunk = 128;
            ink::ProgramLimits kl_limits{static_cast<std::int32_t>(tokens.size()) + 1, kChunk};
            ink::TextProgram a(*model, kl_limits, stream), b(*model, kl_limits, stream);
            const auto valid = static_cast<std::int32_t>(config.unpadded_vocab_size);
            for (const auto& [label, sel] : {std::pair{"lora", ink::AdapterSelection{&dlora, nullptr}},
                                              std::pair{"ablation", ink::AdapterSelection{nullptr, &dablation}}}) {
                a.reset(), b.reset();
                b.set_adapters(sel);
                double kld = 0, kld_max = 0;
                std::size_t n = 0, top1 = 0;
                std::vector<double> lp, lq;
                for (std::size_t s = 0; s < tokens.size(); s += kChunk) {
                    const auto piece = std::span(tokens).subspan(s, std::min<std::size_t>(kChunk, tokens.size() - s));
                    const auto la = download(a.forward(piece), stream), lb = download(b.forward(piece), stream);
                    const std::size_t rows = la.size() / piece.size();
                    for (std::size_t t = 0; t < piece.size(); ++t) {
                        log_softmax(la.data() + t * rows, valid, lp);
                        log_softmax(lb.data() + t * rows, valid, lq);
                        double k = 0;
                        for (std::int32_t v = 0; v < valid; ++v) { k += std::exp(lp[static_cast<std::size_t>(v)]) * (lp[static_cast<std::size_t>(v)] - lq[static_cast<std::size_t>(v)]); }
                        kld += k, kld_max = std::max(kld_max, k), ++n;
                        top1 += std::max_element(lp.begin(), lp.end()) - lp.begin() == std::max_element(lq.begin(), lq.end()) - lq.begin();
                    }
                }
                std::printf("KLD %s vs base over %zu tokens: mean %.4f max %.3f, top-1 agreement %.3f\n", label, n, kld / double(n), kld_max,
                            double(top1) / double(n));
            }
        }
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL: %s\n", error.what());
        return 1;
    }
    std::printf(failures ? "FAILED\n" : "PASS\n");
    return failures ? 1 : 0;
}
