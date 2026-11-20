// Written for NInfer-3090.
//
// The DeepSeek-V4.1-Flash FFN sublayer (src/models/dsv4/execution/ffn.h) on the REAL layer-0 weights of the
// ops-level fixture (tools/dsv4/export_ffn_fixture.py: 8 tokens, the whole 384-expert bank), against the NumPy
// oracle's mixes and next residual streams, and across the ways the experts can be served:
//
//   resident bank in device memory, and in pinned host memory     vs the oracle (the ops test's bounds)
//   the same tokens one column at a time (T = 1, the decode shape)  == the 8-column result, bit for bit
//   collapse with the attention's pre (the block wiring) given explicitly == the own-pre result
//   FileExpertSource over the real artifact (layer 0's bank there is the fixture's bank)
//                                                               == the resident result, bit for bit;
//                                                                  cache counters; cold and warm decode ms
//
// Needs NINFER_DSV4_FFN_FIXTURE (the exporter's output directory, about 3 GB) and, for the file-backed cases,
// NINFER_TEST_DSV4_ARTIFACT (a converted artifact: .../dsv4.ninfer). Both are the real model's, which the
// repository does not hold.

#include "artifact/reader.h"
#include "core/arena.h"
#include "core/weight.h"
#include "models/dsv4/execution/expert_source.h"
#include "models/dsv4/execution/ffn.h"
#include "models/dsv4/model.h"
#include "ops/op_tester.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
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
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

namespace {

namespace t   = ninfer::test;
namespace d4  = ninfer::models::dsv4;
using ninfer::DType;
using ninfer::QType;
using ninfer::Tensor;
using ninfer::Weight;

constexpr int kS = 4, kH = 5120, kInter = 2304, kExperts = 384;

int failures = 0;

void check(bool ok, const std::string& what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what.c_str());
        ++failures;
    }
}

struct Device {
    void* p          = nullptr;
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

// Reads a file into `destination` (device or pinned host memory) in chunks.
void load_bank(const std::string& path, std::size_t bytes, void* destination, cudaMemcpyKind kind) {
    std::ifstream in(path, std::ios::binary);
    if (!in) { throw std::runtime_error("cannot read " + path); }
    constexpr std::size_t kChunk = 256U << 20;
    std::vector<char> buffer(kChunk);
    for (std::size_t at = 0; at < bytes; at += kChunk) {
        const std::size_t n = std::min(kChunk, bytes - at);
        in.read(buffer.data(), static_cast<std::streamsize>(n));
        if (static_cast<std::size_t>(in.gcount()) != n) { throw std::runtime_error("short read of " + path); }
        t::cuda_check(cudaMemcpy(static_cast<std::uint8_t*>(destination) + at, buffer.data(), n, kind), "bank copy");
    }
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

std::vector<std::uint16_t> download_u16(const void* device, std::size_t count) {
    std::vector<std::uint16_t> raw(count);
    t::cuda_check(cudaMemcpy(raw.data(), device, count * 2, cudaMemcpyDeviceToHost), "download");
    return raw;
}

std::vector<float> to_f32(const std::vector<std::uint16_t>& bits) {
    std::vector<float> out(bits.size());
    for (std::size_t i = 0; i < bits.size(); ++i) { out[i] = t::bf16_to_f32(bits[i]); }
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
    const double rel = std::sqrt(num / den);
    std::printf("%-26s relative L2 %.5f (bound %.5f), worst |error| %.4g\n", stage.c_str(), rel, bound, worst);
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
        {"ggml_rec_iq1_m_iq1_m", QType::GGML_REC_IQ1_M_IQ1_M},         {"ggml_rec_iq2_xxs_iq1_m", QType::GGML_REC_IQ2_XXS_IQ1_M},
        {"ggml_rec_iq2_xxs_iq2_xxs", QType::GGML_REC_IQ2_XXS_IQ2_XXS}, {"ggml_rec_iq2_xs_iq1_m", QType::GGML_REC_IQ2_XS_IQ1_M},
        {"ggml_rec_iq2_xxs_iq2_xs", QType::GGML_REC_IQ2_XXS_IQ2_XS},   {"ggml_rec_iq1_m_iq2_xxs", QType::GGML_REC_IQ1_M_IQ2_XXS},
        {"ggml_rec_iq2_xs_iq2_xxs", QType::GGML_REC_IQ2_XS_IQ2_XXS},   {"ggml_rec_iq2_xs_iq2_xs", QType::GGML_REC_IQ2_XS_IQ2_XS}};
    return kTypes.at(name);
}

using Clock = std::chrono::steady_clock;

} // namespace

int main() {
    try {
        const char* dir_env = std::getenv("NINFER_DSV4_FFN_FIXTURE");
        if (dir_env == nullptr) {
            std::printf("SKIP: NINFER_DSV4_FFN_FIXTURE is not set (export a layer with tools/dsv4/export_ffn_fixture.py)\n");
            return 77;
        }
        if (t::cuda_unavailable()) {
            std::printf("SKIP: no CUDA device\n");
            return 77;
        }
        const std::string dir = std::string(dir_env) + "/";
        cudaStream_t stream   = nullptr;
        t::cuda_check(cudaStreamCreate(&stream), "stream");

        const auto config = read_config(dir + "config.txt");
        const auto number = [&](const char* key) { return std::stod(config.at(key)); };
        const int T = static_cast<int>(number("tokens")), k = static_cast<int>(number("topk"));
        check(number("hc") == kS && number("dim") == kH && number("experts") == kExperts && number("inter") == kInter,
              "the exported layer has V4.1-Flash's geometry");
        const auto ref = [&](const char* name) { return read_vector<float>(dir + name + ".bin"); };

        // Weights, as bound views.
        const auto fn_d = upload_file(dir + "hc_fn.bin"), scale_d = upload_file(dir + "hc_scale.bin"), base_d = upload_file(dir + "hc_base.bin"),
                   bias_d = upload_file(dir + "bias.bin"), router_d = upload_file(dir + "router.bin");
        const auto norm_f32 = read_vector<float>(dir + "ffn_norm.bin");
        std::vector<std::uint16_t> norm_bits(norm_f32.size());
        for (std::size_t i = 0; i < norm_f32.size(); ++i) { norm_bits[i] = t::f32_to_bf16(norm_f32[i]); }
        Device norm_d(norm_bits.size() * 2);
        upload(norm_d, norm_bits.data(), norm_bits.size() * 2);
        const auto sg = upload_file(dir + "shexp_gate.bin"), su = upload_file(dir + "shexp_up.bin"), sd = upload_file(dir + "shexp_down.bin");

        d4::FfnParams params;
        params.hidden = kH, params.streams = kS, params.experts = kExperts, params.top_k = k, params.intermediate = kInter;
        params.norm_eps = static_cast<float>(number("norm_eps")), params.hc_eps = static_cast<float>(number("hc_eps"));
        params.hc_iterations = static_cast<int>(number("hc_iters"));
        params.route_scale   = static_cast<float>(number("route_scale"));
        params.norm_topk     = number("norm_topk_prob") != 0;
        params.clamp_shared  = static_cast<float>(number("clamp_shared"));
        d4::FfnWeights weights;
        weights.hc_fn        = Tensor(fn_d->p, DType::FP32, {kS * kH, 24});
        weights.hc_scale     = Tensor(scale_d->p, DType::FP32, {3});
        weights.hc_base      = Tensor(base_d->p, DType::FP32, {24});
        weights.norm         = Tensor(norm_d.p, DType::BF16, {kH});
        weights.router       = Tensor(router_d->p, DType::BF16, {kH, kExperts});
        weights.router_bias  = Tensor(bias_d->p, DType::FP32, {kExperts});
        weights.shared_gate  = ggml_weight(QType::GGML_Q5_K, kInter, kH, sg->p);
        weights.shared_up    = ggml_weight(QType::GGML_Q5_K, kInter, kH, su->p);
        weights.shared_down  = ggml_weight(QType::GGML_Q5_K, kH, kInter, sd->p);
        weights.shared_policy = ninfer::ops::LinearPolicy::AllowA8;

        const QType format         = record_type(config.at("record_format"));
        const std::uint64_t stride = static_cast<std::uint64_t>(number("record_stride"));
        const std::size_t bank_bytes = static_cast<std::size_t>(stride) * kExperts;
        std::printf("layer %d, %d tokens, %d of %d experts, records %s, bank %.2f GB\n", static_cast<int>(number("layer")), T, k, kExperts,
                    config.at("record_format").c_str(), static_cast<double>(bank_bytes) / 1e9);

        // The bank twice: device memory, and pinned host memory (what the load gives a resident bank).
        Device bank_device(bank_bytes);
        load_bank(dir + "bank.bin", bank_bytes, bank_device.p, cudaMemcpyHostToDevice);
        std::uint8_t* bank_pinned = nullptr;
        t::cuda_check(cudaHostAlloc(reinterpret_cast<void**>(&bank_pinned), bank_bytes, cudaHostAllocMapped | cudaHostAllocPortable), "pinned bank");
        load_bank(dir + "bank.bin", bank_bytes, bank_pinned, cudaMemcpyHostToHost);
        void* bank_pinned_device = nullptr;
        t::cuda_check(cudaHostGetDevicePointer(&bank_pinned_device, bank_pinned, 0), "pinned bank device pointer");

        d4::ExpertBank device_bank;
        device_bank.format = format, device_bank.experts = kExperts, device_bank.record_stride = stride;
        device_bank.host_records = device_bank.device_records = bank_device.bytes();
        d4::ExpertBank pinned_bank = device_bank;
        pinned_bank.host_records   = bank_pinned;
        pinned_bank.device_records = static_cast<const std::uint8_t*>(bank_pinned_device);
        d4::ResidentExpertSource resident_device(std::span<const d4::ExpertBank>(&device_bank, 1), kExperts);
        d4::ResidentExpertSource resident_pinned(std::span<const d4::ExpertBank>(&pinned_bank, 1), kExperts);

        // Streams and the workspace arena.
        const auto x_bits = read_file(dir + "x.bin");
        check(x_bits.size() == static_cast<std::size_t>(T) * kS * kH * 2, "x.bin is [tokens, 4 x hidden] BF16");
        const std::size_t column_bytes = static_cast<std::size_t>(kS) * kH * 2;
        Device residual(x_bits.size());
        ninfer::DeviceArena arena(d4::ffn_workspace_bytes(params, weights, T) + (4U << 20));
        ninfer::DeviceArena mix_arena(1U << 20);
        d4::HcMixes mixes = d4::alloc_hc_mixes(mix_arena, params, T), mixes_b = d4::alloc_hc_mixes(mix_arena, params, T);
        const auto reset = [&] { upload(residual, x_bits.data(), x_bits.size()); };
        const auto download_out = [&] { return download_u16(residual.p, static_cast<std::size_t>(kS) * kH * T); };
        const auto run_all = [&](d4::ExpertSource& source, const Tensor* collapse_pre, d4::HcMixes& m) {
            reset();
            Tensor R(residual.p, DType::BF16, {kS * kH, T});
            d4::ffn_forward(params, weights, 0, source, {R, collapse_pre, m}, arena, stream);
            t::cuda_check(cudaStreamSynchronize(stream), "ffn");
            check(arena.used() == 0, "the FFN returns its scratch to the arena");
        };

        // 1. Resident bank vs the oracle.
        run_all(resident_device, nullptr, mixes);
        const auto out_device = download_out();
        compare("pre", download_f32(mixes.pre.data, static_cast<std::size_t>(kS) * T), ref("pre"), 5e-6);
        compare("post", download_f32(mixes.post.data, static_cast<std::size_t>(kS) * T), ref("post"), 5e-6);
        compare("comb", download_f32(mixes.comb.data, static_cast<std::size_t>(kS) * kS * T), ref("comb"), 5e-6);
        compare("out (device bank)", to_f32(out_device), ref("out"), 0.005);

        run_all(resident_pinned, nullptr, mixes_b);
        check(download_out() == out_device, "the pinned host bank gives the device bank's bits");

        // 2. The attention's pre given explicitly (here the sublayer's own pre from the run above): same bits.
        run_all(resident_device, &mixes.pre, mixes_b);
        check(download_out() == out_device, "collapse_pre given explicitly (the sublayer's own pre) gives the own-pre bits");
        // A different pre collapses to a different input, so the plumbing is live.
        {
            std::vector<float> other(static_cast<std::size_t>(kS) * T, 0.0F);
            for (int c = 0; c < T; ++c) { other[static_cast<std::size_t>(c) * kS] = 1.0F; } // stream 0 only
            Device other_d(other.size() * 4);
            upload(other_d, other.data(), other.size() * 4);
            const Tensor other_pre(other_d.p, DType::FP32, {kS, T});
            run_all(resident_device, &other_pre, mixes_b);
            check(download_out() != out_device, "a different collapse_pre changes the output");
        }

        // 3. Column by column (decode, T = 1) equals the 8-column result.
        {
            d4::HcMixes one = d4::alloc_hc_mixes(mix_arena, params, 1);
            const auto columns = [&](d4::ExpertSource& source) { // the tokens as T = 1 calls; returns ms per call
                reset();
                const auto t0 = Clock::now();
                for (int c = 0; c < T; ++c) {
                    Tensor R(residual.bytes() + static_cast<std::size_t>(c) * column_bytes, DType::BF16, {kS * kH, 1});
                    d4::ffn_forward(params, weights, 0, source, {R, nullptr, one}, arena, stream);
                }
                t::cuda_check(cudaStreamSynchronize(stream), "ffn T=1");
                return std::chrono::duration<double, std::milli>(Clock::now() - t0).count() / T;
            };
            (void)columns(resident_device); // warm-up (module load, first launches)
            const double pinned_ms = columns(resident_pinned);
            const double ms        = columns(resident_device);
            std::printf("resident decode: %.3f ms per layer from device memory, %.3f ms from pinned host memory (zero-copy over PCIe)\n", ms, pinned_ms);
            const auto out_columns = download_out();
            const std::size_t differing = static_cast<std::size_t>(std::inner_product(out_columns.begin(), out_columns.end(), out_device.begin(), 0,
                                                                  std::plus<>(), [](std::uint16_t a, std::uint16_t b) { return a != b ? 1 : 0; }));
            std::printf("T=1 column by column: %zu of %zu values differ from the T=%d result\n", differing,
                        out_columns.size(), T);
            compare("out (T=1 vs oracle)", to_f32(out_columns), ref("out"), 0.005);
            check(differing == 0, "T=1 column by column equals the T=8 result bit for bit");
        }

        // 3b. A wide prefill chunk: the 8 tokens repeated to 512 columns (the narrow Op serves any T; the experts
        // see several columns each). Every repeat is checked against the oracle's 8 tokens.
        constexpr int kRepeats = 64, kWide = 8 * kRepeats;
        std::vector<std::uint8_t> wide_x;
        for (int r = 0; r < kRepeats; ++r) { wide_x.insert(wide_x.end(), x_bits.begin(), x_bits.end()); }
        Device wide_residual(wide_x.size());
        ninfer::DeviceArena wide_arena(d4::ffn_workspace_bytes(params, weights, kWide) + (4U << 20));
        d4::HcMixes wide_mixes = d4::alloc_hc_mixes(mix_arena, params, kWide);
        const auto wide_run = [&](d4::ExpertSource& source, std::vector<std::uint16_t>& out) {
            upload(wide_residual, wide_x.data(), wide_x.size());
            Tensor R(wide_residual.p, DType::BF16, {kS * kH, kWide});
            const auto t0 = Clock::now();
            d4::ffn_forward(params, weights, 0, source, {R, nullptr, wide_mixes}, wide_arena, stream);
            t::cuda_check(cudaStreamSynchronize(stream), "ffn wide");
            const double ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
            out = download_u16(wide_residual.p, static_cast<std::size_t>(kS) * kH * kWide);
            return ms;
        };
        std::vector<std::uint16_t> wide_device;
        (void)wide_run(resident_device, wide_device); // warm-up
        const double wide_ms = wide_run(resident_device, wide_device);
        {
            std::vector<float> expected;
            for (int r = 0; r < kRepeats; ++r) {
                const auto one_pass = ref("out");
                expected.insert(expected.end(), one_pass.begin(), one_pass.end());
            }
            std::printf("T=%d prefill chunk, resident: %.2f ms per layer (%.1f us per token)\n", kWide, wide_ms, 1000.0 * wide_ms / kWide);
            compare("out (T=512 vs oracle)", to_f32(wide_device), expected, 0.005);
        }

        // 4. Experts left in the artifact, read through the file-backed source.
        const char* artifact_env = std::getenv("NINFER_TEST_DSV4_ARTIFACT");
        if (artifact_env == nullptr) {
            std::printf("SKIP file-backed cases: NINFER_TEST_DSV4_ARTIFACT is not set\n");
        } else {
            ninfer::artifact::Reader reader(artifact_env);
            const d4::LoadPlan plan = d4::plan_load(reader, true);
            const auto& banks       = plan.file_banks();
            check(banks.size() == 40 && banks[0].record_stride == stride && banks[0].format == format,
                  "the artifact's layer 0 bank has the fixture's record format and stride");
            d4::FileExpertSource file(reader, banks, kExperts); // the defaults: a 4 GiB cache, a call of up to 2048 columns
            std::printf("record cache: %u slots of %.2f MB\n", file.cache_slots(), static_cast<double>(stride) / 1e6);

            run_all(file, nullptr, mixes);
            const auto first = file.stats();
            check(download_out() == out_device, "file-backed (cold cache) equals the resident result bit for bit");
            std::printf("cold T=%d: %llu hits, %llu misses, %.1f MB read in %.3f s\n", T, static_cast<unsigned long long>(first.hits),
                        static_cast<unsigned long long>(first.misses), static_cast<double>(first.bytes_read) / 1e6, first.read_seconds);
            check(first.hits == 0 && first.misses > 0 && first.misses <= static_cast<std::uint64_t>(k * T), "a cold call misses every distinct routed expert");
            check(first.bytes_read == first.misses * stride, "bytes read are one record per miss");

            file.reset_stats();
            run_all(file, nullptr, mixes);
            const auto second = file.stats();
            check(download_out() == out_device, "file-backed (warm cache) equals the resident result bit for bit");
            check(second.misses == 0 && second.hits == first.misses && second.bytes_read == 0, "the second call is all hits");

            // Decode ms per layer: cold (cache cleared; the artifact's pages may still be in the OS cache) and warm.
            const auto decode = [&](const char* label) {
                reset();
                file.reset_stats();
                d4::HcMixes one = d4::alloc_hc_mixes(mix_arena, params, 1);
                const auto t0   = Clock::now();
                for (int c = 0; c < T; ++c) {
                    Tensor R(residual.bytes() + static_cast<std::size_t>(c) * column_bytes, DType::BF16, {kS * kH, 1});
                    d4::ffn_forward(params, weights, 0, file, {R, nullptr, one}, arena, stream);
                }
                t::cuda_check(cudaStreamSynchronize(stream), "ffn file T=1");
                const double ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count() / T;
                const auto s    = file.stats();
                std::printf("file-backed decode, %s: %.3f ms per layer (%llu hits, %llu misses, %.3f s reading)\n", label, ms,
                            static_cast<unsigned long long>(s.hits), static_cast<unsigned long long>(s.misses), s.read_seconds);
                return s;
            };
            file.clear_cache();
            const auto cold = decode("cold cache");
            check(cold.misses > 0, "cold decode reads records");
            check(download_out() == out_device, "file-backed T=1 (cold) equals the resident result bit for bit");
            const auto warm = decode("warm cache");
            check(warm.misses == 0 && warm.hits == cold.hits + cold.misses, "warm decode is all hits");
            check(download_out() == out_device, "file-backed T=1 (warm) equals the resident result bit for bit");

            // A wide chunk through the file source (every routed expert is already cached; a cold wide chunk
            // would read each distinct expert once).
            std::vector<std::uint16_t> wide_file;
            file.reset_stats();
            const double wide_file_ms = wide_run(file, wide_file);
            std::printf("T=%d prefill chunk, file-backed (warm): %.2f ms per layer\n", kWide, wide_file_ms);
            check(wide_file == wide_device, "file-backed T=512 equals the resident result bit for bit");
        }
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
