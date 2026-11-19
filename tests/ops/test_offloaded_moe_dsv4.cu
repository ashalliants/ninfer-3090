// moe_experts_narrow on DeepSeek-V4.1-Flash's expert shape [E, 5120, 2304]: the narrow route over the eight
// record pairs of the GSQ-RCO IQ2_XXS build (IQ1_M, IQ2_XXS and IQ2_XS gate/up and down, 256-value K-quant blocks),
// with the clamped SwiGLU, against an FP64 oracle over independently decoded records (the exact reference decoder),
// and bit for bit between placements (resident, staged, zero-copy, with the precast input): the route and the
// placement are invisible in the outputs. Plus the refusals.
//
// --small: three formats and fewer column counts (a few seconds).

#include "core/arena.h"
#include "core/device.h"
#include "core/tensor.h"
#include "core/weight.h"
#include "ninfer/ops/offloaded_sparse_moe.h"
#include "offloaded_moe_fixtures.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namespace ninfer;
using namespace ninfer::test::moe;
namespace ops = ninfer::ops;

constexpr int kH = om::kDsv4Shape.hidden, kI = om::kDsv4Shape.intermediate;
constexpr double kLimit = 10.0; // the SwiGLU clamp of every layer of the published builds

bool g_small = false;

void cuda_ok(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "CUDA %s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

struct Pinned {
    std::uint8_t* host   = nullptr;
    std::uint8_t* device = nullptr;
    explicit Pinned(std::size_t bytes) {
        cuda_ok(cudaHostAlloc(reinterpret_cast<void**>(&host), bytes, cudaHostAllocMapped | cudaHostAllocPortable),
                "cudaHostAlloc");
        cuda_ok(cudaHostGetDevicePointer(reinterpret_cast<void**>(&device), host, 0), "cudaHostGetDevicePointer");
    }
    ~Pinned() { cudaFreeHost(host); }
    Pinned(const Pinned&)            = delete;
    Pinned& operator=(const Pinned&) = delete;
};

template <class T>
std::vector<T> download(const void* device, std::size_t count) {
    std::vector<T> out(count);
    cuda_ok(cudaMemcpy(out.data(), device, count * sizeof(T), cudaMemcpyDeviceToHost), "download");
    return out;
}

// A record of random blocks, record_geometry(format, kDsv4Shape).bytes long (the gaps between parts zero).
std::vector<std::uint8_t> random_dsv4_record(QType format, std::mt19937& rng) {
    const auto g  = om::record_geometry(format, om::kDsv4Shape);
    std::vector<std::uint8_t> r(g.bytes, 0);
    const auto gu = ggml_block(g.gate_up), dn = ggml_block(g.down);
    const std::size_t gu_blocks = static_cast<std::size_t>(kI) * kH / gu.values;
    const std::size_t dn_blocks = static_cast<std::size_t>(kH) * kI / dn.values;
    fill_blocks(g.gate_up, r.data(), gu_blocks, block_scale(g.gate_up), rng);
    fill_blocks(g.gate_up, r.data() + g.up_offset, gu_blocks, block_scale(g.gate_up), rng);
    fill_blocks(g.down, r.data() + g.down_offset, dn_blocks, block_scale(g.down), rng);
    return r;
}

std::vector<std::uint16_t> random_dsv4_x(int columns, std::mt19937& rng, float sigma) {
    std::normal_distribution<float> n(0.0F, sigma);
    std::vector<std::uint16_t> x(static_cast<std::size_t>(columns) * kH);
    for (auto& v : x) { v = bf16_bits(n(rng)); }
    return x;
}

// FP64 oracle of one expert on one BF16 column, the formula of offloaded_sparse_moe.h at [5120, 2304] with the
// clamped SwiGLU (the clamp is applied to the FP32 gate and up values, as the kernel does).
std::vector<double> oracle(QType format, const std::uint8_t* record, const std::uint16_t* x, bool clamp) {
    const auto g = om::record_geometry(format, om::kDsv4Shape);
    const auto span_of = [&](std::uint64_t offset, std::uint64_t bytes) {
        return std::span<const std::byte>(reinterpret_cast<const std::byte*>(record) + offset, bytes);
    };
    const auto gate = ninfer::test::ggml::decode_blocks(g.gate_up, span_of(0, g.gate_up_row_bytes * kI));
    const auto up   = ninfer::test::ggml::decode_blocks(g.gate_up, span_of(g.up_offset, g.gate_up_row_bytes * kI));
    std::vector<float> xf(kH);
    for (int i = 0; i < kH; ++i) { xf[i] = bf16_value(x[i]); }
    std::vector<double> xa(kH);
    for (int s = 0; s < kH / 32; ++s) {
        std::int8_t q[32];
        const double d = a8_cast(xf.data() + 32 * s, 32, q);
        for (int j = 0; j < 32; ++j) { xa[32 * s + j] = d * q[j]; }
    }
    std::vector<float> h(kI);
    for (int r = 0; r < kI; ++r) {
        double gv = 0.0, uv = 0.0;
        const float* gr = gate.data() + static_cast<std::size_t>(r) * kH;
        const float* ur = up.data() + static_cast<std::size_t>(r) * kH;
        for (int k = 0; k < kH; ++k) {
            gv += static_cast<double>(gr[k]) * xa[k];
            uv += static_cast<double>(ur[k]) * xa[k];
        }
        if (clamp) {
            gv = std::min(gv, kLimit);
            uv = std::clamp(uv, -kLimit, kLimit);
        }
        h[r] = static_cast<float>(silu(gv) * uv);
    }
    std::vector<double> ha(kI);
    for (int s = 0; s < kI / 32; ++s) {
        std::int8_t q[32];
        const double d = a8_cast(h.data() + 32 * s, 32, q);
        for (int j = 0; j < 32; ++j) { ha[32 * s + j] = d * q[j]; }
    }
    const auto down = ninfer::test::ggml::decode_blocks(g.down, span_of(g.down_offset, g.down_row_bytes * kH));
    std::vector<double> y(kH);
    for (int r = 0; r < kH; ++r) {
        double sum      = 0.0;
        const float* dr = down.data() + static_cast<std::size_t>(r) * kI;
        for (int k = 0; k < kI; ++k) { sum += static_cast<double>(dr[k]) * ha[k]; }
        y[r] = sum;
    }
    return y;
}

struct Routed {
    DeviceBuffer logits, ids, weights, shared_gate, dispatch, route_log;
    ops::MoeRouting routing;
    ops::MoeDispatch view;
    std::vector<std::int32_t> host_ids; // [k * T], entry t * k + slot
    int experts, k, columns;

    Routed(int e, int top_k, int t, std::mt19937& rng)
        : logits(static_cast<std::size_t>(e + 1) * t * 4), ids(static_cast<std::size_t>(top_k) * t * 4),
          weights(static_cast<std::size_t>(top_k) * t * 4), shared_gate(static_cast<std::size_t>(t) * 4),
          dispatch(ops::moe_dispatch_bytes(e, top_k * t)), route_log(static_cast<std::size_t>(top_k) * t * 4),
          experts(e), k(top_k), columns(t) {
        routing.ids         = Tensor(ids.p, DType::I32, {top_k, t});
        routing.weights     = Tensor(weights.p, DType::FP32, {top_k, t});
        routing.shared_gate = Tensor(shared_gate.p, DType::FP32, {t});
        view                = ops::carve_moe_dispatch(dispatch.p, e, top_k * t);
        std::normal_distribution<float> n(0.0F, 2.0F);
        std::vector<float> l(static_cast<std::size_t>(e + 1) * t);
        for (auto& v : l) { v = n(rng); }
        logits.copy_from_host(l.data(), l.size() * 4);
        const Tensor lt(logits.p, DType::FP32, {e + 1, t});
        cuda_ok(cudaMemset(route_log.p, 0xFF, route_log.bytes), "poison the route log");
        ops::moe_route(lt, top_k, routing, nullptr);
        ops::moe_dispatch(routing, e, view, static_cast<std::int32_t*>(route_log.p), nullptr);
        cuda_ok(cudaDeviceSynchronize(), "route/dispatch");
        host_ids = download<std::int32_t>(ids.p, static_cast<std::size_t>(top_k) * t);
    }
};

struct Placement {
    const char* name;
    int resident_every; // 0: none resident, 1: all, n: every n-th
    int staging_slots;
    int zero_copy_misses;
    bool precast;
};

constexpr Placement kPlacements[] = {
    {"device", 1, 0, 0, false},
    {"zero-copy", 0, 0, 0, false},
    {"staged-4", 0, 4, 0, false},
    {"mixed-staged-3-precast", 3, 3, 0, true},
};

QType kFormats[] = {QType::GGML_REC_IQ1_M_IQ1_M,     QType::GGML_REC_IQ2_XXS_IQ1_M,   QType::GGML_REC_IQ2_XXS_IQ2_XXS,
                    QType::GGML_REC_IQ2_XS_IQ1_M,    QType::GGML_REC_IQ2_XXS_IQ2_XS,  QType::GGML_REC_IQ1_M_IQ2_XXS,
                    QType::GGML_REC_IQ2_XS_IQ2_XXS,  QType::GGML_REC_IQ2_XS_IQ2_XS};

const char* record_name(QType f) {
    switch (f) {
    case QType::GGML_REC_IQ1_M_IQ1_M: return "iq1_m+iq1_m";
    case QType::GGML_REC_IQ2_XXS_IQ1_M: return "iq2_xxs+iq1_m";
    case QType::GGML_REC_IQ2_XXS_IQ2_XXS: return "iq2_xxs+iq2_xxs";
    case QType::GGML_REC_IQ2_XS_IQ1_M: return "iq2_xs+iq1_m";
    case QType::GGML_REC_IQ2_XXS_IQ2_XS: return "iq2_xxs+iq2_xs";
    case QType::GGML_REC_IQ1_M_IQ2_XXS: return "iq1_m+iq2_xxs";
    case QType::GGML_REC_IQ2_XS_IQ2_XXS: return "iq2_xs+iq2_xxs";
    case QType::GGML_REC_IQ2_XS_IQ2_XS: return "iq2_xs+iq2_xs";
    default: return "?";
    }
}

struct Bank {
    QType format;
    int experts;
    std::uint64_t stride;
    std::vector<std::vector<std::uint8_t>> records;
    std::unique_ptr<Pinned> pinned;
    std::unique_ptr<DeviceBuffer> frames;
};

Bank make_bank(QType format, int experts, std::mt19937& rng) {
    Bank b{format, experts, 0, {}, nullptr, nullptr};
    for (int e = 0; e < experts; ++e) { b.records.push_back(random_dsv4_record(format, rng)); }
    b.stride = (om::record_geometry(format, om::kDsv4Shape).bytes + 255) / 256 * 256;
    std::vector<std::uint8_t> flat(static_cast<std::size_t>(experts) * b.stride, 0);
    for (int e = 0; e < experts; ++e) {
        std::memcpy(flat.data() + static_cast<std::size_t>(e) * b.stride, b.records[e].data(), b.records[e].size());
    }
    b.pinned = std::make_unique<Pinned>(flat.size());
    std::memcpy(b.pinned->host, flat.data(), flat.size());
    b.frames = std::make_unique<DeviceBuffer>(flat.size());
    b.frames->copy_from_host(flat.data(), flat.size());
    return b;
}

// One call of moe_experts_narrow in a placement: BF16 outputs [k * T][H].
std::vector<std::uint16_t> run(const Bank& bank, const Placement& p, const Routed& r, const DeviceBuffer& x_dev, int columns) {
    const int k        = r.k;
    const int max_jobs = std::min(bank.experts, k * columns);
    std::vector<std::int32_t> frames(bank.experts, -1);
    for (int e = 0; e < bank.experts; ++e) {
        if (p.resident_every > 0 && e % p.resident_every == 0) { frames[e] = e; }
    }
    DeviceBuffer d_frames(frames.size() * 4);
    d_frames.copy_from_host(frames.data(), frames.size() * 4);
    std::unique_ptr<DeviceBuffer> staging;
    if (p.staging_slots > 0) { staging = std::make_unique<DeviceBuffer>(bank.stride * p.staging_slots); }
    DeviceBuffer workspace(ops::moe_experts_narrow_workspace_bytes(om::kDsv4Shape, max_jobs, k * columns, columns, p.precast));
    cuda_ok(cudaMemset(workspace.p, 0xA5, workspace.bytes), "poison the workspace");
    DeviceBuffer outputs(static_cast<std::size_t>(k) * columns * kH * 2);
    cuda_ok(cudaMemset(outputs.p, 0xFF, outputs.bytes), "poison the outputs");
    ops::MoeExpertSource source;
    source.format           = bank.format;
    source.frame_base       = static_cast<const std::uint8_t*>(bank.frames->p);
    source.frames           = static_cast<const std::int32_t*>(d_frames.p);
    source.host_records     = bank.pinned->device;
    source.record_stride    = bank.stride;
    source.staging_base     = staging ? static_cast<std::uint8_t*>(staging->p) : nullptr;
    source.staging_slots    = p.staging_slots;
    source.zero_copy_misses = p.zero_copy_misses;
    source.precast_input    = p.precast;
    const Tensor x(x_dev.p, DType::BF16, {kH, columns});
    Tensor out(outputs.p, DType::BF16, {kH, k * columns});
    ops::moe_experts_narrow(om::kDsv4Shape, x, r.view, source, k, max_jobs, workspace.p, out, nullptr);
    cuda_ok(cudaDeviceSynchronize(), p.name);
    return download<std::uint16_t>(outputs.p, static_cast<std::size_t>(k) * columns * kH);
}

void test_format(QType format, std::mt19937& rng) {
    constexpr int kExperts = 12, kTopK = 6;
    const Bank bank = make_bank(format, kExperts, rng);
    for (const int columns : g_small ? std::vector<int>{1, 3} : std::vector<int>{1, 3, 8, 11}) {
        const Routed r(kExperts, kTopK, columns, rng);
        // Activations large enough that the clamp bites on part of the rows (|gate| or |up| above 10).
        const auto x = random_dsv4_x(columns, rng, 6.0F);
        DeviceBuffer x_dev(x.size() * 2);
        x_dev.copy_from_host(x.data(), x.size() * 2);
        std::vector<std::vector<std::uint16_t>> results;
        for (const Placement& p : kPlacements) { results.push_back(run(bank, p, r, x_dev, columns)); }
        for (std::size_t i = 1; i < results.size(); ++i) {
            check(results[i] == results[0], std::string(record_name(format)) + ": placement " + kPlacements[i].name +
                                                " differs from " + kPlacements[0].name + " at T=" + std::to_string(columns));
        }
        // The FP64 oracle on a sample of entries (decoding a 5120 x 2304 record is the cost): the first column's
        // slots, and the last column's first.
        std::map<int, int> sample; // entry -> 1
        for (int slot = 0; slot < kTopK; ++slot) { sample[slot] = 1; }
        sample[(columns - 1) * kTopK] = 1;
        double worst_norm = 0.0, worst_elem = 0.0;
        int clamped_entries = 0;
        for (const auto& [entry, unused] : sample) {
            (void)unused;
            const int expert = r.host_ids[entry], column = entry / kTopK;
            const auto want  = oracle(format, bank.records[expert].data(), x.data() + static_cast<std::size_t>(column) * kH, true);
            const auto got   = std::span<const std::uint16_t>(results[0]).subspan(static_cast<std::size_t>(entry) * kH, kH);
            const auto c     = measure(got, want);
            check(accept(c), std::string(record_name(format)) + ": entry " + std::to_string(entry) + " T=" + std::to_string(columns) +
                                 " is outside the A8 criterion against the FP64 oracle (normwise " + std::to_string(c.normwise) +
                                 ", worst " + std::to_string(c.worst) + ")");
            worst_norm = std::max(worst_norm, c.normwise);
            worst_elem = std::max(worst_elem, c.worst);
            // The clamp matters in this data: the same expert unclamped gives a different answer.
            const auto unclamped = oracle(format, bank.records[expert].data(), x.data() + static_cast<std::size_t>(column) * kH, false);
            if (!accept(measure(got, unclamped))) { ++clamped_entries; }
        }
        std::printf("%-16s T=%-2d %zu placements bit-identical; %zu entries vs FP64 (clamped): normwise %.2e, worst %.2e; "
                    "the clamp changes %d of them\n",
                    record_name(format), columns, results.size(), sample.size(), worst_norm, worst_elem, clamped_entries);
        if (format == kFormats[0] && columns == 1) {
            check(clamped_entries > 0, "the test data never reaches the clamp: raise the activation scale");
        }
    }
}

template <class F>
bool refuses(F&& f) {
    try {
        f();
    } catch (const std::invalid_argument&) { return true; }
    return false;
}

void test_refusals(std::mt19937& rng) {
    const Bank bank = make_bank(QType::GGML_REC_IQ2_XXS_IQ2_XXS, 4, rng);
    const Routed r(4, 2, 1, rng);
    const auto x = random_dsv4_x(1, rng, 1.0F);
    DeviceBuffer x_dev(x.size() * 2);
    x_dev.copy_from_host(x.data(), x.size() * 2);
    DeviceBuffer d_frames(4 * 4);
    const std::int32_t frames[4] = {0, 1, 2, 3};
    d_frames.copy_from_host(frames, sizeof(frames));
    DeviceBuffer workspace(ops::moe_experts_narrow_workspace_bytes(om::kDsv4Shape, 2, 2, 1));
    DeviceBuffer outputs(static_cast<std::size_t>(2) * kH * 2);
    ops::MoeExpertSource source;
    source.format        = bank.format;
    source.frame_base    = static_cast<const std::uint8_t*>(bank.frames->p);
    source.frames        = static_cast<const std::int32_t*>(d_frames.p);
    source.host_records  = bank.pinned->device;
    source.record_stride = bank.stride;
    const Tensor xt(x_dev.p, DType::BF16, {kH, 1});
    Tensor out(outputs.p, DType::BF16, {kH, 2});
    const auto call = [&](om::MoeShape shape, const ops::MoeExpertSource& s, const Tensor& xin, Tensor& o) {
        ops::moe_experts_narrow(shape, xin, r.view, s, 2, 2, workspace.p, o, nullptr);
    };
    call(om::kDsv4Shape, source, xt, out); // the baseline call works
    cuda_ok(cudaDeviceSynchronize(), "baseline");

    check(refuses([&] { call(om::MoeShape{2560, 640}, source, xt, out); }), "an expert shape with no narrow route built is refused");
    check(refuses([&] { (void)ops::moe_experts_narrow_workspace_bytes(om::MoeShape{4096, 1024}, 2, 2, 1); }),
          "the workspace of an unknown shape is refused");
    ops::MoeExpertSource registers = source;
    registers.gate_up_variant      = ops::GateUpVariant::Registers;
    registers.precast_input        = true;
    check(refuses([&] { call(om::kDsv4Shape, registers, xt, out); }), "the register gate/up variant is refused");
    ops::MoeExpertSource cpu = source;
    om::MissRequest request{};
    std::uint16_t cpu_x[8]{};
    cpu.cpu.request = &request;
    cpu.cpu.x       = cpu_x;
    cpu.cpu.max_jobs = 1;
    check(refuses([&] { call(om::kDsv4Shape, cpu, xt, out); }), "a CPU channel is refused");
    // A Qwen record format (Q2_0 down, a valid geometry at this shape) has no down kernel here.
    ops::MoeExpertSource qwen_format = source;
    qwen_format.format               = QType::GGML_REC_IQ2_XXS_Q2_0;
    check(refuses([&] { call(om::kDsv4Shape, qwen_format, xt, out); }), "a Q2_0 down is refused at the V4 shape");
    // An x of the wrong hidden size.
    const Tensor narrow_x(x_dev.p, DType::BF16, {2560, 1});
    check(refuses([&] { call(om::kDsv4Shape, source, narrow_x, out); }), "an x of another hidden size is refused");
    // record_geometry: the Qwen overload keeps to Qwen's formats, the shape overload takes the K-quant downs.
    check(refuses([] { (void)om::record_geometry(QType::GGML_REC_IQ2_XXS_IQ2_XXS); }), "record_geometry(format) refuses a K-quant down");
    check(!refuses([] { (void)om::record_geometry(QType::GGML_REC_IQ2_XXS_IQ2_XXS, om::kDsv4Shape); }),
          "record_geometry(format, shape) accepts it");
    check(refuses([] { (void)om::record_geometry(QType::GGML_REC_IQ2_XXS_IQ2_XXS, om::MoeShape{2560, 640}); }),
          "an intermediate size that does not tile the K-quant down is refused (640 is 2.5 blocks of 256)");
    std::printf("refusals: ok\n");
}

} // namespace

int main(int argc, char** argv) {
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--small") { g_small = true; }
    }
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
        std::printf("SKIP: no usable CUDA device\n");
        return 77;
    }
    std::mt19937 rng(5120);
    test_refusals(rng);
    const std::vector<QType> formats = g_small ? std::vector<QType>{kFormats[0], kFormats[2], kFormats[7]}
                                               : std::vector<QType>(std::begin(kFormats), std::end(kFormats));
    for (const QType format : formats) { test_format(format, rng); }
    std::printf("offloaded_moe_dsv4: passed\n");
    return 0;
}
