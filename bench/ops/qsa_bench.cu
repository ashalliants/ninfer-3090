// Public benchmark of the QSA Ops (include/ninfer/ops/qsa.h) at the registered geometry: the
// index-query preparation, pooled keys, block selection and attention of one call of T columns
// ending at the last position of a context, each captured alone, and the four in one graph.
// Inputs are random (random pooled keys select scattered blocks); the cache is Int8Group64 with a
// shuffled page table.

#include "ninfer/ops/qsa.h"

#include "ninfer_bench_common.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::bench;

namespace {

constexpr int kDi = 128, kIndexHeads = 4, kR = 4, kTop = 512, kPage = 64, kD = 256, kHq = 24,
              kHkv = 2;

ops::QsaIndexerGeometry geometry() {
    return {.index_heads      = kIndexHeads,
            .index_head_dim   = kDi,
            .rotary_dim       = 64,
            .block_tokens     = kR,
            .budget_tokens    = 2048,
            .theta            = 1.0e7F,
            .eps              = 1.0e-6F,
            .unit_offset_norm = false};
}

std::vector<int> parse_list(const std::string& value) {
    std::vector<int> out;
    std::istringstream input(value);
    std::string item;
    while (std::getline(input, item, ',')) {
        std::size_t end = 0;
        const int n     = std::stoi(item, &end);
        if (n < 1 || end != item.size()) { throw std::invalid_argument("lists hold positive integers"); }
        out.push_back(n);
    }
    if (out.empty()) { throw std::invalid_argument("empty list"); }
    return out;
}

DeviceBuffer upload_i32(const std::vector<std::int32_t>& v) {
    DeviceBuffer d(v.size() * sizeof(std::int32_t));
    d.copy_from_host(v.data(), d.bytes);
    return d;
}

void run(int tokens, int context, int envelope_keys, bool cold, int warmup, int repeat) {
    if (tokens > context) { throw std::invalid_argument("T exceeds the context"); }
    const int first = context - tokens;
    const int pages = (context + kPage - 1) / kPage;
    std::mt19937 rng(static_cast<unsigned>(context * 31 + tokens));
    std::vector<std::int32_t> table(pages);
    std::iota(table.begin(), table.end(), 0);
    std::shuffle(table.begin(), table.end(), rng);
    std::vector<std::int32_t> positions(tokens), rope(3 * static_cast<std::size_t>(tokens)), block_rope(3);
    for (int t = 0; t < tokens; ++t) {
        positions[t] = first + t;
        for (int a = 0; a < 3; ++a) { rope[static_cast<std::size_t>(a) * tokens + t] = first + t; }
    }
    for (int a = 0; a < 3; ++a) { block_rope[a] = first - first % kR; }

    DeviceBuffer d_table = upload_i32(table), d_positions = upload_i32(positions), d_rope = upload_i32(rope),
                 d_block_rope = upload_i32(block_rope);
    DeviceBuffer d_index_q   = make_bf16(static_cast<std::size_t>(kDi) * kIndexHeads * tokens, 11U, -1.0F, 1.0F);
    DeviceBuffer d_raw       = make_bf16(static_cast<std::size_t>(kDi) * tokens, 12U, -1.0F, 1.0F);
    DeviceBuffer d_weight    = make_bf16(kDi, 13U, 0.5F, 1.5F);
    DeviceBuffer d_tail      = make_bf16(static_cast<std::size_t>(kDi) * (kR - 1), 14U, -1.0F, 1.0F);
    DeviceBuffer d_pooled    = make_bf16(static_cast<std::size_t>(kDi / kR) * kPage * pages, 15U, -1.0F, 1.0F);
    DeviceBuffer d_q         = make_bf16(static_cast<std::size_t>(kD) * kHq * tokens, 16U, -1.0F, 1.0F);
    DeviceBuffer d_out(static_cast<std::size_t>(kD) * kHq * tokens * 2);
    DeviceBuffer d_selected(static_cast<std::size_t>(kTop) * tokens * 4), d_counts(static_cast<std::size_t>(tokens) * 4);
    const std::size_t rows = static_cast<std::size_t>(pages) * kHkv * kPage;
    DeviceBuffer d_k = make_zeros(rows * kD), d_v = make_zeros(rows * kD);
    DeviceBuffer d_ks = make_bf16(rows * 4, 17U, 0.004F, 0.012F), d_vs = make_bf16(rows * 4, 18U, 0.004F, 0.012F);
    {
        // Random codes; scales as FP16 bit patterns of small positive values.
        std::vector<std::int8_t> codes(rows * kD);
        for (auto& c : codes) { c = static_cast<std::int8_t>(static_cast<int>(rng() % 255) - 127); }
        d_k.copy_from_host(codes.data(), codes.size());
        for (auto& c : codes) { c = static_cast<std::int8_t>(static_cast<int>(rng() % 255) - 127); }
        d_v.copy_from_host(codes.data(), codes.size());
        std::vector<std::uint16_t> scales(rows * 4, 0x1C00); // 2^-8 in FP16
        d_ks.copy_from_host(scales.data(), scales.size() * 2);
        d_vs.copy_from_host(scales.data(), scales.size() * 2);
    }

    PagedKVLayerView cache;
    cache.k_pages       = Tensor(d_k.p, DType::I8, {kD, kPage, kHkv, pages});
    cache.v_pages       = Tensor(d_v.p, DType::I8, {kD, kPage, kHkv, pages});
    cache.k_scale_pages = Tensor(d_ks.p, DType::FP16, {4, kPage, kHkv, pages});
    cache.v_scale_pages = Tensor(d_vs.p, DType::FP16, {4, kPage, kHkv, pages});
    cache.block_table   = Tensor(d_table.p, DType::I32, {pages});
    cache.head_dim      = kD;
    cache.num_kv_heads  = kHkv;
    cache.storage       = KvCacheStorage::Int8Group64;

    int device = 0, sms = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device));
    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    const DeviceExecutionView exec{stream, sms};
    const ops::QsaExecutionEnvelope envelope{static_cast<std::uint32_t>(std::max(envelope_keys, context))};
    const ops::AttentionHeadGeometry heads{kD, kHq, kHkv};
    const std::size_t scratch =
        std::max(ops::qsa_select_blocks_workspace_capacity_bytes(geometry(), envelope, tokens, tokens, exec),
                 ops::qsa_attention_workspace_capacity_bytes(heads, geometry(), KvCacheStorage::Int8Group64,
                                                             tokens, tokens, exec));
    WorkspaceArena workspace(scratch + 256);

    Tensor index_q(d_index_q.p, DType::BF16, {kDi, kIndexHeads, tokens});
    Tensor raw(d_raw.p, DType::BF16, {kDi, tokens});
    Tensor weight(d_weight.p, DType::BF16, {kDi});
    Tensor tail(d_tail.p, DType::BF16, {kDi, kR - 1});
    Tensor pooled(d_pooled.p, DType::BF16, {kDi / kR, kPage, 1, pages});
    Tensor table_tensor(d_table.p, DType::I32, {pages});
    Tensor positions_tensor(d_positions.p, DType::I32, {tokens});
    Tensor rope_tensor(d_rope.p, DType::I32, {tokens, 3});
    Tensor block_rope_tensor(d_block_rope.p, DType::I32, {3});
    Tensor q(d_q.p, DType::BF16, {kD, kHq, tokens});
    Tensor out(d_out.p, DType::BF16, {kD, kHq, tokens});
    Tensor selected(d_selected.p, DType::I32, {kTop, tokens});
    Tensor counts(d_counts.p, DType::I32, {tokens});

    const auto index_query = [&](cudaStream_t s) { ops::qsa_index_query(rope_tensor, weight, geometry(), index_q, s); };
    const auto pool = [&](cudaStream_t s) {
        ops::qsa_pool_keys(raw, positions_tensor, rope_tensor, block_rope_tensor, weight, geometry(), table_tensor,
                           tail, pooled, s);
    };
    const auto select = [&](cudaStream_t s) {
        ops::qsa_select_blocks(index_q, positions_tensor, table_tensor, pooled, geometry(), envelope, workspace,
                               selected, counts, exec.on_stream(s));
    };
    const auto attention = [&](cudaStream_t s) {
        ops::qsa_attention(q, positions_tensor, selected, counts, heads, geometry(), 0.0625F, cache, workspace, out,
                           exec.on_stream(s));
    };
    // The attention stage reads a real selection.
    select(stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    L2FlushBuffer flush(256ULL << 20);
    const auto time = [&](const char* stage, const std::function<void(cudaStream_t)>& body) {
        TimedGraph graph;
        graph.capture(stream, body);
        const ColdTiming timing = cold ? measure_cold_graph(graph, flush, stream, warmup, repeat)
                                       : measure_graph(graph, stream, warmup, repeat);
        std::printf("%s,%d,%d,%u,%s,%.2f,%.2f,%.2f,%zu\n", stage, tokens, context, envelope.max_visible_keys,
                    cold ? "cold" : "warm",
                    timing.median_us, timing.min_us, timing.p95_us, graph.nodes());
    };
    time("index_query", index_query);
    time("pool_keys", pool);
    time("select", select);
    time("attention", attention);
    time("all", [&](cudaStream_t s) {
        index_query(s);
        pool(s);
        select(s);
        attention(s);
    });
    CUDA_CHECK(cudaStreamDestroy(stream));
}

} // namespace

int main(int argc, char** argv) {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
        std::printf("SKIP: no usable CUDA device\n");
        return 0;
    }
    try {
        std::vector<int> tokens   = {1, 2048};
        std::vector<int> contexts = {8192, 32768, 131072};
        bool cold                 = true;
        int warmup = 5, repeat = 30, envelope = 0;
        for (int i = 1; i < argc; ++i) {
            const std::string flag = argv[i];
            if (flag == "--help") {
                std::printf("usage: %s [--tokens T,...] [--contexts L,...] [--cache cold|warm] [--warmup N] "
                            "[--repeat N]\n",
                            argv[0]);
                return 0;
            }
            if (i + 1 == argc) { throw std::invalid_argument("missing option value"); }
            const std::string value = argv[++i];
            if (flag == "--tokens") {
                tokens = parse_list(value);
            } else if (flag == "--contexts") {
                contexts = parse_list(value);
            } else if (flag == "--cache") {
                if (value != "cold" && value != "warm") { throw std::invalid_argument("--cache cold|warm"); }
                cold = value == "cold";
            } else if (flag == "--envelope") {
                envelope = std::stoi(value);
            } else if (flag == "--warmup") {
                warmup = std::stoi(value);
            } else if (flag == "--repeat") {
                repeat = std::stoi(value);
            } else {
                throw std::invalid_argument("unknown option " + flag);
            }
        }
        std::printf("stage,T,context,envelope,cache,median_us,min_us,p95_us,graph_nodes\n");
        for (const int context : contexts) {
            for (const int t : tokens) { run(t, context, envelope, cold, warmup, repeat); }
        }
    } catch (const std::exception& error) {
        std::fprintf(stderr, "error: %s\n", error.what());
        return 2;
    }
    return 0;
}
