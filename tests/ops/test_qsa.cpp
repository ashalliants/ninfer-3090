// QSA Ops (include/ninfer/ops/qsa.h) against independent FP64 and exact oracles, at the real
// registered geometry (24 query heads, 2 KV heads of 256, a 4 x 128 indexer, 64 rotary dimensions,
// budget 2048 in blocks of 4, the Int8Group64 cache).
//
// Adapted from Infernix fd2aa93c9f8bc716d7ec1fea3c36abba067ad8de tests/ops/test_qsa.cpp and
// tests/ops/test_qsa_select.cu (Apache-2.0): the FP64 key and attention oracles, the selection
// case families (budget boundaries, exact ties, zero-score ties, concentrated and near-tie keys)
// and the grid-valued inputs that make a selection exactly decidable. Modified for NInfer: the
// selection is checked only against FP64 (the FP32 scores are private here), and attention is
// checked both on explicit selections and end to end.
//
//   qsa_index_query  FP64 RMSNorm (plain and unit-offset gain) and interleaved M-RoPE at positions
//                    up to 2^18 with distinct axes.
//   qsa_pool_keys    pooled keys vs FP64 RoPE(norm(bf16(mean))) over several call splittings, the
//                    tail state exactly, untouched pooled slots exactly.
//   qsa_select       count, ascending unique ids, FP64 dominance within the contract's bound, and
//                    lower ids first inside every exact-tie class, at 511/512/513/514 blocks, the
//                    tail, 2K-65K blocks, groups of columns, ties, zero scores, concentrated keys.
//   qsa_attention    FP64 softmax attention over the decoded cache for explicit selections
//                    (decode, split and unsplit widths, short counts) and for the selection the
//                    FP64 scores decide (end to end), dense prefixes equal causal attention, CUDA
//                    Graph replay with new positions, output guards and an unchanged cache.

#include "ninfer/ops/qsa.h"

#include "core/arena.h"
#include "core/paged_kv_cache.h"
#include "ops/op_tester.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <functional>
#include <map>
#include <numeric>
#include <random>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

namespace {

namespace ops = ninfer::ops;
namespace t   = ninfer::test;
using ninfer::DeviceBuffer;
using ninfer::DeviceExecutionView;
using ninfer::DType;
using ninfer::KvCacheStorage;
using ninfer::PagedKVLayerView;
using ninfer::Tensor;
using ninfer::WorkspaceArena;

constexpr int kDi          = 128;
constexpr int kIndexHeads  = 4;
constexpr int kRotary      = 64;
constexpr int kR           = 4;
constexpr int kBudget      = 2048;
constexpr int kTop         = kBudget / kR;
constexpr int kPage        = 64;
constexpr int kSlot        = kDi / kR;
constexpr int kD           = 256;
constexpr int kHq          = 24;
constexpr int kHkv         = 2;
constexpr int kGroupHeads  = kHq / kHkv;
constexpr double kTheta    = 1.0e7;
constexpr float kEps       = 1.0e-6F;
constexpr std::uint16_t kCanary = 0x7fc1;

ops::QsaIndexerGeometry geometry(bool unit_offset) {
    return {.index_heads      = kIndexHeads,
            .index_head_dim   = kDi,
            .rotary_dim       = kRotary,
            .block_tokens     = kR,
            .budget_tokens    = kBudget,
            .theta            = static_cast<float>(kTheta),
            .eps              = kEps,
            .unit_offset_norm = unit_offset};
}

constexpr ops::AttentionHeadGeometry kHeads{.head_dim = kD, .query_heads = kHq, .kv_heads = kHkv};

// Index queries and pooled keys are normalized, rotated vectors evaluated in FP32 and stored as
// BF16: one RNE rounding (relative RMS about 2^-9 / sqrt(3) = 1.1e-3, at most 2^-9 of an element),
// plus FP32 norm and rotation arithmetic well below that. The gross bound is two BF16 steps.
constexpr t::ReductionCriterion kIndexVectorCriterion{
    /*relative_l2*/ 2.0e-3,
    /*gross_absolute*/ 1.0e-3,
    /*gross_relative_to_max_reference*/ t::kBf16GrossRelativeFloor,
};

// QSA attention over the Int8Group64 cache, both routes: FP16 rounding of the Hadamard-prepared
// query, exact FP16 key codes against FP32 key-scale products, one FP16 rounding in P x V (the split
// route rounds probability x value scale, the wide route value code x scale and the probability),
// FP32 accumulation, BF16 output. These are the same terms at the same precisions, so one criterion
// covers both. It is the causal INT8-G64 suite's (whose query is INT8 instead of FP16); measured
// relative L2 is 1.63e-3..1.69e-3 on both routes (NINFER_OP_REPORT_STATS=1), 0.52-0.54 of the bound.
constexpr t::ReductionCriterion kAttentionCriterion{
    /*relative_l2*/ 3.15e-3,
    /*gross_absolute*/ 1.1e-3,
    /*gross_relative_to_max_reference*/ t::kBf16GrossRelativeFloor,
};

int g_failures = 0;

void expect(bool ok, const std::string& what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what.c_str());
        ++g_failures;
    }
}

DeviceExecutionView execution(cudaStream_t stream = nullptr) {
    static const std::int32_t sms = [] {
        int device = 0, count = 0;
        t::cuda_check(cudaGetDevice(&device), "cudaGetDevice");
        t::cuda_check(cudaDeviceGetAttribute(&count, cudaDevAttrMultiProcessorCount, device),
                      "multiprocessor count");
        return static_cast<std::int32_t>(count);
    }();
    return {stream, sms};
}

// ------------------------------------------------------------------------------------- values

double from_bf16(std::uint16_t h) { return t::bf16_to_f32(h); }

// The nearest BF16 of a double, ties to even.
std::uint16_t to_bf16(double v) {
    if (v == 0.0 || !std::isfinite(v)) { return t::f32_to_bf16(static_cast<float>(v)); }
    int exponent          = 0;
    const double mantissa = std::frexp(v, &exponent);
    const double rounded  = std::nearbyint(std::ldexp(mantissa, 8));
    return t::f32_to_bf16(static_cast<float>(std::ldexp(rounded, exponent - 8)));
}

double from_fp16(std::uint16_t bits) {
    __half h;
    std::memcpy(&h, &bits, sizeof(bits));
    return static_cast<double>(__half2float(h));
}

std::uint16_t to_fp16(float v) {
    const __half h = __float2half_rn(v);
    std::uint16_t bits;
    std::memcpy(&bits, &h, sizeof(bits));
    return bits;
}

std::vector<std::uint16_t> random_bf16(std::mt19937& rng, std::size_t n, double scale) {
    std::normal_distribution<double> d(0.0, scale);
    std::vector<std::uint16_t> v(n);
    for (auto& x : v) { x = to_bf16(d(rng)); }
    return v;
}

// BF16 multiples of 2^-4 in [-1, 1]: block scores over them are exact in FP32, so the selection
// is decided exactly, including equal scores.
std::vector<std::uint16_t> grid_bf16(std::mt19937& rng, std::size_t n) {
    std::uniform_int_distribution<int> step(-16, 16);
    std::vector<std::uint16_t> v(n);
    for (auto& x : v) { x = to_bf16(step(rng) / 16.0); }
    return v;
}

template <class T>
DeviceBuffer upload(const std::vector<T>& v) {
    return t::to_device(v);
}

template <class T>
std::vector<T> download(const DeviceBuffer& d, std::size_t n) {
    return t::from_device<T>(d, n);
}

void parallel_for(int n, const std::function<void(int)>& body) {
    const int workers = std::max(1, std::min<int>(n, static_cast<int>(std::thread::hardware_concurrency())));
    std::vector<std::thread> pool;
    for (int w = 0; w < workers; ++w) {
        pool.emplace_back([&, w] {
            for (int i = w; i < n; i += workers) { body(i); }
        });
    }
    for (auto& th : pool) { th.join(); }
}

// --------------------------------------------------------------------------- FP64 key oracle

std::vector<double> normalize(const std::vector<double>& x, const std::vector<std::uint16_t>& weight,
                              bool unit_offset) {
    double squares = 0.0;
    for (const double v : x) { squares += v * v; }
    const double inv = 1.0 / std::sqrt(squares / kDi + static_cast<double>(kEps));
    std::vector<double> out(kDi);
    for (int d = 0; d < kDi; ++d) {
        const double w = from_bf16(weight[d]);
        out[d]         = x[d] * inv * (unit_offset ? 1.0 + w : w);
    }
    return out;
}

using Rope = std::array<int, 3>;

void rotate(std::vector<double>& x, const Rope& rope) {
    constexpr int half = kRotary / 2;
    for (int i = 0; i < half; ++i) {
        const double inv   = std::pow(kTheta, -2.0 * i / kRotary);
        const double angle = rope[i % 3] * inv;
        const double c = std::cos(angle), s = std::sin(angle);
        const double a = x[i], b = x[i + half];
        x[i]        = a * c - b * s;
        x[i + half] = a * s + b * c;
    }
}

// ------------------------------------------------------------------------------- index query

void test_index_query(int columns, bool unit_offset, int max_position, bool text_axes,
                      std::uint32_t seed) {
    std::mt19937 rng(seed);
    const auto q      = random_bf16(rng, static_cast<std::size_t>(kDi) * kIndexHeads * columns, 2.0);
    const auto weight = random_bf16(rng, kDi, unit_offset ? 0.3 : 1.0);
    std::uniform_int_distribution<int> position(0, max_position);
    std::vector<std::int32_t> rope(3 * static_cast<std::size_t>(columns));
    for (int c = 0; c < columns; ++c) {
        const int p = position(rng);
        for (int a = 0; a < 3; ++a) {
            rope[static_cast<std::size_t>(a) * columns + c] = text_axes ? p : position(rng);
        }
    }
    DeviceBuffer dq = upload(q), dw = upload(weight), dp = upload(rope);
    Tensor tq(dq.p, DType::BF16, {kDi, kIndexHeads, columns});
    ops::qsa_index_query(Tensor(dp.p, DType::I32, {columns, 3}), Tensor(dw.p, DType::BF16, {kDi}),
                         geometry(unit_offset), tq, nullptr);
    t::cuda_synchronize();
    const auto got = t::from_device_bf16(dq, q.size());
    std::vector<double> ref;
    ref.reserve(q.size());
    for (int c = 0; c < columns; ++c) {
        for (int h = 0; h < kIndexHeads; ++h) {
            std::vector<double> x(kDi);
            for (int d = 0; d < kDi; ++d) {
                x[d] = from_bf16(q[(static_cast<std::size_t>(c) * kIndexHeads + h) * kDi + d]);
            }
            auto out = normalize(x, weight, unit_offset);
            rotate(out, {rope[c], rope[columns + c], rope[2 * static_cast<std::size_t>(columns) + c]});
            ref.insert(ref.end(), out.begin(), out.end());
        }
    }
    g_failures += t::verify_reduction("qsa_index_query T=" + std::to_string(columns) +
                                          (unit_offset ? " unit-offset" : " plain") +
                                          " max_pos=" + std::to_string(max_position) +
                                          (text_axes ? " text" : " mrope"),
                                      got, ref, kIndexVectorCriterion);
}

// ---------------------------------------------------------------------- pooled keys and tails

struct PoolCase {
    std::string name;
    int first = 0;           // the sequence's first position handled by calls
    std::vector<int> widths; // consecutive calls
    bool unit_offset = true;
};

void test_pool(const PoolCase& c, std::uint32_t seed) {
    std::mt19937 rng(seed);
    int end = c.first;
    for (const int w : c.widths) { end += w; }
    const int pages = (end + kPage - 1) / kPage + 1;
    std::vector<std::int32_t> table(pages);
    std::iota(table.begin(), table.end(), 0);
    std::shuffle(table.begin(), table.end(), rng);
    const auto raw    = random_bf16(rng, static_cast<std::size_t>(end) * kDi, 1.0);
    const auto weight = random_bf16(rng, kDi, 0.3);
    // RoPE positions: distinct axes per token, so a rotation by the wrong token or axis shows.
    std::vector<Rope> rope(end);
    for (int p = 0; p < end; ++p) { rope[p] = {p, p + 7 * (p % 5), p + 3 * (p % 11)}; }

    DeviceBuffer d_table = upload(table), d_weight = upload(weight);
    DeviceBuffer d_pooled = upload(std::vector<std::uint16_t>(static_cast<std::size_t>(kSlot) * kPage * pages, kCanary));
    // The tail starts as the latest block's keys before `first`, other columns random.
    auto tail = random_bf16(rng, static_cast<std::size_t>(kDi) * (kR - 1), 1.0);
    for (int q = c.first - c.first % kR; q < c.first; ++q) {
        std::copy_n(raw.begin() + static_cast<std::ptrdiff_t>(q) * kDi, kDi,
                    tail.begin() + static_cast<std::ptrdiff_t>(q % kR) * kDi);
    }
    DeviceBuffer d_tail = upload(tail);
    Tensor pooled(d_pooled.p, DType::BF16, {kSlot, kPage, 1, pages});
    Tensor tail_tensor(d_tail.p, DType::BF16, {kDi, kR - 1});

    int start = c.first;
    for (std::size_t call = 0; call < c.widths.size(); ++call) {
        const int width = c.widths[call];
        std::vector<std::uint16_t> keys(raw.begin() + static_cast<std::ptrdiff_t>(start) * kDi,
                                        raw.begin() + static_cast<std::ptrdiff_t>(start + width) * kDi);
        std::vector<std::int32_t> positions(width), rope_columns(3 * static_cast<std::size_t>(width)),
            block_start(3);
        for (int j = 0; j < width; ++j) {
            positions[j] = start + j;
            for (int a = 0; a < 3; ++a) {
                rope_columns[static_cast<std::size_t>(a) * width + j] = rope[start + j][a];
            }
        }
        for (int a = 0; a < 3; ++a) { block_start[a] = rope[start - start % kR][a]; }
        DeviceBuffer dk = upload(keys), dp = upload(positions), dr = upload(rope_columns),
                     db = upload(block_start);
        ops::qsa_pool_keys(Tensor(dk.p, DType::BF16, {kDi, width}), Tensor(dp.p, DType::I32, {width}),
                           Tensor(dr.p, DType::I32, {width, 3}), Tensor(db.p, DType::I32, {3}),
                           Tensor(d_weight.p, DType::BF16, {kDi}), geometry(c.unit_offset),
                           Tensor(d_table.p, DType::I32, {pages}), tail_tensor, pooled, nullptr);
        t::cuda_synchronize();
        start += width;

        // Tail: the latest block's positions q % R < R - 1 up to the last position.
        const int last = start - 1, base = last - last % kR;
        const auto got_tail = download<std::uint16_t>(d_tail, tail.size());
        for (int q = base; q <= std::min(last, base + kR - 2); ++q) {
            expect(std::equal(got_tail.begin() + static_cast<std::ptrdiff_t>(q % kR) * kDi,
                              got_tail.begin() + static_cast<std::ptrdiff_t>(q % kR + 1) * kDi,
                              raw.begin() + static_cast<std::ptrdiff_t>(q) * kDi),
                   c.name + ": tail column " + std::to_string(q % kR) + " after call " +
                       std::to_string(call) + " holds position " + std::to_string(q));
        }
    }

    // Every block completed by a call against FP64; every other slot untouched.
    const auto got = download<std::uint16_t>(d_pooled, static_cast<std::size_t>(kSlot) * kPage * pages);
    std::vector<double> actual, reference;
    std::vector<char> pooled_block(pages * kPage / kR, 0);
    for (int b = 0; b < end / kR; ++b) {
        if (kR * b + kR - 1 < c.first) { continue; } // completed before the first call
        std::vector<double> mean(kDi);
        for (int d = 0; d < kDi; ++d) {
            double sum = 0.0;
            for (int j = 0; j < kR; ++j) { sum += from_bf16(raw[static_cast<std::size_t>(kR * b + j) * kDi + d]); }
            mean[d] = from_bf16(to_bf16(sum / kR));
        }
        auto ref = normalize(mean, weight, c.unit_offset);
        rotate(ref, rope[kR * b]);
        const int token = kR * b;
        const std::size_t at = (static_cast<std::size_t>(table[token / kPage]) * kPage + token % kPage) * kSlot;
        for (int d = 0; d < kDi; ++d) {
            actual.push_back(from_bf16(got[at + d]));
            reference.push_back(ref[d]);
        }
        pooled_block[static_cast<std::size_t>(table[token / kPage]) * (kPage / kR) + (token % kPage) / kR] = 1;
    }
    g_failures += t::verify_reduction("qsa_pool_keys " + c.name, actual, reference, kIndexVectorCriterion);
    std::size_t touched = 0;
    for (std::size_t slot_block = 0; slot_block < pooled_block.size(); ++slot_block) {
        if (pooled_block[slot_block]) { continue; }
        for (int e = 0; e < kDi; ++e) { touched += got[slot_block * kDi + e] != kCanary; }
    }
    expect(touched == 0, c.name + ": pooled slots of blocks no call completed stay untouched");
}

// ---------------------------------------------------------------------------- selection

enum class Keys { Random, Ties, ZeroHeavy, Concentrated, NearTies, Grid };

struct SelectFixture {
    int columns = 0, pages = 0, blocks = 0;
    std::vector<std::int32_t> positions, table;
    std::vector<std::uint16_t> q;      // [columns][4][128]
    std::vector<std::uint16_t> keys;   // [blocks][128] (logical block order)
    DeviceBuffer d_q, d_pooled, d_table;
};

std::vector<std::uint16_t> perturbed(std::mt19937& rng, const std::vector<std::uint16_t>& base, double mean,
                                     double noise) {
    std::normal_distribution<double> d(0.0, 1.0);
    std::vector<std::uint16_t> v(base.size());
    for (std::size_t i = 0; i < v.size(); ++i) { v[i] = to_bf16(mean * from_bf16(base[i]) + noise * d(rng)); }
    return v;
}

SelectFixture build_select(const std::vector<std::int32_t>& positions, Keys keys, int extra,
                           std::uint32_t seed) {
    SelectFixture f;
    std::mt19937 rng(seed);
    f.columns   = static_cast<int>(positions.size());
    f.positions = positions;
    const int max_position = *std::max_element(positions.begin(), positions.end());
    f.blocks    = (max_position + 1) / kR;
    f.pages     = (f.blocks * kR + kPage - 1) / kPage + 1;
    const auto u = random_bf16(rng, kDi, 1.0);
    for (int i = 0; i < f.columns * kIndexHeads; ++i) {
        const auto v = keys == Keys::ZeroHeavy ? perturbed(rng, u, 1.0, 0.1)
                       : keys == Keys::Grid    ? grid_bf16(rng, kDi)
                                               : random_bf16(rng, kDi, 1.0);
        f.q.insert(f.q.end(), v.begin(), v.end());
    }
    std::vector<std::vector<std::uint16_t>> period;
    for (int i = 0; i < extra && keys == Keys::Ties; ++i) { period.push_back(random_bf16(rng, kDi, 1.0)); }
    std::vector<char> positive(f.blocks, 0);
    if (keys == Keys::ZeroHeavy) {
        std::vector<int> ids(f.blocks);
        std::iota(ids.begin(), ids.end(), 0);
        std::shuffle(ids.begin(), ids.end(), rng);
        for (int i = 0; i < std::min(extra, f.blocks); ++i) { positive[ids[i]] = 1; }
    }
    const auto base = random_bf16(rng, kDi, 1.0);
    f.keys.reserve(static_cast<std::size_t>(f.blocks) * kDi);
    for (int b = 0; b < f.blocks; ++b) {
        std::vector<std::uint16_t> v;
        switch (keys) {
        case Keys::Random: v = random_bf16(rng, kDi, 1.0); break;
        case Keys::Grid: v = grid_bf16(rng, kDi); break;
        case Keys::Ties: v = period[b % extra]; break;
        case Keys::ZeroHeavy: v = perturbed(rng, u, positive[b] ? 1.0 : -1.0, 0.3); break;
        case Keys::Concentrated: v = perturbed(rng, base, 1.0, 0.02); break;
        case Keys::NearTies:
            if (b % 2 == 0) {
                v = random_bf16(rng, kDi, 1.0);
            } else {
                v.assign(f.keys.end() - kDi, f.keys.end());
                const int e = (b / 2) % kDi;
                v[e]        = static_cast<std::uint16_t>(v[e] + 1);
            }
            break;
        }
        f.keys.insert(f.keys.end(), v.begin(), v.end());
    }
    f.table.resize(f.pages);
    std::iota(f.table.begin(), f.table.end(), 0);
    std::shuffle(f.table.begin(), f.table.end(), rng);
    std::vector<std::uint16_t> pooled(static_cast<std::size_t>(kSlot) * kPage * f.pages, 0);
    for (int b = 0; b < f.blocks; ++b) {
        const int token = kR * b;
        const std::size_t at = (static_cast<std::size_t>(f.table[token / kPage]) * kPage + token % kPage) * kSlot;
        std::copy_n(f.keys.begin() + static_cast<std::ptrdiff_t>(b) * kDi, kDi, pooled.begin() + static_cast<std::ptrdiff_t>(at));
    }
    f.d_q      = upload(f.q);
    f.d_pooled = upload(pooled);
    f.d_table  = upload(f.table);
    return f;
}

struct Fp64Scores {
    std::vector<double> score, bound;
    std::vector<char> zero; // every head's dot provably negative: the evaluated score is exactly 0
};

Fp64Scores fp64_scores(const SelectFixture& f, int column) {
    const int n = (f.positions[column] + 1) / kR;
    Fp64Scores out;
    out.score.resize(n);
    out.bound.resize(n);
    out.zero.resize(n);
    const double scale = 1.0 / std::sqrt(static_cast<double>(kDi));
    const std::uint16_t* q = f.q.data() + static_cast<std::size_t>(column) * kIndexHeads * kDi;
    parallel_for(n, [&](int b) {
        const std::uint16_t* k = f.keys.data() + static_cast<std::size_t>(b) * kDi;
        double s = 0.0, a = 0.0;
        bool zero = true;
        for (int h = 0; h < kIndexHeads; ++h) {
            double dot = 0.0, magnitude = 0.0;
            for (int d = 0; d < kDi; ++d) {
                const double p = from_bf16(q[h * kDi + d]) * from_bf16(k[d]);
                dot += p;
                magnitude += std::fabs(p);
            }
            s += std::max(dot, 0.0);
            a += magnitude;
            zero = zero && dot + std::ldexp(magnitude, -17) < 0.0;
        }
        out.score[b] = s * scale;
        out.bound[b] = std::ldexp(a * scale, -17);
        out.zero[b]  = zero ? 1 : 0;
    });
    return out;
}

struct SelectResult {
    std::vector<std::int32_t> selected, counts;
};

SelectResult run_select(const SelectFixture& f, ops::QsaExecutionEnvelope envelope, cudaStream_t stream,
                        const std::string& tag) {
    const auto exec = execution(stream);
    const std::size_t scratch =
        ops::qsa_select_blocks_workspace_capacity_bytes(geometry(true), envelope, f.columns, f.columns, exec);
    WorkspaceArena arena(scratch + 256);
    // Garbage scratch: the Op must not depend on its contents.
    t::cuda_check(cudaMemset(arena.base(), 0x5A, arena.capacity()), "garbage scratch");
    DeviceBuffer d_positions = upload(f.positions);
    t::GuardedDeviceBuffer selected(sizeof(std::int32_t) * kTop * f.columns);
    t::GuardedDeviceBuffer counts(sizeof(std::int32_t) * f.columns);
    selected.fill(0x7F);
    counts.fill(0x7F);
    Tensor selected_tensor(selected.data(), DType::I32, {kTop, f.columns});
    Tensor counts_tensor(counts.data(), DType::I32, {f.columns});
    ops::qsa_select_blocks(Tensor(f.d_q.p, DType::BF16, {kDi, kIndexHeads, f.columns}),
                           Tensor(d_positions.p, DType::I32, {f.columns}),
                           Tensor(f.d_table.p, DType::I32, {f.pages}),
                           Tensor(f.d_pooled.p, DType::BF16, {kSlot, kPage, 1, f.pages}), geometry(true),
                           envelope, arena, selected_tensor, counts_tensor, exec);
    t::cuda_synchronize(stream);
    SelectResult r;
    r.selected.resize(static_cast<std::size_t>(kTop) * f.columns);
    r.counts.resize(f.columns);
    selected.copy_to_host(r.selected.data(), r.selected.size() * 4);
    counts.copy_to_host(r.counts.data(), r.counts.size() * 4);
    expect(selected.verify_guards(tag + " selected") == 0, tag + ": selected guards intact");
    expect(counts.verify_guards(tag + " counts") == 0, tag + ": counts guards intact");
    return r;
}

// Hash of a key vector, to find exact-tie classes of identical keys.
std::uint64_t key_hash(const std::uint16_t* k) {
    std::uint64_t h = 1469598103934665603ULL;
    for (int d = 0; d < kDi; ++d) { h = (h ^ k[d]) * 1099511628211ULL; }
    return h;
}

// Returns the selected flags of a column after checking count, order, FP64 dominance and ties.
std::vector<char> check_selection(const SelectFixture& f, const SelectResult& r, int column,
                                  const std::string& tag) {
    const int n = (f.positions[column] + 1) / kR;
    const std::string where = tag + " column " + std::to_string(column) + " (" + std::to_string(n) + " blocks)";
    const int expected = std::min(n, kTop);
    std::vector<char> in(n, 0);
    if (r.counts[column] != expected) {
        expect(false, where + ": count " + std::to_string(r.counts[column]) + ", expected " + std::to_string(expected));
        return in;
    }
    const std::int32_t* ids = r.selected.data() + static_cast<std::size_t>(column) * kTop;
    bool ascending = true;
    for (int i = 0; i < expected; ++i) {
        ascending = ascending && ids[i] >= 0 && ids[i] < n && (i == 0 || ids[i] > ids[i - 1]);
    }
    expect(ascending, where + ": ids ascend strictly inside [0, n)");
    if (!ascending) { return in; }
    for (int i = 0; i < expected; ++i) { in[ids[i]] = 1; }
    if (n <= kTop) { return in; }
    const Fp64Scores o = fp64_scores(f, column);
    double lowest_selected = INFINITY, highest_unselected = -INFINITY;
    for (int b = 0; b < n; ++b) {
        if (in[b]) {
            lowest_selected = std::min(lowest_selected, o.score[b] + o.bound[b]);
        } else {
            highest_unselected = std::max(highest_unselected, o.score[b] - o.bound[b]);
        }
    }
    expect(highest_unselected <= lowest_selected,
           where + ": an unselected block outranks a selected one beyond the score bound");
    // Exact-tie classes: identical keys, and provably zero scores. Inside a class the selected
    // blocks must be the lowest ids.
    std::map<std::uint64_t, std::vector<int>> classes;
    for (int b = 0; b < n; ++b) {
        classes[o.zero[b] ? 0ULL : key_hash(f.keys.data() + static_cast<std::size_t>(b) * kDi)].push_back(b);
    }
    for (const auto& [hash, members] : classes) {
        if (members.size() < 2) { continue; }
        bool seen_unselected = false, ok = true;
        for (const int b : members) {
            if (!in[b]) { seen_unselected = true; }
            if (in[b] && seen_unselected) { ok = false; }
        }
        expect(ok, where + ": equal scores are taken in ascending block id (class of " +
                       std::to_string(members.size()) + ")");
    }
    return in;
}

struct SelectCase {
    std::string name;
    std::vector<std::int32_t> positions;
    Keys keys = Keys::Random;
    int extra = 0;
    std::uint32_t envelope = 0; // 0: the largest position + 1
};

void test_select(const SelectCase& c, std::uint32_t seed) {
    const SelectFixture f = build_select(c.positions, c.keys, c.extra, seed);
    const std::uint32_t envelope =
        c.envelope != 0 ? c.envelope
                        : static_cast<std::uint32_t>(*std::max_element(c.positions.begin(), c.positions.end()) + 1);
    // Up to 8 columns, the selection route depends on the envelope (above 32K keys: the sliced
    // route): a narrow case under a small envelope also runs under a 128K one, so both routes see
    // every key family and the budget boundaries.
    if (f.columns <= 8 && envelope <= 32768) {
        const SelectResult wide_envelope = run_select(f, {131072}, nullptr, c.name + " (128K envelope)");
        for (int column = 0; column < f.columns; ++column) {
            (void)check_selection(f, wide_envelope, column, c.name + " (128K envelope)");
        }
    }
    const SelectResult r = run_select(f, {envelope}, nullptr, c.name);
    // FP64 on at most 8 spread columns of wide calls; counts and order on all.
    std::vector<int> deep;
    const int want = std::min(f.columns, 8);
    for (int i = 0; i < want; ++i) {
        deep.push_back(want == 1 ? 0 : static_cast<int>(static_cast<std::int64_t>(f.columns - 1) * i / (want - 1)));
    }
    for (int column = 0; column < f.columns; ++column) {
        const int n = (f.positions[column] + 1) / kR;
        const bool full = std::find(deep.begin(), deep.end(), column) != deep.end() || n <= kTop;
        if (full) {
            (void)check_selection(f, r, column, c.name);
        } else {
            const int expected = std::min(n, kTop);
            expect(r.counts[column] == expected, c.name + ": count of column " + std::to_string(column));
        }
    }
    std::printf("  select %-44s %5d columns, %6d..%6d blocks\n", c.name.c_str(), f.columns,
                (c.positions.front() + 1) / kR, (c.positions.back() + 1) / kR);
}

void test_select_graph() {
    // One capture replayed with new positions over garbage scratch.
    std::vector<std::int32_t> positions = {2100, 9000};
    SelectFixture f = build_select({2100, 130000}, Keys::Random, 0, 4242);
    f.positions     = positions;
    cudaStream_t stream = nullptr;
    t::cuda_check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "stream");
    const ops::QsaExecutionEnvelope envelope{131072};
    const auto exec = execution(stream);
    WorkspaceArena arena(ops::qsa_select_blocks_workspace_capacity_bytes(geometry(true), envelope, 2, 2, exec) + 256);
    DeviceBuffer d_positions = upload(positions);
    DeviceBuffer selected(sizeof(std::int32_t) * kTop * 2), counts(sizeof(std::int32_t) * 2);
    Tensor selected_tensor(selected.p, DType::I32, {kTop, 2});
    Tensor counts_tensor(counts.p, DType::I32, {2});
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t exec_graph = nullptr;
    t::cuda_check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal), "begin capture");
    ops::qsa_select_blocks(Tensor(f.d_q.p, DType::BF16, {kDi, kIndexHeads, 2}),
                           Tensor(d_positions.p, DType::I32, {2}), Tensor(f.d_table.p, DType::I32, {f.pages}),
                           Tensor(f.d_pooled.p, DType::BF16, {kSlot, kPage, 1, f.pages}), geometry(true),
                           envelope, arena, selected_tensor, counts_tensor, exec);
    t::cuda_check(cudaStreamEndCapture(stream, &graph), "end capture");
    t::cuda_check(cudaGraphInstantiate(&exec_graph, graph, 0), "instantiate");
    const std::vector<std::vector<std::int32_t>> replays = {{2100, 9000}, {2047, 130000}, {60000, 2051}, {129999, 4000}};
    for (std::size_t i = 0; i < replays.size(); ++i) {
        f.positions = replays[i];
        t::cuda_check(cudaMemcpyAsync(d_positions.p, f.positions.data(), 8, cudaMemcpyHostToDevice, stream), "positions");
        t::cuda_check(cudaMemsetAsync(arena.base(), 0x33 + static_cast<int>(i), arena.capacity(), stream), "garbage");
        t::cuda_check(cudaMemsetAsync(selected.p, 0x7F, selected.bytes, stream), "garbage selected");
        t::cuda_check(cudaGraphLaunch(exec_graph, stream), "graph launch");
        t::cuda_synchronize(stream);
        SelectResult r;
        r.selected = download<std::int32_t>(selected, static_cast<std::size_t>(kTop) * 2);
        r.counts   = download<std::int32_t>(counts, 2);
        for (int column = 0; column < 2; ++column) {
            (void)check_selection(f, r, column, "graph replay " + std::to_string(i));
        }
    }
    t::cuda_check(cudaGraphExecDestroy(exec_graph), "destroy exec");
    t::cuda_check(cudaGraphDestroy(graph), "destroy graph");
    t::cuda_check(cudaStreamDestroy(stream), "destroy stream");
    std::printf("  select graph replays %zu\n", replays.size());
}

// ---------------------------------------------------------------------------- attention

// The normalized Sylvester transform of one 256-vector (its own inverse).
void hadamard(std::vector<double>& x) {
    for (int span = 1; span < kD; span <<= 1) {
        for (int base = 0; base < kD; base += 2 * span) {
            for (int i = base; i < base + span; ++i) {
                const double a = x[i], b = x[i + span];
                x[i]        = a + b;
                x[i + span] = a - b;
            }
        }
    }
    for (auto& v : x) { v /= 16.0; }
}

// A paged Int8Group64 cache with random codes and FP16 scales. Logical pages map onto a smaller
// physical pool (pages may repeat), so long contexts stay small; the Op only reads.
struct Cache {
    int physical_pages = 0, logical_pages = 0;
    std::vector<std::int8_t> k, v;          // [pages][Hkv][64][256]
    std::vector<std::uint16_t> ks, vs;      // [pages][Hkv][64][4]
    std::vector<std::int32_t> table;
    DeviceBuffer dk, dv, dks, dvs, dtable;

    Cache(int positions, int physical, std::mt19937& rng) {
        logical_pages  = (positions + kPage - 1) / kPage;
        physical_pages = std::min(physical, logical_pages);
        const std::size_t rows = static_cast<std::size_t>(physical_pages) * kHkv * kPage;
        k.resize(rows * kD);
        v.resize(rows * kD);
        ks.resize(rows * 4);
        vs.resize(rows * 4);
        std::uniform_int_distribution<int> code(-127, 127);
        std::uniform_real_distribution<float> scale(0.004F, 0.012F);
        for (auto& x : k) { x = static_cast<std::int8_t>(code(rng)); }
        for (auto& x : v) { x = static_cast<std::int8_t>(code(rng)); }
        for (auto& x : ks) { x = to_fp16(scale(rng)); }
        for (auto& x : vs) { x = to_fp16(scale(rng)); }
        table.resize(logical_pages);
        std::vector<std::int32_t> ids(physical_pages);
        std::iota(ids.begin(), ids.end(), 0);
        std::shuffle(ids.begin(), ids.end(), rng);
        for (int i = 0; i < logical_pages; ++i) {
            table[i] = i < physical_pages ? ids[i] : static_cast<std::int32_t>(rng() % physical_pages);
        }
        dk = upload(k), dv = upload(v), dks = upload(ks), dvs = upload(vs), dtable = upload(table);
    }

    PagedKVLayerView view() const {
        PagedKVLayerView view;
        view.k_pages       = Tensor(dk.p, DType::I8, {kD, kPage, kHkv, physical_pages});
        view.v_pages       = Tensor(dv.p, DType::I8, {kD, kPage, kHkv, physical_pages});
        view.k_scale_pages = Tensor(dks.p, DType::FP16, {4, kPage, kHkv, physical_pages});
        view.v_scale_pages = Tensor(dvs.p, DType::FP16, {4, kPage, kHkv, physical_pages});
        view.block_table   = Tensor(dtable.p, DType::I32, {logical_pages});
        view.head_dim      = kD;
        view.num_kv_heads  = kHkv;
        view.storage       = KvCacheStorage::Int8Group64;
        return view;
    }

    std::size_t row(int position, int head) const {
        const int page = table[position / kPage];
        return (static_cast<std::size_t>(page) * kHkv + head) * kPage + position % kPage;
    }

    // Key in original coordinates (the stored row is Hadamard-prepared) and value, exactly.
    std::vector<double> key(int position, int head) const {
        const std::size_t r = row(position, head);
        std::vector<double> x(kD);
        for (int d = 0; d < kD; ++d) { x[d] = k[r * kD + d] * from_fp16(ks[r * 4 + d / 64]); }
        hadamard(x);
        return x;
    }
    std::vector<double> value(int position, int head) const {
        const std::size_t r = row(position, head);
        std::vector<double> x(kD);
        for (int d = 0; d < kD; ++d) { x[d] = v[r * kD + d] * from_fp16(vs[r * 4 + d / 64]); }
        return x;
    }
};

// FP64 attention of one column over token set J.
void attention_oracle(const Cache& cache, const std::vector<std::uint16_t>& q, int column,
                      const std::vector<int>& tokens, double scale, double* out /*[24][256]*/) {
    for (int kvh = 0; kvh < kHkv; ++kvh) {
        std::vector<std::vector<double>> keys, values;
        keys.reserve(tokens.size());
        values.reserve(tokens.size());
        for (const int p : tokens) {
            keys.push_back(cache.key(p, kvh));
            values.push_back(cache.value(p, kvh));
        }
        for (int r = 0; r < kGroupHeads; ++r) {
            const int h = kvh * kGroupHeads + r;
            const std::uint16_t* qh = q.data() + (static_cast<std::size_t>(column) * kHq + h) * kD;
            std::vector<double> s(tokens.size());
            double m = -INFINITY;
            for (std::size_t j = 0; j < tokens.size(); ++j) {
                double dot = 0.0;
                for (int d = 0; d < kD; ++d) { dot += from_bf16(qh[d]) * keys[j][d]; }
                s[j] = dot * scale;
                m    = std::max(m, s[j]);
            }
            double sum = 0.0;
            for (auto& x : s) {
                x = std::exp(x - m);
                sum += x;
            }
            double* o = out + static_cast<std::size_t>(h) * kD;
            for (int d = 0; d < kD; ++d) { o[d] = 0.0; }
            for (std::size_t j = 0; j < tokens.size(); ++j) {
                const double w = s[j] / sum;
                for (int d = 0; d < kD; ++d) { o[d] += w * values[j][d]; }
            }
        }
    }
}

std::vector<int> attended(int p, int count, const std::int32_t* ids) {
    std::vector<int> tokens;
    for (int i = 0; i < count; ++i) {
        for (int j = 0; j < kR; ++j) { tokens.push_back(ids[i] * kR + j); }
    }
    for (int x = (p + 1) / kR * kR; x <= p; ++x) { tokens.push_back(x); }
    return tokens;
}

struct AttentionRun {
    std::vector<double> out;
    bool guards_ok = true;
};

AttentionRun run_attention(const Cache& cache, const std::vector<std::uint16_t>& q,
                           const std::vector<std::int32_t>& positions,
                           const std::vector<std::int32_t>& selected,
                           const std::vector<std::int32_t>& counts, double scale) {
    const int columns = static_cast<int>(positions.size());
    const auto exec   = execution();
    const std::size_t scratch = ops::qsa_attention_workspace_capacity_bytes(
        kHeads, geometry(true), KvCacheStorage::Int8Group64, columns, columns, exec);
    WorkspaceArena arena(scratch + 256);
    t::cuda_check(cudaMemset(arena.base(), 0x5A, arena.capacity()), "garbage scratch");
    DeviceBuffer dq = upload(q), dp = upload(positions), ds = upload(selected), dc = upload(counts);
    t::GuardedDeviceBuffer out(sizeof(std::uint16_t) * kD * kHq * columns);
    out.fill(0x7F);
    Tensor out_tensor(out.data(), DType::BF16, {kD, kHq, columns});
    ops::qsa_attention(Tensor(dq.p, DType::BF16, {kD, kHq, columns}), Tensor(dp.p, DType::I32, {columns}),
                       Tensor(ds.p, DType::I32, {kTop, columns}), Tensor(dc.p, DType::I32, {columns}), kHeads,
                       geometry(true), static_cast<float>(scale), cache.view(), arena, out_tensor, exec);
    t::cuda_synchronize();
    AttentionRun run;
    std::vector<std::uint16_t> raw(static_cast<std::size_t>(kD) * kHq * columns);
    out.copy_to_host(raw.data(), raw.size() * 2);
    run.out.resize(raw.size());
    for (std::size_t i = 0; i < raw.size(); ++i) { run.out[i] = from_bf16(raw[i]); }
    run.guards_ok = out.verify_guards("qsa_attention out") == 0;
    return run;
}

std::vector<double> oracle_all(const Cache& cache, const std::vector<std::uint16_t>& q,
                               const std::vector<std::int32_t>& positions,
                               const std::vector<std::int32_t>& selected,
                               const std::vector<std::int32_t>& counts, double scale) {
    const int columns = static_cast<int>(positions.size());
    std::vector<double> ref(static_cast<std::size_t>(kD) * kHq * columns);
    parallel_for(columns, [&](int c) {
        const auto tokens = attended(positions[c], counts[c], selected.data() + static_cast<std::size_t>(c) * kTop);
        attention_oracle(cache, q, c, tokens, scale, ref.data() + static_cast<std::size_t>(c) * kHq * kD);
    });
    return ref;
}

enum class Selection { Dense, Random, Short };

struct AttentionCase {
    std::string name;
    int first;   // position of the first column
    int columns; // consecutive positions
    Selection selection = Selection::Random;
};

void test_attention(const AttentionCase& c, std::uint32_t seed) {
    std::mt19937 rng(seed);
    const int end = c.first + c.columns;
    Cache cache(end, 640, rng);
    const auto q = random_bf16(rng, static_cast<std::size_t>(kD) * kHq * c.columns, 1.0);
    std::vector<std::int32_t> positions(c.columns), counts(c.columns);
    std::vector<std::int32_t> selected(static_cast<std::size_t>(kTop) * c.columns, -1);
    for (int i = 0; i < c.columns; ++i) {
        const int p = c.first + i, n = (p + 1) / kR;
        positions[i] = p;
        std::int32_t* ids = selected.data() + static_cast<std::size_t>(i) * kTop;
        if (c.selection == Selection::Dense && n > kTop) {
            throw std::logic_error("a dense case beyond the budget");
        }
        int count = std::min(n, kTop);
        if (c.selection == Selection::Short) { count = std::min(count, 1 + static_cast<int>(rng() % 100)); }
        std::vector<int> all(n);
        std::iota(all.begin(), all.end(), 0);
        if (c.selection != Selection::Dense) { std::shuffle(all.begin(), all.end(), rng); }
        std::sort(all.begin(), all.begin() + count);
        std::copy_n(all.begin(), count, ids);
        counts[i] = count;
    }
    const double scale = 1.0 / 16.0;
    const auto k_before = download<std::int8_t>(cache.dk, cache.k.size());
    const AttentionRun run = run_attention(cache, q, positions, selected, counts, scale);
    const auto ref         = oracle_all(cache, q, positions, selected, counts, scale);
    g_failures += t::verify_reduction("qsa_attention " + c.name, run.out, ref, kAttentionCriterion);
    expect(run.guards_ok, c.name + ": output guards intact");
    expect(download<std::int8_t>(cache.dk, cache.k.size()) == k_before, c.name + ": key plane unchanged");
}

// Select (decided exactly by grid-valued indexer inputs) then attention, against FP64 selection
// and FP64 attention.
void test_end_to_end(const std::string& name, int first, int columns, std::uint32_t seed) {
    std::vector<std::int32_t> positions(columns);
    std::iota(positions.begin(), positions.end(), first);
    const SelectFixture f = build_select(positions, Keys::Grid, 0, seed);
    const std::uint32_t envelope = static_cast<std::uint32_t>(positions.back() + 1);
    const SelectResult r = run_select(f, {envelope}, nullptr, name);
    std::mt19937 rng(seed + 1);
    Cache cache(positions.back() + 1, 640, rng);
    const auto q = random_bf16(rng, static_cast<std::size_t>(kD) * kHq * columns, 1.0);
    // The FP64 selection: top scores, lower ids on equal scores (grid inputs make them exact).
    std::vector<std::int32_t> expected(static_cast<std::size_t>(kTop) * columns, -1), counts(columns);
    for (int c = 0; c < columns; ++c) {
        const int n = (positions[c] + 1) / kR;
        std::vector<int> ids(n);
        std::iota(ids.begin(), ids.end(), 0);
        if (n > kTop) {
            const Fp64Scores o = fp64_scores(f, c);
            std::stable_sort(ids.begin(), ids.end(), [&](int a, int b) { return o.score[a] > o.score[b]; });
            ids.resize(kTop);
            std::sort(ids.begin(), ids.end());
        }
        counts[c] = static_cast<int>(ids.size());
        std::copy(ids.begin(), ids.end(), expected.begin() + static_cast<std::ptrdiff_t>(c) * kTop);
        expect(r.counts[c] == counts[c] &&
                   std::equal(ids.begin(), ids.end(), r.selected.begin() + static_cast<std::ptrdiff_t>(c) * kTop),
               name + ": column " + std::to_string(c) + " selects exactly the FP64 top blocks");
    }
    const AttentionRun run = run_attention(cache, q, positions, r.selected, r.counts, 1.0 / 16.0);
    const auto ref         = oracle_all(cache, q, positions, expected, counts, 1.0 / 16.0);
    g_failures += t::verify_reduction("qsa end to end " + name, run.out, ref, kAttentionCriterion);
    expect(run.guards_ok, name + ": output guards intact");
}

// One captured decode call replayed with new positions and selections.
void test_attention_graph() {
    std::mt19937 rng(99);
    Cache cache(40000, 640, rng);
    const auto q = random_bf16(rng, static_cast<std::size_t>(kD) * kHq, 1.0);
    cudaStream_t stream = nullptr;
    t::cuda_check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "stream");
    const auto exec = execution(stream);
    WorkspaceArena arena(ops::qsa_attention_workspace_capacity_bytes(kHeads, geometry(true),
                                                                     KvCacheStorage::Int8Group64, 1, 1, exec) + 256);
    DeviceBuffer dq = upload(q), dp(4), ds(4 * kTop), dc(4), dout(2 * kD * kHq);
    Tensor out(dout.p, DType::BF16, {kD, kHq, 1});
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t exec_graph = nullptr;
    t::cuda_check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal), "begin capture");
    ops::qsa_attention(Tensor(dq.p, DType::BF16, {kD, kHq, 1}), Tensor(dp.p, DType::I32, {1}),
                       Tensor(ds.p, DType::I32, {kTop, 1}), Tensor(dc.p, DType::I32, {1}), kHeads, geometry(true),
                       1.0F / 16.0F, cache.view(), arena, out, exec);
    t::cuda_check(cudaStreamEndCapture(stream, &graph), "end capture");
    t::cuda_check(cudaGraphInstantiate(&exec_graph, graph, 0), "instantiate");
    for (const int p : {5, 2050, 2051, 39999, 17003}) {
        const int n = (p + 1) / kR, count = std::min(n, kTop);
        std::vector<int> all(n);
        std::iota(all.begin(), all.end(), 0);
        std::shuffle(all.begin(), all.end(), rng);
        std::sort(all.begin(), all.begin() + count);
        std::vector<std::int32_t> ids(kTop, -1);
        std::copy_n(all.begin(), count, ids.begin());
        const std::vector<std::int32_t> positions = {p}, counts = {count};
        t::cuda_check(cudaMemcpyAsync(dp.p, positions.data(), 4, cudaMemcpyHostToDevice, stream), "p");
        t::cuda_check(cudaMemcpyAsync(ds.p, ids.data(), 4 * kTop, cudaMemcpyHostToDevice, stream), "ids");
        t::cuda_check(cudaMemcpyAsync(dc.p, counts.data(), 4, cudaMemcpyHostToDevice, stream), "count");
        t::cuda_check(cudaMemsetAsync(arena.base(), 0x77, arena.capacity(), stream), "garbage");
        t::cuda_check(cudaGraphLaunch(exec_graph, stream), "graph launch");
        t::cuda_synchronize(stream);
        const auto raw = download<std::uint16_t>(dout, static_cast<std::size_t>(kD) * kHq);
        std::vector<double> got(raw.size());
        for (std::size_t i = 0; i < raw.size(); ++i) { got[i] = from_bf16(raw[i]); }
        const auto ref = oracle_all(cache, q, positions, ids, counts, 1.0 / 16.0);
        g_failures += t::verify_reduction("qsa_attention graph replay p=" + std::to_string(p), got, ref,
                                          kAttentionCriterion);
    }
    t::cuda_check(cudaGraphExecDestroy(exec_graph), "destroy exec");
    t::cuda_check(cudaGraphDestroy(graph), "destroy graph");
    t::cuda_check(cudaStreamDestroy(stream), "destroy stream");
}

// Contract violations the wrapper must reject.
void test_rejections() {
    const auto exec = execution();
    auto bad        = geometry(true);
    bad.block_tokens = 8;
    bool threw      = false;
    try {
        (void)ops::qsa_select_blocks_workspace_capacity_bytes(bad, {4096}, 1, 1, exec);
    } catch (const std::invalid_argument&) { threw = true; }
    expect(threw, "an unregistered indexer geometry is rejected");
    threw = false;
    try {
        (void)ops::qsa_attention_workspace_capacity_bytes(kHeads, geometry(true), KvCacheStorage::BFloat16, 1, 1, exec);
    } catch (const std::invalid_argument&) { threw = true; }
    expect(threw, "a cache storage other than Int8Group64 is rejected");
    threw = false;
    try {
        (void)ops::qsa_attention_workspace_capacity_bytes({256, 24, 4}, geometry(true), KvCacheStorage::Int8Group64, 1,
                                                          1, exec);
    } catch (const std::invalid_argument&) { threw = true; }
    expect(threw, "an unregistered head geometry is rejected");
    threw = false;
    try {
        (void)ops::qsa_select_blocks_workspace_capacity_bytes(geometry(true), {262145}, 1, 1, exec);
    } catch (const std::invalid_argument&) { threw = true; }
    expect(threw, "an envelope beyond 262144 keys is rejected");

    // rope_positions must be exactly [T, 3] (axis-major); a same-numel [T, 1, 3] is interleaved.
    {
        const std::vector<std::int32_t> rope(6, 0);
        const std::vector<std::uint16_t> q(static_cast<std::size_t>(kDi) * kIndexHeads * 2, 0);
        const std::vector<std::uint16_t> weight(kDi, 0);
        DeviceBuffer dr = upload(rope), dq = upload(q), dw = upload(weight);
        Tensor tq(dq.p, DType::BF16, {kDi, kIndexHeads, 2});
        threw = false;
        try {
            ops::qsa_index_query(Tensor(dr.p, DType::I32, {2, 1, 3}), Tensor(dw.p, DType::BF16, {kDi}),
                                 geometry(true), tq, nullptr);
        } catch (const std::invalid_argument&) { threw = true; }
        expect(threw, "rope_positions shaped [T, 1, 3] is rejected");
    }
}

int run(bool small) {
    if (t::cuda_unavailable()) {
        std::printf("SKIP: no usable CUDA device\n");
        return 77;
    }
    test_rejections();

    // Index query: plain and unit-offset gains, text and distinct-axis positions up to 2^18.
    test_index_query(1, true, 8191, true, 1);
    test_index_query(7, false, 262143, false, 2);
    if (!small) {
        test_index_query(300, true, 262143, false, 3);
        test_index_query(2048, false, 131071, true, 4);
    }

    // Pooled keys: aligned and unaligned starts, decode steps, blocks straddling calls.
    test_pool({"aligned prefill then decode", 0, {64, 1, 1, 1, 1, 3, 2, 5}, true}, 10);
    test_pool({"unaligned start, odd widths", 4002, {1, 2, 3, 1, 7, 64, 1}, false}, 11);
    if (!small) { test_pool({"one wide call", 37, {2048 + 5}, true}, 12); }

    // Selection.
    std::vector<SelectCase> select_cases = {
        {"budget n=511", {2043}},
        {"budget n=512", {2047}},
        {"budget n=512 tail 3", {2050}},
        {"budget n=513", {2051}},
        {"budget n=514", {2055}},
        {"decode 2K blocks (8K keys)", {8191}},
        {"ties period 97", {16000, 16100}, Keys::Ties, 97},
        {"ties period 3", {4000}, Keys::Ties, 3},
        {"zero scores at the threshold", {2078, 2079}, Keys::ZeroHeavy, 470},
        {"zero scores, all blocks", {40000}, Keys::ZeroHeavy, 0},
        {"concentrated", {24000}, Keys::Concentrated},
        {"near ties", {20000}, Keys::NearTies},
    };
    if (!small) {
        std::vector<std::int32_t> straddle(300), wide(129), deep(64);
        std::iota(straddle.begin(), straddle.end(), 1900);
        std::iota(wide.begin(), wide.end(), 20000);
        std::iota(deep.begin(), deep.end(), 131000);
        select_cases.push_back({"decode 8K blocks (32K keys)", {32767}});
        select_cases.push_back({"decode 32K blocks (128K keys)", {131071}});
        select_cases.push_back({"decode 65K blocks (262K keys)", {262143}});
        select_cases.push_back({"decode small context, 262K envelope", {3000}, Keys::Random, 0, 262144});
        select_cases.push_back({"prefill 300 across the budget", straddle});
        select_cases.push_back({"prefill 129 columns (two groups)", wide});
        select_cases.push_back({"prefill 64 at 32K blocks", deep});
        select_cases.push_back({"ties period 701, 40K blocks", {160000}, Keys::Ties, 701});
        select_cases.push_back({"concentrated 30K blocks", {120000}, Keys::Concentrated});
    }
    std::uint32_t seed = 100;
    for (const auto& c : select_cases) { test_select(c, seed++); }
    test_select_graph();

    // Attention on explicit selections.
    std::vector<AttentionCase> attention_cases = {
        {"decode p=0 dense", 0, 1, Selection::Dense},
        {"decode p=3 dense", 3, 1, Selection::Dense},
        {"decode p=2050 dense (2051 keys)", 2050, 1, Selection::Dense},
        {"decode p=2051 selected", 2051, 1},
        {"decode p=32767 selected", 32767, 1},
        {"decode p=131071 selected", 131071, 1},
        {"decode p=9000 short counts", 9000, 1, Selection::Short},
        {"prefill 64 from 0 dense (= causal)", 0, 64, Selection::Dense},
    };
    if (!small) {
        attention_cases.push_back({"split edge 40 columns", 6000, 40});
        attention_cases.push_back({"unsplit 41 columns", 6000, 41});
        attention_cases.push_back({"prefill 96 straddling the budget", 2000, 96});
        attention_cases.push_back({"prefill 160 at 128K", 130900, 160});
        attention_cases.push_back({"prefill 48 short counts", 30000, 48, Selection::Short});
    }
    for (const auto& c : attention_cases) { test_attention(c, seed++); }
    test_end_to_end("decode p=5000", 5000, 1, seed++);
    test_end_to_end("prefill 24 straddling the budget", 2040, 24, seed++);
    if (!small) {
        test_end_to_end("prefill 64 at 20K", 20000, 64, seed++);
        test_attention_graph();
    }

    if (g_failures != 0) {
        std::fprintf(stderr, "qsa: %d check(s) failed\n", g_failures);
        return 1;
    }
    std::printf("qsa: PASS\n");
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    try {
        const bool small = argc == 2 && std::string_view(argv[1]) == "--small";
        if (argc > 2 || (argc == 2 && !small)) {
            std::fprintf(stderr, "usage: ninfer_qsa_test [--small]\n");
            return 2;
        }
        return run(small);
    } catch (const std::exception& error) {
        std::fprintf(stderr, "qsa: threw: %s\n", error.what());
        return 1;
    }
}
