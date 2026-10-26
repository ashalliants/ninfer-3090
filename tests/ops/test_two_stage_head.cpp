// two_stage_head (--lm-head-two-stage) on the 27B's vocabulary shape, [248320, 5120], against the
// dense Q8 head (ops::linear) and the FP64 oracle:
//   - dense_bits: at T = 1..16 verify columns by B = 1..8 rows, every row of every column holds
//     either the dense head's value bit for bit or an upper bound on it that is below the column's
//     64th best dense value; so the top 64 rows of every column are the dense head's, bit for bit;
//   - argmax and sampling: ops::argmax and ops::sample (temperature 0.7, top-k 20, top-p 0.9)
//     choose the same tokens from both outputs;
//   - fp64_sets: where T and B are at their ends, the top-20 and top-64 row sets agree with FP64
//     except across a gap within the Q8 A16 Linear criterion;
//   - copy_bits: sampled rows that were not rescored hold the host interval's upper bound,
//     proxy_interval, rounded up to BF16, bit for bit;
//   - stats: no round of these falls back or is excluded, and the selected count is printed;
//   - exclusion: a round with a penalty or grammar row equals the dense head in every row;
//   - fallback: a copy whose scales are 4x too large fails the column check, and the whole head runs;
//   - the workspace stays inside two_stage_head_workspace_capacity_bytes and hidden is preserved.
// The head has 300 "hot" rows whose codes carry a positive offset, so the columns have a clear top
// above a Gaussian bulk as a language model's logits do.

#include "ninfer/ops/two_stage_head.h"

#include "artifact/companion.h"
#include "core/device.h"
#include "core/weight_view.h"
#include "ninfer/ops/argmax.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/sampling.h"
#include "ops/op_tester.h"
#include "ops/two_stage_head/proxy_codec.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

namespace ph = ninfer::ops::proxy_head;
namespace hc = ninfer::ops::hc_int8;

constexpr std::int32_t kRows    = 248320;
constexpr std::int32_t kColumns = 5120;
constexpr std::int32_t kValid   = 248077; // the Qwen3.5 public token count
constexpr int kHotRows          = 300;

constexpr ReductionCriterion kQ8A16Criterion{1.0 / 256.0, 1.0 / 256.0, 2.0 / 256.0};

std::uint64_t mix(std::uint64_t x) {
    x ^= x >> 33;
    x *= 0xff51afd7ed558ccdULL;
    x ^= x >> 33;
    x *= 0xc4ceb9fe1a85ec53ULL;
    x ^= x >> 33;
    return x;
}

float bf16_value(std::uint16_t bits) { return bf16_to_f32(bits); }

// BF16 rounded toward +inf (finite inputs), as the kernel stores upper bounds.
std::uint16_t bf16_round_up(float x) {
    std::uint32_t u;
    std::memcpy(&u, &x, sizeof u);
    std::uint32_t high = u >> 16;
    if ((u & 0xFFFFU) != 0 && (u & 0x80000000U) == 0) { ++high; }
    return static_cast<std::uint16_t>(high);
}

struct Head {
    std::vector<std::byte> payload; // the stored row-split Q8 object
    WeightGeometry geometry;
    std::vector<std::byte> proxy;

    [[nodiscard]] std::int8_t code(std::int64_t row, std::int64_t column) const {
        return static_cast<std::int8_t>(std::to_integer<unsigned>(payload[static_cast<std::size_t>(row * kColumns + column)]));
    }
    [[nodiscard]] float scale(std::int64_t row, std::int64_t group) const {
        const std::size_t at = geometry.scale_offset + static_cast<std::size_t>((row * (kColumns / 32) + group) * 2);
        return hc::f16_to_f32(static_cast<std::uint16_t>(std::to_integer<unsigned>(payload[at]) |
                                                         (std::to_integer<unsigned>(payload[at + 1]) << 8U)));
    }
    [[nodiscard]] double dot(std::int64_t row, const std::vector<std::uint16_t>& hidden, int t) const {
        double sum = 0.0;
        for (std::int64_t c = 0; c < kColumns; ++c) {
            sum += static_cast<double>(code(row, c)) * scale(row, c / 32) *
                   bf16_value(hidden[static_cast<std::size_t>(t) * kColumns + static_cast<std::size_t>(c)]);
        }
        return sum;
    }
};

Head make_head() {
    Head head;
    const std::vector<std::uint64_t> shape{kRows, kColumns};
    head.geometry = weight_geometry(QType::Q8_G32_FP16, QuantLayout::RowSplit, shape);
    head.payload.assign(head.geometry.bytes, std::byte{0});
    std::vector<int> hot_offset(kRows, 0);
    for (int i = 0; i < kHotRows; ++i) { hot_offset[mix(0xA0000ULL + i) % kRows] = 1 + static_cast<int>(mix(0xB0000ULL + i) % 12); }
    for (std::int64_t row = 0; row < kRows; ++row) {
        for (std::int64_t c = 0; c < kColumns; ++c) {
            const std::uint64_t h = mix(static_cast<std::uint64_t>(row) * kColumns + c);
            // A sum of four bytes: roughly Gaussian, standard deviation ~37 codes.
            int v = static_cast<int>(h & 0xFF) + static_cast<int>((h >> 8) & 0xFF) + static_cast<int>((h >> 16) & 0xFF) +
                    static_cast<int>((h >> 24) & 0xFF) - 510;
            v     = v / 4 + hot_offset[static_cast<std::size_t>(row)];
            head.payload[static_cast<std::size_t>(row * kColumns + c)] =
                static_cast<std::byte>(static_cast<std::int8_t>(std::clamp(v, -127, 127)));
        }
        for (std::int64_t g = 0; g < kColumns / 32; ++g) {
            const float s = 0.0011F * (0.8F + 0.4F * static_cast<float>(mix(0xC0000000ULL + row * 160 + g) % 1000) / 1000.0F);
            const std::uint16_t bits = hc::f32_to_f16_rn(s);
            const std::size_t at     = head.geometry.scale_offset + static_cast<std::size_t>((row * (kColumns / 32) + g) * 2);
            head.payload[at]         = static_cast<std::byte>(bits & 0xFFU);
            head.payload[at + 1]     = static_cast<std::byte>(bits >> 8U);
        }
    }
    head.proxy.resize(ph::proxy_bytes(kRows, kColumns));
    artifact::build_companion(artifact::Companion::TwoBitHeadProxy, shape, head.payload, head.proxy);
    return head;
}

Weight device_weight(const Head& head, void* device) {
    Weight w{};
    w.qtype           = QType::Q8_G32_FP16;
    w.layout          = QuantLayout::RowSplit;
    w.scale_dtype     = DType::FP16;
    w.payload         = device;
    w.payload_bytes   = head.payload.size();
    w.qdata           = device;
    w.qhigh           = nullptr;
    w.scales          = static_cast<std::byte*>(device) + head.geometry.scale_offset;
    w.group_size      = 32;
    w.group           = 32;
    w.ndim            = 2;
    w.shape[0]        = kRows;
    w.shape[1]        = kColumns;
    w.shape[2]        = 1;
    w.shape[3]        = 1;
    w.padded_shape[0] = kRows;
    w.padded_shape[1] = kColumns;
    w.padded_shape[2] = 1;
    w.padded_shape[3] = 1;
    w.n               = kRows;
    w.k               = kColumns;
    return w;
}

std::vector<std::uint16_t> make_hidden(int columns, std::uint64_t seed) {
    std::vector<std::uint16_t> hidden(static_cast<std::size_t>(columns) * kColumns);
    for (std::size_t i = 0; i < hidden.size(); ++i) {
        const std::uint64_t h = mix(seed * 0x9E3779B97F4A7C15ULL + i);
        // Mean 0.3, so the hot rows' offset raises their scores; spread ~1.
        const float v = 0.3F + static_cast<float>(static_cast<double>(h % 20001U) / 10000.0 - 1.0) * 1.7F;
        hidden[i]     = f32_to_bf16(v);
    }
    return hidden;
}

// The top `count` rows of column t by BF16 value, lower id first on a tie.
std::vector<std::int32_t> top_rows(const std::vector<std::uint16_t>& logits, int t, int count) {
    std::vector<std::int32_t> order(kValid);
    std::iota(order.begin(), order.end(), 0);
    const auto value = [&](std::int32_t r) { return bf16_value(logits[static_cast<std::size_t>(t) * kRows + static_cast<std::size_t>(r)]); };
    std::partial_sort(order.begin(), order.begin() + count, order.end(), [&](std::int32_t a, std::int32_t b) {
        return value(a) > value(b) || (value(a) == value(b) && a < b);
    });
    order.resize(static_cast<std::size_t>(count));
    return order;
}

struct Fixture {
    Head head;
    DeviceBuffer weights;
    DeviceBuffer copy;
    Weight weight{};
    ops::ProxyHead proxy{};
    DeviceArena workspace;

    Fixture()
        : head(make_head()), weights(head.payload.size()), copy(head.proxy.size()),
          workspace(ops::two_stage_head_workspace_capacity_bytes(kRows, kColumns, 128) +
                    ops::linear_workspace_capacity_bytes(QType::Q8_G32_FP16, kRows, kColumns, ops::LinearPolicy::A16Only, 1, 128) +
                    ops::sampling_workspace_capacity_bytes(kValid, 1, 8) + (1U << 20)) {
        weights.copy_from_host(head.payload.data(), head.payload.size());
        copy.copy_from_host(head.proxy.data(), head.proxy.size());
        weight = device_weight(head, weights.p);
        auto* base = static_cast<std::byte*>(copy.p);
        proxy      = {base, base + ph::proxy_code_bytes(kRows, kColumns), base + ph::proxy_residual_offset(kRows, kColumns),
                      kRows, kColumns};
    }
};

struct Run {
    std::vector<std::uint16_t> dense;
    std::vector<std::uint16_t> staged;
    std::int32_t stats[3] = {0, 0, 0};
    std::vector<std::int32_t> dense_argmax, staged_argmax, dense_sampled, staged_sampled;
};

Run run(Fixture& f, const std::vector<std::uint16_t>& hidden, int columns, const std::vector<ops::SamplingConfig>& configs,
        int columns_per_config, int& failures, const std::string& label, bool sample) {
    DeviceBuffer x(hidden.size() * 2);
    x.copy_from_host(hidden.data(), hidden.size() * 2);
    DeviceBuffer dense(static_cast<std::size_t>(kRows) * columns * 2);
    DeviceBuffer staged(static_cast<std::size_t>(kRows) * columns * 2);
    DeviceBuffer stats(3 * sizeof(std::int32_t));
    DeviceBuffer device_configs(std::max<std::size_t>(configs.size(), 1) * sizeof(ops::SamplingConfig));
    if (!configs.empty()) { device_configs.copy_from_host(configs.data(), configs.size() * sizeof(ops::SamplingConfig)); }
    Tensor xt(x.p, DType::BF16, {kColumns, columns});
    Tensor dense_t(dense.p, DType::BF16, {kRows, columns});
    Tensor staged_t(staged.p, DType::BF16, {kRows, columns});
    Tensor stats_t(stats.p, DType::I32, {3});
    ops::linear(xt, f.weight, dense_t, ops::LinearPolicy::A16Only, f.workspace, nullptr);
    f.workspace.reset();
    f.workspace.reset_peak();
    ops::two_stage_head(xt, f.weight, f.proxy, kValid, staged_t,
                        {configs.empty() ? nullptr : static_cast<const ops::SamplingConfig*>(device_configs.p), columns_per_config},
                        f.workspace, nullptr, {}, &stats_t);
    cuda_synchronize();
    if (f.workspace.peak_used() > ops::two_stage_head_workspace_capacity_bytes(kRows, kColumns, columns)) {
        std::cerr << label << ": workspace above its stated capacity\n";
        ++failures;
    }
    f.workspace.reset();
    Run out;
    if (sample) {
        // ops::argmax and ops::sample on both outputs, one sampled row per column.
        std::vector<ops::SamplingConfig> sampling(static_cast<std::size_t>(columns));
        for (int c = 0; c < columns; ++c) {
            sampling[static_cast<std::size_t>(c)].temperature = 0.7F;
            sampling[static_cast<std::size_t>(c)].top_k       = 20;
            sampling[static_cast<std::size_t>(c)].top_p       = 0.9F;
            sampling[static_cast<std::size_t>(c)].seed        = 1234U + static_cast<unsigned>(c);
        }
        DeviceBuffer sampling_d(sampling.size() * sizeof(ops::SamplingConfig));
        sampling_d.copy_from_host(sampling.data(), sampling.size() * sizeof(ops::SamplingConfig));
        std::vector<std::int32_t> positions(static_cast<std::size_t>(columns), 77);
        DeviceBuffer positions_d(positions.size() * 4);
        positions_d.copy_from_host(positions.data(), positions.size() * 4);
        Tensor positions_t(positions_d.p, DType::I32, {columns});
        for (int side = 0; side < 2; ++side) {
            Tensor& logits = side == 0 ? dense_t : staged_t;
            DeviceBuffer ids(static_cast<std::size_t>(columns) * 4);
            Tensor ids_t(ids.p, DType::I32, {columns});
            std::vector<std::int32_t> argmax(static_cast<std::size_t>(columns)), sampled(static_cast<std::size_t>(columns));
            ops::argmax(logits, ids_t, kValid, nullptr);
            cuda_synchronize();
            ids.copy_to_host(argmax.data(), argmax.size() * 4);
            ops::sample(logits, ids_t, kValid, static_cast<const ops::SamplingConfig*>(sampling_d.p), positions_t,
                        ops::kSamplePurposeDecode, f.workspace, nullptr);
            cuda_synchronize();
            f.workspace.reset();
            ids.copy_to_host(sampled.data(), sampled.size() * 4);
            (side == 0 ? out.dense_argmax : out.staged_argmax)   = argmax;
            (side == 0 ? out.dense_sampled : out.staged_sampled) = sampled;
        }
        if (out.dense_argmax != out.staged_argmax || out.dense_sampled != out.staged_sampled) {
            std::cerr << label << ": argmax or sampled tokens differ from the dense head's\n";
            ++failures;
        }
    }
    std::vector<std::uint16_t> after(hidden.size());
    x.copy_to_host(after.data(), after.size() * 2);
    if (after != hidden) {
        std::cerr << label << ": hidden was modified\n";
        ++failures;
    }
    out.dense.resize(static_cast<std::size_t>(kRows) * columns);
    out.staged.resize(out.dense.size());
    dense.copy_to_host(out.dense.data(), out.dense.size() * 2);
    staged.copy_to_host(out.staged.data(), out.staged.size() * 2);
    stats.copy_to_host(out.stats, sizeof(out.stats));
    return out;
}

// Every row equals the dense head's bits, or is an upper bound on it below the 64th best.
int check_dense_bits(const Run& r, int columns, bool whole, const std::string& label) {
    int failures = 0;
    for (int t = 0; t < columns; ++t) {
        const auto top     = top_rows(r.dense, t, 64);
        const float kth    = bf16_value(r.dense[static_cast<std::size_t>(t) * kRows + static_cast<std::size_t>(top.back())]);
        std::size_t exact  = 0;
        for (std::int32_t row = 0; row < kRows; ++row) {
            const std::size_t at = static_cast<std::size_t>(t) * kRows + static_cast<std::size_t>(row);
            if (r.staged[at] == r.dense[at]) {
                ++exact;
                continue;
            }
            const float staged = bf16_value(r.staged[at]);
            const float dense  = bf16_value(r.dense[at]);
            if (whole || (row < kValid && !(staged >= dense && staged < kth))) {
                std::cerr << label << " column " << t << " row " << row << ": stored " << staged << " vs dense " << dense
                          << " (64th best " << kth << ")\n";
                ++failures;
                break;
            }
        }
        if (top_rows(r.staged, t, 64) != top) {
            std::cerr << label << " column " << t << ": the top 64 differ from the dense head's\n";
            ++failures;
        }
        (void)exact;
    }
    return failures;
}

// The top-20 and top-64 sets against FP64 (a swap only across a gap within the criterion's gross
// bound), and sampled rows that were not rescored hold the host interval's bits.
int check_fp64_and_copy(const Fixture& f, const Run& r, const std::vector<std::uint16_t>& hidden, int columns,
                        const std::string& label) {
    int failures = 0;
    for (int t = 0; t < columns; ++t) {
        std::vector<double> exact(kValid);
        double max_abs = 0.0;
        for (std::int32_t row = 0; row < kValid; ++row) {
            exact[static_cast<std::size_t>(row)] = f.head.dot(row, hidden, t);
            max_abs                              = std::max(max_abs, std::fabs(exact[static_cast<std::size_t>(row)]));
        }
        const double tie = kQ8A16Criterion.gross_relative_to_max_reference * max_abs + kQ8A16Criterion.gross_absolute;
        for (const int k : {20, 64}) {
            const auto top = top_rows(r.staged, t, k);
            std::vector<double> sorted(exact);
            std::nth_element(sorted.begin(), sorted.begin() + (k - 1), sorted.end(), std::greater<>());
            const double kth = sorted[static_cast<std::size_t>(k - 1)];
            std::vector<bool> in(kValid, false);
            for (const auto row : top) {
                in[static_cast<std::size_t>(row)] = true;
                if (exact[static_cast<std::size_t>(row)] < kth - 2.0 * tie) {
                    std::cerr << label << " column " << t << ": top-" << k << " row " << row << " is well below FP64's\n";
                    ++failures;
                }
            }
            for (std::int32_t row = 0; row < kValid; ++row) {
                if (!in[static_cast<std::size_t>(row)] && exact[static_cast<std::size_t>(row)] > kth + 2.0 * tie) {
                    std::cerr << label << " column " << t << ": row " << row << " is FP64 top-" << k << " but missing\n";
                    ++failures;
                    break;
                }
            }
        }
        std::vector<double> got, want;
        for (const auto row : top_rows(r.staged, t, 64)) {
            got.push_back(bf16_value(r.staged[static_cast<std::size_t>(t) * kRows + static_cast<std::size_t>(row)]));
            want.push_back(exact[static_cast<std::size_t>(row)]);
        }
        failures += verify_reduction(label + " top-64 column " + std::to_string(t), got, want, kQ8A16Criterion);
        std::vector<std::int8_t> q(kColumns);
        std::vector<ph::ActivationGroup> groups(kColumns / 32);
        std::vector<float> x(32);
        for (int g = 0; g < kColumns / 32; ++g) {
            for (int i = 0; i < 32; ++i) {
                x[static_cast<std::size_t>(i)] = bf16_value(hidden[static_cast<std::size_t>(t) * kColumns + static_cast<std::size_t>(g * 32 + i)]);
            }
            ph::quantize_activation_group(x.data(), q.data() + g * 32, &groups[static_cast<std::size_t>(g)]);
        }
        for (int s = 0; s < 64; ++s) {
            const auto row             = static_cast<std::int64_t>(mix(0xD0000ULL + static_cast<std::uint64_t>(t) * 64 + s) % kValid);
            const std::size_t at       = static_cast<std::size_t>(t) * kRows + static_cast<std::size_t>(row);
            const std::uint16_t upper  = bf16_round_up(ph::proxy_interval(f.head.proxy, kRows, row, kColumns, q.data(), groups.data()).upper);
            if (r.staged[at] != r.dense[at] && r.staged[at] != upper) {
                std::cerr << label << " column " << t << " row " << row << ": stored " << bf16_value(r.staged[at])
                          << " is neither dense nor the host upper bound " << bf16_value(upper) << '\n';
                ++failures;
                break;
            }
        }
    }
    return failures;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    int failures = 0;
    Fixture f;
    for (int width = 1; width <= 16; ++width) {
        for (int batch = 1; batch <= 8; ++batch) {
            const int columns = width * batch;
            const auto hidden = make_hidden(columns, static_cast<std::uint64_t>(width * 16 + batch));
            const std::vector<ops::SamplingConfig> configs(static_cast<std::size_t>(batch));
            const std::string label = "two_stage_head T=" + std::to_string(width) + " B=" + std::to_string(batch);
            const Run r = run(f, hidden, columns, configs, width, failures, label, width == 1);
            failures += check_dense_bits(r, columns, false, label);
            if (r.stats[1] != 0 || r.stats[2] != 0 || r.stats[0] > kValid) {
                std::cerr << label << ": excluded " << r.stats[1] << ", fallback " << r.stats[2] << ", selected " << r.stats[0] << '\n';
                ++failures;
            }
            std::cout << label << ": rows selected " << r.stats[0] << " of " << kValid << '\n';
            if ((width == 1 || width == 16) && (batch == 1 || batch == 8) && !(width == 16 && batch == 8)) {
                failures += check_fp64_and_copy(f, r, hidden, columns, label);
            }
        }
    }
    {
        // Three rows of two columns: row 0 has a grammar mask and row 1 penalties, so the round
        // runs the whole dense head.
        std::vector<ops::SamplingConfig> configs(3);
        static std::uint32_t mask_word = 0xFFFFFFFFU;
        configs[0].mask.words   = &mask_word; // only compared with null on the device
        configs[1].token_counts = reinterpret_cast<std::int32_t*>(&mask_word);
        const auto hidden       = make_hidden(6, 0xE1U);
        const Run r             = run(f, hidden, 6, configs, 2, failures, "exclusion", false);
        failures += check_dense_bits(r, 6, true, "exclusion");
        if (r.stats[1] != 1) {
            std::cerr << "exclusion: the round was not reported excluded\n";
            ++failures;
        }
    }
    {
        // A copy whose scales are 4x too large: its intervals no longer hold, the listed rows'
        // dense scores fall short of the threshold, the column check fails and the whole head runs.
        std::vector<std::byte> broken = f.head.proxy;
        auto* scales                  = broken.data() + ph::proxy_code_bytes(kRows, kColumns);
        for (std::size_t i = 0; i < ph::proxy_scale_bytes(kRows, kColumns); i += 2) {
            auto bits = static_cast<std::uint16_t>(std::to_integer<unsigned>(scales[i]) | (std::to_integer<unsigned>(scales[i + 1]) << 8U));
            if ((bits & 0x7C00U) != 0 && (bits & 0x7C00U) < 0x7400U) { bits = static_cast<std::uint16_t>(bits + 0x0800U); }
            scales[i]     = static_cast<std::byte>(bits & 0xFFU);
            scales[i + 1] = static_cast<std::byte>(bits >> 8U);
        }
        // ...and its residuals zero, so the intervals are narrow around the inflated scores.
        std::fill(broken.begin() + static_cast<std::ptrdiff_t>(ph::proxy_residual_offset(kRows, kColumns)), broken.end(),
                  std::byte{0});
        f.copy.copy_from_host(broken.data(), broken.size());
        const auto hidden = make_hidden(4, 0xF00DU);
        const Run r       = run(f, hidden, 4, {}, 1, failures, "fallback", false);
        failures += check_dense_bits(r, 4, true, "fallback");
        if (r.stats[2] != 1) {
            std::cerr << "fallback: the column check did not fall back\n";
            ++failures;
        }
        f.copy.copy_from_host(f.head.proxy.data(), f.head.proxy.size());
    }
    std::cout << (failures ? "FAIL" : "OK") << " two_stage_head\n";
    return failures ? 1 : 0;
}
