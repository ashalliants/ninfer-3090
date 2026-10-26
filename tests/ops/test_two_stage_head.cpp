// two_stage_head (--lm-head-two-stage) on the 27B's vocabulary shape, [248320, 5120], against the
// dense Q8 head (ops::linear) and the FP64 oracle:
//   - agreement: at T = 1..16 verify columns by B = 1..8 rows, argmax and the top 16 rows of every
//     column equal the dense head's (a swap is allowed only between rows whose FP64 scores are
//     within two BF16 steps, a tie for any exact route);
//   - exact_rows: where B = 1 or T = 1, the column's top 64 stored rows meet the Q8 A16 Linear
//     criterion against FP64, and 64 sampled rows per column hold either the copy's score bit for
//     bit (proxy_row_score rounded up to BF16) or an exactly rescored value;
//   - exclusion: a column whose config has penalties or a grammar mask is rescored in every row;
//   - fallback: a copy whose scale is 4x too large over-estimates the rows bounding the threshold,
//     the column check fails, and every row is rescored;
//   - the workspace stays inside two_stage_head_workspace_capacity_bytes and hidden is preserved.
// The head has 300 "hot" rows whose codes carry a positive offset, so the columns have a clear top
// above a Gaussian bulk as a language model's logits do.

#include "ninfer/ops/two_stage_head.h"

#include "artifact/companion.h"
#include "core/device.h"
#include "core/weight_view.h"
#include "ninfer/ops/linear.h"
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

// BF16 rounded toward +inf (finite inputs), as the kernel stores approximate scores.
std::uint16_t bf16_round_up(float x) {
    std::uint32_t u;
    std::memcpy(&u, &x, sizeof u);
    const std::uint32_t low = u & 0xFFFFU;
    std::uint32_t high      = u >> 16;
    if (low != 0 && (u & 0x80000000U) == 0) { ++high; }
    return static_cast<std::uint16_t>(high);
}

struct Head {
    std::vector<std::byte> payload; // the stored row-split Q8 object
    WeightGeometry geometry;
    std::vector<std::byte> proxy;

    [[nodiscard]] std::int8_t code(std::int64_t row, std::int64_t column) const {
        return static_cast<std::int8_t>(
            std::to_integer<unsigned>(payload[static_cast<std::size_t>(row * kColumns + column)]));
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
    for (int i = 0; i < kHotRows; ++i) {
        hot_offset[mix(0xA0000ULL + i) % kRows] = 1 + static_cast<int>(mix(0xB0000ULL + i) % 12);
    }
    for (std::int64_t row = 0; row < kRows; ++row) {
        for (std::int64_t c = 0; c < kColumns; ++c) {
            const std::uint64_t h = mix(static_cast<std::uint64_t>(row) * kColumns + c);
            // A sum of four bytes: roughly Gaussian, standard deviation ~37 codes.
            int v = static_cast<int>(h & 0xFF) + static_cast<int>((h >> 8) & 0xFF) +
                    static_cast<int>((h >> 16) & 0xFF) + static_cast<int>((h >> 24) & 0xFF) - 510;
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
    w.qtype            = QType::Q8_G32_FP16;
    w.layout           = QuantLayout::RowSplit;
    w.scale_dtype      = DType::FP16;
    w.payload          = device;
    w.payload_bytes    = head.payload.size();
    w.qdata            = device;
    w.qhigh            = nullptr;
    w.scales           = static_cast<std::byte*>(device) + head.geometry.scale_offset;
    w.group_size       = 32;
    w.group            = 32;
    w.ndim             = 2;
    w.shape[0]         = kRows;
    w.shape[1]         = kColumns;
    w.shape[2]         = 1;
    w.shape[3]         = 1;
    w.padded_shape[0]  = kRows;
    w.padded_shape[1]  = kColumns;
    w.padded_shape[2]  = 1;
    w.padded_shape[3]  = 1;
    w.n                = kRows;
    w.k                = kColumns;
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
    const auto value = [&](std::int32_t r) {
        return bf16_value(logits[static_cast<std::size_t>(t) * kRows + static_cast<std::size_t>(r)]);
    };
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
                    ops::linear_workspace_capacity_bytes(QType::Q8_G32_FP16, kRows, kColumns,
                                                         ops::LinearPolicy::A16Only, 1, 128) +
                    (1U << 20)) {
        weights.copy_from_host(head.payload.data(), head.payload.size());
        copy.copy_from_host(head.proxy.data(), head.proxy.size());
        weight = device_weight(head, weights.p);
        proxy  = {copy.p, static_cast<std::byte*>(copy.p) + ph::proxy_code_bytes(kRows, kColumns), kRows,
                  kColumns};
    }
};

struct Run {
    std::vector<std::uint16_t> dense;
    std::vector<std::uint16_t> staged;
};

Run run(Fixture& f, const std::vector<std::uint16_t>& hidden, int columns,
        const std::vector<ops::SamplingConfig>& configs, int columns_per_config,
        const ops::TwoStageHeadRule& rule, int& failures, const std::string& label) {
    DeviceBuffer x(hidden.size() * 2);
    x.copy_from_host(hidden.data(), hidden.size() * 2);
    DeviceBuffer dense(static_cast<std::size_t>(kRows) * columns * 2);
    DeviceBuffer staged(static_cast<std::size_t>(kRows) * columns * 2);
    DeviceBuffer device_configs(std::max<std::size_t>(configs.size(), 1) * sizeof(ops::SamplingConfig));
    if (!configs.empty()) {
        device_configs.copy_from_host(configs.data(), configs.size() * sizeof(ops::SamplingConfig));
    }
    Tensor xt(x.p, DType::BF16, {kColumns, columns});
    Tensor dense_t(dense.p, DType::BF16, {kRows, columns});
    Tensor staged_t(staged.p, DType::BF16, {kRows, columns});
    ops::linear(xt, f.weight, dense_t, ops::LinearPolicy::A16Only, f.workspace, nullptr);
    f.workspace.reset();
    f.workspace.reset_peak();
    ops::two_stage_head(xt, f.weight, f.proxy, kValid, staged_t,
                        {configs.empty() ? nullptr : static_cast<const ops::SamplingConfig*>(device_configs.p),
                         columns_per_config},
                        f.workspace, nullptr, rule);
    cuda_synchronize();
    if (f.workspace.peak_used() > ops::two_stage_head_workspace_capacity_bytes(kRows, kColumns, columns)) {
        std::cerr << label << ": workspace above its stated capacity\n";
        ++failures;
    }
    f.workspace.reset();
    std::vector<std::uint16_t> after(hidden.size());
    x.copy_to_host(after.data(), after.size() * 2);
    if (after != hidden) {
        std::cerr << label << ": hidden was modified\n";
        ++failures;
    }
    Run out;
    out.dense.resize(static_cast<std::size_t>(kRows) * columns);
    out.staged.resize(out.dense.size());
    dense.copy_to_host(out.dense.data(), out.dense.size() * 2);
    staged.copy_to_host(out.staged.data(), out.staged.size() * 2);
    return out;
}

// Argmax and top 16 against the dense head.
int check_agreement(const Fixture& f, const Run& r, const std::vector<std::uint16_t>& hidden, int columns,
                    const std::string& label) {
    int failures = 0;
    for (int t = 0; t < columns; ++t) {
        const auto want = top_rows(r.dense, t, 16);
        const auto got  = top_rows(r.staged, t, 16);
        for (int i = 0; i < 16; ++i) {
            if (want[static_cast<std::size_t>(i)] == got[static_cast<std::size_t>(i)]) { continue; }
            const double a   = f.head.dot(want[static_cast<std::size_t>(i)], hidden, t);
            const double b   = f.head.dot(got[static_cast<std::size_t>(i)], hidden, t);
            const double tie = 2.0 * std::ldexp(1.0, std::ilogb(std::max(std::fabs(a), 1.0)) - 7);
            if (std::fabs(a - b) > tie) {
                std::cerr << label << " column " << t << " rank " << i << ": dense row " << want[static_cast<std::size_t>(i)]
                          << " (" << a << ") vs two-stage row " << got[static_cast<std::size_t>(i)] << " (" << b << ")\n";
                ++failures;
                break;
            }
        }
    }
    return failures;
}

// The top 64 stored rows meet the Q8 A16 criterion, and sampled rows hold the copy's bits or an
// exact value. With `all_exact`, every sampled row must be exact.
int check_rows(const Fixture& f, const Run& r, const std::vector<std::uint16_t>& hidden, int columns,
               bool all_exact, const std::string& label) {
    int failures = 0;
    for (int t = 0; t < columns; ++t) {
        std::vector<double> got;
        std::vector<double> want;
        for (const std::int32_t row : top_rows(r.staged, t, 64)) {
            got.push_back(bf16_value(r.staged[static_cast<std::size_t>(t) * kRows + static_cast<std::size_t>(row)]));
            want.push_back(f.head.dot(row, hidden, t));
        }
        failures += verify_reduction(label + " top-64 column " + std::to_string(t), got, want, kQ8A16Criterion);
        std::vector<std::int8_t> q(kColumns);
        std::vector<float> d(kColumns / 32);
        std::vector<float> x(32);
        for (int g = 0; g < kColumns / 32; ++g) {
            for (int i = 0; i < 32; ++i) {
                x[static_cast<std::size_t>(i)] =
                    bf16_value(hidden[static_cast<std::size_t>(t) * kColumns + static_cast<std::size_t>(g * 32 + i)]);
            }
            ph::quantize_activation_group(x.data(), q.data() + g * 32, &d[static_cast<std::size_t>(g)]);
        }
        int approximate = 0;
        for (int s = 0; s < 64; ++s) {
            const auto row    = static_cast<std::int64_t>(mix(0xD0000ULL + static_cast<std::uint64_t>(t) * 64 + s) % kValid);
            const std::uint16_t stored = r.staged[static_cast<std::size_t>(t) * kRows + static_cast<std::size_t>(row)];
            const std::uint16_t copy   = bf16_round_up(ph::proxy_row_score(f.head.proxy, kRows, row, kColumns, q.data(), d.data()));
            const double exact         = f.head.dot(row, hidden, t);
            const bool is_exact        = std::fabs(bf16_value(stored) - exact) <= 2.0 / 256.0 * std::max(1.0, std::fabs(exact));
            approximate += stored == copy && !is_exact ? 1 : 0;
            if (!(is_exact || (!all_exact && stored == copy))) {
                std::cerr << label << " column " << t << " row " << row << ": stored " << bf16_value(stored)
                          << " is neither the copy's score " << bf16_value(copy) << " nor exact " << exact << '\n';
                ++failures;
                break;
            }
        }
        if (all_exact && approximate != 0) { ++failures; }
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
    const ops::TwoStageHeadRule rule{};
    for (int width = 1; width <= 16; ++width) {
        for (int batch = 1; batch <= 8; ++batch) {
            const int columns = width * batch;
            const auto hidden = make_hidden(columns, static_cast<std::uint64_t>(width * 16 + batch));
            const std::vector<ops::SamplingConfig> configs(static_cast<std::size_t>(batch));
            const std::string label = "two_stage_head T=" + std::to_string(width) + " B=" + std::to_string(batch);
            const Run r = run(f, hidden, columns, configs, width, rule, failures, label);
            failures += check_agreement(f, r, hidden, columns, label);
            if (width == 1 || batch == 1) { failures += check_rows(f, r, hidden, columns, false, label); }
        }
    }
    {
        // Three rows of two columns: row 0 has a grammar mask and row 1 penalties, so their four
        // columns are exact in every row; row 2 takes the two-stage head.
        std::vector<ops::SamplingConfig> configs(3);
        static std::uint32_t mask_word = 0xFFFFFFFFU;
        configs[0].mask.words   = &mask_word; // only compared with null on the device
        configs[1].token_counts = reinterpret_cast<std::int32_t*>(&mask_word);
        const auto hidden       = make_hidden(6, 0xE1U);
        const Run r             = run(f, hidden, 6, configs, 2, rule, failures, "exclusion");
        failures += check_agreement(f, r, hidden, 6, "exclusion");
        std::vector<std::uint16_t> excluded(r.staged.begin(), r.staged.begin() + 4 * static_cast<std::size_t>(kRows));
        failures += check_rows(f, Run{r.dense, excluded}, hidden, 4, true, "exclusion");
    }
    {
        // A copy whose scales are 4x too large: the threshold's bounding rows are over-estimated,
        // the column check fails and every row is rescored.
        std::vector<std::byte> broken = f.head.proxy;
        auto* scales = broken.data() + ph::proxy_code_bytes(kRows, kColumns);
        for (std::size_t i = 0; i < ph::proxy_scale_bytes(kRows, kColumns); i += 2) {
            std::uint16_t bits = static_cast<std::uint16_t>(std::to_integer<unsigned>(scales[i]) |
                                                            (std::to_integer<unsigned>(scales[i + 1]) << 8U));
            if ((bits & 0x7C00U) != 0 && (bits & 0x7C00U) < 0x7400U) { bits = static_cast<std::uint16_t>(bits + 0x0800U); }
            scales[i]     = static_cast<std::byte>(bits & 0xFFU);
            scales[i + 1] = static_cast<std::byte>(bits >> 8U);
        }
        f.copy.copy_from_host(broken.data(), broken.size());
        const auto hidden = make_hidden(4, 0xF00DU);
        const Run r       = run(f, hidden, 4, {}, 1, rule, failures, "fallback");
        failures += check_agreement(f, r, hidden, 4, "fallback");
        failures += check_rows(f, r, hidden, 4, true, "fallback");
        f.copy.copy_from_host(f.head.proxy.data(), f.head.proxy.size());
    }
    std::cout << (failures ? "FAIL" : "OK") << " two_stage_head\n";
    return failures ? 1 : 0;
}
