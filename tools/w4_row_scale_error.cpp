// Would handing prefill to cuBLAS cost weight quality?
//
// cuBLAS reduces over the whole of K, so it cannot see a scale per 64 columns: a weight given to it
// must carry one scale per row. That sounds like a clear loss against this fork's group-64 scales,
// and it is the reason the dequantise-to-int8 route was set aside. But the materialised form also
// has *four more bits per code* -- int8 rather than int4 -- and 4 bits is 16x of headroom. If the
// group scales within a row span less than that, one row scale plus 8-bit codes represents the same
// numbers essentially exactly, and the objection dissolves.
//
// So this measures, on the real artifact rather than by argument:
//
//   * the spread of group scales within a row, max/min, which is what has to fit in the headroom,
//   * the relative L2 error of reconstructing each weight as int8-with-one-row-scale, against the
//     exact values its Q4/Q5 group-64 encoding represents.
//
// The second number is the one that matters. It is an upper bound on what the route costs on the
// weight side; the activation side (one scale per token rather than per group) is a separate
// question this does not touch.
//
// Build: see tools/CMakeLists.txt -- it links ninfer_artifact.
//
//   ninfer_w4_row_scale_error <model.ninfer> [tensor-name-substring]

#include "artifact/reader.h"
#include "artifact/schema.h"
#include "core/weight.h"
#include "core/weight_view.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <string_view>
#include <vector>

namespace {

using namespace ninfer;
using namespace ninfer::artifact;

float half_to_float(std::uint16_t bits) {
    const std::uint32_t sign     = static_cast<std::uint32_t>(bits & 0x8000u) << 16;
    const std::uint32_t exponent = (bits >> 10) & 0x1fu;
    const std::uint32_t mantissa = bits & 0x3ffu;
    std::uint32_t out            = 0;
    if (exponent == 0) {
        if (mantissa == 0) {
            out = sign;
        } else { // subnormal: normalise it
            std::uint32_t e = exponent + 1;
            std::uint32_t m = mantissa;
            while ((m & 0x400u) == 0) {
                m <<= 1;
                --e;
            }
            m &= 0x3ffu;
            out = sign | ((e + 112) << 23) | (m << 13);
        }
    } else if (exponent == 31) {
        out = sign | 0x7f800000u | (mantissa << 13);
    } else {
        out = sign | ((exponent + 112) << 23) | (mantissa << 13);
    }
    float value = 0.0F;
    std::memcpy(&value, &out, 4);
    return value;
}

struct Stats {
    double sum_sq_error = 0.0;
    double sum_sq_value = 0.0;
    double worst_row_relative = 0.0;
    double spread_max = 1.0;
    std::vector<std::uint64_t> spread_buckets; // <2x, <4x, <8x, <16x, <32x, >=32x
};

// One row's exact values, as the group-64 encoding represents them.
void decode_row(const std::byte* codes, const std::byte* high, const std::byte* scales,
                QType format, int groups, std::vector<float>& out, std::vector<float>& group_scale) {
    const auto* code_bytes = reinterpret_cast<const std::uint8_t*>(codes);
    const auto* high_bytes = reinterpret_cast<const std::uint8_t*>(high);
    const auto* scale_bits = reinterpret_cast<const std::uint16_t*>(scales);
    const int high_per_group = format == QType::Q5_G64_FP16 ? 8 : (format == QType::Q6_G64_FP16 ? 16 : 0);
    out.resize(static_cast<std::size_t>(groups) * 64);
    group_scale.resize(static_cast<std::size_t>(groups));

    for (int g = 0; g < groups; ++g) {
        const float scale = half_to_float(scale_bits[g]);
        group_scale[static_cast<std::size_t>(g)] = std::fabs(scale);
        for (int j = 0; j < 64; ++j) {
            const std::uint8_t byte = code_bytes[g * 32 + j / 2];
            int code                = (j % 2 == 0) ? (byte & 0x0f) : (byte >> 4);
            if (high_per_group) {
                const int bit = (high_bytes[g * high_per_group + j / 8] >> (j % 8)) & 1;
                code |= bit << 4;
                if (format == QType::Q6_G64_FP16) {
                    const int bit2 = (high_bytes[g * high_per_group + 8 + j / 8] >> (j % 8)) & 1;
                    code |= bit2 << 5;
                }
            }
            // Two's complement in the code's own width, matching Q4Codec/Q5Codec in
            // ops/common/rowsplit_a8_mma.cuh: (v ^ half) - half, not an offset-binary v - half.
            const int half = high_per_group ? (format == QType::Q6_G64_FP16 ? 32 : 16) : 8;
            out[static_cast<std::size_t>(g) * 64 + j] =
                static_cast<float>((code ^ half) - half) * scale;
        }
    }
}

void analyse(const Reader& reader, ObjectHandle handle, const std::string& name) {
    const auto& g = reader.geometry(handle);
    if (g.shape.size() != 2 || !is_row_split(g.layout) || !g.group_size) { return; }
    if (g.format != QType::Q4_G64_FP16 && g.format != QType::Q5_G64_FP16 &&
        g.format != QType::Q6_G64_FP16) {
        return;
    }
    const auto bytes  = reader.read_object(handle);
    const auto rows   = static_cast<int>(g.shape[0]);
    const auto groups = static_cast<int>(g.padded_columns / g.group_size);

    Stats stats;
    stats.spread_buckets.assign(6, 0);
    std::vector<float> values;
    std::vector<float> group_scale;
    // Sampling keeps this a minute rather than an hour; rows are homogeneous enough that a stride
    // over thousands of them settles the question.
    const int stride = std::max(1, rows / 2048);
    int sampled      = 0;
    for (int row = 0; row < rows; row += stride) {
        decode_row(bytes.data() + static_cast<std::size_t>(row) * g.code_bytes_per_row,
                   g.high_bytes ? bytes.data() + g.high_offset +
                                      static_cast<std::size_t>(row) * g.high_bytes_per_row
                                : nullptr,
                   bytes.data() + g.scale_offset +
                       static_cast<std::size_t>(row) * g.scale_bytes_per_row,
                   g.format, groups, values, group_scale);
        ++sampled;

        float lo = 0.0F;
        float hi = 0.0F;
        for (const float s : group_scale) {
            if (s <= 0.0F) { continue; }
            if (lo == 0.0F || s < lo) { lo = s; }
            if (s > hi) { hi = s; }
        }
        const double spread = lo > 0.0F ? static_cast<double>(hi) / lo : 1.0;
        stats.spread_max    = std::max(stats.spread_max, spread);
        const int bucket    = spread < 2 ? 0 : spread < 4 ? 1 : spread < 8 ? 2 : spread < 16 ? 3
                                                                        : spread < 32 ? 4 : 5;
        ++stats.spread_buckets[static_cast<std::size_t>(bucket)];

        // One scale per row. Plain absmax/127 is the obvious choice and the wrong one: clipping a
        // few outliers buys back more than it costs, which is why transcode_row_split already
        // searches ratios per group rather than taking absmax. The same search applies here, so
        // the number below is what a materialiser would actually achieve, not a strawman.
        float absmax = 0.0F;
        for (const float v : values) { absmax = std::max(absmax, std::fabs(v)); }
        double row_error = 0.0;
        double row_value = 0.0;
        for (const float v : values) { row_value += static_cast<double>(v) * v; }
        if (absmax > 0.0F) {
            double best = -1.0;
            for (int step = 0; step < 25; ++step) {
                const float ratio     = 0.70F + 0.02F * static_cast<float>(step);
                const float row_scale = absmax * ratio / 127.0F;
                double error          = 0.0;
                for (const float v : values) {
                    const float q = std::clamp(std::round(v / row_scale), -127.0F, 127.0F);
                    const float r = q * row_scale;
                    error += static_cast<double>(v - r) * (v - r);
                }
                if (best < 0.0 || error < best) { best = error; }
            }
            row_error = best;
        }
        stats.sum_sq_error += row_error;
        stats.sum_sq_value += row_value;
        if (row_value > 0.0) {
            stats.worst_row_relative =
                std::max(stats.worst_row_relative, std::sqrt(row_error / row_value));
        }
    }

    const double relative =
        stats.sum_sq_value > 0.0 ? std::sqrt(stats.sum_sq_error / stats.sum_sq_value) : 0.0;
    std::printf("%-46s rows=%-7d groups=%-5d sampled=%-5d  relL2=%.3e  worstRow=%.3e  "
                "spreadMax=%.1fx  [<2x %llu, <4x %llu, <8x %llu, <16x %llu, <32x %llu, >=32x %llu]\n",
                name.c_str(), rows, groups, sampled, relative, stats.worst_row_relative,
                stats.spread_max,
                static_cast<unsigned long long>(stats.spread_buckets[0]),
                static_cast<unsigned long long>(stats.spread_buckets[1]),
                static_cast<unsigned long long>(stats.spread_buckets[2]),
                static_cast<unsigned long long>(stats.spread_buckets[3]),
                static_cast<unsigned long long>(stats.spread_buckets[4]),
                static_cast<unsigned long long>(stats.spread_buckets[5]));
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <model.ninfer> [tensor-name-substring]\n", argv[0]);
        return 2;
    }
    const std::string filter = argc > 2 ? argv[2] : "";
    try {
        Reader reader(argv[1]);
        std::printf("# int8 with one scale per row, against the exact group-64 values\n");
        std::printf("# relL2 is the reconstruction error the cuBLAS route would pay on weights\n");
        const auto& directory = reader.directory();
        for (std::size_t i = 0; i < directory.objects.size(); ++i) {
            const std::string name = object_id(directory.objects[i]);
            if (!filter.empty() && name.find(filter) == std::string::npos) { continue; }
            // Resource objects and unsupported formats have no geometry to ask about.
            try {
                analyse(reader, ObjectHandle{i}, name);
            } catch (const std::exception&) {
            }
        }
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}
