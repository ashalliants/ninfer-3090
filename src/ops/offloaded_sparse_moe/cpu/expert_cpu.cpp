// Host routes of offloaded_sparse_moe: the record geometry, the exact sub-block decode, the scalar
// rowdot (the CPU reference route) and the expert composition both ISAs share.
//
// The sub-block decoders transcribe ggml's reference dequantize_row_iq2_xxs / iq2_s / iq1_m / q2_0
// (llama.cpp b11316 ggml/src/ggml-quants.c; MIT, notice in third_party/ggml/LICENSE) into integer
// weights and multipliers, so that c * m * w is the reference value exactly.

#include "ops/offloaded_sparse_moe/cpu/expert_cpu.h"

#include "core/weight_view.h"
#include "ggml-common-tables.h"

#include <array>
#include <stdexcept>
#include <vector>

#if defined(__x86_64__) || defined(_M_X64)
#    if defined(_MSC_VER)
#        include <intrin.h>
#    else
#        include <cpuid.h>
#    endif
#    define NINFER_MOE_X86 1
#endif

namespace ninfer::ops::offloaded_moe {

RecordGeometry record_geometry(QType format) {
    if (!is_ggml_record(format)) {
        throw std::invalid_argument("offloaded_sparse_moe: not a GGML expert record format");
    }
    const std::array<std::uint64_t, 3> shape = {1, kHidden, kIntermediate};
    const auto g     = weight_geometry(format, QuantLayout::GgmlExpertRecord, shape);
    const auto parts = ggml_record_parts(format);
    if (parts.down != QType::GGML_Q2_0) {
        throw std::invalid_argument("offloaded_sparse_moe: record down part must be Q2_0");
    }
    RecordGeometry out;
    out.format            = format;
    out.gate_up           = parts.gate_up;
    out.down              = parts.down;
    out.up_offset         = g.record_up_offset;
    out.down_offset       = g.record_down_offset;
    out.bytes             = g.record_bytes;
    out.gate_up_row_bytes = ggml_record_part(g, 0, ExpertPart::Gate).row_bytes;
    out.down_row_bytes    = ggml_record_part(g, 0, ExpertPart::Down).row_bytes;
    return out;
}

namespace detail {

using namespace ::ninfer::ggml_tables;

namespace {

// IQ2_XXS, 66 bytes per 256 values: fp16 d; per 32 values 4 grid bytes and a word of 4 x 7-bit
// sign indices plus a 4-bit scale ls: value = d (0.5 + ls) / 4 * grid * sign = (d / 8)(2 ls + 1) w.
ExpertSub iq2_xxs(const std::uint8_t* b, int ib) {
    ExpertSub out{};
    const std::uint8_t* q    = b + 2 + 8 * ib;
    const std::uint32_t aux1 = load_u32(q + 4);
    for (int l = 0; l < 4; ++l) {
        const std::uint64_t grid = iq2xxs_grid[q[l]];
        const std::uint8_t signs = ksigns_iq2xs[(aux1 >> (7 * l)) & 127];
        for (int j = 0; j < 8; ++j) {
            const int g          = static_cast<int>((grid >> (8 * j)) & 0xFF);
            out.w[8 * l + j]     = static_cast<std::int8_t>((signs & kmask_iq2xs[j]) != 0 ? -g : g);
        }
    }
    out.m[0] = out.m[1] = 2 * static_cast<int>(aux1 >> 28) + 1;
    out.c               = canon::mul_rn(fp16_to_f32(load_u16(b)), 0.125F);
    return out;
}

// IQ2_S, 82 bytes per 256 values: fp16 d, qs[32] grid low bytes, signs[32], qh[8], scales[8]; the
// two halves of 16 values have their own 4-bit scales.
ExpertSub iq2_s(const std::uint8_t* b, int ib) {
    ExpertSub out{};
    const int qh       = b[66 + ib];
    const int scale    = b[74 + ib];
    for (int l = 0; l < 4; ++l) {
        const int index          = b[2 + 4 * ib + l] | ((qh << (8 - 2 * l)) & 0x300);
        const std::uint64_t grid = iq2s_grid[index];
        const std::uint8_t signs = b[34 + 4 * ib + l];
        for (int j = 0; j < 8; ++j) {
            const int g      = static_cast<int>((grid >> (8 * j)) & 0xFF);
            out.w[8 * l + j] = static_cast<std::int8_t>((signs & kmask_iq2xs[j]) != 0 ? -g : g);
        }
    }
    out.m[0] = 2 * (scale & 0xF) + 1;
    out.m[1] = 2 * (scale >> 4) + 1;
    out.c    = canon::mul_rn(fp16_to_f32(load_u16(b)), 0.125F);
    return out;
}

// IQ1_M, 56 bytes per 256 values: qs[32], qh[16], four u16 scales holding 3-bit sub-scales and the
// fp16 d in their top nibbles. value = d (2 s + 1)(grid +- 1/8) = (d / 8)(2 s + 1)(8 grid +- 1).
ExpertSub iq1_m(const std::uint8_t* b, int ib) {
    ExpertSub out{};
    const std::uint16_t sc[4] = {load_u16(b + 48), load_u16(b + 50), load_u16(b + 52), load_u16(b + 54)};
    const auto d16 = static_cast<std::uint16_t>((sc[0] >> 12) | ((sc[1] >> 8) & 0x00F0) |
                                                ((sc[2] >> 4) & 0x0F00) | (sc[3] & 0xF000));
    const int shift = 6 * (ib % 2);
    const int qh0   = b[32 + 2 * ib];
    const int qh1   = b[33 + 2 * ib];
    const int index[4] = {b[4 * ib + 0] | ((qh0 << 8) & 0x700), b[4 * ib + 1] | ((qh0 << 4) & 0x700),
                          b[4 * ib + 2] | ((qh1 << 8) & 0x700), b[4 * ib + 3] | ((qh1 << 4) & 0x700)};
    const int delta[4] = {(qh0 & 0x08) ? -1 : 1, (qh0 & 0x80) ? -1 : 1, (qh1 & 0x08) ? -1 : 1,
                          (qh1 & 0x80) ? -1 : 1};
    for (int l = 0; l < 4; ++l) {
        const std::uint64_t grid = iq1s_grid[index[l]];
        for (int j = 0; j < 8; ++j) {
            const int g      = static_cast<std::int8_t>((grid >> (8 * j)) & 0xFF);
            out.w[8 * l + j] = static_cast<std::int8_t>(8 * g + delta[l]);
        }
    }
    out.m[0] = 2 * ((sc[ib / 2] >> shift) & 7) + 1;
    out.m[1] = 2 * ((sc[ib / 2] >> (shift + 3)) & 7) + 1;
    out.c    = canon::mul_rn(fp16_to_f32(d16), 0.125F);
    return out;
}

// Q2_0, 18 bytes per 64 values: fp16 d, 2-bit codes; value = (code - 1) d. Half h of the block.
ExpertSub q2_0(const std::uint8_t* b, int half) {
    ExpertSub out{};
    for (int j = 0; j < 32; ++j) {
        const int v = 32 * half + j;
        out.w[j]    = static_cast<std::int8_t>(((b[2 + v / 4] >> (2 * (v % 4))) & 3) - 1);
    }
    out.m[0] = out.m[1] = 1;
    out.c               = fp16_to_f32(load_u16(b));
    return out;
}

} // namespace

ExpertSub decode_sub(QType format, const std::uint8_t* row, int s) {
    switch (format) {
    case QType::GGML_IQ2_XXS: return iq2_xxs(row + static_cast<std::size_t>(s / 8) * 66, s % 8);
    case QType::GGML_IQ2_S: return iq2_s(row + static_cast<std::size_t>(s / 8) * 82, s % 8);
    case QType::GGML_IQ1_M: return iq1_m(row + static_cast<std::size_t>(s / 8) * 56, s % 8);
    case QType::GGML_Q2_0: return q2_0(row + static_cast<std::size_t>(s / 2) * 18, s % 2);
    default: throw std::invalid_argument("offloaded_sparse_moe: not an expert block format");
    }
}

void row_values_scalar(QType format, const std::uint8_t* matrix, std::size_t row_bytes, int k, int r0,
                       int r1, const A8Act* acts, int ncols, float* out) {
    const int subs = k / canon::kA8Values;
    for (int r = r0; r < r1; ++r) {
        const std::uint8_t* row = matrix + static_cast<std::size_t>(r) * row_bytes;
        float partial[kMaxColumns][canon::kLanes];
        for (int c = 0; c < ncols; ++c) {
            for (float& p : partial[c]) { p = 0.0F; }
        }
        for (int s = 0; s < subs; ++s) {
            const ExpertSub sub = decode_sub(format, row, s);
            for (int c = 0; c < ncols; ++c) {
                const std::int8_t* q = acts[c].q + canon::kA8Values * s;
                std::int32_t half[2] = {0, 0};
                for (int j = 0; j < 32; ++j) { half[j / 16] += sub.w[j] * q[j]; }
                const std::int32_t p = sub.m[0] * half[0] + sub.m[1] * half[1];
                float& lane          = partial[c][s % canon::kLanes];
                lane                 = canon::accumulate(lane, sub.c, acts[c].d[s], p);
            }
        }
        for (int c = 0; c < ncols; ++c) {
            out[static_cast<std::size_t>(r - r0) * ncols + c] = canon::reduce_lanes(partial[c]);
        }
    }
}

void row_values(CpuIsa isa, QType format, const std::uint8_t* matrix, std::size_t row_bytes, int k, int r0,
                int r1, const A8Act* acts, int ncols, float* out) {
    if (ncols < 1 || ncols > kMaxColumns) { throw std::invalid_argument("offloaded_sparse_moe: ncols must be in [1, 8]"); }
    if (isa == CpuIsa::kAvx2) {
        row_values_avx2(format, matrix, row_bytes, k, r0, r1, acts, ncols, out);
    } else {
        row_values_scalar(format, matrix, row_bytes, k, r0, r1, acts, ncols, out);
    }
}

void quantize_bf16(const std::uint16_t* x, int k, std::int8_t* q, float* d) {
    for (int g = 0; g < k / canon::kA8Values; ++g) {
        d[g] = canon::a8_quantize_bf16(x + canon::kA8Values * g, q + canon::kA8Values * g);
    }
}

void quantize_f32(const float* x, int k, std::int8_t* q, float* d) {
    for (int g = 0; g < k / canon::kA8Values; ++g) {
        d[g] = canon::a8_quantize_f32(x + canon::kA8Values * g, q + canon::kA8Values * g);
    }
}

void gate_up_unit(CpuIsa isa, const RecordGeometry& geometry, const std::uint8_t* record, const A8Act* x,
                  int ncols, int u, std::int8_t* const* hq, float* const* hd) {
    float gate[kUnitRows * kMaxColumns];
    float up[kUnitRows * kMaxColumns];
    const int r0 = kUnitRows * u, r1 = r0 + kUnitRows;
    row_values(isa, geometry.gate_up, record, geometry.gate_up_row_bytes, kHidden, r0, r1, x, ncols, gate);
    row_values(isa, geometry.gate_up, record + geometry.up_offset, geometry.gate_up_row_bytes, kHidden, r0, r1,
               x, ncols, up);
    for (int c = 0; c < ncols; ++c) {
        float h[kUnitRows];
        for (int i = 0; i < kUnitRows; ++i) {
            h[i] = canon::swiglu_f32(gate[i * ncols + c], up[i * ncols + c]);
        }
        hd[c][u] = canon::a8_quantize_f32(h, hq[c] + kUnitRows * u);
    }
}

void down_rows(CpuIsa isa, const RecordGeometry& geometry, const std::uint8_t* record, const A8Act* h, int ncols,
               int r0, int r1, std::uint16_t* const* y) {
    constexpr int kChunk = 16;
    float values[kChunk * kMaxColumns];
    for (int r = r0; r < r1; r += kChunk) {
        const int e = r + kChunk < r1 ? r + kChunk : r1;
        row_values(isa, geometry.down, record + geometry.down_offset, geometry.down_row_bytes, kIntermediate, r,
                   e, h, ncols, values);
        for (int i = r; i < e; ++i) {
            for (int c = 0; c < ncols; ++c) {
                y[c][i] = canon::f32_to_bf16_rn(values[static_cast<std::size_t>(i - r) * ncols + c]);
            }
        }
    }
}

} // namespace detail

namespace {

struct CpuFeatures {
    bool avx2 = false;
};

CpuFeatures detect_features() {
    CpuFeatures f;
#if defined(NINFER_MOE_X86)
    unsigned b7 = 0, c1 = 0;
    bool osxsave = false;
    unsigned long long xcr0 = 0;
#    if defined(_MSC_VER)
    int r[4];
    __cpuid(r, 0);
    if (r[0] < 7) { return f; }
    __cpuid(r, 1);
    c1      = static_cast<unsigned>(r[2]);
    osxsave = (c1 & (1U << 27)) != 0;
    xcr0    = osxsave ? _xgetbv(0) : 0;
    __cpuidex(r, 7, 0);
    b7 = static_cast<unsigned>(r[1]);
#    else
    if (__get_cpuid_max(0, nullptr) < 7) { return f; }
    unsigned a = 0, b = 0, c = 0, d = 0;
    __cpuid(1, a, b, c, d);
    c1      = c;
    osxsave = (c1 & (1U << 27)) != 0;
    if (osxsave) {
        unsigned lo = 0, hi = 0;
        __asm__ volatile("xgetbv" : "=a"(lo), "=d"(hi) : "c"(0));
        xcr0 = (static_cast<unsigned long long>(hi) << 32) | lo;
    }
    __cpuid_count(7, 0, a, b, c, d);
    b7 = b;
#    endif
    const bool ymm  = (xcr0 & 0x6) == 0x6;
    const bool fma  = (c1 & (1U << 12)) != 0;
    const bool f16c = (c1 & (1U << 29)) != 0;
    f.avx2          = ymm && fma && f16c && (b7 & (1U << 5)) != 0;
#endif
    return f;
}

const CpuFeatures& features() {
    static const CpuFeatures f = detect_features();
    return f;
}

} // namespace

const char* cpu_isa_name(CpuIsa isa) {
    switch (isa) {
    case CpuIsa::kScalar: return "scalar";
    case CpuIsa::kAvx2: return "avx2";
    }
    return "unknown";
}

bool cpu_isa_supported(CpuIsa isa) {
    switch (isa) {
    case CpuIsa::kScalar: return true;
    case CpuIsa::kAvx2: return features().avx2;
    }
    return false;
}

CpuIsa best_cpu_isa() { return cpu_isa_supported(CpuIsa::kAvx2) ? CpuIsa::kAvx2 : CpuIsa::kScalar; }

void expert_forward_cpu(CpuIsa isa, QType format, const std::uint8_t* record, int ncols,
                        const std::uint16_t* const* x, std::uint16_t* const* y) {
    if (ncols < 1 || ncols > kMaxColumns) { throw std::invalid_argument("offloaded_sparse_moe: ncols must be in [1, 8]"); }
    if (!cpu_isa_supported(isa)) { throw std::invalid_argument("offloaded_sparse_moe: unsupported CPU ISA"); }
    const RecordGeometry geometry = record_geometry(format);
    std::vector<std::int8_t> xq(static_cast<std::size_t>(ncols) * kHidden);
    std::vector<float> xd(static_cast<std::size_t>(ncols) * detail::kGateUpGroups);
    std::vector<std::int8_t> hq(static_cast<std::size_t>(ncols) * kIntermediate);
    std::vector<float> hd(static_cast<std::size_t>(ncols) * detail::kHGroups);
    detail::A8Act xa[kMaxColumns], ha[kMaxColumns];
    std::int8_t* hq_ptr[kMaxColumns];
    float* hd_ptr[kMaxColumns];
    for (int c = 0; c < ncols; ++c) {
        std::int8_t* q = xq.data() + static_cast<std::size_t>(c) * kHidden;
        float* d       = xd.data() + static_cast<std::size_t>(c) * detail::kGateUpGroups;
        detail::quantize_bf16(x[c], kHidden, q, d);
        xa[c]     = {q, d};
        hq_ptr[c] = hq.data() + static_cast<std::size_t>(c) * kIntermediate;
        hd_ptr[c] = hd.data() + static_cast<std::size_t>(c) * detail::kHGroups;
        ha[c]     = {hq_ptr[c], hd_ptr[c]};
    }
    for (int u = 0; u < detail::kUnits; ++u) { detail::gate_up_unit(isa, geometry, record, xa, ncols, u, hq_ptr, hd_ptr); }
    detail::down_rows(isa, geometry, record, ha, ncols, 0, kHidden, y);
}

} // namespace ninfer::ops::offloaded_moe
