// The AVX2 rowdot of offloaded_sparse_moe's CPU route. Bit-identical to row_values_scalar.
//
// Adapted from Strata src/kernels/cpu/iq_avx2.cpp and src/kernels/cpu/q2_avx2.cpp (Strata commit
// 99f3dbd, MIT, Copyright (c) 2026 Niko1221 and the Strata contributors; notice in
// third_party/strata/LICENSE), which transcribe ggml-cpu's AVX2 dot products (ggml/src/ggml-cpu/
// arch/x86/quants.c, MIT, Copyright (c) 2023-2026 The ggml authors; third_party/ggml/LICENSE).
//
// Modified for NInfer: each 32-value sub-block is decoded once into unsigned magnitudes, a sign
// vector and its two integer half multipliers (Strata's split, with IQ1_M added as 8 grid +- 1 and
// Q2_0 as code - 1); every column then forms the exact int32 sum P_s of the sub-block, and eight
// sub-blocks' P_s are gathered into one vector so that the canonical A8 step (one FP32 product and
// one fma per sub-block into lane s % 32, then the fixed butterfly) runs eight lanes at a time
// (ops/common/canonical_ggml.h). Strata's per-super-block integer accumulation and float order are
// not used: activations carry one scale per 32 values.

#include "ops/offloaded_sparse_moe/cpu/expert_cpu.h"

#if defined(__x86_64__) || defined(_M_X64)

#    include "ggml-common-tables.h"

#    include <immintrin.h>

#    include <stdexcept>

#    if defined(_MSC_VER) && !defined(__clang__)
#        define NINFER_AVX2
#    else
#        define NINFER_AVX2 __attribute__((target("avx2,fma,f16c")))
#    endif

namespace ninfer::ops::offloaded_moe::detail {
namespace {

using namespace ::ninfer::ggml_tables;

// Byte k of v[i] is 0xFF when bit k of ksigns_iq2xs[i] is set, else 0x01 (ggml's keven_signs_q2xs).
struct EvenSigns {
    std::uint64_t v[128];
    EvenSigns() {
        for (int i = 0; i < 128; ++i) {
            std::uint64_t r = 0;
            for (int k = 0; k < 8; ++k) {
                r |= static_cast<std::uint64_t>(((ksigns_iq2xs[i] >> k) & 1) ? 0xFF : 0x01) << (8 * k);
            }
            v[i] = r;
        }
    }
};
const EvenSigns even_signs;

// A decoded sub-block: unsigned magnitudes, the sign source (its sign multiplies the activation,
// zero clears it) and the int16 half multipliers (lanes 0-7: m_0, lanes 8-15: m_1).
struct Sub {
    __m256i aw, sg, m;
};

// 32 sign bits -> bytes of -1 (bit set) or +1 (ggml's bit_selector pattern).
NINFER_AVX2 inline __m256i sign_vector(std::uint32_t bits) {
    const __m128i bm   = _mm_set1_epi32(static_cast<int>(bits));
    const __m128i lo   = _mm_shuffle_epi8(bm, _mm_setr_epi8(0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1));
    const __m128i hi   = _mm_shuffle_epi8(bm, _mm_setr_epi8(2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3));
    const __m256i all  = _mm256_inserti128_si256(_mm256_castsi128_si256(lo), hi, 1);
    const __m256i sel  = _mm256_setr_epi8(1, 2, 4, 8, 16, 32, 64, static_cast<char>(0x80), 1, 2, 4, 8, 16, 32, 64,
                                          static_cast<char>(0x80), 1, 2, 4, 8, 16, 32, 64, static_cast<char>(0x80), 1,
                                          2, 4, 8, 16, 32, 64, static_cast<char>(0x80));
    const __m256i set  = _mm256_cmpeq_epi8(_mm256_and_si256(all, sel), sel);
    return _mm256_or_si256(set, _mm256_set1_epi8(1));
}

NINFER_AVX2 inline __m256i half_multipliers(int m0, int m1) {
    return _mm256_inserti128_si256(_mm256_castsi128_si256(_mm_set1_epi16(static_cast<short>(m0))),
                                   _mm_set1_epi16(static_cast<short>(m1)), 1);
}

NINFER_AVX2 inline __m256i grid4(const std::uint64_t* grid, int i0, int i1, int i2, int i3) {
    return _mm256_set_epi64x(static_cast<long long>(grid[i3]), static_cast<long long>(grid[i2]),
                             static_cast<long long>(grid[i1]), static_cast<long long>(grid[i0]));
}

// Sub-blocks [8 ch, 8 ch + n) of a row (n = 8, or the row's last partial chunk): subs[j] and the
// FP32 scale c of each in cs[j] (cs[j] = 0 for j >= n).
template <QType F>
struct Fmt;

template <>
struct Fmt<QType::GGML_IQ2_XXS> {
    NINFER_AVX2 static void decode(const std::uint8_t* row, int ch, int, Sub* subs, float* cs) {
        const std::uint8_t* b = row + static_cast<std::size_t>(ch) * 66;
        const float c         = canon::mul_rn(fp16_to_f32(load_u16(b)), 0.125F);
        for (int ib = 0; ib < 8; ++ib) {
            const std::uint8_t* q  = b + 2 + 8 * ib;
            const std::uint32_t w1 = load_u32(q + 4);
            subs[ib].aw = grid4(iq2xxs_grid, q[0], q[1], q[2], q[3]);
            subs[ib].sg = _mm256_set_epi64x(static_cast<long long>(even_signs.v[(w1 >> 21) & 127]),
                                            static_cast<long long>(even_signs.v[(w1 >> 14) & 127]),
                                            static_cast<long long>(even_signs.v[(w1 >> 7) & 127]),
                                            static_cast<long long>(even_signs.v[w1 & 127]));
            const int m = 2 * static_cast<int>(w1 >> 28) + 1;
            subs[ib].m  = _mm256_set1_epi16(static_cast<short>(m));
            cs[ib]      = c;
        }
    }
};

template <>
struct Fmt<QType::GGML_IQ2_S> {
    NINFER_AVX2 static void decode(const std::uint8_t* row, int ch, int, Sub* subs, float* cs) {
        const std::uint8_t* b = row + static_cast<std::size_t>(ch) * 82;
        const float c         = canon::mul_rn(fp16_to_f32(load_u16(b)), 0.125F);
        for (int ib = 0; ib < 8; ++ib) {
            const std::uint8_t* qs = b + 2 + 4 * ib;
            const int qh           = b[66 + ib];
            subs[ib].aw = grid4(iq2s_grid, qs[0] | ((qh << 8) & 0x300), qs[1] | ((qh << 6) & 0x300),
                                qs[2] | ((qh << 4) & 0x300), qs[3] | ((qh << 2) & 0x300));
            subs[ib].sg     = sign_vector(load_u32(b + 34 + 4 * ib));
            const int scale = b[74 + ib];
            subs[ib].m      = half_multipliers(2 * (scale & 0xF) + 1, 2 * (scale >> 4) + 1);
            cs[ib]          = c;
        }
    }
};

template <>
struct Fmt<QType::GGML_IQ1_M> {
    NINFER_AVX2 static void decode(const std::uint8_t* row, int ch, int, Sub* subs, float* cs) {
        const std::uint8_t* b     = row + static_cast<std::size_t>(ch) * 56;
        const std::uint16_t sc[4] = {load_u16(b + 48), load_u16(b + 50), load_u16(b + 52), load_u16(b + 54)};
        const auto d16 = static_cast<std::uint16_t>((sc[0] >> 12) | ((sc[1] >> 8) & 0x00F0) |
                                                    ((sc[2] >> 4) & 0x0F00) | (sc[3] & 0xF000));
        const float c  = canon::mul_rn(fp16_to_f32(d16), 0.125F);
        constexpr long long kPlus = 0x0101010101010101LL, kMinus = -1LL;
        for (int ib = 0; ib < 8; ++ib) {
            const int qh0       = b[32 + 2 * ib];
            const int qh1       = b[33 + 2 * ib];
            const std::uint8_t* qs = b + 4 * ib;
            const __m256i g  = grid4(iq1s_grid, qs[0] | ((qh0 << 8) & 0x700), qs[1] | ((qh0 << 4) & 0x700),
                                     qs[2] | ((qh1 << 8) & 0x700), qs[3] | ((qh1 << 4) & 0x700));
            const __m256i dv = _mm256_set_epi64x((qh1 & 0x80) ? kMinus : kPlus, (qh1 & 0x08) ? kMinus : kPlus,
                                                 (qh0 & 0x80) ? kMinus : kPlus, (qh0 & 0x08) ? kMinus : kPlus);
            const __m256i g2 = _mm256_add_epi8(g, g);
            const __m256i g4 = _mm256_add_epi8(g2, g2);
            const __m256i w  = _mm256_add_epi8(_mm256_add_epi8(g4, g4), dv); // 8 grid +- 1
            subs[ib].aw      = _mm256_abs_epi8(w);
            subs[ib].sg      = w;
            const int shift  = 6 * (ib % 2);
            subs[ib].m       = half_multipliers(2 * ((sc[ib / 2] >> shift) & 7) + 1,
                                                2 * ((sc[ib / 2] >> (shift + 3)) & 7) + 1);
            cs[ib]           = c;
        }
    }
};

// 16 bytes of 2-bit codes (value i in byte i / 4, bits 2 (i % 4)) -> 64 codes in value order.
NINFER_AVX2 inline void unpack_q2(const std::uint8_t* codes, __m256i& lo, __m256i& hi) {
    const __m128i b  = _mm_loadu_si128(reinterpret_cast<const __m128i*>(codes));
    const __m128i m3 = _mm_set1_epi8(3);
    const __m128i c0 = _mm_and_si128(b, m3);
    const __m128i c1 = _mm_and_si128(_mm_srli_epi16(b, 2), m3);
    const __m128i c2 = _mm_and_si128(_mm_srli_epi16(b, 4), m3);
    const __m128i c3 = _mm_and_si128(_mm_srli_epi16(b, 6), m3);
    const __m128i a0 = _mm_unpacklo_epi8(c0, c1), a1 = _mm_unpacklo_epi8(c2, c3);
    const __m128i b0 = _mm_unpackhi_epi8(c0, c1), b1 = _mm_unpackhi_epi8(c2, c3);
    lo = _mm256_set_m128i(_mm_unpackhi_epi16(a0, a1), _mm_unpacklo_epi16(a0, a1));
    hi = _mm256_set_m128i(_mm_unpackhi_epi16(b0, b1), _mm_unpacklo_epi16(b0, b1));
}

template <>
struct Fmt<QType::GGML_Q2_0> {
    NINFER_AVX2 static void decode(const std::uint8_t* row, int ch, int n, Sub* subs, float* cs) {
        const __m256i one = _mm256_set1_epi8(1);
        const __m256i m   = _mm256_set1_epi16(1);
        for (int i = 0; i < n / 2; ++i) {
            const std::uint8_t* b = row + static_cast<std::size_t>(4 * ch + i) * 18;
            __m256i lo, hi;
            unpack_q2(b + 2, lo, hi);
            const __m256i w[2] = {_mm256_sub_epi8(lo, one), _mm256_sub_epi8(hi, one)};
            const float c      = fp16_to_f32(load_u16(b));
            for (int h = 0; h < 2; ++h) {
                subs[2 * i + h].aw = _mm256_abs_epi8(w[h]);
                subs[2 * i + h].sg = w[h];
                subs[2 * i + h].m  = m;
                cs[2 * i + h]      = c;
            }
        }
    }
};

// Lane j of the result = the sum of the eight int32 lanes of v[j].
NINFER_AVX2 inline __m256i gather_sums(const __m256i* v) {
    const __m256i h01 = _mm256_hadd_epi32(v[0], v[1]);
    const __m256i h23 = _mm256_hadd_epi32(v[2], v[3]);
    const __m256i h45 = _mm256_hadd_epi32(v[4], v[5]);
    const __m256i h67 = _mm256_hadd_epi32(v[6], v[7]);
    const __m256i q0  = _mm256_hadd_epi32(h01, h23);
    const __m256i q1  = _mm256_hadd_epi32(h45, h67);
    return _mm256_add_epi32(_mm256_permute2x128_si256(q0, q1, 0x20), _mm256_permute2x128_si256(q0, q1, 0x31));
}

// canon::reduce_lanes on partials p[0..32) held as four vectors of eight.
NINFER_AVX2 inline float reduce_lanes(const __m256* p) {
    const __m256 c  = _mm256_add_ps(_mm256_add_ps(p[0], p[2]), _mm256_add_ps(p[1], p[3]));
    const __m128 s4 = _mm_add_ps(_mm256_castps256_ps128(c), _mm256_extractf128_ps(c, 1));
    const __m128 s2 = _mm_add_ps(s4, _mm_movehl_ps(s4, s4));
    const __m128 s1 = _mm_add_ss(s2, _mm_shuffle_ps(s2, s2, 1));
    return _mm_cvtss_f32(s1);
}

template <QType F, int NC>
NINFER_AVX2 void rows(const std::uint8_t* matrix, std::size_t row_bytes, int k, int r0, int r1,
                      const A8Act* acts, float* out) {
    const int subs   = k / canon::kA8Values;
    const int chunks = (subs + 7) / 8;
    for (int r = r0; r < r1; ++r) {
        const std::uint8_t* row = matrix + static_cast<std::size_t>(r) * row_bytes;
        for (std::size_t at = 0; at < row_bytes; at += 64) {
            _mm_prefetch(reinterpret_cast<const char*>(row + row_bytes + at), _MM_HINT_T0);
        }
        __m256 partial[NC][4];
        for (int c = 0; c < NC; ++c) {
            for (int l = 0; l < 4; ++l) { partial[c][l] = _mm256_setzero_ps(); }
        }
        for (int ch = 0; ch < chunks; ++ch) {
            const int n = subs - 8 * ch < 8 ? subs - 8 * ch : 8;
            Sub sub[8];
            alignas(32) float cs[8] = {};
            Fmt<F>::decode(row, ch, n, sub, cs);
            const __m256 cvec = _mm256_load_ps(cs);
            for (int c = 0; c < NC; ++c) {
                const std::int8_t* q = acts[c].q + 256 * ch;
                __m256i v[8];
                for (int j = 0; j < 8; ++j) {
                    if (j < n) {
                        const __m256i qv = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q + 32 * j));
                        const __m256i pr = _mm256_maddubs_epi16(sub[j].aw, _mm256_sign_epi8(qv, sub[j].sg));
                        v[j]             = _mm256_madd_epi16(pr, sub[j].m);
                    } else {
                        v[j] = _mm256_setzero_si256();
                    }
                }
                const __m256i p = gather_sums(v);
                __m256 dvec;
                if (n == 8) {
                    dvec = _mm256_loadu_ps(acts[c].d + 8 * ch);
                } else {
                    alignas(32) float ds[8] = {};
                    for (int j = 0; j < n; ++j) { ds[j] = acts[c].d[8 * ch + j]; }
                    dvec = _mm256_load_ps(ds);
                }
                __m256& lane = partial[c][ch % 4];
                lane         = _mm256_fmadd_ps(_mm256_mul_ps(cvec, dvec), _mm256_cvtepi32_ps(p), lane);
            }
        }
        for (int c = 0; c < NC; ++c) { out[static_cast<std::size_t>(r - r0) * NC + c] = reduce_lanes(partial[c]); }
    }
}

template <QType F>
void rows_any(int ncols, const std::uint8_t* matrix, std::size_t row_bytes, int k, int r0, int r1,
              const A8Act* acts, float* out) {
    switch (ncols) {
    case 1: rows<F, 1>(matrix, row_bytes, k, r0, r1, acts, out); return;
    case 2: rows<F, 2>(matrix, row_bytes, k, r0, r1, acts, out); return;
    case 3: rows<F, 3>(matrix, row_bytes, k, r0, r1, acts, out); return;
    case 4: rows<F, 4>(matrix, row_bytes, k, r0, r1, acts, out); return;
    case 5: rows<F, 5>(matrix, row_bytes, k, r0, r1, acts, out); return;
    case 6: rows<F, 6>(matrix, row_bytes, k, r0, r1, acts, out); return;
    case 7: rows<F, 7>(matrix, row_bytes, k, r0, r1, acts, out); return;
    case 8: rows<F, 8>(matrix, row_bytes, k, r0, r1, acts, out); return;
    default: throw std::invalid_argument("offloaded_sparse_moe: ncols must be in [1, 8]");
    }
}

} // namespace

void row_values_avx2(QType format, const std::uint8_t* matrix, std::size_t row_bytes, int k, int r0, int r1,
                     const A8Act* acts, int ncols, float* out) {
    switch (format) {
    case QType::GGML_IQ2_XXS: rows_any<QType::GGML_IQ2_XXS>(ncols, matrix, row_bytes, k, r0, r1, acts, out); return;
    case QType::GGML_IQ2_S: rows_any<QType::GGML_IQ2_S>(ncols, matrix, row_bytes, k, r0, r1, acts, out); return;
    case QType::GGML_IQ1_M: rows_any<QType::GGML_IQ1_M>(ncols, matrix, row_bytes, k, r0, r1, acts, out); return;
    case QType::GGML_Q2_0: rows_any<QType::GGML_Q2_0>(ncols, matrix, row_bytes, k, r0, r1, acts, out); return;
    default: throw std::invalid_argument("offloaded_sparse_moe: not an expert block format");
    }
}

} // namespace ninfer::ops::offloaded_moe::detail

#else

#    include <stdexcept>

namespace ninfer::ops::offloaded_moe::detail {

void row_values_avx2(QType, const std::uint8_t*, std::size_t, int, int, int, const A8Act*, int, float*) {
    throw std::invalid_argument("offloaded_sparse_moe: AVX2 is not available on this target");
}

} // namespace ninfer::ops::offloaded_moe::detail

#endif
