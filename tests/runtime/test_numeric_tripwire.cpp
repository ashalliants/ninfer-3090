// Host-only checks for the numeric tripwire: the record codec, the naive oracle that the CUDA Op is
// qualified against (tests/ops/test_numeric_tripwire.cpp), the site table, the log line and the
// repeat throttle. Needs no GPU. The kernel itself is only exercised by the CUDA test.
#include "models/qwen3_5/execution/numeric_tripwire_report.h"
#include "ninfer/ops/numeric_tripwire_record.h"
#include "ops/numeric_tripwire_oracle.h"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namespace ninfer;
using namespace ninfer::models::qwen3_5::execution;
using namespace ninfer::test;

void require(bool condition, const char* message) {
    if (!condition) { throw std::runtime_error(message); }
}

bool contains(const std::string& text, const std::string& part) {
    return text.find(part) != std::string::npos;
}

constexpr float kInf = std::numeric_limits<float>::infinity();
constexpr float kNan = std::numeric_limits<float>::quiet_NaN();

void test_record_codec() {
    ops::NumericTripwireRecord clean{};
    require(!ops::decode_numeric_tripwire_record(clean), "all-zero record means nothing seen");

    // Lower index wins the atomic maximum; its own value rides along.
    const std::uint64_t low  = ops::numeric_tripwire_pack(5, std::bit_cast<std::uint32_t>(2.5F));
    const std::uint64_t high = ops::numeric_tripwire_pack(90000, std::bit_cast<std::uint32_t>(9.0F));
    require(low > high, "a lower index must pack to a larger key");

    ops::NumericTripwireRecord record{};
    record.site_plus_one = 8;
    record.count         = 41;
    record.best          = low;
    const auto finding   = ops::decode_numeric_tripwire_record(record);
    require(finding.has_value(), "record with a site decodes");
    require(finding->site == 7 && finding->element_index == 5 && finding->value == 2.5F &&
                finding->offending_count == 41,
            "decoded fields");

    // Index 0 and the largest accepted index must both stay distinguishable from "empty".
    for (const std::uint32_t index : {0U, 0x7ffffffeU}) {
        require(ops::numeric_tripwire_pack(index, 0) != 0, "a real key is never zero");
        record.best = ops::numeric_tripwire_pack(index, std::bit_cast<std::uint32_t>(kNan));
        const auto round_trip = ops::decode_numeric_tripwire_record(record);
        require(round_trip && round_trip->element_index == index && std::isnan(round_trip->value),
                "index and NaN payload round trip");
    }
}

void test_bit_converters() {
    require(tripwire_half_bits_to_float(0x7bffU) == 65504.0F, "fp16 max");
    require(tripwire_half_bits_to_float(0xfbffU) == -65504.0F, "fp16 -max");
    require(tripwire_half_bits_to_float(0x7c00U) == kInf, "fp16 +inf");
    require(tripwire_half_bits_to_float(0xfc00U) == -kInf, "fp16 -inf");
    require(std::isnan(tripwire_half_bits_to_float(0x7e00U)), "fp16 nan");
    require(std::isnan(tripwire_half_bits_to_float(0xfd23U)), "fp16 nan with payload and sign");
    require(tripwire_half_bits_to_float(0x0001U) == std::ldexp(1.0F, -24), "fp16 smallest subnormal");
    require(tripwire_half_bits_to_float(0x3c00U) == 1.0F, "fp16 one");
    require(tripwire_bf16_bits_to_float(0x7f7fU) == 3.3895314e38F, "bf16 max");
    require(tripwire_bf16_bits_to_float(0x7f80U) == kInf, "bf16 +inf");
    require(std::isnan(tripwire_bf16_bits_to_float(0x7fc1U)), "bf16 nan");

    // Encoders against the decoders, including rounding at the top of the fp16 range.
    const float samples[] = {0.0F, 1.0F, -1.0F, 0.1F, 100.0F, 65504.0F, -65504.0F, 1e-7F, 6.1e-5F};
    for (const float s : samples) {
        const float back = tripwire_half_bits_to_float(tripwire_float_to_half_bits(s));
        require(std::fabs(back - s) <= std::fabs(s) * 1.0F / 1024.0F + std::ldexp(1.0F, -24),
                "fp16 round trip within half an ulp");
    }
    require(tripwire_float_to_half_bits(65504.0F) == 0x7bffU, "fp16 encodes the maximum exactly");
    require(tripwire_float_to_half_bits(65520.0F) == 0x7c00U, "65520 rounds up to infinity");
    require(tripwire_float_to_half_bits(65519.0F) == 0x7bffU, "65519 rounds down to the maximum");
    require(tripwire_float_to_half_bits(-kInf) == 0xfc00U, "fp16 encodes -inf");
    require(tripwire_float_to_bf16_bits(3.3895314e38F) == 0x7f7fU, "bf16 encodes the maximum");
    require(tripwire_float_to_bf16_bits(kInf) == 0x7f80U, "bf16 encodes +inf");
}

TripwireOracleScan scan(const std::vector<float>& values, float limit) {
    return tripwire_oracle_scan(values, limit);
}

void test_oracle_cases() {
    // Clean tensor.
    {
        std::vector<float> v(1000, 3.0F);
        const auto s = scan(v, 65504.0F);
        require(s.offending == 0 && !s.first_index, "clean tensor reports nothing");
    }
    // Single NaN, with a sign and payload that a naive `x != x` test must still catch.
    {
        std::vector<float> v(4096, 1.0F);
        v[777] = std::bit_cast<float>(0xffc12345U);
        const auto s = scan(v, 65504.0F);
        require(s.offending == 1 && s.first_index == 777U && std::isnan(s.first_value), "single NaN");
    }
    // Infinities of both signs, with the limit finite and infinite.
    {
        std::vector<float> v(64, 0.5F);
        v[40] = -kInf;
        v[3]  = kInf;
        for (const float limit : {65504.0F, kInf}) {
            const auto s = scan(v, limit);
            require(s.offending == 2 && s.first_index == 3U && s.first_value == kInf,
                    "infinity offends whatever the limit");
        }
    }
    // Limit exceeded; exactly the limit does not offend, either sign.
    {
        std::vector<float> v(100, 1.0F);
        v[9]  = 50.0F;
        v[10] = -50.0F;
        v[11] = 50.001F;
        v[12] = -51.0F;
        const auto s = scan(v, 50.0F);
        require(s.offending == 2 && s.first_index == 11U, "only values beyond the limit offend");
    }
    // Several kinds at once; the lowest index decides the first finding, the count totals all.
    {
        std::vector<float> v(600000, 0.25F);
        v[599999] = kNan;
        v[262147] = 1e30F;
        v[3]      = -kInf;
        v[262211] = kNan;
        const auto s = scan(v, 65504.0F);
        require(s.offending == 4 && s.first_index == 3U && s.first_value == -kInf,
                "multiple findings: lowest index first, count of all");
    }
    // Tail elements past a full block of 256.
    for (const std::size_t n : {1U, 255U, 256U, 257U, 1000003U}) {
        std::vector<float> v(n, 1.0F);
        v[n - 1] = kNan;
        const auto s = scan(v, 65504.0F);
        require(s.offending == 1 && s.first_index == n - 1, "bad value in the last element");
    }
    // FP16 boundary: the largest finite value is not offending at the same limit, one step over is.
    {
        const std::vector<float> at_max = {tripwire_half_bits_to_float(0x7bffU),
                                           tripwire_half_bits_to_float(0xfbffU)};
        require(scan(at_max, 65504.0F).offending == 0, "+-65504 is within a 65504 limit");
        require(scan(at_max, 65503.0F).offending == 2, "+-65504 exceeds a 65503 limit");
        const std::vector<float> inf = {tripwire_half_bits_to_float(0x7c00U)};
        require(scan(inf, 65504.0F).offending == 1, "fp16 infinity offends");
    }
    // BF16 and FP32 boundaries with an infinite limit: finite maxima pass, infinity does not.
    {
        const std::vector<float> maxima = {tripwire_bf16_bits_to_float(0x7f7fU),
                                           std::numeric_limits<float>::max(),
                                           -std::numeric_limits<float>::max()};
        require(scan(maxima, kInf).offending == 0, "finite maxima pass an infinite limit");
        const std::vector<float> one_over = {1.0F, std::nextafter(1.0F, 2.0F)};
        const auto s                      = scan(one_over, 1.0F);
        require(s.offending == 1 && s.first_index == 1U, "one ulp over the limit offends");
    }
}

void test_oracle_record_semantics() {
    TripwireOracleRecord record;
    const auto clean = scan(std::vector<float>(10, 1.0F), 65504.0F);
    record.apply(3, clean);
    require(record.site_plus_one == 0 && record.count == 0, "clean scan leaves the record alone");

    std::vector<float> a(100, 1.0F);
    a[60] = kNan;
    record.apply(5, scan(a, 65504.0F));
    std::vector<float> b(100, 1.0F);
    b[2] = kInf;
    record.apply(9, scan(b, 65504.0F)); // a later site: counted, does not take over
    require(record.site_plus_one == 6 && record.best_index == 60 && std::isnan(record.best_value),
            "the first site to see a bad value keeps the record");
    require(record.count == 2, "count totals every site");
    std::vector<float> c(100, 1.0F);
    c[10] = kNan;
    record.apply(5, scan(c, 65504.0F)); // the same site again keeps the lowest index
    require(record.best_index == 10 && record.count == 3, "same site keeps the lowest index");
}

void test_site_table_and_message() {
    const std::uint32_t gdn    = tripwire_site(TripwireProbe::GdnOutput, 16);
    const std::uint32_t hidden = tripwire_site(TripwireProbe::FinalHidden);
    const std::uint32_t logits = tripwire_site(TripwireProbe::FinalLogits);
    require(gdn != hidden && hidden != logits && gdn != logits, "site ids are distinct");
    require(contains(describe_tripwire_site(gdn), "linear-attention layer 16"), "gdn site name");
    require(contains(describe_tripwire_site(hidden), "final hidden state"), "hidden site name");
    require(contains(describe_tripwire_site(logits), "final logits"), "logits site name");
    require(contains(describe_tripwire_site(0x00990001U), "unknown probe site 0x00990001"),
            "unknown site is still printable");
    require(tripwire_probes_gdn_layer(0) && tripwire_probes_gdn_layer(16) &&
                tripwire_probes_gdn_layer(32) && !tripwire_probes_gdn_layer(1) &&
                !tripwire_probes_gdn_layer(47),
            "every 16th linear-attention layer is probed");

    require(group_thousands(0) == "0" && group_thousands(999) == "999" &&
                group_thousands(1000) == "1,000" && group_thousands(1234567) == "1,234,567",
            "thousands separators");

    ops::NumericTripwireFinding finding;
    finding.site            = logits;
    finding.element_index   = 2 * 248320 + 17;
    finding.value           = kNan;
    finding.offending_count = 12345;
    const TripwireGeometry geometry{.vocab_size = 248320, .hidden_size = 5120};
    const std::string line = format_tripwire_message(finding, 65504.0F, geometry);
    require(line.find('\n') == std::string::npos, "one physical line");
    require(contains(line, "numeric tripwire | first bad value at final logits"), line.c_str());
    require(contains(line, "NaN in element 496,657 (token id 17, batch column 2)"), line.c_str());
    require(contains(line, "12,345 bad element(s) this round"), line.c_str());
    require(contains(line, "generation continues"), line.c_str());
    require(!contains(line, "earlier round"), "no suppression note without suppression");

    finding.site          = gdn;
    finding.element_index = 5;
    finding.value         = 70000.0F;
    const std::string over = format_tripwire_message(finding, 65504.0F, geometry, 300);
    require(contains(over, "linear-attention layer 16"), over.c_str());
    require(contains(over, "70000 (above the limit 65504) in element 5"), over.c_str());
    require(contains(over, "300 earlier round(s) not logged"), over.c_str());

    finding.value = -kInf;
    require(contains(format_tripwire_message(finding, 65504.0F, geometry), "-Inf in element 5"),
            "negative infinity wording");
}

void test_throttle() {
    TripwireReportThrottle throttle;
    require(throttle.admit(7) == std::optional<std::uint32_t>(0), "first sighting is logged");
    for (std::uint32_t i = 0; i + 1U < TripwireReportThrottle::kRepeatEveryRounds; ++i) {
        require(!throttle.admit(7), "the same site stays quiet");
    }
    // Rounds 2..256 were quiet; round 257 repeats the line and reports the skipped ones.
    const auto repeat = throttle.admit(7);
    require(repeat && *repeat == TripwireReportThrottle::kRepeatEveryRounds - 1U,
            "the same site repeats every kRepeatEveryRounds flagged rounds");
    require(!throttle.admit(7), "quiet again after a repeat");
    const auto changed = throttle.admit(8);
    require(changed && *changed == 1U, "a different site is logged at once");
}

} // namespace

int main() {
    try {
        test_record_codec();
        test_bit_converters();
        test_oracle_cases();
        test_oracle_record_semantics();
        test_site_table_and_message();
        test_throttle();
    } catch (const std::exception& error) {
        std::cerr << "FAIL numeric tripwire host test: " << error.what() << '\n';
        return 1;
    }
    std::cout << "OK numeric tripwire host test\n";
    return 0;
}
