#pragma once

// Host-only half of the numeric tripwire (ops::numeric_tripwire_scan): which probe sites exist,
// what they are called, how a finding is worded and how often it may be repeated. Standard library
// and the dependency-free record header only, so tests/runtime/test_numeric_tripwire.cpp builds
// without CUDA.

#include "ninfer/ops/numeric_tripwire_record.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <optional>
#include <string>

namespace ninfer::models::qwen3_5::execution {

// Where a probe sits. A site id is `probe << 16 | index`; ids exist only inside one build and one
// process, so nothing may store or compare them across runs.
enum class TripwireProbe : std::uint32_t {
    GdnOutput   = 1, // a linear-attention layer's output (the recurrent state read by the new token)
    FinalHidden = 2, // the final normalised hidden state
    FinalLogits = 3, // the output head's scores
};

[[nodiscard]] constexpr std::uint32_t tripwire_site(TripwireProbe probe,
                                                    std::uint32_t index = 0) noexcept {
    return (static_cast<std::uint32_t>(probe) << 16U) | (index & 0xffffU);
}

// Only every 16th linear-attention layer is probed (0, 16, 32 of a 48-layer 27B): a NaN in any
// layer's state reaches the next probe or the final hidden state within the same step, and each
// probe costs one small kernel.
inline constexpr std::uint32_t kTripwireGdnProbeStride = 16;

[[nodiscard]] constexpr bool tripwire_probes_gdn_layer(std::uint32_t gdn_index) noexcept {
    return gdn_index % kTripwireGdnProbeStride == 0;
}

// Tensor extents the final probes need to turn a flat element index into "token T of column C".
struct TripwireGeometry {
    std::uint32_t vocab_size  = 0;
    std::uint32_t hidden_size = 0;
};

[[nodiscard]] inline std::string describe_tripwire_site(std::uint32_t site) {
    const std::uint32_t index = site & 0xffffU;
    switch (static_cast<TripwireProbe>(site >> 16U)) {
    case TripwireProbe::GdnOutput:
        return "GDN layer output (the state itself is not read back; a bad state shows here in "
               "the same step), linear-attention layer " +
               std::to_string(index) + " counting from 0";
    case TripwireProbe::FinalHidden:
        return "final hidden state (after the last norm, before the output head)";
    case TripwireProbe::FinalLogits:
        return "final logits (the output head's score for every vocabulary token)";
    }
    char buffer[40];
    std::snprintf(buffer, sizeof(buffer), "unknown probe site 0x%08x", site);
    return buffer;
}

[[nodiscard]] inline std::string group_thousands(std::uint64_t value) {
    std::string digits = std::to_string(value);
    std::string out;
    for (std::size_t i = 0; i < digits.size(); ++i) {
        if (i != 0 && (digits.size() - i) % 3 == 0) { out += ','; }
        out += digits[i];
    }
    return out;
}

[[nodiscard]] inline std::string format_tripwire_value(float value, float abs_limit) {
    if (std::isnan(value)) { return "NaN"; }
    if (std::isinf(value)) { return value > 0.0F ? "+Inf" : "-Inf"; }
    char buffer[96];
    std::snprintf(buffer, sizeof(buffer), "%.9g (above the limit %.9g)",
                  static_cast<double>(value), static_cast<double>(abs_limit));
    return buffer;
}

[[nodiscard]] inline std::string describe_tripwire_element(std::uint32_t site, std::uint32_t index,
                                                           const TripwireGeometry& geometry) {
    std::string out = "element " + group_thousands(index);
    const auto probe = static_cast<TripwireProbe>(site >> 16U);
    if (probe == TripwireProbe::FinalLogits && geometry.vocab_size != 0) {
        out += " (token id " + std::to_string(index % geometry.vocab_size) + ", batch column " +
               std::to_string(index / geometry.vocab_size) + ")";
    } else if (probe == TripwireProbe::FinalHidden && geometry.hidden_size != 0) {
        out += " (channel " + std::to_string(index % geometry.hidden_size) + ", batch column " +
               std::to_string(index / geometry.hidden_size) + ")";
    }
    return out;
}

// One operational log line, pipe-separated like the other records (docs/maintainer/logging.md).
// `suppressed_rounds` counts earlier flagged rounds that were not logged.
[[nodiscard]] inline std::string
format_tripwire_message(const ops::NumericTripwireFinding& finding, float abs_limit,
                        const TripwireGeometry& geometry, std::uint32_t suppressed_rounds = 0) {
    std::string out = "numeric tripwire | first bad value at " + describe_tripwire_site(finding.site) +
                      " | " + format_tripwire_value(finding.value, abs_limit) + " in " +
                      describe_tripwire_element(finding.site, finding.element_index, geometry) +
                      " | " + group_thousands(finding.offending_count) +
                      " bad element(s) this round, counting every probe";
    if (suppressed_rounds != 0) {
        out += " | " + group_thousands(suppressed_rounds) + " earlier round(s) not logged";
    }
    out += " | generation continues, output may be damaged";
    return out;
}

// A NaN in a recurrent state stays there, so every later step would report it. Log a new site at
// once and the same site again only after kRepeatEveryRounds flagged rounds.
class TripwireReportThrottle {
public:
    static constexpr std::uint32_t kRepeatEveryRounds = 256;

    // Called once per flagged round. Returns the number of flagged rounds skipped since the last
    // line when this round should be logged, and nothing when it should stay quiet.
    [[nodiscard]] std::optional<std::uint32_t> admit(std::uint32_t site) noexcept {
        if (!last_site_ || *last_site_ != site || skipped_ + 1U >= kRepeatEveryRounds) {
            const std::uint32_t skipped = skipped_;
            last_site_                  = site;
            skipped_                    = 0;
            return skipped;
        }
        ++skipped_;
        return std::nullopt;
    }

private:
    std::optional<std::uint32_t> last_site_;
    std::uint32_t skipped_ = 0;
};

} // namespace ninfer::models::qwen3_5::execution
