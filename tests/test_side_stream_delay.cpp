// The test-only side-stream delay's variable (core/side_stream_delay.h; CPU only, the parser is
// header only): unset or empty is off, decimal microseconds up to the cap are taken, anything else
// is refused naming the variable.
#include "core/side_stream_delay.h"

#include <cstdio>
#include <stdexcept>
#include <string>

namespace {

int failures = 0;

void check(bool condition, const char* message) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", message);
        ++failures;
    }
}

bool refused(const char* text) {
    try {
        (void)ninfer::parse_side_stream_delay_us(text);
    } catch (const std::invalid_argument& error) {
        return std::string(error.what()).find(ninfer::kSideStreamDelayVariable) != std::string::npos;
    }
    return false;
}

} // namespace

int main() {
    using ninfer::parse_side_stream_delay_us;
    check(parse_side_stream_delay_us(nullptr) == 0, "unset is off");
    check(parse_side_stream_delay_us("") == 0, "empty is off");
    check(parse_side_stream_delay_us("0") == 0, "0 is off");
    check(parse_side_stream_delay_us("2000") == 2000, "microseconds are taken");
    check(parse_side_stream_delay_us("100000") == ninfer::kMaxSideStreamDelayUs, "the cap is taken");
    check(refused("100001"), "past the cap is refused naming the variable");
    check(refused("99999999999999999999999"), "an overflowing value is refused");
    check(refused("-1") && refused("2ms") && refused(" 5") && refused("1.5"), "non-digits are refused");
    check(ninfer::kGateSideStreamDelayUs > 0 && ninfer::kGateSideStreamDelayUs <= ninfer::kMaxSideStreamDelayUs,
          "the gates' delay is within the cap");
    if (failures != 0) { return 1; }
    std::puts("ninfer_side_stream_delay_test: ok");
    return 0;
}
