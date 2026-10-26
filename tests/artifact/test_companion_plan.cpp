// Device companions (artifact/companion.h), host only: the 2-bit head copy --lm-head-two-stage
// derives from a row-split Q8 output head is a planned term of the weight arena.
//   - companion_is_planned: the copy gets a placement of its own right after its source, sized
//     proxy_bytes(N, K), and the arena grows by exactly that term (plus alignment);
//   - companion_refusals: no device demand, a non-Q8 source, a transcoded source and a source that
//     keeps Host bytes are all refused;
//   - companion_fit_refusal: startup names --lm-head-two-stage only when the weights fit without
//     the copy but not with it;
//   - companion_build: build_companion writes what the codec's pack_proxy_rows writes.

#include "artifact/binder.h"
#include "artifact/companion.h"
#include "artifact/fixture.h"
#include "artifact/reader.h"
#include "core/weight_view.h"
#include "ops/two_stage_head/proxy_codec.h"

#include <cstdint>
#include <iostream>
#include <random>
#include <string>
#include <vector>

namespace {

using namespace ninfer;
using namespace ninfer::artifact;
using namespace ninfer::test::artifact_fixture;

void require(bool condition, const std::string& message) {
    ninfer::test::artifact_fixture::require(condition, message.c_str());
}

constexpr std::uint64_t kRows     = 16;
constexpr std::uint64_t kColumns  = 128;
constexpr std::uint64_t kHeadAt   = 1536; // logical payload offset of the head object

std::uint64_t head_bytes() {
    const std::vector<std::uint64_t> shape{kRows, kColumns};
    return weight_geometry(QType::Q8_G32_FP16, QuantLayout::RowSplit, shape).bytes;
}

// The shared fixture plus a fourth file holding a [16,128] row-split Q8 head.
void add_head(Fixture& fixture) {
    const std::uint64_t bytes = head_bytes();
    fixture.root["objects"].push_back({{"id", "head"},
                                       {"kind", "tensor"},
                                       {"shape", {kRows, kColumns}},
                                       {"format", "q8_g32_fp16"},
                                       {"layout", "row_split_k128_v1"},
                                       {"offset", kHeadAt},
                                       {"bytes", bytes}});
    fixture.root["bindings"]["head"] = {{"object", "head"}};
    fixture.root["files"].push_back({{"path", "head.bin"}, {"payload_bytes", kHeadAt - 1344 + bytes}});
    fixture.payload.resize(static_cast<std::size_t>(kHeadAt + bytes));
    std::mt19937 rng(38U);
    for (std::uint64_t i = 0; i < kRows * kColumns; ++i) {
        fixture.payload[kHeadAt + i] = static_cast<std::byte>(rng() & 0xFFU);
    }
    const auto geometry = weight_geometry(QType::Q8_G32_FP16, QuantLayout::RowSplit,
                                          std::vector<std::uint64_t>{kRows, kColumns});
    for (std::uint64_t g = 0; g < kRows * kColumns / 32; ++g) {
        const std::uint16_t scale = static_cast<std::uint16_t>(0x2000U + (rng() & 0x0FFFU)); // small positive
        put_word(fixture.payload, kHeadAt + geometry.scale_offset + 2 * g, scale, 2);
    }
}

void companion_is_planned() {
    Fixture fixture;
    add_head(fixture);
    fixture.write();
    Reader reader(fixture.entry);
    const auto head = reader.find("head");
    std::uint64_t without = 0;
    {
        Binder binder(reader);
        (void)binder.parameter("head", {kRows, kColumns});
        without = std::move(binder).finish().device_capacity(0);
    }
    Binder binder(reader);
    (void)binder.parameter("head", {kRows, kColumns});
    binder.companion_device(head, Companion::TwoBitHeadProxy);
    binder.companion_device(head, Companion::TwoBitHeadProxy); // a repeat is accepted
    const auto plan        = std::move(binder).finish();
    const std::uint64_t copy = ops::proxy_head::proxy_bytes(kRows, kColumns);
    require(copy == kRows * kColumns / 4 + kRows * kColumns / 32 * 2, "proxy size formula changed");
    require(companion_bytes(Companion::TwoBitHeadProxy, std::vector<std::uint64_t>{kRows, kColumns}) == copy,
            "companion_bytes differs from the codec");
    require(plan.device_objects.size() == 2, "the companion has no placement of its own");
    const auto& object    = plan.device_objects[0];
    const auto& companion = plan.device_objects[1];
    require(object.object == head && !object.companion && object.bytes == head_bytes(),
            "the source placement changed");
    require(companion.object == head && companion.companion == Companion::TwoBitHeadProxy &&
                companion.bytes == copy && companion.offset % 256 == 0 &&
                companion.offset >= object.offset + object.bytes && companion.rank == 0,
            "the companion is not placed right after its source");
    const std::uint64_t expected = (without + 255) / 256 * 256 + copy;
    require(plan.device_capacity(0) == expected,
            "the arena did not grow by exactly the companion term: " +
                std::to_string(plan.device_capacity(0)) + " vs " + std::to_string(expected));
    require(planned_companion_bytes(plan) == copy && planned_companion_bytes(plan, 1) == 0,
            "planned_companion_bytes miscounts");
}

void companion_refusals() {
    Fixture fixture;
    add_head(fixture);
    fixture.write();
    Reader reader(fixture.entry);
    const auto head = reader.find("head");
    {
        Binder binder(reader);
        rejects([&] { binder.companion_device(head, Companion::TwoBitHeadProxy); },
                "a companion without a device demand was accepted");
    }
    {
        Binder binder(reader);
        (void)binder.parameter("matrix", {2, 130});
        rejects([&] { binder.companion_device(reader.find("q5"), Companion::TwoBitHeadProxy); },
                "a companion of a Q5 matrix was accepted");
    }
    {
        Binder binder(reader);
        (void)binder.parameter("head", {kRows, kColumns});
        binder.transcode_device(head, QType::Q6_G64_FP16);
        rejects([&] { binder.companion_device(head, Companion::TwoBitHeadProxy); },
                "a companion of a transcoded head was accepted");
    }
    {
        Binder binder(reader);
        (void)binder.parameter("head", {kRows, kColumns});
        binder.companion_device(head, Companion::TwoBitHeadProxy);
        rejects([&] { binder.transcode_device(head, QType::Q4_G64_FP16); },
                "transcoding a head with a companion was accepted");
    }
    {
        Binder binder(reader);
        (void)binder.parameter("head", {kRows, kColumns});
        binder.companion_device(head, Companion::TwoBitHeadProxy);
        (void)binder.host_object(head);
        rejects([&] { (void)std::move(binder).finish(); },
                "a head with a companion that also keeps Host bytes was accepted");
    }
}

void companion_fit_refusal_names_the_flag() {
    constexpr std::uint64_t gib = 1ULL << 30;
    require(companion_fit_refusal("--lm-head-two-stage", 20 * gib, gib / 2, 21 * gib).empty(),
            "a plan that fits was refused");
    require(companion_fit_refusal("--lm-head-two-stage", 20 * gib, gib / 2, 10 * gib).empty(),
            "a plan that does not fit even without the copy blamed the flag");
    require(companion_fit_refusal("--lm-head-two-stage", 20 * gib, 0, 19 * gib).empty(),
            "a plan without a companion blamed the flag");
    const std::string why = companion_fit_refusal("--lm-head-two-stage", 20 * gib, gib, 19 * gib + 1);
    require(why.find("--lm-head-two-stage") != std::string::npos &&
                why.find("1024 MiB") != std::string::npos && why.find("Start without") != std::string::npos,
            "the refusal does not name the flag, the copy's size and what to do: " + why);
    const std::string advice = companion_memory_advice("--lm-head-two-stage", 397U << 20);
    require(advice.find("397 MiB") != std::string::npos &&
                advice.find("--lm-head-two-stage") != std::string::npos,
            "the memory advice does not name the flag and the size: " + advice);
}

void companion_build() {
    Fixture fixture;
    add_head(fixture);
    fixture.write();
    Reader reader(fixture.entry);
    const auto source = reader.read_object(reader.find("head"));
    const std::vector<std::uint64_t> shape{kRows, kColumns};
    std::vector<std::byte> built(companion_bytes(Companion::TwoBitHeadProxy, shape));
    build_companion(Companion::TwoBitHeadProxy, shape, source, built);
    const auto geometry = weight_geometry(QType::Q8_G32_FP16, QuantLayout::RowSplit, shape);
    std::vector<std::byte> direct(built.size(), std::byte{0});
    ops::proxy_head::pack_proxy_rows(
        std::span<const std::byte>(source).subspan(0, kRows * kColumns),
        std::span<const std::byte>(source).subspan(geometry.scale_offset, kRows * kColumns / 32 * 2),
        kRows, kColumns, direct, 0, kRows);
    require(built == direct, "build_companion differs from pack_proxy_rows");
    std::vector<std::byte> wrong(built.size() - 1);
    rejects([&] { build_companion(Companion::TwoBitHeadProxy, shape, source, wrong); },
            "a destination of the wrong size was accepted");
}

} // namespace

int main() {
    try {
        companion_is_planned();
        companion_refusals();
        companion_fit_refusal_names_the_flag();
        companion_build();
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
    std::cout << "artifact companions: all checks passed\n";
    return 0;
}
