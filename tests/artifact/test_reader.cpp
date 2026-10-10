#include "artifact/binder.h"
#include "artifact/reader.h"
#include "artifact/fixture.h"
#include "artifact/formats.h"
#include "artifact/layouts.h"
#include "core/weight_view.h"

#include <algorithm>
#include <array>
#include <bit>
#include <iostream>
#include <string>

namespace {

using namespace ninfer;
using namespace ninfer::artifact;
using namespace ninfer::test::artifact_fixture;

void file_set_and_bindings() {
    Fixture fixture;
    fixture.write(true);
    Reader reader(fixture.entry);
    const auto bytes = reader.read_object(reader.find("q5"));
    require(bytes.size() == 528 &&
                std::equal(bytes.begin(), bytes.end(), fixture.payload.begin() + 256),
            "cross-file parent bytes changed");
    require(reader.read_range(1344, 0).empty(), "empty terminal range failed");
    rejects([&] { (void)reader.read_range(1344, 1); }, "out-of-range read accepted");
    rejects([&] { (void)reader.geometry(reader.find("unused")); },
            "unknown selected encoding accepted");
    rejects([&] { (void)reader.read_range(1280, 64); }, "required missing part accepted");

    Binder binder(reader);
    const auto full = binder.parameter("matrix", {2, 130});
    const auto row  = binder.parameter("row", {1, 130});
    require(full.binding.parts[0].object == row.binding.parts[0].object, "shared parent lost");
    rejects([&] { (void)binder.parameter("matrix", {1, 260}); },
            "whole-object shape was silently reshaped");
    const auto reshaped = binder.parameter("row", {130});
    require(reshaped.binding.elements == 130, "Part reshape changed logical elements");
    const auto context =
        binder.values(binder.use("row", "context").auxiliaries.at("input_divisor"));
    const auto query = binder.values(binder.use("row", "query").auxiliaries.at("input_divisor"));
    require(std::bit_cast<std::uint32_t>(query.scalar_f32()) == 0x40000000 &&
                std::bit_cast<std::uint32_t>(context.scalar_f32()) == 0x40400000,
            "Use values were mixed");
    require(binder.use("row", "query").activation_policy == ActivationPolicy::AllowA4 &&
                binder.use("row", "context").activation_policy == ActivationPolicy::AllowA8,
            "Use policy changed");
    (void)binder.resource("text", "tokenizer.json");
    auto plan = std::move(binder).finish();
    require(plan.device_objects.size() == 1 && plan.device_capacity(0) == 528 &&
                plan.host_objects.size() == 1,
            "selected parent deduplication or residency failed");
}

void geometry_and_views() {
    const std::array<std::uint16_t, 12> direct_words{
        0, 0x8000, 0x3f80, 0x4000, 0x4040, 0x4080, 0x40a0, 0x40c0, 0x40e0, 0x4100, 0x4110, 0x4120};
    const auto* direct_bytes = reinterpret_cast<const std::byte*>(direct_words.data());
    const WeightParent direct_parent{weight_geometry(QType::BF16, QuantLayout::Contiguous,
                                                     std::array<std::uint64_t, 3>{2, 2, 3}),
                                     direct_bytes};
    const WeightView direct_view{{2, 3}, {{&direct_parent, 3, 6}, {&direct_parent, 6, 9}}};
    const auto direct = native_weight(direct_view);
    require(direct.n == 2 && direct.k == 3 && direct.padded_shape[1] == 3 &&
                direct.qdata == direct_bytes + 6 && direct.payload == direct_bytes &&
                direct.payload_bytes == sizeof(direct_words),
            "Direct reshape lost its logical shape, element offset or owning parent");
    Fixture fixture;
    WeightParent parent{weight_geometry(QType::Q5_G64_FP16, QuantLayout::RowSplit,
                                        std::array<std::uint64_t, 2>{2, 130}),
                        fixture.payload.data() + 256};
    const WeightView row{{1, 130}, {{&parent, 130, 260}}};
    const auto planes = weight_row_planes(row.parts.front());
    require(parent.geometry.bytes == 528 && planes.codes == parent.data + 128 &&
                planes.high == parent.data + 288 && planes.scales == parent.data + 520,
            "Q5 view was interpreted as a standalone payload");
    const auto weight = native_weight(row);
    rejects<std::invalid_argument>([&] { (void)native_weight(WeightView{{2, 65}, row.parts}); },
                                   "quantized view changed K without preserving encoding groups");
    require(weight.qdata == planes.codes && weight.qhigh == planes.high &&
                weight.scales == planes.scales,
            "native RowSplit view lost its planes");
    require(std::to_integer<int>(planes.codes[0]) == 0x52 &&
                std::to_integer<int>(planes.scales[1]) == 0x40,
            "row view selected the wrong represented values");

    std::vector<std::byte> fp8(16896);
    WeightParent fp8_parent{weight_geometry(QType::FP8_E4M3FN_ROW_BF16, QuantLayout::RowScale,
                                            std::array<std::uint64_t, 2>{256, 64}),
                            fp8.data()};
    WeightView fp8_rows{{128, 64}, {{&fp8_parent, 0, 128 * 64}}};
    require(weight_row_planes(fp8_rows.parts.front()).scales == fp8.data() + 16384,
            "FP8 view recomputed the parent scale base");
    rejects<std::invalid_argument>([&] { (void)native_weight(fp8_rows); },
                                   "complete-parent ABI accepted an FP8 submatrix");

    std::vector<std::byte> nvfp4(4612);
    WeightParent nv_parent{weight_geometry(QType::NVFP4, QuantLayout::BlockScaleK16M128x4,
                                           std::array<std::uint64_t, 2>{128, 64}),
                           nvfp4.data(), 2.0F};
    require(weight_scale_offset(nv_parent.geometry, 31, 0) == 4096 + 496 &&
                weight_scale_offset(nv_parent.geometry, 32, 0) == 4096 + 4,
            "NVFP4 parent swizzle origin was lost");
    const auto nv_rows = weight_row_planes({&nv_parent, 31 * 64, 33 * 64});
    require(nv_rows.swizzled_scales && nv_rows.row_begin == 31 &&
                nv_rows.scales == nvfp4.data() + 4096,
            "NVFP4 submatrix lost its parent coordinates");
    const WeightView complete{{128, 64}, {{&nv_parent, 0, 128 * 64}}};
    const auto query   = native_weight(complete, 3.0F);
    const auto context = native_weight(complete, 4.0F);
    require(query.weight_scale_divisor == 2 && query.input_scale_divisor == 3 &&
                context.input_scale_divisor == 4 && nv_parent.weight_scale_divisor == 2,
            "per-use native parameters changed the parent");
}

/// GGML block geometry, at the real shapes it must hold: the dense matrices, the 51.2 G-value
// n-gram table (whose element count and byte offsets overflow 32 bits) and the expert records.
void ggml_geometry() {
    constexpr std::array names = {"ggml_q8_0",  "ggml_q6_k",   "ggml_iq2_xxs",
                                  "ggml_iq4_nl", "ggml_iq3_s", "ggml_iq2_s",
                                  "ggml_iq4_xs", "ggml_iq1_m", "ggml_q2_0"};
    for (const auto* name : names) {
        const auto format = parse_format(name);
        require(is_ggml_block(format) && !is_ggml_record(format) && format_name(format) == name,
                "GGML format name does not round-trip");
    }
    require(parse_layout("ggml_blocks_v1") == QuantLayout::GgmlBlocks &&
                layout_name(QuantLayout::GgmlBlocks) == "ggml_blocks_v1",
            "GGML layout name does not round-trip");
    rejects([] { (void)parse_layout("ggml_rows_page4k_v1"); },
            "the superseded paged layout is still accepted");

    // A GGML block matrix: 512 x 1280 rows of 2560 IQ2_S values.
    const auto bank = weight_geometry(QType::GGML_IQ2_S, QuantLayout::GgmlBlocks,
                                      std::array<std::uint64_t, 2>{512 * 1280, 2560});
    require(bank.bytes == 537395200 && bank.code_bytes_per_row == 820 &&
                ggml_row_offset(bank, 1280) == 1049600 && bank.alignment == 256 &&
                bank.group_size == 256 && bank.scale_bytes == 0,
            "GGML block geometry differs from 820 B rows");
    require(weight_geometry(QType::GGML_IQ2_S, QuantLayout::GgmlBlocks,
                            std::array<std::uint64_t, 3>{512, 1280, 2560})
                    .bytes == bank.bytes &&
                weight_geometry(QType::GGML_Q6_K, QuantLayout::GgmlBlocks,
                                std::array<std::uint64_t, 1>{256})
                        .bytes == 210,
            "GGML block geometry depends on how the leading axes are split");

    // The n-gram (PLE) table as one matrix: 320,001,536 rows of 160 IQ4_NL values.
    const std::array<std::uint64_t, 2> ple{320001536, 160};
    const auto flat = weight_geometry(QType::GGML_IQ4_NL, QuantLayout::GgmlBlocks, ple);
    require(flat.elements == 51200245760ULL && flat.bytes == 28800138240ULL &&
                ggml_row_offset(flat, 320001535) == 28800138150ULL,
            "PLE geometry overflowed");
    rejects<std::invalid_argument>([&] { (void)ggml_row_offset(flat, 320001536); },
                                   "a row past the table was addressed");

    // The native Weight ABI is int32 per axis: the whole table still fits, a 2^31-row matrix
    // is refused rather than truncated.
    std::array<std::byte, 90> row{};
    const WeightParent table{flat, row.data()};
    const auto native = native_weight(WeightView{{320001536, 160}, {{&table, 0, flat.elements}}});
    require(native.n == 320001536 && native.k == 160 && native.qdata == row.data() &&
                native.scales == nullptr && native.group == 32 &&
                native.qtype == QType::GGML_IQ4_NL && native.layout == QuantLayout::GgmlBlocks,
            "GGML native Weight lost its rows or block size");
    const WeightParent wide{weight_geometry(QType::GGML_Q8_0, QuantLayout::GgmlBlocks,
                                            std::array<std::uint64_t, 2>{1ULL << 31, 32}),
                            row.data()};
    rejects<std::invalid_argument>(
        [&] { (void)native_weight(WeightView{{1ULL << 31, 32}, {{&wide, 0, 1ULL << 36}}}); },
        "a row count beyond int32 reached the native Weight");
    rejects<std::invalid_argument>([&] { (void)weight_scale_offset(flat, 0, 0); },
                                   "a GGML parent reported a separate scale plane");

    // Expert records of one qwen4exp layer: [512 experts, hidden 2560, intermediate 640]. Every
    // part is a multiple of 256 B, so records are back to back (spec section 2.1 byte counts).
    struct RecordCase {
        const char* name;
        QType format;
        QType gate_up;
        std::uint64_t gate_bytes;
        std::uint64_t record_bytes;
    };
    constexpr std::array records = {
        RecordCase{"ggml_rec_iq2_s_q2_0", QType::GGML_REC_IQ2_S_Q2_0, QType::GGML_IQ2_S, 524800,
                   1510400},
        RecordCase{"ggml_rec_iq2_xxs_q2_0", QType::GGML_REC_IQ2_XXS_Q2_0, QType::GGML_IQ2_XXS,
                   422400, 1305600},
        RecordCase{"ggml_rec_iq1_m_q2_0", QType::GGML_REC_IQ1_M_Q2_0, QType::GGML_IQ1_M, 358400,
                   1177600},
    };
    require(parse_layout("ggml_expert_record_v1") == QuantLayout::GgmlExpertRecord &&
                layout_name(QuantLayout::GgmlExpertRecord) == "ggml_expert_record_v1",
            "expert record layout name does not round-trip");
    const std::array<std::uint64_t, 3> layer{512, 2560, 640};
    std::uint64_t model_bytes = 0;
    for (const auto& item : records) {
        require(parse_format(item.name) == item.format && format_name(item.format) == item.name &&
                    is_ggml_record(item.format) && !is_ggml_block(item.format) &&
                    ggml_record_parts(item.format).gate_up == item.gate_up &&
                    ggml_record_parts(item.format).down == QType::GGML_Q2_0,
                "expert record format does not round-trip");
        const auto g = weight_geometry(item.format, QuantLayout::GgmlExpertRecord, layer);
        require(g.record_up_offset == item.gate_bytes &&
                    g.record_down_offset == 2 * item.gate_bytes &&
                    g.record_bytes == item.record_bytes && g.record_stride == item.record_bytes &&
                    g.bytes == 512 * item.record_bytes && g.alignment == 256 &&
                    g.elements == 512ULL * 2560 * 640,
                (std::string(item.name) + ": record geometry differs from the GGUF").c_str());
        const auto down = ggml_record_part(g, 511, ExpertPart::Down);
        require(down.format == QType::GGML_Q2_0 && down.rows == 2560 && down.k == 640 &&
                    down.row_bytes == 180 &&
                    down.offset == 511 * item.record_bytes + 2 * item.gate_bytes &&
                    down.offset + down.rows * down.row_bytes == g.bytes,
                "the last expert's down part is not the end of the bank");
        const auto up = ggml_record_part(g, 3, ExpertPart::Up);
        require(up.format == item.gate_up && up.rows == 640 && up.k == 2560 &&
                    up.row_bytes * 640 == item.gate_bytes &&
                    up.offset == 3 * item.record_bytes + item.gate_bytes,
                "the up part is not after the gate part");
        rejects<std::invalid_argument>([&] { (void)ggml_record_part(g, 512, ExpertPart::Gate); },
                                       "an expert past the bank was addressed");
        rejects<std::invalid_argument>([&] { (void)weight_scale_offset(g, 0, 0); },
                                       "an expert record reported a separate scale plane");
        const WeightParent parent{g, row.data()};
        rejects<std::invalid_argument>(
            [&] { (void)native_weight(WeightView{{512, 2560, 640}, {{&parent, 0, g.elements}}}); },
            "an expert bank reached the native Weight");
        model_bytes += (item.format == QType::GGML_REC_IQ2_S_Q2_0     ? 34
                        : item.format == QType::GGML_REC_IQ2_XXS_Q2_0 ? 11
                                                                       : 3) *
                       g.bytes;
    }
    require(model_bytes == 35454976000ULL, "the 48 banks differ from the GGUF's expert bytes");
    // Parts that are not 256 B multiples are padded: 64 IQ2_XXS rows of 66 B are 4224 B.
    const auto padded = weight_geometry(QType::GGML_REC_IQ2_XXS_Q2_0, QuantLayout::GgmlExpertRecord,
                                        std::array<std::uint64_t, 3>{3, 256, 64});
    require(padded.record_up_offset == 4352 && padded.record_down_offset == 8704 &&
                padded.record_bytes == 8704 + 256 * 18 && padded.record_stride == 13312 &&
                padded.bytes == 3 * 13312,
            "record parts are not on 256-byte boundaries");

    const auto invalid = [](QType format, QuantLayout layout, std::vector<std::uint64_t> shape) {
        rejects<std::invalid_argument>([&] { (void)weight_geometry(format, layout, shape); },
                                       "invalid GGML geometry accepted");
    };
    invalid(QType::GGML_Q6_K, QuantLayout::GgmlBlocks, {4, 128}); // K % 256
    invalid(QType::GGML_Q8_0, QuantLayout::RowSplit, {2, 128});
    invalid(QType::GGML_Q8_0, QuantLayout::Contiguous, {2, 32});
    invalid(QType::BF16, QuantLayout::GgmlBlocks, {2, 32});
    invalid(QType::GGML_REC_IQ2_S_Q2_0, QuantLayout::GgmlBlocks, {2, 256, 64});
    invalid(QType::GGML_IQ2_S, QuantLayout::GgmlExpertRecord, {2, 256, 64});
    invalid(QType::GGML_REC_IQ2_S_Q2_0, QuantLayout::GgmlExpertRecord, {512, 2560});   // rank 3
    invalid(QType::GGML_REC_IQ2_S_Q2_0, QuantLayout::GgmlExpertRecord, {2, 640, 640});  // H % 256
    invalid(QType::GGML_REC_IQ1_M_Q2_0, QuantLayout::GgmlExpertRecord, {2, 2560, 96}); // I % 64
    invalid(QType::GGML_REC_IQ1_M_Q2_0, QuantLayout::RowSplit, {2, 2560});

    // The object record must carry the layout's size.
    TensorObject object{"bank", {2, 256, 64}, "ggml_rec_iq2_xxs_q2_0", "ggml_expert_record_v1",
                        512, 2 * 13312};
    require(describe_tensor(object).record_stride == 13312, "record object was not described");
    object.bytes = 2 * 13312 - 256;
    rejects([&] { (void)describe_tensor(object); }, "a bank without its last record's tail accepted");
}

void invalid_directories() {
    Fixture fixture;
    const auto bad = [&](auto mutate) {
        Json root = fixture.root;
        mutate(root);
        rejects([&] { (void)parse_directory(root, "model.ninfer"); }, "invalid directory accepted");
    };
    bad([](Json& root) { root["bindings"]["row"]["parts"][0]["range"] = {129, 261}; });
    bad([](Json& root) { root["uses"].push_back(root["uses"][0]); });
    bad([](Json& root) { root["objects"][1]["offset"] = 4; });
    bad([](Json& root) { root["objects"][2]["id"] = "q5"; });
    bad([](Json& root) { root["files"][1]["path"] = "../other"; });
    bad([](Json& root) { root["files"][1]["payload_bytes"] = 1.5; });
    bad([](Json& root) { root["components"]["text"]["resources"]["tokenizer.json"] = "q5"; });
    rejects([] { (void)parse_json("{\"a\":{\"x\":1,\"x\":2}}", "duplicate"); },
            "duplicate JSON key accepted");
    fixture.write();
    {
        std::fstream file(fixture.entry, std::ios::binary | std::ios::in | std::ios::out);
        file.seekp(7);
        file.put(2);
    }
    rejects([&] { Reader reader(fixture.entry); }, "v2 magic accepted");
    fixture.write();
    {
        std::fstream file(fixture.directory / "weights-second.bin",
                          std::ios::binary | std::ios::in | std::ios::out);
        file.seekp(16);
        file.put(0);
    }
    Reader wrong_part(fixture.entry);
    rejects([&] { (void)wrong_part.read_object(wrong_part.find("q5")); },
            "foreign continuation accepted");
    fixture.write();
    Reader shortened(fixture.entry);
    std::filesystem::resize_file(fixture.entry, std::filesystem::file_size(fixture.entry) - 1);
    rejects([&] { (void)shortened.read_range(499, 1); }, "premature EOF accepted");
}

} // namespace

int main(int argc, char** argv) {
    try {
        file_set_and_bindings();
        geometry_and_views();
        ggml_geometry();
        invalid_directories();
        // Optional production-writer fixture or explicitly selected real artifact.
        if (argc == 2) {
            Reader reader(argv[1]);
            std::uint64_t bytes = 0;
            for (std::size_t i = 0; i < reader.directory().objects.size(); ++i) {
                reader.validate_object({i});
                bytes =
                    checked_add(bytes, object_bytes(reader.directory().objects[i]), "object bytes");
            }
            std::cout << "v3 objects=" << reader.directory().objects.size() << " bytes=" << bytes
                      << '\n';
        }
        std::cout << "artifact reader and binding checks passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
