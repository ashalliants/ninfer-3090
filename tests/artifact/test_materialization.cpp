#include "artifact/binder.h"
#include "artifact/fixture.h"
#include "artifact/formats.h"
#include "artifact/views.h"
#include "core/device.h"
#include "core/evictable_weight_pool.h"
#include "ops/ggml_blocks_decode.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cstring>
#include <iostream>
#include <map>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <vector>

namespace ninfer::test {
void materialization_cuda_errors(DeviceContext& device);
}

namespace {

using namespace ninfer;
using namespace ninfer::artifact;
using namespace ninfer::test::artifact_fixture;

void materialization(DeviceContext& device) {
    Fixture fixture;
    fixture.write(true);
    std::optional<MaterializedArtifact> backing;
    ParameterReference row;
    ObjectHandle quantized;
    ObjectHandle divisors;
    ObjectHandle resource;
    const std::byte* original_host = nullptr;
    {
        Reader reader(fixture.entry);
        Binder binder(reader);
        (void)binder.parameter("matrix", {2, 130});
        row       = binder.parameter("row", {1, 130});
        quantized = reader.find("q5");
        divisors  = reader.find("divisors");
        (void)binder.parameter("values", {2}, Residency::Host, QType::FP32);
        binder.require_device(divisors);
        resource      = binder.resource("text", "tokenizer.json");
        original_host = binder.host_object(divisors).data();
        backing.emplace(materialize(reader, std::move(binder).finish(), device));
    }
    const auto& stats = backing->stats();
    require(stats.device_object_count == 2 && stats.h2d_bytes == 536 &&
                stats.device_capacity_bytes == 776 && stats.retained_host_bytes == 13,
            "resident parents were duplicated or file gaps were allocated");
    require(backing->host_bytes(divisors).data() == original_host,
            "retained Host bytes moved after creating borrowed resource views");
    require(backing->host_bytes(resource).size() == 5,
            "Host resource was lost after Reader destruction");
    const auto view   = bind_view(row, *backing);
    const auto native = native_weight(view);
    require(native.n == 1 && native.k == 130 &&
                native.payload == backing->device_parent(quantized).data,
            "row view no longer refers to its owning parent");
    std::vector<std::byte> downloaded(528);
    CUDA_CHECK(cudaMemcpy(downloaded.data(), backing->device_parent(quantized).data,
                          downloaded.size(), cudaMemcpyDeviceToHost));
    require(std::equal(downloaded.begin(), downloaded.end(), fixture.payload.begin() + 256),
            "cross-file upload changed encoded parent bytes");
    std::array<std::byte, 8> values{};
    CUDA_CHECK(cudaMemcpy(values.data(), backing->device_parent(divisors).data, values.size(),
                          cudaMemcpyDeviceToHost));
    require(std::equal(values.begin(), values.end(), backing->host_bytes(divisors).begin()),
            "Host/device demand did not retain identical bytes");
    std::array<std::byte, 1> code{};
    CUDA_CHECK(cudaMemcpy(code.data(), native.qdata, 1, cudaMemcpyDeviceToHost));
    require(code[0] == std::byte{0x52}, "native row pointer addressed a different row");
}

// Overlay Vision residency plans: an evict-ranked object lands in a chunk-aligned arena tail that
// an eviction pool can borrow whole, and a pinned object lands in the page-locked block instead of
// device memory.
void overlay_placement(DeviceContext& device) {
    constexpr std::uint64_t kChunk = EvictableWeightPool::kChunkBytes;
    Fixture fixture;
    fixture.write(true);
    {
        Reader reader(fixture.entry);
        Binder binder(reader);
        const auto matrix   = binder.parameter("matrix", {2, 130});
        const auto divisors = reader.find("divisors");
        binder.require_device(divisors);
        binder.evict_device(divisors, 700);
        const auto plan = std::move(binder).finish(kChunk);
        require(plan.device_objects.size() == 2 && plan.device_objects.front().offset == 0 &&
                    plan.evictable_tail_offset == kChunk && plan.evictable_tail_bytes == 8 &&
                    plan.device_objects.back().offset == kChunk &&
                    plan.device_capacity(0) == kChunk + 8,
                "evict-ranked object was not planned into a chunk-aligned arena tail");
        if (!EvictableWeightPool::supported(device)) {
            std::cout << "note: VMM unsupported, overlay transaction not exercised\n";
        } else {
            auto pool = std::make_unique<EvictableWeightPool>(
                device, EvictableWeightPool::Config{
                            .arena_bytes = static_cast<std::size_t>(plan.device_capacity(0)),
                            .evictable_tail_bytes =
                                static_cast<std::size_t>(plan.evictable_tail_bytes),
                        });
            auto backing = materialize(reader, MaterializationPlan(plan), device, nullptr,
                                       std::move(pool));
            EvictableWeightPool* const live = backing.weight_pool();
            require(live != nullptr, "pool-backed materialization dropped its pool");
            live->capture_window_mirror(kChunk, device.transfer_stream);
            const std::byte* const resident  = backing.device_parent(reader.find("q5")).data;
            const std::byte* const evictable = backing.device_parent(divisors).data;
            std::array<std::byte, 8> before{};
            CUDA_CHECK(cudaMemcpy(before.data(), evictable, before.size(),
                                  cudaMemcpyDeviceToHost));
            {
                auto transaction = live->evict(kChunk, device.stream);
                require(transaction.leased().bytes == kChunk,
                        "overlay window borrowed an unexpected extent");
                CUDA_CHECK(cudaMemsetAsync(transaction.leased().data, 0x5A, kChunk, device.stream));
                CUDA_CHECK(cudaStreamSynchronize(device.stream));
            }
            require(!live->poisoned(), "closing the overlay window poisoned the pool");
            require(backing.device_parent(divisors).data == evictable &&
                        backing.device_parent(reader.find("q5")).data == resident,
                    "weight addresses moved across an overlay window");
            std::array<std::byte, 8> after{};
            CUDA_CHECK(cudaMemcpy(after.data(), evictable, after.size(), cudaMemcpyDeviceToHost));
            require(after == before, "evict-ranked bytes were not restored from the mirror");
            std::array<std::byte, 4> head{};
            CUDA_CHECK(cudaMemcpy(head.data(), resident, head.size(), cudaMemcpyDeviceToHost));
            require(head[0] == std::byte{0x31}, "a resident object was inside the borrowed chunk");
            (void)matrix;
        }
    }
    {
        Reader reader(fixture.entry);
        Binder binder(reader);
        const auto row = binder.parameter("row", {1, 130}, Residency::Pinned);
        const auto plan = std::move(binder).finish();
        require(plan.device_capacity(0) == 0 && plan.pinned_objects.size() == 1 &&
                    plan.pinned_capacity_bytes == 528,
                "pinned residency did not plan the page-locked block");
        auto backing = materialize(reader, MaterializationPlan(plan), device);
        require(backing.stats().pinned_bytes == 528 && backing.pinned_block().size() == 528 &&
                    !backing.has_device(reader.find("q5")),
                "pinned object received device backing or was not accounted");
        const auto view = bind_view(row, backing);
        require(view.parts.front().parent->data == backing.pinned_block().data(),
                "pinned view does not address the pinned block");
        require(std::equal(backing.pinned_block().begin(), backing.pinned_block().end(),
                           fixture.payload.begin() + 256),
                "pinned block content differs from the artifact payload");
    }
    {
        Reader reader(fixture.entry);
        Binder binder(reader);
        (void)binder.parameter("matrix", {2, 130});
        binder.require_pinned(reader.find("q5"));
        rejects([&] { (void)std::move(binder).finish(); },
                "an object with both device and pinned placement was accepted");
    }
}

void failure_and_host_only(DeviceContext& device) {
    Fixture fixture;
    fixture.write();
    {
        std::fstream part(fixture.directory / "weights-second.bin",
                          std::ios::binary | std::ios::in | std::ios::out);
        part.seekp(16);
        part.put(0);
    }
    {
        Reader reader(fixture.entry);
        Binder binder(reader);
        (void)binder.parameter("matrix", {2, 130});
        rejects([&] { (void)materialize(reader, std::move(binder).finish(), device); },
                "invalid required continuation was uploaded");
    }
    fixture.write();
    Reader reader(fixture.entry);
    Binder binder(reader);
    const auto resource = binder.resource("text", "tokenizer.json");
    auto backing        = materialize(reader, std::move(binder).finish(), device);
    require(backing.stats().device_capacity_bytes == 0 && backing.stats().h2d_bytes == 0 &&
                backing.host_bytes(resource).size() == 5,
            "Host-only demand allocated device weights");
    materialization(device);
}

void file_roundtrip(DeviceContext& device, const std::filesystem::path& path, bool writer_fixture) {
    std::optional<MaterializedArtifact> storage;
    std::map<std::size_t, std::vector<std::byte>> expected;
    std::vector<ParameterReference> parameters;
    {
        Reader reader(path);
        require(reader.file_bytes() < 32ULL * 1024 * 1024,
                "this complete-byte fixture check is limited to small artifacts");
        Binder binder(reader);
        for (const auto& [name, binding] : reader.directory().bindings) {
            auto shape = binding.whole_object
                             ? reader.directory().tensor(binding.parts.front().object).shape
                             : Shape{binding.elements};
            parameters.push_back(binder.parameter(name, std::move(shape)));
        }
        for (const auto& [key, use] : reader.directory().uses) {
            for (const auto& [name, auxiliary] : use.auxiliaries) {
                (void)binder.values(auxiliary);
            }
        }
        for (const auto& [name, component] : reader.directory().components) {
            for (const auto& [role, object] : component.resources) {
                (void)binder.resource(name, role);
            }
        }
        auto plan = std::move(binder).finish();
        for (const auto& placement : plan.device_objects) {
            auto bytes = reader.read_object(placement.object);
            if (writer_fixture && reader.directory().tensor(placement.object).id == "matrix") {
                require(bytes.size() == 130 * 130 * 2, "Python writer changed tensor dimensions");
                for (std::size_t i = 0; i < bytes.size(); ++i) {
                    require(bytes[i] == std::byte((i * 37 + 11) % 251),
                            "C++ reading differs from known Python writer input bytes");
                }
            }
            expected.emplace(placement.object.index, std::move(bytes));
        }
        storage.emplace(materialize(reader, std::move(plan), device));
    }
    for (const auto& [index, bytes] : expected) {
        std::vector<std::byte> actual(bytes.size());
        CUDA_CHECK(cudaMemcpy(actual.data(), storage->device_parent({index}).data, actual.size(),
                              cudaMemcpyDeviceToHost));
        require(actual == bytes, "small artifact upload differs from source bytes");
    }
    for (const auto& parameter : parameters) {
        const auto view        = bind_view(parameter, *storage);
        std::uint64_t elements = 0;
        for (const auto& part : view.parts) {
            elements += part.end - part.begin;
            require(part.parent->data != nullptr, "resolved view lost its parent");
        }
        require(elements == weight_element_count(view.shape),
                "resolved view lost logical coverage");
    }
    std::cout << path.filename().string() << ": all bound parent bytes and logical views passed\n";
}

// Expected bytes of writer_interop.py's GGML objects, a byte sequence `(i * 37 + 11) % 251`: row
// r of a GgmlBlocks object holds bytes [r * row_bytes, (r + 1) * row_bytes) of it, and the parts
// of an expert record object, in order (expert 0 gate, up, down, expert 1 gate, ...), hold its
// consecutive runs, with zero gaps between them.
std::byte interop_byte(std::uint64_t index) { return std::byte((index * 37 + 11) % 251); }

constexpr std::array kExpertParts = {ExpertPart::Gate, ExpertPart::Up, ExpertPart::Down};

std::vector<ObjectHandle> ggml_objects(const Reader& reader) {
    std::vector<ObjectHandle> out;
    for (std::size_t i = 0; i < reader.directory().objects.size(); ++i) {
        if (!std::holds_alternative<TensorObject>(reader.directory().objects[i])) { continue; }
        const auto format = reader.geometry({i}).format;
        if (is_ggml_block(format) || is_ggml_record(format)) { out.push_back({i}); }
    }
    return out;
}

// Copies a materialized parent's bytes back to the host, whichever residency holds it.
std::vector<std::byte> parent_bytes(const WeightParent& parent, bool device) {
    std::vector<std::byte> out(static_cast<std::size_t>(parent.geometry.bytes));
    if (device) {
        CUDA_CHECK(cudaMemcpy(out.data(), parent.data, out.size(), cudaMemcpyDeviceToHost));
    } else {
        std::memcpy(out.data(), parent.data, out.size());
    }
    return out;
}

// The writer's GGML objects survive Pinned and Host residency byte for byte, every row and
// record part sits where the Python writer put it, and every one decodes; Device residency is
// covered by file_roundtrip.
void ggml_writer_residencies(DeviceContext& device, const std::filesystem::path& path) {
    Reader reader(path);
    const auto objects = ggml_objects(reader);
    require(objects.size() == 12, "writer fixture lost a GGML format or layout");
    for (const auto residency : {Residency::Pinned, Residency::Host}) {
        Binder binder(reader);
        for (const auto object : objects) {
            if (residency == Residency::Pinned) {
                binder.require_pinned(object);
            } else {
                (void)binder.host_object(object);
            }
        }
        const auto backing = materialize(reader, std::move(binder).finish(), device);
        for (const auto object : objects) {
            const auto& parent = residency == Residency::Pinned ? backing.pinned_parent(object)
                                                                : backing.host_parent(object);
            const auto& g      = parent.geometry;
            const auto bytes   = parent_bytes(parent, false);
            require(bytes == reader.read_object(object), "GGML parent bytes changed in residency");
            if (residency == Residency::Pinned) {
                require((parent.data - backing.pinned_block().data()) % g.alignment == 0,
                        "pinned GGML parent lost its layout alignment");
            }
            if (g.layout == QuantLayout::GgmlExpertRecord) {
                std::uint64_t cursor = 0;
                for (std::uint64_t expert = 0; expert < g.shape[0]; ++expert) {
                    for (const auto part : kExpertParts) {
                        const auto encoded = test::ggml::record_part_bytes(g, bytes, expert, part);
                        for (std::size_t i = 0; i < encoded.size(); ++i) {
                            require(encoded[i] == interop_byte(cursor + i),
                                    "C++ record addressing differs from the Python writer's parts");
                        }
                        cursor += encoded.size();
                        (void)test::ggml::decode_record_part(g, bytes, expert, part);
                    }
                }
                continue;
            }
            const auto rows = g.elements / g.padded_columns;
            for (std::uint64_t row = 0; row < rows; ++row) {
                const auto encoded = test::ggml::row_bytes(g, bytes, row);
                for (std::size_t i = 0; i < encoded.size(); ++i) {
                    require(encoded[i] == interop_byte(row * g.code_bytes_per_row + i),
                            "C++ GGML row addressing differs from the Python writer's rows");
                }
            }
            (void)test::ggml::decode_parent(g, bytes);
        }
    }
    std::cout << path.filename().string() << ": GGML objects passed Pinned and Host residency\n";
}

// Little-endian record writer for --ggml-samples.
class SampleWriter {
public:
    explicit SampleWriter(const std::filesystem::path& path)
        : file_(path, std::ios::binary | std::ios::trunc) {
        file_.exceptions(std::ios::badbit | std::ios::failbit);
    }

    void word(std::uint64_t value, unsigned bytes) {
        for (unsigned i = 0; i < bytes; ++i) { file_.put(static_cast<char>(value >> (8 * i))); }
    }

    void raw(std::span<const std::byte> bytes) {
        file_.write(reinterpret_cast<const char*>(bytes.data()),
                    static_cast<std::streamsize>(bytes.size()));
    }

private:
    std::ofstream file_;
};

// One sampled GGML block of a parent: its payload offset and block format.
struct BlockSite {
    std::uint64_t offset = 0;
    QType format         = QType::BF16;
};

// Block `index` of a parent, counting a GgmlBlocks parent's blocks in row order and an expert
// bank's blocks record by record (gate, up, down).
BlockSite block_site(const WeightGeometry& g, std::uint64_t index) {
    if (g.layout == QuantLayout::GgmlBlocks) {
        const auto block          = ggml_block(g.format);
        const auto blocks_per_row = g.padded_columns / block.values;
        return {ggml_row_offset(g, index / blocks_per_row) + index % blocks_per_row * block.bytes,
                g.format};
    }
    std::uint64_t per_record = 0;
    for (const auto part : kExpertParts) {
        const auto p = ggml_record_part(g, 0, part);
        per_record += p.rows * p.k / ggml_block(p.format).values;
    }
    auto rest = index % per_record;
    for (const auto part : kExpertParts) {
        const auto p      = ggml_record_part(g, index / per_record, part);
        const auto blocks = p.rows * p.k / ggml_block(p.format).values;
        if (rest < blocks) { return {p.offset + rest * ggml_block(p.format).bytes, p.format}; }
        rest -= blocks;
    }
    throw std::logic_error("block index outside its record");
}

std::uint64_t block_count(const WeightGeometry& g) {
    if (g.layout == QuantLayout::GgmlBlocks) { return g.elements / ggml_block(g.format).values; }
    std::uint64_t per_record = 0;
    for (const auto part : kExpertParts) {
        const auto p = ggml_record_part(g, 0, part);
        per_record += p.rows * p.k / ggml_block(p.format).values;
    }
    return g.shape[0] * per_record;
}

// Real-artifact check (tests/artifact/ggml_artifact_real.py drives it): every GGML object is
// materialized, rotating Device, Pinned and Host residency, compared whole against the file, and
// sampled blocks are decoded here and written out for the Python decoder to compare exactly.
// Every expert record part is also addressed once, which checks the gaps between parts are zero.
int ggml_samples(DeviceContext& device, const std::filesystem::path& path,
                 const std::filesystem::path& out) {
    Reader reader(path);
    const auto objects = ggml_objects(reader);
    if (objects.empty()) {
        std::cout << path.filename().string() << " has no GGML block tensors; skipped\n";
        return 77;
    }
    const auto residency_of = [](std::size_t i) {
        return std::array{Residency::Device, Residency::Pinned, Residency::Host}[i % 3];
    };
    Binder binder(reader);
    for (std::size_t i = 0; i < objects.size(); ++i) {
        switch (residency_of(i)) {
        case Residency::Device:
            binder.require_device(objects[i]);
            break;
        case Residency::Pinned:
            binder.require_pinned(objects[i]);
            break;
        default:
            (void)binder.host_object(objects[i]);
        }
    }
    const auto backing = materialize(reader, std::move(binder).finish(), device);
    SampleWriter samples(out);
    std::uint64_t compared   = 0;
    std::size_t sample_count = 0;
    for (std::size_t i = 0; i < objects.size(); ++i) {
        const auto object    = objects[i];
        const auto residency = residency_of(i);
        const auto& parent   = residency == Residency::Device   ? backing.device_parent(object)
                               : residency == Residency::Pinned ? backing.pinned_parent(object)
                                                                : backing.host_parent(object);
        const auto& g        = parent.geometry;
        const auto bytes     = parent_bytes(parent, residency == Residency::Device);
        require(bytes == reader.read_object(object), "materialized GGML object differs from file");
        compared += bytes.size();
        if (g.layout == QuantLayout::GgmlExpertRecord) {
            for (std::uint64_t expert = 0; expert < g.shape[0]; ++expert) {
                for (const auto part : kExpertParts) {
                    (void)test::ggml::record_part_bytes(g, bytes, expert, part);
                }
            }
        }
        // First, last and 30 spread pseudo-random blocks per object.
        const auto count = block_count(g);
        std::vector<std::uint64_t> picks{0, count - 1};
        std::uint64_t state = 0x9E3779B97F4A7C15ULL ^ i;
        for (int n = 0; n < 30; ++n) {
            state = state * 6364136223846793005ULL + 1442695040888963407ULL;
            picks.push_back((state >> 11) % count);
        }
        const auto& id = reader.directory().tensor(object).id;
        for (const auto pick : picks) {
            const auto site    = block_site(g, pick);
            const auto block   = ggml_block(site.format);
            const auto encoded = std::span<const std::byte>(bytes).subspan(site.offset, block.bytes);
            const auto values  = test::ggml::decode_blocks(site.format, encoded);
            const auto name    = format_name(site.format);
            samples.word(id.size(), 4);
            samples.raw(std::as_bytes(std::span(id)));
            samples.word(static_cast<std::uint64_t>(residency), 4);
            samples.word(site.offset, 8);
            samples.word(name.size(), 4);
            samples.raw(std::as_bytes(std::span(name)));
            samples.word(block.bytes, 4);
            samples.raw(encoded);
            samples.word(values.size(), 4);
            for (const float value : values) { samples.word(std::bit_cast<std::uint32_t>(value), 4); }
            ++sample_count;
        }
    }
    std::cout << path.filename().string() << ": " << objects.size() << " GGML objects ("
              << compared << " B) materialized and compared, " << sample_count
              << " blocks decoded\n";
    return 0;
}

void staging_reuse(DeviceContext& device) {
    // More than one full staging ring, with distinct pages and a partial final block.
    constexpr std::size_t bytes = 5ULL * 64 * 1024 * 1024 + 1024;
    Fixture fixture;
    fixture.payload.clear();
    fixture.root    = {{"components", {{"text", {{"config", Json::object()}}}}},
                       {"objects", Json::array({{{"id", "large"},
                                                 {"kind", "tensor"},
                                                 {"shape", {bytes / 2}},
                                                 {"format", "bf16"},
                                                 {"layout", "contiguous_le_v1"},
                                                 {"offset", 0},
                                                 {"bytes", bytes}}})},
                       {"bindings", {{"large", {{"object", "large"}}}}},
                       {"uses", Json::array()},
                       {"files", Json::array({{{"path", nullptr}, {"payload_bytes", bytes}}})}};
    const auto text = fixture.root.dump();
    std::array<std::byte, 4096> header{};
    const std::array<unsigned char, 8> magic{'N', 'I', 'N', 'F', 'E', 'R', 0, 3};
    for (std::size_t i = 0; i < magic.size(); ++i) { header[i] = std::byte(magic[i]); }
    put_word(header, 8, text.size(), 8);
    std::memcpy(header.data() + 32, text.data(), text.size());
    std::ofstream file(fixture.entry, std::ios::binary | std::ios::trunc);
    file.exceptions(std::ios::badbit | std::ios::failbit);
    file.write(reinterpret_cast<const char*>(header.data()), header.size());
    std::array<std::byte, 4096> page{};
    for (std::size_t offset = 0; offset < bytes; offset += page.size()) {
        for (std::size_t i = 0; i < page.size(); ++i) {
            page[i] = std::byte(((offset + i) * 17 + (offset / 4096) * 13) % 251);
        }
        file.write(reinterpret_cast<const char*>(page.data()),
                   std::min(page.size(), bytes - offset));
    }
    file.close();
    Reader reader(fixture.entry);
    Binder binder(reader);
    (void)binder.parameter("large", {bytes / 2});
    auto backing     = materialize(reader, std::move(binder).finish(), device);
    const auto* base = backing.device_parent(reader.find("large")).data;
    std::vector<std::byte> chunk(1024 * 1024);
    for (std::size_t offset = 0; offset < bytes; offset += chunk.size()) {
        const auto count = std::min(chunk.size(), bytes - offset);
        CUDA_CHECK(cudaMemcpy(chunk.data(), base + offset, count, cudaMemcpyDeviceToHost));
        for (std::size_t i = 0; i < count; ++i) {
            const auto position = offset + i;
            require(chunk[i] == std::byte((position * 17 + (position / 4096) * 13) % 251),
                    "staging slot reuse overwrote an in-flight or later block");
        }
    }
}

// Two ranks on one card: the expert-offload split's planning, its separate per-rank arenas and its
// per-rank upload stream, exercised without a second GPU. A repeated device id is supported for
// exactly this reason -- it frees no memory, but every placement and transfer path is the real one.
void pipeline_rank_placement() {
    const std::array<int, 2> ids{0, 0};
    DeviceContext split{std::span<const int>(ids)};
    Fixture fixture;
    fixture.write(true);
    Reader reader(fixture.entry);
    const auto matrix   = reader.find("q5");
    const auto divisors = reader.find("divisors");
    std::vector<std::byte> expected = reader.read_object(divisors);

    Binder binder(reader);
    (void)binder.parameter("matrix", {2, 130});
    binder.require_device(divisors);
    binder.device_rank(divisors, 1);
    const auto plan = std::move(binder).finish();
    require(plan.device_rank_count() == 2 && plan.device_capacity(0) == 528 &&
                plan.device_capacity(1) == 8,
            "per-rank device capacities were not planned independently");
    for (const auto& placement : plan.device_objects) {
        require((placement.object.index == divisors.index) == (placement.rank == 1) &&
                    placement.offset == 0,
                "each rank's objects must start at the base of that rank's own arena");
    }

    auto backing = materialize(reader, MaterializationPlan(plan), split);
    require(backing.stats().device_capacity_bytes == 528 &&
                backing.stats().offloaded_device_capacity_bytes == 8 &&
                backing.stats().h2d_bytes == 536,
            "split materialization did not report one arena per rank");
    require(backing.device_parent(matrix).data != backing.device_parent(divisors).data,
            "both ranks were served from one allocation");
    std::vector<std::byte> actual(expected.size());
    CUDA_CHECK(cudaMemcpy(actual.data(), backing.device_parent(divisors).data, actual.size(),
                          cudaMemcpyDeviceToHost));
    require(actual == expected, "the offloaded rank received the wrong bytes");

    {
        Binder rejected(reader);
        rejects([&] { rejected.device_rank(divisors, 1); },
                "a pipeline rank was accepted without a device placement");
    }
    {
        Binder rejected(reader);
        rejected.require_device(divisors);
        rejected.device_rank(divisors, 1);
        rejects([&] { rejected.device_rank(divisors, 0); },
                "one object was accepted on two different devices");
    }
    {
        // Overlay Vision with a layer split: the evictable tail is rank 0's alone and a later
        // stage's resident weights are planned on their own arena, neither shifting nor joining it.
        Binder overlay(reader);
        (void)overlay.parameter("matrix", {2, 130});
        overlay.device_rank(matrix, 1);
        overlay.require_device(divisors);
        overlay.evict_device(divisors, 700);
        const auto overlay_plan = std::move(overlay).finish(EvictableWeightPool::kChunkBytes);
        require(overlay_plan.device_rank_count() == 2 && overlay_plan.evictable_tail_offset == 0 &&
                    overlay_plan.evictable_tail_bytes == 8 &&
                    overlay_plan.device_capacity(0) == 8 && overlay_plan.device_capacity(1) == 528,
                "an evictable tail and a later stage's weights were not planned per rank");
    }
    {
        // An offloaded rank holds expert blocks and nothing a Vision window could borrow, so the
        // evictable tail and a non-primary rank must not be planned together.
        Binder rejected(reader);
        rejected.require_device(divisors);
        rejected.evict_device(divisors, 1);
        rejected.device_rank(divisors, 1);
        rejects([&] { (void)std::move(rejected).finish(); },
                "an evictable placement was accepted away from the primary device");
    }
}

} // namespace

int main(int argc, char** argv) {
    try {
        int count         = 0;
        const auto result = cudaGetDeviceCount(&count);
        if (result == cudaErrorNoDevice || result == cudaErrorInsufficientDriver ||
            (result == cudaSuccess && count == 0)) {
            return 77;
        }
        CUDA_CHECK(result);
        DeviceContext device;
        if (argc == 3 && (std::string_view(argv[1]) == "--artifact" ||
                          std::string_view(argv[1]) == "--writer-fixture")) {
            const bool writer_fixture = std::string_view(argv[1]) == "--writer-fixture";
            file_roundtrip(device, argv[2], writer_fixture);
            if (writer_fixture) { ggml_writer_residencies(device, argv[2]); }
            return 0;
        }
        if (argc == 4 && std::string_view(argv[1]) == "--ggml-samples") {
            return ggml_samples(device, argv[2], argv[3]);
        }
        if (argc != 1) {
            throw std::invalid_argument(
                "expected [--artifact|--writer-fixture PATH] or [--ggml-samples PATH OUT]");
        }
        materialization(device);
        failure_and_host_only(device);
        overlay_placement(device);
        pipeline_rank_placement();
#if defined(NINFER_TEST_LINK_WRAP)
        ninfer::test::materialization_cuda_errors(device);
#else
        // CUDA fault injection needs GNU ld --wrap; MSVC has no link-time interposition, so those
        // cases are not built here. The remaining checks still run against the real runtime.
        std::cout << "artifact materialization CUDA fault injection skipped (no --wrap)\n";
#endif
        staging_reuse(device);
        std::cout << "artifact materialization checks passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
