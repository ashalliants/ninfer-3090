// Written for NInfer-3090. See engram_sidecar.h.

#include "models/dsv4/execution/engram_sidecar.h"

#include "artifact/schema.h"
#include "models/qwen3_5/frontend/digest.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iterator>
#include <string>
#include <string_view>
#include <vector>

namespace ninfer::models::dsv4 {

using artifact::ArtifactError;
using artifact::Json;

namespace {

constexpr const char* kFormat    = "ninfer-dsv4-engram";
constexpr std::uint32_t kVersion = 1;

std::vector<std::uint8_t> read_bytes(const std::filesystem::path& path, std::uint64_t limit) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) { throw ArtifactError("Engram folder: cannot open " + path.string()); }
    std::vector<std::uint8_t> bytes((std::istreambuf_iterator<char>(stream)), std::istreambuf_iterator<char>());
    if (bytes.size() > limit) { throw ArtifactError("Engram folder: " + path.string() + " is larger than expected"); }
    return bytes;
}

std::vector<std::uint8_t> read_prefix(const std::filesystem::path& path, std::size_t count) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) { throw ArtifactError("Engram volume: cannot open " + path.string()); }
    std::vector<std::uint8_t> bytes(count);
    stream.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(count));
    if (static_cast<std::size_t>(stream.gcount()) != count) { throw ArtifactError("Engram volume: " + path.string() + " is shorter than its header"); }
    return bytes;
}

std::string sha256_of(std::string_view text) { return qwen3_5::frontend::sha256_hex(qwen3_5::frontend::sha256(text)); }

std::string sha256_of(const std::vector<std::uint8_t>& bytes) {
    return qwen3_5::frontend::sha256_hex(qwen3_5::frontend::sha256(std::span<const std::uint8_t>(bytes)));
}

template <class T>
std::vector<T> numbers(const Json& value, const char* what) {
    if (!value.is_array()) { throw ArtifactError(std::string("Engram manifest: ") + what + " is not a list"); }
    std::vector<T> out;
    for (const Json& v : value) {
        if (!v.is_number_integer()) { throw ArtifactError(std::string("Engram manifest: ") + what + " holds a non-integer"); }
        out.push_back(v.get<T>());
    }
    return out;
}

} // namespace

EngramSidecar load_engram_sidecar(const std::filesystem::path& directory, const TextConfig& config) {
    const std::filesystem::path manifest_path = directory / kEngramManifestName;
    const auto fail                           = [&](const std::string& why) {
        return ArtifactError("Engram folder " + directory.string() + ": " + why);
    };
    if (!std::filesystem::is_regular_file(manifest_path)) {
        throw fail(std::string("no ") + kEngramManifestName + " (write the folder with python3 -m tools.convert.dsv4_engram_volume)");
    }
    Json manifest;
    try {
        const auto bytes = read_bytes(manifest_path, 1U << 24);
        manifest         = Json::parse(bytes.begin(), bytes.end());
    } catch (const Json::exception& error) {
        throw fail(std::string("the manifest is not valid JSON (") + error.what() + ")");
    }
    EngramSidecar out;
    try {
        // The digest covers everything the checksums below do not (the multipliers, bucket counts, layers): it is the sha256
        // of the manifest without its own "digest", in the canonical form the writer used (sorted keys, no whitespace).
        Json body = manifest;
        body.erase("digest");
        if (manifest.at("digest").get<std::string>() != sha256_of(std::string_view(body.dump()))) {
            throw fail("the manifest's digest does not match its contents (edited or damaged)");
        }
        if (manifest.at("format").get<std::string>() != kFormat || manifest.at("version").get<std::uint32_t>() != kVersion) {
            throw fail(std::string("the manifest is not a version ") + std::to_string(kVersion) + " " + kFormat + " manifest");
        }
        const Json& hash = manifest.at("hash");
        std::vector<std::uint32_t> layers;
        for (std::uint32_t l = 0; l < config.layers.size(); ++l) {
            if (config.layers[l].engram) { layers.push_back(l); }
        }
        const EngramConfig& e = config.engram;
        if (numbers<std::uint32_t>(hash.at("layers"), "layers") != layers) { throw fail("its Engram layers differ from the model's"); }
        if (hash.at("ngram_size").get<std::uint32_t>() != e.ngram_size || hash.at("heads").get<std::uint32_t>() != e.heads ||
            hash.at("row_width").get<std::uint32_t>() != e.row_width) {
            throw fail("its n-gram size, heads or row width differ from the model's");
        }
        if (hash.at("pad_token_id").get<std::int32_t>() != e.pad_token) { throw fail("its pad token differs from the model's"); }
        if (numbers<std::uint64_t>(hash.at("head_vocab_sizes"), "head_vocab_sizes") != e.head_vocab_sizes) {
            throw fail("its bucket counts differ from the model's (a folder written from another model)");
        }
        out.params.multipliers = numbers<std::int64_t>(hash.at("multipliers"), "multipliers");

        const std::uint64_t count = hash.at("token_map_count").get<std::uint64_t>();
        if (count != config.vocab_size) {
            throw fail("its token map has " + std::to_string(count) + " entries, the model's vocabulary " + std::to_string(config.vocab_size));
        }
        const Json& map     = hash.at("token_map");
        const auto map_path = directory / map.at("file").get<std::string>();
        const auto raw      = read_bytes(map_path, count * 4);
        if (raw.size() != count * 4) { throw fail(map_path.filename().string() + " has the wrong size"); }
        if (sha256_of(raw) != map.at("sha256").get<std::string>()) {
            throw fail(map_path.filename().string() + " differs from the checksum in the manifest");
        }
        out.params.token_map.resize(count);
        std::memcpy(out.params.token_map.data(), raw.data(), raw.size()); // little-endian int32, as the host is

        const Json& volume = manifest.at("volume");
        out.volume_path    = directory / volume.at("file").get<std::string>();
        out.table          = parse_engram_table(volume.at("table"), e.row_width);
        if (e.table_rows != 0 && out.table.rows != e.table_rows) { throw fail("its volume's rows differ from the artifact's engram_table"); }
        std::error_code ec;
        if (!std::filesystem::is_regular_file(out.volume_path) || std::filesystem::file_size(out.volume_path, ec) != out.table.file_bytes) {
            throw fail(out.volume_path.filename().string() + " is missing or not " + std::to_string(out.table.file_bytes) + " bytes");
        }
        if (sha256_of(read_prefix(out.volume_path, out.table.header_bytes)) != volume.at("header_sha256").get<std::string>()) {
            throw fail(out.volume_path.filename().string() + "'s header differs from the manifest's checksum");
        }
    } catch (const Json::exception& error) {
        throw fail(std::string("the manifest is malformed (") + error.what() + ")");
    }
    return out;
}

} // namespace ninfer::models::dsv4
