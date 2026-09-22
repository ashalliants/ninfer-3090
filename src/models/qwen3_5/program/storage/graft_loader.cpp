#include "models/qwen3_5/program/storage/graft_loader.h"

#include <fstream>
#include <cstring>

namespace ninfer::models::qwen3_5 {

// Magic bytes: 8-byte signature followed by version, then model ID length + UTF-8 string.
inline constexpr std::uint64_t kGrafitMagic = 0x5048414E544F4D4BL; // PHANTOMK
inline constexpr std::uint32_t kGrafitVersion = 1;

// Internal binary layout per layer:
//   layer_index(I32) | head_dim(I32) | num_kv_heads(I32) | slots(U32) | k_data(BF16 row-major) | v_data(BF16 row-major)
LoadedGraft GraftLoader::load_file(std::filesystem::path const& path, FileSpec spec) {
    std::ifstream f(path, std::ios::binary);
    if (!f.good()) { throw std::runtime_error("graft file unreadable: " + path.string()); }

    std::uint64_t magic;
    f.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    if (magic != kGrafitMagic) {
        throw std::runtime_error("invalid graft magic in " + path.string());
    }

    std::uint32_t version;
    f.read(reinterpret_cast<char*>(&version), sizeof(version));
    if (version != kGrafitVersion) {
        throw std::runtime_error("unsupported graft version " + std::to_string(version) +
                                 ", expected " + std::to_string(kGrafitVersion));
    }

    // Model ID: U32 length followed by UTF-8 bytes
    std::uint32_t model_id_len;
    f.read(reinterpret_cast<char*>(&model_id_len), sizeof(model_id_len));
    std::vector<char> model_id_buf(model_id_len + 1);
    f.read(model_id_buf.data(), model_id_len);
    model_id_buf[model_id_len] = '\0';
    std::string model_id(model_id_buf.data(), model_id_len);
    if (model_id != spec.model_id) {
        throw std::runtime_error(
            "graft model-id mismatch: file=" + model_id + " expected=" + spec.model_id);
    }

    // Layer count
    std::int32_t layer_count;
    f.read(reinterpret_cast<char*>(&layer_count), sizeof(layer_count));
    if (layer_count <= 0 || layer_count > 256) {
        throw std::runtime_error("invalid layer count in graft");
    }

    LoadedGraft graft;
    graft.model_id_       = model_id;
    graft.layers_.resize(static_cast<std::size_t>(layer_count));

    std::uint32_t global_max_slots = 0;

    for (std::int32_t i = 0; i < layer_count; ++i) {
        auto& ld = graft.layers_[static_cast<std::size_t>(i)];
        std::int32_t li = 0;
        f.read(reinterpret_cast<char*>(&li), sizeof(li));
        if (li != i) {
            throw std::runtime_error("graft layer_index mismatch at index " + std::to_string(i));
        }
        ld.layer_index  = li;
        f.read(reinterpret_cast<char*>(&ld.head_dim), sizeof(ld.head_dim));
        f.read(reinterpret_cast<char*>(&ld.num_kv_heads), sizeof(ld.num_kv_heads));
        f.read(reinterpret_cast<char*>(&ld.slots), sizeof(ld.slots));

        if (ld.slots == 0) {
            throw std::runtime_error("graft layer has zero slots");
        }
        if (ld.head_dim <= 0) {
            throw std::runtime_error("graft layer has invalid head_dim");
        }

        auto elem_bytes = ld.slots * ld.head_dim * 2ULL; // BF16 = 2 bytes

        // Allocate pinned host buffers
        void* k_void = nullptr;
        void* v_void = nullptr;

        cudaHostAlloc(&k_void, elem_bytes, cudaHostAllocDefault);
        cudaHostAlloc(&v_void, elem_bytes, cudaHostAllocDefault);

        std::memset(k_void, 0, elem_bytes);
        std::memset(v_void, 0, elem_bytes);

        f.read(reinterpret_cast<char*>(k_void), elem_bytes);
        f.read(reinterpret_cast<char*>(v_void), elem_bytes);

        // Own the raw allocation so it's freed when LayerGrafitData is destroyed.
        ld.k_raw.reset(static_cast<std::uint8_t*>(k_void));
        ld.v_raw.reset(static_cast<std::uint8_t*>(v_void));

        // Wrap in Tensor structs referencing pinned memory.
        ld.k_tensor  = Tensor(ld.k_raw.get(), DType::BF16,
                              {static_cast<std::int32_t>(ld.slots), ld.head_dim});
        ld.v_tensor  = Tensor(ld.v_raw.get(), DType::BF16,
                              {static_cast<std::int32_t>(ld.slots), ld.head_dim});

        if (ld.slots > global_max_slots) {
            global_max_slots = ld.slots;
        }
    }

    graft.max_slots_ = global_max_slots;
    return graft;
}

LoadedGraft GraftLoader::load(std::filesystem::path const& path,
                              std::string_view expected_model_id) {
    return load_file(path, FileSpec{static_cast<std::string>(expected_model_id)});
}

bool GraftLoader::probe_header(std::filesystem::path const& path,
                               std::string& out_model_id) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f.good()) { return false; }

    auto size = f.tellg();
    // Minimum valid header: magic(8) + version(4) + model_id_len(4) + 1-byte model_id(1) = 17
    if (size < 0 || size < 17) { return false; }

    f.seekg(0, std::ios::beg);
    std::uint64_t magic;
    f.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    if (magic != kGrafitMagic) { return false; }

    std::uint32_t version;
    f.read(reinterpret_cast<char*>(&version), sizeof(version));
    if (version != kGrafitVersion) { return false; }

    std::uint32_t len;
    f.read(reinterpret_cast<char*>(&len), sizeof(len));
    if (len == 0 || len > 256) { return false; }

    std::vector<char> buf(len + 1);
    f.read(buf.data(), len);
    buf[len] = '\0';
    out_model_id.assign(buf.data(), len);
    return true;
}

} // namespace ninfer::models::qwen3_5
