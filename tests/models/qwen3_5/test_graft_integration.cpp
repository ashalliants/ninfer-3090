// phantom-kv graft integration tests
#include "models/qwen3_5/program/storage/graft_loader.h"
#include "core/device.h"

#include <cuda_runtime.h>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <span>
#include <string>
#include <vector>

namespace {

namespace q36 = ninfer::models::qwen3_5;

inline constexpr std::uint64_t kGrafitMagic = 0x5048414E544F4D4BL; // "PHANTOMK" LE
inline constexpr std::uint32_t kGrafitVersion = 1;

int failures       = 0;
std::string tmp_dir;
bool g_cuda_available = false;

void expect(bool condition, std::string_view message) {
    if (condition) return;
    ++failures;
    std::cerr << "FAIL: " << message << '\n';
}

bool throws_fn(std::function<void()> fn) {
    try { fn(); return false; } catch (...) { return true; }
}

void setup() {
    tmp_dir = std::filesystem::temp_directory_path().string() + "/ninfer_graft_test";
    std::filesystem::create_directories(tmp_dir);
    auto err = cudaSetDevice(0);
    if (err != cudaSuccess) return;
    int *dptr = nullptr;
    err = cudaMalloc(&dptr, 4);
    if (err == cudaSuccess) { cudaFree(dptr); g_cuda_available = true; }
}

void cleanup() {
    if (g_cuda_available) { cudaDeviceReset(); }
    std::filesystem::remove_all(tmp_dir);
}

void write_graft_file(std::string const& path,
                      std::string const& model_id,
                      int slot_count, int head_dim,
                      std::vector<std::pair<std::vector<uint8_t>, std::vector<uint8_t>>> const& layer_kvs) {
    std::ofstream f(path, std::ios::binary);
    f.write(reinterpret_cast<char const*>(&kGrafitMagic), sizeof(kGrafitMagic));
    std::uint32_t ver = kGrafitVersion;
    f.write(reinterpret_cast<char const*>(&ver), sizeof(ver));
    std::uint32_t mid_len = static_cast<std::uint32_t>(model_id.size());
    f.write(reinterpret_cast<char const*>(&mid_len), sizeof(mid_len));
    f.write(model_id.data(), mid_len);
    std::int32_t lc = static_cast<std::int32_t>(layer_kvs.size());
    f.write(reinterpret_cast<char const*>(&lc), sizeof(lc));
    for (size_t i = 0; i < layer_kvs.size(); ++i) {
        std::int32_t li = static_cast<std::int32_t>(i);
        f.write(reinterpret_cast<char const*>(&li), sizeof(li));
        std::int32_t hd = head_dim;
        f.write(reinterpret_cast<char const*>(&hd), sizeof(hd));
        std::int32_t kvh = 2;
        f.write(reinterpret_cast<char const*>(&kvh), sizeof(kvh));
        std::uint32_t sl = static_cast<std::uint32_t>(slot_count);
        f.write(reinterpret_cast<char const*>(&sl), sizeof(sl));
        f.write(reinterpret_cast<char const*>(layer_kvs[i].first.data()),
                static_cast<std::streamsize>(layer_kvs[i].first.size()));
        f.write(reinterpret_cast<char const*>(layer_kvs[i].second.data()),
                static_cast<std::streamsize>(layer_kvs[i].second.size()));
    }
}

void test_loader_missing_file() {
    expect(throws_fn([&] { q36::GraftLoader::load("nonexistent.phantom", ""); }),
           "missing file should throw");
}

void test_loader_wrong_magic() {
    const std::string path = tmp_dir + "/wrong_magic.phantom";
    std::ofstream f(path, std::ios::binary);
    std::uint64_t bad = 0xDEADBEEFCAFEBABEULL;
    f.write(reinterpret_cast<char const*>(&bad), sizeof(bad));
    f.close();
    expect(throws_fn([&] { q36::GraftLoader::load(path, ""); }), "wrong magic should throw");
}

void test_loader_short_header() {
    const std::string path = tmp_dir + "/short_header.phantom";
    std::ofstream f(path, std::ios::binary);
    f.put(0x4B);
    f.close();
    expect(throws_fn([&] { q36::GraftLoader::load(path, ""); }), "truncated header should throw");
}

void test_probe_bad_file() {
    const std::string path = tmp_dir + "/probe_bad.dat";
    std::ofstream f(path, std::ios::binary);
    f.write("NOT_A_GRAFT", 11);
    f.close();
    std::string mid;
    bool result = q36::GraftLoader::probe_header(path, mid);
    expect(!result, "probe_header should return false for invalid file");
}

void test_loader_round_trip() {
    if (!g_cuda_available) {
        std::cout << "SKIP: round-trip requires CUDA device\n";
        return;
    }
    const std::string path   = tmp_dir + "/roundtrip.phantom";
    const int slots          = 16;
    const int hd             = 64;
    const size_t data_size   = static_cast<size_t>(slots) * hd * 2ULL;

    std::vector<std::pair<std::vector<uint8_t>, std::vector<uint8_t>>> layer_kvs(1);
    layer_kvs[0].first.resize(data_size);
    layer_kvs[0].second.resize(data_size);
    for (size_t i = 0; i < data_size; ++i) {
        layer_kvs[0].first[i] = static_cast<uint8_t>(i & 0xFF);
        layer_kvs[0].second[i] = static_cast<uint8_t>((static_cast<int>(i ^ 0xAA)) & 0xFF);
    }
    write_graft_file(path, "test_model", slots, hd, layer_kvs);

    bool threw = false;
    try {
        auto graft = q36::GraftLoader::load(path, "test_model");
        expect(graft.layers() == 1, "should have 1 layer");
        expect(graft.max_slots() == 16, "max_slots should be 16");
        const auto& layer = graft.layer(0);
        expect(layer.layer_index == 0, "layer_index == 0");
        expect(layer.slots == 16, "slots == 16");
        expect(layer.head_dim == 64, "head_dim == 64");
        expect(layer.k_tensor.data != nullptr, "k_tensor data != null");
        expect(layer.v_tensor.data != nullptr, "v_tensor data != null");
        if (layer.k_tensor.data && layer.v_tensor.data) {
            const uint8_t* kb = reinterpret_cast<const uint8_t*>(layer.k_tensor.data);
            const uint8_t* vb = reinterpret_cast<const uint8_t*>(layer.v_tensor.data);
            expect(kb[0] == 0x00, "K[0] should be 0x00");
            expect(kb[100] == 100u, "K[100] pattern matches");
            // K/V should differ
            bool any_diff = false;
            for (size_t i = 0; i < data_size && !any_diff; ++i)
                if (kb[i] != vb[i]) any_diff = true;
            expect(any_diff, "K and V tensors should differ");
        }
    } catch (...) { threw = true; }
    expect(!threw, "round-trip should succeed");
}

void test_probe_valid_file() {
    const std::string path = tmp_dir + "/probe_valid.phantom";
    {
        std::ofstream f(path, std::ios::binary);
        f.write(reinterpret_cast<char const*>(&kGrafitMagic), sizeof(kGrafitMagic));
        std::uint32_t ver = kGrafitVersion;
        f.write(reinterpret_cast<char const*>(&ver), sizeof(ver));
        std::uint32_t ml = 8;
        f.write(reinterpret_cast<char const*>(&ml), sizeof(ml));
        f.write("probe_id", 8);
        f.close();
    }
    bool threw = false;
    std::string mid;
    try {
        bool ok = q36::GraftLoader::probe_header(path, mid);
        expect(ok, "probe_header should succeed for valid file");
        expect(mid == "probe_id", "probe_header model_id should match");
    } catch (...) { threw = true; }
    expect(!threw, "probe_header should not throw");
}

void test_probe_incomplete_file() {
    const std::string path = tmp_dir + "/probe_short.phantom";
    {
        std::ofstream f(path, std::ios::binary);
        f.write(reinterpret_cast<char const*>(&kGrafitMagic), sizeof(kGrafitMagic));
        std::uint32_t ver = kGrafitVersion;
        f.write(reinterpret_cast<char const*>(&ver), 2); // truncate
        f.close();
    }
    std::string mid;
    bool result = q36::GraftLoader::probe_header(path, mid);
    expect(!result, "probe_header should return false for incomplete file");
}

} // anonymous namespace

int main() {
    std::cout << "=== Phantom-KV Graft Integration Tests ===\n\n";
    setup();

    std::cout << "[Loader Validation]\n";
    test_loader_missing_file();
    test_loader_wrong_magic();
    test_loader_short_header();
    test_probe_bad_file();
    test_loader_round_trip();
    test_probe_valid_file();
    test_probe_incomplete_file();

    cleanup();

    if (failures > 0) {
        std::cerr << "\n" << failures << " test(s) FAILED\n";
        return 1;
    }
    std::cout << "\nAll tests passed.\n";
    return 0;
}
