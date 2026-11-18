// The Qwen3.8-Flash-Next refusal-steering control vector reader (models/qwen4_exp/steering.h):
// GGUF files written here in the llama.cpp control-vector layout, one accepted and every way the
// reader refuses.

#include "models/qwen4_exp/steering.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::models::qwen4_exp;

namespace {

constexpr std::int32_t kHidden = 8;
constexpr std::int32_t kLayers = 6;

int check(bool condition, const char* message) {
    if (!condition) { std::cerr << "FAIL: " << message << '\n'; }
    return condition ? 0 : 1;
}

struct Writer {
    std::vector<std::uint8_t> bytes;

    template <typename T>
    void pod(T value) {
        const auto* p = reinterpret_cast<const std::uint8_t*>(&value);
        bytes.insert(bytes.end(), p, p + sizeof(T));
    }

    void text(const std::string& s) {
        pod<std::uint64_t>(s.size());
        bytes.insert(bytes.end(), s.begin(), s.end());
    }
};

struct Spec {
    std::string architecture = "controlvector";
    std::string model_hint   = "qwen4exp";
    std::uint32_t version    = 3;
    std::vector<std::pair<std::string, std::vector<float>>> tensors;
    std::uint32_t type        = 0; // F32
    std::uint32_t dims        = 1;
    std::uint64_t offset_bias = 0;
};

std::vector<float> ramp(float scale) {
    std::vector<float> v(kHidden);
    for (int i = 0; i < kHidden; ++i) v[i] = scale * static_cast<float>(i + 1);
    return v;
}

std::vector<std::uint8_t> build(const Spec& spec) {
    Writer w;
    w.bytes.insert(w.bytes.end(), {'G', 'G', 'U', 'F'});
    w.pod<std::uint32_t>(spec.version);
    w.pod<std::uint64_t>(spec.tensors.size());
    w.pod<std::uint64_t>(spec.model_hint.empty() ? 1 : 2);
    w.text("general.architecture");
    w.pod<std::uint32_t>(8);
    w.text(spec.architecture);
    if (!spec.model_hint.empty()) {
        w.text("controlvector.model_hint");
        w.pod<std::uint32_t>(8);
        w.text(spec.model_hint);
    }
    std::uint64_t offset = 0;
    for (const auto& [name, values] : spec.tensors) {
        w.text(name);
        w.pod<std::uint32_t>(spec.dims);
        w.pod<std::uint64_t>(values.size());
        w.pod<std::uint32_t>(spec.type);
        w.pod<std::uint64_t>(offset + spec.offset_bias);
        offset += values.size() * sizeof(float);
    }
    while (w.bytes.size() % 32 != 0) w.bytes.push_back(0);
    for (const auto& [name, values] : spec.tensors) {
        const auto* p = reinterpret_cast<const std::uint8_t*>(values.data());
        w.bytes.insert(w.bytes.end(), p, p + values.size() * sizeof(float));
    }
    return w.bytes;
}

std::filesystem::path write_file(const std::filesystem::path& dir, const std::string& name,
                                 const std::vector<std::uint8_t>& bytes) {
    const auto path = dir / name;
    std::ofstream out(path, std::ios::binary);
    out.write(reinterpret_cast<const char*>(bytes.data()),
              static_cast<std::streamsize>(bytes.size()));
    return path;
}

// Returns the refusal text, or an empty string when the file was accepted.
std::string refusal(const std::filesystem::path& path) {
    try {
        (void)load_steering_vector(GraftSource{.name = "red", .path = path}, kHidden, kLayers);
    } catch (const std::invalid_argument& error) { return error.what(); }
    return {};
}

} // namespace

int main() {
    int failures   = 0;
    const auto dir = std::filesystem::temp_directory_path() / "ninfer-steering-test";
    std::filesystem::remove_all(dir);
    std::filesystem::create_directories(dir);

    // A valid file: layers 1 and 3 carry directions; the loader normalises them.
    Spec good;
    good.tensors         = {{"direction.1", ramp(3.0f)}, {"direction.3", ramp(-0.5f)}};
    const auto good_path = write_file(dir, "good.gguf", build(good));
    const SteeringVector vector =
        load_steering_vector({.name = "red", .path = good_path}, kHidden, kLayers);
    failures += check(vector.name == "red" && vector.hidden == kHidden && vector.layers == kLayers,
                      "the vector did not keep its name, hidden size and layer count");
    failures += check(vector.active == std::vector<std::uint8_t>{0, 1, 0, 1, 0, 0},
                      "only the layers with a tensor were not marked active");
    failures += check(vector.active_layers() == 2, "active_layers miscounted");
    failures += check(vector.directions.size() == static_cast<std::size_t>(kLayers) * kHidden,
                      "the table is not [layers][hidden]");
    for (const std::int32_t layer : {1, 3}) {
        double norm = 0.0;
        for (int i = 0; i < kHidden; ++i) {
            const double v = vector.directions[static_cast<std::size_t>(layer) * kHidden + i];
            norm += v * v;
        }
        failures +=
            check(std::fabs(norm - 1.0) < 1.0e-6, "a direction was not normalised to unit length");
    }
    // The sign and shape of the direction survive: layer 3 was a negative ramp.
    failures +=
        check(vector.directions[3 * kHidden] < 0.0f && vector.directions[3 * kHidden + 7] < 0.0f &&
                  vector.directions[1 * kHidden] > 0.0f,
              "normalisation lost the direction's sign");
    bool zeros = true;
    for (int i = 0; i < kHidden; ++i) zeros = zeros && vector.directions[2 * kHidden + i] == 0.0f;
    failures += check(zeros, "a layer without a tensor was not left at zero");
    failures += check(vector.digest != 0, "the digest was zero");

    // The digest follows the file's bytes: the same file twice agrees, a changed value differs.
    const SteeringVector again =
        load_steering_vector({.name = "other", .path = good_path}, kHidden, kLayers);
    failures += check(again.digest == vector.digest, "the same bytes gave two digests");
    Spec changed = good;
    changed.tensors[0].second[0] += 1.0f;
    const SteeringVector other = load_steering_vector(
        {.name = "red", .path = write_file(dir, "changed.gguf", build(changed))}, kHidden, kLayers);
    failures += check(other.digest != vector.digest, "different bytes gave the same digest");

    // Refusals, each naming the problem.
    auto expect = [&](const char* label, const Spec& spec, const char* fragment) {
        const std::string text =
            refusal(write_file(dir, std::string(label) + ".gguf", build(spec)));
        failures += check(!text.empty() && text.find(fragment) != std::string::npos &&
                              text.find("graft 'red'") != std::string::npos,
                          label);
        if (text.empty() || text.find(fragment) == std::string::npos)
            std::cerr << "  got: " << text << '\n';
    };
    Spec s         = good;
    s.architecture = "llama";
    expect("a file of another architecture", s, "not 'controlvector'");
    s            = good;
    s.model_hint = "qwen3";
    expect("a vector made for another model", s, "not for qwen4exp");
    s         = good;
    s.version = 1;
    expect("an unsupported GGUF version", s, "GGUF version 1");
    s         = good;
    s.tensors = {{"direction.1", std::vector<float>(kHidden + 1, 1.0f)}};
    expect("a tensor of the wrong width", s, "hidden size is 8");
    s         = good;
    s.tensors = {{"direction.0", ramp(1.0f)}};
    expect("layer zero", s, "layers 1 to 5");
    s         = good;
    s.tensors = {{"direction.6", ramp(1.0f)}};
    expect("a layer past the last block", s, "layers 1 to 5");
    s         = good;
    s.tensors = {{"direction.2", ramp(1.0f)}, {"direction.2", ramp(2.0f)}};
    expect("a layer given twice", s, "appears twice");
    s         = good;
    s.tensors = {{"weights.1", ramp(1.0f)}};
    expect("a tensor with another name", s, "not named direction.N");
    s         = good;
    s.tensors = {{"direction.1", std::vector<float>(kHidden, 0.0f)}};
    expect("the zero vector", s, "zero vector");
    s         = good;
    auto bad  = ramp(1.0f);
    bad[2]    = std::nanf("");
    s.tensors = {{"direction.1", bad}};
    expect("a non-finite value", s, "not finite");
    s      = good;
    s.type = 1; // F16
    expect("a tensor that is not F32", s, "not F32");
    s      = good;
    s.dims = 2;
    expect("a tensor with two dimensions", s, "not 1");
    s             = good;
    s.offset_bias = 1 << 20;
    expect("an offset past the end of the file", s, "past the end");
    s = good;
    s.tensors.clear();
    expect("a file with no tensors", s, "0 tensors");

    // Damaged files: not GGUF, truncated at every length, and a missing file.
    failures +=
        check(refusal(write_file(dir, "text.gguf", {'n', 'o', 'p', 'e', '!', '!', '!', '!'}))
                      .find("not a GGUF") != std::string::npos,
              "a text file was not refused as not GGUF");
    const auto whole         = build(good);
    int accepted_truncations = 0;
    for (std::size_t cut = 0; cut < whole.size(); ++cut) {
        const std::vector<std::uint8_t> part(whole.begin(),
                                             whole.begin() + static_cast<std::ptrdiff_t>(cut));
        try {
            (void)load_steering_vector({.name = "red", .path = write_file(dir, "cut.gguf", part)},
                                       kHidden, kLayers);
            ++accepted_truncations;
        } catch (const std::invalid_argument&) {}
    }
    failures += check(accepted_truncations == 0, "a truncated file was accepted");
    failures += check(refusal(dir / "missing.gguf").find("cannot read") != std::string::npos,
                      "a missing file was not refused as unreadable");

    std::filesystem::remove_all(dir);
    std::cout << (failures ? "FAIL" : "OK") << " steering\n";
    return failures ? 1 : 0;
}
