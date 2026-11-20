// The dense-model control-vector reader (models/qwen3_5/frontend/control_vector.h): GGUF files
// written here in the GLP and llama.cpp layouts, what each one means, and every way the reader
// refuses a file it cannot apply as written. NINFER_TEST_CONTROL_VECTOR, when set, also reads a
// real file against Qwen3.8-27B's shape (hidden 5120, 64 layers).

#include "models/qwen3_5/frontend/control_vector.h"
#include "models/qwen3_5/frontend/digest.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>
#include <vector>

using namespace ninfer;
using namespace ninfer::models;
using namespace ninfer::models::qwen3_5;

namespace {

constexpr std::int32_t kHidden = 8;
constexpr std::int32_t kLayers = 6;

int check(bool condition, const std::string& message) {
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

// A metadata value: a string, a u32, an f32 or a bool.
using Meta = std::variant<std::string, std::uint32_t, float, bool>;

struct Spec {
    std::vector<std::pair<std::string, Meta>> meta = {{"general.architecture", std::string("controlvector")},
                                                      {"controlvector.model_hint", std::string("qwen3_5")}};
    std::vector<std::pair<std::string, std::vector<float>>> tensors;
    std::uint32_t type = 0; // F32
    std::uint32_t dims = 1;
    bool magic         = true;
    std::size_t cut    = 0; // bytes dropped from the end

    Spec& set(const std::string& key, Meta value) {
        for (auto& [k, v] : meta) {
            if (k == key) {
                v = std::move(value);
                return *this;
            }
        }
        meta.emplace_back(key, std::move(value));
        return *this;
    }
    Spec& erase(const std::string& key) {
        std::erase_if(meta, [&](const auto& entry) { return entry.first == key; });
        return *this;
    }
};

std::vector<float> ramp(float scale) {
    std::vector<float> v(kHidden);
    for (int i = 0; i < kHidden; ++i) v[static_cast<std::size_t>(i)] = scale * static_cast<float>(i + 1);
    return v;
}

// The bytes of `tensors` in layer order: what glp.content_sha256 hashes.
std::string content_sha256(const std::vector<std::pair<std::string, std::vector<float>>>& tensors) {
    std::vector<std::pair<int, const std::vector<float>*>> ordered;
    for (const auto& [name, values] : tensors) ordered.emplace_back(std::stoi(name.substr(10)), &values);
    std::sort(ordered.begin(), ordered.end());
    std::vector<std::uint8_t> bytes;
    for (const auto& [layer, values] : ordered) {
        const auto* p = reinterpret_cast<const std::uint8_t*>(values->data());
        bytes.insert(bytes.end(), p, p + values->size() * sizeof(float));
    }
    return frontend::sha256_hex(frontend::sha256(bytes));
}

std::vector<std::uint8_t> build(const Spec& spec) {
    Writer w;
    if (spec.magic) {
        w.bytes.insert(w.bytes.end(), {'G', 'G', 'U', 'F'});
    } else {
        w.bytes.insert(w.bytes.end(), {'G', 'G', 'M', 'L'});
    }
    w.pod<std::uint32_t>(3);
    w.pod<std::uint64_t>(spec.tensors.size());
    w.pod<std::uint64_t>(spec.meta.size());
    for (const auto& [key, value] : spec.meta) {
        w.text(key);
        if (const auto* s = std::get_if<std::string>(&value)) {
            w.pod<std::uint32_t>(8);
            w.text(*s);
        } else if (const auto* u = std::get_if<std::uint32_t>(&value)) {
            w.pod<std::uint32_t>(4);
            w.pod<std::uint32_t>(*u);
        } else if (const auto* f = std::get_if<float>(&value)) {
            w.pod<std::uint32_t>(6);
            w.pod<float>(*f);
        } else {
            w.pod<std::uint32_t>(7);
            w.pod<std::uint8_t>(std::get<bool>(value) ? 1 : 0);
        }
    }
    std::uint64_t offset = 0;
    for (const auto& [name, values] : spec.tensors) {
        w.text(name);
        w.pod<std::uint32_t>(spec.dims);
        w.pod<std::uint64_t>(values.size());
        w.pod<std::uint32_t>(spec.type);
        w.pod<std::uint64_t>(offset);
        offset += values.size() * sizeof(float);
    }
    while (w.bytes.size() % 32 != 0) w.bytes.push_back(0);
    for (const auto& [name, values] : spec.tensors) {
        const auto* p = reinterpret_cast<const std::uint8_t*>(values.data());
        w.bytes.insert(w.bytes.end(), p, p + values.size() * sizeof(float));
    }
    w.bytes.resize(w.bytes.size() - std::min(spec.cut, w.bytes.size()));
    return w.bytes;
}

std::filesystem::path g_dir;
int g_files = 0;

std::filesystem::path write(const Spec& spec) {
    const auto path = g_dir / ("v" + std::to_string(g_files++) + ".gguf");
    const std::vector<std::uint8_t> bytes = build(spec);
    std::ofstream out(path, std::ios::binary);
    out.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    return path;
}

ControlVectorTarget target(Architecture architecture = Architecture::Qwen3_5) {
    return {.architecture = architecture, .hidden = kHidden, .layers = kLayers};
}

ControlVector load(const Spec& spec, Architecture architecture = Architecture::Qwen3_5) {
    return load_control_vector(GraftSource{.name = "cv", .path = write(spec)}, target(architecture));
}

// The refusal text, or empty when the file was accepted.
std::string refusal(const Spec& spec, Architecture architecture = Architecture::Qwen3_5) {
    try {
        (void)load(spec, architecture);
    } catch (const std::invalid_argument& error) { return error.what(); }
    return {};
}

// A valid GLP project file: layers 1 and 3.
Spec glp_project() {
    Spec spec;
    spec.set("glp.spec_version", std::uint32_t{1})
        .set("glp.mode", std::string("project"))
        .set("glp.rank", std::uint32_t{1})
        .set("glp.alpha_default", 1.5f)
        .set("glp.hook_point", std::string("residual_stream_post_layer"))
        .set("glp.derived_at", std::string("residual_stream_post_layer"))
        .set("glp.layer_ids_zero_based", std::string("1,3"))
        .set("controlvector.layer_count", std::uint32_t{2});
    spec.tensors = {{"direction.1", ramp(3.0f)}, {"direction.3", ramp(-0.5f)}};
    spec.set("glp.content_sha256", content_sha256(spec.tensors));
    return spec;
}

double norm_of(const ControlVector& v, int layer) {
    double norm = 0.0;
    for (int i = 0; i < kHidden; ++i) {
        const double x = v.directions[static_cast<std::size_t>(layer) * kHidden + static_cast<std::size_t>(i)];
        norm += x * x;
    }
    return std::sqrt(norm);
}

int test_glp_project() {
    const ControlVector v = load(glp_project());
    int failures          = 0;
    failures += check(v.format == ControlVectorFormat::Glp && v.mode == ControlVectorMode::Project,
                      "a GLP project file did not read as GLP project");
    failures += check(v.active == std::vector<std::uint8_t>{0, 1, 0, 1, 0, 0},
                      "direction.N did not apply at zero-based layer N");
    failures += check(v.active_layers() == 2, "active_layers miscounted");
    failures += check(std::fabs(norm_of(v, 1) - 1.0) < 1e-6 && std::fabs(norm_of(v, 3) - 1.0) < 1e-6,
                      "a project direction was not normalised");
    failures += check(norm_of(v, 0) == 0.0 && norm_of(v, 5) == 0.0, "an unsteered layer holds a direction");
    failures += check(v.directions[3 * kHidden] < 0.0f, "normalisation lost a direction's sign");
    failures += check(v.project_coefficient(1) == 1.5f && v.add_coefficient(1) == 0.0f &&
                          v.project_coefficient(0) == 0.0f,
                      "glp.alpha_default did not become the projection strength");
    failures += check(v.spec_version == 1 && v.hook_point == "residual_stream_post_layer",
                      "provenance was not kept");
    failures += check(v.digest != 0, "the digest is zero");
    failures += check(v.describe().find("project") != std::string::npos, "describe() does not name the mode");
    return failures;
}

int test_glp_scales_and_add() {
    int failures = 0;
    // Version 2: alpha_L = alpha_default * dir_scales[0] * layer_scales[L].
    Spec scaled = glp_project();
    scaled.set("glp.spec_version", std::uint32_t{2})
        .set("glp.dir_scales", std::string("2"))
        .set("glp.layer_scales", std::string("1:0.5, 3:0.25"));
    const ControlVector s = load(scaled);
    failures += check(s.project_coefficient(1) == 1.5f && s.project_coefficient(3) == 0.75f,
                      "dir_scales and layer_scales did not multiply alpha_default");
    // A layer_scales entry for an unsteered layer is harmless; one past the model is not.
    Spec beyond = glp_project();
    beyond.set("glp.spec_version", std::uint32_t{2}).set("glp.layer_scales", std::string("9:1"));
    failures += check(!refusal(beyond).empty(), "a layer_scales entry past the model was accepted");

    // glp.mode add: stored values, not normalised, scaled by alpha.
    Spec add = glp_project();
    add.set("glp.mode", std::string("add"));
    const ControlVector a = load(add);
    failures += check(a.mode == ControlVectorMode::Add && a.add_coefficient(3) == 1.5f &&
                          a.project_coefficient(3) == 0.0f,
                      "a GLP add file did not read as additive at alpha");
    failures += check(a.directions[1 * kHidden + 7] == 24.0f, "an add vector was rescaled");
    // A GLP file without glp.mode is additive (the spec's legacy reading).
    Spec unmoded = glp_project();
    unmoded.erase("glp.mode");
    failures += check(load(unmoded).mode == ControlVectorMode::Add, "a GLP file without glp.mode was not additive");
    // Project and add of the same tensors are different files: different digests.
    failures += check(a.digest != load(glp_project()).digest, "two different files share a digest");
    return failures;
}

int test_llama_cpp() {
    Spec spec;
    spec.tensors = {{"direction.2", ramp(0.25f)}, {"direction.5", std::vector<float>(kHidden, 0.0f)}};
    const ControlVector v = load(spec);
    int failures          = 0;
    failures += check(v.format == ControlVectorFormat::LlamaCpp && v.mode == ControlVectorMode::Add,
                      "a plain controlvector file did not read as llama.cpp additive");
    failures += check(v.add_coefficient(2) == 1.0f && v.add_coefficient(5) == 1.0f && v.add_coefficient(1) == 0.0f,
                      "a llama.cpp vector's strength is not 1 on its layers");
    failures += check(v.directions[2 * kHidden + 3] == 1.0f, "a llama.cpp vector was rescaled");
    failures += check(v.active[5] == 1, "an additive zero vector was refused or dropped");
    // No model hint is fine; a Qwen3.5 MoE target takes its own hints only.
    Spec unhinted = spec;
    unhinted.erase("controlvector.model_hint");
    failures += check(refusal(unhinted).empty(), "a file without a model hint was refused");
    failures += check(!refusal(spec, Architecture::Qwen3_5Moe).empty(),
                      "a qwen3_5 vector loaded on a Qwen3.5 MoE model");
    Spec moe = spec;
    moe.set("controlvector.model_hint", std::string("qwen35moe"));
    failures += check(refusal(moe, Architecture::Qwen3_5Moe).empty(), "a qwen35moe vector was refused on MoE");
    return failures;
}

int test_refusals() {
    int failures     = 0;
    const auto refuse = [&](Spec spec, const std::string& what, const std::string& needle) {
        const std::string text = refusal(spec);
        failures += check(!text.empty(), what + " was accepted");
        if (!text.empty() && !needle.empty()) {
            failures += check(text.find(needle) != std::string::npos,
                              what + ": the refusal does not say why: " + text);
        }
    };
    refuse(glp_project().set("general.architecture", std::string("llama")), "another architecture", "controlvector");
    refuse(glp_project().set("controlvector.model_hint", std::string("qwen4exp")), "another model's hint", "made for");
    refuse(glp_project().set("glp.mode", std::string("ablate")), "an unknown mode", "glp.mode");
    refuse(glp_project().set("glp.spec_version", std::uint32_t{4}), "an unknown spec version", "spec_version");
    refuse(glp_project().set("glp.hook_point", std::string("attn_out")), "another hook point", "hook_point");
    refuse(glp_project().set("glp.rank", std::uint32_t{2}), "a rank-2 file", "rank");
    refuse(glp_project().set("glp.dir_scales", std::string("2")), "version-2 keys in version 1", "spec_version 2");
    refuse(glp_project().set("glp.layer_ids_zero_based", std::string("1,4")), "a disagreeing layer list",
           "layer_ids_zero_based");
    refuse(glp_project().set("glp.layer_ids_zero_based", std::string("")), "an empty layer list",
           "layer_ids_zero_based");
    refuse(glp_project().set("glp.layer_ids_zero_based", std::string("  ")), "a blank layer list",
           "layer_ids_zero_based");
    refuse(glp_project().set("controlvector.layer_count", std::uint32_t{3}), "a wrong layer count", "layer_count");
    refuse(glp_project().set("glp.content_sha256", std::string(64, 'a')), "a wrong content digest", "content_sha256");
    refuse(glp_project().set("glp.content_sha256", std::string("")), "an empty content digest", "content_sha256");
    refuse(glp_project().set("glp.content_sha256", std::string(64, 'g')), "a non-hex content digest", "content_sha256");
    {
        Spec dspark;
        dspark.set("dspark.mode", std::string("project"));
        dspark.tensors = {{"direction.1", ramp(1.0f)}};
        refuse(dspark, "a dspark-only file", "dspark");
    }
    const auto with_tensors = [](std::vector<std::pair<std::string, std::vector<float>>> tensors) {
        Spec spec;
        spec.tensors = std::move(tensors);
        return spec;
    };
    refuse(with_tensors({{"direction.0", ramp(1.0f)}}), "direction.0", "direction.0");
    refuse(with_tensors({{"direction.6", ramp(1.0f)}}), "a layer past the model", "layers 1 to 5");
    refuse(with_tensors({{"direction.1", std::vector<float>(kHidden + 1, 1.0f)}}), "a wrong width", "hidden size");
    refuse(with_tensors({{"direction.1", ramp(1.0f)}, {"direction.1", ramp(2.0f)}}), "a duplicate layer", "twice");
    refuse(with_tensors({{"direction.1.0", ramp(1.0f)}}), "a subspace tensor", "rank");
    refuse(with_tensors({{"steer.1", ramp(1.0f)}}), "a misnamed tensor", "direction.N");
    refuse(with_tensors({{"direction.x", ramp(1.0f)}}), "a non-numeric layer", "direction.N");
    {
        std::vector<float> bad = ramp(1.0f);
        bad[2]                 = std::numeric_limits<float>::quiet_NaN();
        refuse(with_tensors({{"direction.1", bad}}), "a NaN", "finite");
    }
    {
        Spec zero = glp_project();
        zero.tensors[0].second.assign(kHidden, 0.0f);
        zero.set("glp.content_sha256", content_sha256(zero.tensors));
        refuse(zero, "a zero project direction", "zero vector");
    }
    {
        Spec f16 = with_tensors({{"direction.1", ramp(1.0f)}});
        f16.type = 1;
        refuse(f16, "an F16 tensor", "F32");
        Spec two_d = with_tensors({{"direction.1", ramp(1.0f)}});
        two_d.dims = 2;
        refuse(two_d, "a 2-D tensor", "dimensions");
        Spec cut = with_tensors({{"direction.1", ramp(1.0f)}});
        cut.cut  = 4;
        refuse(cut, "a truncated file", "past the end");
        Spec empty;
        refuse(empty, "a file without tensors", "tensors");
        Spec magic = with_tensors({{"direction.1", ramp(1.0f)}});
        magic.magic = false;
        refuse(magic, "a file without the GGUF magic", "GGUF");
    }
    return failures;
}

int test_detection() {
    int failures = 0;
    Spec spec;
    spec.tensors              = {{"direction.1", ramp(1.0f)}};
    const auto gguf           = write(spec);
    const auto other          = g_dir / "graft.bin";
    std::ofstream(other, std::ios::binary) << "SAFETENSORS-ish";
    const auto missing        = g_dir / "missing.gguf";
    failures += check(is_control_vector_file(gguf), "a GGUF file is not a control vector");
    failures += check(!is_control_vector_file(other), "a non-GGUF file is a control vector");
    failures += check(!is_control_vector_file(missing), "a missing file is a control vector");
    const std::vector<GraftSource> sources = {
        {.name = "persona", .path = other}, {.name = "cv", .path = gguf}, {.name = "gone", .path = missing}};
    const auto vectors = control_vector_sources(sources);
    const auto prompts = prompt_graft_sources(sources);
    failures += check(vectors.size() == 1 && vectors[0].name == "cv", "control_vector_sources picked wrongly");
    failures += check(prompts.size() == 2 && prompts[0].name == "persona" && prompts[1].name == "gone",
                      "prompt_graft_sources picked wrongly");
    return failures;
}

// The published Qwen3.8-27B refusal vector, when the environment names it.
int test_real_file() {
    const char* path = std::getenv("NINFER_TEST_CONTROL_VECTOR");
    if (path == nullptr || *path == '\0') {
        std::cout << "NINFER_TEST_CONTROL_VECTOR unset: real-file check skipped\n";
        return 0;
    }
    const ControlVector v = load_control_vector(GraftSource{.name = "uncensored", .path = path},
                                                {.architecture = Architecture::Qwen3_5, .hidden = 5120, .layers = 64});
    int failures = 0;
    failures += check(v.format == ControlVectorFormat::Glp && v.mode == ControlVectorMode::Project,
                      "the real vector is not GLP project");
    std::cout << v.describe() << '\n';
    for (int l = 0; l < 64; ++l) {
        const bool steered = v.active[static_cast<std::size_t>(l)] != 0;
        if (steered && std::fabs(v.project_coefficient(l) - v.alpha_default) > 0.0f) {
            failures += check(false, "layer " + std::to_string(l) + " is not at alpha_default");
        }
    }
    // A wrong model shape is refused.
    bool refused = false;
    try {
        (void)load_control_vector(GraftSource{.name = "uncensored", .path = path},
                                  {.architecture = Architecture::Qwen3_5, .hidden = 4096, .layers = 64});
    } catch (const std::invalid_argument&) { refused = true; }
    failures += check(refused, "the real vector loaded on a 4096-wide model");
    refused = false;
    try {
        (void)load_control_vector(GraftSource{.name = "uncensored", .path = path},
                                  {.architecture = Architecture::Qwen3_5, .hidden = 5120, .layers = 40});
    } catch (const std::invalid_argument&) { refused = true; }
    failures += check(refused, "the real vector loaded on a 40-layer model");
    return failures;
}

} // namespace

int main() {
    g_dir = std::filesystem::temp_directory_path() / "ninfer-control-vector-test";
    std::filesystem::remove_all(g_dir);
    std::filesystem::create_directories(g_dir);
    int failures = 0;
    failures += test_glp_project();
    failures += test_glp_scales_and_add();
    failures += test_llama_cpp();
    failures += test_refusals();
    failures += test_detection();
    failures += test_real_file();
    std::filesystem::remove_all(g_dir);
    if (failures != 0) {
        std::cerr << failures << " failure(s)\n";
        return 1;
    }
    std::cout << "control vector reader: all checks passed\n";
    return 0;
}
