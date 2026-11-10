#include "runtime/engine/context_store_binding.h"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

namespace {

int check(bool condition, const char* message) {
    if (condition) { return 0; }
    std::cerr << message << '\n';
    return 1;
}

// A unique directory removed on scope exit, including when a check throws.
struct TempDir {
    std::filesystem::path path;
    TempDir()
        : path(std::filesystem::temp_directory_path() /
               ("ninfer_context_store_binding_" +
                std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()))) {
        std::filesystem::create_directories(path);
    }
    ~TempDir() {
        std::error_code ignored;
        std::filesystem::remove_all(path, ignored);
    }
    TempDir(const TempDir&)            = delete;
    TempDir& operator=(const TempDir&) = delete;
};

} // namespace

// A stored session binds to the artifact's id (LoadSummary::artifact_id, which construct_model and
// construct_qwen4_exp fill from the header id), never to the artifact file's time or size.
int main() {
    using ninfer::context_store_binding;
    int failures = 0;
    ninfer::LoadSummary load;
    load.architecture = "arch";
    load.model_name   = "model";
    load.artifact_id  = "71" + std::string(30, '0');

    // A directory of its own, named by the clock, so a parallel run never shares or removes it.
    const TempDir dir;
    const auto path = dir.path / "artifact.bin";
    { std::ofstream(path) << "x"; }
    ninfer::EngineOptions options;
    options.artifact_path = path;
    const std::string written = context_store_binding(options, load);

    // A copy of the same artifact: its file time differs (a download stamps the finish time) and
    // its id is the same, so the binding matches.
    std::filesystem::last_write_time(path, std::filesystem::last_write_time(path) -
                                               std::chrono::hours(24 * 365));
    failures += check(context_store_binding(options, load) == written,
                      "a copy of the artifact with another file time changed the binding");

    // A different conversion has a different id, so the binding differs.
    ninfer::LoadSummary other = load;
    other.artifact_id         = "72" + std::string(30, '0');
    failures += check(context_store_binding(options, other) != written,
                      "a different artifact id shared the binding");

    // An engine path that left the id empty must not match one that filled it.
    other.artifact_id.clear();
    failures += check(context_store_binding(options, other) != written,
                      "an empty artifact id shared the binding of a filled one");

    return failures;
}
