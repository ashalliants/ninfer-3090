// Rewrites a Qwen3.8-Flash-Next server's saved expert state as the default state shipped beside the
// model, for a release.
//
// A running server saves its VRAM expert cache's state (every expert's use count and the resident
// ones, best first) beside the model, tied to that copy of the artifact by path, size and
// modification time. A copy of the state that ships with the model cannot be tied to those, so it
// is tied to the artifact id instead, which every copy of one conversion shares. A fresh install
// then warm-starts from `<model>.expert-state.default` until it has saved a state of its own.
//
// Build: see tools/CMakeLists.txt -- it links ninfer_artifact and compiles expert_state.cpp.
//
//   ninfer-qwen4-exp-expert-state-export <session-state> <out> <model.ninfer>
//       Reads the artifact's id from the file and refuses a session state saved for another copy.
//   ninfer-qwen4-exp-expert-state-export <session-state> <out> --artifact-id <32 hex digits>
//       Takes the id as given and does not check which artifact the session state was saved for.
//
// The session state is `<model>.expert-state` (or what --expert-state named) after a representative
// session. Copy the output to `<model>.expert-state.default` in the release.

#include "artifact/framing.h"
#include "artifact/reader.h"
#include "models/qwen4_exp/program/expert_cache/expert_state.h"

#include <cstdio>
#include <exception>
#include <filesystem>
#include <string>

namespace {

int usage() {
    std::fprintf(stderr,
                 "usage: ninfer-qwen4-exp-expert-state-export <session-state> <out> <model.ninfer>\n"
                 "       ninfer-qwen4-exp-expert-state-export <session-state> <out> --artifact-id <32 hex digits>\n");
    return 2;
}

} // namespace

int main(int argc, char** argv) {
    namespace es = ninfer::models::qwen4_exp::expert_cache;
    try {
        std::string id, expected;
        if (argc == 4) {
            const std::filesystem::path artifact = argv[3];
            const ninfer::artifact::Reader reader(artifact);
            id       = ninfer::artifact::artifact_id_hex(reader.artifact_id());
            expected = es::session_state_identity(artifact);
        } else if (argc == 5 && std::string(argv[3]) == "--artifact-id") {
            id = argv[4];
        } else {
            return usage();
        }
        const std::string error = es::export_expert_state(argv[1], argv[2], id, expected);
        if (!error.empty()) {
            std::fprintf(stderr, "error: %s\n", error.c_str());
            return 1;
        }
        std::printf("wrote %s for artifact id %s\n", argv[2], id.c_str());
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
