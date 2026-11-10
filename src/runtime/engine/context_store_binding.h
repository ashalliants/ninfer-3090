#pragma once

#include "ninfer/types.h"

#include <cstdint>
#include <filesystem>
#include <string>
#include <string_view>

namespace ninfer {

// The image format every binding names: bump it together with kImageVersion in
// models/qwen3_5/program/storage/checkpoint_image.cpp. Format 2 carries the StateImage geometry
// (GDN state type, DFlash local rings) explicitly instead of only its total size.
inline constexpr std::string_view kCheckpointImageFormat = "checkpoint-image-2";

// What a stored checkpoint image binds to: the model, its weight formats and quantization, the
// artifact file, the KV, recurrent-state and speculative settings its layout was built for, and the
// image format. Images under any other binding are never read, so they age out of the store as
// misses.
//
// The speculative part is the backend, draft window and proposal head. A DFlash or DFlash2 image
// holds the draft model's own context (the local K/V rings in each StateImage, and DFlash's draft KV
// pages), computed by the artifact's draft weights, so it is restored only into the same backend
// and artifact. Restoring just the target part is not offered: the draft context could only be
// rebuilt by prefilling the whole history again, which is what a miss does anyway. N-gram copy
// drafting is deliberately absent: its index is rebuilt from each request's own tokens and no
// checkpoint holds any of it.
inline std::string context_store_binding(const EngineOptions& options, const LoadSummary& load) {
    std::string binding = load.architecture + '\n' + load.model_name + '\n';
    for (const std::string& format : load.weight_formats) { binding += format + ','; }
    binding += '\n' + load.prefill_signature + '\n';
    binding += std::to_string(static_cast<unsigned>(options.kv_cache)) + ',' +
               std::to_string(static_cast<unsigned>(options.speculative.backend)) + ',' +
               std::to_string(options.speculative.draft_tokens) + ',' +
               std::to_string(static_cast<unsigned>(options.speculative.proposal_head)) + ',' +
               (options.gdn_state_fp16 ? "gdn-fp16" : "gdn-fp32") + ',' +
               (options.enable_vision ? "vision" : "text") + '\n';
    // The artifact by the id its conversion wrote: a re-converted artifact never matches, and every
    // copy of the same one does, on any machine. Not its file time: a download stamps the copy with
    // when it finished, so no two machines sharing a bucket would ever agree.
    binding += load.artifact_id;
    // Prompt grafts: a stored session that starts from a graft holds that graft's installed K/V and
    // state under its placeholder ids, so a changed graft file must not match it. Any change to
    // the configured grafts makes the store's images misses.
    for (const GraftSource& graft : options.grafts) {
        std::error_code graft_size_error;
        std::error_code graft_time_error;
        const std::uintmax_t graft_size = std::filesystem::file_size(graft.path, graft_size_error);
        const auto graft_written = std::filesystem::last_write_time(graft.path, graft_time_error);
        binding += "\ngraft " + graft.name + ' ' +
                   (graft_size_error ? std::string("?") : std::to_string(graft_size)) + ' ' +
                   (graft_time_error ? std::string("?")
                                     : std::to_string(graft_written.time_since_epoch().count()));
    }
    // The checkpoint image format; stores written by an older engine become misses.
    binding += '\n';
    binding += kCheckpointImageFormat;
    return binding;
}

} // namespace ninfer
