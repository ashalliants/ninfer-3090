#pragma once

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <string>

// The one-shot CLI's entry point, as a function. The `ninfer` executable calls it from a one-line
// main(); `ninfer-serve` calls it when it is started under the name `ninfer`, so a release archive
// can ship the CLI as a link to the server instead of a second copy of every kernel.
int ninfer_cli_main(int argc, char** argv);

// True when `argv0` names the executable `ninfer` or `ninfer.exe` (any directory, any case). Only
// the file name decides: `ninfer-serve`, `ninfer-cli` and the like are not the CLI.
inline bool ninfer_started_as_cli(const char* argv0) {
    if (argv0 == nullptr) { return false; }
    std::string name = std::filesystem::path(argv0).filename().string();
    std::transform(name.begin(), name.end(), name.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return name == "ninfer" || name == "ninfer.exe";
}
