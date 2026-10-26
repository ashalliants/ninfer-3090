#pragma once

// ninfer-serve --config FILE: serve options written down as a JSON object, read into the words the
// command line would hold. The option parser (serve_options.cpp) puts these words before the
// command line's own, so a command-line option wins over the file's, and the file over the
// defaults. Which keys are options is the parser's to say: this reader checks only the file's
// shape.

#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

namespace ninfer::serve {

struct ConfigWords {
    // Per entry, in the file's order: "--key", then its value when it has one. A string or number
    // is one value, true is the bare flag, false leaves the key out, and an array repeats the flag
    // once per element.
    std::vector<std::string> words;
    // The key each word came from (parallel to `words`).
    std::vector<std::string> keys;
    // Whether each word is a value rather than a flag (parallel to `words`).
    std::vector<bool> values;
    // Keys set to false: they add no word, but are still options the parser must know.
    std::vector<std::string> false_keys;
};

// Throws std::invalid_argument naming the file for one that cannot be read or is not a JSON object,
// and for a key with leading dashes, a nested object, a null, an array of anything but strings and
// numbers, an array for an option that keeps only its last value, and the keys api-key (a secret
// belongs on the command line or in the environment, not in a file) and config (no nesting).
[[nodiscard]] ConfigWords read_serve_config(const std::filesystem::path& path);

// The serve options that collect every occurrence instead of keeping the last (--graft,
// --allowed-host, --context-store-peer), named without dashes: the only keys an array may set.
[[nodiscard]] bool repeatable_serve_option(std::string_view key);

} // namespace ninfer::serve
