#include "serve/serve_config_file.h"

#include <nlohmann/json.hpp>

#include <fstream>
#include <sstream>
#include <stdexcept>

namespace ninfer::serve {
namespace {

using Json = nlohmann::ordered_json; // the file's order, so the words read as it is written

std::invalid_argument refuse(const std::filesystem::path& path, const std::string& what) {
    return std::invalid_argument(path.string() + ": " + what);
}

// One value as a command-line word: a string as written, a number as JSON writes it.
std::string value_word(const Json& value) {
    return value.is_string() ? value.get<std::string>() : value.dump();
}

} // namespace

bool repeatable_serve_option(std::string_view key) {
    // The flags parse_serve_options appends to a list on every occurrence. A flag missing here
    // cannot take an array, which fails loudly rather than dropping values.
    return key == "graft" || key == "allowed-host" || key == "context-store-peer";
}

ConfigWords read_serve_config(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) { throw refuse(path, "cannot read the config file"); }
    std::stringstream text;
    text << file.rdbuf();
    Json root;
    try {
        root = Json::parse(text.str());
    } catch (const Json::parse_error& error) {
        throw refuse(path, std::string("not valid JSON: ") + error.what());
    }
    if (!root.is_object()) {
        throw refuse(path, "must hold a JSON object of serve options (flag names without dashes)");
    }
    ConfigWords out;
    const auto add = [&](const std::string& key, std::string word, bool value) {
        out.words.push_back(std::move(word));
        out.keys.push_back(key);
        out.values.push_back(value);
    };
    for (const auto& [key, value] : root.items()) {
        if (key.empty()) { throw refuse(path, "empty key"); }
        if (key.front() == '-') {
            throw refuse(path, "\"" + key + "\" must be written without leading dashes");
        }
        // The key is a secret: it belongs on the command line (or a launcher's environment), never
        // in a file that is copied, shared and checked in. The value is never repeated.
        if (key == "api-key") {
            throw refuse(path, "\"api-key\" cannot be set in a config file; pass --api-key on the "
                               "command line");
        }
        if (key == "config") { throw refuse(path, "\"config\" cannot name another config file"); }
        const std::string flag = "--" + key;
        if (value.is_boolean()) {
            // True is the bare flag; false leaves the option at its default.
            if (value.get<bool>()) {
                add(key, flag, false);
            } else {
                out.false_keys.push_back(key);
            }
        } else if (value.is_array()) {
            if (!repeatable_serve_option(key)) {
                throw refuse(path, "\"" + key + "\" keeps only its last value and cannot take an "
                                   "array");
            }
            for (const Json& element : value) {
                if (!element.is_string() && !element.is_number()) {
                    throw refuse(path, "\"" + key + "\" takes an array of strings or numbers");
                }
                add(key, flag, false);
                add(key, value_word(element), true);
            }
        } else if (value.is_string() || value.is_number()) {
            add(key, flag, false);
            add(key, value_word(value), true);
        } else {
            throw refuse(path, "\"" + key + "\" takes a string, a number, true, false or (for a "
                               "repeatable option) an array, not " +
                                   (value.is_null() ? "null" : "an object"));
        }
    }
    return out;
}

} // namespace ninfer::serve
