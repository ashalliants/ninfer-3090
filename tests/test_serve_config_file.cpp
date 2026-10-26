// ninfer-serve --config FILE, the reader alone (serve/serve_config_file.h), host only:
//   - config_strings_numbers_bools_arrays: strings and numbers become "--key value", true the bare
//     flag, false no word (but a false key), an array the flag once per element; the file's order
//     is kept and every word knows its key and whether it is a value;
//   - config_refuses_api_key: an api-key entry is refused, naming the file and the key;
//   - config_refuses_shapes: config inside a config, a nested object, a null, leading dashes, an
//     empty key, an array of objects, an array for an option that keeps its last value, a file
//     that is not an object, invalid JSON and a missing file are refused, each naming the file;
//   - repeatable_options: --graft, --allowed-host and --context-store-peer collect, others not.

#include "serve/serve_config_file.h"

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using ninfer::serve::ConfigWords;
using ninfer::serve::read_serve_config;
using ninfer::serve::repeatable_serve_option;

int failures = 0;

void check(bool ok, const std::string& what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what.c_str());
        ++failures;
    }
}

std::filesystem::path write_file(const std::string& name, const std::string& text) {
    const std::filesystem::path directory =
        std::filesystem::temp_directory_path() / "ninfer_serve_config_file_test";
    std::filesystem::create_directories(directory);
    const std::filesystem::path path = directory / name;
    std::ofstream(path, std::ios::binary) << text;
    return path;
}

// The message read_serve_config throws for this text, or "" when it reads it.
std::string refusal(const std::string& name, const std::string& text) {
    try {
        (void)read_serve_config(write_file(name, text));
    } catch (const std::invalid_argument& error) { return error.what(); }
    return "";
}

void config_strings_numbers_bools_arrays() {
    const std::filesystem::path path = write_file(
        "shapes.json", R"({"max-context": 65536, "kv-dtype": "int8", "top-p": 0.95, "seed": -1,
                          "vision": true, "cors": false, "graft": ["a=a.safetensors", "b=b"],
                          "allowed-host": []})");
    const ConfigWords config = read_serve_config(path);
    const std::vector<std::string> words{"--max-context", "65536",  "--kv-dtype",
                                         "int8",          "--top-p", "0.95",
                                         "--seed",        "-1",      "--vision",
                                         "--graft",       "a=a.safetensors", "--graft",
                                         "b=b"};
    check(config.words == words, "the words differ from the file's entries");
    const std::vector<std::string> keys{"max-context", "max-context", "kv-dtype", "kv-dtype",
                                        "top-p",       "top-p",       "seed",     "seed",
                                        "vision",      "graft",       "graft",    "graft",
                                        "graft"};
    check(config.keys == keys, "a word names the wrong key");
    const std::vector<bool> values{false, true, false, true, false, true, false,
                                   true,  false, false, true, false, true};
    check(config.values == values, "a word is marked as a value wrongly");
    check(config.false_keys == std::vector<std::string>{"cors"}, "a false key was not reported");
}

void config_refuses_api_key() {
    const std::string message = refusal("secret.json", R"({"api-key": "sk-1", "port": 8080})");
    check(message.find("secret.json") != std::string::npos &&
              message.find("\"api-key\"") != std::string::npos,
          "api-key in a config file was not refused naming the file and key: " + message);
    check(message.find("sk-1") == std::string::npos, "the refusal repeats the key's value");
}

void config_refuses_shapes() {
    const auto refused = [](const std::string& name, const std::string& text,
                            const std::string& needle) {
        const std::string message = refusal(name, text);
        check(!message.empty() && message.find(name) != std::string::npos &&
                  message.find(needle) != std::string::npos,
              name + " was not refused naming the file and \"" + needle + "\": " + message);
    };
    refused("nested-config.json", R"({"config": "other.json"})", "\"config\"");
    refused("nested-object.json", R"({"spec": {"backend": "mtp"}})", "\"spec\"");
    refused("null.json", R"({"port": null})", "\"port\"");
    refused("dashes.json", R"({"--port": 8080})", "\"--port\"");
    refused("empty-key.json", R"({"": 1})", "empty key");
    refused("array-of-objects.json", R"({"graft": [{"name": "a"}]})", "\"graft\"");
    refused("array-of-bools.json", R"({"allowed-host": [true]})", "\"allowed-host\"");
    refused("array-not-repeatable.json", R"({"port": [8080, 8081]})", "\"port\"");
    refused("not-object.json", R"(["--port", "8080"])", "JSON object");
    refused("invalid.json", R"({"port": 8080,})", "JSON");
    const std::string missing =
        (std::filesystem::temp_directory_path() / "ninfer_serve_config_file_test" / "absent.json")
            .string();
    std::filesystem::remove(missing);
    std::string message;
    try {
        (void)read_serve_config(missing);
    } catch (const std::invalid_argument& error) { message = error.what(); }
    check(message.find("absent.json") != std::string::npos,
          "a missing file was not refused: " + message);
    check(refusal("empty.json", "{}").empty(), "an empty object was refused");
}

void repeatable_options() {
    for (const char* key : {"graft", "allowed-host", "context-store-peer"}) {
        check(repeatable_serve_option(key), std::string(key) + " is not repeatable");
    }
    for (const char* key : {"port", "default-graft", "context-store", "--graft", ""}) {
        check(!repeatable_serve_option(key), std::string(key) + " is repeatable");
    }
}

} // namespace

int main() {
    config_strings_numbers_bools_arrays();
    config_refuses_api_key();
    config_refuses_shapes();
    repeatable_options();
    std::filesystem::remove_all(std::filesystem::temp_directory_path() /
                                "ninfer_serve_config_file_test");
    if (failures != 0) {
        std::fprintf(stderr, "%d check(s) failed\n", failures);
        return 1;
    }
    std::puts("serve config file checks passed");
    return 0;
}
