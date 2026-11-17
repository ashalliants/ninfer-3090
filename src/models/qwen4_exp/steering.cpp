#include "models/qwen4_exp/steering.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <string_view>
#include <system_error>
#include <unordered_set>

namespace ninfer::models::qwen4_exp {
namespace {

// A control vector is tens of kilobytes to a few megabytes; anything near this is not one.
constexpr std::uint64_t kMaxFileBytes   = 64ULL << 20;
constexpr std::uint64_t kMaxStringBytes = 1ULL << 16;
constexpr std::uint32_t kGgufTypeF32    = 0;

// GGUF value types (the format's gguf_type).
enum GgufValue : std::uint32_t {
    U8     = 0,
    I8     = 1,
    U16    = 2,
    I16    = 3,
    U32    = 4,
    I32    = 5,
    F32    = 6,
    Bool   = 7,
    String = 8,
    Array  = 9,
    U64    = 10,
    I64    = 11,
    F64    = 12
};

std::uint64_t scalar_bytes(std::uint32_t type) {
    switch (type) {
    case U8:
    case I8:
    case Bool:
        return 1;
    case U16:
    case I16:
        return 2;
    case U32:
    case I32:
    case F32:
        return 4;
    case U64:
    case I64:
    case F64:
        return 8;
    default:
        return 0;
    }
}

class Cursor {
public:
    Cursor(const std::vector<std::uint8_t>& bytes, const std::string& who)
        : bytes_(bytes), who_(who) {}

    [[nodiscard]] std::uint64_t position() const noexcept { return at_; }

    void skip(std::uint64_t count) {
        need(count);
        at_ += count;
    }

    template <typename T>
    T read() {
        need(sizeof(T));
        T value;
        std::memcpy(&value, bytes_.data() + at_, sizeof(T));
        at_ += sizeof(T);
        return value;
    }

    std::string string() {
        const auto length = read<std::uint64_t>();
        if (length > kMaxStringBytes) {
            fail("a string is " + std::to_string(length) + " bytes long");
        }
        need(length);
        std::string out(reinterpret_cast<const char*>(bytes_.data() + at_),
                        static_cast<std::size_t>(length));
        at_ += length;
        return out;
    }

    [[noreturn]] void fail(const std::string& what) const {
        throw std::invalid_argument("graft '" + who_ +
                                    "' is not a usable Qwen3.8-Flash-Next control vector: " + what);
    }

private:
    void need(std::uint64_t count) const {
        if (count > bytes_.size() || at_ > bytes_.size() - count) {
            fail("the file ends inside its header");
        }
    }

    const std::vector<std::uint8_t>& bytes_;
    const std::string& who_;
    std::uint64_t at_ = 0;
};

// Skips one metadata value of `type`; returns it as text when it is a string.
std::string read_value(Cursor& in, std::uint32_t type, int depth = 0) {
    if (type == String) { return in.string(); }
    if (type == Array) {
        if (depth != 0) { in.fail("nested metadata arrays"); }
        const auto element = in.read<std::uint32_t>();
        const auto count   = in.read<std::uint64_t>();
        if (element == String) {
            for (std::uint64_t i = 0; i < count; ++i) { (void)in.string(); }
        } else {
            const std::uint64_t width = scalar_bytes(element);
            if (width == 0) {
                in.fail("a metadata array of unknown type " + std::to_string(element));
            }
            if (count > std::numeric_limits<std::uint64_t>::max() / width) {
                in.fail("a metadata array is too long");
            }
            in.skip(count * width);
        }
        return {};
    }
    const std::uint64_t width = scalar_bytes(type);
    if (width == 0) { in.fail("metadata of unknown type " + std::to_string(type)); }
    in.skip(width);
    return {};
}

std::uint64_t fnv1a(const std::vector<std::uint8_t>& bytes) {
    std::uint64_t hash = 14695981039346656037ULL;
    for (const std::uint8_t byte : bytes) {
        hash ^= byte;
        hash *= 1099511628211ULL;
    }
    return hash == 0 ? 1 : hash;
}

struct TensorInfo {
    std::int32_t layer   = 0;
    std::uint64_t offset = 0;
};

} // namespace

std::int32_t SteeringVector::active_layers() const noexcept {
    return static_cast<std::int32_t>(std::count(active.begin(), active.end(), std::uint8_t{1}));
}

SteeringVector load_steering_vector(const GraftSource& source, std::int32_t hidden,
                                    std::int32_t layers) {
    const std::string& who = source.name;
    auto refuse            = [&](const std::string& what) -> void {
        throw std::invalid_argument("graft '" + who +
                                               "' is not a usable Qwen3.8-Flash-Next control vector: " + what);
    };

    std::error_code error;
    const std::uint64_t size = std::filesystem::file_size(source.path, error);
    if (error) {
        throw std::invalid_argument("graft '" + who + "': cannot read " + source.path.string() +
                                    ": " + error.message());
    }
    if (size > kMaxFileBytes) {
        refuse("the file is " + std::to_string(size) + " bytes; a control vector is far smaller");
    }
    std::vector<std::uint8_t> bytes(static_cast<std::size_t>(size));
    {
        std::ifstream file(source.path, std::ios::binary);
        if (!file.read(reinterpret_cast<char*>(bytes.data()),
                       static_cast<std::streamsize>(bytes.size()))) {
            throw std::invalid_argument("graft '" + who + "': cannot read " + source.path.string());
        }
    }

    Cursor in(bytes, who);
    if (bytes.size() < 4 || std::memcmp(bytes.data(), "GGUF", 4) != 0) {
        refuse("it is not a GGUF file (a Qwen3.8-Flash-Next --graft is a GGUF control vector)");
    }
    in.skip(4);
    const auto version = in.read<std::uint32_t>();
    if (version != 2 && version != 3) {
        refuse("GGUF version " + std::to_string(version) + " is not 2 or 3");
    }
    const auto tensor_count   = in.read<std::uint64_t>();
    const auto metadata_count = in.read<std::uint64_t>();
    if (tensor_count == 0 || tensor_count > 4096) {
        refuse(std::to_string(tensor_count) + " tensors");
    }
    if (metadata_count > 4096) { refuse(std::to_string(metadata_count) + " metadata entries"); }

    std::string architecture, model_hint;
    std::uint64_t alignment = 32;
    for (std::uint64_t i = 0; i < metadata_count; ++i) {
        const std::string key = in.string();
        const auto type       = in.read<std::uint32_t>();
        if (key == "general.alignment" && type == U32) {
            alignment = in.read<std::uint32_t>();
            continue;
        }
        const std::string value = read_value(in, type);
        if (key == "general.architecture") { architecture = value; }
        if (key == "controlvector.model_hint") { model_hint = value; }
    }
    if (architecture != "controlvector") {
        refuse("general.architecture is '" + architecture + "', not 'controlvector'");
    }
    if (!model_hint.empty() && model_hint != "qwen4exp") {
        refuse("it was made for '" + model_hint + "', not for qwen4exp");
    }
    if (alignment == 0 || alignment > 4096 || (alignment & (alignment - 1)) != 0) {
        refuse("general.alignment " + std::to_string(alignment) +
               " is not a power of two up to 4096");
    }

    std::vector<TensorInfo> tensors;
    std::unordered_set<std::int32_t> seen;
    constexpr std::string_view prefix = "direction.";
    for (std::uint64_t i = 0; i < tensor_count; ++i) {
        const std::string name = in.string();
        const auto dims        = in.read<std::uint32_t>();
        if (dims != 1) {
            refuse("tensor '" + name + "' has " + std::to_string(dims) + " dimensions, not 1");
        }
        const auto width  = in.read<std::uint64_t>();
        const auto type   = in.read<std::uint32_t>();
        const auto offset = in.read<std::uint64_t>();
        if (name.compare(0, prefix.size(), prefix) != 0 || name.size() == prefix.size() ||
            name.find_first_not_of("0123456789", prefix.size()) != std::string::npos ||
            name.size() > prefix.size() + 6) {
            refuse("tensor '" + name + "' is not named direction.N");
        }
        const std::int32_t layer = std::stoi(name.substr(prefix.size()));
        if (layer < 1 || layer >= layers) {
            refuse("tensor '" + name + "' is layer " + std::to_string(layer) +
                   "; this model has layers 1 to " + std::to_string(layers - 1));
        }
        if (type != kGgufTypeF32) { refuse("tensor '" + name + "' is not F32"); }
        if (width != static_cast<std::uint64_t>(hidden)) {
            refuse("tensor '" + name + "' has " + std::to_string(width) +
                   " values; the model's hidden size is " + std::to_string(hidden));
        }
        if (!seen.insert(layer).second) {
            refuse("layer " + std::to_string(layer) + " appears twice");
        }
        tensors.push_back({layer, offset});
    }

    const std::uint64_t base = (in.position() + alignment - 1) / alignment * alignment;
    const std::uint64_t row  = static_cast<std::uint64_t>(hidden) * sizeof(float);

    SteeringVector out;
    out.name   = who;
    out.hidden = hidden;
    out.layers = layers;
    out.directions.assign(static_cast<std::size_t>(layers) * hidden, 0.0f);
    out.active.assign(static_cast<std::size_t>(layers), 0);
    for (const TensorInfo& tensor : tensors) {
        if (tensor.offset > bytes.size() || base > bytes.size() - tensor.offset ||
            row > bytes.size() - base - tensor.offset) {
            refuse("tensor direction." + std::to_string(tensor.layer) +
                   " lies past the end of the file");
        }
        float* destination =
            out.directions.data() + static_cast<std::size_t>(tensor.layer) * hidden;
        std::memcpy(destination, bytes.data() + base + tensor.offset,
                    static_cast<std::size_t>(row));
        double norm = 0.0;
        for (std::int32_t i = 0; i < hidden; ++i) {
            if (!std::isfinite(destination[i])) {
                refuse("direction." + std::to_string(tensor.layer) +
                       " holds a value that is not finite");
            }
            norm += static_cast<double>(destination[i]) * destination[i];
        }
        norm = std::sqrt(norm);
        if (!(norm > 0.0) || !std::isfinite(norm)) {
            refuse("direction." + std::to_string(tensor.layer) + " is the zero vector");
        }
        for (std::int32_t i = 0; i < hidden; ++i) {
            destination[i] = static_cast<float>(destination[i] / norm);
        }
        out.active[static_cast<std::size_t>(tensor.layer)] = 1;
    }
    out.digest = fnv1a(bytes);
    return out;
}

} // namespace ninfer::models::qwen4_exp
