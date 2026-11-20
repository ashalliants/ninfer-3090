#include "models/qwen3_5/frontend/control_vector.h"

#include "models/qwen3_5/frontend/digest.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <locale>
#include <map>
#include <optional>
#include <set>
#include <sstream>
#include <string_view>
#include <system_error>

namespace ninfer::models::qwen3_5 {
namespace {

// A control vector is kilobytes to a few megabytes (1 MB for Qwen3.8-27B's 49 layers); anything
// near this is not one.
constexpr std::uint64_t kMaxFileBytes   = 64ULL << 20;
constexpr std::uint64_t kMaxStringBytes = 1ULL << 20;
constexpr std::uint64_t kMaxEntries     = 4096;
constexpr std::uint32_t kGgmlF32        = 0;
constexpr std::string_view kPostLayer   = "residual_stream_post_layer";

// GGUF metadata value types (gguf_type).
enum GgufType : std::uint32_t {
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

// One metadata value: its type, and its contents when it is a scalar or a string.
struct Value {
    std::uint32_t type = 0;
    std::string text;
    double number = 0.0; // numeric and boolean scalars
    bool integral = false;
};

class Refusal {
public:
    explicit Refusal(const std::string& name) : name_(name) {}

    [[noreturn]] void operator()(const std::string& what) const {
        throw std::invalid_argument("graft '" + name_ + "' is not a usable control vector: " + what);
    }

private:
    const std::string& name_;
};

class Cursor {
public:
    Cursor(const std::vector<std::uint8_t>& bytes, const Refusal& refuse)
        : bytes_(bytes), refuse_(refuse) {}

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
            refuse_("a string is " + std::to_string(length) + " bytes long");
        }
        need(length);
        std::string out(reinterpret_cast<const char*>(bytes_.data() + at_),
                        static_cast<std::size_t>(length));
        at_ += length;
        return out;
    }

    Value value(std::uint32_t type) {
        Value out;
        out.type = type;
        switch (type) {
        case String:
            out.text = string();
            return out;
        case Array: {
            const auto element = read<std::uint32_t>();
            const auto count   = read<std::uint64_t>();
            if (element == String) {
                for (std::uint64_t i = 0; i < count; ++i) { (void)string(); }
                return out;
            }
            if (element == Array) { refuse_("nested metadata arrays"); }
            const std::uint64_t width = scalar_bytes(element);
            if (width == 0) {
                refuse_("a metadata array of unknown type " + std::to_string(element));
            }
            if (count > std::numeric_limits<std::uint64_t>::max() / width) {
                refuse_("a metadata array is too long");
            }
            skip(count * width);
            return out;
        }
        case U8: out.number = read<std::uint8_t>(); out.integral = true; return out;
        case I8: out.number = read<std::int8_t>(); out.integral = true; return out;
        case U16: out.number = read<std::uint16_t>(); out.integral = true; return out;
        case I16: out.number = read<std::int16_t>(); out.integral = true; return out;
        case U32: out.number = read<std::uint32_t>(); out.integral = true; return out;
        case I32: out.number = read<std::int32_t>(); out.integral = true; return out;
        case U64: out.number = static_cast<double>(read<std::uint64_t>()); out.integral = true; return out;
        case I64: out.number = static_cast<double>(read<std::int64_t>()); out.integral = true; return out;
        case F32: out.number = read<float>(); return out;
        case F64: out.number = read<double>(); return out;
        case Bool: out.number = read<std::uint8_t>() != 0 ? 1.0 : 0.0; return out;
        default:
            refuse_("metadata of unknown type " + std::to_string(type));
        }
    }

private:
    void need(std::uint64_t count) const {
        if (count > bytes_.size() || at_ > bytes_.size() - count) {
            refuse_("the file ends inside its header");
        }
    }

    const std::vector<std::uint8_t>& bytes_;
    const Refusal& refuse_;
    std::uint64_t at_ = 0;
};

std::vector<std::uint8_t> read_file(const GraftSource& source) {
    std::error_code error;
    const std::uint64_t size = std::filesystem::file_size(source.path, error);
    if (error) {
        throw std::invalid_argument("graft '" + source.name + "': cannot read " +
                                    source.path.string() + ": " + error.message());
    }
    if (size > kMaxFileBytes) {
        Refusal(source.name)("the file is " + std::to_string(size) +
                             " bytes; a control vector is far smaller");
    }
    std::vector<std::uint8_t> bytes(static_cast<std::size_t>(size));
    std::ifstream file(source.path, std::ios::binary);
    if (!file.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()))) {
        throw std::invalid_argument("graft '" + source.name + "': cannot read " + source.path.string());
    }
    return bytes;
}

// The model hints (controlvector.model_hint) a vector for `architecture` may carry: the Hugging
// Face model_type and the llama.cpp architecture name.
std::vector<std::string> accepted_hints(Architecture architecture) {
    switch (architecture) {
    case Architecture::Qwen3_5:
        return {"qwen3_5", "qwen3_5_text", "qwen35"};
    case Architecture::Qwen3_5Moe:
        return {"qwen3_5_moe", "qwen3_5_moe_text", "qwen35moe"};
    default:
        throw std::invalid_argument("control vectors are for the Qwen3.5 / 3.6 / 3.8 text models");
    }
}

std::string trim(std::string_view text) {
    const auto begin = text.find_first_not_of(" \t");
    if (begin == std::string_view::npos) { return {}; }
    const auto end = text.find_last_not_of(" \t");
    return std::string(text.substr(begin, end - begin + 1));
}

std::vector<std::string> split_commas(std::string_view text) {
    std::vector<std::string> out;
    std::size_t begin = 0;
    while (true) {
        const auto comma = text.find(',', begin);
        out.push_back(trim(text.substr(begin, comma == std::string_view::npos ? comma : comma - begin)));
        if (comma == std::string_view::npos) { return out; }
        begin = comma + 1;
    }
}

std::optional<std::int64_t> parse_integer(const std::string& text) {
    if (text.empty() || text.size() > 9 || text.find_first_not_of("0123456789") != std::string::npos) {
        return std::nullopt;
    }
    return std::stoll(text);
}

std::optional<double> parse_float(const std::string& text) {
    if (text.empty()) { return std::nullopt; }
    std::istringstream in(text);
    in.imbue(std::locale::classic());
    double value = 0.0;
    in >> value;
    if (in.fail() || !in.eof()) { return std::nullopt; }
    return value;
}

struct TensorInfo {
    std::int32_t layer   = 0;
    std::uint64_t offset = 0;
};

} // namespace

std::int32_t ControlVector::active_layers() const noexcept {
    return static_cast<std::int32_t>(std::count(active.begin(), active.end(), std::uint8_t{1}));
}

float ControlVector::project_coefficient(std::int32_t layer) const {
    return mode == ControlVectorMode::Project ? strength.at(static_cast<std::size_t>(layer)) : 0.0f;
}

float ControlVector::add_coefficient(std::int32_t layer) const {
    return mode == ControlVectorMode::Add ? strength.at(static_cast<std::size_t>(layer)) : 0.0f;
}

std::string ControlVector::describe() const {
    std::ostringstream out;
    out.imbue(std::locale::classic());
    out << name << ": " << (format == ControlVectorFormat::Glp ? "GLP" : "llama.cpp")
        << " control vector, " << (mode == ControlVectorMode::Project ? "project" : "add")
        << ", alpha " << alpha_default << ", " << active_layers() << " layers";
    std::int32_t first = -1, last = -1;
    for (std::int32_t l = 0; l < layers; ++l) {
        if (active[static_cast<std::size_t>(l)] == 0) { continue; }
        if (first < 0) { first = l; }
        last = l;
    }
    if (first >= 0) { out << " (" << first << "-" << last << ")"; }
    if (!base_model.empty()) {
        out << ", derived on " << base_model;
        if (!base_revision.empty()) { out << " @ " << base_revision.substr(0, 12); }
    }
    return out.str();
}

bool is_control_vector_file(const std::filesystem::path& path) noexcept {
    try {
        std::ifstream file(path, std::ios::binary);
        char magic[4] = {};
        if (!file.read(magic, 4)) { return false; }
        return std::memcmp(magic, "GGUF", 4) == 0;
    } catch (...) {
        return false;
    }
}

std::vector<GraftSource> control_vector_sources(const std::vector<GraftSource>& grafts) {
    std::vector<GraftSource> out;
    for (const GraftSource& source : grafts) {
        if (is_control_vector_file(source.path)) { out.push_back(source); }
    }
    return out;
}

std::vector<GraftSource> prompt_graft_sources(const std::vector<GraftSource>& grafts) {
    std::vector<GraftSource> out;
    for (const GraftSource& source : grafts) {
        if (!is_control_vector_file(source.path)) { out.push_back(source); }
    }
    return out;
}

ControlVector load_control_vector(const GraftSource& source, const ControlVectorTarget& target) {
    const Refusal refuse(source.name);
    if (target.hidden <= 0 || target.layers < 2) {
        throw std::invalid_argument("load_control_vector: the target model has no layers to steer");
    }
    const std::vector<std::string> hints = accepted_hints(target.architecture);
    const std::vector<std::uint8_t> bytes = read_file(source);

    if (bytes.size() < 4 || std::memcmp(bytes.data(), "GGUF", 4) != 0) {
        refuse("it is not a GGUF file");
    }
    Cursor in(bytes, refuse);
    in.skip(4);
    const auto version = in.read<std::uint32_t>();
    if (version != 2 && version != 3) {
        refuse("GGUF version " + std::to_string(version) + " is not 2 or 3");
    }
    const auto tensor_count   = in.read<std::uint64_t>();
    const auto metadata_count = in.read<std::uint64_t>();
    if (tensor_count == 0 || tensor_count > kMaxEntries) {
        refuse("it holds " + std::to_string(tensor_count) + " tensors");
    }
    if (metadata_count > kMaxEntries) {
        refuse(std::to_string(metadata_count) + " metadata entries");
    }

    std::map<std::string, Value> meta;
    for (std::uint64_t i = 0; i < metadata_count; ++i) {
        std::string key  = in.string();
        const auto type  = in.read<std::uint32_t>();
        Value value      = in.value(type);
        if (!meta.emplace(std::move(key), std::move(value)).second) {
            refuse("a metadata key appears twice");
        }
    }
    const auto text = [&](const char* key) -> std::string {
        const auto it = meta.find(key);
        if (it == meta.end()) { return {}; }
        if (it->second.type != String) { refuse(std::string(key) + " is not a string"); }
        return it->second.text;
    };
    const auto number = [&](const char* key) -> std::optional<Value> {
        const auto it = meta.find(key);
        if (it == meta.end()) { return std::nullopt; }
        if (it->second.type == String || it->second.type == Array) {
            refuse(std::string(key) + " is not a number");
        }
        return it->second;
    };
    const auto integer = [&](const char* key, std::int64_t fallback) -> std::int64_t {
        const auto it = meta.find(key);
        if (it == meta.end()) { return fallback; }
        if (it->second.type == String) {
            if (const auto parsed = parse_integer(trim(it->second.text))) { return *parsed; }
            refuse(std::string(key) + " is not an integer");
        }
        const std::optional<Value> v = number(key);
        if (!v->integral) { refuse(std::string(key) + " is not an integer"); }
        return static_cast<std::int64_t>(v->number);
    };

    const std::string architecture = text("general.architecture");
    if (architecture != "controlvector") {
        refuse("general.architecture is '" + architecture + "', not 'controlvector'");
    }
    std::uint64_t alignment = 32;
    if (meta.count("general.alignment") != 0) {
        const std::int64_t value = integer("general.alignment", 32);
        if (value <= 0 || value > 4096 || (value & (value - 1)) != 0) {
            refuse("general.alignment " + std::to_string(value) + " is not a power of two up to 4096");
        }
        alignment = static_cast<std::uint64_t>(value);
    }

    ControlVector out;
    out.name       = source.name;
    out.hidden     = target.hidden;
    out.layers     = target.layers;
    out.model_hint = text("controlvector.model_hint");
    if (!out.model_hint.empty() && std::find(hints.begin(), hints.end(), out.model_hint) == hints.end()) {
        std::string expected;
        for (const std::string& hint : hints) { expected += (expected.empty() ? "" : ", ") + hint; }
        refuse("it was made for '" + out.model_hint + "', not for this model (" + expected + ")");
    }

    // GLP: the glp.* keys decide what the file does. Without them it is a llama.cpp (additive)
    // control vector.
    const bool glp = std::any_of(meta.begin(), meta.end(), [](const auto& entry) {
        return entry.first.compare(0, 4, "glp.") == 0;
    });
    const bool dspark = std::any_of(meta.begin(), meta.end(), [](const auto& entry) {
        return entry.first.compare(0, 7, "dspark.") == 0;
    });
    if (!glp && dspark) {
        refuse("it carries only dspark.* keys, a pre-GLP internal build; re-export it as GLP");
    }
    double dir_scale = 1.0;
    std::map<std::int32_t, double> layer_scales;
    std::vector<std::int32_t> declared_layers;
    bool layers_declared = false; // glp.layer_ids_zero_based is present, even if it lists none
    std::string content_sha256;
    bool content_declared = false; // glp.content_sha256 is present (it must then be a digest)
    if (glp) {
        out.format = ControlVectorFormat::Glp;
        const std::string mode = meta.count("glp.mode") != 0 ? text("glp.mode") : std::string("add");
        if (mode == "project") {
            out.mode = ControlVectorMode::Project;
        } else if (mode == "add") {
            out.mode = ControlVectorMode::Add;
        } else {
            refuse("glp.mode '" + mode + "' is not one this reader implements (project, add)");
        }
        // glp.spec_version is required: it says which rank, scaling and hook rules the other keys
        // follow, so a file without it is refused rather than read as version 1.
        if (meta.count("glp.spec_version") == 0) {
            refuse("glp.spec_version is missing; a GLP file must say which version it follows");
        }
        const std::int64_t spec = integer("glp.spec_version", 0);
        if (spec < 1 || spec > 3) {
            refuse("glp.spec_version " + std::to_string(spec) + " is not one this reader implements (1-3)");
        }
        out.spec_version = static_cast<std::uint32_t>(spec);
        out.hook_point   = text("glp.hook_point");
        out.derived_at   = text("glp.derived_at");
        if (!out.hook_point.empty() && out.hook_point != kPostLayer) {
            refuse("glp.hook_point '" + out.hook_point + "' is not where this model steers (" +
                   std::string(kPostLayer) + ")");
        }
        const std::int64_t rank = integer("glp.rank", 1);
        if (rank != 1) {
            refuse("glp.rank " + std::to_string(rank) +
                   " is a subspace vector; this reader applies one direction per layer");
        }
        const bool has_dir_scales   = meta.count("glp.dir_scales") != 0;
        const bool has_layer_scales = meta.count("glp.layer_scales") != 0;
        if (spec == 1 && (has_dir_scales || has_layer_scales)) {
            refuse("glp.dir_scales and glp.layer_scales need glp.spec_version 2");
        }
        if (const std::optional<Value> alpha = number("glp.alpha_default")) {
            if (!std::isfinite(alpha->number)) { refuse("glp.alpha_default is not finite"); }
            out.alpha_default = static_cast<float>(alpha->number);
        }
        if (has_dir_scales) {
            const std::vector<std::string> parts = split_commas(text("glp.dir_scales"));
            const std::optional<double> scale    = parts.size() == 1 ? parse_float(parts[0]) : std::nullopt;
            if (!scale || !std::isfinite(*scale)) {
                refuse("glp.dir_scales is not one finite number (glp.rank is 1)");
            }
            dir_scale = *scale;
        }
        if (has_layer_scales) {
            for (const std::string& pair : split_commas(text("glp.layer_scales"))) {
                const auto colon = pair.find(':');
                const auto layer = colon == std::string::npos ? std::nullopt : parse_integer(trim(pair.substr(0, colon)));
                const auto scale = colon == std::string::npos ? std::nullopt : parse_float(trim(pair.substr(colon + 1)));
                if (!layer || !std::isfinite(scale.value_or(std::numeric_limits<double>::quiet_NaN()))) {
                    refuse("glp.layer_scales entry '" + pair + "' is not LAYER:SCALE");
                }
                const std::int64_t id = layer.value_or(-1);
                if (id >= target.layers) {
                    refuse("glp.layer_scales names layer " + std::to_string(id) +
                           "; this model has layers 0 to " + std::to_string(target.layers - 1));
                }
                if (!layer_scales.emplace(static_cast<std::int32_t>(id), scale.value_or(1.0)).second) {
                    refuse("glp.layer_scales names layer " + std::to_string(id) + " twice");
                }
            }
        }
        if (meta.count("glp.layer_ids_zero_based") != 0) {
            layers_declared        = true;
            const std::string list = text("glp.layer_ids_zero_based");
            if (!trim(list).empty()) {
                for (const std::string& item : split_commas(list)) {
                    const auto layer = parse_integer(item);
                    if (!layer) { refuse("glp.layer_ids_zero_based is not a list of layer numbers"); }
                    declared_layers.push_back(static_cast<std::int32_t>(*layer));
                }
                std::sort(declared_layers.begin(), declared_layers.end());
            }
        }
        content_declared = meta.count("glp.content_sha256") != 0;
        content_sha256   = text("glp.content_sha256");
        if (content_declared &&
            (content_sha256.size() != 64 ||
             !std::all_of(content_sha256.begin(), content_sha256.end(),
                          [](unsigned char c) { return std::isxdigit(c) != 0; }))) {
            refuse("glp.content_sha256 is not a 64-digit hex SHA-256");
        }
        out.base_model = text("general.base_model.0.name");
        if (out.base_model.empty()) { out.base_model = text("glp.base_model"); }
        const std::string organization = text("general.base_model.0.organization");
        if (!organization.empty() && !out.base_model.empty()) {
            out.base_model = organization + "/" + out.base_model;
        }
        out.base_revision = text("general.base_model.0.version");
        if (out.base_revision.empty()) { out.base_revision = text("glp.base_revision"); }
    } else {
        out.format        = ControlVectorFormat::LlamaCpp;
        out.mode          = ControlVectorMode::Add;
        out.alpha_default = 1.0f;
    }

    std::vector<TensorInfo> tensors;
    std::set<std::int32_t> seen;
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
        if (name.compare(0, prefix.size(), prefix) != 0 || name.size() == prefix.size()) {
            refuse("tensor '" + name + "' is not named direction.N");
        }
        const std::string suffix = name.substr(prefix.size());
        if (suffix.find('.') != std::string::npos) {
            refuse("tensor '" + name + "' is a further direction of a subspace (rank > 1) vector");
        }
        const auto layer = parse_integer(suffix);
        if (!layer) { refuse("tensor '" + name + "' is not named direction.N"); }
        if (*layer == 0) { refuse("tensor direction.0 is invalid: block 0 cannot be steered"); }
        if (*layer >= target.layers) {
            refuse("tensor '" + name + "' is layer " + std::to_string(*layer) +
                   "; this model has layers 1 to " + std::to_string(target.layers - 1));
        }
        if (type != kGgmlF32) { refuse("tensor '" + name + "' is not F32"); }
        if (width != static_cast<std::uint64_t>(target.hidden)) {
            refuse("tensor '" + name + "' has " + std::to_string(width) +
                   " values; the model's hidden size is " + std::to_string(target.hidden));
        }
        if (!seen.insert(static_cast<std::int32_t>(*layer)).second) {
            refuse("layer " + std::to_string(*layer) + " appears twice");
        }
        tensors.push_back({static_cast<std::int32_t>(*layer), offset});
    }
    if (meta.count("controlvector.layer_count") != 0) {
        const std::int64_t count = integer("controlvector.layer_count", 0);
        if (count != static_cast<std::int64_t>(tensors.size())) {
            refuse("controlvector.layer_count is " + std::to_string(count) + " but the file holds " +
                   std::to_string(tensors.size()) + " directions");
        }
    }
    if (layers_declared && declared_layers != std::vector<std::int32_t>(seen.begin(), seen.end())) {
        refuse("glp.layer_ids_zero_based does not list the layers its direction tensors name; the "
               "names are what would be applied, so the file is inconsistent");
    }

    const std::uint64_t base = (in.position() + alignment - 1) / alignment * alignment;
    const std::uint64_t row  = static_cast<std::uint64_t>(target.hidden) * sizeof(float);
    const auto hidden        = static_cast<std::size_t>(target.hidden);
    out.directions.assign(static_cast<std::size_t>(target.layers) * hidden, 0.0f);
    out.strength.assign(static_cast<std::size_t>(target.layers), 0.0f);
    out.active.assign(static_cast<std::size_t>(target.layers), 0);
    std::sort(tensors.begin(), tensors.end(),
              [](const TensorInfo& a, const TensorInfo& b) { return a.layer < b.layer; });
    std::vector<std::uint8_t> content; // the tensors' bytes in layer order (glp.content_sha256)
    content.reserve(tensors.size() * row);
    for (const TensorInfo& tensor : tensors) {
        const std::string label = "direction." + std::to_string(tensor.layer);
        if (tensor.offset > bytes.size() || base > bytes.size() - tensor.offset ||
            row > bytes.size() - base - tensor.offset) {
            refuse("tensor " + label + " lies past the end of the file");
        }
        const std::uint8_t* raw = bytes.data() + base + tensor.offset;
        content.insert(content.end(), raw, raw + row);
        float* destination = out.directions.data() + static_cast<std::size_t>(tensor.layer) * hidden;
        std::memcpy(destination, raw, static_cast<std::size_t>(row));
        double norm = 0.0;
        for (std::size_t i = 0; i < hidden; ++i) {
            if (!std::isfinite(destination[i])) { refuse(label + " holds a value that is not finite"); }
            norm += static_cast<double>(destination[i]) * destination[i];
        }
        norm = std::sqrt(norm);
        if (out.mode == ControlVectorMode::Project) {
            // The projection is only a projection along a unit direction: normalise, as the
            // reference readers do, so a non-unit export cannot silently rescale alpha.
            if (!(norm > 0.0) || !std::isfinite(norm)) { refuse(label + " is the zero vector"); }
            for (std::size_t i = 0; i < hidden; ++i) {
                destination[i] = static_cast<float>(destination[i] / norm);
            }
        }
        const auto scale = layer_scales.find(tensor.layer);
        const double strength =
            static_cast<double>(out.alpha_default) * dir_scale * (scale == layer_scales.end() ? 1.0 : scale->second);
        if (!std::isfinite(strength) || std::fabs(strength) > std::numeric_limits<float>::max()) {
            refuse(label + "'s strength is not finite");
        }
        out.strength[static_cast<std::size_t>(tensor.layer)] = static_cast<float>(strength);
        out.active[static_cast<std::size_t>(tensor.layer)]   = 1;
    }
    if (content_declared) {
        std::string declared = content_sha256;
        std::transform(declared.begin(), declared.end(), declared.begin(),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        const std::string actual = frontend::sha256_hex(frontend::sha256(content));
        if (declared != actual) {
            refuse("glp.content_sha256 is " + content_sha256.substr(0, 16) +
                   "... but the direction tensors hash to " + actual.substr(0, 16) +
                   "...; the file is damaged or was edited");
        }
    }
    const frontend::Sha256Digest digest = frontend::sha256(bytes);
    std::uint64_t mixed                 = 0;
    for (int i = 0; i < 8; ++i) { mixed = (mixed << 8U) | digest[static_cast<std::size_t>(i)]; }
    out.digest = mixed == 0 ? 1 : mixed;
    return out;
}

} // namespace ninfer::models::qwen3_5
