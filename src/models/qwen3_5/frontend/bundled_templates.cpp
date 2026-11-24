#include "models/qwen3_5/frontend/bundled_templates.h"

#include <cstddef>

namespace ninfer::embedded {
extern const unsigned char qwen3_6_chat_template[];
extern const std::size_t qwen3_6_chat_template_size;
extern const unsigned char qwen3_8_chat_template[];
extern const std::size_t qwen3_8_chat_template_size;
} // namespace ninfer::embedded

namespace ninfer::models::qwen3_5 {

std::optional<std::string> bundled_template_for_stock(std::string_view artifact_source) {
    if (artifact_source.find("raise_exception('System message must be at the beginning.')") ==
        std::string_view::npos) {
        return std::nullopt;
    }
    if (artifact_source.find("reasoning_effort") != std::string_view::npos) {
        return std::string(reinterpret_cast<const char*>(embedded::qwen3_8_chat_template),
                           embedded::qwen3_8_chat_template_size);
    }
    return std::string(reinterpret_cast<const char*>(embedded::qwen3_6_chat_template),
                       embedded::qwen3_6_chat_template_size);
}

} // namespace ninfer::models::qwen3_5
