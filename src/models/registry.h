#pragma once

#include <string_view>

namespace ninfer::models {

// Qwen4Exp is Qwen3.8-Flash-Next (models/qwen4_exp): its own model, Program and Engine core. Dsv4 is
// DeepSeek-V4.1-Flash (models/dsv4), likewise; it has no resolve_architecture entry because its
// artifact is recognised by models::dsv4::is_dsv4, not by the Qwen3.5 config reader.
enum class Architecture { Qwen3_5, Qwen3_5Moe, Qwen4Exp, Dsv4 };

[[nodiscard]] Architecture resolve_architecture(std::string_view architecture,
                                                std::string_view model_type);
[[nodiscard]] std::string_view architecture_name(Architecture architecture) noexcept;

} // namespace ninfer::models
