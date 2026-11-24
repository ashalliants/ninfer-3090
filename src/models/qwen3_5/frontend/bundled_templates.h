#pragma once

#include <optional>
#include <string>
#include <string_view>

namespace ninfer::models::qwen3_5 {

// Artifacts converted without `--resource chat_template.jinja=...` carry the stock Qwen template.
// It raises "System message must be at the beginning." for a system or developer message anywhere
// but first, which Chat, Responses and Messages clients send routinely. If `artifact_source` is
// such a stock template, returns the bundled NInfer template of the same model family (Qwen3.8
// when it has reasoning effort, otherwise Qwen3.6); otherwise nullopt, and the artifact's own
// template is used unchanged. The choice depends only on the template text, so every request on a
// server renders the same way and cached prefixes stay valid.
[[nodiscard]] std::optional<std::string> bundled_template_for_stock(std::string_view artifact_source);

} // namespace ninfer::models::qwen3_5
