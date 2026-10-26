#pragma once

// The local dashboard served at GET /dashboard: one static HTML file (src/serve/dashboard/
// dashboard.html, inline script and style, nothing fetched from elsewhere) embedded at build time.
// It polls /metrics, /v1/load and /slots on its own origin and draws them.

#include <string_view>

namespace ninfer::serve {

// The page's bytes.
[[nodiscard]] std::string_view dashboard_page() noexcept;

// The Content-Security-Policy the page is served with. Everything is denied but the page's own
// inline script and style, fetches to its own origin and data: images (the favicon).
[[nodiscard]] std::string_view dashboard_content_security_policy() noexcept;

// GET or HEAD /dashboard. The page holds no data, so it is served without the API key (and while
// the model loads); its script sends the key the user types with every request it makes. The
// Host and Origin checks still apply.
[[nodiscard]] bool is_dashboard_request(std::string_view method, std::string_view path) noexcept;

} // namespace ninfer::serve
