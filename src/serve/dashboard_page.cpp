#include "serve/dashboard_page.h"

#include <cstddef>

namespace ninfer::embedded {
// Generated from dashboard/dashboard.html by cmake/EmbedFile.cmake.
extern const unsigned char dashboard_html[];
extern const std::size_t dashboard_html_size;
} // namespace ninfer::embedded

namespace ninfer::serve {

std::string_view dashboard_page() noexcept {
    return {reinterpret_cast<const char*>(ninfer::embedded::dashboard_html),
            ninfer::embedded::dashboard_html_size};
}

std::string_view dashboard_content_security_policy() noexcept {
    // 'unsafe-inline' rather than hashes or nonces: the page is fixed at build time and its
    // script never turns server data into markup (it writes text nodes and draws on canvases, and
    // test_dashboard_page checks that it uses no HTML sink), so there is nothing injected for a
    // hash to keep out, while a hash would have to be recomputed by hand on every edit of the
    // page. Everything a hash would still guard against (loading script or style from elsewhere,
    // sending data elsewhere, framing) is denied below.
    return "default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; "
           "connect-src 'self'; img-src data:; base-uri 'none'; form-action 'none'; "
           "frame-ancestors 'none'";
}

bool is_dashboard_request(std::string_view method, std::string_view path) noexcept {
    return (method == "GET" || method == "HEAD") && path == "/dashboard";
}

} // namespace ninfer::serve
