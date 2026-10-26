// The dashboard ninfer-serve serves at GET /dashboard (serve/dashboard_page.h), host only:
//   - page_embedded: the page is in the binary, is an HTML document, polls /slots, /metrics and
//     /v1/load, and names no external URL (it must work offline and fetch nothing elsewhere);
//   - page_writes_no_markup: the script never parses server data as HTML (no innerHTML,
//     outerHTML, insertAdjacentHTML or document.write), which is what makes the inline script
//     safe under the page's Content-Security-Policy;
//   - content_policy: the policy denies everything by default, lets the page fetch from its own
//     origin only, and forbids framing;
//   - keyless_route: GET and HEAD /dashboard are served without the key; other methods, other
//     paths and look-alike paths are not.
#include "serve/dashboard_page.h"

#include <cstdio>
#include <string_view>

namespace {

int failures = 0;

void check(bool condition, const char* what) {
    if (condition) { return; }
    std::fprintf(stderr, "FAIL: %s\n", what);
    ++failures;
}

bool contains(std::string_view text, std::string_view needle) {
    return text.find(needle) != std::string_view::npos;
}

void page_embedded() {
    const std::string_view page = ninfer::serve::dashboard_page();
    check(!page.empty(), "page_embedded: the page is not empty");
    check(page.substr(0, 15) == "<!doctype html>", "page_embedded: starts with <!doctype html>");
    check(contains(page, "</html>"), "page_embedded: the whole file is embedded");
    check(contains(page, "'/slots'"), "page_embedded: polls /slots");
    check(contains(page, "'/metrics'"), "page_embedded: polls /metrics");
    check(contains(page, "'/v1/load'"), "page_embedded: polls /v1/load");
    check(!contains(page, "http://"), "page_embedded: no http:// URL");
    check(!contains(page, "https://"), "page_embedded: no https:// URL");
    check(!contains(page, "//cdn"), "page_embedded: no protocol-relative CDN URL");
}

void page_writes_no_markup() {
    const std::string_view page = ninfer::serve::dashboard_page();
    for (const char* sink : {"innerHTML", "outerHTML", "insertAdjacentHTML", "document.write"}) {
        check(!contains(page, sink), "page_writes_no_markup: no HTML sink in the script");
    }
}

void content_policy() {
    const std::string_view policy = ninfer::serve::dashboard_content_security_policy();
    check(contains(policy, "default-src 'none'"), "content_policy: default-src 'none'");
    check(contains(policy, "connect-src 'self'"), "content_policy: connect-src 'self'");
    check(contains(policy, "frame-ancestors 'none'"), "content_policy: frame-ancestors 'none'");
    check(contains(policy, "form-action 'none'"), "content_policy: form-action 'none'");
    check(!contains(policy, "http"), "content_policy: names no other origin");
    check(!contains(policy, "*"), "content_policy: no wildcard source");
}

void keyless_route() {
    using ninfer::serve::is_dashboard_request;
    check(is_dashboard_request("GET", "/dashboard"), "keyless_route: GET /dashboard");
    check(is_dashboard_request("HEAD", "/dashboard"), "keyless_route: HEAD /dashboard");
    check(!is_dashboard_request("POST", "/dashboard"), "keyless_route: not POST /dashboard");
    check(!is_dashboard_request("DELETE", "/dashboard"), "keyless_route: not DELETE /dashboard");
    check(!is_dashboard_request("get", "/dashboard"), "keyless_route: methods are case-sensitive");
    check(!is_dashboard_request("GET", "/dashboard/"), "keyless_route: not /dashboard/");
    check(!is_dashboard_request("GET", "/dashboardx"), "keyless_route: not /dashboardx");
    check(!is_dashboard_request("GET", "/dashboard/../metrics"),
          "keyless_route: not a path below it");
    check(!is_dashboard_request("GET", "/Dashboard"), "keyless_route: paths are case-sensitive");
    check(!is_dashboard_request("GET", "/metrics"), "keyless_route: not /metrics");
    check(!is_dashboard_request("GET", "/slots"), "keyless_route: not /slots");
    check(!is_dashboard_request("GET", ""), "keyless_route: not an empty path");
}

} // namespace

int main() {
    page_embedded();
    page_writes_no_markup();
    content_policy();
    keyless_route();
    if (failures != 0) {
        std::fprintf(stderr, "%d check(s) failed\n", failures);
        return 1;
    }
    std::puts("dashboard page checks passed");
    return 0;
}
