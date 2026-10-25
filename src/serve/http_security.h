#pragma once

// Request checks that keep a local server local: the Host header against DNS rebinding, the
// Origin header against cross-site requests to a server without a key, and key comparison in
// constant time. Plain string logic with no Engine or HTTP-library dependency.

#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace ninfer::serve {

// True when both strings are equal. The time taken depends on the lengths only, never on where
// the first difference is, so a caller probing an API key learns nothing from response times.
[[nodiscard]] bool constant_time_equal(std::string_view lhs, std::string_view rhs) noexcept;

// The names a request may use to reach this server, lower-case, each with and without ":port".
struct HostPolicy {
    std::vector<std::string> names;
};

// localhost, 127.0.0.1 and [::1]; the bind host when it is not a wildcard (0.0.0.0, ::); the
// machine name when it is; and each --allowed-host. Every name appears bare and as name:port.
[[nodiscard]] HostPolicy make_host_policy(std::string_view bind_host, int port,
                                          std::span<const std::string> allowed,
                                          std::string_view machine_name);

// The Host check. Case-insensitive, and a trailing dot on the name is ignored. The port is not
// compared (a tunnel or port forward reaches the server on another one). A missing or empty
// header passes (HTTP/1.0 tools send none), and so does an IPv4 or bracketed IPv6 literal: DNS
// rebinding needs a name the attacker controls, so an address cannot carry it, and a server
// bound to a wildcard does not know every address that reaches it.
[[nodiscard]] bool host_allowed(const HostPolicy& policy, std::string_view host_header);

// The Origin check. True when the server has a key or --cors (both decide browser access on
// their own), when there is no Origin (a non-browser client), or when the origin's
// scheme://host[:port] names this server: host and port (80/443 by default) are in the policy.
// "null" (file:// pages, sandboxed frames) and other schemes are refused.
[[nodiscard]] bool origin_allowed(const HostPolicy& policy, std::string_view origin, bool has_key,
                                  bool cors);

// True when the server listens beyond loopback with no key: anyone who can reach the address can
// use it.
[[nodiscard]] bool exposed_without_key(std::string_view bind_host, bool has_key);

// The relay from the router to a model drops these request headers: credentials (the model needs
// none) and Origin (the router has checked it; the model, keyless on loopback, would refuse it).
[[nodiscard]] bool dropped_when_relaying(std::string_view header_name);

// This machine's host name, empty when the system will not say.
[[nodiscard]] std::string local_machine_name();

} // namespace ninfer::serve
