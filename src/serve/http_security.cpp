#include "serve/http_security.h"

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <unistd.h>
#endif

#include <algorithm>
#include <array>
#include <cstddef>
#include <optional>

namespace ninfer::serve {
namespace {

char lower(char value) {
    return value >= 'A' && value <= 'Z' ? static_cast<char>(value - 'A' + 'a') : value;
}

bool iequal(std::string_view lhs, std::string_view rhs) {
    return lhs.size() == rhs.size() && std::equal(lhs.begin(), lhs.end(), rhs.begin(),
                                                  [](char a, char b) { return lower(a) == lower(b); });
}

bool is_digit(char value) { return value >= '0' && value <= '9'; }

bool is_hex(char value) {
    const char low = lower(value);
    return is_digit(value) || (low >= 'a' && low <= 'f');
}

std::string_view trim(std::string_view text) {
    while (!text.empty() && (text.front() == ' ' || text.front() == '\t')) { text.remove_prefix(1); }
    while (!text.empty() && (text.back() == ' ' || text.back() == '\t')) { text.remove_suffix(1); }
    return text;
}

// A port: one to five digits, at most 65535.
std::optional<int> parse_port(std::string_view text) {
    if (text.empty() || text.size() > 5 || !std::all_of(text.begin(), text.end(), is_digit)) {
        return std::nullopt;
    }
    int value = 0;
    for (const char digit : text) { value = value * 10 + (digit - '0'); }
    if (value > 65535) { return std::nullopt; }
    return value;
}

// Four dotted decimal octets, each at most 255.
bool is_ipv4_literal(std::string_view text) {
    int octets = 0;
    while (true) {
        std::size_t digits = 0;
        int value          = 0;
        while (digits < text.size() && is_digit(text[digits])) {
            value = value * 10 + (text[digits] - '0');
            ++digits;
        }
        if (digits == 0 || digits > 3 || value > 255) { return false; }
        ++octets;
        text.remove_prefix(digits);
        if (text.empty()) { return octets == 4; }
        if (text.front() != '.' || octets == 4) { return false; }
        text.remove_prefix(1);
    }
}

// "[...]" around hex digits, colons and (for an embedded IPv4 tail) dots, with two colons or more.
bool is_bracketed_ipv6_literal(std::string_view text) {
    if (text.size() < 4 || text.front() != '[' || text.back() != ']') { return false; }
    const std::string_view inner = text.substr(1, text.size() - 2);
    return std::count(inner.begin(), inner.end(), ':') >= 2 &&
           std::all_of(inner.begin(), inner.end(),
                       [](char c) { return is_hex(c) || c == ':' || c == '.'; });
}

// Lower case, IPv6 in brackets, and one trailing dot (the DNS root) dropped from a name.
std::string normalize_name(std::string_view name) {
    std::string out(name);
    std::transform(out.begin(), out.end(), out.begin(), lower);
    if (out.find(':') != std::string::npos && out.front() != '[') { out = "[" + out + "]"; }
    if (out.size() > 1 && out.back() == '.') { out.pop_back(); }
    return out;
}

struct Authority {
    std::string host;        // normalized
    std::optional<int> port; // absent when the authority has none (or an empty one)
};

// host[:port] or [v6][:port]. Refuses anything else, including an unbracketed IPv6 address, a
// non-numeric port and characters a host name cannot hold.
std::optional<Authority> parse_authority(std::string_view text) {
    std::string_view host;
    std::string_view rest;
    if (!text.empty() && text.front() == '[') {
        const std::size_t close = text.find(']');
        if (close == std::string_view::npos) { return std::nullopt; }
        host = text.substr(0, close + 1);
        rest = text.substr(close + 1);
        if (!is_bracketed_ipv6_literal(host)) { return std::nullopt; }
    } else {
        const std::size_t colon = text.find(':');
        host = text.substr(0, colon);
        rest = colon == std::string_view::npos ? std::string_view{} : text.substr(colon);
        const bool name_chars = std::all_of(host.begin(), host.end(), [](char c) {
            const char low = lower(c);
            return is_digit(c) || (low >= 'a' && low <= 'z') || c == '-' || c == '.' || c == '_';
        });
        if (host.empty() || host == "." || !name_chars) { return std::nullopt; }
    }
    Authority authority{.host = normalize_name(host), .port = std::nullopt};
    if (!rest.empty()) {
        if (rest.front() != ':') { return std::nullopt; }
        rest.remove_prefix(1);
        if (!rest.empty()) {
            authority.port = parse_port(rest);
            if (!authority.port) { return std::nullopt; }
        }
    }
    return authority;
}

bool is_wildcard(std::string_view bind_host) {
    return bind_host.empty() || bind_host == "0.0.0.0" || bind_host == "::" || bind_host == "[::]";
}

bool is_loopback(std::string_view bind_host) {
    if (iequal(bind_host, "localhost") || bind_host == "::1" || bind_host == "[::1]") {
        return true;
    }
    return is_ipv4_literal(bind_host) && bind_host.starts_with("127.");
}

bool contains(const HostPolicy& policy, std::string_view name) {
    return std::find(policy.names.begin(), policy.names.end(), name) != policy.names.end();
}

} // namespace

bool constant_time_equal(std::string_view lhs, std::string_view rhs) noexcept {
    // Every byte of the longer string is visited, and a difference in length is folded into the
    // same accumulator, so neither the content nor the position of a mismatch changes the work.
    const std::size_t length = std::max(lhs.size(), rhs.size());
    unsigned difference      = lhs.size() == rhs.size() ? 0U : 1U;
    for (std::size_t i = 0; i < length; ++i) {
        const auto a = static_cast<unsigned char>(i < lhs.size() ? lhs[i] : 0);
        const auto b = static_cast<unsigned char>(i < rhs.size() ? rhs[i] : 0);
        difference |= static_cast<unsigned>(a ^ b);
    }
    return difference == 0U;
}

HostPolicy make_host_policy(std::string_view bind_host, int port,
                            std::span<const std::string> allowed, std::string_view machine_name) {
    HostPolicy policy;
    const std::string suffix = ":" + std::to_string(port);
    const auto add           = [&](std::string_view raw) {
        if (raw.empty()) { return; }
        const std::string name = normalize_name(raw);
        if (name.empty() || contains(policy, name)) { return; }
        policy.names.push_back(name);
        policy.names.push_back(name + suffix);
    };
    add("localhost");
    add("127.0.0.1");
    add("[::1]");
    if (is_wildcard(bind_host)) {
        add(machine_name);
    } else {
        add(bind_host);
    }
    for (const std::string& name : allowed) { add(name); }
    return policy;
}

bool host_allowed(const HostPolicy& policy, std::string_view host_header) {
    const std::string_view header = trim(host_header);
    if (header.empty()) { return true; }
    const std::optional<Authority> authority = parse_authority(header);
    if (!authority) { return false; }
    if (is_ipv4_literal(authority->host) || is_bracketed_ipv6_literal(authority->host)) {
        return true;
    }
    return contains(policy, authority->host);
}

bool origin_allowed(const HostPolicy& policy, std::string_view origin, bool has_key, bool cors) {
    if (has_key || cors) { return true; }
    origin = trim(origin);
    if (origin.empty()) { return true; }
    int default_port = 0;
    std::string_view rest;
    constexpr std::string_view http  = "http://";
    constexpr std::string_view https = "https://";
    if (origin.size() > http.size() && iequal(origin.substr(0, http.size()), http)) {
        default_port = 80;
        rest         = origin.substr(http.size());
    } else if (origin.size() > https.size() && iequal(origin.substr(0, https.size()), https)) {
        default_port = 443;
        rest         = origin.substr(https.size());
    } else {
        return false; // "null", file:, extension and other schemes
    }
    const std::optional<Authority> authority = parse_authority(rest);
    if (!authority) { return false; }
    return contains(policy,
                    authority->host + ":" + std::to_string(authority->port.value_or(default_port)));
}

bool exposed_without_key(std::string_view bind_host, bool has_key) {
    return !has_key && !is_loopback(bind_host);
}

bool dropped_when_relaying(std::string_view header_name) {
    return iequal(header_name, "authorization") || iequal(header_name, "x-api-key") ||
           iequal(header_name, "origin");
}

std::string local_machine_name() {
#if defined(_WIN32)
    std::array<char, 256> buffer{};
    DWORD size = static_cast<DWORD>(buffer.size());
    if (GetComputerNameExA(ComputerNameDnsHostname, buffer.data(), &size) == 0) { return {}; }
    return std::string(buffer.data(), size);
#else
    std::array<char, 256> buffer{};
    if (gethostname(buffer.data(), buffer.size() - 1) != 0) { return {}; }
    return std::string(buffer.data());
#endif
}

} // namespace ninfer::serve
