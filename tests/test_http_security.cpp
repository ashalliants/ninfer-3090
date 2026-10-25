// ninfer-serve's request checks (serve/http_security.h), host only:
//   - ct_equal_cases: equal, different, different lengths and empty strings;
//   - host_loopback_ok: localhost, 127.0.0.1 and [::1], with and without a port, in any case and
//     with a trailing dot, and a missing Host header;
//   - host_rebind_rejected: a name the server does not answer to (evil.example:8080) is refused,
//     also when it merely ends in or contains an allowed name, carries userinfo or a space;
//   - host_lan_ip_ok_when_bound: bound to 192.168.1.5, Host 192.168.1.5:8080 passes;
//   - host_wildcard_machine_name: bound to 0.0.0.0 or ::, the machine name passes, and so does a
//     LAN address the server cannot know in advance; bound to one address, the machine name does
//     not;
//   - host_allowed_flag: each --allowed-host passes, with any port;
//   - origin_cases: a key or --cors allows any origin; without either, a missing origin and the
//     server's own origins pass, and a foreign origin, another port, "null" and other schemes
//     are refused;
//   - fetch_site_cases: Sec-Fetch-Site: cross-site is refused without a key or --cors; the other
//     values and its absence pass;
//   - allowed_host_values: --allowed-host takes names and addresses the Host check can match, and
//     refuses wildcards, ports, schemes, paths, userinfo and characters no name holds;
//   - exposure_warning_cases: 0.0.0.0, :: and a LAN address without a key warn; loopback, or a
//     key, does not;
//   - relay_drops_origin_and_credentials: the router drops Origin, Authorization and x-api-key
//     (in any case) when relaying, and keeps the rest.

#include "serve/http_security.h"

#include <cstdio>
#include <string>
#include <vector>

namespace {

using ninfer::serve::constant_time_equal;
using ninfer::serve::dropped_when_relaying;
using ninfer::serve::exposed_without_key;
using ninfer::serve::fetch_site_allowed;
using ninfer::serve::host_allowed;
using ninfer::serve::HostPolicy;
using ninfer::serve::make_host_policy;
using ninfer::serve::origin_allowed;
using ninfer::serve::valid_allowed_host;

int failures = 0;

void check(bool ok, const std::string& what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what.c_str());
        ++failures;
    }
}

const std::vector<std::string> kNone;

HostPolicy loopback_policy() { return make_host_policy("127.0.0.1", 8080, kNone, "gpubox"); }

void ct_equal_cases() {
    check(constant_time_equal("secret", "secret"), "equal strings");
    check(!constant_time_equal("secret", "secreT"), "last byte differs");
    check(!constant_time_equal("secret", "Secret"), "first byte differs");
    check(!constant_time_equal("secret", "secret2"), "longer right");
    check(!constant_time_equal("secret2", "secret"), "longer left");
    check(!constant_time_equal("", "secret"), "empty left");
    check(!constant_time_equal("secret", ""), "empty right");
    check(constant_time_equal("", ""), "both empty");
    const std::string with_nul("a\0b", 3);
    check(!constant_time_equal(with_nul, "a"), "embedded NUL is compared, not a terminator");
}

void host_loopback_ok() {
    const HostPolicy policy = loopback_policy();
    for (const char* host : {"localhost", "localhost:8080", "127.0.0.1", "127.0.0.1:8080", "[::1]",
                             "[::1]:8080", "LOCALHOST:8080", "LocalHost", "localhost.:8080",
                             "localhost.", " localhost:8080 ", "localhost:"}) {
        check(host_allowed(policy, host), std::string("loopback host refused: ") + host);
    }
    check(host_allowed(policy, ""), "a missing Host header (HTTP/1.0) is allowed");
    // A tunnel (ssh -L 9000:127.0.0.1:8080) arrives with its own port.
    check(host_allowed(policy, "localhost:9000"), "loopback name on another port");
}

void host_rebind_rejected() {
    const HostPolicy policy = loopback_policy();
    for (const char* host :
         {"evil.example:8080", "evil.example", "EVIL.example", "localhost.evil.example:8080",
          "evillocalhost:8080", "127.0.0.1.evil.example", "localhost:8080.evil.example",
          "localhost:abc", "localhost:8080:8080", "localhost..", "gpubox:8080", "::1",
          "[::1", "[evil.example]:8080", "1.2.3:8080", "256.1.1.1:8080", "1.2.3.4.5",
          "user@localhost", "user@localhost:8080", "local host:8080", "localhost :8080",
          "local\thost"}) {
        check(!host_allowed(policy, host), std::string("rebinding host accepted: ") + host);
    }
}

void host_lan_ip_ok_when_bound() {
    const HostPolicy policy = make_host_policy("192.168.1.5", 8080, kNone, "gpubox");
    check(host_allowed(policy, "192.168.1.5:8080"), "bound LAN address with port");
    check(host_allowed(policy, "192.168.1.5"), "bound LAN address without port");
    check(host_allowed(policy, "localhost:8080"), "loopback still allowed");
    check(!host_allowed(policy, "gpubox:8080"), "machine name only when bound to a wildcard");
    check(!host_allowed(policy, "evil.example:8080"), "foreign name");

    // Bound to a name, the name passes in any case.
    const HostPolicy named = make_host_policy("GpuBox.lan", 8080, kNone, "gpubox");
    check(host_allowed(named, "gpubox.lan:8080"), "bound name, lower case");
    check(host_allowed(named, "GPUBOX.LAN"), "bound name, upper case");

    // Bound to an IPv6 address, written bare or bracketed.
    const HostPolicy v6 = make_host_policy("fe80::5", 8080, kNone, "gpubox");
    check(host_allowed(v6, "[fe80::5]:8080"), "bound IPv6 address");
    check(host_allowed(v6, "[FE80::5]"), "bound IPv6 address, upper case");
}

void host_wildcard_machine_name() {
    for (const char* wildcard : {"0.0.0.0", "::", "[::]"}) {
        const HostPolicy policy = make_host_policy(wildcard, 8080, kNone, "GpuBox");
        const std::string at    = std::string(" (bound to ") + wildcard + ")";
        check(host_allowed(policy, "gpubox:8080"), "machine name" + at);
        check(host_allowed(policy, "GPUBOX"), "machine name, upper case, no port" + at);
        check(host_allowed(policy, "localhost:8080"), "loopback" + at);
        check(host_allowed(policy, "192.168.1.5:8080"), "LAN address" + at);
        check(host_allowed(policy, "10.0.0.7"), "other LAN address" + at);
        check(host_allowed(policy, "[fe80::5]:8080"), "IPv6 LAN address" + at);
        check(!host_allowed(policy, "evil.example:8080"), "foreign name" + at);
        check(!host_allowed(policy, "0.0.0.0.evil.example"), "wildcard text in a name" + at);
    }
    // No machine name (the system would not say): the wildcard adds nothing, loopback still works.
    const HostPolicy unnamed = make_host_policy("0.0.0.0", 8080, kNone, "");
    check(host_allowed(unnamed, "localhost:8080"), "loopback without a machine name");
    check(!host_allowed(unnamed, ":8080"), "empty name with a port");
}

void host_allowed_flag() {
    const std::vector<std::string> allowed{"ninfer.lan", "My-Proxy"};
    const HostPolicy policy = make_host_policy("127.0.0.1", 8080, allowed, "gpubox");
    check(host_allowed(policy, "ninfer.lan:8080"), "--allowed-host with the port");
    check(host_allowed(policy, "ninfer.lan"), "--allowed-host without a port");
    check(host_allowed(policy, "my-proxy:443"), "--allowed-host, other case and port");
    check(!host_allowed(policy, "sub.ninfer.lan:8080"), "a subdomain is a different name");
    check(!host_allowed(policy, "evil.example:8080"), "foreign name with --allowed-host set");
}

void origin_cases() {
    const HostPolicy policy = loopback_policy();
    const char* foreign     = "http://evil.example";
    check(origin_allowed(policy, foreign, /*has_key=*/true, /*cors=*/false), "key set");
    check(origin_allowed(policy, foreign, false, /*cors=*/true), "--cors set");
    check(origin_allowed(policy, "null", true, false), "null origin with a key");
    check(!origin_allowed(policy, foreign, false, false), "foreign origin without a key");
    check(!origin_allowed(policy, "https://evil.example:8080", false, false),
          "foreign origin on the server's port");
    check(origin_allowed(policy, "", false, false), "no Origin (non-browser client)");
    for (const char* own : {"http://localhost:8080", "http://127.0.0.1:8080", "http://[::1]:8080",
                            "HTTP://LOCALHOST:8080", "http://localhost.:8080"}) {
        check(origin_allowed(policy, own, false, false), std::string("own origin refused: ") + own);
    }
    for (const char* other :
         {"http://localhost:3000", "http://localhost", "https://localhost",
          "http://localhost:8080/", "http://user@localhost:8080", "null", "file://",
          "chrome-extension://abcdef", "localhost:8080", "http://", "http://:8080",
          "http://localhost:80800", "http://localhost.evil.example:8080"}) {
        check(!origin_allowed(policy, other, false, false),
              std::string("foreign origin accepted: ") + other);
    }
    // The default ports count: a server on 80 is reached from http://name, on 443 from https.
    const HostPolicy on_80 = make_host_policy("127.0.0.1", 80, kNone, "gpubox");
    check(origin_allowed(on_80, "http://localhost", false, false), "http default port");
    check(!origin_allowed(on_80, "https://localhost", false, false), "https is port 443");
    const HostPolicy on_443 = make_host_policy("127.0.0.1", 443, kNone, "gpubox");
    check(origin_allowed(on_443, "https://localhost", false, false), "https default port");
    // Unlike the Host check, an address is not enough: the page must be this server's.
    const HostPolicy wildcard = make_host_policy("0.0.0.0", 8080, kNone, "gpubox");
    check(origin_allowed(wildcard, "http://gpubox:8080", false, false), "machine-name origin");
    check(!origin_allowed(wildcard, "http://192.168.1.99:8080", false, false),
          "another LAN page is a different site");
    const std::vector<std::string> allowed{"ninfer.lan"};
    const HostPolicy flagged = make_host_policy("127.0.0.1", 8080, allowed, "gpubox");
    check(origin_allowed(flagged, "http://ninfer.lan:8080", false, false),
          "--allowed-host origin on the server's port");
    check(!origin_allowed(flagged, "http://ninfer.lan:9000", false, false),
          "--allowed-host origin on another port");
}

void fetch_site_cases() {
    check(!fetch_site_allowed("cross-site", false, false), "cross-site without a key or --cors");
    check(!fetch_site_allowed(" Cross-Site ", false, false), "cross-site in another case");
    check(fetch_site_allowed("cross-site", true, false), "cross-site with a key");
    check(fetch_site_allowed("cross-site", false, true), "cross-site with --cors");
    for (const char* site : {"", "same-origin", "same-site", "none"}) {
        check(fetch_site_allowed(site, false, false), std::string("refused: ") + site);
    }
}

void allowed_host_values() {
    for (const char* good : {"gpubox", "gpubox.lan", "GpuBox.Local", "my_box", "192.168.1.5",
                             "[fe80::5]", "gpubox.lan."}) {
        check(valid_allowed_host(good), std::string("valid --allowed-host refused: ") + good);
    }
    for (const char* bad : {"", "*", "*.lan", "gpu%box", "gpubox?", "gpu box", "user@gpubox",
                            "http://gpubox", "gpubox:8080", "gpubox:", "gpubox/v1", "fe80::5",
                            "[fe80::5]:8080", "[gpubox]", ".", "gpu\tbox"}) {
        check(!valid_allowed_host(bad), std::string("invalid --allowed-host accepted: ") + bad);
    }
}

void exposure_warning_cases() {
    check(exposed_without_key("0.0.0.0", false), "0.0.0.0 without a key warns");
    check(exposed_without_key("::", false), ":: without a key warns");
    check(exposed_without_key("192.168.1.5", false), "a LAN address without a key warns");
    check(exposed_without_key("gpubox.lan", false), "a LAN name without a key warns");
    check(!exposed_without_key("0.0.0.0", true), "a key silences the warning");
    for (const char* loopback : {"127.0.0.1", "127.0.0.2", "localhost", "LOCALHOST", "::1",
                                 "[::1]"}) {
        check(!exposed_without_key(loopback, false),
              std::string("loopback warned: ") + loopback);
    }
    check(exposed_without_key("127.0.0.1.evil", false), "a name starting 127. is not loopback");
}

void relay_drops_origin_and_credentials() {
    for (const char* dropped : {"Origin", "origin", "ORIGIN", "Authorization", "x-api-key",
                                "X-Api-Key"}) {
        check(dropped_when_relaying(dropped), std::string("relayed: ") + dropped);
    }
    for (const char* kept : {"Accept", "anthropic-version", "x-request-id", "Referer",
                             "User-Agent", "Originator"}) {
        check(!dropped_when_relaying(kept), std::string("dropped: ") + kept);
    }
}

} // namespace

int main() {
    ct_equal_cases();
    host_loopback_ok();
    host_rebind_rejected();
    host_lan_ip_ok_when_bound();
    host_wildcard_machine_name();
    host_allowed_flag();
    origin_cases();
    fetch_site_cases();
    allowed_host_values();
    exposure_warning_cases();
    relay_drops_origin_and_credentials();
    if (failures != 0) {
        std::fprintf(stderr, "%d check(s) failed\n", failures);
        return 1;
    }
    std::puts("http security checks passed");
    return 0;
}
