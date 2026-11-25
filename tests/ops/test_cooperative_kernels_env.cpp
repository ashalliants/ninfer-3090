// NINFER_NO_COOPERATIVE_KERNELS (ops/common/cooperative_kernels.h), host only: which values turn
// cooperative kernels off, and that the process reads the variable once. The schedules each op
// picks with them off are checked by that op's own test (ninfer_gdn_gating_proj_test,
// verify_no_cooperative_routes).

#include "ops/common/cooperative_kernels.h"

#include <cstdio>
#include <cstdlib>
#include <string>

namespace {

int failures = 0;

// setenv / unsetenv are POSIX; the Windows CRT has _putenv_s, where an empty value removes the
// variable.
int set_env(const char* name, const char* value) {
#if defined(_WIN32)
    return _putenv_s(name, value);
#else
    return ::setenv(name, value, 1);
#endif
}

void unset_env(const char* name) {
#if defined(_WIN32)
    (void)_putenv_s(name, "");
#else
    (void)::unsetenv(name);
#endif
}

void check(bool ok, const std::string& what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what.c_str());
        ++failures;
    }
}

} // namespace

int main() {
    using ninfer::ops::cooperative_kernels_disabled_by;
    // Only "1" opts in, as with the other NINFER_* opt-ins (NINFER_SM70_VERIFY_PROMPT_ATTN=1).
    check(!cooperative_kernels_disabled_by(nullptr), "unset disabled cooperative kernels");
    check(!cooperative_kernels_disabled_by(""), "an empty value disabled cooperative kernels");
    check(!cooperative_kernels_disabled_by("0"), "0 disabled cooperative kernels");
    check(!cooperative_kernels_disabled_by("10"), "10 disabled cooperative kernels");
    check(cooperative_kernels_disabled_by("1"), "1 did not disable cooperative kernels");
    static_assert(cooperative_kernels_disabled_by("1") && !cooperative_kernels_disabled_by("0"));

    // The process value: whatever the environment says when first asked, and fixed after that, so
    // a plan resolved early and one resolved late always agree.
    if (set_env("NINFER_NO_COOPERATIVE_KERNELS", "1") != 0) {
        std::fprintf(stderr, "FAIL: setenv\n");
        return 1;
    }
    check(!ninfer::ops::cooperative_kernels_allowed(),
          "NINFER_NO_COOPERATIVE_KERNELS=1 left cooperative kernels allowed");
    unset_env("NINFER_NO_COOPERATIVE_KERNELS");
    check(!ninfer::ops::cooperative_kernels_allowed(),
          "the setting changed after it was first read");

    std::printf("%s cooperative_kernels_env\n", failures == 0 ? "OK" : "FAIL");
    return failures == 0 ? 0 : 1;
}
