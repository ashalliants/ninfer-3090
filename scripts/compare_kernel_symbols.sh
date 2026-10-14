#!/usr/bin/env bash
# Compare device kernel entries (name + REG/STACK/SHARED/LOCAL) between two CMake builds of
# ninfer_ops, to show a translation-unit split moved code without changing it.
#
# Usage: scripts/compare_kernel_symbols.sh <build-dir-A> <build-dir-B> [kernel-name-regex]
#   Default regex covers the INT8-family small-T attention kernels:
#   'small_t_i8_tiled_kernel|small_t_reduce_output_kernel'
# Exit status 0 when the two sets are identical, 1 when they differ, 2 on usage/tool errors.
set -euo pipefail

if [ "$#" -lt 2 ]; then
    sed -n '2,8p' "$0" >&2
    exit 2
fi
command -v cuobjdump >/dev/null || { echo "cuobjdump not found (needs the CUDA toolkit)" >&2; exit 2; }

PATTERN="${3:-small_t_i8_tiled_kernel|small_t_reduce_output_kernel}"

# Prints "<mangled kernel name> REG=.. STACK=.. SHARED=.. LOCAL=.." for every matching entry
# in every ninfer_ops object file under the build dir, sorted.
dump() {
    local build="$1"
    # The target lives in src/ops, so its objects are under src/ops/CMakeFiles/ninfer_ops.dir.
    # Search for the directory rather than hardcoding it; Linux builds emit .o, Windows .obj.
    local objdir
    objdir="$(find "$build" -type d -name ninfer_ops.dir -path '*CMakeFiles*' -print -quit)"
    [ -n "$objdir" ] || { echo "no ninfer_ops.dir under $build" >&2; exit 2; }
    find "$objdir" \( -name '*.o' -o -name '*.obj' \) -print0 |
        xargs -0 -n1 cuobjdump --dump-resource-usage 2>/dev/null |
        awk -v pat="$PATTERN" '
            /^ *Function / { name = $2; sub(/:$/, "", name); next }
            name != "" && /REG:/ {
                if (name ~ pat) {
                    reg = stack = shared = lcl = ""
                    for (i = 1; i <= NF; i++) {
                        if ($i ~ /^REG:/)    reg = substr($i, 5)
                        if ($i ~ /^STACK:/)  stack = substr($i, 7)
                        if ($i ~ /^SHARED:/) shared = substr($i, 8)
                        if ($i ~ /^LOCAL:/)  lcl = substr($i, 7)
                    }
                    print name, "REG=" reg, "STACK=" stack, "SHARED=" shared, "LOCAL=" lcl
                }
                name = ""
            }' |
        sort
}

A="$(mktemp)"
B="$(mktemp)"
trap 'rm -f "$A" "$B"' EXIT
dump "$1" >"$A"
dump "$2" >"$B"

echo "A ($1): $(wc -l <"$A" | tr -d ' ') kernel entries, $(cut -d' ' -f1 "$A" | sort -u | wc -l | tr -d ' ') unique"
echo "B ($2): $(wc -l <"$B" | tr -d ' ') kernel entries, $(cut -d' ' -f1 "$B" | sort -u | wc -l | tr -d ' ') unique"

if [ ! -s "$A" ] || [ ! -s "$B" ]; then
    echo "no matching kernels found; check the regex or that the builds are sm_8x with device code" >&2
    exit 2
fi
if diff -u "$A" "$B"; then
    echo "IDENTICAL: same kernel entries with the same register/shared/stack/local usage"
else
    echo "DIFFERENT: see diff above (a duplicate entry means a kernel is instantiated in two TUs)" >&2
    exit 1
fi
