"""Show that two builds of a CUDA translation unit contain the same machine code, kernel by kernel.

A refactor that only threads a template parameter through (so every kernel's mangled name changes) must leave the
SASS of each kernel exactly as it was; scripts/compare_kernel_symbols.sh checks names and resource usage, this checks
the instructions. Symbol names, local labels and instruction addresses are ignored; a kernel is identified by the
text of its instructions, so the comparison is a multiset of kernel bodies.

    cuobjdump -sass before/unit.cu.o > before.txt        # the object of the unit, before and after the change
    cuobjdump -sass after/unit.cu.o  > after.txt
    python scripts/compare_kernel_sass.py before.txt after.txt

Exit status 0 when every kernel body matches, 1 when any differs (the first few are listed).
"""
import collections
import hashlib
import re
import sys


def bodies(path):
    out = collections.Counter()
    name, lines = None, []
    sizes = {}

    def flush():
        if name is not None:
            body = "\n".join(lines)
            body = re.sub(r"_Z\w+", "SYM", body)  # mangled names, including call targets and descriptors
            body = re.sub(r"\.L_\d+", ".L", body)  # local labels renumber with code order, not meaning
            digest = hashlib.sha256(body.encode()).hexdigest()[:16]
            out[digest] += 1
            sizes[digest] = len(lines)

    for line in open(path, errors="replace"):
        m = re.match(r"\s*Function : (\S+)", line)
        if m:
            flush()
            name, lines = m.group(1), []
            continue
        if name is None:
            continue
        # keep the instruction text, drop the address comments and encodings
        text = re.sub(r"/\*[0-9a-fx]+\*/", "", line).strip()
        if text:
            lines.append(text)
    flush()
    return out, sizes


a, sa = bodies(sys.argv[1])
b, sb = bodies(sys.argv[2])
only_a, only_b = a - b, b - a
print(f"{sum(a.values())} kernels in {sys.argv[1]}, {sum(b.values())} in {sys.argv[2]}")
print(f"identical bodies: {sum((a & b).values())}; only in first: {sum(only_a.values())}; only in second: {sum(only_b.values())}")
for d in list(only_a)[:5]:
    print("  only first:", d, sa[d], "lines")
for d in list(only_b)[:5]:
    print("  only second:", d, sb[d], "lines")
sys.exit(0 if not only_a and not only_b else 1)
