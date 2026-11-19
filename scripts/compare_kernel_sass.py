"""Show that two builds of a CUDA translation unit contain the same machine code, kernel by kernel.

A refactor that only threads a template parameter through (so every kernel's mangled name changes) must leave the
SASS of each kernel exactly as it was; scripts/compare_kernel_symbols.sh checks names and resource usage, this checks
the instructions. Symbol names, local labels and instruction addresses are ignored; each kernel is identified by the
text of its instructions.

Kernels are compared in the order the object holds them (the compiler emits them in source order, which a change
that only adds a template parameter does not disturb), so a kernel is matched to the one at its own position, never to
another kernel that happens to have the same body: two kernels that traded bodies are a difference.

    cuobjdump -sass before/unit.cu.o > before.txt        # the object of the unit, before and after the change
    cuobjdump -sass after/unit.cu.o  > after.txt
    python scripts/compare_kernel_sass.py before.txt after.txt
    python scripts/compare_kernel_sass.py --allow-added before.txt after.txt     # new kernels are fine

Exit status 0 when every kernel matches, 1 when any differs (the first few are listed), 2 when an input holds no
kernel (an empty dump is a failed disassembly, not evidence). With --allow-added the second dump may hold kernels the
first does not (a change that adds an instantiation); every kernel of the first must still be in the second, unchanged,
in the same relative order.
"""
import hashlib
import re
import sys


def kernels(path):
    """[(digest, instruction count)] of every kernel in the dump, in order."""
    out, name, lines = [], None, []

    def flush():
        if name is not None:
            body = "\n".join(lines)
            body = re.sub(r"_Z\w+", "SYM", body)  # mangled names, including call targets and descriptors
            body = re.sub(r"\.L_\d+", ".L", body)  # local labels renumber with code order, not meaning
            out.append((hashlib.sha256(body.encode()).hexdigest()[:16], len(lines)))

    for line in open(path, errors="replace"):
        m = re.match(r"\s*Function : (\S+)", line)
        if m:
            flush()
            name, lines = m.group(1), []
            continue
        if name is None:
            continue
        # keep the instruction text: drop every comment (addresses, and on older architectures the instruction
        # encodings, which are padded to the widest line of the whole dump and so differ between dumps)
        text = re.sub(r"/\*.*?\*/", "", line).strip()
        if text:
            lines.append(text)
    flush()
    return out


def main(argv):
    args = [arg for arg in argv if not arg.startswith("--")]
    allow_added = "--allow-added" in argv
    if len(args) != 2:
        print(__doc__, file=sys.stderr)
        return 2
    a, b = kernels(args[0]), kernels(args[1])
    print(f"{len(a)} kernels in {args[0]}, {len(b)} in {args[1]}")
    if not a or not b:
        print("an input holds no kernel: not a valid cuobjdump -sass dump of a CUDA object", file=sys.stderr)
        return 2
    if allow_added:
        # Every kernel of the first, in order, must be the next matching kernel of the second.
        at = 0
        for index, (digest, size) in enumerate(a):
            while at < len(b) and b[at][0] != digest:
                at += 1
            if at == len(b):
                print(f"  kernel {index} of the first ({digest}, {size} instructions) is not in the second, in order",
                      file=sys.stderr)
                return 1
            at += 1
        print(f"all {len(a)} kernels of the first are in the second in order, unchanged; added in the second: {len(b) - len(a)}")
        return 0
    differing = [i for i in range(min(len(a), len(b))) if a[i][0] != b[i][0]]
    print(f"identical kernels: {len(a) - len(differing) - max(0, len(a) - len(b))}; differing: {len(differing)}; "
          f"only in first: {max(0, len(a) - len(b))}; only in second: {max(0, len(b) - len(a))}")
    for i in differing[:5]:
        print(f"  kernel {i}: {a[i][0]} ({a[i][1]} instructions) vs {b[i][0]} ({b[i][1]} instructions)")
    return 0 if not differing and len(a) == len(b) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
