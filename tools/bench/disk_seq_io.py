#!/usr/bin/env python3
"""Raw sequential write and read speed of one drive, with the OS file cache bypassed.

Feeds the disk-tier cost model (context store hydration, write-on-evict): what a streaming reader
could get from the drive, as opposed to the store's end-to-end read path. For each block size the
script writes a scratch file of `--size-gib` with unbuffered, write-through I/O, reads it back
unbuffered, and deletes it. With `--buffered` it then reads the file twice more through the OS
file cache (the first of those comes from disk, the second from RAM, which is what a warm store
read sees).

Windows: CreateFileW with FILE_FLAG_NO_BUFFERING (+ FILE_FLAG_WRITE_THROUGH for writes) and
page-aligned buffers. Linux: O_DIRECT (+ O_DSYNC for writes). Every transfer is one block of the
given size at an aligned offset, issued one at a time (queue depth 1).

    python tools/bench/disk_seq_io.py --dir %TEMP% --size-gib 4 --block-mib 4 8 16 32
    python tools/bench/disk_seq_io.py --dir /tmp --size-gib 2 --repeats 3 --json out.json

The file holds random bytes, so a compressing controller cannot shortcut it. Use a directory on
the drive under test; the script refuses to overwrite an existing file and always removes its own
scratch file, even on failure. Disk activity from other programs during the run lowers the
result: say so when quoting a number.
"""

from __future__ import annotations

import argparse
import json
import mmap
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any

MIB = 1024 * 1024
GIB = 1024 * MIB


class _WindowsFile:
    GENERIC_READ = 0x80000000
    GENERIC_WRITE = 0x40000000
    CREATE_NEW = 1
    OPEN_EXISTING = 3
    FILE_FLAG_NO_BUFFERING = 0x20000000
    FILE_FLAG_WRITE_THROUGH = 0x80000000
    FILE_FLAG_SEQUENTIAL_SCAN = 0x08000000
    FILE_ATTRIBUTE_NORMAL = 0x80

    def __init__(self, path: Path, write: bool, unbuffered: bool) -> None:
        import ctypes
        from ctypes import wintypes

        self._ctypes = ctypes
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.CreateFileW.restype = wintypes.HANDLE
        k32.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
                                    wintypes.LPVOID, wintypes.DWORD, wintypes.DWORD,
                                    wintypes.HANDLE]
        k32.ReadFile.argtypes = [wintypes.HANDLE, wintypes.LPVOID, wintypes.DWORD,
                                 ctypes.POINTER(wintypes.DWORD), wintypes.LPVOID]
        k32.WriteFile.argtypes = [wintypes.HANDLE, wintypes.LPCVOID, wintypes.DWORD,
                                  ctypes.POINTER(wintypes.DWORD), wintypes.LPVOID]
        k32.CloseHandle.argtypes = [wintypes.HANDLE]
        self._k32 = k32
        self._wintypes = wintypes
        flags = self.FILE_ATTRIBUTE_NORMAL
        if unbuffered:
            flags |= self.FILE_FLAG_NO_BUFFERING
            if write:
                flags |= self.FILE_FLAG_WRITE_THROUGH
        else:
            flags |= self.FILE_FLAG_SEQUENTIAL_SCAN
        access = self.GENERIC_WRITE if write else self.GENERIC_READ
        disposition = self.CREATE_NEW if write else self.OPEN_EXISTING
        handle = k32.CreateFileW(str(path), access, 0, None, disposition, flags, None)
        if handle is None or handle == wintypes.HANDLE(-1).value:
            raise OSError(ctypes.get_last_error(), f"CreateFileW failed for {path}")
        self._handle = handle

    def _address(self, buffer: mmap.mmap) -> int:
        return self._ctypes.addressof(self._ctypes.c_char.from_buffer(buffer))

    def write(self, buffer: mmap.mmap, size: int) -> None:
        done = self._wintypes.DWORD(0)
        if not self._k32.WriteFile(self._handle, self._address(buffer), size,
                                   self._ctypes.byref(done), None) or done.value != size:
            raise OSError(self._ctypes.get_last_error(), "WriteFile failed")

    def read(self, buffer: mmap.mmap, size: int) -> int:
        done = self._wintypes.DWORD(0)
        if not self._k32.ReadFile(self._handle, self._address(buffer), size,
                                  self._ctypes.byref(done), None):
            raise OSError(self._ctypes.get_last_error(), "ReadFile failed")
        return done.value

    def close(self) -> None:
        self._k32.CloseHandle(self._handle)


class _PosixFile:
    def __init__(self, path: Path, write: bool, unbuffered: bool) -> None:
        flags = (os.O_WRONLY | os.O_CREAT | os.O_EXCL) if write else os.O_RDONLY
        if unbuffered:
            flags |= getattr(os, "O_DIRECT", 0)
            if write:
                flags |= getattr(os, "O_DSYNC", 0)
        self._fd = os.open(path, flags, 0o600)

    def write(self, buffer: mmap.mmap, size: int) -> None:
        view = memoryview(buffer)[:size]
        while view:
            view = view[os.write(self._fd, view):]

    def read(self, buffer: mmap.mmap, size: int) -> int:
        return os.readv(self._fd, [memoryview(buffer)[:size]])

    def close(self) -> None:
        os.close(self._fd)


def open_file(path: Path, write: bool, unbuffered: bool) -> Any:
    if sys.platform == "win32":
        return _WindowsFile(path, write, unbuffered)
    return _PosixFile(path, write, unbuffered)


def drop_cache_hint(path: Path) -> None:
    """Best effort on Linux; unbuffered I/O on Windows never fills the cache in the first place."""
    if hasattr(os, "posix_fadvise"):
        fd = os.open(path, os.O_RDONLY)
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)


def read_all(path: Path, buffer: mmap.mmap, block: int, unbuffered: bool) -> tuple[int, float]:
    handle = open_file(path, write=False, unbuffered=unbuffered)
    try:
        start = time.perf_counter()
        total = 0
        while True:
            got = handle.read(buffer, block)
            if got <= 0:
                break
            total += got
        return total, time.perf_counter() - start
    finally:
        handle.close()


def one_pass(path: Path, size: int, block: int, buffered: bool) -> dict[str, float]:
    buffer = mmap.mmap(-1, block)  # page aligned, as unbuffered I/O requires
    blocks = size // block
    result: dict[str, float] = {}
    owned = False  # only unlink a file our own exclusive create made
    try:
        handle = open_file(path, write=True, unbuffered=True)
        owned = True
        try:
            elapsed = 0.0
            for _ in range(blocks):
                buffer[:] = os.urandom(block)  # fresh incompressible block, outside the timing
                start = time.perf_counter()
                handle.write(buffer, block)
                elapsed += time.perf_counter() - start
            result["write_gbps"] = blocks * block / elapsed / 1e9
        finally:
            handle.close()
        drop_cache_hint(path)
        total, seconds = read_all(path, buffer, block, unbuffered=True)
        if total != blocks * block:
            raise RuntimeError(f"read {total} bytes, wrote {blocks * block}")
        result["read_gbps"] = total / seconds / 1e9
        if buffered:
            for label in ("buffered_cold_gbps", "buffered_warm_gbps"):
                total, seconds = read_all(path, buffer, block, unbuffered=False)
                result[label] = total / seconds / 1e9
    finally:
        buffer.close()
        if owned:
            path.unlink(missing_ok=True)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dir", type=Path, required=True,
                        help="directory on the drive under test (a scratch file is made there)")
    parser.add_argument("--size-gib", type=float, default=4.0, help="scratch file size (default 4)")
    parser.add_argument("--block-mib", type=int, nargs="+", default=[4, 8, 16, 32],
                        help="block sizes to sweep, MiB (default 4 8 16 32)")
    parser.add_argument("--repeats", type=int, default=2, help="passes per block size (default 2)")
    parser.add_argument("--buffered", action="store_true",
                        help="also read through the OS file cache (from disk, then from RAM)")
    parser.add_argument("--json", type=Path, help="write every pass and the medians here")
    args = parser.parse_args()

    if not args.dir.is_dir():
        print(f"error: {args.dir} is not a directory", file=sys.stderr)
        return 2
    rows: list[dict[str, Any]] = []
    header = f"{'block MiB':>9} {'pass':>4} {'write GB/s':>10} {'read GB/s':>9}"
    if args.buffered:
        header += f" {'buf disk':>8} {'buf RAM':>8}"
    print(header)
    for block_mib in args.block_mib:
        block = block_mib * MIB
        size = max(block, int(args.size_gib * GIB) // block * block)
        for repeat in range(args.repeats):
            path = args.dir / f"disk_seq_io_{os.getpid()}_{block_mib}_{repeat}.tmp"
            if path.exists():
                print(f"error: {path} exists; refusing to overwrite", file=sys.stderr)
                return 2
            result = one_pass(path, size, block, args.buffered)
            rows.append({"block_mib": block_mib, "pass": repeat, "bytes": size, **result})
            line = (f"{block_mib:>9} {repeat:>4} {result['write_gbps']:>10.2f} "
                    f"{result['read_gbps']:>9.2f}")
            if args.buffered:
                line += (f" {result['buffered_cold_gbps']:>8.2f} "
                         f"{result['buffered_warm_gbps']:>8.2f}")
            print(line, flush=True)
    medians: dict[int, dict[str, float]] = {}
    for block_mib in args.block_mib:
        selected = [r for r in rows if r["block_mib"] == block_mib]
        medians[block_mib] = {key: statistics.median(r[key] for r in selected)
                              for key in selected[0] if key.endswith("_gbps")}
    print("medians (GB/s):", json.dumps(medians))
    if args.json:
        args.json.write_text(json.dumps({"size_gib": args.size_gib, "passes": rows,
                                         "medians": medians}, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
