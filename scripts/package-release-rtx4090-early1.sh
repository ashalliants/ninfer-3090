#!/usr/bin/env bash
# Linux counterpart to package-release-rtx4090-early1.ps1.
#
# This one existed only as PowerShell, and check-linux-scripts.sh had to name it in a
# `windows_only` exemption list so the counterpart rule would pass at all. An exemption list is a
# thing that rots: the next Windows-only packaging script gets added to it rather than paired, and
# nothing says so out loud. Writing the counterpart empties the list instead.
#
# It mirrors the PowerShell exactly, with the two differences the platforms force:
#   * the Ninja/Makefile builds Linux uses are single-config, so products sit at apps/ and bench/
#     rather than under an apps/Release/ subdirectory that MSBuild creates;
#   * no DLLs are copied -- the shared-library story on Linux is the system CUDA runtime, which is
#     not shipped in the archive.
#
#   NINFER_BUILD_ROOT=~/ninfer/build-sm89 scripts/package-release-rtx4090-early1.sh
set -euo pipefail

release_tag='0.6.0-rtx4090-early1'
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
build_root="${NINFER_BUILD_ROOT:-$repo_root/build-sm89}"
dist_root="$repo_root/dist"
product_name="ninfer-rtx4090-linux-x64-$release_tag"
product_root="$dist_root/$product_name"
archive_name="$product_name.tar.gz"
archive_path="$dist_root/$archive_name"
checksum_path="$dist_root/SHA256SUMS-v0.6.0-rtx4090-early1-linux.txt"

mkdir -p -- "$dist_root"
# Same guard as the PowerShell: refuse to write anywhere but the expected directory, since what
# follows is an rm -rf.
case "$product_root" in "$dist_root/$product_name") ;; *) exit 1 ;; esac
rm -rf -- "$product_root"
rm -f -- "$archive_path"
mkdir -- "$product_root"

for product in 'apps/ninfer:ninfer' 'apps/ninfer-serve:ninfer-serve' 'bench/ninfer_bench:ninfer_bench'; do
  source_path="$build_root/${product%%:*}"
  destination="$product_root/${product#*:}"
  [[ -f "$source_path" ]] || { printf 'Missing release product: %s\n' "$source_path" >&2; exit 1; }
  cp -- "$source_path" "$destination"
done

# This release predates the repository VERSION file, so the tag is written rather than copied --
# as the PowerShell does.
printf '%s\n' "$release_tag" > "$product_root/VERSION"
cp -- "$repo_root/LICENSE" "$product_root/"
cp -- "$repo_root/docs/rtx-4090-early.md" "$product_root/README.md"
cp -- "$repo_root/RELEASE_NOTES_0.6.0_RTX4090_EARLY1.md" "$product_root/"

(
  cd -- "$product_root"
  mapfile -d '' files < <(find . -maxdepth 1 -type f ! -name SHA256SUMS.txt -print0 | LC_ALL=C sort -z)
  sha256sum -- "${files[@]}" > SHA256SUMS.txt
)
tar -C "$dist_root" -czf "$archive_path" "$product_name"
(
  cd -- "$dist_root"
  sha256sum -- "$archive_name" > "$(basename -- "$checksum_path")"
)
du -h -- "$archive_path" "$checksum_path"
