#!/usr/bin/env bash
# Build the Linux release archive for the version in the repository's VERSION file.
#
#   NINFER_BUILD_ROOT=~/ninfer/build-linux scripts/package-release.sh
#   scripts/build.sh --package         configure + build, then this
#   NINFER_PACKAGE_TARGET=v100 NINFER_BUILD_ROOT=~/ninfer/build-v100 scripts/package-release.sh
#                                      the Tesla V100 (sm_70) archive from a scripts/v100_build.sh-style tree
#
# There is one packager, not one per release. Everything version-specific is derived from VERSION,
# whose content is the full release tag (for example 0.10.0-rtx3090): the text before the first `-`
# names RELEASE_NOTES_<version>.md and the checksum file, and the whole tag names the archive. To cut
# a release, bump VERSION and write its release notes; there is nothing here to copy and edit.
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

[[ -f "$repo_root/VERSION" ]] || { printf 'Missing %s\n' "$repo_root/VERSION" >&2; exit 1; }
release_tag="$(tr -d '[:space:]' < "$repo_root/VERSION")"
[[ -n "$release_tag" ]] || { printf '%s is empty\n' "$repo_root/VERSION" >&2; exit 1; }
release_version="${release_tag%%-*}"
release_notes="docs/release-notes/RELEASE_NOTES_$release_version.md"
# Checked before anything is deleted or built into dist, so a forgotten release-notes file costs
# nothing rather than a finished archive that has to be thrown away.
[[ -f "$repo_root/$release_notes" ]] || {
  printf 'Missing %s: write the release notes for %s before packaging it\n' \
    "$repo_root/$release_notes" "$release_tag" >&2
  exit 1
}

# One packager, two cards: the RTX 3090 archive (sm_86, the default) and the Tesla V100 archive (sm_70).
# A build holds one architecture, so each card has its own tree and its own archive.
target="${NINFER_PACKAGE_TARGET:-rtx3090}"
case "$target" in
  rtx3090) product_prefix="ninfer-rtx3090-linux-x64"; readme_source="docs/release-archive-linux.md"
           checksum_suffix="linux"; extra_scripts=(); products=('apps/ninfer-serve:ninfer-serve' 'bench/ninfer_bench:ninfer_bench') ;;
  v100)    product_prefix="ninfer-v100-linux-x64"; readme_source="docs/release-archive-v100.md"
           checksum_suffix="linux-v100"; extra_scripts=(run-v100.sh)
           products=('apps/ninfer-serve:ninfer-serve' 'bench/ninfer_bench:ninfer_bench') ;;
  *) printf 'NINFER_PACKAGE_TARGET must be rtx3090 or v100, not %s\n' "$target" >&2; exit 1 ;;
esac
build_root="${NINFER_BUILD_ROOT:-$repo_root/build-linux}"
dist_root="$repo_root/dist"
product_name="$product_prefix-$release_tag"
product_root="$dist_root/$product_name"
archive_name="$product_name.tar.gz"
archive_path="$dist_root/$archive_name"
checksum_path="$dist_root/SHA256SUMS-v$release_version-$checksum_suffix.txt"

mkdir -p -- "$dist_root"
case "$product_root" in "$dist_root/$product_name") ;; *) exit 1 ;; esac
rm -rf -- "$product_root"
rm -f -- "$archive_path"
# The checksum goes too, and before anything can fail. It is written last, so leaving a previous
# run's copy in place means a mid-run failure can leave `dist` holding a new archive beside a
# checksum for the old one -- and publishing that pair is worse than publishing neither.
rm -f -- "$checksum_path"
mkdir -- "$product_root"

for product in "${products[@]}"; do
  source_path="$build_root/${product%%:*}"
  destination="$product_root/${product#*:}"
  [[ -f "$source_path" ]] || { printf 'Missing release product: %s\n' "$source_path" >&2; exit 1; }
  # A tree built for the other card would package without complaint and then fail on first use, so
  # when cuobjdump is available the archive's server must hold machine code for this card.
  if [[ "$product" == 'apps/ninfer-serve:ninfer-serve' ]] && command -v cuobjdump >/dev/null; then
    case "$target" in rtx3090) want_arch=sm_86 ;; v100) want_arch=sm_70 ;; esac
    # Captured first: `grep -q` closing the pipe early would fail cuobjdump under pipefail.
    elf_list="$(cuobjdump --list-elf "$source_path")"
    grep -q "$want_arch" <<<"$elf_list" || {
      printf '%s holds no %s code: it was built for another card (target %s)\n' "$source_path" "$want_arch" "$target" >&2
      exit 1
    }
  fi
  cp -- "$source_path" "$destination"
done
# The CLI. `ninfer-serve` runs it when started under the name `ninfer` (apps/serve/main.cpp), so the archive
# carries one copy of every kernel and `ninfer` is a link to it: two separate 1.37 GB sm_70 binaries put the V100
# tarball over GitHub's 2 GiB asset limit (2,246,329,497 bytes with three, v0.20.0). A build from before that
# change has no such mode, so it still gets its own `ninfer` binary.
cli_probe="$(mktemp -d)"
ln -s "$product_root/ninfer-serve" "$cli_probe/ninfer"
if [[ "$("$cli_probe/ninfer" --version 2>/dev/null | head -n 1)" == "ninfer "[0-9]* ]]; then
  ln -s ninfer-serve "$product_root/ninfer"
  printf 'ninfer is a link to ninfer-serve (one copy of the kernels)\n'
else
  [[ -f "$build_root/apps/ninfer" ]] || { printf 'Missing release product: %s\n' "$build_root/apps/ninfer" >&2; exit 1; }
  cp -- "$build_root/apps/ninfer" "$product_root/ninfer"
  printf 'this build has no CLI mode in ninfer-serve: shipping a separate ninfer\n'
fi
rm -rf -- "$cli_probe"
cp -- "$repo_root/VERSION" "$repo_root/LICENSE" "$repo_root/$release_notes" "$product_root/"
# Apache-2.0 section 4(d): the conversation projector (lossy prefill's summary strategy) is a port of
# hermes-vcc-engine, so its NOTICE ships with every binary that contains it. LICENSE above is the
# Apache-2.0 text it refers to.
cp -- "$repo_root/src/runtime/engine/compaction/NOTICE" "$product_root/THIRD_PARTY_NOTICES.txt"
# The archive README must describe the archive. docs/rtx-3090-linux.md is a guide to *building* from
# source and points at ./scripts/download-model.sh; the packager copies that to the archive root, so
# a user following it from inside the archive got a missing-file error.
cp -- "$repo_root/$readme_source" "$product_root/README.md"
# Explicit rather than globbed, so a new script is shipped only once someone has decided it belongs
# in the archive.
cp -- "$repo_root"/scripts/{run.sh,download-model.sh} "$product_root/"
for extra in ${extra_scripts[@]+"${extra_scripts[@]}"}; do cp -- "$repo_root/scripts/$extra" "$product_root/"; done
# run.sh serves each profile from its --config file, which it looks for in profiles/ beside itself.
mkdir -- "$product_root/profiles"
cp -- "$repo_root"/scripts/profiles/*.json "$product_root/profiles/"
(
  cd -- "$product_root"
  mapfile -d '' files < <( { find . -maxdepth 1 \( -type f -o -type l \) ! -name SHA256SUMS.txt -print0
                             find ./profiles -maxdepth 1 -type f -print0; } | LC_ALL=C sort -z)
  sha256sum -- "${files[@]}" > SHA256SUMS.txt
)
tar -C "$dist_root" -czf "$archive_path" "$product_name"
(
  cd -- "$dist_root"
  sha256sum -- "$archive_name" > "$(basename -- "$checksum_path")"
)
du -h -- "$archive_path" "$checksum_path"
