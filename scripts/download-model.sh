#!/usr/bin/env bash
set -euo pipefail

# Downloads one pinned model artifact, verifies it, and only then promotes it to its final name.
#
#   download-model.sh <model>
#
#   qwen38-27b       Qwen3.8-27B, 17.7 GiB. Recommended. The default 27B for every benchmark in this
#                    repository.
#                    This fork's own conversion (WarlaxZ/Qwen3.8-27B-NInfer-3090, recipe
#                    qwen3_8_27b): the imatrix-weighted grouped_search encoder, a 4-bit embedding, a
#                    6-bit head and 4-bit mixer outputs and MLP down in layers 36-63. It carries the
#                    vision, MTP and DFlash2 weights, so one file serves --vision, --spec mtp and
#                    --spec dflash2. Executables older than the 5120x17408 Q4 linear_add route refuse
#                    it at startup. Figures published before 2026-10 were measured against the
#                    upstream neroued pin 1cbd84e7 (v2 18dfc887), 19.0 GiB.
#   bonsai2-27b      Ternary Bonsai 2 27B, 8.9 GiB. Prism ML's ternary Qwen3.8-27B (2.125 bits per
#                    weight, Hadamard-rotated) converted by this fork's bonsai2_27b_ternary recipe
#                    (WarlaxZ/Ternary-Bonsai-2-27B-NInfer-3090), with the MTP and DFlash2 heads
#                    ProCreations trained on it, the vision tower and a proposal head. Needs this
#                    fork's t2_g128_fp16 weight format: older executables refuse it at startup.
#   qwen36-27b       Qwen3.6-27B groupwise-int, 16.3 GiB, target_key qwen3_6_27b. A different model
#                    family from qwen3_8_27b, which is why having the latter does not satisfy the
#                    former: four real-model tests -- ninfer_qwen3_6_27b_prefix_real_test,
#                    _score_real_test, _load_plan_test and the Qwen3.6 27B half of the engine suite --
#                    skip without it.
#   qwen36-35b-a3b   Qwen3.6-35B-A3B, 21.2 GiB, the RTX 3090-compatible vision model. The official v3
#                    artifact revision: its artifact-manifest.json lists the text, vision, mtp and
#                    dflash components, and the DFlash bundle (from z-lab/Qwen3.6-35B-A3B-DFlash) is
#                    what --spec dflash needs.
#   qwen38-flash-next  Qwen3.8-Flash-Next (preview; the server runs it on Windows only), this fork's
#                    exact conversion of the ISTA DASLab GSQ-RCO IQ2_XS GGUF with Strata's MTP drafter
#                    and the vision tower (WarlaxZ/Qwen3.8-Flash-Next-NInfer-3090). Three files: the
#                    model, its .part-0001 continuation (both read once, at load) and the 27.1 GiB
#                    n-gram volume, read on every prompt. NINFER_NGRAM_DIR, or the answer to a question
#                    on a terminal, puts the volume in another folder (default: beside the model): a
#                    faster drive, since prefill from NVMe measured 1.1-1.4x that from a SATA SSD.
#                    The launcher is then told with NINFER_NGRAM_VOLUME, which the script prints.
#
# To repin a model, edit its block below: take the size and sha256 from what HuggingFace reports for
# the new revision (X-Linked-Size and X-Linked-ETag on the resolve URL, or the revision's
# artifact-manifest.json), and for qwen36-35b-a3b check the manifest still lists a dflash component.
# Changing a pin makes the staging hazard below live rather than hypothetical.
#
# Environment: NINFER_MODEL_DIR (default: models/ beside this script), NINFER_SKIP_SHA256=1 to accept
# a file on size alone. qwen38-27b also fetches the private `godmode` graft when HF_TOKEN (or
# `hf auth login`) grants access to it; NINFER_GRAFT_DIR and NINFER_GRAFTS=off as in run.bat.
#
# Uses aria2c (16 parallel ranges) when it is on PATH, since single-stream curl against these
# artifacts has been observed to throttle to ~1 MB/s or stall outright; falls back to curl
# otherwise. `apt-get install aria2` / `brew install aria2` to opt in.

usage() {
  printf 'usage: %s <model>\n\n  qwen38-27b (recommended)\n  bonsai2-27b\n  qwen36-27b\n  qwen36-35b-a3b\n  qwen38-flash-next (preview)\n' \
    "${0##*/}" >&2
}

# Only qwen38-flash-next has continuation parts and an n-gram volume.
part_sizes=() part_sha256s=() ngram_name=''
case "${1:-}" in
  qwen38-27b)
    artifact='qwen3_8_27b.ninfer'
    owner='WarlaxZ'
    repo='Qwen3.8-27B-NInfer-3090'
    revision='d47f2732d369acaec76dc44228f67c72b081df2f'
    expected_size=18982458624
    expected_sha256='7f2a0086a071ee932c66106e1552d118a85cfec8ddef13c67a54cd64abb39706'
    label='the Qwen3.8-27B model (17.7 GiB)'
    tests_variable='NINFER_QWEN3_8_27B_WEIGHTS'
    ;;
  bonsai2-27b)
    artifact='bonsai2_27b.ninfer'
    owner='WarlaxZ'
    repo='Ternary-Bonsai-2-27B-NInfer-3090'
    revision='983a9be22f3623f3a2799655741d2a92db86d37d'
    expected_size=9520051456
    expected_sha256='975126774d7e1f9ffa36b654c0a6fc0db42c773ba4dd5f1196a9ddc6a46d4634'
    label='the Ternary Bonsai 2 27B model (8.9 GiB)'
    tests_variable='NINFER_TEST_ARTIFACT'
    ;;
  qwen36-27b)
    artifact='qwen3_6_27b.ninfer'
    owner='neroued'
    repo='Qwen3.6-27B-NInfer'
    revision='3e3d9a3951c452c1ca80bd7a2860c7f3bfc5a829'
    expected_size=17495538688
    expected_sha256='9b610a7d051e7c4dbf89adb604bd269d248b643c8f6c7ef75bdb871af92c6f6b'
    label='the Qwen3.6-27B model (16.3 GiB)'
    tests_variable='NINFER_QWEN3_6_27B_WEIGHTS'
    ;;
  qwen36-35b-a3b)
    artifact='qwen3_6_35b_a3b.ninfer'
    owner='neroued'
    repo='Qwen3.6-35B-A3B-NInfer'
    revision='ee4495803bc4f8015b8a7e22d4cf9b67de8e27c6'
    expected_size=22790484480
    expected_sha256='3e33297645dc33557751be1a3c407a74ed7c00f34909b5d4e8cfdce91b3dbe84'
    label='the RTX 3090-compatible Qwen3.6-35B-A3B vision model (21.2 GiB)'
    tests_variable='NINFER_QWEN3_6_35B_A3B_WEIGHTS'
    ;;
  # The n-gram volume is pinned to the commit that uploaded it; the model file and its part are pinned
  # to the revision that published the model card with them. The volume is unchanged in that revision.
  qwen38-flash-next)
    artifact='qwen4exp-iq2xs-mtp-vision.ninfer'
    owner='WarlaxZ'
    repo='Qwen3.8-Flash-Next-NInfer-3090'
    revision='c7dfb9694fc3cb3d40298afbfe2e3a8c172241fa'
    expected_size=32000000000
    expected_sha256='87c871e6cc1b6392e387104e8dfb159283a8a64247f669fbf33355b19b769ef4'
    part_sizes=(9012140032)
    part_sha256s=('1798c63c2f74a05cac71a609701f34368f0ba1e1eeebbf1b7641e83d0dd8d811')
    ngram_name='qwen4exp-iq2xs-mtp-vision.ninfer.ngram'
    ngram_revision='95b1b0169bd25db96ceb772a8a67cc6e87088a1f'
    ngram_size=29127258112
    ngram_sha256='7623459b2385eef4505891c2ff133b3dd8ad4ecd0ad334c901878f3fa1d1b051'
    label='the Qwen3.8-Flash-Next model (38.2 GiB in two files)'
    tests_variable='NINFER_TEST_QWEN4EXP_ARTIFACT'
    ;;
  -h|--help) usage; exit 0 ;;
  *) usage; exit 2 ;;
esac

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
model_dir="${NINFER_MODEL_DIR:-$root/models}"
model="$model_dir/$artifact"

file_size() { wc -c < "$1" | tr -d '[:space:]'; }

# Verifies a file against expected_size and, unless NINFER_SKIP_SHA256=1, expected_sha256.
# Used both for an existing "$model" (so a same-sized-but-corrupt file is not accepted forever
# just because it happened to pass once, or was replaced out from under this script) and for a
# freshly downloaded "$part" -- one check that cannot drift out of sync with itself. A missing
# sha256sum/shasum fails closed rather than silently promoting an unverified file: the whole
# reason these artifacts are checksummed is to catch a same-sized-but-corrupt file, and silently
# skipping that would defeat it, not just once, but for every future run on that host.
verify() {
  [ "$(file_size "$1")" = "$expected_size" ] || return 1
  [ "${NINFER_SKIP_SHA256:-0}" = '1' ] && return 0
  local actual
  # Hashed from stdin: given a name with a backslash (a Windows path under Git Bash), sha256sum
  # escapes its output line with a leading "\", which would make every hash a mismatch.
  if command -v sha256sum >/dev/null 2>&1; then
    actual="$(sha256sum < "$1" | cut -d' ' -f1)"
  elif command -v shasum >/dev/null 2>&1; then
    actual="$(shasum -a 256 < "$1" | cut -d' ' -f1)"
  else
    printf 'No sha256sum or shasum found; cannot verify %s. Set NINFER_SKIP_SHA256=1 to accept it unverified.\n' "$1" >&2
    return 1
  fi
  [ "$actual" = "$expected_sha256" ]
}

# The qwen38-27b `godmode` graft lives in a private repo. Fetching it is best-effort: with no token,
# or a token without access, the model download still succeeds and the graft is simply skipped.
# Both files are pinned to a revision and sha256, and sit in artifacts/grafts/ beside the repo root,
# where run.bat looks (NINFER_GRAFT_DIR overrides). NINFER_GRAFTS=off skips the fetch.
fetch_graft() {
  [ "${1:-}" = 'qwen38-27b' ] && [ "${NINFER_GRAFTS:-}" != 'off' ] || return 0
  local repo='WarlaxZ/Qwen3.8-27B-godmode-graft' rev='97a02e9bbb37d76af63256152c07fc4eeb4c40c9'
  local dir="${NINFER_GRAFT_DIR:-$root/../artifacts/grafts}"
  local names=(godmode_q38_trained.bin godmode_q38_trained.json)
  local shas=(a0ddecc4ca5db8b84f600d60fbf238c97e21a74069b7182e0e0e595d7a22c391
              2a832ea95cc1dcee1dff2b292b10ef98913594eaf1f74121b01dd03a23dee3b1)
  local i complete=1
  # The graft is a pair the launcher trusts by file presence and the loader needs whole, so every
  # failure below leaves neither file behind: a lone or mismatched file would break a later launch.
  for i in 0 1; do
    { [ -f "$dir/${names[i]}" ] && graft_sha_ok "$dir/${names[i]}" "${shas[i]}"; } || complete=0
  done
  if [ "$complete" = 1 ]; then
    printf 'Graft ready: %s/%s\n' "$dir" "${names[0]}"
    return 0
  fi
  # The token is HF_TOKEN, else the file huggingface_hub reads: HF_TOKEN_PATH, or token under HF_HOME,
  # which itself defaults to huggingface/ under XDG_CACHE_HOME and then ~/.cache.
  local token="${HF_TOKEN:-}" tokfile
  if [ -n "${HF_TOKEN_PATH:-}" ]; then tokfile="$HF_TOKEN_PATH"
  else tokfile="${HF_HOME:-${XDG_CACHE_HOME:-$HOME/.cache}/huggingface}/token"; fi
  [ -n "$token" ] || { [ -f "$tokfile" ] && token="$(tr -d '[:space:]' < "$tokfile")"; } || true
  if [ -z "$token" ]; then
    graft_drop_half "$dir"
    printf 'Graft: skipped (no Hugging Face token; set HF_TOKEN or run `hf auth login` if you have access).\n'
    return 0
  fi
  if ! mkdir -p -- "$dir" 2>/dev/null; then
    printf 'Graft: skipped (cannot create %s).\n' "$dir" >&2
    return 0
  fi
  # Stage both files and verify both; only then publish the pair.
  for i in 0 1; do
    if ! curl -sSL --fail -H "Authorization: Bearer $token" --output "$dir/${names[i]}.part" \
         "https://huggingface.co/$repo/resolve/$rev/${names[i]}" 2>/dev/null; then
      graft_discard "$dir"
      printf 'Graft: skipped (no access to %s with this token).\n' "$repo"
      return 0
    fi
    if ! graft_sha_ok "$dir/${names[i]}.part" "${shas[i]}"; then
      graft_discard "$dir"
      printf 'Graft: %s failed sha256 verification; not installed.\n' "${names[i]}" >&2
      return 0
    fi
  done
  # Any pair already here is invalid or partial (a verified one returned above). Clear it, binary first,
  # so the new pair never sits beside a leftover half of the old one. Then install the sidecar first and
  # the binary last: run.bat and the launcher trust the .bin, so it must never be visible without its .json.
  rm -f -- "$dir/${names[0]}" "$dir/${names[1]}" 2>/dev/null || true
  for i in 1 0; do
    if ! mv -f -- "$dir/${names[i]}.part" "$dir/${names[i]}" 2>/dev/null; then
      graft_discard "$dir"
      printf 'Graft: skipped (cannot install into %s).\n' "$dir" >&2
      return 0
    fi
  done
  printf 'Graft ready: %s/%s\n' "$dir" "${names[0]}"
}

# Without a token nothing can be fetched, but a lone half of the pair or a staged .part left by an
# interrupted fetch would still break a launch (the launcher trusts the .bin, the loader needs the
# .json), so drop those. A whole pair is left alone even if it does not match the pinned hashes: it
# may be one the user supplied.
graft_drop_half() {
  local bin="$1/godmode_q38_trained.bin" json="$1/godmode_q38_trained.json"
  rm -f -- "$bin.part" "$json.part" 2>/dev/null || true
  if [ -f "$bin" ] && [ ! -f "$json" ]; then rm -f -- "$bin" 2>/dev/null || true
  elif [ -f "$json" ] && [ ! -f "$bin" ]; then rm -f -- "$json" 2>/dev/null || true; fi
}

# Removes the staged and installed graft files so a failed fetch never leaves a partial or mixed pair.
graft_discard() {
  rm -f -- "$1"/godmode_q38_trained.bin "$1"/godmode_q38_trained.json \
           "$1"/godmode_q38_trained.bin.part "$1"/godmode_q38_trained.json.part 2>/dev/null || true
}

graft_sha_ok() {
  local actual
  if command -v sha256sum >/dev/null 2>&1; then actual="$(sha256sum -- "$1" | cut -d' ' -f1)"
  elif command -v shasum >/dev/null 2>&1; then actual="$(shasum -a 256 -- "$1" | cut -d' ' -f1)"
  else return 1; fi
  [ "$actual" = "$2" ]
}

# Fetches one pinned file: folder, name, size, sha256, revision, description, and kind (model, part or
# volume). Returns non-zero unless the file ends at its final name, verified. A file already there that
# verifies is kept without any download.
#
# The download is staged under a revision-scoped name so that curl -C - can only ever resume the same
# file. curl -C - resumes by appending at the current file length, without checking what wrote those
# bytes, so a leftover partial from a different revision -- for instance the previous pin's, on any
# machine that ran an older script -- would be spliced onto the head of this one: a file of entirely
# plausible size that is corrupt throughout. A name that carries the revision means a resume can only
# ever continue the same file, and the checks below are what promote it to the final name.
fetch_file() {
  local dir="$1" name="$2" expected_size="$3" expected_sha256="$4" rev="$5" what="$6" kind="$7"
  local path="$dir/$name" part="$dir/$name.$rev.part" url status=0
  if [ -f "$path" ]; then
    if verify "$path"; then
      if [ "$kind" = model ]; then printf 'Model already present: %s\n' "$path"
      else printf 'Already present: %s\n' "$path"; fi
      return 0
    fi
    printf '%s\n' "Existing $path did not verify against revision $rev; fetching the pinned one." >&2
    # A v2 file from an earlier release does not load any more, but it does not need to be fetched
    # again either: tools/upgrade_ninfer_v2_to_v3.py rewrites it locally in a couple of minutes.
    if [ "$kind" = model ] && [ -z "$ngram_name" ]; then
      printf '%s\n' "If it is the v2 file from an earlier release, you can upgrade it instead of downloading:" \
        "  python tools/upgrade_ninfer_v2_to_v3.py $path $path.v3 && mv $path.v3 $path" >&2
    fi
  fi

  printf '%s\n' "Downloading $what..."
  url="https://huggingface.co/$owner/$repo/resolve/$rev/$name"
  if command -v aria2c >/dev/null 2>&1; then
    # 16 parallel ranges over one file, not single-stream curl. On the Vast rental boxes this repo
    # downloads at ~98 MB/s over aria2c against the same throttled-to-~1MB/s (and once outright
    # stalled at 11 of 21 GiB) single-stream curl -- see scripts/multi-gpu-testing/README.md. -c
    # resumes into the same .part name curl would use, and HuggingFace serves ranges fine.
    aria2c -x16 -s16 -c --file-allocation=none -d "$dir" -o "$(basename -- "$part")" "$url" || status=$?
  else
    curl -L -C - --fail --output "$part" "$url" || status=$?
  fi
  if [ "$status" -ne 0 ]; then
    printf '%s\n' 'Download failed. Run this script again to resume.' >&2
    return 1
  fi

  if ! verify "$part"; then
    printf 'Downloaded file at %s failed verification against revision %s. Delete it and run this script again.\n' \
      "$part" "$rev" >&2
    return 1
  fi
  mv -f -- "$part" "$path"
}

# The folder for the n-gram volume: NINFER_NGRAM_DIR, else the answer to a question when there is a
# terminal to ask on, else the model folder.
choose_ngram_dir() {
  local answer=''
  ngram_dir="${NINFER_NGRAM_DIR:-$model_dir}"
  if [ -z "${NINFER_NGRAM_DIR:-}" ] && [ -t 0 ]; then
    printf '\n%s\n%s\n%s\n\n  Model folder: %s\n\n' \
      'Qwen3.8-Flash-Next reads its 27.1 GiB n-gram volume on every prompt, and the model only once,' \
      'at load. Prompts were processed 1.1-1.4x faster with the volume on an NVMe drive than on a' \
      'SATA SSD, so put it on your fastest drive (28 GiB free). The model can stay where it is.' \
      "$model_dir"
    read -r -p 'Folder for the n-gram volume (Enter = the model folder): ' answer || answer=''
    [ -n "$answer" ] && ngram_dir="$answer"
  fi
  if ! mkdir -p -- "$ngram_dir" 2>/dev/null; then
    printf 'Cannot create the folder %s for the n-gram volume.\n' "$ngram_dir" >&2
    return 1
  fi
  printf 'N-gram volume folder: %s\n' "$ngram_dir"
}

mkdir -p -- "$model_dir"

# Ask where the n-gram volume goes before anything is fetched, so the downloads can run unattended.
if [ -n "$ngram_name" ]; then choose_ngram_dir || exit 1; fi

fetch_file "$model_dir" "$artifact" "$expected_size" "$expected_sha256" "$revision" "$label" model || exit 1
for i in "${!part_sizes[@]}"; do
  fetch_file "$model_dir" "$artifact.part-$(printf '%04d' $((i + 1)))" "${part_sizes[i]}" "${part_sha256s[i]}" \
    "$revision" "part $((i + 1)) of the model" part || exit 1
done
if [ -n "$ngram_name" ]; then
  fetch_file "$ngram_dir" "$ngram_name" "$ngram_size" "$ngram_sha256" "$ngram_revision" \
    'the n-gram volume (27.1 GiB)' volume || exit 1
fi

printf 'Model ready: %s\n' "$model"
if [ -n "$ngram_name" ]; then
  printf 'N-gram volume ready: %s\n' "$ngram_dir/$ngram_name"
  # The server reads <model>.ngram beside the model unless --ngram-volume (in run.bat,
  # NINFER_NGRAM_VOLUME) names another file. Flash-Next is served on Windows only.
  if [ "$ngram_dir" != "$model_dir" ]; then
    printf 'It is not beside the model, so serve with --ngram-volume "%s"\n' "$ngram_dir/$ngram_name"
    printf '(run.bat on Windows: set NINFER_NGRAM_VOLUME to that file).\n'
  fi
fi
printf 'Point the tests at it with:  export %s=%s\n' "$tests_variable" "$model"
fetch_graft "$1"
