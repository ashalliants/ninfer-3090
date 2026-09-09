#!/usr/bin/env bash
set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
tmp="$(mktemp -d)"
trap 'rm -rf -- "$tmp"' EXIT

for script in "$root"/*.sh; do
  bash -n "$script"
done

# CRLF in a shell script is not cosmetic. Inside a `case` block the stray CR becomes part of the
# `in` token and Linux bash refuses the file outright -- which is exactly what happened to
# run-qwen36-35b-a3b-c1-maxctx.sh and run-qwen38-c1-maxctx.sh, the two launchers the README
# recommends as the Linux entry point. `bash -n` above catches that particular shape, but a CRLF
# script without a `case` parses and then misbehaves at runtime instead, so check the bytes.
#
# -U forces binary matching. Without it, Git Bash's grep translates CR away and reports every file
# clean on a Windows checkout, which is how this survived as long as it did.
#
# Scoped to the whole repository rather than scripts/, because .gitattributes states the policy
# repository-wide and eval/ already holds three shell scripts that a scripts/-only check would
# never look at. Driven off `git ls-files` so a new directory is covered the day it appears
# instead of the day someone remembers to add a glob here.
repo_root="$(git -C "$root" rev-parse --show-toplevel 2>/dev/null || printf '%s' "$root/..")"
crlf=''
while IFS= read -r candidate; do
  [[ -f "$repo_root/$candidate" ]] || continue
  if LC_ALL=C grep -qU $'\r' "$repo_root/$candidate"; then
    crlf+="  $candidate"$'\n'
  fi
done < <(git -C "$repo_root" ls-files '*.sh' '*.bash' 2>/dev/null)

if [[ -n "$crlf" ]]; then
  printf 'Shell scripts with CRLF line endings (must be LF; see .gitattributes):\n%s' "$crlf" >&2
  printf 'Linux bash rejects these outright inside a `case` block.\n' >&2
  exit 1
fi
# Every Windows script needs a Bash sibling. There used to be an exemption list here holding
# package-release-rtx4090-early1.ps1; the counterpart was written instead, so the list is gone
# rather than empty. An exemption list is where the next Windows-only script quietly goes.
#
# Accumulate rather than exiting on the first miss. This loop used to `exit 1` immediately, and
# because it was failing on a cosmetic counterpart rule, none of the downloader coverage below it
# ran at all -- which is how the curl fixture further down went stale unnoticed. A guard that
# stops at its first complaint disables everything after it.
missing=''
for windows_script in "$root"/*.bat "$root"/*.ps1; do
  counterpart="${windows_script%.*}.sh"
  [[ -f "$counterpart" ]] || missing+="  $(basename -- "$counterpart")"$'
'
done
if [[ -n "$missing" ]]; then
  printf 'Missing Bash counterpart for a Windows script:
%s' "$missing" >&2
  exit 1
fi

# The executable bit, read from the index rather than the filesystem. On a Windows checkout every
# file looks executable to bash, and under WSL /mnt/c reports 777 for everything, so `-x` on the
# working tree cannot see a mode-644 script here at all -- while a real Linux checkout gets a file
# that will not run. scripts/package-release-v090.sh, the current release's Linux packager, sat at
# 644 for exactly that reason. `git ls-files -s` reports what is committed, which is the thing
# that actually reaches a Linux user.
not_executable=''
while IFS= read -r entry; do
  mode="${entry%% *}"
  path="${entry#*$'	'}"
  [[ "$mode" == '100755' ]] || not_executable+="  $path (mode $mode)"$'
'
done < <(git -C "$repo_root" ls-files -s '*.sh' '*.bash' 2>/dev/null)
if [[ -n "$not_executable" ]]; then
  printf 'Shell scripts committed without the executable bit:
%s' "$not_executable" >&2
  printf 'Fix with: git update-index --chmod=+x <path>
' >&2
  exit 1
fi

cat > "$tmp/ninfer-serve" <<'SERVER'
#!/usr/bin/env bash
printf '%s\n' "$@" > "$NINFER_TEST_ARGS"
SERVER
chmod +x "$tmp/ninfer-serve"
touch "$tmp/qwen3_8_27b.ninfer" "$tmp/qwen3_6_35b_a3b.ninfer"

# Hermetic fixtures. A developer's shell may already export NINFER_MODEL, NINFER_HOST and friends
# -- the machine this was written on exports both -- and inheriting them makes every assertion below
# describe that box rather than the layout under test. The first draft of the archive case failed
# exactly that way, against a NINFER_MODEL naming a directory that no longer exists. Clear every
# NINFER_* and pass back only what a case sets deliberately.
ninfer_clear=()
while IFS= read -r name; do
  ninfer_clear+=(-u "$name")
done < <(env | sed -n 's/^\(NINFER_[A-Za-z0-9_]*\)=.*/\1/p')
clear_env() { env ${ninfer_clear[@]+"${ninfer_clear[@]}"} "$@"; }

launchers=(
  'run-qwen38-c1.sh:qwen3_8_27b.ninfer:--max-context:65536'
  'run-qwen38-c8.sh:qwen3_8_27b.ninfer:--max-concurrency:8'
  'run-qwen38-vision.sh:qwen3_8_27b.ninfer:--vision:--spec'
  'run-qwen36-35b-vision.sh:qwen3_6_35b_a3b.ninfer:--vision:--no-thinking'
)
for entry in "${launchers[@]}"; do
  IFS=: read -r script model expected value <<< "$entry"
  args="$tmp/${script%.sh}.args"
  clear_env NINFER_SERVER="$tmp/ninfer-serve" NINFER_TEST_ARGS="$args" \
    "$root/$script" "$tmp/$model" >/dev/null
  grep -Fx -- "$expected" "$args" >/dev/null
  grep -Fx -- "$value" "$args" >/dev/null
done

# The two -maxctx launchers ship *inside* the release archive as well as living here, and README
# calls them the Linux entry point. They resolved both the server and the artifact from
# `dirname(script)/..` -- the archive's parent once packaged -- so the documented quick start exited
# before serving, with a "Missing ninfer-serve" naming a path outside the archive. Nothing tested
# that, because every case above hands the launcher an explicit NINFER_SERVER and model path.
#
# So drive them the way a user does: unpack and run, flat layout, no environment overrides beyond
# the stub hook. The launcher must find the binary beside itself and the artifact under its own
# models/, not one directory up.
archive="$tmp/archive"
mkdir -- "$archive" "$archive/models"
cp -- "$tmp/ninfer-serve" "$archive/ninfer-serve"
for entry in 'run-qwen38-c1-maxctx.sh:qwen3_8_27b.ninfer' \
             'run-qwen36-35b-a3b-c1-maxctx.sh:qwen3_6_35b_a3b.ninfer'; do
  IFS=: read -r script model <<< "$entry"
  : > "$archive/models/$model"
  cp -- "$root/$script" "$archive/$script"
  args="$tmp/${script%.sh}.archive.args"
  ( cd -- "$archive" && clear_env NINFER_TEST_ARGS="$args" "./$script" >/dev/null )
  if ! grep -Fx -- "$archive/models/$model" "$args" >/dev/null; then
    printf '%s did not resolve its artifact inside the archive: %s\n' "$script" "$(head -1 -- "$args")" >&2
    exit 1
  fi
done

# The other half of the two-candidate lookup: a checkout must still prefer its own build tree and
# models/ directory. This is the case that regresses if someone "simplifies" the fallback away.
checkout="$tmp/checkout"
mkdir -p -- "$checkout/scripts" "$checkout/build-linux/apps" "$checkout/models"
cp -- "$tmp/ninfer-serve" "$checkout/build-linux/apps/ninfer-serve"
: > "$checkout/models/qwen3_8_27b.ninfer"
cp -- "$root/run-qwen38-c1-maxctx.sh" "$checkout/scripts/run-qwen38-c1-maxctx.sh"
args="$tmp/run-qwen38-c1-maxctx.checkout.args"
( cd -- "$checkout" && clear_env NINFER_TEST_ARGS="$args" ./scripts/run-qwen38-c1-maxctx.sh >/dev/null )
if ! grep -Fx -- "$checkout/models/qwen3_8_27b.ninfer" "$args" >/dev/null; then
  printf 'run-qwen38-c1-maxctx.sh did not resolve the checkout artifact: %s\n' "$(head -1 -- "$args")" >&2
  exit 1
fi


mkdir -- "$tmp/bin" "$tmp/models"
# NINFER_TEST_FAKE_SIZE makes the fixture produce a file of exactly the pinned downloaders'
# expected_size (via truncate, so this stays instant regardless of how large the real artifact
# is) instead of an empty one: download-qwen36-35b-a3b.sh now verifies size (and, unless
# NINFER_SKIP_SHA256=1, sha256) before promoting the file, so an empty fixture output fails
# verification and never reaches the file-existence assertions below.
cat > "$tmp/bin/curl" <<'CURL'
#!/usr/bin/env bash
while (( $# )); do
  if [[ "$1" == '--output' ]]; then
    output="$2"
    shift 2
  else
    shift
  fi
done
: > "$output"
if [[ -n "${NINFER_TEST_FAKE_SIZE:-}" ]]; then
  truncate -s "$NINFER_TEST_FAKE_SIZE" "$output"
fi
CURL
chmod +x "$tmp/bin/curl"
# Stubs sha256sum for the checksum-rejection fixture below. verify() hashes whatever it is given,
# and a real sha256sum reads every logical byte even of a sparse file -- tens of GB per downloader,
# which is instant to allocate but not free to read, and turned this fixture into a multi-minute
# hash of empty space. The stub returns a fixed value that cannot match any pinned expected_sha256
# without reading the file at all, which is enough to exercise verify()'s mismatch-rejection branch
# -- the property under test is the script's response to a failed comparison, not whether
# sha256sum itself hashes correctly.
cat > "$tmp/bin/sha256sum" <<'HASH'
#!/usr/bin/env bash
printf -v hash '%064d' 0
printf '%s  %s\n' "$hash" "${*: -1}"
HASH
chmod +x "$tmp/bin/sha256sum"
# Every downloader now pins a revision and verifies size and sha256 before promoting, so they are
# all driven the same way: hand the fixture the script's own expected_size and skip the hash, which
# proves the promotion path without needing a real 17 GB payload.
expected_size_of() { sed -n 's/^expected_size=\([0-9]\+\)$/\1/p' "$root/$1.sh"; }
for downloader in download-qwen38-27b download-qwen36-27b download-qwen36-35b-a3b; do
  case "$downloader" in
    download-qwen38-27b) model='qwen3_8_27b.ninfer' ;;
    download-qwen36-27b) model='qwen3_6_27b.ninfer' ;;
    download-qwen36-35b-a3b) model='qwen3_6_35b_a3b.ninfer' ;;
  esac
  size="$(expected_size_of "$downloader")"
  if [[ -z "$size" ]]; then
    printf '%s has no expected_size to verify against\n' "$downloader" >&2
    exit 1
  fi
  PATH="$tmp/bin:$PATH" NINFER_MODEL_DIR="$tmp/models" NINFER_SKIP_SHA256=1 \
    NINFER_TEST_FAKE_SIZE="$size" "$root/$downloader.sh" >/dev/null
  if [[ ! -f "$tmp/models/$model" ]]; then
    printf '%s did not promote a payload matching its pin to %s\n' "$downloader" "$model" >&2
    exit 1
  fi
done

# The other half of the contract. Above proves a payload matching the pin is promoted; this proves
# one that does not is refused, which is the property the staging and checksum work exists for and
# the one a regression would silently remove. Without NINFER_TEST_FAKE_SIZE the fixture writes an
# empty file, so every pinned downloader should reject it, leave the real artifact path alone, and
# keep the revision-scoped .part it was told to delete.
rm -f -- "$tmp/models"/*.ninfer
for downloader in download-qwen38-27b download-qwen36-35b-a3b download-qwen36-27b; do
  case "$downloader" in
    download-qwen38-27b) model='qwen3_8_27b.ninfer' ;;
    download-qwen36-35b-a3b) model='qwen3_6_35b_a3b.ninfer' ;;
    download-qwen36-27b) model='qwen3_6_27b.ninfer' ;;
  esac

  if PATH="$tmp/bin:$PATH" NINFER_MODEL_DIR="$tmp/models" \
       "$root/$downloader.sh" >/dev/null 2>&1; then
    printf '%s accepted a payload that does not match its pin\n' "$downloader" >&2
    exit 1
  fi
  if [[ -e "$tmp/models/$model" ]]; then
    printf '%s promoted an unverified payload to %s\n' "$downloader" "$model" >&2
    exit 1
  fi
  if ! compgen -G "$tmp/models/$model."*".part" >/dev/null; then
    printf '%s did not stage its download under a revision-scoped name\n' "$downloader" >&2
    exit 1
  fi
  rm -f -- "$tmp/models/$model."*".part"
done

# The size check alone is not the checksum contract: a payload of the right size but wrong content
# must be rejected too, and every case above either skips the hash (NINFER_SKIP_SHA256=1) or never
# reaches it (wrong size fails first). Drive the fixture with the correct size and the hash check
# left on -- the stubbed sha256sum above always returns a value that cannot match a real pinned
# hash, so this exercises verify()'s mismatch-rejection branch without hashing tens of GB of
# logical (sparse) data to get there.
for downloader in download-qwen38-27b download-qwen36-27b download-qwen36-35b-a3b; do
  case "$downloader" in
    download-qwen38-27b) model='qwen3_8_27b.ninfer' ;;
    download-qwen36-27b) model='qwen3_6_27b.ninfer' ;;
    download-qwen36-35b-a3b) model='qwen3_6_35b_a3b.ninfer' ;;
  esac
  size="$(expected_size_of "$downloader")"

  if PATH="$tmp/bin:$PATH" NINFER_MODEL_DIR="$tmp/models" NINFER_TEST_FAKE_SIZE="$size" \
       "$root/$downloader.sh" >/dev/null 2>&1; then
    printf '%s accepted a size-correct payload with the wrong checksum\n' "$downloader" >&2
    exit 1
  fi
  if [[ -e "$tmp/models/$model" ]]; then
    printf '%s promoted a payload that failed checksum verification to %s\n' "$downloader" "$model" >&2
    exit 1
  fi
  rm -f -- "$tmp/models/$model."*".part"
done

printf '%s\n' 'Linux script checks passed.'
