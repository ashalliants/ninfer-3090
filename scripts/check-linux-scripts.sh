#!/usr/bin/env bash
set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
tmp="$(mktemp -d)"
trap 'rm -rf -- "$tmp"' EXIT

for script in "$root"/*.sh; do
  bash -n "$script"
done
for windows_script in "$root"/*.bat "$root"/*.ps1; do
  counterpart="${windows_script%.*}.sh"
  if [[ ! -x "$counterpart" ]]; then
    printf 'Missing Bash counterpart: %s\n' "$counterpart" >&2
    exit 1
  fi
done

cat > "$tmp/ninfer-serve" <<'SERVER'
#!/usr/bin/env bash
printf '%s\n' "$@" > "$NINFER_TEST_ARGS"
SERVER
chmod +x "$tmp/ninfer-serve"
touch "$tmp/qwen3_8_27b.ninfer" "$tmp/qwen3_6_35b_a3b.ninfer"

launchers=(
  'run-qwen38-c1.sh:qwen3_8_27b.ninfer:--max-context:65536'
  'run-qwen38-c8.sh:qwen3_8_27b.ninfer:--max-concurrency:8'
  'run-qwen38-vision.sh:qwen3_8_27b.ninfer:--vision:--spec'
  'run-qwen36-35b-vision.sh:qwen3_6_35b_a3b.ninfer:--vision:--no-thinking'
)
for entry in "${launchers[@]}"; do
  IFS=: read -r script model expected value <<< "$entry"
  args="$tmp/${script%.sh}.args"
  NINFER_SERVER="$tmp/ninfer-serve" NINFER_TEST_ARGS="$args" \
    "$root/$script" "$tmp/$model" >/dev/null
  grep -Fx -- "$expected" "$args" >/dev/null
  grep -Fx -- "$value" "$args" >/dev/null
done

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
PATH="$tmp/bin:$PATH" NINFER_MODEL_DIR="$tmp/models" "$root/download-qwen38-27b.sh" >/dev/null
qwen36_35b_expected_size="$(sed -n 's/^expected_size=\([0-9]\+\)$/\1/p' "$root/download-qwen36-35b-a3b.sh")"
PATH="$tmp/bin:$PATH" NINFER_MODEL_DIR="$tmp/models" NINFER_SKIP_SHA256=1 \
  NINFER_TEST_FAKE_SIZE="$qwen36_35b_expected_size" "$root/download-qwen36-35b-a3b.sh" >/dev/null
[[ -f "$tmp/models/qwen3_8_27b.ninfer" ]]
[[ -f "$tmp/models/qwen3_6_35b_a3b.ninfer" ]]

printf '%s\n' 'Linux script checks passed.'
