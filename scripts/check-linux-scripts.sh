#!/usr/bin/env bash
set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
tmp="$(mktemp -d)"
trap 'rm -rf -- "$tmp"' EXIT

for script in "$root"/*.sh; do
  bash -n "$script"
done

# CRLF in a shell script is not cosmetic. Inside a `case` block the stray CR becomes part of the
# `in` token and Linux bash refuses the file outright -- which is exactly what happened to the two
# launchers the README recommended as the Linux entry point. `bash -n` above catches that particular shape, but a CRLF
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
# Every Windows script needs a Bash sibling. There used to be an exemption list here; the
# counterpart was written instead, so the list is gone rather than empty. An exemption list is where the next Windows-only script quietly goes.
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
# that will not run. A release's Linux packager once sat at 644 for exactly that reason. `git ls-files -s` reports what is committed, which is the thing
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

# The launcher passes each profile's fixed flags as a `--config` file (scripts/profiles/) and the
# rest on the command line, so what ninfer-serve acts on is the two together. expand.py works that
# out by the rules of src/serve/serve_config_file.h -- the file's words first, then the command
# line's; the last value of an option wins, except for the options that collect -- with Python's own
# JSON parser rather than the launcher's reader. It prints the model, then one option per line
# ("--flag value" or "--flag"), sorted.
cat > "$tmp/expand.py" <<'EXPAND'
import json, sys
model, words = sys.argv[1], sys.argv[2:]
if '--config' in words:
    at = words.index('--config')
    with open(words[at + 1], encoding='utf-8') as file:
        profile = json.load(file)
    del words[at:at + 2]
    expanded = []
    for key, value in profile.items():
        for item in value if isinstance(value, list) else [value]:
            if item is True:
                expanded.append('--' + key)
            elif item is not False:
                expanded += ['--' + key, str(item)]
    words = expanded + words
options = {}
at = 0
while at < len(words):
    flag = words[at]
    if not flag.startswith('--'):
        sys.exit('expand.py: not an option: ' + flag)
    unit = flag
    if at + 1 < len(words) and not words[at + 1].startswith('--'):
        unit += ' ' + words[at + 1]
        at += 1
    at += 1
    if flag in ('--graft', '--allowed-host', '--context-store-peer'):
        options.setdefault(flag, []).append(unit)
    else:
        options[flag] = [unit]
print(model)
for flag in sorted(options):
    print('\n'.join(options[flag]))
EXPAND
expand() { python3 -I "$tmp/expand.py" "$@"; }

# Every profile file keeps the layout both launchers read their banner values from: one
# "key": value pair per line, exactly as json.dumps(..., indent=2) writes it, with scalar values
# that are not empty and hold no space, comma, colon or quote. Integers stay integers: the server
# reads 1024.0 as 1024, but the banner would show "1024.0".
for profile in "$root"/profiles/*.json; do
  python3 -I - "$profile" <<'LAYOUT' || exit 1
import json, sys
path = sys.argv[1]
text = open(path, encoding='utf-8').read()
profile = json.loads(text)
problems = []
if text != json.dumps(profile, indent=2) + '\n':
    problems.append('not laid out as json.dumps(..., indent=2) writes it, one pair per line')
for key, value in profile.items():
    if isinstance(value, (list, dict)) or value is None or isinstance(value, float):
        problems.append(f'"{key}" is not a string, an integer or true')
    elif value is False:
        problems.append(f'"{key}" is false: leave the key out instead')
    elif isinstance(value, str) and any(c in value for c in ' ,:"'):
        problems.append(f'"{key}" holds a space, comma, colon or quote')
    elif value == '':
        # The banner's line readers would show an empty value as a blank.
        problems.append(f'"{key}" is empty: leave the key out instead')
for problem in problems:
    print(f'{path}: {problem}', file=sys.stderr)
sys.exit(1 if problems else 0)
LAYOUT
done

# scripts/run.sh serves every shipped profile, chosen by `run.sh <model> [profile]`. The README and
# the launcher's own header tell users which flag set to reach for, so pin each set as a whole: a
# flag dropped from one -- or a load-time transcode the 27B artifact no longer needs (it stores the
# embedding as Q4 and the head as Q6) -- fails here instead of at a user's first start.
record() { # record <label> [NAME=value ...] -- <run.sh arguments>; prints the file of options in force
  local label="$1"; shift
  local recorded="$tmp/run.$label.args" assignments=() raw
  while [[ "$1" != '--' ]]; do assignments+=("$1"); shift; done
  shift
  clear_env NINFER_SERVER="$tmp/ninfer-serve" NINFER_TEST_ARGS="$recorded.raw" NINFER_MODEL_DIR="$tmp" \
    ${assignments[@]+"${assignments[@]}"} "$root/run.sh" "$@" >/dev/null
  mapfile -t raw < "$recorded.raw"
  expand "${raw[@]}" > "$recorded"
  printf '%s' "$recorded"
}
units() { # units <words...>: one option per line, each flag with its value
  local word unit=''
  for word in "$@"; do
    if [[ "$word" == --* && -n "$unit" ]]; then printf '%s\n' "$unit"; unit="$word"
    elif [[ -n "$unit" ]]; then unit+=" $word"
    else unit="$word"; fi
  done
  [[ -z "$unit" ]] || printf '%s\n' "$unit"
}
expect_flags() { # expect_flags <label> <recorded> <flag sequence>...: every option in each is in force
  local label="$1" recorded="$2" sequence unit words; shift 2
  for sequence in "$@"; do
    read -ra words <<< "$sequence"
    while IFS= read -r unit; do
      grep -Fxq -- "$unit" "$recorded" && continue
      printf 'run.sh (%s) is missing "%s"\noptions in force:\n%s\n' "$label" "$unit" "$(cat -- "$recorded")" >&2
      exit 1
    done < <(units "${words[@]}")
  done
}
expect_options() { # expect_options <label> <recorded> <words...>: exactly these options are in force
  local label="$1" recorded="$2" want got; shift 2
  want="$(units "$@" | LC_ALL=C sort)"
  got="$(tail -n +2 -- "$recorded" | LC_ALL=C sort)"
  [[ "$want" == "$got" ]] && return 0
  printf 'run.sh (%s): the options in force differ from the profile (< expected, > got):\n%s\n' "$label" \
    "$(diff <(printf '%s\n' "$want") <(printf '%s\n' "$got"))" >&2
  exit 1
}
expect_config() { # expect_config <label> <recorded> <profile>: run.sh passed --config profiles/<profile>.json
  local label="$1" recorded="$2" want="$root/profiles/$3.json"
  if [[ "$(grep -A1 -Fx -- '--config' "$recorded.raw" | tail -n 1)" != "$want" ]]; then
    printf 'run.sh (%s) did not pass --config %s\nargs: %s\n' "$label" "$want" "$(tr '\n' ' ' < "$recorded.raw")" >&2
    exit 1
  fi
}
refuse_flag() { # refuse_flag <label> <recorded> <flag>
  if awk -v flag="$3" '$1 == flag { found = 1 } END { exit !found }' "$2"; then
    printf 'run.sh (%s) must not pass %s\n' "$1" "$3" >&2
    exit 1
  fi
}
expect_exit() { # expect_exit <status> <label> <command...>: the command exits with exactly <status>
  local want="$1" label="$2" got=0; shift 2
  "$@" >/dev/null 2>&1 || got=$?
  [[ "$got" == "$want" ]] && return 0
  printf 'run.sh (%s) exited %s, expected %s\n' "$label" "$got" "$want" >&2
  exit 1
}

# Every profile and variant, whole: the file it is served from and every option in force, as
# measured (docs/maintainer/launcher-profiles.md). Moving a flag between a profile file and the
# launcher must leave these exactly as they are.
common_tuned=(--host 127.0.0.1 --port 8080 --kv-dtype rk4v4 --gdn-state-fp16 --prefill-cublas
  --max-pending-requests 16 --pending-timeout-ms 600000 --vision --vision-residency overlay
  --host-context-mib 8192 --min-p 0.03 --presence-penalty 0.5)
recorded="$(record whole_dflash2 -- qwen38-27b)"
expect_config '27B default' "$recorded" qwen38-27b-tuned-dflash2
expect_options '27B default' "$recorded" "${common_tuned[@]}" --fit-context 32768 \
  --max-concurrency 1 --max-context 262144 --kv-capacity 262144 \
  --spec dflash2 --draft-tokens 7 --lm-head-draft --ngram-draft-tokens 15 --prefill-chunk 4096
recorded="$(record whole_mtp NINFER_SPEC=mtp -- qwen38-27b)"
expect_config '27B NINFER_SPEC=mtp' "$recorded" qwen38-27b-tuned-mtp
expect_options '27B NINFER_SPEC=mtp' "$recorded" "${common_tuned[@]}" --fit-context 32768 \
  --max-concurrency 2 --max-context 262144 --kv-capacity 262144 \
  --spec mtp --draft-tokens 3 --lm-head-draft --prefill-chunk 2048
recorded="$(record whole_none NINFER_SPEC=none -- qwen38-27b)"
expect_config '27B NINFER_SPEC=none' "$recorded" qwen38-27b-tuned-none
expect_options '27B NINFER_SPEC=none' "$recorded" "${common_tuned[@]}" --fit-context 32768 \
  --max-concurrency 2 --max-context 262144 --kv-capacity 262144 --prefill-chunk 2048
recorded="$(record whole_35b -- qwen36-35b-a3b)"
expect_config '35B default' "$recorded" qwen36-35b-a3b-tuned-mtp
expect_options '35B default' "$recorded" "${common_tuned[@]}" --fit-context 32768 \
  --max-concurrency 3 --max-context 262144 --kv-capacity 262144 \
  --spec mtp --draft-tokens 3 --lm-head-draft --mtp-experts-q4 --prefill-chunk 4096
recorded="$(record whole_35b_none NINFER_SPEC=none -- qwen36-35b-a3b)"
expect_config '35B NINFER_SPEC=none' "$recorded" qwen36-35b-a3b-tuned-none
expect_options '35B NINFER_SPEC=none' "$recorded" "${common_tuned[@]}" --fit-context 32768 \
  --max-concurrency 3 --max-context 262144 --kv-capacity 262144 --prefill-chunk 4096
recorded="$(record whole_int8 -- qwen38-27b int8)"
expect_config '27B int8' "$recorded" qwen38-27b-int8
expect_options '27B int8' "$recorded" --host 127.0.0.1 --port 8080 \
  --max-context 65536 --kv-capacity 65536 --max-concurrency 1 --max-pending-requests 16 \
  --pending-timeout-ms 600000 --prefill-chunk 1024 --kv-dtype int8 --spec mtp --draft-tokens 3 --lm-head-draft
recorded="$(record whole_c8 -- qwen38-27b c8)"
expect_config '27B c8' "$recorded" qwen38-27b-c8
expect_options '27B c8' "$recorded" --host 127.0.0.1 --port 8080 \
  --max-context 8192 --kv-capacity 16384 --max-concurrency 8 --max-pending-requests 32 \
  --pending-timeout-ms 600000 --prefill-chunk 512 --kv-dtype int8 --spec mtp --draft-tokens 3 --lm-head-draft \
  --device-state-slots 8 --host-context-mib 8192
# The environment's overrides go after --config and win over the file's values; the banner reports
# the values in force, the file's where nothing overrides them.
recorded="$(record whole_overrides NINFER_SPEC=mtp NINFER_DRAFT_TOKENS=11 NINFER_CONTEXT=131072 \
  NINFER_KV_DTYPE=rk8v4 NINFER_PREFILL_CHUNK=1024 -- qwen38-27b)"
expect_options '27B overrides over the file' "$recorded" --host 127.0.0.1 --port 8080 --gdn-state-fp16 \
  --prefill-cublas --max-pending-requests 16 --pending-timeout-ms 600000 --vision --vision-residency overlay \
  --host-context-mib 8192 --min-p 0.03 --presence-penalty 0.5 \
  --max-concurrency 2 --max-context 131072 --kv-capacity 131072 \
  --kv-dtype rk8v4 --spec mtp --draft-tokens 11 --lm-head-draft --prefill-chunk 1024
refuse_flag '27B NINFER_SPEC=none NINFER_DRAFT_TOKENS=11' \
  "$(record none_draft NINFER_SPEC=none NINFER_DRAFT_TOKENS=11 -- qwen38-27b)" '--draft-tokens'
banner_of() { # banner_of [NAME=value ...] -- <run.sh arguments>
  local assignments=()
  while [[ "$1" != '--' ]]; do assignments+=("$1"); shift; done
  shift
  clear_env NINFER_SERVER="$tmp/ninfer-serve" NINFER_TEST_ARGS="$tmp/unused.args" NINFER_MODEL_DIR="$tmp" \
    ${assignments[@]+"${assignments[@]}"} "$root/run.sh" "$@"
}
for entry in \
  '#qwen38-27b#rk4v4  |  DFlash2 K=7 + draft head + n-gram copies#chunk 4096' \
  'NINFER_SPEC=mtp#qwen38-27b#rk4v4  |  MTP3 + draft head, two lanes#chunk 2048' \
  'NINFER_DRAFT_TOKENS=11#qwen38-27b#DFlash2 K=11 + draft head#chunk 4096' \
  'NINFER_PREFILL_CHUNK=1024#qwen38-27b#rk4v4  |  DFlash2 K=7#chunk 1024' \
  'NINFER_KV_DTYPE=rk8v4#qwen36-35b-a3b#rk8v4  |  MTP3 + draft head  |  vision#' \
  '#qwen36-35b-a3b#rk4v4  |  MTP3 + draft head  |  vision#'; do
  IFS='#' read -r assignment model want_label want_prefill <<< "$entry"
  banner="$(banner_of ${assignment:+"$assignment"} -- "$model")"
  [[ "$banner" == *"$want_label"* && ( -z "$want_prefill" || "$banner" == *"$want_prefill"* ) ]] || {
    printf 'run.sh banner (%s %s) does not show "%s" and "%s":\n%s\n' "$assignment" "$model" \
      "$want_label" "$want_prefill" "$banner" >&2
    exit 1
  }
done

# Qwen3.8-27B `tuned`: the two measured flag sets and the speculation-free variant.
recorded="$(record dflash2 -- qwen38-27b)"
expect_flags '27B default' "$recorded" \
  '--spec dflash2 --draft-tokens 7 --lm-head-draft --ngram-draft-tokens 15' \
  '--prefill-cublas --prefill-chunk 4096' \
  '--kv-dtype rk4v4' '--gdn-state-fp16' \
  '--vision --vision-residency overlay' '--max-context 262144' '--max-concurrency 1'
refuse_flag '27B default' "$recorded" '--lm-head-q6'
refuse_flag '27B default' "$recorded" '--embedding-q4'

# N-gram copies: on with DFlash2 at one and two lanes, off at three unless asked for, never without
# DFlash2.
refuse_flag '27B NINFER_NGRAM=off' "$(record ngram_off NINFER_NGRAM=off -- qwen38-27b)" '--ngram-draft-tokens'
expect_flags '27B DFlash2 two lanes' \
  "$(record ngram_c2 NINFER_CONCURRENCY=2 -- qwen38-27b)" '--ngram-draft-tokens 15'
refuse_flag '27B DFlash2 two lanes, NINFER_NGRAM=off' \
  "$(record ngram_c2_off NINFER_CONCURRENCY=2 NINFER_NGRAM=off -- qwen38-27b)" '--ngram-draft-tokens'
refuse_flag '27B DFlash2 three lanes' "$(record ngram_c3 NINFER_CONCURRENCY=3 -- qwen38-27b)" '--ngram-draft-tokens'
expect_flags '27B DFlash2 three lanes, NINFER_NGRAM=on' \
  "$(record ngram_c3_on NINFER_CONCURRENCY=3 NINFER_NGRAM=on -- qwen38-27b)" '--ngram-draft-tokens 15'
refuse_flag '27B NINFER_SPEC=mtp' "$(record ngram_mtp NINFER_SPEC=mtp -- qwen38-27b)" '--ngram-draft-tokens'

recorded="$(record mtp NINFER_SPEC=mtp -- qwen38-27b tuned)"
expect_flags '27B NINFER_SPEC=mtp' "$recorded" \
  '--spec mtp --draft-tokens 3 --lm-head-draft' \
  '--prefill-cublas --prefill-chunk 2048' \
  '--kv-dtype rk4v4' '--gdn-state-fp16' \
  '--vision --vision-residency overlay' '--max-context 262144' '--max-concurrency 2'
refuse_flag '27B NINFER_SPEC=mtp' "$recorded" '--lm-head-q6'
refuse_flag '27B NINFER_SPEC=mtp' "$recorded" '--embedding-q4'

recorded="$(record none NINFER_SPEC=none -- qwen38-27b)"
expect_flags '27B NINFER_SPEC=none' "$recorded" '--prefill-cublas --prefill-chunk 2048'
refuse_flag '27B NINFER_SPEC=none' "$recorded" '--embedding-q4'
refuse_flag '27B NINFER_SPEC=none' "$recorded" '--spec'
refuse_flag '27B NINFER_SPEC=none' "$recorded" '--lm-head-q6'

# NINFER_ALLOWED_HOST: each comma- or space-separated name becomes one --allowed-host, in every
# profile, and the banner shows them; unset, there is none.
expect_flags '27B NINFER_ALLOWED_HOST' \
  "$(record allowed NINFER_ALLOWED_HOST='gpubox.lan, ninfer.example,,box2' -- qwen38-27b)" \
  '--allowed-host gpubox.lan' '--allowed-host ninfer.example' '--allowed-host box2'
expect_flags '27B int8 NINFER_ALLOWED_HOST' \
  "$(record allowed_int8 NINFER_ALLOWED_HOST=gpubox.lan -- qwen38-27b int8)" '--allowed-host gpubox.lan'
refuse_flag '27B default' "$(record allowed_none -- qwen38-27b)" '--allowed-host'
banner="$(banner_of NINFER_ALLOWED_HOST=gpubox.lan -- qwen38-27b)"
[[ "$banner" == *'Allowed host names: gpubox.lan'* ]] || {
  printf 'run.sh banner does not show NINFER_ALLOWED_HOST:\n%s\n' "$banner" >&2
  exit 1
}

# An explicit override beats the profile's default; vision can be turned off.
recorded="$(record override NINFER_CONTEXT=65536 NINFER_PREFILL_CHUNK=1024 NINFER_CONCURRENCY=2 \
  NINFER_VISION=off -- qwen38-27b)"
expect_flags '27B overrides' "$recorded" '--max-context 65536' '--prefill-chunk 1024' '--max-concurrency 2'
refuse_flag '27B overrides' "$recorded" '--vision'

# The Host context budget is the pinned-RAM override a box short on memory needs, and its default
# must not move.
expect_flags '27B default host context' "$(record host8192 -- qwen38-27b)" '--host-context-mib 8192'
expect_flags '27B host context override' "$(record host1024 NINFER_HOST_CONTEXT_MIB=1024 -- qwen38-27b)" '--host-context-mib 1024'

# NINFER_CHAT_TEMPLATE forwards straight to --chat-template, and is absent when unset.
expect_flags '27B chat template override' \
  "$(record chat_template NINFER_CHAT_TEMPLATE="$tmp/custom.jinja" -- qwen38-27b)" \
  "--chat-template $tmp/custom.jinja"
refuse_flag '27B no chat template override' "$(record no_chat_template -- qwen38-27b)" '--chat-template'

# The reasoning-loop guard is opt-in: unset or off passes nothing (the server default is off), stop
# and conclude pass through on both models, the reference profiles stay minimal, and the banner
# says which mode is in force.
refuse_flag '27B reasoning loop unset' "$(record loop_unset -- qwen38-27b)" '--reasoning-loop'
refuse_flag '27B NINFER_REASONING_LOOP=off' "$(record loop_off NINFER_REASONING_LOOP=off -- qwen38-27b)" '--reasoning-loop'
expect_flags '27B NINFER_REASONING_LOOP=stop' \
  "$(record loop_stop NINFER_REASONING_LOOP=stop -- qwen38-27b)" '--reasoning-loop stop'
expect_flags '35B NINFER_REASONING_LOOP=conclude' \
  "$(record loop_conclude35 NINFER_REASONING_LOOP=conclude -- qwen36-35b-a3b)" '--reasoning-loop conclude'
refuse_flag '27B int8 NINFER_REASONING_LOOP=stop' \
  "$(record loop_int8 NINFER_REASONING_LOOP=stop -- qwen38-27b int8)" '--reasoning-loop'
banner="$(clear_env NINFER_SERVER="$tmp/ninfer-serve" NINFER_TEST_ARGS="$tmp/unused.args" \
  NINFER_MODEL_DIR="$tmp" NINFER_REASONING_LOOP=conclude "$root/run.sh" qwen38-27b)"
[[ "$banner" == *'Reasoning-loop guard: conclude'* ]] || {
  printf 'run.sh banner does not report NINFER_REASONING_LOOP: %s\n' "$banner" >&2
  exit 1
}
banner="$(clear_env NINFER_SERVER="$tmp/ninfer-serve" NINFER_TEST_ARGS="$tmp/unused.args" \
  NINFER_MODEL_DIR="$tmp" "$root/run.sh" qwen38-27b)"
[[ "$banner" != *'Reasoning-loop guard'* ]] || {
  printf 'run.sh banner reports a reasoning-loop guard nobody asked for: %s\n' "$banner" >&2
  exit 1
}

# The banner reports what is being served: an explicit vision residency shows up in it, not the
# default. (The stub server prints nothing, so stdout here is the launcher's own.)
banner="$(clear_env NINFER_SERVER="$tmp/ninfer-serve" NINFER_TEST_ARGS="$tmp/unused.args" \
  NINFER_MODEL_DIR="$tmp" NINFER_VISION_RESIDENCY=resident "$root/run.sh" qwen38-27b)"
[[ "$banner" == *'vision (resident)'* && "$banner" != *'vision (overlay)'* ]] || {
  printf 'run.sh banner ignored NINFER_VISION_RESIDENCY: %s\n' "$banner" >&2
  exit 1
}

# The reference profiles are fixed and minimal: no vision, and only c8 sizes its context cache,
# per lane (two retained conversations and two host states per lane, one device state per lane).
recorded="$(record int8 -- qwen38-27b int8)"
expect_flags '27B int8' "$recorded" '--max-context 65536' '--max-concurrency 1' '--kv-dtype int8' \
  '--spec mtp --draft-tokens 3 --lm-head-draft'
refuse_flag '27B int8' "$recorded" '--vision'
recorded="$(record c8 -- qwen38-27b c8)"
expect_flags '27B c8' "$recorded" '--max-concurrency 8' '--max-context 8192' '--kv-capacity 16384' \
  '--kv-dtype int8' '--spec mtp --draft-tokens 3 --lm-head-draft' \
  '--device-state-slots 8' '--host-context-mib 8192'
refuse_flag '27B c8' "$recorded" '--vision'

# Qwen3.6-35B-A3B `tuned`: the cuBLAS prefill route at chunk 4096 (it engages on the 35B's dense
# projections from chunk 2048), no Q4 embedding, and its own draft head flags.
recorded="$(record 35b -- qwen36-35b-a3b)"
expect_flags '35B default' "$recorded" \
  '--spec mtp --draft-tokens 3 --lm-head-draft --mtp-experts-q4' \
  '--kv-dtype rk4v4' '--gdn-state-fp16' '--prefill-cublas --prefill-chunk 4096' \
  '--vision --vision-residency overlay' '--max-context 262144' '--max-concurrency 3' '--host-context-mib 8192'
refuse_flag '35B default' "$recorded" '--embedding-q4'
recorded="$(record 35bnone NINFER_SPEC=none -- qwen36-35b-a3b)"
refuse_flag '35B NINFER_SPEC=none' "$recorded" '--spec'

# Bad input is refused rather than falling through to a profile: no model, an unknown model, a
# profile the model does not have, an unknown speculation setting -- and --help succeeds.
serve_env=(NINFER_SERVER="$tmp/ninfer-serve" NINFER_MODEL_DIR="$tmp" NINFER_TEST_ARGS="$tmp/unused.args")
expect_exit 2 'no model' clear_env "${serve_env[@]}" "$root/run.sh"
expect_exit 2 'unknown model' clear_env "${serve_env[@]}" "$root/run.sh" gpt
expect_exit 2 'profile the model lacks' clear_env "${serve_env[@]}" "$root/run.sh" qwen36-35b-a3b c8
expect_exit 2 'unknown profile' clear_env "${serve_env[@]}" "$root/run.sh" qwen38-27b fastest
expect_exit 2 'unknown NINFER_SPEC' clear_env "${serve_env[@]}" NINFER_SPEC=bogus "$root/run.sh" qwen38-27b
expect_exit 2 'unknown NINFER_NGRAM' clear_env "${serve_env[@]}" NINFER_NGRAM=yes "$root/run.sh" qwen38-27b
expect_exit 2 'NINFER_NGRAM=on without DFlash2' clear_env "${serve_env[@]}" NINFER_SPEC=mtp NINFER_NGRAM=on \
  "$root/run.sh" qwen38-27b
expect_exit 2 'NINFER_NGRAM=on on the 35B' clear_env "${serve_env[@]}" NINFER_NGRAM=on "$root/run.sh" qwen36-35b-a3b
expect_exit 2 'unknown NINFER_NGRAM on the 35B' clear_env "${serve_env[@]}" NINFER_NGRAM=yes "$root/run.sh" qwen36-35b-a3b
expect_exit 2 'unknown NINFER_REASONING_LOOP' clear_env "${serve_env[@]}" NINFER_REASONING_LOOP=on \
  "$root/run.sh" qwen38-27b
# The Flash-Next n-gram volume override belongs to run.bat; these models have no volume to point at.
touch "$tmp/volume.ngram"
expect_exit 2 'NINFER_NGRAM_VOLUME on the 27B' clear_env "${serve_env[@]}" NINFER_NGRAM_VOLUME="$tmp/volume.ngram" \
  "$root/run.sh" qwen38-27b
expect_exit 2 'NINFER_NGRAM_VOLUME on the 35B' clear_env "${serve_env[@]}" NINFER_NGRAM_VOLUME="$tmp/volume.ngram" \
  "$root/run.sh" qwen36-35b-a3b
refuse_flag '27B default' "$(record no_volume -- qwen38-27b)" '--ngram-volume'
expect_exit 0 '--help' clear_env "${serve_env[@]}" "$root/run.sh" --help

# A desktop holding VRAM can leave too little for the default context, and the tuned profile is
# meant to start anyway: its context is a ceiling the server fits to the free memory
# (--fit-context, with the floor below which it refuses). A context the caller chose is theirs, and
# NINFER_FALLBACK=off turns the fit off. The reference profiles are fixed shapes.
expect_flags '27B default fits its context' "$(record fit -- qwen38-27b)" \
  '--max-context 262144 --kv-capacity 262144' '--fit-context 32768'
expect_flags '35B default fits its context' "$(record fit35 -- qwen36-35b-a3b)" '--fit-context 32768'
refuse_flag '27B NINFER_CONTEXT' "$(record fit_context NINFER_CONTEXT=131072 -- qwen38-27b)" '--fit-context'
refuse_flag '27B NINFER_KV_CAPACITY' "$(record fit_kv NINFER_KV_CAPACITY=300000 -- qwen38-27b)" '--fit-context'
refuse_flag '27B NINFER_FALLBACK=off' "$(record fit_off NINFER_FALLBACK=off -- qwen38-27b)" '--fit-context'
refuse_flag '27B int8' "$(record fit_int8 -- qwen38-27b int8)" '--fit-context'
refuse_flag '27B c8' "$(record fit_c8 -- qwen38-27b c8)" '--fit-context'
banner="$(clear_env NINFER_SERVER="$tmp/ninfer-serve" NINFER_TEST_ARGS="$tmp/unused.args" \
  NINFER_MODEL_DIR="$tmp" "$root/run.sh" qwen38-27b)"
[[ "$banner" == *'context up to 262144'* ]] || {
  printf 'run.sh banner does not say the context is a ceiling: %s\n' "$banner" >&2
  exit 1
}

# Host retry. A machine short of RAM can fail to pin the Host context budget; the tuned profile then
# retries with half of it, down to none. The stub refuses (with the engine's own message) until the
# requested budget fits, and logs every attempt it is given.
# It reads the options in force (expand.py), so a budget from the profile file counts as well.
cat > "$tmp/ninfer-serve-tight" <<'TIGHT'
#!/usr/bin/env bash
read -ra options <<< "$(python3 -I "$NINFER_TEST_EXPAND" "$@" | tr '\n' ' ')"
host=''; previous=''
for argument in ${options[@]+"${options[@]}"}; do [[ "$previous" == '--host-context-mib' ]] && host="$argument"; previous="$argument"; done
printf '%s\n' "${options[*]}" >> "$NINFER_TEST_LOG"
if (( ${host:-0} > ${NINFER_TEST_FITS:-0} )); then
  echo "${NINFER_TEST_FAILURE:-FATAL server failed during startup | cudaMallocHost failed to pin 8589934592 bytes}"
  exit 1
fi
TIGHT
chmod +x "$tmp/ninfer-serve-tight"
attempts() { # attempts <fits> [NAME=value ...] -> log of every attempt; sets $ladder_status, $ladder_out
  local fits="$1"; shift
  ladder_log="$tmp/ladder.$RANDOM.log"; : > "$ladder_log"
  ladder_status=0
  ladder_out="$(clear_env NINFER_SERVER="$tmp/ninfer-serve-tight" NINFER_MODEL_DIR="$tmp" \
    NINFER_TEST_EXPAND="$tmp/expand.py" NINFER_TEST_LOG="$ladder_log" NINFER_TEST_FITS="$fits" ${@+"$@"} "$root/run.sh" qwen38-27b 2>&1)" ||
    ladder_status=$?
}
field() { # field <flag>: that flag's value on each logged attempt, space separated
  awk -v flag="$1" '{ for (i = 1; i < NF; i++) if ($i == flag) printf "%s ", $(i + 1) } END { print "" }' "$ladder_log"
}
expect_eq() { # expect_eq <label> <got> <want>
  [[ "$2" == "$3" ]] && return 0
  printf 'run.sh retry (%s): got "%s", expected "%s"\n' "$1" "$2" "$3" >&2
  exit 1
}

attempts 2048
expect_eq 'host budget halves until it pins' "$(field --host-context-mib)" '8192 4096 2048 '
expect_eq 'the context is left to the fit' "$(field --max-context)" '262144 262144 262144 '
expect_eq 'every retry still fits its context' "$(field --fit-context)" '32768 32768 32768 '
expect_eq 'it started, so it exits 0' "$ladder_status" '0'

attempts -1
expect_eq 'never pins: down to none, then stops' "$(field --host-context-mib)" '8192 4096 2048 0 '
expect_eq 'never pins: reports the failure' "$ladder_status" '1'

# Values the caller chose are theirs, and a switch turns the retry off: one attempt, loud failure.
attempts 0 NINFER_HOST_CONTEXT_MIB=8192
expect_eq 'explicit host context is not second-guessed' "$(field --host-context-mib)" '8192 '
attempts 0 NINFER_FALLBACK=off
expect_eq 'NINFER_FALLBACK=off' "$(field --host-context-mib)" '8192 '
attempts -1 NINFER_AUTO_HOST_CACHE=on
expect_eq 'auto host cache is not retried' "$(wc -l < "$ladder_log" | tr -d ' ')" '1'

# Any other startup failure is not something a smaller budget fixes.
attempts 0 NINFER_TEST_FAILURE='FATAL server failed during startup | artifact is corrupt'
expect_eq 'other failures do not retry' "$(field --host-context-mib)" '8192 '

# When not even the fit's floor fits the GPU, the server refuses and the launcher says, in plain
# words, what to do -- once, without retrying anything.
attempts 0 NINFER_TEST_FAILURE='FATAL server failed during startup | not enough free GPU memory to start: ...'
expect_eq 'a GPU memory refusal does not retry' "$(field --host-context-mib)" '8192 '
expect_eq 'a GPU memory refusal fails' "$ladder_status" '1'
[[ "$ladder_out" == *'Close other programs that use the GPU'* ]] || {
  printf 'run.sh gave no advice on a GPU memory refusal:\n%s\n' "$ladder_out" >&2
  exit 1
}

# NINFER_CHAT_TEMPLATE plays no part in the retry, but it must still ride along on every attempt --
# a value set once should not silently drop off a later one.
attempts 2048 NINFER_CHAT_TEMPLATE="$tmp/custom.jinja"
expect_eq 'chat template rides every retry' "$(field --chat-template)" \
  "$tmp/custom.jinja $tmp/custom.jinja $tmp/custom.jinja "

# The reference profiles are fixed shapes and never retry.
ladder_log="$tmp/ladder.fixed.log"; : > "$ladder_log"
clear_env NINFER_SERVER="$tmp/ninfer-serve-tight" NINFER_MODEL_DIR="$tmp" NINFER_TEST_LOG="$ladder_log" \
  NINFER_TEST_EXPAND="$tmp/expand.py" NINFER_TEST_FITS=-1 "$root/run.sh" qwen38-27b c8 >/dev/null 2>&1 || true
expect_eq 'c8 profile has no retry' "$(field --host-context-mib)" '8192 '

# run.sh ships *inside* the release archive as well as living here, and README calls it the Linux
# entry point. It once resolved both the server and the artifact from `dirname(script)/..` -- the
# archive's parent once packaged -- so the documented quick start exited before serving, with a
# "Missing ninfer-serve" naming a path outside the archive. Nothing tested that, because every case
# above hands the launcher an explicit NINFER_SERVER and model directory.
#
# So drive it the way a user does: unpack and run, flat layout, no environment overrides beyond the
# stub hook. The launcher must find the binary beside itself and the artifact under its own
# models/, not one directory up.
archive="$tmp/archive"
mkdir -- "$archive" "$archive/models"
cp -- "$tmp/ninfer-serve" "$archive/ninfer-serve"
cp -- "$root/run.sh" "$archive/run.sh"
cp -R -- "$root/profiles" "$archive/profiles"
for entry in 'qwen38-27b:qwen3_8_27b.ninfer' 'qwen36-35b-a3b:qwen3_6_35b_a3b.ninfer'; do
  IFS=: read -r key model <<< "$entry"
  : > "$archive/models/$model"
  args="$tmp/run.$key.archive.args"
  ( cd -- "$archive" && clear_env NINFER_TEST_ARGS="$args" ./run.sh "$key" >/dev/null )
  if ! grep -Fx -- "$archive/models/$model" "$args" >/dev/null; then
    printf 'run.sh %s did not resolve its artifact inside the archive: %s\n' "$key" "$(head -1 -- "$args")" >&2
    exit 1
  fi
done

# The probe must select the candidate that holds the artifact, not the first models/ directory that
# happens to exist. An archive unpacked below a directory with its own (empty) models/ would
# otherwise stop at the parent and fail while its own artifact sat beside the launcher.
decoy="$tmp/decoy"
mkdir -p -- "$decoy/models" "$decoy/inner/models"
cp -- "$tmp/ninfer-serve" "$decoy/inner/ninfer-serve"
: > "$decoy/inner/models/qwen3_8_27b.ninfer"
cp -- "$root/run.sh" "$decoy/inner/run.sh"
cp -R -- "$root/profiles" "$decoy/inner/profiles"
args="$tmp/run.decoy.args"
( cd -- "$decoy/inner" && clear_env NINFER_TEST_ARGS="$args" ./run.sh qwen38-27b >/dev/null )
if ! grep -Fx -- "$decoy/inner/models/qwen3_8_27b.ninfer" "$args" >/dev/null; then
  printf 'run.sh stopped at an empty parent models/: %s\n' "$(head -1 -- "$args")" >&2
  exit 1
fi

# An explicit NINFER_MODEL_DIR must be honoured verbatim, never probed past. The two-candidate
# fallback above is for when the caller said nothing; applying it to a directory the caller *named*
# reports a missing artifact under a path they never mentioned, and hides their typo. Assert the
# error names the directory that was actually asked for.
err="$tmp/explicit-dir.err"
if ( cd -- "$archive" && clear_env NINFER_MODEL_DIR="$tmp/nope" NINFER_TEST_ARGS="$tmp/unused.args" \
       ./run.sh qwen38-27b >/dev/null 2>"$err" ); then
  printf 'run.sh accepted a nonexistent NINFER_MODEL_DIR\n' >&2
  exit 1
fi
if ! grep -Fq -- "$tmp/nope/qwen3_8_27b.ninfer" "$err"; then
  printf 'run.sh ignored an explicit NINFER_MODEL_DIR: %s\n' "$(head -1 -- "$err")" >&2
  exit 1
fi

# The other half of the two-candidate lookup: a checkout must still prefer its own build tree and
# models/ directory. This is the case that regresses if someone "simplifies" the fallback away.
checkout="$tmp/checkout"
mkdir -p -- "$checkout/scripts" "$checkout/build-linux/apps" "$checkout/models"
cp -- "$tmp/ninfer-serve" "$checkout/build-linux/apps/ninfer-serve"
: > "$checkout/models/qwen3_8_27b.ninfer"
cp -- "$root/run.sh" "$checkout/scripts/run.sh"
cp -R -- "$root/profiles" "$checkout/scripts/profiles"
args="$tmp/run.checkout.args"
( cd -- "$checkout" && clear_env NINFER_TEST_ARGS="$args" ./scripts/run.sh qwen38-27b >/dev/null )
if ! grep -Fx -- "$checkout/models/qwen3_8_27b.ninfer" "$args" >/dev/null; then
  printf 'run.sh did not resolve the checkout artifact: %s\n' "$(head -1 -- "$args")" >&2
  exit 1
fi
# Both layouts read the profile files beside the launcher: scripts/profiles/ here, profiles/ at the
# archive root (package-release.sh ships them there).
if ! grep -Fx -- "$checkout/scripts/profiles/qwen38-27b-tuned-dflash2.json" "$args" >/dev/null; then
  printf 'run.sh did not read the checkout profile: %s\n' "$(tr '\n' ' ' < "$args")" >&2
  exit 1
fi
if ! grep -Fx -- "$archive/profiles/qwen38-27b-tuned-dflash2.json" "$tmp/run.qwen38-27b.archive.args" >/dev/null; then
  printf 'run.sh did not read the archive profile: %s\n' "$(tr '\n' ' ' < "$tmp/run.qwen38-27b.archive.args")" >&2
  exit 1
fi
# A launcher without its profiles refuses to start, naming the file, rather than serving defaults.
rm -r -- "$checkout/scripts/profiles"
err="$tmp/no-profiles.err"
status=0
( cd -- "$checkout" && clear_env NINFER_TEST_ARGS="$tmp/unused.args" ./scripts/run.sh qwen38-27b int8 \
    >/dev/null 2>"$err" ) || status=$?
if [[ "$status" != 1 ]] || ! grep -Fq -- "$checkout/scripts/profiles/qwen38-27b-int8.json" "$err"; then
  printf 'run.sh without profiles/ exited %s: %s\n' "$status" "$(cat -- "$err")" >&2
  exit 1
fi


mkdir -- "$tmp/bin" "$tmp/models"
# Everything below drives download-model.sh. Run it as a user with no Hugging Face login, so the
# optional graft fetch after a model is in place has no token to use whatever this machine holds.
unset HF_TOKEN HF_TOKEN_PATH HF_HOME XDG_CACHE_HOME
mkdir -- "$tmp/no-login-home"
export HOME="$tmp/no-login-home"
# NINFER_TEST_FAKE_SIZE makes the fixture produce a file of exactly the pinned downloaders'
# expected_size (via truncate, so this stays instant regardless of how large the real artifact
# is) instead of an empty one: download-model.sh verifies size (and, unless
# NINFER_SKIP_SHA256=1, sha256) before promoting the file, so an empty fixture output fails
# verification and never reaches the file-existence assertions below.
# NINFER_TEST_FAKE_SIZES ("name=size ...") sizes each file by the name the URL ends in, for a model
# that fetches several files; a name it does not list gets NINFER_TEST_FAKE_SIZE, else stays empty.
cat > "$tmp/bin/curl" <<'CURL'
#!/usr/bin/env bash
url=''
while (( $# )); do
  if [[ "$1" == '--output' ]]; then
    output="$2"
    shift 2
  else
    [[ "$1" == https://* ]] && url="$1"
    shift
  fi
done
: > "$output"
size="${NINFER_TEST_FAKE_SIZE:-}"
for entry in ${NINFER_TEST_FAKE_SIZES:-}; do
  [[ "${url##*/}" == "${entry%%=*}" ]] && size="${entry#*=}"
done
if [[ -n "$size" ]]; then
  truncate -s "$size" "$output"
fi
CURL
chmod +x "$tmp/bin/curl"
# download-model.sh prefers aria2c over curl when aria2c is on PATH, and GitHub's hosted runner
# image ships aria2c -- so without a stub here, every case below that reaches the fetch branch
# would download the real multi-GB artifact from Hugging Face instead of exercising the fixture,
# which is what made this suite take minutes instead of seconds. -d/-o mirror aria2c's own flags
# (see download-model.sh's invocation), not curl's --output.
cat > "$tmp/bin/aria2c" <<'ARIA2C'
#!/usr/bin/env bash
dir='.' out='' url=''
while (( $# )); do
  case "$1" in
    -d) dir="$2"; shift 2 ;;
    -o) out="$2"; shift 2 ;;
    https://*) url="$1"; shift ;;
    *) shift ;;
  esac
done
output="$dir/$out"
: > "$output"
size="${NINFER_TEST_FAKE_SIZE:-}"
for entry in ${NINFER_TEST_FAKE_SIZES:-}; do
  [[ "${url##*/}" == "${entry%%=*}" ]] && size="${entry#*=}"
done
if [[ -n "$size" ]]; then
  truncate -s "$size" "$output"
fi
ARIA2C
chmod +x "$tmp/bin/aria2c"
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
# One downloader serves every model and pins each one's revision, size and sha256 in a case block,
# verifying size and sha256 before promoting. Every model is driven the same way: hand the fixture the
# model's own expected_size and skip the hash, which proves the promotion path without needing a real
# 19 GB payload.
downloader="$root/download-model.sh"
models=(qwen38-27b qwen36-27b qwen36-35b-a3b)
artifact_of() {
  case "$1" in
    qwen38-27b) printf 'qwen3_8_27b.ninfer' ;;
    qwen36-27b) printf 'qwen3_6_27b.ninfer' ;;
    qwen36-35b-a3b) printf 'qwen3_6_35b_a3b.ninfer' ;;
  esac
}
# The pin lives in download-model.sh's `  <model>)` block, so read it from there rather than repeating it.
expected_size_of() {
  sed -n "/^  $1)\$/,/^    ;;\$/ s/^ *expected_size=\\([0-9]\\+\\)\$/\\1/p" "$downloader"
}
for key in "${models[@]}"; do
  model="$(artifact_of "$key")"
  size="$(expected_size_of "$key")"
  if [[ -z "$size" ]]; then
    printf 'download-model.sh has no expected_size to verify %s against\n' "$key" >&2
    exit 1
  fi
  PATH="$tmp/bin:$PATH" NINFER_MODEL_DIR="$tmp/models" NINFER_SKIP_SHA256=1 \
    NINFER_TEST_FAKE_SIZE="$size" "$downloader" "$key" >/dev/null
  if [[ ! -f "$tmp/models/$model" ]]; then
    printf 'download-model.sh %s did not promote a payload matching its pin to %s\n' "$key" "$model" >&2
    exit 1
  fi
done

# A model name is required and must be one of the pinned ones. Neither case may touch the network or
# the models directory: exit 2 with a usage message that names every model, before any curl or
# aria2c runs.
# The models directory it is pointed at does not exist, so creating it would be caught below.
for bad in '' 'qwen38-27' 'qwen3_8_27b'; do
  if [[ -z "$bad" ]]; then bad_args=(); else bad_args=("$bad"); fi
  status=0
  usage_output="$(PATH="$tmp/bin:$PATH" NINFER_MODEL_DIR="$tmp/never-created" "$downloader" \
    ${bad_args[@]+"${bad_args[@]}"} 2>&1 >/dev/null)" || status=$?
  if [[ "$status" != 2 ]]; then
    printf 'download-model.sh %s exited %s, expected 2\n' "${bad:-<no model>}" "$status" >&2
    exit 1
  fi
  for key in "${models[@]}" qwen38-flash-next; do
    if [[ "$usage_output" != *"$key"* ]]; then
      printf 'download-model.sh usage for %s omits %s\n' "${bad:-<no model>}" "$key" >&2
      exit 1
    fi
  done
  if [[ -e "$tmp/never-created" ]]; then
    printf 'download-model.sh created the model directory before validating its argument\n' >&2
    exit 1
  fi
done

# An artifact that already verifies is left alone: no download, not even an attempted one. Both
# stubs in this directory fail every call, so a script that fetched again -- via either path --
# would exit non-zero instead of quietly reaching the real network.
mkdir -- "$tmp/failing-curl"
printf '#!/usr/bin/env bash\nexit 22\n' > "$tmp/failing-curl/curl"
chmod +x "$tmp/failing-curl/curl"
cp -- "$tmp/failing-curl/curl" "$tmp/failing-curl/aria2c"
for key in "${models[@]}"; do
  model="$(artifact_of "$key")"
  size="$(expected_size_of "$key")"
  # Sparse, like the fixture's own payloads: instant to allocate whatever the pinned size.
  truncate -s "$size" "$tmp/models/$model"
  # Capture rather than pipe into grep -q: the script prints more after this line, and a reader that
  # exits at the first match would hand it SIGPIPE, failing the pipeline under pipefail.
  present_output="$(PATH="$tmp/failing-curl:$PATH" NINFER_MODEL_DIR="$tmp/models" NINFER_SKIP_SHA256=1 \
    "$downloader" "$key" 2>&1)" || present_output=''
  if [[ "$present_output" != *'Model already present'* ]]; then
    printf 'download-model.sh %s did not accept an artifact that already verifies\n' "$key" >&2
    exit 1
  fi
done

# The opposite: a file at the final path that does not verify is replaced, not trusted. An empty
# file is the wrong size, so the script must fetch (the fixture supplies a payload of the pinned
# size) and end with the pinned-size artifact rather than the corrupt one.
for key in "${models[@]}"; do
  model="$(artifact_of "$key")"
  size="$(expected_size_of "$key")"
  : > "$tmp/models/$model"
  PATH="$tmp/bin:$PATH" NINFER_MODEL_DIR="$tmp/models" NINFER_SKIP_SHA256=1 \
    NINFER_TEST_FAKE_SIZE="$size" "$downloader" "$key" >/dev/null 2>&1
  if [[ "$(wc -c < "$tmp/models/$model" | tr -d '[:space:]')" != "$size" ]]; then
    printf 'download-model.sh %s kept a corrupt artifact instead of re-fetching it\n' "$key" >&2
    exit 1
  fi
done

# The other half of the contract. Above proves a payload matching the pin is promoted; this proves
# one that does not is refused, which is the property the staging and checksum work exists for and
# the one a regression would silently remove. Without NINFER_TEST_FAKE_SIZE the fixture writes an
# empty file, so every pinned model should be rejected, leave the real artifact path alone, and
# keep the revision-scoped .part it was told to delete.
rm -f -- "$tmp/models"/*.ninfer
for key in "${models[@]}"; do
  model="$(artifact_of "$key")"

  if PATH="$tmp/bin:$PATH" NINFER_MODEL_DIR="$tmp/models" \
       "$downloader" "$key" >/dev/null 2>&1; then
    printf 'download-model.sh %s accepted a payload that does not match its pin\n' "$key" >&2
    exit 1
  fi
  if [[ -e "$tmp/models/$model" ]]; then
    printf 'download-model.sh %s promoted an unverified payload to %s\n' "$key" "$model" >&2
    exit 1
  fi
  if ! compgen -G "$tmp/models/$model."*".part" >/dev/null; then
    printf 'download-model.sh %s did not stage its download under a revision-scoped name\n' "$key" >&2
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
for key in "${models[@]}"; do
  model="$(artifact_of "$key")"
  size="$(expected_size_of "$key")"

  if PATH="$tmp/bin:$PATH" NINFER_MODEL_DIR="$tmp/models" NINFER_TEST_FAKE_SIZE="$size" \
       "$downloader" "$key" >/dev/null 2>&1; then
    printf 'download-model.sh %s accepted a size-correct payload with the wrong checksum\n' "$key" >&2
    exit 1
  fi
  if [[ -e "$tmp/models/$model" ]]; then
    printf 'download-model.sh %s promoted a payload that failed checksum verification to %s\n' "$key" "$model" >&2
    exit 1
  fi
  rm -f -- "$tmp/models/$model."*".part"
done

# qwen38-flash-next fetches three pinned files through the same verify-then-promote routine: the model
# and its .part-0001 into the models folder, and the n-gram volume into NINFER_NGRAM_DIR (with no
# terminal to ask on and no NINFER_NGRAM_DIR, the models folder). The fixture serves each file at its
# own pinned size, read from the downloader's block like the others. Every run has stdin closed, so the
# script must never wait on its question.
fn_fail() {
  printf 'download-model.sh qwen38-flash-next: %s\n' "$1" >&2
  exit 1
}
fn_artifact='qwen4exp-iq2xs-mtp-vision.ninfer'
fn_part="$fn_artifact.part-0001"
fn_ngram="$fn_artifact.ngram"
fn_block="$(sed -n '/^  qwen38-flash-next)$/,/^    ;;$/p' "$downloader")"
fn_size="$(expected_size_of qwen38-flash-next)"
fn_part_size="$(sed -n 's/^ *part_sizes=(\([0-9]\+\))$/\1/p' <<<"$fn_block")"
fn_ngram_size="$(sed -n 's/^ *ngram_size=\([0-9]\+\)$/\1/p' <<<"$fn_block")"
if [[ -z "$fn_size" || -z "$fn_part_size" || -z "$fn_ngram_size" ]] ||
   ! grep -q "^ *artifact='$fn_artifact'\$" <<<"$fn_block" || ! grep -q "^ *ngram_name='$fn_ngram'\$" <<<"$fn_block"; then
  fn_fail 'its block no longer pins the model, one part and the volume this check expects'
fi
fn_sizes="$fn_artifact=$fn_size $fn_part=$fn_part_size $fn_ngram=$fn_ngram_size"
fn_size_ok() { [[ -f "$1" && "$(wc -c < "$1" | tr -d '[:space:]')" == "$2" ]]; }
fn_models="$tmp/fn-models"
fn_volumes="$tmp/fn-volumes"
fn_run() {  # PATH directory, then extra environment; prints the output, returns the status
  local stubs="$1"
  shift
  PATH="$stubs:$PATH" NINFER_MODEL_DIR="$fn_models" NINFER_NGRAM_DIR="$fn_volumes" NINFER_SKIP_SHA256=1 \
    "$@" "$downloader" qwen38-flash-next 2>&1 </dev/null
}

# A fresh fetch promotes all three at their pinned sizes, the volume into its own folder only, and the
# closing lines tell the server where the volume is.
fn_output="$(fn_run "$tmp/bin" env NINFER_TEST_FAKE_SIZES="$fn_sizes")" || fn_fail "a fresh fetch failed: $fn_output"
fn_size_ok "$fn_models/$fn_artifact" "$fn_size" || fn_fail 'did not promote the model at its pinned size'
fn_size_ok "$fn_models/$fn_part" "$fn_part_size" || fn_fail 'did not promote the part at its pinned size'
fn_size_ok "$fn_volumes/$fn_ngram" "$fn_ngram_size" || fn_fail 'did not promote the volume into NINFER_NGRAM_DIR'
[[ -e "$fn_models/$fn_ngram" ]] && fn_fail 'also put the volume beside the model'
[[ "$fn_output" == *"--ngram-volume \"$fn_volumes/$fn_ngram\""* ]] ||
  fn_fail "did not name the volume outside the model folder: $fn_output"
compgen -G "$fn_models/*.part" >/dev/null && fn_fail 'left a staged .part behind after promoting'

# Everything present and verifying: kept, each reported, and nothing fetched (both stubs fail).
fn_output="$(fn_run "$tmp/failing-curl" env)" || fn_fail "re-fetched files that already verify: $fn_output"
[[ "$fn_output" == *"Model already present: $fn_models/$fn_artifact"* &&
   "$fn_output" == *"Already present: $fn_models/$fn_part"* &&
   "$fn_output" == *"Already present: $fn_volumes/$fn_ngram"* ]] ||
  fn_fail "did not report all three files present: $fn_output"

# A corrupt (empty) part is fetched again; the verified model and volume are kept.
: > "$fn_models/$fn_part"
fn_output="$(fn_run "$tmp/bin" env NINFER_TEST_FAKE_SIZES="$fn_sizes")" || fn_fail "did not replace a corrupt part: $fn_output"
fn_size_ok "$fn_models/$fn_part" "$fn_part_size" || fn_fail 'kept a corrupt part'
[[ "$fn_output" == *'Model already present'* && "$fn_output" == *"Already present: $fn_volumes/$fn_ngram"* ]] ||
  fn_fail "fetched the verified model or volume again: $fn_output"

# A part that does not match its pin (the fixture leaves it empty) fails the run, is not promoted, and
# stays staged under its revision-scoped name for the next run.
rm -f -- "$fn_models/$fn_part"
if fn_output="$(fn_run "$tmp/bin" env NINFER_TEST_FAKE_SIZES="$fn_artifact=$fn_size $fn_ngram=$fn_ngram_size")"; then
  fn_fail "accepted a part that does not match its pin: $fn_output"
fi
[[ -e "$fn_models/$fn_part" ]] && fn_fail 'promoted a part that does not match its pin'
compgen -G "$fn_models/$fn_part.*.part" >/dev/null || fn_fail 'did not stage the part under a revision-scoped name'
rm -f -- "$fn_models/$fn_part."*".part"

# A size-correct model with the wrong checksum is refused (hashing on; the stub sha256sum never matches
# a real pin): the run fails at the model, promotes nothing and fetches nothing after it.
fn_models="$tmp/fn-models-hash"
fn_volumes="$tmp/fn-volumes-hash"
if fn_output="$(PATH="$tmp/bin:$PATH" NINFER_MODEL_DIR="$fn_models" NINFER_NGRAM_DIR="$fn_volumes" \
     NINFER_TEST_FAKE_SIZES="$fn_sizes" "$downloader" qwen38-flash-next 2>&1 </dev/null)"; then
  fn_fail "accepted a model whose checksum does not match: $fn_output"
fi
[[ -e "$fn_models/$fn_artifact" ]] && fn_fail 'promoted a model that failed its checksum'
[[ -e "$fn_models/$fn_part" || -e "$fn_volumes/$fn_ngram" ]] && fn_fail 'kept fetching after the model failed its checksum'

# With no terminal and no NINFER_NGRAM_DIR the volume goes beside the model, and no override is printed.
fn_models="$tmp/fn-models-beside"
fn_output="$(PATH="$tmp/bin:$PATH" NINFER_MODEL_DIR="$fn_models" NINFER_SKIP_SHA256=1 \
  NINFER_TEST_FAKE_SIZES="$fn_sizes" "$downloader" qwen38-flash-next 2>&1 </dev/null)" ||
  fn_fail "a fetch with no volume folder failed: $fn_output"
fn_size_ok "$fn_models/$fn_ngram" "$fn_ngram_size" || fn_fail 'did not put the volume beside the model by default'
[[ "$fn_output" == *'--ngram-volume'* ]] && fn_fail "printed a volume override for a volume beside the model: $fn_output"

# The qwen38-27b godmode graft is fetched best-effort after the model is in place. It is a pair the
# launcher trusts by the .bin alone and the loader needs whole with its .json, so the contract is
# all-or-nothing: whatever happens, a failed fetch must leave a whole verified pair or neither file,
# and must never fail the model download. These cases run the real fetch_graft against a stub curl
# that serves tiny payloads (the model itself is a sparse file that already verifies, so nothing
# large is fetched) and a stub sha256sum that recognises those payloads; no network is touched.
# The graft's two pins sit in fetch_graft; the model pins earlier in the file are different hashes.
graft_shas="$(sed -n '/^fetch_graft() {$/,/^}$/p' "$downloader" | grep -oE '[0-9a-f]{64}')"
graft_bin_sha="$(sed -n 1p <<<"$graft_shas")"
graft_json_sha="$(sed -n 2p <<<"$graft_shas")"
if [[ -z "$graft_bin_sha" || -z "$graft_json_sha" ]]; then
  printf 'download-model.sh has no graft pins to verify against\n' >&2
  exit 1
fi
mkdir -- "$tmp/graft-bin" "$tmp/graft-home" "$tmp/graft-models"
cat > "$tmp/graft-bin/curl" <<'CURL'
#!/usr/bin/env bash
# NINFER_TEST_GRAFT_MODE: ok (default), deny_json (the sidecar request is refused) or bad_json
# (the sidecar arrives with the wrong content). Authorization headers are logged to GRAFT_LOG.
header='' output='' url=''
while (( $# )); do
  case "$1" in
    -H) header="$2"; shift 2 ;;
    --output) output="$2"; shift 2 ;;
    -*) shift ;;
    *) url="$1"; shift ;;
  esac
done
printf '%s %s\n' "$header" "$url" >> "$GRAFT_LOG"
if [[ "${NINFER_TEST_GRAFT_MODE:-ok}" == deny_json && "$url" == *.json ]]; then exit 22; fi
if [[ "${NINFER_TEST_GRAFT_MODE:-ok}" == bad_json && "$url" == *.json ]]; then printf 'bad' > "$output"; exit 0; fi
printf 'good' > "$output"
CURL
cat > "$tmp/graft-bin/sha256sum" <<HASH
#!/usr/bin/env bash
file="\${*: -1}"
if [[ "\$(cat "\$file")" == good ]]; then
  case "\${file%.part}" in
    *.bin) printf '%s  %s\n' '$graft_bin_sha' "\$file" ;;
    *) printf '%s  %s\n' '$graft_json_sha' "\$file" ;;
  esac
else
  printf '%064d  %s\n' 0 "\$file"
fi
HASH
chmod +x "$tmp/graft-bin/curl" "$tmp/graft-bin/sha256sum"
graft_size="$(expected_size_of qwen38-27b)"
truncate -s "$graft_size" "$tmp/graft-models/$(artifact_of qwen38-27b)"
gdir="$tmp/graft-dir"
export GRAFT_LOG="$tmp/graft.log"

# Runs the downloader for qwen38-27b with no token configured anywhere unless the caller adds one.
run_graft() {
  env -u HF_TOKEN -u HF_TOKEN_PATH -u HF_HOME -u XDG_CACHE_HOME -u NINFER_GRAFTS \
    HOME="$tmp/graft-home" PATH="$tmp/graft-bin:$PATH" NINFER_MODEL_DIR="$tmp/graft-models" \
    NINFER_SKIP_SHA256=1 NINFER_GRAFT_DIR="$gdir" "$@" "$downloader" qwen38-27b 2>&1
}
graft_files() { (cd "$gdir" 2>/dev/null && ls -A | tr '\n' ' '); }
graft_fail() { printf 'download-model.sh graft: %s\n' "$1" >&2; exit 1; }
graft_reset() { rm -rf -- "$gdir"; : > "$GRAFT_LOG"; }
pair='godmode_q38_trained.bin godmode_q38_trained.json '

# No token anywhere: nothing is fetched, a lone half or a staged .part is dropped, and the model
# download still succeeds.
graft_reset; mkdir -- "$gdir"; printf 'good' > "$gdir/godmode_q38_trained.bin"; printf 'x' > "$gdir/godmode_q38_trained.json.part"
out="$(run_graft)" || graft_fail 'a missing token failed the model download'
[[ "$out" == *'no Hugging Face token'* ]] || graft_fail 'a missing token was not reported'
[[ -s "$GRAFT_LOG" ]] && graft_fail 'a missing token still reached curl'
[[ -z "$(graft_files)" ]] || graft_fail "a missing token left a broken pair behind: $(graft_files)"
# A whole pair that does not match the pinned hashes may be the user's own; without a token it stays.
printf 'mine' > "$gdir/godmode_q38_trained.bin"; printf 'mine' > "$gdir/godmode_q38_trained.json"
run_graft >/dev/null || graft_fail 'a missing token failed the model download'
[[ "$(graft_files)" == "$pair" ]] || graft_fail 'a whole unverified pair was deleted without a token'

# Token discovery follows huggingface_hub: HF_TOKEN, HF_TOKEN_PATH, HF_HOME/token, then
# XDG_CACHE_HOME/huggingface/token. Each must be honoured, and the pair installs whole.
mkdir -p -- "$tmp/graft-tokens/home" "$tmp/graft-tokens/xdg/huggingface"
printf 'tok-path\n' > "$tmp/graft-tokens/path-token"
printf 'tok-home\n' > "$tmp/graft-tokens/home/token"
printf 'tok-xdg\n' > "$tmp/graft-tokens/xdg/huggingface/token"
check_token() { # <expected token> <env assignment>...
  local expected="$1"; shift
  graft_reset
  out="$(run_graft "$@")" || graft_fail "the download failed with $*"
  grep -Fq "Bearer $expected " "$GRAFT_LOG" || graft_fail "token $expected was not used with $*"
  [[ "$(graft_files)" == "$pair" ]] || graft_fail "no whole pair after $*: $(graft_files)"
  [[ "$out" == *'Graft ready'* ]] || graft_fail "graft not reported ready with $*"
}
check_token env-token HF_TOKEN=env-token HF_TOKEN_PATH="$tmp/graft-tokens/path-token"
check_token tok-path HF_TOKEN_PATH="$tmp/graft-tokens/path-token" HF_HOME="$tmp/graft-tokens/home"
check_token tok-home HF_HOME="$tmp/graft-tokens/home" XDG_CACHE_HOME="$tmp/graft-tokens/xdg"
check_token tok-xdg XDG_CACHE_HOME="$tmp/graft-tokens/xdg"

# A verified pair is accepted without any fetch, and a second run is idempotent.
: > "$GRAFT_LOG"
out="$(run_graft HF_TOKEN=t)" || graft_fail 'a verified pair failed the model download'
[[ "$out" == *'Graft ready'* && ! -s "$GRAFT_LOG" ]] || graft_fail 'a verified pair was fetched again'

# Access denied for the second file, with a stale lone .bin on disk: neither file remains.
graft_reset; mkdir -- "$gdir"; printf 'stale' > "$gdir/godmode_q38_trained.bin"
out="$(run_graft HF_TOKEN=t NINFER_TEST_GRAFT_MODE=deny_json)" || graft_fail 'a denied sidecar failed the model download'
[[ "$out" == *'no access'* ]] || graft_fail 'a denied sidecar was not reported'
[[ -z "$(graft_files)" ]] || graft_fail "a denied sidecar left files behind: $(graft_files)"

# A sidecar with the wrong hash is not installed, and the binary that arrived first goes with it.
graft_reset
out="$(run_graft HF_TOKEN=t NINFER_TEST_GRAFT_MODE=bad_json)" || graft_fail 'a bad sidecar hash failed the model download'
[[ "$out" == *'failed sha256 verification'* ]] || graft_fail 'a bad sidecar hash was not reported'
[[ -z "$(graft_files)" ]] || graft_fail "a bad sidecar hash left files behind: $(graft_files)"

# A mismatched pair already on disk is replaced by the verified one.
graft_reset; mkdir -- "$gdir"; printf 'old' > "$gdir/godmode_q38_trained.bin"; printf 'old' > "$gdir/godmode_q38_trained.json"
run_graft HF_TOKEN=t >/dev/null || graft_fail 'replacing a mismatched pair failed the model download'
[[ "$(graft_files)" == "$pair" && "$(cat "$gdir/godmode_q38_trained.bin")" == good ]] || graft_fail 'a mismatched pair was not replaced'

# A directory that cannot be created skips the graft; the model download result is unchanged.
: > "$tmp/graft-afile"
out="$(run_graft HF_TOKEN=t NINFER_GRAFT_DIR="$tmp/graft-afile/sub")" || graft_fail 'an uncreatable graft directory failed the model download'
[[ "$out" == *'cannot create'* ]] || graft_fail 'an uncreatable graft directory was not reported'

# NINFER_GRAFTS=off skips the graft entirely.
graft_reset
run_graft HF_TOKEN=t NINFER_GRAFTS=off >/dev/null || graft_fail 'NINFER_GRAFTS=off failed the model download'
[[ ! -s "$GRAFT_LOG" && ! -e "$gdir" ]] || graft_fail 'NINFER_GRAFTS=off still fetched'

printf '%s\n' 'Linux script checks passed.'
