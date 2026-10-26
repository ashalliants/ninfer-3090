#!/usr/bin/env bash
# Builds, packages and smoke-tests the LINUX release archive on the tracked dev instance (a rented
# RTX 3090 running Ubuntu 24.04 + CUDA 12.8.1, the same baseline the Linux archive documents), then
# pulls the archive and its checksum file back into <repo-dir>/dist/.
#   ./vast-release-linux.sh [repo-dir]
# Prerequisites: vast-dev-launch.sh has finished (instance ready, model fetched), and the release
# commit is checked out and CLEAN in repo-dir -- VERSION bumped and docs/release-notes/RELEASE_NOTES_X.Y.Z.md committed,
# because the archive is a `git archive` of that exact commit and the packager refuses without them.
# The build runs detached on the instance and is polled, so a dropped SSH connection costs nothing.
# Does not destroy the instance: pull everything you need, then ./vast-destroy.sh.
set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo="$(cd -- "${1:-$HOME/ash/ninfer-3090}" && pwd)"

# The archive is `git archive HEAD` on the instance: uncommitted edits to VERSION or the notes would be
# synced but not archived, and the packager would then build a different release than the one named.
if [[ -n "$(git -C "$repo" status --porcelain)" ]]; then
  echo "Refusing to release from a dirty tree: commit (or stash) first. Ignored files are fine." >&2
  exit 1
fi
commit="$(git -C "$repo" rev-parse HEAD)"

echo "==> Syncing the release commit to the instance"
"$root/vast-dev-sync.sh" "$repo"
remote_head="$("$root/vast-dev-ssh.sh" 'git -C /root/ninfer-3090 rev-parse HEAD' | tr -d '\r\n')"
if [[ "$remote_head" != "$commit" ]]; then
  echo "The instance is at ${remote_head:-<unknown>}, not the release commit $commit; not building." >&2
  exit 1
fi

echo "==> Writing the release script on the instance"
"$root/vast-dev-ssh.sh" 'cat > /root/release-linux.sh && chmod +x /root/release-linux.sh' <<'REMOTE_EOF'
#!/usr/bin/env bash
# Runs on the instance. Status file: last line DONE or FAILED.
set -euo pipefail
status=/root/release-linux.status
echo RUNNING > "$status"
trap 'rc=$?; if (( rc != 0 )); then echo FAILED >> "$status"; fi' EXIT

export PATH="/usr/local/cuda/bin:$PATH"
export CC=/usr/bin/gcc-13 CXX=/usr/bin/g++-13 CUDACXX=/usr/local/cuda/bin/nvcc CUDAHOSTCXX=/usr/bin/g++-13
repo=/root/ninfer-3090
rel=/root/ninfer-rel
smoke=/root/ninfer-smoke

echo "==> [$(date -Is)] exporting $(git -C "$repo" rev-parse --short HEAD)"
rm -rf "$rel" "$smoke"
mkdir -p "$rel" "$smoke"
git -C "$repo" archive HEAD | tar -x -C "$rel"
cd "$rel"

command -v jq >/dev/null || { apt-get update -qq && apt-get install --yes --no-install-recommends jq; }

mem_gib=$(( $(awk '/MemTotal/ {print $2}' /proc/meminfo) / 1048576 ))
jobs=$(( mem_gib / 4 ))
(( jobs > $(nproc) )) && jobs=$(nproc)
(( jobs > 32 )) && jobs=32   # nproc reports the host's cores, not the share this container gets
(( jobs < 1 )) && jobs=1

echo "==> [$(date -Is)] configuring (-j$jobs)"
cmake -S . -B build-linux -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER="$CC" -DCMAKE_CXX_COMPILER="$CXX" \
  -DCMAKE_CUDA_COMPILER="$CUDACXX" -DCMAKE_CUDA_HOST_COMPILER="$CUDAHOSTCXX" \
  -DCMAKE_CUDA_ARCHITECTURES=86 -DBUILD_TESTING=OFF -DNINFER_BUILD_BENCHMARKS=ON
# One --target per invocation: `cmake --build --target A B` can silently build only A on some generators.
for target in ninfer ninfer-serve ninfer_bench; do
  echo "==> [$(date -Is)] building $target"
  cmake --build build-linux --parallel "$jobs" --target "$target"
done

echo "==> [$(date -Is)] packaging"
scripts/package-release.sh
ls -l dist

archive="$(ls dist/ninfer-rtx3090-linux-x64-*.tar.gz)"
echo "==> [$(date -Is)] unpacking $archive for the checks"
tar -xzf "$archive" -C "$smoke"
dir="$(ls -d "$smoke"/ninfer-rtx3090-linux-x64-*)"

echo "==> checks: version, help, ldd"
"$dir/ninfer-serve" --version
"$dir/ninfer-serve" --help >/dev/null
"$dir/run.sh" --help >/dev/null || true
ldd "$dir/ninfer-serve" | tee /root/release-ldd.txt
if grep -q 'not found' /root/release-ldd.txt; then
  echo "ldd reports unresolved libraries (see above)"; exit 1
fi
(cd "$dir" && sha256sum -c SHA256SUMS.txt)

echo "==> [$(date -Is)] smoke test: default profile, one chat completion on the real card"
[[ -f /root/.model-ready ]] || { echo "model not ready (/root/.model-ready missing)"; exit 1; }
# A port of this run's own: with the default 8080, any other server already listening there would answer
# the readiness probe and the chat request, and the smoke test would pass without touching the archive.
port=$(( 20000 + RANDOM % 20000 ))
while ss -ltn "sport = :$port" | grep -q LISTEN; do port=$(( 20000 + RANDOM % 20000 )); done
echo "smoke test port: $port"
(cd "$dir" && NINFER_PORT="$port" NINFER_MODEL_DIR=/root/models nohup ./run.sh qwen38-27b > /root/smoke-serve.log 2>&1 & echo $! > /root/smoke-serve.pid)
pid="$(cat /root/smoke-serve.pid)"
stop_server() { pkill -P "$pid" 2>/dev/null || true; kill "$pid" 2>/dev/null || true; }
trap 'rc=$?; stop_server; if (( rc != 0 )); then echo FAILED >> "$status"; fi' EXIT
ready=0
for _ in $(seq 1 120); do
  if curl -fs http://127.0.0.1:$port/v1/models > /root/smoke-models.json 2>/dev/null; then ready=1; break; fi
  kill -0 "$pid" 2>/dev/null || { echo "server exited early:"; tail -40 /root/smoke-serve.log; exit 1; }
  sleep 5
done
(( ready )) || { echo "server not ready after 10 minutes:"; tail -40 /root/smoke-serve.log; exit 1; }
model_id="$(jq -r '.data[0].id' /root/smoke-models.json)"
echo "served model id: $model_id"
reply="$(curl -fs http://127.0.0.1:$port/v1/chat/completions -H 'Content-Type: application/json' \
  -d "{\"model\":\"$model_id\",\"max_tokens\":200,\"messages\":[{\"role\":\"user\",\"content\":\"Say hello in five words.\"}]}")"
echo "$reply" | jq '{finish: .choices[0].finish_reason, content: .choices[0].message.content, reasoning: (.choices[0].message.reasoning_content // "" | .[0:200])}'
echo "$reply" | jq -e '(.choices[0].message.content // "") + (.choices[0].message.reasoning_content // "") | length > 0' >/dev/null \
  || { echo "empty completion"; exit 1; }
echo "SMOKE TEST PASSED" | tee /root/release-smoke-ok.txt

stop_server
echo "==> [$(date -Is)] done"
echo DONE >> "$status"
REMOTE_EOF

echo "==> Starting the release build on the instance (detached)"
"$root/vast-dev-ssh.sh" 'rm -f /root/release-linux.status; nohup /root/release-linux.sh > /root/release-linux.log 2>&1 < /dev/null & sleep 2; tail -2 /root/release-linux.log'

echo "==> Polling every 60 s (log: /root/release-linux.log on the instance)"
while true; do
  sleep 60
  state="$("$root/vast-dev-ssh.sh" 'tail -n1 /root/release-linux.status 2>/dev/null || echo WAITING' || echo UNREACHABLE)"
  echo "    $(date +%H:%M:%S) $state: $("$root/vast-dev-ssh.sh" 'tail -n1 /root/release-linux.log' 2>/dev/null | cut -c1-120)"
  case "$state" in
    DONE) break ;;
    FAILED)
      echo "Release build FAILED. Last 60 log lines:" >&2
      "$root/vast-dev-ssh.sh" 'tail -n 60 /root/release-linux.log' >&2
      exit 1 ;;
  esac
done

echo "==> Pulling the archive back to $repo/dist"
mkdir -p "$repo/dist"
names="$("$root/vast-dev-ssh.sh" 'cd /root/ninfer-rel/dist && ls *linux*.tar.gz SHA256SUMS-*linux.txt' | tr -d '\r')"
for name in $names; do
  "$root/vast-dev-ssh.sh" "cat /root/ninfer-rel/dist/$name" > "$repo/dist/$name"
done
"$root/vast-dev-ssh.sh" 'cat /root/release-ldd.txt' > "$repo/dist/release-ldd-linux.txt"
# Only the pair just pulled: dist/ may hold older releases' files, whose archives are not there.
( cd "$repo/dist" && for sums in $(printf '%s\n' $names | grep '^SHA256SUMS-'); do { sha256sum -c "$sums" 2>/dev/null || shasum -a 256 -c "$sums"; }; done )
ls -l "$repo/dist"
echo "Release commit: $commit"
echo "Linux archive built, checked and smoke-tested on the real card. Destroy the instance with ./vast-destroy.sh when you are done."
