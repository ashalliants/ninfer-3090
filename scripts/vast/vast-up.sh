#!/usr/bin/env bash
# Rent a Vast RTX 3090 for a Linux release build, install the build dependencies, arm an on-box kill
# switch, and fetch the 27B model the smoke test serves. Returns when the instance is ready.
#   ./vast-up.sh [repo-dir] [kill-after-hours]   (repo-dir defaults to this repository; the box destroys
#                                                itself after the hours, default 3, whatever happens)
# Needs the `vastai` CLI logged in. A release takes about an hour; destroy the box with ./vast-destroy.sh
# as soon as the archive is back.
#
# The offer is chosen by its estimated total for the job, not its hourly rate: hosts bill bandwidth per GB,
# and on the v0.17.0 release box (a $0.25/h host at $0.039/GB) downloads cost more than the GPU time
# ($0.43 for 11 GB against $0.27 for 1.1 h, $0.85 in all). The estimate's inputs, with what one release
# with the default qwen38-27b smoke test uses:
#   NINFER_VAST_EST_HOURS  hours the box is up                                  (default 1)
#   NINFER_VAST_DOWN_GB    GB downloaded: the 17.7 GiB model, apt packages,      (default 21)
#                          the previous release's archive if you diff its ldd
#   NINFER_VAST_UP_GB      GB uploaded: about 1 GB per Linux archive pulled back (default 2)
# A Bonsai 2 27B smoke test downloads about 11 GB instead.
#
# A host can lower the card's power limit and the offer listing does not say so: the v0.17 box was a 3090
# limited to 220 W of its 350 W default and ran its weight kernels at 54-57% of the card's read bandwidth
# (docs/maintainer/perf-model.md). So once a box is running, power_guard.py reads its limit, and a card
# below the floor is destroyed and the next-cheapest offer tried:
#   NINFER_VAST_MIN_POWER_PCT  lowest acceptable power limit, as a percent of the card's default limit
#                              (default 100, a full-power card; 80 is the lowest worth measuring on; 0 is off,
#                              for a build that does not care how fast the card is)
#   NINFER_VAST_MAX_HOSTS      how many offers to try before giving up (default 4); each rejected host
#                              bills for the minutes it took to boot
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$root/vast-common.sh"
repo="$(cd -- "${1:-$root/../..}" && pwd)"
hours="${2:-3}"

if [[ -s "$state_file" ]]; then
  echo "Instance $(cat "$state_file") is already tracked; destroy it first (vast-destroy.sh)." >&2; exit 1
fi

disk_gb=80
min_power_pct="${NINFER_VAST_MIN_POWER_PCT:-100}"
max_hosts="${NINFER_VAST_MAX_HOSTS:-4}"
[[ "$min_power_pct" =~ ^[0-9]+$ ]] && (( min_power_pct <= 100 )) || { echo "NINFER_VAST_MIN_POWER_PCT must be 0-100, got '$min_power_pct'." >&2; exit 1; }
[[ "$max_hosts" =~ ^[1-9][0-9]*$ ]] || { echo "NINFER_VAST_MAX_HOSTS must be a positive number, got '$max_hosts'." >&2; exit 1; }
# Verified single-3090 offers with a fast link, CUDA 12.8-capable drivers and >=16 effective cores (a release
# build is 15 min on 24 cores, an hour on 6), ranked by estimated total. Spain hosts were the reliable ones in
# 2026-10. The cheapest max_hosts are the candidates, in order.
# Vast lists a gpu_max_power (watts) per offer. Its docs call it the "GPU power limit" without saying whether
# that is the enforced limit or the card's rating, so it only narrows the search (a 3090's default is
# 350 W); the check after boot is the one that decides. Each candidate prints its value, so the first real
# run shows which it is.
power_filter=""
if (( min_power_pct > 0 )); then power_filter=" gpu_max_power>=$(( 350 * min_power_pct / 100 ))"; fi
offers="$(vastai search offers "gpu_name=RTX_3090 num_gpus=1 reliability>0.98 disk_space>=100 cuda_max_good>=12.8 rentable=true inet_down>=1000${power_filter}" -o dph --raw |
  NINFER_VAST_DISK_GB="$disk_gb" NINFER_VAST_MAX_HOSTS="$max_hosts" python3 -I -c '
import json, os, sys
hours = float(os.environ.get("NINFER_VAST_EST_HOURS", "1"))
down = float(os.environ.get("NINFER_VAST_DOWN_GB", "21"))
up = float(os.environ.get("NINFER_VAST_UP_GB", "2"))
disk = float(os.environ["NINFER_VAST_DISK_GB"])
def cost(o):
    gpu = o["dph_base"] * hours
    storage = o["storage_cost"] * disk / 730 * hours   # storage_cost is per GB-month
    net = o["inet_down_cost"] * down + o["inet_up_cost"] * up
    return gpu + storage + net, gpu, storage, net
offers = sorted((o for o in json.load(sys.stdin) if o["cpu_cores_effective"] >= 16), key=lambda o: cost(o)[0])
if not offers: sys.exit("no suitable offer")
print("estimate for %.1f h, %.0f GB down, %.0f GB up:" % (hours, down, up), file=sys.stderr)
candidates = offers[:int(os.environ["NINFER_VAST_MAX_HOSTS"])]
for o in candidates:
    total, gpu, storage, net = cost(o)
    print("  offer %s: %s, $%.2f (GPU $%.2f + disk $%.2f + bandwidth $%.2f; $%.3f/h, $%.4f/GB down, $%.4f/GB up), %d cores, %d Mbps down, listed power %s W"
          % (o["id"], o["geolocation"], total, gpu, storage, net, o["dph_base"], o["inet_down_cost"], o["inet_up_cost"], o["cpu_cores_effective"], o["inet_down"], o.get("gpu_max_power", "?")), file=sys.stderr)
for o in candidates:
    print(o["id"])')" || { echo "No suitable offer. The search required gpu_max_power >= $(( 350 * min_power_pct / 100 )) W; if that emptied it, lower NINFER_VAST_MIN_POWER_PCT (0 drops the filter and the check)." >&2; exit 1; }
[[ -n "$offers" ]] || { echo "No suitable offer." >&2; exit 1; }
if (( min_power_pct > 0 )); then
  echo "Renting the first one that boots with a power limit of at least ${min_power_pct}% of the card's default."
else
  echo "Renting the first one (power-limit check off)."
fi

onstart="$(mktemp)"
cat > "$onstart" <<ONSTART
#!/bin/bash
# Dead-man switch first, before anything can fail.
cat > /root/killswitch.sh <<'KS'
#!/bin/bash
sleep $(( hours * 3600 ))
for i in 1 2 3 4 5; do
  curl -fsS --connect-timeout 10 --max-time 30 -X DELETE "https://console.vast.ai/api/v0/instances/\$CONTAINER_ID/" \\
    -H "Authorization: Bearer \$CONTAINER_API_KEY" && break
  sleep 10
done
poweroff -f || halt -f
KS
chmod +x /root/killswitch.sh
env | grep -E '^(CONTAINER_ID|CONTAINER_API_KEY)=' | sed 's/^/export /' > /root/vastenv.sh
nohup setsid bash -c 'source /root/vastenv.sh; export CONTAINER_ID CONTAINER_API_KEY; /root/killswitch.sh' >/dev/null 2>&1 &
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq git cmake ninja-build build-essential pkg-config gcc-13 g++-13 aria2 jq python3 rsync \\
  libavcodec-dev libavformat-dev libavutil-dev libswscale-dev libavfilter-dev libswresample-dev \\
  libcurl4-openssl-dev libzstd-dev ccache python3-jinja2 python3-jsonschema >/dev/null 2>&1
touch /root/deps.done
ONSTART

# Any failure destroys the instance being tried: it bills. The state file goes only once the destroy is
# confirmed; if it fails the file stays, so vast-destroy.sh can retry and a new vast-up.sh refuses to rent.
new=""
destroy_instance() {
  if echo y | vastai destroy instance "$1" >&2; then rm -f "$state_file"; return 0; fi
  echo "Destroy FAILED: instance $1 is still billing. Run vast-destroy.sh until it succeeds." >&2
  return 1
}
trap 'rc=$?; if (( rc != 0 )) && [[ -n "$new" ]]; then echo "vast-up failed: destroying instance $new" >&2; destroy_instance "$new" || true; fi' EXIT

# Try the candidates cheapest first. A host is kept only if it boots within 8 minutes and its card may draw
# enough power; otherwise it is destroyed and the next one tried.
chosen=""
tried=0
for offer in $offers; do
  tried=$((tried + 1))
  new="$(vastai create instance "$offer" --image vastai/base-image:cuda-12.8.1-auto --disk "$disk_gb" --ssh --direct \
    --onstart "$onstart" --label ninfer-release --raw | python3 -I -c 'import json,sys; print(json.load(sys.stdin)["new_contract"])')"
  echo "$new" > "$state_file"
  echo "Rented instance $new (offer $offer). Kill switch: ${hours} h. Waiting for it to boot..."

  status=""
  for i in $(seq 1 32); do
    status="$(vastai show instance "$new" --raw | python3 -I -c 'import json,sys; print(json.load(sys.stdin).get("actual_status"))')"
    [[ "$status" == running ]] && break
    sleep 15
  done
  if [[ "$status" != running ]]; then
    echo "Offer $offer: still '$status' after 8 minutes; destroying it and trying the next host." >&2
    destroy_instance "$new" || exit 1
    new=""; continue
  fi

  # The offer cannot tell us the card's power limit; the box can. sshd may need a moment after "running".
  smi=""
  for i in $(seq 1 18); do
    if smi="$("$root/vast-dev-ssh.sh" 'nvidia-smi --query-gpu=power.limit,power.default_limit --format=csv,noheader,nounits' </dev/null 2>/dev/null)"; then break; fi
    smi=""; sleep 10
  done
  power_rc=0
  verdict="$(printf '%s\n' "$smi" | python3 -I "$root/power_guard.py" --min-pct "$min_power_pct")" || power_rc=$?
  if (( power_rc == 0 )); then echo "Offer $offer: $verdict."; chosen="$new"; break; fi
  echo "Offer $offer rejected: $verdict. Destroying it and trying the next host." >&2
  destroy_instance "$new" || exit 1
  new=""
done
rm -f "$onstart"
[[ -n "$chosen" ]] || { echo "None of the $tried offers tried booted with a power limit of at least ${min_power_pct}% of the card's default. Run again (new offers appear all the time), raise NINFER_VAST_MAX_HOSTS, or accept a capped card with NINFER_VAST_MIN_POWER_PCT=80 (0 for a build that does not time anything)." >&2; exit 1; }

for i in $(seq 1 60); do "$root/vast-dev-ssh.sh" 'test -f /root/deps.done' 2>/dev/null && break; sleep 10; done
"$root/vast-dev-ssh.sh" 'test -f /root/deps.done' || { echo "Dependencies did not install (see /root/onstart log)." >&2; exit 1; }
"$root/vast-dev-ssh.sh" 'nvidia-smi -L; nproc'

echo "==> Fetching the 27B model for the smoke test (about 19 GB)"
# The downloader comes from the release commit in repo-dir, not from a branch on GitHub: the release is the
# only source of release-time behaviour, and nothing mutable is executed as root on the box.
git -C "$repo" show HEAD:scripts/download-model.sh | "$root/vast-dev-ssh.sh" 'rm -f /root/.model-ready; mkdir -p /root/models /root/dl && cat > /root/dl/download-model.sh'
"$root/vast-dev-ssh.sh" 'cd /root/dl && NINFER_MODEL_DIR=/root/models bash download-model.sh qwen38-27b > /root/model-dl.log 2>&1 && touch /root/.model-ready' || true
"$root/vast-dev-ssh.sh" 'test -f /root/.model-ready' || { echo "Model download failed (tail /root/model-dl.log on the instance, now destroyed)." >&2; exit 1; }
echo "Ready. Next: ./vast-release-linux.sh <repo-dir>"
trap - EXIT
