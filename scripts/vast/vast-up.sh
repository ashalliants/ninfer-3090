#!/usr/bin/env bash
# Rent a Vast RTX 3090 for a Linux release build, install the build dependencies, arm an on-box kill
# switch, and fetch the 27B model the smoke test serves. Returns when the instance is ready.
#   ./vast-up.sh [kill-after-hours]      (default 3; the box destroys itself after that, whatever happens)
# Needs the `vastai` CLI logged in. Cost is about $0.4/hour; a release takes about an hour. Destroy it
# with ./vast-destroy.sh as soon as the archive is back.
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$root/vast-common.sh"
hours="${1:-3}"

if [[ -s "$state_file" ]]; then
  echo "Instance $(cat "$state_file") is already tracked; destroy it first (vast-destroy.sh)." >&2; exit 1
fi

# Cheapest verified single-3090 offer with a fast link, CUDA 12.8-capable drivers and >=16 effective cores
# (a release build is 15 min on 24 cores, an hour on 6). Spain hosts were the reliable ones in 2026-10.
offer="$(vastai search offers 'gpu_name=RTX_3090 num_gpus=1 reliability>0.98 disk_space>=100 cuda_max_good>=12.8 rentable=true inet_down>=1000' -o dph --raw |
  python3 -I -c '
import json, sys
offers = [o for o in json.load(sys.stdin) if o["cpu_cores_effective"] >= 16]
if not offers: sys.exit("no suitable offer")
o = offers[0]
print(o["id"], file=sys.stdout)
print("offer %s: machine %s, %s, $%.3f/h, %d cores, %d Mbps down, $%.4f/GB down" % (o["id"], o["machine_id"], o["geolocation"], o["dph_total"], o["cpu_cores_effective"], o["inet_down"], o["inet_down_cost"]), file=sys.stderr)')"

onstart="$(mktemp)"
cat > "$onstart" <<ONSTART
#!/bin/bash
# Dead-man switch first, before anything can fail.
cat > /root/killswitch.sh <<'KS'
#!/bin/bash
sleep $(( hours * 3600 ))
for i in 1 2 3 4 5; do
  curl -s -X DELETE "https://console.vast.ai/api/v0/instances/\$CONTAINER_ID/" -H "Authorization: Bearer \$CONTAINER_API_KEY" && break
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

new="$(vastai create instance "$offer" --image vastai/base-image:cuda-12.8.1-auto --disk 80 --ssh --direct \
  --onstart "$onstart" --label ninfer-release --raw | python3 -I -c 'import json,sys; print(json.load(sys.stdin)["new_contract"])')"
rm -f "$onstart"
echo "$new" > "$state_file"
echo "Rented instance $new. Kill switch: ${hours} h. Waiting for it to boot..."

# A host that never leaves "loading" burns money: give up after 8 minutes (destroy and re-run).
for i in $(seq 1 32); do
  status="$(vastai show instance "$new" --raw | python3 -I -c 'import json,sys; print(json.load(sys.stdin).get("actual_status"))')"
  [[ "$status" == running ]] && break
  sleep 15
done
[[ "$status" == running ]] || { echo "Instance still '$status' after 8 minutes: ./vast-destroy.sh and try again." >&2; exit 1; }
for i in $(seq 1 60); do "$root/vast-dev-ssh.sh" 'test -f /root/deps.done' 2>/dev/null && break; sleep 10; done
"$root/vast-dev-ssh.sh" 'test -f /root/deps.done' || { echo "Dependencies did not install (see /root/onstart log)." >&2; exit 1; }
"$root/vast-dev-ssh.sh" 'nvidia-smi -L; nproc'

echo "==> Fetching the 27B model for the smoke test (about 19 GB)"
"$root/vast-dev-ssh.sh" 'rm -f /root/.model-ready; mkdir -p /root/models /root/dl && cd /root/dl &&
  curl -fsSL https://raw.githubusercontent.com/ashalliants/ninfer-3090/master/scripts/download-model.sh -o download-model.sh &&
  NINFER_MODEL_DIR=/root/models bash download-model.sh qwen38-27b > /root/model-dl.log 2>&1 && touch /root/.model-ready'
"$root/vast-dev-ssh.sh" 'test -f /root/.model-ready' || { echo "Model download failed: tail /root/model-dl.log on the instance." >&2; exit 1; }
echo "Ready. Next: ./vast-release-linux.sh <repo-dir>"
