#!/usr/bin/env bash
# Run a command on the release instance (stdin is passed through): ./vast-dev-ssh.sh 'nvidia-smi -L'
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$root/vast-common.sh"
vast_ssh_target
exec ssh "${vast_ssh_opts[@]}" -p "$vast_port" "$vast_user@$vast_host" "$@"
