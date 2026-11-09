#!/usr/bin/env bash
# Destroy the release instance this directory's scripts rented (and only that one).
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$root/vast-common.sh"
id="$(instance_id)"
echo y | vastai destroy instance "$id"
rm -f "$state_file"
echo "destroyed $id"
