# Sourced by the vast-*.sh scripts. The release instance's id lives in one state file, so these scripts
# can only ever touch the instance they rented themselves, never another one on the same Vast account.
state_dir="${NINFER_VAST_STATE:-$HOME/.cache/ninfer-release}"
state_file="$state_dir/instance_id"
mkdir -p "$state_dir"

instance_id() {
  [[ -s "$state_file" ]] || { echo "No release instance: run vast-up.sh first." >&2; return 1; }
  cat "$state_file"
}

# ssh options shared by the ssh and sync helpers.
vast_ssh_target() {
  local url; url="$(vastai ssh-url "$(instance_id)")"
  url="${url#ssh://}"
  vast_user="${url%%@*}"; local hostport="${url#*@}"
  vast_host="${hostport%:*}"; vast_port="${hostport##*:}"
}
vast_ssh_opts=(-i "${NINFER_VAST_KEY:-$HOME/.ssh/id_ed25519}" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null
               -o LogLevel=ERROR -o ConnectTimeout=25 -o ServerAliveInterval=30)
