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
# The key Vast has for this account: NINFER_VAST_KEY, else ~/.ssh/id_rsa_ash (the one release-away-from-3090.md
# documents), else ~/.ssh/id_ed25519. IdentityAgent=none and IdentitiesOnly stop ssh offering other keys.
vast_key="${NINFER_VAST_KEY:-}"
if [[ -z "$vast_key" ]]; then
  for k in "$HOME/.ssh/id_rsa_ash" "$HOME/.ssh/id_ed25519"; do [[ -f "$k" ]] && { vast_key="$k"; break; }; done
fi
[[ -f "$vast_key" ]] || { echo "No ssh key for Vast: set NINFER_VAST_KEY." >&2; return 1 2>/dev/null || exit 1; }
vast_ssh_opts=(-i "$vast_key" -o IdentityAgent=none -o IdentitiesOnly=yes -o StrictHostKeyChecking=no
               -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -o ConnectTimeout=25 -o ServerAliveInterval=30)
