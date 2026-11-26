# Sourced by scripts/build.sh and scripts/v100_build.sh.
#
# ccache (base_dir) hashes paths relative to the build directory, so a tree shares cache entries
# with other worktrees' builds only when it sits directly in its checkout like the default: every
# such tree hashes `-I../include`. A tree anywhere else (~/build-v100, say) still builds, but misses
# every object another worktree compiled. Measured on the V100 box: a rebuild after a flag change
# took about 25 minutes there and 12 seconds in <checkout>/build-v100.
#
# note_build_dir_outside_checkout BUILD_DIR CHECKOUT OPTION DEFAULT_NAME
# prints a note on stderr when BUILD_DIR's real location is not directly inside CHECKOUT. Symlinks
# are resolved, dangling ones included (ccache sees the compiler's physical working directory), so a
# link at <checkout>/build-v100 that points elsewhere is noted too. Call it after every override is parsed.
note_build_dir_outside_checkout() {
  local build_dir="$1" checkout="$2" option="$3" default_name="$4" real real_parent real_checkout shown
  # realpath -m follows symlinks even when the target (or the tree itself) does not exist yet, so a
  # link to a directory cmake has not created is resolved too.
  real="$(realpath -m -- "$build_dir")"
  real_parent="$(dirname -- "$real")"
  shown="$build_dir"
  [[ "$real" != "$(realpath -ms -- "$build_dir")" ]] && shown="$build_dir -> $real"
  real_checkout="$(realpath -- "$checkout")"
  if [[ "$real_parent" != "$real_checkout" ]]; then
    printf 'note: the build tree %s (%s) is not directly inside this checkout (%s), so ccache\n' \
      "$shown" "$option" "$checkout" >&2
    printf "      cannot share objects with other worktrees' builds; build in a real directory at %s/%s\n" \
      "$checkout" "$default_name" >&2
  fi
}
