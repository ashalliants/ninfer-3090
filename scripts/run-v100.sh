#!/usr/bin/env bash
# ------------------------------------------------------------------------------------------------
# Serve a model on one Tesla V100 (sm_70): run.sh with the V100's own defaults.
#
#   run-v100.sh <model> [profile]          e.g.  run-v100.sh qwen38-flash-next
#
# Every run.sh variable (NINFER_*) still applies and wins over what this script picks; run.sh's
# header documents them. What this adds:
#
#   - the server listens on every interface (NINFER_HOST=0.0.0.0; run.sh's default is 127.0.0.1),
#     so other machines on the network reach it at http://<this machine's IP or name>:8080
#     (NINFER_HOST=127.0.0.1 keeps it local). Set NINFER_API_KEY too: without a key anyone who can
#     reach the port can use the model, and the server says so at startup;
#   - the sm_70 server: NINFER_SERVER, else the first ninfer-serve found in build-v100/ beside the
#     checkout, in ../build-v100/, or in /workspace/build-v100/ (scripts/v100_build.sh's default);
#   - Qwen3.8-Flash-Next: run.sh's `tuned` profile unchanged. Measured on a V100-PCIE-32GB
#     (2026-11-17, docs/maintainer/volta-port.md): it beats a minimal flag set (MTP3 decode 68.9
#     against 63 tok/s, a 2K-token prefill 293 against 251 tok/s), and leaving out any one of its
#     flags is neutral or slower (MTP joint acceptance -4 tok/s, the shared-expert stream -1.7, CPU
#     prefill share 0 -> 226 tok/s prefill, 0.6 -> 285);
#   - Qwen3.8-Flash-Next from Unsloth's UD-IQ3_XXS (qwen38-flash-next-iq3): the same flags, with its own
#     limits. It runs with --partial-ram-residency by default (the launcher's setting: its 42 GiB of
#     experts pin only what the VRAM cache does not hold, and the mode takes no images and no elastic KV
#     pool), so its context defaults to 32,768 tokens instead of 262,144. NINFER_PARTIAL_RAM_RESIDENCY=off
#     pins every expert (about 50 GiB of available RAM; the test host has 59 GiB) and brings back the
#     full context and images. Not run on a V100: its kernels compile for sm_70 and its op tests are
#     the ones scripts/v100_validate.sh --tests runs (docs/tesla-v100.md, "Qwen3.8-Flash-Next, UD-IQ3_XXS");
#   - Qwen3.8-27B: MTP3 with INT8 KV (NINFER_SPEC=mtp, NINFER_KV_DTYPE=int8) instead of run.sh's
#     DFlash2 with rk4v4. On the V100 MTP3 decodes 76 tok/s against DFlash2's 66 (code), and INT8
#     KV is the cache validated on the card;
#   - Flash-Next prefill calls of up to 1024 tokens hand the CPU every expert miss it can take
#     (--prefill-cpu-all-columns 1024; NINFER_PREFILL_CPU_ALL_COLUMNS=0 turns it off), and longer
#     calls hand it cold experts routed up to 64 tokens at share 0.5 (--prefill-cpu-max-columns 64
#     --prefill-cpu-share 0.5; NINFER_PREFILL_CPU_MAX_COLUMNS=8 NINFER_PREFILL_CPU_SHARE=0.4 restore
#     the tuned profile's);
#   - Flash-Next holds as much of the 29 GB n-gram table in RAM as the RAM ledger spares after the
#     experts, the context cache and a 2 GiB margin beyond the reserve (--ngram-ram-mib auto; ~13
#     GiB, half the table, on a 59 GiB host; ~6 GiB stays available), loaded in ~6 s after startup.
#     A prompt's row read halves (3.9K tokens 0.86 -> 0.50 s): 4-8% faster prefill (939 tokens
#     1.82-2.00 -> 1.68-1.71 s), the same output. NINFER_NGRAM_RAM_MIB=0 turns it off (when other
#     programs need that RAM), or N caps it at N MiB;
#   - Flash-Next serves a 262,144-token context instead of run.sh's 32,768 (NINFER_CONTEXT
#     overrides). The INT8 KV pool backs 496 MiB at startup and grows into the expert cache's frames
#     only as a conversation gets long (3.6 GiB at 256K tokens), so the cache starts 222 frames (1%)
#     smaller than at 8K; a 256K-token checkpoint (3.6 GiB) still fits the 4 GiB Host context tier;
#   - Flash-Next MTP drafts step by step (--mtp-staged-drafts; NINFER_MTP_STAGED_DRAFTS=off turns it
#     off): +2-4% decode, the same output;
#   - Flash-Next MTP verification rounds compute part of their uncached experts on the CPU
#     (--cpu-verify-misses; NINFER_CPU_VERIFY_MISSES=off turns it off), the same output. About a
#     quarter of the routed experts do not fit in VRAM, so a prompt's first pass misses 3-7% of
#     the experts it routes to. Over this card's 3.3 GB/s link, that first pass decoded 48-63
#     tok/s and now decodes 68-80 (2026-11-18). With a warm cache it is within 2% either way;
#     The CPU threads are nproc - 6, clamped to 6..10 (NINFER_CPU_VERIFY_WORKERS overrides): on an
#     8-core/16-thread host 10 or 12 measured +1-3% over the default 6, 14 lost to oversubscription;
#   - Flash-Next keeps only the token embedding's first 8192 rows in VRAM (--hot-embedding-rows
#     8192; NINFER_HOT_EMBEDDING_ROWS=0 keeps the whole table): 216 more expert frames, the same
#     output, a prompt's first pass +1-1.5%;
#   - the memory-lock limit: Flash-Next pins 33 GiB of experts. When `ulimit -l` is below that and
#     cannot be raised here, NINFER_ALLOW_UNLOCKED_HOST_MEMORY=on is set (the server then continues
#     and still fails loudly if the pin itself fails). To lift the limit for good, as root:
#       printf '%s - memlock unlimited\n' "$USER" > /etc/security/limits.d/90-ninfer-memlock.conf
#     then log in again;
#   - Flash-Next runs prefill's BF16 and GGML linears and prompt attention on the FP16 tensor cores
#     (NINFER_SM70_TENSOR_CORES=1; =0 turns it off): a 3.9K-token prompt 6.9 -> 6.0 s, close to but
#     not bit for bit the plain server's output (docs/maintainer/volta-port.md "Flash-Next: the
#     tensor-core route");
#   - checks before starting: the card is a V100 (a warning otherwise), no other process is on it
#     (a warning: the server sizes itself to the free memory), and the power limit (the card's
#     default is 250 W; a lower cap slows decode, which is clock-bound on Volta).
#
# Temperatures: a passive V100 needs real airflow. Sustained prefill holds it at its power cap;
# watch `nvidia-smi -l 1` the first time on a new cooling setup.
# ------------------------------------------------------------------------------------------------
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
root="$(cd -- "$script_dir/.." && pwd)"
model_key="${1:-}"

if [[ -z "${NINFER_SERVER:-}" ]]; then
  # scripts/v100_build.sh's tree: BUILD_DIR, else its default $HOME/build-v100; then build-v100/
  # beside or above the checkout, and a rented box's /workspace.
  for candidate in ${BUILD_DIR:+"$BUILD_DIR/apps/ninfer-serve"} "$HOME/build-v100/apps/ninfer-serve" \
                   "$root/build-v100/apps/ninfer-serve" "$root/../build-v100/apps/ninfer-serve" \
                   "/workspace/build-v100/apps/ninfer-serve"; do
    if [[ -x "$candidate" ]]; then
      NINFER_SERVER="$(cd -- "$(dirname -- "$candidate")" && pwd)/ninfer-serve"
      break
    fi
  done
  if [[ -z "${NINFER_SERVER:-}" ]]; then
    printf 'No sm_70 ninfer-serve found (looked in $BUILD_DIR, ~/build-v100, build-v100/ beside and\n' >&2
    printf 'above the checkout, and /workspace/build-v100/). Build one with scripts/v100_build.sh, or\n' >&2
    printf 'set NINFER_SERVER (or BUILD_DIR).\n' >&2
    exit 1
  fi
  export NINFER_SERVER
fi

# Serve the network as well as this machine (see the header); NINFER_HOST overrides it.
export NINFER_HOST="${NINFER_HOST:-0.0.0.0}"
if [[ "$NINFER_HOST" != 127.0.0.1 && "$NINFER_HOST" != localhost && "$NINFER_HOST" != ::1 && -z "${NINFER_API_KEY:-}" ]]; then
  printf 'Note: listening on %s without NINFER_API_KEY; anyone who can reach the port can use the model.\n' "$NINFER_HOST"
fi

case "$model_key" in
  qwen38-27b)
    export NINFER_SPEC="${NINFER_SPEC:-mtp}"
    export NINFER_KV_DTYPE="${NINFER_KV_DTYPE:-int8}" ;;
  qwen38-flash-next|qwen38-flash-next-iq3)
    # The tuned profile's CPU share (0.4) suits long prompts; calls of up to 1024 tokens give the CPU
    # every miss it can take instead, since their missed experts see a few tokens each and the CPU
    # computes them faster than this link copies them (TTFT of a short prompt ~300 -> ~250 ms, a
    # 342-token prompt 1.6 -> 1.5 s, 2K prompts unchanged).
    export NINFER_PREFILL_CPU_ALL_COLUMNS="${NINFER_PREFILL_CPU_ALL_COLUMNS:-1024}"
    # Longer calls let the CPU take cold experts routed up to 64 tokens (as jobs of 8) instead of
    # copying them over the link inside their layer, at share 0.5 instead of the profile's 0.4: a
    # 2K-token prompt 3.78 -> 3.36 s, 3.6K 5.17 -> 4.85 s, short prompts unchanged, the same output.
    export NINFER_PREFILL_CPU_MAX_COLUMNS="${NINFER_PREFILL_CPU_MAX_COLUMNS:-64}"
    export NINFER_PREFILL_CPU_SHARE="${NINFER_PREFILL_CPU_SHARE:-0.5}"
    # The n-gram table's blocks in the RAM the ledger spares last (after the experts, the context
    # cache and a 2 GiB margin): a prompt's row read halves on this drive (the same output).
    export NINFER_NGRAM_RAM_MIB="${NINFER_NGRAM_RAM_MIB:-auto}"
    # The model's full 262,144-token context: the KV pool is backed as it fills, borrowing expert
    # frames only then (run.sh's Flash-Next default is 32,768).
    v100_context=262144
    v100_pin_gib=36 # the 33 GiB expert pin and its neighbours
    v100_experts_gib=33
    if [[ "$model_key" == 'qwen38-flash-next-iq3' ]]; then
      # --partial-ram-residency (run.sh's default for this build) refuses the elastic KV pool the full
      # context needs; with it off every expert is pinned (42.3 GiB bank) beside the n-gram RAM and cache.
      v100_experts_gib=42
      if [[ "${NINFER_PARTIAL_RAM_RESIDENCY:-on}" == 'on' ]]; then v100_context=32768; v100_pin_gib=42
      else v100_pin_gib=46; fi
    fi
    export NINFER_CONTEXT="${NINFER_CONTEXT:-$v100_context}"
    # MTP drafts one step at a time and stops once no further draft could be verified (the same
    # output): a round mostly needs one or two of the three drafts, and a step is ~1.2 ms here.
    export NINFER_MTP_STAGED_DRAFTS="${NINFER_MTP_STAGED_DRAFTS:-on}"
    # A verification round's misses go partly to CPU threads beside the GPU instead of over the link
    # (the same output): a prompt's first pass, 48-63 -> 68-80 tok/s on this card.
    export NINFER_CPU_VERIFY_MISSES="${NINFER_CPU_VERIFY_MISSES:-on}"
    if [[ -z "${NINFER_CPU_VERIFY_WORKERS:-}" ]]; then
      threads="$(nproc 2>/dev/null || echo 12)"
      workers=$(( threads - 6 )); (( workers < 6 )) && workers=6; (( workers > 10 )) && workers=10
      export NINFER_CPU_VERIFY_WORKERS="$workers"
    fi
    # The embedding's rare rows are read from pinned RAM, giving their VRAM to the expert cache (the
    # UD-IQ3_XXS build's Q6_K embedding too: 480 MiB freed at 8192 rows instead of 311).
    export NINFER_HOT_EMBEDDING_ROWS="${NINFER_HOT_EMBEDDING_ROWS:-8192}"
    # Prefill's BF16 and GGML linears (cuBLAS) and prompt attention (native m8n8k4) on the FP16 tensor
    # cores instead of emulated Ampere MMA: a 3.9K-token prompt 6.9 -> 6.0 s, more accurate per op,
    # not bit for bit the same output. NINFER_SM70_TENSOR_CORES=0 turns it off.
    export NINFER_SM70_TENSOR_CORES="${NINFER_SM70_TENSOR_CORES:-1}"
    lock_kib="$(ulimit -l)"
    need_kib=$(( v100_pin_gib * 1024 * 1024 ))
    if [[ "$lock_kib" != 'unlimited' ]] && (( lock_kib < need_kib )); then
      if ulimit -l unlimited 2>/dev/null || ulimit -l "$need_kib" 2>/dev/null; then
        :
      elif [[ -z "${NINFER_ALLOW_UNLOCKED_HOST_MEMORY:-}" ]]; then
        printf 'Memory-lock limit is %s KiB, below the %s GiB Flash-Next pins; continuing with\n' "$lock_kib" "$v100_experts_gib"
        printf 'NINFER_ALLOW_UNLOCKED_HOST_MEMORY=on (see the top of this script to lift the limit).\n'
        export NINFER_ALLOW_UNLOCKED_HOST_MEMORY=on
      fi
    fi ;;
esac

if command -v nvidia-smi >/dev/null 2>&1; then
  gpu_name="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || true)"
  case "$gpu_name" in
    *V100*) ;;
    '') printf 'Warning: nvidia-smi lists no GPU.\n' >&2 ;;
    *) printf 'Warning: this launcher is for a Tesla V100; the GPU is %s (use run.sh).\n' "$gpu_name" >&2 ;;
  esac
  others="$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null || true)"
  if [[ -n "$others" ]]; then
    printf 'Warning: other processes are using the GPU; the server sizes itself to what is left:\n%s\n' "$others" >&2
  fi
  power="$(nvidia-smi --query-gpu=power.limit --format=csv,noheader,nounits 2>/dev/null | head -1 || true)"
  if [[ -n "$power" ]] && awk -v p="$power" 'BEGIN { exit !(p < 249) }'; then
    printf 'Note: the power limit is %s W (the V100 default is 250 W; raise it with sudo nvidia-smi -pl 250\n' "$power"
    printf 'if the card is cooled for it).\n'
  fi
fi

exec "$script_dir/run.sh" "$@"
